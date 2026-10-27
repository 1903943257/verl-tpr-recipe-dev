# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin integration helpers between verl's MegatronEngine and TPR."""

from __future__ import annotations

import copy
import inspect
import os
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec, get_module
from megatron.core.utils import get_model_config
from torch import Tensor

from verl_tpr.attention import TPRSelfAttention
from verl.utils import tensordict_utils as tu
from verl.utils.megatron_utils import unwrap_model
from verl.utils.metric import AggregationType, Metric

from .executor import SegmentEventProfiler, SegmentExecutor, prepare_replay_model, validate_replay_plan
from .ring import aggregate_cp_loss, resolve_engine_cp_runtime
from .tree import ForestExecutionPlan, TreeExecutionPlan, assert_tp_plan_agreement, digest_forest_plan

if TYPE_CHECKING:
    from verl.workers.engine.megatron.transformer_impl import MegatronEngine


def run_forest(
    forest: ForestExecutionPlan,
    run_tree: Callable[[TreeExecutionPlan], Tensor],
    finalize_gradients: Callable[[], None],
    no_sync_context: Callable[[], Any] | None = None,
) -> Tensor:
    """Run all trees under one native no-sync scope and finalize once."""
    if not forest.trees:
        raise ValueError("forest must contain at least one executable tree")
    total = None
    with nullcontext() if no_sync_context is None else no_sync_context():
        for tree in forest.trees:
            # No objective anywhere below this root: neither replay nor dKV is
            # needed. This is a global-tree decision, never a CP-local count.
            if not tree.logical_loss_tokens:
                continue
            loss = run_tree(tree)
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise TypeError("run_tree must return a scalar Tensor")
            total = loss.detach() if total is None else total + loss.detach()
    if total is None:
        raise ValueError("forest has no supervised logical response tokens")
    # A failed tree never reaches native gradient finalization.
    finalize_gradients()
    return total


class _PPOMetrics:
    """Preserve native SUM metrics and weight segment MEANs by logical tokens."""

    def __init__(self):
        self.values: dict[str, list[Any]] = {}
        self.kinds: dict[str, set[bool]] = {}

    def add(self, metrics, token_weight):
        for name, value in metrics.items():
            wrapped = isinstance(value, Metric)
            self.kinds.setdefault(name, set()).add(wrapped)
            if wrapped:
                if value.aggregation is AggregationType.MEAN:
                    value = value.aggregate() * token_weight
                elif value.aggregation is not AggregationType.SUM:
                    raise NotImplementedError(f"TPR PPO metric {name} requires unsupported {value.aggregation}")
            self.values.setdefault(name, []).append(value)

    def scalar_lists(self, cp_runtime=None):
        # V1 Trainer uses np.mean(list), so wrapped metrics must return ONE
        # additive scalar. Plain numeric diagnostics retain their list semantics.
        result = {}
        for name, values in self.values.items():
            if len(self.kinds[name]) != 1:
                raise TypeError(f"TPR metric {name} mixes Metric and numeric values")
            values = [float(value.aggregate() if isinstance(value, Metric) else value) for value in values]
            result[name] = [sum(values)] if True in self.kinds[name] else values
        if cp_runtime is None or not cp_runtime.enabled:
            return result

        # Ranks with no owned refs may have no metric keys at all. Exchange the
        # small, already-host-side scalar dictionaries once, not one collective
        # per key (which would deadlock on an empty rank). SUMs and weighted
        # MEANs are additive across CP; plain diagnostics keep list semantics.
        import torch.distributed as dist

        local = {name: (True in self.kinds[name], values) for name, values in result.items()}
        peers = [None] * cp_runtime.size
        dist.all_gather_object(peers, local, group=cp_runtime.group)
        result, kinds = {}, {}
        for peer in peers:
            for name, (wrapped, values) in peer.items():
                if name in kinds and kinds[name] != wrapped:
                    raise TypeError(f"TPR metric {name} mixes Metric and numeric values across CP")
                kinds[name] = wrapped
                result.setdefault(name, []).extend(values)
        return {name: [sum(values)] if kinds[name] else values for name, values in result.items()}


def _assert_finite_tree_loss(loss: Tensor, *, tp_size: int, cp_size: int) -> None:
    finite = torch.isfinite(loss.detach()).to(torch.int64)
    if tp_size * cp_size > 1:
        import torch.distributed as dist
        from megatron.core import parallel_state

        group = (parallel_state.get_tensor_and_context_parallel_group() if cp_size > 1
                 else parallel_state.get_tensor_model_parallel_group())
        if group is None:
            raise RuntimeError("TPR cannot resolve the loss-check model-parallel group")
        # All model-parallel peers reject before another tree or finalization;
        # a rank-local exception here would strand finite peers in collectives.
        dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
    if not bool(finite.item()):
        raise FloatingPointError("non-finite TPR PPO tree loss; gradients will not be finalized")


def install_tpr_module_spec(provider: Any, *, bridge: Any | None = None) -> None:
    """Install the TPR attention spec for the actual model-construction path.

    Non-vanilla uses the provider. Vanilla MBridge builds its layer spec lazily
    through the bridge instance, so patch that instance before get_model().
    """
    if provider is None:
        if bridge is None:
            raise NotImplementedError(
                "TPR model construction requires either a non-vanilla provider "
                "or the vanilla bridge instance"
            )
        install_tpr_vanilla_mbridge_spec(bridge)
        return
    provider.transformer_layer_spec = make_tpr_module_spec_provider(
        provider.transformer_layer_spec
    )



def _validate_tpr_parallel_sizes(sizes: dict[str, int], *, ppo: bool) -> int:
    """Support native VERL DP=2 in the tested TP2×RingCP2 PPO topology.

    DP replicas retain independent TPR forests: no cross-DP Prefix KV.
    Standard VERL equal-cardinality dispatch and Megatron optimizer finalize
    remain responsible for grouping and gradient synchronization.
    """
    tp, cp, dp = sizes["TP"], sizes["CP"], sizes["DP"]
    production_dp2 = ppo and (tp, cp, dp) == (2, 2, 2)
    # Keep the isolated 10K parity matrix opt-in; do not broaden production.
    parity_10k_dp2 = (
        ppo and dp == 2 and tp in (1, 2) and cp in (1, 2)
        and os.getenv("TPR_EXPERIMENTAL_OFFLINE_DP2") == "1"
        and os.getenv("TPR_RUN_10K_PARALLEL_PARITY") == "1"
        and os.getenv("TPR_10K_PARITY_ONLY") == "1"
    )
    supported_cp = (
        (cp in (1, 2) if tp == 1 else cp in (1, 2, 4))
        if ppo else (cp >= 1 if tp == 1 else cp in (1, 2, 4))
    )
    if (tp not in (1, 2) or not supported_cp
            or any(sizes[k] != 1 for k in ("PP", "EP"))
            or (dp != 1 and not (production_dp2 or parity_10k_dp2))):
        raise NotImplementedError(
            "TPR PPO supports TP1 CP1/2 DP1, TP2 CP1/2/4 DP1 and TP2 CP2 DP2; "
            f"PP=EP=1; other DP topologies and non-PPO DP2 are gated, got {sizes}"
        )
    return tp


def _validate_tpr_cp_combination(
    tp_size: int, cp_runtime: Any, config: Any, *, ppo: bool
) -> None:
    if cp_runtime.size <= 1:
        return
    if (ppo or tp_size > 1) and cp_runtime.backend_name != "ring":
        raise NotImplementedError(
            "TPR TP+CP PPO requires the tested Ring backend; "
            f"got {cp_runtime.backend_name!r}"
        )
    if tp_size > 1 and getattr(config, "sequence_parallel", False):
        raise NotImplementedError(
            "TP2+CP requires sequence_parallel=False; combined TP+SP+CP "
            "has not been validated"
        )


def _trajectory_keys_from_minibatch(data: Any) -> tuple[str, ...]:
    """Resolve stable per-row identities WITHOUT asking the VERL Trainer to build a plan.

    tpr_trajectory_keys preserves exact TQ identity when propagated to the actor
    mini-batch. The usual VERL uid column is an acceptable fallback: it
    provides the shared-sample grouping boundary, while row numbers guarantee
    unique trajectory keys after mini-batch sampling/shuffling.
    """
    count = len(data["input_ids"])
    explicit = tu.get_non_tensor_data(data, key="tpr_trajectory_keys", default=None)
    if explicit is not None:
        if hasattr(explicit, "tolist"):
            explicit = explicit.tolist()
        keys = tuple(explicit)
        if len(keys) != count or not all(isinstance(key, str) for key in keys):
            raise ValueError("tpr_trajectory_keys must contain one string per mini-batch row")
        return keys

    uid = data.get("uid", None)
    if uid is None:
        raise ValueError(
            "TPR PPO requires tpr_trajectory_keys or a per-row uid in the actor "
            "mini-batch. Preserve the TQ identity across VERL preprocessing."
        )
    if hasattr(uid, "tolist"):
        values = uid.tolist()
    elif hasattr(uid, "unbind"):
        values = list(uid.unbind())
    else:
        values = list(uid)
    if len(values) != count:
        raise ValueError(f"uid row count {len(values)} != input_ids rows {count}")
    result = []
    for row, value in enumerate(values):
        value = tu.unwrap_non_tensor_data(value)
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, (str, int)) or not str(value):
            raise ValueError(f"uid row {row} must be a nonempty string or integer")
        result.append(f"{value}_tpr_{row}")
    return tuple(result)


def _assert_dense_model(model: Any) -> None:
    # Inspect the actual decoder slots, not just the presence of one supported
    # module somewhere in a potentially mixed model. Native VERL is unchanged.
    layers = getattr(getattr(model, "decoder", None), "layers", ())
    if not layers or any(not isinstance(getattr(layer, "self_attention", None), TPRSelfAttention)
                         for layer in layers):
        raise TypeError("TPR PPO requires every decoder layer to use TPRSelfAttention")


def run_tpr_forward_backward_batch(
    engine: MegatronEngine,
    data: Any,
    loss_function: Any,
    *,
    forward_only: bool = False,
) -> dict[str, Any]:
    """Native VERL PPO mini-batch -> TPR Forest schedule -> one gradient finalize.

    The Engine MUST compute batch_num_tokens / dp_size first. It MUST invoke
    this function BEFORE prepare_micro_batches, so shared trajectories remain
    in the same logical mini-batch. Current training scope: dense Qwen/GPT,
    TP=1/2, CP=1/2/4, PP=EP=1. Production DP2 supports TP2/CP2
    with native VERL dispatch and Megatron gradient finalization; other
    DP>1 combinations remain explicitly gated.
    Ring required for CP>1,
    and native SP is restricted to CP=1 at TP2.
    Vanilla token-mean PPO; no native /M schedule
    scaling is applied because SegmentExecutor performs autograd itself.
    """
    from .objective import SegmentPPOObjectiveAdapter
    from .tree import build_tree_execution_plans

    if forward_only:
        raise NotImplementedError("TPR PPO schedule is training-only; use native forward_only")
    if not engine.engine_config.tpr_enabled:
        raise RuntimeError("TPR PPO requires engine_config.tpr_enabled=True at model creation")
    if loss_function is None:
        raise ValueError("TPR PPO requires the original VERL actor loss_function")
    if len(engine.module) != 1:
        raise NotImplementedError("TPR PPO requires exactly one Megatron model chunk (PP=VPP=1)")

    sizes = {
        "TP": engine.engine_config.tensor_model_parallel_size,
        "PP": engine.engine_config.pipeline_model_parallel_size,
        "EP": engine.engine_config.expert_model_parallel_size,
        "CP": engine.engine_config.context_parallel_size,
        "DP": engine.get_data_parallel_size(),
    }
    tp_size = _validate_tpr_parallel_sizes(sizes, ppo=True)
    if engine.engine_config.virtual_pipeline_model_parallel_size not in (None, 1):
        raise NotImplementedError("TPR PPO does not support virtual PP")
    if engine.model_config.mtp.enable or engine.enable_routing_replay:
        raise NotImplementedError("TPR PPO does not yet support MTP or router replay")

    wrapped_model = engine.module[0]
    model = unwrap_model(wrapped_model)
    config = get_model_config(wrapped_model)
    if not model.training:
        raise RuntimeError("TPR PPO requires model.train()")
    # Native Full Checkpoint cannot export TPR KV autograd roots by itself.
    # Only allow the isolated experimental adapter after installation checks.
    # Preserve PR7's default OFF behavior; only installed full/uniform/1 is opt-in.
    if getattr(config, "recompute_granularity", None) is not None:
        from .recompute import assert_tpr_full_recompute_installed

        assert_tpr_full_recompute_installed(model, config)
    unsupported = {
        "calculate_per_token_loss": getattr(config, "calculate_per_token_loss", False),
        "FP8": getattr(config, "fp8", None) not in (None, False),
        "MoE": getattr(config, "num_moe_experts", None) is not None,
        "CUDA graph": getattr(config, "cuda_graph_impl", None) not in (None, "none"),
        "native CPU offload": getattr(config, "cpu_offloading", False),
        "fused LM head": bool(getattr(engine.engine_config, "use_fused_kernels", False)),
    }
    if any(unsupported.values()):
        raise NotImplementedError(
            "TPR PPO phase-4 does not support: "
            + ", ".join(k for k, enabled in unsupported.items() if enabled)
        )
    _assert_dense_model(model)
    prepared_model = prepare_replay_model(model)

    token_count = tu.get_non_tensor_data(data, key="batch_num_tokens", default=None)
    dp_size = tu.get_non_tensor_data(data, key="dp_size", default=None)
    if (token_count is None or token_count <= 0
            or dp_size != sizes["DP"]):
        raise ValueError(
            "Engine must attach positive GLOBAL batch_num_tokens and the "
            f"actual dp_size={sizes['DP']} before TPR routing"
        )

    detailed_profile = os.getenv("TPR_EXEC_PROFILE", "0") == "1"
    stage_ms = {}
    profile_origin = time.perf_counter()

    def record_stage(name, start):
        if detailed_profile:
            stage_ms[name] = stage_ms.get(name, 0.0) + (time.perf_counter() - start) * 1000

    tree_policy = os.getenv("TPR_TREE_POLICY", "qwen_turn")
    try:
        turn_min_tokens = int(os.getenv("TPR_TURN_MIN_SEGMENT_TOKENS", "128"))
    except ValueError as exc:
        raise ValueError("TPR_TURN_MIN_SEGMENT_TOKENS must be a positive integer") from exc
    keys = _trajectory_keys_from_minibatch(data)
    forest = build_tree_execution_plans(
        keys, data, tree_policy=tree_policy,
        turn_min_segment_tokens=turn_min_tokens,
    )
    if not forest.trees or not forest.logical_loss_tokens:
        raise ValueError("TPR PPO mini-batch has no supervised response tokens")
    record_stage("tree_build_cpu_ms", profile_origin)
    tp_check_start = time.perf_counter()
    if tp_size * sizes["CP"] > 1:
        assert_tp_plan_agreement(digest_forest_plan(forest), tp_size=tp_size, cp_size=sizes["CP"])
    record_stage("tp_plan_collective_ms", tp_check_start)
    cp_runtime = resolve_engine_cp_runtime(engine, config)
    _validate_tpr_cp_combination(tp_size, cp_runtime, config, ppo=True)
    # Reject unsupported physical shapes for the whole forest before any
    # forward/backward can leave the mini-batch with partial gradients.
    for tree_plan in forest.trees:
        if tree_plan.logical_loss_tokens:
            validate_replay_plan(
                tree_plan.segment_plan, prepared_model, cp_size=cp_runtime.size,
                active_segment_ids=tree_plan.active_segment_ids,
            )
    # DP2+CP2 has twice the per-rank local sequence of the CP4 baseline.
    # Default this *isolated* experiment to recomputable PPO logprob chunks;
    # ordinary TPR and all other topologies remain unchanged.
    default_lp_chunk = "512" if sizes["DP"] == 2 and sizes["CP"] == 2 else "0"
    try:
        lp_chunk_tokens = int(os.getenv("TPR_PPO_LOGPROB_CHUNK_TOKENS", default_lp_chunk))
    except ValueError as exc:
        raise ValueError("TPR_PPO_LOGPROB_CHUNK_TOKENS must be a nonnegative integer") from exc
    if lp_chunk_tokens < 0:
        raise ValueError("TPR_PPO_LOGPROB_CHUNK_TOKENS must be nonnegative")
    objective = SegmentPPOObjectiveAdapter(
        data, loss_function, dp_group=engine.get_data_parallel_group(),
        log_prob_chunk_size=lp_chunk_tokens or None,
    )

    # Native Megatron uses /M inside its PP schedule. That schedule is bypassed;
    # the objective already carries the full VERL global denominator, so do
    # NOT multiply or divide by the number of physical tree segments here.
    no_sync_func = getattr(config, "no_sync_func", None)
    if isinstance(no_sync_func, list):
        raise NotImplementedError("TPR PPO does not support per-chunk no_sync callbacks")
    finalize = getattr(config, "finalize_model_grads_func", None)
    if not callable(finalize):
        raise RuntimeError("TPR PPO requires native finalize_model_grads_func")

    metrics = _PPOMetrics()
    event_profiler = SegmentEventProfiler() if detailed_profile else None
    tree_number = 0
    leaf_batch_stats = {"model_forward_calls": 0, "grouped_leaf_calls": 0, "grouped_leaf_segments": 0}

    def run_one_tree(tree_plan):
        nonlocal tree_number
        setup_start = time.perf_counter() if detailed_profile else None
        shard_for_segment = (
            None if cp_runtime.backend is None
            else lambda segment: cp_runtime.backend.make_sequence_shard(segment.length)
        )
        callback, counts = objective.bind_tree(tree_plan, shard_for_segment=shard_for_segment)
        executor = SegmentExecutor(
            model,
            tree_plan.segment_plan,
            expected_layer_numbers=prepared_model.layer_numbers,
            prepared_model=prepared_model,
            active_segment_ids=tree_plan.active_segment_ids,
            loss_scale_func=getattr(config, "grad_scale_func", None),
            segment_loss_fn=callback,
            segment_loss_term_counts=counts,
            cp_group=cp_runtime.group,
            cp_backend=cp_runtime.backend,
        )
        record_stage("executor_setup_cpu_ms", setup_start)
        loss = executor.run(event_profiler=event_profiler, tree_index=tree_number)
        for name in leaf_batch_stats:
            leaf_batch_stats[name] += getattr(executor, name)
        tree_number += 1
        metrics_start = time.perf_counter() if detailed_profile else None
        _assert_finite_tree_loss(loss, tp_size=tp_size, cp_size=cp_runtime.size)
        for segment_id, segment_metrics in executor.segment_loss_metrics:
            token_weight = counts[segment_id] / forest.logical_loss_tokens
            metrics.add(segment_metrics, token_weight)
        record_stage("segment_metrics_cpu_ms", metrics_start)
        return loss

    run_start = time.perf_counter() if detailed_profile else None
    loss = run_forest(
        forest, run_one_tree,
        lambda: finalize(engine.module, None, force_all_reduce=True),
        no_sync_func,
    )
    record_stage("tree_run_and_finalize_host_ms", run_start)
    if os.getenv("TPR_LEAF_BATCH_TRACE", "0") == "1":
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(
            f"TPR_LEAF_BATCH_RUN rank={rank} "
            f"configured_k={os.getenv('TPR_LEAF_BATCH_SIZE', '1')} "
            f"max_leaf_tokens={os.getenv('TPR_LEAF_BATCH_MAX_TOKENS', '8192')} "
            f"cp={cp_runtime.size} "
            f"trees={tree_number} "
            f"model_forward_calls={leaf_batch_stats['model_forward_calls']} "
            f"grouped_leaf_calls={leaf_batch_stats['grouped_leaf_calls']} "
            f"grouped_leaf_segments={leaf_batch_stats['grouped_leaf_segments']} "
            f"activated={int(leaf_batch_stats['grouped_leaf_calls'] > 0)}",
            flush=True,
        )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("non-finite TPR PPO mini-batch loss")
    if objective.capture_log_probs:
        # Explicitly opt-in through tpr_capture_log_probs for numerical gates.
        # Never materialize all per-token logprobs on the host during training.
        engine._tpr_captured_log_probs = dict(objective.debug_new_log_probs)
    if objective.capture_logit_parts:
        engine._tpr_captured_logit_parts = dict(objective.debug_logit_parts)

    logical_input_tokens = sum(int(row.numel()) for row in data["input_ids"].unbind())
    unique_tree_tokens = sum(
        segment.length for tree in forest.trees
        for segment in tree.segment_plan.segments.values()
    )
    active_tree_tokens = sum(
        tree.segment_plan.get(segment_id).length for tree in forest.trees for segment_id in tree.active_segment_ids
    )
    # All segments share the original global batch denominator. Pass the
    # same additive objective to both legacy reduction slots.
    _, global_normalized_loss = aggregate_cp_loss(
        loss, loss, cp_runtime
    )
    loss = float(global_normalized_loss.item())
    if detailed_profile:
        record_stage("whole_entry_host_ms", profile_origin)
        # NPU events must only be materialized after caller's one final NPU sync.
        engine._tpr_segment_event_profiler = event_profiler
        engine._tpr_stage_profile = stage_ms
        engine._tpr_segment_peak_allocated_bytes = event_profiler.max_allocated_bytes
        engine._tpr_segment_peak_reserved_bytes = event_profiler.max_reserved_bytes
    diagnostics = {
        "tpr/forest_trees": len(forest.trees),
        "tpr/physical_segments": forest.segment_count,
        "tpr/logical_loss_tokens": forest.logical_loss_tokens,
        "tpr/logical_input_tokens": logical_input_tokens,
        "tpr/unique_tree_tokens": unique_tree_tokens,
        "tpr/topology_token_reuse_ratio": logical_input_tokens / unique_tree_tokens,
        # The historical keys above describe the unpruned topology. These are
        # the globally supervised closure, not Push+Pop forward token totals.
        "tpr/active_tree_tokens": active_tree_tokens,
        "tpr/active_physical_segments": sum(len(tree.active_segment_ids) for tree in forest.trees),
        "tpr/model_forward_calls": leaf_batch_stats["model_forward_calls"],
        "tpr/grouped_leaf_calls": leaf_batch_stats["grouped_leaf_calls"],
        "tpr/grouped_leaf_segments": leaf_batch_stats["grouped_leaf_segments"],
    }
    metric_lists = metrics.scalar_lists(cp_runtime)
    metric_lists.update({name: [float(value)] for name, value in diagnostics.items()})

    # VERL postprocess expects list-valued micro-batch metrics and a list of
    # loss contributions. The sum of all physical segment objectives is the
    # entire mini-batch objective, with no additional micro-batch factor.
    return {
        "model_output": {},
        "loss": [loss],
        "metrics": metric_lists,
    }


# Megatron ModuleSpec installation is an Engine concern, not a model kernel.
_SpecT = TypeVar("_SpecT")


def replace_self_attention_with_tpr(transformer_layer_spec: _SpecT) -> _SpecT:
    """Copy a dense attention spec while preserving its configuration.

    Both forms used by the local verl model builders are accepted: a single
    TransformerLayer ``ModuleSpec`` and a TransformerBlock submodule object
    containing ``layer_specs``.  Only the nested attention module class is
    changed; its params, submodules, and metainfo are preserved.
    """

    if transformer_layer_spec is None:
        raise TypeError("transformer_layer_spec must not be None")

    copied_spec = copy.deepcopy(transformer_layer_spec)
    layer_specs = _get_layer_specs(copied_spec)

    for index, layer_spec in enumerate(layer_specs):
        if not isinstance(layer_spec, ModuleSpec):
            raise TypeError(
                f"layer_specs[{index}] must be a ModuleSpec, "
                f"got {type(layer_spec).__name__}"
            )
        layer_submodules = layer_spec.submodules
        if not hasattr(layer_submodules, "self_attention"):
            raise ValueError(f"layer_specs[{index}] does not define self_attention submodules")

        attention_spec = layer_submodules.self_attention
        if not isinstance(attention_spec, ModuleSpec):
            raise TypeError(
                f"layer_specs[{index}].submodules.self_attention must be a ModuleSpec, "
                f"got {type(attention_spec).__name__}"
            )

        attention_module = get_module(attention_spec)
        if attention_module is TPRSelfAttention:
            continue
        if attention_module is not SelfAttention:
            module_name = getattr(attention_module, "__name__", repr(attention_module))
            raise TypeError(
                f"layer_specs[{index}] uses unsupported attention module {module_name}; "
                "TPR requires Megatron SelfAttention"
            )

        layer_submodules.self_attention = replace(attention_spec, module=TPRSelfAttention)

    return copied_spec


def make_tpr_module_spec_provider(
    original: _SpecT | Callable[..., _SpecT],
) -> _SpecT | Callable[..., _SpecT]:
    """Wrap a Megatron-Bridge layer-spec provider with the TPR replacement."""

    if not callable(original):
        return replace_self_attention_with_tpr(original)
    accepts_vp_stage = "vp_stage" in inspect.signature(original).parameters

    def tpr_spec_provider(config: Any, vp_stage: int | None = None) -> _SpecT:
        spec = original(config, vp_stage=vp_stage) if accepts_vp_stage else original(config)
        return replace_self_attention_with_tpr(spec)

    return tpr_spec_provider



def install_tpr_vanilla_mbridge_spec(bridge: Any) -> None:
    """Patch only this vanilla MBridge instance's lazy TransformerLayer spec.

    mbridge 0.15.1 resolves the spec inside _model_provider; install before
    bridge.get_model() so the constructed model uses TPRSelfAttention.
    """
    original = getattr(bridge, "_get_transformer_layer_spec", None)
    if not callable(original):
        raise TypeError(
            "Vanilla MBridge TPR requires callable "
            "bridge._get_transformer_layer_spec; got "
            f"{type(original).__name__}"
        )
    if getattr(original, "_tpr_vanilla_mbridge_spec", False):
        return

    @wraps(original)
    def tpr_layer_spec(*args: Any, **kwargs: Any) -> Any:
        return replace_self_attention_with_tpr(original(*args, **kwargs))

    tpr_layer_spec._tpr_vanilla_mbridge_spec = True
    bridge._get_transformer_layer_spec = tpr_layer_spec


def _get_layer_specs(transformer_layer_spec: object) -> list[ModuleSpec]:
    is_spec = isinstance(transformer_layer_spec, ModuleSpec)
    container = transformer_layer_spec.submodules if is_spec else transformer_layer_spec
    if hasattr(container, "layer_specs"):
        layer_specs = container.layer_specs
    elif is_spec and hasattr(container, "self_attention"):
        layer_specs = [transformer_layer_spec]
    elif is_spec:
        raise ValueError(
            "ModuleSpec must describe either a TransformerLayer with self_attention "
            "or a TransformerBlock with layer_specs"
        )
    else:
        raise TypeError(
            "transformer_layer_spec must be a TransformerLayer ModuleSpec or an object "
            f"containing layer_specs, got {type(transformer_layer_spec).__name__}"
        )

    if not isinstance(layer_specs, list) or not layer_specs:
        raise ValueError("layer_specs must be a non-empty list")
    return layer_specs
