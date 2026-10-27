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

"""Public TPR Ring CP API and Engine backend; see ring_layout/ring_attention.

The public and diagnostic import names remain available here. For profiling or
monkeypatching, instrument the *owning* implementation module.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import Tensor

# Keep the historical diagnostic and CP1 SBH import surface available.
from verl_tpr.rectangular_attention import _compressed_causal_mask

from .ring_layout import (
    SequenceRange,
    _validate_integer,
    _validate_parallel_coordinates,
    _normalize_dimension,
    _validate_global_offset,
    SequenceShard,
    round_up_sequence_length,
    iter_valid_sequence_slices,
    resolve_cp_group,
    _group_world_size_and_rank,
    _TND_LAYOUT,
    _RIGHT_DOWN_CAUSAL_MODE,
    _FULL_ATTENTION_MODE,
    _MAX_TOKENS,
    RingBlockKind,
    RingLocalKVBlock,
    _RingAttentionConfig,
    _RingStaticLayout,
    _RingRangeSlice,
    make_ring_sequence_shard,
    classify_ring_block,
    _load_mindspeed_ring_primitives,
    _process_group_global_ranks,
    _validate_ring_shard,
    _validate_kv_pair,
    _prepare_ring_layout,
    ring_merge_settings,
    _normalize_inputs,
)

from .ring_attention import (
    _circulate_kv,
    _iter_range_slices,
    _can_coalesce_prefix_kv,
    _prefix_full_slices,
    _observe_ring_storage,
    _can_coalesce_prefix_query,
    _source_merged_prefix_kv,
    _slice_tnd_result,
    _trace_merged_query,
    _range_validity,
    _physical_block_attention_mask,
    _block_attention_parameters,
    _block_attention_forward,
    _merge_attention,
    _finalize_attention_result,
    _flatten_tnd_softmax,
    _unflatten_tnd_softmax,
    _block_attention_backward,
    _reduce_ring_gradients_to_owner,
    _source_schedule,
    _emit_ring_block_stats,
    _RingTPRAttention,
    _trace_ring_block,
    _trace_ring_storage,
)

if TYPE_CHECKING:
    from .state import KVStack, PastKVAnchors


def ordinary_ring_cp_attention(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    current_shard: SequenceShard,
    cp_group: Any,
    softmax_scale: float | None = None,
    allow_trailing_padding: bool = False,
) -> Tensor:
    """Whole causal attention using paired MindSpeed native Ring.

    Explicit entry point: an empty prefix alone does NOT identify whole mode
    (a segmented execution's root also has no prefix). Keep that root on the
    existing TPR schedule until the prefix extension is migrated separately.

    MindSpeed owns streaming KV, causal grouping, output/stat correction,
    reverse backward and owner dKV accumulation. No extra gradient SUM/cast.
    """
    _, config = _normalize_inputs(
        query, current_key, current_value, prefix_blocks=(),
        current_shard=current_shard, cp_group=cp_group,
        softmax_scale=softmax_scale,
    )
    if current_shard.global_length != current_shard.padded_length:
        if not allow_trailing_padding:
            raise ValueError("ordinary whole Ring requires an unpadded sequence")
        # The SegmentExecutor's Ring sequence shard has only VIRTUAL right-
        # padding after the final real token. For causal attention every real
        # query is earlier than these virtual tokens, so padded keys cannot
        # contribute to any real query; the PPO mask excludes padded queries.
        # Never allow this option when Prefix blocks are present.
        if not (0 < current_shard.global_length < current_shard.padded_length):
            raise ValueError("native whole Ring requires strictly trailing padding")
    from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import (
        ringattn_context_parallel,
    )

    cp_para = {
        "causal": True,
        "cp_group": cp_group,
        "cp_size": config.cp_size,
        "rank": config.cp_rank,
        "cp_global_ranks": list(config.global_ranks),
        "cp_inner_ranks": [dist.get_rank()],
        "cp_outer_ranks": list(config.global_ranks),
        "cp_dkv_outer_ranks": list(config.global_ranks),
        "megatron_cp_in_bnsd": False,
        "cache_policy": None,
        "pse_type": 1,
    }
    # Native ordinary causal Ring consumes SBH, retaining GQA's smaller KV H.
    q, k, v = (x.flatten(2).contiguous() for x in (query, current_key, current_value))
    output = ringattn_context_parallel(
        q, k, v, config.query_heads, cp_para,
        softmax_scale=config.softmax_scale, dropout_p=0.0,
    )
    if current_shard.valid_local_length != current_shard.local_length:
        # Native causal Ring can process physical trailing Query tokens, but
        # they must never contribute to the PPO loss or either gradient path.
        # Masking the output (rather than only the objective) also protects
        # Root Push/Pop when a caller supplies a nonzero padded dO.
        valid = torch.arange(current_shard.local_length, device=output.device) < current_shard.valid_local_length
        output = output.masked_fill(~valid[:, None, None], 0)
    return output


def ring_cp_attention(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    prefix_blocks: Sequence[RingLocalKVBlock] = (),
    current_shard: SequenceShard,
    cp_group: Any,
    softmax_scale: float | None = None,
    _layout: _RingStaticLayout | None = None,
    _trace_layer: int | None = None,
) -> Tensor:
    """Run blockwise TPR attention over MindSpeed causal Ring shards."""

    blocks, config = _normalize_inputs(
        query,
        current_key,
        current_value,
        prefix_blocks=prefix_blocks,
        current_shard=current_shard,
        cp_group=cp_group,
        softmax_scale=softmax_scale,
        _layout=_layout,
        _trace_layer=_trace_layer,
    )
    flat_kv = []
    for block in blocks:
        flat_kv.extend((block.key, block.value))
    flat_kv.extend((current_key, current_value))
    return _RingTPRAttention.apply(query, *flat_kv, config)


class RingCPAttentionBackend:
    """Bind one TPR Segment and ordered Prefix blocks to MindSpeed Ring CP."""

    def __init__(
        self,
        *,
        global_prefix_length: int,
        current_shard: SequenceShard,
        prefix_blocks_by_layer: Mapping[int, Sequence[RingLocalKVBlock]],
        cp_group: Any,
        use_native_root: bool = False,
    ) -> None:
        if type(use_native_root) is not bool:
            raise TypeError("use_native_root must be a bool")
        if (
            not isinstance(global_prefix_length, int)
            or isinstance(global_prefix_length, bool)
            or global_prefix_length < 0
        ):
            raise ValueError("global_prefix_length must be a non-negative integer")
        normalized = {
            layer: tuple(blocks)
            for layer, blocks in sorted(prefix_blocks_by_layer.items())
        }
        if not normalized:
            raise ValueError("prefix_blocks_by_layer must contain every TPR layer")
        path_signature = None
        for layer, blocks in normalized.items():
            if not isinstance(layer, int) or isinstance(layer, bool) or layer <= 0:
                raise ValueError(f"layer numbers must be positive integers, got {layer!r}")
            signature = tuple(block.segment_id for block in blocks)
            if path_signature is None:
                path_signature = signature
            elif signature != path_signature:
                raise ValueError("Prefix path must be identical across Ring attention layers")
            if sum(block.shard.global_length for block in blocks) != global_prefix_length:
                raise ValueError(f"layer {layer} Prefix blocks do not cover the Prefix length")
        if use_native_root and (global_prefix_length != 0 or any(normalized.values())):
            raise ValueError("native Root Ring requires zero Prefix length and no Prefix KV blocks")
        first_blocks = next(iter(normalized.values()))
        layout = _prepare_ring_layout(first_blocks, current_shard, cp_group)
        for layer, blocks in normalized.items():
            if tuple((block.segment_id, block.shard) for block in blocks) != layout.prefix_signature:
                raise ValueError(f"layer {layer} Ring Prefix placement differs from other layers")
        self._ring_layout = layout
        self._global_prefix_length = global_prefix_length
        self._current_shard = current_shard
        self._prefix_blocks_by_layer = MappingProxyType(normalized)
        self._cp_group = cp_group
        self._parallel_size = layout.cp_size
        self._first_layer = min(normalized)
        self._use_native_root = use_native_root

    @property
    def global_prefix_length(self) -> int:
        return self._global_prefix_length

    @property
    def global_suffix_length(self) -> int:
        return self._current_shard.global_length

    @property
    def local_suffix_length(self) -> int:
        return self._current_shard.local_length

    @property
    def parallel_size(self) -> int:
        return self._parallel_size

    def attention(
        self,
        layer_number: int,
        query: Tensor,
        new_key: Tensor,
        new_value: Tensor,
        *,
        softmax_scale: float | None,
    ) -> Tensor:
        try:
            blocks = self._prefix_blocks_by_layer[layer_number]
        except KeyError as exc:
            raise KeyError(f"Ring CP context has no layer {layer_number}") from exc
        if self._use_native_root:
            if (
                layer_number == self._first_layer
                and self._ring_layout.cp_rank == 0
                and os.getenv("TPR_RING_BLOCK_STATS", "0") == "1"
            ):
                print("TPR_FAST_ROOT_RING " + json.dumps({
                    "backend": "mindspeed_native",
                    "root_length": self._current_shard.global_length,
                    "padded_length": self._current_shard.padded_length,
                    "grad_enabled": torch.is_grad_enabled(),
                }, sort_keys=True), flush=True)
            return ordinary_ring_cp_attention(
                query, new_key, new_value,
                current_shard=self._current_shard,
                cp_group=self._cp_group,
                softmax_scale=softmax_scale,
                allow_trailing_padding=True,
            )
        return ring_cp_attention(
            query,
            new_key,
            new_value,
            prefix_blocks=blocks,
            current_shard=self._current_shard,
            cp_group=self._cp_group,
            softmax_scale=softmax_scale,
            _layout=self._ring_layout,
            _trace_layer=(
                layer_number if os.getenv("TPR_RING_BLOCK_STATS", "0") == "1"
                and layer_number == self._first_layer else None
            ),
        )


def _prefix_blocks_by_layer(
    kv_stack: KVStack,
    anchors: PastKVAnchors | None,
    expected_layer_numbers: tuple[int, ...],
) -> dict[int, tuple[RingLocalKVBlock, ...]]:
    """Select one ordered ancestor path; preserve original or anchored KV leaves."""
    sources = (
        tuple((segment_id, kv_stack.get(segment_id).kv) for segment_id in kv_stack.segment_ids)
        if anchors is None
        else tuple((entry.segment_id, entry) for entry in anchors.entries)
    )
    return {
        layer: tuple(
            RingLocalKVBlock(segment_id, state.shard, *state.key_values[layer])
            for segment_id, state in sources
        )
        for layer in expected_layer_numbers
    }


# Production CP policy: only the Prefix-aware Ring implementation is admitted.
class RingCPBackend:
    backend_name = "ring"

    def __init__(self, group: Any, *, parallel_size: int | None = None, parallel_rank: int | None = None):
        if (parallel_size is None) != (parallel_rank is None):
            raise ValueError("parallel_size and parallel_rank must be provided together")
        if parallel_size is None:
            parallel_size, parallel_rank = resolve_cp_group(group)
        if parallel_size <= 1 or not 0 <= parallel_rank < parallel_size:
            raise ValueError(f"invalid Ring CP coordinate {parallel_rank}/{parallel_size}")
        self._cp_group = group
        self._parallel_size = parallel_size
        self._parallel_rank = parallel_rank

    @property
    def parallel_size(self) -> int:
        return self._parallel_size

    @property
    def parallel_rank(self) -> int:
        return self._parallel_rank

    def validate_segment_length(self, global_length: int) -> None:
        self.make_sequence_shard(global_length)

    def make_sequence_shard(self, global_length: int) -> SequenceShard:
        return make_ring_sequence_shard(
            global_length, cp_rank=self.parallel_rank, cp_size=self.parallel_size,
        )

    def make_attention_backend(
        self, kv_stack: KVStack, *, expected_layer_numbers: tuple[int, ...],
        current_shard: SequenceShard, past_anchors: PastKVAnchors | None,
        use_native_root: bool = False,
    ) -> RingCPAttentionBackend:
        return RingCPAttentionBackend(
            global_prefix_length=(
                kv_stack.prefix_length if past_anchors is None
                else past_anchors.global_prefix_length
            ),
            current_shard=current_shard,
            prefix_blocks_by_layer=_prefix_blocks_by_layer(
                kv_stack, past_anchors, expected_layer_numbers,
            ),
            cp_group=self._cp_group,
            use_native_root=use_native_root,
        )


@dataclass(frozen=True, slots=True)
class TPREngineCPRuntime:
    group: Any | None
    backend: RingCPBackend | None
    size: int
    rank: int

    @property
    def enabled(self) -> bool:
        return self.size > 1

    @property
    def backend_name(self) -> str:
        return "none" if self.backend is None else self.backend.backend_name


def resolve_engine_cp_runtime(engine: Any, model_config: Any) -> TPREngineCPRuntime:
    configured_size = engine.engine_config.context_parallel_size
    if not isinstance(configured_size, int) or isinstance(configured_size, bool) or configured_size <= 0:
        raise ValueError(f"context_parallel_size must be positive, got {configured_size!r}")
    backend_name = getattr(engine.engine_config, "tpr_cp_backend", None)
    if configured_size == 1:
        if backend_name not in (None, "auto"):
            raise ValueError(f"tpr_cp_backend={backend_name!r} requires CP>1")
        return TPREngineCPRuntime(group=None, backend=None, size=1, rank=0)
    if backend_name not in (None, "auto", "ring", "megatron_cp_algo"):
        raise NotImplementedError("Production TPR only supports Ring CP")
    if getattr(engine.engine_config, "dynamic_context_parallel", False):
        raise NotImplementedError("Dynamic CP is not supported by TPR")
    from megatron.core import parallel_state

    if not dist.is_initialized() or not parallel_state.model_parallel_is_initialized():
        raise RuntimeError("TPR CP requires initialized torch.distributed and Megatron groups")
    group = parallel_state.get_context_parallel_group()
    size, rank = resolve_cp_group(group)
    if size != configured_size:
        raise RuntimeError(f"TPR CP topology mismatch: configured {configured_size}, actual {size}")
    return TPREngineCPRuntime(
        group=group,
        backend=RingCPBackend(group, parallel_size=size, parallel_rank=rank),
        size=size, rank=rank,
    )


def aggregate_cp_loss(
    loss_sum: Tensor, normalized_loss: Tensor, runtime: TPREngineCPRuntime,
) -> tuple[Tensor, Tensor]:
    global_loss_sum = loss_sum.detach().clone()
    global_normalized_loss = normalized_loss.detach().clone()
    if runtime.enabled:
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=runtime.group)
        dist.all_reduce(global_normalized_loss, op=dist.ReduceOp.SUM, group=runtime.group)
    return global_loss_sum, global_normalized_loss
