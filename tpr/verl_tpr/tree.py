# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Turn-boundary token sharing and native PPO query ownership.

The compressed SegmentPlan is the only persistent physical topology. Temporary
Turn trie nodes are discarded after assigning each logical response token to
its predicting query, including child-first tokens predicted by the parent.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor


SegmentId = int
QWEN3_IM_END_TOKEN_ID = 151645


def _validate_segment_id(value: int, *, field_name: str = "segment_id") -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    segment_id: SegmentId
    parent_id: SegmentId | None
    token_ids: Tensor
    position_start: int
    prefix_length: int

    def __post_init__(self) -> None:
        _validate_segment_id(self.segment_id)
        if self.parent_id is not None:
            _validate_segment_id(self.parent_id, field_name="parent_id")
        _validate_segment_id(self.position_start, field_name="position_start")
        _validate_segment_id(self.prefix_length, field_name="prefix_length")
        if self.prefix_length != self.position_start:
            raise ValueError("prefix_length must equal position_start")
        if not isinstance(self.token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if (self.token_ids.device.type != "cpu" or self.token_ids.dtype != torch.long
                or self.token_ids.ndim != 1 or not self.token_ids.numel()):
            raise ValueError("token_ids must be a non-empty one-dimensional CPU torch.long tensor")
        if bool((self.token_ids < 0).any()):
            raise ValueError("token_ids must be non-negative")
        object.__setattr__(self, "token_ids", self.token_ids.detach().clone())

    @property
    def length(self) -> int:
        return self.token_ids.numel()

    @property
    def position_end(self) -> int:
        return self.position_start + self.length


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    segments: Mapping[SegmentId, SegmentSpec] | Sequence[SegmentSpec]
    root_id: SegmentId
    _children: Mapping[SegmentId, tuple[SegmentId, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_segment_id(self.root_id, field_name="root_id")
        source = self.segments.values() if isinstance(self.segments, Mapping) else self.segments
        segments = {}
        for segment in source:
            if not isinstance(segment, SegmentSpec):
                raise TypeError("segments must contain SegmentSpec objects")
            if segment.segment_id in segments:
                raise ValueError(f"duplicate segment_id {segment.segment_id}")
            segments[segment.segment_id] = segment
        roots = [s.segment_id for s in segments.values() if s.parent_id is None]
        if roots != [self.root_id]:
            raise ValueError(f"plan must have exactly root {self.root_id}, got roots={roots}")
        children = {segment_id: [] for segment_id in segments}
        for segment in segments.values():
            if segment.parent_id is None:
                if segment.position_start != 0:
                    raise ValueError("root segment must start at position zero")
                continue
            parent = segments.get(segment.parent_id)
            if parent is None or segment.position_start != parent.position_end:
                raise ValueError(f"segment {segment.segment_id} must start at parent end")
            children[parent.segment_id].append(segment.segment_id)
        # Positive segment lengths and increasing parent positions exclude cycles.
        object.__setattr__(self, "segments", MappingProxyType(segments))
        object.__setattr__(self, "_children", MappingProxyType({k: tuple(v) for k, v in children.items()}))

    def get(self, segment_id: SegmentId) -> SegmentSpec:
        return self.segments[segment_id]

    def children_of(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        return tuple(self.segments[child] for child in self._children[segment_id])

    def path_to(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        path = []
        while segment_id is not None:
            segment = self.get(segment_id)
            path.append(segment)
            segment_id = segment.parent_id
        return tuple(reversed(path))

    def depth_of(self, segment_id: SegmentId) -> int:
        return len(self.path_to(segment_id)) - 1


@dataclass(frozen=True, slots=True)
class SegmentObjectiveRef:
    segment_id: int
    query_offset: int
    target_token_id: int
    sample_row: int
    response_offset: int


@dataclass(frozen=True, slots=True)
class TreeExecutionPlan:
    uid: str
    tree_index: int
    member_rows: tuple[int, ...]
    segment_plan: SegmentPlan
    objective_refs: tuple[SegmentObjectiveRef, ...]
    active_segment_ids: frozenset[SegmentId] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Global query owners and their ancestors, never CP-local counts.
        active = set()
        for ref in self.objective_refs:
            segment_id = ref.segment_id
            while segment_id is not None and segment_id not in active:
                active.add(segment_id)
                segment_id = self.segment_plan.get(segment_id).parent_id
        object.__setattr__(self, "active_segment_ids", frozenset(active))

    @property
    def key(self) -> str:
        return f"{self.uid}:{self.tree_index}"

    def refs_for_segment(self, segment_id: int) -> tuple[SegmentObjectiveRef, ...]:
        self.segment_plan.get(segment_id)
        return tuple(ref for ref in self.objective_refs if ref.segment_id == segment_id)

    @property
    def logical_loss_tokens(self) -> int:
        return len(self.objective_refs)

@dataclass(frozen=True, slots=True)
class ForestExecutionPlan:
    trees: tuple[TreeExecutionPlan, ...]
    logical_loss_tokens: int

    def __post_init__(self):
        if sum(tree.logical_loss_tokens for tree in self.trees) != self.logical_loss_tokens:
            raise ValueError("forest logical_loss_tokens disagrees with objective refs")

    @property
    def segment_count(self) -> int:
        return sum(len(tree.segment_plan.segments) for tree in self.trees)


def _rows(batch: Any, key: str) -> list[Tensor]:
    try:
        value = batch[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"batch is missing {key!r}") from exc
    values = list(value) if isinstance(value, (list, tuple)) else list(value.unbind())
    if not all(isinstance(item, Tensor) and item.ndim == 1 for item in values):
        raise ValueError(f"{key!r} must contain one-dimensional tensor rows")
    return values


def _parse_key(key: str) -> tuple[str, str, int]:
    if not isinstance(key, str):
        raise TypeError(f"trajectory key must be str, got {type(key)!r}")
    fields = key.rsplit("_", 2)
    if len(fields) != 3 or not fields[0] or not fields[1]:
        raise ValueError(f"unexpected trajectory key format: {key!r}")
    try:
        return fields[0], fields[1], int(fields[2])
    except ValueError as exc:
        raise ValueError(f"trajectory key has non-integer index: {key!r}") from exc


@dataclass(slots=True)
class _TurnNode:
    end: int
    source_row: int
    member_rows: list[int] = field(default_factory=list)
    terminal_rows: list[int] = field(default_factory=list)
    children: dict[tuple[int, ...], _TurnNode] = field(default_factory=dict)


def _turn_units(tokens: list[int], prompt_length: int, minimum: int):
    boundaries = {i + 1 for i, token in enumerate(tokens) if token == QWEN3_IM_END_TOKEN_ID}
    if 0 < prompt_length < len(tokens):
        boundaries.add(prompt_length)
    boundaries.add(len(tokens))
    previous = 0
    for end in sorted(boundaries):
        if end < len(tokens) and end - previous < minimum:
            continue
        yield tuple(tokens[previous:end]), end
        previous = end


def _compress(node: _TurnNode, parent_id: int | None, start: int, inputs: list[Tensor],
              segments: list[SegmentSpec], paths: dict[int, tuple[int, ...]], path: tuple[int, ...]) -> None:
    # Explicit DFS preserves insertion order without a Python recursion limit
    # on long conversations with terminal rows at many successive Turn nodes.
    pending = [(node, parent_id, start, path)]
    while pending:
        node, parent_id, start, path = pending.pop()
        while not node.terminal_rows and len(node.children) == 1:
            node = next(iter(node.children.values()))
        segment_id = len(segments)
        segments.append(SegmentSpec(segment_id, parent_id, inputs[node.source_row][start:node.end], start, start))
        path = (*path, segment_id)
        paths.update((row, path) for row in node.terminal_rows)
        pending.extend((child, segment_id, node.end, path) for child in reversed(node.children.values()))


def build_tree_execution_plans(
    keys: list[str] | tuple[str, ...], batch: Any, *,
    require_loss_mask_alignment: bool = False,
    tree_policy: str = "qwen_turn", turn_min_segment_tokens: int = 128,
) -> ForestExecutionPlan:
    """Build the sole executable topology and retain one ref per masked response.

    Turn signatures are token-exact, and only complete ChatML messages or the
    captured prompt endpoint can create a shared internal node. Duplicate rows
    share physical tokens but retain independent PPO references and weights.
    """
    if tree_policy != "qwen_turn":
        raise ValueError("Production TPR only supports qwen_turn")
    if (not isinstance(turn_min_segment_tokens, int) or isinstance(turn_min_segment_tokens, bool)
            or turn_min_segment_tokens < 1):
        raise ValueError("turn_min_segment_tokens must be a positive integer")
    inputs, masks = _rows(batch, "input_ids"), _rows(batch, "response_mask")
    if len(keys) != len(inputs) or len(keys) != len(masks):
        raise ValueError("keys, input_ids and response_mask must have identical row counts")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate trajectory key")
    prompts = _rows(batch, "prompts") if "prompts" in batch else None
    responses = _rows(batch, "responses") if "responses" in batch else None
    loss_masks = _rows(batch, "loss_mask") if require_loss_mask_alignment else None
    for values in (prompts, responses, loss_masks):
        if values is not None and len(values) != len(keys):
            raise ValueError("optional batch field rows do not match keys")

    groups: dict[str, _TurnNode] = {}
    selected = []
    cpu_inputs = []
    for row, (key, tokens, mask) in enumerate(zip(keys, inputs, masks, strict=True)):
        if (not tokens.numel() or tokens.dtype == torch.bool or torch.is_floating_point(tokens)
                or torch.is_complex(tokens) or bool((tokens < 0).any())):
            raise ValueError(f"input_ids[{row}] must contain non-empty non-negative integer token ids")
        cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long)
        cpu_inputs.append(cpu_tokens)
        if mask.numel() > tokens.numel():
            raise ValueError(f"response_mask[{row}] is longer than input_ids[{row}]")
        response_start = tokens.numel() - mask.numel()
        if responses is not None:
            response = responses[row]
            if response.numel() != mask.numel():
                raise ValueError(f"responses/response_mask length mismatch on row {row}")
            if not torch.equal(tokens[response_start:], response):
                raise ValueError(f"response is not an input_ids suffix on row {row}")
        offsets = torch.nonzero(mask.to(torch.bool), as_tuple=False).flatten().tolist()
        if offsets and response_start + offsets[0] == 0:
            raise ValueError(f"row {row} has no preceding query token for first supervised token")
        selected.append(offsets)
        if loss_masks is not None:
            loss_mask = loss_masks[row]
            if loss_mask.numel() == tokens.numel():
                if bool(loss_mask[:response_start].to(bool).any()):
                    raise ValueError(f"row {row} loss_mask includes prompt tokens")
                loss_mask = loss_mask[response_start:]
            if loss_mask.numel() != mask.numel() or not torch.equal(loss_mask.to(bool), mask.to(bool)):
                raise ValueError(f"row {row} response_mask and loss_mask disagree")
        prompt_length = 0
        if prompts is not None:
            prompt = prompts[row]
            prompt_length = prompt.numel()
            if prompt_length > tokens.numel() or not torch.equal(tokens[:prompt_length].cpu(), prompt.cpu()):
                raise ValueError(f"turn builder prompt is not an exact input prefix for row={row}")
        uid, _, _ = _parse_key(key)
        node = groups.setdefault(uid, _TurnNode(0, row))
        for signature, end in _turn_units(cpu_tokens.tolist(), prompt_length, turn_min_segment_tokens):
            node = node.children.setdefault(signature, _TurnNode(end, row))
            node.member_rows.append(row)
        node.terminal_rows.append(row)

    result = []
    for uid, root in groups.items():
        for tree_index, child in enumerate(root.children.values()):
            segments, paths = [], {}
            _compress(child, None, 0, cpu_inputs, segments, paths, ())
            plan = SegmentPlan(segments, root_id=0)
            refs = []
            for row in child.member_rows:
                response_start = inputs[row].numel() - masks[row].numel()
                path = paths[row]
                ends = [plan.get(segment_id).position_end for segment_id in path]
                for offset in selected[row]:
                    target = response_start + offset
                    segment_id = path[bisect_right(ends, target - 1)]
                    refs.append(SegmentObjectiveRef(
                        segment_id, target - 1 - plan.get(segment_id).position_start,
                        int(cpu_inputs[row][target]), row, offset,
                    ))
            result.append(TreeExecutionPlan(uid, tree_index, tuple(child.member_rows), plan, tuple(refs)))
    return ForestExecutionPlan(tuple(result), sum(map(len, selected)))


def _write_record(hasher, record) -> None:
    hasher.update(json.dumps(record, separators=(",", ":")).encode())


def _write_plan(hasher, plan: SegmentPlan) -> None:
    _write_record(hasher, plan.root_id)
    # Insertion order determines DFS order and the collective call sequence.
    for segment in plan.segments.values():
        _write_record(hasher, (segment.segment_id, segment.parent_id, segment.position_start, segment.length))
        hasher.update(segment.token_ids.contiguous().numpy().tobytes())


def digest_segment_plan(plan: SegmentPlan) -> str:
    hasher = hashlib.sha256(b"TPR-PPO-turn-v2")
    _write_plan(hasher, plan)
    return hasher.hexdigest()


def digest_forest_plan(forest: ForestExecutionPlan) -> str:
    hasher = hashlib.sha256(b"TPR-PPO-turn-v2")
    _write_record(hasher, forest.logical_loss_tokens)
    for tree in forest.trees:
        _write_record(hasher, (tree.uid, tree.tree_index, tree.member_rows))
        _write_plan(hasher, tree.segment_plan)
        for ref in tree.objective_refs:
            _write_record(hasher, (ref.segment_id, ref.query_offset, ref.target_token_id,
                                   ref.sample_row, ref.response_offset))
    return hasher.hexdigest()


def assert_tp_plan_agreement(digest: str, *, tp_size: int, cp_size: int = 1) -> None:
    """Reject divergent plans on the native TP group (TP×CP when CP is enabled).

    Called ONCE per TPR minibatch, *before* model forward or TP collectives.
    Uses only a 32-byte SHA256 signature per rank, not model tensors / KV.
    Every peer receives the same success or mismatch result.
    """
    if tp_size < 1 or cp_size < 1:
        raise ValueError("tp_size and cp_size must be positive")
    size = tp_size * cp_size
    if size == 1:
        return
    if len(digest) != 64:
        raise ValueError("SHA256 digest must be a 64-character hex string")

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("TPR TP/CP requires an initialized distributed process group")
    axis = "TP×CP" if cp_size > 1 else "TP"
    group = (parallel_state.get_tensor_and_context_parallel_group() if cp_size > 1
             else parallel_state.get_tensor_model_parallel_group())
    if group is None:
        raise RuntimeError(f"TPR cannot resolve native Megatron {axis} group")
    if dist.get_world_size(group=group) != size:
        raise RuntimeError(f"TPR {axis} group size differs from Engine config")
    # HCCL AllGather supports int64, so store SHA256 as eight int64 words
    # (each word is a non-negative 32-bit unsigned value).
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    device = torch.device("cpu")
    if dist.get_backend(group) != "gloo":
        from verl.utils.device import get_device_id, get_device_name

        device = torch.device(get_device_name(), get_device_id())
    local = torch.tensor(words, dtype=torch.int64, device=device)
    signatures = [torch.empty_like(local) for _ in range(size)]
    dist.all_gather(signatures, local, group=group)
    if any(not torch.equal(value, local) for value in signatures):
        rank = dist.get_rank(group=group)
        raise RuntimeError(
            f"TPR {axis} schedule mismatch on rank={rank}: forest, SegmentPlan, "
            "topology or PPO refs differ across model-parallel peers"
        )
