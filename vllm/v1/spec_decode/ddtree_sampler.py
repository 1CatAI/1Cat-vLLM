# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy DDTree sampler over compact target logits."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from vllm.v1.spec_decode.ddtree_payload import (
    DDTreeDraftPayload,
    tree_from_payload,
)
from vllm.v1.spec_decode.ddtree_verify import (
    DDTreeVerificationResult,
    greedy_verify_payload_from_compact_logits,
    greedy_verify_payload_from_target_tokens,
)

PLACEHOLDER_TOKEN_ID = -1
_SAMPLING_EPS = 1e-5
_MAX_TOPK_FIRST_STOCHASTIC_K = 256
logger = init_logger(__name__)


def _ddtree_debug_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_DEBUG", "0") == "1"


def _ddtree_worker_profile_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_WORKER_PROFILE", "0") == "1"


def _ddtree_trace_path() -> str | None:
    return os.getenv("VLLM_DFLASH_DDTREE_TRACE_JSONL")


def _ddtree_verify_row_trace_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE", "0") == "1"


def _ddtree_verify_row_trace_topk() -> int:
    raw = os.getenv("VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_TOPK", "8")
    try:
        return max(0, int(raw))
    except ValueError:
        return 8


def _write_ddtree_trace_event(
    event: str,
    payload: Mapping[str, object],
) -> None:
    trace_path = _ddtree_trace_path()
    if not trace_path:
        return
    record = {
        "event": event,
        "pid": os.getpid(),
        **payload,
    }
    try:
        with open(trace_path, "a", encoding="utf-8") as trace_file:
            json.dump(record, trace_file, ensure_ascii=True, sort_keys=True)
            trace_file.write("\n")
    except OSError:
        logger.exception("Failed to write DDTree trace event to %s", trace_path)


def _candidate_rank(
    payload: DDTreeDraftPayload,
    *,
    depth: int,
    token_id: int,
) -> int | None:
    if depth < 1 or depth > len(payload.topk_token_ids_by_depth):
        return None
    try:
        return payload.topk_token_ids_by_depth[depth - 1].index(token_id)
    except ValueError:
        return None


def _verifier_row_trace(
    payload: DDTreeDraftPayload,
    compact_logits: torch.Tensor,
    result: DDTreeVerificationResult,
) -> dict[str, object]:
    """Return row/path metadata for offline verifier-row recompute checks."""

    tree = tree_from_payload(payload)
    rows: list[dict[str, object]] = []
    for node in tree.nodes:
        children = tree.child_by_token(node.index)
        parent_index = node.parent_index
        parent_row = None if parent_index is None else int(parent_index)
        path_token_ids = [
            int(token) for token in tree.path_token_ids(node.index)
        ]
        rows.append(
            {
                "row": int(node.index),
                "parent_row": parent_row,
                "depth": int(node.depth),
                "row_token_id": None if node.index == 0 else int(node.token_id),
                "path_token_ids": path_token_ids,
                "child_token_to_row": {
                    str(int(token)): int(child)
                    for token, child in children.items()
                },
            }
        )

    accepted_path_token_ids: list[int] = []
    for node_index in result.accepted_node_indices:
        if node_index == 0:
            continue
        if 0 <= node_index < len(tree.nodes):
            accepted_path_token_ids.append(int(tree.nodes[node_index].token_id))

    trace: dict[str, object] = {
        "verifier_rows": rows,
        "accepted_path_token_ids": accepted_path_token_ids,
        "bonus_parent_row": int(result.accepted_node_indices[-1]),
        "row_trace_note": (
            "Each non-root row should match target recompute on prefix plus "
            "path_token_ids. Row 0 is the root verifier row."
        ),
    }

    topk = min(_ddtree_verify_row_trace_topk(), compact_logits.shape[-1])
    if topk > 0:
        values, token_ids = torch.topk(
            compact_logits.detach().float(),
            k=topk,
            dim=-1,
        )
        trace["logits_topk_k"] = topk
        trace["logits_topk_token_ids"] = token_ids.detach().cpu().tolist()
        trace["logits_topk_values"] = values.detach().cpu().tolist()

    return trace


def _debug_log_verification(
    *,
    req_id: str,
    payload: DDTreeDraftPayload,
    compact_logits: torch.Tensor,
    result: DDTreeVerificationResult,
) -> None:
    debug_enabled = _ddtree_debug_enabled()
    if not debug_enabled and not _ddtree_trace_path():
        return

    tree = tree_from_payload(payload)
    target_argmax = compact_logits.argmax(dim=-1).detach().cpu().tolist()
    cursor = 0
    rows: list[str] = []
    trace_rows: list[dict[str, object]] = []
    while True:
        if cursor >= len(target_argmax):
            rows.append(f"cursor={cursor} missing_compact_row")
            trace_rows.append({"row": cursor, "missing_compact_row": True})
            break
        next_token = int(target_argmax[cursor])
        children = tree.child_by_token(cursor)
        child_index = children.get(next_token)
        depth = tree.nodes[cursor].depth + 1
        rank = _candidate_rank(payload, depth=depth, token_id=next_token)
        child_tokens = tuple(children.keys())[:8]
        rows.append(
            f"row={cursor} parent={cursor} depth={depth} target={next_token} "
            f"child={child_index} topk_rank={rank} child_tokens={child_tokens}"
        )
        trace_rows.append(
            {
                "row": cursor,
                "parent": tree.nodes[cursor].parent_index,
                "depth": tree.nodes[cursor].depth,
                "target": next_token,
                "matched_child": child_index,
                "topk_rank": rank,
                "child_tokens_head": list(child_tokens),
            }
        )
        if child_index is None:
            break
        cursor = child_index

    trace_payload: dict[str, object] = {
        "req_id": req_id,
        "accepted_node_indices": list(result.accepted_node_indices),
        "accepted_token_ids": list(result.accepted_token_ids),
        "bonus_token_id": result.bonus_token_id,
        "output_token_ids": list(result.output_token_ids),
        "target_argmax": [int(token_id) for token_id in target_argmax],
        "flat_draft_token_ids": list(payload.flat_draft_token_ids),
        "top1_chain_token_ids": list(payload.top1_chain_token_ids),
        "tree_token_ids": list(payload.tree_token_ids),
        "parent_indices": list(payload.parent_indices),
        "node_depths": list(payload.node_depths),
        "walk": trace_rows,
    }
    if _ddtree_verify_row_trace_enabled():
        trace_payload.update(_verifier_row_trace(payload, compact_logits, result))

    _write_ddtree_trace_event(
        "sampler_verify",
        trace_payload,
    )
    if not debug_enabled:
        return

    logger.info(
        "DFLASH_DDTREE_DEBUG verify req=%s accepted=%s output=%s "
        "flat=%s top1=%s tree_head=%s walk=%s",
        req_id,
        result.accepted_node_indices,
        result.output_token_ids,
        payload.flat_draft_token_ids,
        payload.top1_chain_token_ids,
        payload.tree_token_ids[: min(16, len(payload.tree_token_ids))],
        rows,
    )


def _debug_log_stochastic_verification(
    *,
    req_id: str,
    payload: DDTreeDraftPayload,
    compact_logits: torch.Tensor,
    target_tokens: Sequence[int],
    result: DDTreeVerificationResult,
) -> None:
    debug_enabled = _ddtree_debug_enabled()
    if not debug_enabled and not _ddtree_trace_path():
        return

    tree = tree_from_payload(payload)
    cursor = 0
    rows: list[str] = []
    trace_rows: list[dict[str, object]] = []
    while True:
        if cursor >= len(target_tokens):
            rows.append(f"cursor={cursor} missing_sampled_row")
            trace_rows.append({"row": cursor, "missing_sampled_row": True})
            break
        next_token = int(target_tokens[cursor])
        children = tree.child_by_token(cursor)
        child_index = children.get(next_token)
        depth = tree.nodes[cursor].depth + 1
        rank = _candidate_rank(payload, depth=depth, token_id=next_token)
        child_tokens = tuple(children.keys())[:8]
        trace_rows.append(
            {
                "row": cursor,
                "parent": tree.nodes[cursor].parent_index,
                "depth": tree.nodes[cursor].depth,
                "sampled_target": next_token,
                "matched_child": child_index,
                "topk_rank": rank,
                "child_tokens_head": list(child_tokens),
            }
        )
        rows.append(
            f"row={cursor} parent={tree.nodes[cursor].parent_index} "
            f"depth={tree.nodes[cursor].depth} sampled={next_token} "
            f"child={child_index} topk_rank={rank} "
            f"child_tokens={child_tokens}"
        )
        if child_index is None:
            break
        cursor = child_index

    trace_payload: dict[str, object] = {
        "req_id": req_id,
        "accepted_node_indices": list(result.accepted_node_indices),
        "accepted_token_ids": list(result.accepted_token_ids),
        "bonus_token_id": result.bonus_token_id,
        "output_token_ids": list(result.output_token_ids),
        "sampled_target_tokens": [int(token_id) for token_id in target_tokens],
        "flat_draft_token_ids": list(payload.flat_draft_token_ids),
        "top1_chain_token_ids": list(payload.top1_chain_token_ids),
        "tree_token_ids": list(payload.tree_token_ids),
        "parent_indices": list(payload.parent_indices),
        "node_depths": list(payload.node_depths),
        "tree_budget": int(payload.budget),
        "tree_top_k": int(payload.top_k),
        "chain_seed": bool(payload.chain_seed),
        "tree_mode": payload.tree_mode,
        "walk": trace_rows,
    }
    if _ddtree_verify_row_trace_enabled():
        trace_payload.update(_verifier_row_trace(payload, compact_logits, result))

    _write_ddtree_trace_event("stochastic_sampler_verify", trace_payload)
    if not debug_enabled:
        return

    logger.info(
        "DFLASH_DDTREE_DEBUG stochastic verify req=%s accepted=%s "
        "output=%s flat=%s top1=%s tree_head=%s walk=%s",
        req_id,
        result.accepted_node_indices,
        result.output_token_ids,
        payload.flat_draft_token_ids,
        payload.top1_chain_token_ids,
        payload.tree_token_ids[: min(16, len(payload.tree_token_ids))],
        rows,
    )


@dataclass(frozen=True)
class DDTreeGreedySamplerResult:
    sampler_output: SamplerOutput
    verification_results: tuple[DDTreeVerificationResult, ...]


def _compact_top_token_ids_1d(compact_top_token_ids: torch.Tensor) -> torch.Tensor:
    if compact_top_token_ids.ndim == 1:
        return compact_top_token_ids
    if compact_top_token_ids.ndim == 2:
        if compact_top_token_ids.shape[1] < 1:
            raise ValueError("compact_top_token_ids must have at least one column")
        return compact_top_token_ids[:, 0]
    raise ValueError("compact_top_token_ids must have shape [rows] or [rows, k]")


def _padded_int_tensor(
    rows: Sequence[tuple[int, ...]],
    *,
    device: torch.device,
) -> torch.Tensor:
    max_len = max((len(row) for row in rows), default=1)
    if not rows:
        return torch.full(
            (0, max_len),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=device,
        )

    padded_rows = [
        tuple(row) + (PLACEHOLDER_TOKEN_ID,) * (max_len - len(row))
        for row in rows
    ]
    return torch.tensor(padded_rows, dtype=torch.int32, device=device)


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


@triton.jit
def _ddtree_single_top_token_sampler_kernel(
    compact_top_tokens,
    compact_input_ids,
    parent_ids,
    sampled_token_ids,
    accepted_node_indices,
    MAX_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = offsets < MAX_ROWS
    parents = tl.load(parent_ids + offsets, mask=row_mask, other=0)
    parents = tl.where(parents < 0, 0, parents)
    tree_tokens = tl.load(compact_input_ids + offsets, mask=row_mask, other=-1)

    tl.store(sampled_token_ids + offsets, -1, mask=row_mask)
    tl.store(accepted_node_indices + offsets, -1, mask=row_mask)
    tl.store(accepted_node_indices, 0)

    cursor = tl.full((), 0, tl.int32)
    active = tl.full((), True, tl.int1)
    for output_pos in tl.static_range(0, MAX_ROWS):
        next_token = tl.load(compact_top_tokens + cursor)
        tl.store(
            sampled_token_ids + output_pos,
            tl.where(active, next_token, -1).to(tl.int32),
        )
        if output_pos + 1 < MAX_ROWS:
            matches = (
                active
                & (offsets > 0)
                & row_mask
                & (parents == cursor)
                & (tree_tokens == next_token)
            )
            child = tl.max(tl.where(matches, offsets, 0), axis=0).to(tl.int32)
            matched = active & (child > 0)
            tl.store(
                accepted_node_indices + output_pos + 1,
                tl.where(matched, child, -1),
            )
            cursor = tl.where(matched, child, cursor)
            active = matched


def warmup_ddtree_single_top_token_sampler(
    *,
    device: torch.device,
    max_rows: int,
) -> bool:
    """JIT-compile the single-request DDTree top-token sampler."""

    if device.type != "cuda":
        return False
    if os.getenv("VLLM_DFLASH_DDTREE_TRITON_SAMPLER", "1") == "0":
        return False
    max_rows = int(max_rows)
    if max_rows <= 0:
        return False

    compact_top_tokens = torch.zeros(max_rows, dtype=torch.int64, device=device)
    compact_input_ids = torch.ones(max_rows, dtype=torch.int32, device=device)
    parent_ids = torch.zeros((1, max_rows), dtype=torch.int32, device=device)
    sampled_token_ids = torch.empty((1, max_rows), dtype=torch.int32, device=device)
    accepted_node_indices = torch.empty(
        (1, max_rows),
        dtype=torch.int32,
        device=device,
    )
    _ddtree_single_top_token_sampler_kernel[(1,)](
        compact_top_tokens,
        compact_input_ids,
        parent_ids,
        sampled_token_ids,
        accepted_node_indices,
        MAX_ROWS=max_rows,
        BLOCK_SIZE=_next_power_of_2(max_rows),
    )
    torch.cuda.synchronize(device)
    return True


def greedy_sample_ddtree_payloads(
    *,
    req_ids: Sequence[str],
    payload_by_req_id: Mapping[str, DDTreeDraftPayload],
    compact_logits: torch.Tensor,
    num_draft_tokens: Sequence[int],
) -> DDTreeGreedySamplerResult | None:
    """Sample accepted DDTree paths from compact target logits.

    ``compact_logits`` is expected to use the same per-request row order as
    flat speculative decode today: root row followed by one row per scheduled
    draft/tree node. Until the verifier input path is tree-shaped, callers must
    only pass payloads whose tree node count equals the scheduled flat draft
    length.
    """

    if compact_logits.ndim != 2:
        raise ValueError("compact_logits must have shape [rows, vocab]")
    if len(req_ids) != len(num_draft_tokens):
        raise ValueError("req_ids and num_draft_tokens length mismatch")

    rows_consumed = 0
    output_token_rows: list[tuple[int, ...]] = []
    accepted_node_rows: list[tuple[int, ...]] = []
    results: list[DDTreeVerificationResult] = []
    for req_id, draft_len in zip(req_ids, num_draft_tokens, strict=True):
        num_rows = int(draft_len) + 1
        payload = payload_by_req_id.get(req_id)
        if payload is None or payload.num_tree_nodes != int(draft_len):
            return None
        req_logits = compact_logits[rows_consumed : rows_consumed + num_rows]
        result = greedy_verify_payload_from_compact_logits(
            payload=payload,
            compact_logits=req_logits,
        )
        _debug_log_verification(
            req_id=req_id,
            payload=payload,
            compact_logits=req_logits,
            result=result,
        )
        if _ddtree_debug_enabled():
            logger.info(
                "DFLASH_DDTREE_DEBUG sampler req=%s draft_len=%d "
                "accepted_nodes=%s accepted_tokens=%s bonus=%s output=%s",
                req_id,
                int(draft_len),
                result.accepted_node_indices,
                result.accepted_token_ids,
                result.bonus_token_id,
                result.output_token_ids,
            )
        results.append(result)
        output_token_rows.append(result.output_token_ids)
        accepted_node_rows.append(result.accepted_node_indices)
        rows_consumed += num_rows

    if rows_consumed != compact_logits.shape[0]:
        raise ValueError(
            f"compact_logits row mismatch: consumed {rows_consumed}, "
            f"got {compact_logits.shape[0]}"
        )

    sampled_token_ids = _padded_int_tensor(
        output_token_rows,
        device=compact_logits.device,
    )
    accepted_node_indices = _padded_int_tensor(
        accepted_node_rows,
        device=compact_logits.device,
    )
    return DDTreeGreedySamplerResult(
        sampler_output=SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            ddtree_accepted_node_indices=accepted_node_indices,
        ),
        verification_results=tuple(results),
    )


def _expand_request_param_to_rows(
    param: torch.Tensor | None,
    row_req_indices: torch.Tensor,
) -> torch.Tensor | None:
    if param is None:
        return None
    return param.index_select(0, row_req_indices.to(device=param.device))


def _expand_generators_to_rows(
    generators: Mapping[int, torch.Generator],
    row_req_indices: Sequence[int],
) -> dict[int, torch.Generator]:
    if not generators:
        return {}
    row_generators: dict[int, torch.Generator] = {}
    for row_index, req_index in enumerate(row_req_indices):
        generator = generators.get(int(req_index))
        if generator is not None:
            row_generators[row_index] = generator
    return row_generators


def _sample_compact_target_tokens(
    *,
    compact_logits: torch.Tensor,
    row_req_indices: Sequence[int],
    temperature: torch.Tensor | None,
    top_k: torch.Tensor | None,
    top_k_cpu: Sequence[int] | None,
    top_p: torch.Tensor | None,
    generators: Mapping[int, torch.Generator],
) -> torch.Tensor:
    if temperature is None:
        return compact_logits.argmax(dim=-1)

    req_indices = torch.tensor(
        row_req_indices,
        dtype=torch.long,
        device=compact_logits.device,
    )
    row_temperature = _expand_request_param_to_rows(temperature, req_indices)
    assert row_temperature is not None

    greedy_mask = row_temperature < _SAMPLING_EPS
    greedy_tokens = None
    if greedy_mask.any():
        greedy_tokens = compact_logits.argmax(dim=-1)

    safe_temperature = torch.where(greedy_mask, 1.0, row_temperature)

    row_top_k = _expand_request_param_to_rows(top_k, req_indices)
    row_top_p = _expand_request_param_to_rows(top_p, req_indices)
    row_top_k_cpu = (
        None
        if top_k_cpu is None
        else [int(top_k_cpu[int(req_index)]) for req_index in row_req_indices]
    )
    topk_first_tokens = _sample_compact_target_tokens_topk_first(
        compact_logits=compact_logits,
        safe_temperature=safe_temperature,
        greedy_mask=greedy_mask,
        greedy_tokens=greedy_tokens,
        row_top_k=row_top_k,
        row_top_k_cpu=row_top_k_cpu,
        row_top_p=row_top_p,
        row_generators=_expand_generators_to_rows(generators, row_req_indices),
    )
    if topk_first_tokens is not None:
        return topk_first_tokens

    sample_logits = compact_logits.float().clone()
    sample_logits.div_(safe_temperature.view(-1, 1))
    sample_logits = apply_top_k_top_p(sample_logits, row_top_k, row_top_p)
    probs = sample_logits.softmax(dim=-1, dtype=torch.float32)
    random_tokens = random_sample(
        probs,
        _expand_generators_to_rows(generators, row_req_indices),
    )
    if greedy_tokens is None:
        return random_tokens
    return torch.where(greedy_mask, greedy_tokens, random_tokens)


def _sample_compact_target_tokens_topk_first(
    *,
    compact_logits: torch.Tensor,
    safe_temperature: torch.Tensor,
    greedy_mask: torch.Tensor,
    greedy_tokens: torch.Tensor | None,
    row_top_k: torch.Tensor | None,
    row_top_k_cpu: Sequence[int] | None,
    row_top_p: torch.Tensor | None,
    row_generators: Mapping[int, torch.Generator],
) -> torch.Tensor | None:
    """Sample from a small top-k slice without materializing vocab-sized probs."""

    if row_top_k is None or row_top_k_cpu is None:
        return None
    if not row_top_k_cpu or row_top_k.numel() == 0:
        return None

    max_top_k = max(row_top_k_cpu)

    if max_top_k <= 0 or max_top_k > _MAX_TOPK_FIRST_STOCHASTIC_K:
        return None
    if max_top_k >= compact_logits.shape[-1]:
        return None

    topk_values, topk_indices = torch.topk(
        compact_logits,
        k=max_top_k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    return _sample_target_tokens_from_topk(
        topk_values=topk_values,
        topk_indices=topk_indices,
        safe_temperature=safe_temperature,
        greedy_mask=greedy_mask,
        greedy_tokens=greedy_tokens,
        row_top_k=row_top_k,
        row_top_k_cpu=row_top_k_cpu,
        row_top_p=row_top_p,
        row_generators=row_generators,
    )


def _sample_target_tokens_from_topk(
    *,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    safe_temperature: torch.Tensor,
    greedy_mask: torch.Tensor,
    greedy_tokens: torch.Tensor | None,
    row_top_k: torch.Tensor | None,
    row_top_k_cpu: Sequence[int] | None,
    row_top_p: torch.Tensor | None,
    row_generators: Mapping[int, torch.Generator],
) -> torch.Tensor:
    max_top_k = int(topk_values.shape[-1])
    if greedy_tokens is None:
        greedy_tokens = topk_indices[:, 0]

    topk_values = topk_values.float()
    topk_values.div_(safe_temperature.view(-1, 1))

    if (
        row_top_k is not None
        and row_top_k_cpu is not None
        and any(k != max_top_k for k in row_top_k_cpu)
    ):
        positions = torch.arange(
            max_top_k,
            dtype=row_top_k.dtype,
            device=row_top_k.device,
        )
        topk_values.masked_fill_(
            positions.unsqueeze(0) >= row_top_k.view(-1, 1),
            -float("inf"),
        )

    if row_top_p is not None:
        probs_for_p = topk_values.softmax(dim=-1, dtype=torch.float32)
        remove_mask = probs_for_p.cumsum(dim=-1) > row_top_p.view(-1, 1)
        shifted_remove_mask = remove_mask.clone()
        shifted_remove_mask[:, 1:] = remove_mask[:, :-1]
        shifted_remove_mask[:, 0] = False
        topk_values.masked_fill_(shifted_remove_mask, -float("inf"))

    probs = topk_values.softmax(dim=-1, dtype=torch.float32)
    sampled_offsets = random_sample(probs, dict(row_generators))
    random_tokens = topk_indices.gather(1, sampled_offsets.view(-1, 1)).view(-1)
    if greedy_tokens is None:
        return random_tokens
    return torch.where(greedy_mask, greedy_tokens, random_tokens)


def _sample_compact_topk_target_tokens(
    *,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    row_req_indices: Sequence[int],
    temperature: torch.Tensor | None,
    top_k: torch.Tensor | None,
    top_k_cpu: Sequence[int] | None,
    top_p: torch.Tensor | None,
    generators: Mapping[int, torch.Generator],
) -> torch.Tensor:
    if temperature is None:
        return topk_indices[:, 0]

    req_indices = torch.tensor(
        row_req_indices,
        dtype=torch.long,
        device=topk_values.device,
    )
    row_temperature = _expand_request_param_to_rows(temperature, req_indices)
    assert row_temperature is not None
    greedy_mask = row_temperature < _SAMPLING_EPS
    safe_temperature = torch.where(greedy_mask, 1.0, row_temperature)
    greedy_tokens = topk_indices[:, 0] if greedy_mask.any() else None
    row_top_k = _expand_request_param_to_rows(top_k, req_indices)
    row_top_k_cpu = (
        None
        if top_k_cpu is None
        else [int(top_k_cpu[int(req_index)]) for req_index in row_req_indices]
    )
    row_top_p = _expand_request_param_to_rows(top_p, req_indices)
    return _sample_target_tokens_from_topk(
        topk_values=topk_values,
        topk_indices=topk_indices,
        safe_temperature=safe_temperature,
        greedy_mask=greedy_mask,
        greedy_tokens=greedy_tokens,
        row_top_k=row_top_k,
        row_top_k_cpu=row_top_k_cpu,
        row_top_p=row_top_p,
        row_generators=_expand_generators_to_rows(generators, row_req_indices),
    )


def stochastic_sample_ddtree_payloads(
    *,
    req_ids: Sequence[str],
    payload_by_req_id: Mapping[str, DDTreeDraftPayload],
    compact_logits: torch.Tensor,
    num_draft_tokens: Sequence[int],
    temperature: torch.Tensor | None,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    generators: Mapping[int, torch.Generator],
    top_k_cpu: Sequence[int] | None = None,
) -> DDTreeGreedySamplerResult | None:
    """Sample target verifier rows, then follow the DDTree official path.

    This matches the official stochastic DDTree algorithm: target logits
    produce one sampled posterior token for every verifier row, the sampler
    follows matching child edges, and the first non-child target sample is the
    bonus token. The draft distribution is used only to build the tree; it does
    not participate in rejection probabilities.
    """

    if compact_logits.ndim != 2:
        raise ValueError("compact_logits must have shape [rows, vocab]")
    if len(req_ids) != len(num_draft_tokens):
        raise ValueError("req_ids and num_draft_tokens length mismatch")

    profile_enabled = _ddtree_worker_profile_enabled()
    profile_t0 = time.perf_counter() if profile_enabled else 0.0

    rows_consumed = 0
    row_req_indices: list[int] = []
    payload_rows: list[tuple[str, int, int, int, DDTreeDraftPayload]] = []
    validate_t0 = time.perf_counter() if profile_enabled else 0.0
    for req_index, (req_id, draft_len) in enumerate(
        zip(req_ids, num_draft_tokens, strict=True)
    ):
        num_rows = int(draft_len) + 1
        payload = payload_by_req_id.get(req_id)
        if payload is None or payload.num_tree_nodes != int(draft_len):
            return None
        row_start = rows_consumed
        rows_consumed += num_rows
        row_req_indices.extend([req_index] * num_rows)
        payload_rows.append(
            (req_id, int(draft_len), row_start, rows_consumed, payload)
        )

    if rows_consumed != compact_logits.shape[0]:
        raise ValueError(
            f"compact_logits row mismatch: consumed {rows_consumed}, "
            f"got {compact_logits.shape[0]}"
        )
    validate_ms = (
        (time.perf_counter() - validate_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    sample_t0 = time.perf_counter() if profile_enabled else 0.0
    target_tokens_tensor = _sample_compact_target_tokens(
        compact_logits=compact_logits,
        row_req_indices=row_req_indices,
        temperature=temperature,
        top_k=top_k,
        top_k_cpu=top_k_cpu,
        top_p=top_p,
        generators=generators,
    )
    sample_ms = (
        (time.perf_counter() - sample_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    copy_t0 = time.perf_counter() if profile_enabled else 0.0
    target_tokens = target_tokens_tensor.detach().cpu().tolist()
    copy_ms = (
        (time.perf_counter() - copy_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    output_token_rows: list[tuple[int, ...]] = []
    accepted_node_rows: list[tuple[int, ...]] = []
    results: list[DDTreeVerificationResult] = []
    verify_t0 = time.perf_counter() if profile_enabled else 0.0
    for req_id, draft_len, row_start, row_end, payload in payload_rows:
        req_target_tokens = target_tokens[row_start:row_end]
        result = greedy_verify_payload_from_target_tokens(
            payload=payload,
            target_tokens=req_target_tokens,
        )
        _debug_log_stochastic_verification(
            req_id=req_id,
            payload=payload,
            compact_logits=compact_logits[row_start:row_end],
            target_tokens=req_target_tokens,
            result=result,
        )
        if _ddtree_debug_enabled():
            logger.info(
                "DFLASH_DDTREE_DEBUG stochastic_sampler req=%s draft_len=%d "
                "accepted_nodes=%s accepted_tokens=%s bonus=%s output=%s",
                req_id,
                int(draft_len),
                result.accepted_node_indices,
                result.accepted_token_ids,
                result.bonus_token_id,
                result.output_token_ids,
            )
        results.append(result)
        output_token_rows.append(result.output_token_ids)
        accepted_node_rows.append(result.accepted_node_indices)
    verify_ms = (
        (time.perf_counter() - verify_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    tensor_t0 = time.perf_counter() if profile_enabled else 0.0
    sampled_token_ids = _padded_int_tensor(
        output_token_rows,
        device=compact_logits.device,
    )
    accepted_node_indices = _padded_int_tensor(
        accepted_node_rows,
        device=compact_logits.device,
    )
    tensor_ms = (
        (time.perf_counter() - tensor_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    if profile_enabled:
        logger.info(
            "DFLASH_DDTREE_WORKER_PROFILE stochastic_sampler "
            "total_ms=%.3f validate_ms=%.3f sample_ms=%.3f "
            "d2h_sample_tokens_ms=%.3f verify_walk_ms=%.3f "
            "output_tensor_ms=%.3f rows=%d reqs=%d device=%s dtype=%s "
            "top_k=%s top_p=%s generators=%d",
            (time.perf_counter() - profile_t0) * 1000.0,
            validate_ms,
            sample_ms,
            copy_ms,
            verify_ms,
            tensor_ms,
            int(compact_logits.shape[0]),
            len(req_ids),
            compact_logits.device,
            compact_logits.dtype,
            top_k is not None,
            top_p is not None,
            len(generators),
        )

    return DDTreeGreedySamplerResult(
        sampler_output=SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            ddtree_accepted_node_indices=accepted_node_indices,
        ),
        verification_results=tuple(results),
    )


def stochastic_sample_ddtree_payloads_from_topk(
    *,
    req_ids: Sequence[str],
    payload_by_req_id: Mapping[str, DDTreeDraftPayload],
    target_topk_token_ids: torch.Tensor,
    target_topk_logits: torch.Tensor,
    num_draft_tokens: Sequence[int],
    temperature: torch.Tensor | None,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    generators: Mapping[int, torch.Generator],
    top_k_cpu: Sequence[int] | None = None,
) -> DDTreeGreedySamplerResult | None:
    """Sample DDTree verifier rows from already-merged target top-k logits."""

    if target_topk_token_ids.ndim != 2 or target_topk_logits.ndim != 2:
        raise ValueError("target top-k tensors must have shape [rows, top_k]")
    if target_topk_token_ids.shape != target_topk_logits.shape:
        raise ValueError("target top-k ids/logits shape mismatch")
    if len(req_ids) != len(num_draft_tokens):
        raise ValueError("req_ids and num_draft_tokens length mismatch")

    profile_enabled = _ddtree_worker_profile_enabled()
    profile_t0 = time.perf_counter() if profile_enabled else 0.0

    rows_consumed = 0
    row_req_indices: list[int] = []
    payload_rows: list[tuple[str, int, int, int, DDTreeDraftPayload]] = []
    validate_t0 = time.perf_counter() if profile_enabled else 0.0
    for req_index, (req_id, draft_len) in enumerate(
        zip(req_ids, num_draft_tokens, strict=True)
    ):
        num_rows = int(draft_len) + 1
        payload = payload_by_req_id.get(req_id)
        if payload is None or payload.num_tree_nodes != int(draft_len):
            return None
        row_start = rows_consumed
        rows_consumed += num_rows
        row_req_indices.extend([req_index] * num_rows)
        payload_rows.append(
            (req_id, int(draft_len), row_start, rows_consumed, payload)
        )

    if rows_consumed != target_topk_logits.shape[0]:
        raise ValueError(
            f"target top-k row mismatch: consumed {rows_consumed}, "
            f"got {target_topk_logits.shape[0]}"
        )
    validate_ms = (
        (time.perf_counter() - validate_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    sample_t0 = time.perf_counter() if profile_enabled else 0.0
    target_tokens_tensor = _sample_compact_topk_target_tokens(
        topk_values=target_topk_logits,
        topk_indices=target_topk_token_ids,
        row_req_indices=row_req_indices,
        temperature=temperature,
        top_k=top_k,
        top_k_cpu=top_k_cpu,
        top_p=top_p,
        generators=generators,
    )
    sample_ms = (
        (time.perf_counter() - sample_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    copy_t0 = time.perf_counter() if profile_enabled else 0.0
    target_tokens = target_tokens_tensor.detach().cpu().tolist()
    copy_ms = (
        (time.perf_counter() - copy_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    output_token_rows: list[tuple[int, ...]] = []
    accepted_node_rows: list[tuple[int, ...]] = []
    results: list[DDTreeVerificationResult] = []
    verify_t0 = time.perf_counter() if profile_enabled else 0.0
    for req_id, draft_len, row_start, row_end, payload in payload_rows:
        req_target_tokens = target_tokens[row_start:row_end]
        result = greedy_verify_payload_from_target_tokens(
            payload=payload,
            target_tokens=req_target_tokens,
        )
        if _ddtree_debug_enabled():
            logger.info(
                "DFLASH_DDTREE_DEBUG stochastic_topk_sampler req=%s "
                "draft_len=%d accepted_nodes=%s accepted_tokens=%s "
                "bonus=%s output=%s",
                req_id,
                int(draft_len),
                result.accepted_node_indices,
                result.accepted_token_ids,
                result.bonus_token_id,
                result.output_token_ids,
            )
        results.append(result)
        output_token_rows.append(result.output_token_ids)
        accepted_node_rows.append(result.accepted_node_indices)
    verify_ms = (
        (time.perf_counter() - verify_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    tensor_t0 = time.perf_counter() if profile_enabled else 0.0
    sampled_token_ids = _padded_int_tensor(
        output_token_rows,
        device=target_topk_logits.device,
    )
    accepted_node_indices = _padded_int_tensor(
        accepted_node_rows,
        device=target_topk_logits.device,
    )
    tensor_ms = (
        (time.perf_counter() - tensor_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    if profile_enabled:
        logger.info(
            "DFLASH_DDTREE_WORKER_PROFILE stochastic_topk_sampler "
            "total_ms=%.3f validate_ms=%.3f sample_ms=%.3f "
            "d2h_sample_tokens_ms=%.3f verify_walk_ms=%.3f "
            "output_tensor_ms=%.3f rows=%d reqs=%d top_k_width=%d "
            "device=%s dtype=%s top_k=%s top_p=%s generators=%d",
            (time.perf_counter() - profile_t0) * 1000.0,
            validate_ms,
            sample_ms,
            copy_ms,
            verify_ms,
            tensor_ms,
            int(target_topk_logits.shape[0]),
            len(req_ids),
            int(target_topk_logits.shape[1]),
            target_topk_logits.device,
            target_topk_logits.dtype,
            top_k is not None,
            top_p is not None,
            len(generators),
        )

    return DDTreeGreedySamplerResult(
        sampler_output=SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            ddtree_accepted_node_indices=accepted_node_indices,
        ),
        verification_results=tuple(results),
    )


def greedy_sample_ddtree_payloads_from_top_tokens(
    *,
    req_ids: Sequence[str],
    payload_by_req_id: Mapping[str, DDTreeDraftPayload],
    compact_top_tokens: torch.Tensor,
    num_draft_tokens: Sequence[int],
) -> DDTreeGreedySamplerResult | None:
    """Sample accepted DDTree paths from compact verifier-row top tokens."""

    profile_enabled = _ddtree_worker_profile_enabled()
    profile_t0 = time.perf_counter() if profile_enabled else 0.0
    compact_top_tokens = _compact_top_token_ids_1d(compact_top_tokens)
    compact_ms = (
        (time.perf_counter() - profile_t0) * 1000.0
        if profile_enabled
        else 0.0
    )
    if len(req_ids) != len(num_draft_tokens):
        raise ValueError("req_ids and num_draft_tokens length mismatch")

    rows_consumed = 0
    payload_rows: list[tuple[str, int, int, int, DDTreeDraftPayload]] = []
    validate_t0 = time.perf_counter() if profile_enabled else 0.0
    for req_id, draft_len in zip(req_ids, num_draft_tokens, strict=True):
        num_rows = int(draft_len) + 1
        payload = payload_by_req_id.get(req_id)
        if payload is None or payload.num_tree_nodes != int(draft_len):
            return None
        row_start = rows_consumed
        rows_consumed += num_rows
        payload_rows.append(
            (req_id, int(draft_len), row_start, rows_consumed, payload)
        )

    if rows_consumed != compact_top_tokens.shape[0]:
        raise ValueError(
            f"compact_top_tokens row mismatch: consumed {rows_consumed}, "
            f"got {compact_top_tokens.shape[0]}"
        )
    validate_ms = (
        (time.perf_counter() - validate_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    copy_t0 = time.perf_counter() if profile_enabled else 0.0
    target_tokens = compact_top_tokens.detach().cpu().tolist()
    copy_ms = (
        (time.perf_counter() - copy_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    output_token_rows: list[tuple[int, ...]] = []
    accepted_node_rows: list[tuple[int, ...]] = []
    results: list[DDTreeVerificationResult] = []
    verify_t0 = time.perf_counter() if profile_enabled else 0.0
    for req_id, draft_len, row_start, row_end, payload in payload_rows:
        req_top_tokens = target_tokens[row_start:row_end]
        result = greedy_verify_payload_from_target_tokens(
            payload=payload,
            target_tokens=req_top_tokens,
        )
        if _ddtree_debug_enabled():
            logger.info(
                "DFLASH_DDTREE_DEBUG sampler_top_tokens req=%s draft_len=%d "
                "accepted_nodes=%s accepted_tokens=%s bonus=%s output=%s",
                req_id,
                int(draft_len),
                result.accepted_node_indices,
                result.accepted_token_ids,
                result.bonus_token_id,
                result.output_token_ids,
            )
        results.append(result)
        output_token_rows.append(result.output_token_ids)
        accepted_node_rows.append(result.accepted_node_indices)
    verify_ms = (
        (time.perf_counter() - verify_t0) * 1000.0
        if profile_enabled
        else 0.0
    )

    tensor_t0 = time.perf_counter() if profile_enabled else 0.0
    sampled_token_ids = _padded_int_tensor(
        output_token_rows,
        device=compact_top_tokens.device,
    )
    accepted_node_indices = _padded_int_tensor(
        accepted_node_rows,
        device=compact_top_tokens.device,
    )
    tensor_ms = (
        (time.perf_counter() - tensor_t0) * 1000.0
        if profile_enabled
        else 0.0
    )
    if profile_enabled:
        logger.info(
            "DFLASH_DDTREE_WORKER_PROFILE top_token_sampler "
            "total_ms=%.3f compact_ms=%.3f validate_ms=%.3f "
            "d2h_top_tokens_ms=%.3f verify_walk_ms=%.3f "
            "output_tensor_ms=%.3f rows=%d reqs=%d device=%s",
            (time.perf_counter() - profile_t0) * 1000.0,
            compact_ms,
            validate_ms,
            copy_ms,
            verify_ms,
            tensor_ms,
            int(compact_top_tokens.shape[0]),
            len(req_ids),
            compact_top_tokens.device,
        )
    return DDTreeGreedySamplerResult(
        sampler_output=SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            ddtree_accepted_node_indices=accepted_node_indices,
        ),
        verification_results=tuple(results),
    )


def greedy_sample_ddtree_payloads_from_top_tokens_gpu(
    *,
    compact_top_tokens: torch.Tensor,
    compact_input_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    num_draft_tokens: Sequence[int],
) -> SamplerOutput | None:
    """Device-side greedy DDTree walk for the top-token verifier path.

    ``parent_ids`` uses the root-plus-tree compact row convention: column 0 is
    the synthetic root, and non-root columns 1..N correspond to compact verifier
    rows 1..N. ``compact_input_ids`` follows the same compact row order.
    """

    if compact_top_tokens.device.type != "cuda":
        return None
    if compact_input_ids.device != compact_top_tokens.device:
        return None
    if parent_ids.device != compact_top_tokens.device:
        return None

    compact_top_tokens = _compact_top_token_ids_1d(compact_top_tokens)
    if compact_input_ids.ndim != 1:
        raise ValueError("compact_input_ids must have shape [rows]")
    if compact_input_ids.shape[0] != compact_top_tokens.shape[0]:
        raise ValueError(
            "compact_input_ids and compact_top_tokens row mismatch: "
            f"{compact_input_ids.shape[0]} != {compact_top_tokens.shape[0]}"
        )

    num_reqs = len(num_draft_tokens)
    if num_reqs <= 0:
        return SamplerOutput(
            sampled_token_ids=torch.full(
                (0, 1),
                PLACEHOLDER_TOKEN_ID,
                dtype=torch.int32,
                device=compact_top_tokens.device,
            ),
            logprobs_tensors=None,
            ddtree_accepted_node_indices=torch.full(
                (0, 1),
                PLACEHOLDER_TOKEN_ID,
                dtype=torch.int32,
                device=compact_top_tokens.device,
            ),
        )
    if parent_ids.ndim != 2 or parent_ids.shape[0] < num_reqs:
        raise ValueError("parent_ids must have shape [reqs, tree_slots]")

    draft_lens_cpu = [int(draft_len) for draft_len in num_draft_tokens]
    if any(draft_len < 0 for draft_len in draft_lens_cpu):
        raise ValueError("num_draft_tokens must be non-negative")
    max_draft_len = max(draft_lens_cpu, default=0)
    max_rows = max_draft_len + 1
    expected_rows = sum(draft_len + 1 for draft_len in draft_lens_cpu)
    if compact_top_tokens.shape[0] != expected_rows:
        raise ValueError(
            f"compact_top_tokens row mismatch: expected {expected_rows}, "
            f"got {compact_top_tokens.shape[0]}"
        )
    if parent_ids.shape[1] < max_rows:
        raise ValueError(
            f"parent_ids width {parent_ids.shape[1]} is smaller than "
            f"required compact rows {max_rows}"
        )

    profile_enabled = _ddtree_worker_profile_enabled()
    profile_t0 = time.perf_counter() if profile_enabled else 0.0
    device = compact_top_tokens.device

    if num_reqs == 1:
        if os.getenv("VLLM_DFLASH_DDTREE_TRITON_SAMPLER", "1") != "0":
            sampled_token_ids = torch.empty(
                (1, max_rows),
                dtype=torch.int32,
                device=device,
            )
            accepted_node_indices = torch.empty(
                (1, max_rows),
                dtype=torch.int32,
                device=device,
            )
            _ddtree_single_top_token_sampler_kernel[(1,)](
                compact_top_tokens,
                compact_input_ids,
                parent_ids,
                sampled_token_ids,
                accepted_node_indices,
                MAX_ROWS=max_rows,
                BLOCK_SIZE=_next_power_of_2(max_rows),
            )
            if profile_enabled:
                logger.info(
                    "DFLASH_DDTREE_WORKER_PROFILE gpu_top_token_sampler "
                    "enqueue_wall_ms=%.3f rows=%d reqs=%d max_rows=%d "
                    "device=%s mode=single_triton",
                    (time.perf_counter() - profile_t0) * 1000.0,
                    int(compact_top_tokens.shape[0]),
                    num_reqs,
                    max_rows,
                    device,
                )
            return SamplerOutput(
                sampled_token_ids=sampled_token_ids,
                logprobs_tensors=None,
                ddtree_accepted_node_indices=accepted_node_indices,
            )

        parents = parent_ids[0, :max_rows]
        if parents.dtype != torch.int32:
            parents = parents.to(torch.int32)
        parents = torch.where(parents < 0, torch.zeros_like(parents), parents)
        top_tokens = compact_top_tokens[:max_rows].to(torch.int64)
        tree_tokens = compact_input_ids[:max_rows].to(torch.int64)
        rows_i32 = torch.arange(max_rows, dtype=torch.int32, device=device)
        child_candidate = rows_i32 > 0
        sampled_token_ids = torch.empty(
            (1, max_rows),
            dtype=torch.int32,
            device=device,
        )
        sampled_token_ids.fill_(PLACEHOLDER_TOKEN_ID)
        accepted_node_indices = torch.empty(
            (1, max_rows),
            dtype=torch.int32,
            device=device,
        )
        accepted_node_indices.fill_(PLACEHOLDER_TOKEN_ID)
        accepted_node_indices[:, 0] = 0

        cursor = torch.zeros((), dtype=torch.int32, device=device)
        active = torch.ones((), dtype=torch.bool, device=device)
        zero = torch.zeros((), dtype=torch.int32, device=device)
        for output_pos in range(max_rows):
            next_token = top_tokens.gather(0, cursor.to(torch.int64).view(1))[0]
            sampled_token_ids[:, output_pos] = torch.where(
                active,
                next_token.to(torch.int32),
                sampled_token_ids[:, output_pos],
            )
            if output_pos + 1 >= max_rows:
                break
            matches = (
                active
                & child_candidate
                & (parents == cursor)
                & (tree_tokens == next_token)
            )
            child = torch.where(matches, rows_i32, torch.zeros_like(rows_i32)).max()
            matched = active & (child > 0)
            accepted_node_indices[:, output_pos + 1] = torch.where(
                matched,
                child,
                accepted_node_indices[:, output_pos + 1],
            )
            cursor = torch.where(matched, child, cursor)
            active = matched

        if profile_enabled:
            logger.info(
                "DFLASH_DDTREE_WORKER_PROFILE gpu_top_token_sampler "
                "enqueue_wall_ms=%.3f rows=%d reqs=%d max_rows=%d "
                "device=%s mode=single",
                (time.perf_counter() - profile_t0) * 1000.0,
                int(compact_top_tokens.shape[0]),
                num_reqs,
                max_rows,
                device,
            )
        return SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            ddtree_accepted_node_indices=accepted_node_indices,
        )

    row_offsets_cpu: list[int] = []
    row_offset = 0
    for draft_len in draft_lens_cpu:
        row_offsets_cpu.append(row_offset)
        row_offset += draft_len + 1

    draft_lens = torch.tensor(draft_lens_cpu, dtype=torch.int64, device=device)
    row_offsets = torch.tensor(row_offsets_cpu, dtype=torch.int64, device=device)
    rows_long = torch.arange(max_rows, dtype=torch.int64, device=device)
    rows_i32 = rows_long.to(torch.int32)
    valid_rows = rows_long.unsqueeze(0) <= draft_lens.unsqueeze(1)
    compact_rows = row_offsets.unsqueeze(1) + rows_long.unsqueeze(0)
    compact_rows = compact_rows.clamp_max(max(expected_rows - 1, 0))

    top_tokens = compact_top_tokens[compact_rows].to(torch.int64)
    tree_tokens = compact_input_ids[compact_rows].to(torch.int64)
    top_tokens = torch.where(valid_rows, top_tokens, PLACEHOLDER_TOKEN_ID)
    tree_tokens = torch.where(valid_rows, tree_tokens, PLACEHOLDER_TOKEN_ID)

    parents = parent_ids[:num_reqs, :max_rows].to(torch.int32)
    parents = torch.where(parents < 0, torch.zeros_like(parents), parents)
    sampled_token_ids = torch.full(
        (num_reqs, max_rows),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    accepted_node_indices = torch.full(
        (num_reqs, max_rows),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    accepted_node_indices[:, 0] = 0

    cursor = torch.zeros((num_reqs,), dtype=torch.int32, device=device)
    active = torch.ones((num_reqs,), dtype=torch.bool, device=device)
    child_rows = rows_i32.unsqueeze(0).expand(num_reqs, max_rows)
    child_candidate = child_rows > 0

    for output_pos in range(max_rows):
        cursor_long = cursor.to(torch.int64).unsqueeze(1)
        next_tokens = top_tokens.gather(1, cursor_long).squeeze(1)
        sampled_token_ids[:, output_pos] = torch.where(
            active,
            next_tokens.to(torch.int32),
            sampled_token_ids[:, output_pos],
        )
        if output_pos + 1 >= max_rows:
            break

        matches = (
            active.unsqueeze(1)
            & child_candidate
            & valid_rows
            & (parents == cursor.unsqueeze(1))
            & (tree_tokens == next_tokens.unsqueeze(1))
        )
        child = torch.where(matches, child_rows, torch.zeros_like(child_rows)).max(
            dim=1
        ).values
        matched = active & (child > 0)
        accepted_node_indices[:, output_pos + 1] = torch.where(
            matched,
            child,
            accepted_node_indices[:, output_pos + 1],
        )
        cursor = torch.where(matched, child, cursor)
        active = matched

    if profile_enabled:
        logger.info(
            "DFLASH_DDTREE_WORKER_PROFILE gpu_top_token_sampler "
            "enqueue_wall_ms=%.3f rows=%d reqs=%d max_rows=%d device=%s",
            (time.perf_counter() - profile_t0) * 1000.0,
            int(compact_top_tokens.shape[0]),
            num_reqs,
            max_rows,
            device,
        )

    return SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        logprobs_tensors=None,
        ddtree_accepted_node_indices=accepted_node_indices,
    )
