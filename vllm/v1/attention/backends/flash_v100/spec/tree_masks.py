# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree-verification visibility and captured metadata contracts."""

from __future__ import annotations

import torch

from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata


def parent_metadata_requires_branch(
    attn_metadata: TritonAttentionMetadata,
    query_start_loc: torch.Tensor,
) -> bool:
    parent_ids = getattr(attn_metadata, "ddtree_parent_ids", None)
    num_tree_tokens_cpu = getattr(attn_metadata, "ddtree_num_tree_tokens_cpu", None)
    if parent_ids is None or num_tree_tokens_cpu is None:
        return False

    num_reqs = min(
        int(parent_ids.shape[0]),
        int(num_tree_tokens_cpu.numel()),
        max(0, len(query_start_loc) - 1),
    )
    if num_reqs <= 0:
        return False
    return bool(torch.any(num_tree_tokens_cpu[:num_reqs] > 0).item())


def triton_seq_lens_match(
    attn_metadata: TritonAttentionMetadata,
    seq_lens: torch.Tensor,
    num_reqs: int,
) -> bool:
    metadata_seq_lens = getattr(attn_metadata, "seq_lens", None)
    if metadata_seq_lens is None:
        return False
    if num_reqs <= 0:
        return True
    if metadata_seq_lens[:num_reqs].data_ptr() == seq_lens[:num_reqs].data_ptr():
        return True
    if _routing.is_cuda_graph_capturing(metadata_seq_lens):
        return bool(
            getattr(attn_metadata, "ddtree_seq_lens_restored_for_triton", False)
        )
    return bool(
        torch.equal(
            metadata_seq_lens[:num_reqs].detach().cpu(),
            seq_lens[:num_reqs].detach().cpu(),
        )
    )


def triton_query_start_loc_match(
    attn_metadata: TritonAttentionMetadata,
    query_start_loc: torch.Tensor,
    num_reqs: int,
) -> bool:
    metadata_query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if metadata_query_start_loc is None:
        return False
    num_boundaries = num_reqs + 1
    if num_boundaries <= 1:
        return True
    if (
        metadata_query_start_loc[:num_boundaries].data_ptr()
        == query_start_loc[:num_boundaries].data_ptr()
    ):
        return True
    if _routing.is_cuda_graph_capturing(metadata_query_start_loc):
        return bool(
            getattr(
                attn_metadata,
                "ddtree_query_start_loc_restored_for_triton",
                False,
            )
        )
    return bool(
        torch.equal(
            metadata_query_start_loc[:num_boundaries].detach().cpu(),
            query_start_loc[:num_boundaries].detach().cpu(),
        )
    )


def triton_parent_ids_for_query(
    parent_ids: torch.Tensor,
    num_tree_tokens_cpu: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    *,
    is_capturing: bool,
) -> torch.Tensor | None:
    if num_tree_tokens_cpu is None or parent_ids.ndim != 2:
        return parent_ids

    max_q_len = int(parent_ids.shape[1])
    num_reqs = min(
        int(parent_ids.shape[0]),
        int(num_tree_tokens_cpu.numel()),
        max(0, len(query_start_loc) - 1),
    )
    if num_reqs <= 0 or max_q_len <= 0:
        return parent_ids

    query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
    rows_needing_causal_parent: list[int] = []
    for req_idx in range(num_reqs):
        q_len = int(query_lens[req_idx].item())
        if q_len <= 0:
            continue
        if q_len > max_q_len:
            return None

        tree_len = int(num_tree_tokens_cpu[req_idx].item())
        if tree_len <= 0:
            rows_needing_causal_parent.append(req_idx)
            continue
        if q_len > tree_len + 1:
            return None

    if not rows_needing_causal_parent:
        return parent_ids
    if is_capturing:
        return None

    causal_parent_row = torch.arange(
        max_q_len,
        device=parent_ids.device,
        dtype=parent_ids.dtype,
    )
    causal_parent_row = torch.clamp(causal_parent_row - 1, min=0)
    triton_parent_ids = parent_ids.clone()
    triton_parent_ids[rows_needing_causal_parent, :] = causal_parent_row
    return triton_parent_ids


def build_visibility_mask(
    *,
    q_len: int,
    seq_len: int,
    prefix_len: int,
    tree_len: int,
    parent_row: torch.Tensor | None,
    device: torch.device,
    window_size: tuple[int, int],
) -> torch.Tensor:
    visible = torch.zeros((q_len, seq_len), dtype=torch.bool, device=device)
    if q_len <= 0 or seq_len <= 0:
        return visible

    for q_offset in range(q_len):
        logical_q_idx = prefix_len + q_offset
        if q_offset == 0 or tree_len <= 0:
            visible[q_offset, : min(logical_q_idx + 1, seq_len)] = True
            continue

        if q_offset > tree_len or parent_row is None:
            visible[q_offset, : min(logical_q_idx + 1, seq_len)] = True
            continue

        visible[q_offset, : min(prefix_len, seq_len)] = True
        if prefix_len < seq_len:
            visible[q_offset, prefix_len] = True
        if logical_q_idx < seq_len:
            visible[q_offset, logical_q_idx] = True

        ancestor = q_offset
        max_slots = int(parent_row.shape[0])
        for _ in range(max_slots):
            if ancestor < 0 or ancestor >= max_slots:
                break
            parent = int(parent_row[ancestor].item())
            parent = 0 if parent < 0 else parent
            parent_pos = prefix_len + parent
            if 0 <= parent_pos < seq_len:
                visible[q_offset, parent_pos] = True
            if parent <= 0:
                break
            ancestor = parent

    left, right = window_size
    if left >= 0 or right >= 0:
        q_pos = torch.arange(q_len, device=device) + prefix_len
        k_pos = torch.arange(seq_len, device=device)
        if left >= 0:
            visible &= k_pos.unsqueeze(0) >= q_pos.unsqueeze(1) - left
        if right >= 0:
            visible &= k_pos.unsqueeze(0) <= q_pos.unsqueeze(1) + right
    return visible


# The facade resolves old names to live public bindings, not copied values.
COMPATIBILITY_ALIASES = {
    "_ddtree_parent_metadata_requires_branch": "parent_metadata_requires_branch",
    "_ddtree_triton_seq_lens_match": "triton_seq_lens_match",
    "_ddtree_triton_query_start_loc_match": "triton_query_start_loc_match",
    "_ddtree_triton_parent_ids_for_query": "triton_parent_ids_for_query",
    "_build_ddtree_visibility_mask": "build_visibility_mask",
}


def parent_ids_cpu(
    attn_metadata: TritonAttentionMetadata,
    adopt,
) -> torch.Tensor | None:
    parent_ids = getattr(attn_metadata, "ddtree_parent_ids", None)
    if parent_ids is None:
        return None
    parent_ids_cpu = getattr(attn_metadata, "ddtree_parent_ids_cpu", None)
    if parent_ids_cpu is None:
        parent_ids_cpu = parent_ids.detach().cpu()
        adopt(attn_metadata).ddtree_parent_ids_cpu = parent_ids_cpu
    return parent_ids_cpu
