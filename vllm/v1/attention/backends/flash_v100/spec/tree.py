# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 speculative tree metadata owner."""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100.spec.metadata_contracts import metadata_view
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def attach_metadata(
    self: Any,
    attn_metadata: TritonAttentionMetadata,
    *,
    ddtree_parent_ids: torch.Tensor | None,
    ddtree_num_tree_tokens_cpu: torch.Tensor | None,
) -> None:
    flash_metadata = metadata_view(attn_metadata)
    flash_metadata.ddtree_parent_ids = None
    flash_metadata.ddtree_parent_ids_cpu = None
    flash_metadata.ddtree_num_tree_tokens_cpu = None
    flash_metadata.ddtree_seq_lens_restored_for_triton = False
    flash_metadata.ddtree_query_start_loc_restored_for_triton = False
    if ddtree_parent_ids is None:
        return
    if ddtree_num_tree_tokens_cpu is None:
        raise ValueError(
            "ddtree_num_tree_tokens_cpu is required with ddtree_parent_ids"
        )
    if ddtree_parent_ids.ndim != 2:
        raise ValueError("ddtree_parent_ids must have shape [batch, slots]")
    if ddtree_num_tree_tokens_cpu.ndim != 1:
        raise ValueError("ddtree_num_tree_tokens_cpu must be a 1D tensor")
    num_reqs = int(attn_metadata.query_start_loc.numel() - 1)
    if ddtree_parent_ids.shape[0] < num_reqs:
        raise ValueError(
            "ddtree_parent_ids must cover active requests: "
            f"{ddtree_parent_ids.shape[0]} < {num_reqs}"
        )
    if ddtree_num_tree_tokens_cpu.numel() < num_reqs:
        raise ValueError(
            "ddtree_num_tree_tokens_cpu must cover active requests: "
            f"{ddtree_num_tree_tokens_cpu.numel()} < {num_reqs}"
        )
    flash_metadata.ddtree_parent_ids = ddtree_parent_ids
    flash_metadata.ddtree_num_tree_tokens_cpu = ddtree_num_tree_tokens_cpu
    if bool(torch.any(ddtree_num_tree_tokens_cpu[:num_reqs] > 0).item()):
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if seq_lens is not None and seq_lens_cpu is not None:
            if _routing._is_cuda_graph_capturing(seq_lens):
                seq_lens[:num_reqs].copy_(
                    seq_lens_cpu[:num_reqs].to(
                        device=seq_lens.device,
                        dtype=seq_lens.dtype,
                    ),
                    non_blocking=True,
                )
            flash_metadata.ddtree_seq_lens_restored_for_triton = True
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        if query_start_loc is not None and query_start_loc_cpu is not None:
            num_boundaries = num_reqs + 1
            if _routing._is_cuda_graph_capturing(query_start_loc):
                query_start_loc[:num_boundaries].copy_(
                    query_start_loc_cpu[:num_boundaries].to(
                        device=query_start_loc.device,
                        dtype=query_start_loc.dtype,
                    ),
                    non_blocking=True,
                )
            flash_metadata.ddtree_query_start_loc_restored_for_triton = True


_attach_ddtree_metadata = attach_metadata
