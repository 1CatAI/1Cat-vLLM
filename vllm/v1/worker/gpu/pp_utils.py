# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

import torch

from vllm.distributed.parallel_state import get_pp_group


def pp_broadcast(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    max_sample_len: int = 1,
    draft_token_ids: torch.Tensor | None = None,
    max_draft_len: int = 0,
) -> None:
    pp = get_pp_group()
    assert pp.is_last_rank

    assert sampled_token_ids.dtype == torch.int64
    assert sampled_token_ids.ndim == 2
    num_reqs, sample_len = sampled_token_ids.shape
    if sample_len > max_sample_len:
        raise ValueError(
            f"sampled token width {sample_len} exceeds PP receive width "
            f"{max_sample_len}"
        )
    if max_draft_len < 0:
        raise ValueError("max_draft_len must be non-negative")
    if max_draft_len:
        if draft_token_ids is None:
            raise ValueError("draft_token_ids are required when max_draft_len > 0")
        if draft_token_ids.dtype != torch.int64 or draft_token_ids.ndim != 2:
            raise ValueError("draft_token_ids must be a 2D int64 tensor")
        if draft_token_ids.shape[0] != num_reqs:
            raise ValueError("sampled and draft token batch sizes must match")
        draft_len = draft_token_ids.shape[1]
        if draft_len > max_draft_len:
            raise ValueError(
                f"draft token width {draft_len} exceeds PP receive width "
                f"{max_draft_len}"
            )
    else:
        draft_len = 0

    packet_width = max_sample_len + max_draft_len
    if sample_len < max_sample_len or max_draft_len:
        # NCCL collectives require every rank to use the same element count.
        # Keep accepted and next-step draft IDs in one packet so PP speculative
        # decoding adds bytes to the existing collective instead of another
        # latency-bearing collective.
        token_ids = sampled_token_ids.new_full((num_reqs, packet_width), -1)
        token_ids[:, :sample_len].copy_(sampled_token_ids)
        if draft_len:
            token_ids[
                :, max_sample_len : max_sample_len + draft_len
            ].copy_(draft_token_ids)
    else:
        token_ids = sampled_token_ids
    torch.distributed.broadcast(
        token_ids.contiguous(), src=pp.last_rank, group=pp.device_group
    )

    combined = torch.stack((num_sampled, num_rejected), dim=0)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)


def pp_receive(
    num_reqs: int, max_sample_len: int = 1, max_draft_len: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    pp = get_pp_group()
    assert not pp.is_last_rank

    token_ids = torch.empty(
        num_reqs,
        max_sample_len + max_draft_len,
        dtype=torch.int64,
        device=pp.device,
    )
    torch.distributed.broadcast(token_ids, src=pp.last_rank, group=pp.device_group)

    combined = torch.empty(2, num_reqs, dtype=torch.int32, device=pp.device)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)
    num_sampled, num_rejected = combined.unbind(dim=0)
    sampled_tokens = token_ids[:, :max_sample_len]
    draft_tokens = token_ids[:, max_sample_len:] if max_draft_len else None
    return sampled_tokens, num_sampled, num_rejected, draft_tokens
