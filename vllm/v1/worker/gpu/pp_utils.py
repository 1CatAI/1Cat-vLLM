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
    if sample_len < max_sample_len:
        # NCCL collectives require every rank to use the same element count.
        # A non-speculative prefill emits one token even when later verify
        # steps can emit 1 + num_speculative_tokens.
        padded_sampled_token_ids = sampled_token_ids.new_full(
            (num_reqs, max_sample_len), -1
        )
        padded_sampled_token_ids[:, :sample_len].copy_(sampled_token_ids)
        sampled_token_ids = padded_sampled_token_ids
    torch.distributed.broadcast(
        sampled_token_ids.contiguous(), src=pp.last_rank, group=pp.device_group
    )

    combined = torch.stack((num_sampled, num_rejected), dim=0)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)


def pp_receive(
    num_reqs: int, max_sample_len: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pp = get_pp_group()
    assert not pp.is_last_rank

    sampled_tokens = torch.empty(
        num_reqs, max_sample_len, dtype=torch.int64, device=pp.device
    )
    torch.distributed.broadcast(sampled_tokens, src=pp.last_rank, group=pp.device_group)

    combined = torch.empty(2, num_reqs, dtype=torch.int32, device=pp.device)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)
    num_sampled, num_rejected = combined.unbind(dim=0)
    return sampled_tokens, num_sampled, num_rejected
