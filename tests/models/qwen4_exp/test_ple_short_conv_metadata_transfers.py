# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends import short_conv_attn


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "query_lengths,drafts,spec_rows,non_spec_rows",
    [
        ([5], [4], [0], []),
        ([5, 5, 5, 5], [4, 4, 4, 4], [0, 1, 2, 3], []),
        ([5, 0], [4, -1], [0], []),
        ([5, 1, 5, 3], [4, -1, 4, -1], [0, 2], [1, 3]),
    ],
)
def test_ple_request_indices_preserve_state_and_acceptance_order(
    device, query_lengths, drafts, spec_rows, non_spec_rows, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setattr(
        short_conv_attn, "mamba_get_block_table_tensor", lambda table, *_: table
    )
    monkeypatch.setattr(
        short_conv_attn, "compute_causal_conv1d_metadata", lambda *_a, **_k: (None,) * 3
    )
    builder = short_conv_attn.PleShortConvAttentionMetadataBuilder.__new__(
        short_conv_attn.PleShortConvAttentionMetadataBuilder
    )
    builder.use_spec_decode = True
    builder.use_full_cuda_graph = False
    builder.num_spec = 4
    builder.kv_cache_spec = None
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none")
    )
    starts = torch.tensor([0, *query_lengths], dtype=torch.int32).cumsum(0).int()
    n = len(query_lengths)
    slots = torch.arange(10, 10 + n, dtype=torch.int32).reshape(n, 1)
    accepted = (torch.arange(n, dtype=torch.int32) % 5 + 1).to(device)
    batch = SimpleNamespace(
        query_start_loc=starts.to(device),
        query_start_loc_cpu=starts,
        num_reqs=n,
        num_actual_tokens=sum(query_lengths),
        block_table_tensor=slots.to(device),
        seq_lens=torch.tensor(query_lengths, dtype=torch.int32, device=device) + 16,
        compute_num_computed_tokens=lambda: torch.full((n,), 16, device=device),
    )
    metadata = builder.build(
        0,
        batch,
        num_accepted_tokens=accepted,
        num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.spec_state_indices_tensor.cpu(), slots[spec_rows, 0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        metadata.num_accepted_tokens.cpu(), accepted.cpu()[spec_rows], rtol=0, atol=0
    )
    torch.testing.assert_close(
        metadata.state_indices_tensor.cpu(), slots[non_spec_rows, 0], rtol=0, atol=0
    )
