# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.v1.attention.backends import turboquant_attn as attention


def test_large_continuation_uses_bounded_decode_chunks(monkeypatch):
    implementation = attention.TurboQuantAttentionImpl.__new__(
        attention.TurboQuantAttentionImpl
    )
    implementation.use_flash_attn_prefill = False
    implementation.use_flash_v100_dense_prefill = False
    implementation.scale = 1.0
    implementation.tq_config = SimpleNamespace(
        key_mse_bits=2,
        key_packed_size=32,
        effective_value_quant_bits=4,
        key_fp8=False,
        norm_correction=False,
    )
    calls = []

    def observe_lengths(**kwargs):
        query = kwargs["query"]
        calls.append(query.shape[0])
        return kwargs["seq_lens"].to(query.dtype).view(-1, 1, 1).expand_as(query)

    def reject_dense(*args, **kwargs):
        raise AssertionError(
            "The large continuation must not materialize dense history"
        )

    monkeypatch.setattr(
        attention, "triton_turboquant_decode_attention", observe_lengths
    )
    monkeypatch.setattr(implementation, "_continuation_prefill", reject_dense)
    rows, context = 257, 512
    metadata = attention.TurboQuantMetadata(
        seq_lens=torch.tensor([context], dtype=torch.int32),
        slot_mapping=torch.arange(rows),
        block_table=torch.zeros(1, 32, dtype=torch.int32),
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
        num_actual_tokens=rows,
        max_query_len=rows,
        max_seq_len=context,
        query_start_loc_cpu=torch.tensor([0, rows], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([context], dtype=torch.int32),
    )
    query = torch.zeros(rows, 1, 128, dtype=torch.float16)
    result = implementation._prefill_attention(
        query,
        query,
        query,
        torch.empty(32, 16, 1, 96, dtype=torch.uint8),
        metadata,
        torch.eye(128),
        torch.tensor([-1.0, -0.25, 0.25, 1.0]),
    )
    assert calls == [128, 128, 1]
    expected = torch.arange(context - rows + 1, context + 1).to(query.dtype)
    torch.testing.assert_close(result[:, 0, 0], expected, rtol=0, atol=0)
