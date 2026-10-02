# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends import turboquant_attn as attention


def _metadata(rows, seq_lens, cpu_copy=True):
    return attention.TurboQuantMetadata(
        seq_lens=seq_lens,
        slot_mapping=torch.arange(rows),
        block_table=torch.zeros(1, 32, dtype=torch.int32),
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
        num_actual_tokens=rows,
        max_query_len=rows,
        max_seq_len=rows + 1,
        query_start_loc_cpu=torch.tensor([0, rows], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([rows + 1], dtype=torch.int32) if cpu_copy else None,
    )


@pytest.mark.parametrize("rows", [1, 5, 128])
@pytest.mark.parametrize("cpu_copy", [True, False])
def test_capture_metadata_selects_continuation(rows, cpu_copy):
    metadata = _metadata(rows, torch.tensor([rows]), cpu_copy)
    metadata.max_seq_len = rows
    if metadata.seq_lens_cpu is not None:
        metadata.seq_lens_cpu.fill_(rows)
    builder = attention.TurboQuantMetadataBuilder.__new__(
        attention.TurboQuantMetadataBuilder
    )
    builder.build = lambda *args: metadata
    result = builder.build_for_cudagraph_capture(None)
    assert result.max_seq_len == rows + 1
    assert result.seq_lens.tolist() == [rows + 1]
    if cpu_copy:
        assert result.seq_lens_cpu.tolist() == [rows + 1]


@pytest.mark.parametrize("rows", [1, 5, 128])
def test_captured_continuation_reads_live_sequence_lengths(monkeypatch, rows):
    # Replace only the GPU kernel; its output exposes the lengths supplied by
    # the real wrapper. CPU hints remain frozen like CUDA graph capture hints.
    def observe_lengths(**kwargs):
        query = kwargs["query"]
        return kwargs["seq_lens"].to(query.dtype).view(-1, 1, 1).expand_as(query)

    monkeypatch.setattr(
        attention, "triton_turboquant_decode_attention", observe_lengths
    )
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
    metadata = _metadata(rows, torch.tensor([rows + 1], dtype=torch.int32))

    class Continuation(torch.nn.Module):
        def forward(self, query, seq_lens):
            metadata.seq_lens = seq_lens
            return implementation._prefill_attention(
                query,
                query,
                query,
                torch.empty(32, 16, 1, 96, dtype=torch.uint8),
                metadata,
                torch.eye(128),
                torch.tensor([-1.0, -0.25, 0.25, 1.0]),
            )

    query = torch.zeros(rows, 1, 128, dtype=torch.float16)
    compiled = torch.jit.trace(
        Continuation(), (query, metadata.seq_lens), check_trace=False
    )
    for context in (rows + 1, rows + 38, rows + 75):
        actual = compiled(query, torch.tensor([context], dtype=torch.int32))
        expected = torch.arange(context - rows + 1, context + 1).to(query.dtype)
        torch.testing.assert_close(actual[:, 0, 0], expected, rtol=0, atol=0)
