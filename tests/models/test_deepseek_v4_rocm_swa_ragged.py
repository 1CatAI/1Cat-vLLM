# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

WINDOW_SIZE = 128
# cdiv(window_size + num_speculative_tokens, 128) * 128 for DSpark
DSPARK_INDEX_WIDTH = 256


@pytest.mark.parametrize(
    "device_type",
    [
        "cpu",
        pytest.param(
            current_platform.device_type,
            marks=pytest.mark.skipif(
                not current_platform.is_cuda_alike(),
                reason="the native ragged pack requires CUDA or ROCm",
            ),
            id="native",
        ),
    ],
)
@pytest.mark.parametrize(
    "noncausal_index_width",
    [0, DSPARK_INDEX_WIDTH],
    ids=["causal", "dspark"],
)
def test_rocm_swa_builder_keeps_full_decode_rows(
    monkeypatch: pytest.MonkeyPatch, noncausal_index_width: int, device_type: str
):
    """The ROCm SWA builder copies the ragged decode indices into its graph
    buffer. DSpark's non-causal rows are wider than the window; both the
    buffer and the copied slice have to follow the row width."""
    from vllm.models.deepseek_v4.amd.rocm import (
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
        DeepseekSparseSWAMetadataBuilder,
    )

    device = torch.device(device_type)
    # Every schedulable token is a decode token, so the graph buffer is
    # filled to its size.
    max_tokens = 4
    width = max(WINDOW_SIZE, noncausal_index_width)
    indices = torch.randint(
        0, 1 << 20, (max_tokens, 1, width), dtype=torch.int32, device=device
    )
    lens = torch.tensor(
        [width, width - 7, width // 2 + 1, 3],
        dtype=torch.int32,
        device=device,
    )

    def fake_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens)
        )
        self.window_size = WINDOW_SIZE
        self.noncausal_index_width = noncausal_index_width
        self.device = device

    def fake_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return DeepseekSparseSWAMetadata(
            block_table=torch.empty((max_tokens, 0), dtype=torch.int32),
            slot_mapping=torch.empty(max_tokens, dtype=torch.int64),
            block_size=256,
            causal=noncausal_index_width == 0,
            decode_swa_indices=indices,
            decode_swa_lens=lens,
            num_decodes=max_tokens,
            num_decode_tokens=max_tokens,
        )

    monkeypatch.setattr(DeepseekSparseSWAMetadataBuilder, "__init__", fake_init)
    monkeypatch.setattr(DeepseekSparseSWAMetadataBuilder, "build", fake_build)
    if device_type == "cpu":
        from vllm.models.deepseek_v4.amd import rocm

        def cpu_pack(dense, lengths):
            # Only replace the Triton pack. Allocation, slice sizing, graph
            # buffer copies, and returned metadata execute the real builder.
            packed = dense.new_empty(dense.numel())
            prefix = torch.cat(
                [dense[row, : int(length)] for row, length in enumerate(lengths)]
            )
            packed[: prefix.numel()].copy_(prefix)
            indptr = lengths.new_zeros(lengths.numel() + 1)
            torch.cumsum(lengths, dim=0, out=indptr[1:])
            return packed, indptr

        monkeypatch.setattr(rocm, "build_ragged_indices_from_dense", cpu_pack)

    builder = DeepseekV4ROCMAiterSparseSWAMetadataBuilder()
    metadata = builder.build(0, None)

    expected = torch.cat([indices[i, 0, : int(n)] for i, n in enumerate(lens)])
    expected_indptr = torch.zeros(max_tokens + 1, dtype=torch.int32, device=device)
    torch.cumsum(lens, dim=0, out=expected_indptr[1:])

    assert metadata.decode_swa_ragged_indptr is not None
    assert metadata.decode_swa_ragged_indices is not None
    torch.testing.assert_close(metadata.decode_swa_ragged_indptr, expected_indptr)
    torch.testing.assert_close(
        metadata.decode_swa_ragged_indices[: expected.numel()], expected
    )
    # The returned slice lives in the persistent graph buffer.
    assert (
        metadata.decode_swa_ragged_indices.data_ptr()
        == builder.decode_swa_ragged_indices_buffer.data_ptr()
    )
    buffer_pointer = builder.decode_swa_ragged_indices_buffer.data_ptr()
    indices.add_(1)
    repeated = builder.build(0, None)
    assert repeated.decode_swa_ragged_indices is not None
    assert repeated.decode_swa_ragged_indices.data_ptr() == buffer_pointer
    expected = torch.cat([indices[i, 0, : int(n)] for i, n in enumerate(lens)])
    torch.testing.assert_close(
        repeated.decode_swa_ragged_indices[: expected.numel()], expected
    )
