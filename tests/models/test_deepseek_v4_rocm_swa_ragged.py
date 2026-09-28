# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the ragged pack is a Triton kernel",
)

WINDOW_SIZE = 128
# cdiv(window_size + num_speculative_tokens, 128) * 128 for DSpark
DSPARK_INDEX_WIDTH = 256


@pytest.mark.parametrize(
    "noncausal_index_width",
    [0, DSPARK_INDEX_WIDTH],
    ids=["causal", "dspark"],
)
def test_rocm_swa_builder_keeps_full_decode_rows(
    monkeypatch: pytest.MonkeyPatch, noncausal_index_width: int
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

    device = torch.device(current_platform.device_type)
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
