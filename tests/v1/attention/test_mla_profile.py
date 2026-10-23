# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile allocations must match the selected MLA execution path."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention import mla_attention
from vllm.v1.attention.backend import SparseMLAAttentionImpl

pytestmark = pytest.mark.cpu_test


class _SparseImpl(SparseMLAAttentionImpl):
    def __init__(self):
        pass

    def forward_mqa(self, *args, **kwargs):
        raise AssertionError("Profiling must not run attention without metadata")


class _ProfileLayer(SimpleNamespace):
    @property
    def chunked_prefill_workspace_size(self):
        assert not isinstance(self.impl, SparseMLAAttentionImpl)
        return 11


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("quantized_output", [False, True])
def test_profile_reserves_dense_projection_only_for_dense_impl(
    monkeypatch, sparse, quantized_output
):
    layer = _ProfileLayer(
        impl=_SparseImpl() if sparse else SimpleNamespace(),
        num_heads=2,
        qk_nope_head_dim=3,
        v_head_dim=4,
    )
    q = torch.ones((3, 2, 3), dtype=torch.float16)
    latent = torch.ones((3, 5), dtype=torch.float16)
    pe = torch.empty((3, 1, 0), dtype=torch.float16)
    output = torch.ones((3, 8), dtype=torch.float16)
    if quantized_output:
        output = output.to(mla_attention._FP8_DTYPE)
    output_scale = torch.ones(1) if quantized_output else None
    allocations = []
    original_empty = torch.empty

    def recording_empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        allocations.append((tuple(result.shape), result.dtype, result.device.type))
        return result

    monkeypatch.setattr(torch, "empty", recording_empty)
    result = mla_attention.MLAAttention.forward_impl(
        layer, q, latent, pe, latent, None, output, output_scale=output_scale
    )
    expected: list[tuple[tuple[int, ...], torch.dtype, str]] = (
        [((3, 8), torch.float16, "cpu")] if quantized_output else []
    )
    if not sparse:
        expected.append(((11, 2, 7), torch.float16, "cpu"))
    assert allocations == expected
    assert result is output
    torch.testing.assert_close(result.float(), torch.zeros_like(result.float()))


def test_flashinfer_sparse_profile_reserves_shared_runtime_workspace(monkeypatch):
    pytest.importorskip("flashinfer.decode")
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as backend

    # Exercise its real lazy allocation on CPU with a small workspace, without
    # loading FlashInfer kernels or allocating the full production buffer.
    monkeypatch.setattr(backend, "FLASHINFER_MLA_SPARSE_WORKSPACE_BUFFER_SIZE", 256)
    monkeypatch.setattr(backend, "_fi_sparse_workspace", None)
    implementations = [
        backend.FlashInferMLASparseImpl.__new__(backend.FlashInferMLASparseImpl)
        for _ in range(2)
    ]
    q = torch.ones((3, 2, 3), dtype=torch.float16)
    latent = torch.ones((3, 5), dtype=torch.float16)
    pe = torch.empty((3, 1, 0), dtype=torch.float16)
    for impl in implementations:
        impl._workspace_buffer = None
        layer = _ProfileLayer(impl=impl, num_heads=2, v_head_dim=4, qk_nope_head_dim=3)
        output = torch.ones((3, 8), dtype=torch.float16)
        mla_attention.MLAAttention.forward_impl(
            layer, q, latent, pe, latent, None, output
        )
        assert impl._workspace_buffer is backend._fi_sparse_workspace
        assert impl._workspace_buffer.numel() == 256
        assert impl._workspace_buffer.dtype == torch.uint8
        before = impl._workspace_buffer
        impl.reserve_profile_workspace(torch.device("cpu"))
        assert impl._workspace_buffer is before
    assert implementations[0]._workspace_buffer is implementations[1]._workspace_buffer
