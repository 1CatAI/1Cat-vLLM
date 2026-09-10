# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared attention contracts and non-H3 DiT shapes, without model imports."""

import pytest
import torch

from vllm.model_executor.layers import sm70_attention as ops


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan"), 1e100])
def test_invalid_scale_does_not_load_native_code(monkeypatch, scale):
    def unexpected_load():
        pytest.fail("invalid scale must fail before extension loading")

    monkeypatch.setattr(ops, "flashattn_extension", unexpected_load)
    monkeypatch.setattr(ops, "flashinfer_extension", unexpected_load)
    for backend in ("FLASH_ATTN_V100", "FLASHINFER_SM70"):
        with pytest.raises(ValueError, match="scale"):
            ops.noncausal_attention(None, None, None, scale=scale, backend=backend)


def test_explicit_backend_and_geometry_contract():
    for backend, tile in (("AUTO", 64), ("FLASHINFER_SM70", 128)):
        with pytest.raises(ValueError):
            ops.noncausal_attention(
                None, None, None, scale=0.1, backend=backend, query_tile=tile
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM70 GPU")
@pytest.mark.parametrize(
    "backend,tile",
    [("FLASH_ATTN_V100", 64), ("FLASH_ATTN_V100", 128), ("FLASHINFER_SM70", 64)],
)
@pytest.mark.parametrize("batch,length,heads", [(1, 1537, 24), (2, 65, 8)])
def test_gpu_non_h3_shapes_preserve_native_output(backend, tile, batch, length, heads):
    torch.manual_seed(206)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # Strided token storage and non-H3 head counts exercise the shared contract.
    q, k, v = [
        torch.randn(batch, length * 2, heads, 128, device="cuda", dtype=torch.float16)[
            :, ::2
        ]
        for _ in range(3)
    ]
    k[:, length // 2 :] *= 4
    scale = 128**-0.5
    actual = ops.noncausal_attention(
        q, k, v, scale=scale, backend=backend, query_tile=tile
    )
    native = (
        ops.flashattn_extension()
        if backend == "FLASH_ATTN_V100"
        else ops.flashinfer_extension()
    )
    args = (scale, 0, tile) if tile != 64 else (scale,)
    expected = native.forward(q.contiguous(), k.contiguous(), v.contiguous(), *args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    rows = torch.linspace(0, length - 1, min(length, 33), device="cuda").long()
    qh, kh, vh = (x.transpose(1, 2).float() for x in (q[:, rows], k, v))
    reference = (((qh @ kh.transpose(-1, -2)) * scale).softmax(-1) @ vh).transpose(1, 2)
    assert torch.isfinite(actual).all()
    relative_l2 = (actual[:, rows].float() - reference).norm() / reference.norm()
    assert relative_l2 < 0.001
    with pytest.raises(RuntimeError, match="FP16"):
        ops.noncausal_attention(
            q.float(),
            k.float(),
            v.float(),
            scale=scale,
            backend=backend,
            query_tile=tile,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM70 GPU")
def test_gpu_cross_attention_contract():
    torch.manual_seed(207)
    q = torch.randn(2, 33, 8, 128, device="cuda", dtype=torch.float16)
    k, v = [
        torch.randn(2, 129, 8, 128, device="cuda", dtype=torch.float16)
        for _ in range(2)
    ]
    actual = ops.noncausal_attention(
        q, k, v, scale=0.1, backend="FLASH_ATTN_V100", query_tile=128
    )
    expected = ops.flashattn_extension().forward(q, k, v, 0.1, 0, 128)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="matching"):
        ops.noncausal_attention(q, k, v, scale=0.1, backend="FLASHINFER_SM70")
