# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F

import flashinfer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
        reason="requires physical SM70; CPU results are not kernel validation",
    ),
]


@pytest.mark.parametrize("width", [72, 128, 2560, 8192])
@pytest.mark.parametrize("gemma", [False, True])
def test_norm_graph_reload_and_separate_instances(width, gemma):
    torch.manual_seed(13)
    flashinfer.sm70.prepare()
    fn = flashinfer.gemma_rmsnorm if gemma else flashinfer.rmsnorm
    x = torch.randn(20, width, dtype=torch.float16, device="cuda")
    weight = torch.randn(width, dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    fn(x, weight, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(x, weight, out=out)
    for _ in range(3):
        x.normal_()
        weight.normal_()
        graph.replay()
        reference = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        reference *= weight.float() + float(gemma)
        torch.testing.assert_close(out, reference.half(), rtol=1e-3, atol=1e-3)
    second = fn(x, weight)
    assert second.data_ptr() != out.data_ptr()
    torch.testing.assert_close(second, out, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,activation",
    [
        ("silu_and_mul", F.silu),
        ("gelu_and_mul", F.gelu),
        ("gelu_tanh_and_mul", lambda x: F.gelu(x, approximate="tanh")),
    ],
)
def test_activation(name, activation):
    x = torch.randn(20, 5120, dtype=torch.float16, device="cuda")
    out = getattr(flashinfer, name)(x)
    reference = activation(x[:, :2560].float()) * x[:, 2560:].float()
    torch.testing.assert_close(out, reference.half(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("interleave", [False, True])
def test_partial_rope_inplace_strided_and_positions(interleave):
    flashinfer.sm70.prepare()
    qkv = torch.randn(20, 3, 4, 128, dtype=torch.float16, device="cuda")
    q, k = qkv[:, 0], qkv[:, 1]
    positions = torch.arange(127990, 128010, device="cuda", dtype=torch.int64)
    q0, k0 = q.clone(), k.clone()
    for original in (q0, k0):
        frequency = 10000.0 ** (-torch.arange(32, device="cuda").float() / 32)
        angle = positions.float()[:, None, None] * frequency
        part = original[..., :64].float()
        a, b = (
            (part[..., ::2], part[..., 1::2])
            if interleave
            else (part[..., :32], part[..., 32:])
        )
        rotated = torch.stack(
            (a * angle.cos() - b * angle.sin(), b * angle.cos() + a * angle.sin()), -1
        )
        if not interleave:
            rotated = rotated.transpose(-1, -2)
        original[..., :64] = rotated.flatten(-2).half()
    flashinfer.apply_rope_pos_ids_inplace(
        q, k, positions, rotary_dim=64, interleave=interleave
    )
    torch.testing.assert_close(q, q0, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k, k0, rtol=1e-3, atol=1e-3)


def test_unsupported_dtype_and_pdl_fail_closed():
    x = torch.ones(1, 128, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="FP16"):
        flashinfer.rmsnorm(x, x[0])
    with pytest.raises(ValueError, match="unavailable"):
        flashinfer.rmsnorm(x.half(), x[0].half(), enable_pdl=True)


@pytest.mark.parametrize("gemma", [False, True])
def test_residual_rounding_and_empty_rows(gemma):
    x = torch.randn(5, 2560, dtype=torch.float16, device="cuda")
    residual, weight = torch.randn_like(x), torch.randn_like(x[0])
    total = x.float() + residual.float()
    reference = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6)
    reference *= weight.float() + float(gemma)
    fn = flashinfer.gemma_fused_add_rmsnorm if gemma else flashinfer.fused_add_rmsnorm
    fn(x, residual, weight)
    torch.testing.assert_close(residual, total.half(), rtol=0, atol=0)
    torch.testing.assert_close(x, reference.half(), rtol=1e-3, atol=1e-3)
    fn(x[:0], residual[:0], weight)
    assert flashinfer.rmsnorm(x[:0], weight).shape == (0, 2560)
