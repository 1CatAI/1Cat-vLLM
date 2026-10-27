# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded V4.1 FP16 sublayer checks, without a resident FP32 model."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.mhc.deepseek_v41 import (
    _row_projection,
    collapse_hc,
    expand_hc,
    fp32_projection,
    hc_mixes,
    route_experts,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_hc_k_parallelism_is_bitwise_original_order_after_graph_updates(tokens, dtype):
    generator = torch.Generator(device="cuda").manual_seed(4191)
    x = torch.randn(tokens, 20480, generator=generator, device="cuda").to(dtype)
    if dtype == torch.float32:
        x.mul_(2.0**19)
    weight = torch.randn(24, 20480, generator=generator, device="cuda") * 0.005
    eps = 3e-9
    expected = torch.empty(tokens, 24, dtype=torch.float32, device="cuda")

    def reference():
        _row_projection[(tokens, 24)](
            x,
            weight,
            expected,
            K=20480,
            N=24,
            NORMALIZE=True,
            EPS=eps,
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )

    fp32_projection(x, weight, normalized=True, eps=eps)
    reference()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fp32_projection(x, weight, normalized=True, eps=eps)
    for multiplier in (-0.75, 1.0):
        x.mul_(multiplier)
        weight.mul_(0.9)
        graph.replay()
        reference()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 6, 64])
def test_v41_mhc_coefficients_delayed_collapse_and_post(tokens):
    generator = torch.Generator().manual_seed(41)
    residual = torch.randn(tokens, 4, 5120, generator=generator).half()
    fn = torch.randn(24, 20480, generator=generator) * 0.005
    scales = torch.tensor([0.3, 0.2, 0.1])
    base = torch.randn(24, generator=generator) * 0.4
    norm = torch.rand(5120, generator=generator) + 0.5
    x = residual.flatten(1).float()
    mixes = F.linear(x, fn) * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)
    expected_pre = torch.sigmoid(mixes[:, :4] * scales[0] + base[:4]) + 1e-6
    expected_post = 2 * torch.sigmoid(mixes[:, 4:8] * scales[1] + base[4:8])
    expected_comb = (mixes[:, 8:] * scales[2] + base[8:]).view(-1, 4, 4).softmax(
        -1
    ) + 1e-6
    expected_comb /= expected_comb.sum(-2, keepdim=True) + 1e-6
    for _ in range(19):
        expected_comb /= expected_comb.sum(-1, keepdim=True) + 1e-6
        expected_comb /= expected_comb.sum(-2, keepdim=True) + 1e-6
    gpu_residual = residual.cuda()
    pre, post, comb = hc_mixes(gpu_residual, fn.cuda(), scales.cuda(), base.cuda())
    for actual, expected in (
        (pre, expected_pre),
        (post, expected_post),
        (comb, expected_comb),
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)

    # The current sublayer consumes an independently supplied previous mix.
    # Identity selects stream zero, not the freshly calculated current pre.
    previous = torch.zeros(tokens, 4)
    previous[:, 0] = 1
    collapsed = collapse_hc(gpu_residual, previous.cuda())
    torch.testing.assert_close(collapsed.cpu(), residual[:, 0], rtol=0, atol=0)
    normalized = collapse_hc(gpu_residual, previous.cuda(), norm.cuda())
    f = residual[:, 0].float()
    expected = (
        f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-20) * norm
    ).half()
    torch.testing.assert_close(normalized.cpu(), expected, rtol=0.001, atol=2e-6)

    output = expand_hc(normalized, gpu_residual, post, comb)
    # Source index precedes destination index in comb. A transpose would
    # still look doubly stochastic but would produce a different residual.
    expected = (
        torch.einsum("tij,tih->tjh", comb.cpu(), residual.float())
        + post.cpu()[..., None] * normalized.cpu()[:, None].float()
    ).half()
    # Compare this FP16 sublayer boundary using the verified coefficients
    # and actual preceding FP16 store. An upstream half-ULP rounding choice
    # can be amplified by cancellation in the next operation; it is not a
    # different post-mix arithmetic error allowance.
    torch.testing.assert_close(output.cpu(), expected, rtol=0.002, atol=3e-6)


@pytest.mark.parametrize("normalized,n,k", [(False, 384, 5120), (True, 24, 20480)])
def test_v41_fp32_projection_and_routes_are_bitwise_batch_invariant(normalized, n, k):
    generator = torch.Generator(device="cuda").manual_seed(41)
    x = torch.randn(6, k, device="cuda", generator=generator).half()
    weight = torch.randn(n, k, device="cuda", generator=generator) * 0.005
    actual = fp32_projection(x, weight, normalized=normalized)
    separate = torch.cat(
        [fp32_projection(x[i : i + 1], weight, normalized=normalized) for i in range(6)]
    )
    torch.testing.assert_close(actual, separate, rtol=0, atol=0)
    expected = F.linear(x.float(), weight)
    if normalized:
        expected *= torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    if not normalized:
        bias = torch.randn(n, device="cuda", generator=generator) * 0.01
        probability, ids = route_experts(x, weight, bias)
        rows = [route_experts(x[i : i + 1], weight, bias) for i in range(6)]
        torch.testing.assert_close(ids, torch.cat([r[1] for r in rows]), rtol=0, atol=0)
        torch.testing.assert_close(
            probability, torch.cat([r[0] for r in rows]), rtol=0, atol=0
        )


def test_v41_mhc_graph_replay_and_fp16_before_norm_boundary():
    generator = torch.Generator(device="cuda").manual_seed(42)
    residual = torch.randn(6, 4, 5120, device="cuda", generator=generator).half()
    pre = torch.rand(6, 4, device="cuda", generator=generator)
    norm = torch.rand(5120, device="cuda", generator=generator)
    collapse_hc(residual, pre, norm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = collapse_hc(residual, pre, norm)
    residual.mul_(-0.75)
    pre.mul_(0.7)
    graph.replay()
    # Explicit FP16 collapse before normalization is observable here.
    f = (residual.float() * pre[..., None]).sum(1).half().float()
    expected = (
        f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-20) * norm
    ).half()
    torch.testing.assert_close(output, expected, rtol=0.001, atol=2e-6)


@pytest.mark.parametrize("tokens", [1, 6])
@pytest.mark.parametrize("wide_branch", [False, True])
def test_fp32_hc_carries_wide_residuals_and_normalizes_before_fp16(tokens, wide_branch):
    generator = torch.Generator(device="cuda").manual_seed(710)
    residual = torch.randn(tokens, 4, 5120, device="cuda", generator=generator)
    residual.mul_(2.0**19)
    pre = torch.rand(tokens, 4, device="cuda", generator=generator)
    norm = torch.rand(5120, device="cuda", generator=generator) + 0.5
    collapsed = (residual * pre[..., None]).sum(1)
    assert torch.isinf(collapsed.half()).any()  # The original failure boundary.

    def run():
        return collapse_hc(residual, pre, norm)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    residual.mul_(-0.75)
    graph.replay()
    f = (residual * pre[..., None]).sum(1)
    expected = (
        f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-20) * norm
    ).half()
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.001, atol=2e-6)

    post = torch.rand(tokens, 4, device="cuda", generator=generator)
    comb = torch.rand(tokens, 4, 4, device="cuda", generator=generator)
    branch = actual.float() * (2.0**18) if wide_branch else actual
    if wide_branch:
        assert torch.isinf(branch.half()).any()
    expanded = expand_hc(branch, residual, post, comb)
    assert expanded.dtype == torch.float32
    assert torch.isfinite(expanded).all()
    products = comb.double()[..., None] * residual.double()[:, :, None]
    addition = post.double()[..., None] * branch.double()[:, None]
    reference = products.sum(1) + addition
    # A five-term FP32 sum is bounded by its own product magnitudes. This
    # checks the FP32 store, independently of the old FP16 error allowance.
    bound = (
        8 * torch.finfo(torch.float32).eps * (products.abs().sum(1) + addition.abs())
    )
    assert ((expanded.double() - reference).abs() <= bound).all()
