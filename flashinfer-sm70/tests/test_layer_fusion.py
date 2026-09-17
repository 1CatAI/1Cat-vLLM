# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from benchmarks.kernels.benchmark_sm70_flashinfer_gdn_conv import (
    capture,
    check_exclusive,
)
from benchmarks.kernels.flashinfer_sm70_gdn_conv import FusedGDN, build
from benchmarks.kernels.flashinfer_sm70_hc_norm import HCNorm
from benchmarks.kernels.flashinfer_sm70_hc_norm import build as build_hc


def test_conv_product_rounding_is_not_fp32_multiply():
    x = torch.tensor([1.0009765625], dtype=torch.float16)
    assert (x * x).float().item() != (x.float() * x.float()).item()


@pytest.mark.parametrize("rows", [0, -1, 65])
def test_invalid_rows_fail_before_gpu_allocation(rows):
    with pytest.raises(ValueError):
        FusedGDN(rows, device="cpu")


@pytest.mark.parametrize("hq,hv", [(0, 12), (4, 0), (4, 7), (-1, 12)])
def test_invalid_heads_fail_before_gpu_allocation(hq, hv):
    with pytest.raises(ValueError):
        FusedGDN(4, hq, hv, device="cpu")


@pytest.fixture(scope="module")
def cuda_build():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    check_exclusive()
    torch.manual_seed(7)
    build()


@pytest.fixture(scope="module")
def cuda_hc_build():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    check_exclusive()
    build_hc()


def oracle(x, weights, qkv, cw, conv, A, dt, state, indices):
    ba = (x @ weights).half().float()
    output = torch.zeros(x.shape[0], 12, 128, device=x.device, dtype=torch.float32)
    for row in range(x.shape[0]):
        slot = int(indices[row])
        if slot < 0:
            continue
        # Explicit FP16 product boundary observed in production conv PTX.
        y = (conv[slot, :, 0] * cw[:, 0]).float()
        y += (conv[slot, :, 1] * cw[:, 1]).float()
        y += (conv[slot, :, 2] * cw[:, 2]).float()
        y += (qkv[row] * cw[:, 3]).float()
        mixed = (y / (1 + (-y).exp())).half().float()
        conv[slot, :, :2] = conv[slot, :, 1:].clone()
        conv[slot, :, 2] = qkv[row]
        q = mixed[:512].reshape(4, 128).repeat_interleave(3, 0)
        k = mixed[512:1024].reshape(4, 128).repeat_interleave(3, 0)
        v = mixed[1024:].reshape(12, 128)
        q *= (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        k *= (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        decay = (
            -A.exp() * torch.nn.functional.softplus(ba[row, 12:] + dt.float())
        ).exp()
        beta = ba[row, :12].sigmoid()
        old = state[slot] * decay[:, None, None]
        delta = (v - (old * k[:, None, :]).sum(-1)) * beta[:, None]
        state[slot] = old + delta[:, :, None] * k[:, None, :]
        output[row] = (state[slot] * q[:, None, :]).sum(-1) / (128**0.5)
    return output


@pytest.mark.parametrize(
    "rows,sd_layout", [(1, True), (4, False), (8, True), (16, True)]
)
def test_gdn_graph_dynamic_slots_and_history(cuda_build, rows, sd_layout):
    pool, width = rows + 2, 2560
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    weights = torch.randn(2560, 24, device="cuda", dtype=torch.float16) * 0.01
    qkv_storage = torch.randn(rows, 4096, device="cuda", dtype=torch.float16)
    qkv = qkv_storage[:, :width]
    cw = torch.randn(width, 4, device="cuda", dtype=torch.float16) * 0.2
    bias = torch.empty(0, device="cuda", dtype=torch.float16)
    conv = torch.randn(pool, width, 3, device="cuda", dtype=torch.float16) * 0.2
    if sd_layout:
        conv = conv.transpose(1, 2).contiguous().transpose(1, 2)
    state = torch.randn(pool, 12, 128, 128, device="cuda", dtype=torch.float32) * 0.02
    A = torch.randn(12, device="cuda", dtype=torch.float32)
    dt = torch.randn(12, device="cuda", dtype=torch.float16)
    indices = torch.arange(rows, device="cuda", dtype=torch.int32)
    candidate = FusedGDN(rows)
    call = lambda: candidate(x, weights, qkv, cw, bias, conv, A, dt, state, indices)
    graph = capture(call)
    expected_c, expected_s = conv.clone(), state.clone()
    for cycle in range(8):
        x.normal_()
        qkv.normal_()
        indices.copy_(torch.randperm(pool, device="cuda")[:rows])
        if cycle % 3 == 2:
            indices[-1] = -1
        c0, s0 = conv.clone(), state.clone()
        # The oracle retains its own history: copying candidate state into it
        # each cycle would hide accumulated recurrent drift.
        expected_o = oracle(x, weights, qkv, cw, expected_c, A, dt, expected_s, indices)
        eager = call().clone()
        eager_state, eager_conv = state.clone(), conv.clone()
        conv.copy_(c0)
        state.copy_(s0)
        candidate.partial.fill_(float("nan"))
        candidate.output.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(candidate.output, eager, atol=0, rtol=0)
        torch.testing.assert_close(state, eager_state, atol=0, rtol=0)
        torch.testing.assert_close(conv, eager_conv, atol=0, rtol=0)
        torch.testing.assert_close(conv, expected_c, atol=0, rtol=0)
        torch.testing.assert_close(state, expected_s, atol=1e-4, rtol=2e-3)
        torch.testing.assert_close(
            candidate.output.float(), expected_o, atol=1e-4, rtol=2e-3
        )


@pytest.mark.parametrize(
    "rows,groups,width,r_dtype,b_dtype,w_dtype,shared,warps",
    [
        (4, 4, 2560, torch.float16, torch.float16, torch.float16, False, 2),
        (8, 4, 320, torch.float32, torch.float16, torch.float16, True, 1),
        (16, 2, 520, torch.float16, torch.float32, torch.float32, False, 4),
        (1, 3, 4096, torch.float32, torch.float32, torch.float32, True, 8),
    ],
)
def test_hc_graph_rounding_and_shared_weights(
    cuda_hc_build, rows, groups, width, r_dtype, b_dtype, w_dtype, shared, warps
):
    check_hc_graph(rows, groups, width, r_dtype, b_dtype, w_dtype, shared, warps)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("warps", [4, 8])
def test_hc_register_graph(cuda_hc_build, dtype, warps):
    check_hc_graph(
        16, 4, 2560, dtype, torch.float16, torch.float16, True, warps, registers=True
    )


def check_hc_graph(
    rows, groups, width, r_dtype, b_dtype, w_dtype, shared, warps, registers=False
):
    torch.manual_seed(19)
    r = torch.randn(rows, groups * width, device="cuda", dtype=r_dtype)
    b = torch.randn(rows, width, device="cuda", dtype=b_dtype)
    inj = torch.randn(rows, groups, device="cuda", dtype=w_dtype)
    w = torch.randn(width if shared else groups * width, device="cuda", dtype=w_dtype)
    candidate = HCNorm(r, warps, registers=registers)
    call = lambda: candidate(r, b, inj, w)
    graph = capture(call)
    for scale in (0.25, 1.0, 3.0):
        r.normal_().mul_(scale)
        b.normal_().mul_(scale)
        inj.normal_()
        # Independent wider-precision oracle, with the same explicit
        # materialized residual boundary; never normalize the unrounded sum.
        gate = 2 * (inj.double() / groups).sigmoid()
        combined = (
            r.double().reshape(rows, groups, width)
            + b.double()[:, None, :] * gate[:, :, None]
        ).to(r_dtype)
        y = combined.double()
        y *= (y.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        y *= 1 + w.double().reshape(1, 1 if shared else groups, width)
        eager = [t.clone() for t in call()]
        candidate.combined.fill_(float("nan"))
        candidate.normalized.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        for actual, ref in zip((candidate.combined, candidate.normalized), eager):
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)
        atol = 2e-3 if r_dtype == torch.float16 else 2e-5
        for actual, ref in zip(eager, (combined, y.to(r_dtype))):
            torch.testing.assert_close(
                actual.reshape(rows, groups, width), ref, atol=atol, rtol=2e-3
            )
