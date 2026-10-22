# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused M9-M32 b/a route against the QPN8 projection plus a dense GEMM."""

import pytest
import torch

K, N, QKV_N, BA_N = 5120, 4096, 2560, 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("rows", [9, 12, 16, 17, 24, 31, 32])
def test_fused_batch_matches_separate_projections(rows):
    op = getattr(torch.ops._C, "fp8_qpn8_dispatch_ba_split_sm70_out", None)
    if op is None or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires the SM70 native ops")
    torch.manual_seed(5)
    device = "cuda"
    codes = torch.randint(0, 0x60, (K, N), dtype=torch.uint8, device=device)
    scales = (torch.rand(1, N, device=device) * 0.01 + 0.001).to(torch.float16)
    ba_weight = (torch.randn(BA_N, K, device=device) * 0.02).to(torch.float16)
    x = (torch.randn(rows, K, device=device) * 0.5).to(torch.float16)
    qkv, z, b, a = (
        x.new_full((rows, n), float("nan")) for n in (QKV_N, N - QKV_N, 12, 12)
    )
    op(
        qkv,
        z,
        b,
        a,
        x.new_empty((rows, N)),
        x.new_empty((rows, BA_N)),
        0,
        x,
        codes,
        scales,
        ba_weight,
    )
    expected = x.new_empty((rows, N))
    torch.ops._C.fp8_qpn8_gemm_sm70_out(expected, x, codes, scales, 16, 2, True, False)
    torch.cuda.synchronize()
    assert torch.equal(qkv, expected[:, :QKV_N])
    assert torch.equal(z, expected[:, QKV_N:])
    expected_ba = x.float() @ ba_weight.float().t()
    error = (torch.cat([b, a], dim=1).float() - expected_ba).abs().max()
    # One FP16 rounding of the largest value, plus the accumulation order.
    assert error <= 2 ** -10 * expected_ba.abs().max()
