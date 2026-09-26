# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protect exact split reduction and graph input lifetime for batched FP4."""

import pytest
import torch


@pytest.mark.parametrize("rows", [9, 15, 16, 17, 24, 31, 32])
@pytest.mark.parametrize("k,width", [(4096, 2048), (4352, 5120), (5120, 4352)])
@pytest.mark.parametrize("gated", [False, True])
def test_qpn2_m32_changed_input_graph(rows, k, width, gated):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    torch.manual_seed(930)
    n = width * (2 if gated else 1)
    codes = torch.randint(16, (k, n), device="cuda", dtype=torch.uint8)
    raw_scales = (torch.rand(n, k // 16, device="cuda") * 4 + 0.125).to(
        torch.float8_e4m3fn
    )
    global_scale = 0.01234567
    weight, scales, meta = torch.ops._C.nvfp4_sm70_prepare(
        codes, (raw_scales.t().float() * global_scale).half().contiguous(), 16, False
    )
    compact = torch.ops._C.nvfp4_qpn2_prepare_scales_sm70(raw_scales)
    x = torch.empty(rows, k, device="cuda", dtype=torch.float16)
    out = torch.empty(rows, width, device="cuda", dtype=x.dtype)
    reference = torch.empty_like(out)

    def call(a=x, y=out):
        torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
            y,
            a,
            weight,
            compact,
            global_scale,
            8 if gated else 16,
            2,
            scales,
            16,
            int(meta[0]),
            int(meta[1]),
            gated,
            0,
        )

    x.normal_(0, 0.1)
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for amplitude in (0.1, 0.3, 0.01, 0.0):
        x.normal_(0, amplitude)
        for start in range(0, rows, 8):
            call(x[start : start + 8], reference[start : start + 8])
        for invoke in (call, graph.replay):
            out.fill_(float("nan"))
            invoke()
            assert torch.isfinite(out).all()
            assert torch.equal(out.view(torch.int16), reference.view(torch.int16))
