# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch tiles must handle tails and reread inputs during graph replay."""

import pytest
import torch


@pytest.mark.parametrize("rows", [33, 63, 64, 65])
@pytest.mark.parametrize("kind", ["fp4", "fp8_channel", "fp8_block"])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("k,width", [(2048, 2048), (4352, 2112)])
def test_quantized_batch_supply_graph(monkeypatch, rows, kind, gated, k, width):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    # Exercise the normal native registry, with tuning enabled before its first
    # use. N=2112 and M=33/63 cover tails; M=65 retains the established route.
    monkeypatch.setenv("VLLM_SM70_FP8_DENSE_TUNE_MAX_M", "64")
    monkeypatch.setenv("VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M", "64")
    import vllm._C  # noqa: F401

    torch.manual_seed(20260926)
    n = width * (2 if gated else 1)
    columns = torch.arange(32, device="cuda")
    if gated:
        columns = torch.cat((columns, columns + width))
    if kind == "fp4":
        codes = torch.randint(16, (k, n), device="cuda", dtype=torch.uint8)
        scales = (torch.rand(k // 16, n, device="cuda") * 0.02 + 0.002).half()
        weight, packed_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
            codes, scales, 16, gated
        )
        values = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
            dtype=torch.float16,
        )
        effective = (
            values[codes[:, columns].long()]
            * scales[:, columns].repeat_interleave(16, 0)
        ).double()
        op, group_size = torch.ops._C.nvfp4_gemm_sm70_out, 16
    else:
        raw = (torch.randn(n, k, device="cuda") * 16).to(torch.float8_e4m3fn)
        shape = (n, 1) if kind == "fp8_channel" else ((n + 127) // 128, k // 128)
        scales = torch.rand(shape, device="cuda") * 0.005 + 0.002
        weight, packed_scales, meta = torch.ops._C.fp8_sm70_prepare(
            raw, scales, 128, gated
        )
        selected = (
            scales[columns].half()
            if kind == "fp8_channel"
            else scales[columns // 128].half().repeat_interleave(128, 1)
        )
        effective = (raw[columns].half() * selected).double().T
        op, group_size = torch.ops._C.fp8_gemm_sm70_out, 128
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.1
    out = torch.empty(rows, width, device="cuda", dtype=torch.float16)

    def call():
        op(out, x, weight, packed_scales, group_size, int(meta[0]), int(meta[1]), gated)

    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for amplitude in (0.1, 0.3, 0.01):
        x.normal_(0, amplitude)
        call()
        eager = out.clone()
        out.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(out).all()
        assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
        reference = x.double() @ effective
        if gated:
            gate, up = reference.half().double().chunk(2, -1)
            reference = torch.nn.functional.silu(gate).half().double() * up
        actual = out[:, :32].double()
        assert ((actual - reference).norm() / reference.norm()).item() < 0.004
