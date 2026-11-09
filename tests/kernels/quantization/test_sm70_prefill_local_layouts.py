# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32-accumulated 8K prefill for wider local FP8 projection shards."""

import pytest
import torch


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("projection", ["out", "down", "gate_up"])
def test_block_fp8_cutlass_local_layout_graph(monkeypatch, capfd, tp, projection):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    monkeypatch.setenv("VLLM_SM70_FP8_PREFILL_CUTLASS", "1")
    monkeypatch.setenv("VLLM_SM70_PROFILE_TRACE", "1")
    gated = projection == "gate_up"
    k, n = {
        "out": (6144 // tp, 5120),
        "down": (17408 // tp, 5120),
        "gate_up": (5120, 34816 // tp),
    }[projection]
    torch.manual_seed(9351)
    weight = (torch.randn(n, k, device="cuda") * 16).to(torch.float8_e4m3fn)
    scales = torch.rand(n // 128, k // 128, device="cuda") * 0.005 + 0.002
    packed, packed_scales, meta = torch.ops._C.fp8_sm70_prepare(
        weight, scales, 128, gated
    )
    workspace = torch.empty(k, n, device="cuda", dtype=torch.float16)
    x = torch.randn(8192, k, device="cuda", dtype=torch.float16) * 0.1
    out = torch.empty(8192, n // 2 if gated else n, device="cuda", dtype=x.dtype)

    def call():
        torch.ops._C.fp8_gemm_sm70_prefill_dispatch_out(
            out,
            workspace.data_ptr(),
            x,
            packed,
            packed_scales,
            128,
            int(meta[0]),
            int(meta[1]),
            gated,
            3920,
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    columns = torch.arange(32, device="cuda")
    if gated:
        columns = torch.cat((columns, columns + n // 2))
    # TurboMind first rounds each block scale to FP16, then multiplies the
    # exactly represented FP8 value by that scale in FP16.
    effective = (
        weight[columns].half() * scales[columns // 128].half().repeat_interleave(128, 1)
    ).double()
    sample_rows = torch.arange(0, 8192, 127, device="cuda")
    for sigma in (0.1, 0.3):
        x.normal_(0, sigma)
        out.fill_(float("nan"))
        graph.replay()
        reference = x[sample_rows].double() @ effective.T
        if gated:
            gate, up = reference.half().double().chunk(2, -1)
            reference = torch.nn.functional.silu(gate).half().double() * up
        actual = out[sample_rows, :32].double()
        assert torch.isfinite(out).all()
        relative_l2 = (actual - reference).norm() / reference.norm()
        assert relative_l2.item() < 0.001

    # The native route log is once per shape family; the first case proves
    # this test reaches CUTLASS instead of accepting a silent cuBLAS fallback.
    if tp == 1 and projection == "out":
        assert "CUTLASS projection route M=8192" in capfd.readouterr().err
