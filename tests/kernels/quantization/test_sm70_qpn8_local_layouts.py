# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Changed-input graph checks for local TP1/TP2/TP4 projection layouts."""

import pytest
import torch


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("projection", ["out", "qkv", "gate_up"])
@pytest.mark.parametrize("rows", [8, 16, 32])
def test_channel_fp8_qpn8_local_layout_graph(tp, projection, rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    gated = projection == "gate_up"
    k, n = {
        "out": (6144 // tp, 5120),
        "qkv": (5120, 14336 // tp),
        "gate_up": (5120, 34816 // tp),
    }[projection]
    split = 8 if gated or k == 1536 else 16
    torch.manual_seed(1971)
    weight = (torch.randn(n, k, device="cuda") * 16).to(torch.float8_e4m3fn)
    scales = torch.rand(n, 1, device="cuda") * 0.005 + 0.002
    codes, packed_scales = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scales)
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.1
    out = torch.empty(rows, n // 2 if gated else n, device="cuda", dtype=x.dtype)

    def call():
        # No dense workspace: falling back instead of hitting QPN8 must fail.
        torch.ops._C.fp8_qpn8_dispatch_sm70_out(
            out, 0, x, codes, packed_scales, split, 2, False, gated
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    columns = torch.arange(64, device="cuda")
    if gated:
        columns = torch.cat((columns, columns + n // 2))
    effective = (
        (weight[columns].half() / 256) * (scales[columns] * 256).half()
    ).double()
    for scale in (0.1, 0.3):
        x.normal_(0, scale)
        out.fill_(float("nan"))
        graph.replay()
        ref = x.double() @ effective.T
        if gated:
            gate, up = ref.half().double().chunk(2, dim=-1)
            ref = torch.nn.functional.silu(gate).half().double() * up
        actual = out[:, :64].double()
        assert torch.isfinite(out).all()
        relative_l2 = (actual - ref).norm() / ref.norm().clamp_min(1e-12)
        assert float(relative_l2) < 0.001
