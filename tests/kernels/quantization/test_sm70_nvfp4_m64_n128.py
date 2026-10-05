# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M64 activation reuse must retain W4A16 rounding and graph scratch lifetime."""

import pytest
import torch


@pytest.mark.parametrize("gated", [False, True])
@torch.inference_mode()
def test_m64_n128_basis_and_shared_graph_scratch(gated):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    torch.manual_seed(105)
    m, k, width = (64, 5120, 4352) if gated else (64, 4352, 5120)
    n = width * (2 if gated else 1)
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    raw = torch.tensor([0, 2**-9, 0.5, 7, 192, 448], device="cuda")
    raw = raw[torch.randint(0, 6, (n, k // 16), device="cuda")]
    raw = raw.to(torch.float8_e4m3fn)
    compact = torch.ops._C.nvfp4_qpn2_prepare_scales_sm70(raw)
    global_scale = 0.000156947542564
    effective = (raw.float() * global_scale).half()
    tm_weight, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
        codes.T.contiguous(), effective.T.contiguous(), 16, False
    )
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float16,
    )
    weights = (lookup[codes.long()] * effective.repeat_interleave(16, 1)).half()
    x = torch.zeros(m, k, device="cuda", dtype=torch.float16)
    outputs = [
        torch.empty(m, width, device="cuda", dtype=torch.float16) for _ in range(4)
    ]

    def run():
        for out in outputs:
            torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
                out,
                x,
                tm_weight,
                compact,
                global_scale,
                8,
                2,
                tm_scales,
                16,
                int(meta[0]),
                int(meta[1]),
                gated,
                0,
                True,
            )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Cover every K partition and both signs/scales. Replay an earlier graph
    # after another capture to catch stale or overwritten partials.
    second = torch.cuda.CUDAGraph()
    with torch.cuda.graph(second):
        run()
    columns = torch.linspace(0, k - 1, m, device="cuda").long()
    for offset in (0, 7, 31):
        selected = (columns + offset) % k
        x.zero_()
        x[torch.arange(m, device="cuda"), selected] = 1
        second.replay()
        graph.replay()
        expected = weights[:, selected].T.contiguous()
        if gated:
            gate, up = expected.chunk(2, dim=-1)
            expected = torch.nn.functional.silu(gate.float()).half() * up
        for out in outputs:
            torch.testing.assert_close(out, expected, rtol=0, atol=2e-6)
