# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiled gated prefill keeps both packed layouts and FP16 scale rounding."""

import pytest
import torch


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("m", [1024, 4096])
@torch.inference_mode()
def test_tiled_gate_up_graph_matches_effective_weights(tp, shared, m):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    torch.manual_seed(7101)
    k, n = 128, 34816 // tp
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    raw = torch.randint(1, 8, (n, k // 16), device="cuda").to(torch.float8_e4m3fn)
    global_scale = 0.00314159
    weight, scales = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw)
    effective_scales = (raw.float() * global_scale).half()
    if shared:
        tm_weight, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
            codes.T.contiguous(), effective_scales.T.contiguous(), 16, False
        )
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    out = torch.empty(m, n // 2, device="cuda", dtype=torch.float16)

    def run():
        if shared:
            torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
                out,
                x,
                tm_weight,
                scales,
                global_scale,
                8,
                2,
                tm_scales,
                16,
                int(meta[0]),
                int(meta[1]),
                True,
                1024,
            )
        else:
            torch.ops._C.nvfp4_qpn4_prefill_sm70_out(
                out,
                0,
                x,
                weight.view(k, n // 2),
                scales.view(k // 16, n),
                global_scale,
                True,
                True,
            )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Include both sides of each tile boundary and the final partial tile.
    columns = sorted(
        {0, n // 2 - 1}
        | {
            c
            for boundary in range(4096, n // 2, 4096)
            for c in (boundary - 1, boundary)
        }
    )
    selected = torch.tensor(columns + [c + n // 2 for c in columns], device="cuda")
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    selected_codes = codes[selected].long()
    weights = magnitudes[selected_codes & 7] * torch.where(
        selected_codes & 8 != 0, -1, 1
    )
    weights = (weights * effective_scales[selected].repeat_interleave(16, 1)).half()
    for _ in range(3):
        x.normal_(0, 0.3)
        out.fill_(float("nan"))
        graph.replay()
        reference = (x[::127].double() @ weights.T.double()).half()
        gate, up = reference.chunk(2, -1)
        expected = torch.nn.functional.silu(gate) * up
        actual = out[::127, columns]
        assert torch.isfinite(out).all()
        torch.testing.assert_close(actual, expected, rtol=0.002, atol=2e-5)


@pytest.mark.parametrize("shared", [False, True])
@torch.inference_mode()
def test_tiled_down_projection_graph_column_boundaries(shared):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    torch.manual_seed(7102)
    m, k, n = 4096, 8192, 8192
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    raw = torch.randint(1, 8, (n, k // 16), device="cuda").to(torch.float8_e4m3fn)
    scale = 0.00314159
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    weight, scales = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw)
    if shared:
        tm_weight, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
            codes.T.contiguous(), (raw.float() * scale).T.half().contiguous(), 16, False
        )
    x = torch.zeros(m, k, device="cuda", dtype=torch.float16)
    out = torch.empty(m, n, device="cuda", dtype=torch.float16)

    def run():
        if shared:
            torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
                out,
                x,
                tm_weight,
                scales,
                scale,
                8,
                2,
                tm_scales,
                16,
                int(meta[0]),
                int(meta[1]),
                False,
                1024,
            )
        else:
            torch.ops._C.nvfp4_qpn4_prefill_sm70_out(
                out,
                0,
                x,
                weight.view(k, n // 2),
                scales.view(k // 16, n),
                scale,
                True,
                False,
            )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    # Changed one-hot inputs test every output column, including tile boundaries,
    # with an exact oracle independent of cuBLAS algorithm selection.
    rows = torch.arange(m, device="cuda")
    for offset in (0, k - m):
        columns = rows + offset
        x.zero_()
        x[rows, columns] = 1
        out.fill_(float("nan"))
        graph.replay()
        selected_codes = codes[:, columns].long()
        expected = magnitudes[selected_codes & 7] * torch.where(
            selected_codes & 8 != 0, -1, 1
        )
        expected = (expected * (raw.float() * scale).half()[:, columns // 16]).half()
        torch.testing.assert_close(out, expected.T, rtol=0, atol=0)
