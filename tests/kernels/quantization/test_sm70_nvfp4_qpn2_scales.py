# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QPN2 must preserve the effective NVFP4 weights used by TurboMind."""

import pytest
import torch


@pytest.mark.parametrize(
    "global_scale", [0.000156947542564, 0.000502813432831, 2**-20, 0.5]
)
@pytest.mark.parametrize("rows", [8, 16, 32])
@pytest.mark.parametrize("shared", [False, True])
def test_qpn2_basis_vectors_preserve_dequantized_weights(global_scale, rows, shared):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    n, k = 32, 128
    codes = torch.arange(n * k, device="cuda").reshape(n, k).remainder(16)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    raw_scales = (
        torch.tensor([0, 2**-9, 0.5, 1.5, 7, 24, 192, 448], device="cuda")
        .to(torch.float8_e4m3fn)
        .repeat(n, 1)
    )
    weight, scales = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw_scales)
    magnitudes = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda", dtype=torch.float64
    )
    # Independent codebook and the existing W4A16 contract: FP32 global/group
    # scale product rounded once to FP16, followed by FP16 weight rounding.
    effective_scales = (raw_scales.float() * global_scale).half().double()
    expected = magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    expected = (expected * effective_scales.repeat_interleave(16, 1)).half()
    if shared:
        tm_weight, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
            codes.T.to(torch.uint8).contiguous(),
            effective_scales.T.half().contiguous(),
            16,
            False,
        )
        k_ld, q_ld = int(meta[0]), int(meta[1])
    x = torch.zeros(rows, k, dtype=torch.float16, device="cuda")
    out = torch.empty(rows, n, dtype=torch.float16, device="cuda")

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
                k_ld,
                q_ld,
                False,
                0,
            )
        else:
            torch.ops._C.nvfp4_qpn2_gemm_sm70_out(
                out, x, weight, scales, global_scale, 8, 2
            )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for begin in range(0, k, rows):
        columns = torch.arange(begin, begin + rows, device="cuda")
        x.zero_()
        x[torch.arange(rows, device="cuda"), columns] = 1
        graph.replay()
        torch.testing.assert_close(out, expected[:, columns].T, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [64, 256])
@torch.inference_mode()
def test_compact_scales_reuse_graph_scratch_without_changing_outputs(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    torch.manual_seed(71)
    n = k = 2048
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    raw = torch.randint(1, 8, (n, k // 16), device="cuda").to(torch.float8_e4m3fn)
    compact = torch.ops._C.nvfp4_qpn2_prepare_scales_sm70(raw)
    tm_weight, scales, meta = torch.ops._C.nvfp4_sm70_prepare(
        codes.T.contiguous(), (raw.float() * 0.125).T.half().contiguous(), 16, False
    )
    _, scales2, _ = torch.ops._C.nvfp4_sm70_prepare(
        codes.T.contiguous(), (raw.float() * 0.25).T.half().contiguous(), 16, False
    )
    k_ld, q_ld = int(meta[0]), int(meta[1])
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    outputs = [
        torch.empty(rows, n, device="cuda", dtype=torch.float16) for _ in range(24)
    ]
    references = [torch.empty_like(outputs[0]) for _ in range(2)]

    def run():
        for i, out in enumerate(outputs):
            torch.ops._C.nvfp4_qpn2_compact_tm_gemm_sm70_out(
                out, x, tm_weight, compact, 0.125 * (1 + i % 2), k_ld, q_ld, False
            )

    run()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # One scale matrix per stream/shape, not one per layer in the captured graph.
    assert torch.cuda.memory_allocated() - before < 4 * scales.nbytes + 2 * 2**20
    for _ in range(3):
        x.normal_()
        graph.replay()
        for reference, scale in zip(references, [scales, scales2]):
            torch.ops._C.nvfp4_gemm_sm70_out(
                reference, x, tm_weight, scale, 16, k_ld, q_ld, False
            )
        for i, out in enumerate(outputs):
            torch.testing.assert_close(out, references[i % 2], rtol=0, atol=0)
