# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One native packing must preserve weights and bound graph scratch."""

import pytest
import torch


@pytest.mark.parametrize("rows", [8, 16, 32, 64, 256])
@torch.inference_mode()
def test_native_decode_and_prefill_basis(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    from vllm.model_executor.layers.quantization.utils import (
        sm70_nvfp4_native,  # noqa: F401
    )

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
    effective_scale = (raw_scales.float() * 0.000502813432831).half().double()
    expected = magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    expected = (expected * effective_scale.repeat_interleave(16, 1)).half()
    x = torch.zeros(rows, k, dtype=torch.float16, device="cuda")
    out = torch.empty(rows, n, dtype=torch.float16, device="cuda")

    def run():
        torch.ops.vllm.sm70_nvfp4_native_dispatch(
            out, x, weight, scales, 0.000502813432831, 8, 2, False
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for begin in range(0, k, min(rows, k)):
        columns = (torch.arange(rows, device="cuda") + begin) % k
        x.zero_()
        x[torch.arange(rows, device="cuda"), columns] = 1
        graph.replay()
        torch.testing.assert_close(out, expected[:, columns].T, rtol=0, atol=0)


@torch.inference_mode()
def test_native_prefill_graphs_share_dense_workspace():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    from vllm.model_executor.layers.quantization.utils import (
        sm70_nvfp4_native,  # noqa: F401
    )

    torch.manual_seed(71)
    n = k = 2048
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    raw = torch.ones(n, k // 16, device="cuda").to(torch.float8_e4m3fn)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous()
    weight, scales = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw)
    x = torch.randn(64, k, device="cuda", dtype=torch.float16)
    outputs = [torch.empty(64, n, device="cuda", dtype=x.dtype) for _ in range(24)]

    def run():
        for out in outputs:
            torch.ops.vllm.sm70_nvfp4_native_dispatch(
                out, x, weight, scales, 0.125, 8, 2, False
            )

    run()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # The dense weight scratch is shared, not retained once per captured layer.
    assert torch.cuda.memory_allocated() - before < 2 * 2**20
    reference = torch.empty_like(outputs[0])
    for _ in range(3):
        x.normal_()
        graph.replay()
        torch.ops.vllm.sm70_nvfp4_native_dispatch(
            reference, x, weight, scales, 0.125, 8, 2, False
        )
        for out in outputs:
            torch.testing.assert_close(out, reference, rtol=0, atol=0)
