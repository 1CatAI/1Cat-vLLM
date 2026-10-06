# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One native packing must preserve weights and bound graph scratch."""

import pytest
import torch


@pytest.mark.parametrize("rows", [1, 8, 16, 24, 32, 64, 256])
@pytest.mark.parametrize("bundled", [False, True])
@torch.inference_mode()
def test_native_decode_and_prefill_basis(rows, bundled):
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
    if bundled:
        weight, scales = torch.ops._C.nvfp4_qpn2_bundle_sm70(weight, scales)
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


@pytest.mark.parametrize("bundled", [False, True])
@torch.inference_mode()
def test_native_prefill_graphs_share_dense_workspace(bundled):
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
    if bundled:
        weight, scales = torch.ops._C.nvfp4_qpn2_bundle_sm70(weight, scales)
    x = torch.randn(64, k, device="cuda", dtype=torch.float16)
    outputs = [torch.empty(64, n, device="cuda", dtype=x.dtype) for _ in range(24)]
    # Distinct packed scales expose accidental reuse of another layer's
    # compact operands. Keep the group factors exactly representable.
    layer_scales = [scales.clone() for _ in outputs]
    if bundled:
        layer_weights = [
            torch.ops._C.nvfp4_qpn2_bundle_sm70(
                weight.contiguous().view(n, k // 2),
                s.contiguous().view(n, k // 16),
            )
            for s in layer_scales
        ]
    else:
        layer_weights = [(weight, s) for s in layer_scales]
    for i, (_, s) in enumerate(layer_weights):
        s.fill_(0x38 + i % 8)

    def run():
        for out, (w, s) in zip(outputs, layer_weights):
            torch.ops.vllm.sm70_nvfp4_native_dispatch(out, x, w, s, 0.125, 8, 2, False)

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
        for out, (w, s) in zip(outputs, layer_weights):
            torch.ops.vllm.sm70_nvfp4_native_dispatch(
                reference, x, w, s, 0.125, 8, 2, False
            )
            torch.testing.assert_close(out, reference, rtol=0, atol=0)


@torch.inference_mode()
def test_bundled_storage_and_invalid_views():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    torch.manual_seed(79)
    n, k = 64, 128
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(0, 127, (n, k // 16), dtype=torch.uint8, device="cuda")
    c, s = torch.ops._C.nvfp4_qpn2_bundle_sm70(codes, scales)
    assert c.stride() == (k // 16 * 288, 288, 1)
    assert s.data_ptr() == c.data_ptr() + 256
    assert c.untyped_storage().data_ptr() == s.untyped_storage().data_ptr()
    assert c.untyped_storage().nbytes() == codes.numel() + scales.numel()
    assert torch.equal(c.contiguous().view_as(codes), codes)
    assert torch.equal(s.contiguous().view_as(scales), scales)

    x = torch.zeros(8, k, dtype=torch.float16, device="cuda")
    out = torch.empty(8, n, dtype=x.dtype, device=x.device)
    # A clone with matching shape is not the adjacent scale view.
    with pytest.raises(RuntimeError, match="contiguous or bundled views"):
        torch.ops._C.nvfp4_qpn2_gemm_sm70_out(out, x, c, s.clone(), 0.125, 8, 2)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        torch.ops._C.nvfp4_qpn2_bundle_sm70(codes, scales[:, :1].contiguous())


@pytest.mark.parametrize("rows", [16, 24, 32])
@pytest.mark.parametrize("gated", [False, True])
@torch.inference_mode()
def test_native_batches_preserve_turbomind_reduction(rows, gated):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    torch.manual_seed(73)
    n = k = 4096
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    raw = torch.randint(1, 8, (n, k // 16), device="cuda").to(torch.float8_e4m3fn)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous()
    native, compact = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw)
    weight, scales, meta = torch.ops._C.nvfp4_sm70_prepare(
        codes.T.contiguous(), (raw.float() * 0.125).T.half().contiguous(), 16, False
    )
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.125
    out = torch.empty(rows, n // 2 if gated else n, device="cuda", dtype=x.dtype)
    reference = torch.empty_like(out)
    split = 8 if gated else 16
    torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
        reference,
        x,
        weight,
        compact,
        0.125,
        split,
        2,
        scales,
        16,
        int(meta[0]),
        int(meta[1]),
        gated,
        256,
    )
    op = (
        torch.ops._C.nvfp4_qpn2_gated_sm70_out
        if gated
        else torch.ops._C.nvfp4_qpn2_gemm_sm70_out
    )
    op(out, x, native, compact, 0.125, split, 2)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    bundled, bundled_scales = torch.ops._C.nvfp4_qpn2_bundle_sm70(native, compact)
    op(out, x, bundled, bundled_scales, 0.125, split, 2)
    assert torch.equal(out.view(torch.int16), reference.view(torch.int16))
