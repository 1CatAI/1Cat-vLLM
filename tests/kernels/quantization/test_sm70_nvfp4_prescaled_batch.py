# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time scale folding must preserve storage and the chosen reduction."""

import pytest
import torch

from vllm.model_executor.layers.quantization.sm70_turbomind import (
    _prescale_nvfp4_batch_scales,
)


def test_nvfp4_scale_folding_all_representable_values():
    positive = torch.arange(0x4400, dtype=torch.int32)
    bits = torch.cat((positive, positive + 0x8000)).to(torch.int16)
    scales = bits.view(torch.float16).clone()
    pointer = scales.data_ptr()
    assert _prescale_nvfp4_batch_scales(scales)
    assert pointer == scales.data_ptr()
    assert torch.isfinite(scales).all()
    assert torch.equal((scales / 16384).view(torch.int16), bits)


@pytest.mark.parametrize("value", [4.0, -4.0, float("inf"), float("nan")])
def test_nvfp4_scale_folding_rejects_without_mutation(value):
    scales = torch.tensor([0.001, value], dtype=torch.float16)
    original = scales.view(torch.int16).clone()
    assert not _prescale_nvfp4_batch_scales(scales)
    assert torch.equal(scales.view(torch.int16), original)


@pytest.mark.parametrize("rows", [33, 64, 65, 128, 1024, 8192])
@pytest.mark.parametrize("gated", [False, True])
def test_nvfp4_prescaled_same_partition_graph(monkeypatch, rows, gated):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    monkeypatch.setenv("VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M", "64")
    import vllm._C  # noqa: F401

    torch.manual_seed(2079)
    k, n = 4352, 4096
    codes = torch.randint(16, (k, n), device="cuda", dtype=torch.uint8)
    values = torch.tensor(
        [0, 2**-24, 2**-14, 0.005, 0.1, 1, 3.998046875],
        device="cuda",
        dtype=torch.float16,
    )
    scales = values[torch.randint(values.numel(), (k // 16, n), device="cuda")]
    weight, packed, meta = torch.ops._C.nvfp4_sm70_prepare(codes, scales, 16, gated)
    shifted = packed.clone()
    assert _prescale_nvfp4_batch_scales(shifted)
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.01
    reference = torch.empty(rows, n // 2 if gated else n, device="cuda", dtype=x.dtype)
    actual = torch.empty_like(reference)

    def call(out, scale, prescaled):
        op = (
            torch.ops._C.nvfp4_gemm_sm70_prescaled_out
            if prescaled
            else torch.ops._C.nvfp4_gemm_sm70_out
        )
        op(out, x, weight, scale, 16, int(meta[0]), int(meta[1]), gated)

    call(reference, packed, False)
    call(actual, shifted, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(actual, shifted, True)
    for amplitude in (0.01, 0.03):
        x.normal_(0, amplitude)
        call(reference, packed, False)
        actual.fill_(float("nan"))
        graph.replay()
        assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))


@pytest.mark.parametrize("gated", [False, True])
def test_nvfp4_prescaled_opaque_dispatch_dynamic_rows(monkeypatch, gated):
    """One compiled operator must retain QPN2 at small M and TM at large M."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    monkeypatch.setenv("VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M", "64")
    from vllm import _sm70_ops as ops

    torch.manual_seed(2081)
    k, n = 4352, 4096
    codes = torch.randint(16, (k, n), device="cuda", dtype=torch.uint8)
    raw_scales = torch.rand(n, k // 16, device="cuda").add_(0.1).to(torch.float8_e4m3fn)
    global_scale = 0.015625
    weight, scales, meta = ops.nvfp4_sm70_prepare(
        codes, (raw_scales.t().float() * global_scale).half().contiguous(), 16, False
    )
    compact = ops.nvfp4_qpn2_prepare_scales_sm70(raw_scales)
    k_ld, q_ld = int(meta[0]), int(meta[1])
    shifted = scales.clone()
    assert _prescale_nvfp4_batch_scales(shifted)

    def call(x, s, prescaled):
        out = torch.empty(
            x.shape[0], n // 2 if gated else n, device=x.device, dtype=x.dtype
        )
        ops.nvfp4_qpn2_tm_dispatch_sm70_out(
            out,
            x,
            weight,
            compact,
            global_scale,
            8 if gated else 16,
            2,
            s,
            16,
            k_ld,
            q_ld,
            gated,
            0,
            prescaled,
        )
        return out

    compiled = torch.compile(call, backend="aot_eager", fullgraph=True, dynamic=True)
    for rows in (64, 8, 16, 32, 33, 65, 128, 1024):
        x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.03
        expected = call(x, scales, False)
        actual = compiled(x, shifted, True)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
