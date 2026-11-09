# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw-scale expansion and decode admission must cover E4M3 edge encodings."""

import pytest
import torch

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    _raw_scales_match_prepared,
)


@pytest.fixture(autouse=True)
def require_sm70_raw_scale_ops():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    if not sm70_ops.has_nvfp4_qpn_raw_scale_dispatch():
        pytest.skip("requires raw-scale native operators")


@pytest.mark.parametrize(
    "stage,interleaved", [("w13", False), ("w13", True), ("w2", False)]
)
def test_raw_scale_strict_expansion_all_e4m3_codes(stage, interleaved):
    shape = (512, 160, 320) if stage == "w13" else (512, 10, 2560)
    # CPU conversion is an independent reference, including signed zeros,
    # subnormals and NaNs; no SM70 native FP8 arithmetic is assumed.
    codebook = torch.arange(256, dtype=torch.uint8)
    values = codebook.view(torch.float8_e4m3fn).float().cuda()
    codes = codebook.cuda().repeat(torch.Size(shape).numel() // 256).view(shape)
    globals_ = (
        torch.tensor(
            [0.0, 2**-24, 2**-10, 2**-4, 1.0, 2.0, -1.0, 256.0],
            device="cuda",
            dtype=torch.float32,
        )
        .repeat(128 if stage == "w13" else 64)
        .view(512, -1)
    )
    if stage == "w13":
        columns = torch.arange(shape[-1], device="cuda")
        slots = ((columns // 32) % 2) if interleaved else (columns >= 160).long()
        global_by_column = globals_[:, slots].unsqueeze(1)
    else:
        global_by_column = globals_.unsqueeze(1)
    expected = (values[codes.long()] * global_by_column).half()
    out = torch.empty(shape, dtype=torch.float16, device="cuda")
    sm70_ops.nvfp4_expand_raw_scales_sm70_out(out, codes, globals_, interleaved, False)
    finite = ~expected.isnan()
    # Bitwise comparison also preserves signed zero; NaN payload is immaterial.
    assert torch.equal(
        out.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )
    assert torch.equal(out.isnan(), expected.isnan())

    # Replay must read current code/global buffers, not capture-time values.
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        sm70_ops.nvfp4_expand_raw_scales_sm70_out(
            out, codes, globals_, interleaved, False
        )
    codes.fill_(64)  # E4M3 2.0
    globals_.fill_(2**-4)
    graph.replay()
    assert torch.equal(out, torch.full_like(out, 0.125))


@pytest.mark.parametrize("code", [0, 1, 7, 128, 129, 255])
def test_raw_scale_admission_rejects_unsupported_fast_encodings(code):
    shape = (512, 10, 2560)
    codes = torch.full(shape, code, dtype=torch.uint8, device="cuda")
    globals_ = torch.full((512, 1), 2**-10, dtype=torch.float32, device="cuda")
    prepared = torch.empty(shape, dtype=torch.float16, device="cuda")
    sm70_ops.nvfp4_expand_raw_scales_sm70_out(prepared, codes, globals_, False, False)
    assert not _raw_scales_match_prepared(
        torch.empty_like(prepared), codes, globals_, prepared, False
    )


def test_raw_scale_admission_checks_effective_hmma_scale():
    shape = (512, 10, 2560)
    codes = torch.full(shape, 64, dtype=torch.uint8, device="cuda")
    globals_ = torch.full((512, 1), 2**-10, dtype=torch.float32, device="cuda")
    prepared = torch.full(shape, 2**-9, dtype=torch.float16, device="cuda")
    out = torch.empty_like(prepared)
    assert _raw_scales_match_prepared(out, codes, globals_, prepared, False)
    assert torch.equal(out, prepared * 16384.0)
