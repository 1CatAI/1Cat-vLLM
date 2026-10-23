# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the M8 paired route against the unchanged separate-layout kernel."""

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)
@pytest.mark.parametrize(("rows", "chains"), [(7, 1), (8, 1), (8, 2), (32, 2)])
def test_paired_gated_graph_preserves_bits_and_fallbacks(rows, chains):
    from vllm import _sm70_ops as ops

    torch.manual_seed(123)
    weight = torch.randint(256, (8704, 2560), device="cuda", dtype=torch.uint8)
    scales = torch.randint(1, 120, (8704, 320), device="cuda", dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    codes, packed_scales = ops.nvfp4_qpn2_prepare_sm70(weight, scales)
    bundled_codes, bundled_scales = ops.nvfp4_qpn2_bundle_sm70(codes, packed_scales)
    x = torch.empty(rows, 5120, device="cuda", dtype=torch.float16)
    actual = torch.empty(rows, 4352, device="cuda", dtype=torch.float16)
    reference = torch.empty_like(actual)

    def call():
        ops.nvfp4_qpn2_gated_sm70_out(
            actual, x, bundled_codes, bundled_scales, 1 / 6400, 8, chains
        )
        ops.nvfp4_qpn2_gated_sm70_out(
            reference, x, codes, packed_scales, 1 / 6400, 8, chains
        )

    x.normal_(0, 0.125)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.normal_(0, amplitude)
        actual.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(actual).all()
        assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
