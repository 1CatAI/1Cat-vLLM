# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import vllm._C  # noqa: F401


@pytest.mark.parametrize(
    "width,columns,bundled", [(5120, 62080, False), (4352, 5120, True)]
)
@pytest.mark.parametrize("rows", [1, 3, 5, 7, 8])
@pytest.mark.parametrize("global_scale", [0.0, 1e-12, 1e-4, 0.01])
def test_qpn2_mtp_native_matches_padded_rows(
    rows, width, columns, bundled, global_scale
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    if not hasattr(torch.ops._C, "nvfp4_qpn2_bundle_sm70"):
        pytest.skip("requires the QPN2 native layout operators")
    torch.manual_seed(62147)
    packed = torch.randint(
        0, 256, (columns, width // 2), device="cuda", dtype=torch.uint8
    )
    scale_bytes = torch.randint(
        0, 127, (columns, width // 16), device="cuda", dtype=torch.uint8
    )
    codes, scales = torch.ops._C.nvfp4_qpn2_prepare_sm70(
        packed, scale_bytes.view(torch.float8_e4m3fn)
    )
    if bundled:
        codes, scales = torch.ops._C.nvfp4_qpn2_bundle_sm70(codes, scales)
    x = torch.randn(16, width, device="cuda", dtype=torch.float16) * 0.1
    reference = torch.empty(16, columns, device="cuda", dtype=torch.float16)
    output = torch.empty(rows, columns, device="cuda", dtype=torch.float16)
    operation = torch.ops._C.nvfp4_qpn2_gemm_sm70_out
    operation(reference, x, codes, scales, global_scale, 16, 2)
    operation(output, x[:rows], codes, scales, global_scale, 16, 2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation(output, x[:rows], codes, scales, global_scale, 16, 2)
    for _ in range(3):
        x.normal_().mul_(0.1)
        operation(reference, x, codes, scales, global_scale, 16, 2)
        graph.replay()
        assert torch.equal(output.view(torch.int16), reference[:rows].view(torch.int16))
