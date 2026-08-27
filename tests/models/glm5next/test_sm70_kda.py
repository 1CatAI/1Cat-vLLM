# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm._sm70_ops as sm70_ops
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.get_device_capability() == DeviceCapability(7, 0)
        and hasattr(torch.ops._C, "sm70_glm_kda_fg_b_out")
    ),
    reason="native NVIDIA V100/SM70 GLM KDA CUDA op required",
)
def test_sm70_glm_kda_fused_fg_b_matches_fp16_and_graph() -> None:
    torch.manual_seed(20260827)
    device = current_platform.device_type
    f_input = torch.randn((1, 128), device=device, dtype=torch.float16).mul_(0.1)
    g_input = torch.randn((1, 128), device=device, dtype=torch.float16).mul_(0.1)
    f_weight = torch.randn((2048, 128), device=device, dtype=torch.float16).mul_(0.01)
    g_weight = torch.randn((2048, 128), device=device, dtype=torch.float16).mul_(0.01)
    f_out = torch.empty((1, 2048), device=device, dtype=torch.float16)
    g_out = torch.empty_like(f_out)

    def run() -> None:
        sm70_ops.sm70_glm_kda_fg_b_out(
            f_out, g_out, f_input, g_input, f_weight, g_weight
        )

    run()
    torch.accelerator.synchronize()
    expected = (F.linear(f_input, f_weight), F.linear(g_input, g_weight))
    torch.testing.assert_close(f_out, expected[0], rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(g_out, expected[1], rtol=2e-3, atol=2e-4)
    eager = (f_out.clone(), g_out.clone())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.accelerator.synchronize()

    torch.testing.assert_close(f_out, eager[0], rtol=0, atol=0)
    torch.testing.assert_close(g_out, eager[1], rtol=0, atol=0)
