# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise grouped FP8 preparation and apply against an independent oracle."""

import pytest
import torch

from vllm._sm70.policy import NativeBindings
from vllm.config.kernel import Sm70Fp8Config
from vllm.config.sm70_native import capture_linear_native_config
from vllm.model_executor.kernels.linear.qpn.fp8 import TurboMindFp8LinearKernel
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="requires SM70 native FP8 kernels",
)


@pytest.mark.parametrize("groups", [2, 8])
@pytest.mark.parametrize("tokens", [1, 2, 5, 16, 17])
@torch.no_grad()
def test_grouped_fp8_apply_matches_fp64_and_changed_input_graph(
    monkeypatch, groups, tokens
):
    monkeypatch.setenv("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", "0")
    monkeypatch.setenv("VLLM_SM70_FP8_GROUPED_BMM_DECODE", "1")
    torch.manual_seed(123)
    n, k = 256, 512
    codes = (torch.randn(groups * n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    scales = (
        torch.arange(groups * (n // 128) * (k // 128), device="cuda") % 5
    ).float().reshape(groups * (n // 128), k // 128) / 32 + 0.125
    dense = codes.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    dense = dense.reshape(groups, n, k).double()
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(codes, requires_grad=False))
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scales, requires_grad=False)
    )
    layer.orig_dtype = torch.float16
    layer.is_bmm = True
    layer.bmm_batch_size = groups
    kernel = object.__new__(TurboMindFp8LinearKernel)
    kernel.policy = Sm70Fp8Config(qpn8=False, qpn8_pp2_tp4=False)
    kernel.policy.resolve()
    kernel.native_ops = NativeBindings(capture_linear_native_config("fp8").values)
    kernel.weight_block_size = [128, 128]
    kernel.is_scale_e8m0 = False
    kernel.process_weights_after_loading(layer)
    assert layer.sm70_fp8_bmm_grouped_decode
    x = torch.randn(tokens, groups, k, device="cuda", dtype=torch.float16)
    bias = torch.randn(groups * n, device="cuda", dtype=torch.float16) / 16

    def oracle():
        return (
            torch.einsum("mgk,gnk->mgn", x.double(), dense)
            + bias.reshape(groups, n).double()
        ).half()

    for _ in range(2):
        result = kernel.apply_weights(layer, x, bias)
    assert torch.isfinite(result).all()
    # FP32 accumulation/reordering followed by FP16 output rounding; this also
    # covers the ordinary per-group fallback immediately above the M16 cutoff.
    torch.testing.assert_close(result, oracle(), atol=2e-3, rtol=2e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernel.apply_weights(layer, x, bias)
    for scale in (0.25, -0.5):
        x.mul_(scale)
        graph.replay()
        assert torch.isfinite(captured).all()
        torch.testing.assert_close(captured, oracle(), atol=2e-3, rtol=2e-3)
