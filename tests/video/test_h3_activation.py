# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.models.minimax_h3.activation import silu_prepare_fp16
from vllm.model_executor.models.minimax_h3.quantization import (
    DiffusionInt8ConvRotConfig,
    Int8ConvRotLayerConfig,
    Int8ConvRotLinearMethod,
    fp16_gemm_input,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires SM70 GPU"
)


def activation():
    with set_current_vllm_config(VllmConfig()):
        return SiluAndMul()


@pytest.mark.parametrize(
    "rows,channels,multiplier", [(17, 257, 1), (129, 3584, 1000), (12352, 3584, 10)]
)
def test_fused_silu_preserves_prepared_values_and_scale(rows, channels, multiplier):
    torch.manual_seed(42)
    hidden = torch.randn(rows, 2 * channels, device="cuda", dtype=torch.float16)
    hidden *= multiplier
    hidden[0].zero_()
    expected = fp16_gemm_input(activation()(hidden.float()))
    actual = silu_prepare_fp16(hidden)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual[0].dtype == torch.float16 and actual[1].dtype == torch.float32
    assert torch.isfinite(actual[0]).all()


def test_fused_silu_prepared_projection_restores_scale_before_reduction():
    torch.manual_seed(123)
    channels = 512
    hidden = torch.randn(2, 33, 2 * channels, device="cuda", dtype=torch.float16)
    hidden *= 1000
    method = Int8ConvRotLinearMethod(
        DiffusionInt8ConvRotConfig(weight_layout="column"),
        Int8ConvRotLayerConfig(True),
        prefix="blocks.0.mlp.fc2",
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.randint(-128, 128, (256, channels), device="cuda", dtype=torch.int8),
            False,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            torch.linspace(1e-4, 1e-2, 256, device="cuda"),
            False,
        ),
    )
    layer.h3_output_fp32 = True
    method.process_weights_after_loading(layer)
    expected = method.apply(layer, activation()(hidden.float()))
    actual = method.apply_prepared(layer, *silu_prepare_fp16(hidden))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (2, 33, 256) and actual.dtype == torch.float32


def test_fused_silu_graph_replay_changes_values_and_scale():
    hidden = torch.randn(129, 7168, device="cuda", dtype=torch.float16)
    act = activation()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        silu_prepare_fp16(hidden)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = silu_prepare_fp16(hidden)
    hidden.mul_(1000)
    graph.replay()
    expected = fp16_gemm_input(act(hidden.float()))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
