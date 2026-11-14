# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM70 Marlin dispatch adds the bias after the GEMM.

Callers pass the bias in Marlin's column order (marlin_permute_bias) while the
SM70 kernels write logical columns, so the dispatch has to undo that order.
The weights are packed the way the loader packs them (gptq_marlin_repack and
awq_marlin_repack), which is the layout the SM70 kernels read.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_zero_points,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    awq_pack,
    gptq_pack,
    gptq_quantize_weights,
    quantize_weights,
)
from vllm.scalar_type import ScalarType, scalar_types

pytestmark = pytest.mark.skipif(
    not ops.sm70_marlin_available(), reason="Needs the SM70 Marlin build."
)

SIZE_K, SIZE_N = 1024, 2048


def _pack(kind: str, group_size: int):
    device = torch.device("cuda")
    weight = torch.randn((SIZE_K, SIZE_N), dtype=torch.float16, device=device)
    empty = torch.empty(0, dtype=torch.int32, device=device)
    if kind == "fp8":
        weight_ref, q_weight, scales = marlin_quant_fp8_torch(weight.T, group_size)
        return scalar_types.float8_e4m3fn, weight_ref, q_weight, scales, None
    groups = SIZE_K if group_size == -1 else group_size
    if kind == "uint4b8":
        quant_type = scalar_types.uint4b8
        weight_ref, q_w, scales, _, _ = gptq_quantize_weights(
            weight, quant_type, groups, False
        )
        q_weight = ops.gptq_marlin_repack(
            gptq_pack(q_w, 4, SIZE_K, SIZE_N), empty, SIZE_K, SIZE_N, 4
        )
        zero_points = None
    else:
        quant_type = scalar_types.uint4
        weight_ref, q_w, scales, zp = quantize_weights(
            weight, quant_type, groups, zero_points=True
        )
        q_weight = ops.awq_marlin_repack(
            awq_pack(q_w, 4, SIZE_K, SIZE_N), SIZE_K, SIZE_N, 4
        )
        zero_points = marlin_zero_points(zp, SIZE_K // groups, SIZE_N, 4)
    scales = marlin_permute_scales(scales, SIZE_K, SIZE_N, group_size)
    return quant_type, weight_ref, q_weight, scales, zero_points


def _gemm(
    quant_type: ScalarType,
    activation: torch.Tensor,
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return ops.marlin_gemm(
        activation,
        None,
        q_weight,
        bias,
        scales,
        None,
        None,
        zero_points,
        None,
        None,
        marlin_make_workspace_new(activation.device),
        quant_type,
        activation.shape[0],
        SIZE_N,
        SIZE_K,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )


def _relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    difference = torch.mean(torch.abs(output.float() - reference.float()))
    return (difference / torch.mean(torch.abs(reference.float()))).item()


@pytest.mark.parametrize(
    "kind,group_size",
    [("fp8", -1), ("fp8", 128), ("uint4b8", 128), ("uint4b8", -1), ("uint4", 128)],
)
@pytest.mark.parametrize("size_m", [1, 16, 256])
def test_sm70_marlin_bias_is_added_in_logical_columns(
    kind: str, group_size: int, size_m: int
) -> None:
    torch.manual_seed(20261114)
    quant_type, weight_ref, q_weight, scales, zero_points = _pack(kind, group_size)
    activation = torch.randn((size_m, SIZE_K), dtype=torch.float16, device="cuda")
    bias = torch.randn(SIZE_N, dtype=torch.float16, device="cuda") * 10

    without_bias = _gemm(quant_type, activation, q_weight, scales, zero_points, None)
    with_bias = _gemm(
        quant_type, activation, q_weight, scales, zero_points, marlin_permute_bias(bias)
    )
    unpermuted = _gemm(quant_type, activation, q_weight, scales, zero_points, bias)

    # The bias is added in fp16 to the finished output.
    expected = (without_bias.float() + bias.float()).half()
    torch.testing.assert_close(with_bias, expected, rtol=0, atol=0)
    # Without Marlin's permutation the columns get the wrong bias.
    assert not torch.equal(unpermuted, expected)
    reference = torch.matmul(activation, weight_ref) + bias
    assert _relative_error(with_bias, reference) < 0.04


def test_sm70_marlin_bias_survives_graph_replay() -> None:
    torch.manual_seed(20261115)
    quant_type, weight_ref, q_weight, scales, _ = _pack("fp8", 128)
    activation = torch.randn((16, SIZE_K), dtype=torch.float16, device="cuda")
    bias = torch.randn(SIZE_N, dtype=torch.float16, device="cuda") * 10
    marlin_bias = marlin_permute_bias(bias)

    _gemm(quant_type, activation, q_weight, scales, None, marlin_bias)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _gemm(quant_type, activation, q_weight, scales, None, marlin_bias)

    for _ in range(3):
        activation.copy_(torch.randn_like(activation))
        graph.replay()
        eager = _gemm(quant_type, activation, q_weight, scales, None, marlin_bias)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
