# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 NVFP4 block/global scale convention (Aggressive vs Medium GPTQ).

Aggressive CT exports use large FP8 block scales and a large disk global
(CT then stores global as 1/disk). Medium/TC GPTQ leaves tiny block scales;
multiplying by inverted global underflows. These tests drive the real
combine helpers.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter


def _layer(block: torch.Tensor, global_scale: torch.Tensor) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight_scale = Parameter(block, requires_grad=False)
    layer.weight_global_scale = Parameter(global_scale, requires_grad=False)
    return layer


def test_tiny_block_scales_detected():
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    tiny = _layer(
        torch.tensor([[0.0037, 0.01]], dtype=torch.float32),
        torch.tensor([6048.0], dtype=torch.float32),
    )
    big = _layer(
        torch.tensor([[22.0, 40.0]], dtype=torch.float32),
        torch.tensor([6048.0], dtype=torch.float32),
    )
    assert tm.nvfp4_block_scales_are_tiny(tiny) is True
    assert tm.nvfp4_block_scales_are_tiny(big) is False


def test_normalize_sets_global_one_for_tiny_blocks_leaves_scale():
    """Claude/g1-fix: set global=1.0, never weight_scale *= global (fp8 overflow)."""
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    block = torch.tensor([[0.078125, 0.01]], dtype=torch.float32)
    layer = _layer(block.clone(), torch.tensor([6048.0], dtype=torch.float32))
    tm.normalize_nvfp4_global_scale_for_sm70(layer)
    assert abs(float(layer.weight_global_scale.item()) - 1.0) < 1e-6
    # block scales untouched
    assert torch.equal(layer.weight_scale.data, block)


def test_normalize_reciprocates_global_for_aggressive_blocks():
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    layer = _layer(
        torch.tensor([[22.0, 40.0]], dtype=torch.float32),
        torch.tensor([6048.0], dtype=torch.float32),
    )
    tm.normalize_nvfp4_global_scale_for_sm70(layer)
    assert abs(float(layer.weight_global_scale.item()) - (1.0 / 6048.0)) < 1e-9


def test_combine_after_normalize_aligns_conventions():
    """After normalize, both conventions land near the same TM effective scale."""
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    med = _layer(torch.full((4, 8), 0.0037), torch.tensor([6048.0]))
    agg = _layer(torch.full((4, 8), 22.0), torch.tensor([6048.0]))
    tm.normalize_nvfp4_global_scale_for_sm70(med)
    tm.normalize_nvfp4_global_scale_for_sm70(agg)
    c_med = tm.combine_nvfp4_scales_for_sm70_tm(med).float().mean().item()
    c_agg = tm.combine_nvfp4_scales_for_sm70_tm(agg).float().mean().item()
    ratio = c_med / c_agg
    assert 0.5 < ratio < 2.0, (c_med, c_agg, ratio)


def test_combine_safety_net_if_caller_skipped_normalize():
    """If global is still 1/disk (tiny) with tiny blocks, skip multiply."""
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    g_inv = torch.tensor([1.0 / 6048.0], dtype=torch.float32)
    tiny_block = torch.tensor([[0.0037, 0.0040]], dtype=torch.float32)
    layer = _layer(tiny_block, g_inv)
    combined = tm.combine_nvfp4_scales_for_sm70_tm(layer)
    expected = tiny_block.t().to(torch.float16)
    assert torch.allclose(combined.float(), expected.float(), rtol=1e-3, atol=1e-5)


def test_dequant_global_is_one_when_unnormalized_tiny():
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    g_inv = torch.tensor([1.0 / 6048.0], dtype=torch.float32)
    layer = _layer(torch.tensor([[0.01]], dtype=torch.float32), g_inv)
    g = tm.resolve_nvfp4_global_for_dequant(layer)
    assert abs(float(g.item()) - 1.0) < 1e-6
