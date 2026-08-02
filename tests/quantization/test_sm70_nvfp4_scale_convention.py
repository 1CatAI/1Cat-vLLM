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
        torch.tensor([1.0 / 6048.0], dtype=torch.float32),
    )
    big = _layer(
        torch.tensor([[22.0, 40.0]], dtype=torch.float32),
        torch.tensor([1.0 / 6048.0], dtype=torch.float32),
    )
    assert tm.nvfp4_block_scales_are_tiny(tiny) is True
    assert tm.nvfp4_block_scales_are_tiny(big) is False


def test_combine_skips_global_for_tiny_block_scales():
    """Medium convention: effective scale ≈ block (not block/G)."""
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    # After CT reciprocal: global is already 1/disk_global.
    g_inv = torch.tensor([1.0 / 6048.0], dtype=torch.float32)
    tiny_block = torch.tensor([[0.0037, 0.0040]], dtype=torch.float32)
    layer = _layer(tiny_block, g_inv)

    combined = tm.combine_nvfp4_scales_for_sm70_tm(layer)
    # weight_scale is transposed in combine; input was [1, 2] -> t -> [2, 1]
    assert combined.dtype == torch.float16
    # Must NOT be tiny_block * g_inv (~6e-7)
    expected = tiny_block.t().to(torch.float16)
    assert torch.allclose(combined.float(), expected.float(), rtol=1e-3, atol=1e-5)
    assert float(combined.abs().max()) > 1e-3


def test_combine_multiplies_global_for_aggressive_block_scales():
    """Aggressive convention: effective = block * (1/disk_global)."""
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    g_inv = torch.tensor([1.0 / 6048.0], dtype=torch.float32)
    big_block = torch.tensor([[22.0, 40.0]], dtype=torch.float32)
    layer = _layer(big_block, g_inv)

    combined = tm.combine_nvfp4_scales_for_sm70_tm(layer)
    expected = (big_block.t().float() * g_inv).to(torch.float16)
    assert torch.allclose(combined.float(), expected.float(), rtol=1e-3, atol=1e-6)
    # Same order of magnitude as Medium's tiny block scales (~0.003–0.007)
    assert 1e-4 < float(combined.abs().mean()) < 0.1


def test_medium_and_aggressive_effective_scales_align():
    """Sanity: after combine, both conventions land near the same magnitude."""
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    g_inv = torch.tensor([1.0 / 6048.0], dtype=torch.float32)
    # Field measurement: aggressive ≈ medium_disk_block * disk_global
    # After CT: med uses tiny blocks; agg uses large blocks * g_inv.
    med = _layer(torch.full((4, 8), 0.0037), g_inv)
    agg = _layer(torch.full((4, 8), 22.0), g_inv)
    c_med = tm.combine_nvfp4_scales_for_sm70_tm(med).float().mean().item()
    c_agg = tm.combine_nvfp4_scales_for_sm70_tm(agg).float().mean().item()
    # Within ~20% (fp8 quantization noise on real weights is larger)
    ratio = c_med / c_agg
    assert 0.5 < ratio < 2.0, (c_med, c_agg, ratio)


def test_dequant_global_is_one_for_tiny_blocks():
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    g_inv = torch.tensor([1.0 / 6048.0], dtype=torch.float32)
    layer = _layer(torch.tensor([[0.01]], dtype=torch.float32), g_inv)
    g = tm.resolve_nvfp4_global_for_dequant(layer)
    assert g.numel() == 1
    assert abs(float(g.item()) - 1.0) < 1e-6


def test_dequant_global_preserved_for_large_blocks():
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    g_inv = torch.tensor([1.0 / 6048.0], dtype=torch.float32)
    layer = _layer(torch.tensor([[22.0]], dtype=torch.float32), g_inv)
    g = tm.resolve_nvfp4_global_for_dequant(layer)
    assert abs(float(g.item()) - float(g_inv.item())) < 1e-9
