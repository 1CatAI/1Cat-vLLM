# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.models.z_image.precision import (
    FP32FeedForward,
    FP32OutputLinear,
    preserve_projection_range,
)


def test_projection_preserves_outliers_for_following_normalization():
    linear = torch.nn.Linear(4, 2, bias=False, dtype=torch.float16)
    linear.weight.data.fill_(400)
    value = torch.full((1, 3, 4), 400, dtype=torch.float16)
    result = FP32OutputLinear(linear)(value)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.full((1, 3, 2), 640000.0))
    assert not torch.isfinite(result.half()).all()


def test_gating_retains_values_above_fp16_range():
    original = torch.nn.Module()
    original.w1 = torch.nn.Linear(2, 2, bias=False, dtype=torch.float16)
    original.w2 = torch.nn.Linear(2, 2, bias=False, dtype=torch.float16)
    original.w3 = torch.nn.Linear(2, 2, bias=False, dtype=torch.float16)
    for layer in (original.w1, original.w2, original.w3):
        layer.weight.data.copy_(torch.eye(2))
    value = torch.tensor([[[400.0, -400.0]]], dtype=torch.float16)
    result = FP32FeedForward(original)(value)
    expected = torch.nn.functional.silu(value.float()) * value.float()
    assert torch.equal(result, expected)
    assert result[0, 0, 0] > torch.finfo(torch.float16).max


def test_qk_projection_is_normalized_before_returning_to_half():
    from diffusers.models.normalization import RMSNorm

    block = torch.nn.Module()
    block.attention = attention = torch.nn.Module()
    attention.to_out = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    for projection, norm in (("to_q", "norm_q"), ("to_k", "norm_k")):
        linear = torch.nn.Linear(2, 2, bias=False, dtype=torch.float16)
        linear.weight.data.fill_(400)
        setattr(attention, projection, linear)
        setattr(attention, norm, RMSNorm(2, eps=1e-5).half())
    preserve_projection_range(block)
    value = torch.full((1, 3, 2), 400, dtype=torch.float16)
    for projection, norm in (("to_q", "norm_q"), ("to_k", "norm_k")):
        projected = getattr(attention, projection)(value)
        assert projected.max() > torch.finfo(torch.float16).max
        normalized = getattr(attention, norm)(projected)
        assert normalized.dtype == torch.float16
        assert torch.equal(normalized, torch.ones_like(value))


def test_base_retains_modulation_and_residual_range_through_final_layer():
    from diffusers.models.transformers.transformer_z_image import (
        FinalLayer,
        ZImageTransformerBlock,
    )

    torch.manual_seed(42)
    block = ZImageTransformerBlock(0, 16, 1, 1, 1e-5, True).half()
    final = FinalLayer(16, 4).half()
    block.attention_norm1.weight.data.fill_(500)
    block.adaLN_modulation[0].weight.data.zero_()
    block.adaLN_modulation[0].bias.data.fill_(500)
    value = torch.randn(1, 4, 16).half()
    conditioning = torch.ones(1, 16).half()
    frequencies = torch.ones(1, 4, 8, dtype=torch.complex64)
    assert not torch.isfinite(block.attention_norm1(value) * 501).all()
    preserve_projection_range(torch.nn.ModuleList([block, final]), attention_fp32=True)
    result = final(block(value, None, frequencies, conditioning), conditioning)
    assert result.shape == (1, 4, 4)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
