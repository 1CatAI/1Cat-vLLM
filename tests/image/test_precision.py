# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.models.z_image.precision import (
    FP32FeedForward,
    FP32OutputLinear,
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
