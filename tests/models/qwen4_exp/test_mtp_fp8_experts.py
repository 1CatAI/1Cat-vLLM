# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import mtp_fp8_experts as impl


@pytest.fixture
def should_do_global_cleanup_after_test():
    # These tests only create CPU tensors and never initialize a process group.
    return False


@pytest.mark.parametrize("shape", [(320, 2560), (2560, 160), (16, 16)])
def test_online_rows_keep_shape_and_bound_rounding(shape):
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(shape, generator=generator, dtype=torch.float16)
    weight[0].zero_()
    original = weight.clone()
    quantized, scale = impl.quantize_expert_rows(weight)
    recovered = quantized.float() * scale
    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.shape == weight.shape
    assert scale.shape == (shape[0], 1)
    assert torch.equal(weight, original)
    assert torch.isfinite(recovered).all()
    assert torch.equal(recovered[0], weight[0].float())
    # E4M3's largest adjacent spacing is 32, so rounding contributes at
    # most half a step at the row scale (plus scale-rounding saturation).
    assert ((recovered - weight.float()).abs() <= 17 * scale).all()
    assert quantized.numel() + scale.numel() * scale.element_size() < weight.numel() * 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_checkpoint_rejected(value):
    weight = torch.zeros(8, 8, dtype=torch.float16)
    weight[0, 0] = value
    with pytest.raises(ValueError, match="non-finite"):
        impl.quantize_expert_rows(weight)


def test_padding_preserves_gated_expert_and_still_reduces_storage():
    generator = torch.Generator().manual_seed(9)
    gate_up = torch.randn(320, 64, generator=generator)
    down = torch.randn(64, 160, generator=generator)
    x = torch.randn(3, 64, generator=generator)
    padded_gate_up = impl.pad_expert_matrix(gate_up, True)
    padded_down = impl.pad_expert_matrix(down, False)
    assert padded_gate_up.shape == (512, 64)
    assert padded_down.shape == (64, 256)
    gate, up = (x @ gate_up.T).chunk(2, dim=-1)
    padded_gate, padded_up = (x @ padded_gate_up.T).chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate) * up) @ down.T
    actual = (torch.nn.functional.silu(padded_gate) * padded_up) @ padded_down.T
    torch.testing.assert_close(actual, expected)
    # Payload savings survive the required padding: 160/256 * FP16/FP8.
    assert padded_gate_up.numel() + padded_down.numel() < 2 * (
        gate_up.numel() + down.numel()
    )


def test_only_unquantized_draft_experts_are_overridden(monkeypatch):
    unquantized = Mock(spec=impl.UnquantizedFusedMoEMethod)
    fallback = SimpleNamespace(
        packed_modules_mapping={}, get_quant_method=lambda layer, prefix: unquantized
    )
    wrapped = impl.MTPExpertFp8Config(fallback)
    expert = Mock(spec=impl.RoutedExperts)
    fp8 = object()
    monkeypatch.setattr(impl, "MTPFp8SM70MoEMethod", lambda layer: fp8)
    assert wrapped.get_quant_method(expert, "mtp.layers.48.mlp.experts") is fp8
    assert wrapped.get_quant_method(expert, "model.layers.0.mlp.experts") is unquantized
    for prefix in ("mtp.layers.48.mlp.gate", "lm_head", "model.embed_tokens"):
        assert wrapped.get_quant_method(torch.nn.Linear(4, 4), prefix) is unquantized
    assert fallback.get_quant_method(expert, "mtp.layers.48.mlp.experts") is unquantized
    fallback.get_quant_method = lambda layer, prefix: object()
    with pytest.raises(ValueError, match="unquantized checkpoint"):
        wrapped.get_quant_method(expert, "mtp.layers.48.mlp.experts")
