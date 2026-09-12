# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed ModelOpt draft experts must retain checkpoint block scaling."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.models.qwen4_exp.nvidia import mtp
from vllm.models.qwen4_exp.nvidia.mtp_fp8 import (
    dequantize_mtp_fp8_experts,
    dequantize_mtp_fp8_weight,
)


def mixed_config(group_size=128):
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "quantized_layers": {
                "mtp.layers.0.mlp.experts": {
                    "quant_algo": "FP8_BLOCK_SCALES",
                    "group_size": group_size,
                },
                "model.language_model.layers.0.mlp.experts": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
            },
        }
    )


def test_draft_mixed_assignments_follow_runtime_layer_offsets():
    quant = mixed_config()
    original = dict(quant.quantized_layers)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=object())
    )
    with (
        patch.object(mtp, "get_draft_quant_config", return_value=quant),
        patch.object(mtp, "configure_quant_config"),
        patch.object(mtp, "replace", return_value=SimpleNamespace()),
    ):
        draft = mtp._make_draft_vllm_config(config, 48)
    assert draft.quant_config is quant
    assert quant._resolve_quant_algo("mtp.layers.48.mlp.experts") == "FP8_BLOCK_SCALES"
    assert quant._resolve_quant_algo("mtp.layers.0.mlp.experts") is None
    assert (
        quant.quantized_layers["model.language_model.layers.0.mlp.experts"]
        == original["model.language_model.layers.0.mlp.experts"]
    )
    assert "mtp.layers.0.mlp.experts" in original


def test_sm70_draft_experts_use_unquantized_dispatch_without_changing_target():
    quant = mixed_config()
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=object())
    )
    with (
        patch.object(mtp, "get_draft_quant_config", return_value=quant),
        patch.object(mtp, "replace", return_value=SimpleNamespace()),
        patch.object(mtp.sm70_tm, "is_exact_sm70_cuda_platform", return_value=True),
    ):
        draft = mtp._make_draft_vllm_config(config, 48)
        assert mtp._sm70_mtp_fp8_blocks(draft.quant_config) == {
            "mtp.layers.48.mlp.experts": 128
        }
    assert (
        quant.get_quant_method(Mock(spec=RoutedExperts), "mtp.layers.48.mlp.experts")
        is None
    )
    assert not quant.is_layer_excluded("model.language_model.layers.0.mlp.experts")
    assert (
        quant._resolve_quant_algo("model.language_model.layers.0.mlp.experts")
        == "NVFP4"
    )


@pytest.mark.parametrize("group_size", [None, 0, -128, "128"])
def test_mixed_block_fp8_rejects_invalid_group_size(group_size):
    quant = mixed_config(group_size)
    with (
        patch.object(mtp.sm70_tm, "is_exact_sm70_cuda_platform", return_value=True),
        pytest.raises(ValueError, match="Unsupported MTP FP8 block size"),
    ):
        mtp._sm70_mtp_fp8_blocks(quant)


def test_non_sm70_preserves_native_quantization():
    with patch.object(mtp.sm70_tm, "is_exact_sm70_cuda_platform", return_value=False):
        assert mtp._sm70_mtp_fp8_blocks(mixed_config()) == {}


@pytest.mark.parametrize("shape", [(640, 2560), (2560, 640), (129, 131)])
def test_dequantization_matches_blockwise_reference_and_tp4_slices(shape):
    rng = torch.Generator().manual_seed(19)
    weight = (torch.randn(shape, generator=rng) * 4).to(torch.float8_e4m3fn)
    scales = torch.rand(tuple((dim + 127) // 128 for dim in shape), generator=rng).to(
        torch.bfloat16
    )
    expected = torch.empty(shape, dtype=torch.float16)
    for row in range(scales.shape[0]):
        for col in range(scales.shape[1]):
            rows, cols = (
                slice(row * 128, (row + 1) * 128),
                slice(col * 128, (col + 1) * 128),
            )
            expected[rows, cols] = weight[rows, cols].float() * scales[row, col].float()
    actual = dequantize_mtp_fp8_weight(weight, scales, 128)
    assert actual.dtype == torch.float16
    assert torch.equal(actual, expected)
    if 640 in shape:
        shard_dim = shape.index(640)
        for rank in range(4):
            assert torch.equal(
                actual.narrow(shard_dim, rank * 160, 160),
                expected.narrow(shard_dim, rank * 160, 160),
            )


@pytest.mark.parametrize("scale_first", [False, True])
def test_expert_pairing_and_unrelated_weight_passthrough(scale_first):
    prefix = "model.layers.0.mlp.experts"
    name = prefix + ".3.gate_proj"
    weight = torch.ones((640, 128)).to(torch.float8_e4m3fn)
    scale = torch.arange(1, 6, dtype=torch.float32).reshape(5, 1)
    pair = [(name + ".weight", weight), (name + ".weight_scale_inv", scale)]
    if scale_first:
        pair.reverse()
    other = ("model.layers.0.self_attn.q_proj.weight", torch.ones(1))
    output = list(dequantize_mtp_fp8_experts([pair[0], other, pair[1]], {prefix: 128}))
    assert output[0][0] == other[0] and output[0][1] is other[1]
    assert output[1][0] == name + ".weight"
    assert torch.equal(output[1][1], dequantize_mtp_fp8_weight(weight, scale, 128))


@pytest.mark.parametrize("suffix", ["weight", "weight_scale_inv"])
def test_missing_pair_fails_closed(suffix):
    prefix = "model.layers.0.mlp.experts"
    with pytest.raises(ValueError, match="Unpaired MTP FP8"):
        list(
            dequantize_mtp_fp8_experts(
                [(prefix + ".0.down_proj." + suffix, torch.ones(1))], {prefix: 128}
            )
        )


def test_invalid_scale_shape_fails_closed():
    weight = torch.ones((640, 128)).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="scale shape/dtype mismatch"):
        dequantize_mtp_fp8_weight(weight, torch.ones((4, 1)), 128)
