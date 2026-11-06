# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequantize_mxfp8_weight_pairs,
)


def _fp8(rows: int, cols: int) -> torch.Tensor:
    return torch.full((rows, cols), 1.5).to(torch.float8_e4m3fn)


def _names(items) -> list[str]:
    return [item[0] for item in items]


def test_block_fp8_scale_before_its_weight_is_handed_on_after_it():
    # GLM-5.3-Flash-FP4-Local stores some block-FP8 scales in an earlier shard
    # than their weights; the pair must still reach the loader side by side.
    stream = [
        ("m.q_a_proj.weight_scale", torch.ones(1, 1, 1, 1)),
        ("m.norm.weight", torch.ones(4, dtype=torch.bfloat16)),
        ("m.q_a_proj.weight", _fp8(128, 128)),
    ]
    out = list(dequantize_mxfp8_weight_pairs(stream))
    assert _names(out) == [
        "m.norm.weight",
        "m.q_a_proj.weight",
        "m.q_a_proj.weight_scale",
    ]
    assert out[1][1].dtype == torch.float8_e4m3fn


def test_block_fp8_weight_before_its_scale_stays_next_to_it():
    stream = [
        ("m.o_proj.weight", _fp8(128, 128)),
        ("m.norm.weight", torch.ones(4, dtype=torch.bfloat16)),
        ("m.o_proj.weight_scale_inv", torch.ones(1, 1)),
    ]
    out = list(dequantize_mxfp8_weight_pairs(stream))
    assert _names(out) == [
        "m.norm.weight",
        "m.o_proj.weight",
        "m.o_proj.weight_scale_inv",
    ]


@pytest.mark.parametrize("scale_first", [False, True])
def test_mxfp8_pair_becomes_one_dequantized_weight(scale_first):
    weight = ("m.k_proj.weight", _fp8(2, 64))
    # E8M0 128 = 2^1: every 1.5 becomes 3.0.
    scale = ("m.k_proj.weight_scale", torch.full((2, 2), 128, dtype=torch.uint8))
    stream = [scale, weight] if scale_first else [weight, scale]
    out = list(dequantize_mxfp8_weight_pairs(stream))
    assert _names(out) == ["m.k_proj.weight"]
    assert out[0][1].dtype == torch.bfloat16
    assert torch.equal(out[0][1], torch.full((2, 64), 3.0, dtype=torch.bfloat16))


def test_nvfp4_block_scales_pass_through_in_place():
    stream = [
        ("m.experts.0.up_proj.weight_scale", _fp8(4, 1)),
        ("m.experts.0.up_proj.weight", torch.zeros(4, 8, dtype=torch.uint8)),
        ("m.experts.0.up_proj.weight_scale_2", torch.ones(())),
    ]
    assert _names(dequantize_mxfp8_weight_pairs(stream)) == _names(stream)


def test_float_scale_after_an_unquantized_weight_passes_through():
    stream = [
        ("m.proj.weight", torch.zeros(4, 8, dtype=torch.int8)),
        ("m.norm.weight", torch.ones(4, dtype=torch.bfloat16)),
        ("m.proj.weight_scale", torch.ones(4, 1)),
    ]
    assert _names(dequantize_mxfp8_weight_pairs(stream)) == _names(stream)


def test_a_scale_without_its_weight_fails_loudly():
    stream = [("m.q_b_proj.weight_scale", torch.ones(1, 1, 1, 1))]
    with pytest.raises(ValueError, match="without their weights"):
        list(dequantize_mxfp8_weight_pairs(stream))


def test_pairs_survive_a_split_by_top_level_prefix():
    # AutoWeightsLoader hands each run of one top-level prefix to its module in
    # a separate call; a pair that straddles a visual run would be parted.
    stream = [
        (
            "model.language_model.layers.11.self_attn.q_b_proj.weight_scale",
            torch.ones(1, 1, 1, 1),
        ),
        ("model.visual.blocks.0.norm1.weight", torch.ones(4, dtype=torch.bfloat16)),
        ("model.language_model.layers.11.self_attn.q_b_proj.weight", _fp8(128, 128)),
    ]
    runs = [
        _names(group)
        for _, group in itertools.groupby(
            dequantize_mxfp8_weight_pairs(stream),
            key=lambda item: item[0].split(".")[1],
        )
    ]
    assert [
        "model.language_model.layers.11.self_attn.q_b_proj.weight",
        "model.language_model.layers.11.self_attn.q_b_proj.weight_scale",
    ] in runs


@pytest.mark.parametrize("scale_first", [False, True])
def test_a_kept_mxfp8_pair_passes_through_in_stream_order(scale_first):
    weight = ("m.o_proj.weight", _fp8(2, 64))
    scale = ("m.o_proj.weight_scale", torch.full((2, 2), 128, dtype=torch.uint8))
    stream = [scale, weight] if scale_first else [weight, scale]
    out = list(
        dequantize_mxfp8_weight_pairs(
            stream, keep_quantized=lambda module: module == "m.o_proj"
        )
    )
    assert len(out) == len(stream)
    assert all(got is expected for got, expected in zip(out, stream))


def test_modules_the_model_does_not_keep_are_still_dequantized():
    kept_weight = ("m.o_proj.weight", _fp8(2, 64))
    kept_scale = ("m.o_proj.weight_scale", torch.full((2, 2), 128, dtype=torch.uint8))
    other_weight = ("m.lm_head.weight", _fp8(2, 64))
    other_scale = ("m.lm_head.weight_scale", torch.full((2, 2), 128, dtype=torch.uint8))
    out = list(
        dequantize_mxfp8_weight_pairs(
            [kept_weight, other_weight, kept_scale, other_scale],
            keep_quantized=lambda module: module == "m.o_proj",
        )
    )
    assert _names(out) == [
        "m.o_proj.weight",
        "m.o_proj.weight_scale",
        "m.lm_head.weight",
    ]
    assert out[2][1].dtype == torch.bfloat16
    assert torch.equal(out[2][1], torch.full((2, 64), 3.0, dtype=torch.bfloat16))
