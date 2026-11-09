# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.model_loader.gguf_adapters import get_gguf_adapter
from vllm.model_executor.model_loader.gguf_adapters.deepseek_v41 import (
    DeepseekV41Adapter,
)
from vllm.transformers_utils.configs.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
)


@pytest.fixture
def adapter():
    config = DeepseekV41TextConfig(
        n_routed_experts=384,
        num_experts_per_tok=6,
        hc_mult=4,
        q_lora_rank=1280,
        o_lora_rank=1024,
        o_groups=8,
        index_n_heads=32,
        index_head_dim=128,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        engram_layer_ids=[1, 14],
        tie_word_embeddings=False,
    )
    return get_gguf_adapter(DeepseekV41Config(text_config=config), tp_size=4)


def metadata(adapter):
    return {
        name: SimpleNamespace(
            shape=shape,
            tensor_type=gguf.GGMLQuantizationType.IQ1_M
            if "_exps.weight" in name
            else gguf.GGMLQuantizationType.F32,
        )
        for name, shape in adapter.expected.items()
    }


def test_complete_backbone_has_1008_unique_names_and_no_norm_offset(adapter):
    assert isinstance(adapter, DeepseekV41Adapter)
    names = adapter.build_name_map(metadata(adapter))
    assert len(names) == len(set(names.values())) == 1008
    assert names["blk.1.engram_wkv.weight"] == "model.layers.1.engram.wkv.weight"
    assert names["blk.39.hc_ffn_fn.weight"] == "model.layers.39.hc_ffn_fn"
    assert names["engram_multipliers.weight"] == "model.engram_multipliers"
    assert adapter.is_linear(names["blk.0.attn_output_a.weight"])
    assert not adapter.is_linear(names["blk.0.ffn_gate_inp.weight"])


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "format", "pair"])
def test_incomplete_or_incompatible_backbones_are_rejected(adapter, change):
    tensors = metadata(adapter)
    if change == "missing":
        del tensors["blk.39.hc_ffn_fn.weight"]
    elif change == "extra":
        tensors["blk.40.attn_norm.weight"] = tensors["blk.0.attn_norm.weight"]
    elif change == "shape":
        tensors["blk.0.attn_q_b.weight"].shape = (1280, 16384)
    elif change == "format":
        tensors[
            "blk.0.ffn_down_exps.weight"
        ].tensor_type = gguf.GGMLQuantizationType.Q5_K
    else:
        tensors[
            "blk.0.ffn_up_exps.weight"
        ].tensor_type = gguf.GGMLQuantizationType.IQ2_XS
    with pytest.raises(ValueError):
        adapter.build_name_map(tensors)


def test_fp32_norm_is_preserved_without_half_rounding_or_qwen_shift(adapter):
    raw, target = "blk.0.attn_norm.weight", "model.layers.0.attn_norm.weight"
    values = np.full((5120,), 1.000123, dtype=np.float32)
    tensor = SimpleNamespace(tensor_type=gguf.GGMLQuantizationType.F32, data=values)
    loaded = dict(adapter.weights({raw: tensor}, {raw: target}, torch.float16))
    assert loaded[target].dtype == torch.float32
    np.testing.assert_array_equal(loaded[target].numpy(), values)


def test_hash_multipliers_retain_every_int64_bit(adapter):
    raw, target = "engram_multipliers.weight", "model.engram_multipliers"
    values = np.array(
        [[2**62 + 1, -(2**62) + 3, 17, 19], [23, 29, 31, 37]], dtype=np.int64
    )
    tensor = SimpleNamespace(tensor_type=gguf.GGMLQuantizationType.I64, data=values)
    loaded = dict(adapter.weights({raw: tensor}, {raw: target}, torch.float16))
    assert loaded[target].dtype == torch.int64
    np.testing.assert_array_equal(loaded[target].numpy(), values)


@pytest.mark.parametrize("layer,dtype", [(2, torch.float32), (20, torch.float16)])
def test_compressor_keeps_ratio_two_scores_unrounded(adapter, layer, dtype):
    raw = f"blk.{layer}.attn_compressor_kv.weight"
    name = f"model.layers.{layer}.attn.compressor.wkv.weight"
    values = torch.tensor([[1.25, -0.5], [0.03125, 0.0625]]).bfloat16()
    tensor = SimpleNamespace(
        tensor_type=gguf.GGMLQuantizationType.BF16,
        data=values.view(torch.uint16).numpy(),
    )
    loaded = dict(adapter.weights({raw: tensor}, {raw: name}, torch.float16))
    assert loaded[name.removesuffix("weight") + "qweight"].dtype == dtype
    descriptor = int(loaded[name.removesuffix("weight") + "qweight_type"])
    assert descriptor == int(
        gguf.GGMLQuantizationType.F32
        if dtype == torch.float32
        else gguf.GGMLQuantizationType.F16
    )


def test_shared_down_restores_fp16_before_unaligned_576_column_tp_split(adapter):
    kind = gguf.GGMLQuantizationType.Q5_K
    _, size = gguf.GGML_QUANT_SIZES[kind]
    values = np.random.default_rng(41).integers(0, 256, (3, 9, size), dtype=np.uint8)
    values[..., :4] = np.array([0.03125, 0.03125], dtype="<f2").view(np.uint8)
    tensor = SimpleNamespace(
        shape=(2304, 3), data=values.reshape(3, -1), tensor_type=kind
    )
    name = "model.layers.0.ffn.shared_experts.down_proj.weight"
    raw = "blk.0.ffn_down_shexp.weight"
    loaded = dict(adapter.weights({raw: tensor}, {raw: name}, torch.float16))
    expected = torch.from_numpy(gguf.quants.dequantize(tensor.data, kind)).half()
    torch.testing.assert_close(
        loaded[name.removesuffix("weight") + "qweight"], expected, rtol=0, atol=0
    )
    assert int(loaded[name.removesuffix("weight") + "qweight_type"]) == int(
        gguf.GGMLQuantizationType.F16
    )
    assert (
        "shared_down_tp_boundary"
        in adapter.fallback_reasons[name.removesuffix(".weight")]
    )


def test_grouped_output_projection_flattens_only_output_rows(adapter):
    raw, target = "blk.0.attn_output_a.weight", "model.layers.0.attn.wo_a.weight"
    data = np.broadcast_to(
        np.arange(8, dtype=np.uint8)[:, None, None], (8, 1024, 2816)
    ).copy()
    tensor = SimpleNamespace(tensor_type=gguf.GGMLQuantizationType.Q5_K, data=data)
    loaded = dict(adapter.weights({raw: tensor}, {raw: target}, torch.float16))
    weight = loaded[target.removesuffix(".weight") + ".qweight"]
    assert weight.shape == (8192, 2816) and weight.dtype == torch.uint8
    np.testing.assert_array_equal(weight.reshape(data.shape).numpy(), data)


def test_float_projection_descriptor_matches_the_converted_storage(adapter):
    raw, target = (
        "blk.2.indexer.proj.weight",
        "model.layers.2.attn.indexer.weights_proj.weight",
    )
    values = torch.full((8, 8), 0.03125, dtype=torch.bfloat16)
    tensor = SimpleNamespace(
        tensor_type=gguf.GGMLQuantizationType.BF16,
        data=values.view(torch.uint16).numpy(),
    )
    loaded = dict(adapter.weights({raw: tensor}, {raw: target}, torch.float16))
    prefix = target.removesuffix(".weight")
    assert int(loaded[prefix + ".qweight_type"]) == gguf.GGMLQuantizationType.F16
    assert loaded[prefix + ".qweight"].dtype == torch.float16
    torch.testing.assert_close(
        loaded[prefix + ".qweight"].float(), values.float(), rtol=0, atol=0
    )
