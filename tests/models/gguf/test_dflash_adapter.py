# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.model_loader.gguf_adapters import get_gguf_adapter
from vllm.model_executor.model_loader.gguf_adapters.dflash import DFlashAdapter
from vllm.transformers_utils.gguf_config import (
    gguf_config_dict,
    gguf_config_from_metadata,
)


def metadata():
    return json.loads(Path(__file__).with_name("dflash2_metadata.json").read_text())


def test_metadata_recovers_dflash2_arithmetic_and_layer_contract():
    config = gguf_config_from_metadata(metadata())
    assert config.architectures == ["DFlash2DraftModel"]
    assert config.dtype == torch.bfloat16
    assert (config.hidden_size, config.intermediate_size) == (5120, 17408)
    assert (config.num_attention_heads, config.num_key_value_heads) == (32, 8)
    assert config.head_dim == 128
    assert config.layer_types == ["sliding_attention"] * 5
    assert config.sliding_window == 2048 and config.is_causal is False
    assert config.dflash_config["target_layer_ids"] == [5, 19, 33, 47, 61]
    assert config.dflash_config["mask_token_id"] == 248070
    assert config.dflash_config["block_size"] == 8
    assert isinstance(get_gguf_adapter(config, tp_size=4), DFlashAdapter)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("dflash.target_layers", [0, 20], "1-based"),
        ("dflash.target_layers", [20, 6], "ordered"),
        ("tokenizer.ggml.mask_token_id", 248320, "mask token"),
        ("dflash.conv_group_size", 17, "incompatible"),
        ("dflash.attention.sliding_window_pattern", [True], "match layers"),
        ("dflash.attention.sliding_window", 0, "positive window"),
    ],
)
def test_reject_incompatible_metadata(key, value, match):
    data = metadata()
    data[key] = value
    with pytest.raises(ValueError, match=match):
        gguf_config_dict(data)


def test_real_checkpoint_directory_has_no_unmapped_tensors():
    directory = json.loads(
        Path(__file__).with_name("dflash2_tensor_directory.json").read_text()
    )
    adapter = get_gguf_adapter(gguf_config_from_metadata(metadata()), tp_size=4)
    mapping = adapter.build_name_map({t["name"]: None for t in directory})
    assert len(mapping) == 81
    assert mapping["fc.weight"] == "fc.weight"
    assert mapping["enc.output_norm.weight"] == "hidden_norm.weight"
    assert mapping["blk.4.ffn_conv_base"] == "layers.4.mlp_conv.base_kernel"
    assert not any("embed_tokens" in name or "lm_head" in name for name in mapping)
    with pytest.raises(ValueError, match="Unmapped DFlash"):
        adapter.build_name_map({"blk.5.attn_q.weight": None})


def test_explicit_hf_draft_config_selects_native_adapter():
    config = gguf_config_from_metadata(metadata())
    del config.gguf_architecture
    assert isinstance(get_gguf_adapter(config, tp_size=4), DFlashAdapter)


def test_dense_dequantization_retains_owned_output_without_an_extra_copy(monkeypatch):
    adapter = get_gguf_adapter(gguf_config_from_metadata(metadata()), tp_size=4)
    values = np.ones((4, 32), dtype=np.float32)
    monkeypatch.setattr(gguf.quants, "dequantize", lambda *_: values)
    tensor = SimpleNamespace(
        tensor_type=gguf.GGMLQuantizationType.Q8_0,
        data=np.zeros((4, 34), dtype=np.uint8),
        shape=[32, 4],
    )
    weight = dict(
        adapter.weights(
            {"selector_hidden.weight": tensor},
            {"selector_hidden.weight": "candidate_selector.hidden_projection.weight"},
            torch.float32,
        )
    )["candidate_selector.hidden_projection.weight"]
    assert weight.data_ptr() == values.ctypes.data


def test_dense_auxiliaries_decode_without_qwen35_norm_offsets():
    adapter = get_gguf_adapter(gguf_config_from_metadata(metadata()), tp_size=4)
    data = gguf.quants.quantize(
        np.arange(4 * 32, dtype=np.float32).reshape(4, 32) / 128,
        gguf.GGMLQuantizationType.Q8_0,
    )
    packed = SimpleNamespace(
        data=data, tensor_type=gguf.GGMLQuantizationType.Q8_0, shape=[32, 4]
    )
    norm = np.full(32, 1.25, dtype=np.float32)
    tensors = {
        "selector_predecessor.weight": packed,
        "blk.0.attn_conv_proj.weight": packed,
        "blk.0.attn_q.weight": packed,
        "output_norm.weight": SimpleNamespace(
            data=norm, tensor_type=gguf.GGMLQuantizationType.F32, shape=[32]
        ),
    }
    weights = dict(
        adapter.weights(tensors, adapter.build_name_map(tensors), torch.float16)
    )
    expected = torch.from_numpy(
        gguf.quants.dequantize(data, gguf.GGMLQuantizationType.Q8_0)
    ).half()
    assert torch.equal(weights["candidate_selector.predecessor_codebook"], expected)
    assert torch.equal(
        weights["layers.0.attention_conv.kernel_projection.weight"], expected
    )
    assert torch.equal(weights["norm.weight"], torch.from_numpy(norm).half())
    assert torch.equal(
        weights["layers.0.self_attn.q_proj.qweight"], torch.from_numpy(data)
    )
    assert "layers.0.self_attn.q_proj.qweight_type" in weights
    assert "candidate_selector.predecessor_codebook.qweight_type" not in weights
