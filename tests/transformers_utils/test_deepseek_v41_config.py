# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import pytest

from vllm.transformers_utils.configs.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
)


@pytest.fixture
def checkpoint_config():
    return {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "dtype": "bfloat16",
        "pad_token_id": 2,
        "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "n_routed_experts": 384,
            "num_experts_per_tok": 6,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "engram_layer_ids": [1, 14],
            "engram_pad_token_id": 2,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [37, 38, 39],
            "dspark_n_routed_experts": 128,
            "dspark_num_experts_per_tok": 3,
            "rope_scaling": {"rope_type": "yarn", "factor": 16},
        },
        "vision_config": {"model_type": "deepseek_v41_vision", "hidden_size": 1024},
    }


def test_nested_checkpoint_roundtrip_keeps_v41_and_dspark_identity(
    checkpoint_config, tmp_path
):
    original = deepcopy(checkpoint_config)
    config = DeepseekV41Config.from_dict(checkpoint_config)
    assert checkpoint_config == original
    assert config.model_type == "deepseek_v41"
    assert config.architectures == ["DeepseekV41ForCausalLM"]
    text = config.get_text_config()
    assert isinstance(text, DeepseekV41TextConfig)
    assert text.hidden_size == 5120 and text.moe_intermediate_size == 2304
    assert text.num_hidden_layers == 40
    assert text.n_routed_experts == 384 and text.num_experts_per_tok == 6
    assert text.dspark_block_size == 5
    assert text.dspark_target_layer_ids == [37, 38, 39]
    assert text.dspark_n_routed_experts == 128
    assert text.dspark_num_experts_per_tok == 3
    assert text.kv_source_layer_ids == [2, 8, 14, 20]
    assert len(text.compress_ratios) == 43  # Includes all three draft layers.
    config.save_pretrained(tmp_path)
    restored = DeepseekV41Config.from_pretrained(tmp_path, local_files_only=True)
    assert restored.to_dict() == config.to_dict()


def test_vllm_config_loader_uses_the_separate_v41_class(checkpoint_config, tmp_path):
    from vllm.transformers_utils.config import get_config, get_hf_text_config

    DeepseekV41Config(**checkpoint_config).save_pretrained(tmp_path)
    config = get_config(tmp_path, trust_remote_code=False)
    assert isinstance(config, DeepseekV41Config)
    assert get_hf_text_config(config).model_type == "deepseek_v41_text"
    assert config.architectures == ["DeepseekV41ForCausalLM"]


def test_config_lists_are_not_aliased_into_the_input(checkpoint_config):
    config = DeepseekV41Config(**checkpoint_config)
    config.text_config.kv_source_layer_ids.append(38)
    assert config.vision_config is not None
    config.vision_config["hidden_size"] = 1
    assert checkpoint_config["text_config"]["kv_source_layer_ids"] == [2, 8, 14, 20]
    assert checkpoint_config["vision_config"]["hidden_size"] == 1024


def test_explicit_text_config_and_rope_parameters_are_preserved():
    rope = {"rope_type": "default", "rope_theta": 10000}
    text = DeepseekV41TextConfig(rope_parameters=rope)
    config = DeepseekV41Config(text_config=text)
    assert config.get_text_config() is text
    assert text.rope_parameters == rope


@pytest.mark.parametrize(
    "text",
    [
        {"model_type": "deepseek_v4"},
        {"model_type": "deepseek_v3"},
        "deepseek_v41_text",
    ],
)
def test_v41_cannot_silently_load_another_text_architecture(text):
    with pytest.raises((TypeError, ValueError)):
        DeepseekV41Config(text_config=text)
