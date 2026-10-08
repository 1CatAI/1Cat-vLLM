# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig


def _glm53_config(architecture: str) -> Glm5NextConfig:
    return Glm5NextConfig(
        architectures=[architecture],
        text_config={
            "num_nextn_predict_layers": 1,
            "hc_mult": 4,
            "kv_lora_rank": 512,
        },
    )


@pytest.mark.parametrize(
    "architecture",
    ["Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"],
)
def test_glm53_mtp_draft_config(architecture: str) -> None:
    config = SpeculativeConfig.hf_config_override(_glm53_config(architecture))

    assert config.model_type == "glm5_next_mtp"
    assert config.architectures == ["Glm5NextMTPModel"]
    assert "Glm5NextMTPModel" in ModelRegistry.get_supported_archs()
    assert config.n_predict == 1
    assert config.is_mtp_draft
    # The draft reads the collapsed trunk state, not the 4-stream residual.
    assert config.hc_mult == 1
    # Text-config fields stay reachable through the multimodal wrapper.
    assert config.kv_lora_rank == 512


def test_other_mtp_overrides_unchanged() -> None:
    config = PretrainedConfig(
        architectures=["DeepseekV3ForCausalLM"],
        num_nextn_predict_layers=1,
    )
    config.model_type = "deepseek_v3"

    config = SpeculativeConfig.hf_config_override(config)

    assert config.model_type == "deepseek_mtp"
    assert config.architectures == ["DeepSeekMTPModel"]
    assert not getattr(config, "is_mtp_draft", False)
