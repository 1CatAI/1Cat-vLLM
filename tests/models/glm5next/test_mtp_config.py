# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig
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
    assert config.mtp_draft_pipeline_parallel_size == 1
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


def test_glm53_mtp_override_is_idempotent() -> None:
    config = SpeculativeConfig.hf_config_override(_glm53_config("Glm5NextForCausalLM"))
    assert SpeculativeConfig.hf_config_override(config) is config
    assert config.model_type == "glm5_next_mtp"
    assert config.architectures == ["Glm5NextMTPModel"]
    assert config.hc_mult == 1


def test_config_import_does_not_import_model_implementation():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig; "
                "assert 'vllm.models.glm5next.nvidia.model' not in sys.modules; "
                "assert Glm5NextConfig().get_text_config().hidden_size == 4096"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "repository_name",
    ["GLM-NVFP4", "GLM-NVFP4-DFlash2", "GLM-eagle-assets", "GLM-eagle3-assets"],
)
def test_native_mtp_metadata_precedes_repository_name(tmp_path, repository_name):
    directory = tmp_path / repository_name
    directory.mkdir()
    _glm53_config("Glm5NextForConditionalGeneration").to_json_file(
        directory / "config.json"
    )
    target = ModelConfig(
        model=str(directory),
        dtype="float16",
        max_model_len=16384,
        enforce_eager=True,
    )
    config = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        method="mtp",
        num_speculative_tokens=4,
    )
    assert config.method == "mtp"
    assert config.draft_model_config.hf_config.architectures == ["Glm5NextMTPModel"]
    assert config.draft_model_config.hf_text_config.num_attention_heads == 64
    assert config.draft_model_config.max_model_len == 16384
