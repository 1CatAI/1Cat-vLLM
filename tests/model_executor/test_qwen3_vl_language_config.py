# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3VLConfig

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models import qwen3_vl

pytestmark = pytest.mark.cpu_test


def test_language_submodel_has_its_own_architecture(tmp_path, monkeypatch):
    config = Qwen3VLConfig()
    config.architectures = ["Qwen3VLForConditionalGeneration"]
    config.save_pretrained(tmp_path)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    engine = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, enforce_eager=True
        )
    )
    original = engine.model_config.hf_config.text_config
    assert original.architectures is None
    captured = []

    class Vision(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class Language(nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            captured.append(vllm_config)
            self.model = SimpleNamespace(start_layer=0)

        def make_empty_intermediate_tensors(self, *args, **kwargs):
            raise AssertionError("This configuration test does not run inference")

    monkeypatch.setattr(qwen3_vl, "Qwen3_VisionTransformer", Vision)
    monkeypatch.setattr(qwen3_vl, "Qwen3LLMForCausalLM", Language)
    monkeypatch.setattr(
        qwen3_vl, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True)
    )
    with torch.device("cpu"):
        qwen3_vl.Qwen3VLForConditionalGeneration(vllm_config=engine)

    (language,) = captured
    assert language.model_config.architectures == ["Qwen3ForCausalLM"]
    assert language.model_config.hf_config.model_type == "qwen3_vl_text"
    assert engine.model_config.is_multimodal_model
    assert original.architectures is None
