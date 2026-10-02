# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs
from vllm.config.model import ModelConfig
from vllm.config.vllm import _configure_sm70_dflash2_graph_cache


@pytest.fixture
def release(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
    monkeypatch.delenv("VLLM_USE_AOT_COMPILE", raising=False)
    model = NS(
        architectures=["Qwen3_5ForConditionalGeneration"],
        dtype=torch.float16,
        quantization="compressed-tensors",
        model_arch_config=NS(quantization_config={"format": "nvfp4-pack-quantized"}),
        hf_text_config=NS(
            hidden_size=5120,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=256,
        ),
    )
    model.is_nvfp4_quantized = lambda: ModelConfig.is_nvfp4_quantized(model)
    spec = NS(
        method="dflash",
        num_speculative_tokens=7,
        draft_model_config=NS(hf_config=NS(dflash_config={"selector_top_k": 16})),
    )
    parallel = NS(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        enable_dbo=False,
        ubatch_size=1,
    )
    return model, spec, parallel, NS(cache_dtype="fp8_e4m3")


def test_release_uses_graph_cache_without_disabling_compilation_cache(release):
    assert _configure_sm70_dflash2_graph_cache(*release)
    assert not envs.VLLM_USE_AOT_COMPILE
    assert not envs.VLLM_DISABLE_COMPILE_CACHE


@pytest.mark.parametrize("override", ["0", "1"])
def test_explicit_aot_choice_is_preserved(release, monkeypatch, override):
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", override)
    assert _configure_sm70_dflash2_graph_cache(*release)
    assert (override == "1") == envs.VLLM_USE_AOT_COMPILE


@pytest.mark.parametrize(
    "change", ["cache_off", "tp2", "e5m2", "fp8", "q5", "target_only"]
)
def test_unqualified_routes_keep_their_existing_aot_default(
    release, monkeypatch, change
):
    model, spec, parallel, cache = release
    if change == "cache_off":
        monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    elif change == "tp2":
        parallel.tensor_parallel_size = 2
    elif change == "e5m2":
        cache.cache_dtype = "fp8_e5m2"
    elif change == "fp8":
        model.model_arch_config.quantization_config = {"format": "float-quantized"}
    elif change == "q5":
        spec.num_speculative_tokens = 5
    else:
        spec = None
    assert not _configure_sm70_dflash2_graph_cache(model, spec, parallel, cache)
    assert "VLLM_USE_AOT_COMPILE" not in os.environ
    assert envs.VLLM_USE_AOT_COMPILE  # Existing SM70 graph default.
