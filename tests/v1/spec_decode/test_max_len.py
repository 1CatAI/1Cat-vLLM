# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test whether spec decoding handles the max model length properly."""

import copy
from types import SimpleNamespace
from typing import Any

import pytest
from transformers import PretrainedConfig

from tests.utils import get_attn_backend_list_based_on_platform
from vllm import LLM, SamplingParams
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.platforms import current_platform
from vllm.sampling_params import StructuredOutputsParams

_PROMPTS = [
    "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1",
    "Repeat the following sentence 10 times: Consistency is key to mastering any skill.",  # noqa: E501
    "Who won the Turing Award in 2018, and for what contribution? Describe in detail.",  # noqa: E501
]


@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 10])
def test_ngram_max_len(num_speculative_tokens: int):
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=100,
        enforce_eager=True,  # For faster initialization.
        speculative_config={
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": num_speculative_tokens,
        },
    )
    sampling_params = SamplingParams(max_tokens=100, ignore_eos=True)
    llm.generate(_PROMPTS, sampling_params)


@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 10])
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_max_len(
    monkeypatch: pytest.MonkeyPatch, num_speculative_tokens: int, attn_backend: str
):
    if attn_backend == "ROCM_AITER_FA" and current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        enforce_eager=True,  # For faster initialization.
        speculative_config={
            "method": "eagle",
            "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
            "num_speculative_tokens": num_speculative_tokens,
            "max_model_len": 80,
        },
        max_model_len=200,
        attention_config={"backend": attn_backend},
    )
    sampling_params = SamplingParams(max_tokens=200, ignore_eos=True)
    outputs = llm.generate(_PROMPTS, sampling_params)
    for o in outputs:
        assert o.outputs[0].finish_reason == "length", (
            "This test is only meaningful if the output is truncated due to max length"
        )

    sampling_params = SamplingParams(
        max_tokens=200,
        structured_outputs=StructuredOutputsParams(regex="^" + "a b c d e " * 15 + "$"),
    )
    output = llm.generate(_PROMPTS, sampling_params)
    for o in output:
        assert o.prompt_token_ids is not None
        assert (
            len(o.prompt_token_ids)
            < 80
            < len(o.prompt_token_ids) + len(o.outputs[0].token_ids)
            <= 200
        ), (
            "This test is only meaningful if the output "
            "is longer than the eagle max length"
        )
        assert o.outputs[0].text == "a b c d e " * 15


@pytest.mark.parametrize("spec_max_model_len", [80, 150])
def test_mtp_speculative_config_max_model_len(spec_max_model_len: int):
    """Regression test for #41456: max_model_len in speculative config
    should be respected for the draft model."""
    model_config = ModelConfig(
        model="XiaomiMiMo/MiMo-7B-Base",
        runner="generate",
        max_model_len=200,
        trust_remote_code=True,
    )
    spec_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        method="mtp",
        num_speculative_tokens=1,
        max_model_len=spec_max_model_len,
    )
    assert spec_config.draft_model_config.max_model_len == spec_max_model_len


def _native_mtp_yarn_config(
    *,
    max_model_len: int = 1_000_000,
    rope_parameters: dict[str, Any] | None = None,
) -> tuple[Any, PretrainedConfig]:
    if rope_parameters is None:
        rope_parameters = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 262_144,
        }
    target_hf_config = PretrainedConfig(max_position_embeddings=262_144)
    # Keep deliberately invalid fixtures out of Transformers' constructor
    # validation so these cases exercise vLLM's own inheritance checks.
    target_hf_config.rope_parameters = copy.deepcopy(rope_parameters)
    draft_hf_config = PretrainedConfig(max_position_embeddings=262_144)
    target_model_config = SimpleNamespace(
        model="test/native-mtp",
        max_model_len=1_000_000,
        hf_config=target_hf_config,
    )
    config = SimpleNamespace(
        method="mtp",
        model=target_model_config.model,
        max_model_len=max_model_len,
        target_model_config=target_model_config,
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
    )
    return config, draft_hf_config


def test_native_mtp_inherits_validated_target_yarn() -> None:
    config, draft_hf_config = _native_mtp_yarn_config()
    target_rope = config.target_model_config.hf_config.rope_parameters

    SpeculativeConfig._inherit_target_rope_for_extended_native_mtp(config)

    assert draft_hf_config.rope_parameters == target_rope
    assert draft_hf_config.rope_parameters is not target_rope
    target_rope["factor"] = 2.0
    assert draft_hf_config.rope_parameters["factor"] == 4.0


@pytest.mark.parametrize(
    ("rope_parameters", "match"),
    [
        ({}, "explicit target YaRN"),
        (
            {
                "rope_type": "dynamic",
                "factor": 4.0,
                "original_max_position_embeddings": 262_144,
            },
            "explicit target YaRN",
        ),
        (
            {
                "rope_type": "yarn",
                "factor": "invalid",
                "original_max_position_embeddings": 262_144,
            },
            "requires numeric",
        ),
        (
            {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 131_072,
            },
            "to match the drafter",
        ),
    ],
)
def test_native_mtp_rejects_unvalidated_rope(
    rope_parameters: dict[str, Any],
    match: str,
) -> None:
    config, _ = _native_mtp_yarn_config(rope_parameters=rope_parameters)

    with pytest.raises(ValueError, match=match):
        SpeculativeConfig._inherit_target_rope_for_extended_native_mtp(config)


def test_native_mtp_rejects_length_above_validated_yarn_limit() -> None:
    config, _ = _native_mtp_yarn_config(max_model_len=1_048_577)
    config.target_model_config.max_model_len = 1_048_577

    with pytest.raises(ValueError, match="validated target YaRN limit=1048576"):
        SpeculativeConfig._inherit_target_rope_for_extended_native_mtp(config)


def test_native_mtp_does_not_modify_unextended_drafter() -> None:
    config, draft_hf_config = _native_mtp_yarn_config(max_model_len=262_144)
    original = copy.deepcopy(draft_hf_config.to_dict())

    SpeculativeConfig._inherit_target_rope_for_extended_native_mtp(config)

    assert draft_hf_config.to_dict() == original


@pytest.mark.parametrize("factor", [0.0, 1.0, float("nan"), float("inf"), 1e308])
def test_native_mtp_rejects_invalid_or_overflowing_yarn_factor(factor: float) -> None:
    config, _ = _native_mtp_yarn_config()
    config.target_model_config.hf_config.rope_parameters["factor"] = factor
    with pytest.raises(ValueError):
        SpeculativeConfig._inherit_target_rope_for_extended_native_mtp(config)


def test_native_mtp_respects_target_limit_below_yarn_limit() -> None:
    config, _ = _native_mtp_yarn_config(max_model_len=800_001)
    config.target_model_config.max_model_len = 800_000
    with pytest.raises(ValueError, match="validated target YaRN limit=800000"):
        SpeculativeConfig._inherit_target_rope_for_extended_native_mtp(config)


@pytest.mark.parametrize("key", ["factor", "original_max_position_embeddings"])
def test_native_mtp_rejects_missing_yarn_parameters(key: str) -> None:
    config, _ = _native_mtp_yarn_config()
    config.target_model_config.hf_config.rope_parameters.pop(key)
    with pytest.raises(ValueError, match="requires numeric"):
        SpeculativeConfig._inherit_target_rope_for_extended_native_mtp(config)


@pytest.mark.parametrize("rope_type", ["linear", "yarn"])
def test_native_mtp_preserves_existing_matching_scaled_context(rope_type: str) -> None:
    from vllm.config.model import _get_and_verify_max_len

    config, draft_hf_config = _native_mtp_yarn_config()
    rope = {
        "rope_type": rope_type,
        "factor": 4.0,
        "original_max_position_embeddings": 262_144,
    }
    config.target_model_config.hf_config.rope_parameters = copy.deepcopy(rope)
    draft_hf_config.rope_parameters = copy.deepcopy(rope)
    original_rope = draft_hf_config.rope_parameters

    def derive_limit(requested: int) -> int:
        return _get_and_verify_max_len(
            hf_config=draft_hf_config,
            model_arch_config=SimpleNamespace(
                derived_max_model_len_and_key=(262_144, "max_position_embeddings")
            ),
            tokenizer_config=None,
            max_model_len=requested,
            disable_sliding_window=False,
            sliding_window=None,
        )

    config.draft_model_config.get_and_verify_max_len = derive_limit
    assert derive_limit(-1) == 1_048_576
    SpeculativeConfig._inherit_target_rope_for_extended_native_mtp(config)
    assert draft_hf_config.rope_parameters is original_rope


@pytest.mark.parametrize("field,value", [("method", "eagle"), ("model", "other/draft")])
def test_native_mtp_leaves_unrelated_drafters_unchanged(field: str, value: str) -> None:
    config, draft_hf_config = _native_mtp_yarn_config()
    setattr(config, field, value)
    original = copy.deepcopy(draft_hf_config.to_dict())
    SpeculativeConfig._inherit_target_rope_for_extended_native_mtp(config)
    assert draft_hf_config.to_dict() == original
