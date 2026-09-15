# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import torch

from vllm.config.vllm import (
    _SM70_QWEN38_27B_FP8_C32_DEFAULTS,
    _apply_sm70_qwen38_27b_fp8_c32_defaults,
)


def _config(**overrides):
    values = dict(
        model_config=SimpleNamespace(
            architectures=("Qwen3_5ForConditionalGeneration",),
            dtype=torch.float16,
            quantization="fp8",
            hf_text_config=SimpleNamespace(
                hidden_size=5120,
                num_hidden_layers=64,
                num_attention_heads=24,
                num_key_value_heads=4,
                head_dim=256,
                full_attention_interval=4,
            ),
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            enable_dbo=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=(1, 2, 4, 8, 16, 24, 32),
            max_cudagraph_capture_size=32,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_exact_c32_contract_applies_defaults(monkeypatch):
    for name in _SM70_QWEN38_27B_FP8_C32_DEFAULTS:
        monkeypatch.delenv(name, raising=False)

    applied = _apply_sm70_qwen38_27b_fp8_c32_defaults(_config(), is_sm70=True)

    assert set(applied) == set(_SM70_QWEN38_27B_FP8_C32_DEFAULTS)
    assert {
        name: os.environ[name] for name in _SM70_QWEN38_27B_FP8_C32_DEFAULTS
    } == _SM70_QWEN38_27B_FP8_C32_DEFAULTS


def test_explicit_override_is_preserved(monkeypatch):
    for name in _SM70_QWEN38_27B_FP8_C32_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_G6_DUAL_CTA", "0")

    applied = _apply_sm70_qwen38_27b_fp8_c32_defaults(_config(), is_sm70=True)

    assert "VLLM_FLASH_V100_XQA_G6_DUAL_CTA" not in applied
    assert os.environ["VLLM_FLASH_V100_XQA_G6_DUAL_CTA"] == "0"


def test_route_rejects_nonmatching_or_uncaptured_c32(monkeypatch):
    for name in _SM70_QWEN38_27B_FP8_C32_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    model = _config().model_config
    model.quantization = None
    assert not _apply_sm70_qwen38_27b_fp8_c32_defaults(
        _config(model_config=model), is_sm70=True
    )

    compilation = SimpleNamespace(
        cudagraph_capture_sizes=(1, 2, 4, 8, 16),
        max_cudagraph_capture_size=32,
    )
    assert not _apply_sm70_qwen38_27b_fp8_c32_defaults(
        _config(compilation_config=compilation), is_sm70=True
    )
    assert not any(name in os.environ for name in _SM70_QWEN38_27B_FP8_C32_DEFAULTS)
