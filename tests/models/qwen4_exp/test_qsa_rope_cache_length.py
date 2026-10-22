# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.models.qwen4_exp.nvidia.qsa import _qsa_rope_cache_length

ROPE = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default",
}


def _mm_config(language_model_only=True, videos=999):
    return SimpleNamespace(
        language_model_only=language_model_only,
        get_limit_per_prompt=lambda modality: (
            0 if language_model_only else {"video": videos}.get(modality, 999)
        ),
    )


def _configs(max_model_len, language_model_only=True, videos=999):
    config = SimpleNamespace(max_position_embeddings=262144, rope_parameters=ROPE)
    model = SimpleNamespace(
        max_model_len=max_model_len,
        multimodal_config=_mm_config(language_model_only, videos),
    )
    return config, model


def test_text_only_mrope_cache_covers_exactly_the_engine_positions():
    config, model = _configs(262144)
    length = _qsa_rope_cache_length(config, model)
    assert length == 65536  # MRotaryEmbedding stores 4x: 262144 rows
    config, model = _configs(1000)
    assert _qsa_rope_cache_length(config, model) * 4 >= 1000
    config, model = _configs(262144, language_model_only=False)
    assert _qsa_rope_cache_length(config, model) == 262144


def test_image_only_mrope_cache_is_bounded_like_text():
    # Image positions advance at most one per token; only video can exceed
    # max_model_len, so the vision wrapper without video keeps the short table.
    config, model = _configs(262144, language_model_only=False, videos=0)
    assert _qsa_rope_cache_length(config, model) == 65536


def test_bounded_mrope_rows_match_the_full_cache(default_vllm_config):
    def cache(max_position):
        return MRotaryEmbedding(
            256,
            64,
            max_position,
            10000000,
            True,
            torch.float16,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        ).cos_sin_cache

    short, full = cache(2048), cache(8192)
    assert short.shape[0] == 8192 and full.shape[0] == 32768
    assert torch.equal(short, full[: short.shape[0]])
