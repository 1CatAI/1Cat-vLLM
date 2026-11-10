# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal limits in the compile-cache hash.

A zero limit changes the traced language-model graph, so configs that differ
only in which modalities are disabled must not share a compile cache. Non-zero
limits do not change the graph and must keep the hash unchanged, so that
existing caches stay valid.
"""

import json

import pytest

from vllm.config.model import ModelConfig, mm_limit_hash_factors
from vllm.config.multimodal import MultiModalConfig


def _mm(**kwargs) -> MultiModalConfig:
    return MultiModalConfig(**kwargs)


@pytest.mark.parametrize(
    "limits",
    [
        {},
        {"image": 16, "video": 1, "audio": 16},
        {"image": 8},
        {"video": {"count": 1, "num_frames": 32, "width": 512, "height": 512}},
    ],
)
def test_non_zero_limits_add_no_factor(limits):
    assert mm_limit_hash_factors(_mm(limit_per_prompt=limits)) == {}
    assert (
        mm_limit_hash_factors(_mm(limit_per_prompt=limits, enable_mm_embeds=True)) == {}
    )


def test_zero_limits_are_hashed_as_a_sorted_set():
    factors = mm_limit_hash_factors(_mm(limit_per_prompt={"video": 0, "image": 0}))
    assert factors == {
        "mm_zero_limit_modalities": ["image", "video"],
        "mm_enable_mm_embeds": False,
    }
    assert (
        mm_limit_hash_factors(_mm(limit_per_prompt={"image": 0, "video": 0})) == factors
    )


def test_option_form_zero_count_counts_as_zero():
    factors = mm_limit_hash_factors(
        _mm(limit_per_prompt={"video": {"count": 0, "num_frames": 8}})
    )
    assert factors["mm_zero_limit_modalities"] == ["video"]


def test_enable_mm_embeds_distinguishes_zero_limit_configs():
    off = mm_limit_hash_factors(_mm(limit_per_prompt={"image": 0, "video": 0}))
    on = mm_limit_hash_factors(
        _mm(limit_per_prompt={"image": 0, "video": 0}, enable_mm_embeds=True)
    )
    assert off != on


@pytest.fixture
def vl_model_dir(tmp_path):
    """A local Qwen2-VL config, enough for ModelConfig without network."""
    config = {
        "architectures": ["Qwen2VLForConditionalGeneration"],
        "model_type": "qwen2_vl",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
        "vision_config": {
            "embed_dim": 32,
            "depth": 1,
            "num_heads": 2,
            "hidden_size": 64,
            "in_chans": 3,
        },
        "torch_dtype": "float16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _hash(model_dir, limits=None, **kwargs):
    cfg = ModelConfig(
        model=model_dir,
        skip_tokenizer_init=True,
        limit_mm_per_prompt=limits,
        **kwargs,
    )
    return cfg.compute_hash()


def test_compute_hash_unchanged_without_zero_limits(vl_model_dir, monkeypatch):
    baseline = _hash(vl_model_dir)
    # The hash without any zero limit is exactly the previous hash: drop the
    # new factor and compare.
    import vllm.config.model as model_mod

    monkeypatch.setattr(model_mod, "mm_limit_hash_factors", lambda _cfg: {})
    assert _hash(vl_model_dir) == baseline
    assert _hash(vl_model_dir, {"image": 16, "video": 1}) == baseline
    monkeypatch.undo()
    assert _hash(vl_model_dir, {"image": 16, "video": 1}) == baseline
    assert _hash(vl_model_dir, {"image": 2, "video": 4}) == baseline


def test_compute_hash_splits_zero_limit_configs(vl_model_dir):
    baseline = _hash(vl_model_dir)
    text_only = _hash(vl_model_dir, {"image": 0, "video": 0})
    image_off = _hash(vl_model_dir, {"image": 0, "video": 1})
    embeds = _hash(vl_model_dir, {"image": 0, "video": 0}, enable_mm_embeds=True)
    assert len({baseline, text_only, image_off, embeds}) == 4
