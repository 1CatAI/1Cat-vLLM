# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.layers import rotary_embedding as rope

pytestmark = pytest.mark.skip_global_cleanup


def _config(video: int | None, *, multimodal: bool = True) -> VllmConfig:
    config = VllmConfig()
    limits = {"image": 1}
    if video is not None:
        limits["video"] = video
    config.model_config = SimpleNamespace(
        multimodal_config=MultiModalConfig(limit_per_prompt=limits)
        if multimodal
        else None,
        max_model_len=512,
    )
    return config


def _rope(**kwargs):
    return rope.get_rope(
        head_size=256,
        max_position=512,
        dtype=torch.float16,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            **kwargs,
        },
    )


@pytest.mark.parametrize(("video", "rows"), [(0, 512), (1, 2048), (None, 2048)])
def test_video_capacity_and_exact_cache_prefix(monkeypatch, video, rows):
    monkeypatch.setattr(rope, "_ROPE_DICT", {})
    with set_current_vllm_config(_config(1)):
        reference = _rope()
    with set_current_vllm_config(_config(video)):
        actual = _rope()
    assert actual.cos_sin_cache.shape == (rows, 64)
    assert torch.equal(actual.cos_sin_cache, reference.cos_sin_cache[:rows])
    positions = torch.tensor([[0, 511], [511, 0], [510, 511]])
    query = torch.randn(2, 6 * 256, dtype=torch.float16)
    key = torch.randn(2, 256, dtype=torch.float16)
    expected = reference.forward_native(positions, query.clone(), key.clone())
    observed = actual.forward_native(positions, query.clone(), key.clone())
    assert all(torch.equal(x, y) for x, y in zip(expected, observed))


def test_video_engine_does_not_reuse_bounded_cache(monkeypatch):
    monkeypatch.setattr(rope, "_ROPE_DICT", {})
    with set_current_vllm_config(_config(0)):
        bounded = _rope()
        assert _rope() is bounded
    with set_current_vllm_config(_config(1)):
        full = _rope()
        assert _rope() is full
    assert full is not bounded
    assert full.cos_sin_cache.shape[0] == 2048


def test_text_only_engine_keeps_served_context(monkeypatch):
    monkeypatch.setattr(rope, "_ROPE_DICT", {})
    config = _config(0, multimodal=False)
    config.model_config.max_model_len = 1024
    with set_current_vllm_config(config):
        assert _rope().cos_sin_cache.shape[0] == 1024


def test_unknown_engine_keeps_standalone_capacity(monkeypatch):
    monkeypatch.setattr(rope, "_ROPE_DICT", {})
    with set_current_vllm_config(VllmConfig()):
        assert _rope().cos_sin_cache.shape[0] == 2048


def test_yarn_retains_existing_frequency_and_capacity(monkeypatch):
    monkeypatch.setattr(rope, "_ROPE_DICT", {})
    args = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 512}
    with set_current_vllm_config(_config(1)):
        reference = _rope(**args)
    with set_current_vllm_config(_config(0)):
        actual = _rope(**args)
    assert actual is reference
