# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for #412: the SM70 gates in ``VllmConfig`` must consider
every visible device, not only device 0."""

import os
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from vllm import platforms
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.vllm import apply_prefix_anchored_swa_constraints

SM70 = (7, 0)
SM75 = (7, 5)

# The unconditional part of the SM70 Flash-V100 baseline defaults.
BASELINE_ENV = (
    "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE",
    "VLLM_SM70_GDN_KKT_SCHEDULE",
    "VLLM_SM70_GDN_DELTA_H_SCHEDULE",
    "VLLM_SM70_GDN_CHUNK_O_SCHEDULE",
    "VLLM_SM70_FLA_RECURRENT_SCHEDULE",
    "VLLM_SM70_FUSED_SIGMOID_GATING_SCHED",
    "VLLM_SM70_GEMMA_RMS_NORM_COMPILE_NATIVE",
    "VLLM_SM70_GDN_DECODE_FLASHQLA",
)


def _fake_platform(capabilities: list[tuple[int, int]], is_cuda: bool = True):
    return SimpleNamespace(
        is_cuda=lambda: is_cuda,
        device_count=lambda: len(capabilities),
        is_device_capability=lambda capability, device_id=0: (
            capabilities[device_id] == capability
        ),
    )


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        pytest.param([SM70], True, id="single-sm70"),
        pytest.param([SM75, SM70], True, id="sm75-first"),
        pytest.param([SM70, SM75], True, id="sm70-first"),
        pytest.param([SM75, SM75], False, id="homogeneous-sm75"),
        pytest.param([], False, id="no-devices"),
    ],
)
def test_any_visible_device_is_capability(monkeypatch, capabilities, expected):
    from vllm.config.vllm import _any_visible_device_is_capability

    monkeypatch.setattr(platforms, "current_platform", _fake_platform(capabilities))
    assert _any_visible_device_is_capability(SM70) is expected


def test_any_visible_device_is_capability_requires_cuda(monkeypatch):
    from vllm.config.vllm import _any_visible_device_is_capability

    monkeypatch.setattr(
        platforms, "current_platform", _fake_platform([SM70], is_cuda=False)
    )
    assert _any_visible_device_is_capability(SM70) is False


def test_prefix_anchored_swa_accepts_sm70_behind_sm75(monkeypatch):
    """The engine contract only needs an SM70 device somewhere in the grid."""
    monkeypatch.setattr(platforms, "current_platform", _fake_platform([SM75, SM70]))
    cfg = SimpleNamespace(
        attention_config=SimpleNamespace(prefix_anchored_decode_window=64),
    )
    # Passing the capability gate means reaching the model-config check.
    with pytest.raises(ValueError, match="requires a model config"):
        cfg.model_config = None
        apply_prefix_anchored_swa_constraints(cfg)


@pytest.fixture
def isolated_baseline_env() -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in BASELINE_ENV}
    for name in BASELINE_ENV:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _patch_visible_devices(monkeypatch, capabilities: list[tuple[int, int]]):
    # Patch the platform instance, not its class: other tests leave
    # instance-level attributes behind that would shadow a class patch.
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_count", lambda: len(capabilities))
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: capabilities[device_id] == capability,
    )
    assert current_platform.device_count() == len(capabilities)
    assert current_platform.is_device_capability((7, 0)) is (capabilities[0] == (7, 0))


def _build_vllm_config() -> VllmConfig:
    model_config = ModelConfig(model="facebook/opt-125m", dtype="float16", seed=42)
    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=16, gpu_memory_utilization=0.9, cache_dtype="auto"
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=10,
            max_num_batched_tokens=512,
            max_model_len=512,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        parallel_config=ParallelConfig(),
    )


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        pytest.param([SM75, SM70], True, id="sm70-stage-behind-sm75"),
        pytest.param([SM75, SM75], False, id="homogeneous-sm75"),
    ],
)
def test_sm70_baseline_defaults_follow_any_visible_device(
    monkeypatch, isolated_baseline_env, capabilities, expected
):
    """PP stage 0 on an sm75 card, stage 1 on sm70: the Flash-V100 baseline
    defaults must still be applied; a homogeneous sm75 box must stay clean."""
    _patch_visible_devices(monkeypatch, capabilities)

    _build_vllm_config()

    applied = {name for name in BASELINE_ENV if os.environ.get(name) == "1"}
    assert (applied == set(BASELINE_ENV)) is expected
    if not expected:
        assert not applied
