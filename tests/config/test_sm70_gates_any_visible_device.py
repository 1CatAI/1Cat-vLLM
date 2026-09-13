# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for #412: the SM70 gates in ``VllmConfig`` must consider
every visible device, not only device 0."""

import os
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

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
SM80 = (8, 0)

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


def _placement_config(world_size=1, **overrides):
    values = dict(
        distributed_executor_backend="mp" if world_size > 1 else "uni",
        data_parallel_backend="mp",
        world_size=world_size,
        local_world_size=world_size,
        nnodes_within_dp=1,
        data_parallel_rank_local=0,
        data_parallel_index=0,
        tensor_parallel_size=1,
        pipeline_parallel_size=world_size,
    )
    values.update(overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**values),
        device_config=SimpleNamespace(device=torch.device("cuda")),
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
def test_any_participating_device_is_capability(monkeypatch, capabilities, expected):
    from vllm.config.vllm import _any_participating_device_is_capability

    monkeypatch.setattr(platforms, "current_platform", _fake_platform(capabilities))
    cfg = _placement_config(world_size=len(capabilities))
    assert _any_participating_device_is_capability(cfg, SM70) is expected


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        pytest.param([SM70], True, id="volta"),
        pytest.param([SM75], True, id="turing"),
        pytest.param([SM80], False, id="ampere"),
        pytest.param([SM80, SM75], True, id="turing-behind-ampere"),
        pytest.param([SM75, SM70], True, id="mixed-pre-ampere"),
        pytest.param([], False, id="no-devices"),
    ],
)
def test_any_participating_device_is_pre_ampere(monkeypatch, capabilities, expected):
    """The SM70 baseline is a pre-Ampere tuning: Turing counts, Ampere does not."""
    from vllm.config.vllm import _any_participating_device_is_pre_ampere

    monkeypatch.setattr(platforms, "current_platform", _fake_platform(capabilities))
    cfg = _placement_config(world_size=len(capabilities))
    assert _any_participating_device_is_pre_ampere(cfg) is expected


def test_any_participating_device_is_capability_requires_cuda(monkeypatch):
    from vllm.config.vllm import _any_participating_device_is_capability

    monkeypatch.setattr(
        platforms, "current_platform", _fake_platform([SM70], is_cuda=False)
    )
    assert _any_participating_device_is_capability(_placement_config(), SM70) is False


def test_prefix_anchored_swa_requires_supported_hardware_on_every_rank(monkeypatch):
    """Do not relax backend support because only one participant is SM70."""
    monkeypatch.setattr(platforms, "current_platform", _fake_platform([SM75, SM70]))
    cfg = _placement_config(world_size=2)
    cfg.attention_config = SimpleNamespace(prefix_anchored_decode_window=64)
    with pytest.raises(ValueError, match="every participating rank"):
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


def _build_vllm_config(pp_size: int = 1) -> VllmConfig:
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
        parallel_config=ParallelConfig(
            pipeline_parallel_size=pp_size,
            distributed_executor_backend="mp" if pp_size > 1 else "uni",
        ),
    )


@pytest.mark.parametrize(
    ("capabilities", "pp_size", "expected"),
    [
        pytest.param([SM75, SM70], 2, True, id="sm70-stage-behind-sm75"),
        pytest.param([SM75, SM75], 2, True, id="homogeneous-sm75-is-pre-ampere"),
        pytest.param([SM70, SM70], 2, True, id="homogeneous-sm70"),
        pytest.param([SM80, SM70], 1, False, id="unused-sm70-must-not-change-defaults"),
        pytest.param([SM80, SM80], 2, False, id="ampere-must-stay-clean"),
    ],
)
def test_sm70_baseline_defaults_follow_any_visible_device(
    monkeypatch, isolated_baseline_env, capabilities, pp_size, expected
):
    """The Flash-V100 baseline follows the participating devices, and it is a
    pre-Ampere tuning: a mixed sm75/sm70 deployment gets it, and so does a
    homogeneous sm75 one. An sm70 card that no rank uses must not pull it in,
    and Ampere and later must stay clean."""
    _patch_visible_devices(monkeypatch, capabilities)

    _build_vllm_config(pp_size)

    applied = {name for name in BASELINE_ENV if os.environ.get(name) == "1"}
    assert (applied == set(BASELINE_ENV)) is expected
    if not expected:
        assert not applied


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, (0,)),
        ({"data_parallel_rank_local": 2}, (2,)),
        ({"data_parallel_rank_local": None, "data_parallel_index": 1}, (1,)),
        ({"world_size": 4, "local_world_size": 2, "nnodes_within_dp": 2}, (0, 1)),
        ({"distributed_executor_backend": "external_launcher"}, (3,)),
        ({"distributed_executor_backend": "ray"}, (0,)),
    ],
)
def test_participation_matches_executor_assignment(monkeypatch, overrides, expected):
    from vllm.config.vllm import _participating_cuda_device_ids

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(platforms, "current_platform", _fake_platform([SM75] * 4))
    assert _participating_cuda_device_ids(_placement_config(**overrides)) == expected


def test_explicit_uniproc_device_is_preserved(monkeypatch):
    from vllm.config.vllm import _participating_cuda_device_ids

    monkeypatch.setattr(platforms, "current_platform", _fake_platform([SM75, SM70]))
    cfg = _placement_config()
    cfg.device_config.device = torch.device("cuda:1")
    assert _participating_cuda_device_ids(cfg) == (1,)
