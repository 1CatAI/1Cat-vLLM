# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A checkpoint's KV-cache quantization directive is honored only on Ampere+.

The directive describes how the weights were made; on Volta and Turing there
is no FP8 hardware, so under ``--kv-cache-dtype auto`` the KV cache keeps the
model dtype. An explicit ``--kv-cache-dtype`` is never touched.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm import platforms
from vllm.config import CacheConfig, VllmConfig
from vllm.config import vllm as vllm_config_module
from vllm.config.vllm import checkpoint_kv_quant_allowed

SM70 = (7, 0)
SM75 = (7, 5)
SM80 = (8, 0)
SM90 = (9, 0)


def _fake_platform(capabilities: list[tuple[int, int]]):
    return SimpleNamespace(
        is_cuda=lambda: True,
        device_count=lambda: len(capabilities),
        is_device_capability=lambda capability, device_id=0: (
            capabilities[device_id] == capability
        ),
    )


def _placement_config(world_size: int):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp" if world_size > 1 else "uni",
            data_parallel_backend="mp",
            world_size=world_size,
            local_world_size=world_size,
            nnodes_within_dp=1,
            data_parallel_rank_local=0,
            data_parallel_index=0,
            tensor_parallel_size=world_size,
            pipeline_parallel_size=1,
        ),
        device_config=SimpleNamespace(device=torch.device("cuda")),
    )


@pytest.mark.parametrize(
    ("capabilities", "world_size", "expected"),
    [
        pytest.param([SM80], 1, True, id="ampere"),
        pytest.param([SM90, SM90], 2, True, id="hopper-pair"),
        pytest.param([SM70], 1, False, id="volta"),
        pytest.param([SM75], 1, False, id="turing"),
        pytest.param([SM75, SM70], 2, False, id="mixed-pre-ampere"),
        pytest.param([SM80, SM75], 2, False, id="ampere-with-turing"),
        # Only participating devices count: the Turing card is visible but
        # not part of this single-GPU engine.
        pytest.param([SM80, SM75], 1, True, id="turing-visible-not-used"),
    ],
)
def test_policy_follows_participating_devices(
    monkeypatch, capabilities, world_size, expected
):
    monkeypatch.setattr(platforms, "current_platform", _fake_platform(capabilities))
    assert checkpoint_kv_quant_allowed(_placement_config(world_size)) is expected


def _checkpoint_resolved_cache_config() -> CacheConfig:
    cache_config = CacheConfig(cache_dtype="fp8_e4m3")
    cache_config.cache_dtype_from_checkpoint = True
    return cache_config


def test_checkpoint_directive_dropped_on_pre_ampere(monkeypatch):
    monkeypatch.setattr(
        vllm_config_module, "_any_participating_device_is_pre_ampere", lambda cfg: True
    )
    config = VllmConfig(cache_config=_checkpoint_resolved_cache_config())
    assert config.cache_config.cache_dtype == "auto"
    assert config.cache_config.cache_dtype_from_checkpoint is False


def test_checkpoint_directive_kept_on_ampere(monkeypatch):
    monkeypatch.setattr(
        vllm_config_module, "_any_participating_device_is_pre_ampere", lambda cfg: False
    )
    config = VllmConfig(cache_config=_checkpoint_resolved_cache_config())
    assert config.cache_config.cache_dtype == "fp8_e4m3"
    assert config.cache_config.cache_dtype_from_checkpoint is True


def test_explicit_request_is_never_touched(monkeypatch):
    monkeypatch.setattr(
        vllm_config_module, "_any_participating_device_is_pre_ampere", lambda cfg: True
    )
    config = VllmConfig(cache_config=CacheConfig(cache_dtype="fp8_e4m3"))
    assert config.cache_config.cache_dtype == "fp8_e4m3"
