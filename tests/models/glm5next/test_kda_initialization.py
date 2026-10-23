# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA runtime owners must exist before either model runner starts warmup."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.fla.ops import gdn_chunk_kernels as factory
from vllm.model_executor.layers.mamba.gdn import base
from vllm.models.glm5next.nvidia import kda
from vllm.runtime_resources import runtime_resources_for
from vllm.transformers_utils.configs.glm5_next import Glm5NextTextConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def cpu_construction(monkeypatch):
    # Execute the real GLM constructor and real projection constructors without
    # requiring distributed initialization, CUDA allocations, or a checkpoint.
    for module in (linear, parameter, base):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(kda.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _construct(vconfig, index, schema):
    if schema == "flat":
        config = Glm5NextTextConfig(
            hidden_size=8,
            num_hidden_layers=2,
            linear_head_dim=4,
            linear_num_heads=2,
            linear_conv_kernel_dim=4,
        )
    else:
        config = KimiLinearConfig(
            hidden_size=8,
            num_hidden_layers=2,
            linear_attn_config={
                "kda_layers": [1, 2],
                "full_attn_layers": [],
                "head_dim": 4,
                "num_heads": 2,
                "short_conv_kernel_size": 4,
            },
        )
    with set_current_vllm_config(vconfig), torch.device("cpu"):
        return kda.Glm5NextLinearAttention(
            config, vconfig, prefix=f"model.layers.{index}.self_attn"
        )


@pytest.mark.parametrize("schema", ["flat", "legacy"])
@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_constructor_binds_shared_per_engine_kda_owners(
    cpu_construction, monkeypatch, schema, order
):
    created = []

    def create_chunk(schedule):
        assert schedule.resolved
        return factory.GdnChunkKernels(object(), object(), object())

    def create_kda(schedule, delta_h):
        owner = factory.KdaChunkKernels(
            SimpleNamespace(cache={}, enabled=schedule.kda_prefill_enabled),
            delta_h,
            object(),
        )
        created.append(owner)
        return owner

    # Only replace CUDA autotuner construction; the production binder, resource
    # map, scheduling capture, and complete GLM constructors remain real.
    monkeypatch.setattr(factory, "create_chunk_kernels", create_chunk)
    monkeypatch.setattr(factory, "create_kda_kernels", create_kda)
    engines = []
    layers = []
    for enabled in order:
        engine = VllmConfig(device_config=DeviceConfig(device="cpu"))
        engine.kernel_config.gdn.schedule.kda_prefill_enabled = enabled
        engines.append(engine)
        layers.append([_construct(engine, i, schema) for i in range(2)])

    assert len(created) == 2
    for engine, pair, enabled in zip(engines, layers, order):
        owner = pair[0].chunk_kernels
        assert pair[1].chunk_kernels is owner
        resources = runtime_resources_for(engine)
        assert resources["kda_chunk_kernels"] is owner
        assert owner.delta_h is resources["gdn_chunk_kernels"].delta_h
        assert owner.recompute.enabled is enabled
        with set_current_vllm_config(engine):
            assert factory.resolve_kda_kernels() is owner
        assert all(p.device.type == "cpu" for layer in pair for p in layer.parameters())

    first, second = (pair[0].chunk_kernels for pair in layers)
    assert first is not second
    first.recompute.cache["shape"] = "winner"
    assert second.recompute.cache == {}
    # Constructor ownership is independent of which engine most recently ran.
    with set_current_vllm_config(engines[1]):
        assert factory.resolve_kda_kernels(first) is first


def test_constructor_initializes_no_cuda_resource_entries(cpu_construction):
    engine = VllmConfig(device_config=DeviceConfig(device="cpu"))
    layer = _construct(engine, 0, "flat")
    resources = runtime_resources_for(engine)
    assert layer.chunk_kernels is None
    assert resources["kda_chunk_kernels"] is None
    assert resources["gdn_chunk_kernels"] is None
    assert engine.kernel_config.gdn.schedule.resolved
