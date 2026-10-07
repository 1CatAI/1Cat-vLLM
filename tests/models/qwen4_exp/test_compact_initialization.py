# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_gdn_norm_follows_parameter_device(monkeypatch, default_vllm_config, device):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.mamba.gdn import base
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as module

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU/meta model construction must not select a GPU")

    for source in (base, linear, parameter):
        monkeypatch.setattr(source, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(source, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(module.current_platform, "current_device", forbidden)
    monkeypatch.setattr(
        module, "_resolve_gdn_prefill_backend", lambda _: ("triton", "triton")
    )
    monkeypatch.setattr(module, "_log_gdn_backend_decision", lambda *args: None)
    config = SimpleNamespace(
        hidden_size=256,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    from vllm.config import get_current_vllm_config

    runtime = get_current_vllm_config()
    with torch.device(device):
        attention = module.QwenGatedDeltaNetAttention(config, runtime, "layers.0")
    assert attention.norm.weight.device == attention.dt_bias.device
    assert attention.norm.weight.device.type == device
