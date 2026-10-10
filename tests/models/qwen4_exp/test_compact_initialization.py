# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP
from vllm.models.qwen4_exp.nvidia.qsa import _qsa_rope_cache_length


@pytest.mark.parametrize("text_only", [False, True])
@pytest.mark.parametrize("rope_type", ["default", "dynamic"])
def test_rope_cache_limit_preserves_multimodal_and_dynamic_contract(
    text_only, rope_type
):
    config = SimpleNamespace(
        max_position_embeddings=262144, rope_parameters={"rope_type": rope_type}
    )
    model_config = SimpleNamespace(
        max_model_len=2048,
        multimodal_config=SimpleNamespace(language_model_only=text_only),
    )
    expected = 2048 if text_only and rope_type == "default" else 262144
    assert _qsa_rope_cache_length(config, model_config) == expected


def test_default_rope_short_cache_has_identical_values(default_vllm_config):
    short = RotaryEmbedding(64, 64, 2048, 10000, True, torch.float16)
    full = RotaryEmbedding(64, 64, 4096, 10000, True, torch.float16)
    torch.testing.assert_close(
        short.cos_sin_cache, full.cos_sin_cache[:2048], rtol=0, atol=0
    )


@pytest.mark.parametrize("shared", [False, True])
def test_mtp_checkpoint_skip_matches_io_sharing(shared):
    model = object.__new__(Qwen4ExpMTP)
    nn.Module.__init__(model)
    model.share_target_io_weights = shared
    for name in [
        "embed_tokens.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "mtp.shared_head.head.weight",
    ]:
        assert model.skip_checkpoint_weight(name) == shared
    assert not model.skip_checkpoint_weight("mtp.fc_embedding.weight")
    assert model.skip_checkpoint_weight("model.layers.1.mlp.experts.w13_weight")


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("offload_worker", [False, True])
def test_hc_staging_avoids_replicated_device_allocations(
    monkeypatch, default_vllm_config, sharded, offload_worker
):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.model_executor.model_loader.utils import device_loading_context
    from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
    from vllm.models.qwen4_exp.nvidia import hyperconnection as module

    runtime = SimpleNamespace(
        kernel_config=SimpleNamespace(
            hc_weight_storage="sharded" if sharded else "replicated"
        )
    )
    monkeypatch.setattr(module, "get_current_vllm_config_or_none", lambda: runtime)
    monkeypatch.setattr(module, "is_offload_process", lambda: offload_worker)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 4)
    with torch.device("meta"):
        hc = module.GatedResidual(
            HyperConnectionConfig(
                hidden_size=8, hc_lowrank=4, params_dtype=torch.float16
            )
        )
    staged = sharded and not offload_worker
    for layer in (hc.input_mix_weight_down_block_inject, hc.input_mix_weight_up):
        assert layer.weight.device.type == ("cpu" if staged else "meta")
        assert getattr(layer.weight, "_vllm_keep_on_cpu", False) == staged
        if staged:
            with device_loading_context(layer, torch.device("cuda")):
                assert layer.weight.device.type == "cpu"


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
    monkeypatch.setattr(runtime, "model_config", SimpleNamespace(dtype=torch.float16))
    with torch.device(device):
        attention = module.QwenGatedDeltaNetAttention(config, runtime, "layers.0")
    assert attention.norm.weight.device == attention.dt_bias.device
    assert attention.norm.weight.device.type == device
