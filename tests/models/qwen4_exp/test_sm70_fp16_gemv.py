# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.models.qwen2_moe import (
    _sm70_force_shared_expert_silu_custom_op,
    _sm70_fused_shared_expert_gate_module_supported,
    _sm70_fused_shared_expert_gate_shape_supported,
)
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import _plan_for


def test_qwen38_sm70_fp16_gemv_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "VLLM_SM70_QWEN38_FP16_GEMV"
    monkeypatch.delenv(name, raising=False)
    assert not envs.VLLM_SM70_QWEN38_FP16_GEMV
    monkeypatch.setenv(name, "1")
    assert envs.VLLM_SM70_QWEN38_FP16_GEMV


def test_qwen38_sm70_fp16_hc_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "VLLM_SM70_QWEN38_FUSED_HC_FP16"
    monkeypatch.delenv(name, raising=False)
    assert not envs.VLLM_SM70_QWEN38_FUSED_HC_FP16
    monkeypatch.setenv(name, "1")
    assert envs.VLLM_SM70_QWEN38_FUSED_HC_FP16


def test_qwen38_sm70_fp16_gdn_input_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16"
    monkeypatch.delenv(name, raising=False)
    assert not envs.VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16
    monkeypatch.setenv(name, "1")
    assert envs.VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16


@pytest.mark.parametrize("tokens", (1, 4, 8, 16))
def test_qwen38_sm70_fused_shared_gate_accepts_small_batch(tokens: int) -> None:
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    out = torch.empty_like(x)
    assert _sm70_fused_shared_expert_gate_shape_supported(x, out)


def test_qwen38_sm70_fused_shared_gate_rejects_unsupported_shape() -> None:
    x = torch.empty(17, 2560, dtype=torch.float16)
    assert not _sm70_fused_shared_expert_gate_shape_supported(x, torch.empty_like(x))
    assert not _sm70_fused_shared_expert_gate_shape_supported(
        x[:16], torch.empty(15, 2560, dtype=torch.float16)
    )
    assert not _sm70_fused_shared_expert_gate_shape_supported(
        x[:16].float(), torch.empty(16, 2560, dtype=torch.float16)
    )


def test_qwen38_sm70_fused_shared_gate_uses_local_tp_shape() -> None:
    gate_up = SimpleNamespace(
        input_size_per_partition=2560,
        output_partition_sizes=(160, 160),
    )
    down = SimpleNamespace(
        input_size_per_partition=160,
        output_size_per_partition=2560,
    )
    assert _sm70_fused_shared_expert_gate_module_supported(gate_up, down)

    gate_up.output_partition_sizes = (640, 640)
    assert not _sm70_fused_shared_expert_gate_module_supported(gate_up, down)


def test_qwen38_sm70_shared_expert_custom_silu_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))
    assert _sm70_force_shared_expert_silu_custom_op("layers.0.mlp.shared_expert")
    assert not _sm70_force_shared_expert_silu_custom_op("layers.0.mlp.experts")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert not _sm70_force_shared_expert_silu_custom_op("layers.0.mlp.shared_expert")


@pytest.mark.parametrize(
    ("prefix", "shape"),
    [
        (
            "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject",
            (336, 10240),
        ),
        ("model.layers.0.linear_attn.in_proj_qkvz", (4096, 2560)),
        ("model.layers.0.linear_attn.in_proj_ba", (24, 2560)),
        ("model.layers.0.linear_attn.out_proj", (2560, 1536)),
        ("model.layers.3.self_attn.qkv_proj", (3584, 2560)),
        ("model.layers.3.self_attn.o_proj", (2560, 1536)),
        ("model.layers.3.self_attn.indexer.index_qk_proj", (640, 2560)),
        ("model.layers.3.mlp.gate", (512, 2560)),
    ],
)
def test_qwen38_sm70_fp16_gemv_exact_role_allowlist(
    prefix: str, shape: tuple[int, int]
) -> None:
    assert _plan_for(prefix, shape) is not None


@pytest.mark.parametrize(
    ("prefix", "shape"),
    [
        ("model.layers.0.attn_hyper_connection.input_mix_weight_up", (10240, 320)),
        ("model.layers.0.mlp.shared_expert.gate_up_proj", (320, 2560)),
        ("model.layers.0.mlp.shared_expert_gate", (1, 2560)),
        ("model.layers.0.linear_attn.out_proj", (2560, 2560)),
        ("other.linear_attn.in_proj_qkvz", (4096, 2561)),
    ],
)
def test_qwen38_sm70_fp16_gemv_rejects_other_roles(
    prefix: str, shape: tuple[int, int]
) -> None:
    assert _plan_for(prefix, shape) is None
