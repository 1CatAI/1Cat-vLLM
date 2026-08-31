# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.envs as envs
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
