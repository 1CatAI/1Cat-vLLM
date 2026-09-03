# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import _plan_for
from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import (
    _qwen38_hc_down_local_shard_kernel,
    _qwen38_hc_down_silu_inject_kernel,
    _qwen38_hc_up_gate_mix_kernel,
    _qwen38_hc_up_gate_mix_row4_kernel,
    _qwen38_hc_up_local_gate_kernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON


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


@pytest.mark.skipif(
    not current_platform.is_device_capability((7, 0)) or not HAS_TRITON,
    reason="Qwen3.8 HC row-tile kernel requires CUDA SM70 and Triton",
)
def test_qwen38_sm70_hc_up_row4_is_bitwise() -> None:
    lora = torch.empty(1, 320, dtype=torch.float16, device="cuda")
    weight = torch.randn(10240, 320, dtype=torch.float16, device="cuda")
    branches = torch.empty(1, 10240, dtype=torch.float16, device="cuda")
    reference = torch.empty(1, 2560, dtype=torch.float16, device="cuda")
    actual = torch.empty_like(reference)

    for seed in range(8):
        torch.manual_seed(seed)
        lora.normal_()
        branches.normal_()
        _qwen38_hc_up_gate_mix_kernel[(2560,)](
            lora,
            weight,
            branches,
            reference,
            K=320,
            HC_DIMENSION=2560,
            HC_COUNT=4,
            BLOCK_K=512,
            num_warps=2,
        )
        _qwen38_hc_up_gate_mix_row4_kernel[(640,)](
            lora,
            weight,
            branches,
            actual,
            K=320,
            HC_DIMENSION=2560,
            HC_COUNT=4,
            BLOCK_N=4,
            BLOCK_K=512,
            num_warps=8,
        )
        torch.accelerator.synchronize()
        assert torch.equal(actual, reference)


@pytest.mark.skipif(
    not current_platform.is_device_capability((7, 0)) or not HAS_TRITON,
    reason="Qwen3.8 HC TP4 shards require CUDA SM70 and Triton",
)
def test_qwen38_sm70_hc_tp4_compute_shards_are_bitwise() -> None:
    x = torch.empty(1, 10240, dtype=torch.float16, device="cuda")
    down_weight = torch.randn(336, 10240, dtype=torch.float16, device="cuda")
    up_weight = torch.randn(10240, 320, dtype=torch.float16, device="cuda")
    reference_lora = torch.empty(1, 320, dtype=torch.float16, device="cuda")
    reference_injection = torch.empty(1, 4, dtype=torch.float16, device="cuda")
    reference_block = torch.empty(1, 2560, dtype=torch.float16, device="cuda")

    for seed in range(4):
        torch.manual_seed(seed)
        x.normal_()
        _qwen38_hc_down_silu_inject_kernel[(324,)](
            x,
            down_weight,
            reference_lora,
            reference_injection,
            K=10240,
            BLOCK_K=256,
            RANK_VALUE=320,
            HC_COUNT=4,
            num_warps=4,
        )
        _qwen38_hc_up_gate_mix_row4_kernel[(640,)](
            reference_lora,
            up_weight,
            x,
            reference_block,
            K=320,
            HC_DIMENSION=2560,
            HC_COUNT=4,
            BLOCK_N=4,
            BLOCK_K=512,
            num_warps=8,
        )

        local_down = []
        local_gates = []
        for rank in range(4):
            shard = torch.empty(1, 88, dtype=torch.float16, device="cuda")
            _qwen38_hc_down_local_shard_kernel[(88,)](
                x,
                down_weight,
                shard,
                TP_RANK=rank,
                num_warps=4,
            )
            local_down.append(shard)
        gathered_lora = torch.cat([shard[..., :80] for shard in local_down], dim=-1)
        gathered_injection = torch.cat(
            [shard[..., 80:81] for shard in local_down], dim=-1
        )
        for rank in range(4):
            gate = torch.empty(1, 2560, dtype=torch.float16, device="cuda")
            _qwen38_hc_up_local_gate_kernel[(320,)](
                gathered_lora,
                up_weight,
                gate,
                TP_RANK=rank,
                BLOCK_N=8,
                num_warps=8,
            )
            local_gates.append(gate)
        actual_block = hc_gate_mix(x, torch.cat(local_gates, dim=-1), 4)
        torch.accelerator.synchronize()

        assert torch.equal(gathered_lora, reference_lora)
        assert torch.equal(gathered_injection, reference_injection)
        assert torch.equal(actual_block, reference_block)
