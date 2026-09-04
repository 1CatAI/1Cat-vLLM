# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm import envs
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _QWEN38_COMPACT_GROUPED_MAX_SLOTS,
    _use_qwen38_compact_grouped_decode,
)
from vllm.model_executor.warmup import awq_sm70_warmup as warmup

pytestmark = pytest.mark.skip_global_cleanup


def _qwen38_layer() -> SimpleNamespace:
    return SimpleNamespace(
        moe_config=SimpleNamespace(tp_size=4),
        sm70_awq_qwen38_compact_grouped_decode=True,
        sm70_awq_moe_batched_gemm=True,
        sm70_awq_group_size=32,
        sm70_num_experts=512,
        sm70_hidden_logical_size=2560,
        sm70_intermediate_size=160,
        sm70_w13_k_dim=2560,
        sm70_w13_n_dim=320,
        sm70_w2_k_dim=160,
        sm70_w2_n_dim=2560,
    )


def test_qwen38_awq_compact_grouped_decode_defaults_on_with_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE"
    monkeypatch.delenv(name, raising=False)
    assert envs.VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE

    monkeypatch.setenv(name, "0")
    assert not envs.VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE


def test_qwen38_awq_compact_grouped_decode_gate_is_exact() -> None:
    layer = _qwen38_layer()

    assert _QWEN38_COMPACT_GROUPED_MAX_SLOTS == 80
    assert not _use_qwen38_compact_grouped_decode(layer, 1, 10)
    assert _use_qwen38_compact_grouped_decode(layer, 2, 10)
    assert _use_qwen38_compact_grouped_decode(layer, 4, 10)
    assert _use_qwen38_compact_grouped_decode(layer, 8, 10)
    assert not _use_qwen38_compact_grouped_decode(layer, 9, 10)
    assert not _use_qwen38_compact_grouped_decode(layer, 8, 8)

    layer.moe_config.tp_size = 2
    assert not _use_qwen38_compact_grouped_decode(layer, 4, 10)
    layer.moe_config.tp_size = 4

    layer.sm70_awq_group_size = 128
    assert not _use_qwen38_compact_grouped_decode(layer, 4, 10)
    layer.sm70_awq_group_size = 32

    layer.sm70_w2_n_dim = 2592
    assert not _use_qwen38_compact_grouped_decode(layer, 4, 10)
    layer.sm70_w2_n_dim = 2560

    layer.sm70_awq_qwen38_compact_grouped_decode = False
    assert not _use_qwen38_compact_grouped_decode(layer, 4, 10)


def _warmup_layer() -> nn.Module:
    layer = nn.Module()
    layer._awq_moe_buf_top_k = 10
    layer.sm70_num_experts = 512
    layer.sm70_w13_k_dim = 2560
    layer.sm70_w13_n_dim = 320
    layer.sm70_w2_k_dim = 160
    layer.sm70_w2_n_dim = 2560
    layer.sm70_awq_moe_w13_interleaved = False
    layer.sm70_awq_qwen38_compact_grouped_decode = True
    layer.sm70_awq_compact_grouped_max_slots = 80
    layer.w13_tm_scales = torch.empty((80, 320), dtype=torch.float16)
    layer.w13_strided_ptrs_w = torch.empty(1, dtype=torch.uint8)
    layer.w13_strided_ptrs_s = torch.empty(1, dtype=torch.uint8)
    layer.w2_strided_ptrs_w = torch.empty(1, dtype=torch.uint8)
    layer.w2_strided_ptrs_s = torch.empty(1, dtype=torch.uint8)
    return layer


def test_awq_warmup_uses_compact_groups_only_through_c8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _warmup_layer()
    dense_calls: list[tuple] = []
    compact_calls: list[tuple] = []
    monkeypatch.setattr(
        torch.ops._C,
        "awq_moe_compact_grouped_dense_stage_sm70_out",
        object(),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops._C,
        "awq_moe_dense_stage_sm70_out",
        object(),
        raising=False,
    )
    monkeypatch.setattr(
        warmup.sm70_ops,
        "awq_moe_dense_stage_sm70_out",
        lambda *args: dense_calls.append(args),
    )
    monkeypatch.setattr(
        warmup.sm70_ops,
        "awq_moe_compact_grouped_dense_stage_sm70_out",
        lambda *args: compact_calls.append(args),
    )
    monkeypatch.setattr(
        warmup,
        "_silu_and_mul_w13",
        lambda layer, out, gate_up: out.zero_(),
    )

    assert warmup._warmup_moe_dense_stage_layers([layer], [1, 4, 8, 9]) == 8

    assert [call[6] for call in compact_calls] == [40, 40, 80, 80]
    assert all(call[2].tolist() == list(range(call[6] + 1)) for call in compact_calls)
    assert all(call[3].tolist() == list(range(call[6])) for call in compact_calls)
    assert [call[6] for call in dense_calls] == [512, 512, 512, 512]
