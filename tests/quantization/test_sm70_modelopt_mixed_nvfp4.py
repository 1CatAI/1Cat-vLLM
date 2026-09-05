# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow SM70 admission and shape gates for ModelOpt mixed NVFP4."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    _QWEN38_DYNAMIC_QPN_BATCH_W13_SPLIT_K,
    _QWEN38_QPN_BATCH_W13_SPLIT_K,
    _QWEN38_QPN_M1_W13_SPLIT_K,
    ModelOptNvFp4SM70MoEMethod,
    _mtp_weighted_reduce,
    _prepare_compact_expert_groups,
    _prepare_compact_slot_groups,
    _prepare_single_token_slots,
    _raw_scales_match_prepared,
    _single_token_weighted_reduce,
    _use_compact_grouped,
    _use_qwen38_indexed_prefill,
    _use_qwen38_qpn_batch_decode,
    _use_qwen38_qpn_batch_fused_w2,
    _use_qwen38_qpn_batch_fused_w13,
    _use_qwen38_qpn_m1_decode,
    _use_qwen38_qpn_mtp5_decode,
    _validate_weight_layout,
    validate_nvfp4_sm70_moe_contract,
)


def test_qwen38_qpn_batch_uses_screened_splits():
    assert _QWEN38_QPN_BATCH_W13_SPLIT_K == {2: 10, 4: 5, 8: 4, 16: 1}


def test_qwen38_qpn_batch_covers_dynamic_graph_widths():
    assert set(_QWEN38_DYNAMIC_QPN_BATCH_W13_SPLIT_K) == set(range(2, 17))
    for tokens, split_k in _QWEN38_QPN_BATCH_W13_SPLIT_K.items():
        assert _QWEN38_DYNAMIC_QPN_BATCH_W13_SPLIT_K[tokens] == split_k


def test_qwen38_qpn_m1_w13_uses_same_precision_split16_plan():
    assert _QWEN38_QPN_M1_W13_SPLIT_K == 16


@pytest.mark.parametrize(
    ("top_k", "compact_tokens", "dense_tokens"),
    [(8, 10, 11), (10, 8, 9)],
)
def test_nvfp4_compact_work_limit_is_routed_rows(top_k, compact_tokens, dense_tokens):
    assert _use_compact_grouped(compact_tokens, top_k)
    assert not _use_compact_grouped(dense_tokens, top_k)


def _mixed_config() -> ModelOptMixedPrecisionConfig:
    fp8 = ModelOptFp8Config("FP8", True, None, [])
    nvfp4 = ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
    )
    w4a16 = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
    )
    return ModelOptMixedPrecisionConfig(
        kv_cache_quant_method=None,
        exclude_modules=[],
        quantized_layers={"model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4"}},
        fp8_config=fp8,
        nvfp4_config=nvfp4,
        w4a16_nvfp4_config=w4a16,
    )


def _moe_contract(**overrides):
    values = {
        "num_experts": 256,
        "experts_per_token": 8,
        "hidden_dim": 2048,
        "intermediate_size_per_partition": 128,
        "tp_size": 4,
        "moe_parallel_config": SimpleNamespace(use_all2all_kernels=False),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _qwen4_moe_contract(**overrides):
    values = {
        "num_experts": 512,
        "experts_per_token": 10,
        "hidden_dim": 2560,
        "intermediate_size_per_partition": 160,
        "tp_size": 4,
        "moe_parallel_config": SimpleNamespace(use_all2all_kernels=False),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_mixed_min_capability_requires_exact_sm70_and_both_turbomind_routes():
    with (
        patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=True),
        patch.object(sm70_tm, "use_turbomind", side_effect=[True, True]),
    ):
        assert ModelOptMixedPrecisionConfig.get_min_capability() == 70

    with (
        patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=True),
        patch.object(sm70_tm, "use_turbomind", side_effect=[True, False]),
    ):
        assert ModelOptMixedPrecisionConfig.get_min_capability() == 89

    with patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=False):
        assert ModelOptMixedPrecisionConfig.get_min_capability() == 89


def test_nvfp4_grouped_prefill_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL", raising=False)
    assert envs.VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL

    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL", "0")
    assert not envs.VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL


def test_qwen38_fast_prefill_defaults_on_and_can_be_disabled(monkeypatch):
    name = "VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL"
    monkeypatch.delenv(name, raising=False)
    assert envs.VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL

    monkeypatch.setenv(name, "0")
    assert not envs.VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL


def test_qwen38_w2_direct_reduce_defaults_on_and_can_be_disabled(monkeypatch):
    name = "VLLM_SM70_NVFP4_QWEN38_MOE_W2_DIRECT_REDUCE"
    monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_NVFP4_QWEN38_MOE_W2_DIRECT_REDUCE
        monkeypatch.setenv(name, "0")
        envs.disable_envs_cache()
        assert not envs.VLLM_SM70_NVFP4_QWEN38_MOE_W2_DIRECT_REDUCE
    finally:
        envs.disable_envs_cache()

    name = "VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL"
    monkeypatch.delenv(name, raising=False)
    assert envs.VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL

    monkeypatch.setenv(name, "0")
    assert not envs.VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL

    name = "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE"
    monkeypatch.delenv(name, raising=False)
    assert not envs.VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE

    monkeypatch.setenv(name, "1")
    assert envs.VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_experts", 128),
        ("experts_per_token", 6),
        ("hidden_dim", 4096),
        ("intermediate_size_per_partition", 96),
        ("tp_size", 2),
    ],
)
def test_nvfp4_moe_contract_rejects_unvalidated_shapes(field, value):
    validate_nvfp4_sm70_moe_contract(_moe_contract())
    with pytest.raises(NotImplementedError):
        validate_nvfp4_sm70_moe_contract(_moe_contract(**{field: value}))


def test_nvfp4_moe_contract_accepts_qwen4_exp_tp4():
    validate_nvfp4_sm70_moe_contract(_qwen4_moe_contract())


def test_qwen38_qpn_m1_decode_is_default_on_and_exact_shape_only(monkeypatch):
    layer = SimpleNamespace(
        moe_config=_qwen4_moe_contract(),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
    )
    x = torch.empty(1, 2560, dtype=torch.float16)
    topk_ids = torch.empty(1, 10, dtype=torch.int32)

    monkeypatch.delenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE", raising=False)
    assert _use_qwen38_qpn_m1_decode(layer, x, topk_ids)

    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE", "0")
    assert not _use_qwen38_qpn_m1_decode(layer, x, topk_ids)

    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE", "1")
    assert _use_qwen38_qpn_m1_decode(layer, x, topk_ids)
    assert not _use_qwen38_qpn_m1_decode(layer, x.repeat(2, 1), topk_ids)
    assert not _use_qwen38_qpn_m1_decode(layer, x, topk_ids.long())

    layer.moe_config.tp_size = 2
    assert not _use_qwen38_qpn_m1_decode(layer, x, topk_ids)


def test_qwen38_indexed_prefill_is_default_on_and_exact_shape_only(monkeypatch):
    layer = SimpleNamespace(
        moe_config=_qwen4_moe_contract(),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
    )
    x = torch.empty(128, 2560, dtype=torch.float16)
    topk_ids = torch.empty(128, 10, dtype=torch.int32)

    monkeypatch.delenv("VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL", raising=False)
    assert _use_qwen38_indexed_prefill(layer, x, topk_ids)

    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL", "0")
    assert not _use_qwen38_indexed_prefill(layer, x, topk_ids)

    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL", "1")
    assert _use_qwen38_indexed_prefill(layer, x, topk_ids)
    assert not _use_qwen38_indexed_prefill(layer, x[:127], topk_ids[:127])
    assert not _use_qwen38_indexed_prefill(layer, x, topk_ids[:, :9])

    layer.moe_config.tp_size = 2
    assert not _use_qwen38_indexed_prefill(layer, x, topk_ids)


@pytest.mark.parametrize("tokens", (2, 4, 8, 16))
def test_qwen38_qpn_batch_decode_defaults_on_and_is_shape_gated(monkeypatch, tokens):
    layer = SimpleNamespace(
        moe_config=_qwen4_moe_contract(),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
    )
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    topk_ids = torch.empty(tokens, 10, dtype=torch.int32)

    monkeypatch.delenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE", raising=False)
    assert _use_qwen38_qpn_batch_decode(layer, x, topk_ids)

    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE", "0")
    assert not _use_qwen38_qpn_batch_decode(layer, x, topk_ids)

    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE", "1")
    assert _use_qwen38_qpn_batch_decode(layer, x, topk_ids)
    assert not _use_qwen38_qpn_batch_decode(layer, x[:, :-1], topk_ids)
    assert not _use_qwen38_qpn_batch_decode(layer, x, topk_ids.long())

    layer.moe_config.tp_size = 2
    assert not _use_qwen38_qpn_batch_decode(layer, x, topk_ids)


@pytest.mark.parametrize("tokens", (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15))
@pytest.mark.parametrize("raw_scale", (False, True))
def test_qwen38_qpn_dynamic_widths_are_independent_of_scale_storage(
    monkeypatch, tokens, raw_scale
):
    layer = SimpleNamespace(
        moe_config=_qwen4_moe_contract(),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
        sm70_nvfp4_qwen38_raw_scale=raw_scale,
    )
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    topk_ids = torch.empty(tokens, 10, dtype=torch.int32)
    monkeypatch.delenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE", raising=False)
    name = "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE"
    monkeypatch.delenv(name, raising=False)
    assert not _use_qwen38_qpn_batch_decode(layer, x, topk_ids)

    monkeypatch.setenv(name, "1")
    assert _use_qwen38_qpn_batch_decode(layer, x, topk_ids)
    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE", "0")
    assert not _use_qwen38_qpn_batch_decode(layer, x, topk_ids)


@pytest.mark.parametrize("interleaved", (False, True))
@pytest.mark.parametrize("mismatch", (None, "prefill", "decode"))
def test_raw_scale_admission_checks_prefill_and_effective_decode(
    monkeypatch, interleaved, mismatch
):
    # A decode-scale mismatch can disappear after dividing by 2**14 and
    # rounding to stored FP16 subnormals. Admission must compare BEFORE that
    # lossy division, exactly where the QPN HMMA consumes the scale.
    prepared = torch.tensor([2**-24], dtype=torch.float16)
    effective = prepared * 16384.0
    changed_effective = torch.nextafter(
        effective, torch.full_like(effective, torch.inf)
    )
    assert torch.equal(changed_effective / 16384.0, prepared)
    workspace = torch.empty_like(prepared)
    codes = torch.zeros(1, dtype=torch.uint8)
    globals_ = torch.ones(1, dtype=torch.float32)
    calls = []

    def expand(out, actual_codes, actual_globals, actual_interleaved, fast):
        assert actual_codes is codes
        assert actual_globals is globals_
        assert actual_interleaved == interleaved
        calls.append(fast)
        if fast:
            out.copy_(changed_effective if mismatch == "decode" else effective)
        else:
            out.copy_(prepared * 2 if mismatch == "prefill" else prepared)

    monkeypatch.setattr(sm70_ops, "nvfp4_expand_raw_scales_sm70_out", expand)
    assert _raw_scales_match_prepared(
        workspace, codes, globals_, prepared, interleaved
    ) == (mismatch is None)
    assert calls == ([False] if mismatch == "prefill" else [False, True])


@pytest.mark.parametrize("tokens", (4, 8, 16))
def test_qwen38_qpn_batch_fused_w13_defaults_on_and_is_shape_gated(monkeypatch, tokens):
    layer = SimpleNamespace(
        moe_config=_qwen4_moe_contract(),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
        sm70_nvfp4_qwen38_fused_swiglu_prefill=True,
        swiglu_limit=None,
    )
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    topk_ids = torch.empty(tokens, 10, dtype=torch.int32)
    monkeypatch.setattr(
        sm70_ops,
        "has_nvfp4_qpn_w13_swiglu_batch_dispatch",
        lambda: True,
    )

    name = "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W13"
    monkeypatch.delenv(name, raising=False)
    assert _use_qwen38_qpn_batch_fused_w13(layer, x, topk_ids)

    monkeypatch.setenv(name, "0")
    assert not _use_qwen38_qpn_batch_fused_w13(layer, x, topk_ids)

    monkeypatch.setenv(name, "1")
    assert not _use_qwen38_qpn_batch_fused_w13(layer, x[:1], topk_ids[:1])
    layer.sm70_nvfp4_qwen38_fused_swiglu_prefill = False
    assert _use_qwen38_qpn_batch_fused_w13(layer, x, topk_ids)


@pytest.mark.parametrize("tokens", (2, 4, 8, 16))
def test_qwen38_qpn_batch_fused_w2_defaults_on_and_is_shape_gated(monkeypatch, tokens):
    layer = SimpleNamespace(
        moe_config=_qwen4_moe_contract(),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
    )
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    topk_ids = torch.empty(tokens, 10, dtype=torch.int32)
    monkeypatch.setattr(
        sm70_ops,
        "has_nvfp4_qpn_w2_reduce_dispatch",
        lambda: True,
    )

    name = "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W2"
    monkeypatch.delenv(name, raising=False)
    assert _use_qwen38_qpn_batch_fused_w2(layer, x, topk_ids)

    monkeypatch.setenv(name, "0")
    assert not _use_qwen38_qpn_batch_fused_w2(layer, x, topk_ids)

    monkeypatch.setenv(name, "1")
    assert not _use_qwen38_qpn_batch_fused_w2(layer, x[:1], topk_ids[:1])
    monkeypatch.setattr(
        sm70_ops,
        "has_nvfp4_qpn_w2_reduce_dispatch",
        lambda: False,
    )
    assert not _use_qwen38_qpn_batch_fused_w2(layer, x, topk_ids)


def test_qwen38_qpn_mtp5_decode_is_opt_in_and_exact_shape_only(monkeypatch):
    layer = SimpleNamespace(
        moe_config=_qwen4_moe_contract(),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
    )
    x = torch.empty(5, 2560, dtype=torch.float16)
    topk_ids = torch.empty(5, 10, dtype=torch.int32)

    monkeypatch.delenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE", raising=False)
    assert not _use_qwen38_qpn_mtp5_decode(layer, x, topk_ids)

    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE", "1")
    assert _use_qwen38_qpn_mtp5_decode(layer, x, topk_ids)
    assert not _use_qwen38_qpn_mtp5_decode(layer, x[:4], topk_ids[:4])
    assert not _use_qwen38_qpn_mtp5_decode(layer, x, topk_ids.long())

    layer.moe_config.tp_size = 2
    assert not _use_qwen38_qpn_mtp5_decode(layer, x, topk_ids)


def test_nvfp4_moe_contract_rejects_shape_consistent_unvalidated_tp8():
    with pytest.raises(NotImplementedError, match="tensor parallel"):
        validate_nvfp4_sm70_moe_contract(
            _moe_contract(tp_size=8, intermediate_size_per_partition=64)
        )


def _meta_layer() -> SimpleNamespace:
    experts, hidden, intermediate = 256, 2048, 128
    return SimpleNamespace(
        local_num_experts=experts,
        moe_config=_moe_contract(),
        w13_weight=torch.empty(
            experts, 2 * intermediate, hidden // 2, dtype=torch.uint8, device="meta"
        ),
        w13_weight_scale=torch.empty(
            experts,
            2 * intermediate,
            hidden // 16,
            dtype=torch.float8_e4m3fn,
            device="meta",
        ),
        w13_weight_scale_2=torch.empty(experts, 2, device="meta"),
        w2_weight=torch.empty(
            experts, hidden, intermediate // 2, dtype=torch.uint8, device="meta"
        ),
        w2_weight_scale=torch.empty(
            experts,
            hidden,
            intermediate // 16,
            dtype=torch.float8_e4m3fn,
            device="meta",
        ),
        w2_weight_scale_2=torch.empty(experts, device="meta"),
    )


def test_nvfp4_moe_weight_layout_is_explicit():
    layer = _meta_layer()
    _validate_weight_layout(layer)

    layer.w2_weight = torch.empty(256, 2048, 32, dtype=torch.uint8, device="meta")
    with pytest.raises(ValueError, match="w2_weight"):
        _validate_weight_layout(layer)


def test_nvfp4_sm70_moe_owns_routing_without_generic_modular_wrapper():
    method = ModelOptNvFp4SM70MoEMethod(
        quant_config=ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
        ),
        moe_config=_moe_contract(),
    )

    assert method.maybe_make_prepare_finalize() is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an exact SM70 CUDA device",
)
@pytest.mark.parametrize("total_slots", (8, 72, 80))
def test_nvfp4_compact_groups_keep_duplicate_expert_slots_independent(total_slots):
    sorted_expert_ids = (
        torch.arange(total_slots, dtype=torch.int32, device="cuda") // 3
    ) % 256
    compact_offsets = torch.empty(total_slots + 1, dtype=torch.int32, device="cuda")
    active_expert_ids = torch.empty(total_slots, dtype=torch.int32, device="cuda")

    _prepare_compact_slot_groups(sorted_expert_ids, compact_offsets, active_expert_ids)

    assert torch.equal(
        compact_offsets.cpu(),
        torch.arange(total_slots + 1, dtype=torch.int32, device="cpu"),
    )
    assert torch.equal(active_expert_ids.cpu(), sorted_expert_ids.cpu())


@pytest.mark.parametrize("total_slots", (81, 100))
def test_nvfp4_compact_groups_reject_work_above_80_rows(total_slots):
    sorted_expert_ids = torch.empty(total_slots, dtype=torch.int32)
    compact_offsets = torch.empty(total_slots + 1, dtype=torch.int32)
    active_expert_ids = torch.empty(total_slots, dtype=torch.int32)

    with pytest.raises(ValueError, match="active-expert slots"):
        _prepare_compact_slot_groups(
            sorted_expert_ids, compact_offsets, active_expert_ids
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an exact SM70 CUDA device",
)
def test_nvfp4_compact_expert_groups_merge_duplicate_rows_and_pad_tail():
    sorted_expert_ids = torch.tensor(
        [3, 3, 7, 9, 9, 9, 15, 21], dtype=torch.int32, device="cuda"
    )
    compact_offsets = torch.empty(9, dtype=torch.int32, device="cuda")
    active_expert_ids = torch.empty(8, dtype=torch.int32, device="cuda")

    _prepare_compact_expert_groups(
        sorted_expert_ids, compact_offsets, active_expert_ids
    )

    assert compact_offsets.cpu().tolist() == [0, 2, 3, 6, 7, 8, 8, 8, 8]
    assert active_expert_ids.cpu().tolist() == [3, 7, 9, 15, 21, 0, 0, 0]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an exact SM70 CUDA device",
)
def test_nvfp4_single_token_direct_routing_is_exact_and_graph_dynamic():
    top_k = 10
    hidden = 2560
    x = torch.randn(1, hidden, dtype=torch.float16, device="cuda")
    topk_ids = torch.tensor(
        [[401, 7, 310, 99, 211, 3, 470, 120, 55, 256]],
        dtype=torch.int64,
        device="cuda",
    )
    weights = torch.softmax(torch.randn(1, top_k, device="cuda"), dim=-1)
    expert_output = torch.randn(top_k, hidden, dtype=torch.float16, device="cuda")
    expanded = torch.empty_like(expert_output)
    active_ids = torch.empty(top_k, dtype=torch.int32, device="cuda")
    output = torch.empty(1, hidden, dtype=torch.float16, device="cuda")
    reference = torch.empty_like(output)
    identity = torch.arange(top_k, dtype=torch.int32, device="cuda").view(1, -1)

    _prepare_single_token_slots(x, topk_ids, expanded, active_ids)
    _single_token_weighted_reduce(expert_output, weights, output)
    torch.ops._moe_C.moe_unpermute(
        expert_output, weights, identity, None, top_k, reference
    )
    torch.accelerator.synchronize()
    assert torch.equal(expanded, x.expand(top_k, -1))
    assert torch.equal(active_ids, topk_ids[0].int())
    assert torch.equal(output, reference)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _prepare_single_token_slots(x, topk_ids, expanded, active_ids)
        _single_token_weighted_reduce(expert_output, weights, output)

    x.add_(0.25)
    topk_ids.copy_(topk_ids.roll(1, dims=1))
    weights.copy_(weights.roll(2, dims=1))
    expert_output.mul_(0.5)
    graph.replay()
    torch.ops._moe_C.moe_unpermute(
        expert_output, weights, identity, None, top_k, reference
    )
    torch.accelerator.synchronize()
    assert torch.equal(expanded, x.expand(top_k, -1))
    assert torch.equal(active_ids, topk_ids[0].int())
    assert torch.equal(output, reference)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an exact SM70 CUDA device",
)
def test_nvfp4_mtp5_direct_weighted_reduce_is_exact_and_graph_dynamic():
    tokens = 5
    top_k = 10
    hidden = 2560
    weights = torch.softmax(
        torch.randn(tokens, top_k, device="cuda", dtype=torch.float32), dim=-1
    )
    expert_output = torch.randn(
        tokens * top_k, hidden, dtype=torch.float16, device="cuda"
    )
    output = torch.empty(tokens, hidden, dtype=torch.float16, device="cuda")
    reference = torch.empty_like(output)
    identity = torch.arange(tokens * top_k, dtype=torch.int32, device="cuda").view(
        tokens, top_k
    )

    _mtp_weighted_reduce(expert_output, weights, output)
    torch.ops._moe_C.moe_unpermute(
        expert_output, weights, identity, None, top_k, reference
    )
    torch.accelerator.synchronize()
    assert torch.equal(output, reference)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _mtp_weighted_reduce(expert_output, weights, output)

    weights.copy_(weights.roll(2, dims=1))
    expert_output.mul_(0.5)
    graph.replay()
    torch.ops._moe_C.moe_unpermute(
        expert_output, weights, identity, None, top_k, reference
    )
    torch.accelerator.synchronize()
    assert torch.equal(output, reference)


def test_mixed_w4a16_moe_requires_turbomind_on_sm70():
    config = _mixed_config()

    class FakeRoutedExperts:
        moe_config = _moe_contract()

    with (
        patch.object(modelopt, "RoutedExperts", FakeRoutedExperts),
        patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=True),
        patch.object(sm70_tm, "should_use_nvfp4_moe_turbomind", return_value=False),
        pytest.raises(NotImplementedError, match="TurboMind"),
    ):
        config.get_quant_method(FakeRoutedExperts(), "model.layers.0.mlp.experts")


def test_pure_nvfp4_qwen4_moe_uses_turbomind_w4a16_on_sm70():
    config = ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
    )

    class FakeRoutedExperts:
        moe_config = _qwen4_moe_contract()

    with (
        patch.object(modelopt, "RoutedExperts", FakeRoutedExperts),
        patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=True),
        patch.object(sm70_tm, "should_use_nvfp4_moe_turbomind", return_value=True),
    ):
        method = config.get_quant_method(
            FakeRoutedExperts(), "model.layers.0.mlp.experts"
        )

    assert isinstance(method, ModelOptNvFp4SM70MoEMethod)
    assert method.use_a16
