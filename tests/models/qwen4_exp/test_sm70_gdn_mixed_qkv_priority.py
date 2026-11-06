# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only dispatch checks for the upstream SM70 mixed-QKV verifier."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.gdn_projection import GdnProjectionConfig
from vllm.model_executor.layers.mamba.gdn import (
    qwen_gdn_linear_attn as qwen_gdn,
)
from vllm.model_executor.layers.mamba.gdn import sm70_verify_dispatch as dispatch


def _planner_case(monkeypatch, *, tokens, state_dtype, enabled=True, contiguous=False):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(
        qwen_gdn.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )
    for name in (
        "VLLM_SM70_QWEN_GDN_CONTEXT_CORE",
        "VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS",
        "VLLM_SM70_GDN_Z_CONTIGUOUS",
        "VLLM_SM70_QWEN_GDN_INPUT_PROJECTION_OP",
        "VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP",
    ):
        monkeypatch.setattr(qwen_gdn.envs, name, False)
    monkeypatch.setattr(
        qwen_gdn, "_sm70_gdn_projection_dump_requested", lambda _: False
    )
    monkeypatch.setattr(qwen_gdn, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(qwen_gdn, "_ddtree_parent_ids_require_branch", lambda *_: False)
    monkeypatch.setattr(
        dispatch._gdn_verify_fused, "_UNITS_OVERRIDE", frozenset({"u2"})
    )

    sequences = max(1, tokens // 5)
    metadata = qwen_gdn.GDNAttentionMetadata(
        num_prefill_tokens=0,
        num_decode_tokens=0,
        num_spec_decode_tokens=tokens,
        num_actual_tokens=tokens,
        spec_sequence_masks=torch.ones(sequences, dtype=torch.bool),
        num_spec_decodes=sequences,
        num_prefills=0,
        num_decodes=0,
        ddtree_parent_ids=None,
        ddtree_num_tree_tokens_cpu=None,
    )
    state = torch.empty((1,), dtype=state_dtype)
    layer = SimpleNamespace(kv_cache=(torch.empty((1,)), state))
    context = SimpleNamespace(
        attn_metadata={"layer": metadata},
        no_compile_layers={"layer": layer},
    )
    monkeypatch.setattr(qwen_gdn, "get_forward_context", lambda: context)

    norm = SimpleNamespace(
        group_size=None,
        norm_before_gate=True,
        activation="sigmoid",
        weight=torch.empty((128,), dtype=torch.float16),
    )
    model = SimpleNamespace(
        gdn_policy=SimpleNamespace(
            projection=GdnProjectionConfig(
                context_core=False,
                mixed_qkv_contiguous=False,
                z_contiguous=False,
                input_projection=False,
                output_projection=False,
                input_core=False,
            )
        ),
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
        sm70_qwen38_fp16_fused_input=False,
        auto_sm70_qwen_gdn_full_forward=True,
        enable_sm70_dflash2_fused_gdn_verify=False,
        enable_sm70_dflash2_fused_gdn_norm=False,
        enable_sm70_dflash2_fused_gdn_split=False,
        enable_sm70_dflash2_fused_gdn_combined_split=False,
        enable_sm70_dflash2_fused_qkv_pack=False,
        enable_sm70_gdn_rmsnorm_onepass=False,
        enable_sm70_fused_sigmoid_mixed_qkv=enabled,
        key_dim=2048,
        value_dim=6144,
        num_k_heads=16,
        num_v_heads=48,
        tp_size=4,
        head_k_dim=128,
        head_v_dim=128,
        conv1d=SimpleNamespace(weight=torch.empty((1,), dtype=torch.float16)),
        activation="silu",
        norm=norm,
    )

    if contiguous:
        mixed_qkv = torch.empty((tokens, 2560), dtype=torch.float16)
    else:
        mixed_qkv = torch.empty((tokens, 4096), dtype=torch.float16)[:, :2560]
    return model, metadata, mixed_qkv


@pytest.mark.parametrize("tokens", [5, 20])
def test_upstream_mixed_qkv_eligible_verify_preempts_local_fusion(monkeypatch, tokens):
    model, _, mixed_qkv = _planner_case(
        monkeypatch, tokens=tokens, state_dtype=torch.float32
    )

    units = dispatch.plan(qwen_gdn, model, "layer", mixed_qkv)

    assert (
        dispatch.block_reason(qwen_gdn, model, "layer", mixed_qkv)
        == "upstream_mixed_qkv"
    )
    assert units == frozenset()


@pytest.mark.parametrize(
    "tokens,state_dtype,enabled,contiguous",
    [
        (4, torch.float32, True, False),
        (5, torch.float16, True, False),
        (5, torch.float32, False, False),
        (8, torch.float32, True, False),
    ],
)
def test_unqualified_mixed_qkv_keeps_local_fusion(
    monkeypatch, tokens, state_dtype, enabled, contiguous
):
    model, _, mixed_qkv = _planner_case(
        monkeypatch,
        tokens=tokens,
        state_dtype=state_dtype,
        enabled=enabled,
        contiguous=contiguous,
    )

    units = dispatch.plan(qwen_gdn, model, "layer", mixed_qkv)

    assert dispatch.block_reason(qwen_gdn, model, "layer", mixed_qkv) is None
    assert units == frozenset({"u2"})


def test_upstream_mixed_qkv_contiguous_verifier_preempts_local_fusion(monkeypatch):
    model, _, mixed_qkv = _planner_case(
        monkeypatch,
        tokens=8,
        state_dtype=torch.float32,
        contiguous=True,
    )

    units = dispatch.plan(qwen_gdn, model, "layer", mixed_qkv)

    assert (
        dispatch.block_reason(qwen_gdn, model, "layer", mixed_qkv)
        == "upstream_mixed_qkv"
    )
    assert units == frozenset()


def test_fallback_admission_uses_layer_projection_policy(monkeypatch):
    layer, _, mixed_qkv = _planner_case(
        monkeypatch, tokens=8, state_dtype=torch.float32, enabled=False
    )
    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_CONTEXT_CORE", "1")
    assert dispatch.block_reason(qwen_gdn, layer, "layer", mixed_qkv) is None
    layer.gdn_policy.projection.context_core = True
    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_CONTEXT_CORE", "0")
    assert dispatch.block_reason(qwen_gdn, layer, "layer", mixed_qkv) == "env"
