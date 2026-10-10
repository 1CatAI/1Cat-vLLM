# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 packed GDN verification beside a prefill chunk in one batch."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.config.gdn import GdnConfig, GdnProfileConfig
from vllm.model_executor.layers.fla.ops.gdn_prefill import GdnPrefill
from vllm.model_executor.layers.fla.ops.gdn_preparation import GdnPreparation
from vllm.model_executor.layers.fla.ops.gdn_profiling import GdnPrefillProfiler
from vllm.model_executor.layers.fla.ops.gdn_selector import (
    GDN_BACKEND_STAGES,
    GdnExecutionPlan,
)
from vllm.model_executor.layers.fla.ops.gdn_stages import GdnHeadContract
from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as mod
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

WIDTH = 8
CONV_WIDTH = 4
QKV_DIM = 2560
TP = 4


@pytest.mark.parametrize("prefill_len", [40, 300])
@pytest.mark.parametrize("prefill_first", [True, False])
@pytest.mark.parametrize("num_spec", [1, 3])
def test_packed_verify_beside_prefill_matches_generic_route(
    monkeypatch, prefill_len, prefill_first, num_spec
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")

    torch.manual_seed(20261003)
    device, dtype = "cuda", torch.float16
    spec_tokens = num_spec * WIDTH
    tokens = spec_tokens + prefill_len
    slots = (num_spec + 1) * 16

    # Request order: the prefill request is first or last, so the speculative
    # tokens are not the leading rows of the layer output.
    spec_start = prefill_len if prefill_first else 0
    prefill_start = 0 if prefill_first else spec_tokens
    spec_token_indx = torch.arange(
        spec_start, spec_start + spec_tokens, device=device, dtype=torch.int64
    )
    non_spec_token_indx = torch.arange(
        prefill_start, prefill_start + prefill_len, device=device, dtype=torch.int64
    )
    batch = num_spec + 1
    mask = [False] + [True] * num_spec if prefill_first else [True] * num_spec + [False]
    spec_sequence_masks = torch.tensor(mask, device=device, dtype=torch.bool)
    non_spec_cu = torch.tensor([0, prefill_len], device=device, dtype=torch.int32)
    non_spec_cu_cpu = non_spec_cu.cpu()
    spec_cu = torch.arange(num_spec + 1, device=device, dtype=torch.int32) * WIDTH
    conv_slots = WIDTH + 2
    spec_indices = (
        torch.tensor([0, 6, 2, 5, 1, 3, 4, 7], device=device, dtype=torch.int32)[None]
        + torch.arange(num_spec, device=device, dtype=torch.int32)[:, None] * 16
        + 16
    )
    accepted = torch.ones(num_spec, device=device, dtype=torch.int32)
    selectors = (torch.arange(num_spec, device=device, dtype=torch.int32) % WIDTH) + 1
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        non_spec_cu_cpu, device=torch.device(device)
    )
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=prefill_len,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_spec,
        num_spec_decode_tokens=spec_tokens,
        num_actual_tokens=tokens,
        has_initial_state=torch.zeros(1, device=device, dtype=torch.bool),
        spec_query_start_loc=spec_cu,
        non_spec_query_start_loc=non_spec_cu,
        spec_state_indices_tensor=spec_indices,
        non_spec_state_indices_tensor=torch.tensor(
            [15], device=device, dtype=torch.int32
        ),
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=accepted,
        spec_state_slot_selectors=selectors,
        chunk_indices=prepare_chunk_indices(non_spec_cu_cpu, FLA_CHUNK_SIZE).to(device),
        chunk_offsets=prepare_chunk_offsets(non_spec_cu_cpu, FLA_CHUNK_SIZE).to(device),
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    del batch
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"test": metadata}),
    )
    monkeypatch.setattr(mod, "is_conv_state_dim_first", lambda: True)

    conv = SimpleNamespace(
        weight=torch.randn(QKV_DIM, 1, CONV_WIDTH, device=device, dtype=dtype) * 0.5,
        bias=None,
    )
    policy = GdnConfig()
    policy.resolve()
    profiling = GdnProfileConfig(enabled=False)
    profiling.resolve()
    heads = GdnHeadContract(16, 48, 128, 128, 4)
    profiler = GdnPrefillProfiler(profiling)
    prefill = GdnPrefill(
        GdnExecutionPlan(
            "triton", GDN_BACKEND_STAGES["triton"], True, False, False, None
        ),
        profiler,
    )
    common = dict(
        gdn_policy=policy,
        gdn_heads=heads,
        gdn_preparation=GdnPreparation(heads),
        _gdn_profiler=profiler,
        verification_update=None,
        _can_use_sm70_gdn_preprocess=lambda *args: False,
        prefix="test",
        tp_size=TP,
        num_k_heads=16,
        num_v_heads=48,
        key_dim=2048,
        value_dim=6144,
        head_k_dim=128,
        head_v_dim=128,
        conv1d=conv,
        activation="silu",
        enable_packed_recurrent_decode=False,
        enable_sm70_fused_sigmoid_mixed_qkv=False,
        enable_sm70_legacy_prefill_prep=False,
        compare_sm70_fused_sigmoid_mixed_qkv=False,
        enable_sm70_dflash2_tp2_gdn_bv2=False,
        enable_sm70_dflash2_fused_qkv_pack=False,
        gdn_prefill_backend="triton",
        chunk_gated_delta_rule=prefill,
        A_log=torch.randn(12, device=device, dtype=torch.float32),
        dt_bias=torch.randn(12, device=device, dtype=dtype),
    )
    generic = SimpleNamespace(
        **common,
        enable_sm70_dflash2_fused_gdn_verify=False,
        _can_use_dflash2_packed_gdn_verify=lambda **kwargs: False,
    )
    packed = SimpleNamespace(**common, enable_sm70_dflash2_fused_gdn_verify=True)
    for layer in (generic, packed):
        layer.rearrange_mixed_qkv = MethodType(
            mod.QwenGatedDeltaNetAttention.rearrange_mixed_qkv, layer
        )
    packed._can_use_dflash2_packed_gdn_verify = MethodType(
        mod.QwenGatedDeltaNetAttention._can_use_dflash2_packed_gdn_verify, packed
    )
    packed._forward_dflash2_packed_gdn_verify = MethodType(
        mod.QwenGatedDeltaNetAttention._forward_dflash2_packed_gdn_verify, packed
    )

    routes: list[str] = []
    original = mod.QwenGatedDeltaNetAttention._forward_dflash2_packed_gdn_verify

    def record(self, *args, **kwargs):
        routes.append("packed")
        return original(self, *args, **kwargs)

    packed._forward_dflash2_packed_gdn_verify = MethodType(record, packed)

    mixed = torch.randn(tokens, QKV_DIM, device=device, dtype=dtype) * 0.1
    a = torch.randn(tokens, 12, device=device, dtype=dtype)
    b = torch.randn_like(a)
    conv_seed = torch.randn(slots, QKV_DIM, conv_slots, device=device, dtype=dtype)
    ssm_seed = torch.randn(slots, 12, 128, 128, device=device, dtype=torch.float32)
    ssm_seed *= 0.02
    outputs, convs, ssms = [], [], []
    for layer in (generic, packed):
        conv_state = conv_seed.clone()
        ssm_state = ssm_seed.clone()
        out = torch.full((tokens + 3, 12, 128), 42, device=device, dtype=dtype)
        mod.QwenGatedDeltaNetAttention._forward_core(
            layer, mixed.clone(), b.clone(), a.clone(), out, (conv_state, ssm_state)
        )
        torch.accelerator.synchronize()
        outputs.append(out)
        convs.append(conv_state)
        ssms.append(ssm_state)

    assert routes == ["packed"], routes
    # The tail past the live tokens is never written.
    assert torch.all(outputs[1][tokens:] == 42)
    # Prefill rows do not depend on the verification route.
    prefill_rows = non_spec_token_indx
    assert torch.equal(outputs[0][prefill_rows], outputs[1][prefill_rows])
    # Verification rows, conv state and every recurrent state slot agree.
    torch.testing.assert_close(
        outputs[1][spec_token_indx], outputs[0][spec_token_indx], atol=2e-3, rtol=2e-3
    )
    assert torch.isfinite(outputs[1][:tokens]).all()
    assert torch.equal(convs[0], convs[1])
    torch.testing.assert_close(ssms[1], ssms[0], atol=2e-4, rtol=2e-3)
