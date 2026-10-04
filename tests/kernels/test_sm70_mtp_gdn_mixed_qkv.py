# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP uses the existing fused recurrence with a different QKV loader."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as mod
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


@pytest.mark.parametrize("batch,width", [(1, 2), (1, 5), (2, 5), (4, 4), (4, 5)])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("row_stride", [2560, 4096])
def test_standard_mtp_core_keeps_output_conv_and_ssm_bits(
    monkeypatch, batch, width, dim_first, row_stride
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")

    if row_stride != 2560 and (batch, width) not in ((1, 5), (4, 5)):
        pytest.skip("Only measured M5/M20 row-strided verifier batches")

    torch.manual_seed(20260927)
    device, dtype = "cuda", torch.float16
    tokens, slots = batch * width, batch * 8
    inputs = [
        torch.empty(tokens, row_stride, device=device, dtype=dtype)[:, :2560]
        for _ in range(2)
    ]
    outputs = [
        torch.empty(tokens + 2, 12, 128, device=device, dtype=dtype) for _ in range(2)
    ]
    a = torch.randn(tokens, 12, device=device, dtype=dtype)
    b = torch.randn_like(a)
    conv_seed = torch.randn(slots, 2560, width + 2, device=device, dtype=dtype)
    conv_states = [conv_seed.clone() for _ in range(2)]
    ssm_seed = torch.randn(slots, 12, 128, 128, device=device, dtype=torch.float32)
    ssm_states = [ssm_seed.clone() for _ in range(2)]
    # Include live slot zero, nonmonotonic slots, and untouched state canaries.
    indices = torch.tensor([0, 6, 2, 5, 1], device=device, dtype=torch.int32)[:width]
    indices = (
        indices + torch.arange(batch, device=device, dtype=torch.int32)[:, None] * 8
    )
    accepted = torch.ones(batch, device=device, dtype=torch.int32)
    selectors = torch.ones_like(accepted)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=batch,
        num_spec_decode_tokens=tokens,
        num_actual_tokens=tokens,
        spec_query_start_loc=torch.arange(batch + 1, device=device, dtype=torch.int32)
        * width,
        spec_state_indices_tensor=indices,
        spec_sequence_masks=torch.ones(batch, device=device, dtype=torch.bool),
        num_accepted_tokens=accepted,
        spec_state_slot_selectors=selectors,
    )
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"test": metadata}),
    )
    monkeypatch.setattr(mod, "is_conv_state_dim_first", lambda: dim_first)
    conv = SimpleNamespace(
        weight=torch.randn(2560, 1, 4, device=device, dtype=dtype), bias=None
    )
    common = dict(
        prefix="test",
        tp_size=4,
        num_k_heads=16,
        num_v_heads=48,
        key_dim=2048,
        value_dim=6144,
        head_k_dim=128,
        head_v_dim=128,
        conv1d=conv,
        activation="silu",
        enable_packed_recurrent_decode=False,
        enable_sm70_dflash2_fused_qkv_pack=False,
        A_log=torch.randn(12, device=device, dtype=torch.float32),
        dt_bias=torch.randn(12, device=device, dtype=dtype),
        _can_use_dflash2_packed_gdn_verify=lambda **kwargs: False,
        _can_use_sm70_gdn_preprocess=lambda *args: False,
    )
    layers = [
        SimpleNamespace(**common, enable_sm70_fused_sigmoid_mixed_qkv=enabled)
        for enabled in (False, True)
    ]
    for layer in layers:
        layer.rearrange_mixed_qkv = MethodType(
            mod.QwenGatedDeltaNetAttention.rearrange_mixed_qkv, layer
        )

    calls = [0, 0]
    for arm, name in enumerate(
        (
            "fused_sigmoid_gating_delta_rule_update",
            "fused_sigmoid_gating_delta_rule_update_mixed_qkv",
        )
    ):
        original = getattr(mod, name)

        def record(*args, _arm=arm, _original=original, **kwargs):
            calls[_arm] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(mod, name, record)

    def run(arm):
        conv_cache = (
            conv_states[arm] if dim_first else conv_states[arm].transpose(-1, -2)
        )
        mod.QwenGatedDeltaNetAttention._forward_core(
            layers[arm], inputs[arm], b, a, outputs[arm], (conv_cache, ssm_states[arm])
        )

    graphs = []
    for arm in range(2):
        inputs[arm].normal_()
        for _ in range(3):
            run(arm)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(arm)
        graphs.append(graph)
    assert calls == [4, 4]

    for replay in range(10):
        inputs[0].normal_(0, (0.001, 0.1, 1.0)[replay % 3])
        inputs[1].copy_(inputs[0])
        a.normal_()
        b.normal_()
        # Selectors can differ from accepted counts in the state-slot bridge.
        selectors.copy_((torch.arange(batch, device=device) + replay) % width + 1)
        conv_seed.normal_()
        ssm_seed.normal_(0, 0.02)
        for arm in range(2):
            conv_states[arm].copy_(conv_seed)
            ssm_states[arm].copy_(ssm_seed)
            outputs[arm].fill_(42)
            graphs[arm].replay()
        for tensors, bits in (
            (outputs, torch.int16),
            (conv_states, torch.int16),
            (ssm_states, torch.int32),
        ):
            assert torch.equal(tensors[0].view(bits), tensors[1].view(bits))
        assert torch.all(outputs[1][tokens:] == 42)
