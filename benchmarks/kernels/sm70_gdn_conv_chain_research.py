# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Leased whole-layer research route for convolution, recurrence and norm."""

from pathlib import Path
from types import MethodType

import torch
from torch.utils.cpp_extension import load

from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

_extension = None


def build():
    global _extension
    if _extension is None:
        _extension = load(
            "sm70_gdn_conv_tile_chain_screen",
            [
                str(
                    Path(__file__).parents[1]
                    / "csrc/sm70_gdn_conv_tile_chain_screen.cu"
                )
            ],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
            verbose=True,
        )
    return _extension


def _core(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv: torch.Tensor,
    state: torch.Tensor,
    core: torch.Tensor,
    norms: torch.Tensor,
    flags: torch.Tensor,
    epochs: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    meta = get_forward_context().attn_metadata[layer.prefix]
    m = qkv.shape[0]
    assert m in (1, 5) and meta.num_actual_tokens == m
    if m == 5:
        assert meta.num_spec_decodes == 1 and meta.num_spec_decode_tokens == 5
        ids = meta.spec_state_indices_tensor[0]
        accepted = (
            meta.spec_state_slot_selectors
            if meta.spec_state_slot_selectors is not None
            else meta.num_accepted_tokens
        )
    else:
        assert meta.num_decodes == 1 and meta.num_prefills == 0
        ids = meta.non_spec_state_indices_tensor[:1]
        accepted = layer._gdn_conv_chain_accepted
    output = torch.empty_like(z)
    build().run(
        qkv,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        z,
        layer.norm.weight,
        conv,
        layer.conv1d.weight.view(2560, 4),
        state,
        ids,
        accepted,
        core,
        norms,
        flags,
        epochs,
        output,
        layer.norm.eps,
    )
    return output


def _fake(qkv, z, b, a, conv, state, core, norms, flags, epochs, layer_name):
    return torch.empty_like(z)


direct_register_custom_op(
    op_name="sm70_gdn_conv_chain_research",
    op_func=_core,
    fake_impl=_fake,
    mutates_args=["conv", "state", "core", "norms", "flags", "epochs"],
)


def attach(layer):
    from vllm.model_executor.layers.mamba.mamba_utils import (
        is_conv_state_dim_first,
    )

    build()
    assert layer.tp_size == 4 and not layer.gqa_interleaved_layout
    assert not layer.disable_tp_for_ba_proj and layer.conv1d.bias is None
    assert layer.conv1d.weight.shape == (2560, 1, 4)
    assert layer.norm.norm_before_gate and layer.norm.group_size is None
    assert layer.norm.activation in ("silu", "swish")
    assert layer.sm70_qwen38_fp16_fused_input
    layer._gdn_conv_chain_original = layer.forward
    layer._gdn_conv_chain_enabled = False
    device = layer.A_log.device
    for name, tensor in (
        ("core", torch.empty(5, 12, 128, dtype=torch.float16, device=device)),
        ("norms", torch.empty(5, 192, dtype=torch.float32, device=device)),
        ("flags", torch.zeros(5, 192, dtype=torch.int32, device=device)),
        ("epochs", torch.zeros(204, dtype=torch.int32, device=device)),
        ("accepted", torch.ones(1, dtype=torch.int32, device=device)),
    ):
        layer.register_buffer(f"_gdn_conv_chain_{name}", tensor, persistent=False)

    def forward(self, hidden_states, output):
        if not self._gdn_conv_chain_enabled:
            return self._gdn_conv_chain_original(hidden_states, output)
        qkv, z, b, a = torch.ops.vllm.qwen38_sm70_fp16_gdn_input(
            hidden_states,
            self.in_proj_qkvz.weight,
            self.in_proj_ba.weight,
            getattr(self.in_proj_qkvz, "_sm70_qwen38_gdn_packed", None),
            getattr(self.in_proj_ba, "_sm70_qwen38_gdn_packed", None),
        )
        conv = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
        normalized = torch.ops.vllm.sm70_gdn_conv_chain_research(
            qkv,
            z.reshape(-1, 12, 128),
            b,
            a,
            conv,
            self.kv_cache[1],
            self._gdn_conv_chain_core,
            self._gdn_conv_chain_norms,
            self._gdn_conv_chain_flags,
            self._gdn_conv_chain_epochs,
            _encode_layer_name(self.prefix),
        )
        projected, _ = self.out_proj(normalized.flatten(1))
        if output is not None:
            output.copy_(projected)
        return projected

    layer.forward = MethodType(forward, layer)
