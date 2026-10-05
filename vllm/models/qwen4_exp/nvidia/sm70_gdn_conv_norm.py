# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 convolution, recurrent update and sigmoid normalization transaction."""

from typing import cast

import torch

from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def _forward(
    hidden: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    context = get_forward_context()
    layer = context.no_compile_layers[_resolve_layer_name(layer_name)]
    all_metadata = context.attn_metadata
    if isinstance(all_metadata, list):
        all_metadata = all_metadata[0]
    if not isinstance(all_metadata, dict):
        output.zero_()
        return
    metadata = cast(GDNAttentionMetadata, all_metadata[layer.prefix])
    count = hidden.shape[0]
    if metadata.num_actual_tokens == 0:
        output.zero_()
        return
    ordinary = (
        count == 1
        and metadata.num_actual_tokens == 1
        and metadata.num_decodes == 1
        and metadata.num_prefills == 0
    )
    speculative = (
        count == 5
        and metadata.num_actual_tokens == 5
        and metadata.num_spec_decodes == 1
        and metadata.num_spec_decode_tokens == 5
        and metadata.num_prefills == 0
    )
    if not (ordinary or speculative):
        # A one-row prefill still uses the original complete forward method.
        layer._forward_method(hidden, output)
        return
    if speculative:
        if metadata.spec_state_indices_tensor is None:
            layer._forward_method(hidden, output)
            return
        indices = metadata.spec_state_indices_tensor[0]
        accepted = metadata.spec_state_slot_selectors
        if accepted is None:
            accepted = metadata.num_accepted_tokens
    else:
        if metadata.non_spec_state_indices_tensor is None:
            layer._forward_method(hidden, output)
            return
        indices = metadata.non_spec_state_indices_tensor[:1]
        accepted = layer._sm70_gdn_conv_norm_accepted
    if accepted is None:
        layer._forward_method(hidden, output)
        return
    # The hybrid cache exposes FP16 and FP32 views of one byte allocation.
    # Keep that owner inside the opaque transaction, as QSA does for its
    # caches. The explicit output mutation retains this transaction in AOT;
    # per-layer workspaces belong to the same opaque owner as its caches.
    conv, state = layer.kv_cache
    qkv, z, b, a = torch.ops.vllm.qwen38_sm70_fp16_gdn_input(
        hidden,
        layer.in_proj_qkvz.weight,
        layer.in_proj_ba.weight,
        getattr(layer.in_proj_qkvz, "_sm70_qwen38_gdn_packed", None),
        getattr(layer.in_proj_ba, "_sm70_qwen38_gdn_packed", None),
    )
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    if not is_conv_state_dim_first():
        conv = conv.transpose(-1, -2)
    normalized = z.new_empty((count, 12, 128))
    torch.ops._C.qwen38_gdn_conv_norm_sm70_out(
        qkv,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        z.reshape(count, 12, 128),
        layer.norm.weight,
        conv,
        layer.conv1d.weight.view(2560, 4),
        state,
        indices,
        accepted,
        layer._sm70_gdn_conv_norm_core,
        layer._sm70_gdn_conv_norm_norms,
        layer._sm70_gdn_conv_norm_flags,
        layer._sm70_gdn_conv_norm_epochs,
        normalized,
        layer.norm.eps,
        False,
    )
    result, _ = layer.out_proj(normalized.flatten(1))
    output.copy_(result)


def _fake(
    hidden: torch.Tensor, output: torch.Tensor, layer_name: LayerNameType
) -> None:
    pass


direct_register_custom_op(
    op_name="qwen38_sm70_gdn_conv_norm_forward",
    op_func=_forward,
    fake_impl=_fake,
    mutates_args=["output"],
)


def prepare_gdn_conv_norm(module: torch.nn.Module, vllm_config) -> int:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.platforms import current_platform

    from .sm70_fp16_gemv import _exact_runtime_contract

    if not _exact_runtime_contract(
        vllm_config
    ) or not current_platform.is_device_capability(70):
        return 0
    if getattr(vllm_config.parallel_config, "use_ubatching", False):
        return 0
    if vllm_config.cache_config.mamba_ssm_cache_dtype != "float32":
        return 0
    prepared = 0
    for layer in module.modules():
        if not isinstance(layer, QwenGatedDeltaNetAttention):
            continue
        if hasattr(layer, "_sm70_gdn_conv_norm"):
            continue
        if (
            layer.tp_size != 4
            or layer.gqa_interleaved_layout
            or layer.disable_tp_for_ba_proj
            or not layer.sm70_qwen38_fp16_fused_input
            or layer.conv1d.bias is not None
            or not layer.norm.norm_before_gate
            or layer.norm.group_size is not None
            or layer.norm.activation != "sigmoid"
        ):
            continue
        weights = (
            layer.in_proj_qkvz.weight,
            layer.in_proj_ba.weight,
            layer.out_proj.weight,
            layer.conv1d.weight,
            layer.norm.weight,
            layer.dt_bias,
        )
        shapes = ((4096, 2560), (24, 2560), (2560, 1536), (2560, 1, 4), (128,), (12,))
        if any(w.shape != shape for w, shape in zip(weights, shapes)):
            continue
        if not all(
            w.is_cuda and w.dtype == torch.float16 and w.is_contiguous()
            for w in weights
        ):
            continue
        if layer.A_log.dtype != torch.float32 or layer.A_log.shape != (12,):
            continue
        device = layer.A_log.device
        for name, value in (
            ("core", torch.empty(5, 12, 128, dtype=torch.float16, device=device)),
            ("norms", torch.empty(5, 192, dtype=torch.float32, device=device)),
            ("flags", torch.zeros(5, 192, dtype=torch.int32, device=device)),
            ("epochs", torch.zeros(204, dtype=torch.int32, device=device)),
            ("accepted", torch.ones(1, dtype=torch.int32, device=device)),
        ):
            layer.register_buffer(
                f"_sm70_gdn_conv_norm_{name}", value, persistent=False
            )
        layer._sm70_gdn_conv_norm = True
        prepared += 1
    return prepared


def conv_norm_forward(layer, hidden_states, output):
    layer_name = _encode_layer_name(layer.prefix)
    if output is None:
        output = torch.empty_like(hidden_states)
    torch.ops.vllm.qwen38_sm70_gdn_conv_norm_forward(
        hidden_states,
        output,
        layer_name,
    )
    return output
