# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in integration probe for the validated FlashInfer SM70 components.

Libraries are built and loaded before graph capture. This module does not JIT
compile, quantize weights, change scheduler state ownership, or replace HC/M1.
All scratch tensors are call-local so different graph/ubatch streams cannot
overwrite a shared mutable workspace. Only the zero page and weights persist.
"""

import os

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.platforms import current_platform

logger = init_logger(__name__)
_QSA_ZERO: dict[torch.device, torch.Tensor] = {}


def copy_derived_buffer(layer, name, value):
    existing = getattr(layer, name, None)
    if existing is None:
        layer.register_buffer(name, value.clone().contiguous(), persistent=False)
    else:
        if (existing.shape, existing.dtype, existing.device) != (
            value.shape,
            value.dtype,
            value.device,
        ):
            raise RuntimeError(
                "FlashInfer derived geometry changed; rebuild CUDA graphs"
            )
        existing.copy_(value)


def uniform_decode(metadata, rows: int) -> bool:
    """Metadata, not a server max-seqs or prefill-budget setting, owns routing."""
    return bool(
        metadata is not None
        and 2 <= rows <= 64
        and metadata.num_prefills == 0
        and metadata.num_prefill_tokens == 0
        and metadata.num_spec_decodes == 0
        and metadata.num_spec_decode_tokens == 0
        and 0 < metadata.num_decodes <= rows
        and metadata.num_decode_tokens == metadata.num_decodes
        and metadata.non_spec_state_indices_tensor is not None
        and metadata.non_spec_state_indices_tensor.numel() >= rows
    )


def prepare(model: torch.nn.Module, device: torch.device) -> None:
    if (
        device.type != "cuda"
        or not current_platform.is_device_capability(70)
        or get_current_vllm_config().speculative_config is not None
    ):
        return
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    for key in ("VLLM_SM70_FLASHINFER_GDN_LIBRARY", "VLLM_SM70_FLASHINFER_QSA_LIBRARY"):
        path = os.environ.get(key)
        if path:
            torch.ops.load_library(path)
    if hasattr(torch.ops._C_flashinfer_qsa_sm70, "run"):
        concrete_device = torch.empty(0, device=device).device
        if concrete_device not in _QSA_ZERO:
            _QSA_ZERO[concrete_device] = torch.zeros(
                256, dtype=torch.float16, device=concrete_device
            )
    count = 0
    for layer in model.modules():
        if not isinstance(layer, QwenGatedDeltaNetAttention):
            continue
        qh, vh = layer.num_k_heads // layer.tp_size, layer.num_v_heads // layer.tp_size
        ba = getattr(layer.in_proj_ba, "weight", None)
        qkvz = getattr(layer.in_proj_qkvz, "weight", None)
        if (
            ba is None
            or qkvz is None
            or ba.dtype != torch.float16
            or qkvz.dtype != torch.float16
            or ba.device.type != "cuda"
            or qkvz.device != ba.device
            or layer.gqa_interleaved_layout
            or layer.disable_tp_for_ba_proj
            or layer.head_k_dim != 128
            or layer.head_v_dim != 128
            or ba.ndim != 2
            or ba.shape != (2 * vh, qkvz.shape[1])
            or qkvz.shape[0] != (2 * qh + 2 * vh) * 128
            or layer.A_log.dtype != torch.float32
            or layer.dt_bias.dtype != torch.float16
            or layer.conv1d.weight.dtype != torch.float16
            or layer.activation != "silu"
        ):
            continue
        namespace = f"_C_flashinfer_gdn_sm70_h{ba.shape[1]}_q{qh}_v{vh}"
        ops = getattr(torch.ops, namespace)
        if not hasattr(ops, "run"):
            continue
        copy_derived_buffer(layer, "_sm70_fi_ba", ba.t())
        copy_derived_buffer(
            layer,
            "_sm70_fi_bias",
            layer.conv1d.bias if layer.conv1d.bias is not None else ba.new_empty(0),
        )
        layer._sm70_fi_op = ops.run
        layer._sm70_fi_ready = True
        count += 1
    logger.info(
        "Prepared FlashInfer SM70 GDN probe for %d layers; no M1/HC change.", count
    )


def try_gdn(layer, hidden, z_out, core_out, conv_cache, state, metadata) -> bool:
    rows = hidden.shape[0]
    if not getattr(layer, "_sm70_fi_ready", False) or not uniform_decode(
        metadata, rows
    ):
        return False
    conv = conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)
    vh = layer.num_v_heads // layer.tp_size
    qh = layer.num_k_heads // layer.tp_size
    width = (2 * qh + vh) * 128
    if (
        hidden.dtype != torch.float16
        or not hidden.is_contiguous()
        or state.dtype != torch.float32
        or state.ndim != 4
        or state.shape[1:] != (vh, 128, 128)
        or state.stride()[1:] != (128 * 128, 128, 1)
        or conv.dtype != torch.float16
        or conv.ndim != 3
        or conv.shape[1:] != (width, 3)
        or conv.shape[0] != state.shape[0]
    ):
        return False
    # This boundary owns both projection and recurrence: do not also compute
    # the separate BA GEMM, which would erase the gate-fusion benefit.
    qkvz, _ = layer.in_proj_qkvz(hidden)
    qkv = qkvz[:, :width]
    z_out.copy_(qkvz[:, width:].reshape_as(z_out))
    conv_weight = layer.conv1d.weight.reshape(width, 4)
    conv_out = hidden.new_empty((rows, width))
    partial = torch.empty(
        rows * 2 * vh * 160, device=hidden.device, dtype=torch.float32
    )
    layer._sm70_fi_op(
        hidden,
        layer._sm70_fi_ba,
        qkv,
        conv_weight,
        layer._sm70_fi_bias,
        conv,
        layer.A_log,
        layer.dt_bias,
        state,
        metadata.non_spec_state_indices_tensor[:rows],
        core_out,
        conv_out,
        partial,
    )
    logger.info_once("Selected FlashInfer SM70 fused GDN batch route, rows=%d.", rows)
    return True


def try_qsa(q, k, v, indices, table, requests, out):
    zero = _QSA_ZERO.get(q.device)
    if (
        zero is None
        or not 4 <= q.shape[0] <= 16
        or q.dtype != torch.float16
        or k.dtype != torch.float16
        or v.dtype != torch.float16
        or q.shape[2] != 256
        or q.shape[1] > 32
        or q.shape[1] // k.shape[2] not in (1, 2, 4, 6, 8)
        or not out.is_contiguous()
        or k.stride() != v.stride()
        or any(
            t.data_ptr() % 16 or any(s % 8 for s in t.stride()[:-1]) for t in (q, k, v)
        )
    ):
        return None
    rows, heads, dim = q.shape
    splits = 16 if rows >= 16 else 32
    width = ((indices.shape[1] + splits - 1) // splits) * splits
    offsets = torch.empty((rows, width), device=q.device, dtype=torch.int64)
    metadata = torch.empty(
        rows + 2 + 2 * rows * splits, device=q.device, dtype=torch.int32
    )
    partial = torch.empty(
        (rows, splits, heads, dim), device=q.device, dtype=torch.float32
    )
    lse = torch.empty((rows, splits, heads), device=q.device, dtype=torch.float32)
    torch.ops._C_flashinfer_qsa_sm70.run(
        q,
        k,
        v,
        indices,
        table,
        requests,
        offsets,
        metadata,
        zero,
        partial,
        lse,
        out,
        splits,
    )
    logger.info_once("Selected FlashInfer SM70 sparse QSA batch route, rows=%d.", rows)
    return out
