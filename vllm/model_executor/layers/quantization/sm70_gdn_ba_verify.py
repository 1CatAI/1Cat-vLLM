# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse the SM70 DFlash2 verifier's channel FP8 and FP16 projections."""

import torch

from vllm import _sm70_ops as sm70_ops
from vllm._sm70.policy import NativeBindings, call_native, register_policy_op
from vllm.config.sm70_native import capture_linear_native_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.platforms import current_platform


def bind_gdn_ba_verify_policy(vllm_config) -> NativeBindings | None:
    """Capture the engine's FP8 linear policy for the fused verify projection.

    The QKVZ projection it replaces runs under the same "fp8" family policy.
    Called while the layer is built, so token execution reads no configuration.
    LoRA changes the projections the fusion replaces, so it stays off then.
    """
    if vllm_config.lora_config is not None or not (
        current_platform.is_device_capability(70)
        and hasattr(torch.ops._C, "fp8_qpn8_dispatch_ba_split_sm70_out")
    ):
        return None
    return NativeBindings(capture_linear_native_config("fp8").values)


def _dispatch_gdn_ba_verify(
    qkv_out: torch.Tensor,
    z_out: torch.Tensor,
    b_out: torch.Tensor,
    a_out: torch.Tensor,
    qkvz_staging: torch.Tensor,
    ba_staging: torch.Tensor,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    ba_weight: torch.Tensor,
    native_policy: list[str] | None = None,
) -> None:
    operation = (
        sm70_ops.fp8_qpn8_dispatch_ba_masked_sm70_out
        if x.shape[0] < 8
        else sm70_ops.fp8_qpn8_dispatch_ba_split_sm70_out
    )
    # Verify rows (2..32 at K = 5120) never take the dense large-M route, so
    # no dense workspace address is needed.
    call_native(
        operation,
        native_policy,
        qkv_out,
        z_out,
        b_out,
        a_out,
        qkvz_staging,
        ba_staging,
        0,
        x,
        codes,
        scales,
        ba_weight,
    )


def _dispatch_gdn_ba_verify_fake(
    qkv_out: torch.Tensor,
    z_out: torch.Tensor,
    b_out: torch.Tensor,
    a_out: torch.Tensor,
    qkvz_staging: torch.Tensor,
    ba_staging: torch.Tensor,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    ba_weight: torch.Tensor,
    native_policy: list[str] | None = None,
) -> None:
    return None


register_policy_op(
    "sm70_gdn_ba_verify_dispatch",
    "(Tensor(a!) qkv_out, Tensor(b!) z_out, Tensor(c!) b_out, Tensor(d!) a_out, "
    "Tensor(e!) qkvz_staging, Tensor(f!) ba_staging, Tensor x, Tensor codes, "
    "Tensor scales, Tensor ba_weight, str[]? native_policy=None) -> ()",
    _dispatch_gdn_ba_verify,
    _dispatch_gdn_ba_verify_fake,
)


def _gdn_ba_shard_supported(qkvz_width: int, heads: int) -> bool:
    if heads <= 0 or qkvz_width != heads * 1024 // 3 or heads % 3:
        return False
    if qkvz_width == 4096:
        return True  # Original binary contract.
    revision = getattr(torch.ops._C, "fp8_qpn8_ba_split_geometry_revision", None)
    return revision is not None and revision() >= 1


def apply_gdn_ba_verify(layer, x, outputs=None):
    qkvz = layer.in_proj_qkvz
    ba = layer.in_proj_ba
    if (
        not getattr(layer, "enable_sm70_dflash2_fused_gdn_verify", False)
        or getattr(layer, "sm70_gdn_ba_verify_native", None) is None
        or not isinstance(qkvz, LinearBase)
        or not isinstance(ba, LinearBase)
        or not current_platform.is_device_capability(70)
    ):
        return None
    heads = layer.num_v_heads // layer.tp_size
    z_width = heads * layer.head_v_dim
    qkv_width = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
    width = qkv_width + z_width
    if (
        not getattr(qkvz, "sm70_fp8_qpn8", False)
        or x.ndim != 2
        or x.shape[1] != 5120
        or not 2 <= x.shape[0] <= 32
        or x.dtype != torch.float16
        or not x.is_contiguous()
        or layer.head_k_dim != 128
        or layer.head_v_dim != 128
        or not _gdn_ba_shard_supported(width, heads)
        or qkvz.weight.shape != (5120, width)
        or qkvz.weight_scale_inv.shape != (1, width)
        or ba.weight.shape != (2 * heads, 5120)
        or ba.weight.dtype != torch.float16
        or not ba.weight.is_contiguous()
        or layer.gqa_interleaved_layout
        or layer.disable_tp_for_ba_proj
        or not hasattr(torch.ops._C, "fp8_qpn8_dispatch_ba_split_sm70_out")
    ):
        return None
    rows = x.shape[0]
    masked_rows = hasattr(torch.ops._C, "fp8_qpn8_dispatch_ba_masked_sm70_out")
    if rows < 8 and not masked_rows:
        # MTP verifies 1 + drafts < 8 rows. Below eight the native route
        # splits into QPN8 plus a cuBLAS b/a GEMM and a copy; zero rows pad
        # the block to the fused M8 launch, which reads the weights once.
        padded = x.new_zeros((8, x.shape[1]))
        padded[:rows] = x
        fused = apply_gdn_ba_verify(layer, padded)
        if fused is None:
            return None
        qkv, z, b, a = (part[:rows] for part in fused)
        if outputs is None:
            return qkv, z, b, a
        for out, part in zip(outputs, (qkv, z, b, a)):
            out.copy_(part)
        return outputs
    if outputs is None:
        qkv = x.new_empty((rows, qkv_width))
        z = x.new_empty((rows, z_width))
        b = x.new_empty((rows, heads))
        a = torch.empty_like(b)
    else:
        qkv, z, b, a = outputs
        if any(
            tensor.shape != shape
            or tensor.dtype != x.dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
            for tensor, shape in zip(
                outputs,
                ((rows, qkv_width), (rows, z_width), (rows, heads), (rows, heads)),
            )
        ):
            return None
    torch.ops.vllm.sm70_gdn_ba_verify_dispatch(
        qkv,
        z,
        b,
        a,
        x.new_empty((rows, width)),
        x.new_empty((rows, 2 * heads)),
        x,
        qkvz.weight,
        qkvz.weight_scale_inv,
        ba.weight,
        layer.sm70_gdn_ba_verify_native.arguments,
    )
    return qkv, z, b, a
