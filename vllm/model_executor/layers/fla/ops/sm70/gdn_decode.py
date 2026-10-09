# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 GDN decode provider; only tensor and head contracts cross this boundary."""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fla.ops.gdn_stages import GdnHeadContract
from vllm.model_executor.layers.fla.ops.sm70.gdn_prefill import (
    _flashqla_sm70_decode_available,
)


def mixed_qkv_decode_layout(mixed_qkv):
    if mixed_qkv.dim() != 2 or mixed_qkv.stride(1) != 1:
        return "unsupported"
    if mixed_qkv.stride(0) < mixed_qkv.shape[1]:
        return "unsupported"
    return "compact" if mixed_qkv.stride(0) == mixed_qkv.shape[1] else "row_strided"


@dataclass(frozen=True)
class FlashQlaDecodeAdmission:
    heads: GdnHeadContract
    enabled: bool
    capability: tuple[int, int] | None
    device_index: int | None = None

    @classmethod
    def bind(cls, heads, enabled):
        capability = (
            torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        )
        device = (
            torch.accelerator.current_device_index() if capability is not None else None
        )
        return cls(heads, bool(enabled), capability, device)

    def rejection(self, mixed_qkv, state_indices, tokens):
        return flashqla_decode_rejection(
            self.heads,
            self.enabled,
            mixed_qkv,
            state_indices,
            tokens,
            capability=self.capability,
            device_index=self.device_index,
        )


def flashqla_decode_rejection(
    contract, enabled, mixed_qkv, state_indices, tokens, *, capability, device_index
):
    if not enabled:
        return "env_disabled"  # retained diagnostic compatibility name
    if tokens <= 0 or state_indices is None:
        return "no_decode_tokens_or_state_indices"
    if state_indices.dtype != torch.int32:
        return "state_indices_dtype"
    if (
        not mixed_qkv.is_cuda
        or mixed_qkv.dtype != torch.float16
        or mixed_qkv_decode_layout(mixed_qkv) == "unsupported"
    ):
        return "mixed_qkv_contract"
    if contract.head_k_dim != 128 or contract.head_v_dim != 128:
        return "head_dim"
    if (
        contract.num_k_heads % contract.tp_size
        or contract.num_v_heads % contract.tp_size
    ):
        return "head_tp_divisibility"
    # Normal engine steps stay on the bound device. Preserve the old admission
    # order and tensor-device check when a standalone caller moves the layer.
    if mixed_qkv.device.index != device_index:
        capability = torch.cuda.get_device_capability(mixed_qkv.device)
    if capability not in ((7, 0), (7, 5)):
        return "device_capability"
    if not _flashqla_sm70_decode_available():
        return "flashqla_decode_import"
    return None


def flashqla_decode(
    contract: GdnHeadContract,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    num_decode_tokens: int,
    cu_seqlens: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
) -> torch.Tensor:
    del cu_seqlens
    from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import (
        gdn_decode_mixed_qkv_global_state_sm70,
    )

    state_indices = state_indices[:num_decode_tokens].contiguous()
    if core_attn_out is None:
        core_attn_out = mixed_qkv.new_empty(
            (
                num_decode_tokens,
                contract.local_v_heads,
                contract.head_v_dim,
            )
        )
    out = core_attn_out[:num_decode_tokens]
    kernel_out = (
        out
        if out.is_contiguous()
        else torch.empty(out.shape, dtype=out.dtype, device=out.device)
    )
    gdn_decode_mixed_qkv_global_state_sm70(
        mixed_qkv=mixed_qkv[:num_decode_tokens],
        a=a[:num_decode_tokens].contiguous(),
        b=b[:num_decode_tokens].contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        state=ssm_state,
        state_indices=state_indices,
        output=kernel_out,
        scale=contract.head_k_dim**-0.5,
        use_qk_l2norm_in_kernel=True,
    )
    if kernel_out is not out:
        out.copy_(kernel_out)
    return out.unsqueeze(0)
