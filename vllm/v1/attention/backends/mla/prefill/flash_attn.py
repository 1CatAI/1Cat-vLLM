# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention backend for MLA prefill."""

import functools
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None  # type: ignore[assignment]


def _is_sm70_flash_v100_platform() -> bool:
    """Return whether this process is running on an exact SM70 CUDA device."""
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and tuple(capability) == (7, 0)


@functools.cache
def _flash_v100_dense_prefill_lse_usable() -> bool:
    """Return whether the SM70 dense prefill entry can emit softmax LSE."""
    if not _is_sm70_flash_v100_platform():
        return False
    try:
        from vllm.v1.attention.backends.flash_v100 import (
            flash_v100_dense_prefill_lse_available,
        )
    except ImportError:
        return False
    return flash_v100_dense_prefill_lse_available()


# Head dimensions instantiated by fused_mha_forward.cu. Keep this in sync with
# the kernel dispatch switch so unsupported models are rejected during selection.
_SM70_DENSE_PREFILL_HEAD_DIMS = frozenset({16, 32, 64, 128, 256})


class FlashAttnPrefillBackend(MLAPrefillBackend):
    """FlashAttention backend for MLA prefill."""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def is_available(cls) -> bool:
        # The generic CUDA availability predicate can be true on V100 even when
        # it resolves to an unsupported FA2 stub. Select the in-tree implementation
        # explicitly on SM70 and retain the normal FA2/FA3 behavior elsewhere.
        if _is_sm70_flash_v100_platform():
            return _flash_v100_dense_prefill_lse_usable()
        return is_flash_attn_varlen_func_available()

    @classmethod
    def validate_configuration(cls, device_capability, selector_config) -> list[str]:
        invalid_reasons = super().validate_configuration(
            device_capability, selector_config
        )
        if _is_sm70_flash_v100_platform():
            if selector_config.dtype != torch.float16:
                invalid_reasons.append(
                    "SM70 Flash-V100 MLA prefill route requires float16 "
                    f"(got {selector_config.dtype})"
                )
            if selector_config.qk_head_dim != selector_config.v_head_dim:
                invalid_reasons.append(
                    "SM70 Flash-V100 MLA prefill route requires uniform qk/v head "
                    f"dims (got qk={selector_config.qk_head_dim}, "
                    f"v={selector_config.v_head_dim})"
                )
            elif selector_config.qk_head_dim not in _SM70_DENSE_PREFILL_HEAD_DIMS:
                invalid_reasons.append(
                    "SM70 Flash-V100 MLA prefill route has no compiled kernel tile "
                    f"for head dim {selector_config.qk_head_dim} "
                    f"(compiled: {sorted(_SM70_DENSE_PREFILL_HEAD_DIMS)})"
                )
        return invalid_reasons

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        if _is_sm70_flash_v100_platform():
            assert _flash_v100_dense_prefill_lse_usable(), (
                "The SM70 FlashAttention MLA prefill route requires the "
                "Flash-V100 dense LSE entry."
            )
            self.flash_attn_varlen_func = None
            self.vllm_flash_attn_version = None
            self.requires_v_padding = False
            self._is_vllm_fa = True
            return

        # Handle the differences between the flash_attn_varlen from
        # flash_attn and the one from vllm_flash_attn.
        assert flash_attn_varlen_func is not None, (
            "FlashAttnPrefillBackend requires flash_attn_varlen_func. "
            "Ensure FlashAttnPrefillBackend.is_available() is checked first."
        )

        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = get_flash_attn_version(head_size=qk_head_dim)
        if self.vllm_flash_attn_version is not None:
            self.flash_attn_varlen_func = functools.partial(
                flash_attn_varlen_func, fa_version=self.vllm_flash_attn_version
            )

        # Determine if we need to pad V
        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim for attention backends that do
        # not support different headdims.
        # FA3 on Hopper (SM90) and FA4 natively handle diff headdims.
        device_capability = current_platform.get_device_capability()
        self.requires_v_padding = self.vllm_flash_attn_version is None or not (
            (
                self.vllm_flash_attn_version == 3
                and device_capability is not None
                and device_capability[0] == 9
            )
            or self.vllm_flash_attn_version == 4
        )

        # Track whether we're using vllm's FA or upstream (for ROCm)
        self._is_vllm_fa = current_platform.is_cuda() or current_platform.is_xpu()

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if _is_sm70_flash_v100_platform():
            cu_q = kwargs.get("cu_seqlens_q")
            cu_k = kwargs.get("cu_seqlens_k")
            causal = bool(kwargs.get("causal", False))
            window_size = tuple(kwargs.get("window_size", (-1, -1)))
            if not _flash_v100_dense_prefill_lse_usable():
                raise RuntimeError("Flash-V100 dense LSE prefill is unavailable")
            if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
                raise TypeError("SM70 Flash-V100 MLA prefill requires fp16 Q/K/V")
            if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
                raise ValueError("SM70 Flash-V100 MLA prefill expects [T, H, D] Q/K/V")
            if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
                raise ValueError(
                    "SM70 Flash-V100 MLA prefill requires uniform Q/K/V head dims"
                )
            if k.shape[1] != v.shape[1] or q.shape[1] % k.shape[1] != 0:
                raise ValueError(
                    "SM70 Flash-V100 MLA prefill has invalid Q/K/V head mapping"
                )
            if cu_q is None or cu_k is None or cu_q.numel() != cu_k.numel():
                raise ValueError(
                    "SM70 Flash-V100 MLA prefill requires matching Q/K metadata"
                )
            if float(kwargs.get("dropout_p", 0.0)) != 0.0:
                raise ValueError("SM70 Flash-V100 MLA prefill does not support dropout")
            if kwargs.get("alibi_slopes") is not None:
                raise ValueError("SM70 Flash-V100 MLA prefill does not support ALiBi")
            if float(kwargs.get("softcap", 0.0)) != 0.0:
                raise ValueError("SM70 Flash-V100 MLA prefill does not support softcap")

            from vllm.v1.attention.backends.flash_v100 import (
                flash_v100_dense_prefill_lse,
            )

            q_c = q.contiguous()
            out = torch.empty_like(q_c)
            lse = torch.empty(
                (q_c.shape[1], q_c.shape[0]),
                dtype=torch.float32,
                device=q_c.device,
            )
            flash_v100_dense_prefill_lse(
                q_c,
                k.contiguous(),
                v.contiguous(),
                out,
                lse,
                cu_q,
                cu_k,
                q_c.shape[0],
                float(softmax_scale)
                if softmax_scale is not None
                else q.shape[-1] ** -0.5,
                causal=causal,
                window_size=window_size,
            )
            if return_softmax_lse:
                return out, lse
            return out

        maybe_padded_v = v
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        if self._is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
        else:
            # ROCm leverages the upstream flash_attn, which takes a parameter
            # called "return_attn_probs" instead of return_softmax_lse
            kwargs["return_attn_probs"] = return_softmax_lse
        if envs.VLLM_BATCH_INVARIANT:
            kwargs["num_splits"] = 1

        flash_attn_varlen_func = self.flash_attn_varlen_func
        assert flash_attn_varlen_func is not None
        attn_out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there are multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Unpad output back to v_head_dim if we padded V
        if self.requires_v_padding:
            attn_out = attn_out[..., : v.shape[-1]]

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._prefill_metadata.chunked_context is not None
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.chunked_context.max_seq_lens[chunk_idx],
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
        )
