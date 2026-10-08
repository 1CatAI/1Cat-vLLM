# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared storage accounting for the existing dense K/V codecs.

Attention schedules do not belong here. Packed MLA and TurboQuant layouts
continue to own their accounting until their separate codec migration.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch

from vllm.utils.torch_utils import get_dtype_size, nvfp4_kv_cache_full_dim


class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    NONE = 0
    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)
    INT8_PER_TOKEN_HEAD = 2  # per-token-head dynamic scales for int8
    FP8_PER_TOKEN_HEAD = 3  # per-token-head dynamic scales for fp8
    NVFP4 = 4  # packed fp4 data + fp8 block scales

    @property
    def is_per_token_head(self) -> bool:
        """True for any per-token-head quantization mode."""
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
        )

    @property
    def is_nvfp4(self) -> bool:
        """True for NVFP4 packed quantization mode."""
        return self == KVQuantMode.NVFP4


def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


def kv_cache_uses_per_token_head_scales(kv_cache_dtype: str) -> bool:
    """Return True if *kv_cache_dtype* needs per-token-head scales."""
    return get_kv_quant_mode(kv_cache_dtype).is_per_token_head


@dataclass(frozen=True)
class KVCacheCodec:
    """Physical storage contract of an existing dense K/V format.

    ``dtype`` is the allocated payload dtype, not the semantic FP8 format.
    Per-tensor scales live on the attention layer, outside the token pool;
    they are not charged once per token. Token/head FP32 scales are inline
    after each head's payload, even though kernels receive separate views.
    """

    dtype: torch.dtype
    quant_mode: KVQuantMode = KVQuantMode.NONE

    def payload_bytes_per_token(
        self, num_kv_heads: int, head_size: int, head_size_v: int | None = None
    ) -> int:
        if head_size_v is None:
            head_size_v = head_size
        if self.quant_mode.is_nvfp4:
            head_size = nvfp4_kv_cache_full_dim(head_size)
            head_size_v = nvfp4_kv_cache_full_dim(head_size_v)
        return num_kv_heads * (head_size + head_size_v) * get_dtype_size(self.dtype)

    @property
    def scale_bytes_per_head(self) -> int:
        return get_dtype_size(torch.float32) if self.quant_mode.is_per_token_head else 0

    def scale_bytes_per_token(self, num_kv_heads: int) -> int:
        return 2 * num_kv_heads * self.scale_bytes_per_head

    def bytes_per_token(
        self, num_kv_heads: int, head_size: int, head_size_v: int | None = None
    ) -> int:
        return self.payload_bytes_per_token(
            num_kv_heads, head_size, head_size_v
        ) + self.scale_bytes_per_token(num_kv_heads)

    @property
    def scale_padding_elements(self) -> int:
        return self.scale_bytes_per_head // get_dtype_size(self.dtype)

    def per_token_head_scale_layout(
        self, block_size: int, num_kv_heads: int, head_size: int
    ) -> "KVScaleLayout":
        """FP32 view strides/offsets for ``(block, 2, token, head, data+scale)``.

        These offsets refer to the start of the raw allocation, as in the
        existing Triton backend. No allocation or device read occurs here.
        """
        assert self.quant_mode.is_per_token_head
        dtype_bytes = get_dtype_size(self.dtype)
        scale_bytes = self.scale_bytes_per_head
        payload_bytes = head_size * dtype_bytes
        head_bytes = payload_bytes + scale_bytes
        assert payload_bytes % scale_bytes == 0
        side_bytes = block_size * num_kv_heads * head_bytes
        return KVScaleLayout(
            strides=(
                2 * side_bytes // scale_bytes,
                num_kv_heads * head_bytes // scale_bytes,
                head_bytes // scale_bytes,
            ),
            k_offset=payload_bytes // scale_bytes,
            v_offset=(side_bytes + payload_bytes) // scale_bytes,
        )


@dataclass(frozen=True)
class KVScaleLayout:
    """Strides and K/V storage offsets in FP32 elements."""

    strides: tuple[int, int, int]
    k_offset: int
    v_offset: int
