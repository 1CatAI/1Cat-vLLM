# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache storage formats as codecs.

An attention operator is written once against a codec; a route admits the
codecs it implements instead of comparing ``kv_cache_dtype`` strings. Adding a
KV format (for example grouped INT8) means adding a codec here and teaching the
operators that should support it, not editing every route predicate.

The codec describes storage only. It is independent of weight quantization and
of the acceleration path that reads it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class KVCodec:
    """One KV-cache storage format."""

    # Canonical name, also the ``kv_cache_dtype`` string native operators take.
    name: str
    # ``--kv-cache-dtype`` spellings that resolve to this codec.
    aliases: tuple[str, ...]
    # Tensor dtype of the paged K/V cache.
    storage_dtype: torch.dtype
    # Element dtype the stored bytes are reinterpreted as, if they are packed.
    element_dtype: torch.dtype | None = None
    # Whether reading the cache needs per-tensor scales (k_scale/v_scale).
    quantized: bool = False

    def stores(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> bool:
        """Whether both cache tensors use this codec's storage dtype."""
        return (
            key_cache.dtype == self.storage_dtype
            and value_cache.dtype == self.storage_dtype
        )

    def dequantize(
        self, cache: torch.Tensor, scale: float, out_dtype: torch.dtype
    ) -> torch.Tensor:
        """Reference dequantization of contiguous cache bytes."""
        if not self.quantized:
            return cache
        assert self.element_dtype is not None
        return cache.view(self.element_dtype).to(out_dtype) * scale

    def __repr__(self) -> str:
        return f"KVCodec({self.name})"


# The unquantized 16-bit cache. Native operators name it "auto".
FP16 = KVCodec("auto", ("auto", "float16"), torch.float16)
BF16 = KVCodec("bfloat16", ("bfloat16",), torch.bfloat16)
FP8_E4M3 = KVCodec(
    "fp8_e4m3",
    ("fp8", "fp8_e4m3"),
    torch.uint8,
    torch.float8_e4m3fn,
    quantized=True,
)
FP8_E5M2 = KVCodec(
    "fp8_e5m2", ("fp8_e5m2",), torch.uint8, torch.float8_e5m2, quantized=True
)
INT8_PER_TOKEN_HEAD = KVCodec(
    "int8_per_token_head",
    ("int8", "int8_per_token_head"),
    torch.int8,
    torch.int8,
    quantized=True,
)

KV_CODECS: tuple[KVCodec, ...] = (FP16, BF16, FP8_E4M3, FP8_E5M2, INT8_PER_TOKEN_HEAD)
_BY_ALIAS = {alias: codec for codec in KV_CODECS for alias in codec.aliases}


def resolve_kv_codec(kv_cache_dtype: str | None) -> KVCodec | None:
    """Map a ``kv_cache_dtype`` string to its codec; ``None`` if unknown."""
    if kv_cache_dtype is None:
        return None
    return _BY_ALIAS.get(kv_cache_dtype)


def canonical_kv_cache_dtype(kv_cache_dtype: str) -> str:
    """The canonical spelling of a known format; unknown names are kept."""
    codec = resolve_kv_codec(kv_cache_dtype)
    return kv_cache_dtype if codec is None else codec.name


_PTH_BASE_OFFSETS: dict[tuple[int, int, int], int] = {}


def _record_pth_base(storage_ptr: int, block_stride: int, base_offset: int, nkv: int):
    key = (storage_ptr, block_stride, base_offset)
    if key not in _PTH_BASE_OFFSETS and len(_PTH_BASE_OFFSETS) >= 512:
        _PTH_BASE_OFFSETS.pop(next(iter(_PTH_BASE_OFFSETS)))
    _PTH_BASE_OFFSETS[key] = nkv


def split_per_token_head_kv_cache(
    kv_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return block-planar (key_cache, value_cache, k_scale_cache, v_scale_cache).

    Given a padded paged carrier ``(num_blocks, 2, block_size, nkv, head_size + pad)``
    (or ``(2, num_blocks, ...)``), places the ``head_size`` payload bytes of each
    ``(block, kv_half)`` contiguously first and the FP32 per-(token, head) scales
    immediately after inside the same ``(block, kv_half)`` storage slice.

    This preserves 16-byte alignment on every token's K/V vector while keeping
    each physical block self-contained for paged allocation, prefix caching,
    interleaved packed pages, and padded hybrid pages.
    """
    if kv_cache.ndim != 5:
        raise ValueError(
            f"Expected 5D per-token-head KV cache, got {tuple(kv_cache.shape)}"
        )
    if kv_cache.shape[1] == 2:
        num_blocks, _, block_size, num_kv_heads, padded_dim = kv_cache.shape
        block_stride_elems = kv_cache.stride(0)
        kv_half_stride_elems = kv_cache.stride(1)
    elif kv_cache.shape[0] == 2:
        _, num_blocks, block_size, num_kv_heads, padded_dim = kv_cache.shape
        kv_half_stride_elems = kv_cache.stride(0)
        block_stride_elems = kv_cache.stride(1)
    else:
        raise ValueError(
            f"Expected K/V axis of size 2 at dim 0 or 1, got {tuple(kv_cache.shape)}"
        )
    elem_bytes = kv_cache.element_size()
    scale_bytes = torch.float32.itemsize
    expected_pad = scale_bytes // elem_bytes
    if padded_dim != head_size + expected_pad:
        raise ValueError(
            f"Expected padded head dim {head_size + expected_pad}, got {padded_dim}"
        )
    base_bytes = kv_cache.storage_offset() * elem_bytes
    block_stride_bytes = block_stride_elems * elem_bytes
    kv_half_bytes = kv_half_stride_elems * elem_bytes
    payload_bytes = block_size * num_kv_heads * head_size * elem_bytes
    if (
        base_bytes % scale_bytes != 0
        or block_stride_bytes % scale_bytes != 0
        or kv_half_bytes % scale_bytes != 0
        or payload_bytes % scale_bytes != 0
    ):
        raise ValueError("Per-token-head KV cache storage is not 4-byte aligned")

    storage = kv_cache.untyped_storage()
    storage_ptr = int(storage.data_ptr())
    raw_elem = torch.empty(0, dtype=kv_cache.dtype, device=kv_cache.device).set_(
        storage
    )
    raw_f32 = torch.empty(0, dtype=torch.float32, device=kv_cache.device).set_(storage)

    k_base_elems = base_bytes // elem_bytes
    v_base_elems = (base_bytes + kv_half_bytes) // elem_bytes
    _record_pth_base(storage_ptr, block_stride_elems, k_base_elems, num_kv_heads)
    _record_pth_base(storage_ptr, block_stride_elems, v_base_elems, num_kv_heads)

    payload_shape = (num_blocks, block_size, num_kv_heads, head_size)
    payload_stride = (
        block_stride_elems,
        num_kv_heads * head_size,
        head_size,
        1,
    )
    key_cache = torch.as_strided(
        raw_elem,
        size=payload_shape,
        stride=payload_stride,
        storage_offset=k_base_elems,
    )
    value_cache = torch.as_strided(
        raw_elem,
        size=payload_shape,
        stride=payload_stride,
        storage_offset=v_base_elems,
    )

    scale_shape = (num_blocks, block_size, num_kv_heads)
    scale_stride = (
        block_stride_bytes // scale_bytes,
        num_kv_heads,
        1,
    )
    k_scale_cache = torch.as_strided(
        raw_f32,
        size=scale_shape,
        stride=scale_stride,
        storage_offset=(base_bytes + payload_bytes) // scale_bytes,
    )
    v_scale_cache = torch.as_strided(
        raw_f32,
        size=scale_shape,
        stride=scale_stride,
        storage_offset=(base_bytes + kv_half_bytes + payload_bytes) // scale_bytes,
    )
    return key_cache, value_cache, k_scale_cache, v_scale_cache


def per_token_head_scales_for_kv_views(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct FP32 scale views from 4D block-planar K/V payload views."""
    num_blocks, block_size, kv_heads_slice, head_size = key_cache.shape
    elem_bytes = key_cache.element_size()
    scale_bytes = torch.float32.itemsize
    block_stride_elems = key_cache.stride(0)
    total_kv_heads = max(kv_heads_slice, key_cache.stride(1) // head_size)
    storage = key_cache.untyped_storage()
    storage_ptr = int(storage.data_ptr())

    kv_lo = 0
    if kv_heads_slice < total_kv_heads:
        k_off = key_cache.storage_offset()
        for cand in range(total_kv_heads - kv_heads_slice + 1):
            if (storage_ptr, block_stride_elems, k_off - cand * head_size) in (
                _PTH_BASE_OFFSETS
            ):
                kv_lo = cand
                break
        else:
            kv_lo = (k_off % (total_kv_heads * head_size)) // head_size

    k_base_bytes = (key_cache.storage_offset() - kv_lo * head_size) * elem_bytes
    v_base_bytes = (value_cache.storage_offset() - kv_lo * head_size) * elem_bytes
    payload_bytes = block_size * total_kv_heads * head_size * elem_bytes
    block_stride_bytes = block_stride_elems * elem_bytes
    raw_f32 = torch.empty(0, dtype=torch.float32, device=key_cache.device).set_(storage)
    scale_shape = (num_blocks, block_size, kv_heads_slice)
    scale_stride = (block_stride_bytes // scale_bytes, total_kv_heads, 1)
    k_scales = torch.as_strided(
        raw_f32,
        size=scale_shape,
        stride=scale_stride,
        storage_offset=(k_base_bytes + payload_bytes) // scale_bytes + kv_lo,
    )
    v_scales = torch.as_strided(
        raw_f32,
        size=scale_shape,
        stride=scale_stride,
        storage_offset=(v_base_bytes + payload_bytes) // scale_bytes + kv_lo,
    )
    return k_scales, v_scales
