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

KV_CODECS: tuple[KVCodec, ...] = (FP16, BF16, FP8_E4M3, FP8_E5M2)
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
