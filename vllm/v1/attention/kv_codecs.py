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
from typing import Literal

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
    # Whether reading the cache needs scales.
    quantized: bool = False
    native_id: int | None = None
    encoding_version: int = 1
    scale_granularity: Literal["none", "tensor", "token_head"] = "none"
    scale_dtype: torch.dtype | None = None
    rounding: Literal["none", "nearest_even", "truncate"] = "none"
    layout: Literal["dense", "inline_scale", "planar_scales"] = "dense"

    @property
    def cache_identity(self) -> tuple[str, int, str]:
        """Versioned storage identity; equal byte widths are not compatible."""
        return (self.name, self.encoding_version, self.layout)

    @property
    def per_token_head(self) -> bool:
        return self.scale_granularity == "token_head"

    def stores(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> bool:
        """Whether both cache tensors use this codec's storage dtype."""
        return (
            key_cache.dtype == self.storage_dtype
            and value_cache.dtype == self.storage_dtype
        )

    def supported_by(self, operation) -> bool:
        """Query installed bindings; legacy callables support tensor-scale KV only."""
        if operation is None:
            return False
        formats = getattr(operation, "kv_cache_codecs", None)
        if formats is None:
            return self in (FP16, FP8_E4M3, FP8_E5M2)
        return self.name in formats

    def dequantize(
        self,
        cache: torch.Tensor,
        scale: float | torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reference dequantization of contiguous cache bytes."""
        if not self.quantized:
            return cache
        assert self.element_dtype is not None
        if self.per_token_head:
            if not isinstance(scale, torch.Tensor) or scale.shape != cache.shape[:-1]:
                raise ValueError("Per-token-head decoding needs one scale per row")
            # Keep the scale multiplication in FP32. Casting a scale or an
            # unscaled INT8 product to FP16 can overflow for finite inputs.
            decoded = cache.view(self.element_dtype).float() * scale[..., None]
            if self.layout == "planar_scales":
                # Even an FP32 scale rounded from FLT_MAX/127 can reconstruct
                # just above FLT_MAX. Saturate finite rows at the destination
                # boundary; preserve the NaN scale used for invalid rows.
                limit = torch.finfo(out_dtype).max
                decoded = torch.where(
                    torch.isfinite(scale[..., None]),
                    decoded.clamp(-limit, limit),
                    decoded,
                )
            return decoded.to(out_dtype)
        return cache.view(self.element_dtype).to(out_dtype) * scale

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference dynamic INT8 encoder; not used by the execution hot path.

        A non-finite row receives a NaN scale, propagating invalid inputs to
        its readers instead of silently converting them into finite values.
        Legacy truncation retains its original multiplication order.
        """
        if self.element_dtype != torch.int8 or not self.per_token_head:
            raise ValueError(f"{self.name} has no dynamic INT8 reference encoder")
        x = values.float()
        scale = (x.abs().amax(dim=-1) / 127.0).clamp_min(1e-6)
        if self.rounding == "nearest_even":
            finite = torch.isfinite(x).all(dim=-1)
            scale = torch.where(finite, scale, float("nan"))
            q = (x / scale[..., None]).round()
        else:
            q = (x * scale.reciprocal()[..., None]).trunc()
        return q.clamp(-128, 127).to(torch.int8), scale

    def __repr__(self) -> str:
        return f"KVCodec({self.name})"


# The unquantized 16-bit cache. Native operators name it "auto".
FP16 = KVCodec("auto", ("auto", "float16"), torch.float16, native_id=0)
BF16 = KVCodec("bfloat16", ("bfloat16",), torch.bfloat16)
FP8_E4M3 = KVCodec(
    "fp8_e4m3",
    ("fp8", "fp8_e4m3"),
    torch.uint8,
    torch.float8_e4m3fn,
    quantized=True,
    native_id=1,
    scale_granularity="tensor",
    scale_dtype=torch.float32,
    rounding="nearest_even",
)
FP8_E5M2 = KVCodec(
    "fp8_e5m2",
    ("fp8_e5m2",),
    torch.uint8,
    torch.float8_e5m2,
    quantized=True,
    native_id=2,
    scale_granularity="tensor",
    scale_dtype=torch.float32,
    rounding="nearest_even",
)
INT8 = KVCodec(
    "int8",
    ("int8",),
    torch.int8,
    torch.int8,
    quantized=True,
    native_id=3,
    scale_granularity="token_head",
    scale_dtype=torch.float32,
    rounding="nearest_even",
    layout="planar_scales",
)
INT8_PER_TOKEN_HEAD = KVCodec(
    "int8_per_token_head",
    ("int8_per_token_head",),
    torch.int8,
    torch.int8,
    quantized=True,
    native_id=3,
    scale_granularity="token_head",
    scale_dtype=torch.float32,
    rounding="truncate",
    layout="inline_scale",
)
FP8_PER_TOKEN_HEAD = KVCodec(
    "fp8_per_token_head",
    ("fp8_per_token_head",),
    torch.uint8,
    torch.float8_e4m3fn,
    quantized=True,
    scale_granularity="token_head",
    scale_dtype=torch.float32,
    rounding="nearest_even",
    layout="inline_scale",
)

KV_CODECS: tuple[KVCodec, ...] = (
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    INT8,
    INT8_PER_TOKEN_HEAD,
    FP8_PER_TOKEN_HEAD,
)
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
