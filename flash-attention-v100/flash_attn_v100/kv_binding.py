# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind a codec to an installed algorithm without duplicating execution flow."""

from dataclasses import dataclass
from typing import Any

import torch

_LEGACY_CODECS = frozenset(("auto", "float16", "fp8", "fp8_e4m3", "fp8_e5m2"))
_ROW_SCALE_CODECS = frozenset(("int8", "int8_per_token_head"))
_FP32_PARTIAL_CODECS = _ROW_SCALE_CODECS | {"fp8", "fp8_e4m3"}


def codec_names(native: Any, algorithm: str) -> frozenset[str]:
    if int(getattr(native, f"{algorithm}_kv_abi", 0)) == 1 and callable(
        getattr(native, f"{algorithm}_kv_fwd", None)
    ):
        return frozenset(getattr(native, f"{algorithm}_kv_codecs", ()))
    return (
        _LEGACY_CODECS
        if callable(getattr(native, f"{algorithm}_fwd", None))
        else frozenset()
    )


@dataclass(frozen=True)
class KVBinding:
    operation: Any
    scale_args: (
        tuple[torch.Tensor | None, torch.Tensor | None]
        | tuple[torch.Tensor | None, torch.Tensor | None, int, int]
        | tuple[()]
    )
    partial_dtype: torch.dtype


def bind_kv(
    native: Any,
    algorithm: str,
    codec: str,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    *,
    partial_dtype: torch.dtype | None = None,
    page_mapping: tuple[int, int] | None = None,
) -> KVBinding:
    # Check the binary before allocating workspaces or entering graph capture.
    if codec not in codec_names(native, algorithm):
        raise RuntimeError(
            f"Rebuild Flash-V100: {algorithm} has no explicit KV binding for {codec}"
        )
    row_scaled = codec in _ROW_SCALE_CODECS
    if row_scaled:
        if key_scale is None or value_scale is None:
            raise ValueError("Token/head KV decoding requires both scale planes")
    else:
        if key_scale is not None or value_scale is not None:
            raise ValueError("This KV codec uses tensor scales, not scale planes")
    explicit = int(getattr(native, f"{algorithm}_kv_abi", 0)) == 1 and callable(
        getattr(native, f"{algorithm}_kv_fwd", None)
    )
    view_binding = getattr(native, f"{algorithm}_kv_view_abi", 0) >= 1
    scale_args: (
        tuple[torch.Tensor | None, torch.Tensor | None]
        | tuple[torch.Tensor | None, torch.Tensor | None, int, int]
        | tuple[()]
    )
    if view_binding:
        if not explicit:
            raise RuntimeError("Native cache-view capability requires a KV binding")
        mapping = page_mapping if page_mapping is not None else (1, 0)
        scale_args = (key_scale, value_scale, mapping[0], mapping[1])
    elif page_mapping is not None:
        raise RuntimeError("Native codec binding does not support mapped KV pages")
    else:
        scale_args = (key_scale, value_scale) if explicit else ()
    return KVBinding(
        getattr(native, f"{algorithm}_{'kv_' if explicit else ''}fwd"),
        scale_args,
        partial_dtype
        or (torch.float32 if codec in _FP32_PARTIAL_CODECS else torch.float16),
    )
