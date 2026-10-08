# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Legacy format conversion boundary for Flash-V100.

Keep dense fallback rounding unchanged: convert to FP16 before scaling.
Native reader/writer and allocation migration are separate acceptance steps.
"""

from __future__ import annotations

import torch


def _normalize_flash_v100_kv_cache_dtype(kv_cache_dtype: str) -> str:
    # Newer vLLM resolves an explicit FP16 cache to "float16". The vendored
    # Flash-V100 extension uses "auto" for the same unquantized FP16 layout.
    return "auto" if kv_cache_dtype == "float16" else kv_cache_dtype


def _uses_fp8_kv_cache(kv_cache_dtype: str) -> bool:
    return isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8")


def _fp8_dtype_from_cache_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FLASH_ATTN_V100 fp8 dtype: {kv_cache_dtype}")


def _dequantize_fp8_contiguous_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _uses_fp8_kv_cache(kv_cache_dtype):
        return key, value
    fp8_dtype = _fp8_dtype_from_cache_dtype(kv_cache_dtype)
    key = key.view(fp8_dtype).to(torch.float16) * k_scale
    value = value.view(fp8_dtype).to(torch.float16) * v_scale
    return key, value
