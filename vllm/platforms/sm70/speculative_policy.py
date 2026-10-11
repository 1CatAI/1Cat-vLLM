# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared speculative cache defaults for the SM70 runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType


def resolve_dflash_draft_cache_dtype(
    target_dtype: str, explicit_dtype: CacheDType | None, *, sm70: bool
) -> CacheDType | None:
    """One implicit draft storage policy for both runner generations.

    ``None`` means inherit target storage. An explicit draft choice always
    wins; only the existing SM70 quantized-target default selects FP16.
    """
    from vllm.utils.torch_utils import kv_cache_uses_per_token_head_scales

    if explicit_dtype is not None:
        return explicit_dtype
    if sm70 and (
        target_dtype.startswith("fp8")
        or kv_cache_uses_per_token_head_scales(target_dtype)
    ):
        return "auto"
    return None
