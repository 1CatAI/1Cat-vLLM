# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared one-shot implementation state (one owner per binding)."""

from __future__ import annotations

_warned_feature_fallback = False
_warned_decode_fallback = False
_warned_decode_strict_fallback = False
_logged_prefill_flash = False
_logged_prefill_prefix_flash = False
_logged_prefill_prefix_contig_dense = False
_logged_prefill_prefix_bfla = False
_logged_prefill_prefix_splitkv = False
_logged_prefill_paged_cache = False
_logged_prefill_smallq_decode = False
_logged_prefill_prefix_decode_rows = False
_logged_prefill_prefix_decode_rows_grouped = False
_logged_prefill_smallq_decode_xqa = False
_logged_prefill_smallq_grouped_verify = False
_logged_prefill_smallq_grouped_verify_gate = False
_logged_prefill_triton_safe = False
_logged_decode_flash = False
_logged_decode_dense_reference = False
_logged_decode_dense_cache = False
_logged_decode_paged_prefill = False
_logged_decode_paged_prefill_bhmd = False
_logged_decode_paged_prefill_bhmd_q_clone = False
_logged_decode_wmma_wrapper = False
_logged_fp8_prefill_bridge = False
_logged_prefill_compare = False
_logged_dflash_prefix_dump = False
_logged_prefill_ddtree_dense = False
_logged_prefill_ddtree_triton = False
_logged_prefill_ddtree_triton_fallback = False
