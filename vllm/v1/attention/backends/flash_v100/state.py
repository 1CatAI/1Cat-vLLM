# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared one-shot implementation state (one owner per binding)."""

from __future__ import annotations

import sys
import types

from vllm.logger import log_once_seen, set_log_once_state

_warned_feature_fallback = False
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
_logged_fp8_prefill_bridge = False
_logged_prefill_compare = False
_logged_dflash_prefix_dump = False
_logged_prefill_ddtree_dense = False
_logged_prefill_ddtree_triton = False
_logged_prefill_ddtree_triton_fallback = False


LOG_KEYS = {
    "_logged_decode_dense_cache": "flash_v100._logged_decode_dense_cache",
    "_logged_decode_dense_reference": "flash_v100._logged_decode_dense_reference",
    "_logged_decode_flash": "flash_v100._logged_decode_flash",
    "_logged_decode_paged_prefill": "flash_v100._logged_decode_paged_prefill",
    "_logged_decode_paged_prefill_bhmd": "flash_v100._logged_decode_paged_prefill_bhmd",
    "_logged_decode_paged_prefill_bhmd_q_clone": (
        "flash_v100._logged_decode_paged_prefill_bhmd_q_clone"
    ),
    "_logged_decode_wmma_wrapper": "flash_v100._logged_decode_wmma_wrapper",
    "_warned_decode_fallback": "flash_v100._warned_decode_fallback",
    "_warned_decode_strict_fallback": "flash_v100._warned_decode_strict_fallback",
}


class _LogStateModule(types.ModuleType):
    """Legacy flags are live views of the logger's process-wide event keys."""

    def __getattr__(self, name):
        if name in LOG_KEYS:
            return log_once_seen(LOG_KEYS[name])
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in LOG_KEYS:
            set_log_once_state(LOG_KEYS[name], value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _LogStateModule
