# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission for the single-request E4M3 FP32 small-Q route."""

from vllm.v1.attention.ops.sm70_grouped import (  # noqa: F401
    GROUP_ROWS,
    MAX_GROUPS_PER_CALL,
    grouped_e4m3_fp32_allowed,
    grouped_e4m3_fp32_groups_allowed,
)


def load_grouped_e4m3_fp32():
    try:
        from flash_attn_v100 import (
            flash_attn_grouped_e4m3_fp32_available,
            flash_attn_grouped_e4m3_fp32_paged,
        )
    except ImportError:
        return None
    if not flash_attn_grouped_e4m3_fp32_available():
        return None
    from vllm.v1.attention.ops.sm70_grouped_long import wrap_long_attention

    return wrap_long_attention(flash_attn_grouped_e4m3_fp32_paged)
