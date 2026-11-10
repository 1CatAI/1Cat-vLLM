# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission for the single-request E4M3 FP32 small-Q route."""

from vllm.v1.attention.ops.sm70_grouped import (  # noqa: F401
    GROUP_ROWS,
    MAX_CONTEXT,
    MAX_GROUPS_PER_CALL,
    grouped_e4m3_fp32_allowed,
    grouped_e4m3_fp32_groups_allowed,
)


def grouped_context_limit() -> int:
    """Only the qualified long graph may exceed the ordinary E4M3 provider."""
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )
    from vllm.v1.attention.ops.sm70_grouped_long import (
        builtin_dynamic_context,
        long_attention_graph_contract,
    )

    descriptor = (
        get_forward_context().batch_descriptor
        if is_forward_context_available()
        else None
    )
    bucket = getattr(descriptor, "attention_context_bucket", None)
    if bucket is not None and bucket > 0 and builtin_dynamic_context():
        capacity, _ = long_attention_graph_contract(bucket)
        if capacity is not None and bucket <= capacity:
            return capacity
    return MAX_CONTEXT


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
