# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared SM70 MoE route selection for quantized TurboMind kernels."""

from dataclasses import dataclass
from enum import Enum


class Sm70MoeStageRoute(str, Enum):
    BATCHED = "batched"
    PER_EXPERT_DISPATCH = "per_expert_dispatch"
    DENSE = "dense"
    ACTIVE_DENSE = "active_dense"
    INDEXED = "indexed"
    COMPACT = "compact"
    INDEXED_PREFILL = "indexed_prefill"
    ACTIVE_GROUPED = "active_grouped"
    CHUNKED = "chunked"


@dataclass(frozen=True)
class Sm70MoeRoutePlan:
    use_batched_moe_gemm: bool
    use_batched_strict_w13: bool
    use_batched_exact_w2: bool
    use_batched_active_exact_w2: bool
    w13: Sm70MoeStageRoute
    w2: Sm70MoeStageRoute
    chunk_tokens: int = 0
    weighted_reduce: bool = False
    strict: bool = False
    batched_indexed: bool = False


def select_sm70_quantized_moe_route(
    *,
    batched_enabled: bool,
    num_tokens: int,
    total_slots: int,
    batched_decode_max_tokens: int = 0,
    strict_dense_w13: bool = False,
    exact_w2: bool = False,
    active_exact_w2: bool = False,
    w13_per_expert_dispatch: bool = False,
    w2_per_expert_dispatch: bool = False,
) -> Sm70MoeRoutePlan:
    """Pick the quantization-independent SM70 MoE stage route.

    This only chooses the compute order. AWQ and FP8 keep their own thin
    adapters because their packed weights and scale descriptors differ.
    """
    use_batched_strict_w13 = batched_enabled and strict_dense_w13
    use_batched_for_shape = (
        batched_decode_max_tokens <= 0 or num_tokens <= batched_decode_max_tokens
    )
    use_batched_moe_gemm = (
        batched_enabled and not use_batched_strict_w13 and use_batched_for_shape
    )
    if not use_batched_moe_gemm:
        return Sm70MoeRoutePlan(
            use_batched_moe_gemm=False,
            use_batched_strict_w13=use_batched_strict_w13,
            use_batched_exact_w2=False,
            use_batched_active_exact_w2=False,
            w13=Sm70MoeStageRoute.DENSE,
            w2=Sm70MoeStageRoute.DENSE,
        )

    w13 = (
        Sm70MoeStageRoute.PER_EXPERT_DISPATCH
        if w13_per_expert_dispatch
        else Sm70MoeStageRoute.BATCHED
    )
    request_active_exact_w2 = active_exact_w2
    use_batched_active_exact_w2 = request_active_exact_w2 and total_slots <= 128
    use_batched_exact_w2 = exact_w2 or (
        request_active_exact_w2 and not use_batched_active_exact_w2
    )
    if use_batched_active_exact_w2:
        w2 = Sm70MoeStageRoute.ACTIVE_DENSE
    elif use_batched_exact_w2:
        w2 = Sm70MoeStageRoute.DENSE
    elif w2_per_expert_dispatch:
        w2 = Sm70MoeStageRoute.PER_EXPERT_DISPATCH
    else:
        w2 = Sm70MoeStageRoute.BATCHED

    return Sm70MoeRoutePlan(
        use_batched_moe_gemm=True,
        use_batched_strict_w13=False,
        use_batched_exact_w2=use_batched_exact_w2,
        use_batched_active_exact_w2=use_batched_active_exact_w2,
        w13=w13,
        w2=w2,
    )


def select_single_token_plan(
    *,
    compact_w13: bool,
    indexed_w13: bool,
    indexed_w2: bool,
    weighted_reduce: bool,
    strict: bool = False,
    batched_indexed: bool = False,
) -> Sm70MoeRoutePlan:
    """Keep compact > indexed > dense and the legacy strict-W2 override."""
    return Sm70MoeRoutePlan(
        use_batched_moe_gemm=False,
        use_batched_strict_w13=strict,
        use_batched_exact_w2=False,
        use_batched_active_exact_w2=False,
        w13=Sm70MoeStageRoute.COMPACT
        if compact_w13
        else Sm70MoeStageRoute.INDEXED
        if indexed_w13 and not strict
        else Sm70MoeStageRoute.ACTIVE_DENSE,
        w2=Sm70MoeStageRoute.INDEXED
        if indexed_w2 and not strict
        else Sm70MoeStageRoute.ACTIVE_DENSE,
        weighted_reduce=weighted_reduce,
        strict=strict,
        batched_indexed=batched_indexed,
    )
