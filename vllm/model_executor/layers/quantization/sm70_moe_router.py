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


@dataclass(frozen=True)
class Sm70MoeRoutePlan:
    use_batched_moe_gemm: bool
    use_batched_strict_w13: bool
    use_batched_exact_w2: bool
    use_batched_active_exact_w2: bool
    w13: Sm70MoeStageRoute
    w2: Sm70MoeStageRoute


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
