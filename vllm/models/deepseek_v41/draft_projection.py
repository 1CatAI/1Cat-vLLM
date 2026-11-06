# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shard independent output rows without changing the full-call split-K."""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProjectionPlan:
    first: int
    last: int
    split: int

    def __post_init__(self) -> None:
        if (
            self.first < 0
            or self.last <= self.first
            or self.first % 32
            or self.last % 32
            or self.split < 1
        ):
            raise ValueError(
                "projection plan requires aligned increasing rows and positive split"
            )


def projection_plan(n: int, k: int, m: int, world: int, rank: int) -> ProjectionPlan:
    """Validated row slice and the original replicated GEMV's split count."""
    if world not in (1, 2, 4, 8) or not 0 <= rank < world:
        raise ValueError("draft projection requires TP 1/2/4/8 and a valid rank")
    if n <= 0 or k <= 0 or n % (32 * world) or k % 32 or not 1 <= m <= 8:
        raise ValueError("draft projection requires aligned N/K and 1..8 input rows")
    rows = n // world
    ctas = (n + 31) // 32
    split = max(1, min(max(1, k // 512), (240 + ctas - 1) // ctas))
    return ProjectionPlan(rank * rows, (rank + 1) * rows, split)


def admit_projection(
    world: int,
    gemv_ready: bool,
    biased: bool,
    n: int,
    k: int,
    *,
    weight: torch.Tensor,
    scales: torch.Tensor | None,
) -> None:
    """Refuse unsupported immutable model/provider conditions at construction."""
    if world not in (2, 4, 8):
        raise ValueError("V4.1 draft projection sharding requires TP 2/4/8")
    if not gemv_ready:
        raise ValueError("V4.1 draft projection sharding requires the v41k FP8 GEMV")
    if biased:
        raise ValueError("V4.1 sharded main projection must be bias-free")
    projection_plan(n, k, 1, world, 0)
    if (
        weight.dtype != torch.float8_e4m3fn
        or scales is None
        or scales.dtype != torch.float8_e8m0fnu
    ):
        raise ValueError("draft projection requires raw E4M3 weights and UE8M0 scales")
    if (
        weight.shape != (n, k)
        or scales.shape != (n // 32, k // 32)
        or not (weight.is_contiguous() and scales.is_contiguous())
    ):
        raise ValueError("draft projection requires aligned contiguous weight/scales")


def sharded_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    rank: int,
    world: int,
    gather: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    n, k = weight.shape
    plan = projection_plan(n, k, x.shape[0], world, rank)
    if (
        x.ndim != 2
        or x.shape[1] != k
        or x.dtype != torch.float16
        or not x.is_contiguous()
    ):
        raise ValueError("draft projection requires contiguous FP16 [M, K] input")
    if weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float8_e8m0fnu:
        raise ValueError("draft projection requires raw E4M3 weights and UE8M0 scales")
    if scales.shape != (n // 32, k // 32) or not (
        weight.is_contiguous() and scales.is_contiguous()
    ):
        raise ValueError("draft projection requires aligned contiguous weight/scales")
    local_weight = weight[plan.first : plan.last]
    local_scales = scales[plan.first // 32 : plan.last // 32]
    result = torch.empty(
        x.shape[0], plan.last - plan.first, dtype=x.dtype, device=x.device
    )
    partials = torch.empty(
        plan.split * result.numel() if plan.split > 1 else 1,
        dtype=torch.float32,
        device=x.device,
    )
    torch.ops.v41k.fp8_gemv(
        result,
        x,
        local_weight.view(torch.uint8),
        local_scales.view(torch.uint8),
        partials,
        plan.split,
        None,
    )
    return gather(result)
