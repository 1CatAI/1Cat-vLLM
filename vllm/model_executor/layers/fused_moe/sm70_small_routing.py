# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch expert alignment without a sort or inverse-sort pipeline."""

from dataclasses import dataclass

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@dataclass(frozen=True)
class Sm70SmallRoutingCapability:
    operator: str = "sm70_small_expert_route"
    graph_safe: bool = True
    min_m: int = 1
    max_m: int = 32

    def reason(self, x, ids, experts):
        if not x.is_cuda or not current_platform.is_device_capability(70):
            return "requires_sm70"
        if x.dtype != torch.float16 or ids.dtype not in (torch.int32, torch.int64):
            return "requires_fp16_activations_and_integer_routes"
        if x.ndim != 2 or ids.ndim != 2 or x.shape[0] != ids.shape[0]:
            return "invalid_route_geometry"
        if not self.min_m <= x.shape[0] <= self.max_m:
            return "outside_small_batch_band"
        if x.shape[1] != 2560 or experts != 512 or not 1 <= ids.shape[1] <= 16:
            return "unmeasured_expert_geometry"
        if not x.is_contiguous() or not ids.is_contiguous() or x.device != ids.device:
            return "requires_contiguous_colocated_routes"
        return None


SM70_SMALL_ROUTING = Sm70SmallRoutingCapability()


@dataclass(frozen=True)
class Sm70PrefillRoutingCapability(Sm70SmallRoutingCapability):
    min_m: int = 33
    max_m: int = 16384


SM70_PREFILL_ROUTING = Sm70PrefillRoutingCapability()


def _prefill_enabled():
    config = get_current_vllm_config_or_none()
    return config is None or config.kernel_config.sm70_gguf.prefill_routing


@triton.jit
def _prefill_gather_inverse(
    X,
    ORDER,
    ROUTED,
    INVERSE,
    R: tl.constexpr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, h = index // H, index % H
    original = tl.load(ORDER + row, row < R, other=0)
    value = tl.load(X + (original // TOP_K) * H + h, index < R * H, other=0)
    tl.store(ROUTED + index, value, index < R * H)
    # ORDER is a permutation, so each inverse entry has exactly one writer.
    tl.store(INVERSE + original, row, (row < R) & (h == 0))


@triton.jit
def _route_and_gather(
    X,
    IDS,
    ROUTED,
    OFFSETS,
    SORTED_IDS,
    INVERSE,
    R: tl.constexpr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TILE_R: tl.constexpr,
    TILE_H: tl.constexpr,
):
    # Each program derives the same stable positions independently. This
    # allows activation gathering without cross-CTA synchronization or a
    # second launch. Work is bounded by the small routing capability.
    all_r = tl.arange(0, BLOCK_R)
    all_ids = tl.load(IDS + all_r, all_r < R, other=0).to(tl.int32)
    counts = tl.histogram(all_ids, BLOCK_E, mask=all_r < R)
    ends = tl.cumsum(counts)
    starts = ends - counts
    r = tl.program_id(0) * TILE_R + tl.arange(0, TILE_R)
    ids = tl.load(IDS + r, r < R, other=0).to(tl.int32)
    preceding = tl.sum(
        ((all_r[None, :] < r[:, None]) & (all_ids[None, :] == ids[:, None])).to(
            tl.int32
        ),
        axis=1,
    )
    positions = tl.gather(starts, ids, 0) + preceding
    h = tl.program_id(1) * TILE_H + tl.arange(0, TILE_H)
    values = tl.load(
        X + (r // TOP_K)[:, None] * H + h[None, :],
        (r[:, None] < R) & (h[None, :] < H),
        other=0,
    )
    tl.store(
        ROUTED + positions[:, None] * H + h[None, :],
        values,
        (r[:, None] < R) & (h[None, :] < H),
    )
    if tl.program_id(1) == 0:
        tl.store(INVERSE + r, positions, r < R)
        tl.store(SORTED_IDS + positions, ids, r < R)
        if tl.program_id(0) == 0:
            e = tl.arange(0, BLOCK_E)
            tl.store(OFFSETS + e, starts)
            tl.store(OFFSETS + BLOCK_E, R)


@triton.jit
def _weighted_route_value(
    DOWN,
    INVERSE,
    WEIGHTS,
    token,
    h,
    M: tl.constexpr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    K: tl.constexpr,
):
    route = token * TOP_K + K
    position = tl.load(INVERSE + route, token < M, other=0)
    value = tl.load(DOWN + position * H + h, token < M, other=0).to(tl.float32)
    weight = tl.load(WEIGHTS + route, token < M, other=0).to(tl.float32)
    return value * weight


@triton.jit
def _unroute_weighted_sum(
    DOWN,
    INVERSE,
    WEIGHTS,
    OUT,
    M: tl.constexpr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, h = index // H, index % H
    # Torch Reduce.cuh uses vt0=4 independent accumulators for this
    # non-contiguous reduction axis and short top-k. Preserve its order,
    # including separate FP32 products, without materializing the product.
    acc0 = tl.full((BLOCK,), 0, tl.float32)
    acc1 = tl.full((BLOCK,), 0, tl.float32)
    acc2 = tl.full((BLOCK,), 0, tl.float32)
    acc3 = tl.full((BLOCK,), 0, tl.float32)
    for base in tl.static_range(0, TOP_K, 4):
        acc0 += _weighted_route_value(
            DOWN, INVERSE, WEIGHTS, token, h, M, H, TOP_K, base
        )
        if base + 1 < TOP_K:
            acc1 += _weighted_route_value(
                DOWN, INVERSE, WEIGHTS, token, h, M, H, TOP_K, base + 1
            )
        if base + 2 < TOP_K:
            acc2 += _weighted_route_value(
                DOWN, INVERSE, WEIGHTS, token, h, M, H, TOP_K, base + 2
            )
        if base + 3 < TOP_K:
            acc3 += _weighted_route_value(
                DOWN, INVERSE, WEIGHTS, token, h, M, H, TOP_K, base + 3
            )
    acc = ((acc0 + acc1) + acc2) + acc3
    tl.store(OUT + index, acc, token < M)


def _small_route(
    x: torch.Tensor, ids: torch.Tensor, experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, top_k = ids.shape
    r, h = m * top_k, x.shape[1]
    if SM70_SMALL_ROUTING.reason(x, ids, experts) is not None:
        if SM70_PREFILL_ROUTING.reason(x, ids, experts) is None and _prefill_enabled():
            sorted_ids, order = ids.reshape(-1).int().sort(stable=True)
            boundaries = torch.arange(experts + 1, device=x.device, dtype=torch.int32)
            offsets = torch.searchsorted(sorted_ids, boundaries).int()
            routed = x.new_empty((r, h))
            inverse = torch.empty(r, device=x.device, dtype=torch.int32)
            _prefill_gather_inverse[(triton.cdiv(r * h, 1024),)](
                x, order, routed, inverse, r, h, top_k, 1024, num_warps=4
            )
            return routed, offsets, sorted_ids.long(), inverse
        sorted_ids, order = ids.reshape(-1).long().sort()
        boundaries = torch.arange(experts + 1, device=x.device)
        offsets = torch.searchsorted(sorted_ids, boundaries).to(torch.int32)
        routed = x[order // top_k].contiguous()
        return routed, offsets, sorted_ids.long(), order.argsort().to(torch.int32)
    routed = x.new_empty((r, h))
    offsets = torch.empty(experts + 1, device=x.device, dtype=torch.int32)
    sorted_ids = torch.empty(r, device=x.device, dtype=torch.int64)
    inverse = torch.empty(r, device=x.device, dtype=torch.int32)
    _route_and_gather[(triton.cdiv(r, 8), triton.cdiv(h, 256))](
        x,
        ids,
        routed,
        offsets,
        sorted_ids,
        inverse,
        r,
        h,
        top_k,
        triton.next_power_of_2(r),
        experts,
        8,
        256,
        num_warps=4,
    )
    logger.info_once("SM70 fused expert alignment/gather enabled for M=%d.", m)
    return routed, offsets, sorted_ids, inverse


def _small_route_fake(x: torch.Tensor, ids: torch.Tensor, experts: int):
    r = ids.numel()
    return (
        x.new_empty((r, x.shape[1])),
        torch.empty(experts + 1, device=x.device, dtype=torch.int32),
        torch.empty(r, device=x.device, dtype=torch.int64),
        torch.empty(r, device=x.device, dtype=torch.int32),
    )


def _small_unroute(
    down: torch.Tensor, inverse: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    m, top_k = weights.shape
    h = down.shape[1]
    if not (
        down.is_cuda
        and current_platform.is_device_capability(70)
        and down.dtype == torch.float16
        and (1 <= m <= 32 or (33 <= m <= 16384 and _prefill_enabled()))
        and h == 2560
        and 1 <= top_k <= 16
        and down.is_contiguous()
        and inverse.is_contiguous()
        and weights.is_contiguous()
    ):
        restored = down[inverse.long()].view(m, top_k, h)
        return (restored.float() * weights[..., None].float()).sum(1).to(down.dtype)
    out = down.new_empty((m, h))
    _unroute_weighted_sum[(triton.cdiv(m * h, 256),)](
        down,
        inverse,
        weights,
        out,
        m,
        h,
        top_k,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def _small_unroute_fake(
    down: torch.Tensor, inverse: torch.Tensor, weights: torch.Tensor
):
    return down.new_empty((weights.shape[0], down.shape[1]))


direct_register_custom_op(
    op_name="sm70_small_expert_route",
    op_func=_small_route,
    fake_impl=_small_route_fake,
)
direct_register_custom_op(
    op_name="sm70_small_expert_unroute",
    op_func=_small_unroute,
    fake_impl=_small_unroute_fake,
)
