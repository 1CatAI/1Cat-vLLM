# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resource-driven planning primitives for the FlashInfer SM70 backend."""

from collections.abc import Sequence
from dataclasses import dataclass


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _round_up(value: int, alignment: int) -> int:
    return _cdiv(value, alignment) * alignment


@dataclass(frozen=True)
class SM70DeviceLimits:
    sm_count: int = 72
    registers_per_sm: int = 65_536
    shared_bytes_per_sm: int = 98_304
    max_threads_per_sm: int = 2_048
    max_warps_per_sm: int = 64
    max_ctas_per_sm: int = 32
    register_allocation_unit_per_warp: int = 256
    shared_allocation_unit: int = 256


DEFAULT_SM70_LIMITS = SM70DeviceLimits()


@dataclass(frozen=True)
class SM70KernelResources:
    threads_per_cta: int
    registers_per_thread: int
    shared_bytes_per_cta: int

    def resident_ctas(self, limits: SM70DeviceLimits) -> int:
        if self.threads_per_cta <= 0 or self.threads_per_cta % 32 != 0:
            raise ValueError("threads_per_cta must be a positive multiple of 32")
        if not 0 < self.registers_per_thread <= 255:
            raise ValueError("registers_per_thread must be in [1, 255]")
        if not 0 <= self.shared_bytes_per_cta <= limits.shared_bytes_per_sm:
            raise ValueError("shared_bytes_per_cta exceeds the SM70 SM limit")

        warps_per_cta = self.threads_per_cta // 32
        registers_per_warp = _round_up(
            self.registers_per_thread * 32,
            limits.register_allocation_unit_per_warp,
        )
        registers_per_cta = registers_per_warp * warps_per_cta
        allocated_shared = _round_up(
            self.shared_bytes_per_cta,
            limits.shared_allocation_unit,
        )

        register_limit = limits.registers_per_sm // registers_per_cta
        shared_limit = (
            limits.max_ctas_per_sm
            if allocated_shared == 0
            else limits.shared_bytes_per_sm // allocated_shared
        )
        return min(
            limits.max_ctas_per_sm,
            limits.max_threads_per_sm // self.threads_per_cta,
            limits.max_warps_per_sm // warps_per_cta,
            register_limit,
            shared_limit,
        )


@dataclass(frozen=True)
class FlashInferSM70Plan:
    split_kv: bool
    kv_chunk_size: int
    logical_ctas: int
    padded_ctas: int
    resident_ctas_per_sm: int
    grid_capacity: int

    @property
    def waves(self) -> float:
        return self.logical_ctas / self.grid_capacity


@dataclass(frozen=True)
class FlashInferSM70PlannerDecision:
    """Planner result plus an explicit, unpromoted delegate boundary."""

    stage: str
    workload: str
    plan: FlashInferSM70Plan
    planner_candidate: str
    dispatch: str
    delegate_backend: str
    delegate_dispatch: str
    kernel_promoted: bool

    @property
    def route_proof(self) -> dict[str, object]:
        return {
            "selected_backend": "FLASHINFER_SM70",
            "planner_stage": self.stage,
            "workload": self.workload,
            "planner_candidate": self.planner_candidate,
            "dispatch": self.dispatch,
            "delegate_backend": self.delegate_backend,
            "delegate_dispatch": self.delegate_dispatch,
            "kernel_promoted": self.kernel_promoted,
            "plan": {
                "split_kv": self.plan.split_kv,
                "kv_chunk_size": self.plan.kv_chunk_size,
                "logical_ctas": self.plan.logical_ctas,
                "padded_ctas": self.plan.padded_ctas,
                "resident_ctas_per_sm": self.plan.resident_ctas_per_sm,
                "grid_capacity": self.plan.grid_capacity,
            },
        }


def make_delegate_decision(
    *,
    stage: str,
    workload: str,
    plan: FlashInferSM70Plan,
) -> FlashInferSM70PlannerDecision:
    """Record a resource candidate without claiming the delegate's branch."""
    planner_candidates = {
        "prefill": "accepted_bm32_resource_envelope",
        "decode": "accepted_decode_resource_envelope",
    }
    if workload not in planner_candidates:
        raise ValueError(f"unsupported SM70 workload: {workload}")
    return FlashInferSM70PlannerDecision(
        stage=stage,
        workload=workload,
        plan=plan,
        planner_candidate=planner_candidates[workload],
        dispatch="delegate",
        delegate_backend="FLASH_ATTN_V100",
        delegate_dispatch="flash_v100_runtime_dispatch",
        kernel_promoted=False,
    )


def _grid_capacity(
    resources: SM70KernelResources,
    limits: SM70DeviceLimits,
) -> tuple[int, int]:
    resident_ctas = resources.resident_ctas(limits)
    if resident_ctas < 1:
        raise ValueError("kernel resources do not permit one resident CTA")
    return resident_ctas, resident_ctas * limits.sm_count


def plan_prefill(
    *,
    packed_qo_len: int,
    kv_len: int,
    num_kv_heads: int,
    cta_tile_q: int,
    kv_tile_size: int,
    resources: SM70KernelResources,
    limits: SM70DeviceLimits = DEFAULT_SM70_LIMITS,
    enable_cuda_graph: bool = False,
    fixed_split_size: int | None = None,
) -> FlashInferSM70Plan:
    """Plan FlashInfer-style prefill work from the compiled kernel envelope."""
    if min(packed_qo_len, kv_len, num_kv_heads, cta_tile_q, kv_tile_size) <= 0:
        raise ValueError("prefill dimensions and tile sizes must be positive")

    resident_ctas, capacity = _grid_capacity(resources, limits)
    query_tiles = _cdiv(packed_qo_len, cta_tile_q) * num_kv_heads

    if fixed_split_size is not None:
        if fixed_split_size <= 0:
            raise ValueError("fixed_split_size must be positive")
        chunk_size = _round_up(fixed_split_size, kv_tile_size)
    elif query_tiles >= capacity:
        chunk_size = _round_up(kv_len, kv_tile_size)
    else:
        max_splits = max(1, capacity // query_tiles)
        max_splits = min(max_splits, _cdiv(kv_len, kv_tile_size))
        chunk_size = _round_up(_cdiv(kv_len, max_splits), kv_tile_size)

    split_count = _cdiv(kv_len, chunk_size)
    logical_ctas = query_tiles * split_count
    padded_ctas = (
        max(capacity, logical_ctas)
        if enable_cuda_graph and split_count > 1
        else logical_ctas
    )
    return FlashInferSM70Plan(
        split_kv=split_count > 1,
        kv_chunk_size=chunk_size,
        logical_ctas=logical_ctas,
        padded_ctas=padded_ctas,
        resident_ctas_per_sm=resident_ctas,
        grid_capacity=capacity,
    )


def _decode_ctas(
    seq_lens: Sequence[int],
    num_kv_heads: int,
    chunk_size: int,
) -> int:
    return sum(_cdiv(seq_len, chunk_size) for seq_len in seq_lens) * num_kv_heads


def _adaptive_decode_chunk_size(
    seq_lens: Sequence[int],
    num_kv_heads: int,
    kv_tile_size: int,
    capacity: int,
) -> int:
    max_seq_len = max(seq_lens)
    upper = _round_up(max_seq_len, kv_tile_size)
    if len(seq_lens) * num_kv_heads >= capacity:
        return upper

    low_tiles = 1
    high_tiles = upper // kv_tile_size
    while low_tiles < high_tiles:
        mid_tiles = (low_tiles + high_tiles) // 2
        chunk_size = mid_tiles * kv_tile_size
        if _decode_ctas(seq_lens, num_kv_heads, chunk_size) <= capacity:
            high_tiles = mid_tiles
        else:
            low_tiles = mid_tiles + 1
    return low_tiles * kv_tile_size


def plan_decode(
    *,
    seq_lens: Sequence[int],
    num_kv_heads: int,
    kv_tile_size: int,
    resources: SM70KernelResources,
    limits: SM70DeviceLimits = DEFAULT_SM70_LIMITS,
    enable_cuda_graph: bool = False,
    fixed_split_size: int | None = None,
) -> FlashInferSM70Plan:
    """Plan decode partitions while allowing a quality-locked split size."""
    if not seq_lens or any(seq_len <= 0 for seq_len in seq_lens):
        raise ValueError("seq_lens must contain positive lengths")
    if num_kv_heads <= 0 or kv_tile_size <= 0:
        raise ValueError("num_kv_heads and kv_tile_size must be positive")

    resident_ctas, capacity = _grid_capacity(resources, limits)
    if fixed_split_size is not None:
        if fixed_split_size <= 0:
            raise ValueError("fixed_split_size must be positive")
        chunk_size = _round_up(fixed_split_size, kv_tile_size)
    else:
        chunk_size = _adaptive_decode_chunk_size(
            seq_lens,
            num_kv_heads,
            kv_tile_size,
            capacity,
        )

    logical_ctas = _decode_ctas(seq_lens, num_kv_heads, chunk_size)
    split_kv = any(seq_len > chunk_size for seq_len in seq_lens)
    padded_ctas = (
        max(capacity, logical_ctas) if enable_cuda_graph and split_kv else logical_ctas
    )
    return FlashInferSM70Plan(
        split_kv=split_kv,
        kv_chunk_size=chunk_size,
        logical_ctas=logical_ctas,
        padded_ctas=padded_ctas,
        resident_ctas_per_sm=resident_ctas,
        grid_capacity=capacity,
    )
