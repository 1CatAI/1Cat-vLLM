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
class FlashInferSM70PrefillWorkDescription:
    """Immutable host description of FlashInfer-style prefill work."""

    planning_mode: str
    qo_lens: tuple[int, ...]
    packed_qo_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    num_qo_heads: int
    num_kv_heads: int
    head_dim_vo: int
    cta_tile_q: int
    kv_tile_size: int
    enable_cuda_graph: bool
    valid_work_items: int
    padded_work_items: int
    request_indices: tuple[int, ...]
    qo_tile_indices: tuple[int, ...]
    kv_tile_indices: tuple[int, ...]
    merge_indptr: tuple[int, ...]
    o_indptr: tuple[int, ...]
    block_valid_mask: tuple[bool, ...]
    int_workspace_bytes: int
    float_workspace_bytes: int
    tmp_v_bytes: int
    tmp_s_bytes: int
    merge_indptr_bytes: int
    block_valid_mask_bytes: int
    plan: FlashInferSM70Plan

    @property
    def is_exact(self) -> bool:
        return self.planning_mode == "exact"

    @property
    def split_kv(self) -> bool:
        return self.plan.split_kv

    @property
    def kv_chunk_size(self) -> int:
        return self.plan.kv_chunk_size

    @property
    def valid_ctas(self) -> int:
        return self.plan.logical_ctas

    @property
    def padded_ctas(self) -> int:
        return self.plan.padded_ctas

    @property
    def total_num_rows(self) -> int:
        return sum(self.qo_lens)


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


def _aligned_workspace_bytes(
    allocations: Sequence[tuple[int, int]],
) -> int:
    offset = 0
    for size, alignment in allocations:
        offset = _round_up(offset, alignment)
        offset += size
    return _round_up(offset, 16) if offset else 0


def _prefill_ctas(
    query_tiles: Sequence[int],
    kv_lens: Sequence[int],
    kv_chunk_size: int,
    num_kv_heads: int,
) -> int:
    work_items = sum(
        num_query_tiles * _cdiv(max(kv_len, 1), kv_chunk_size)
        for num_query_tiles, kv_len in zip(query_tiles, kv_lens)
    )
    return work_items * num_kv_heads


def plan_prefill_requests(
    *,
    qo_lens: Sequence[int],
    kv_lens: Sequence[int],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_vo: int,
    cta_tile_q: int,
    kv_tile_size: int,
    resources: SM70KernelResources,
    limits: SM70DeviceLimits = DEFAULT_SM70_LIMITS,
    enable_cuda_graph: bool = False,
    fixed_split_size: int | None = None,
    planning_mode: str = "exact",
) -> FlashInferSM70PrefillWorkDescription:
    """Describe per-request work using FlashInfer's prefill scheduler order."""
    qo_lens_tuple = tuple(int(length) for length in qo_lens)
    kv_lens_tuple = tuple(int(length) for length in kv_lens)
    if not qo_lens_tuple or len(qo_lens_tuple) != len(kv_lens_tuple):
        raise ValueError("qo_lens and kv_lens must have the same non-zero length")
    if any(length < 0 for length in (*qo_lens_tuple, *kv_lens_tuple)):
        raise ValueError("qo_lens and kv_lens must be non-negative")
    if min(num_qo_heads, num_kv_heads, head_dim_vo, cta_tile_q, kv_tile_size) <= 0:
        raise ValueError("head counts, dimensions, and tile sizes must be positive")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if planning_mode not in {"exact", "upper_bound"}:
        raise ValueError("planning_mode must be 'exact' or 'upper_bound'")

    resident_ctas, capacity = _grid_capacity(resources, limits)
    gqa_group_size = num_qo_heads // num_kv_heads
    packed_qo_lens = tuple(length * gqa_group_size for length in qo_lens_tuple)
    query_tiles = tuple(_cdiv(length, cta_tile_q) for length in packed_qo_lens)

    if fixed_split_size is not None:
        if fixed_split_size <= 0:
            raise ValueError("fixed_split_size must be positive")
        kv_chunk_size = _round_up(fixed_split_size, kv_tile_size)
    else:
        low_tiles = 1
        high_tiles = max(
            1,
            max(_cdiv(max(length, 1), kv_tile_size) for length in kv_lens_tuple),
        )
        while low_tiles < high_tiles:
            mid_tiles = (low_tiles + high_tiles) // 2
            candidate_chunk_size = mid_tiles * kv_tile_size
            if (
                _prefill_ctas(
                    query_tiles,
                    kv_lens_tuple,
                    candidate_chunk_size,
                    num_kv_heads,
                )
                <= capacity
            ):
                high_tiles = mid_tiles
            else:
                low_tiles = mid_tiles + 1
        kv_chunk_size = low_tiles * kv_tile_size

    kv_chunks = tuple(_cdiv(max(length, 1), kv_chunk_size) for length in kv_lens_tuple)
    has_multiple_kv_chunks = any(num_chunks > 1 for num_chunks in kv_chunks)
    split_kv = has_multiple_kv_chunks
    request_indices: list[int] = []
    qo_tile_indices: list[int] = []
    kv_tile_indices: list[int] = []
    merge_indptr = [0]
    o_indptr = [0]

    for request_idx, (qo_len, num_query_tiles, num_kv_chunks) in enumerate(
        zip(qo_lens_tuple, query_tiles, kv_chunks)
    ):
        for qo_tile_idx in range(num_query_tiles):
            for kv_tile_idx in range(num_kv_chunks):
                request_indices.append(request_idx)
                qo_tile_indices.append(qo_tile_idx)
                kv_tile_indices.append(kv_tile_idx)
        for _ in range(qo_len):
            merge_indptr.append(merge_indptr[-1] + num_kv_chunks)
        o_indptr.append(o_indptr[-1] + qo_len * num_kv_chunks)

    valid_work_items = len(request_indices)
    padded_work_items = valid_work_items
    if enable_cuda_graph:
        total_num_tiles_q_upper_bound = (
            _cdiv(sum(packed_qo_lens), cta_tile_q) + len(qo_lens_tuple) - 1
        )
        padded_work_items = max(
            capacity // num_kv_heads,
            total_num_tiles_q_upper_bound,
        )
        if valid_work_items > padded_work_items:
            raise ValueError(
                "prefill work exceeds the fixed CUDA graph padding capacity"
            )

    valid_ctas = valid_work_items * num_kv_heads
    padded_ctas = padded_work_items * num_kv_heads
    plan = FlashInferSM70Plan(
        split_kv=split_kv,
        kv_chunk_size=kv_chunk_size,
        logical_ctas=valid_ctas,
        padded_ctas=padded_ctas,
        resident_ctas_per_sm=resident_ctas,
        grid_capacity=capacity,
    )

    index_bytes = 4
    int_allocations = [
        (padded_work_items * index_bytes, 16),
        (padded_work_items * index_bytes, 16),
        (padded_work_items * index_bytes, 16),
        ((len(qo_lens_tuple) + 1) * index_bytes, 16),
        (index_bytes, 1),
    ]
    if enable_cuda_graph:
        int_allocations.append((index_bytes, 16))

    total_num_rows = sum(qo_lens_tuple)
    merge_indptr_bytes = (total_num_rows + 1) * index_bytes if split_kv else 0
    block_valid_mask_bytes = padded_work_items if split_kv else 0
    if split_kv:
        int_allocations.extend(
            [
                (merge_indptr_bytes, 16),
                (block_valid_mask_bytes, 16),
            ]
        )

    float_bytes = 4
    tmp_v_bytes = (
        num_qo_heads * padded_work_items * cta_tile_q * head_dim_vo * float_bytes
        if split_kv
        else 0
    )
    tmp_s_bytes = (
        num_qo_heads * padded_work_items * cta_tile_q * float_bytes if split_kv else 0
    )
    float_allocations = [(tmp_v_bytes, 16), (tmp_s_bytes, 16)] if split_kv else []

    return FlashInferSM70PrefillWorkDescription(
        planning_mode=planning_mode,
        qo_lens=qo_lens_tuple,
        packed_qo_lens=packed_qo_lens,
        kv_lens=kv_lens_tuple,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_vo=head_dim_vo,
        cta_tile_q=cta_tile_q,
        kv_tile_size=kv_tile_size,
        enable_cuda_graph=enable_cuda_graph,
        valid_work_items=valid_work_items,
        padded_work_items=padded_work_items,
        request_indices=tuple(request_indices),
        qo_tile_indices=tuple(qo_tile_indices),
        kv_tile_indices=tuple(kv_tile_indices),
        merge_indptr=tuple(merge_indptr) if split_kv else (),
        o_indptr=tuple(o_indptr),
        block_valid_mask=(
            tuple(index < valid_work_items for index in range(padded_work_items))
            if split_kv
            else ()
        ),
        int_workspace_bytes=_aligned_workspace_bytes(int_allocations),
        float_workspace_bytes=_aligned_workspace_bytes(float_allocations),
        tmp_v_bytes=tmp_v_bytes,
        tmp_s_bytes=tmp_s_bytes,
        merge_indptr_bytes=merge_indptr_bytes,
        block_valid_mask_bytes=block_valid_mask_bytes,
        plan=plan,
    )


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
