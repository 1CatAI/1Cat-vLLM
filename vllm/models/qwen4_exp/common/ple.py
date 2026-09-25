# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen4Exp PLE helpers."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.distributed.utils import get_layers_outside_first_pp_rank
from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def check_ple_layers_on_first_pp_rank(text_config: Any, pp_size: int) -> None:
    """Refuse a pipeline split that puts a PLE layer beyond the first rank.

    The n-gram context and ``query_start_loc`` a PLE layer consumes are only
    prepared on the first pipeline rank, and later ranks receive no input ids
    at all. ``ple_layer_ids`` are 1-based: id ``L`` attaches the PLE module to
    decoder layer ``L - 1`` (see ``Qwen4ExpDecoderLayer``), so that is the
    index the partition has to keep on rank 0.
    """
    if pp_size <= 1:
        return
    ple_decoder_layers = [int(layer_id) - 1 for layer_id in text_config.ple_layer_ids]
    misplaced, first_rank_end = get_layers_outside_first_pp_rank(
        ple_decoder_layers, int(text_config.num_hidden_layers), pp_size
    )
    if misplaced:
        raise RuntimeError(
            "N-gram PLE embedding requires every PLE layer on the first pipeline "
            f"rank, which holds decoder layers 0..{first_rank_end - 1}, but the "
            f"PLE modules of decoder layers {misplaced} fall on a later stage. "
            "Either run with pipeline_parallel_size=1 or move the split with "
            "VLLM_PP_LAYER_PARTITION so those layers stay on rank 0."
        )


@dataclass(frozen=True)
class PLEShardOverlap:
    """Source and destination slices for one checkpoint embedding shard."""

    source_start: int
    destination_start: int
    row_count: int


def compute_ple_shard_overlap(
    *,
    checkpoint_start: int,
    checkpoint_rows: int,
    tp_start: int,
    tp_end: int,
) -> PLEShardOverlap | None:
    """Compute the overlap of a checkpoint shard and one TP vocabulary range."""

    if checkpoint_start < 0 or checkpoint_rows < 0:
        raise ValueError("checkpoint shard bounds must be non-negative")
    if tp_start < 0 or tp_end < tp_start:
        raise ValueError("invalid TP vocabulary range")
    checkpoint_end = checkpoint_start + checkpoint_rows
    overlap_start = max(checkpoint_start, tp_start)
    overlap_end = min(checkpoint_end, tp_end)
    if overlap_start >= overlap_end:
        return None
    return PLEShardOverlap(
        source_start=overlap_start - checkpoint_start,
        destination_start=overlap_start - tp_start,
        row_count=overlap_end - overlap_start,
    )


def copy_ple_embedding_shard_(
    destination: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> int:
    """Copy the overlapping rows of a PLE checkpoint shard into a TP table."""

    if destination.ndim == 0 or loaded_weight.ndim != destination.ndim:
        raise ValueError("destination and loaded weight must have matching ranks")
    if destination.shape[1:] != loaded_weight.shape[1:]:
        raise ValueError(
            "embedding shard dimensions do not match: "
            f"{tuple(destination.shape[1:])} != {tuple(loaded_weight.shape[1:])}"
        )
    if destination.shape[0] < tp_end - tp_start:
        raise ValueError("destination does not cover the requested TP range")
    overlap = compute_ple_shard_overlap(
        checkpoint_start=checkpoint_start,
        checkpoint_rows=loaded_weight.shape[0],
        tp_start=tp_start,
        tp_end=tp_end,
    )
    if overlap is None:
        return 0
    source = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
    target = destination.narrow(0, overlap.destination_start, overlap.row_count)
    with torch.no_grad():
        # copy_ converts device and dtype itself. Staging through .to() would
        # leave a device-side copy of the slice cached by the allocator, one
        # checkpoint shard (0.37 GiB for Qwen3.8-Flash-Next) per card.
        target.copy_(source)
    return overlap.row_count


@dataclass(frozen=True)
class PLEPlacement:
    """How many PLE table rows live in each tier of one tensor-parallel rank.

    The tiers are consecutive ranges of the rank's rows: device memory first,
    then pinned host memory, then the disk tier the PLE offload worker reads
    from the mapped checkpoint.
    """

    vram_rows: int
    host_rows: int
    disk_rows: int

    @property
    def local_rows(self) -> int:
        """Rows the rank holds itself."""
        return self.vram_rows + self.host_rows

    @property
    def total_rows(self) -> int:
        return self.local_rows + self.disk_rows


def plan_ple_placement(
    *,
    total_rows: int,
    row_bytes: int,
    host_budget_bytes: int,
    vram_budget_bytes: int | None,
    disk_allowed: bool = False,
) -> PLEPlacement:
    """Split the PLE rows of one rank into device, host and disk tiers.

    The table is addressed by hashes, so every row is equally likely to be read
    and the split points carry no meaning beyond capacity -- so every tier is
    filled before the next, fastest first: the device up to its measured
    budget, then the pinned host share, and only then the disk tier, which is
    the mapped checkpoint and needs no budget. Without a device budget (no
    cascade) the host share is taken first and the device holds the rest
    unmeasured, as it always has. Rows are never dropped: a remainder with no
    tier left to hold it is an error.
    """

    if total_rows < 0 or row_bytes <= 0:
        raise ValueError("total_rows must be non-negative and row_bytes positive")
    if host_budget_bytes < 0:
        raise ValueError("host budget must be non-negative")
    if vram_budget_bytes is not None and vram_budget_bytes < 0:
        raise ValueError("device budget must be non-negative")
    if vram_budget_bytes is None:
        host_rows = min(total_rows, host_budget_bytes // row_bytes)
        vram_rows = total_rows - host_rows
    else:
        vram_rows = min(total_rows, vram_budget_bytes // row_bytes)
        host_rows = min(total_rows - vram_rows, host_budget_bytes // row_bytes)
    disk_rows = total_rows - host_rows - vram_rows
    if disk_rows and not disk_allowed:
        raise ValueError(
            f"The PLE table does not fit: {disk_rows} rows "
            f"({disk_rows * row_bytes} bytes) remain beyond the device and "
            "host tiers. Raise VLLM_QWEN4EXP_PLE_HOST_GIB, or set "
            "VLLM_QWEN4EXP_PLE_DISK=1 to read the rest from the checkpoint on disk."
        )
    return PLEPlacement(vram_rows=vram_rows, host_rows=host_rows, disk_rows=disk_rows)


@dataclass(frozen=True)
class PLERemotePlacement:
    """Rows of one tensor-parallel rank that the PLE offload worker serves.

    ``tp_start`` and ``tp_end`` are the rank's vocabulary range as the
    checkpoint counts it, without padding. ``local_rows`` are the rows the
    rank keeps itself, counted from its first row (device tier followed by
    the pinned-host tier), so every rank-local id at or beyond it is read
    from the worker's output buffer, which the worker fills from the mapped
    checkpoint.
    """

    tp_start: int
    tp_end: int
    local_rows: int

    def __post_init__(self) -> None:
        if self.tp_start < 0 or self.tp_end < self.tp_start:
            raise ValueError("invalid TP vocabulary range")
        if self.local_rows < 0:
            raise ValueError("local_rows must be non-negative")

    @property
    def remote_rows(self) -> int:
        """Rows of this rank the worker has to serve."""
        return max(0, self.tp_end - self.tp_start - self.local_rows)


@dataclass(frozen=True)
class PLEDiskSegment:
    """Global row range ``[start, end)`` the worker reads from the checkpoint."""

    start: int
    end: int


def plan_ple_disk_segments(
    placements: Sequence[PLERemotePlacement],
) -> list[PLEDiskSegment]:
    """Row ranges the worker reads from the checkpoint, addressed by global id.

    Tensor-parallel ranks own disjoint vocabulary ranges, so the segments are
    disjoint and every row id falls into at most one of them.
    """

    ranges = sorted((placement.tp_start, placement.tp_end) for placement in placements)
    for (_, previous_end), (next_start, _) in pairwise(ranges):
        if next_start < previous_end:
            raise ValueError(f"tensor-parallel vocabulary ranges overlap: {ranges}")
    return [
        PLEDiskSegment(
            start=placement.tp_start + placement.local_rows, end=placement.tp_end
        )
        for placement in placements
        if placement.remote_rows
    ]


def ple_disk_mask(
    ids: torch.Tensor, segments: Sequence[PLEDiskSegment]
) -> torch.Tensor:
    """Which row ids the worker has to read from the mapped checkpoint."""

    mask = torch.zeros_like(ids, dtype=torch.bool)
    for segment in segments:
        mask |= (ids >= segment.start) & (ids < segment.end)
    return mask


def copy_ple_embedding_shard_tiers_(
    tiers: Sequence[tuple[int, torch.Tensor | None]],
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> int:
    """Copy one checkpoint shard into a table split across consecutive tiers.

    Each tier is a row count and the tensor holding those rows, or ``None``
    for rows another process holds. The tiers cover the TP-local range in
    order, so every tier is a plain sub-range of the same vocabulary interval
    and the single-target copy above handles each of them unchanged.
    """

    if tp_start < 0 or tp_end < tp_start:
        raise ValueError("invalid TP vocabulary range")
    if sum(rows for rows, _ in tiers) < tp_end - tp_start:
        raise ValueError("tiers do not cover the requested TP range")
    copied = 0
    tier_start = tp_start
    for rows, destination in tiers:
        # Storage includes vocabulary padding, while checkpoint TP bounds do
        # not. Padding can lie in any tier and must not reject a valid
        # checkpoint.
        tier_end = min(tp_end, tier_start + rows)
        if destination is not None and tier_start < tier_end:
            copied += copy_ple_embedding_shard_(
                destination,
                loaded_weight,
                checkpoint_start=checkpoint_start,
                tp_start=tier_start,
                tp_end=tier_end,
            )
        tier_start = tier_end
    return copied


def kv_cache_bytes_for_max_model_len(vllm_config: "VllmConfig") -> int:
    """Bytes this rank's KV cache needs to serve ``max_model_len``.

    Sums what every KV-owning layer of this pipeline stage declares. The specs
    are the engine's own source of truth for that number -- the same ones the
    allocator consults later. This is a per-layer estimate: hybrid cache
    grouping and allocator padding can require additional capacity.
    """

    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)  # type: ignore[type-abstract]
    total = 0
    for layer in layers.values():
        spec = layer.get_kv_cache_spec(vllm_config)
        if spec is not None:
            total += spec.max_memory_usage_bytes(vllm_config)
    return total


def auto_ple_host_budget_bytes(
    *,
    table_bytes: int,
    device_total_bytes: int,
    device_allocated_bytes: int,
    gpu_memory_utilization: float,
    kv_cache_bytes: int,
    reserve_bytes: int,
) -> int:
    """Host bytes needed so the requested context still fits beside the table.

    The table stays in device memory and only what the context claims is
    spilled -- never more. Spilling beyond that would be wasted: with pipeline
    parallelism the weakest stage caps the block count for all of them, so
    surplus room on this rank buys nothing while pinned host memory is the
    scarcer resource.
    """

    if table_bytes < 0 or device_total_bytes <= 0:
        raise ValueError("table and device sizes must be non-negative")
    usable = int(device_total_bytes * gpu_memory_utilization)
    room_for_table = usable - device_allocated_bytes - kv_cache_bytes - reserve_bytes
    if room_for_table >= table_bytes:
        return 0
    return table_bytes - max(0, room_for_table)


def _meminfo_bytes(key: str) -> int | None:
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def available_host_bytes() -> int | None:
    """Host memory the kernel currently reports as available, or None."""

    return _meminfo_bytes("MemAvailable")


def total_host_bytes() -> int | None:
    """Physical host memory as the kernel reports it, or None."""

    return _meminfo_bytes("MemTotal")


def cap_host_budget_bytes(
    *,
    budget_bytes: int,
    available_bytes: int,
    reserve_bytes: int,
    ranks_sharing_host: int,
) -> int:
    """Bound a rank's pinned-host budget by its fair share of the host.

    Every tensor-parallel rank of the stage that owns the table pins its own
    share, and all of them draw on the same host memory. Reading MemAvailable
    per rank therefore double-books it: on a 30 GB host with 20 GB available,
    two ranks each saw room for 7 GiB, pinned 14 GiB together and pushed the
    engine processes, the checkpoint loading and everything else into swap
    (2026-09-06). The share is what remains after the reserve, divided by the
    ranks; a budget above it is cut to the share. What no longer spills stays
    on the device, and if the context then does not fit, the KV allocator
    reports the reachable max_model_len -- host memory is the hard limit,
    context the negotiable one.
    """

    if ranks_sharing_host <= 0:
        raise ValueError("ranks_sharing_host must be positive")
    share = max(0, available_bytes - reserve_bytes) // ranks_sharing_host
    return min(budget_bytes, share)


def env_gib_bytes(name: str) -> int | None:
    """Bytes of a GiB-valued PLE variable, or None when it is unset."""

    value_gib = getattr(envs, name)
    if value_gib is None:
        return None
    if not math.isfinite(value_gib) or value_gib < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value_gib}")
    return int(value_gib * 1024**3)


def ple_host_budget_bytes() -> int | None:
    """Configured host bytes per rank for the PLE table, or None to derive them."""

    return env_gib_bytes("VLLM_QWEN4EXP_PLE_HOST_GIB")


def ple_host_reserve_bytes(host_total_bytes: int) -> int:
    """Host memory the placement leaves to everything else.

    The engine processes, the checkpoint loading and other tenants of the
    host need room that no single rank can measure; on a 30 GB host the
    default keeps 7.5 GiB.
    """

    reserve = env_gib_bytes("VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB")
    if reserve is not None:
        return reserve
    return host_total_bytes // 4


def ple_vram_reserve_bytes(device_total_bytes: int) -> int:
    """Device memory the automatic placement keeps free.

    It covers the activation peak and the graph pool, which the engine only
    measures after the weights are placed -- so they cannot be read here. On
    a 48 GB card the gap between (weights + KV) and the utilization budget
    stayed between 1.8 and 2.3 GiB; the default leaves a slightly wider
    margin. Overshooting costs host memory, undershooting makes the KV
    allocator fail late.
    """

    reserve = env_gib_bytes("VLLM_QWEN4EXP_PLE_VRAM_RESERVE_GIB")
    if reserve is not None:
        return reserve
    return min(int(device_total_bytes * 0.08), 4 * 1024**3)


def ple_cascade_configured() -> bool:
    """Whether the overflow cascade is on: rows beyond the device and host
    tiers are read from the mapped checkpoint by the PLE offload worker."""

    return bool(envs.VLLM_QWEN4EXP_PLE_DISK)


def check_ple_host_share(text_config: Any, ranks_sharing_host: int) -> None:
    """Refuse a configured pinned-host share the host cannot hold.

    Runs once before any rank starts. The ranks place their tables at the same
    time, so a rank that reads the host memory while a sibling already pins its
    share would count that share twice and refuse a configuration that fits.
    The reserve covers what the loading claims afterwards.
    """

    if not getattr(text_config, "ple_layer_ids", None):
        return
    budget = ple_host_budget_bytes()
    available = available_host_bytes()
    total = total_host_bytes()
    if not budget or available is None or total is None:
        return
    reserve = ple_host_reserve_bytes(total)
    share = cap_host_budget_bytes(
        budget_bytes=budget,
        available_bytes=available,
        reserve_bytes=reserve,
        ranks_sharing_host=ranks_sharing_host,
    )
    if share < budget:
        raise ValueError(
            f"VLLM_QWEN4EXP_PLE_HOST_GIB asks for {format_gib(budget)} GiB of "
            f"pinned host memory per rank, but each of the {ranks_sharing_host} "
            f"tensor-parallel ranks may pin at most {format_gib(share)} GiB: "
            f"{format_gib(available)} GiB available, {format_gib(reserve)} GiB "
            "kept in reserve. Lower VLLM_QWEN4EXP_PLE_HOST_GIB or "
            "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB."
        )
    logger.info(
        "Qwen4Exp PLE host share %s GiB per rank fits: %d ranks, %s GiB "
        "available, %s GiB kept in reserve.",
        format_gib(budget),
        ranks_sharing_host,
        format_gib(available),
        format_gib(reserve),
    )
