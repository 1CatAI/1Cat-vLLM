# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen4Exp PLE helpers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.utils import get_layers_outside_first_pp_rank

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# Checkpoint tensors of the PLE table: ``...ngram_embedding.shard_<i>.weight``.
PLE_SHARD_PREFIX = "ngram_embedding.shard_"


def is_ple_checkpoint_shard(name: str) -> bool:
    """Whether a checkpoint tensor is a shard of the PLE embedding table.

    The ranks read only their rows of these shards and the disk tier serves
    the rest from the mapped checkpoint, so they must stay memory-mapped
    under direct I/O (see map_checkpoint_weight of the models).
    """
    return f".{PLE_SHARD_PREFIX}" in name or name.startswith(PLE_SHARD_PREFIX)


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
        target.copy_(source.to(device=target.device, dtype=target.dtype))
    return overlap.row_count


@dataclass(frozen=True)
class PLEPlacement:
    """How many PLE table rows live in device memory and how many on the host."""

    vram_rows: int
    host_rows: int

    @property
    def total_rows(self) -> int:
        return self.vram_rows + self.host_rows


def plan_ple_placement(
    *,
    total_rows: int,
    row_bytes: int,
    host_budget_bytes: int,
) -> PLEPlacement:
    """Place as many PLE rows as possible in device memory.

    The table is addressed by hashes, so every row is equally likely to be read
    and the split point carries no meaning beyond capacity: whatever does not
    fit in device memory goes to the host. Rows are never dropped, so the
    caller must provide a host budget large enough for the remainder.
    """

    if total_rows < 0 or row_bytes <= 0:
        raise ValueError("total_rows must be non-negative and row_bytes positive")
    if host_budget_bytes < 0:
        raise ValueError("host budget must be non-negative")
    host_rows = min(total_rows, host_budget_bytes // row_bytes)
    return PLEPlacement(vram_rows=total_rows - host_rows, host_rows=host_rows)


def copy_ple_embedding_shard_split_(
    vram_table: torch.Tensor,
    host_table: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> int:
    """Copy one checkpoint shard into a table split across device and host.

    The split point is a row index in the TP-local range, so both halves are
    plain sub-ranges of the same vocabulary interval and the single-target
    copy above handles each of them unchanged.
    """

    if tp_start < 0 or tp_end < tp_start:
        raise ValueError("invalid TP vocabulary range")
    if vram_table.shape[0] + host_table.shape[0] < tp_end - tp_start:
        raise ValueError("split tables do not cover the requested TP range")
    # Storage includes vocabulary padding, while checkpoint TP bounds do not.
    # Padding can lie in either half and must not reject a valid checkpoint.
    boundary = min(tp_end, tp_start + vram_table.shape[0])
    copied = 0
    if vram_table.shape[0]:
        copied += copy_ple_embedding_shard_(
            vram_table,
            loaded_weight,
            checkpoint_start=checkpoint_start,
            tp_start=tp_start,
            tp_end=boundary,
        )
    if boundary < tp_end:
        copied += copy_ple_embedding_shard_(
            host_table,
            loaded_weight,
            checkpoint_start=checkpoint_start,
            tp_start=boundary,
            tp_end=tp_end,
        )
    return copied


def kv_cache_bytes_for_max_model_len(vllm_config: "VllmConfig") -> int:
    """Bytes this rank's KV cache needs to serve ``max_model_len``.

    Use the allocator's grouped layout so hybrid padding and shared pools are
    included before deciding how much of the PLE table can stay on device.

    The PLE table is placed while the model loads, but the worker settles the
    hybrid block size only after load_model(). Settle it here first: with the
    provisional block size the specs describe pages the allocator never uses,
    and CSA+linear models fail the cache geometry check outright. The call is
    idempotent, so the worker's later one changes nothing.
    """

    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.platforms import current_platform
    from vllm.v1.core.kv_cache_utils import (
        _max_memory_usage_bytes_from_groups,
        get_kv_cache_groups,
    )

    current_platform.update_block_size_for_backend(vllm_config)
    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)  # type: ignore[type-abstract]
    specs = {}
    for name, layer in layers.items():
        spec = layer.get_kv_cache_spec(vllm_config)
        if spec is not None:
            specs[name] = spec
    groups = get_kv_cache_groups(vllm_config, specs)
    return _max_memory_usage_bytes_from_groups(vllm_config, groups)


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
