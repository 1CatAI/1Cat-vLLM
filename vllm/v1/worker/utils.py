# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product as iprod
from typing import Any, cast

import torch

from vllm.config import CacheConfig, VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import PAGED_MQA_PAGE_SIZES
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImplBase,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.block_table import get_block_table_width

logger = init_logger(__name__)


def configure_attention_workspace(
    vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroupSpec]
) -> None:
    """Bind backend workspace bounds from resolved physical cache pages."""
    layers = get_layers_from_vllm_config(
        vllm_config, cast(type[Any], AttentionLayerBase)
    )
    for group in kv_cache_groups:
        for name in group.layer_names:
            impl = getattr(layers.get(name), "impl", None)
            if not isinstance(impl, AttentionImplBase):
                continue
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            impl.configure_kv_cache_workspace(vllm_config, spec)


# Shared host pools stay mapped and registered for the worker's lifetime.
_SHARED_HOST_POOLS: list[Any] = []


def _shared_replica_host_pool(pool_size: int, device: torch.device):
    """One host KV pool per replicated KV head instead of one per TP rank.

    With fewer KV heads than TP ranks, consecutive ranks hold the same KV head
    and write identical history. Those ranks map one shared-memory pool and
    register it with CUDA, which divides host KV memory by the replication.
    Returns None (private pool) when the layout does not replicate heads or
    the ranks cannot share host memory.
    """
    import mmap
    import os
    import uuid

    import torch.distributed as dist

    from vllm.config import get_current_vllm_config_or_none
    from vllm.distributed.parallel_state import get_tp_group

    config = get_current_vllm_config_or_none()
    if config is None or not getattr(
        config.kernel_config, "qsa_host_kv_share_replicas", True
    ):
        return None
    parallel = config.parallel_config
    tp = parallel.tensor_parallel_size
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
    ):
        return None
    try:
        kv_heads = config.model_config.get_total_num_kv_heads()
    except Exception:  # noqa: BLE001 - models without a KV head count
        return None
    if tp <= 1 or kv_heads >= tp or tp % kv_heads:
        return None
    group = get_tp_group()
    replicas = tp // kv_heads
    rank = group.rank_in_group
    leader = rank % replicas == 0
    tokens = [uuid.uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(tokens, src=group.ranks[0], group=group.cpu_group)
    path = f"/dev/shm/vllm-host-kv-{tokens[0]}-{rank // replicas}"
    created = False
    try:
        if leader:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            created = True
            try:
                os.ftruncate(fd, pool_size)
            finally:
                os.close(fd)
        ok = True
    except OSError:
        ok = False
    try:
        flags = [None] * group.world_size
        dist.all_gather_object(flags, ok, group=group.cpu_group)
        if not all(flags):
            return None
        fd = os.open(path, os.O_RDWR)
        try:
            shared = mmap.mmap(fd, pool_size, mmap.MAP_SHARED)
        finally:
            os.close(fd)
        dist.barrier(group=group.cpu_group)
    finally:
        if leader and created:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
    from vllm.model_executor.layers.quantization.gguf_expert_cache import _native

    host = torch.frombuffer(shared, dtype=torch.int8)
    native = _native()
    native.host_register(host.data_ptr(), pool_size)
    view = native.mapped_view(host, device.index)
    _SHARED_HOST_POOLS.append((shared, host, view))
    logger.info(
        "Host KV pool shared by %d TP ranks per KV head (%s per pool).",
        replicas,
        format_gib(pool_size),
    )
    return view


def allocate_host_kv_cache_pool(
    kv_cache_config: KVCacheConfig, device: torch.device
) -> dict[str, torch.Tensor]:
    """Pack host cache banks into one pinned allocation per worker.

    The caching pinned allocator rounds each allocation to a power of two, so
    the pool is one exact cudaHostAlloc instead of per-layer pinned tensors.
    Aligned views retain the mapped pool owner and keep stable graph pointers.
    """
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    banks = [t for t in kv_cache_config.kv_cache_tensors if t.host_backed]
    if not banks:
        return {}
    aligned_sizes = [(t.size + 255) // 256 * 256 for t in banks]
    pool_size = sum(aligned_sizes)
    logger.info(
        "Host KV pinned pool: payload=%d allocation=%d banks=%d bytes",
        sum(t.size for t in banks),
        pool_size,
        len(banks),
    )
    pool = _shared_replica_host_pool(pool_size, device)
    if pool is None:
        with torch.accelerator.device_index(device.index):
            # A pageable source makes the view helper cudaHostAlloc exactly
            # pool_size bytes. The caching pinned allocator would round this
            # one allocation up to a power of two (2.3 GB -> 4 GiB per worker).
            # The untouched source pages are never resident; zero the pool on
            # device.
            host = torch.empty(pool_size, dtype=torch.int8, device="cpu")
            pool = get_accelerator_view_from_cpu_tensor(host)
            del host
            pool.zero_()
    tensors: dict[str, torch.Tensor] = {}
    offset = 0
    for bank, aligned_size in zip(banks, aligned_sizes):
        view = pool.narrow(0, offset, bank.size)
        for name in bank.shared_by:
            tensors[name] = view
        offset += aligned_size
    return tensors


def _resolve_zeroer_kernel_layout(
    spec: AttentionSpec,
    group_kernel_block_size: int,
) -> tuple[int, int]:
    """Return the physical kernel block size and virtual-block ratio.

    Compressed attention specs store fewer rows than their logical scheduler
    block size. Their cache view is shaped with ``storage_block_size`` as one
    physical page, so zeroing must mirror that layout instead of multiplying
    the page span by the logical compression ratio a second time.
    """
    storage_block_size = spec.storage_block_size
    kernel_block_size = (
        storage_block_size
        if storage_block_size != spec.block_size
        else group_kernel_block_size
    )
    if storage_block_size % kernel_block_size:
        raise ValueError(
            "KV storage block size must be divisible by its kernel block size: "
            f"{storage_block_size} vs {kernel_block_size}."
        )
    return kernel_block_size, storage_block_size // kernel_block_size


def compressed_kernel_block_size(spec: AttentionSpec) -> int:
    storage = spec.storage_block_size
    max_page = max(PAGED_MQA_PAGE_SIZES)
    min_page = min(PAGED_MQA_PAGE_SIZES)
    if storage <= max_page:
        return storage
    return max_page if storage % max_page == 0 else min_page


def _infer_segment_block_strides(
    seg_addrs: list[int],
    page_size_el: int,
) -> list[int]:
    """Infer logical block strides from block-major segment address runs.

    Dense segments advance by one page. An interleaved pool exposes one segment
    base per cell, exactly ``page_size_el * 4`` bytes apart, and advances to the
    next logical block after every segment in that address run. Treat separate
    runs independently so multiple pools do not disable one another.
    """
    if page_size_el <= 0:
        raise ValueError(f"page_size_el must be positive, got {page_size_el}.")
    if len(set(seg_addrs)) != len(seg_addrs):
        raise ValueError("Segment addresses must be unique.")

    block_strides = [page_size_el] * len(seg_addrs)
    if len(seg_addrs) < 2:
        return block_strides

    cell_bytes = page_size_el * 4
    ordered = sorted((addr, index) for index, addr in enumerate(seg_addrs))
    run_start = 0
    for run_end in range(1, len(ordered) + 1):
        if (
            run_end < len(ordered)
            and ordered[run_end][0] - ordered[run_end - 1][0] == cell_bytes
        ):
            continue
        run = ordered[run_start:run_end]
        if len(run) > 1:
            run_stride = page_size_el * len(run)
            for _, original_index in run:
                block_strides[original_index] = run_stride
        run_start = run_end
    return block_strides


@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    seg_num_blocks_ptr,
    block_ids_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  For backends
    where blocks are outermost (block_dim=0) there is one segment per
    buffer.  For backends where K/V is outermost (block_dim=1) there are
    two segments per buffer (one for K, one for V).

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different CUDA allocations.
    seg_block_strides_ptr holds the per-segment block-to-block stride in
    elements, kept separate from the PAGE_SIZE_EL span each block zeros (they
    differ for interleaved layouts, see init_meta).

    Programs are mapped as (block_index, seg_index, chunk_index). Segment page
    sizes may differ across hybrid cache groups.

    Block IDs carry no cache-group identity. A segment whose tensor holds
    fewer blocks than the pool (a bounded recurrent-state range) never owns
    IDs past its own capacity, so those are skipped rather than written past
    the end of the allocation.
    """
    block_index = tl.program_id(0)
    seg_index = tl.program_id(1)
    chunk_index = tl.program_id(2)
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    chunk_offset = chunk_index.to(tl.int64) * BLOCK_SIZE
    if chunk_offset >= page_size_el:
        return
    block_id = tl.load(block_ids_ptr + block_index)
    if block_id < 0 or block_id >= tl.load(seg_num_blocks_ptr + seg_index):
        return
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    block_stride_el = tl.load(seg_block_strides_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    block_offset = block_id.to(tl.int64) * block_stride_el.to(tl.int64)
    cols = chunk_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    tl.store(
        ptr + block_offset + cols,
        tl.zeros([BLOCK_SIZE], dtype=tl.int32),
        mask=cols < page_size_el,
    )


@triton.jit
def _copy_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    seg_num_blocks_ptr,
    seg_group_masks_ptr,
    src_ids_ptr,
    dst_ids_ptr,
    pair_groups_ptr,
    HAS_PAIR_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy whole KV cache blocks across matching segments."""
    pair_index = tl.program_id(0)
    seg_index = tl.program_id(1)
    if HAS_PAIR_GROUPS:
        pair_group = tl.load(pair_groups_ptr + pair_index)
        seg_mask = tl.load(seg_group_masks_ptr + seg_index)
        if (
            pair_group >= 0
            and pair_group < 62
            and seg_mask >= 0
            and ((seg_mask >> pair_group) & 1) == 0
        ):
            return
    chunk_index = tl.program_id(2)
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    chunk_offset = chunk_index.to(tl.int64) * BLOCK_SIZE
    if chunk_offset >= page_size_el:
        return
    src_id = tl.load(src_ids_ptr + pair_index)
    dst_id = tl.load(dst_ids_ptr + pair_index)
    num_blocks = tl.load(seg_num_blocks_ptr + seg_index)
    if src_id < 0 or src_id >= num_blocks or dst_id < 0 or dst_id >= num_blocks:
        return
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    block_stride_el = tl.load(seg_block_strides_ptr + seg_index).to(tl.int64)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    cols = chunk_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = cols < page_size_el
    values = tl.load(ptr + src_id.to(tl.int64) * block_stride_el + cols, mask=mask)
    tl.store(ptr + dst_id.to(tl.int64) * block_stride_el + cols, values, mask=mask)


@triton.jit
def _invalidate_hot_kv_blocks_kernel(
    owner_page_slots_ptrs,
    owner_tags_ptrs,
    owner_stamps_ptrs,
    owner_pages_per_block_ptr,
    owner_num_blocks_ptr,
    owner_group_masks_ptr,
    dst_ids_ptr,
    pair_groups_ptr,
    HAS_PAIR_GROUPS: tl.constexpr,
    PAGE_CHUNK: tl.constexpr,
):
    """Invalidate HostQSAKV device hot-page mappings for copied target blocks."""
    pair_index = tl.program_id(0)
    owner_index = tl.program_id(1)
    if HAS_PAIR_GROUPS:
        pair_group = tl.load(pair_groups_ptr + pair_index)
        owner_mask = tl.load(owner_group_masks_ptr + owner_index)
        if (
            pair_group >= 0
            and pair_group < 62
            and owner_mask >= 0
            and ((owner_mask >> pair_group) & 1) == 0
        ):
            return
    dst_id = tl.load(dst_ids_ptr + pair_index)
    num_blocks = tl.load(owner_num_blocks_ptr + owner_index)
    if dst_id < 0 or dst_id >= num_blocks:
        return
    chunk_index = tl.program_id(2)
    pages_per_block = tl.load(owner_pages_per_block_ptr + owner_index)
    if chunk_index * PAGE_CHUNK >= pages_per_block:
        return
    p_offs = chunk_index * PAGE_CHUNK + tl.arange(0, PAGE_CHUNK)
    p_mask = p_offs < pages_per_block
    pages_i64 = dst_id.to(tl.int64) * pages_per_block.to(tl.int64) + p_offs.to(tl.int64)
    pages_i32 = pages_i64.to(tl.int32)
    page_slots = tl.cast(
        tl.load(owner_page_slots_ptrs + owner_index),
        tl.pointer_type(tl.int32),
    )
    tags = tl.cast(
        tl.load(owner_tags_ptrs + owner_index),
        tl.pointer_type(tl.int32),
    )
    stamps = tl.cast(
        tl.load(owner_stamps_ptrs + owner_index),
        tl.pointer_type(tl.int32),
    )
    registered = tl.load(page_slots + pages_i64, mask=p_mask, other=-1)
    tl.store(page_slots + pages_i64, -1, mask=p_mask)
    slots = tl.maximum(registered, 0).to(tl.int64)
    tag_vals = tl.load(tags + slots, mask=p_mask & (registered >= 0), other=-1)
    owned = p_mask & (registered >= 0) & (tag_vals == pages_i32)
    tl.store(tags + slots, -1, mask=owned)
    tl.store(stamps + slots, 0, mask=owned)


def _group_bitmask(gid: int) -> int:
    return (1 << gid) if 0 <= gid < 62 else -1


def _merge_group_bitmask(cur: int, gid: int) -> int:
    bit = _group_bitmask(gid)
    if cur < 0 or bit < 0:
        return -1
    return cur | bit


class KVBlockZeroer:
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Call :meth:`init_meta` once after KV caches are allocated to precompute
    segment addresses, then call :meth:`zero_block_ids` each step to zero
    newly-allocated blocks.
    """

    def __init__(self, device: torch.device, pin_memory: bool):
        self.device = device
        self.pin_memory = pin_memory
        self._meta: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]
            | None
        ) = None
        self._copy_group_masks: torch.Tensor | None = None
        self._host_copy_meta: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
                int,
                int,
            ]
            | None
        ) = None
        self._hot_inv_meta: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
                int,
                int,
            ]
            | None
        ) = None
        self._host_union_group_mask: int = 0

    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
        host_backed_layers: set[str] | None = None,
    ) -> None:
        """One-time precomputation for zero_block_ids and copy_block_ids.

        Builds absolute-address table for the Triton zeroing/copying kernels.
        Each entry is the absolute byte address of a segment start on the
        GPU (or mapped pinned host pool), so segments in different allocations
        work correctly.

        Host-backed layers (pinned host history mapped into the device address
        space) are not zeroed, since readers only touch slots below each
        request's length, avoiding PCIe zeroing traffic. For partial prefix
        cache block copies, host-backed segments and HostQSAKV per-token scales
        are copied only when the owning cache group requests a copy, and any
        stale hot-page entries for the destination block are invalidated.
        """
        host_backed_layers = host_backed_layers or set()
        seen_ptrs: dict[int, int] = {}
        seg_addrs: list[int] = []
        seg_page_sizes: list[int] = []
        # Logical blocks each segment's tensor actually holds.
        seg_num_blocks: list[int] = []
        seg_group_masks: list[int] = []

        host_seen_ptrs: dict[int, int] = {}
        host_seg_addrs: list[int] = []
        host_seg_strides: list[int] = []
        host_seg_page_sizes: list[int] = []
        host_seg_num_blocks: list[int] = []
        host_seg_group_masks: list[int] = []
        host_union_mask = 0

        hot_seen_ptrs: dict[int, int] = {}
        hot_page_slots_addrs: list[int] = []
        hot_tags_addrs: list[int] = []
        hot_stamps_addrs: list[int] = []
        hot_pages_per_block: list[int] = []
        hot_num_blocks: list[int] = []
        hot_group_masks: list[int] = []

        def add_segment(
            address: int, page_size_el: int, num_blocks: int, gid: int
        ) -> None:
            if (index := seen_ptrs.get(address)) is not None:
                seg_page_sizes[index] = max(seg_page_sizes[index], page_size_el)
                seg_num_blocks[index] = min(seg_num_blocks[index], num_blocks)
                seg_group_masks[index] = _merge_group_bitmask(
                    seg_group_masks[index], gid
                )
                return
            seen_ptrs[address] = len(seg_addrs)
            seg_addrs.append(address)
            seg_page_sizes.append(page_size_el)
            seg_num_blocks.append(num_blocks)
            seg_group_masks.append(_group_bitmask(gid))

        def add_host_segment(
            address: int,
            stride_el: int,
            page_size_el: int,
            num_blocks: int,
            gid: int,
        ) -> None:
            nonlocal host_union_mask
            host_union_mask = _merge_group_bitmask(host_union_mask, gid)
            if (index := host_seen_ptrs.get(address)) is not None:
                host_seg_page_sizes[index] = max(
                    host_seg_page_sizes[index], page_size_el
                )
                host_seg_strides[index] = max(host_seg_strides[index], stride_el)
                host_seg_num_blocks[index] = min(host_seg_num_blocks[index], num_blocks)
                host_seg_group_masks[index] = _merge_group_bitmask(
                    host_seg_group_masks[index], gid
                )
                return
            host_seen_ptrs[address] = len(host_seg_addrs)
            host_seg_addrs.append(address)
            host_seg_strides.append(stride_el)
            host_seg_page_sizes.append(page_size_el)
            host_seg_num_blocks.append(num_blocks)
            host_seg_group_masks.append(_group_bitmask(gid))

        def add_hot_owner(
            page_slots_addr: int,
            tags_addr: int,
            stamps_addr: int,
            pages_per_block: int,
            num_blocks: int,
            gid: int,
        ) -> None:
            if pages_per_block <= 0 or num_blocks <= 0:
                return
            if (index := hot_seen_ptrs.get(page_slots_addr)) is not None:
                hot_group_masks[index] = _merge_group_bitmask(
                    hot_group_masks[index], gid
                )
                return
            hot_seen_ptrs[page_slots_addr] = len(hot_page_slots_addrs)
            hot_page_slots_addrs.append(page_slots_addr)
            hot_tags_addrs.append(tags_addr)
            hot_stamps_addrs.append(stamps_addr)
            hot_pages_per_block.append(pages_per_block)
            hot_num_blocks.append(num_blocks)
            hot_group_masks.append(_group_bitmask(gid))

        def iter_attn_tensor_segments(
            spec: AttentionSpec,
            backend: type[AttentionBackend],
            kv: torch.Tensor,
            group_kbs: int,
        ) -> list[tuple[int, int, int, int]]:
            kernel_bs, ratio = _resolve_zeroer_kernel_layout(spec, group_kbs)
            block_dim = backend.get_kv_cache_block_dim(
                kernel_bs,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=spec.layout_cache_dtype(cache_dtype),
            )
            el = kv.element_size()
            page_offset_el = kv.storage_offset() % kv.stride(block_dim)
            dp = kv.data_ptr() - page_offset_el * el

            cur_bytes = kv.stride(block_dim) * el
            assert cur_bytes % 4 == 0
            kernel_block_el = cur_bytes // 4
            cur_page_el = kernel_block_el * ratio

            block_stride_bytes = cur_bytes
            outer_dims = [
                d for d in range(block_dim) if kv.stride(d) * el > block_stride_bytes
            ]
            outer_strides = [kv.stride(d) * el for d in outer_dims]
            num_blocks = kv.shape[block_dim] // ratio
            out: list[tuple[int, int, int, int]] = []
            for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                out.append((dp + off_bytes, cur_page_el, cur_page_el, num_blocks))
            return out

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            gid = group.kv_cache_group_id
            if gid >= len(kernel_block_sizes):
                continue

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                layer_obj = static_forward_context[layer_name]
                if layer_name in host_backed_layers:
                    host_kv = getattr(layer_obj, "host_kv", None)
                    if host_kv is not None:
                        host_page_size = int(getattr(host_kv, "page_size", 0) or 0)
                        host_ratio = 1
                        if (
                            isinstance(spec, AttentionSpec)
                            and host_page_size > 0
                            and spec.storage_block_size > host_page_size
                        ):
                            assert spec.storage_block_size % host_page_size == 0
                            host_ratio = spec.storage_block_size // host_page_size
                        assert int(host_kv.blocks) % host_ratio == 0
                        host_num_blocks = int(host_kv.blocks) // host_ratio
                        hist = host_kv.history
                        if isinstance(hist, torch.Tensor) and hist.is_cuda:
                            hist_bytes = (
                                hist.stride(0) * hist.element_size() * host_ratio
                            )
                            assert hist_bytes % 4 == 0
                            hist_el = hist_bytes // 4
                            add_host_segment(
                                hist.data_ptr(),
                                hist_el,
                                hist_el,
                                host_num_blocks,
                                gid,
                            )
                        if (
                            getattr(host_kv, "fp8", False)
                            and isinstance(host_kv.scales, torch.Tensor)
                            and host_kv.scales.is_cuda
                        ):
                            scales_el = host_page_size * 2 * host_ratio
                            add_host_segment(
                                host_kv.scales.data_ptr(),
                                scales_el,
                                scales_el,
                                host_num_blocks,
                                gid,
                            )
                        if (
                            isinstance(
                                getattr(host_kv, "page_slots", None), torch.Tensor
                            )
                            and host_kv.page_slots.is_cuda
                            and int(host_kv.page_slots.numel()) > 0
                        ):
                            add_hot_owner(
                                host_kv.page_slots.data_ptr(),
                                host_kv.tags.data_ptr(),
                                host_kv.stamps.data_ptr(),
                                (host_page_size // 4) * host_ratio,
                                host_num_blocks,
                                gid,
                            )
                    elif isinstance(spec, AttentionSpec) and not isinstance(
                        spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
                    ):
                        kv = layer_obj.kv_cache
                        if isinstance(kv, torch.Tensor) and kv.is_cuda:
                            for (
                                addr,
                                stride_el,
                                page_el,
                                n_blk,
                            ) in iter_attn_tensor_segments(
                                spec,
                                group.backend,
                                kv,
                                kernel_block_sizes[gid],
                            ):
                                add_host_segment(addr, stride_el, page_el, n_blk, gid)
                    continue

                kv = layer_obj.kv_cache
                if isinstance(spec, AttentionSpec) and not isinstance(
                    spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
                ):
                    if not isinstance(kv, torch.Tensor):
                        continue
                    for addr, _, cur_page_el, num_blocks in iter_attn_tensor_segments(
                        spec,
                        group.backend,
                        kv,
                        kernel_block_sizes[gid],
                    ):
                        add_segment(addr, cur_page_el, num_blocks, gid)
                elif isinstance(spec, MambaSpec):
                    if not isinstance(kv, (list, tuple)) or not kv:
                        continue
                    first_state = kv[0]
                    if not isinstance(first_state, torch.Tensor):
                        continue
                    dp = first_state.data_ptr()

                    page_bytes = spec.page_size_bytes
                    assert page_bytes % 4 == 0
                    cur_page_el = page_bytes // 4
                    # A bounded state pool exposes fewer pages than the
                    # scheduler's block-ID range.
                    add_segment(dp, cur_page_el, first_state.shape[0], gid)

        self._host_union_group_mask = host_union_mask
        if host_seg_addrs:
            h_max_el = max(host_seg_page_sizes)
            h_blk = min(1 << (h_max_el - 1).bit_length(), 1024)
            self._host_copy_meta = (
                torch.tensor(host_seg_addrs, dtype=torch.uint64, device=self.device),
                torch.tensor(host_seg_strides, dtype=torch.int64, device=self.device),
                torch.tensor(
                    host_seg_page_sizes, dtype=torch.int64, device=self.device
                ),
                torch.tensor(
                    host_seg_num_blocks, dtype=torch.int64, device=self.device
                ),
                torch.tensor(
                    host_seg_group_masks, dtype=torch.int64, device=self.device
                ),
                (h_max_el + h_blk - 1) // h_blk,
                h_blk,
                len(host_seg_addrs),
            )
        else:
            self._host_copy_meta = None

        if hot_page_slots_addrs:
            max_pages = max(hot_pages_per_block)
            page_chunk = min(1 << max(0, max_pages - 1).bit_length(), 512)
            self._hot_inv_meta = (
                torch.tensor(
                    hot_page_slots_addrs, dtype=torch.uint64, device=self.device
                ),
                torch.tensor(hot_tags_addrs, dtype=torch.uint64, device=self.device),
                torch.tensor(hot_stamps_addrs, dtype=torch.uint64, device=self.device),
                torch.tensor(
                    hot_pages_per_block, dtype=torch.int64, device=self.device
                ),
                torch.tensor(hot_num_blocks, dtype=torch.int64, device=self.device),
                torch.tensor(hot_group_masks, dtype=torch.int64, device=self.device),
                (max_pages + page_chunk - 1) // page_chunk,
                page_chunk,
                len(hot_page_slots_addrs),
            )
        else:
            self._hot_inv_meta = None

        logger.info(
            "KV block zeroer: %d segments, %.1f MiB per block (skipped %d "
            "host-backed layers; %d host copy segments, %d hot-cache owners)",
            len(seg_addrs),
            sum(seg_page_sizes) * 4 / 2**20,
            len(host_backed_layers),
            len(host_seg_addrs),
            len(hot_page_slots_addrs),
        )
        if not seg_addrs:
            self._meta = None
            self._copy_group_masks = None
            return

        max_page_size_el = max(seg_page_sizes)
        blk_size = min(1 << (max_page_size_el - 1).bit_length(), 1024)

        # Dense layouts space blocks page_size_el apart. Block-major
        # interleaved pools expose segment starts one cell apart and advance by
        # the number of segments in that contiguous address run. Infer each
        # run separately because a process may own more than one KV pool.
        n_segs = len(seg_addrs)
        block_strides = [0] * n_segs
        for page_size_el in set(seg_page_sizes):
            indices = [
                index
                for index, size in enumerate(seg_page_sizes)
                if size == page_size_el
            ]
            inferred = _infer_segment_block_strides(
                [seg_addrs[index] for index in indices], page_size_el
            )
            for index, stride in zip(indices, inferred):
                block_strides[index] = stride
        seg_block_strides = torch.tensor(
            block_strides,
            dtype=torch.int64,
            device=self.device,
        )

        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            seg_block_strides,
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            torch.tensor(seg_num_blocks, dtype=torch.int64, device=self.device),
            (max_page_size_el + blk_size - 1) // blk_size,
            blk_size,
            n_segs,
        )
        self._copy_group_masks = torch.tensor(
            seg_group_masks, dtype=torch.int64, device=self.device
        )

    def copy_block_ids(
        self,
        pairs: list[tuple[int, int]],
        group_ids: list[int] | None = None,
    ) -> None:
        """Copy whole KV cache blocks (source, destination) across matching groups."""
        if not pairs:
            return
        if self._meta is None and self._host_copy_meta is None:
            raise RuntimeError("KV block copies need initialized block metadata")
        has_groups = group_ids is not None and len(group_ids) == len(pairs)
        src = async_tensor_h2d(
            [p[0] for p in pairs],
            dtype=torch.int64,
            device=self.device,
            pin_memory=self.pin_memory,
        )
        dst = async_tensor_h2d(
            [p[1] for p in pairs],
            dtype=torch.int64,
            device=self.device,
            pin_memory=self.pin_memory,
        )
        grp = (
            async_tensor_h2d(
                group_ids,
                dtype=torch.int64,
                device=self.device,
                pin_memory=self.pin_memory,
            )
            if has_groups and group_ids is not None
            else src
        )
        if self._meta is not None:
            (
                seg_addrs,
                seg_block_strides,
                seg_page_sizes,
                seg_num_blocks,
                max_chunks,
                blk_size,
                n_segs,
            ) = self._meta
            seg_group_masks = (
                self._copy_group_masks
                if self._copy_group_masks is not None
                else seg_num_blocks
            )
            _copy_kv_blocks_kernel[(len(pairs), n_segs, max_chunks)](
                seg_addrs,
                seg_block_strides,
                seg_page_sizes,
                seg_num_blocks,
                seg_group_masks,
                src,
                dst,
                grp,
                HAS_PAIR_GROUPS=has_groups,
                BLOCK_SIZE=blk_size,
            )

        if self._host_copy_meta is None and self._hot_inv_meta is None:
            return

        if has_groups and group_ids is not None and self._host_union_group_mask >= 0:
            filtered = [
                (p, g)
                for p, g in zip(pairs, group_ids)
                if g < 0 or g >= 62 or ((self._host_union_group_mask >> g) & 1) != 0
            ]
            if not filtered:
                return
            h_src = async_tensor_h2d(
                [p[0] for p, _ in filtered],
                dtype=torch.int64,
                device=self.device,
                pin_memory=self.pin_memory,
            )
            h_dst = async_tensor_h2d(
                [p[1] for p, _ in filtered],
                dtype=torch.int64,
                device=self.device,
                pin_memory=self.pin_memory,
            )
            h_grp = async_tensor_h2d(
                [g for _, g in filtered],
                dtype=torch.int64,
                device=self.device,
                pin_memory=self.pin_memory,
            )
            n_host_pairs = len(filtered)
            h_has_groups = True
        else:
            h_src, h_dst, h_grp = src, dst, grp
            n_host_pairs = len(pairs)
            h_has_groups = has_groups

        if self._host_copy_meta is not None:
            (
                h_addrs,
                h_strides,
                h_pages,
                h_nblocks,
                h_masks,
                h_chunks,
                h_blk,
                h_nsegs,
            ) = self._host_copy_meta
            _copy_kv_blocks_kernel[(n_host_pairs, h_nsegs, h_chunks)](
                h_addrs,
                h_strides,
                h_pages,
                h_nblocks,
                h_masks,
                h_src,
                h_dst,
                h_grp,
                HAS_PAIR_GROUPS=h_has_groups,
                BLOCK_SIZE=h_blk,
            )

        if self._hot_inv_meta is not None:
            (
                ps_ptrs,
                tag_ptrs,
                stamp_ptrs,
                pages_per_blk,
                hot_nblocks,
                hot_masks,
                hot_chunks,
                page_chunk,
                n_owners,
            ) = self._hot_inv_meta
            _invalidate_hot_kv_blocks_kernel[(n_host_pairs, n_owners, hot_chunks)](
                ps_ptrs,
                tag_ptrs,
                stamp_ptrs,
                pages_per_blk,
                hot_nblocks,
                hot_masks,
                h_dst,
                h_grp,
                HAS_PAIR_GROUPS=h_has_groups,
                PAGE_CHUNK=page_chunk,
            )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
        (
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            seg_num_blocks,
            max_chunks,
            blk_size,
            n_segs,
        ) = self._meta
        n_blocks = len(block_ids)
        # Each nonblocking H2D copy needs its own pinned source allocation.
        # Reusing one host buffer lets a later scheduler step overwrite block
        # IDs while an earlier DMA is still in flight, which can clear the
        # wrong KV pages. The pinned allocator keeps this temporary source
        # alive until its transfer completes.
        idx = async_tensor_h2d(
            block_ids,
            dtype=torch.int64,
            device=self.device,
            pin_memory=self.pin_memory,
        )
        grid = (n_blocks, n_segs, max_chunks)
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            seg_num_blocks,
            idx,
            BLOCK_SIZE=blk_size,
        )

    def warmup_kernel(self) -> bool:
        """JIT the zero/copy/invalidation kernels before the first real step."""
        if self._meta is None and self._host_copy_meta is None:
            return False
        if self._meta is not None:
            self.zero_block_ids([0])
        self.copy_block_ids([(-1, -1)], [-1])
        return True


@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    # When ubatching is enabled we will have a metadata builder for each ubatch
    # so that if they use internal persistent buffers for cudagraphs, and they
    # won't have to worry about conflicting with the other ubatches.
    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
    )

    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ):
        builder_cls = self.backend.get_builder_cls()
        uses_physical_block_table = (
            isinstance(self.kv_cache_spec, AttentionSpec)
            and self.kv_cache_spec.storage_block_size != self.kv_cache_spec.block_size
            and builder_cls.uses_physical_block_table
        )
        if kernel_block_size is None or (uses_physical_block_table):
            kv_cache_spec_builder = self.kv_cache_spec
        elif (
            isinstance(self.kv_cache_spec, AttentionSpec)
            and self.kv_cache_spec.storage_block_size != self.kv_cache_spec.block_size
        ):
            compress_ratio = (
                self.kv_cache_spec.block_size // self.kv_cache_spec.storage_block_size
            )
            kv_cache_spec_builder = self.kv_cache_spec.copy_with_new_block_size(
                compressed_kernel_block_size(self.kv_cache_spec) * compress_ratio
            )
        else:
            kv_cache_spec_builder = self.kv_cache_spec.copy_with_new_block_size(
                kernel_block_size
            )
        builder_kwargs = {}
        if builder_cls.requires_block_table_width:
            max_num_blocks = self.kv_cache_spec.max_num_blocks_per_req(
                vllm_config, vllm_config.model_config.max_model_len
            )
            metadata_block_size = (
                self.kv_cache_spec.block_size
                if uses_physical_block_table
                else kernel_block_size
            )
            builder_kwargs["block_table_width"] = get_block_table_width(
                max_num_blocks,
                self.kv_cache_spec.block_size,
                metadata_block_size,
            )
        self.metadata_builders = [
            builder_cls(
                kv_cache_spec_builder,
                self.layer_names,
                vllm_config,
                device,
                **builder_kwargs,
            )
            for _ in range(num_metadata_builders)
        ]
        if kernel_block_size is not None:
            for builder in self.metadata_builders:
                builder.kernel_block_size = kernel_block_size  # type: ignore[attr-defined]

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        assert len(self.metadata_builders) > ubatch_id
        return self.metadata_builders[ubatch_id]


def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
) -> int:
    """
    Select a block size that is supported by all backends and is a factor of
    kv_manager_block_size.

    If kv_manager_block_size is supported by all backends, return it directly.
    Otherwise, return the max supported size.

    Args:
        kv_manager_block_size: Block size of KV cache.
        backends: List of attention backend classes.

    Returns:
        The selected block size.

    Raises:
        ValueError: If no valid block size found.
    """

    def block_size_is_supported(
        backends: list[type[AttentionBackend]], block_size: int
    ) -> bool:
        """Check if the block size is supported by all backends."""
        for backend in backends:
            is_supported = False
            for supported_size in backend.get_supported_kernel_block_sizes():
                if isinstance(supported_size, int):
                    if block_size == supported_size:
                        is_supported = True
                elif isinstance(supported_size, MultipleOf):
                    if block_size % supported_size.base == 0:
                        is_supported = True
                else:
                    raise ValueError(f"Unknown supported size: {supported_size}")
            if not is_supported:
                return False
        return True

    # Case 1: if the block_size of kv cache manager is supported by all backends,
    # return it directly.
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # Case 2: otherwise, the block_size must be an `int`-format supported size of
    # at least one backend. Iterate over all `int`-format supported sizes in
    # descending order and return the first one that is supported by all backends.
    # Simple proof:
    # If the supported size b is in MultipleOf(x_i) format for all attention
    # backends i, and b a factor of kv_manager_block_size, then
    # kv_manager_block_size also satisfies MultipleOf(x_i) for all i. We will
    # return kv_manager_block_size in case 1.
    all_int_supported_sizes = set(
        supported_size
        for backend in backends
        for supported_size in backend.get_supported_kernel_block_sizes()
        if isinstance(supported_size, int)
    )

    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        if kv_manager_block_size % supported_size != 0:
            continue
        if block_size_is_supported(backends, supported_size):
            return supported_size
    raise ValueError(f"No common block size for {kv_manager_block_size}. ")


def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int]:
    """
    Generate kernel_block_sizes that matches each block_size.

    For attention backends that support virtual block splitting,
    use the supported block sizes from the backend.
    For other backends (like Mamba), use the same block size (no splitting).

    Args:
        kv_cache_config: The KV cache configuration.
        attn_groups: Attention groups indexed by KV cache group id.

    Returns:
        List of kernel block sizes for each cache group.
    """
    kernel_block_sizes = []
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # pick an arbitrary one to dispatch.
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        if isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
        elif isinstance(kv_cache_spec, MambaSpec):
            # This is likely Mamba or other non-attention cache, no splitting.
            kernel_block_sizes.append(kv_cache_spec.block_size)
        else:
            raise NotImplementedError(
                f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    return kernel_block_sizes


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.embed_multimodal`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )


def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    if init_snapshot.free_memory < requested_memory:
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
            f"utilization or reduce GPU memory used by other processes."
        )

    return requested_memory


def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    """
    Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
    """
    if not shared_kv_cache_layers:
        return

    layer_to_kv_cache_group: dict[str, KVCacheGroupSpec] = {}
    for kv_cache_group in kv_cache_groups:
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group[layer_name] = kv_cache_group

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
        tgt_kv_cache_group.layer_names.append(layer_name)

        if runner_only_attn_layers is not None:
            runner_only_attn_layers.add(layer_name)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.

            # TODO - analyze where runner_kv_caches is used and the right
            # way to ensure it properly reflects multiple attention layers
            # in the same decoder block.
            if (
                current_platform.is_cuda_alike()
                or current_platform.is_xpu()
                or current_platform.is_cpu()
            ):
                # We know that the GPU / CPU runner is not impacted by this
                # case. Some test code depends on runner_kv_caches, but
                # not in a way that's impacted by ignoring this.
                pass
            else:
                raise NotImplementedError
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].bind_kv_cache(kv_cache)


def clear_layer_kv_caches(layers: Iterable[Any]) -> None:
    """Detach KV/state tensors installed on model layers by ``bind_kv_cache``.

    Models can outlive their runner through compilation hooks or external
    references. Clearing only the runner-owned list would otherwise leave the
    same allocations reachable from attention and Mamba layers.
    """
    for layer in layers:
        if not hasattr(layer, "kv_cache"):
            continue
        kv_cache = layer.kv_cache
        layer.kv_cache = torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []

        impl = getattr(layer, "impl", None)
        if impl is not None:
            if hasattr(impl, "_k_scale_cache"):
                impl._k_scale_cache = None
            if hasattr(impl, "_v_scale_cache"):
                impl._v_scale_cache = None


def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool:
    """Check if the residual tensor is scattered for sequence parallelism.

    The residual tensor is scattered across tensor parallel ranks when sequence
    parallelism and tensor parallelism is enabled. SP is only supported in
    full-graph compilation mode.
    """
    if not vllm_config.compilation_config.pass_config.enable_sp:
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size

    if tp == 1:
        return False

    assert (
        vllm_config.compilation_config.use_inductor_graph_partition
        or not vllm_config.compilation_config.splitting_ops
    ), "Sequence parallelism requires full-graph compilation"

    # When sequence parallelism is enabled, we always pad num_input_tokens
    # to be a multiple of tensor_parallel_size (tp) earlier.
    assert num_input_tokens % tp == 0

    return True
