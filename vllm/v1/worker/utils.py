# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product as iprod
from typing import Any

import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import PAGED_MQA_PAGE_SIZES
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.block_table import get_block_table_width

logger = init_logger(__name__)


def _resolve_zeroer_kernel_layout(
    spec: FullAttentionSpec,
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
    """
    block_index = tl.program_id(0)
    seg_index = tl.program_id(1)
    chunk_index = tl.program_id(2)
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    chunk_offset = chunk_index.to(tl.int64) * BLOCK_SIZE
    if chunk_offset >= page_size_el:
        return
    block_id = tl.load(block_ids_ptr + block_index)
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
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int] | None
        ) = None
        self._id_cap: int = 0
        self._ids_pinned: torch.Tensor | None = None
        self._ids_gpu: torch.Tensor | None = None

    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None:
        """One-time precomputation for zero_block_ids.

        Builds absolute-address table for the Triton zeroing kernel.
        Each entry is the absolute byte address of a segment start on the
        GPU, so segments in different CUDA allocations work correctly.

        Block IDs from the scheduler reference logical blocks whose size
        may differ from the kernel block size (virtual block splitting).
        PAGE_SIZE_EL accounts for this ratio so that
        ``block_id * PAGE_SIZE_EL`` lands at the correct offset.

        Full-attention layers may expose K/V as separate outer dimensions, so
        they can contribute multiple segments. Mamba/GDN layers expose typed
        state tensor views over one raw page per block; the first state tensor
        starts at the page base, so one segment per layer zeros the whole page.
        """
        seen_ptrs: dict[int, int] = {}
        seg_addrs: list[int] = []
        seg_page_sizes: list[int] = []

        def add_segment(address: int, page_size_el: int) -> None:
            if (index := seen_ptrs.get(address)) is not None:
                seg_page_sizes[index] = max(seg_page_sizes[index], page_size_el)
                return
            seen_ptrs[address] = len(seg_addrs)
            seg_addrs.append(address)
            seg_page_sizes.append(page_size_el)

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv = static_forward_context[layer_name].kv_cache
                if isinstance(spec, FullAttentionSpec):
                    if not isinstance(kv, torch.Tensor):
                        continue
                    kernel_bs, ratio = _resolve_zeroer_kernel_layout(
                        spec,
                        kernel_block_sizes[group.kv_cache_group_id],
                    )
                    block_dim = group.backend.get_kv_cache_block_dim(
                        kernel_bs,
                        spec.num_kv_heads,
                        spec.head_size,
                        cache_dtype_str=cache_dtype,
                    )
                    dp = kv.data_ptr()

                    el = kv.element_size()
                    cur_bytes = kv.stride(block_dim) * el
                    assert cur_bytes % 4 == 0
                    kernel_block_el = cur_bytes // 4
                    cur_page_el = kernel_block_el * ratio

                    block_stride_bytes = cur_bytes
                    outer_dims = [
                        d
                        for d in range(block_dim)
                        if kv.stride(d) * el > block_stride_bytes
                    ]
                    outer_strides = [kv.stride(d) * el for d in outer_dims]
                    for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                        off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                        add_segment(dp + off_bytes, cur_page_el)
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
                    add_segment(dp, cur_page_el)

        if not seg_addrs:
            self._meta = None
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

        self._id_cap = 8192
        self._ids_pinned = torch.empty(
            self._id_cap,
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            seg_block_strides,
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            (max_page_size_el + blk_size - 1) // blk_size,
            blk_size,
            n_segs,
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
        (
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            max_chunks,
            blk_size,
            n_segs,
        ) = self._meta
        n_blocks = len(block_ids)
        if n_blocks > self._id_cap:
            self._id_cap = n_blocks * 2
            self._ids_pinned = torch.empty(
                self._id_cap,
                dtype=torch.int64,
                pin_memory=self.pin_memory,
            )
            self._ids_gpu = torch.empty(
                self._id_cap, dtype=torch.int64, device=self.device
            )
        assert self._ids_pinned is not None and self._ids_gpu is not None
        self._ids_pinned[:n_blocks].numpy()[:] = block_ids
        idx = self._ids_gpu[:n_blocks]
        idx.copy_(self._ids_pinned[:n_blocks], non_blocking=True)
        grid = (n_blocks, n_segs, max_chunks)
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            idx,
            BLOCK_SIZE=blk_size,
        )

    def warmup_kernel(self) -> bool:
        """JIT the zero-block kernel before the first real scheduler step."""
        if self._meta is None:
            return False
        self.zero_block_ids([0])
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
        if kernel_block_size is None:
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
        builder_cls = self.backend.get_builder_cls()
        builder_kwargs = {}
        if builder_cls.requires_block_table_width:
            max_num_blocks = self.kv_cache_spec.max_num_blocks_per_req(
                vllm_config, vllm_config.model_config.max_model_len
            )
            builder_kwargs["block_table_width"] = get_block_table_width(
                max_num_blocks, self.kv_cache_spec.block_size, kernel_block_size
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
