# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)
from vllm.v1.kv_offload.worker.worker import OffloadingHandler


def mamba_state_slots(
    num_token_blocks: int,
    block_size: int,
    alignment_tokens: int,
    mamba_spec: MambaSpec,
    retention_interval: int | None,
    reference_tokens: int,
) -> int:
    """Recurrent-state slots needed alongside ``num_token_blocks`` token slots.

    The demand is read off the same ``reachable_block_mask`` the GPU and the
    store path use: for a request of ``reference_tokens`` tokens, count the
    boundary states the mask retains (its replay boundary plus periodic
    checkpoints), allow one shared-prefix junction per request, and multiply
    by the number of such requests the token pool can hold. Dense retention
    keeps every boundary and therefore one state slot per token slot.
    """
    if retention_interval is None:
        return num_token_blocks
    blocks_per_request = max(1, cdiv(reference_tokens, block_size))
    mask = MambaManager.reachable_block_mask(
        start_block=0,
        end_block=blocks_per_request,
        alignment_tokens=alignment_tokens,
        kv_cache_spec=mamba_spec,
        use_eagle=False,
        retention_interval=retention_interval,
        reachable_boundaries=(max(reference_tokens - 1, 0),),
    )
    states_per_request = blocks_per_request if mask is None else sum(mask)
    # A request may additionally pin one shared-prefix junction state.
    states_per_request += 1
    requests_in_pool = max(1, cdiv(num_token_blocks, blocks_per_request))
    return min(num_token_blocks, requests_in_pool * states_per_request)


class CPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        cacheable_groups = {
            i: group
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
            if group.kv_cache_spec.prefix_cacheable
        }
        # Equal-sized Mamba/attention blocks need matching checkpoint coverage.
        # Keep the existing layout for single groups and mixed block geometry.
        self.partition_by_group = (
            kv_cache_config.num_blocks > 0
            and len(cacheable_groups) > 1
            and any(
                isinstance(group.kv_cache_spec, MambaSpec)
                for group in cacheable_groups.values()
            )
            and len(
                {group.kv_cache_spec.block_size for group in cacheable_groups.values()}
            )
            == 1
        )
        # Mamba state slots follow the prefix-cache retention policy (see
        # CacheConfig.prefix_cache_retention_interval): sparse retention keeps
        # only reachable boundary states, so the state pools are sized from the
        # retention mask for a reference request length instead of one slot
        # per token block. The reference defaults to the model's max length;
        # workloads dominated by shorter prompts can lower it.
        self.retention_interval = kv_cache_config.prefix_cache_retention_interval
        self.state_slots_reference_tokens = int(
            self.extra_config.get(
                "mamba_state_slots_reference_tokens",
                vllm_config.model_config.max_model_len,
            )
        )
        if self.state_slots_reference_tokens <= 0:
            raise ValueError("mamba_state_slots_reference_tokens must be positive")

        self.cpu_group_page_sizes: dict[int, int] = {}
        self.cpu_group_num_blocks: dict[int, int] = {}
        world_size = vllm_config.parallel_config.world_size
        if self.partition_by_group:
            mamba_groups: set[int] = set()
            mamba_spec: MambaSpec | None = None
            for i, group in cacheable_groups.items():
                layer_names = set(group.layer_names)
                # Scheduler specs can replace a heterogeneous UniformType spec
                # by its largest layer spec. Physical tensor sizes/shared_by are
                # preserved and give identical budgets on scheduler and workers.
                page_size = sum(
                    tensor.size // kv_cache_config.num_blocks
                    for tensor in kv_cache_config.kv_cache_tensors
                    if not layer_names.isdisjoint(tensor.shared_by)
                )
                self.cpu_group_page_sizes[i] = page_size * self.block_size_factor
                if isinstance(group.kv_cache_spec, MambaSpec):
                    mamba_groups.add(i)
                    mamba_spec = group.kv_cache_spec
            # Physical storage contains only that group's tensors, rather than
            # every shared GPU tensor for each independently allocated group key.
            self.cpu_page_size_per_worker = sum(self.cpu_group_page_sizes.values())
            token_page = sum(
                page
                for i, page in self.cpu_group_page_sizes.items()
                if i not in mamba_groups
            )
            state_page = sum(self.cpu_group_page_sizes[i] for i in mamba_groups)
            budget_per_worker = int(cpu_bytes_to_use) // world_size

            assert mamba_spec is not None
            block_size = mamba_spec.block_size * self.block_size_factor
            alignment_tokens = block_size

            def state_slots(num_token_blocks: int) -> int:
                return mamba_state_slots(
                    num_token_blocks,
                    block_size,
                    alignment_tokens,
                    mamba_spec,
                    self.retention_interval,
                    self.state_slots_reference_tokens,
                )

            def bytes_per_worker(num_token_blocks: int) -> int:
                return (
                    num_token_blocks * token_page
                    + state_slots(num_token_blocks) * state_page
                )

            # Largest token-block count whose token pages plus the state slots
            # the retention mask demands fit the per-worker budget.
            num_blocks = 0
            if token_page + state_page > 0:
                num_blocks = budget_per_worker // (token_page + state_page)
                while bytes_per_worker(num_blocks + 1) <= budget_per_worker:
                    num_blocks += 1
                while (
                    num_blocks > 0 and bytes_per_worker(num_blocks) > budget_per_worker
                ):
                    num_blocks -= 1
            self.num_blocks = num_blocks
            for i in self.cpu_group_page_sizes:
                self.cpu_group_num_blocks[i] = (
                    state_slots(num_blocks) if i in mamba_groups else num_blocks
                )
        else:
            total_gpu_kv_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
            self.cpu_page_size_per_worker = (
                total_gpu_kv_bytes
                // kv_cache_config.num_blocks
                * self.block_size_factor
                if kv_cache_config.num_blocks > 0
                else 0
            )
            kv_bytes_per_offloaded_block = self.cpu_page_size_per_worker * world_size
            self.num_blocks = (
                int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
                if kv_bytes_per_offloaded_block > 0
                else 0
            )

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handlers: CpuGpuOffloadingHandlers | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            def create_manager(num_blocks: int) -> CPUOffloadingManager:
                return CPUOffloadingManager(
                    num_blocks=num_blocks,
                    cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                    enable_events=enable_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )

            self._manager = (
                GroupedCPUOffloadingManager(
                    {
                        i: create_manager(num_blocks)
                        for i, num_blocks in self.cpu_group_num_blocks.items()
                    }
                )
                if self.partition_by_group
                else create_manager(self.num_blocks)
            )
        return self._manager

    def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuGpuOffloadingHandlers:
        return CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            group_page_sizes=(
                self.cpu_group_page_sizes if self.partition_by_group else None
            ),
            group_num_blocks=(
                self.cpu_group_num_blocks if self.partition_by_group else None
            ),
        )

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike GPUs"
                )
            self._handlers = self.create_handlers(kv_caches)

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler
