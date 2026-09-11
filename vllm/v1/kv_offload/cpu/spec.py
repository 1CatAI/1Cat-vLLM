# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.platforms import current_platform
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
        self.cpu_group_page_sizes: dict[int, int] = {}
        if self.partition_by_group:
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
            # Each pool has the same number of token blocks. Physical storage
            # contains only that group's tensors, rather than every shared GPU
            # tensor for each independently allocated group key.
            self.cpu_page_size_per_worker = sum(self.cpu_group_page_sizes.values())
        else:
            total_gpu_kv_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
            self.cpu_page_size_per_worker = (
                total_gpu_kv_bytes
                // kv_cache_config.num_blocks
                * self.block_size_factor
                if kv_cache_config.num_blocks > 0
                else 0
            )
        kv_bytes_per_offloaded_block = (
            self.cpu_page_size_per_worker * vllm_config.parallel_config.world_size
        )
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

            def create_manager() -> CPUOffloadingManager:
                return CPUOffloadingManager(
                    num_blocks=self.num_blocks,
                    cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                    enable_events=enable_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )

            self._manager = (
                GroupedCPUOffloadingManager(
                    {i: create_manager() for i in self.cpu_group_page_sizes}
                )
                if self.partition_by_group
                else create_manager()
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
