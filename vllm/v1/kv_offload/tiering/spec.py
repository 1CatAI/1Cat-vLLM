# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingSpec: Spec for multi-tier KV cache offloading.

This spec creates a TieringOffloadingManager with a CPU primary tier
and configurable secondary tiers (e.g., Storage, Network).

Configuration via kv_connector_extra_config:
  - cpu_bytes_to_use: (required) Bytes to allocate for CPU primary tier
  - block_size: (optional) Block size for offloaded blocks (default: GPU block size)
  - eviction_policy: (optional) Primary tier eviction policy: "lru" or
    "arc" (default: "lru")
  - secondary_tiers: (optional) List of secondary tier configurations
    Each secondary tier config is a dict with:
      - type: (required) Type of secondary tier (e.g., "example", "storage", "network")
      - Additional tier-specific parameters are passed directly to the tier
        constructor. See each tier's documentation for supported parameters.

Example configuration:
{
    "cpu_bytes_to_use": 10737418240,  # 10 GB
    "block_size": 16,
    "eviction_policy": "lru",
    "secondary_tiers": [
        {
            "type": "example",
            "custom_param": 67
        }
    ]
}
"""

from contextlib import ExitStack

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingManager
from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.cpu.manager import GroupedCPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

logger = init_logger(__name__)


class TieringOffloadingSpec(CPUOffloadingSpec):
    """
    Spec for multi-tier KV cache offloading.

    Creates a TieringOffloadingManager with:
    - Primary tier: CPU (LRU or ARC eviction policy)
    - Secondary tiers: Configurable via extra_config

    The CPU primary tier has direct GPU access and serves as the gateway for
    all GPU↔offload operations. Secondary tiers cannot directly access GPU
    memory and must transfer data through the primary tier.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        # Redeclare for mypy: parent sets this but `--follow-imports skip` hides it
        self._manager: OffloadingManager | None = None

        # Parse secondary tier configurations
        self.secondary_tier_configs = self.extra_config.get("secondary_tiers", [])
        if not isinstance(self.secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        self.persistent_layout: dict | None = None
        if self.partition_by_group:
            parallel = vllm_config.parallel_config
            if (
                parallel.pipeline_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
                or parallel.nnodes != 1
            ):
                raise ValueError("Grouped tiering currently requires single-node TP")
            backend = vllm_config.attention_config.backend
            if backend is None:
                raise ValueError(
                    "Grouped tiering requires an explicit attention backend"
                )
            self.persistent_layout = {
                "version": "grouped-worker-interleaved-v1",
                "config_hash": vllm_config.compute_hash(),
                "attention_backend": str(backend),
                "model_revision": vllm_config.model_config.revision,
                "group_pages": self.cpu_group_page_sizes,
                "tensors": [
                    {
                        "page_size": tensor.size // kv_cache_config.num_blocks,
                        "shared_by": list(tensor.shared_by),
                    }
                    for tensor in kv_cache_config.kv_cache_tensors
                ],
            }

        # Scheduler-side mmap (rank=None); kept for cleanup
        self._scheduler_mmap: SharedOffloadRegion | None = None

    def _create_group_regions(self, rank: int | None) -> dict[int, SharedOffloadRegion]:
        """Use the same group names and row geometry on scheduler and workers."""
        world_size = self.vllm_config.parallel_config.world_size
        regions: dict[int, SharedOffloadRegion] = {}
        try:
            for group, page_size in self.cpu_group_page_sizes.items():
                num_blocks = self.cpu_group_num_blocks[group]
                regions[group] = SharedOffloadRegion(
                    instance_id=f"{self.vllm_config.instance_id}_g{group}",
                    total_size_bytes=page_size * world_size * num_blocks,
                    num_blocks=num_blocks,
                    rank=rank,
                    num_workers=world_size,
                    cpu_page_size=page_size,
                )
        except Exception:
            for region in regions.values():
                region.cleanup()
            raise
        return regions

    def _get_grouped_manager(self, enable_events: bool) -> OffloadingManager:
        if int(self.extra_config.get("store_threshold", 0)) >= 2:
            raise ValueError(
                "store_threshold is not supported for TieringOffloadingSpec"
            )
        # ExitStack unwinds tiers before primary views, including partial setup.
        with ExitStack() as cleanup:
            regions = self._create_group_regions(None)
            for region in regions.values():
                cleanup.callback(region.cleanup)
            managers: dict[int, OffloadingManager] = {}
            for group, region in regions.items():
                primary = CPUPrimaryTierOffloadingManager(
                    num_blocks=self.cpu_group_num_blocks[group],
                    cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                    enable_events=enable_events,
                    mmap_region=region,
                )
                cleanup.callback(primary.shutdown)
                tiers = []
                for config in self.secondary_tier_configs:
                    tier = SecondaryTierFactory.create_secondary_tier(
                        config, primary.get_kv_memoryview(), self
                    )
                    cleanup.callback(tier.shutdown)
                    tiers.append(tier)
                managers[group] = TieringOffloadingManager(
                    primary_tier=primary,
                    secondary_tiers=tiers,
                    enable_events=enable_events,
                )
            manager = GroupedCPUOffloadingManager(managers)
            cleanup.pop_all()
            return manager

    @override
    def get_manager(self) -> OffloadingManager:
        """
        Get the TieringOffloadingManager.

        Creates a TieringOffloadingManager with:
        - Primary tier: CPU (LRU or ARC)
        - Secondary tiers: As configured in extra_config

        Returns:
            TieringOffloadingManager instance
        """
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            if self.partition_by_group:
                self._manager = self._get_grouped_manager(enable_events)
                return self._manager

            # Create scheduler-side SharedOffloadRegion (rank=None) so the
            # primary tier can eagerly create a memoryview over _base.
            world_size = self.vllm_config.parallel_config.world_size
            scheduler_mmap = SharedOffloadRegion(
                instance_id=self.vllm_config.instance_id,
                total_size_bytes=self.cpu_page_size_per_worker
                * world_size
                * self.num_blocks,
                num_blocks=self.num_blocks,
                rank=None,
                num_workers=world_size,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
            self._scheduler_mmap = scheduler_mmap

            # Create primary tier (CPU-based)
            assert len(self.gpu_block_size) == 1
            primary_tier = CPUPrimaryTierOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=enable_events,
                mmap_region=scheduler_mmap,
            )

            # Create secondary tiers
            primary_kv_view = primary_tier.get_kv_memoryview()
            secondary_tiers = []
            for i, tier_config in enumerate(self.secondary_tier_configs):
                try:
                    tier = SecondaryTierFactory.create_secondary_tier(
                        tier_config, primary_kv_view, self
                    )
                    secondary_tiers.append(tier)
                    logger.info(
                        "Created secondary tier #%d (%s)",
                        i,
                        tier.tier_type,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to create secondary tier from config %s: %s",
                        tier_config,
                        e,
                    )
                    raise

            # Create TieringOffloadingManager. GPU↔CPU transfers use the inherited
            # get_handlers(); secondary tier transfers are handled by the
            # secondary tier managers and need no additional handlers here.
            tiering_manager = TieringOffloadingManager(
                primary_tier=primary_tier,
                secondary_tiers=secondary_tiers,
                enable_events=enable_events,
            )
            if int(self.extra_config.get("store_threshold", 0)) >= 2:
                raise ValueError(
                    "store_threshold is not supported for TieringOffloadingSpec"
                )
            self._manager = tiering_manager

            logger.info(
                "Created TieringOffloadingManager with primary tier "
                "(%s, %s blocks) and %s secondary tier(s)",
                self.eviction_policy,
                self.num_blocks,
                len(secondary_tiers),
            )

        return self._manager

    @override
    def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuGpuOffloadingHandlers:
        world_size = self.vllm_config.parallel_config.world_size
        rank = torch.accelerator.current_device_index()
        if self.partition_by_group:
            return CpuGpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
                group_page_sizes=self.cpu_group_page_sizes,
                group_mmap_regions=self._create_group_regions(rank),
                group_num_blocks=self.cpu_group_num_blocks,
            )
        worker_mmap = SharedOffloadRegion(
            instance_id=self.vllm_config.instance_id,
            total_size_bytes=self.cpu_page_size_per_worker
            * world_size
            * self.num_blocks,
            num_blocks=self.num_blocks,
            rank=rank,
            num_workers=world_size,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        return CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            mmap_region=worker_mmap,
        )
