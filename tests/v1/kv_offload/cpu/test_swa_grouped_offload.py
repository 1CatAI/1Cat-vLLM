# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    SchedulerOffloadConfig,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec, swa_tail_slots


def _make_hybrid_configs(
    *,
    cpu_bytes_to_use: int = 8 * (1 << 30),
    world_size: int = 4,
    max_model_len: int = 131072,
    extra_config: dict | None = None,
):
    """Build an Inkling2-like Full+SWA hybrid config with mixed block sizes (4 and 64)."""
    num_gpu_blocks = 1000
    groups: list[KVCacheGroupSpec] = []
    tensors: list[KVCacheTensor] = []

    # Groups 0..5: conv-state sliding window (block_size=4, window=4)
    for i in range(6):
        lname = f"conv_{i}"
        groups.append(
            KVCacheGroupSpec(
                layer_names=[lname],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=4,
                    num_kv_heads=2,
                    head_size=128,
                    dtype=torch.float16,
                    sliding_window=4,
                ),
            )
        )
        tensors.append(
            KVCacheTensor(size=num_gpu_blocks * 4 * 1024, shared_by=[lname])
        )

    # Groups 6..10: local SWA attention (block_size=64, window=1024)
    for i in range(5):
        lname = f"swa_{i}"
        groups.append(
            KVCacheGroupSpec(
                layer_names=[lname],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=64,
                    num_kv_heads=2,
                    head_size=128,
                    dtype=torch.float16,
                    sliding_window=1024,
                ),
            )
        )
        tensors.append(
            KVCacheTensor(size=num_gpu_blocks * 512 * 1024, shared_by=[lname])
        )

    # Group 11: full attention (block_size=64)
    groups.append(
        KVCacheGroupSpec(
            layer_names=["global_0"],
            kv_cache_spec=FullAttentionSpec(
                block_size=64,
                num_kv_heads=2,
                head_size=128,
                dtype=torch.float16,
            ),
        )
    )
    tensors.append(
        KVCacheTensor(size=num_gpu_blocks * 576 * 1024, shared_by=["global_0"])
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )
    extra = {"cpu_bytes_to_use": cpu_bytes_to_use}
    if extra_config:
        extra.update(extra_config)

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(
            world_size=world_size,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            block_size=64,
            hash_block_size=4,
            enable_prefix_caching=True,
            user_specified_block_size=False,
            mamba_block_size=None,
            mamba_cache_mode="none",
            prefix_cache_retention_interval=None,
        ),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
    )
    return vllm_config, kv_cache_config


def test_swa_tail_slots_sizing():
    # 500K tokens at block_size=64 -> 7813 full blocks; window=1024 -> 16 blocks; ref=16K -> 256 blocks
    slots = swa_tail_slots(num_token_blocks=7813, sw_blocks=16, ref_blocks=256)
    assert slots == 31 * 18
    assert slots < 7813 // 10
    assert swa_tail_slots(num_token_blocks=0, sw_blocks=16, ref_blocks=256) == 0
    assert swa_tail_slots(num_token_blocks=10, sw_blocks=16, ref_blocks=256) == 10


def test_cpu_offloading_spec_full_plus_swa_mixed_block_sizes():
    budget = 8 * (1 << 30)
    world_size = 4
    vllm_cfg, kv_cfg = _make_hybrid_configs(
        cpu_bytes_to_use=budget, world_size=world_size
    )
    spec = CPUOffloadingSpec(vllm_cfg, kv_cfg)

    assert spec.partition_by_group is True
    assert spec._swa_partition is True
    assert spec.swa_tail_only_offload is True

    # Full-attention group 11 gets the bulk of the budget; SWA groups get tail slots.
    assert spec.cpu_group_num_blocks[11] == spec.num_blocks
    assert spec.num_blocks > 2000  # >128K tokens in 8 GiB vs ~650 blocks monolithic
    for i in range(6):
        assert spec.cpu_group_num_blocks[i] < spec.num_blocks // 20
    for i in range(6, 11):
        assert spec.cpu_group_num_blocks[i] < spec.num_blocks // 5

    # Per-worker byte budget must not be exceeded.
    total_per_worker = sum(
        spec.cpu_group_num_blocks[i] * spec.cpu_group_page_sizes[i]
        for i in spec.cpu_group_num_blocks
    )
    assert total_per_worker <= budget // world_size

    sched_cfg = SchedulerOffloadConfig.from_spec(spec)
    assert sched_cfg.swa_tail_only_offload is True


def test_cpu_offloading_spec_swa_tail_only_opt_out():
    vllm_cfg, kv_cfg = _make_hybrid_configs(
        extra_config={"swa_tail_only_offload": False}
    )
    spec = CPUOffloadingSpec(vllm_cfg, kv_cfg)

    assert spec.partition_by_group is False
    assert spec._swa_partition is False
    sched_cfg = SchedulerOffloadConfig.from_spec(spec)
    assert sched_cfg.swa_tail_only_offload is False
