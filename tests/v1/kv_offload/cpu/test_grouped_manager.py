# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    OffloadingSpec,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.gpu_worker import partition_kv_caches
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec, mamba_state_slots

CTX = ReqContext("test")
GROUPS = (0, 2, 3, 4, 5)


@pytest.mark.parametrize("error_code", [1, 2])
def test_mmap_registration_failure_is_fatal(monkeypatch, error_code):
    from vllm.v1.kv_offload.cpu import gpu_worker as worker

    region = SimpleNamespace(
        rank=3,
        mmap_path="/dev/shm/test-offload.mmap",
        total_size_bytes=4096,
        _base=torch.zeros(4096, dtype=torch.int8),
        is_pinned=False,
    )
    calls = []

    def register(ptr, size, flags):
        calls.append((ptr, size, flags))
        return SimpleNamespace(value=error_code)

    monkeypatch.setattr(
        torch.cuda, "cudart", lambda: SimpleNamespace(cudaHostRegister=register)
    )
    with pytest.raises(RuntimeError, match="KV batch transfers require pinned memory"):
        worker.pin_mmap_region(region)
    assert calls == [(region._base.data_ptr(), 4096, 0)]
    assert not region.is_pinned


def key(group, index):
    return make_offload_key(index.to_bytes(8, "big"), group)


def manager(capacity=2, policy="lru"):
    return GroupedCPUOffloadingManager(
        {
            g: CPUOffloadingManager(capacity, cache_policy=policy, enable_events=True)
            for g in GROUPS
        }
    )


def store(m, keys):
    out = m.prepare_store(keys, CTX)
    assert out is not None
    m.complete_store(out.keys_to_store, CTX)
    return out


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_large_context_capacity(policy):
    # Measured Flash-Next geometry: 16 GiB buys 419 all-tensor slots, or
    # 107 slots per group when each key pays for only its own tensors.
    old = CPUOffloadingManager(419, cache_policy=policy)
    new = manager(107, policy)
    a = [key(g, i) for g in GROUPS for i in range(48)]
    b = [key(g, i + 100) for g in GROUPS for i in range(48)]
    for m in (old, new):
        store(m, a)
        store(m, b)
    assert not all(old.lookup(k, CTX) for k in a)
    assert all(new.lookup(k, CTX) for k in a + b)
    assert list(new.prepare_load(a, CTX).block_ids) == list(range(48)) * 5
    new.complete_load(a, CTX)


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_group_pool_lifecycle_and_order(policy):
    m = manager(policy=policy)
    keys = [key(5, 1), key(0, 2), key(5, 3)]
    out = m.prepare_store(keys, CTX)
    assert out.keys_to_store == keys
    assert list(out.store_spec.block_ids) == [0, 0, 1]
    assert m.lookup(keys[0], CTX) is None
    m.complete_store(keys, CTX)
    assert list(m.prepare_load(keys, CTX).block_ids) == [0, 0, 1]
    # A blocked group defers the entire operation, releasing other reservations.
    partial = m.prepare_store([key(5, 4), key(0, 4)], CTX)
    assert partial is None
    assert m.lookup(key(0, 4), CTX) is False
    m.complete_load(keys, CTX)
    m.touch([keys[0]], CTX)
    out = store(m, [key(5, 4)])
    assert out.evicted_keys == [keys[2]]
    # Eviction recycles the old slot within the bounded group pool.
    assert list(out.store_spec.block_ids) == [1]
    assert m.lookup(keys[0], CTX) is True
    assert m.lookup(keys[2], CTX) is False
    assert m.lookup(keys[1], CTX) is True
    assert list(m.take_events())
    assert not list(m.take_events())
    m.reset_cache()
    assert all(m.lookup(k, CTX) is False for k in keys)
    assert list(store(m, [keys[0]]).store_spec.block_ids) == [0]


def test_group_views_preserve_gpu_aliases_but_separate_cpu_indices():
    tensor = CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16)
    caches = CanonicalKVCaches(
        [tensor],
        [
            [CanonicalKVCacheRef(0, 16)],
            [CanonicalKVCacheRef(0, 16)],
            [CanonicalKVCacheRef(0, 8)],
        ],
    )
    split = partition_kv_caches(caches, {0: 32, 2: 32}, 2)
    assert split.tensors[0].tensor is split.tensors[1].tensor
    assert split.group_data_refs[0][0].tensor_idx == 0
    assert split.group_data_refs[1] == []
    assert split.group_data_refs[2][0].tensor_idx == 1
    assert split.group_data_refs[2][0].page_size_bytes == 8
    with pytest.raises(AssertionError, match="byte budget"):
        partition_kv_caches(caches, {0: 31, 2: 32}, 2)


FLASH_NEXT_PAGES = {
    0: 10235904,
    2: 9633792,
    3: 9633792,
    4: 9633792,
    5: 802816,
}


def _flash_next_spec(
    monkeypatch, layout="equal", retention_interval=None, **extra_config
):
    def init(self, config, caches):
        self.vllm_config = config
        self.kv_cache_config = caches
        self.extra_config = {"cpu_bytes_to_use": 16 * 1024**3, **extra_config}
        self.block_size_factor = 1

    monkeypatch.setattr(OffloadingSpec, "__init__", init)
    fa = FullAttentionSpec(
        block_size=784, num_kv_heads=1, head_size=256, dtype=torch.float16
    )
    small = FullAttentionSpec(
        block_size=784, num_kv_heads=1, head_size=16, dtype=torch.float16
    )
    uniform = UniformTypeKVCacheSpecs(
        block_size=784,
        kv_cache_specs={
            **{f"fa{i}": fa for i in range(12)},
            **{f"index{i}": small for i in range(12)},
        },
    )
    mamba = MambaSpec(
        block_size=784,
        shapes=((1,),),
        dtypes=(torch.float16,),
        page_size_padded=802816,
        mamba_cache_mode="align",
    )
    group = lambda spec, count: SimpleNamespace(
        kv_cache_spec=spec, layer_names=list(range(count))
    )
    groups = [group(uniform, 24), group(SimpleNamespace(prefix_cacheable=False), 12)]
    groups += [group(mamba, n) for n in (12, 12, 12, 1)]
    groups[0].layer_names = list(uniform.kv_cache_specs)
    for g in (2, 3, 4):
        groups[g].layer_names = [f"m{g}_{i}" for i in range(12)]
    groups[5].layer_names = ["ple"]
    tensors = [
        SimpleNamespace(
            size=104 * 802816,
            shared_by=[f"fa{i}", f"m2_{i}", f"m3_{i}", f"m4_{i}"]
            + (["ple"] if i == 0 else []),
        )
        for i in range(12)
    ] + [SimpleNamespace(size=104 * 50176, shared_by=[f"index{i}"]) for i in range(12)]
    if layout == "scheduler":
        groups[0].kv_cache_spec = fa
    if layout == "mixed":
        groups[2] = group(replace(mamba, block_size=392), 12)
    elif layout == "attention_only":
        groups = [group(uniform, 24), group(uniform, 24)]
    return CPUOffloadingSpec(
        SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=4),
            model_config=SimpleNamespace(max_model_len=65536),
        ),
        SimpleNamespace(
            kv_cache_groups=groups,
            num_blocks=104,
            kv_cache_tensors=tensors,
            prefix_cache_retention_interval=retention_interval,
        ),
    )


@pytest.mark.parametrize("layout", ["equal", "scheduler", "mixed", "attention_only"])
def test_actual_flash_next_group_budget(monkeypatch, layout):
    spec = _flash_next_spec(monkeypatch, layout)
    if layout not in ("equal", "scheduler"):
        assert not spec.partition_by_group
        assert spec.num_blocks == 419
        assert spec.cpu_group_num_blocks == {}
        return
    assert spec.cpu_group_page_sizes == FLASH_NEXT_PAGES
    # Dense retention (None): one state slot per token slot, 107 each.
    assert spec.num_blocks == 107
    assert spec.retention_interval is None
    assert spec.cpu_group_num_blocks == dict.fromkeys(FLASH_NEXT_PAGES, 107)
    assert spec.cpu_page_size_per_worker * 4 * spec.num_blocks <= 16 * 1024**3
    assert spec.num_blocks >= 2 * 48


def _grouped_bytes(spec):
    return 4 * sum(
        spec.cpu_group_page_sizes[g] * n for g, n in spec.cpu_group_num_blocks.items()
    )


def test_flash_next_semantic_retention_budget(monkeypatch):
    # retention 0 keeps only the replay boundary (+ one junction allowance)
    # per request. Sized for 64K requests (84 blocks): 2 states per request.
    spec = _flash_next_spec(monkeypatch, retention_interval=0)
    assert spec.partition_by_group
    assert spec.retention_interval == 0
    assert spec.state_slots_reference_tokens == 65536
    token_slots = spec.num_blocks
    requests = -(-token_slots // 84)
    state_slots = requests * 2
    assert spec.cpu_group_num_blocks == {
        0: token_slots,
        2: state_slots,
        3: state_slots,
        4: state_slots,
        5: state_slots,
    }
    # Nearly the whole 16 GiB now buys token pages: > 3.5x the dense layout.
    assert token_slots >= 380
    assert _grouped_bytes(spec) <= 16 * 1024**3
    spec.cpu_group_num_blocks[0] += 1
    assert _grouped_bytes(spec) > 16 * 1024**3


def test_flash_next_periodic_retention_budget(monkeypatch):
    # retention 8 blocks: periodic states every 8 blocks plus replay/junction.
    spec = _flash_next_spec(monkeypatch, retention_interval=8 * 784)
    token_slots = spec.num_blocks
    # 84-block reference request: boundaries at blocks 8,16,...,80 (10),
    # the replay boundary at block 81 (index 80 already counted? no: replay
    # boundary 65535 -> block 83 -> index 82) plus the junction allowance.
    per_request = mamba_state_slots(
        84,
        784,
        784,
        spec.kv_cache_config.kv_cache_groups[2].kv_cache_spec,
        8 * 784,
        65536,
    )
    assert per_request == 10 + 1 + 1
    assert spec.cpu_group_num_blocks[2] == min(
        token_slots, -(-token_slots // 84) * per_request
    )
    assert token_slots > 107
    assert _grouped_bytes(spec) <= 16 * 1024**3


def test_state_slots_follow_reference_length(monkeypatch):
    # A shorter sizing reference reserves more state slots per token slot.
    long = _flash_next_spec(monkeypatch, retention_interval=0)
    short = _flash_next_spec(
        monkeypatch,
        retention_interval=0,
        mamba_state_slots_reference_tokens=8 * 784,
    )
    assert short.cpu_group_num_blocks[2] > long.cpu_group_num_blocks[2]
    assert short.num_blocks < long.num_blocks
    assert _grouped_bytes(short) <= 16 * 1024**3
    with pytest.raises(ValueError, match="reference_tokens"):
        _flash_next_spec(
            monkeypatch, retention_interval=0, mamba_state_slots_reference_tokens=0
        )


def test_private_cpu_allocations_obey_group_budget(monkeypatch):
    import vllm.v1.kv_offload.cpu.gpu_worker as worker

    monkeypatch.setattr(worker, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        worker, "SingleDirectionOffloadingHandler", lambda **kw: SimpleNamespace(**kw)
    )
    tensor = CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16)
    caches = CanonicalKVCaches(
        [tensor],
        [
            [CanonicalKVCacheRef(0, 16)],
            [CanonicalKVCacheRef(0, 16)],
            [CanonicalKVCacheRef(0, 8)],
        ],
    )
    handlers = worker.CpuGpuOffloadingHandlers(
        caches, 2, 2, group_page_sizes={0: 32, 2: 32}
    )
    cpu = handlers.gpu_to_cpu_handler.cpu_tensors
    assert sum(t.numel() for t in cpu) == 2 * (32 + 32)
    cpu[0][0].fill_(7)
    cpu[1][0].fill_(3)
    assert torch.all(cpu[0][0] == 7)
    assert torch.all(cpu[1][0] == 3)
    assert torch.all(cpu[0][1] == 0)
    assert handlers.cpu_to_gpu_handler.cpu_tensors is cpu


def test_private_cpu_allocations_use_group_slot_counts(monkeypatch):
    import vllm.v1.kv_offload.cpu.gpu_worker as worker

    monkeypatch.setattr(worker, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        worker, "SingleDirectionOffloadingHandler", lambda **kw: SimpleNamespace(**kw)
    )
    tensor = CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16)
    caches = CanonicalKVCaches(
        [tensor],
        [[CanonicalKVCacheRef(0, 16)], [], [CanonicalKVCacheRef(0, 8)]],
    )
    handlers = worker.CpuGpuOffloadingHandlers(
        caches,
        2,
        4,
        group_page_sizes={0: 32, 2: 32},
        group_num_blocks={0: 4, 2: 1},
    )
    cpu = handlers.gpu_to_cpu_handler.cpu_tensors
    assert [t.shape for t in cpu] == [(4, 32), (1, 32)]
    with pytest.raises(AssertionError):
        worker.CpuGpuOffloadingHandlers(caches, 2, 4, group_num_blocks={0: 4})


def test_grouped_worker_views_share_scheduler_rows(monkeypatch):
    import uuid

    import vllm.v1.kv_offload.cpu.gpu_worker as worker
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    monkeypatch.setattr(worker, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        worker, "SingleDirectionOffloadingHandler", lambda **kw: SimpleNamespace(**kw)
    )
    tensor = CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16)
    caches = CanonicalKVCaches(
        [tensor],
        [[CanonicalKVCacheRef(0, 16)], [], [CanonicalKVCacheRef(0, 8)]],
    )
    scheduler = {}
    regions = []
    handlers = []
    try:
        for group in (0, 2):
            instance = f"grouped-worker-test-{uuid.uuid4().hex}"
            args = dict(
                instance_id=instance,
                total_size_bytes=8192,
                num_blocks=2,
                num_workers=2,
                cpu_page_size=2048,
            )
            scheduler[group] = SharedOffloadRegion(rank=None, **args)
            regions.append(
                {rank: SharedOffloadRegion(rank=rank, **args) for rank in (0, 1)}
            )
        for rank in (0, 1):
            handler = worker.CpuGpuOffloadingHandlers(
                caches,
                2,
                2,
                group_page_sizes={0: 2048, 2: 2048},
                group_mmap_regions={g: regions[i][rank] for i, g in enumerate((0, 2))},
            )
            handlers.append(handler)
            for i, cpu in enumerate(handler.gpu_to_cpu_handler.cpu_tensors):
                assert cpu.stride() == (4096, 1)
                cpu[1].fill_(10 + rank + i * 20)
        for i, group in enumerate((0, 2)):
            with scheduler[group].create_kv_memoryview().cast("B") as view:
                for rank in (0, 1):
                    offset = 4096 + rank * 2048
                    assert (
                        bytes(view[offset : offset + 32])
                        == bytes([10 + rank + i * 20]) * 32
                    )
                    view[offset : offset + 32] = bytes([90 + rank]) * 32
            for rank in (0, 1):
                assert torch.all(
                    handlers[rank].gpu_to_cpu_handler.cpu_tensors[i][1] == 90 + rank
                )
    finally:
        for handler in handlers:
            handler.gpu_to_cpu_handler.cpu_tensors.clear()
        for group_regions in regions:
            for region in group_regions.values():
                region.cleanup()
        for region in scheduler.values():
            region.cleanup()


@pytest.mark.parametrize("failure", ["pin", "view", "handler"])
def test_grouped_worker_initialization_failure_releases_regions(monkeypatch, failure):
    import uuid
    from pathlib import Path

    import vllm.v1.kv_offload.cpu.gpu_worker as worker
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    def fail(*args, **kwargs):
        raise RuntimeError("injected initialization failure")

    tensor = CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16)
    caches = CanonicalKVCaches(
        [tensor], [[CanonicalKVCacheRef(0, 16)], [], [CanonicalKVCacheRef(0, 16)]]
    )
    regions = {
        group: SharedOffloadRegion(
            instance_id=f"grouped-failure-test-{uuid.uuid4().hex}",
            total_size_bytes=4096,
            num_blocks=2,
            rank=0,
            num_workers=1,
            cpu_page_size=2048,
        )
        for group in (0, 2)
    }
    mappings = [region.mmap_obj for region in regions.values()]
    monkeypatch.setattr(worker, "is_pin_memory_available", lambda: failure == "pin")
    monkeypatch.setattr(worker, "pin_mmap_region", fail)
    monkeypatch.setattr(worker, "SingleDirectionOffloadingHandler", fail)
    if failure == "view":
        monkeypatch.setattr(regions[2], "create_next_view", fail)
    try:
        with pytest.raises(RuntimeError, match="injected initialization failure"):
            worker.CpuGpuOffloadingHandlers(
                caches,
                2,
                2,
                group_page_sizes={0: 2048, 2: 2048},
                group_mmap_regions=regions,
            )
        assert all(mapping.closed for mapping in mappings)
        for region in regions.values():
            assert region.fd is None
            assert region.mmap_obj is None
            assert not Path(region.mmap_path).exists()
    finally:
        for region in regions.values():
            region.cleanup()


@pytest.mark.parametrize("policy", ["lru", "arc"])
@pytest.mark.parametrize("blocked_first", [True, False])
def test_deferred_group_releases_reservations_and_retries(policy, blocked_first):
    m = manager(capacity=1, policy=policy)
    pinned = key(5, 1)
    store(m, [pinned])
    m.prepare_load([pinned], CTX)
    keys = [key(5, 2), key(0, 2)]
    if not blocked_first:
        keys.reverse()
    for _ in range(3):
        assert m.prepare_store(keys, CTX) is None
        assert all(m.lookup(k, CTX) is False for k in keys)
        assert m.lookup(pinned, CTX) is True
    m.complete_load([pinned], CTX)
    out = store(m, keys)
    assert out.keys_to_store == keys
    assert all(m.lookup(k, CTX) is True for k in keys)


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_deferred_store_preserves_existing_and_inflight_keys(policy):
    m = manager(capacity=3, policy=policy)
    ready, inflight, fresh = [key(0, i) for i in range(3)]
    pinned = [key(5, i) for i in range(3)]
    store(m, [ready, *pinned])
    m.prepare_store([inflight], CTX)
    m.prepare_load(pinned, CTX)
    assert m.prepare_store([ready, inflight, fresh, key(5, 3)], CTX) is None
    assert m.lookup(ready, CTX) is True
    assert m.lookup(inflight, CTX) is None
    assert m.lookup(fresh, CTX) is False
    m.complete_store([inflight], CTX)
    assert m.lookup(inflight, CTX) is True
    m.complete_load(pinned, CTX)
    assert store(m, [fresh, key(5, 3)]).keys_to_store == [fresh, key(5, 3)]
