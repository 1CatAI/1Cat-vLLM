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
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

CTX = ReqContext("test")
GROUPS = (0, 2, 3, 4, 5)


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
    # Both group-5 slots are pinned; group 0 can still accept its own data.
    partial = m.prepare_store([key(5, 4), key(0, 4)], CTX)
    assert partial.keys_to_store == [key(0, 4)]
    m.complete_store(partial.keys_to_store, CTX, success=False)
    assert m.lookup(key(0, 4), CTX) is False
    m.complete_load(keys, CTX)
    m.touch([keys[0]], CTX)
    out = store(m, [key(5, 4)])
    assert out.evicted_keys == [keys[2]]
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


@pytest.mark.parametrize("layout", ["equal", "scheduler", "mixed", "attention_only"])
def test_actual_flash_next_group_budget(monkeypatch, layout):
    def init(self, config, caches):
        self.vllm_config = config
        self.kv_cache_config = caches
        self.extra_config = {"cpu_bytes_to_use": 16 * 1024**3}
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
    spec = CPUOffloadingSpec(
        SimpleNamespace(parallel_config=SimpleNamespace(world_size=4)),
        SimpleNamespace(
            kv_cache_groups=groups,
            num_blocks=104,
            kv_cache_tensors=tensors,
        ),
    )
    if layout not in ("equal", "scheduler"):
        assert not spec.partition_by_group
        assert spec.num_blocks == 419
        return
    assert spec.cpu_group_page_sizes == {
        0: 10235904,
        2: 9633792,
        3: 9633792,
        4: 9633792,
        5: 802816,
    }
    assert spec.num_blocks == 107
    assert spec.cpu_page_size_per_worker * 4 * spec.num_blocks <= 16 * 1024**3
    assert spec.num_blocks >= 2 * 48


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
