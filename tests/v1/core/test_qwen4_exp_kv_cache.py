# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    _get_csa_linear_tensor_layout,
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_max_concurrency_for_kv_cache_config,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.utils import AttentionGroup

pytestmark = pytest.mark.skip_global_cleanup


class _ModelConfig:
    max_model_len = 8192

    def get_num_kv_heads(self, parallel_config) -> int:
        del parallel_config
        return 1

    def get_total_num_hidden_layers(self) -> int:
        return 8


def _vllm_config():
    return SimpleNamespace(
        model_config=_ModelConfig(),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=8192,
            max_num_seqs=2,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            mamba_cache_mode="none",
        ),
    )


def _qwen4_exp_cache_specs():
    specs = {}
    for layer in (3, 7):
        prefix = f"model.layers.{layer}.self_attn"
        specs[prefix] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=256,
            head_size_v=256,
            dtype=torch.float16,
        )
        specs[f"{prefix}.compressed"] = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            compress_ratio=4,
        )
        specs[f"{prefix}.compressor_state"] = CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float16,
        )

    for layer in (0, 1, 2, 4, 5, 6):
        specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
            block_size=16,
            shapes=((1, 64),),
            dtypes=(torch.float16,),
        )
    specs["model.layers.2.ple"] = MambaSpec(
        block_size=16,
        shapes=((1, 64),),
        dtypes=(torch.float16,),
        tp_replicated=True,
    )
    return specs


def test_qwen4_exp_csa_linear_cache_layout() -> None:
    groups = get_kv_cache_groups(_vllm_config(), _qwen4_exp_cache_specs())
    layout = _get_csa_linear_tensor_layout(groups)

    assert layout is not None
    assert [len(group.layer_names) for group in groups] == [4, 2, 2, 2, 2, 1]
    assert len(layout.main_kv_names) == 2
    assert len(layout.compressed_names) == 2
    assert len(layout.compressor_state_names) == 2
    assert len(layout.mamba_groups) == 4

    cache_config = get_kv_cache_config_from_groups(
        _vllm_config(), groups, available_memory=1 << 30
    )
    assert len(cache_config.kv_cache_tensors) == 4
    assert all(len(tensor.shared_by) >= 2 for tensor in cache_config.kv_cache_tensors)

    scheduler_config = generate_scheduler_kv_cache_config([cache_config])
    scheduler_config.num_blocks = 32
    assert isinstance(
        scheduler_config.kv_cache_groups[0].kv_cache_spec, FullAttentionSpec
    )
    assert isinstance(
        scheduler_config.kv_cache_groups[1].kv_cache_spec, CircularBufferSpec
    )
    coordinator = get_kv_cache_coordinator(
        scheduler_config,
        max_model_len=8192,
        max_in_flight_tokens=128,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=4,
    )
    assert isinstance(coordinator.single_type_managers[1], CircularBufferManager)

    prefix_coordinator = get_kv_cache_coordinator(
        scheduler_config,
        max_model_len=8192,
        max_in_flight_tokens=128,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=4,
    )
    assert all(
        not isinstance(spec, CircularBufferSpec)
        for spec, _, _ in prefix_coordinator.attention_groups
    )


def test_qwen4_exp_circular_cache_stores_keys_without_unused_values() -> None:
    spec = CircularBufferSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.float16,
    )

    assert spec.real_page_size_bytes == 4 * 128 * 2
    assert spec.max_memory_usage_bytes(_vllm_config()) == spec.page_size_bytes


def _mixed_dcp_specs():
    specs = _qwen4_exp_cache_specs()
    for name, spec in list(specs.items()):
        if type(spec) is FullAttentionSpec:
            # Layer 3 is the target; layer 7 stands in for the replicated draft.
            sharded = "layers.3." in name
            specs[name] = replace(
                spec, block_size=16 if sharded else 32, dcp_sharded=sharded
            )
        elif type(spec) is MLAAttentionSpec:
            specs[name] = replace(spec, block_size=32, dcp_sharded=False)
        else:
            specs[name] = replace(spec, dcp_sharded=False)
    return specs


def test_qwen4_exp_mixed_dcp_pages_preserve_shared_state_stride() -> None:
    config = _vllm_config()
    config.parallel_config.decode_context_parallel_size = 2
    config.cache_config.num_gpu_blocks_override = 3
    groups = get_kv_cache_groups(config, _mixed_dcp_specs())
    layout = _get_csa_linear_tensor_layout(groups)
    assert layout is not None
    assert layout.main_kv_page_sizes == [16_384, 32_768]
    assert layout.compressed_page_sizes == [2_048, 2_048]
    assert layout.bytes_per_block == 53_248
    caches = get_kv_cache_config_from_groups(config, groups, available_memory=1 << 30)
    assert sum(t.size for t in caches.kv_cache_tensors) == 3 * 53_248

    # Use the real worker reshape path. Every owner must see exactly three
    # blocks, and a write into block 1 must start at its shared tensor's page.
    members = {}
    for group_id, group in enumerate(groups):
        spec = group.kv_cache_spec
        specs = (
            spec.kv_cache_specs
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else {name: spec for name in group.layer_names}
        )
        members.update({name: (group_id, s) for name, s in specs.items()})
    for tensor in caches.kv_cache_tensors:
        raw = torch.zeros(tensor.size, dtype=torch.int8)
        for name in tensor.shared_by:
            group_id, spec = members[name]
            assert tensor.size == spec.page_size_bytes * 3
            if not isinstance(spec, MambaSpec):
                continue
            views = _reshape_kv_cache(
                attn_groups=[AttentionGroup(QSAStateBackend, [name], spec, group_id)],
                kv_cache_raw_tensors={name: raw},
                cache_dtype="auto",
                kernel_block_sizes=[16] * len(groups),
                shared_kv_cache_layers={},
            )[name]
            assert views[0].shape[0] == 3
            assert views[0].stride(0) * views[0].element_size() == tensor.size // 3
            raw.zero_()
            views[0][1].fill_(1)
            assert torch.count_nonzero(raw[: spec.page_size_bytes]) == 0
            assert (
                torch.count_nonzero(
                    raw[spec.page_size_bytes : 2 * spec.page_size_bytes]
                )
                > 0
            )
            assert torch.count_nonzero(raw[2 * spec.page_size_bytes :]) == 0

    scheduler = generate_scheduler_kv_cache_config([caches])
    assert scheduler.kv_cache_groups[0].kv_cache_spec.global_block_size(2) == 32
    assert all(not g.kv_cache_spec.dcp_sharded for g in scheduler.kv_cache_groups[2:])


@pytest.mark.parametrize("reverse", [False, True])
def test_qwen4_exp_dcp_group_uses_global_spans(reverse) -> None:
    specs = _mixed_dcp_specs()
    target = specs["model.layers.3.self_attn"]
    draft = specs["model.layers.7.self_attn"]
    members = {"target": target, "draft": draft}
    if reverse:
        members = dict(reversed(list(members.items())))
    uniform = UniformTypeKVCacheSpecs.from_specs(members, dcp_world_size=2)
    assert uniform is not None
    assert uniform.block_size == 16
    assert uniform.dcp_sharded
    assert uniform.global_block_size(2) == 32
    # Equal physical slots are insufficient if their global spans differ.
    assert (
        UniformTypeKVCacheSpecs.from_specs(
            {"target": target, "draft": replace(draft, block_size=16)}, dcp_world_size=2
        )
        is None
    )


@pytest.mark.parametrize("spec_cls", [FullAttentionSpec, MLAAttentionSpec])
def test_qwen4_exp_replicated_cache_memory_and_merge(spec_cls) -> None:
    config = _vllm_config()
    config.parallel_config.decode_context_parallel_size = 2
    spec = spec_cls(
        block_size=16,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.float16,
        dcp_sharded=False,
    )
    assert spec.max_memory_usage_bytes(config) == 512 * spec.page_size_bytes
    assert (
        replace(spec, dcp_sharded=True).max_memory_usage_bytes(config)
        == 256 * spec.page_size_bytes
    )
    assert not spec_cls.merge([spec, spec]).dcp_sharded
    with pytest.raises(AssertionError):
        spec_cls.merge([spec, replace(spec, dcp_sharded=True)])


@pytest.mark.parametrize("dcp,expected_tokens", [(1, 775_096), (2, 1_178_375)])
def test_qwen4_exp_capacity_projection_runs_through_allocator(dcp, expected_tokens):
    config = _vllm_config()
    config.model_config.max_model_len = 262_144
    config.parallel_config.decode_context_parallel_size = dcp
    config.cache_config.mamba_cache_mode = "align"
    span = 1600 * dcp
    specs = {}
    for layer in range(13):
        name = f"model.layers.{layer}.self_attn"
        specs[name] = FullAttentionSpec(
            block_size=1600 if layer < 12 else span,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.uint8,
            dcp_sharded=layer < 12,
        )
        specs[name + ".compressed"] = MLAAttentionSpec(
            block_size=span,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            compress_ratio=4,
            dcp_sharded=False,
        )
        specs[name + ".compressor_state"] = CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float16,
            dcp_sharded=False,
        )
    for layer in range(36):
        specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
            block_size=span,
            shapes=((1, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=3,
        )
    specs["model.layers.0.ple"] = MambaSpec(
        block_size=span,
        shapes=((1, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
        tp_replicated=True,
    )
    # State shapes are synthetic but fit each real QSA page; the allocator
    # pads them exactly as it does for the measured 12 QSA / 36 GDN / 1 draft.
    groups = get_kv_cache_groups(config, specs)
    cache = get_kv_cache_config_from_groups(
        config, groups, available_memory=547 * 11_980_800
    )
    concurrency = get_max_concurrency_for_kv_cache_config(config, cache)
    assert int(concurrency * 262_144) == expected_tokens


def test_dcp_prefix_hit_respects_target_draft_and_state_ownership():
    from tests.v1.core.test_prefix_caching import make_request

    init_none_hash(sha256)
    target = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    draft = replace(target, dcp_sharded=False)
    state = MambaSpec(
        block_size=16, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="all"
    )
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["target"], target),
            KVCacheGroupSpec(["draft"], draft),
            KVCacheGroupSpec(["state"], state),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=128,
        hash_block_size=16,
        dcp_world_size=2,
        enable_caching=True,
    )
    owners = manager.coordinator.single_type_managers
    assert [m.block_size for m in owners] == [32, 16, 16]
    assert [m.dcp_world_size for m in owners] == [2, 1, 1]
    common = list(range(64))
    first = make_request("first", common + [91] * 5, 16, sha256)
    assert manager.allocate_slots(first, 69) is not None
    second = make_request("second", common + [92] * 5, 16, sha256)
    blocks, tokens = manager.get_computed_blocks(second)
    assert tokens == 64
    assert [len(group) for group in blocks.blocks] == [2, 4, 4]
    manager.free(first)


def test_dcp_offload_worker_and_scheduler_keep_the_same_group_budget():
    config = _vllm_config()
    config.parallel_config.decode_context_parallel_size = 2
    config.parallel_config.world_size = 4
    config.cache_config.block_size = 16
    config.cache_config.hash_block_size = 4
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_cache_retention_interval = 0
    config.cache_config.mamba_cache_mode = "align"
    config.kv_transfer_config = SimpleNamespace(
        kv_connector_extra_config={"cpu_bytes_to_use": 16 * 1024**2}
    )
    specs = _mixed_dcp_specs()
    for name, spec in list(specs.items()):
        if isinstance(spec, MambaSpec):
            specs[name] = replace(
                spec, block_size=32, mamba_cache_mode="align", num_speculative_blocks=3
            )
    groups = get_kv_cache_groups(config, specs)
    worker_cache = get_kv_cache_config_from_groups(
        config, groups, available_memory=1 << 20
    )
    scheduler_cache = generate_scheduler_kv_cache_config([worker_cache])
    worker = CPUOffloadingSpec(config, worker_cache)
    scheduler = CPUOffloadingSpec(config, scheduler_cache)
    assert worker.gpu_block_size == scheduler.gpu_block_size == (32, 4, 32, 32, 32, 32)
    assert worker.hash_block_size == scheduler.hash_block_size == 4
    assert worker.partition_by_group and scheduler.partition_by_group
    assert worker.cpu_group_page_sizes == scheduler.cpu_group_page_sizes
    assert worker.cpu_group_num_blocks == scheduler.cpu_group_num_blocks
    assert worker.num_blocks > 0
    used = (
        sum(
            worker.cpu_group_page_sizes[i] * n
            for i, n in worker.cpu_group_num_blocks.items()
        )
        * 4
    )
    assert used <= 16 * 1024**2


def test_qwen4_exp_circular_manager_owns_one_block_per_request() -> None:
    spec = CircularBufferSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.float16,
    )
    block_pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=False,
        hash_block_size=spec.block_size,
    )
    manager = CircularBufferManager(
        spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
    )

    assert manager.get_num_blocks_to_allocate("req", 4096, (), 0, 4096) == 1
    blocks = manager.allocate_new_blocks("req", 4096, 4096)
    assert len(blocks) == 1
    assert manager.req_to_blocks["req"] == blocks
    assert manager.get_num_blocks_to_allocate("req", 8192, (), 4096, 8192) == 0
    assert manager.allocate_new_blocks("req", 8192, 8192) == []


def test_qwen4_exp_compressed_qsa_reshape_uses_storage_block_size() -> None:
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
    )
    num_blocks = 3
    raw = torch.empty(num_blocks * spec.page_size_bytes, dtype=torch.int8)
    group = AttentionGroup(
        QSAStateBackend,
        ["compressed"],
        spec,
        kv_cache_group_id=0,
    )

    caches = _reshape_kv_cache(
        attn_groups=[group],
        kv_cache_raw_tensors={"compressed": raw},
        cache_dtype="auto",
        kernel_block_sizes=[16],
        shared_kv_cache_layers={},
    )

    assert caches["compressed"].shape == (num_blocks, 4, 1, 128)
    assert caches["compressed"].untyped_storage().data_ptr() == raw.data_ptr()


def test_qwen4_exp_qsa_metadata_canonicalizes_expanded_block_table() -> None:
    spec = MLAAttentionSpec(
        block_size=784,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=8,
    )
    group = AttentionGroup(
        QSAStateBackend,
        ["compressed"],
        spec,
        kv_cache_group_id=0,
    )
    group.create_metadata_builders(
        _vllm_config(), torch.device("cpu"), kernel_block_size=16
    )
    builder = group.get_metadata_builder()

    # The QSA builder must retain the actual 98-row cache page rather than the
    # generic 32-row paged-MQA virtualization used by other compressed backends.
    assert builder.kv_cache_spec.block_size == 784
    assert builder.kv_cache_spec.storage_block_size == 98

    expansion = 784 // 16
    # The legacy block-table path pads 11 physical pages to 16 for its
    # 128-token alignment. The persistent QSA buffer must cover that width as
    # well as the unpadded V2 table.
    physical_pages = torch.arange(16, dtype=torch.int32).mul(3).add(7)
    expanded = (
        physical_pages[:, None] * expansion + torch.arange(expansion, dtype=torch.int32)
    ).reshape(1, -1)

    canonical = builder._canonical_block_table(expanded)
    first_ptr = canonical.data_ptr()
    assert torch.equal(canonical, physical_pages[None])

    physical_pages.add_(5)
    expanded.copy_(
        (
            physical_pages[:, None] * expansion
            + torch.arange(expansion, dtype=torch.int32)
        ).reshape(1, -1)
    )
    canonical = builder._canonical_block_table(expanded)
    assert canonical.data_ptr() == first_ptr
    assert torch.equal(canonical, physical_pages[None])
