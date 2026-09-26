# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.common.qsa_cache import (
    QSAStateBackend,
    qsa_dcp_block_geometry,
)
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
from vllm.v1.worker.block_table import MultiGroupBlockTable
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


def test_qwen4_exp_real_dcp_cache_geometry() -> None:
    config = _vllm_config()
    config.cache_config.block_size = 1600
    config.parallel_config.decode_context_parallel_size = 2
    target = qsa_dcp_block_geometry(config, "model.layers.3.self_attn")
    draft = qsa_dcp_block_geometry(config, "mtp.layers.48.self_attn")
    # A block spans block_size global tokens at every DCP size; each rank holds
    # half of a sharded target page, and the replicated draft a whole one.
    assert target == (800, 1600, True)
    assert draft == (1600, 1600, False)
    config.parallel_config.decode_context_parallel_size = 1
    assert qsa_dcp_block_geometry(config, "model.layers.3.self_attn") == (
        1600,
        1600,
        True,
    )


def test_qwen4_exp_worker_block_table_respects_replicated_dcp_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.worker import block_table as block_table_module

    monkeypatch.setattr(
        block_table_module,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    monkeypatch.setattr(block_table_module, "get_total_cp_world_size", lambda: 2)
    tables = MultiGroupBlockTable(
        max_num_reqs=1,
        max_model_len=256,
        max_num_batched_tokens=4,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 16],
        kernel_block_sizes=[16, 16],
        dcp_sharded=[True, False],
    )
    assert [t.max_num_blocks_per_req for t in tables.block_tables] == [8, 16]
    assert [(t.dcp_world_size, t.dcp_rank) for t in tables.block_tables] == [
        (2, 1),
        (1, 0),
    ]


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


@pytest.mark.parametrize(
    "dcp,target_slots,expected_tokens",
    [
        (1, 1600, 775_096),
        # 1600 target slots per rank: a 3200-token span, one page per layer.
        (2, 1600, 1_178_375),
        # 800 target slots per rank: DCP1's 1600-token span, two target
        # layers per physical page, six GDN groups over seven owners.
        (2, 800, 1_215_037),
    ],
)
def test_qwen4_exp_capacity_projection_runs_through_allocator(
    dcp, target_slots, expected_tokens
):
    config = _vllm_config()
    config.model_config.max_model_len = 262_144
    config.parallel_config.decode_context_parallel_size = dcp
    config.cache_config.mamba_cache_mode = "align"
    span = target_slots * dcp
    specs = {}
    for layer in range(13):
        name = f"model.layers.{layer}.self_attn"
        specs[name] = FullAttentionSpec(
            block_size=target_slots if layer < 12 else span,
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
        # One real MTP3 GDN state: 817,152 bytes.
        specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
            block_size=span,
            shapes=((1, 408_576),),
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
    # The allocator pads the states exactly as it does for the measured
    # 12 QSA / 36 GDN / 1 draft.
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


def test_qwen4_exp_dcp2_selector_canonicalizes_local_group_expansion() -> None:
    config = _vllm_config()
    config.parallel_config.decode_context_parallel_size = 2
    spec = MLAAttentionSpec(
        block_size=3200,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=16,
        dcp_sharded=False,
    )
    group = AttentionGroup(
        QSAStateBackend,
        ["model.layers.3.self_attn.indexer.compressed_key_cache"],
        spec,
        kv_cache_group_id=0,
    )
    group.create_metadata_builders(config, torch.device("cpu"), kernel_block_size=16)
    builder = group.get_metadata_builder()
    physical = torch.tensor([4, 7, 9], dtype=torch.int32)
    expansion = 1600 // 16
    expanded = (
        physical[:, None] * expansion + torch.arange(expansion, dtype=torch.int32)
    ).reshape(1, -1)
    assert torch.equal(builder._canonical_block_table(expanded), physical[None])


def _packed_dcp_specs(span: int = 16, state_width: int = 5000):
    """DCP2 specs for a ``span``-token block: two sharded target layers whose
    half pages cannot hold a recurrent state alone, one replicated draft
    layer, and states between the two page sizes."""
    specs = {}
    for prefix, sharded in (
        ("model.layers.3.self_attn", True),
        ("model.layers.7.self_attn", True),
        ("mtp.layers.8.self_attn", False),
    ):
        specs[prefix] = FullAttentionSpec(
            block_size=span // 2 if sharded else span,
            num_kv_heads=1,
            head_size=256,
            head_size_v=256,
            dtype=torch.float16,
            dcp_sharded=sharded,
        )
        specs[f"{prefix}.compressed"] = MLAAttentionSpec(
            block_size=span,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            compress_ratio=4,
            dcp_sharded=False,
        )
        specs[f"{prefix}.compressor_state"] = CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float16,
            dcp_sharded=False,
        )
    for layer in (0, 1, 2, 4, 5, 6):
        specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
            block_size=span,
            shapes=((1, state_width),),
            dtypes=(torch.float16,),
            dcp_sharded=False,
        )
    specs["model.layers.2.ple"] = MambaSpec(
        block_size=span,
        shapes=((1, 64),),
        dtypes=(torch.float16,),
        tp_replicated=True,
        dcp_sharded=False,
    )
    return specs


def test_qwen4_exp_dcp_packs_sharded_pages_that_cannot_hold_a_state() -> None:
    config = _vllm_config()
    config.parallel_config.decode_context_parallel_size = 2
    config.cache_config.num_gpu_blocks_override = 3
    groups = get_kv_cache_groups(config, _packed_dcp_specs())
    layout = _get_csa_linear_tensor_layout(groups)
    assert layout is not None
    # 8 KiB sharded pages are packed in pairs; the 16 KiB draft page is alone.
    assert layout.main_kv_page_sizes == [8_192, 8_192, 16_384]
    assert layout.main_kv_owners == [[0, 1], [2]]
    assert [layout.owner_page_size(i) for i in range(2)] == [16_384, 16_384]
    # Six recurrent states over two physical owners need three groups.
    gdn_groups = [
        g for g in groups if any(n.endswith("linear_attn") for n in g.layer_names)
    ]
    assert len(gdn_groups) == 3
    caches = get_kv_cache_config_from_groups(config, groups, available_memory=1 << 30)
    main = caches.kv_cache_tensors[:2]
    assert main[0].packed_members == [
        "model.layers.3.self_attn",
        "model.layers.7.self_attn",
    ]
    assert main[1].packed_members is None
    assert [t.size for t in main] == [16_384 * 3, 16_384 * 3]
    # Packing does not change the bytes a pool block costs.
    assert sum(t.size for t in caches.kv_cache_tensors) == (
        layout.bytes_per_block * caches.num_blocks
    )


@pytest.mark.parametrize("cache_layout", ["NHD", "HND"])
def test_qwen4_exp_packed_members_interleave_kernel_blocks(
    monkeypatch, cache_layout: str
) -> None:
    """Both members of a packed page tile it exactly, one kernel block at a
    time, through the real worker reshape path. The QSA target backend keeps
    FlashAttention's block-outermost cache shape in either layout."""
    from vllm.v1.attention.backends import flash_attn
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: cache_layout)

    config = _vllm_config()
    config.parallel_config.decode_context_parallel_size = 2
    config.cache_config.num_gpu_blocks_override = 3
    # 32 local slots per sharded page (32 KiB), 64 KiB draft page, 40 KB state.
    specs = _packed_dcp_specs(span=64, state_width=20_000)
    groups = get_kv_cache_groups(config, specs)
    caches = get_kv_cache_config_from_groups(config, groups, available_memory=1 << 30)
    tensor = caches.kv_cache_tensors[0]
    members = tensor.packed_members
    assert members is not None
    raw = torch.zeros(tensor.size, dtype=torch.int8)
    spec = specs[members[0]]
    kernel_block = 16
    views = _reshape_kv_cache(
        attn_groups=[AttentionGroup(FlashAttentionBackend, members, spec, 0)],
        kv_cache_raw_tensors={name: raw for name in members},
        cache_dtype="auto",
        kernel_block_sizes=[kernel_block],
        shared_kv_cache_layers={},
        packed_members={name: (i, len(members)) for i, name in enumerate(members)},
    )
    first, second = (views[name] for name in members)
    # Three physical blocks of 32 local slots are six 16-slot kernel blocks.
    assert first.shape[0] == second.shape[0] == 3 * 32 // kernel_block
    kernel_bytes = 2 * kernel_block * 256 * 2  # K and V, one head, fp16
    assert first.stride(0) * first.element_size() == 2 * kernel_bytes
    assert second.storage_offset() * second.element_size() == kernel_bytes

    first.fill_(1)
    second.fill_(2)
    chunks = raw.view(torch.float16).view(-1, kernel_bytes // 2)
    # The members alternate chunk by chunk and together cover every byte.
    assert torch.equal(chunks[0::2], torch.ones_like(chunks[0::2]))
    assert torch.equal(chunks[1::2], torch.full_like(chunks[1::2], 2))

    # A single kernel-block write lands at its interleaved offset: member 1's
    # kernel block 3 is chunk 3 * 2 + 1, inside physical block 1, which is
    # the pool block the block table expands kernel blocks 2 and 3 from.
    raw.zero_()
    second[3].fill_(5)
    written = (chunks != 0).any(dim=1).nonzero().flatten().tolist()
    assert written == [3 * 2 + 1]
    chunks_per_physical_block = tensor.size // 3 // kernel_bytes
    assert written[0] // chunks_per_physical_block == 3 // (32 // kernel_block)


def test_qwen4_exp_offload_registers_a_packed_page_once(monkeypatch) -> None:
    """The offload worker copies a packed physical page as one reference that
    covers every member's kernel blocks of that pool block."""
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.attention.backends import flash_attn
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.kv_offload.base import OffloadingSpec

    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: "NHD")
    config = _vllm_config()
    config.parallel_config.decode_context_parallel_size = 2
    config.cache_config.num_gpu_blocks_override = 3
    specs = _packed_dcp_specs(span=64, state_width=20_000)
    groups = get_kv_cache_groups(config, specs)
    caches = get_kv_cache_config_from_groups(config, groups, available_memory=1 << 30)
    layout = _get_csa_linear_tensor_layout(groups)
    assert layout is not None
    owner_tensors = caches.kv_cache_tensors[: len(layout.main_kv_owners)]
    packed = {
        name: (index, len(t.packed_members))
        for t in owner_tensors
        for index, name in enumerate(t.packed_members or [])
    }
    # Main K/V and recurrent states go through the real worker reshape path;
    # the side caches only need to expose their storage.
    kv_caches = {}
    for tensor in caches.kv_cache_tensors:
        raw = torch.zeros(tensor.size, dtype=torch.int8)
        for group_id, group in enumerate(groups):
            spec = group.kv_cache_spec
            layer_specs = (
                spec.kv_cache_specs
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else {name: spec for name in group.layer_names}
            )
            for name in set(tensor.shared_by) & set(layer_specs):
                layer_spec = layer_specs[name]
                if type(layer_spec) is FullAttentionSpec:
                    backend = FlashAttentionBackend
                elif isinstance(layer_spec, MambaSpec):
                    backend = QSAStateBackend
                else:
                    kv_caches[name] = raw
                    continue
                kv_caches.update(
                    _reshape_kv_cache(
                        attn_groups=[AttentionGroup(backend, [name], layer_spec, 0)],
                        kv_cache_raw_tensors={name: raw},
                        cache_dtype="auto",
                        kernel_block_sizes=[16],
                        shared_kv_cache_layers={},
                        packed_members=packed,
                    )
                )

    spec = MagicMock(spec=OffloadingSpec)
    spec.kv_cache_config = caches
    spec.vllm_config = MagicMock()
    spec.get_handlers.return_value = iter([])
    worker = OffloadingConnectorWorker(spec=spec)
    worker.worker = MagicMock()
    worker.register_kv_caches(kv_caches)
    canonical = spec.get_handlers.call_args[0][0]

    page = 65_536
    # The two physical owners come first, one canonical tensor each.
    assert [t.tensor.shape for t in canonical.tensors[:2]] == [(3, page), (3, page)]
    main_names = set(layout.main_kv_names)
    main_group = next(
        i for i, g in enumerate(groups) if main_names & set(g.layer_names)
    )
    main_refs = [r for r in canonical.group_data_refs[main_group] if r.tensor_idx < 2]
    # One whole-page reference per physical owner, not one per layer.
    assert [(r.tensor_idx, r.page_size_bytes) for r in main_refs] == [
        (0, page),
        (1, page),
    ]
    assert sum(r.page_size_bytes for r in main_refs) == sum(layout.main_kv_page_sizes)

    # Both members' kernel blocks of pool block 1 (kernel blocks 2 and 3 at 32
    # local slots per page) lie inside the bytes the reference moves.
    first, second = (kv_caches[name] for name in owner_tensors[0].packed_members)
    first[2:4].fill_(1)
    second[2:4].fill_(2)
    block = canonical.tensors[0].tensor[1]
    assert torch.count_nonzero(canonical.tensors[0].tensor[0]) == 0
    assert torch.count_nonzero(canonical.tensors[0].tensor[2]) == 0
    values = block.view(torch.float16)
    assert torch.count_nonzero(values) == values.numel()
