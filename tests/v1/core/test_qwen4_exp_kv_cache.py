# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import (
    _get_csa_linear_tensor_layout,
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.utils import AttentionGroup


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
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
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
