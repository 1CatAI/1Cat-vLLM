# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup


class _IndexerMetadataBuilder:
    uses_physical_block_table = False


class _IndexerBackend:
    @staticmethod
    def get_builder_cls():
        return _IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        cache_dtype_str="auto",
    ):
        del cache_dtype_str
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order():
        return (0, 1, 2)


def test_glm53_indexer_cache_uses_physical_pool_pages():
    spec = MLAAttentionSpec(
        block_size=1152,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        compress_ratio=4,
    )
    layer_name = "model.layers.3.self_attn.indexer.k_cache"
    group = AttentionGroup(
        backend=_IndexerBackend,
        layer_names=[layer_name],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )

    runner = object.__new__(GPUModelRunner)
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([layer_name], spec)],
    )
    runner.attn_groups = [[group]]
    runner.runner_only_attn_layers = set()
    runner.cache_config = SimpleNamespace(cache_dtype="auto")

    raw = torch.empty(spec.page_size_bytes, dtype=torch.int8)
    cache = runner._reshape_kv_cache_tensors({layer_name: raw}, [64])[layer_name]

    # 1152 scheduler tokens / kpool=4 = 288 stored entries, split into the
    # 32-entry pages accepted by the paged MQA kernel.
    assert cache.shape == (9, 32, 132)
    assert cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()


def test_glm53_v2_indexer_cache_uses_physical_pool_pages():
    spec = MLAAttentionSpec(
        block_size=1152,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        compress_ratio=4,
    )
    layer_name = "model.layers.3.self_attn.indexer.k_cache"
    group = AttentionGroup(
        backend=_IndexerBackend,
        layer_names=[layer_name],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    raw = torch.empty(spec.page_size_bytes, dtype=torch.int8)

    cache = _reshape_kv_cache(
        attn_groups=[group],
        kv_cache_raw_tensors={layer_name: raw},
        cache_dtype="auto",
        kernel_block_sizes=[64],
        shared_kv_cache_layers={},
    )[layer_name]

    assert cache.shape == (9, 32, 132)
    assert cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
