# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    MambaSpec,
    MLAAttentionSpec,
)


def test_glm53_pp2_kpool_tail_shares_indexer_storage(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    model_config = SimpleNamespace(
        max_model_len=32768,
        get_total_num_hidden_layers=lambda: 45,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            num_gpu_blocks_override=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048,
            disable_hybrid_kv_cache_manager=False,
        ),
    )

    block_size = 1152
    mamba_block_size = 4096
    mamba_spec = MambaSpec(
        shapes=((3, 2048), (3, 2048), (3, 2048), (16, 128, 128)),
        dtypes=(torch.float16, torch.float16, torch.float16, torch.float32),
        block_size=mamba_block_size,
    )
    specs = {}
    for layer_idx in range(45):
        prefix = f"model.layers.{layer_idx}.self_attn"
        if layer_idx % 4 == 3:
            specs[f"{prefix}.mla_attn"] = MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.float16,
            )
            specs[f"{prefix}.indexer.k_cache"] = MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=132,
                dtype=torch.uint8,
                compress_ratio=4,
            )
            specs[f"{prefix}.indexer.tail_cache"] = KpoolTailSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=256,
                head_size_v=0,
                dtype=torch.bfloat16,
                sliding_window=4,
            )
        else:
            specs[prefix] = mamba_spec

    groups = kv_cache_utils.get_kv_cache_groups(vllm_config, specs)
    layout = kv_cache_utils._glm5_next_tensor_layout(groups)
    assert layout is not None
    _, mamba_groups, mla_names, idx_names, mla_page, idx_page, tail_names, _ = layout
    assert len(mamba_groups) == 4
    assert [len(group.layer_names) for group in mamba_groups] == [9, 9, 8, 8]
    assert all(
        group.kv_cache_spec.block_size == mamba_block_size for group in mamba_groups
    )
    assert len(mla_names) == len(idx_names) == len(tail_names) == 11
    participating_block_sizes = [
        group.kv_cache_spec.block_size
        for group in groups
        if group.kv_cache_spec.prefix_cacheable
    ]
    assert min(participating_block_sizes) == block_size
    assert min(group.kv_cache_spec.block_size for group in groups) == 4

    bytes_per_block = 11 * (mla_page + idx_page)
    cache_config = kv_cache_utils.get_kv_cache_config_from_groups(
        vllm_config, groups, bytes_per_block * 100
    )
    assert cache_config.num_blocks == 100
    assert len(cache_config.kv_cache_tensors) == 22
    for idx_name, tail_name in zip(idx_names, tail_names):
        tensor = next(
            t for t in cache_config.kv_cache_tensors if idx_name in t.shared_by
        )
        assert tensor.shared_by == [idx_name, tail_name]
