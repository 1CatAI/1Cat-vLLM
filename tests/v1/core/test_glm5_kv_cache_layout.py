# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)


def test_kpool_tail_admission_uses_in_flight_keyword() -> None:
    spec = KpoolTailSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=256,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=4,
    )

    assert (
        spec.max_admission_blocks_per_request(
            max_in_flight_tokens=2048,
            max_model_len=32768,
        )
        == 1
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


def test_glm53_dflash_keeps_compressed_layout_and_allocates_draft_pages():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=4096,
            get_total_num_hidden_layers=lambda: 4,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            num_gpu_blocks_override=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=128,
            disable_hybrid_kv_cache_manager=False,
        ),
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
    )
    block_size = 64
    specs = {
        "model.layers.0.mla_attn": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
        ),
        "model.layers.0.indexer.k_cache": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            compress_ratio=4,
        ),
        "model.layers.1.self_attn": MambaSpec(
            block_size=block_size,
            shapes=((1024,),),
            dtypes=(torch.uint8,),
        ),
    }
    draft_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        dtype=torch.float16,
        sliding_window=2048,
    )
    for layer_idx in range(5):
        specs[f"speculator.model.layers.{layer_idx}.self_attn.attn"] = draft_spec

    groups = kv_cache_utils.get_kv_cache_groups(vllm_config, specs)
    layout = kv_cache_utils._glm5_next_tensor_layout(groups)

    assert layout is not None
    auxiliary_groups = kv_cache_utils._glm5_next_auxiliary_attention_groups(groups)
    assert len(auxiliary_groups) == 1
    assert len(auxiliary_groups[0].layer_names) == 5

    _, _, mla_names, idx_names, mla_page, idx_page, _, _ = layout
    bytes_per_block = mla_page + idx_page + 5 * draft_spec.page_size_bytes
    assert kv_cache_utils._pool_bytes_per_block(groups) == bytes_per_block

    cache_config = kv_cache_utils.get_kv_cache_config_from_groups(
        vllm_config, groups, bytes_per_block * 100
    )
    assert cache_config.num_blocks == 100
    assert len(cache_config.kv_cache_tensors) == len(mla_names) + len(idx_names) + 5
    draft_tensors = [
        tensor
        for tensor in cache_config.kv_cache_tensors
        if tensor.shared_by[0].startswith("speculator.")
    ]
    assert len(draft_tensors) == 5
    assert all(
        tensor.size == draft_spec.page_size_bytes * 100 for tensor in draft_tensors
    )
