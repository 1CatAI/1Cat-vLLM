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


def test_glm53_dflash_packs_only_draft_pages_that_fit_mla_slots():
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
    auxiliary_spec = auxiliary_groups[0].kv_cache_spec
    assert auxiliary_spec.block_size == 32
    assert auxiliary_spec.page_size_bytes == mla_page
    bytes_per_block = mla_page + idx_page + 4 * mla_page
    assert kv_cache_utils._pool_bytes_per_block(groups) == bytes_per_block

    cache_config = kv_cache_utils.get_kv_cache_config_from_groups(
        vllm_config, groups, bytes_per_block * 100
    )
    assert cache_config.num_blocks == 100
    assert len(cache_config.kv_cache_tensors) == len(mla_names) + len(idx_names) + 4
    packed_draft_tensors = [
        tensor
        for tensor in cache_config.kv_cache_tensors
        if tensor.shared_by[0] in mla_names
        and any(name.startswith("speculator.") for name in tensor.shared_by)
    ]
    assert len(packed_draft_tensors) == 1
    draft_tensors = [
        tensor
        for tensor in cache_config.kv_cache_tensors
        if tensor.shared_by[0].startswith("speculator.")
    ]
    assert len(draft_tensors) == 4
    assert all(tensor.size == mla_page * 100 for tensor in draft_tensors)


def test_glm53_dflash_tp4_reblocks_and_packs_all_draft_pages():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=262144,
            get_total_num_hidden_layers=lambda: 21,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        cache_config=SimpleNamespace(
            mamba_cache_mode="none",
            num_gpu_blocks_override=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=128,
            disable_hybrid_kv_cache_manager=False,
        ),
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
    )
    target_block_size = 2304
    specs = {}
    for layer_idx in range(5):
        prefix = f"model.layers.{4 * layer_idx + 3}.self_attn"
        specs[f"{prefix}.mla_attn"] = MLAAttentionSpec(
            block_size=target_block_size,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            cache_dtype_str="fp8_e4m3",
            model_version="glm5_next",
        )
        specs[f"{prefix}.indexer.k_cache"] = MLAAttentionSpec(
            block_size=target_block_size,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            compress_ratio=4,
        )
        specs[f"model.layers.{4 * layer_idx}.self_attn"] = MambaSpec(
            block_size=262144,
            shapes=((1198080,),),
            dtypes=(torch.uint8,),
            num_speculative_blocks=7,
        )
        specs[f"speculator.model.layers.{layer_idx}.self_attn.attn"] = (
            SlidingWindowSpec(
                block_size=target_block_size,
                num_kv_heads=2,
                head_size=128,
                head_size_v=128,
                dtype=torch.float16,
                sliding_window=2048,
            )
        )

    groups = kv_cache_utils.get_kv_cache_groups(vllm_config, specs)
    layout = kv_cache_utils._glm5_next_tensor_layout(groups)
    assert layout is not None
    _, _, mla_names, idx_names, mla_page, idx_page, _, _ = layout
    auxiliary_group = kv_cache_utils._glm5_next_auxiliary_attention_groups(groups)[0]
    assert auxiliary_group.kv_cache_spec.block_size == 1152
    assert auxiliary_group.kv_cache_spec.page_size_bytes == mla_page == 1198080

    bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
    assert bytes_per_block == 6370560
    assert kv_cache_utils._pool_bytes_per_block(groups) == bytes_per_block

    cache_config = kv_cache_utils.get_kv_cache_config_from_groups(
        vllm_config, groups, bytes_per_block * 200
    )
    assert cache_config.num_blocks == 200
    assert len(cache_config.kv_cache_tensors) == len(mla_names) + len(idx_names)
    for layer_idx, mla_name in enumerate(mla_names):
        tensor = next(
            tensor
            for tensor in cache_config.kv_cache_tensors
            if mla_name in tensor.shared_by
        )
        assert f"speculator.model.layers.{layer_idx}.self_attn.attn" in tensor.shared_by
