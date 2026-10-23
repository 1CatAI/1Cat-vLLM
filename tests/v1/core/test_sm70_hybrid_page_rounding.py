# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("mamba_mode", ["none", "align"])
def test_rounded_hybrid_pages_preserve_precision_and_admission(tp, mamba_mode):
    full = FullAttentionSpec(
        block_size=1648,
        num_kv_heads=4 // tp,
        head_size=256,
        dtype=torch.uint8,
    )
    mamba = MambaSpec(
        block_size=32768,
        shapes=((10, 10240 // tp), (48 // tp, 128, 128)),
        dtypes=(torch.float16, torch.float32),
        num_speculative_blocks=7,
        mamba_cache_mode=mamba_mode,
        page_size_padded=full.page_size_bytes,
    )
    draft = SlidingWindowSpec(
        block_size=1648,
        num_kv_heads=8 // tp,
        head_size=128,
        dtype=torch.float16,
        sliding_window=2048,
    )
    specs = {f"full.{i}": full for i in range(16)}
    specs.update({f"mamba.{i}": mamba for i in range(48)})
    specs.update({f"draft.{i}": draft for i in range(5)})
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=262144),
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN_V100),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(
            mamba_cache_mode=mamba_mode, user_specified_mamba_block_size=True
        ),
        max_in_flight_tokens=1031,
    )
    compact = kv_cache_utils.unify_kv_cache_spec_page_size(specs, config)
    page = 1664 * full.page_size_bytes // full.block_size
    assert compact["full.0"] == replace(full, block_size=1664)
    assert compact["draft.0"] == replace(draft, block_size=832)
    assert compact["mamba.0"] == replace(mamba, page_size_padded=page)
    assert {spec.page_size_bytes for spec in compact.values()} == {page}
    assert specs["draft.0"] == draft  # Worker specs are immutable inputs.
    groups = kv_cache_utils.get_kv_cache_groups(config, specs)
    required = kv_cache_utils._max_memory_usage_bytes_from_groups(config, groups)

    config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
    original = kv_cache_utils.unify_kv_cache_spec_page_size(specs, config)
    assert original["draft.0"] == draft
    assert original["full.0"].block_size == 3296
    original_groups = kv_cache_utils.get_kv_cache_groups(config, specs)
    original_required = kv_cache_utils._max_memory_usage_bytes_from_groups(
        config, original_groups
    )
    assert original_required - required > 0.55 * 2**30 * 2 / tp

    config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN_V100
    config.speculative_config = SimpleNamespace(
        attention_backend=AttentionBackendEnum.TRITON_ATTN
    )
    assert kv_cache_utils.unify_kv_cache_spec_page_size(specs, config) == original


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_non_integral_hybrid_pages_preserve_logical_blocks(heads):
    target = FullAttentionSpec(
        block_size=2048,
        num_kv_heads=heads,
        head_size=256,
        dtype=torch.int8,
        kv_quant_mode=get_kv_quant_mode("int8_per_token_head"),
    )
    draft = SlidingWindowSpec(
        block_size=1024,
        num_kv_heads=2 * heads,
        head_size=128,
        dtype=torch.float16,
        sliding_window=2048,
    )
    state = MambaSpec(
        block_size=2048,
        shapes=((10, 2560 * heads), (12 * heads, 128, 128)),
        dtypes=(torch.float16, torch.float32),
        num_speculative_blocks=7,
        mamba_cache_mode="align",
        page_size_padded=draft.page_size_bytes,
    )
    assert target.page_size_bytes * 64 == draft.page_size_bytes * 65
    specs = {"target": target, "draft": draft, "state": state}
    original = dict(specs)
    unified = kv_cache_utils.unify_kv_cache_spec_page_size(specs)
    page = target.page_size_bytes
    assert unified == {
        "target": target,
        "draft": replace(draft, page_size_padded=page),
        "state": replace(state, page_size_padded=page),
    }
    assert {spec.page_size_bytes for spec in unified.values()} == {page}
    assert specs == original
    assert kv_cache_utils.unify_kv_cache_spec_page_size(unified) == unified


@pytest.mark.parametrize("cache_dtype", ["fp8_e4m3", "int8", "int8_per_token_head"])
@pytest.mark.parametrize("heads", [1, 2])
def test_compact_draft_preserves_prefix_grid_with_scale_overhead(cache_dtype, heads):
    target = FullAttentionSpec(
        block_size=2048,
        num_kv_heads=heads,
        head_size=256,
        dtype=torch.uint8 if cache_dtype.startswith("fp8") else torch.int8,
        kv_quant_mode=get_kv_quant_mode(cache_dtype),
    )
    draft = SlidingWindowSpec(
        block_size=2048,
        num_kv_heads=2 * heads,
        head_size=128,
        dtype=torch.float16,
        sliding_window=2048,
    )
    state = MambaSpec(
        block_size=2048,
        shapes=((10, 2560 * heads), (12 * heads, 128, 128)),
        dtypes=(torch.float16, torch.float32),
        num_speculative_blocks=7,
        mamba_cache_mode="align",
        page_size_padded=target.page_size_bytes,
    )
    specs = {"target": target, "draft": draft, "state": state}
    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN_V100)
    )
    unified = kv_cache_utils.unify_kv_cache_spec_page_size(specs, config)
    assert unified["target"] == target
    assert unified["state"] == state
    assert unified["draft"].block_size == 1024
    assert {s.page_size_bytes for s in unified.values()} == {target.page_size_bytes}
    assert specs["draft"] == draft
    assert kv_cache_utils.unify_kv_cache_spec_page_size(unified, config) == unified
