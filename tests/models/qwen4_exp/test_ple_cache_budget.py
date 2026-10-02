# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest
import torch

import vllm.config
from vllm.models.qwen4_exp.common.ple import kv_cache_bytes_for_max_model_len
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec


@pytest.mark.parametrize(
    "full_layers,other_layers,kind",
    [
        (2, 0, "window"),
        (2, 3, "window"),
        (8, 9, "window"),
        (2, 3, "none"),
        (2, 3, "align"),
        (2, 3, "all"),
    ],
)
def test_automatic_ple_budget_funds_actual_cache_layout(
    monkeypatch, full_layers, other_layers, kind
):
    config = NS(
        model_config=NS(max_model_len=64),
        parallel_config=NS(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        cache_config=NS(
            num_gpu_blocks_override=None,
            mamba_cache_mode="none" if kind == "window" else kind,
        ),
        max_in_flight_tokens=16,
    )
    common = dict(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.float16)
    specs = {f"full-{i}": FullAttentionSpec(**common) for i in range(full_layers)}
    if kind == "window":
        other = SlidingWindowSpec(**common, sliding_window=16)
    else:
        other = MambaSpec(
            block_size=16,
            shapes=((128,),),
            dtypes=(torch.float32,),
            page_size_padded=1024,
            mamba_cache_mode=kind,
        )
    specs.update({f"other-{i}": other for i in range(other_layers)})
    layers = {
        name: NS(get_kv_cache_spec=lambda cfg, spec=spec: spec)
        for name, spec in specs.items()
    }
    # Replace model discovery, retaining the real grouping and allocation math.
    monkeypatch.setattr(
        vllm.config, "get_layers_from_vllm_config", lambda *args: layers
    )
    budget = kv_cache_bytes_for_max_model_len(config)
    groups = get_kv_cache_groups(config, specs)
    cache = get_kv_cache_config_from_groups(config, groups, budget)
    assert get_max_concurrency_for_kv_cache_config(config, cache) >= 1
    layer_bytes = sum(spec.max_memory_usage_bytes(config) for spec in specs.values())
    if other_layers:
        assert budget > layer_bytes  # Shared pools charge padded hybrid slots.
    else:
        assert budget == layer_bytes


def test_automatic_ple_budget_without_cache_layers(monkeypatch):
    monkeypatch.setattr(vllm.config, "get_layers_from_vllm_config", lambda *args: {})
    config = NS(scheduler_config=NS(disable_hybrid_kv_cache_manager=False))
    assert kv_cache_bytes_for_max_model_len(config) == 0
