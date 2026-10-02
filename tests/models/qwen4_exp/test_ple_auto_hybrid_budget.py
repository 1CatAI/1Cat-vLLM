# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.models.qwen4_exp.nvidia import ple_layer as ple


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("dp", [1, 2])
def test_auto_hybrid_placement_leaves_headroom_and_caps_all_local_ranks(
    monkeypatch, hybrid, dp
):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_QWEN38_HYBRID_PLE", str(int(hybrid)))
    monkeypatch.setattr(ple, "_ple_host_budget_bytes", lambda: None)
    monkeypatch.setattr(ple, "_ple_vram_reserve_bytes", lambda _total: 0)
    monkeypatch.setattr(ple, "_ple_host_reserve_bytes", lambda _total: 0)
    monkeypatch.setattr(ple, "kv_cache_bytes_for_max_model_len", lambda _cfg: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1000, 1000))
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 600)
    monkeypatch.setattr(ple, "total_host_bytes", lambda: 600)
    monkeypatch.setattr(
        ple,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=8192),
            cache_config=SimpleNamespace(gpu_memory_utilization=0.9),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=4,
                local_world_size=4,
                data_parallel_size_local=dp,
            ),
        ),
    )
    table = SimpleNamespace(_meta_weight_shape=(100, 4), embedding_dim=4)
    budget = ple.Qwen4ExpPinnedHostEmbedding._resolve_host_budget(
        table, torch.device("cuda:0")
    )
    # Ordinary placement fits the whole table on-device. Hybrid leaves device
    # headroom, but cannot pin more than the fair share of actual host memory.
    assert budget == (600 // (4 * dp) if hybrid else 0)
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 10000)
    assert ple.Qwen4ExpPinnedHostEmbedding._resolve_host_budget(
        table, torch.device("cuda:0")
    ) == (400 if hybrid else 0)
    envs.disable_envs_cache()


def test_explicit_host_budget_is_preserved(monkeypatch):
    monkeypatch.setattr(ple, "_ple_host_budget_bytes", lambda: 123)
    table = SimpleNamespace()
    assert (
        ple.Qwen4ExpPinnedHostEmbedding._resolve_host_budget(
            table, torch.device("cuda:0")
        )
        == 123
    )
