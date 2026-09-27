# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.models.qwen4_exp.nvidia import ple_layer as ple

Embedding = ple.Qwen4ExpPinnedHostEmbedding
ROWS, ROW_BYTES = 100, 4


def _table() -> SimpleNamespace:
    # The device measurement and the host cap run for real on the stubs below.
    table = SimpleNamespace(
        _meta_weight_shape=(ROWS, ROW_BYTES), embedding_dim=ROW_BYTES
    )
    table._device_spill_bytes = MethodType(Embedding._device_spill_bytes, table)
    table._cap_derived_host_budget = MethodType(
        Embedding._cap_derived_host_budget, table
    )
    return table


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("dp", [1, 2])
def test_auto_hybrid_placement_leaves_headroom_and_caps_all_local_ranks(
    monkeypatch, hybrid, dp
):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_QWEN38_HYBRID_PLE", str(int(hybrid)))
    monkeypatch.setattr(ple, "ple_host_budget_bytes", lambda: None)
    monkeypatch.setattr(ple, "ple_cascade_configured", lambda: False)
    monkeypatch.setattr(ple, "ple_vram_reserve_bytes", lambda _total: 0)
    monkeypatch.setattr(ple, "ple_host_reserve_bytes", lambda _total: 0)
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
    table = _table()
    placement = Embedding._plan_placement(table, torch.device("cuda:0"))
    # Ordinary placement fits the whole table on-device. Hybrid leaves device
    # headroom, but cannot pin more than the fair share of actual host memory.
    assert placement.host_rows == ((600 // (4 * dp)) // ROW_BYTES if hybrid else 0)
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 10000)
    placement = Embedding._plan_placement(table, torch.device("cuda:0"))
    assert placement.host_rows == (ROWS if hybrid else 0)
    envs.disable_envs_cache()


def test_explicit_host_budget_is_preserved(monkeypatch):
    monkeypatch.setattr(ple, "ple_host_budget_bytes", lambda: 123)
    monkeypatch.setattr(ple, "ple_cascade_configured", lambda: False)
    table = SimpleNamespace(
        _meta_weight_shape=(ROWS, ROW_BYTES), embedding_dim=ROW_BYTES
    )
    placement = Embedding._plan_placement(table, torch.device("cuda:0"))
    assert placement.host_rows == 123 // ROW_BYTES


def test_hybrid_host_budget_does_not_group_provisional_cache_pages(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_QWEN38_HYBRID_PLE", "1")
    monkeypatch.setattr(ple, "ple_host_budget_bytes", lambda: None)
    monkeypatch.setattr(ple, "ple_cascade_configured", lambda: False)
    monkeypatch.setattr(ple, "available_host_bytes", lambda: None)
    monkeypatch.setattr(ple, "total_host_bytes", lambda: None)
    monkeypatch.setattr(ple, "get_current_vllm_config", lambda: SimpleNamespace())

    def unresolved_layout(*_args):
        raise ValueError("CSA+linear layer 3 violates cache geometry.")

    monkeypatch.setattr(ple, "kv_cache_bytes_for_max_model_len", unresolved_layout)
    monkeypatch.setattr(torch.cuda, "mem_get_info", unresolved_layout)
    placement = Embedding._plan_placement(_table(), torch.device("cuda:0"))
    assert placement.host_rows == ROWS
    envs.disable_envs_cache()
