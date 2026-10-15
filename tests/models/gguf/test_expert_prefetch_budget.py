# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.config import KernelConfig
from vllm.model_executor.layers.quantization import gguf_expert_storage as storage


@pytest.mark.parametrize("depth", [1, 4])
def test_bank_budget_preserves_loading_order_and_bounded_queue(monkeypatch, depth):
    jobs = [(str(i), i, "w1", (0, 1), None) for i in range(9)]
    monkeypatch.setattr(storage, "_local_slice", lambda tensor, *args: tensor)
    loader = storage._Prefetcher(jobs, depth=depth)
    try:
        assert len(loader.futures) <= depth
        assert loader.take("not-a-bank") is None
        for i in range(9):
            assert loader.take(str(i)) == i
            assert len(loader.futures) <= depth
        assert not loader.futures
    finally:
        loader.pool.shutdown(wait=True)


@pytest.mark.parametrize("depth", [1, 4])
def test_selected_bank_budget_reaches_loader_owner(monkeypatch, depth):
    calls = []
    cfg = SimpleNamespace(expert_load_prefetch_banks=depth)
    monkeypatch.setattr(storage, "_expert_cache_config", lambda: cfg)
    monkeypatch.setattr(storage, "_slice_spec", lambda *args: ((0, 1), None))
    monkeypatch.setattr(
        storage, "_Prefetcher", lambda jobs, depth: calls.append((jobs, depth))
    )
    monkeypatch.setattr(storage, "_PREFETCH", None)
    storage.start_prefetch({"bank": (None, None, "w1")}, {"bank": object()})
    assert calls[0][1] == depth


def test_startup_bank_policy_keeps_computation_hash_and_rejects_bad_budget():
    a = KernelConfig()
    b = KernelConfig()
    assert a.sm70_gguf.expert_load_prefetch_banks == 4
    a.sm70_gguf.active = b.sm70_gguf.active = True
    b.sm70_gguf.expert_load_prefetch_banks = 1
    assert a.compute_hash() == b.compute_hash()
    for value in (0, 5):
        with pytest.raises(ValueError):
            KernelConfig(sm70_gguf={"expert_load_prefetch_banks": value})
