# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Larger local shards must not invalidate already prepared raw pointers."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import awq, fp8, sm70_turbomind


@pytest.mark.parametrize(
    "module,getter,cache_name,packing",
    [
        (
            awq,
            "_get_sm70_awq_prefill_exact_dense_workspace",
            "_sm70_awq_prefill_dense_workspaces",
            8,
        ),
        (
            fp8,
            "_get_sm70_fp8_prefill_exact_dense_workspace",
            "_sm70_fp8_prefill_dense_workspaces",
            1,
        ),
        (
            sm70_turbomind,
            "get_nvfp4_qpn4_dense_workspace",
            "_nvfp4_qpn4_dense_workspaces",
            2,
        ),
    ],
)
def test_local_workspace_growth_retains_live_allocations(
    monkeypatch, module, getter, cache_name, packing
):
    cache = getattr(module, cache_name)
    saved = dict(cache)
    cache.clear()
    allocations = []
    original_empty = torch.empty

    def allocate(shape, *, dtype, device):
        allocations.append(shape[0])
        return original_empty(1, dtype=dtype)

    monkeypatch.setattr(torch, "empty", allocate)
    small = SimpleNamespace(
        device=torch.device("cuda:0"), numel=lambda: 5120 * 8704 // packing
    )
    large = SimpleNamespace(device=small.device, numel=lambda: 5120 * 34816 // packing)
    get = getattr(module, getter)
    try:
        before = get(small)
        pointer = before.data_ptr()
        after = get(large)
        assert after is not before
        assert get(small) is before and before.data_ptr() == pointer
        assert get(large) is after
        assert allocations == [5120 * 8704, 5120 * 34816]
    finally:
        cache.clear()
        cache.update(saved)
