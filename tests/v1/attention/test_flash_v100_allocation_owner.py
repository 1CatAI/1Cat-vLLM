# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Growing allocation preserves retry order and live legacy bindings."""

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import workspace

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("on_cuda", [False, True])
@pytest.mark.parametrize("failures", [0, 1, 2])
def test_allocation_retry_order(monkeypatch, on_cuda, failures):
    calls = []
    result = (torch.empty(1),)
    remaining = failures

    def allocate():
        nonlocal remaining
        calls.append("allocate")
        if remaining:
            remaining -= 1
            raise torch.OutOfMemoryError("controlled allocation failure")
        return result

    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: calls.append("empty_cache")
    )
    actual = workspace.allocate_growing_workspace(allocate, on_cuda=on_cuda)
    assert actual is (None if failures == 2 else result)
    assert calls == (
        ["allocate"]
        if failures == 0
        else ["allocate", *(["empty_cache"] if on_cuda else []), "allocate"]
    )


def test_non_oom_error_propagates_without_retry(monkeypatch):
    calls = []

    def allocate():
        calls.append("allocate")
        raise ValueError("invalid allocation contract")

    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: calls.append("empty_cache")
    )
    with pytest.raises(ValueError, match="invalid allocation contract"):
        workspace.allocate_growing_workspace(allocate, on_cuda=True)
    assert calls == ["allocate"]


def test_legacy_allocation_patch_drives_both_real_consumers(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as legacy
    from vllm.v1.attention.backends.flash_v100 import dense_prefill, kv_layout

    original = workspace.allocate_growing_workspace
    calls = []

    def tracked(allocate, *, on_cuda):
        calls.append(on_cuda)
        return original(allocate, on_cuda=on_cuda)

    monkeypatch.setattr(kv_layout, "_prefill_gather_dense_workspaces", {})
    monkeypatch.setattr(dense_prefill, "_fp8_prefill_bridge_workspaces", {})
    monkeypatch.setattr(legacy, "_allocate_growing_workspace", tracked)
    assert workspace.allocate_growing_workspace is tracked
    assert legacy._allocate_growing_workspace is tracked
    assert "_allocate_growing_workspace" not in vars(dense_prefill)
    cache = torch.empty((1, 16, 1, 4), dtype=torch.float16)
    for get_buffers in (
        kv_layout._get_prefill_gather_dense_workspace,
        dense_prefill._get_fp8_prefill_bridge_workspace,
    ):
        first = get_buffers(cache, 2)
        second = get_buffers(cache, 2)
        assert first is not None and second is not None
        assert [t.data_ptr() for t in first] == [t.data_ptr() for t in second]
    assert calls == [False, False]


def test_declared_legacy_alias_can_be_deleted_and_restored():
    from vllm.v1.attention.backends import flash_attn_v100 as legacy

    original = workspace.allocate_growing_workspace
    try:
        del legacy._allocate_growing_workspace
        assert not hasattr(workspace, "allocate_growing_workspace")
        assert not hasattr(legacy, "_allocate_growing_workspace")
        assert legacy._owners("_allocate_growing_workspace") == [workspace]
    finally:
        legacy._allocate_growing_workspace = original
    assert workspace.allocate_growing_workspace is original
    assert "_allocate_growing_workspace" in dir(legacy)
