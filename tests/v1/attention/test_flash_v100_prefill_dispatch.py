# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned outer dispatch preserves reset, capture and completion semantics."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import impl, prefill, workspace
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("branch", ["triton", "prefix", "paged", "dense"])
def test_reset_precedes_execution_and_none_is_complete(monkeypatch, branch):
    events = []
    config = SimpleNamespace(
        use_triton_prefill=branch == "triton",
        use_prefill_paged_cache=branch == "paged",
        use_flash_v100_prefill_paged=True,
    )

    def complete(name):
        def call(*args):
            events.append(name)
            return None

        return call

    ops = prefill.PrefillDriverOps(
        triton_forward=complete("triton"),
        compare_triton=complete("compare"),
        small_query_enabled=lambda metadata: False,
    )
    owner = prefill.PrefillExecutor(
        prefill.PrefillConfig(config, 0.125, "auto"),
        ops,
        SimpleNamespace(decode_cache=SimpleNamespace(invalidate=complete("reset"))),
        {
            "_flash_v100_prefill_with_prefix": complete("prefix"),
            "_flash_v100_prefill": complete("dense"),
        },
    )
    monkeypatch.setattr(
        prefill._kv_layout,
        "_metadata_expects_more_query_tokens_than_available",
        lambda *args: False,
    )
    monkeypatch.setattr(
        prefill._kv_layout, "_has_prefix_context", lambda metadata: branch == "prefix"
    )
    monkeypatch.setattr(prefill._routing, "_record_route", complete("route"))
    tensor = torch.empty((2, 1, 4))
    result = owner.forward(
        None,
        tensor,
        tensor,
        tensor,
        tensor,
        SimpleNamespace(),
        tensor,
        None,
        None,
        False,
        "layer",
    )
    assert result is None
    assert events == (
        ["reset", "route", "triton"]
        if branch == "triton"
        else ["reset", "dense" if branch == "dense" else "prefix", "compare", "route"]
    )


@pytest.mark.parametrize("prefix_feature", [False, True])
def test_capture_early_return_preserves_resident_cache(monkeypatch, prefix_feature):
    events = []
    cache = workspace.DecodeCache(torch.ones(1), torch.ones(1), 1, 1)
    key = cache.key
    config = SimpleNamespace(use_triton_prefill=False)

    def prefix(*args):
        events.append("prefix")
        return args[-1]

    owner = prefill.PrefillExecutor(
        prefill.PrefillConfig(config, 0.125, "auto"),
        prefill.PrefillDriverOps(
            capture_prefix_kind=lambda *args: prefix_feature,
            record_capture_prefix=lambda: events.append("feature"),
            small_query_enabled=lambda metadata: True,
            record_capture_layout=lambda metadata: events.append("layout"),
        ),
        workspace.V100Workspace(cache),
        {"_flash_v100_prefill_with_prefix": prefix},
    )
    monkeypatch.setattr(
        prefill._kv_layout,
        "_metadata_expects_more_query_tokens_than_available",
        lambda *args: False,
    )
    monkeypatch.setattr(
        prefill._routing, "_record_route", lambda name: events.append(name)
    )
    tensor = torch.empty((2, 1, 4))
    assert (
        owner.forward(
            None,
            tensor,
            tensor,
            tensor,
            tensor,
            SimpleNamespace(),
            tensor,
            None,
            None,
            True,
            "layer",
        )
        is tensor
    )
    assert cache.key is key and cache.length == cache.capacity == 1
    assert events == (
        ["feature", "prefix"]
        if prefix_feature
        else ["prefill_capture_smallq", "layout", "prefix"]
    )


def test_assembly_keeps_original_super_and_live_feature_overrides(monkeypatch):
    instance = object.__new__(impl.FlashAttnV100Impl)
    instance.workspace = workspace.V100Workspace()
    instance.scale = 0.125
    instance.kv_cache_dtype = "auto"
    marker = object()
    instance._small_query_decode_enabled = lambda metadata: marker

    def reference(receiver, *args):
        assert receiver is instance
        return marker

    monkeypatch.setattr(TritonAttentionImpl, "forward", reference)
    owner = instance._new_prefill_executor()
    assert owner.ops.triton_forward() is marker
    assert owner.ops.small_query_enabled(None) is marker
    instance._small_query_decode_enabled = lambda metadata: None
    assert owner.ops.small_query_enabled(None) is marker
    assert instance._new_prefill_executor().ops.small_query_enabled(None) is None
