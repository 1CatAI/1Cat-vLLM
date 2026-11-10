# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The loaded native provider owns request widths and the model context bound."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.config
from vllm.v1.attention.ops import sm70_grouped_long as module


@pytest.mark.parametrize("revision,expected", [(1, (8,)), (2, tuple(range(2, 9)))])
def test_batch_widths_follow_native_revision(revision, expected):
    manifest = {**module.BUILTIN_MANIFEST, "query_rows_revision": revision}
    assert module.long_attention_batch_query_rows(manifest) == expected


@pytest.mark.parametrize("batch", [1, 2, 3, 8, 32])
@pytest.mark.parametrize("width", [2, 3, 5, 7, 8])
@pytest.mark.parametrize("revision", [1, 2])
@pytest.mark.parametrize("capacity", [131072, 524288, None])
def test_shared_provider_uses_model_window_without_speculation_descriptor(
    monkeypatch, batch, width, revision, capacity
):
    native, fallback = Mock(), Mock()
    manifest = {**module.BUILTIN_MANIFEST, "query_rows_revision": revision}
    monkeypatch.setattr(module, "resolve_long_attention", lambda: (native, manifest))
    monkeypatch.setattr(module, "long_attention_max_batch_size", lambda _: 65535)
    monkeypatch.setattr(module, "long_attention_page_supported", lambda *args: True)
    monkeypatch.setattr(module, "is_forward_context_available", lambda: False)
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=capacity))
    lookup = Mock(return_value=config)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lookup)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=0)
    )
    monkeypatch.setattr(module, "_WORKSPACES", {})
    operation = module.wrap_long_attention(fallback)
    # Binding captures the model's served window. The hot path needs no config lookup.
    config.model_config.max_model_len = 1048576
    q = torch.empty(batch * width, 6, 256, dtype=torch.float16, device="meta")
    k = torch.empty(1, 2048, 1, 256, dtype=torch.uint8, device="meta")
    table = torch.empty(batch, 64, dtype=torch.int32, device="meta")
    lengths = torch.empty(batch * width, dtype=torch.int32, device="meta")
    for _ in range(2):
        operation(q, k, k, table, lengths, out=q, softmax_scale=0.0625)
    expected_native = revision >= 2 and capacity != 524288
    assert native.call_count == (2 if expected_native else 0)
    assert fallback.call_count == (0 if expected_native else 2)
    assert hasattr(operation, "request_query_rows") is expected_native
    lookup.assert_called_once()
    if expected_native:
        args = native.call_args.args
        assert args[4] is table
        partial = args[6]
        assert partial.shape == (
            (80, 8, 6, 256) if batch == 1 else (batch, 80, 8, 6, 256)
        )


def test_native_row_revision_preserves_served_context(monkeypatch):
    native = Mock()
    monkeypatch.setattr(module, "builtin_long_attention", lambda: native)
    monkeypatch.setattr(module, "long_attention_enabled", lambda _: True)
    monkeypatch.setattr(module, "builtin_context_limit", lambda capacity: capacity)
    monkeypatch.setattr(
        torch.ops._vllm_fa2_C,
        "sm70_grouped_long_query_rows_revision",
        lambda: 2,
        raising=False,
    )
    operator, manifest = module.resolve_long_attention(
        SimpleNamespace(e4m3_long_manifest=None), capacity=1048576
    )
    assert operator is native
    assert manifest["max_context"] == 1048576
    assert manifest["query_rows_revision"] == 2
