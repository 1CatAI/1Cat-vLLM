# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 served-window admission uses its own native capacity revision."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.test_sm70_grouped_codecs import scenario
from vllm.v1.attention.kv_codecs import FP16
from vllm.v1.attention.ops import sm70_fp16_grouped as fp16
from vllm.v1.attention.ops import sm70_grouped as grouped
from vllm.v1.attention.ops import sm70_grouped_long as long

pytestmark = pytest.mark.cpu_test


def configure(monkeypatch, context, revision):
    native = SimpleNamespace(sm70_grouped_long_context_revision=lambda: 1)
    if revision is not None:
        native.sm70_grouped_fp16_context_revision = lambda: revision
    monkeypatch.setattr(torch.ops, "_vllm_fa2_C", native)
    monkeypatch.setattr(
        grouped,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(model_config=SimpleNamespace(max_model_len=context)),
    )


@pytest.mark.parametrize("revision", [None, 0, 1])
@pytest.mark.parametrize("context", [32768, 262144, 524288, 1048576, 2097152])
def test_fp16_capacity_does_not_inherit_e4m3_revision(monkeypatch, revision, context):
    configure(monkeypatch, context, revision)
    assert fp16.grouped_context_limit() == (
        context if revision == 1 else grouped.MAX_CONTEXT
    )


@pytest.mark.parametrize("context", [0, -1, True, 1.5, 2**31])
def test_fp16_invalid_window_fails_before_launch(monkeypatch, context):
    configure(monkeypatch, context, 1)
    with pytest.raises(ValueError, match="positive int32"):
        fp16.grouped_context_limit()


def test_missing_model_config_preserves_legacy_capacity(monkeypatch):
    configure(monkeypatch, 1048576, 1)
    monkeypatch.setattr(grouped, "get_current_vllm_config_or_none", lambda: None)
    assert fp16.grouped_context_limit() == grouped.MAX_CONTEXT


def test_forward_borrows_window_captured_at_provider_initialization(monkeypatch):
    from vllm.vllm_flash_attn import flash_attn_interface

    configure(monkeypatch, 1048576, 1)
    native = torch.ops._vllm_fa2_C
    native.sm70_grouped_fp16_fwd = object()
    resources: dict[str, int] = {}
    monkeypatch.setattr(fp16, "workspace_cache", lambda name, _: resources)
    monkeypatch.setattr(fp16.current_platform, "is_device_capability", lambda _: True)
    monkeypatch.setattr(flash_attn_interface, "ensure_fa2_library_loaded", lambda: None)
    assert fp16.load_grouped_fp16_fp32() is not None
    # The construction-only global config is absent during graph capture/replay.
    monkeypatch.setattr(grouped, "get_current_vllm_config_or_none", lambda: None)
    assert fp16.grouped_context_limit() == 1048576


@pytest.mark.parametrize("revision", [None, 0, 1])
@pytest.mark.parametrize("context", [262144, 262145, 266240, 266241, 1048576, 2097152])
@pytest.mark.parametrize("groups,rows", [(1, 2), (1, 8), (2, 16), (4, 32)])
def test_fp16_table_coverage_without_e4m3_graph(
    monkeypatch, revision, context, groups, rows
):
    configure(monkeypatch, context, revision)

    def unexpected(*args, **kwargs):
        pytest.fail("FP16 capacity must not depend on the E4M3 long graph")

    monkeypatch.setattr(long, "long_attention_graph_contract", unexpected)
    monkeypatch.setenv("VLLM_SM70_E4M3_LONG_ATTENTION", "0")
    monkeypatch.setenv("VLLM_SM70_E4M3_LONG_ATTENTION_MANIFEST", "missing.json")
    inputs = scenario(FP16, groups=groups, rows=rows)
    instance, q, k, v, table, lengths, metadata, out = inputs
    width = (context + 831) // 832
    for columns in (width, width + 1):
        table.shape = (rows, columns)
        metadata.block_table.shape = (groups, columns)
        result = grouped.grouped_fp32_reason(
            FP16, instance, q, k, v, table, lengths, metadata, out=out
        )
        limit = max(grouped.MAX_CONTEXT, width * 832) if revision == 1 else 266240
        assert result == (None if columns * 832 <= limit else "context_capacity")


def test_page_rounding_cannot_exceed_native_integer_bound(monkeypatch):
    context = 2**31 - 4096
    configure(monkeypatch, context, 1)
    instance, q, k, v, table, lengths, metadata, out = scenario(FP16)
    width = (context + 831) // 832
    table.shape = (8, width)
    metadata.block_table.shape = (1, width)
    assert (
        grouped.grouped_fp32_reason(
            FP16, instance, q, k, v, table, lengths, metadata, out=out
        )
        == "context_capacity"
    )
