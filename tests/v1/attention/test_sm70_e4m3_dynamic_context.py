# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served-window planning, independent native ABIs and conservative fallback."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.flash_v100 import FlashV100Options
from vllm.v1.attention.ops import sm70_grouped_long as long
from vllm.v1.attention.ops import sm70_grouped_scalar as scalar

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("context", [32768, 262144, 1048576, 2097152, 8388608])
@pytest.mark.parametrize("revision", [None, 0, 1])
@pytest.mark.parametrize("is_scalar", [False, True])
def test_native_abi_and_served_window(monkeypatch, context, revision, is_scalar):
    native = SimpleNamespace()
    if revision is not None:
        name = (
            "sm70_scalar_attention_workspace_revision"
            if is_scalar
            else "sm70_grouped_long_context_revision"
        )
        setattr(native, name, lambda: revision)
    monkeypatch.setattr(torch.ops, "_vllm_fa2_C", native)
    monkeypatch.setattr(long, "builtin_long_attention", lambda: object())
    monkeypatch.setattr(
        long,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(model_config=SimpleNamespace(max_model_len=context)),
    )
    expected = context if revision == 1 else min(context, 262144)
    assert long.builtin_context_limit(scalar=is_scalar) == expected
    assert long.builtin_context_limit(context, scalar=is_scalar) == expected


@pytest.mark.parametrize("context", [0, -1, True, 2**31, 1.5])
def test_invalid_window_fails_before_launch(context):
    with pytest.raises(ValueError, match="positive int32"):
        long.builtin_context_limit(context)


@pytest.mark.parametrize("page", [1024, 1648, 2048, 3296])
@pytest.mark.parametrize("context", [32768, 262145, 1048576, 2097152])
def test_table_page_rounding(context, page):
    width = (context + page - 1) // page
    assert long.context_table_supported(torch.empty(1, width), page, context)
    assert not long.context_table_supported(torch.empty(1, width + 1), page, context)
    assert not long.context_table_supported(torch.empty(1, 0), page, context)


def test_graph_uses_explicit_window_and_preserves_opt_out(monkeypatch):
    monkeypatch.setattr(long, "builtin_long_attention", lambda: object())
    monkeypatch.setattr(long, "builtin_dynamic_context", lambda **_: True)
    policy = FlashV100Options(e4m3_long_enabled=True)
    assert long.long_attention_graph_contract(2097152, policy=policy)[0] == 2097152
    policy.e4m3_long_enabled = False
    assert long.long_attention_graph_contract(2097152, policy=policy) == (None, ())


def test_old_scalar_does_not_inherit_new_grouped_abi(monkeypatch):
    monkeypatch.setattr(scalar, "builtin_scalar_tail_attention", lambda: object())
    monkeypatch.setattr(scalar, "builtin_context_limit", lambda *_, **__: 262144)
    monkeypatch.setattr(
        scalar,
        "resolve_long_attention",
        lambda: (object(), dict(long.BUILTIN_MANIFEST, max_context=2097152)),
    )
    scalar.load_scalar_tail_attention.cache_clear()
    try:
        assert scalar.load_scalar_tail_attention("", torch.device("cpu")) is None
    finally:
        scalar.load_scalar_tail_attention.cache_clear()


def test_scalar_plan_tracks_window_and_reuses_only_matching_buffers(monkeypatch):
    context = 32768
    calls = []
    monkeypatch.setattr(
        scalar,
        "builtin_scalar_tail_attention",
        lambda: lambda *args: calls.append(args),
    )
    monkeypatch.setattr(scalar, "builtin_context_limit", lambda value, **_: value)
    monkeypatch.setattr(scalar, "builtin_dynamic_context", lambda **_: True)
    monkeypatch.setattr(
        scalar,
        "resolve_long_attention",
        lambda: (object(), dict(long.BUILTIN_MANIFEST, max_context=context)),
    )
    monkeypatch.setattr(scalar, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        scalar,
        "get_forward_context",
        lambda: SimpleNamespace(
            batch_descriptor=SimpleNamespace(attention_context_bucket=context)
        ),
    )
    scalar.load_scalar_tail_attention.cache_clear()
    try:
        plans = []
        for context in (32768, 2097152, 32768):
            run = scalar.load_scalar_tail_attention("", torch.device("cpu"))
            plans.append(run)
            q = torch.zeros((1, 6, 256), dtype=torch.float16)
            kv = torch.empty((1, 2048, 1, 256), dtype=torch.uint8)
            assert run(
                q,
                kv,
                kv,
                torch.zeros((1, (context + 2047) // 2048), dtype=torch.int32),
                torch.full((1,), context, dtype=torch.int32),
                out=torch.empty_like(q),
                softmax_scale=0.0625,
                k_scale=1.0,
                v_scale=1.0,
                kv_cache_dtype="fp8_e4m3",
                window_size=(-1, -1),
                max_seq_len_hint=context,
                partition_size_hint=None,
                anchor_lens=None,
                anchored_window=0,
            )
            assert calls[-1][6].shape == (1, 6, (context + 1023) // 1024, 256)
        assert plans[0] is plans[2] and plans[0] is not plans[1]
        assert calls[0][6].data_ptr() == calls[2][6].data_ptr()
    finally:
        scalar.load_scalar_tail_attention.cache_clear()


@pytest.mark.parametrize("revision", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("bound", [None, 0, 262144, 1048576, 2097152])
def test_wide_admission_requires_graph_coverage_and_new_native(
    monkeypatch, revision, enabled, bound
):
    import vllm.forward_context as forward
    from tests.v1.attention.test_sm70_grouped_codecs import scenario
    from vllm.v1.attention.kv_codecs import FP8_E4M3
    from vllm.v1.attention.ops.sm70_grouped import grouped_fp32_reason

    monkeypatch.setattr(long, "builtin_dynamic_context", lambda **_: revision)
    monkeypatch.setattr(
        long,
        "long_attention_graph_contract",
        lambda value: (value, (8,)) if enabled else (None, ()),
    )
    monkeypatch.setattr(forward, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward,
        "get_forward_context",
        lambda: SimpleNamespace(
            batch_descriptor=SimpleNamespace(attention_context_bucket=bound),
            runtime_resources={},
        ),
    )
    instance, q, k, v, table, lengths, metadata, out = scenario(FP8_E4M3, page=2048)
    table.shape = (8, 512)
    metadata.block_table.shape = (1, 512)
    reason = grouped_fp32_reason(
        FP8_E4M3, instance, q, k, v, table, lengths, metadata, out=out
    )
    admitted = revision and enabled and bound is not None and bound >= 1048576
    assert reason == (None if admitted else "context_capacity")
