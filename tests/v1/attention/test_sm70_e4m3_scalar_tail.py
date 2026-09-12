# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission checks for the explicitly loaded scalar tail operator."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops import sm70_e4m3_scalar as scalar


@pytest.fixture
def scalar_call(monkeypatch):
    calls = []
    manifest = {
        "share_kv_six_heads": True,
        "compact_page_map": True,
        "e4m3_lut": True,
        "max_context": 262144,
        "module_name": "test_scalar",
        "library_sha256": "test_digest",
    }
    monkeypatch.setattr(
        scalar,
        "load_attention_library",
        lambda _: (SimpleNamespace(run=lambda *args: calls.append(args)), manifest),
    )
    scalar.load_scalar_tail_attention.cache_clear()
    monkeypatch.setattr(scalar, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        scalar,
        "get_forward_context",
        lambda: SimpleNamespace(
            batch_descriptor=SimpleNamespace(attention_context_bucket=262144)
        ),
    )
    run = scalar.load_scalar_tail_attention("test-manifest", torch.device("cpu"))
    q = torch.zeros((1, 6, 256), dtype=torch.float16)
    kv = torch.zeros((1, 3296, 1, 256), dtype=torch.uint8)
    args = (
        q,
        kv,
        kv,
        torch.zeros((1, 80), dtype=torch.int32),
        torch.tensor([262144], dtype=torch.int32),
    )
    kwargs = dict(
        out=torch.empty_like(q),
        softmax_scale=0.0625,
        k_scale=1.0,
        v_scale=1.0,
        kv_cache_dtype="fp8_e4m3",
        window_size=(-1, -1),
        max_seq_len_hint=262144,
        partition_size_hint=None,
        anchor_lens=None,
        anchored_window=0,
    )
    yield run, args, kwargs, calls
    scalar.load_scalar_tail_attention.cache_clear()


@pytest.mark.parametrize(
    "override",
    [
        {"max_seq_len_hint": 262145},
        {"max_seq_len_hint": None},
        {"max_seq_len_hint": torch.tensor(262144)},
        {"window_size": (2048, 0)},
        {"kv_cache_dtype": "fp8_e5m2"},
        {"partition_size_hint": 512},
        {"anchor_lens": torch.tensor([10]), "anchored_window": 2048},
    ],
)
def test_scalar_tail_preserves_unsupported_route(scalar_call, override):
    run, args, kwargs, calls = scalar_call
    assert not run(*args, **(kwargs | override))
    assert not calls


def test_scalar_tail_keeps_workspace_and_forwards_scale(scalar_call):
    run, args, kwargs, calls = scalar_call
    kwargs.update(k_scale=0.5, v_scale=2.0)
    assert run(*args, **kwargs)
    pointers = [t.data_ptr() for t in calls[0][6:10]]
    assert run(*args, **kwargs)
    assert [t.data_ptr() for t in calls[1][6:10]] == pointers
    assert all(t.dtype == torch.float32 for t in calls[0][6:9])
    assert calls[0][9].item() == 256
    assert calls[0][-3:] == (0.0625, 0.5, 2.0)


def test_ordinary_graph_preserves_original_scalar(scalar_call, monkeypatch):
    run, args, kwargs, calls = scalar_call
    monkeypatch.setattr(
        scalar, "get_forward_context", lambda: SimpleNamespace(batch_descriptor=None)
    )
    assert not run(*args, **kwargs)
    assert not calls
