# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe real CPU cache data through diagnostics without an Impl receiver."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import debug_compare, ops, state
from vllm.v1.attention.backends.flash_v100.plan.events import (
    EventStream,
    PrefillDebugEvent,
    prefill_debug,
)

pytestmark = pytest.mark.cpu_test


def test_events_are_ordered_and_propagate_observer_failure():
    stream = EventStream[list[str]]()
    visited: list[str] = []
    stream.subscribe(lambda event: event.append("reference"))

    def fail(event):
        assert event == ["reference"]
        event.append("report")
        raise RuntimeError("dump failed")

    stream.subscribe(fail)
    stream.subscribe(lambda event: event.append("too late"))
    with pytest.raises(RuntimeError, match="dump failed"):
        stream.emit(visited)
    assert visited == ["reference", "report"]


@pytest.fixture
def observation(monkeypatch):
    monkeypatch.setattr(ops, "_get_paged_kv_utils", lambda: None)
    monkeypatch.setattr(state, "_logged_prefill_compare", False)
    monkeypatch.setattr(state, "_logged_dflash_prefix_dump", False)
    payloads = []
    warnings = []
    calls = []
    monkeypatch.setattr(torch, "save", lambda payload, path: payloads.append(payload))
    monkeypatch.setattr(
        debug_compare.logger, "warning", lambda *args: warnings.append(args)
    )
    cache = torch.arange(24, dtype=torch.float16).reshape(3, 2, 2, 1, 2)
    key_cache, value_cache = cache.unbind(1)
    key = key_cache.flatten(0, 1)[[5, 0]].clone()
    value = value_cache.flatten(0, 1)[[5, 0]].clone()
    query = torch.ones((2, 1, 2), dtype=torch.float16)
    layer = torch.nn.Module()
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    layer.is_dflash_draft_attn = False

    def reference(name):
        def run(q, k, v, **kwargs):
            calls.append((name, q.shape, k.clone(), v.clone(), kwargs))
            return torch.zeros_like(q)

        return run

    event = PrefillDebugEvent(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=cache,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_metadata=SimpleNamespace(
            block_table=torch.tensor([[2, 0]]),
            seq_lens=torch.tensor([3]),
            slot_mapping=torch.tensor([5, 0]),
        ),
        out_seq=torch.ones_like(query),
        i=0,
        start=0,
        end=2,
        seq_len=3,
        num_kv_heads=1,
        head_dim=2,
        block_size=2,
        causal=True,
        window_size=(-1, -1),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([3]),
        debug_compare=True,
        dump_enabled=True,
        kv_cache_dtype="auto",
        scale=0.5,
        dense=reference("dense"),
        torch_reference=reference("torch"),
        layer_info=lambda layer: {"layer_name": "observed"},
    )
    return event, payloads, warnings, calls


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("valid_slots", [False, True])
def test_prefix_observers_preserve_reference_and_slot_dumps(
    observation, draft, valid_slots
):
    event, payloads, warnings, calls = observation
    event.layer.is_dflash_draft_attn = draft
    if not valid_slots:
        event.attn_metadata.slot_mapping[0] = -1
    prefill_debug.emit(event)

    assert len(calls) == 1
    name, shape, key, value, kwargs = calls[0]
    assert name == ("torch" if draft else "dense")
    assert shape == ((2, 1, 2) if draft else (1, 2, 1, 2))
    expected_k = event.key_cache.flatten(0, 1)[[4, 5, 0]]
    expected_v = event.value_cache.flatten(0, 1)[[4, 5, 0]]
    assert torch.equal(key.reshape(3, 1, 2), expected_k)
    assert torch.equal(value.reshape(3, 1, 2), expected_v)
    assert kwargs == dict(causal=True, softmax_scale=0.5, window_size=(-1, -1))
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["layer_name"] == "observed"
    assert payload["paged_vs_dense_max"] == 1.0
    assert payload["paged_vs_dense_mean"] == 1.0
    assert payload["slot_k_max"] == (0.0 if valid_slots else None)
    assert payload["tail_v_max"] == (0.0 if valid_slots else None)
    assert torch.equal(payload["k_cont_tail"], expected_k[1:])
    assert torch.equal(payload["slot_mapping"], event.attn_metadata.slot_mapping)
    assert state._logged_prefill_compare and state._logged_dflash_prefix_dump
    assert len(warnings) == 2


def test_nan_dump_and_compare_flag_are_shared_across_observers(observation):
    event, payloads, warnings, _ = observation
    event.dump_enabled = False
    event.out_seq[0, 0, 0] = torch.nan
    prefill_debug.emit(event)
    assert event.reference is not None and event.reference.nan_count == 1
    assert len(payloads) == 1
    assert torch.equal(payloads[0]["key_cache"], event.key_cache)
    assert torch.isnan(payloads[0]["out_seq"]).sum().item() == 1
    assert state._logged_prefill_compare
    assert not state._logged_dflash_prefix_dump
    assert len(warnings) == 2
    # A new subscriber instance still sees the process-shared one-shot gate.
    debug_compare.PrefixReferenceObserver()(event)
    debug_compare.PrefixReportObserver()(event)
    assert len(payloads) == 1 and len(warnings) == 2
