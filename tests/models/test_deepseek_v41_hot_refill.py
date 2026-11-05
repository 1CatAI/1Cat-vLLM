# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-prefix hits past the packed-KV hot ring (1Cat #1341).

Readers take hot slot ``c % hot_rows`` as row ``c`` for every ``c`` in
``(newest - hot_rows, newest]``. The previous request wrote rows past the hit
(its answer and up to 4 rejected drafts); once the ring has wrapped, those sit
in the slots of rows about ``hot_rows`` back. After ``refill_hot(cached)``
every slot a reader can take as current must hold the host row it stands for.
"""

import types

import pytest
import torch

from vllm.models.deepseek_v41.kv_store import V41PackedKV
from vllm.models.deepseek_v41.model import DeepseekV41Model

# Ratio-4 and ratio-128 KV source layers, as in the V4.1-Flash config.
SOURCES = {2: 4, 8: 128, 14: 4, 20: 4}
DRAFTS = 4  # 5 draft tokens: rejected drafts reach processed + 4
WIDTH = 3


def _store(hot_positions, prompt, answer):
    """A store after a request of prompt + answer tokens plus rejected drafts."""
    st = types.SimpleNamespace(
        sources=SOURCES,
        hot_rows={s: max(1, hot_positions // r) for s, r in SOURCES.items()},
    )
    written = prompt + answer + DRAFTS
    st.host = {
        s: torch.zeros(written // r + 1, WIDTH, dtype=torch.int32)
        for s, r in SOURCES.items()
    }
    st.hot = {s: torch.zeros(st.hot_rows[s], WIDTH, dtype=torch.int32) for s in SOURCES}
    for s, r in SOURCES.items():
        for row in range(written // r):
            value = torch.tensor([row, 1, 7], dtype=torch.int32)
            st.hot[s][row % st.hot_rows[s]] = value
            st.host[s][row] = value
    return st


def _stale_slots(st, cached, prompt, answer):
    """Mark the rows past the hit, then count slots that disagree with host."""
    written = prompt + answer + DRAFTS
    for s, r in SOURCES.items():
        for row in range(cached // r, written // r):
            st.hot[s][row % st.hot_rows[s]][1] = 99


def _mismatched(st, cached):
    bad = 0
    for s, r in SOURCES.items():
        hi = cached // r - 1
        rows = st.hot_rows[s]
        for c in range(max(0, hi - rows + 1), hi + 1):
            bad += not torch.equal(st.hot[s][c % rows], st.host[s][c])
    return bad


@pytest.mark.parametrize(
    ("prompt", "answer", "hot_positions"),
    [
        (40000, 64, 32768),
        (39560, 64, 32768),  # the prompt length reported in #1341
        (250000, 64, 32768),
        (500000, 64, 32768),
        (37000, 1, 32768),
        (100, 64, 64),
        (2000, 300, 1024),
    ],
)
def test_refill_restores_every_current_slot(prompt, answer, hot_positions):
    st = _store(hot_positions, prompt, answer)
    _stale_slots(st, prompt, prompt, answer)
    copied = V41PackedKV.refill_hot(st, prompt)
    assert _mismatched(st, prompt) == 0
    # Exactly the rows a reader can take as current are copied.
    want = sum(
        min(st.hot_rows[s], prompt // r)
        for s, r in SOURCES.items()
        if prompt // r - 1 + 256 >= st.hot_rows[s]
    )
    assert copied == want


def test_without_refill_the_wrapped_ring_is_stale():
    # The check above can fail: past the ring, the old rows are read as current.
    st = _store(32768, 40000, 64)
    _stale_slots(st, 40000, 40000, 64)
    assert _mismatched(st, 40000) > 0


def test_below_the_ring_the_hot_slots_are_unchanged():
    # Before the ring wraps, the slots are already right; a refill may copy
    # (the ratio-128 ring is short) but must not change any slot.
    st = _store(32768, 20000, 64)
    _stale_slots(st, 20000, 20000, 64)
    before = {s: t.clone() for s, t in st.hot.items()}
    V41PackedKV.refill_hot(st, 20000)
    assert _mismatched(st, 20000) == 0
    for s in SOURCES:
        assert torch.equal(st.hot[s], before[s])


class _Store:
    def __init__(self):
        self.refills = []

    def refill_hot(self, cached):
        self.refills.append(cached)
        return 0


def _model(store):
    vocab = 16
    return types.SimpleNamespace(
        hash_state=types.SimpleNamespace(
            history=torch.zeros(4, dtype=torch.long),
            token_map=torch.arange(vocab, dtype=torch.int32),
            metadata=torch.zeros(3, dtype=torch.int32),
        ),
        layers=[],
        _v41_shared=types.SimpleNamespace(store=store),
        _v41_hot_refill_logged=False,
    )


@pytest.mark.parametrize(("env", "want"), [(None, [6]), ("1", [6]), ("0", [])])
def test_restore_prefix_refills_the_hot_ring(monkeypatch, env, want):
    if env is None:
        monkeypatch.delenv("VLLM_V41_HOT_REFILL", raising=False)
    else:
        monkeypatch.setenv("VLLM_V41_HOT_REFILL", env)
    store = _Store()
    model = _model(store)
    DeepseekV41Model.restore_prefix(model, 6, [1, 2, 3, 4, 5, 6, 7])
    assert store.refills == want
    assert model.hash_state.history.tolist() == [4, 5, 6, 6]


def test_restore_prefix_without_packed_kv(monkeypatch):
    monkeypatch.setenv("VLLM_V41_HOT_REFILL", "1")
    model = _model(None)
    DeepseekV41Model.restore_prefix(model, 4, [1, 2, 3, 4, 5])
    assert model.hash_state.history.tolist() == [2, 3, 4, 4]


def test_ring_reach_leaves_room_for_rejected_drafts(monkeypatch):
    # Window rings hold 256 rows; a hit at P needs rows [P - 128, P). Rejected
    # drafts reach processed + 4, so a hit may be at most 128 - 4 behind.
    from vllm import envs

    monkeypatch.delenv("VLLM_V41_RING_REACH", raising=False)
    assert int(envs.VLLM_V41_RING_REACH) == 128 - DRAFTS
