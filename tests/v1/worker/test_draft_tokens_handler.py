# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu.spec_decode.utils as spec_utils
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


def _make_handler(monkeypatch=None) -> tuple[DraftTokensHandler, torch.device]:
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        return DraftTokensHandler(device), device

    assert monkeypatch is not None

    class _DummyStream:
        def wait_stream(self, _other) -> None:
            return None

    class _DummyEvent:
        def record(self, *_args, **_kwargs) -> None:
            return None

        def synchronize(self) -> None:
            return None

    monkeypatch.setattr(torch.cuda, "Stream", lambda *_a, **_k: _DummyStream())
    monkeypatch.setattr(torch.cuda, "Event", lambda *_a, **_k: _DummyEvent())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_a, **_k: _DummyStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda *_a, **_k: nullcontext())
    device = torch.device("cpu")
    return DraftTokensHandler(device), device


def test_interleaved_prefill_preserves_structured_draft_tokens(monkeypatch=None):
    handler, device = _make_handler(monkeypatch)

    # Step 1: decode batch with a structured-output request proposes 4 drafts.
    decode_batch = SimpleNamespace(
        req_ids=["req-struct-1", "req-plain-1"],
        has_structured_output_reqs=True,
    )
    decode_drafts = torch.tensor(
        [[763, 220, 21, 15], [10, 11, 12, 13]], dtype=torch.int32, device=device
    )
    handler.set_draft_tokens(decode_batch, decode_drafts)

    # Step 2: pure-phase-separation prefill step for an unrelated request runs
    # before EngineCore calls take_draft_token_ids() for req-struct-1.
    prefill_batch = SimpleNamespace(
        req_ids=["req-prefill-other"],
        has_structured_output_reqs=False,
    )
    prefill_drafts = torch.tensor([[99, 98, 97, 96]], dtype=torch.int32, device=device)
    handler.set_draft_tokens(prefill_batch, prefill_drafts)

    out = handler.get_draft_tokens()
    assert out is not None
    by_req = dict(zip(out.req_ids, out.draft_token_ids))
    assert by_req["req-struct-1"] == [763, 220, 21, 15]
    assert "req-prefill-other" not in by_req

    # Step 3: another structured-output prefill finishes while req-struct-1 has
    # not advanced yet; both requests' latest proposals must remain available.
    struct_prefill_batch = SimpleNamespace(
        req_ids=["req-struct-2"],
        has_structured_output_reqs=True,
    )
    struct_prefill_drafts = torch.tensor(
        [[41, 42, 43, 44]], dtype=torch.int32, device=device
    )
    handler.set_draft_tokens(struct_prefill_batch, struct_prefill_drafts)

    out2 = handler.get_draft_tokens()
    assert out2 is not None
    by_req2 = dict(zip(out2.req_ids, out2.draft_token_ids))
    assert by_req2["req-struct-1"] == [763, 220, 21, 15]
    assert by_req2["req-struct-2"] == [41, 42, 43, 44]

    # Removing finished/preempted requests cleans up both pending and cached state.
    handler.set_draft_tokens(
        SimpleNamespace(req_ids=["req-retired"], has_structured_output_reqs=True),
        torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=device),
    )
    handler.remove_request("req-retired")
    handler.remove_request("req-struct-1")
    handler.remove_request("req-plain-1")
    out3 = handler.get_draft_tokens()
    assert out3 is not None
    by_req3 = dict(zip(out3.req_ids, out3.draft_token_ids))
    assert list(by_req3.keys()) == ["req-struct-2"]
    assert by_req3["req-struct-2"] == [41, 42, 43, 44]
    _ = spec_utils


def test_non_structured_fallback_returns_placeholder_rows(monkeypatch=None):
    handler, device = _make_handler(monkeypatch)
    batch = SimpleNamespace(
        req_ids=["r0", "r1"],
        has_structured_output_reqs=False,
    )
    handler.set_draft_tokens(
        batch, torch.zeros((2, 4), dtype=torch.int32, device=device)
    )
    out = handler.get_draft_tokens()
    assert out is not None
    assert out.req_ids == ["r0", "r1"]
    assert out.draft_token_ids == [[-1, -1, -1, -1], [-1, -1, -1, -1]]
