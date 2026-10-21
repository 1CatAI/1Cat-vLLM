# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for the real V2 PP feedback implementation.

Only CUDA stream/event operations and the distributed transport are replaced.
Packing, buffer views, queue advancement, and generation filtering run unchanged.
"""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pytest
import torch

from vllm.v1.worker.gpu import pp_utils

_ExpectedFeedback: TypeAlias = tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_],
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]


class _Event:
    pass


class _Stream:
    def __init__(self, trace):
        self.trace = trace

    def wait_stream(self, other):
        self.trace.append(("wait_stream", self, other))

    def wait_event(self, event):
        self.trace.append(("wait_event", self, event))

    def record_event(self, event=None):
        event = event or _Event()
        self.trace.append(("record_event", self, event))
        return event


@pytest.fixture
def feedback_pair(monkeypatch):
    trace: list[tuple[object, ...]] = []
    main_stream = _Stream(trace)
    transport: deque[torch.Tensor] = deque()
    sent: list[int] = []
    received: list[int] = []
    sender_active = False
    group = object()

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: main_stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda _device: _Stream(trace))
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)

    def broadcast(tensor, src, group):
        assert src == 6
        assert group is pair.group
        assert trace[-1][0] in ("wait_stream", "broadcast")
        assert tensor.is_contiguous()
        trace.append(("broadcast", tensor.data_ptr()))
        if sender_active:
            transport.append(tensor.clone())
            sent.append(tensor.data_ptr())
        else:
            tensor.copy_(transport.popleft())
            received.append(tensor.data_ptr())

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    def make(depth, speculative_steps=4, packed=True):
        nonlocal sender_active

        def make_group(is_last):
            def sibling(*, group_desc):
                assert group_desc == "pp_sampled_draft_broadcast"
                return group

            return SimpleNamespace(
                world_size=7,
                is_last_rank=is_last,
                last_rank=6,
                make_sibling_device_group=sibling,
            )

        monkeypatch.setattr(pp_utils, "get_pp_group", lambda: make_group(True))
        sender = pp_utils.PPHandler(
            8, speculative_steps, torch.device("cpu"), depth, packed
        )
        monkeypatch.setattr(pp_utils, "get_pp_group", lambda: make_group(False))
        receiver = pp_utils.PPHandler(
            8, speculative_steps, torch.device("cpu"), depth, packed
        )

        def exchange(batch, sampled, accepted, rejected, draft):
            nonlocal sender_active
            sender_active = True
            sender.broadcast(sampled, accepted, rejected, batch, draft)
            sender_active = False
            result = receiver.receive(batch)
            assert not transport
            return result

        pair.sender = sender
        pair.receiver = receiver
        pair.exchange = exchange
        return pair

    pair = SimpleNamespace(
        make=make, group=group, trace=trace, sent=sent, received=received
    )
    return pair


def _batch(indices, *, computed=None, scheduled=None, prefill=None, max_len=None):
    num_reqs = len(indices)

    def array(value, default):
        return np.array(value if value is not None else [default] * num_reqs)

    return SimpleNamespace(
        num_reqs=num_reqs,
        idx_mapping_np=np.array(indices, dtype=np.int32),
        idx_mapping=torch.tensor(indices, dtype=torch.int32),
        num_computed_tokens_np=array(computed, 8),
        num_scheduled_tokens=array(scheduled, 1),
        prefill_len_np=array(prefill, 8),
        max_seq_len_np=array(max_len, 100),
    )


@pytest.mark.parametrize("num_reqs", [0, 1, 3, 8])
@pytest.mark.parametrize("speculative_steps", [0, 4])
def test_packet_views_preserve_dtypes_and_share_storage(num_reqs, speculative_steps):
    token_width = 2 * speculative_steps + 1
    storage = torch.full((8 * (token_width + 1),), -9, dtype=torch.int64)
    buffer = pp_utils._FeedbackBuffer(storage, _Event())
    packet, tokens, counts = buffer.views(num_reqs, token_width)
    tokens.fill_(2**40)
    counts[0].fill_(speculative_steps + 1)
    counts[1].fill_(speculative_steps)

    assert packet.is_contiguous()
    assert packet.numel() * packet.element_size() == num_reqs * (token_width * 8 + 8)
    assert tokens.dtype == torch.int64
    assert counts.dtype == torch.int32
    assert tokens.shape == (num_reqs, token_width)
    assert counts.shape == (2, num_reqs)
    assert (tokens == 2**40).all()
    assert (storage[num_reqs * (token_width + 1) :] == -9).all()
    if num_reqs:
        assert packet.data_ptr() == storage.data_ptr() == tokens.data_ptr()
        assert counts.data_ptr() == storage.data_ptr() + num_reqs * token_width * 8


@pytest.mark.parametrize("depth", [2, 4, 7])
@pytest.mark.parametrize("speculative_steps", [0, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_feedback_ring_reorders_cancels_reuses_and_skips_batches(
    feedback_pair, depth, speculative_steps, packed
):
    pair = feedback_pair.make(depth, speculative_steps, packed)
    receiver = pair.receiver
    expected: deque[_ExpectedFeedback | None] = deque([None] * depth)
    active_receives = 0
    all_discarded = partial_discarded = delivered = 0
    generations = np.zeros(8, dtype=np.int32)

    for step in range(depth * 6 + 3):
        # Cancellation and immediate request-index reuse invalidate old packets.
        # These happen before feedback consumption, as in request reconciliation.
        if step % 3 == 0:
            for idx in (0, 2, 5):
                receiver.on_req_idx_freed(idx)
                generations[idx] += 1
        if step % (2 * depth + 3) == 0:
            for idx in range(8):
                receiver.on_req_idx_freed(idx)
                generations[idx] += 1

        previous = expected.popleft()
        output = receiver.get_prev_sampled_outputs()
        if previous is None:
            assert output is None
        else:
            indices, gen, mask, sampled, draft, accepted, rejected = previous
            keep = (generations[indices] == gen) & mask
            if not keep.any():
                all_discarded += 1
                assert output is None
            else:
                delivered += 1
                partial_discarded += int(not keep.all())
                assert output is not None
                torch.testing.assert_close(
                    output["idx_mapping"],
                    torch.from_numpy(np.where(keep, indices, -1)),
                )
                torch.testing.assert_close(output["sampled_tokens"], sampled)
                torch.testing.assert_close(output["num_sampled"], accepted)
                torch.testing.assert_close(output["num_rejected"], rejected)
                if draft is None:
                    assert output["draft_tokens"] is None
                else:
                    torch.testing.assert_close(output["draft_tokens"], draft)

        mode = step % 6
        if mode == 0:
            batch = _batch([])
        elif mode == 1:
            batch = _batch([5, 0], computed=[0, 0], scheduled=[2, 2])
        elif mode == 2:
            batch = _batch([2, 0], computed=[99, 99])
        elif mode == 3:
            # Mixed decode, unfinished prefill, and final decode.
            batch = _batch([5, 1, 2], computed=[8, 0, 99])
        elif mode == 4:
            batch = _batch([2, 7, 0, 1, 5])
        else:
            batch = _batch([1, 5, 7, 0, 2, 6, 3, 4])

        n = batch.num_reqs
        width = 1 + (step % (speculative_steps + 1))
        sampled = torch.arange(n * width, dtype=torch.int64).view(n, width) + step * 100
        draft = (
            torch.arange(n * speculative_steps, dtype=torch.int64).view(
                n, speculative_steps
            )
            + step * 1000
            if speculative_steps
            else None
        )
        accepted = torch.full((n,), width, dtype=torch.int32)
        rejected = torch.full((n,), speculative_steps + 1 - width, dtype=torch.int32)
        mask = pp_utils.compute_need_sampled_mask(batch)
        calls_before = len(pair.sent)
        all_decode = pair.exchange(batch, sampled, accepted, rejected, draft)
        if mask is None:
            assert not all_decode
            assert len(pair.sent) == calls_before
            expected.append(None)
        else:
            active_receives += 1
            assert all_decode == bool(mask.all())
            assert len(pair.sent) - calls_before == (1 if packed else 2)
            padded = torch.full((n, speculative_steps + 1), -1, dtype=torch.int64)
            padded[:, :width] = sampled
            expected.append(
                (
                    batch.idx_mapping_np.copy(),
                    generations[batch.idx_mapping_np].copy(),
                    mask,
                    padded,
                    draft,
                    accepted,
                    rejected,
                )
            )

        # A reused packet must never overwrite any still-pending slot.
        for slot, pending in zip(receiver.queue, expected):
            if pending is not None:
                assert slot is not None
                torch.testing.assert_close(slot.sampled_tokens, pending[3])
                torch.testing.assert_close(slot.num_sampled, pending[5])
                torch.testing.assert_close(slot.num_rejected, pending[6])
                if pending[4] is not None:
                    torch.testing.assert_close(slot.draft_tokens, pending[4])

    assert delivered and all_discarded and partial_discarded
    assert active_receives > depth
    assert len(pair.sent) == len(pair.received)
    if packed:
        assert len(set(pair.sent)) == len(set(pair.received)) == depth
        recorded = {id(e[2]) for e in pair.trace if e[0] == "record_event"}
        assert len(recorded) == depth


def test_packed_feedback_does_not_allocate_packets_per_batch(
    feedback_pair, monkeypatch
):
    pair = feedback_pair.make(4)
    batch = _batch([0, 1, 2])
    sampled = torch.ones((3, 5), dtype=torch.int64)
    draft = torch.ones((3, 4), dtype=torch.int64)
    accepted = torch.full((3,), 5, dtype=torch.int32)
    rejected = torch.zeros(3, dtype=torch.int32)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("feedback allocated a new packet instead of reusing its ring")

    original_stack = torch.stack

    def stack_into_existing_buffer(tensors, *, dim, out=None):
        assert out is not None
        assert out.is_contiguous()
        assert out.dtype == torch.int32
        assert out.shape == (2, batch.num_reqs)
        assert any(
            out.untyped_storage().data_ptr() == buffer.packet.data_ptr()
            for buffer in pair.sender._buffers
        )
        return original_stack(tensors, dim=dim, out=out)

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    monkeypatch.setattr(torch, "stack", stack_into_existing_buffer)
    for _ in range(12):
        pair.receiver.get_prev_sampled_outputs()
        pair.exchange(batch, sampled, accepted, rejected, draft)


def test_feedback_rejects_capacity_overflow(feedback_pair):
    pair = feedback_pair.make(4)
    with pytest.raises(ValueError, match="buffer capacity"):
        pair.receiver.receive(_batch(list(range(8)) + [0]))


def test_packing_clears_previous_wider_sample(feedback_pair):
    pair = feedback_pair.make(1)
    batch = _batch([0, 1])
    draft = torch.ones((2, 4), dtype=torch.int64)
    counts = torch.ones(2, dtype=torch.int32)
    pair.receiver.get_prev_sampled_outputs()
    pair.exchange(
        batch, torch.full((2, 5), 9, dtype=torch.int64), counts, counts, draft
    )
    pair.receiver.get_prev_sampled_outputs()
    pair.exchange(
        batch, torch.full((2, 1), 8, dtype=torch.int64), counts, counts, draft
    )
    output = pair.receiver.get_prev_sampled_outputs()
    assert output is not None
    torch.testing.assert_close(
        output["sampled_tokens"], torch.tensor([[8, -1, -1, -1, -1]] * 2)
    )


@pytest.mark.parametrize("cancel_all", [False, True])
def test_reuse_orders_main_consumers_before_next_receive(feedback_pair, cancel_all):
    pair = feedback_pair.make(1)
    batch = _batch([0, 1])
    sampled = torch.ones((2, 5), dtype=torch.int64)
    draft = torch.ones((2, 4), dtype=torch.int64)
    counts = torch.ones(2, dtype=torch.int32)
    receiver = pair.receiver
    receiver.get_prev_sampled_outputs()
    pair.exchange(batch, sampled, counts, counts, draft)
    event = receiver.queue[0].event
    receiver.on_req_idx_freed(0)
    if cancel_all:
        receiver.on_req_idx_freed(1)
    before_get = len(pair.trace)
    output = receiver.get_prev_sampled_outputs()
    if cancel_all:
        assert output is None
        # No consumer exists, so no main-stream event wait is necessary.
        assert pair.trace[before_get:] == []
    else:
        assert output is not None
        assert pair.trace[before_get:] == [("wait_event", receiver.main_stream, event)]
    pair.trace.append(("main_consumer",))
    before_reuse = len(pair.trace)
    pair.exchange(batch, sampled, counts, counts, draft)
    assert pair.trace[before_reuse:] == [
        (
            "wait_stream",
            pair.sender.broadcast_stream,
            pair.sender.main_stream,
        ),
        ("broadcast", pair.sent[-1]),
        ("wait_stream", receiver.broadcast_stream, receiver.main_stream),
        ("broadcast", pair.received[-1]),
        ("record_event", receiver.broadcast_stream, event),
    ]
    assert pair.received[0] == pair.received[1]


@pytest.mark.parametrize(
    "bad_counts", [torch.ones(2), torch.ones((2, 1), dtype=torch.int32)]
)
def test_broadcast_rejects_count_contract_before_collective(feedback_pair, bad_counts):
    pair = feedback_pair.make(4)
    with pytest.raises(ValueError, match="1D int32"):
        pair.sender.broadcast(
            torch.ones((2, 1), dtype=torch.int64),
            bad_counts,
            torch.zeros(2, dtype=torch.int32),
            _batch([0, 1]),
            torch.ones((2, 4), dtype=torch.int64),
        )
    assert not pair.sent
    assert not pair.received


@pytest.mark.parametrize("bad_shape", [(2, 4), (1, 4)])
def test_packing_rejects_mismatched_sample_rows_before_collective(
    feedback_pair, bad_shape
):
    pair = feedback_pair.make(4)
    # The draft shape matches the supplied samples, but the input batch and
    # persistent output buffer require three rows.
    with pytest.raises(ValueError, match="token packet output"):
        pair.sender.broadcast(
            torch.ones(bad_shape, dtype=torch.int64),
            torch.ones(3, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32),
            _batch([0, 1, 2]),
            torch.ones((bad_shape[0], 4), dtype=torch.int64),
        )
    assert not pair.sent
