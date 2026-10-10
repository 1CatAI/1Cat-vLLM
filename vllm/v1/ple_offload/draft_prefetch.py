# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefetch PLE rows for drafted tokens while the drafter is still running.

The next verify window of a request is ``[bonus, d1 .. dK]``. Its tokens become
known one draft step at a time, milliseconds before the target forward reaches
the PLE layer. TP0 copies each newly known token into a small host-mapped ring;
the CPU worker hashes it and asks the kernel to read the table pages of its
rows in the background. Only page-cache residency changes, never a result:
the verify request still computes every row from its own inputs.
"""

import ctypes
import os
import threading
import time
from collections.abc import Callable

import numpy as np
import torch
from cuda.bindings import driver as cuda_driver

from vllm.logger import init_logger
from vllm.model_executor.kernels.ple.host_result import _shared_zeros
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# header: round, published window positions, requests
_HEADER = 8
_ROUND, _VALID, _REQS = 0, 1, 2
_MADV_WILLNEED = 3

RowIds = Callable[[list[int], list[int]], np.ndarray]


def ring_shape(max_reqs: int, num_spec: int) -> tuple[int]:
    # header | context+bonus [max_reqs, 3] | drafts [max_reqs, num_spec]
    return (_HEADER + max_reqs * 3 + max_reqs * num_spec,)


# Per-round scalars vary every call; never specialize (or recompile) on them.
@triton.jit(do_not_specialize=["num_reqs", "round_id", "eos"])
def _publish_round_kernel(
    idx_mapping,
    num_computed,
    all_tokens,
    tokens_stride,
    context,
    header,
    num_reqs,
    round_id,
    eos,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    valid = lane < num_reqs
    request = tl.load(idx_mapping + lane, mask=valid, other=0).to(tl.int64)
    end = tl.load(num_computed + request, mask=valid, other=0).to(tl.int64)
    for offset in tl.static_range(3):
        position = end - 2 + offset
        token = tl.load(
            all_tokens + request * tokens_stride + tl.maximum(position, 0),
            mask=valid & (position >= 0),
            other=eos,
        ).to(tl.int64)
        tl.store(context + lane * 3 + offset, token, mask=valid)
    tl.debug_barrier()
    tl.store(header + 2, num_reqs.to(tl.int64))
    tl.store(header + 1, 1)
    tl.debug_barrier()
    # Release the window and its position count before the new round number:
    # a reader that sees this round also sees its count, never a stale one.
    tl.atomic_xchg(header + 0, round_id.to(tl.int64), sem="release", scope="sys")


@triton.jit(do_not_specialize=["rows"])
def _publish_drafts_kernel(
    draft_tokens,
    draft_stride,
    draft_step,
    drafts,
    header,
    rows,
    NUM_SPEC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    valid = lane < rows * NUM_SPEC
    row = lane // NUM_SPEC
    column = lane % NUM_SPEC
    token = tl.load(draft_tokens + row * draft_stride + column, mask=valid, other=0)
    tl.store(drafts + lane, token.to(tl.int64), mask=valid)
    step = tl.load(draft_step).to(tl.int64)
    tl.debug_barrier()
    # Positions bonus..d(step+1) are now known.
    tl.atomic_xchg(header + 1, step + 2, sem="release", scope="sys")


class DraftTokenRing:
    """TP0's host-mapped publication of each round's verify-window tokens.

    Each publication is one kernel that stores straight into the mapped ring.
    Nothing is copied from the host, so the CPU never waits on the stream.
    """

    def __init__(self, device: torch.device, max_reqs: int, num_spec: int):
        self.device = device
        self.max_reqs = max_reqs
        self.num_spec = num_spec
        self.host = _shared_zeros(ring_shape(max_reqs, num_spec), torch.int64)
        self._registered = 0
        self._round = 0
        self._round_block = triton.next_power_of_2(max_reqs)
        self._draft_block = triton.next_power_of_2(max_reqs * num_spec)
        with torch.accelerator.device_index(device.index):
            # Make the device's primary context current before registration.
            torch.cuda.init()
            torch.cuda.current_stream(device)
            size = self.host.numel() * self.host.element_size()
            status = cuda_driver.cuMemHostRegister(self.host.data_ptr(), size, 3)
            if status[0].value != 0:
                raise RuntimeError(f"PLE draft ring registration failed: {status[0]}")
            self._registered = self.host.data_ptr()
            mapped = torch.ops._C.get_cuda_view_from_cpu_tensor(self.host)
            if mapped.data_ptr() != self.host.data_ptr():
                raise RuntimeError("PLE draft ring has no stable mapped CUDA view")
            context_end = _HEADER + max_reqs * 3
            self._header = mapped[:_HEADER]
            self._context = mapped[_HEADER:context_end]
            self._drafts = mapped[context_end:]

    def validate_registration(self) -> None:
        if self.host.data_ptr() != self._registered or not self.host.is_pinned():
            raise RuntimeError("PLE draft ring storage changed during registration")

    def publish_round(
        self,
        idx_mapping: torch.Tensor,
        num_reqs: int,
        all_token_ids: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        eos_token_id: int,
    ) -> None:
        """Publish each request's two context tokens and its bonus token.

        Called on the model stream after the sampled tokens are committed and
        before drafting; position ``num_computed`` holds the bonus token.
        """
        if not 0 < num_reqs <= self.max_reqs:
            return
        self._round += 1
        _publish_round_kernel[(1,)](
            idx_mapping,
            num_computed_tokens,
            all_token_ids,
            all_token_ids.stride(0),
            self._context,
            self._header,
            num_reqs,
            self._round,
            eos_token_id,
            BLOCK=self._round_block,
            num_warps=1,
        )

    def publish_draft_step(
        self, draft_tokens: torch.Tensor, draft_step: torch.Tensor
    ) -> None:
        """Publish drafts through ``draft_step``; safe inside CUDA graphs."""
        if draft_tokens.shape[1] != self.num_spec:
            raise ValueError("PLE draft ring geometry differs from the drafter")
        _publish_drafts_kernel[(1,)](
            draft_tokens,
            draft_tokens.stride(0),
            draft_step,
            self._drafts,
            self._header,
            min(draft_tokens.shape[0], self.max_reqs),
            NUM_SPEC=self.num_spec,
            BLOCK=self._draft_block,
            num_warps=1,
        )

    def close(self) -> None:
        if self._registered:
            with torch.accelerator.device_index(self.device.index):
                torch.accelerator.synchronize()
                cuda_driver.cuMemHostUnregister(self._registered)
            self._registered = 0


class DraftRowPrefetcher:
    """Worker thread turning published tokens into page-cache readahead."""

    def __init__(
        self,
        ring: torch.Tensor,
        max_reqs: int,
        num_spec: int,
        tables: list[tuple[RowIds, int, int, int]],
        poll_s: float = 50e-6,
    ):
        """``tables`` holds ``(row_ids, base address, row bytes, rows)``.

        ``row_ids(tokens, context)`` hashes one request's consecutive window
        tokens, preceded by its two context tokens, into that table's rows.
        """
        self._ring = ring.numpy()
        self._max_reqs = max_reqs
        self._num_spec = num_spec
        self._tables = tables
        self._poll_s = poll_s
        self._page = os.sysconf("SC_PAGE_SIZE")
        self._madvise = ctypes.CDLL(None, use_errno=True).madvise
        self._madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        self._stop = threading.Event()
        self.rounds = self.positions = self.pages = 0
        self._thread = threading.Thread(
            target=self._run, name="ple-draft-prefetch", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)

    def _windows(self, num_reqs: int) -> np.ndarray:
        start = _HEADER
        context = self._ring[start : start + self._max_reqs * 3]
        drafts = self._ring[start + self._max_reqs * 3 :]
        context = context.reshape(self._max_reqs, 3)[:num_reqs]
        drafts = drafts.reshape(self._max_reqs, self._num_spec)[:num_reqs]
        return np.concatenate((context, drafts), axis=1)

    def _advise(self, rows: np.ndarray, base: int, row_bytes: int, total: int):
        page = self._page
        rows = rows[(rows >= 0) & (rows < total)]
        first = (base + rows * row_bytes) // page
        last = (base + rows * row_bytes + row_bytes - 1) // page
        for number in np.unique(np.concatenate((first, last))).tolist():
            # Asynchronous readahead; resident pages return immediately.
            self._madvise(number * page, page, _MADV_WILLNEED)
            self.pages += 1

    def prefetch(self, window: np.ndarray, begin: int, end: int) -> None:
        """Advise rows of window positions ``[begin, end)`` for every request."""
        sequences = window.tolist()  # context0, context1, bonus, d1 .. dK
        for row_ids, base, row_bytes, total in self._tables:
            rows = [
                np.asarray(row_ids(s[2 + begin : 2 + end], s[begin : begin + 2]))
                for s in sequences
            ]
            if rows:
                self._advise(np.concatenate(rows).reshape(-1), base, row_bytes, total)
        self.positions += end - begin

    def _run(self) -> None:
        ring = self._ring
        current, done = -1, 0
        limit = self._num_spec + 1
        while not self._stop.is_set():
            round_id, valid, num_reqs = (int(x) for x in ring[:3])
            if round_id != current:
                current, done = round_id, 0
                self.rounds += 1
            valid = min(valid, limit)
            if valid > done and 0 < num_reqs <= self._max_reqs:
                window = self._windows(num_reqs).copy()
                # A newer round may have overwritten the window meanwhile.
                if int(ring[_ROUND]) == current:
                    try:
                        self.prefetch(window, done, valid)
                    except Exception:  # never disturb the result path
                        logger.exception("PLE draft prefetch failed; disabling it")
                        return
                done = valid
                continue
            time.sleep(self._poll_s)
