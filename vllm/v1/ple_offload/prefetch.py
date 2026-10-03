# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best-effort bounded PLE row warming from a completed draft prefix."""

import queue
import threading

import msgspec
import torch
import zmq

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.ple_offload.protocol import PleOffloadRequest

logger = init_logger(__name__)


@triton.jit
def _pack_prefix(
    request_indices,
    computed,
    sampled,
    last_tokens,
    all_tokens,
    drafts,
    output,
    all_stride: tl.constexpr,
    draft_stride: tl.constexpr,
    output_stride: tl.constexpr,
    context_length: tl.constexpr,
    eos: tl.constexpr,
):
    req = tl.program_id(0)
    state = tl.load(request_indices + req)
    valid = tl.load(sampled + req) > 0
    end = tl.load(computed + state)
    base = output + req * output_stride
    tl.store(base, valid)
    tl.store(base + 1, tl.load(last_tokens + state))
    tl.store(base + 2, tl.load(drafts + req * draft_stride))
    tl.store(base + 3, tl.load(drafts + req * draft_stride + 1))
    for shift in range(context_length):
        position = end - context_length + shift
        token = tl.load(
            all_tokens + state * all_stride + position, mask=position >= 0, other=eos
        )
        tl.store(base + 4 + shift, token)


class DraftPlePrefetcher:
    """Separate staging prevents prefetch from overwriting demand inputs.

    The CPU receiver caches immutable rows only. These notifications never
    publish model results, release a semaphore, or update request history.
    """

    def __init__(self, config, device, ipc_addr, dp_rank):
        self.device = device
        self.dp_rank = dp_rank
        self.context_length = int(config.model_config.hf_text_config.ngram_size) - 1
        self.eos = int(config.model_config.hf_text_config.eos_token_id)
        self.stride = 4 + self.context_length
        capacity = config.max_concurrent_batches + 1
        self._free: queue.Queue[int] = queue.Queue()
        self._pending: queue.Queue[tuple[int, int] | None] = queue.Queue(
            maxsize=capacity
        )
        self._closed = False
        self._failed = False
        self._copy_stream = torch.cuda.Stream(device=device)
        self._gpu = []
        self._cpu = []
        self._events = []
        for index in range(capacity):
            shape = (config.scheduler_config.max_num_seqs, self.stride)
            self._gpu.append(torch.empty(shape, dtype=torch.int32, device=device))
            self._cpu.append(torch.empty(shape, dtype=torch.int32, pin_memory=True))
            self._events.append(torch.cuda.Event())
            self._free.put_nowait(index)
        self._thread = threading.Thread(
            target=self._send, args=(ipc_addr,), name="ple-draft-prefetch", daemon=True
        )
        self._thread.start()

    def enqueue(self, input_batch, states, sampled, drafts):
        if self._closed or self._failed or drafts.shape[1] < 2:
            return
        # Large chunked prefills already have their own disk lookup overlap.
        if input_batch.num_reqs == 0 or input_batch.is_prefilling_np.any():
            return
        try:
            slot = self._free.get_nowait()
        except queue.Empty:
            return
        count = input_batch.num_reqs
        gpu = self._gpu[slot]
        _pack_prefix[(count,)](
            input_batch.idx_mapping,
            states.num_computed_tokens.gpu,
            sampled,
            states.last_sampled_tokens,
            states.all_token_ids.gpu,
            drafts,
            gpu,
            states.all_token_ids.gpu.stride(0),
            drafts.stride(0),
            self.stride,
            self.context_length,
            self.eos,
            num_warps=1,
        )
        main_stream = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(main_stream)
            self._cpu[slot][:count].copy_(gpu[:count], non_blocking=True)
            self._events[slot].record(self._copy_stream)
        self._pending.put_nowait((slot, count))

    def _send(self, ipc_addr):
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.connect(ipc_addr)
        try:
            while (item := self._pending.get()) is not None:
                slot, count = item
                try:
                    with torch.accelerator.device_index(self.device.index):
                        self._events[slot].synchronize()
                    rows = [r for r in self._cpu[slot][:count].tolist() if r[0]]
                    if rows:
                        request = PleOffloadRequest(
                            dp_rank=self.dp_rank,
                            num_tokens=len(rows) * 3,
                            num_reqs=len(rows),
                            prefetch_ids=[token for row in rows for token in row[1:4]],
                            prefetch_context=[row[4:] for row in rows],
                        )
                        socket.send(msgspec.msgpack.encode(request))
                finally:
                    self._free.put_nowait(slot)
        except Exception:
            self._failed = True
            logger.exception("PLE draft prefetch failed; demand lookup remains active")
        finally:
            socket.close()
            context.term()

    def close(self):
        self._closed = True
        if self._thread.is_alive():
            try:
                self._pending.put(None, timeout=5)
            except queue.Full:
                logger.error("Timed out stopping PLE draft prefetch")
            self._thread.join(timeout=5)
