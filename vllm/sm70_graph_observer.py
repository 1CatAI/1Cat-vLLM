# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit benchmark-only CPU stage observer for TP graph launch diagnosis."""

import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any


class CPUStageRecorder:
    def __init__(self, rank, limit=131072):
        self.rank = rank
        self.limit = limit
        self.enabled = False
        self.nvtx = False
        self.events = []
        self.step = 0
        self.dropped = 0

    @contextmanager
    def stage(self, label, metadata=None):
        if not self.enabled:
            yield
            return
        event = {
            "label": label,
            "step": self.step,
            "thread": threading.get_native_id(),
            "start_ns": time.perf_counter_ns(),
            **(metadata or {}),
        }
        if self.nvtx:
            import torch

            torch.cuda.nvtx.range_push("graph_parity." + label)
        try:
            yield
        finally:
            if self.nvtx:
                torch.cuda.nvtx.range_pop()
            event["end_ns"] = time.perf_counter_ns()
            if len(self.events) < self.limit:
                self.events.append(event)
            else:
                self.dropped += 1

    def wrap(self, owner, name, label, metadata=None, advances_step=False):
        if owner is None or not hasattr(owner, name):
            return
        original = getattr(owner, name)

        @wraps(original)
        def observed(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            if advances_step:
                self.step += 1
            fields = metadata(*args, **kwargs) if metadata else None
            with self.stage(label, fields):
                return original(*args, **kwargs)

        setattr(owner, name, observed)

    def read(self, stop=True):
        if stop:
            self.enabled = False
        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "clock": "perf_counter_ns",
            "events": list(self.events),
            "dropped": self.dropped,
        }


class GraphParityWorkerExtension:
    """Attach through worker_extension_cls; ordinary workers remain unchanged."""

    rank: int
    model_runner: Any

    def start_graph_parity_observer(self, nvtx=False):
        if not hasattr(self, "_graph_parity_recorder"):
            from vllm.v1.executor.multiproc_executor import WorkerProc
            from vllm.v1.worker.gpu.async_utils import AsyncOutput

            rec = CPUStageRecorder(self.rank)
            self._graph_parity_recorder = rec
            runner = self.model_runner
            rec.wrap(self, "execute_model", "worker.execute", advances_step=True)
            rec.wrap(self, "sample_tokens", "worker.sample")
            for name in (
                "prepare_inputs",
                "prepare_attn",
                "finish_requests",
                "free_states",
                "add_requests",
                "update_requests",
                "sample",
                "postprocess_sampled",
            ):
                rec.wrap(runner, name, "runner." + name)
            for name in ("prepare_inputs", "prepare_attn", "preprocess_state"):
                rec.wrap(runner.model_state, name, "model_state." + name)
            rec.wrap(runner.block_tables, "apply_staged_writes", "block_tables.apply")
            rec.wrap(runner.speculator, "propose", "draft.propose")
            rec.wrap(runner._ple_offload_connector, "prepare_forward", "ple.prepare")
            rec.wrap(
                runner.cudagraph_manager,
                "run_fullgraph",
                "target.replay",
                metadata=lambda desc: {
                    "tokens": desc.num_tokens,
                    "requests": desc.num_reqs,
                    "uniform": desc.uniform_token_count,
                },
            )
            rec.wrap(AsyncOutput, "get_output", "output.materialize")
            rec.wrap(WorkerProc, "enqueue_output", "output.serialize")
        rec = self._graph_parity_recorder
        rec.events.clear()
        rec.step = rec.dropped = 0
        rec.nvtx = nvtx
        rec.enabled = True
        return {"rank": self.rank, "clock": "perf_counter_ns", "nvtx": nvtx}

    def read_graph_parity_observer(self, stop=True):
        return self._graph_parity_recorder.read(stop)

    def start_graph_parity_capture(self):
        import torch

        torch.cuda.cudart().cudaProfilerStart()

    def stop_graph_parity_capture(self):
        import torch

        torch.accelerator.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
