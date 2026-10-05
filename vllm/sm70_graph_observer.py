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
        self._target = threading.local()

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

    def wrap_target_replay(self, manager, graph_class):
        original_manager = manager.run_fullgraph
        original_replay = graph_class.replay

        @wraps(original_manager)
        def managed(desc):
            if not self.enabled:
                return original_manager(desc)
            fields = {
                "tokens": desc.num_tokens,
                "requests": desc.num_reqs,
                "uniform": desc.uniform_token_count,
            }
            self._target.fields = fields
            try:
                with self.stage("target.manager", fields):
                    return original_manager(desc)
            finally:
                self._target.fields = None

        @wraps(original_replay)
        def replay(graph):
            fields = getattr(self._target, "fields", None)
            if not self.enabled or fields is None:
                return original_replay(graph)
            # Timestamp immediately before the actual replay API, after any
            # manager-side offloader wait. Drafter graphs remain excluded.
            with self.stage("target.replay", fields):
                return original_replay(graph)

        manager.run_fullgraph = managed
        graph_class.replay = replay

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
            import torch

            rec.wrap_target_replay(runner.cudagraph_manager, torch.cuda.CUDAGraph)
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
