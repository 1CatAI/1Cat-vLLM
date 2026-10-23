# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only CPU timeline hooks, imported as a worker extension.

Place this module in the benchmark package. It observes the installed PLE
implementation without replacing its computation or IPC protocol. Results
are buffered until shutdown/RPC; timings include Python hook overhead and
must not be reported as uninstrumented TPOT.
"""

import functools
import json
import os
import resource
import threading
import time
from pathlib import Path

from vllm.v1.ple_offload import connector, worker

_records = []
_sequence = 0
_context = None
_original_cpu_main = worker.PleOffloadWorker.proc_main


def record(stage, start, end, metadata=None):
    _records.append(
        {
            "stage": stage,
            "start_ns": start,
            "end_ns": end,
            "sequence": _context,
            "thread": threading.get_ident(),
            "metadata": metadata,
        }
    )


def observe(owner, name, stage):
    original = getattr(owner, name)

    @functools.wraps(original)
    def observed(*args, **kwargs):
        before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            end = time.perf_counter_ns()
            after = resource.getrusage(resource.RUSAGE_SELF)
            record(
                stage,
                start,
                end,
                {
                    "major_faults": after.ru_majflt - before.ru_majflt,
                    "minor_faults": after.ru_minflt - before.ru_minflt,
                    "block_inputs": after.ru_inblock - before.ru_inblock,
                },
            )

    setattr(owner, name, observed)


class SocketProbe:
    def __init__(self, socket):
        self.socket = socket

    def send(self, *args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return self.socket.send(*args, **kwargs)
        finally:
            record("publish_socket", start, time.perf_counter_ns())


_original_process = connector.PleOffloadConnector._process_request


@functools.wraps(_original_process)
def process_request(self, pending, socket):
    global _sequence, _context
    _sequence += 1
    _context = _sequence
    start = time.perf_counter_ns()
    try:
        return _original_process(self, pending, SocketProbe(socket))
    finally:
        record(
            "notifier_wait_and_publish",
            start,
            time.perf_counter_ns(),
            {
                "tokens": pending.request.num_tokens,
                "requests": pending.request.num_reqs,
            },
        )
        _context = None


def cpu_profile_main(**kwargs):
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding

    observe(Qwen4ExpNGramEmbedding, "compute_ngram_ids", "key_compute")
    observe(Qwen4ExpNGramEmbedding, "_gather_mapped_rows", "mmap_gather")
    observe(Qwen4ExpNGramEmbedding, "_disk_embedding_lookup", "gather_and_stage")
    observe(Qwen4ExpNGramEmbedding, "forward_impl", "lookup_forward")
    observe(worker, "wait_host_resets", "previous_consumer_wait")
    observe(worker, "publish_host_flag", "flag_publish")
    original = worker.PleOffloadRunner._handle_requests

    @functools.wraps(original)
    def handle(self, requests):
        global _sequence, _context
        _sequence += 1
        _context = _sequence
        start = time.perf_counter_ns()
        try:
            return original(self, requests)
        finally:
            record(
                "cpu_request",
                start,
                time.perf_counter_ns(),
                [
                    {
                        "tokens": r.num_tokens,
                        "requests": r.num_reqs,
                        "dp_rank": r.dp_rank,
                    }
                    for r in requests
                ],
            )
            _context = None

    worker.PleOffloadRunner._handle_requests = handle
    try:
        return _original_cpu_main(**kwargs)
    finally:
        Path(f"ple-phases-cpu-{os.getpid()}.json").write_text(
            json.dumps({"instrumented": True, "records": _records}) + "\n"
        )


connector.PleOffloadConnector._process_request = process_request
worker.PleOffloadWorker.proc_main = staticmethod(cpu_profile_main)


class PlePhaseWorkerExtension:
    def ple_phase_records(self):
        return {"instrumented": True, "rank": self.rank, "records": _records}
