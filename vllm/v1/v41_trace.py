# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wall-clock step tracing across processes and nodes (VLLM_V41_TRACE=1).

Default off. Each traced process appends ``event t_ns fields`` lines to
/tmp/v41trace-<host>-<pid>.log; benchmarks/deepseek_v41/native/trace_join.py
aligns the nodes with measured clock offsets and prints per-step timelines.
Points that need the GPU to be done synchronize first (tracing perturbs).
"""

import os
import socket
import time

import vllm.envs as envs

ENABLED = envs.VLLM_V41_TRACE == "1"
# VLLM_V41_TRACE_NOSYNC=1: never synchronize at marks (async-safe timings).
_NOSYNC = envs.VLLM_V41_TRACE_NOSYNC == "1"
_STATE: dict = {"file": None, "lines": [], "role": None}
# GPU marks: timing events on the current stream, resolved without blocking
# (query) and placed on the host clock through one synchronized reference.
_GPU: dict = {"ref": None, "ref_ns": 0, "pending": []}


def set_role(role: str) -> None:
    _STATE["role"] = role


def mark(event: str, sync: bool = False, **fields) -> None:
    if not ENABLED or _STATE["role"] is None:
        return
    if sync and not _NOSYNC:
        import torch

        torch.cuda.current_stream().synchronize()
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    _STATE["lines"].append(f"{_STATE['role']} {event} {time.time_ns()} {extra}\n")
    if len(_STATE["lines"]) >= 256:
        flush()


def gmark(event: str, **fields) -> None:
    """Record when the GPU reaches this point of the current stream."""
    if not ENABLED or _STATE["role"] is None:
        return
    import torch

    if _GPU["ref"] is None:
        ref = torch.cuda.Event(enable_timing=True)
        ref.record()
        ref.synchronize()
        _GPU["ref"], _GPU["ref_ns"] = ref, time.time_ns()
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    _GPU["pending"].append((event, ev, time.time_ns(), extra))
    _resolve()


def _resolve() -> None:
    pending = _GPU["pending"]
    while pending and pending[0][1].query():
        event, ev, issued, extra = pending.pop(0)
        t = _GPU["ref_ns"] + int(_GPU["ref"].elapsed_time(ev) * 1e6)
        _STATE["lines"].append(
            f"{_STATE['role']} g_{event} {t} issued={issued} {extra}\n"
        )
    if len(_STATE["lines"]) >= 256:
        flush()


def flush() -> None:
    if not _STATE["lines"]:
        return
    if _STATE["file"] is None:
        path = f"/tmp/v41trace-{socket.gethostname()}-{os.getpid()}.log"
        _STATE["file"] = open(path, "a", buffering=1 << 16)  # noqa: SIM115
    _STATE["file"].writelines(_STATE["lines"])
    _STATE["file"].flush()
    _STATE["lines"].clear()
