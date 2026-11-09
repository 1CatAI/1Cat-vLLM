# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded CUDA-event phase timing without a CUPTI/node profiler.

Events are initialized before the request. There is no per-round synchronize;
flush happens after generation. Compare endpoint cost/output with the ordinary
control before using these diagnostics. Event ranges may nest/overlap.
"""

import time

import torch


def install(worker):
    if hasattr(worker, "_mtp_phase_events"):
        raise RuntimeError("Phase events already installed")
    runner = worker.model_runner
    draft = runner.speculator
    if draft is None or draft.num_speculative_steps != 4:
        raise ValueError("Phase timing requires the fixed MTP4 control")
    state = {"restore": [], "observations": [], "ordinal": 0}

    def wrap(obj, name, label, capacity=1024):
        original = getattr(obj, name)
        # Warm all lazy CUDA handles outside the measured request.
        pool = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(capacity)
        ]
        for begin, end in pool:
            begin.record()
            end.record()
        cursor = 0

        def observed(*args, **kwargs):
            nonlocal cursor
            if cursor == len(pool):
                raise RuntimeError("Phase event capacity exhausted")
            begin, end = pool[cursor]
            cursor += 1
            annotation = label(*args, **kwargs) if callable(label) else label
            host_start = time.monotonic_ns()
            begin.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                state["observations"].append(
                    (annotation, begin, end, host_start, time.monotonic_ns())
                )

        state["restore"].append((obj, name, original))
        setattr(obj, name, observed)

    def proposal(*args, **kwargs):
        state["ordinal"] = 0
        return "draft_all"

    def decode(desc):
        state["ordinal"] += 1
        if state["ordinal"] > 3:
            raise RuntimeError("Unexpected draft window")
        return f"draft_step/{state['ordinal']}/M{desc.num_tokens}"

    wrap(runner, "execute_model", "target_execute")
    wrap(runner, "prepare_inputs", "prepare")
    wrap(runner, "sample_tokens", "sample_handoff_draft")
    wrap(runner, "sample", "target_head_sample")
    wrap(
        runner.cudagraph_manager,
        "run_fullgraph",
        lambda desc: f"target_forward/M{desc.num_tokens}",
    )
    wrap(draft, "propose", proposal)
    wrap(
        draft.prefill_cudagraph_manager,
        "run_fullgraph",
        lambda desc: f"draft_step/0/M{desc.num_tokens}",
    )
    wrap(draft.decode_cudagraph_manager, "run_fullgraph", decode, capacity=3072)
    torch.accelerator.synchronize()
    worker._mtp_phase_events = state
    return {"rank": worker.rank, "kind": "cuda_events_no_profiler"}


def flush(worker):
    state = worker._mtp_phase_events
    for obj, name, original in reversed(state["restore"]):
        setattr(obj, name, original)
    torch.accelerator.synchronize()
    records = state["observations"]
    origin = min(records, key=lambda r: r[3])[1] if records else None
    result = []
    for label, begin, end, host_start, host_end in records:
        result.append(
            {
                "label": label,
                "gpu_start_ms": origin.elapsed_time(begin),
                "gpu_duration_ms": begin.elapsed_time(end),
                "host_start_ns": host_start,
                "host_end_ns": host_end,
            }
        )
    del worker._mtp_phase_events
    return {"rank": worker.rank, "records": result}
