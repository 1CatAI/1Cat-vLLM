# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate paired MTP4 events and close each rank's nested round intervals.

GPU event origins are independent across ranks. Never combine their timestamps
or add independently maximized phases. This report cannot admit endpoint speed.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def statistics(values):
    if not values:
        raise ValueError("Missing steady-state samples")
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
    }


def rank_phases(worker):
    records = sorted(worker["records"], key=lambda r: r["host_start_ns"])
    executions = [r for r in records if r["label"] == "target_execute"]
    rounds = []
    for index, execution in enumerate(executions[:-1]):
        next_execution = executions[index + 1]
        group = [
            r
            for r in records
            if execution["host_start_ns"]
            <= r["host_start_ns"]
            < next_execution["host_start_ns"]
        ]
        if not any(r["label"] == "target_forward/M5" for r in group):
            continue  # Chunked prefill is outside the decode-round budget.
        duration = next_execution["gpu_start_ms"] - execution["gpu_start_ms"]
        labels = {}
        for r in group:
            labels.setdefault(r["label"], []).append(r["gpu_duration_ms"])

        def one(label, labels=labels):
            values = labels.get(label, [])
            if len(values) != 1:
                raise ValueError(f"Expected one {label}, got {len(values)}")
            return values[0]

        target = one("target_forward/M5")
        draft = [
            one(f"draft_step/{step}/M{5 if step == 0 else 1}") for step in range(4)
        ]
        head = one("target_head_sample")
        draft_all = one("draft_all")
        handoff = one("sample_handoff_draft")
        parts = {
            "target_M5": target,
            **{f"draft_{i}": value for i, value in enumerate(draft)},
            "draft_outside_graphs": draft_all - sum(draft),
            "target_head_sample": head,
            "target_outside_forward": execution["gpu_duration_ms"] - target,
            "handoff_outside_head_draft": handoff - head - draft_all,
            "round_boundary": duration - execution["gpu_duration_ms"] - handoff,
        }
        # Event timestamps have microsecond-scale floating-point rounding.
        if min(parts.values()) < -0.01:
            raise ValueError("Nested intervals overlap or escape the round")
        rounds.append({"complete_round": duration, **parts})
    steady = rounds[1:-1]
    if not steady:
        raise ValueError("Too few complete M5 rounds after excluding transitions")
    return {
        "rank": worker["rank"],
        "steady_rounds": len(steady),
        "ms": {key: statistics([r[key] for r in steady]) for key in steady[0]},
    }


def analyze(report, max_overhead_fraction):
    controls = {
        (r["id"], r["repeat"]): r for r in report.get("phase_event_controls", [])
    }
    cases = []
    for case in report["cases"]:
        control = controls.get((case["id"], case["repeat"]))
        control_ms = (
            1000
            * control["metrics"]["decode_time"]
            / control["spec_decoding"]["num_drafts"]
            if control
            else None
        )
        event_ms = case["complete_round_ms"]
        match = bool(
            control
            and control["token_ids"] == case["token_ids"]
            and control["spec_decoding"] == case["spec_decoding"]
        )
        overhead = event_ms / control_ms - 1 if control_ms else None
        ranks = [rank_phases(worker) for worker in case["phase_events"]]
        if sorted(r["rank"] for r in ranks) != [0, 1, 2, 3]:
            raise ValueError("Expected all four TP ranks exactly once")
        cases.append(
            {
                "id": case["id"],
                "repeat": case["repeat"],
                "ordinary_control_round_ms": control_ms,
                "event_endpoint_round_ms": event_ms,
                "paired_token_and_counter_match": match,
                "overhead_fraction": overhead,
                "calibrated": match and abs(overhead) <= max_overhead_fraction,
                "ranks": ranks,
                "actual_steady_tokens_per_round": case["metrics"][
                    "steady_decode_tokens"
                ]
                / case["spec_decoding"]["num_drafts"],
            }
        )
    if not cases:
        raise ValueError("No measured cases")
    return {
        "source": report["source"],
        "measurement": "same-rank CUDA-event wall; diagnostic, not speed admission",
        "max_overhead_fraction": max_overhead_fraction,
        "calibrated": all(c["calibrated"] for c in cases),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-overhead-fraction", type=float, default=0.02)
    args = parser.parse_args()
    if not 0 <= args.max_overhead_fraction < 1:
        parser.error("overhead fraction must be in [0, 1)")
    result = analyze(json.loads(args.input.read_text()), args.max_overhead_fraction)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))


if __name__ == "__main__":
    main()
