# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute verifier sampling in a retained Torch trace without re-profiling."""

import argparse
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def ledger(trace: Path):
    opener = gzip.open if trace.suffix == ".gz" else open
    with opener(trace, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = sorted(
        (e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"]
    )
    rows, windows = defaultdict(list), []
    for index, kernel in enumerate(kernels):
        if kernel["name"] != "_prepare_inputs":
            continue
        stop = next(
            (
                j
                for j in range(index + 1, len(kernels))
                if kernels[j]["name"]
                in ("_prepare_inputs", "_prepare_dflash_inputs_kernel")
            ),
            None,
        )
        if stop is None or kernels[stop]["name"] != "_prepare_dflash_inputs_kernel":
            continue
        segment = kernels[index:stop]
        start = segment[0]["ts"]
        end = max(e["ts"] + e["dur"] for e in segment)
        gaps = [
            max(0, b["ts"] - a["ts"] - a["dur"]) for a, b in zip(segment, segment[1:])
        ]
        sync = [
            dict(name=e["name"], duration_us=e.get("dur", 0))
            for e in events
            if e.get("cat") == "cuda_runtime"
            and "Synchronize" in e.get("name", "")
            and start <= e.get("ts", -1) < end
        ]
        for entry in segment:
            rows[entry["name"]].append(entry["dur"])
        windows.append(
            dict(
                envelope_us=end - start,
                gpu_service_us=sum(e["dur"] for e in segment),
                positive_gap_us=sum(gaps),
                largest_gap_us=max(gaps, default=0),
                kernel_count=len(segment),
                overlapping_cpu_sync_apis=sync,
                compact_calls=sum(
                    "sparse_topk_rejection_kernel" in e["name"] for e in segment
                ),
                dense_calls=sum(e["name"] == "_rejection_kernel" for e in segment),
            )
        )
    if not windows:
        raise ValueError("No complete verifier-sampling windows found in this trace")
    return dict(
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        trace=str(trace.resolve()),
        steps=len(windows),
        windows=windows,
        boundary="_prepare_inputs to before _prepare_dflash_inputs_kernel",
        interpretation="GPU envelope includes target head, TP selection and state "
        "bookkeeping. CPU sync timestamps alone do not prove host gaps.",
        mean_envelope_us=statistics.fmean(w["envelope_us"] for w in windows),
        kernels=[
            dict(
                name=name,
                calls=len(values),
                calls_per_step=len(values) / len(windows),
                mean_us=statistics.fmean(values),
                service_us_per_step=sum(values) / len(windows),
            )
            for name, values in rows.items()
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = ledger(args.trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    table = [
        (
            f"{result['steps']} complete steps; envelope "
            f"{result['mean_envelope_us']:.2f} us (includes head and bookkeeping)."
        ),
        "",
        "| Kernel | Calls/step | Mean us | Service us/step |",
        "|---|---:|---:|---:|",
    ]
    for row in result["kernels"]:
        table.append(
            f"| {row['name']} | {row['calls_per_step']:.2f} | "
            f"{row['mean_us']:.2f} | {row['service_us_per_step']:.2f} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(table) + "\n")
    print(
        json.dumps(dict(steps=result["steps"], envelope_us=result["mean_envelope_us"]))
    )


if __name__ == "__main__":
    main()
