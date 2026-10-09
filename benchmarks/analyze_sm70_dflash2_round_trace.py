# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute full DFlash2 rounds using GPU graph correlations in a Torch trace.

Kernel service and gaps are diagnostic; use an unprofiled request for latency.
Layer attribution requires the unfused graph's exact collective structure and
fails closed if a different graph structure is encountered.
"""

import argparse
import bisect
import collections
import json
import statistics
from pathlib import Path


def collective(name: str) -> str | None:
    lower = name.lower()
    if "allgather" in lower:
        return "nccl_allgather" if "nccl" in lower else "custom_allgather"
    if "allreduce" in lower:
        return "nccl_allreduce" if "nccl" in lower else "custom_allreduce"
    if "cross_device_reduce" in lower or "hierarchical_reduce_push" in lower:
        return "custom_allreduce"
    return None


def ledger(kernels: list[dict]) -> list[dict]:
    previous_end = None
    result = []
    for event in kernels:
        end = event["ts"] + event["dur"]
        result.append(
            {
                "name": event["name"],
                "ts_us": event["ts"],
                "duration_us": event["dur"],
                "gap_before_us": (
                    None if previous_end is None else event["ts"] - previous_end
                ),
                "grid": event.get("args", {}).get("grid"),
                "block": event.get("args", {}).get("block"),
                "stream": event.get("args", {}).get("stream"),
                "correlation": event.get("args", {}).get("correlation"),
                "collective": collective(event["name"]),
            }
        )
        previous_end = end if previous_end is None else max(previous_end, end)
    return result


def layers(kernels: list[dict], layer_types: list[str]) -> dict:
    positions = [
        i
        for i, event in enumerate(kernels)
        if collective(event["name"]) in ("custom_allreduce", "nccl_allreduce")
    ]
    expected = 1 + 2 * len(layer_types)
    if len(positions) != expected:
        return {
            "status": "unassigned_collective_structure",
            "expected_collectives": expected,
            "observed_collectives": len(positions),
            "kernels": ledger(kernels),
        }
    # One embedding/context collective precedes two collectives per layer.
    start = positions[0] + 1
    prefix = ledger(kernels[:start])
    result = []
    for layer, kind in enumerate(layer_types):
        end = positions[2 + 2 * layer] + 1
        part = kernels[start:end]
        result.append(
            {
                "layer": layer,
                "kind": kind,
                "service_us": sum(event["dur"] for event in part),
                "span_us": part[-1]["ts"] + part[-1]["dur"] - part[0]["ts"],
                "kernels": ledger(part),
            }
        )
        start = end
    return {
        "status": "paired_collectives",
        "boundary_contract": (
            "Prefix collective, then two collectives per layer; "
            "trailing norm/copies remain explicit."
        ),
        "prefix": prefix,
        "layers": result,
        "suffix": ledger(kernels[start:]),
    }


def analyze(trace: Path, config: Path) -> dict:
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = sorted(
        (event for event in events if event.get("cat") == "kernel"),
        key=lambda event: event["ts"],
    )
    groups = collections.defaultdict(list)
    for event in kernels:
        groups[event.get("args", {}).get("correlation")].append(event)
    target = sorted(
        (
            part
            for part in groups.values()
            if sum("delta_rule" in event["name"] for event in part) >= 40
        ),
        key=lambda part: part[0]["ts"],
    )
    if len(target) < 2:
        raise ValueError("Trace lacks two complete target graphs with GDN kernels")
    draft = sorted(
        (
            part
            for part in groups.values()
            if sum("_dflash_rmsnorm" in event["name"] for event in part) >= 10
        ),
        key=lambda part: part[0]["ts"],
    )
    model = json.loads(config.read_text())
    types = model.get("text_config", model)["layer_types"]
    times = [event["ts"] for event in kernels]
    rounds = []
    for number, (current, following) in enumerate(zip(target, target[1:])):
        begin, end = current[0]["ts"], following[0]["ts"]
        part = kernels[
            bisect.bisect_left(times, begin) : bisect.bisect_left(times, end)
        ]
        drafts = [group for group in draft if begin <= group[0]["ts"] < end]
        names = [event["name"] for event in part]
        compact = any("_dflash2_sparse_topk_rejection_kernel" in name for name in names)
        dense = any(
            ("_rejection_kernel" in name and "sparse_topk_rejection" not in name)
            or "_resample_kernel" in name
            for name in names
        )
        route = (
            "mixed"
            if compact and dense
            else "compact"
            if compact
            else "dense"
            if dense
            else "unclassified"
        )
        comm = collections.defaultdict(lambda: {"count": 0, "service_us": 0.0})
        for event in part:
            backend = collective(event["name"])
            if backend:
                comm[backend]["count"] += 1
                comm[backend]["service_us"] += event["dur"]
        rounds.append(
            {
                "round": number,
                "target_graph_correlation": current[0]["args"]["correlation"],
                "target_start_to_next_target_start_ms": (end - begin) / 1000,
                "kernel_service_ms": sum(event["dur"] for event in part) / 1000,
                "rejection_route": route,
                "collectives": dict(comm),
                "target": layers(current, types),
                "draft": [layers(group, ["sliding_attention"] * 5) for group in drafts],
                "kernels": ledger(part),
            }
        )
    # Exclude edge rounds for aggregate timing, but retain every ledger.
    steady = rounds[1:-1] if len(rounds) > 4 else rounds
    routes = collections.Counter(part["rejection_route"] for part in steady)
    all_routes = collections.Counter(part["rejection_route"] for part in rounds)
    return {
        "trace": str(trace),
        "complete_rounds": len(rounds),
        "aggregate_rounds": len(steady),
        "rejection_counts": dict(routes),
        "all_rejection_counts": dict(all_routes),
        "rejection_fractions": {
            key: value / len(steady) for key, value in routes.items()
        },
        "profiled_target_interval_mean_ms": statistics.mean(
            part["target_start_to_next_target_start_ms"] for part in steady
        ),
        "interpretation": (
            "Torch profiling attribution; not an unprofiled speed result. "
            "Service sums can overlap. Layer spans omit gaps between ledger slices."
        ),
        "rounds": rounds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, args.model_config)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rounds"}, indent=2
        )
    )
