# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate compulsory expert reads from route-issued reads in a cost capture.

This consumes untimed routing records. It does not turn storage size into
measured HBM traffic, or claim that concurrent kernel service is additive.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

import regex as re


def expert_reads(records, bytes_per_expert):
    if bytes_per_expert <= 0 or not records:
        raise ValueError("Expert reads require positive storage and route records")
    samples = []
    for record in records:
        ids = [expert for row in record["ids"] for expert in row]
        if not ids or any(expert < 0 for expert in ids):
            raise ValueError("Unwritten routing entries are not expert accesses")
        counts = collections.Counter(ids)
        samples.append(
            {
                "ordinal": record["ordinal"],
                "rows": record["rows"],
                "routes": len(ids),
                "unique_experts": len(counts),
                "tokens_per_expert": dict(sorted(counts.items())),
                "compulsory_weight_bytes": len(counts) * bytes_per_expert,
                "route_issued_weight_bytes": len(ids) * bytes_per_expert,
            }
        )
    return {
        "bytes_per_expert": bytes_per_expert,
        "mean_unique_experts": statistics.mean(s["unique_experts"] for s in samples),
        "mean_compulsory_bytes": statistics.mean(
            s["compulsory_weight_bytes"] for s in samples
        ),
        "mean_route_issued_bytes": statistics.mean(
            s["route_issued_weight_bytes"] for s in samples
        ),
        "samples": samples,
    }


def expert_banks(tensors):
    banks = collections.defaultdict(dict)
    pattern = re.compile(
        r"^(.*)\.gguf_expert_banks\.(w[123])\.(raw_weights|weights|stats)$"
    )
    for tensor in tensors:
        match = pattern.match(tensor["name"])
        if match:
            prefix, projection, storage = match.groups()
            banks[prefix].setdefault(projection, {})[storage] = tensor
    return dict(banks)


def projection_bytes(bank, prefer_raw):
    # IQ gate/up reads original GGML blocks. Down reads TP-sliced canonical
    # storage, whose K padding cannot be inferred by dividing GGUF file bytes.
    selected = (
        [bank["raw_weights"]]
        if prefer_raw and "raw_weights" in bank
        else [bank[key] for key in ("weights", "stats") if key in bank]
    )
    if not selected:
        raise ValueError("Dispatched expert storage missing from inventory")
    experts = selected[0]["shape"][0]
    if any(t["shape"][0] != experts for t in selected):
        raise ValueError("Expert storage has incompatible leading dimensions")
    total = sum(t["logical_bytes"] for t in selected)
    if total % experts:
        raise ValueError("Expert storage is not evenly packed")
    return total // experts, [t["name"] for t in selected]


def draft_groups(records, verification_rows):
    """Exclude bootstrap draft calls and retain complete four-step rounds.

    The first draft processes verifier rows to update attention history, then
    three dependent calls use one row each. Treating all four as M1 omits the
    first call's actual routed experts.
    """
    starts = [
        i for i, record in enumerate(records) if record["rows"] == verification_rows
    ]
    groups = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(records)
        group = records[start:end]
        if [record["rows"] for record in group] != [verification_rows, 1, 1, 1]:
            raise ValueError("Draft routing does not contain complete MTP4 rounds")
        groups.append(group)
    if not groups:
        raise ValueError("No verified four-step draft rounds")
    return groups


def analyze(report, rows=5, trim=8, hbm_gbps=900):
    workers = sorted(report["round_cost"], key=lambda worker: worker["rank"])
    if [worker["rank"] for worker in workers] != [0, 1, 2, 3]:
        raise ValueError("A TP4 ledger requires all four rank inventories")
    reference = workers[0]["routes"]
    if any(worker["routes"] != reference for worker in workers[1:]):
        raise ValueError("Ranks disagree on actual routing; do not average them")
    result = {"scope": "untimed expert byte lower bounds", "per_rank": []}
    for worker in workers:
        banks = expert_banks(worker["tensors"]["target"])
        layers = []
        for prefix, projections in banks.items():
            history = worker["routes"][prefix]
            records = [r for r in history["records"] if r["rows"] == rows]
            if len(records) <= 2 * trim:
                raise ValueError(f"Insufficient central routing records: {prefix}")
            records = records[trim : len(records) - trim if trim else None]
            gate_bytes, gate_names = projection_bytes(projections["w1"], True)
            up_bytes, up_names = projection_bytes(projections["w3"], True)
            down_bytes, down_names = projection_bytes(projections["w2"], False)
            gu = expert_reads(records, gate_bytes + up_bytes)
            down = expert_reads(records, down_bytes)
            layers.append(
                {
                    "layer": prefix,
                    "dropped_records": history["dropped"],
                    "gate_up": gu,
                    "down": down,
                    "selected_storage": gate_names + up_names + down_names,
                    "bandwidth_lower_ms": (
                        gu["mean_compulsory_bytes"] + down["mean_compulsory_bytes"]
                    )
                    / (hbm_gbps * 1e6),
                }
            )
        if len(layers) != 48:
            raise ValueError("Flash-Next target ledger requires 48 expert layers")
        draft_routes = {
            key: value for key, value in worker["routes"].items() if key not in banks
        }
        if len(draft_routes) != 1:
            raise ValueError("Flash-Next MTP ledger requires one draft expert layer")
        draft_prefix, draft_history = next(iter(draft_routes.items()))
        groups = draft_groups(draft_history["records"], rows)
        groups = groups[trim : len(groups) - trim if trim else None]
        if not groups:
            raise ValueError("Insufficient central draft rounds")
        draft_weights = [
            tensor
            for tensor in worker["tensors"]["draft"]
            if tensor["name"].endswith((".experts.w13_weight", ".experts.w2_weight"))
        ]
        if len(draft_weights) != 2:
            raise ValueError("Missing native FP16 draft expert storage")
        draft_per_expert = sum(
            tensor["logical_bytes"] // tensor["shape"][0] for tensor in draft_weights
        )
        draft = {
            "layer": draft_prefix,
            "dropped_records": draft_history["dropped"],
            "rounds": len(groups),
            "steps": [
                expert_reads([group[step] for group in groups], draft_per_expert)
                for step in range(4)
            ],
            "selected_storage": [tensor["name"] for tensor in draft_weights],
        }
        result["per_rank"].append(
            {"rank": worker["rank"], "layers": layers, "draft": draft}
        )
    result["hbm_gbps"] = hbm_gbps
    result["notes"] = [
        "Unique experts define a compulsory-read floor; repeated routes can hit L2.",
        "Canonical and retained original banks are not both counted as reads.",
        "Decoder, communication and dependent draft costs require separate rows.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--trim", type=int, default=8)
    parser.add_argument("--hbm-gbps", type=float, default=900)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            analyze(
                json.loads(args.benchmark.read_text()),
                args.rows,
                args.trim,
                args.hbm_gbps,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
