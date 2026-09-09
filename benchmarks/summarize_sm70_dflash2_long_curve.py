# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze or evaluate a repeated, unprofiled DFlash2 context-cost curve."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

LENGTHS = (1024, 32768, 65536, 131072)
INTERVALS = ((32768, 65536), (65536, 131072), (32768, 131072))


def summarize(paths):
    if len(paths) != 3 or len(set(p.resolve() for p in paths)) != 3:
        raise ValueError("Three distinct startup reports are required")
    reports = [json.loads(p.read_text()) for p in paths]
    startup_ids = {
        tuple(
            row["pid"] for row in sorted(r["initial_routes"], key=lambda x: x["rank"])
        )
        for r in reports
    }
    if len(startup_ids) != 3:
        raise ValueError("Reports do not identify three independent worker startups")
    contract_keys = ("sampling", "corpus_sha256", "context_capacity")
    contract = {key: reports[0][key] for key in contract_keys}
    for report in reports:
        assert report["complete"] and not report["profiler"]
        assert report["require_native_prefill"]
        assert report["require_original_gdn_prefill"]
        assert {key: report[key] for key in contract_keys} == contract
        assert {r["prompt_tokens"] for r in report["cases"]} == set(LENGTHS)
    result = {
        "contract": contract,
        "source_reports": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "rows": {},
    }
    for length in LENGTHS:
        cold, measured, startup_rows = [], [], []
        for report in reports:
            rows = [r for r in report["cases"] if r["prompt_tokens"] == length]
            warmup = [r for r in rows if r["warmup"]]
            repeats = [r for r in rows if not r["warmup"]]
            assert len(warmup) == 1 and len(repeats) == 5
            assert warmup[0]["prefill_computed_tokens"] == length
            assert sorted(r["repeat"] for r in repeats) == list(range(5))
            assert all(r["prompt_sha256"] == rows[0]["prompt_sha256"] for r in rows)
            cold.append(warmup[0]["engine_prefill_s"])
            measured.extend(repeats)
            startup_rows.append(rows)
        row = {
            "cold_prefill_seconds": cold,
            "cold_prefill_tps": length / statistics.median(cold),
            "startup_count": 3,
            "measured_request_count": len(measured),
            "within_startup_same_tokens": all(
                all(r["token_ids"] == rows[0]["token_ids"] for r in rows)
                for rows in startup_rows
            ),
            "between_startup_same_tokens": all(
                rows[0]["token_ids"] == startup_rows[0][0]["token_ids"]
                for rows in startup_rows
            ),
            "finish_reasons": sorted({r["finish_reason"] for r in measured}),
        }
        assert len({rows[0]["prompt_sha256"] for rows in startup_rows}) == 1
        for metric in (
            "complete_round_ms",
            "pure_decode_tps",
            "accepted_drafts_per_round",
            "emitted_tokens_per_round",
            "accepted_over_proposed",
            "ttft_s",
        ):
            values = [r[metric] for r in measured]
            row[metric] = float(statistics.median(values))
            row[metric + "_request_distribution"] = dict(
                zip(("p50", "p90", "p99"), np.percentile(values, [50, 90, 99]).tolist())
            )
        result["rows"][str(length)] = row
    result["percentile_scope"] = (
        "Distributions across request averages; not individual GPU-round percentiles"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a frozen curve")
    result = summarize(args.reports)
    if args.baseline is None:
        result["frozen_prefill_growth_limits"] = {
            f"{a}:{b}": result["rows"][str(a)]["cold_prefill_tps"]
            / result["rows"][str(b)]["cold_prefill_tps"]
            for a, b in INTERVALS
        }
    else:
        baseline = json.loads(args.baseline.read_text())
        assert result["contract"] == baseline["contract"]
        checks = {}
        for length in LENGTHS:
            current = result["rows"][str(length)]["complete_round_ms"]
            previous = baseline["rows"][str(length)]["complete_round_ms"]
            checks[f"absolute_{length}"] = (
                current <= previous if length == 1024 else current < previous
            )
        for a, b in INTERVALS:
            growth = (
                result["rows"][str(b)]["complete_round_ms"]
                / result["rows"][str(a)]["complete_round_ms"]
            )
            limit = baseline["frozen_prefill_growth_limits"][f"{a}:{b}"]
            checks[f"growth_{a}:{b}"] = growth <= limit
        result["curve_checks"] = checks
        result["performance_curve_passed"] = all(checks.values())
        result["quality_and_acceptance_admission"] = "Requires separate paired evidence"
        result["baseline_sha256"] = hashlib.sha256(
            args.baseline.read_bytes()
        ).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
