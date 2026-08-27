# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare paired multi-seed rows from the bounded DFlash2 quality matrix."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import regex as re


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    return parser.parse_args()


def _arm_name(path: Path) -> str:
    match = re.search(r"-(t0|d[0-9]+)$", path.stem)
    if match is None:
        raise ValueError(f"cannot infer matrix arm from {path}")
    return match.group(1)


def _ordered_arm_names(arms: dict[str, Any]) -> list[str]:
    """Order the target control before progressively accelerated DFlash arms."""

    def key(name: str) -> tuple[int, int]:
        if name == "t0":
            return (0, 0)
        return (1, int(name[1:]))

    return sorted(arms, key=key)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _paired_bootstrap_delta(
    pairs: list[tuple[float, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    if not pairs:
        return {"count": 0, "mean_delta": None, "ci95": None}
    deltas = [right - left for left, right in pairs]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    return {
        "count": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
    }


def _mcnemar_exact(left: list[bool], right: list[bool]) -> dict[str, float | int]:
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(record["passed"] for record in records)
    acceptance = [
        float(record["acceptance_length"])
        for record in records
        if record["acceptance_length"] is not None
    ]
    steady_decode = [
        float(record["steady_decode_tps"])
        for record in records
        if record["steady_decode_tps"] is not None
    ]
    return {
        "num_cases": len(records),
        "passed": passed,
        "score": passed / len(records) if records else None,
        "finish_reasons": dict(Counter(record["finish_reason"] for record in records)),
        "failure_reasons": dict(
            Counter(record["reason"] for record in records if not record["passed"])
        ),
        "mean_acceptance_length": (
            sum(acceptance) / len(acceptance) if acceptance else None
        ),
        "mean_steady_decode_tps": (
            sum(steady_decode) / len(steady_decode) if steady_decode else None
        ),
    }


def _load_arms(scores_path: Path) -> dict[str, dict[str, Any]]:
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    arms = {}
    for scored_run in scores["runs"]:
        raw_path = Path(scored_run["result"])
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if len(raw["cases"]) != len(scored_run["cases"]):
            raise ValueError(f"raw/scored case count differs for {raw_path}")

        records = []
        for raw_case, scored_case in zip(
            raw["cases"], scored_run["cases"], strict=True
        ):
            if raw_case["dataset_index"] != scored_case["dataset_index"]:
                raise ValueError(f"raw/scored case order differs for {raw_path}")
            request_metrics = raw_case.get("request_metrics") or {}
            spec_metrics = raw_case.get("spec_decode_metrics") or {}
            records.append(
                {
                    "key": [raw_case.get("request_seed"), raw_case["dataset_index"]],
                    "request_seed": raw_case.get("request_seed"),
                    "dataset_index": raw_case["dataset_index"],
                    "suite": scored_case["suite"],
                    "suite_index": scored_case["suite_index"],
                    "passed": bool(scored_case["passed"]),
                    "reason": scored_case["reason"],
                    "finish_reason": raw_case["finish_reason"],
                    "token_hash": raw_case["token_hash"],
                    "output_tokens": raw_case["output_tokens"],
                    "acceptance_length": spec_metrics.get("acceptance_length"),
                    "steady_decode_tps": request_metrics.get("steady_decode_tps"),
                }
            )
        arm = _arm_name(raw_path)
        if arm in arms:
            raise ValueError(f"duplicate matrix arm {arm}")
        arms[arm] = {
            "result": str(raw_path),
            "contract": raw["contract"],
            "runtime": raw["runtime"],
            "records": records,
        }
    return arms


def _record_map(arm: dict[str, Any]) -> dict[tuple[int | None, int], dict[str, Any]]:
    result = {}
    for record in arm["records"]:
        key = tuple(record["key"])
        if key in result:
            raise ValueError(f"duplicate paired key {key}")
        result[key] = record
    return result


def _compare(
    left_name: str,
    left_arm: dict[str, Any],
    right_name: str,
    right_arm: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    left = _record_map(left_arm)
    right = _record_map(right_arm)
    if set(left) != set(right):
        raise ValueError(f"paired keys differ for {left_name}/{right_name}")
    keys = sorted(left)
    left_pass = [left[key]["passed"] for key in keys]
    right_pass = [right[key]["passed"] for key in keys]
    score_pairs = [(float(a), float(b)) for a, b in zip(left_pass, right_pass)]
    acceptance_pairs = [
        (float(left[key]["acceptance_length"]), float(right[key]["acceptance_length"]))
        for key in keys
        if left[key]["acceptance_length"] is not None
        and right[key]["acceptance_length"] is not None
    ]
    return {
        "left": left_name,
        "right": right_name,
        "num_cases": len(keys),
        "token_hash_equal_cases": sum(
            left[key]["token_hash"] == right[key]["token_hash"] for key in keys
        ),
        "score": _paired_bootstrap_delta(
            score_pairs,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "acceptance_length": _paired_bootstrap_delta(
            acceptance_pairs,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        ),
        "mcnemar_exact": _mcnemar_exact(left_pass, right_pass),
    }


def main() -> int:
    args = _parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    arms = _load_arms(args.scores)

    summaries = {}
    for arm_name, arm in sorted(arms.items()):
        by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in arm["records"]:
            by_suite[record["suite"]].append(record)
            by_seed[str(record["request_seed"])].append(record)
        summaries[arm_name] = {
            "result": arm["result"],
            "aggregate": _summarize(arm["records"]),
            "by_suite": {
                name: _summarize(records) for name, records in sorted(by_suite.items())
            },
            "by_seed": {
                name: _summarize(records) for name, records in sorted(by_seed.items())
            },
        }

    comparisons = [
        _compare(
            left,
            arms[left],
            right,
            arms[right],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for left, right in itertools.combinations(_ordered_arm_names(arms), 2)
    ]

    payload = {
        "format": "sm70_dflash2_quality_matrix_v1",
        "scores": str(args.scores),
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "arms": summaries,
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
