# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare matched no-speculation and DSpark quality artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest_hashes(value: Any, prefix: str = "") -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    if isinstance(value.get("sha256"), str):
        return {prefix: value["sha256"]}
    hashes: dict[str, str] = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else key
        hashes.update(_manifest_hashes(child, child_prefix))
    return hashes


def _compare_humaneval(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    reference_rows = {row["task_id"]: row for row in reference["human_eval"]["records"]}
    candidate_rows = {row["task_id"]: row for row in candidate["human_eval"]["records"]}
    common = sorted(reference_rows.keys() & candidate_rows.keys())
    regressions = [
        task_id
        for task_id in common
        if reference_rows[task_id]["passed"] and not candidate_rows[task_id]["passed"]
    ]
    improvements = [
        task_id
        for task_id in common
        if not reference_rows[task_id]["passed"] and candidate_rows[task_id]["passed"]
    ]
    exact_responses = sum(
        reference_rows[task_id].get("response")
        == candidate_rows[task_id].get("response")
        for task_id in common
    )
    reference_passed = int(reference["human_eval"]["passed"])
    candidate_passed = int(candidate["human_eval"]["passed"])
    missing = sorted(reference_rows.keys() ^ candidate_rows.keys())
    return {
        "reference_passed": reference_passed,
        "candidate_passed": candidate_passed,
        "samples": len(common),
        "exact_response_matches": exact_responses,
        "regressions": regressions,
        "improvements": improvements,
        "missing_or_extra_tasks": missing,
        # Individual flips remain visible for diagnosis, but are not token-
        # identity gates. Acceptance is based on matched inputs and aggregate
        # task quality.
        "passed": not missing and candidate_passed >= reference_passed,
    }


def _compare_longbench(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    reference_sets = reference["longbench"]["datasets"]
    candidate_sets = candidate["longbench"]["datasets"]
    missing = sorted(reference_sets.keys() ^ candidate_sets.keys())
    results: dict[str, Any] = {}
    for dataset in sorted(reference_sets.keys() & candidate_sets.keys()):
        reference_rows = {
            int(row["source_index"]): row for row in reference_sets[dataset]["records"]
        }
        candidate_rows = {
            int(row["source_index"]): row for row in candidate_sets[dataset]["records"]
        }
        common = sorted(reference_rows.keys() & candidate_rows.keys())
        row_mismatch = sorted(reference_rows.keys() ^ candidate_rows.keys())
        input_mismatch = [
            index
            for index in common
            if reference_rows[index].get("answers")
            != candidate_rows[index].get("answers")
        ]
        regressions = [
            {
                "source_index": index,
                "reference_score": float(reference_rows[index]["score"]),
                "candidate_score": float(candidate_rows[index]["score"]),
            }
            for index in common
            if float(candidate_rows[index]["score"]) + tolerance
            < float(reference_rows[index]["score"])
        ]
        exact_predictions = sum(
            reference_rows[index].get("prediction")
            == candidate_rows[index].get("prediction")
            for index in common
        )
        reference_score = float(reference_sets[dataset]["score"])
        candidate_score = float(candidate_sets[dataset]["score"])
        results[dataset] = {
            "reference_score": reference_score,
            "candidate_score": candidate_score,
            "samples": len(common),
            "exact_prediction_matches": exact_predictions,
            "regressions": regressions,
            "missing_or_extra_rows": row_mismatch,
            "input_mismatches": input_mismatch,
            "passed": not row_mismatch
            and not input_mismatch
            and candidate_score + tolerance >= reference_score,
        }
    reference_average = float(reference["longbench"]["average_score"])
    candidate_average = float(candidate["longbench"]["average_score"])
    return {
        "reference_average": reference_average,
        "candidate_average": candidate_average,
        "score_tolerance": tolerance,
        "datasets": results,
        "missing_or_extra_datasets": missing,
        "passed": not missing
        and all(result["passed"] for result in results.values())
        and candidate_average + tolerance >= reference_average,
    }


def _compare_api_quality(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    reference_hashes = _manifest_hashes(reference.get("input_manifest", {}))
    candidate_hashes = _manifest_hashes(candidate.get("input_manifest", {}))
    manifest_equal = bool(reference_hashes) and reference_hashes == candidate_hashes
    reference_contract = reference.get("evaluation_contract")
    candidate_contract = candidate.get("evaluation_contract")
    contract_equal = (
        bool(reference_contract) and reference_contract == candidate_contract
    )
    humaneval = _compare_humaneval(reference, candidate)
    longbench = _compare_longbench(reference, candidate, tolerance)
    return {
        "input_hashes_equal": manifest_equal,
        "reference_input_hashes": reference_hashes,
        "candidate_input_hashes": candidate_hashes,
        "evaluation_contract_equal": contract_equal,
        "reference_evaluation_contract": reference_contract,
        "candidate_evaluation_contract": candidate_contract,
        "humaneval": humaneval,
        "longbench": longbench,
        "passed": manifest_equal
        and contract_equal
        and humaneval["passed"]
        and longbench["passed"],
    }


def _compare_gsm8k(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    reference_hashes = _manifest_hashes(reference.get("input_manifest", {}))
    candidate_hashes = _manifest_hashes(candidate.get("input_manifest", {}))
    manifest_equal = bool(reference_hashes) and reference_hashes == candidate_hashes
    contract_keys = (
        "questions",
        "few_shot",
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "strictly_sequential",
        "train_selection",
        "test_selection",
        "prompt_format",
        "answer_normalization",
        "stop_sequences",
    )
    reference_contract = {
        key: reference.get("contract", {}).get(key) for key in contract_keys
    }
    candidate_contract = {
        key: candidate.get("contract", {}).get(key) for key in contract_keys
    }
    contract_equal = reference_contract == candidate_contract
    reference_rows = {int(row["index"]): row for row in reference["rows"]}
    candidate_rows = {int(row["index"]): row for row in candidate["rows"]}
    common = sorted(reference_rows.keys() & candidate_rows.keys())
    missing = sorted(reference_rows.keys() ^ candidate_rows.keys())
    expected_samples = reference_contract["questions"]
    row_count_matches_contract = (
        isinstance(expected_samples, int)
        and len(reference["rows"]) == expected_samples
        and len(candidate["rows"]) == expected_samples
        and len(reference_rows) == expected_samples
        and len(candidate_rows) == expected_samples
    )

    def row_is_consistent(row: dict[str, Any]) -> bool:
        expected = row.get("expected")
        predicted = row.get("predicted")
        return bool(
            expected is not None
            and bool(row.get("correct"))
            == (predicted is not None and predicted == expected)
            and bool(row.get("invalid")) == (predicted is None)
        )

    rows_self_consistent = all(
        row_is_consistent(row) for row in [*reference["rows"], *candidate["rows"]]
    )
    summaries_self_consistent = all(
        int(result["correct"])
        == sum(bool(row.get("correct")) for row in result["rows"])
        and int(result["invalid"])
        == sum(bool(row.get("invalid")) for row in result["rows"])
        for result in (reference, candidate)
    )
    input_mismatch = [
        index
        for index in common
        if (
            reference_rows[index].get("question"),
            reference_rows[index].get("expected"),
        )
        != (
            candidate_rows[index].get("question"),
            candidate_rows[index].get("expected"),
        )
    ]
    regressions = [
        index
        for index in common
        if reference_rows[index]["correct"] and not candidate_rows[index]["correct"]
    ]
    improvements = [
        index
        for index in common
        if not reference_rows[index]["correct"] and candidate_rows[index]["correct"]
    ]
    exact_predictions = sum(
        reference_rows[index].get("predicted") == candidate_rows[index].get("predicted")
        for index in common
    )
    return {
        "input_hashes_equal": manifest_equal,
        "reference_input_hashes": reference_hashes,
        "candidate_input_hashes": candidate_hashes,
        "evaluation_contract_equal": contract_equal,
        "reference_evaluation_contract": reference_contract,
        "candidate_evaluation_contract": candidate_contract,
        "row_count_matches_contract": row_count_matches_contract,
        "rows_self_consistent": rows_self_consistent,
        "summaries_self_consistent": summaries_self_consistent,
        "reference_correct": int(reference["correct"]),
        "candidate_correct": int(candidate["correct"]),
        "correct_delta": int(candidate["correct"]) - int(reference["correct"]),
        "reference_invalid": int(reference["invalid"]),
        "candidate_invalid": int(candidate["invalid"]),
        "samples": len(common),
        "exact_prediction_matches": exact_predictions,
        "regressions": regressions,
        "improvements": improvements,
        "missing_or_extra_rows": missing,
        "input_mismatches": input_mismatch,
        # Report directional flips without requiring greedy/token identity.
        # Matched aggregate correctness and validity are the quality gate.
        "passed": manifest_equal
        and contract_equal
        and row_count_matches_contract
        and rows_self_consistent
        and summaries_self_consistent
        and not missing
        and not input_mismatch
        and int(candidate["correct"]) >= int(reference["correct"])
        and int(candidate["invalid"]) <= int(reference["invalid"]),
    }


def _compare_needle(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_rows = {row["sample_id"]: row for row in reference}
    candidate_rows = {row["sample_id"]: row for row in candidate}
    common = sorted(reference_rows.keys() & candidate_rows.keys())
    missing = sorted(reference_rows.keys() ^ candidate_rows.keys())
    input_mismatch = [
        sample_id
        for sample_id in common
        if (
            reference_rows[sample_id].get("target_tokens"),
            reference_rows[sample_id].get("depth"),
            reference_rows[sample_id].get("code"),
        )
        != (
            candidate_rows[sample_id].get("target_tokens"),
            candidate_rows[sample_id].get("depth"),
            candidate_rows[sample_id].get("code"),
        )
    ]
    final_regressions = [
        sample_id
        for sample_id in common
        if reference_rows[sample_id]["hit"] and not candidate_rows[sample_id]["hit"]
    ]
    anywhere_regressions = [
        sample_id
        for sample_id in common
        if reference_rows[sample_id]["hit_anywhere"]
        and not candidate_rows[sample_id]["hit_anywhere"]
    ]
    reference_hits = sum(bool(row["hit"]) for row in reference_rows.values())
    candidate_hits = sum(bool(row["hit"]) for row in candidate_rows.values())
    reference_anywhere = sum(
        bool(row["hit_anywhere"]) for row in reference_rows.values()
    )
    candidate_anywhere = sum(
        bool(row["hit_anywhere"]) for row in candidate_rows.values()
    )
    return {
        "reference_final_hits": reference_hits,
        "candidate_final_hits": candidate_hits,
        "reference_anywhere_hits": reference_anywhere,
        "candidate_anywhere_hits": candidate_anywhere,
        "samples": len(common),
        "final_hit_regressions": final_regressions,
        "anywhere_hit_regressions": anywhere_regressions,
        "missing_or_extra_rows": missing,
        "input_mismatches": input_mismatch,
        "passed": not missing
        and not input_mismatch
        and candidate_hits >= reference_hits
        and candidate_anywhere >= reference_anywhere,
    }


def _paired_paths(
    parser: argparse.ArgumentParser,
    reference: Path | None,
    candidate: Path | None,
    label: str,
) -> bool:
    if (reference is None) != (candidate is None):
        parser.error(f"--reference-{label} and --candidate-{label} must be paired")
    return reference is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-api", type=Path)
    parser.add_argument("--candidate-api", type=Path)
    parser.add_argument("--reference-gsm8k", type=Path)
    parser.add_argument("--candidate-gsm8k", type=Path)
    parser.add_argument("--reference-needle", type=Path)
    parser.add_argument("--candidate-needle", type=Path)
    parser.add_argument("--longbench-score-tolerance", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.longbench_score_tolerance < 0:
        parser.error("--longbench-score-tolerance must be non-negative")

    has_api = _paired_paths(parser, args.reference_api, args.candidate_api, "api")
    has_gsm8k = _paired_paths(
        parser, args.reference_gsm8k, args.candidate_gsm8k, "gsm8k"
    )
    has_needle = _paired_paths(
        parser, args.reference_needle, args.candidate_needle, "needle"
    )
    if not (has_api or has_gsm8k or has_needle):
        parser.error("at least one paired quality artifact must be provided")

    result: dict[str, Any] = {
        "contract": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    gates: list[bool] = []
    if has_api:
        assert args.reference_api is not None and args.candidate_api is not None
        result["api_quality"] = _compare_api_quality(
            _load_json(args.reference_api),
            _load_json(args.candidate_api),
            args.longbench_score_tolerance,
        )
        gates.append(result["api_quality"]["passed"])
    if has_gsm8k:
        assert args.reference_gsm8k is not None and args.candidate_gsm8k is not None
        result["gsm8k"] = _compare_gsm8k(
            _load_json(args.reference_gsm8k), _load_json(args.candidate_gsm8k)
        )
        gates.append(result["gsm8k"]["passed"])
    if has_needle:
        assert args.reference_needle is not None and args.candidate_needle is not None
        result["needle"] = _compare_needle(
            _load_jsonl(args.reference_needle), _load_jsonl(args.candidate_needle)
        )
        gates.append(result["needle"]["passed"])
    result["passed"] = all(gates)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
