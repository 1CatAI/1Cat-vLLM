# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate a static FP16 draft-vocabulary base with an online LRU tail."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

_ALIGNMENT_STEP_RE = re.compile(r"_step(?P<step>\d+)_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--base-size", type=int, default=65536)
    parser.add_argument("--tail-size", type=int, action="append")
    parser.add_argument("--alignment-glob")
    parser.add_argument("--alignment-summary", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        rows = payload.get("records", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"Expected records in {path}.")
        records.extend(row for row in rows if isinstance(row, dict))
    return records


def _record_tokens(
    record: dict[str, Any], tokenizer: Any
) -> tuple[list[int], list[int]]:
    choice = record.get("choice")
    choice = choice if isinstance(choice, dict) else {}
    prompt_ids = choice.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        prompt = record.get("prompt")
        prompt_ids = (
            tokenizer.encode(prompt, add_special_tokens=False)
            if isinstance(prompt, str)
            else []
        )
    output_ids = choice.get("token_ids")
    outputs = record.get("outputs")
    if (
        (not isinstance(output_ids, list) or not output_ids)
        and isinstance(outputs, list)
        and outputs
        and isinstance(outputs[0], dict)
    ):
        output_ids = outputs[0].get("token_ids")
    if not isinstance(output_ids, list) or not output_ids:
        text = choice.get("text")
        if (
            not isinstance(text, str)
            and isinstance(outputs, list)
            and outputs
            and isinstance(outputs[0], dict)
        ):
            text = outputs[0].get("text")
        output_ids = (
            tokenizer.encode(text, add_special_tokens=False)
            if isinstance(text, str)
            else []
        )
    return [int(x) for x in prompt_ids], [int(x) for x in output_ids]


class _Tail:
    def __init__(
        self,
        capacity: int,
        base: set[int],
        initial_ids: list[int],
    ) -> None:
        self.capacity = capacity
        self.base = base
        self.ids: OrderedDict[int, None] = OrderedDict()
        self.insert(initial_ids)

    def __contains__(self, token_id: int) -> bool:
        return token_id in self.ids

    def insert(self, token_ids: list[int]) -> None:
        for token_id in token_ids:
            if token_id in self.base:
                continue
            self.ids.pop(token_id, None)
            self.ids[token_id] = None
            if len(self.ids) > self.capacity:
                self.ids.popitem(last=False)


def _output_coverage(
    records: list[dict[str, Any]],
    tokenizer: Any,
    ranking: list[int],
    base_size: int,
    tail_size: int,
) -> dict[str, Any]:
    base = set(ranking[:base_size])
    ranked_tail = ranking[base_size : base_size + tail_size]
    aggregate: dict[str, int] = {
        "samples": 0,
        "tokens": 0,
        "base_hits": 0,
        "active_hits": 0,
        "cold_misses": 0,
        "repeat_misses_after_eviction": 0,
    }
    peak_tail_rows = 0
    for record in records:
        prompt_ids, output_ids = _record_tokens(record, tokenizer)
        tail = _Tail(tail_size, base, ranked_tail)
        tail.insert(prompt_ids)
        seen_missing: set[int] = set()
        aggregate["samples"] += 1
        for token_id in output_ids:
            aggregate["tokens"] += 1
            if token_id in base:
                aggregate["base_hits"] += 1
                aggregate["active_hits"] += 1
            elif token_id in tail:
                aggregate["active_hits"] += 1
            elif token_id in seen_missing:
                aggregate["repeat_misses_after_eviction"] += 1
            else:
                aggregate["cold_misses"] += 1
            if token_id not in base:
                seen_missing.add(token_id)
            tail.insert([token_id])
            peak_tail_rows = max(peak_tail_rows, len(tail.ids))
    total = aggregate["tokens"]
    return {
        **aggregate,
        "base_coverage": aggregate["base_hits"] / total if total else 1.0,
        "active_coverage": aggregate["active_hits"] / total if total else 1.0,
        "missing_occurrences": total - aggregate["active_hits"],
        "peak_tail_rows": peak_tail_rows,
    }


def _load_alignment_steps(pattern: str) -> list[dict[str, Any]]:
    by_step: dict[int, Path] = {}
    for name in sorted(glob.glob(pattern)):
        path = Path(name)
        match = _ALIGNMENT_STEP_RE.search(path.name)
        if match is not None:
            by_step.setdefault(int(match.group("step")), path)
    return [
        torch.load(path, map_location="cpu", weights_only=True)
        for _, path in sorted(by_step.items())
    ]


def _alignment_prompt_ids(path: Path | None, tokenizer: Any) -> list[int]:
    if path is None:
        return []
    records = _load_records([path])
    return _record_tokens(records[0], tokenizer)[0] if records else []


def _alignment_coverage(
    steps: list[dict[str, Any]],
    prompt_ids: list[int],
    ranking: list[int],
    base_size: int,
    tail_size: int,
    *,
    seed_prompt: bool,
    update_observed_output: bool,
    previous_candidate_keys: tuple[str, ...],
) -> dict[str, Any]:
    base = set(ranking[:base_size])
    tail = _Tail(
        tail_size,
        base,
        ranking[base_size : base_size + tail_size],
    )
    if seed_prompt:
        tail.insert(prompt_ids)
    topk_entries = 0
    topk_hits = 0
    topk_mass = 0.0
    covered_topk_mass = 0.0
    sampled_tokens = 0
    sampled_hits = 0
    sampled_misses: Counter[int] = Counter()
    position_sampled_tokens: list[int] = []
    position_sampled_hits: list[int] = []
    position_topk_mass: list[float] = []
    position_covered_topk_mass: list[float] = []
    position_baseline_acceptance_sum: list[float] = []
    position_restricted_acceptance_sum: list[float] = []
    position_acceptance_rows: list[int] = []
    baseline_expected_accepted = 0.0
    restricted_expected_accepted = 0.0

    def ensure_position(position: int) -> None:
        while len(position_sampled_tokens) <= position:
            position_sampled_tokens.append(0)
            position_sampled_hits.append(0)
            position_topk_mass.append(0.0)
            position_covered_topk_mass.append(0.0)
            position_baseline_acceptance_sum.append(0.0)
            position_restricted_acceptance_sum.append(0.0)
            position_acceptance_rows.append(0)

    for payload in steps:
        topk_ids = payload.get("draft_topk_ids")
        topk_probs = payload.get("draft_topk_values")
        target_topk_ids = payload.get("target_topk_ids")
        target_topk_values = payload.get("target_topk_values")
        round_baseline_acceptance: list[float] = []
        round_restricted_acceptance: list[float] = []
        if isinstance(topk_ids, torch.Tensor) and isinstance(topk_probs, torch.Tensor):
            ids_by_position = topk_ids.reshape(-1, topk_ids.shape[-1]).tolist()
            probs_by_position = topk_probs.reshape(-1, topk_probs.shape[-1]).tolist()
            for position, (position_ids, position_probs) in enumerate(
                zip(ids_by_position, probs_by_position)
            ):
                ensure_position(position)
                for token_id, probability in zip(position_ids, position_probs):
                    token_id = int(token_id)
                    probability = float(probability)
                    hit = token_id in base or token_id in tail
                    topk_entries += 1
                    topk_mass += probability
                    position_topk_mass[position] += probability
                    if hit:
                        topk_hits += 1
                        covered_topk_mass += probability
                        position_covered_topk_mass[position] += probability
            if isinstance(target_topk_ids, torch.Tensor) and isinstance(
                target_topk_values, torch.Tensor
            ):
                target_ids_by_position = target_topk_ids.reshape(
                    -1, target_topk_ids.shape[-1]
                ).tolist()
                target_probs_by_position = (
                    target_topk_values.reshape(-1, target_topk_values.shape[-1])
                    .softmax(dim=-1, dtype=torch.float32)
                    .tolist()
                )
                for position, (
                    draft_ids_row,
                    draft_probs_row,
                    target_ids_row,
                    target_probs_row,
                ) in enumerate(
                    zip(
                        ids_by_position,
                        probs_by_position,
                        target_ids_by_position,
                        target_probs_by_position,
                    )
                ):
                    ensure_position(position)
                    target_distribution = {
                        int(token_id): float(probability)
                        for token_id, probability in zip(
                            target_ids_row, target_probs_row
                        )
                    }
                    active_draft_mass = sum(
                        float(probability)
                        for token_id, probability in zip(draft_ids_row, draft_probs_row)
                        if int(token_id) in base or int(token_id) in tail
                    )
                    baseline_acceptance = sum(
                        min(
                            target_distribution.get(int(token_id), 0.0),
                            float(probability),
                        )
                        for token_id, probability in zip(draft_ids_row, draft_probs_row)
                    )
                    restricted_acceptance = (
                        sum(
                            min(
                                target_distribution.get(int(token_id), 0.0),
                                float(probability) / active_draft_mass,
                            )
                            for token_id, probability in zip(
                                draft_ids_row, draft_probs_row
                            )
                            if int(token_id) in base or int(token_id) in tail
                        )
                        if active_draft_mass
                        else 0.0
                    )
                    round_baseline_acceptance.append(baseline_acceptance)
                    round_restricted_acceptance.append(restricted_acceptance)
                    position_baseline_acceptance_sum[position] += baseline_acceptance
                    position_restricted_acceptance_sum[position] += (
                        restricted_acceptance
                    )
                    position_acceptance_rows[position] += 1
        draft_ids = payload.get("draft_token_ids")
        if isinstance(draft_ids, torch.Tensor):
            for position, token_id in enumerate(draft_ids.reshape(-1).tolist()):
                ensure_position(position)
                token_id = int(token_id)
                hit = token_id in base or token_id in tail
                sampled_tokens += 1
                sampled_hits += int(hit)
                position_sampled_tokens[position] += 1
                position_sampled_hits[position] += int(hit)
                if not hit:
                    sampled_misses[token_id] += 1

        baseline_reach_probability = 1.0
        restricted_reach_probability = 1.0
        for baseline_acceptance, restricted_acceptance in zip(
            round_baseline_acceptance, round_restricted_acceptance
        ):
            baseline_reach_probability *= baseline_acceptance
            restricted_reach_probability *= restricted_acceptance
            baseline_expected_accepted += baseline_reach_probability
            restricted_expected_accepted += restricted_reach_probability

        valid_counts = payload.get("output_valid_counts")
        output_ids = payload.get("output_token_ids")
        if (
            update_observed_output
            and isinstance(valid_counts, torch.Tensor)
            and isinstance(output_ids, torch.Tensor)
        ):
            count = int(valid_counts.reshape(-1)[0])
            tail.insert(
                [int(x) for x in output_ids.reshape(-1)[:count].tolist() if x >= 0]
            )
        if previous_candidate_keys:
            candidate_ids: list[int] = []
            for key in previous_candidate_keys:
                ids = payload.get(key)
                candidate_values = payload.get(key.replace("_ids", "_values"))
                if isinstance(ids, torch.Tensor) and isinstance(
                    candidate_values, torch.Tensor
                ):
                    valid = (
                        torch.isfinite(candidate_values)
                        if key == "target_topk_ids"
                        else candidate_values > 0
                    )
                    candidate_ids.extend(
                        int(x) for x in ids[valid].reshape(-1).tolist()
                    )
            tail.insert(candidate_ids)
    return {
        "steps": len(steps),
        "topk_entries": topk_entries,
        "topk_id_coverage": topk_hits / topk_entries if topk_entries else 1.0,
        "topk_probability_mass_coverage": (
            covered_topk_mass / topk_mass if topk_mass else 1.0
        ),
        "sampled_draft_token_coverage": (
            sampled_hits / sampled_tokens if sampled_tokens else 1.0
        ),
        "sampled_draft_token_coverage_by_position": [
            hits / tokens if tokens else 1.0
            for hits, tokens in zip(position_sampled_hits, position_sampled_tokens)
        ],
        "topk_probability_mass_coverage_by_position": [
            covered / total if total else 1.0
            for covered, total in zip(position_covered_topk_mass, position_topk_mass)
        ],
        "sampled_misses": sampled_tokens - sampled_hits,
        "sampled_miss_token_counts": dict(sampled_misses.most_common()),
        "mean_acceptance_probability_by_position": {
            "baseline": [
                total / rows if rows else 0.0
                for total, rows in zip(
                    position_baseline_acceptance_sum, position_acceptance_rows
                )
            ],
            "restricted": [
                total / rows if rows else 0.0
                for total, rows in zip(
                    position_restricted_acceptance_sum, position_acceptance_rows
                )
            ],
        },
        "counterfactual_expected_acceptance_length": {
            "baseline": 1.0 + baseline_expected_accepted / len(steps) if steps else 1.0,
            "restricted": 1.0 + restricted_expected_accepted / len(steps)
            if steps
            else 1.0,
            "method": (
                "baseline-trajectory estimate using per-position sum(min(p,q)) "
                "and sequential reach-probability products"
            ),
        },
        "final_tail_rows": len(tail.ids),
    }


def main() -> int:
    args = _parse_args()
    tail_sizes = args.tail_size or [512, 1024, 2048, 4096, 8192]
    if args.base_size <= 0 or any(size <= 0 for size in tail_sizes):
        raise ValueError("Base and tail sizes must be positive.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    payload = torch.load(args.ranking, map_location="cpu", weights_only=True)
    ranking_tensor = payload.get("global_ranking")
    if not isinstance(ranking_tensor, torch.Tensor) or ranking_tensor.ndim != 1:
        raise ValueError("Ranking artifact is missing global_ranking.")
    ranking = [int(x) for x in ranking_tensor.tolist()]
    if args.base_size + max(tail_sizes) > len(ranking):
        raise ValueError("Base plus tail exceeds the model vocabulary.")

    records = _load_records(args.eval)
    alignment_steps = (
        _load_alignment_steps(args.alignment_glob) if args.alignment_glob else []
    )
    alignment_prompt = _alignment_prompt_ids(args.alignment_summary, tokenizer)
    base_only: dict[str, Any] = {
        "output": _output_coverage(
            records,
            tokenizer,
            ranking,
            args.base_size,
            0,
        )
    }
    if alignment_steps:
        base_only["alignment"] = _alignment_coverage(
            alignment_steps,
            alignment_prompt,
            ranking,
            args.base_size,
            0,
            seed_prompt=False,
            update_observed_output=False,
            previous_candidate_keys=(),
        )
    cases = []
    for tail_size in tail_sizes:
        case: dict[str, Any] = {
            "tail_size": tail_size,
            "output": _output_coverage(
                records,
                tokenizer,
                ranking,
                args.base_size,
                tail_size,
            ),
        }
        if alignment_steps:
            case["alignment_ranked_seed_only"] = _alignment_coverage(
                alignment_steps,
                alignment_prompt,
                ranking,
                args.base_size,
                tail_size,
                seed_prompt=False,
                update_observed_output=False,
                previous_candidate_keys=(),
            )
            case["alignment_prompt_seed_only"] = _alignment_coverage(
                alignment_steps,
                alignment_prompt,
                ranking,
                args.base_size,
                tail_size,
                seed_prompt=True,
                update_observed_output=False,
                previous_candidate_keys=(),
            )
            case["alignment_prompt_observed_output"] = _alignment_coverage(
                alignment_steps,
                alignment_prompt,
                ranking,
                args.base_size,
                tail_size,
                seed_prompt=True,
                update_observed_output=True,
                previous_candidate_keys=(),
            )
            case["alignment_previous_target"] = _alignment_coverage(
                alignment_steps,
                alignment_prompt,
                ranking,
                args.base_size,
                tail_size,
                seed_prompt=True,
                update_observed_output=True,
                previous_candidate_keys=("target_topk_ids",),
            )
            case["alignment_previous_draft"] = _alignment_coverage(
                alignment_steps,
                alignment_prompt,
                ranking,
                args.base_size,
                tail_size,
                seed_prompt=True,
                update_observed_output=True,
                previous_candidate_keys=("draft_topk_ids",),
            )
            case["alignment_previous_target_and_draft"] = _alignment_coverage(
                alignment_steps,
                alignment_prompt,
                ranking,
                args.base_size,
                tail_size,
                seed_prompt=True,
                update_observed_output=True,
                previous_candidate_keys=("target_topk_ids", "draft_topk_ids"),
            )
        cases.append(case)

    result = {
        "model": str(args.model),
        "ranking": str(args.ranking),
        "base_size": args.base_size,
        "base_only": base_only,
        "tail_policy": (
            "next-ranked rows seeded at request start; prompt and observed output "
            "tokens update a per-request LRU"
        ),
        "eval_paths": [str(path) for path in args.eval],
        "alignment_glob": args.alignment_glob,
        "cases": cases,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
