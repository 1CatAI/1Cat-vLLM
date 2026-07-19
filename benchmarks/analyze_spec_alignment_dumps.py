# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize speculative decoding alignment dumps.

The dumps are produced by running vLLM with:

    VLLM_SPEC_DUMP_ALIGNMENT=1 VLLM_SPEC_DUMP_ALIGNMENT_LIMIT=<N>

They are per-rank `.pt` files. TP ranks should normally produce identical
payloads, so this script deduplicates by sampler step before computing the
acceptance and distribution-overlap summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import regex as re
import torch

PLACEHOLDER_TOKEN_ID = -1
_DUMP_RE = re.compile(r"pid(?P<pid>\d+)_step(?P<step>\d+)_")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - idx) + xs[hi] * (idx - lo))


def _tensor_equal(a: Any, b: Any) -> bool:
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return a == b
    if torch.is_floating_point(a):
        return bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5))
    return bool(torch.equal(a, b))


def _load_dumps(pattern: str) -> dict[int, list[tuple[str, Path, dict[str, Any]]]]:
    by_step: dict[int, list[tuple[str, Path, dict[str, Any]]]] = {}
    for name in sorted(glob.glob(pattern)):
        path = Path(name)
        match = _DUMP_RE.search(path.name)
        if match is None:
            continue
        pid = match.group("pid")
        step = int(match.group("step"))
        payload = torch.load(path, map_location="cpu")
        by_step.setdefault(step, []).append((pid, path, payload))
    return by_step


def _dedupe_steps(
    by_step: dict[int, list[tuple[str, Path, dict[str, Any]]]],
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    compare_keys = (
        "draft_token_ids",
        "output_token_ids",
        "draft_acceptance_caps",
        "recovered_residual_mass",
        "draft_token_target_topk_rank",
        "draft_target_probs",
        "draft_token_probs",
    )
    mismatches: list[dict[str, Any]] = []
    selected: list[tuple[int, dict[str, Any]]] = []
    pids: set[str] = set()

    for step, entries in sorted(by_step.items()):
        base_pid, base_path, base_payload = entries[0]
        pids.add(base_pid)
        selected.append((step, base_payload))
        for pid, path, payload in entries[1:]:
            pids.add(pid)
            for key in compare_keys:
                if key not in base_payload or key not in payload:
                    continue
                if not _tensor_equal(base_payload[key], payload[key]):
                    mismatches.append(
                        {
                            "step": step,
                            "pid": pid,
                            "path": str(path),
                            "key": key,
                        }
                    )
                    break

    meta = {
        "num_raw_files": sum(len(entries) for entries in by_step.values()),
        "num_deduped_steps": len(selected),
        "pids": sorted(pids),
        "num_duplicate_mismatches": len(mismatches),
        "duplicate_mismatches": mismatches[:20],
    }
    if selected:
        meta["first_step"] = selected[0][0]
        meta["last_step"] = selected[-1][0]
    return selected, meta


def _step_acceptance(payload: dict[str, Any]) -> tuple[int, int, int]:
    draft_tokens = payload["draft_token_ids"].view(-1).tolist()
    output_tokens = payload["output_token_ids"][0].tolist()
    valid_count = int(payload["output_valid_counts"][0])
    accepted = 0
    for pos, token_id in enumerate(draft_tokens):
        if pos >= len(output_tokens):
            break
        if output_tokens[pos] == PLACEHOLDER_TOKEN_ID:
            break
        if output_tokens[pos] != token_id:
            break
        accepted += 1
    return accepted, valid_count, len(draft_tokens)


def summarize(pattern: str) -> dict[str, Any]:
    by_step = _load_dumps(pattern)
    selected, meta = _dedupe_steps(by_step)

    step_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    for step, payload in selected:
        accepted, valid_count, draft_count = _step_acceptance(payload)
        finish = (
            "all_accept_bonus"
            if accepted == draft_count and valid_count == draft_count + 1
            else "reject_or_recover"
            if valid_count == accepted + 1
            else "other"
        )
        step_rows.append(
            {
                "step": step,
                "accepted_draft_tokens": accepted,
                "valid_output_tokens": valid_count,
                "draft_tokens": draft_count,
                "finish": finish,
            }
        )

        caps = payload.get("draft_acceptance_caps")
        residual = payload.get("recovered_residual_mass")
        ranks = payload.get("draft_token_target_topk_rank")
        target_probs = payload.get("draft_target_probs")
        draft_probs = payload.get("draft_token_probs")
        for pos in range(draft_count):
            row = {
                "step": step,
                "position": pos,
                "accepted_prefix": pos < accepted,
            }
            if caps is not None:
                row["acceptance_cap"] = float(caps[pos])
            if residual is not None:
                row["residual_mass"] = float(residual[pos])
                row["distribution_overlap"] = 1.0 - float(residual[pos])
            if ranks is not None:
                row["target_top10_rank"] = int(ranks[pos])
            if target_probs is not None:
                row["target_prob_of_draft_token"] = float(target_probs[pos])
            if draft_probs is not None:
                row["draft_prob_of_draft_token"] = float(draft_probs[pos])
            token_rows.append(row)

    valid_counts = [float(row["valid_output_tokens"]) for row in step_rows]
    accepted_counts = [float(row["accepted_draft_tokens"]) for row in step_rows]
    accepted_hist = Counter(row["accepted_draft_tokens"] for row in step_rows)
    finish_hist = Counter(row["finish"] for row in step_rows)

    per_position: list[dict[str, Any]] = []
    max_pos = max((row["position"] for row in token_rows), default=-1)
    for pos in range(max_pos + 1):
        rows = [row for row in token_rows if row["position"] == pos]
        caps = [row["acceptance_cap"] for row in rows if "acceptance_cap" in row]
        overlaps = [
            row["distribution_overlap"] for row in rows if "distribution_overlap" in row
        ]
        ranks = [row["target_top10_rank"] for row in rows if "target_top10_rank" in row]
        target_ps = [
            row["target_prob_of_draft_token"]
            for row in rows
            if "target_prob_of_draft_token" in row
        ]
        draft_ps = [
            row["draft_prob_of_draft_token"]
            for row in rows
            if "draft_prob_of_draft_token" in row
        ]
        per_position.append(
            {
                "position": pos,
                "samples": len(rows),
                "observed_prefix_acceptance": _mean(
                    [1.0 if row["accepted_prefix"] else 0.0 for row in rows]
                ),
                "mean_acceptance_cap": _mean(caps),
                "p50_acceptance_cap": _quantile(caps, 0.50),
                "p10_acceptance_cap": _quantile(caps, 0.10),
                "acceptance_cap_lt_0_5_rate": _mean(
                    [1.0 if value < 0.5 else 0.0 for value in caps]
                ),
                "mean_distribution_overlap": _mean(overlaps),
                "p50_distribution_overlap": _quantile(overlaps, 0.50),
                "p10_distribution_overlap": _quantile(overlaps, 0.10),
                "target_top10_hit_rate": _mean(
                    [1.0 if rank > 0 else 0.0 for rank in ranks]
                ),
                "target_top10_miss_rate": _mean(
                    [1.0 if rank < 0 else 0.0 for rank in ranks]
                ),
                "mean_target_prob_of_draft_token": _mean(target_ps),
                "mean_draft_prob_of_draft_token": _mean(draft_ps),
                "draft_prob_gt_target_prob_rate": _mean(
                    [
                        1.0 if draft_p > target_p else 0.0
                        for draft_p, target_p in zip(draft_ps, target_ps)
                    ]
                ),
            }
        )

    return {
        "meta": meta,
        "summary": {
            "mean_output_tokens_per_round": _mean(valid_counts),
            "mean_accepted_draft_tokens_per_round": _mean(accepted_counts),
            "total_output_tokens": int(sum(valid_counts)),
            "total_accepted_draft_tokens": int(sum(accepted_counts)),
            "accepted_draft_count_histogram": dict(sorted(accepted_hist.items())),
            "finish_histogram": dict(sorted(finish_hist.items())),
            "observed_output_tokens_from_prefix_rates": 1.0
            + sum(row["observed_prefix_acceptance"] for row in per_position),
        },
        "per_position": per_position,
        "step_rows": step_rows,
    }


def _format_rate(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{100.0 * value:.1f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    overview = summary["summary"]
    lines = [
        "# Speculative Alignment Dump Summary",
        "",
        "## Inputs",
        "",
        f"- Raw files: `{meta['num_raw_files']}`",
        f"- Deduped sampler steps: `{meta['num_deduped_steps']}`",
        f"- PIDs: `{', '.join(meta['pids'])}`",
        f"- Duplicate mismatches: `{meta['num_duplicate_mismatches']}`",
        f"- Step range: `{meta.get('first_step', '-')}` to "
        f"`{meta.get('last_step', '-')}`",
        "",
        "## Round Summary",
        "",
        "- Mean output tokens per round: "
        f"`{overview['mean_output_tokens_per_round']:.4f}`",
        "- Mean accepted draft tokens per round: "
        f"`{overview['mean_accepted_draft_tokens_per_round']:.4f}`",
        f"- Total output tokens: `{overview['total_output_tokens']}`",
        f"- Total accepted draft tokens: `{overview['total_accepted_draft_tokens']}`",
        f"- Accepted-draft histogram: `{overview['accepted_draft_count_histogram']}`",
        f"- Finish histogram: `{overview['finish_histogram']}`",
        "",
        "## Per-Position Summary",
        "",
        "| pos | prefix accept | mean overlap | p50 overlap | p10 overlap | "
        "mean cap | p50 cap | cap < 0.5 | target top10 miss | "
        "q(draft)>p(target) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["per_position"]:
        lines.append(
            "| "
            f"{row['position']} | "
            f"{_format_rate(row['observed_prefix_acceptance'])} | "
            f"{_format_rate(row['mean_distribution_overlap'])} | "
            f"{_format_rate(row['p50_distribution_overlap'])} | "
            f"{_format_rate(row['p10_distribution_overlap'])} | "
            f"{_format_rate(row['mean_acceptance_cap'])} | "
            f"{_format_rate(row['p50_acceptance_cap'])} | "
            f"{_format_rate(row['acceptance_cap_lt_0_5_rate'])} | "
            f"{_format_rate(row['target_top10_miss_rate'])} | "
            f"{_format_rate(row['draft_prob_gt_target_prob_rate'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="/tmp/spec_alignment_pid*.pt",
        help="Glob for VLLM_SPEC_DUMP_ALIGNMENT payloads.",
    )
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    summary = summarize(args.glob)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = render_markdown(summary)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
