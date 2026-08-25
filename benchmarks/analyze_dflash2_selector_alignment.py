# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and sweep exact DFlash2 selector proposals from alignment dumps.

This tool consumes the B1 records emitted by the GPU-runner compact rejection
path when ``VLLM_SPEC_DUMP_ALIGNMENT=1``. It never changes target sampling.
Counterfactual results are one-step overlap proxies on the recorded prefixes;
only an end-to-end run can establish a new acceptance-length result.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ProposalConfig:
    name: str
    proposal_temperature_scale: float = 1.0
    proposal_top_p: float = 1.0
    unary_scale: float = 1.0
    edge_scale: float = 1.0
    future_beta: float = 0.0
    greedy_mix: float = 0.0
    use_cached_logits: bool = False


@dataclass
class AlignmentRecord:
    path: Path
    step: int
    temperature: float
    top_p: float
    target_topk_ids: torch.Tensor
    target_topk_logits: torch.Tensor
    candidate_ids: torch.Tensor
    realized_logits: torch.Tensor
    unary_logits: torch.Tensor
    lattice_scores: torch.Tensor
    draft_sampled: torch.Tensor
    num_sampled: int


def _compact_probs(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply the compact top-p contract used by the sparse rejection kernel."""
    logits = logits.to(torch.float64)
    order = torch.argsort(logits, descending=True, stable=True)
    sorted_probs = torch.softmax(logits[order], dim=-1)
    cumulative_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    keep_sorted = (top_p >= 1.0) | (cumulative_before < top_p)
    kept = torch.zeros_like(sorted_probs)
    kept[keep_sorted] = sorted_probs[keep_sorted]
    kept /= kept.sum()
    probs = torch.zeros_like(kept)
    probs[order] = kept
    return probs


def _distribution_overlap(
    target_ids: torch.Tensor,
    target_probs: torch.Tensor,
    proposal_ids: torch.Tensor,
    proposal_probs: torch.Tensor,
) -> float:
    matches = target_ids[:, None] == proposal_ids[None, :]
    proposal_on_target = torch.where(
        matches,
        proposal_probs[None, :],
        torch.zeros((), dtype=proposal_probs.dtype),
    ).sum(dim=1)
    return float(torch.minimum(target_probs, proposal_on_target).sum())


def _support_mass(
    target_ids: torch.Tensor,
    target_probs: torch.Tensor,
    proposal_ids: torch.Tensor,
) -> float:
    supported = (target_ids[:, None] == proposal_ids[None, :]).any(dim=1)
    return float(target_probs[supported].sum())


def _transformed_lattice(
    record: AlignmentRecord, config: ProposalConfig
) -> torch.Tensor:
    unary = record.unary_logits[:, None, :].to(torch.float64)
    lattice = record.lattice_scores.to(torch.float64)
    edge = lattice - unary
    denominator = record.temperature * config.proposal_temperature_scale
    return (config.unary_scale * unary + config.edge_scale * edge) / denominator


def _backward_messages(lattice: torch.Tensor) -> torch.Tensor:
    """Compute normalized log-sum-exp future messages for a KxK chain."""
    num_steps, top_k, _ = lattice.shape
    future = torch.zeros((num_steps, top_k), dtype=lattice.dtype)
    for step in range(num_steps - 2, -1, -1):
        values = lattice[step + 1] + future[step + 1][None, :]
        future[step] = torch.logsumexp(values, dim=-1)
        future[step] -= future[step].max()
    return future


def _path_predecessors(record: AlignmentRecord) -> list[int]:
    num_steps = record.candidate_ids.shape[0]
    if record.draft_sampled.numel() < num_steps + 1:
        raise ValueError(f"{record.path}: draft_sampled is missing the anchor row")
    predecessors = [0]
    for step in range(num_steps - 1):
        proposed = int(record.draft_sampled[step + 1])
        matches = torch.nonzero(record.candidate_ids[step] == proposed).flatten()
        if matches.numel() != 1:
            raise ValueError(
                f"{record.path}: proposed token {proposed} is not unique at step {step}"
            )
        predecessors.append(int(matches[0]))
    return predecessors


def _proposal_probs(
    record: AlignmentRecord,
    config: ProposalConfig,
) -> list[torch.Tensor]:
    if config.use_cached_logits:
        rows = [
            _compact_probs(row, config.proposal_top_p) for row in record.realized_logits
        ]
    else:
        lattice = _transformed_lattice(record, config)
        future = _backward_messages(lattice)
        predecessors = _path_predecessors(record)
        rows = [
            _compact_probs(
                lattice[step, predecessor] + config.future_beta * future[step],
                config.proposal_top_p,
            )
            for step, predecessor in enumerate(predecessors)
        ]
    if config.greedy_mix == 0.0:
        return rows
    if not 0.0 <= config.greedy_mix <= 1.0:
        raise ValueError("greedy_mix must be in [0, 1]")
    mixed_rows = []
    for probs in rows:
        mixed = probs * (1.0 - config.greedy_mix)
        mixed[torch.argmax(probs)] += config.greedy_mix
        mixed_rows.append(mixed)
    return mixed_rows


def _target_rows(record: AlignmentRecord) -> list[torch.Tensor]:
    rows = []
    for logits in record.target_topk_logits[: record.candidate_ids.shape[0]]:
        rows.append(_compact_probs(logits / record.temperature, record.top_p))
    return rows


def _load_record(path: Path) -> AlignmentRecord:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "dflash2_selector_alignment_v1":
        raise ValueError(f"{path}: unsupported alignment format")
    candidate_ids = payload["selector_candidate_ids"].to(torch.int64)
    cached_ids = payload["draft_candidate_ids"].to(torch.int64)
    if not torch.equal(candidate_ids, cached_ids):
        raise ValueError(f"{path}: packed selector IDs do not match request-slot cache")
    record = AlignmentRecord(
        path=path,
        step=int(payload["step"]),
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        target_topk_ids=payload["target_topk_ids"].to(torch.int64),
        target_topk_logits=payload["target_topk_logits"].to(torch.float64),
        candidate_ids=candidate_ids,
        realized_logits=payload["draft_realized_logits"].to(torch.float64),
        unary_logits=payload["selector_unary_logits"].to(torch.float64),
        lattice_scores=payload["selector_lattice_scores"].to(torch.float64),
        draft_sampled=payload["draft_sampled"].to(torch.int64).flatten(),
        num_sampled=int(payload["num_sampled"].flatten()[0]),
    )
    return record


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def _completion_proxy(per_position_overlap: list[float]) -> float:
    survival = 1.0
    result = 1.0
    for overlap in per_position_overlap:
        survival *= overlap
        result += survival
    return result


def _evaluate(records: list[AlignmentRecord], config: ProposalConfig) -> dict[str, Any]:
    num_steps = records[0].candidate_ids.shape[0]
    overlaps = [[] for _ in range(num_steps)]
    for record in records:
        target_rows = _target_rows(record)
        proposal_rows = _proposal_probs(record, config)
        for step, (target_probs, proposal_probs) in enumerate(
            zip(target_rows, proposal_rows)
        ):
            overlaps[step].append(
                _distribution_overlap(
                    record.target_topk_ids[step],
                    target_probs,
                    record.candidate_ids[step],
                    proposal_probs,
                )
            )
    means = [_mean(values) for values in overlaps]
    return {
        "config": asdict(config),
        "num_records": len(records),
        "mean_overlap_by_position": means,
        "completion_length_proxy": _completion_proxy(means),
    }


def _sweep_configs() -> list[ProposalConfig]:
    configs = [ProposalConfig(name="current", use_cached_logits=True)]
    seen: set[tuple[float, float, float, float, float, float]] = set()

    def add(
        family: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        unary: float = 1.0,
        edge: float = 1.0,
        beta: float = 0.0,
        greedy_mix: float = 0.0,
    ) -> None:
        key = (temperature, top_p, unary, edge, beta, greedy_mix)
        if key in seen:
            return
        seen.add(key)
        configs.append(
            ProposalConfig(
                name=(
                    f"{family}:t={temperature:g},p={top_p:g},u={unary:g},"
                    f"e={edge:g},b={beta:g},g={greedy_mix:g}"
                ),
                proposal_temperature_scale=temperature,
                proposal_top_p=top_p,
                unary_scale=unary,
                edge_scale=edge,
                future_beta=beta,
                greedy_mix=greedy_mix,
            )
        )

    for temperature in (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3):
        add("temperature", temperature=temperature)
    for top_p in (0.9, 0.95, 0.98, 0.99):
        for temperature in (0.8, 0.9, 1.0):
            add("nucleus", temperature=temperature, top_p=top_p)
    for greedy_mix in (0.1, 0.2, 0.3, 0.4, 0.5):
        for temperature in (0.8, 0.9, 1.0, 1.1):
            add(
                "greedy-mixture",
                temperature=temperature,
                greedy_mix=greedy_mix,
            )
    for unary in (0.75, 1.0, 1.25):
        for edge in (0.5, 0.75, 1.0, 1.25, 1.5):
            for temperature in (0.85, 1.0, 1.15):
                add(
                    "edge-calibration",
                    temperature=temperature,
                    unary=unary,
                    edge=edge,
                )
    for beta in (0.25, 0.5, 0.75, 1.0):
        for temperature in (0.8, 0.9, 1.0, 1.1):
            for edge in (0.75, 1.0, 1.25):
                add(
                    "future-message",
                    temperature=temperature,
                    edge=edge,
                    beta=beta,
                )
    return configs


def _baseline_diagnostics(records: list[AlignmentRecord]) -> dict[str, Any]:
    num_steps = records[0].candidate_ids.shape[0]
    support = [[] for _ in range(num_steps)]
    cached_lattice_max_abs: list[float] = []
    for record in records:
        target_rows = _target_rows(record)
        predecessors = _path_predecessors(record)
        for step, target_probs in enumerate(target_rows):
            support[step].append(
                _support_mass(
                    record.target_topk_ids[step],
                    target_probs,
                    record.candidate_ids[step],
                )
            )
            expected = (
                record.lattice_scores[step, predecessors[step]] / record.temperature
            )
            cached_lattice_max_abs.append(
                float((expected - record.realized_logits[step]).abs().max())
            )
    support_means = [_mean(values) for values in support]
    return {
        "observed_mean_completion_tokens_per_round": _mean(
            [float(record.num_sampled) for record in records]
        ),
        "candidate_support_mass_by_position": support_means,
        "candidate_support_completion_upper_proxy": _completion_proxy(support_means),
        "cached_vs_lattice_max_abs": max(cached_lattice_max_abs, default=math.nan),
    }


def summarize(pattern: str, top_n: int) -> dict[str, Any]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise ValueError(f"no selector alignment dumps matched {pattern!r}")
    records = [_load_record(path) for path in paths]
    tune = [record for record in records if record.step % 2 == 0]
    holdout = [record for record in records if record.step % 2 == 1]
    if not tune or not holdout:
        tune = records
        holdout = records

    configs = _sweep_configs()
    tune_results = [_evaluate(tune, config) for config in configs]
    holdout_by_name = {
        result["config"]["name"]: result
        for result in (_evaluate(holdout, config) for config in configs)
    }
    current_tune = tune_results[0]["completion_length_proxy"]
    current_holdout = holdout_by_name["current"]["completion_length_proxy"]
    position_policy = []
    position_tune_overlaps = []
    position_holdout_overlaps = []
    for position in range(records[0].candidate_ids.shape[0]):
        best = max(
            tune_results,
            key=lambda result: result["mean_overlap_by_position"][position],
        )
        holdout_best = holdout_by_name[best["config"]["name"]]
        position_policy.append(
            {
                "position": position,
                "config": best["config"],
                "tune_overlap": best["mean_overlap_by_position"][position],
                "holdout_overlap": holdout_best["mean_overlap_by_position"][position],
            }
        )
        position_tune_overlaps.append(best["mean_overlap_by_position"][position])
        position_holdout_overlaps.append(
            holdout_best["mean_overlap_by_position"][position]
        )
    ranked = sorted(
        tune_results[1:],
        key=lambda result: result["completion_length_proxy"],
        reverse=True,
    )
    leaders = []
    for result in ranked[:top_n]:
        holdout_result = holdout_by_name[result["config"]["name"]]
        leaders.append(
            {
                "config": result["config"],
                "tune_completion_length_proxy": result["completion_length_proxy"],
                "tune_delta": result["completion_length_proxy"] - current_tune,
                "holdout_completion_length_proxy": holdout_result[
                    "completion_length_proxy"
                ],
                "holdout_delta": (
                    holdout_result["completion_length_proxy"] - current_holdout
                ),
                "tune_mean_overlap_by_position": result["mean_overlap_by_position"],
                "holdout_mean_overlap_by_position": holdout_result[
                    "mean_overlap_by_position"
                ],
            }
        )

    return {
        "contract": {
            "glob": pattern,
            "num_records": len(records),
            "num_tune_records": len(tune),
            "num_holdout_records": len(holdout),
            "split": "even/odd sampler step",
            "counterfactual_scope": "one-step recorded-prefix overlap proxy",
            "end_to_end_claim": False,
        },
        "baseline": _baseline_diagnostics(records),
        "current": {
            "tune": tune_results[0],
            "holdout": holdout_by_name["current"],
        },
        "position_policy": {
            "selection": "best tune overlap independently at each depth",
            "rows": position_policy,
            "tune_completion_length_proxy": _completion_proxy(position_tune_overlaps),
            "tune_delta": (_completion_proxy(position_tune_overlaps) - current_tune),
            "holdout_completion_length_proxy": _completion_proxy(
                position_holdout_overlaps
            ),
            "holdout_delta": (
                _completion_proxy(position_holdout_overlaps) - current_holdout
            ),
        },
        "leaders": leaders,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    contract = summary["contract"]
    baseline = summary["baseline"]
    current = summary["current"]
    position_policy = summary["position_policy"]
    lines = [
        "# DFlash2 Selector Alignment Sweep",
        "",
        (
            f"- Records: `{contract['num_records']}` "
            f"(tune `{contract['num_tune_records']}`, "
            f"holdout `{contract['num_holdout_records']}`)"
        ),
        (
            "- Counterfactual scope: one-step overlap on recorded prefixes; "
            "this is not an end-to-end acceptance claim."
        ),
        (
            "- Observed completion tokens/round: "
            f"`{baseline['observed_mean_completion_tokens_per_round']:.4f}`"
        ),
        (
            "- Current overlap proxy (tune/holdout): "
            f"`{current['tune']['completion_length_proxy']:.4f}` / "
            f"`{current['holdout']['completion_length_proxy']:.4f}`"
        ),
        (
            "- Fixed-top16 support upper proxy: "
            f"`{baseline['candidate_support_completion_upper_proxy']:.4f}`"
        ),
        (
            "- Cached/lattice max absolute mismatch: "
            f"`{baseline['cached_vs_lattice_max_abs']:.3e}`"
        ),
        (
            "- Position-wise held-out proxy/delta: "
            f"`{position_policy['holdout_completion_length_proxy']:.4f}` / "
            f"`{position_policy['holdout_delta']:+.4f}`"
        ),
        "",
        "| candidate | tune proxy | tune delta | holdout proxy | holdout delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["leaders"]:
        lines.append(
            f"| `{row['config']['name']}` | "
            f"{row['tune_completion_length_proxy']:.4f} | "
            f"{row['tune_delta']:+.4f} | "
            f"{row['holdout_completion_length_proxy']:.4f} | "
            f"{row['holdout_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Position-wise calibration policy",
            "",
            "| position | tune overlap | holdout overlap | configuration |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    for row in position_policy["rows"]:
        lines.append(
            f"| {row['position']} | {row['tune_overlap']:.4f} | "
            f"{row['holdout_overlap']:.4f} | `{row['config']['name']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="/tmp/spec_alignment_dflash2_selector_*.pt",
        help="Glob for GPU-runner DFlash2 selector alignment records.",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    summary = summarize(args.glob, args.top)
    markdown = render_markdown(summary)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
