# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare aligned full-vocabulary teacher-forcing logits offline.

Each torch dump contains logits [positions, vocabulary], position_ids,
token_ids, prompt_sha256 and role (target or draft). Sampling transforms must
not be applied. Capture is diagnostic; never time inference with this observer.
"""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.qwen38_distribution_probe import (
    DISTRIBUTION_LIMITS as LIMITS,
)
from benchmarks.qwen38_distribution_probe import (
    distribution_metrics,
    summarize_distribution,
)


def compare_logits(
    control: torch.Tensor, candidate: torch.Tensor, *, chunk_rows: int = 32
) -> dict:
    if control.ndim != 2 or control.shape != candidate.shape:
        raise ValueError("Expected matching [positions, vocabulary] logits")
    if control.shape[0] == 0 or control.shape[1] < 2 or chunk_rows < 1:
        raise ValueError("Need positions, at least two vocabulary items and a chunk")
    if not control.is_floating_point() or not candidate.is_floating_point():
        raise ValueError("Logits must be floating point")
    # Reuse no-MTP's complete-vocabulary FP64 metric, one row at a time.
    rows = [distribution_metrics(a.cpu(), b.cpu()) for a, b in zip(control, candidate)]
    result = summarize_distribution(rows)
    high_margin = [r for r in rows if r["reference_margin"] >= 0.1]
    result.update(
        {
            "positions": control.shape[0],
            "vocabulary": control.shape[1],
            "control_dtype": str(control.dtype),
            "candidate_dtype": str(candidate.dtype),
            "max_centered_logit_error": result["centered_max_logit_error"],
            "p99_row_max_logit_error": result["p99_logit_error"],
            "high_margin_positions": len(high_margin),
            "high_margin_top1_agreement": (
                sum(r["top1_agreement"] for r in high_margin) / len(high_margin)
                if high_margin
                else None
            ),
        }
    )
    result["segments"] = [
        {"start": start, "end": end, **summarize_distribution(rows[start:end])}
        for start, end in zip(
            [0, len(rows) // 3, 2 * len(rows) // 3],
            [len(rows) // 3, 2 * len(rows) // 3, len(rows)],
        )
        if end > start
    ]
    return result


def compare_dumps(control: dict, candidate: dict) -> dict:
    for name in ("prompt_sha256", "role"):
        if control[name] != candidate[name]:
            raise ValueError(f"Teacher-forcing {name} mismatch")
    for name in ("position_ids", "token_ids"):
        if not torch.equal(control[name], candidate[name]):
            raise ValueError(f"Teacher-forcing {name} mismatch")
    rows = control["logits"].shape[0]
    if any(control[name].shape != (rows,) for name in ("position_ids", "token_ids")):
        raise ValueError("Alignment metadata must have one item per logit row")
    if control["role"] not in ("target", "draft"):
        raise ValueError("Teacher-forcing role must be target or draft")
    result = {
        "prompt_sha256": control["prompt_sha256"],
        "role": control["role"],
        **compare_logits(control["logits"], candidate["logits"]),
    }
    if "optimized_token_ids" in candidate:
        # A fused top1 kernel need not materialize its logits. Check the actual
        # accelerated decision as well as the diagnostic full-head logits.
        ids = candidate["optimized_token_ids"]
        if ids.shape != (rows,) or ids.dtype != torch.int64:
            raise ValueError("Optimized tokens must be one int64 ID per position")
        if (ids < 0).any() or (ids >= control["logits"].shape[1]).any():
            raise ValueError("Optimized token outside the full vocabulary")
        logits = control["logits"].float()
        top2 = logits.topk(2, dim=-1).values
        confident = top2[:, 0] - top2[:, 1] >= 0.1
        agree = ids == logits.argmax(dim=-1)
        result["optimized_top1_agreement"] = agree.double().mean().item()
        result["optimized_high_margin_top1_agreement"] = (
            agree[confident].double().mean().item() if confident.any() else None
        )
        result["checks"]["optimized_top1_agreement"] = (
            result["optimized_top1_agreement"] >= LIMITS["top1_agreement"]
        )
        result["passed"] = all(
            value is not False for value in result["checks"].values()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare_dumps(
        torch.load(args.control, map_location="cpu", weights_only=True),
        torch.load(args.candidate, map_location="cpu", weights_only=True),
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


def validate_manifest(tapes, vocabulary):
    """Reject malformed forcing data before expensive model initialization."""
    import hashlib
    import json

    if not isinstance(tapes, list) or not tapes:
        raise ValueError("Teacher-forcing manifest must contain cases")
    seen = set()
    for case in tapes:
        ids = case["token_ids"]
        prompt = case["prompt_length"]
        output = case["output_length"]
        if not isinstance(ids, list) or any(type(t) is not int for t in ids):
            raise ValueError(
                "Teacher-forcing token_ids must be integers, not tokenizer mapping keys"
            )
        if any(t < 0 or t >= vocabulary for t in ids):
            raise ValueError("Teacher-forcing token IDs outside vocabulary")
        if prompt != 8192 or output <= 0 or len(ids) < prompt + output + 8:
            raise ValueError(
                "Teacher-forcing requires 8K prompt and continuation padding"
            )
        actual = hashlib.sha256(json.dumps(ids[:prompt]).encode()).hexdigest()
        if actual != case["prompt_sha256"]:
            raise ValueError("Teacher-forcing prompt SHA mismatch")
        if case["id"] in seen:
            raise ValueError("Duplicate teacher-forcing case ID")
        seen.add(case["id"])
    return tapes
