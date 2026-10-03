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

LIMITS = {
    "mean_kl": 1e-4,
    "p99_kl": 1e-3,
    "max_kl": 1e-2,
    "top1_agreement": 0.995,
    "high_margin_top1_agreement": 0.999,
    "max_logit_error": 0.125,
    "max_centered_logit_error": 0.125,
    "p99_row_max_logit_error": 0.03125,
}


def compare_logits(
    control: torch.Tensor, candidate: torch.Tensor, *, chunk_rows: int = 32
) -> dict:
    if control.ndim != 2 or control.shape != candidate.shape:
        raise ValueError("Expected matching [positions, vocabulary] logits")
    if control.shape[0] == 0 or control.shape[1] < 2 or chunk_rows < 1:
        raise ValueError("Need positions, at least two vocabulary items and a chunk")
    if not control.is_floating_point() or not candidate.is_floating_point():
        raise ValueError("Logits must be floating point")
    if not torch.isfinite(control).all() or not torch.isfinite(candidate).all():
        raise ValueError("Nonfinite teacher-forcing logits")

    kls, agreements, margins, errors, centered_errors, scales = [], [], [], [], [], []
    for start in range(0, control.shape[0], chunk_rows):
        left = control[start : start + chunk_rows].to(device="cpu", dtype=torch.float64)
        right = candidate[start : start + chunk_rows].to(
            device="cpu", dtype=torch.float64
        )
        log_p = left.log_softmax(dim=-1)
        log_q = right.log_softmax(dim=-1)
        # Tiny negative roundoff at an identical distribution is not negative KL.
        kls.append((log_p.exp() * (log_p - log_q)).sum(-1).clamp_min(0))
        agreements.append(left.argmax(-1) == right.argmax(-1))
        top2 = left.topk(2, dim=-1).values
        margins.append(top2[:, 0] - top2[:, 1])
        delta = right - left
        errors.append(delta.abs().amax(-1))
        centered_errors.append((delta - delta.mean(-1, keepdim=True)).abs().amax(-1))
        scales.append(left.abs().amax(-1))
    kl = torch.cat(kls)
    agree = torch.cat(agreements)
    margin = torch.cat(margins)
    error = torch.cat(errors)
    centered = torch.cat(centered_errors)
    magnitude = torch.cat(scales)
    high_margin = margin >= 0.1
    result = {
        "positions": control.shape[0],
        "vocabulary": control.shape[1],
        "control_dtype": str(control.dtype),
        "candidate_dtype": str(candidate.dtype),
        "mean_kl": kl.mean().item(),
        "p99_kl": kl.quantile(0.99).item(),
        "max_kl": kl.max().item(),
        "top1_agreement": agree.double().mean().item(),
        "top1_disagreements": (~agree).sum().item(),
        "high_margin_positions": high_margin.sum().item(),
        "high_margin_top1_agreement": (
            agree[high_margin].double().mean().item() if high_margin.any() else None
        ),
        "max_logit_error": error.max().item(),
        "max_centered_logit_error": centered.max().item(),
        "p99_row_max_logit_error": error.quantile(0.99).item(),
        "max_control_logit_magnitude": magnitude.max().item(),
        "p99_control_logit_magnitude": magnitude.quantile(0.99).item(),
        "median_top2_margin": margin.median().item(),
    }
    result["segments"] = [
        {
            "start": start,
            "end": end,
            "mean_kl": kl[start:end].mean().item(),
            "max_logit_error": error[start:end].max().item(),
            "top1_agreement": agree[start:end].double().mean().item(),
        }
        for start, end in zip(
            [0, control.shape[0] // 3, 2 * control.shape[0] // 3],
            [control.shape[0] // 3, 2 * control.shape[0] // 3, control.shape[0]],
        )
        if end > start
    ]
    checks = {}
    for name, limit in LIMITS.items():
        value = result[name]
        checks[name] = (
            None
            if value is None
            else value >= limit
            if name.endswith("agreement")
            else value <= limit
        )
    result["limits"] = LIMITS.copy()
    result["checks"] = checks
    result["passed"] = all(value is not False for value in checks.values())
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
    return {
        "prompt_sha256": control["prompt_sha256"],
        "role": control["role"],
        **compare_logits(control["logits"], candidate["logits"]),
    }


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
