#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare SM70 tensor dump directories.

The Qwen/GDN graph dump helpers save one file per tensor per TP worker. Worker
PIDs are not stable across runs, so this script aligns same-key tensors by the
lowest cross-rank max-diff instead of matching by PID.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

BoundRule = tuple[str, re.Pattern[str], float]


def _tensor_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {
            "shape_mismatch": True,
            "left_shape": tuple(left.shape),
            "right_shape": tuple(right.shape),
            "max_abs": math.inf,
            "mean_abs": math.inf,
            "cosine": None,
        }

    lf = left.detach().to(torch.float32).reshape(-1)
    rf = right.detach().to(torch.float32).reshape(-1)
    diff = (lf - rf).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    denom = float(torch.linalg.vector_norm(lf) * torch.linalg.vector_norm(rf))
    cosine = float(torch.dot(lf, rf) / denom) if denom > 0 else None
    return {
        "shape_mismatch": False,
        "left_shape": tuple(left.shape),
        "right_shape": tuple(right.shape),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
    }


def _row_stats(left: torch.Tensor, right: torch.Tensor) -> list[dict[str, Any]]:
    if left.ndim == 0 or tuple(left.shape) != tuple(right.shape):
        return []
    rows = min(int(left.shape[0]), 8)
    result = []
    for row in range(rows):
        stats = _tensor_stats(left[row], right[row])
        result.append({"row": row, **stats})
    return result


def _load_records(root: Path) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    records: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - diagnostic utility.
            print(f"skip {path}: {exc}")
            continue
        tensor = payload.get("tensor")
        if not isinstance(tensor, torch.Tensor):
            continue
        step = payload.get("step")
        layer_idx = payload.get("layer_idx")
        label = payload.get("label")
        source = payload.get("source", payload.get("layer_type", "unknown"))
        shape = tuple(payload.get("shape", tuple(tensor.shape)))
        key = (step, layer_idx, source, label, shape)
        records.setdefault(key, []).append({
            "path": str(path),
            "pid": payload.get("pid"),
            "tensor": tensor,
        })
    return records


def _best_pairing(
    left_items: list[dict[str, Any]],
    right_items: list[dict[str, Any]],
) -> tuple[float, list[tuple[int, int, dict[str, Any]]]]:
    n = min(len(left_items), len(right_items))
    if n == 0:
        return math.inf, []
    best_score = math.inf
    best_pairs: list[tuple[int, int, dict[str, Any]]] = []
    # TP is small here; exhaustive permutations keep the implementation simple.
    for right_perm in itertools.permutations(range(len(right_items)), n):
        pairs = []
        score = 0.0
        for left_idx, right_idx in enumerate(right_perm):
            stats = _tensor_stats(
                left_items[left_idx]["tensor"],
                right_items[right_idx]["tensor"],
            )
            score += float(stats["max_abs"])
            pairs.append((left_idx, right_idx, stats))
        if score < best_score:
            best_score = score
            best_pairs = pairs
    return best_score, best_pairs


def _sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("step") if entry.get("step") is not None else -1,
        entry.get("layer_idx") if entry.get("layer_idx") is not None else -1,
        entry.get("source") or "",
        entry.get("label") or "",
        -entry.get("max_abs", -1.0),
    )


def compare_dirs(left: Path, right: Path, top: int) -> dict[str, Any]:
    left_records = _load_records(left)
    right_records = _load_records(right)
    all_keys = sorted(set(left_records) | set(right_records), key=str)
    summaries = []
    missing = []
    for key in all_keys:
        left_items = left_records.get(key, [])
        right_items = right_records.get(key, [])
        step, layer_idx, source, label, shape = key
        if not left_items or not right_items:
            missing.append({
                "step": step,
                "layer_idx": layer_idx,
                "source": source,
                "label": label,
                "shape": shape,
                "left_count": len(left_items),
                "right_count": len(right_items),
            })
            continue
        _, pairs = _best_pairing(left_items, right_items)
        for left_idx, right_idx, stats in pairs:
            left_tensor = left_items[left_idx]["tensor"]
            right_tensor = right_items[right_idx]["tensor"]
            summaries.append({
                "step": step,
                "layer_idx": layer_idx,
                "source": source,
                "label": label,
                "shape": shape,
                "left_pid": left_items[left_idx]["pid"],
                "right_pid": right_items[right_idx]["pid"],
                "left_path": left_items[left_idx]["path"],
                "right_path": right_items[right_idx]["path"],
                **stats,
                "rows": _row_stats(left_tensor, right_tensor),
            })

    by_worst = sorted(
        summaries,
        key=lambda item: (
            bool(item.get("shape_mismatch")),
            float(item.get("max_abs", 0.0)),
        ),
        reverse=True,
    )
    by_time = sorted(summaries, key=_sort_key)
    first_nonzero = next(
        (item for item in by_time if float(item.get("max_abs", 0.0)) != 0.0),
        None,
    )
    return {
        "left": str(left),
        "right": str(right),
        "num_compared": len(summaries),
        "num_missing": len(missing),
        "first_nonzero_by_step": first_nonzero,
        "worst": by_worst[:top],
        "missing": missing[:top],
        "_all_summaries": summaries,
        "_all_missing": missing,
    }


def _parse_bound_rules(raw_values: list[str]) -> list[BoundRule]:
    rules = []
    for raw_value in raw_values:
        pattern, sep, bound = raw_value.rpartition("=")
        if not sep or not pattern:
            raise ValueError(
                f"Invalid bound rule {raw_value!r}; expected PATTERN=FLOAT"
            )
        rules.append((pattern, re.compile(pattern), float(bound)))
    return rules


def _match_bound_rule(
    item: dict[str, Any],
    rules: list[BoundRule],
) -> tuple[float | None, str | None]:
    label = "" if item.get("label") is None else str(item.get("label"))
    source = "" if item.get("source") is None else str(item.get("source"))
    key = f"{source}:{label}"
    for pattern, regex, bound in rules:
        if regex.search(label) or regex.search(key):
            return bound, pattern
    return None, None


def _bound_for_item(
    item: dict[str, Any],
    *,
    global_bound: float | None,
    rules: list[BoundRule],
) -> tuple[float | None, str | None]:
    rule_bound, pattern = _match_bound_rule(item, rules)
    if rule_bound is not None:
        return rule_bound, pattern
    return global_bound, None


def _record_ref(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": item.get("step"),
        "layer_idx": item.get("layer_idx"),
        "source": item.get("source"),
        "label": item.get("label"),
        "shape": item.get("shape"),
        "left_pid": item.get("left_pid"),
        "right_pid": item.get("right_pid"),
        "left_path": item.get("left_path"),
        "right_path": item.get("right_path"),
    }


def _numeric_gate(
    result: dict[str, Any],
    *,
    max_abs_bound: float | None,
    mean_abs_bound: float | None,
    min_cosine: float | None,
    max_abs_rules: list[BoundRule],
    mean_abs_rules: list[BoundRule],
    allow_missing: bool,
) -> dict[str, Any]:
    compared = result["_all_summaries"]
    missing = result["_all_missing"]
    violations = []
    pending = []
    checked_max_abs = 0
    checked_mean_abs = 0
    checked_cosine = 0
    global_max_abs = 0.0
    global_mean_abs = 0.0
    min_seen_cosine = None

    if missing and not allow_missing:
        for item in missing[:20]:
            violations.append({
                "reason": "missing_tensor_dump",
                **item,
            })

    for item in compared:
        max_abs = float(item.get("max_abs", math.inf))
        mean_abs = float(item.get("mean_abs", math.inf))
        cosine = item.get("cosine")
        global_max_abs = max(global_max_abs, max_abs)
        global_mean_abs = max(global_mean_abs, mean_abs)
        if cosine is not None:
            cosine = float(cosine)
            min_seen_cosine = (
                cosine if min_seen_cosine is None else min(min_seen_cosine, cosine)
            )

        if item.get("shape_mismatch"):
            violations.append({
                "reason": "shape_mismatch",
                **_record_ref(item),
                "left_shape": item.get("left_shape"),
                "right_shape": item.get("right_shape"),
            })
            continue

        if not math.isfinite(max_abs) or not math.isfinite(mean_abs):
            violations.append({
                "reason": "non_finite_diff",
                **_record_ref(item),
                "max_abs": item.get("max_abs"),
                "mean_abs": item.get("mean_abs"),
            })
            continue

        item_max_abs_bound, max_abs_rule = _bound_for_item(
            item,
            global_bound=max_abs_bound,
            rules=max_abs_rules,
        )
        if item_max_abs_bound is None:
            if max_abs != 0.0:
                pending.append({
                    "reason": "nonzero_max_abs_without_bound",
                    **_record_ref(item),
                    "max_abs": max_abs,
                })
        else:
            checked_max_abs += 1
            if max_abs > item_max_abs_bound:
                violations.append({
                    "reason": "max_abs_exceeds_bound",
                    **_record_ref(item),
                    "max_abs": max_abs,
                    "bound": item_max_abs_bound,
                    "bound_rule": max_abs_rule,
                })

        item_mean_abs_bound, mean_abs_rule = _bound_for_item(
            item,
            global_bound=mean_abs_bound,
            rules=mean_abs_rules,
        )
        if item_mean_abs_bound is not None:
            checked_mean_abs += 1
            if mean_abs > item_mean_abs_bound:
                violations.append({
                    "reason": "mean_abs_exceeds_bound",
                    **_record_ref(item),
                    "mean_abs": mean_abs,
                    "bound": item_mean_abs_bound,
                    "bound_rule": mean_abs_rule,
                })

        if min_cosine is not None and cosine is not None:
            checked_cosine += 1
            if cosine < min_cosine:
                violations.append({
                    "reason": "cosine_below_bound",
                    **_record_ref(item),
                    "cosine": cosine,
                    "bound": min_cosine,
                })

    if checked_max_abs == 0 and (max_abs_bound is not None or max_abs_rules):
        pending.append("no tensors matched any max_abs bound")
    if checked_mean_abs == 0 and (mean_abs_bound is not None or mean_abs_rules):
        pending.append("no tensors matched any mean_abs bound")
    if checked_cosine == 0 and min_cosine is not None:
        pending.append("no tensors had finite cosine evidence")

    if violations:
        label = "numeric-fail"
    elif pending:
        label = "numeric-pending"
    else:
        label = "numeric-pass"

    return {
        "label": label,
        "num_compared": len(compared),
        "num_missing": len(missing),
        "allow_missing": allow_missing,
        "global_max_abs": global_max_abs,
        "global_max_mean_abs": global_mean_abs,
        "min_seen_cosine": min_seen_cosine,
        "bounds": {
            "max_abs_bound": max_abs_bound,
            "mean_abs_bound": mean_abs_bound,
            "min_cosine": min_cosine,
            "label_max_abs_bounds": [
                {"pattern": pattern, "bound": bound}
                for pattern, _, bound in max_abs_rules
            ],
            "label_mean_abs_bounds": [
                {"pattern": pattern, "bound": bound}
                for pattern, _, bound in mean_abs_rules
            ],
        },
        "num_checked_max_abs": checked_max_abs,
        "num_checked_mean_abs": checked_mean_abs,
        "num_checked_cosine": checked_cosine,
        "violations": violations[:50],
        "num_violations": len(violations),
        "pending": pending[:50],
        "num_pending": len(pending),
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--require-numeric-gate",
        action="store_true",
        help="Exit nonzero unless the configured numeric gate passes.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail the numeric gate when one side lacks a tensor dump.",
    )
    parser.add_argument(
        "--max-abs-bound",
        type=float,
        help="Default max absolute diff bound for compared tensors.",
    )
    parser.add_argument(
        "--mean-abs-bound",
        type=float,
        help="Default mean absolute diff bound for compared tensors.",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        help="Default minimum cosine similarity for compared tensors.",
    )
    parser.add_argument(
        "--label-max-abs-bound",
        action="append",
        default=[],
        metavar="PATTERN=FLOAT",
        help=(
            "Label/source regex-specific max abs bound. The regex is matched "
            "against both label and source:label. May be repeated."
        ),
    )
    parser.add_argument(
        "--label-mean-abs-bound",
        action="append",
        default=[],
        metavar="PATTERN=FLOAT",
        help=(
            "Label/source regex-specific mean abs bound. The regex is matched "
            "against both label and source:label. May be repeated."
        ),
    )
    args = parser.parse_args()

    result = compare_dirs(args.left, args.right, args.top)
    max_abs_rules = _parse_bound_rules(args.label_max_abs_bound)
    mean_abs_rules = _parse_bound_rules(args.label_mean_abs_bound)
    gate_requested = (
        args.require_numeric_gate
        or args.max_abs_bound is not None
        or args.mean_abs_bound is not None
        or args.min_cosine is not None
        or bool(max_abs_rules)
        or bool(mean_abs_rules)
    )
    if gate_requested:
        result["numeric_gate"] = _numeric_gate(
            result,
            max_abs_bound=args.max_abs_bound,
            mean_abs_bound=args.mean_abs_bound,
            min_cosine=args.min_cosine,
            max_abs_rules=max_abs_rules,
            mean_abs_rules=mean_abs_rules,
            allow_missing=args.allow_missing,
        )
    result.pop("_all_summaries", None)
    result.pop("_all_missing", None)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)
    if args.require_numeric_gate:
        gate = result.get("numeric_gate") or {}
        return 0 if gate.get("label") == "numeric-pass" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
