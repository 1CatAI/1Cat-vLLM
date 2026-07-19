# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare SM70 model-quality trace dumps.

This is an offline triage helper. It intentionally does not run a model or
toggle any fast path. Feed it paired no-compact/compact artifacts and it reports
where token, probability, hidden-state, and layer-buffer drift first appears.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import regex as re
import torch

_LAYER_DUMP_RE = re.compile(
    r"pid(?P<pid>\d+)_step(?P<step>\d+)_layer(?P<layer>-?\d+)_"
    r"(?P<tail>.+?)_shape(?P<shape>[^/]+)\.pt$"
)
_PID_RE = re.compile(r"pid(?P<pid>\d+)_")
_SAMPLE_DUMP_RE = re.compile(r"sample_tensors_pid(?P<pid>\d+)_step(?P<step>\d+)\.pt$")


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_json(path: Path | None) -> Any | None:
    if path is None:
        return None
    with path.open() as f:
        return json.load(f)


def _token_ids(payload: Any | None) -> list[list[int]]:
    if not payload:
        return []
    records = payload.get("records") or []
    result: list[list[int]] = []
    for record in records:
        for output in record.get("outputs") or []:
            ids = output.get("token_ids")
            if ids is not None:
                result.append([int(token_id) for token_id in ids])
    return result


def _common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


def _compare_tokens(left: Any | None, right: Any | None) -> dict[str, Any] | None:
    left_ids = _token_ids(left)
    right_ids = _token_ids(right)
    if not left_ids and not right_ids:
        return None
    records = []
    for index, (l_ids, r_ids) in enumerate(zip(left_ids, right_ids, strict=False)):
        prefix = _common_prefix(l_ids, r_ids)
        records.append(
            {
                "record_index": index,
                "left_len": len(l_ids),
                "right_len": len(r_ids),
                "common_prefix_len": prefix,
                "first_left": None if prefix >= len(l_ids) else l_ids[prefix],
                "first_right": None if prefix >= len(r_ids) else r_ids[prefix],
            }
        )
    return {
        "left_record_count": len(left_ids),
        "right_record_count": len(right_ids),
        "records": records,
    }


def _load_margin_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    records: list[dict[str, Any]] = []
    for file in files:
        with file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    records.sort(
        key=lambda item: (
            int(item.get("decode_step", -1)),
            int(item.get("device", -1) if item.get("device") is not None else -1),
            int(item.get("pid", -1)),
        )
    )
    return records


def _float_diff(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))


def _compare_margin_lists(
    left_values: list[Any],
    right_values: list[Any],
) -> dict[str, Any]:
    diffs = []
    for left, right in zip(left_values, right_values, strict=False):
        if isinstance(left, list) and isinstance(right, list):
            diffs.extend(
                abs(float(a) - float(b)) for a, b in zip(left, right, strict=False)
            )
        else:
            diff = _float_diff(left, right)
            if diff is not None:
                diffs.append(diff)
    return {
        "compared_values": len(diffs),
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
    }


def _compare_margins(
    left_dir: Path | None,
    right_dir: Path | None,
) -> dict[str, Any] | None:
    left = _load_margin_records(left_dir)
    right = _load_margin_records(right_dir)
    if not left and not right:
        return None
    pairs = []
    first_token_diff = None
    max_top_value_diff = 0.0
    max_selected_value_diff = 0.0
    max_margin_diff = 0.0
    for index, (l_rec, r_rec) in enumerate(zip(left, right, strict=False)):
        selected_equal = l_rec.get("selected_tokens") == r_rec.get("selected_tokens")
        top_ids_equal = l_rec.get("top_ids") == r_rec.get("top_ids")
        top_value_diff = _compare_margin_lists(
            l_rec.get("top_values") or [], r_rec.get("top_values") or []
        )
        selected_value_diff = _compare_margin_lists(
            l_rec.get("selected_values") or [],
            r_rec.get("selected_values") or [],
        )
        margin_diff = _compare_margin_lists(
            l_rec.get("top1_top2_margins") or [],
            r_rec.get("top1_top2_margins") or [],
        )
        max_top_value_diff = max(
            max_top_value_diff, top_value_diff["max_abs_diff"] or 0
        )
        max_selected_value_diff = max(
            max_selected_value_diff, selected_value_diff["max_abs_diff"] or 0
        )
        max_margin_diff = max(max_margin_diff, margin_diff["max_abs_diff"] or 0)
        if first_token_diff is None and not selected_equal:
            first_token_diff = {
                "pair_index": index,
                "left_step": l_rec.get("decode_step"),
                "right_step": r_rec.get("decode_step"),
                "left_selected_tokens": l_rec.get("selected_tokens"),
                "right_selected_tokens": r_rec.get("selected_tokens"),
                "left_top_ids": l_rec.get("top_ids"),
                "right_top_ids": r_rec.get("top_ids"),
                "left_margins": l_rec.get("top1_top2_margins"),
                "right_margins": r_rec.get("top1_top2_margins"),
            }
        if index < 8 or not selected_equal or not top_ids_equal:
            pairs.append(
                {
                    "pair_index": index,
                    "left_step": l_rec.get("decode_step"),
                    "right_step": r_rec.get("decode_step"),
                    "selected_equal": selected_equal,
                    "top_ids_equal": top_ids_equal,
                    "top_value_diff": top_value_diff,
                    "selected_value_diff": selected_value_diff,
                    "margin_diff": margin_diff,
                }
            )
    return {
        "left_record_count": len(left),
        "right_record_count": len(right),
        "compared_record_count": min(len(left), len(right)),
        "first_selected_token_diff": first_token_diff,
        "max_top_value_abs_diff": max_top_value_diff,
        "max_selected_value_abs_diff": max_selected_value_diff,
        "max_margin_abs_diff": max_margin_diff,
        "sample_pairs": pairs[:32],
    }


def _tensor_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    left_f = left.float()
    right_f = right.float()
    diff = (left_f - right_f).abs()
    nonzero = diff > 0
    return {
        "shape_equal": True,
        "shape": list(left.shape),
        "dtype_left": str(left.dtype),
        "dtype_right": str(right.dtype),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "nonzero_count": int(nonzero.sum().item()) if diff.numel() else 0,
        "numel": int(diff.numel()),
    }


def _safe_load_tensor_payload(path: Path) -> dict[str, Any] | None:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - diagnostic tool.
        return {"load_error": str(exc)}


def _pid_ordinals(files: list[Path]) -> dict[int, int]:
    pids: set[int] = set()
    for file in files:
        match = _PID_RE.search(file.name)
        if match:
            pids.add(int(match.group("pid")))
    return {pid: index for index, pid in enumerate(sorted(pids))}


def _index_layer_dumps(
    path: Path | None,
) -> dict[tuple[int, int, int, str, str, str], Path]:
    if path is None or not path.exists():
        return {}
    files = sorted(path.glob("*.pt"))
    pid_to_worker = _pid_ordinals(files)
    result: dict[tuple[int, int, int, str, str, str], Path] = {}
    for file in files:
        match = _LAYER_DUMP_RE.search(file.name)
        if not match:
            continue
        payload = _safe_load_tensor_payload(file)
        if not payload or "load_error" in payload:
            continue
        pid = int(payload.get("pid", match.group("pid")))
        shape = payload.get("shape")
        if isinstance(shape, (tuple, list)):
            shape_key = "x".join(str(dim) for dim in shape)
        else:
            shape_key = match.group("shape")
        key = (
            pid_to_worker.get(pid, -1),
            int(match.group("step")),
            int(payload.get("layer_idx", match.group("layer"))),
            str(payload.get("layer_type", match.group("tail"))),
            str(payload.get("label", match.group("tail"))),
            shape_key,
        )
        result.setdefault(key, file)
    return result


def _compare_layer_dumps(
    left_dir: Path | None,
    right_dir: Path | None,
    max_rows: int,
) -> dict[str, Any] | None:
    left = _index_layer_dumps(left_dir)
    right = _index_layer_dumps(right_dir)
    if not left and not right:
        return None
    common_keys = sorted(set(left) & set(right))
    by_label: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "first_nonzero": None,
            "max_abs_diff": 0.0,
            "max_record": None,
        }
    )
    rows = []
    for key in common_keys:
        left_payload = _safe_load_tensor_payload(left[key])
        right_payload = _safe_load_tensor_payload(right[key])
        if not left_payload or not right_payload:
            continue
        if "load_error" in left_payload or "load_error" in right_payload:
            stats = {
                "load_error_left": left_payload.get("load_error"),
                "load_error_right": right_payload.get("load_error"),
            }
        else:
            stats = _tensor_stats(left_payload["tensor"], right_payload["tensor"])
        worker, step, layer, layer_type, label, shape = key
        record = {
            "worker": worker,
            "step": step,
            "layer": layer,
            "layer_type": layer_type,
            "label": label,
            "shape": shape,
            **stats,
        }
        label_key = f"{layer_type}:{label}"
        group = by_label[label_key]
        group["count"] += 1
        max_diff = float(stats.get("max_abs_diff") or 0.0)
        if max_diff > 0 and group["first_nonzero"] is None:
            group["first_nonzero"] = record
        if max_diff > group["max_abs_diff"]:
            group["max_abs_diff"] = max_diff
            group["max_record"] = record
        if len(rows) < max_rows or max_diff > 0:
            rows.append(record)
    rows.sort(
        key=lambda item: (
            item["worker"],
            item["step"],
            item["layer"],
            item["layer_type"],
            item["label"],
        )
    )
    by_label_out = dict(sorted(by_label.items()))
    first_nonzero = next(
        (row for row in rows if float(row.get("max_abs_diff") or 0.0) > 0),
        None,
    )
    return {
        "left_file_count": len(left),
        "right_file_count": len(right),
        "common_file_count": len(common_keys),
        "missing_left_count": len(set(right) - set(left)),
        "missing_right_count": len(set(left) - set(right)),
        "first_nonzero": first_nonzero,
        "by_label": by_label_out,
        "rows": rows[:max_rows],
    }


def _index_sample_dumps(path: Path | None) -> dict[tuple[int, int], Path]:
    if path is None or not path.exists():
        return {}
    files = sorted(path.glob("*.pt"))
    pid_to_worker = _pid_ordinals(files)
    result: dict[tuple[int, int], Path] = {}
    for file in files:
        match = _SAMPLE_DUMP_RE.search(file.name)
        if match:
            pid = int(match.group("pid"))
            key = (pid_to_worker.get(pid, -1), int(match.group("step")))
            result.setdefault(key, file)
    return result


def _compare_sample_dumps(
    left_dir: Path | None,
    right_dir: Path | None,
    max_rows: int,
) -> dict[str, Any] | None:
    left = _index_sample_dumps(left_dir)
    right = _index_sample_dumps(right_dir)
    if not left and not right:
        return None
    rows = []
    max_hidden = 0.0
    max_logits = 0.0
    first_nonzero = None
    for worker, step in sorted(set(left) & set(right)):
        key = (worker, step)
        left_payload = _safe_load_tensor_payload(left[key])
        right_payload = _safe_load_tensor_payload(right[key])
        if not left_payload or not right_payload:
            continue
        hidden_stats = _tensor_stats(
            left_payload["sample_hidden_states"],
            right_payload["sample_hidden_states"],
        )
        logits_stats = None
        if (
            left_payload.get("logits") is not None
            and right_payload.get("logits") is not None
        ):
            logits_stats = _tensor_stats(
                left_payload["logits"], right_payload["logits"]
            )
        hidden_diff = float(hidden_stats.get("max_abs_diff") or 0.0)
        logits_diff = float((logits_stats or {}).get("max_abs_diff") or 0.0)
        max_hidden = max(max_hidden, hidden_diff)
        max_logits = max(max_logits, logits_diff)
        record = {
            "worker": worker,
            "step": step,
            "hidden": hidden_stats,
            "logits": logits_stats,
            "left_metadata": _json_safe(left_payload.get("metadata", {})),
            "right_metadata": _json_safe(right_payload.get("metadata", {})),
        }
        if first_nonzero is None and (hidden_diff > 0 or logits_diff > 0):
            first_nonzero = record
        if len(rows) < max_rows or hidden_diff > 0 or logits_diff > 0:
            rows.append(record)
    return {
        "left_file_count": len(left),
        "right_file_count": len(right),
        "common_file_count": len(set(left) & set(right)),
        "first_nonzero": first_nonzero,
        "max_hidden_abs_diff": max_hidden,
        "max_logits_abs_diff": max_logits,
        "rows": rows[:max_rows],
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-tokens-json", type=Path)
    parser.add_argument("--right-tokens-json", type=Path)
    parser.add_argument("--left-margin-dir", type=Path)
    parser.add_argument("--right-margin-dir", type=Path)
    parser.add_argument("--left-layer-dir", type=Path)
    parser.add_argument("--right-layer-dir", type=Path)
    parser.add_argument("--left-sample-dir", type=Path)
    parser.add_argument("--right-sample-dir", type=Path)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = {
        "left_tokens_json": str(args.left_tokens_json)
        if args.left_tokens_json
        else None,
        "right_tokens_json": str(args.right_tokens_json)
        if args.right_tokens_json
        else None,
        "tokens": _compare_tokens(
            _load_json(args.left_tokens_json),
            _load_json(args.right_tokens_json),
        ),
        "margins": _compare_margins(args.left_margin_dir, args.right_margin_dir),
        "sample_tensors": _compare_sample_dumps(
            args.left_sample_dir,
            args.right_sample_dir,
            args.max_rows,
        ),
        "layer_tensors": _compare_layer_dumps(
            args.left_layer_dir,
            args.right_layer_dir,
            args.max_rows,
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
