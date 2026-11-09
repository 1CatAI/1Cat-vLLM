# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit DFlash2 real-request and PP-boundary tensor dumps."""

import argparse
import json
from pathlib import Path
from typing import Any

import regex as re
import torch
import torch.nn.functional as F

_PP_DUMP_RE = re.compile(
    r"pp_aux_(?P<step>\d+)_(?P<role>send|recv)_pp(?P<pp>\d+)_"
    r"tp(?P<tp>\d+)_pid\d+\.pt$"
)


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float().reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(values).all().item()),
        "sum": float(values.sum().item()),
        "sqsum": float(values.square().sum().item()),
        "absmax": float(values.abs().max().item()) if values.numel() else 0.0,
    }


def _compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    result = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
    }
    if actual.shape != expected.shape:
        result["shape_match"] = False
        return result

    result["shape_match"] = True
    result["equal"] = bool(torch.equal(actual, expected))
    actual_float = actual.float().reshape(-1)
    expected_float = expected.float().reshape(-1)
    diff = actual_float - expected_float
    result.update(
        {
            "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
            "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
            "cosine": (
                float(
                    F.cosine_similarity(
                        actual_float.unsqueeze(0), expected_float.unsqueeze(0)
                    ).item()
                )
                if diff.numel()
                else 1.0
            ),
        }
    )
    return result


def _load_pp_dumps(dump_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    dumps = []
    for path in sorted(dump_dir.glob("pp_aux_*.pt")):
        if _PP_DUMP_RE.match(path.name):
            dumps.append((path, _load(path)))
    return dumps


def _audit_pp_transport(dump_dir: Path) -> dict[str, Any]:
    dumps = _load_pp_dumps(dump_dir)
    senders = [(path, payload) for path, payload in dumps if payload["role"] == "send"]
    receivers = [
        (path, payload) for path, payload in dumps if payload["role"] == "recv"
    ]
    used_receivers: set[Path] = set()
    pairs = []
    unmatched_senders = []

    for send_path, sender in senders:
        match = None
        for recv_path, receiver in receivers:
            if recv_path in used_receivers or receiver["tp_rank"] != sender["tp_rank"]:
                continue
            if not torch.equal(receiver["positions"], sender["positions"]):
                continue
            if set(receiver["aux_hidden_states"]) != set(sender["aux_hidden_states"]):
                continue
            match = recv_path, receiver
            break
        if match is None:
            unmatched_senders.append(send_path.name)
            continue

        recv_path, receiver = match
        used_receivers.add(recv_path)
        layer_results = {
            str(boundary): _compare_tensors(
                receiver["aux_hidden_states"][boundary],
                sender["aux_hidden_states"][boundary],
            )
            for boundary in sender["aux_hidden_states"]
        }
        pairs.append(
            {
                "send": send_path.name,
                "recv": recv_path.name,
                "tp_rank": sender["tp_rank"],
                "positions_equal": True,
                "layers": layer_results,
                "all_equal": all(
                    item.get("equal", False) for item in layer_results.values()
                ),
            }
        )

    return {
        "pairs": pairs,
        "all_pairs_equal": bool(pairs) and all(pair["all_equal"] for pair in pairs),
        "unmatched_senders": unmatched_senders,
        "unmatched_receivers": [
            path.name for path, _ in receivers if path not in used_receivers
        ],
    }


def _greedy_selector_path(payload: dict[str, Any]) -> list[dict[str, Any]]:
    dflash2 = payload.get("dflash2") or {}
    candidate_ids = dflash2.get("candidate_ids")
    scores = dflash2.get("lattice_scores")
    if candidate_ids is None or scores is None or candidate_ids.shape[0] == 0:
        return []

    path = []
    predecessor = 0
    for step in range(candidate_ids.shape[1]):
        row = scores[0, step, predecessor]
        successor = int(row.argmax().item())
        path.append(
            {
                "step": step,
                "predecessor": predecessor,
                "successor": successor,
                "token_id": int(candidate_ids[0, step, successor].item()),
                "score": float(row[successor].item()),
            }
        )
        predecessor = successor
    return path


def _summarize_proposals(dump_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(dump_dir.glob("proposal_*.pt")):
        payload = _load(path)
        aux = payload.get("aux_hidden_states") or []
        dflash2 = payload.get("dflash2") or {}
        summaries.append(
            {
                "file": path.name,
                "request_ids": payload["request_ids"],
                "target_input_ids": payload["target_input_ids"].tolist(),
                "target_positions": payload["target_positions"].tolist(),
                "draft_input_ids": payload["draft_input_ids"].tolist(),
                "draft_positions": payload["draft_positions"].tolist(),
                "context_positions": payload["context_positions"].tolist(),
                "draft_tokens": payload["draft_tokens"].tolist(),
                "aux_hidden_states": [
                    {"boundary_index": index, **_tensor_stats(tensor)}
                    for index, tensor in enumerate(aux)
                ],
                "projected_hidden_states": _tensor_stats(
                    payload["projected_hidden_states"]
                ),
                "draft_input_embeds": _tensor_stats(payload["draft_input_embeds"]),
                "backbone_hidden_states": (
                    _tensor_stats(dflash2["backbone_hidden_states"])
                    if "backbone_hidden_states" in dflash2
                    else None
                ),
                "candidate_ids": (
                    dflash2["candidate_ids"].tolist()
                    if "candidate_ids" in dflash2
                    else None
                ),
                "greedy_selector_path": _greedy_selector_path(payload),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = {
        "dump_dir": str(args.dump_dir.resolve()),
        "pp_transport": _audit_pp_transport(args.dump_dir),
        "proposals": _summarize_proposals(args.dump_dir),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
