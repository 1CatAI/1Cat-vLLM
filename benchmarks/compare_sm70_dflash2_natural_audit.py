# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locate the first observed target/proposal difference with real acceptance.

This is a diagnostic comparison, not a quality or acceptance noninferiority
gate. Inputs after the first differing proposal need not be the same. A first
observed state difference still requires an operator-level causality check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.compare_sm70_dflash2_state_audit import (
    sampling_difference,
    tensor_difference,
)


def _load(directory: Path) -> dict:
    records = {}
    for path in directory.glob("*-rank*-step*.pt"):
        row = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        case, rank, step = (row[k] for k in ("case", "rank", "step"))
        key = case, step, rank
        if key in records or row.get("control") != "natural_sampling":
            raise ValueError(f"{path}: duplicate or non-natural observation")
        if row["phase"] != ("prefill" if step == 0 else "verify"):
            raise ValueError(f"{path}: incorrect phase")
        if not row.get("expected_layers") or "capture_epoch" not in row:
            raise ValueError(f"{path}: missing snapshot provenance")
        if rank == 0 and not torch.isfinite(row["native_logits"]).all():
            raise ValueError(f"{path}: nonfinite native logits")
        for field in ("aux_hidden_states", "sampling", "states"):
            if not row.get(field):
                raise ValueError(f"{path}: missing {field}")
        if row["num_sampled"].numel() != 1 or row["num_rejected"].numel() != 1:
            raise ValueError(f"{path}: expected B1 counts")
        n = int(row["num_sampled"].item())
        rejected = int(row["num_rejected"].item())
        if not 1 <= n <= 8 or not 0 <= rejected <= 7:
            raise ValueError(f"{path}: invalid acceptance counts")
        if n > row["sampled_token_ids"].shape[1]:
            raise ValueError(f"{path}: missing accepted output IDs")
        proposal = directory / f"proposal-{case}-tp{rank}-forward{step}.pt"
        if not proposal.exists():
            raise ValueError(f"{path}: missing proposal observation")
        draft = torch.load(proposal, map_location="cpu", weights_only=True, mmap=True)
        if (draft["case"], draft["step"], draft["rank"]) != key:
            raise ValueError(f"{proposal}: proposal identity differs")
        if "draft_tokens" not in draft or "projected_context" not in draft:
            raise ValueError(f"{proposal}: incomplete proposal")
        records[key] = row, draft
    if not records:
        raise ValueError(f"{directory}: empty captures")
    for case in {k[0] for k in records}:
        steps = {k[1] for k in records if k[0] == case}
        if len(steps) < 2 or steps != set(range(max(steps) + 1)):
            raise ValueError(f"{case}: missing prefill or verifier steps")
        for step in steps:
            if {k[2] for k in records if k[:2] == (case, step)} != {0, 1, 2, 3}:
                raise ValueError(f"{case}/{step}: every step requires four TP ranks")
    return records


def _target_tensors(row: dict) -> dict[str, torch.Tensor]:
    result = {
        k: row[k]
        for k in ("positions", "input_ids", "hidden", "num_sampled", "num_rejected")
    }
    for k in ("native_logits", "draft_logits"):
        if row.get(k) is not None:
            result[k] = row[k]
    for k in ("sampling", "states"):
        result.update({f"{k}/{label}": t for label, t in row[k].items()})
    result.update({f"aux/{i}": t for i, t in enumerate(row["aux_hidden_states"])})
    result.update(
        {
            f"layer/{v['layer_idx']}/{v['label']}": v["tensor"]
            for v in row["tensors"].values()
        }
    )
    # Unwritten output padding is not an emitted or accepted token.
    result["accepted_output"] = row["sampled_token_ids"][:, : row["num_sampled"].item()]
    return result


def _proposal_tensors(row: dict) -> dict[str, torch.Tensor]:
    values = {
        k: v
        for k, v in row.items()
        if isinstance(v, torch.Tensor) and k != "idx_mapping"
    }
    if "idx_mapping" in row and row.get("sampling_layout") != "request_gathered_v1":
        # Early captures retained complete arrays indexed by request slot.
        indices = row["idx_mapping"].to(torch.int64)
        for name in ("temperature", "seeds"):
            if name in values:
                values[name] = values[name].index_select(0, indices)
    return values


def compare_natural(left_dir: Path, right_dir: Path) -> dict:
    left, right = _load(left_dir), _load(right_dir)
    cases = {k[0] for k in left}
    if cases != {k[0] for k in right}:
        raise ValueError("Case coverage differs")
    result = {"left": str(left_dir), "right": str(right_dir), "cases": []}
    for case in sorted(cases):
        lengths = [len({k[1] for k in arm if k[0] == case}) for arm in (left, right)]
        first = None
        mappings = []
        for step in range(min(lengths)):
            for phase_index, phase in enumerate(("target", "proposal")):
                differences = []
                for rank in range(4):
                    rows = [arm[case, step, rank][phase_index] for arm in (left, right)]
                    values = [
                        _target_tensors(row)
                        if phase == "target"
                        else _proposal_tensors(row)
                        for row in rows
                    ]
                    if phase == "proposal" and "idx_mapping" in rows[0]:
                        slots = [row["idx_mapping"].tolist() for row in rows]
                        if slots[0] != slots[1]:
                            mappings.append(
                                {"step": step, "rank": rank, "slots": slots}
                            )
                    if values[0].keys() != values[1].keys():
                        raise ValueError(
                            f"{case}/{step}/{rank}: tensor coverage differs"
                        )
                    for name in sorted(values[0]):
                        a, b = (v[name] for v in values)
                        if a.shape != b.shape or a.dtype != b.dtype:
                            diff = {
                                "contract_changed": [
                                    str(a.shape),
                                    str(b.shape),
                                    str(a.dtype),
                                    str(b.dtype),
                                ]
                            }
                        else:
                            diff = tensor_difference(a, b)
                            if diff["bitwise_equal"]:
                                continue
                            if name == "native_logits":
                                diff.update(sampling_difference(a, b))
                        differences.append({"rank": rank, "name": name, **diff})
                if differences:
                    first = {"step": step, "phase": phase, "differences": differences}
                    break
            if first is not None:
                break
        result["cases"].append(
            {
                "case": case,
                "steps_per_arm": lengths,
                "first_observed_difference": first,
                "different_request_slot_mappings": mappings,
                "all_logical_tensors_equal": first is None and lengths[0] == lengths[1],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = compare_natural(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        first = case["first_observed_difference"]
        print(
            case["case"], "equal" if first is None else (first["step"], first["phase"])
        )


if __name__ == "__main__":
    main()
