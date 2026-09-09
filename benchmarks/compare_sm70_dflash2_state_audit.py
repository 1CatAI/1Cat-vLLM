# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete four-rank StateAuditExtension captures, failing on gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(
            f"Tensor contract differs: {left.shape}/{left.dtype}, "
            f"{right.shape}/{right.dtype}"
        )

    def raw_bytes(tensor):
        flat = tensor.contiguous().reshape(-1)
        # A size-one tensor can be "contiguous" with stride 8. Reinterpret
        # only its logical storage, not padding or neighboring metadata rows.
        return flat.as_strided((flat.numel(),), (1,)).view(torch.uint8)

    byte_equal = torch.equal(raw_bytes(left), raw_bytes(right))
    if byte_equal:
        return {"bitwise_equal": True}
    difference = (left.double() - right.double()).abs()
    return {
        "bitwise_equal": False,
        "different_elements": int((left != right).sum()),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "left_nonfinite": int((~torch.isfinite(left)).sum()),
        "right_nonfinite": int((~torch.isfinite(right)).sum()),
    }


def sampling_difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    if left.shape != right.shape:
        raise ValueError("Logit shapes differ")
    count = left.shape[0]

    def probabilities(logits):
        return apply_top_k_top_p_pytorch(
            logits.float().clone(),
            torch.full((count,), 20, dtype=torch.int32),
            torch.full((count,), 0.95),
        ).softmax(-1)

    p, q = probabilities(left), probabilities(right)
    full_tv = (left.float().softmax(-1) - right.float().softmax(-1)).abs().sum(-1) / 2
    return {
        "full_softmax_tv": full_tv.tolist(),
        "sampling_tv": ((p - q).abs().sum(-1) / 2).tolist(),
        "support_changed": ((p > 0) != (q > 0)).any(-1).tolist(),
        "top1_changed": (left.argmax(-1) != right.argmax(-1)).tolist(),
    }


def compare(
    left_dir: Path, right_dir: Path, *, right_verifier_route: str | None = None
) -> dict:
    left_files = {p.name: p for p in left_dir.glob("*-rank*-step*.pt")}
    right_files = {p.name: p for p in right_dir.glob("*-rank*-step*.pt")}
    if not left_files or left_files.keys() != right_files.keys():
        raise ValueError("Missing or mismatched capture files")
    result = {
        "left": str(left_dir),
        "right": str(right_dir),
        "comparisons": [],
        "logits": [],
    }
    coverage: dict[tuple[str, int], set[int]] = {}
    seen_states: dict[tuple[str, int, str], set[str]] = {}
    for name in sorted(left_files):
        left = torch.load(left_files[name], map_location="cpu", weights_only=True)
        right = torch.load(right_files[name], map_location="cpu", weights_only=True)
        for key in ("case", "rank", "step", "phase", "num_draft_tokens"):
            if left[key] != right[key]:
                raise ValueError(f"{name}: {key} differs")
        identity = {key: left[key] for key in ("case", "rank", "step", "phase")}
        if right_verifier_route is not None and right["phase"] == "verify":
            expected = {
                f"route/verify/layer{layer}/{right_verifier_route}"
                for layer in right["expected_layers"]
            }
            if set(right.get("verifier_routes", ())) != expected:
                raise ValueError(f"{name}: missing {right_verifier_route} route hit")
        if left.get("expected_layers") != right.get("expected_layers"):
            raise ValueError(f"{name}: requested layers differ")
        if not left.get("expected_layers") or "capture_epoch" not in left:
            raise ValueError(f"{name}: missing current-forward snapshot provenance")
        for side in (left, right):
            for layer in side["expected_layers"]:
                prefix = f"{side['phase']}/layer{layer}"
                for required in (
                    "/conv/input:",
                    "/conv/output:",
                    "/recurrent/q:",
                    "/recurrent/input_state",
                    "/recurrent/output:",
                ):
                    if not any(k.startswith(prefix + required) for k in side["states"]):
                        raise ValueError(f"{name}: missing {prefix + required}")
        coverage.setdefault((left["case"], left["step"]), set()).add(left["rank"])
        for key in ("positions", "input_ids"):
            if not torch.equal(left[key], right[key]):
                raise ValueError(f"{name}: forced {key} differs")
        groups = {
            "boundary": ({"hidden": left["hidden"]}, {"hidden": right["hidden"]}),
            "state": (left["states"], right["states"]),
            "layer": tuple(
                {
                    f"layer{v['layer_idx']}/{v['label']}": v["tensor"]
                    for v in d["tensors"].values()
                }
                for d in (left, right)
            ),
            "sampling": (left["sampling"], right["sampling"]),
        }
        seen_states.setdefault(
            (left["case"], left["rank"], left["phase"]), set()
        ).update(left["states"])
        for group, (lvalues, rvalues) in groups.items():
            if lvalues.keys() != rvalues.keys():
                raise ValueError(f"{name}: {group} snapshot coverage differs")
            for label in sorted(lvalues):
                difference = tensor_difference(lvalues[label], rvalues[label])
                if not difference["bitwise_equal"]:
                    result["comparisons"].append(
                        {**identity, "group": group, "label": label, **difference}
                    )
        if left["rank"] == 0:
            if not all(torch.isfinite(d["native_logits"]).all() for d in (left, right)):
                raise ValueError(f"{name}: nonfinite native logits")
            result["logits"].append(
                {
                    **identity,
                    "positions": left["positions"].tolist(),
                    **tensor_difference(left["native_logits"], right["native_logits"]),
                    **sampling_difference(
                        left["native_logits"], right["native_logits"]
                    ),
                }
            )
    if any(ranks != {0, 1, 2, 3} for ranks in coverage.values()):
        raise ValueError("Every step requires four TP ranks")
    for key, labels in seen_states.items():
        for required in (
            "/conv/input",
            "/conv/output",
            "/recurrent/input_state",
            "/recurrent/output",
        ):
            if not any(required in label for label in labels):
                raise ValueError(f"{key}: missing {required} evidence")
    for case in {key[0] for key in coverage}:
        steps = {key[1] for key in coverage if key[0] == case}
        if steps != set(range(max(steps) + 1)) or len(steps) < 2:
            raise ValueError(f"{case}: missing prefill or verifier steps")
        for rank in range(4):
            if any(
                (case, rank, phase) not in seen_states
                for phase in ("prefill", "verify")
            ):
                raise ValueError(f"{case}/rank{rank}: missing prefill/verifier states")
    result["summary"] = {
        "files_per_arm": len(left_files),
        "differing_intermediates": len(result["comparisons"]),
        "max_sampling_tv": max(max(row["sampling_tv"]) for row in result["logits"]),
        "support_changed_rows": sum(
            sum(row["support_changed"]) for row in result["logits"]
        ),
        "top1_changed_rows": sum(sum(row["top1_changed"]) for row in result["logits"]),
        "all_logits_bitwise_equal": all(
            row["bitwise_equal"] for row in result["logits"]
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--right-verifier-route", choices=("split", "packed"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = compare(
        args.left, args.right, right_verifier_route=args.right_verifier_route
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
