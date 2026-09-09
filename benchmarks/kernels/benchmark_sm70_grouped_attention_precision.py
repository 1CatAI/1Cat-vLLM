# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject arithmetic candidates before timing when FP64-reference errors grow.

Independent native FP32-output builds expose the actual final accumulator.
Their partial workspaces and converted output must match the original builds;
an inaccurate or aliased diagnostic cannot authorize an arithmetic change.
This operator screen never provides model, recursion or acceptance admission.
"""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_sm70_grouped_attention_long import (
    byte_equal,
    call,
    load_operator,
    make_case,
)
from benchmarks.kernels.benchmark_sm70_grouped_attention_splits import (
    error,
    fp64_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "baseline-precast", "candidate-precast"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    assert torch.cuda.get_device_capability() == (7, 0)
    paths = [
        args.baseline,
        args.candidate,
        args.baseline_precast,
        args.candidate_precast,
    ]
    operators, manifests = zip(*(load_operator(p) for p in paths))
    assert len({id(op) for op in operators}) == 4
    assert all(m.get("splits", 80) == 80 for m in manifests)
    assert all(m["diagnostic_output_fp32"] for m in manifests[2:])
    report = {
        "manifests": manifests,
        "reference": "Independent FP64 QK, softmax and PV on decoded E4M3 operands",
        "scope": "Operator reference screen; no model or arithmetic admission",
        "arithmetic_admitted": False,
        "complete": False,
        "checks": [],
    }

    def save() -> None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for seed in (20260909, 20260910):
        torch.manual_seed(seed)
        for length in (129, 3297, 32768, 131072, 262144):
            case = make_case(8, 3296, length, 4, stride_padding=8 if seed % 2 else 0)
            if seed % 2:
                case["lengths"][-1] = 0
            for arm in (2, 3):
                guard = torch.full(
                    (10, 6, 256), -777.0, dtype=torch.float32, device="cuda"
                )
                case["arms"][arm]["guard"] = guard
                case["arms"][arm]["out"] = guard[1:-1]
            reference = fp64_reference(case, length)
            for arm, operator in enumerate(operators):
                call(case, arm, operator)
                assert torch.isfinite(case["arms"][arm]["out"]).all()
                assert (case["arms"][arm]["guard"][[0, -1]] == -777).all()
            for ordinary, precast in ((0, 2), (1, 3)):
                left, right = case["arms"][ordinary], case["arms"][precast]
                for key in ("partial", "lse"):
                    assert byte_equal(left[key], right[key]), (
                        "Invalid pre-cast diagnostic",
                        seed,
                        length,
                        ordinary,
                        key,
                    )
                assert byte_equal(left["out"], right["out"].half())
            metrics = [error(case["arms"][i]["out"], reference) for i in range(4)]
            nonexpansion = {
                name: all(metrics[b][key] <= metrics[a][key] for key in metrics[a])
                for name, a, b in (("fp16_output", 0, 1), ("precast_fp32", 2, 3))
            }
            row = {
                "seed": seed,
                "length": length,
                "padding_row": bool(seed % 2),
                "stride_padding": 8 if seed % 2 else 0,
                "precast_diagnostics_validated": True,
                "errors": dict(
                    zip(("baseline", "candidate", "base32", "new32"), metrics)
                ),
                "reference_error_nonexpansion": nonexpansion,
                "byte_equal": {
                    key: byte_equal(case["arms"][0][key], case["arms"][1][key])
                    for key in ("out", "partial", "lse")
                },
                "precast_byte_equal": byte_equal(
                    case["arms"][2]["out"], case["arms"][3]["out"]
                ),
            }
            report["checks"].append(row)
            save()
            print(json.dumps(row), flush=True)
            del case, reference
    report["all_reference_errors_nonexpanding"] = all(
        all(r["reference_error_nonexpansion"].values()) for r in report["checks"]
    )
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
