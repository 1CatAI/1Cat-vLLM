# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired MTP5 expert chains, with checkpoint weights and changing routes.

The installed source-built operators are required. Hidden states are synthetic;
optional diagnostic route snapshots retain their real IDs and weights. Reported
times cover grouping, W13/SwiGLU, W2 and ordered reduction, not a model round.
"""

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from benchmarks.kernels.benchmark_qwen38_nvfp4_qpn_mtp5 import mtp_weighted_reduce
from benchmarks.kernels.benchmark_sm70_moe_packed_w13 import (
    checkpoint_weights,
    graph,
    latency,
)
from vllm import _sm70_ops as ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(4), default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--routes-dir", type=Path)
    args = parser.parse_args()
    if not ops.has_nvfp4_grouped_batch_reduce_dispatch():
        parser.error("Build this source tree's normal native extension first")
    torch.set_num_threads(1)
    torch.manual_seed(20260927 + args.rank)
    w13, s13, w2, s2 = checkpoint_weights(args.model, args.layer, args.rank, True)
    x = torch.randn(5, 2560, device="cuda", dtype=torch.float16)
    ids = torch.arange(50, device="cuda", dtype=torch.int32)
    weights = torch.softmax(torch.randn(5, 10, device="cuda"), -1)
    mids = [x.new_empty(50, 160) for _ in range(2)]
    outputs = [torch.empty_like(x) for _ in range(2)]
    gate_up = x.new_empty(50, 320)
    routed = x.new_empty(50, 2560)
    scratch = torch.empty_like(routed)
    rows = torch.empty(50, 8, device="cuda", dtype=torch.int32)
    experts = torch.empty_like(ids)
    sizes = torch.empty_like(ids)
    total = ids.new_empty(1)

    def control():
        ops.nvfp4_moe_qpn_mtp5_sm70_out(gate_up, x, w13, s13, ids, True, 4)
        torch.ops._C.silu_and_mul_interleaved(mids[0], gate_up)
        ops.nvfp4_moe_qpn_mtp5_sm70_out(routed, mids[0], w2, s2, ids, False, 1)
        mtp_weighted_reduce(routed, weights, outputs[0])

    def candidate():
        ops.nvfp4_grouped_w13_sm70_out(
            mids[1], x, w13, s13, ids, rows, experts, sizes, total, 4, True
        )
        ops.nvfp4_grouped_w2_batch_reduce_sm70_out(
            outputs[1],
            scratch,
            mids[1],
            w2,
            s2,
            weights,
            rows,
            experts,
            sizes,
            total,
        )

    graphs = [graph(fn) for fn in (control, candidate)]
    cases = []
    if args.routes_dir:
        for path in sorted(args.routes_dir.glob("*.pt")):
            snapshot = torch.load(path, map_location="cpu", weights_only=False)
            captured = {
                row["label"]: row["tensor"]
                for row in snapshot
                if row["layer_idx"] == args.layer
            }
            cases.append(
                (path.stem, captured["moe_topk_ids"], captured["moe_topk_weights"])
            )
    else:
        for unique in (50, 35, 20, 10):
            cases.append((f"unique{unique}", torch.arange(50) % unique, None))
        invalid = torch.arange(50, dtype=torch.int32)
        invalid[::3] = -1
        invalid[1::7] = 512
        cases.append(("invalid_expert_slots", invalid, None))
    report = dict(
        layer=args.layer,
        tp_weight_rank=args.rank,
        synthetic_hidden_states=True,
        real_routes=args.routes_dir is not None,
        results=[],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for label, route, captured_weights in cases:
        ids.copy_(route.flatten())
        mismatches = []
        for scale in (0.0, 0.001, 0.1, 1.0, 3.0):
            x.normal_(0, scale)
            weights.copy_(
                captured_weights
                if captured_weights is not None
                else torch.softmax(torch.randn_like(weights), -1)
            )
            for out in (*mids, *outputs):
                out.fill_(float("nan"))
            for metadata in (rows, experts, sizes, total):
                metadata.fill_(-123)
            for capture in graphs:
                capture.replay()
            mismatches.append(
                [
                    int(
                        (
                            pair[0].view(torch.int16) != pair[1].view(torch.int16)
                        ).count_nonzero()
                    )
                    for pair in (mids, outputs)
                ]
            )
        record = dict(case=label, mismatches=mismatches)
        report["results"].append(record)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        if any(value for pair in mismatches for value in pair):
            raise AssertionError(f"Changed-input/route bit mismatch: {record}")
        samples = [[], []]
        for repeat in range(7):
            for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                samples[arm].append(latency(graphs[arm], 20))
        record.update(
            control_us=median(samples[0]),
            candidate_us=median(samples[1]),
            samples_us=samples,
        )
        print(json.dumps(record), flush=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
