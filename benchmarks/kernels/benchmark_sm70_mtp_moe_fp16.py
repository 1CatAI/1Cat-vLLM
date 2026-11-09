# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact Flash-Next draft projections using the normal source-built extension.

Checkpoint weights cover four TP slices. Activations and routes are synthetic;
times measure individual projections, not complete draft or model rounds.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import torch
from safetensors import safe_open

from benchmarks.kernels.benchmark_sm70_moe_packed_w13 import graph, latency
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)
from vllm.triton_utils import tl


def checkpoint_weight(model: Path, rank: int, down: bool):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    suffix = "down_proj" if down else "gate_up_proj"
    key = next(k for k in index if k.endswith("mtp.layers.0.mlp.experts." + suffix))
    with safe_open(model / index[key], framework="pt", device="cpu") as handle:
        value = handle.get_slice(key)
        lo = rank * 160
        weight = (
            value[:, :, lo : lo + 160].contiguous()
            if down
            else torch.cat(
                [value[:, lo : lo + 160, :], value[:, 640 + lo : 800 + lo, :]], dim=1
            )
        )
    return weight.half().to("cuda")


def measure(weight, m, down):
    n, k = weight.shape[-2:]
    x = torch.randn(m * 10 if down else m, k, device="cuda", dtype=torch.float16)
    ids = torch.empty(m * 10, device="cuda", dtype=torch.int32).random_(0, 512)
    weights = torch.softmax(torch.randn(m, 10, device="cuda"), -1)
    padded = torch.tensor([m * 20], device="cuda", dtype=torch.int32)
    outputs = [x.new_empty(m, 10, n) for _ in range(2)]
    config = dict(
        BLOCK_SIZE_M=2,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1,
        SPLIT_K=1,
        num_warps=4,
        num_stages=3,
    )

    def control():
        invoke_fused_moe_triton_kernel(
            x,
            weight,
            outputs[0],
            None,
            None,
            weights,
            None,
            ids,
            padded,
            down,
            1 if down else 10,
            config,
            tl.float16,
            False,
            False,
            False,
            False,
            False,
        )

    def candidate():
        torch.ops._C.sm70_mtp_moe_fp16_out(
            outputs[1], x, weight, ids, weights, padded, down
        )

    graphs = [graph(fn, unroll=8) for fn in (control, candidate)]
    mismatches = []
    for scale in (0.0, 0.001, 0.03, 0.1, 1.0, 3.0):
        x.normal_(0, scale)
        ids.random_(0, 512)
        ids[0] = -1
        weights.copy_(torch.softmax(torch.randn_like(weights), -1))
        for output in outputs:
            output.fill_(float("nan"))
        for capture in graphs:
            capture.replay()
        mismatches.append(
            int((outputs[0].view(torch.int16) != outputs[1].view(torch.int16)).sum())
        )
    if any(mismatches):
        raise AssertionError(dict(m=m, down=down, mismatches=mismatches))
    # Time all ten valid routes after checking the invalid-expert path.
    ids.random_(0, 512)
    samples = [[], []]
    for trial in range(5):
        for arm in (0, 1) if trial % 2 == 0 else (1, 0):
            samples[arm].append(latency(graphs[arm], repeats=20, unroll=8))
    return dict(
        m=m,
        down=down,
        mismatches=mismatches,
        control_us=median(samples[0]),
        candidate_us=median(samples[1]),
        samples_us=samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ranks", type=int, nargs="+", choices=range(4), default=range(4)
    )
    args = parser.parse_args()
    if not hasattr(torch.ops._C, "sm70_mtp_moe_fp16_out"):
        parser.error("Build this source tree's normal native extension first")
    torch.set_num_threads(1)
    torch.manual_seed(20260927)
    native = Path(__file__).resolve().parents[2] / "vllm/_C.abi3.so"
    report = dict(
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        real_checkpoint_weights=True,
        synthetic_activations_and_routes=True,
        rows=[],
        complete=False,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for rank in args.ranks:
        for down in (False, True):
            weight = checkpoint_weight(args.model, rank, down)
            for m in (1, 5):
                row = dict(rank=rank, **measure(weight, m, down))
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n")
            del weight
    report["complete"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
