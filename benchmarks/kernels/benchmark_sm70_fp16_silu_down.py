# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold C1 SiLU/down comparison on checkpoint shared-expert matrices."""

import argparse
import fcntl
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.fp16_silu_down import apply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--inputs", type=Path, help="Optional retained linear snapshots"
    )
    args = parser.parse_args()
    if not 0 <= args.rank < args.tensor_parallel_size:
        parser.error("rank must belong to the tensor-parallel group")
    with open("/tmp/gpu0-3.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pids = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0,1,2,3",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        others = {int(p) for p in pids.splitlines() if p.strip()} - {os.getpid()}
        if others:
            raise RuntimeError(f"Other GPU users: {sorted(others)}")
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability() != (7, 0):
            raise RuntimeError("This benchmark requires SM70")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        index = json.loads((args.model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        names = sorted(
            name for name in index if name.endswith(".shared_expert.down_proj.weight")
        )
        inputs, weights = [], []
        for name in names:
            tensors = []
            for part in ("gate", "up", "down"):
                key = name.replace("down_proj.weight", f"{part}_proj.weight")
                with safe_open(args.model / index[key], framework="pt") as checkpoint:
                    tensors.append(checkpoint.get_tensor(key).half())
            gate, up, down = tensors
            if gate.shape[0] % args.tensor_parallel_size:
                raise ValueError("Shared intermediate size is not divisible by TP")
            width = gate.shape[0] // args.tensor_parallel_size
            lo, hi = args.rank * width, (args.rank + 1) * width
            projection = torch.cat((gate[lo:hi], up[lo:hi])).cuda()
            weight = down[:, lo:hi].contiguous().cuda()
            k = projection.shape[1]
            if args.inputs:
                layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
                snapshot = torch.load(
                    args.inputs / f"rank{args.rank}_layer{layer:02d}_gate_up_proj.pt",
                    weights_only=True,
                    map_location="cpu",
                )
                x = snapshot["input"].cuda()
            else:
                x = (((torch.arange(k, device="cuda") % 23) - 11).half() / 16)[None, :]
            inputs.append(torch.nn.functional.linear(x, projection))
            weights.append(weight)
        if not weights:
            raise ValueError("Checkpoint has no shared-expert matrices")

        def reference(x, weight):
            activation = x.new_empty((1, weight.shape[1]))
            ops.silu_and_mul(activation, x)
            return torch.nn.functional.linear(activation, weight)

        checks = []
        for name, x, weight in zip(names, inputs, weights):
            expected, actual = reference(x, weight), apply(x, weight)
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
            checks.append(
                dict(
                    name=name,
                    shape=list(weight.shape),
                    weight_bytes=weight.numel() * 2,
                    unequal=int((actual != expected).sum()),
                    max_error=float((actual - expected).abs().max()),
                )
            )
        total_bytes = sum(weight.numel() * 2 for weight in weights)
        flush = (
            torch.zeros(4 * 1024 * 1024, device="cuda", dtype=torch.int32)
            if total_bytes <= 16 * 1024 * 1024
            else None
        )
        graphs = []
        iterations = 8
        for fused in (False, True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(iterations):
                    for x, weight in zip(inputs, weights):
                        if flush is not None:
                            flush.add_(1)
                        apply(x, weight) if fused else reference(x, weight)
            graphs.append(graph)
        timings: list[list[float]] = [[], []]
        for repeat in range(9):
            for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                graphs[arm].replay()
                end.record()
                end.synchronize()
                timings[arm].append(
                    start.elapsed_time(end) * 1000 / (iterations * len(weights))
                )
        report = dict(
            endpoint=False,
            dense_dtype="float16",
            accumulation="float32",
            m=1,
            gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            tp=args.tensor_parallel_size,
            rank=args.rank,
            real_captured_inputs=args.inputs is not None,
            rotating_weight_bytes=total_bytes,
            includes_flush=flush is not None,
            checks=checks,
            timings=timings,
            median_us=[statistics.median(t) for t in timings],
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("checks", "timings")}
            )
        )
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
