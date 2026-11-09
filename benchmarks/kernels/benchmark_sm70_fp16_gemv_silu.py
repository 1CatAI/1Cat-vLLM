# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare installed HC down kernels on rotating checkpoint FP16 weights.

Run from an installed candidate wheel. The four rank mappings are measured on
one SM70 GPU; this is an operator benchmark, not a TP4 endpoint measurement.
All GPU work holds /tmp/gpu0-3.lock. Whole-model quality remains a separate gate.
"""

import argparse
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

import vllm
from vllm.model_executor.kernels.linear.fp16_gemv_silu import Sm70Fp16GemvSiluKernel
from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import (
    _qwen38_hc_down_local_shard_kernel,
)


def check_exclusive() -> None:
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
        raise RuntimeError(f"GPU 0-3 have other consumers: {sorted(others)}")


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def elapsed(graph, matrices, repeats=128):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (repeats * matrices)


def run(args):
    check_exclusive()
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("Requires SM70")
    torch.set_num_threads(1)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights, names = [], []
    for layer in range(16):
        stem = f"model.language_model.layers.{layer}.attn_hyper_connection"
        tensors = []
        for part in ("input_mix_weight_down.weight", "block_inject_weight.weight"):
            name = f"{stem}.{part}"
            with safe_open(args.model / index[name], framework="pt") as f:
                tensors.append(f.get_tensor(name).half().cuda())
            names.append(name)
        weights.append(torch.cat(tensors).contiguous())
    helper = Path(vllm.__file__).parent / (
        "model_executor/kernels/linear/fp16_gemv_silu.py"
    )
    report = {
        "complete": False,
        "endpoint": False,
        "runtime": vllm.__version__,
        "runtime_path": vllm.__file__,
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "weight_names": names,
        "rotating_weight_bytes": sum(w.numel() * w.element_size() for w in weights),
        "arms": ["existing HC down shard", "generic GEMV/SiLU capability policy"],
        "cases": [],
    }

    def save():
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    for rank in range(4):
        torch.manual_seed(3800 + rank)
        xs = [
            torch.randn(1, 10240, device="cuda", dtype=torch.float16) for _ in weights
        ]
        outputs = [[x.new_empty(1, 88) for x in xs] for _ in range(2)]

        def launch(arm, xs=xs, outputs=outputs, rank=rank):
            for x, weight, out in zip(xs, weights, outputs[arm], strict=True):
                if arm == 0:
                    _qwen38_hc_down_local_shard_kernel[(88,)](
                        x, weight, out, TP_RANK=rank, num_warps=4
                    )
                else:
                    Sm70Fp16GemvSiluKernel.apply_out(
                        x, weight, out, 81, 80, rank * 80, 320 + rank, 4.0
                    )

        graphs = [
            capture(lambda arm=arm, launch=launch: launch(arm)) for arm in range(2)
        ]
        selected = torch.cat(
            [
                torch.arange(rank * 80, (rank + 1) * 80, device="cuda"),
                torch.tensor([320 + rank], device="cuda"),
            ]
        )
        checks = []
        for scale in (0, 0.03, 1, 3):
            for x in xs:
                x.normal_(0, scale)
            for arm in outputs:
                for out in arm:
                    out.fill_(float("nan"))
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            errors = [[], []]
            for i, (x, weight) in enumerate(zip(xs, weights, strict=True)):
                ref = (x.double() @ weight[selected].double().T).half().float()
                ref[:, :80] = torch.nn.functional.silu(ref[:, :80] / 4)
                ref = ref.half().float()
                for arm in range(2):
                    out = outputs[arm][i]
                    actual = out[:, :81].float()
                    relative = float(
                        torch.linalg.vector_norm(actual - ref)
                        / torch.linalg.vector_norm(ref).clamp_min(1e-12)
                    )
                    if not torch.isfinite(out).all() or relative > 0.002:
                        raise RuntimeError(f"FP64 oracle failed: {rank=} {scale=}")
                    if torch.count_nonzero(out[:, 81:]):
                        raise RuntimeError("Output padding was not cleared")
                    errors[arm].append(relative)
            checks.append(
                {"scale": scale, "max_relative_l2_vs_fp64": list(map(max, errors))}
            )
        for x in xs:
            x.normal_()
        samples = [[], []]
        for trial in range(6):
            for offset in range(2):
                arm = (trial + offset) % 2
                samples[arm].append(elapsed(graphs[arm], len(weights)))
        case = {
            "tp_rank_mapping": rank,
            "median_us": list(map(statistics.median, samples)),
            "samples_us": samples,
            "checks": checks,
        }
        report["cases"].append(case)
        save()
        print(rank, case["median_us"], flush=True)
    check_exclusive()
    report["complete"] = True
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open("/tmp/gpu0-3.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(args)
