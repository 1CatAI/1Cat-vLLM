# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare FP16 GEMV on rotating checkpoint shared expert weights.

The four TP4 weight mappings are measured on one SM70 GPU. This is an
operator diagnostic, not a TP4 endpoint benchmark. --kernel-source explicitly
loads research Python for this diagnostic only; omit it for installed-wheel
validation. No model runtime or installed source is modified.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path)
    args = parser.parse_args()
    global Sm70Fp16GemvSiluKernel
    if args.kernel_source:
        spec = importlib.util.spec_from_file_location(
            "research_fp16_kernel", args.kernel_source
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Sm70Fp16GemvSiluKernel = module.Sm70Fp16GemvSiluKernel
    lock_fd = os.open("/tmp/gpu0-3.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
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
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    global_weights = []
    names = []
    for layer in range(16):
        weights = []
        for part in ("gate", "up", "down"):
            stem = f"model.language_model.layers.{layer}.mlp.shared_expert"
            name = f"{stem}.{part}_proj.weight"
            with safe_open(args.model / index[name], framework="pt") as f:
                weights.append(f.get_tensor(name).half())
            names.append(name)
        global_weights.append(weights)
    report = dict(
        complete=False,
        endpoint=False,
        research_source=args.kernel_source is not None,
        kernel_sha256=hashlib.sha256(args.kernel_source.read_bytes()).hexdigest()
        if args.kernel_source
        else None,
        dense_dtype="float16",
        accumulation="float32",
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        weight_names=names,
        cases=[],
    )
    for rank in range(4):
        for projection in ("gate_up", "down"):
            weights = []
            for gate, up, down in global_weights:
                width = gate.shape[0] // 4
                lo, hi = rank * width, (rank + 1) * width
                weight = (
                    torch.cat((gate[lo:hi], up[lo:hi]))
                    if projection == "gate_up"
                    else down[:, lo:hi].contiguous()
                )
                weights.append(weight.cuda())
            n, k = weights[0].shape
            torch.manual_seed(3800 + rank)
            xs = [
                torch.randn(1, k, device="cuda", dtype=torch.float16) for _ in weights
            ]
            outputs = [
                [torch.empty(1, n, device="cuda", dtype=torch.float16) for _ in weights]
                for _ in range(2)
            ]

            def launch(arm, x, w, out, n=n):
                if arm == 0:
                    torch.mm(x, w.t(), out=out)
                else:
                    Sm70Fp16GemvSiluKernel.apply_out(x, w, out, n, 0)

            checks = []
            for m in (1, 5, 7, 17, 33):
                for scale in (0.03, 1.0, 3.0):
                    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * scale
                    ref = (x.cpu().double() @ weights[0].cpu().double().t()).half()
                    errors = []
                    for arm in range(2):
                        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
                        launch(arm, x, weights[0], out)
                        actual = out.cpu()
                        torch.testing.assert_close(actual, ref, atol=0.002, rtol=0.002)
                        errors.append(float((actual.float() - ref.float()).abs().max()))
                    checks.append(dict(m=m, scale=scale, max_errors=errors))
            graphs = []
            for arm in range(2):
                for x, w, out in zip(xs, weights, outputs[arm]):
                    launch(arm, x, w, out)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):
                        for x, w, out in zip(xs, weights, outputs[arm]):
                            launch(arm, x, w, out)
                graphs.append(graph)
            times = [[], []]
            for repeat in range(5):
                for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                    for _ in range(3):
                        graphs[arm].replay()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    times[arm].append(start.elapsed_time(end) * 1000 / 256)
            case = dict(
                rank=rank,
                projection=projection,
                shape=[n, k],
                rotating_weight_bytes=sum(w.numel() * 2 for w in weights),
                checks=checks,
                c1_us=times,
                c1_median_us=[statistics.median(t) for t in times],
            )
            report["cases"].append(case)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(rank, projection, case["c1_median_us"], flush=True)
            del graphs, weights, outputs, xs
    report["complete"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
