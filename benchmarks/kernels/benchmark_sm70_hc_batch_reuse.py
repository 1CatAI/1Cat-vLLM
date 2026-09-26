# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight, high-precision HC batch kernel screen; NOT engine throughput.

The extension is built from this tree into a task-owned JIT directory. It is
research-only: a winning kernel still needs runtime integration, TP4 and
full-engine quality/performance validation before any default changes.
"""

import argparse
import hashlib
import itertools
import json
import os
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors import safe_open
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]


@triton.jit
def mix_reference(X, G, Y, H: tl.constexpr, OFFSET: tl.constexpr):
    # Same arithmetic/FP16 boundaries as ops/hc.py::_hc_gate_mix_kernel.
    # The only generalization is selecting a TP output shard from X.
    row = tl.program_id(0)
    col = tl.program_id(1) * 512 + tl.arange(0, 512)
    acc = tl.zeros((512,), tl.float32)
    for branch in tl.static_range(4):
        gate = tl.load(G + row * 4 * H + branch * H + col, col < H, 0)
        x = tl.load(X + row * 10240 + branch * 2560 + OFFSET + col, col < H, 0)
        acc += tl.sigmoid(gate.to(tl.float32)) * x.to(tl.float32)
    tl.store(Y + row * H + col, acc / 4, col < H)


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def time_graph(graph, count, repeats=24):
    for _ in range(4):
        graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (repeats * count)


def pack_weight(weight):
    """[4, hidden, 320] -> warp-contiguous branch/K fragments, no conversion."""
    if weight.ndim != 3 or weight.shape[0] != 4 or weight.shape[2] != 320:
        raise ValueError("HC up weight must have shape [4, hidden, 320]")
    hidden = weight.shape[1]
    if hidden not in (640, 2560):
        raise ValueError("Only the replicated or TP4 hidden shape is supported")
    return (
        weight.reshape(4, hidden // 8, 8, 20, 2, 8)
        .permute(1, 3, 4, 0, 2, 5)
        .contiguous()
    )


def read_weights(model, pairs, rank, hidden):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    suffix = ".input_mix_weight_up.weight"
    names = sorted(
        name
        for name in index["weight_map"]
        if name.startswith("model.language_model.layers.") and name.endswith(suffix)
    )[:pairs]
    if len(names) != pairs:
        raise ValueError(f"Expected {pairs} HC weights, got {len(names)}")
    offset = rank * hidden if hidden != 2560 else 0
    original, packed, full = [], [], []
    for name in names:
        with safe_open(model / index["weight_map"][name], framework="pt") as file:
            weight = file.get_tensor(name)
        if tuple(weight.shape) != (10240, 320):
            raise ValueError(f"Unexpected HC weight {name}: {weight.shape}")
        weight = weight.half().cuda()
        full.append(weight)
        weight = weight.view(4, 2560, 320)[:, offset : offset + hidden].contiguous()
        original.append(weight.view(4 * hidden, 320))
        packed.append(pack_weight(weight))
    return names, original, packed, full, offset


def build():
    return load(
        name="sm70_hc_batch_reuse_screen",
        sources=[str(ROOT / "benchmarks/csrc/benchmark_sm70_hc_batch_reuse.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


def reference(inputs, weights, branches, gates, outputs, rows, hidden, offset):
    for x, w, b, g, o in zip(inputs, weights, branches, gates, outputs):
        torch.mm(x, w.t(), out=g)
        mix_reference[(rows, triton.cdiv(hidden, 512))](
            b, g, o, hidden, offset, num_warps=4
        )


def run_candidate(
    ext,
    inputs,
    packed,
    branches,
    gates,
    outputs,
    offset,
    paired,
    warps,
    unroll,
    *,
    fused=True,
):
    dest = outputs if fused else gates
    for x, w, b, y in zip(inputs, packed, branches, dest):
        ext.run(x, w, b, y, offset, paired, warps, unroll, fused)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--rows", default="2,4,8,16")
    p.add_argument("--hidden", default="640,2560")
    p.add_argument("--rank", type=int, default=0)
    args = p.parse_args()
    if not args.build_only and (args.model is None or not 0 <= args.rank < 4):
        p.error("--model and a TP rank in [0, 4) are required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    ext = build()
    if args.build_only:
        assert not torch.cuda.is_initialized()
        print("Built on CPU; no CUDA context created", flush=True)
        return
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This screen requires an idle SM70 device")
    torch.manual_seed(20260927)
    source = ROOT / "benchmarks/csrc/benchmark_sm70_hc_batch_reuse.cu"
    result = {
        "contract": "Real HC weights, synthetic inputs, graph micro; not engine TPOT",
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "kernel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "extension_sha256": hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pairs": args.pairs,
        "rank": args.rank,
        "reduced_precision_reduction": False,
        "fp16_accumulation": False,
        "complete": False,
        "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for hidden in map(int, args.hidden.split(",")):
        names, weights, packed, full_weights, offset = read_weights(
            args.model, args.pairs, args.rank, hidden
        )
        for rows in map(int, args.rows.split(",")):
            inputs = [
                torch.empty(rows, 320, device="cuda", dtype=torch.half) for _ in weights
            ]
            branches = [
                torch.empty(rows, 10240, device="cuda", dtype=torch.half)
                for _ in weights
            ]
            gates = [
                torch.empty(rows, hidden * 4, device="cuda", dtype=torch.half)
                for _ in weights
            ]
            outputs = [
                torch.empty(rows, hidden, device="cuda", dtype=torch.half)
                for _ in weights
            ]
            actual_gates = [torch.empty_like(g) for g in gates]
            actual_outputs = [torch.empty_like(o) for o in outputs]

            baseline = partial(
                reference,
                inputs,
                weights,
                branches,
                gates,
                outputs,
                rows,
                hidden,
                offset,
            )

            for x, b in zip(inputs, branches):
                x.normal_(0, 0.1)
                b.normal_()
            bg = capture(baseline)
            combinations = itertools.product(
                (False, True) if rows > 8 else (False,), (1, 4), (4, 8)
            )
            for paired, warps, unroll in combinations:
                candidate = partial(
                    run_candidate,
                    ext,
                    inputs,
                    packed,
                    branches,
                    actual_gates,
                    actual_outputs,
                    offset,
                    paired,
                    warps,
                    unroll,
                )

                cg = capture(candidate)
                gg = capture(partial(candidate, fused=False))
                checks = []
                for scale in (0.0, 0.001, 0.03, 0.1, 1.0, 3.0):
                    for x, b, ag, ao in zip(
                        inputs, branches, actual_gates, actual_outputs
                    ):
                        x.normal_(0, scale)
                        b.normal_(0, scale)
                        ag.fill_(float("nan"))
                        ao.fill_(float("nan"))
                    bg.replay()
                    cg.replay()
                    gg.replay()
                    mismatches = [
                        sum(
                            int((a.view(torch.int16) != b.view(torch.int16)).sum())
                            for a, b in zip(actual, reference)
                        )
                        for actual, reference in (
                            (actual_gates, gates),
                            (actual_outputs, outputs),
                        )
                    ]
                    # Changing N can change cuBLAS's K partition. Admission
                    # also requires parity with the replicated runtime GEMM,
                    # not merely the smaller shard's default heuristic.
                    full_mismatches = 0
                    if hidden != 2560:
                        for x, w, actual in zip(inputs, full_weights, actual_gates):
                            expected = torch.nn.functional.linear(x, w)
                            expected = expected.view(rows, 4, 2560)[
                                :, :, offset : offset + hidden
                            ].reshape(rows, 4 * hidden)
                            full_mismatches += int(
                                (
                                    actual.view(torch.int16)
                                    != expected.view(torch.int16)
                                ).sum()
                            )
                    checks.append(
                        {
                            "scale": scale,
                            "gate_and_mix_mismatches": mismatches,
                            "replicated_projection_mismatches": full_mismatches,
                        }
                    )
                samples = []
                if not any(
                    any(c["gate_and_mix_mismatches"])
                    or c["replicated_projection_mismatches"]
                    for c in checks
                ):
                    for trial in range(6):
                        pair = [None, None]
                        for arm in (0, 1) if trial % 2 == 0 else (1, 0):
                            pair[arm] = time_graph((bg, cg)[arm], len(weights))
                        samples.append(pair)
                medians = (
                    [statistics.median(p[a] for p in samples) for a in (0, 1)]
                    if samples
                    else None
                )
                record = {
                    "rows": rows,
                    "hidden": hidden,
                    "paired": paired,
                    "warps": warps,
                    "unroll": unroll,
                    "checks": checks,
                    "baseline_candidate_us": medians,
                    "paired_samples_us": samples,
                    "weights": names,
                }
                result["cases"].append(record)
                args.out.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in record.items()
                            if k not in ("weights", "paired_samples_us")
                        }
                    ),
                    flush=True,
                )
            del bg, cg, gg, baseline, candidate, inputs, branches, gates, outputs
            del actual_gates, actual_outputs
        del weights, packed, full_weights
    result["complete"] = True
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
