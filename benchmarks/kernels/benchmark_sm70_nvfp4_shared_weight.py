# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare shared TurboMind codes against QPN2 on real TP4 projection shards.

Load the task-built sm70_nvfp4_shared_sidecar first. Its control and shared
kernels are compiled together; TurboMind itself comes from --core-library.
This is an operator benchmark, not an end-to-end DFlash2 speed measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from benchmark_sm70_quasar_nvfp4_oracle import (
    QPN2_CONFIGS,
    _load_projections,
    _unpack_codes,
)


def _paired_latency(control, shared, m):
    # Multiple nodes amortize Python replay overhead for these short kernels.
    nodes = 16 if m <= 32 else 1
    graphs = []
    for call in (control, shared):
        for _ in range(5):
            call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(nodes):
                call()
        graphs.append(graph)

    def measure(graph):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / (10 * nodes)

    # ABBA pairs expose drift and balance which layout runs first.
    samples = []
    for _ in range(8):
        samples.append([measure(graphs[i]) for i in (0, 1, 1, 0)])
    before = statistics.median(s[0] for s in samples)
    after = statistics.median(s[3] for s in samples)
    candidate = statistics.median((s[1] + s[2]) / 2 for s in samples)
    return dict(
        control_before_us=before,
        shared_us=candidate,
        control_after_us=after,
        ratio=candidate / ((before + after) / 2),
        abba_samples_us=samples,
        graph_nodes=nodes,
    )


def _equal_bits(actual, expected):
    return torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def _check_projection(projection, rank, rows):
    packed = projection.packed.cuda()
    raw_scales = projection.scales.cuda()
    logical_n = packed.shape[0]
    n = (logical_n + 31) // 32 * 32
    k = packed.shape[1] * 2
    padded = torch.zeros((n, k // 2), dtype=torch.uint8, device="cuda")
    padded_scales = torch.zeros((n, k // 16), dtype=raw_scales.dtype, device="cuda")
    padded[:logical_n].copy_(packed)
    padded_scales[:logical_n].copy_(raw_scales)
    global_scale = 1.0 / projection.weight_global_divisor
    effective_scales = (padded_scales.t().float() * global_scale).half().contiguous()
    tm_weight, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
        _unpack_codes(padded), effective_scales, 16, False
    )
    codes, scales = torch.ops._qpn2_shared_control.prepare(padded, padded_scales)
    shared_scales = torch.ops._C.nvfp4_qpn2_prepare_scales_sm70(raw_scales)
    assert torch.equal(shared_scales, scales), "Scale-only preparation differs"

    # Inspect real CUDA converter output independently of either GEMM reader.
    lane = torch.arange(32, device="cuda")
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) != 0) * 4
    mapped = (
        tm_weight.reshape(n // 32, k // 16, 2, 32)[..., col]
        .permute(0, 1, 3, 2)
        .contiguous()
        .view(torch.uint8)
    )
    assert torch.equal(mapped.flatten(), codes.flatten()), "4-bit code mapping differs"
    del mapped, effective_scales, padded, padded_scales, packed, raw_scales
    split_k, nacc = QPN2_CONFIGS[(k, n)]
    result = {
        "rank": rank,
        "projection": projection.name,
        "logical_n": logical_n,
        "n": n,
        "k": k,
        "code_bits_equal": True,
        "scale_bits_equal": True,
        "removed_code_bytes": codes.numel(),
        "cases": [],
    }
    for m in rows:
        for gated in [False, True] if projection.name == "mlp_gate_up" else [False]:
            x = torch.randn((m, k), dtype=torch.float16, device="cuda") * 0.1
            old = torch.empty((m, n // 2 if gated else n), device="cuda", dtype=x.dtype)
            new = torch.empty_like(old)

            def control(old=old, x=x, gated=gated):
                torch.ops._qpn2_shared_control.dispatch(
                    old,
                    x,
                    codes,
                    scales,
                    global_scale,
                    split_k,
                    nacc,
                    tm_weight,
                    tm_scales,
                    16,
                    int(meta[0]),
                    int(meta[1]),
                    gated,
                    1024,
                )

            def shared(new=new, x=x, gated=gated):
                torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
                    new,
                    x,
                    tm_weight,
                    shared_scales,
                    global_scale,
                    split_k,
                    nacc,
                    tm_scales,
                    16,
                    int(meta[0]),
                    int(meta[1]),
                    gated,
                    1024,
                )

            control()
            shared()
            assert _equal_bits(new, old), (rank, projection.name, m, gated, "eager")
            # Capture both paths and change inputs between replays.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                control()
                shared()
            for _ in range(2):
                x.normal_(std=0.1)
                graph.replay()
                assert _equal_bits(new, old), (rank, projection.name, m, gated, "graph")
            del graph
            case = {
                "m": m,
                "gated": gated,
                "eager_bits_equal": True,
                "graph_bits_equal": True,
            }
            if rank == 0:
                case.update(_paired_latency(control, shared, m))
            result["cases"].append(case)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--core-library", type=Path, required=True)
    parser.add_argument("--shared-library", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 33, 64, 135, 1019, 1024, 4096],
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("vllm._C", args.core_library)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm._C"] = module
    spec.loader.exec_module(module)
    stable_library = args.core_library.with_name("_C_stable_libtorch.abi3.so")
    torch.ops.load_library(str(stable_library))
    torch.ops.load_library(str(args.shared_library))
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(20260908)
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "model": str(args.model),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "libraries": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [args.core_library, stable_library, args.shared_library]
        },
        "results": [],
    }
    try:
        for rank in args.ranks:
            for projection in _load_projections(args.model, rank, 4):
                result = _check_projection(projection, rank, args.rows)
                report["results"].append(result)
                print(json.dumps(result), flush=True)
                torch.cuda.empty_cache()
        report["passed"] = True
    finally:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
