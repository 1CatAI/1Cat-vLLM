# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual FlashInfer SM70 QSA screening, not a serving performance claim.

All candidates include index preparation and final merge. No GPU allocations
occur within the candidate forward. Run on an exclusively reserved GPU.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.flashinfer_sm70_qsa import (
    UPSTREAM_SHA,
    FlashInferQSA,
    build,
)


def make_case(rows, selected=2051, page=784, kv_heads=1, group=6, length=8192):
    """Independent requests, permuted physical pages, ordered sparse indices."""
    blocks = (length + page - 1) // page
    q = torch.randn(rows, kv_heads * group, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(
        rows * blocks, page, kv_heads, 256, device="cuda", dtype=torch.float16
    )
    v = torch.randn_like(k)
    table = torch.randperm(rows * blocks, device="cuda", dtype=torch.int32)
    table = table.reshape(rows, blocks)
    requests = torch.arange(rows, device="cuda", dtype=torch.int32)
    # Random order is deliberate, with no physical sorting or deduplication.
    indices = torch.stack(
        [
            torch.randperm(length, device="cuda", dtype=torch.int32)[:selected]
            for _ in range(rows)
        ]
    )
    if selected == 2051:
        # Runtime uses 512 four-token blocks and up to three tail positions.
        pages = torch.stack(
            [
                torch.randperm(length // 4 - 1, device="cuda", dtype=torch.int32)[:512]
                for _ in range(rows)
            ]
        )
        indices[:, :2048] = (
            pages[:, :, None] * 4 + torch.arange(4, device="cuda")
        ).reshape(rows, 2048)
        indices[:, 2048:] = -1  # 8192 has zero tail residue.
    return q, k, v, indices, table, requests


def oracle(q, k, v, indices, table, requests):
    """FP32 arithmetic with the same visible-index contract, preserving repeats."""
    out = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    page = k.shape[1]
    group = q.shape[1] // k.shape[2]
    for row in range(q.shape[0]):
        request = int(requests[row])
        if not 0 <= request < table.shape[0]:
            continue
        logical = indices[row].long()
        valid = (logical >= 0) & (logical // page < table.shape[1])
        logical = logical[valid]
        physical = table[request, logical // page].long()
        valid = (physical >= 0) & (physical < k.shape[0])
        logical, physical = logical[valid], physical[valid]
        if logical.numel() == 0:
            continue
        keys = k[physical, logical % page].float().repeat_interleave(group, 1)
        values = v[physical, logical % page].float().repeat_interleave(group, 1)
        scores = torch.einsum("hd,shd->hs", q[row].float(), keys) / 16
        out[row] = torch.einsum("hs,shd->hd", scores.softmax(-1), values)
    return out


def check_exclusive():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Choose exactly one reserved CUDA_VISIBLE_DEVICES GPU")
    report = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            visible,
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    for line in report.splitlines():
        pid, _, name = line.partition(",")
        if (
            pid.strip().isdigit()
            and int(pid) != os.getpid()
            and "snapd-desktop-integration" not in name
        ):
            raise RuntimeError(f"Foreign GPU process; discard timing: {line}")


def capture(call, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            call()
    return graph


def elapsed_us(graph, calls):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / calls


@torch.inference_mode()
def screen(rows, args):
    case = make_case(rows, length=args.length)
    q, k, v, indices, table, req = case
    names, calls = [], []
    reference_sha = None
    if args.compare_triton:
        from vllm.models.qwen4_exp.nvidia.ops import qsa as ops

        # No query_positions supplied: this forces actual Triton sparse QSA,
        # bypassing the native page4 dispatcher. No env heuristic override.
        out = torch.empty_like(q)
        calls.append(
            lambda: ops.qsa_sparse_paged_attention(
                q, k, v, indices, table, req, out=out
            )
        )
        names.append("current_triton")
        reference_sha = hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest()
    candidates = []
    for splits in args.splits:
        candidate = FlashInferQSA(q, indices.shape[1], splits)
        candidates.append(candidate)
        calls.append(lambda candidate=candidate: candidate(*case))
        names.append(f"flashinfer_s{splits}")
    graphs = [capture(call, args.calls) for call in calls]
    checks = []
    for cycle in range(4):
        q.normal_().mul_((0.25, 1.0, 3.0, 1.0)[cycle])
        indices[:, 2048:] = -1
        if cycle:
            indices[:, 2048 : 2048 + cycle] = torch.arange(
                args.length - cycle, args.length, device="cuda"
            )
        table.copy_(torch.randperm(k.shape[0], device="cuda").reshape_as(table))
        expected = oracle(*case)
        for name, call, graph in zip(names, calls, graphs):
            eager = call().clone()
            graph.replay()
            torch.cuda.synchronize()
            # Another call is not used to inspect replay output.
            replay = (
                out
                if name == "current_triton"
                else candidates[
                    args.splits.index(int(name.removeprefix("flashinfer_s")))
                ].output
            )
            torch.testing.assert_close(replay, eager, atol=0, rtol=0)
            torch.testing.assert_close(replay.float(), expected, atol=2e-3, rtol=1e-2)
            relative = (
                (replay.float() - expected).norm() / expected.norm().clamp_min(1e-20)
            ).item()
            if relative > 5e-3:
                raise AssertionError(f"{name}: relative L2 {relative}")
            checks.append(
                {
                    "cycle": cycle,
                    "route": name,
                    "max_abs": (replay.float() - expected).abs().max().item(),
                    "relative_l2": relative,
                }
            )
    # Restore ordinary full-context tail for timing; all arms see same data.
    indices[:, 2048:] = -1
    timings = {name: [] for name in names}
    check_exclusive()
    for iteration in range(args.samples):
        order = list(range(len(names)))
        if iteration % 2:
            order.reverse()
        for index in order:
            timings[names[index]].append(elapsed_us(graphs[index], args.calls))
    check_exclusive()
    return {
        "rows": rows,
        "length": args.length,
        "selection_width": 2051,
        "q_shape": list(q.shape),
        "page_size": k.shape[1],
        "flashinfer_sha": UPSTREAM_SHA,
        "triton_source_sha256": reference_sha,
        "checks": checks,
        "microseconds_samples": timings,
        "median_us": {name: statistics.median(t) for name, t in timings.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--splits", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--compare-triton", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    check_exclusive()
    build()
    for rows in args.rows:
        print(json.dumps(screen(rows, args)), flush=True)


if __name__ == "__main__":
    main()
