# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired exactness/graph screen; no model weights or serving process."""

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    _sm70_qwen38_router_topk_kernel as router,
)


def capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def timing(graph, loops=200):
    samples = []
    for _ in range(5):
        for _ in range(20):
            graph.replay()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(loops):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / loops)
    return {"median_ms": median(samples), "samples_ms": samples}


def paired_timing(control, candidate):
    # Interleave A/B and reverse order to expose clock/cache drift.
    for _ in range(3000):
        control.replay()
        candidate.replay()
    torch.cuda.synchronize()
    samples = {"control": [], "candidate": []}
    for repeat in range(8):
        pairs = [("control", control), ("candidate", candidate)]
        if repeat % 2:
            pairs.reverse()
        for label, graph in pairs:
            for _ in range(20):
                graph.replay()
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(200):
                graph.replay()
            b.record()
            b.synchronize()
            samples[label].append(a.elapsed_time(b) / 200)
    return {
        label: {"median_ms": median(v), "samples_ms": v} for label, v in samples.items()
    }


def router_screen():
    def out(rows):
        return (
            torch.empty(rows, 10, dtype=torch.float32, device="cuda"),
            torch.empty(rows, 10, dtype=torch.int32, device="cuda"),
            torch.empty(rows, 10, dtype=torch.int32, device="cuda"),
        )

    def launch(x, dst, packed):
        router[(x.shape[0],)](
            x,
            *dst,
            E=512,
            K=10,
            M=x.shape[0],
            BLOCK_E=512,
            PACKED_HALF_KEY=packed,
            num_warps=8,
        )

    # Exhaust all raw half encodings, then finite-only permutations and ties.
    raw = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
    exhaustive = raw.view(torch.float16).reshape(128, 512)
    finite = torch.nan_to_num(exhaustive, nan=0, posinf=65504, neginf=-65504)
    cases = [
        exhaustive,
        finite,
        finite.flatten()[torch.randperm(65536, device="cuda")].reshape(128, 512),
    ]
    for scale in (0.001, 0.1, 1.0, 10.0):
        cases.append(torch.randn(128, 512, dtype=torch.float16, device="cuda") * scale)
    cases += [torch.zeros_like(finite), torch.full_like(finite, -float("inf"))]
    cases.append((torch.arange(512, device="cuda") % 7).half().repeat(128, 1))
    samples = 0
    for x in cases:
        a, b = out(x.shape[0]), out(x.shape[0])
        launch(x, a, False)
        launch(x, b, True)
        for ref, got in zip(a, b):
            assert torch.equal(ref.view(torch.int32), got.view(torch.int32))
        samples += x.shape[0]

    x = torch.randn(48, 512, dtype=torch.float16, device="cuda")
    a, b = out(48), out(48)

    def run(dst, packed):
        for i in range(48):
            launch(x[i : i + 1], tuple(t[i : i + 1] for t in dst), packed)

    ga, gb = capture(lambda: run(a, False)), capture(lambda: run(b, True))
    for _ in range(16):
        x.normal_()
        for value in b:
            value.fill_(-777)
        ga.replay()
        gb.replay()
        assert all(
            torch.equal(p.view(torch.int32), q.view(torch.int32)) for p, q in zip(a, b)
        )
    times = paired_timing(ga, gb)
    return {
        "bitwise_rows": samples,
        "changing_graph_replays": 16,
        "control": times["control"],
        "packed32": times["candidate"],
        "calls": 48,
    }


def qsa_screen():
    base = torch.ops._C_qsa_verify.baseline
    fast = torch.ops._C_qsa_sm70.qsa_lexicographic_topk
    assert torch.ops._C_qsa_sm70.decode_specialization_version() == 1
    count = 0
    for rows in (1, 2):
        for length in (0, 1, 511, 512, 2048, 2304, 2305, 4096, 65536):
            width = max(512, length)
            x = torch.randn(rows, width, device="cuda")
            n = torch.full((rows,), length, dtype=torch.int32, device="cuda")
            a = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
            b = torch.empty_like(a)
            for pattern in ("random", "ties", "zeros", "special"):
                if pattern == "ties":
                    x.copy_(torch.arange(width, device="cuda") % 7)
                elif pattern == "zeros":
                    x.zero_()
                    x[:, ::2] = -0.0
                elif pattern == "special":
                    x[:, 0], x[:, 1], x[:, 2] = (
                        float("nan"),
                        float("inf"),
                        -float("inf"),
                    )
                a.fill_(-111)
                b.fill_(-111)
                base(x, n, a, 512)
                fast(x, n, b, 512)
                assert torch.equal(a, b), (rows, length, pattern)
                count += 1
    x = torch.randn(12, 1, 4096, device="cuda")
    n = torch.tensor(
        [[2048 + i * 11] for i in range(12)], dtype=torch.int32, device="cuda"
    )
    a = torch.empty(12, 1, 512, dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)

    def run(op, out):
        for i in range(12):
            op(x[i], n[i], out[i], 512)

    ga, gb = capture(lambda: run(base, a)), capture(lambda: run(fast, b))
    for _ in range(16):
        x.normal_()
        b.fill_(-777)
        ga.replay()
        gb.replay()
        assert torch.equal(a, b)
    times = paired_timing(ga, gb)
    return {
        "exact_cases": count,
        "changing_graph_replays": 16,
        "control": times["control"],
        "decode": times["candidate"],
        "calls": 12,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qsa-library", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(20260905)
    torch.ops.load_library(args.qsa_library)
    result = {"scope": "model-free GPU operator/graph screen, not endpoint speed"}
    result["router"] = router_screen()
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    result["qsa"] = qsa_screen()
    result.update(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
