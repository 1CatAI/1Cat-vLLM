# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare installed SM70 expert operators on real GGUF TP shards and routes.

Run under flock /tmp/gpu0-3.lock on an idle SM70 GPU. No extension overlays,
model startup, or dispatch changes. Inputs are synthetic FP16; M20 concatenates
four captured M5 windows and is not an end-to-end C4 measurement.
"""

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import vllm._C as core

from vllm.model_executor.layers.quantization.gguf_moe_planes import expert_table
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
    GGUFExpertBank,
    _expert_planes,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            fn()
    return graph


def measure(graph, iterations):
    for _ in range(5):
        graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (iterations * 8)


def restore_q8(packet):
    scale = packet[..., :2].contiguous().view(torch.float16).float()
    return (packet[..., 4:].view(torch.int8).float() * scale).flatten(-2)


def error(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.float().norm().clamp_min(1e-30)).item(),
    }


def run_case(args, layer, source, records):
    gate_type, down_type = int(source[0].tensor_type), int(source[2].tensor_type)
    assert gate_type in (18, 21, 22) and down_type in (20, 42)
    assert int(source[1].tensor_type) == gate_type
    # ReaderTensor.shape uses GGML's K,N,E order; packed data is E,N,Kbytes.
    experts, full_n, k = map(int, reversed(source[0].shape))
    assert experts == 512 and k == 2560 and full_n % args.tp == 0
    n, topk = full_n // args.tp, 10
    assert n == 160
    raw = [
        torch.from_numpy(
            np.stack(
                [
                    RawGGUFProjection.from_rows(rows, gate_type)
                    .tp_slice(args.rank, args.tp, axis=0)
                    .data
                    for rows in tensor.data
                ]
            )
        ).cuda()
        for tensor in source[:2]
    ]
    bank = GGUFExpertBank(down_type, experts, torch.device("cuda"), torch.float16)
    for e, rows in enumerate(source[2].data):
        bank.add(e, torch.from_numpy(rows.copy()), args.rank, args.tp, axis=1)
    bank.finalize()
    fmt, gc, gs = _expert_planes(raw[0], gate_type, n, k)
    _, uc, us = _expert_planes(raw[1], gate_type, n, k)
    table = torch.from_numpy(expert_table(gate_type)).cuda()
    gu_bytes = sum(t.numel() * t.element_size() for t in raw) // experts
    mma_bytes = sum(t.numel() * t.element_size() for t in (gc, gs, uc, us)) // experts
    down_bytes = (
        sum(t.numel() * t.element_size() for t in (bank.weights, bank.stats)) // experts
    )
    results = []
    for m in args.m:
        torch.manual_seed(args.seed + layer * 31 + m)
        x = (torch.randn(m, k, device="cuda") * 0.25).half()
        ids = torch.empty((m, topk), dtype=torch.int32, device="cuda")
        probabilities = torch.softmax(torch.randn(m, topk, device="cuda"), -1)
        q8 = torch.empty((m, k // 32, 36), dtype=torch.uint8, device="cuda")
        hidden = {
            name: torch.empty((m, topk, n // 32, 36), dtype=torch.uint8, device="cuda")
            for name in ("raw", "mma")
        }
        output = {name: torch.empty_like(x) for name in hidden}
        h16 = torch.empty((m, topk, n), dtype=torch.float16, device="cuda")

        def set_routes(ordinal, m=m, ids=ids):
            selected = [
                records[(ordinal + i) % len(records)]["ids"] for i in range(m // 5)
            ]
            array = np.concatenate(selected, 0)
            assert array.shape == (m, topk) and np.all((array >= 0) & (array < experts))
            assert all(len(set(row)) == topk for row in array)
            ids.copy_(torch.tensor(array, dtype=torch.int32, device="cuda"))

        def quantize(q8=q8, x=x):
            torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)

        def raw_gu(hidden=hidden, q8=q8, ids=ids):
            torch.ops._C.gguf_dp4a_gate_up_sm70_out(
                hidden["raw"], q8, ids, *raw, gate_type, True
            )

        def mma_gu(hidden=hidden, x=x, ids=ids):
            torch.ops._C.gguf_moe_gate_up_sm70_out(
                hidden["mma"], x, ids, gc, gs, uc, us, fmt, table, 4
            )

        def down(
            name, output=output, hidden=hidden, ids=ids, probabilities=probabilities
        ):
            torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
                output[name],
                hidden[name],
                ids,
                probabilities,
                bank.weight_ptrs,
                bank.stat_ptrs,
                down_type,
                experts,
            )

        def raw_chain():
            quantize()
            raw_gu()
            down("raw")

        def mma_chain():
            mma_gu()
            down("mma")

        set_routes(0)
        raw_chain()
        mma_chain()
        operations = {
            "quantize": quantize,
            "raw_gate": raw_gu,
            "mma_gate": mma_gu,
            "raw_down": lambda: down("raw"),
            "mma_down": lambda: down("mma"),
            "raw_chain": raw_chain,
            "mma_chain": mma_chain,
        }
        graphs = {name: capture(fn) for name, fn in operations.items()}
        checks, weight_cache = [], {}

        def official(e, weight_cache=weight_cache):
            if e not in weight_cache:
                rows = slice(args.rank * n, (args.rank + 1) * n)
                arrays = [dequantize(t.data[e], int(t.tensor_type)) for t in source]
                weight_cache[e] = [
                    torch.from_numpy(
                        np.ascontiguousarray(a[rows] if i < 2 else a[:, rows])
                    ).cuda()
                    for i, a in enumerate(arrays)
                ]
            return weight_cache[e]

        for ordinal in range(min(args.verify_routes, len(records))):
            set_routes(ordinal)
            x.copy_((torch.randn_like(x.float()) * (0.1 + ordinal * 0.05)).half())
            probabilities.copy_(torch.softmax(torch.randn_like(probabilities), -1))
            for packet in hidden.values():
                packet.fill_(255)
            graphs["raw_chain"].replay()
            graphs["mma_chain"].replay()
            check = {"ordinal": ordinal}
            for name in ("raw", "mma"):
                if name == "raw":
                    torch.ops._C.gguf_dp4a_gate_up_sm70_out(
                        h16, q8, ids, *raw, gate_type, True
                    )
                    activation = restore_q8(q8)
                else:
                    torch.ops._C.gguf_moe_gate_up_sm70_out(
                        h16, x, ids, gc, gs, uc, us, fmt, table, 4
                    )
                    activation = x.float()
                href, dref = (
                    torch.empty_like(h16),
                    torch.empty((m, topk, k), dtype=torch.float16, device="cuda"),
                )
                restored = restore_q8(hidden[name])
                for e in ids.unique().tolist():
                    wg, wu, wd = official(e)
                    loc = (ids == e).nonzero()
                    g = (activation[loc[:, 0]] @ wg.T).half()
                    u = (activation[loc[:, 0]] @ wu.T).half()
                    href[loc[:, 0], loc[:, 1]] = torch.nn.functional.silu(g) * u
                    dref[loc[:, 0], loc[:, 1]] = (
                        restored[loc[:, 0], loc[:, 1]] @ wd.T
                    ).half()
                oracle = (dref.float() * probabilities[..., None]).sum(1).half()
                h_error, q_error, d_error = (
                    error(h16, href),
                    error(restored, href),
                    error(output[name], oracle),
                )
                assert h_error["relative_l2"] < 0.002, h_error
                assert q_error["relative_l2"] < 0.02, q_error
                torch.testing.assert_close(output[name], oracle, rtol=0.003, atol=0.003)
                check[name] = {
                    "fp16_hidden": h_error,
                    "q8_hidden": q_error,
                    "down_output": d_error,
                }
            checks.append(check)
        weight_cache.clear()

        samples = {name: [] for name in graphs}
        windows = []
        for ordinal in range(min(args.windows, len(records))):
            set_routes(ordinal)
            raw_chain()
            mma_chain()
            torch.cuda.synchronize()
            counts = Counter(ids.cpu().flatten().tolist())
            windows.append(
                {
                    "ordinal": ordinal,
                    "unique_experts": len(counts),
                    "mma_batches": sum((v + 7) // 8 for v in counts.values()),
                }
            )
            order = list(graphs)
            if ordinal % 2:
                order.reverse()
            for name in order:
                samples[name].append(measure(graphs[name], args.iterations))
        payload = {
            "raw_gate": [m * topk * gu_bytes] * len(windows),
            "mma_gate": [w["mma_batches"] * mma_bytes for w in windows],
            "raw_down": [m * topk * down_bytes] * len(windows),
            "mma_down": [m * topk * down_bytes] * len(windows),
        }
        logical = {
            name: [
                w["unique_experts"]
                * (
                    gu_bytes
                    if name == "raw_gate"
                    else mma_bytes
                    if name == "mma_gate"
                    else down_bytes
                )
                for w in windows
            ]
            for name in payload
        }
        case = {
            "layer": layer,
            "gate_type": gate_type,
            "down_type": down_type,
            "m": m,
            "n": n,
            "k": k,
            "topk": topk,
            "median_us": {name: statistics.median(v) for name, v in samples.items()},
            "mean_us": {name: statistics.mean(v) for name, v in samples.items()},
            "samples_us": samples,
            "route_windows": windows,
            "numerical_checks": checks,
            "paired_chain_saving_us": [
                a - b for a, b in zip(samples["raw_chain"], samples["mma_chain"])
            ],
            "weight_bytes_per_expert": {
                "raw_gate": gu_bytes,
                "mma_gate": mma_bytes,
                "down": down_bytes,
            },
            "logical_unique_gbps": {
                name: sum(v) / sum(samples[name]) / 1000 for name, v in logical.items()
            },
            "issued_payload_gbps": {
                name: sum(v) / sum(samples[name]) / 1000 for name, v in payload.items()
            },
            "bandwidth_scope": "weight payload estimate; not measured DRAM traffic",
            "kernel_count": {"raw_chain": 3, "mma_chain": 2},
        }
        results.append(case)
        print(
            json.dumps(
                {
                    key: case[key]
                    for key in ("layer", "m", "median_us", "logical_unique_gbps")
                }
            ),
            flush=True,
        )
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--routes", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--layers", nargs="+", type=int, default=[17, 0, 1])
    p.add_argument("--m", nargs="+", type=int, choices=[5, 20], default=[5, 20])
    p.add_argument("--tp", type=int, choices=[4], default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--windows", type=int, default=38)
    p.add_argument("--verify-routes", type=int, default=8)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--seed", type=int, default=20261010)
    args = p.parse_args()
    assert 0 <= args.rank < args.tp
    assert min(args.windows, args.verify_routes, args.iterations) > 0
    assert not os.environ.get("LD_PRELOAD")
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    stem = args.model.name.rsplit("-00001-of-", 1)[0]
    readers = [GGUFReader(f) for f in sorted(args.model.parent.glob(stem + "-*.gguf"))]
    assert readers
    tensors = {t.name: t for r in readers for t in r.tensors}
    routes = json.loads(args.routes.read_text())
    report = {
        "scope": "installed native expert operators; no model or dispatch changes",
        "model": args.model.name,
        "tp": args.tp,
        "rank": args.rank,
        "source_sha": os.environ.get("EXPERT_SOURCE_SHA"),
        "benchmark_sha256": fingerprint(__file__),
        "core_sha256": fingerprint(core.__file__),
        "route_sha256": fingerprint(args.routes),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "activation": "seeded synthetic FP16, std 0.1 through 0.45",
        "m20_routes": "four consecutive recorded M5 windows concatenated",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in ("model", "routes", "output")
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for layer in args.layers:
        source = [
            tensors[f"blk.{layer}.ffn_{name}_exps.weight"]
            for name in ("gate", "up", "down")
        ]
        report["cases"].extend(
            run_case(args, layer, source, routes[str(layer)]["records"])
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
