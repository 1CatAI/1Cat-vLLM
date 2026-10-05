# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold TP4 expert layer: Q8 intermediate versus repeated down quantization."""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np
import torch
import vllm._C as core

import vllm
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import GGUFExpertBank
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader


def cold_time(operation, eviction, iterations):
    for _ in range(3):
        eviction.add_(1)
        operation()
    events = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in range(16)
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for start, end in events:
            eviction.add_(1)
            start.record()
            operation()
            end.record()
    samples = []
    for _ in range(iterations):
        graph.replay()
        events[-1][1].synchronize()
        samples.extend(a.elapsed_time(b) * 1000 for a, b in events)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--lanes", type=int, nargs="+", choices=[4, 8, 16, 32], default=[16, 8, 4]
    )
    args = parser.parse_args()
    assert "site-packages" in vllm.__file__, vllm.__file__
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    reader = GGUFReader(args.model)
    tensors = [
        next(
            t
            for t in reader.tensors
            if t.name == f"blk.{args.layer}.ffn_{name}_exps.weight"
        )
        for name in ("gate", "up", "down")
    ]
    kind, down_kind = int(tensors[0].tensor_type), int(tensors[2].tensor_type)
    assert kind == int(tensors[1].tensor_type) and kind in (18, 21, 22)
    assert down_kind in (20, 42)
    banks = []
    for tensor in tensors[:2]:
        rows = [
            RawGGUFProjection.from_rows(r, kind).tp_slice(0, 4, axis=0).data
            for r in tensor.data
        ]
        banks.append(torch.from_numpy(np.stack(rows)).cuda())
    down = GGUFExpertBank(down_kind, 512, torch.device("cuda"), torch.float16)
    for index, row in enumerate(tensors[2].data):
        down.add(index, torch.from_numpy(row.copy()), 0, 4, axis=1)
    down.finalize()
    eviction = torch.zeros(32 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    report = dict(
        version=vllm.__version__,
        layer=args.layer,
        source_type=kind,
        down_type=down_kind,
        tp_size=4,
        shape=[512, 160, 2560],
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        routing="seeded random top10; synthetic distribution",
        cache="32MiB eviction before each operation",
        cases=[],
    )
    for m in (5, 20):
        torch.manual_seed(20261005 + m)
        x = torch.randn((m, 2560), dtype=torch.float16, device="cuda")
        ids = torch.randn((m, 512), device="cuda").topk(10, dim=1).indices.int()
        probabilities = torch.softmax(torch.randn((m, 10), device="cuda"), 1)
        q8 = torch.empty((m, 80, 36), dtype=torch.uint8, device="cuda")
        hidden = torch.empty((m, 10, 160), dtype=torch.float16, device="cuda")
        intermediate = torch.empty((m, 10, 5, 36), dtype=torch.uint8, device="cuda")
        baseline = torch.empty_like(x)
        candidate = torch.empty_like(x)

        def encode(q8=q8, x=x):
            torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)

        def gate(lanes=None, hidden=hidden, intermediate=intermediate, q8=q8, ids=ids):
            torch.ops._C.gguf_dp4a_gate_up_sm70_out(
                hidden if lanes is None else intermediate,
                q8,
                ids,
                *banks,
                kind,
                True,
                16 if lanes is None else lanes,
            )

        def output(
            quantized=False,
            candidate=candidate,
            baseline=baseline,
            intermediate=intermediate,
            hidden=hidden,
            ids=ids,
            probabilities=probabilities,
        ):
            torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
                candidate if quantized else baseline,
                intermediate if quantized else hidden,
                ids,
                probabilities,
                down.weight_ptrs,
                down.stat_ptrs,
                down_kind,
                512,
            )

        def full(lanes=None):
            encode()
            gate(lanes)
            output(lanes is not None)

        full()
        for lanes in args.lanes:
            full(lanes)
            if lanes == 16:
                torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
            else:
                torch.testing.assert_close(candidate, baseline, rtol=0.005, atol=0.005)
            timings = {
                name: []
                for name in (
                    "baseline",
                    "candidate",
                    "old_gate",
                    "q8_gate",
                    "old_down",
                    "q8_down",
                )
            }
            operations = dict(
                baseline=lambda: full(),
                candidate=lambda lanes=lanes: full(lanes),
                old_gate=lambda: gate(),
                q8_gate=lambda lanes=lanes: gate(lanes),
                old_down=lambda: output(),
                q8_down=lambda: output(True),
            )
            for epoch in range(4):
                for name in list(operations)[:: 1 if epoch % 2 == 0 else -1]:
                    timings[name].append(
                        cold_time(operations[name], eviction, args.iterations)
                    )
            report["cases"].append(
                dict(
                    m=m,
                    lanes=lanes,
                    active_experts=ids.unique().numel(),
                    kernels_per_layer=3,
                    epoch_us=timings,
                    median_us={n: statistics.median(t) for n, t in timings.items()},
                    unique_gate_up_bytes=sum(b.numel() for b in banks)
                    // 512
                    * ids.unique().numel(),
                    unique_gate_up_gbps=(
                        sum(b.numel() for b in banks) // 512 * ids.unique().numel()
                    )
                    / statistics.median(timings["q8_gate"])
                    / 1000,
                    clocks=subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=clocks.sm,clocks.mem",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    ).strip(),
                )
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["cases"][-1]), flush=True)


if __name__ == "__main__":
    main()
