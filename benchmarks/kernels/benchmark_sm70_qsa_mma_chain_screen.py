# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor Core selection/expansion/sparse-attention/merge/output-gate screen.

Score construction, projections, rotary transforms and KV writes are excluded.
The native exact selector and paged attention are the complete-chain control.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from benchmarks.kernels.sm70_chain_screen_utils import graph_kernel_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).parents[1] / "csrc/sm70_qsa_mma_chain_screen.cu"
    extension = load(
        "sm70_qsa_mma_chain_screen",
        [str(source)],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        verbose=True,
    )
    if args.build_only:
        args.output.write_text(json.dumps({"library": extension.__file__}) + "\n")
        return
    import vllm._custom_ops  # noqa: F401
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    report = {
        "research_only": True,
        "model_admission": False,
        "finite_score_contract": True,
        "producer_ctas": 64,
        "cache_banks": 12,
        "calls_per_graph": 12,
        "fixed_consumer_ctas": 7,
        "scope": (
            "selection/expansion/sparse attention/merge/gate; "
            "no scores/projections/rotary/KV writes"
        ),
        "widths": [],
    }

    def screen(m):
        torch.manual_seed(4201)
        columns, page = 2052, 784
        positions = torch.arange(8192, 8192 + m, dtype=torch.int64, device="cuda")
        lengths = ((positions + 1) // 4).int()
        scores = torch.randn(m, columns, device="cuda", dtype=torch.float32)
        queries = torch.randn(m, 6, 256, device="cuda", dtype=torch.float16) * 0.1
        cache = torch.randn(11, 2, page, 256, device="cuda", dtype=torch.float16) * 0.1
        table = torch.randperm(11, device="cuda").int().view(1, 11)
        gate = torch.randn_like(queries)
        selected = [
            torch.empty(m, 512, device="cuda", dtype=torch.int32) for _ in range(2)
        ]
        indices = torch.empty(m, 2051, device="cuda", dtype=torch.int32)
        partial = torch.empty(m, 6, 64, 256, device="cuda", dtype=torch.float32)
        lse = torch.empty(m, 6, 64, device="cuda", dtype=torch.float32)
        flags = torch.zeros(m + m * 6 * 64, device="cuda", dtype=torch.int32)
        epochs = torch.zeros(71, device="cuda", dtype=torch.int32)
        outputs = [torch.empty_like(queries) for _ in range(2)]
        requests = torch.zeros(m, device="cuda", dtype=torch.int32)
        seq_lens = torch.tensor([8192 + m], device="cuda", dtype=torch.int32)

        banks = [
            (
                scores.clone(),
                queries.clone(),
                cache.clone(),
                gate.clone(),
                [torch.empty_like(t) for t in selected],
                torch.empty_like(indices),
                torch.empty_like(partial),
                torch.empty_like(lse),
                torch.zeros_like(flags),
                torch.zeros_like(epochs),
                [torch.empty_like(t) for t in outputs],
            )
            for _ in range(12)
        ]

        def control():
            for (
                scores,
                queries,
                cache,
                gate,
                selected,
                indices,
                partial,
                lse,
                flags,
                epochs,
                outputs,
            ) in banks:
                torch.ops._C.qsa_lexicographic_topk(
                    scores, lengths, selected[0], 512, m == 5
                )
                qsa.expand_qsa_block_indices_cuda(
                    selected[0], positions, seq_lens, requests, 4, 2048, indices
                )
                qsa.qsa_sparse_paged_attention(
                    queries,
                    cache[:, 0].unsqueeze(2),
                    cache[:, 1].unsqueeze(2),
                    indices,
                    table,
                    requests,
                    outputs[0],
                    output_gate=gate,
                    query_positions=positions,
                    sequence_lengths=seq_lens,
                )

        def candidate():
            for (
                scores,
                queries,
                cache,
                gate,
                selected,
                indices,
                partial,
                lse,
                flags,
                epochs,
                outputs,
            ) in banks:
                extension.run(
                    scores,
                    lengths,
                    queries,
                    cache,
                    table,
                    positions,
                    gate,
                    selected[1],
                    partial,
                    lse,
                    flags,
                    epochs,
                    outputs[1],
                )

        control()
        candidate()
        torch.cuda.synchronize()
        errors = [
            {
                "selected_ids_equal": bool(torch.equal(bank[4][0], bank[4][1])),
                "max_abs": float(
                    (bank[10][0].float() - bank[10][1].float()).abs().max()
                ),
                "rel_l2": float(
                    (bank[10][0].float() - bank[10][1].float()).norm()
                    / bank[10][0].float().norm()
                ),
            }
            for bank in banks
        ]
        graphs = {}
        for name, fn in (("control", control), ("candidate", candidate)):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                fn()
            graphs[name] = graph
        samples = {name: [] for name in graphs}
        for rep in range(7):
            for name in list(graphs)[:: 1 if rep % 2 == 0 else -1]:
                for _ in range(10):
                    graphs[name].replay()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(50):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 50)
        counts = {
            name: graph_kernel_geometry(graph, args.output, f"M{m}.{name}")
            for name, graph in graphs.items()
        }
        report["widths"].append(
            dict(
                m=m,
                errors=errors,
                samples_ms=samples,
                medians_ms={n: statistics.median(v) for n, v in samples.items()},
                measured_kernel_counts=counts,
            )
        )

    for m in (1, 5):
        screen(m)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
