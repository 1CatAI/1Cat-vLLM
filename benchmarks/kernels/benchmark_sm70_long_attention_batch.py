# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare shipped long q8 attention with the ordinary request-major kernel.

Use a separate KV working set per layer and report kernel-only eager/graph
latency. These measurements do not represent model-forward or decode speed.
"""

import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--page", type=int, default=3296)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda

    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    old = flash_attn_v100_cuda.grouped_e4m3_fp32_paged_fwd
    new = torch.ops._vllm_fa2_C.sm70_grouped_long_fwd
    assert torch.cuda.get_device_capability() == (7, 0)
    assert torch.ops._vllm_fa2_C.sm70_grouped_long_max_batch_size() >= max(args.batches)
    library = Path(flash_attn_interface.__file__).parent / "_vllm_fa2_C.abi3.so"
    report = {
        "scope": "attention kernels only; independent KV working sets per layer",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "candidate_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(
            Path(flash_attn_v100_cuda.__file__).read_bytes()
        ).hexdigest(),
        "page": args.page,
        "layers": args.layers,
        "results": [],
    }
    torch.manual_seed(20260927)
    for length in args.contexts:
        for batch in args.batches:
            pages = (length + args.page - 1) // args.page
            cases = []
            for _ in range(args.layers):
                storage = (
                    torch.randn(
                        batch * pages,
                        2,
                        args.page,
                        1,
                        256,
                        device="cuda",
                        dtype=torch.float16,
                    )
                    .to(torch.float8_e4m3fn)
                    .view(torch.uint8)
                )
                k, v = storage.unbind(1)
                cases.append(
                    (
                        torch.randn(
                            batch * 8, 6, 256, device="cuda", dtype=torch.float16
                        ),
                        k,
                        v,
                        torch.randperm(batch * pages, device="cuda")
                        .int()
                        .reshape(batch, pages),
                    )
                )
            lengths = torch.arange(
                length - 7, length + 1, device="cuda", dtype=torch.int32
            ).repeat(batch)
            prefix = () if batch == 1 else (batch,)
            partial = torch.empty((*prefix, 80, 8, 6, 256), device="cuda")
            lse = torch.empty((*prefix, 80, 8, 6, 2), device="cuda")
            outputs = [torch.empty_like(cases[0][0]) for _ in range(2)]

            def run(
                operator,
                arm,
                cases=cases,
                outputs=outputs,
                lengths=lengths,
                partial=partial,
                lse=lse,
            ):
                for q, k, v, table in cases:
                    operator(
                        q,
                        k,
                        v,
                        outputs[arm],
                        table,
                        lengths,
                        partial,
                        lse,
                        0.0625,
                        1.0,
                        1.0,
                    )

            graphs = []
            for arm, op in enumerate((old, new)):
                run(op, arm)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(op, arm)
                graphs.append(graph)
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0.03, atol=3e-4)
            row = {
                "context": length,
                "batch": batch,
                "max_abs_difference": float((outputs[1] - outputs[0]).abs().max()),
                "equal_fraction": float((outputs[1] == outputs[0]).float().mean()),
            }
            for mode in ("eager", "graph"):
                samples = [[], []]
                for rep in range(5):
                    for arm in [0, 1] if rep % 2 == 0 else [1, 0]:
                        call = (
                            graphs[arm].replay
                            if mode == "graph"
                            else (lambda arm=arm, run=run: run((old, new)[arm], arm))
                        )
                        for _ in range(2):
                            call()
                        start, end = (
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        )
                        start.record()
                        for _ in range(8):
                            call()
                        end.record()
                        end.synchronize()
                        samples[arm].append(start.elapsed_time(end) / 8)
                medians = [statistics.median(values) for values in samples]
                row[mode] = {
                    "baseline_ms": medians[0],
                    "candidate_ms": medians[1],
                    "latency_reduction_pct": 100 * (1 - medians[1] / medians[0]),
                    "samples_ms": samples,
                }
            report["results"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            # Release graph references as well as the last loop's KV views.
            del graphs, graph, run, call, cases, storage, k, v
            del partial, lse, outputs, lengths
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
