# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen address resolution plus unchanged sparse attention/merge, M1 TP4."""

import argparse
import json
from functools import partial
from pathlib import Path
from statistics import median

import torch
from verify_sm70_qsa_router_exact import capture

from vllm.models.qwen4_exp.nvidia.ops import qsa

ORIGINAL_GATE = qsa._use_sm70_qsa_resolved_indices


def paired(ga, gb):
    for _ in range(2500):
        ga.replay()
        gb.replay()
    torch.accelerator.synchronize()
    values = {"control": [], "resolved": []}
    for turn in range(8):
        pairs = [("control", ga), ("resolved", gb)]
        if turn % 2:
            pairs.reverse()
        for label, graph in pairs:
            for _ in range(20):
                graph.replay()
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
            for _ in range(100):
                graph.replay()
            b.record()
            b.synchronize()
            values[label].append(a.elapsed_time(b) / 100)
    return {
        "samples_ms": values,
        "median_ms": {k: median(v) for k, v in values.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--interleaved-kv", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    torch.accelerator.set_device_index(0)
    torch.manual_seed(20260905)
    qsa._SM70_QSA_XQA_PAGE4 = False
    results = []
    for context in (8192, 32768, 262144):
        page, layers, width = 400, 12, 2051
        blocks = (context + page - 1) // page
        if args.interleaved_kv:
            # Match the worker ABI: [blocks, 2, page, KV heads, head dim].
            kv = torch.randn(
                layers, blocks, 2, page, 1, 256, device="cuda", dtype=torch.float16
            )
            k, v = kv.unbind(2)
        else:
            kv = None
            k = torch.randn(
                layers, blocks, page, 1, 256, device="cuda", dtype=torch.float16
            )
            v = torch.randn_like(k)
        queries = torch.randn(layers, 1, 6, 256, device="cuda", dtype=torch.float16)
        gates = torch.randn_like(queries)
        indices = torch.randint(
            context, (layers, 1, width), device="cuda", dtype=torch.int32
        )
        tables = torch.stack(
            [
                torch.randperm(blocks, device="cuda", dtype=torch.int32)
                for _ in range(layers)
            ]
        ).view(layers, 1, blocks)
        requests = torch.zeros(layers, 1, device="cuda", dtype=torch.int32)
        a, b = torch.empty_like(queries), torch.empty_like(queries)

        def run(
            output,
            resolved,
            state=(queries, k, v, indices, tables, requests, gates),
            layers=layers,
        ):
            queries, k, v, indices, tables, requests, gates = state
            qsa._use_sm70_qsa_resolved_indices = (
                ORIGINAL_GATE if resolved else lambda *args: False
            )
            try:
                for i in range(layers):
                    qsa.qsa_sparse_paged_attention(
                        queries[i],
                        k[i],
                        v[i],
                        indices[i],
                        tables[i],
                        requests[i],
                        out=output[i],
                        output_gate=gates[i],
                    )
            finally:
                qsa._use_sm70_qsa_resolved_indices = ORIGINAL_GATE

        ga, gb = capture(partial(run, a, False)), capture(partial(run, b, True))
        for scenario in range(8):
            indices.random_(context)
            queries.normal_()
            tables.copy_(tables.roll(1, dims=-1))
            requests.zero_()
            if scenario == 1:
                indices[:, :, ::7] = -1
            if scenario == 2:
                indices[:, :, ::7] = blocks * page
            if scenario == 3:
                tables[:, :, 0] = -1
            if scenario == 4:
                tables[:, :, 1] = blocks
            if scenario == 5:
                indices.zero_()
            if scenario == 6:
                requests.fill_(-1)
            if scenario == 7:
                requests.fill_(1)
            b.fill_(float("nan"))
            ga.replay()
            gb.replay()
            torch.accelerator.synchronize()
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (
                context,
                scenario,
                (a - b).abs().max().item(),
            )
        # Restore valid, varied metadata for timing, not the all-invalid case.
        requests.zero_()
        indices.random_(context)
        tables.copy_(
            torch.stack(
                [
                    torch.randperm(blocks, device="cuda", dtype=torch.int32)
                    for _ in range(layers)
                ]
            ).view_as(tables)
        )
        result = {
            "context": context,
            "layers": layers,
            "page_size": page,
            "cache_layout": "interleaved_kv" if args.interleaved_kv else "separate_kv",
            "key_strides": list(k[0].stride()),
            "bitwise_graph_scenarios": 8,
            **({} if args.skip_timing else paired(ga, gb)),
        }
        results.append(result)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        del ga, gb, run, kv, k, v, queries, gates, indices, tables, requests, a, b
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
