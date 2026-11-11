# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the fused decode attention epilogue against the existing FP16 path."""

import argparse
import functools
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from vllm.models.deepseek_v41.attn_fuse import attn_post
from vllm.models.deepseek_v41.native import load_v41sk
from vllm.models.deepseek_v41.quant_rope import rope_table

parser = argparse.ArgumentParser()
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
torch.manual_seed(411)
load_v41sk()
device = "cuda"
flush_buffer = torch.randn(16 * 1024 * 1024, device=device)
flush_result = torch.empty((), device=device)
benchmarks = {}
cases = 0
tables = {False: rope_table(), True: rope_table(compressed=True)}
short_table = rope_table(16)

for tokens in (1, 2, 3, 4, 5, 6, 8):
    for position, ratio, count, block in (
        (0, 4, 0, 16),
        (90, 4, 0, 16),
        (255, 4, 512, 16),
        (300, 4, 512, 16),
        (3000, 4, 512, 16),
        (300, 128, 512, 128),
        (12000, 1, 512, 16),
        (16370, 1, 512, 16),
    ):
        for invalid in (False, True):
            positions = torch.arange(
                position, position + tokens, dtype=torch.int64, device=device
            )
            valid = torch.ones(tokens, dtype=torch.bool, device=device)
            if invalid:
                valid[-1] = False
            query = (torch.randn(tokens, 16, 512, device=device) * 0.3).half()
            ring = (torch.randn(256, 512, device=device) * 0.5).half()
            fused_ring = ring.clone()
            values = (torch.randn(tokens, 512, device=device) * 0.5).half()
            rows_per_page = block // ratio
            compressed = (position + tokens) // ratio + 1
            pages = (compressed + rows_per_page - 1) // rows_per_page + 2
            cache = (
                torch.randn(pages, rows_per_page * 512, device=device) * 0.5
            ).half()
            page_table = torch.randperm(pages, device=device).int()
            indices = torch.full((tokens, count), -1, dtype=torch.int32, device=device)
            for token in range(tokens):
                live = (position + token + 1) // ratio
                selected = min(live, count)
                indices[token, :selected] = (
                    torch.randperm(live, device=device)[:selected].sort().values.int()
                )
            sink = torch.randn(16, device=device)
            raw = torch.empty_like(query)
            fused = torch.empty_like(query)
            table = tables[count != 0]
            common = (
                values,
                positions,
                cache,
                page_table,
                positions,
                valid,
                indices,
                sink,
                512**-0.5,
                block,
                ratio,
                128,
                count,
            )
            fused_operation = functools.partial(
                torch.ops.v41sk.attn_ring_dec,
                fused,
                query,
                fused_ring,
                *common,
                table,
            )

            def reference_operation(
                raw=raw,
                query=query,
                ring=ring,
                common=common,
                table=table,
                positions=positions,
                values=values,
                valid=valid,
            ):
                torch.ops.v41sk.attn_ring_dec(raw, query, ring, *common)
                return attn_post(raw, positions, table, 2, ring, values, valid)

            torch.sum(flush_buffer, dim=0, out=flush_result)
            reference = reference_operation().view_as(fused)
            fused_operation()
            cases += 1
            equal = torch.equal(reference.view(torch.int16), fused.view(torch.int16))
            if not equal or not torch.equal(ring, fused_ring):
                difference = (reference.float() - fused.float()).abs()
                raise AssertionError(
                    f"T={tokens} P={position} R={ratio} invalid={invalid}: "
                    f"output_equal={equal}, max_abs={difference.max().item()}, "
                    f"different={(reference != fused).sum().item()}, "
                    f"ring_equal={torch.equal(ring, fused_ring)}"
                )
            if tokens == 6 and not invalid:
                label = f"P{position}_R{ratio}"
                benchmarks[label + "_reference"] = reference_operation
                benchmarks[label + "_fused"] = fused_operation

print(json.dumps({"bitwise_cases": cases, "status": "passed"}), flush=True)
if args.profile:
    torch.cuda.cudart().cudaProfilerStart()
for name, operation in benchmarks.items():
    for _ in range(3):
        torch.sum(flush_buffer, dim=0, out=flush_result)
        operation()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    finish = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.sum(flush_buffer, dim=0, out=flush_result)
        start.record()
        output = operation()
        finish.record()
    samples = []
    torch.cuda.nvtx.range_push(name)
    for _ in range(40):
        graph.replay()
        torch.accelerator.synchronize()
        samples.append(start.elapsed_time(finish) * 1000)
    torch.cuda.nvtx.range_pop()
    print(
        json.dumps(
            {
                "case": name,
                "cold_graph_us": round(statistics.median(samples), 3),
                "p95_us": round(sorted(samples)[37], 3),
            }
        ),
        flush=True,
    )
if args.profile:
    torch.cuda.cudart().cudaProfilerStop()
