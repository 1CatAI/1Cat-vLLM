# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen grouped latent MLA workload; run controls before candidate codecs."""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm.models.glm5next.sm70.fp8_kv import (
    sm70_glm5_kv_insert,
    sm70_glm5_sparse_attention_paged,
    sm70_glm5_sparse_attention_paged_batched_gemm,
    sm70_glm5_sparse_attention_paged_gemm,
)
from vllm.v1.attention.kv_cache_view import MLAPageLayout
from vllm.v1.attention.kv_codecs import resolve_kv_codec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", choices=["fp8_e4m3", "int8"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(20261012)
    codec = resolve_kv_codec(args.codec, representation="mla_latent")
    assert codec is not None and codec.scale_dtype is not None
    page, tokens, width, heads, dim = 64, 4096, 2048, 16, 512
    layout = MLAPageLayout(dim, scale_bytes=codec.scale_dtype.itemsize)
    carrier = torch.zeros(
        (tokens // page, page, layout.slot_bytes),
        dtype=codec.storage_dtype,
        device="cuda",
    )
    values = torch.randn((tokens, dim), device="cuda", dtype=torch.float16) * 0.2
    values *= torch.arange(1, 9, device="cuda").repeat_interleave(64)[None, :]
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    sm70_glm5_kv_insert(values, carrier, slots, kv_cache_dtype=codec.name)
    results = []

    def measure(name, launch, out):
        for _ in range(30):
            launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for _ in range(2000):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(9):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(1000):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 1000)
        results.append(
            dict(
                algorithm=name,
                median_ms=statistics.median(samples),
                samples_ms=samples,
                output_sha256=hashlib.sha256(
                    out.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest(),
            )
        )

    for count in (1, 32):
        measure(
            f"writer-q{count}",
            partial(
                sm70_glm5_kv_insert,
                values[:count],
                carrier,
                slots[:count],
                kv_cache_dtype=codec.name,
            ),
            carrier,
        )
    for algorithm, rows in [("paged", 1), ("paged", 9), ("gemm", 1), ("bmm", 8)]:
        query = (
            torch.randn((rows, heads, dim), device="cuda", dtype=torch.float16) * 0.2
        )
        indices = torch.randperm(tokens, device="cuda")[:width].int().repeat(rows, 1)
        lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
        output = torch.empty_like(query)
        gathered = torch.empty((rows, width, dim), device="cuda", dtype=torch.float16)
        scores = torch.empty((rows, heads, width), device="cuda", dtype=torch.float16)
        probs = torch.empty_like(scores)
        if algorithm == "paged":
            launch = partial(
                sm70_glm5_sparse_attention_paged,
                query,
                carrier,
                indices,
                lengths,
                dim**-0.5,
                output,
                kv_cache_dtype=codec.name,
            )
        elif algorithm == "gemm":
            launch = partial(
                sm70_glm5_sparse_attention_paged_gemm,
                query,
                carrier,
                indices,
                lengths,
                dim**-0.5,
                output,
                gathered[0],
                scores[0],
                probs[0],
                kv_cache_dtype=codec.name,
            )
        else:
            launch = partial(
                sm70_glm5_sparse_attention_paged_batched_gemm,
                query,
                carrier,
                indices,
                lengths,
                dim**-0.5,
                output,
                gathered,
                scores,
                probs,
                kv_cache_dtype=codec.name,
            )
        measure(f"{algorithm}-q{rows}", launch, output)
    record = dict(
        codec=codec.name,
        geometry=dict(
            cache_tokens=tokens,
            selected_tokens=width,
            page_tokens=page,
            heads=heads,
            latent_dim=dim,
        ),
        protocol=dict(ordinary_warmups=30, graph_warmups=2000, repeats=9, replays=1000),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results=results,
    )
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    print([(r["algorithm"], r["median_ms"]) for r in results], flush=True)


if __name__ == "__main__":
    main()
