# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise regression and cold-L2 graph timings for V4.1 decode kernels."""

import argparse
import functools
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ["VLLM_V41_H16"] = "1"
from vllm.models.deepseek_v41 import native

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--save", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--match-groups", action="store_true")
args = parser.parse_args()
torch.manual_seed(1729)
native._H16 = True
native.load_v41sk()
device = "cuda"
flush_buffer = torch.randn(16 * 1024 * 1024, device=device)
flush_result = torch.empty((), device=device)
digests = {}
benchmarks = {}


def flush():
    torch.sum(flush_buffer, dim=0, out=flush_result)


def check(name, operation, outputs):
    flush()
    operation()
    digest = hashlib.sha256()
    values = [tensor.detach().contiguous().cpu() for tensor in outputs]
    if name.startswith(("route/", "gate/")):
        ids = values[-4].flatten()
        permutation, experts, offsets = values[-3:]
        slots = ids.numel()
        assert torch.equal(permutation.sort().values, torch.arange(slots)), name
        assert offsets[0] == 0 and offsets[-1] == slots, name
        assert bool(torch.all(offsets[1:] >= offsets[:-1])), name
        groups = int(torch.count_nonzero(offsets[:-1] < slots))
        assert experts[:groups].unique().numel() == groups, name
        assert bool(torch.all(experts[groups:] == 0)), name
        for group in range(groups):
            begin, end = int(offsets[group]), int(offsets[group + 1])
            rows = permutation[begin:end]
            assert begin < end and bool(torch.all(ids[rows] == experts[group])), name
            assert bool(torch.all(rows[1:] > rows[:-1])), name
        # Group order is internal to the expert kernels. Canonicalize it for
        # the original golden digest while validating stable slots above.
        ordered = ids.argsort(stable=True)
        unique, counts = ids[ordered].unique_consecutive(return_counts=True)
        canonical_experts = torch.zeros_like(experts)
        canonical_experts[:groups] = unique
        canonical_offsets = torch.full_like(offsets, slots)
        canonical_offsets[0] = 0
        canonical_offsets[1 : groups + 1] = counts.cumsum(0)
        values[-3:] = [
            ordered.to(permutation.dtype),
            canonical_experts,
            canonical_offsets,
        ]
    for value in values:
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    digests[name] = digest.hexdigest()


def route_case(logits, bias, top_k):
    tokens = logits.shape[0]
    slots = tokens * top_k
    probabilities = torch.empty(tokens, top_k, device=device)
    ids = torch.empty(tokens, top_k, dtype=torch.int32, device=device)
    permutation = torch.empty(slots, dtype=torch.int32, device=device)
    experts = torch.empty_like(permutation)
    offsets = torch.empty(slots + 1, dtype=torch.int32, device=device)
    outputs = probabilities, ids, permutation, experts, offsets
    operation = functools.partial(
        torch.ops.v41sk.route_plan, logits, bias, *outputs, 1.5, args.match_groups
    )
    return operation, outputs


for experts in (128, 256, 384):
    for tokens in range(1, 9):
        for top_k in range(1, 9):
            if tokens * top_k >= 64:
                continue
            for trial in range(4):
                logits = torch.randn(tokens, experts, device=device) * 3
                bias = torch.randn(experts, device=device) * 0.05
                if trial == 1:
                    logits = logits.half().float()
                elif trial == 2:
                    logits = torch.randint(
                        -3, 3, (tokens, experts), device=device
                    ).float()
                    bias.zero_()
                elif trial == 3:
                    logits.fill_(25)
                    bias.zero_()
                operation, outputs = route_case(logits, bias, top_k)
                name = f"route/E{experts}/T{tokens}/K{top_k}/{trial}"
                check(name, operation, outputs)
                if tokens == 6 and top_k == 6 and trial == 0:
                    benchmarks[f"route_E{experts}_T6_K6"] = operation

for experts, top_k in ((384, 6), (128, 3)):
    gate = SimpleNamespace(
        weight=(torch.randn(experts, 5120, device=device) * 0.02).bfloat16(),
        bias=torch.randn(experts, device=device) * 0.05,
        temperature=1.0,
    )
    assert native.attach_h16_gate(gate)
    packed, tickets, output_scale = gate.v41_h16
    for tokens in range(1, 9):
        x = torch.randn(tokens, 5120, device=device).half()
        logits = torch.empty(tokens, experts, device=device)
        _, outputs = route_case(logits, gate.bias, top_k)
        operation = functools.partial(
            torch.ops.v41sk.gate_route,
            logits,
            x,
            packed,
            tickets,
            gate.bias,
            *outputs,
            5,
            1.5,
            output_scale,
            args.match_groups,
        )
        check(f"gate/E{experts}/T{tokens}", operation, (logits, *outputs))
        assert torch.count_nonzero(tickets).item() == 0
        if tokens in (1, 6):
            benchmarks[f"gate_E{experts}_T{tokens}"] = operation

for tokens in (1, 5, 6, 8):
    for width in (33, 128, 133, 640):
        q = (torch.randn(tokens, 16, 512, device=device) * 0.3).half()
        keys = (torch.randn(tokens, width, 512, device=device) * 0.5).half()
        unused = torch.rand(tokens, width, device=device) < 0.2
        unused[:, min(width, 203) :] = True
        unused[-1] = True
        sink = torch.randn(16, device=device)
        output = torch.empty_like(q)
        operation = functools.partial(
            torch.ops.v41sk.attn_dec, output, q, keys, unused, sink, 512**-0.5
        )
        check(f"attn/T{tokens}/W{width}", operation, (output,))
        if tokens == 6:
            benchmarks[f"attn_T6_W{width}"] = operation

for tokens in (1, 6, 8):
    for position, ratio, top_k, block in (
        (0, 4, 0, 16),
        (90, 4, 0, 16),
        (300, 4, 512, 16),
        (3000, 4, 512, 16),
        (300, 128, 512, 128),
        (12000, 1, 512, 16),
    ):
        positions = torch.arange(
            position, position + tokens, dtype=torch.int64, device=device
        )
        valid = torch.ones(tokens, dtype=torch.bool, device=device)
        valid[-1] = False
        q = (torch.randn(tokens, 16, 512, device=device) * 0.3).half()
        ring = (torch.randn(256, 512, device=device) * 0.5).half()
        values = (torch.randn(tokens, 512, device=device) * 0.5).half()
        per_page = block // ratio
        compressed = (position + tokens) // ratio + 1
        pages = (compressed + per_page - 1) // per_page + 2
        cache = (torch.randn(pages, per_page * 512, device=device) * 0.5).half()
        table = torch.randperm(pages, device=device).int()
        indices = torch.full((tokens, top_k), -1, dtype=torch.int32, device=device)
        for token in range(tokens):
            live = (position + token + 1) // ratio
            count = min(live, top_k)
            indices[token, :count] = (
                torch.randperm(live, device=device)[:count].sort().values.int()
            )
        sink = torch.randn(16, device=device)
        output = torch.empty_like(q)
        operation = functools.partial(
            torch.ops.v41sk.attn_ring_dec,
            output,
            q,
            ring,
            values,
            positions,
            cache,
            table,
            positions,
            valid,
            indices,
            sink,
            512**-0.5,
            block,
            ratio,
            128,
            top_k,
        )
        name = f"ring/T{tokens}/P{position}/R{ratio}/K{top_k}"
        check(name, operation, (output,))
        if tokens == 6:
            benchmarks[f"ring_T6_P{position}_R{ratio}"] = operation

if args.save:
    args.reference.write_text(json.dumps(digests, indent=2) + "\n")
else:
    expected = json.loads(args.reference.read_text())
    failures = [
        name for name, digest in digests.items() if expected.get(name) != digest
    ]
    if failures or set(expected) != set(digests):
        raise AssertionError(f"Bitwise mismatches: {failures[:20]}")
print(
    json.dumps(
        {"bitwise_cases": len(digests), "status": "saved" if args.save else "passed"}
    ),
    flush=True,
)

if args.profile:
    torch.cuda.cudart().cudaProfilerStart()
for name, operation in benchmarks.items():
    for _ in range(3):
        flush()
        operation()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    finish = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flush()
        start.record()
        operation()
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
