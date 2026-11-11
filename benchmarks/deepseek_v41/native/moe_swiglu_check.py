# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise checks and cold-L2 timing of the routed gate/up/SwiGLU pipeline."""

import argparse
import functools
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from vllm.models.deepseek_v41 import native

parser = argparse.ArgumentParser()
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
torch.manual_seed(2729)
native.load_v41sk()
flush_buffer = torch.randn(16 * 1024 * 1024, device="cuda")
flush_result = torch.empty((), device="cuda")
benchmarks = {}
cases = 0


def flush():
    torch.sum(flush_buffer, dim=0, out=flush_result)


def run_case(tokens, experts, intermediate, top_k, clipped=False):
    global cases
    inner = 5120
    slots = tokens * top_k
    if experts == 1:
        ids = torch.zeros(tokens, top_k, dtype=torch.int64, device="cuda")
    else:
        ids = torch.stack(
            [torch.randperm(experts, device="cuda")[:top_k] for _ in range(tokens)]
        )
    permutation, groups, offsets = native._route_plan_torch(ids)
    codes = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, inner // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    scales = torch.randint(
        123 if clipped else 116,
        127 if clipped else 121,
        (experts, 2 * intermediate, inner // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    global_scales = torch.exp2((torch.arange(experts, device="cuda") % 3 - 1).float())
    x = (torch.randn(tokens, inner, device="cuda") * 0.2).half()
    probabilities = torch.rand(tokens, top_k, device="cuda")
    probabilities = (
        probabilities / probabilities.sum(-1, keepdim=True) * 1.5
    ).flatten()
    probabilities[0] = 0
    gate_up = torch.empty(slots, 2 * intermediate, device="cuda", dtype=torch.float16)
    reference = torch.empty(slots, intermediate, device="cuda", dtype=torch.float16)
    prefix = (x, codes, scales, global_scales)
    plan = (permutation, groups, offsets, top_k)

    def original():
        torch.ops.v41sk.skinny_moe_dec_sm70(*prefix, *plan, gate_up, False, 4, 8)
        native._swiglu_prob_kernel[(slots, (intermediate + 255) // 256)](
            gate_up,
            probabilities,
            reference,
            INTER=intermediate,
            BLOCK=256,
            num_warps=4,
        )

    flush()
    original()
    keep = tokens == 6 and experts == 384 and intermediate == 576
    label = f"T{tokens}/E{experts}/I{intermediate}/K{top_k}/clipped{clipped}"
    if keep:
        benchmarks[label + "/original"] = original
    for chunk in (2, 4, 8):
        output = torch.empty_like(reference)
        fused = functools.partial(
            torch.ops.v41sk.moe_w13_swiglu, *prefix, probabilities, *plan, output, chunk
        )
        flush()
        fused()
        equal = torch.equal(reference.view(torch.int16), output.view(torch.int16))
        if not equal:
            difference = (reference.float() - output.float()).abs()
            raise AssertionError(
                (
                    label,
                    chunk,
                    int(torch.count_nonzero(difference)),
                    float(difference.max()),
                )
            )
        cases += 1
        if keep:
            benchmarks[label + f"/fused_X{chunk}"] = fused


for tokens in (1, 5, 6, 8):
    for experts, intermediate, top_k in ((1, 32, 6), (8, 64, 3), (64, 576, 6)):
        for clipped in (False, True):
            run_case(tokens, experts, intermediate, top_k, clipped)
run_case(6, 384, 576, 6, True)

# Exercise the normal MoE entry point, including the prefill fallback and
# repeated graph replays with changing activations.
layer = SimpleNamespace(
    w13_weight=torch.randint(0, 256, (8, 1152, 2560), device="cuda", dtype=torch.uint8),
    w13_weight_scale=torch.randint(
        116, 121, (8, 1152, 160), device="cuda", dtype=torch.uint8
    ),
    v41_g13=torch.ones(8, device="cuda"),
    w2_weight=torch.randint(0, 256, (8, 5120, 288), device="cuda", dtype=torch.uint8),
    w2_weight_scale=torch.randint(
        116, 121, (8, 5120, 18), device="cuda", dtype=torch.uint8
    ),
    v41_g2=torch.ones(8, device="cuda"),
    v41_split13=4,
    v41_split2=6,
)
native._MOE_DEC = True
native._W13_UNITS = None
native._W2_ROUTE = True
for tokens in (1, 6, 8, 9):
    ids = torch.stack([torch.randperm(8, device="cuda")[:6] for _ in range(tokens)])
    layer.v41_plan = (ids, native._route_plan_torch(ids))
    x = (torch.randn(tokens, 5120, device="cuda") * 0.2).half()
    probabilities = torch.softmax(torch.randn(tokens, 6, device="cuda"), -1) * 1.5
    graphs, outputs = [], []
    for enabled in (False, True):
        native._W13_SWIGLU = enabled
        flush()
        native.V41NativeMoEMethod.apply(None, layer, x, probabilities, ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flush()
            output = native.V41NativeMoEMethod.apply(None, layer, x, probabilities, ids)
        graphs.append(graph)
        outputs.append(output)
    for _ in range(3):
        x.copy_((torch.randn_like(x) * 0.2).half())
        for graph in graphs:
            graph.replay()
        assert torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32))
        cases += 1
    for graph in graphs:
        graph.reset()
native._W13_SWIGLU = False

print(json.dumps({"bitwise_cases": cases, "status": "passed"}), flush=True)

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
