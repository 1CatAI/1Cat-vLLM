# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise and cold-L2 checks for V4.1 MXFP4 kernels."""

import argparse
import functools
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from vllm.models.deepseek_v41 import native

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--save", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--match-groups", action="store_true")
args = parser.parse_args()
torch.manual_seed(832)
native.load_v41sk()
flush_buffer = torch.randn(16 * 1024 * 1024, device="cuda")
flush_result = torch.empty((), device="cuda")
digests = {}
benchmarks = {}
cases = 0


def digest(tensor):
    return hashlib.sha256(
        tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def run_case(name, tokens, experts, columns, inner, split, decode_chunk=None):
    global cases
    top_k = 6
    slots = tokens * top_k
    distinct = min(experts, 30)
    ids = torch.arange(slots, device="cuda").reshape(tokens, top_k) % distinct
    permutation, groups, offsets = native._route_plan_torch(ids)
    codes = torch.randint(
        0, 256, (experts, columns, inner // 2), dtype=torch.uint8, device="cuda"
    )
    scales = torch.randint(
        116,
        121,
        (experts, columns, inner // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    global_scales = torch.ones(experts, device="cuda")
    slot_major = name.startswith("w2")
    x = (
        torch.randn(slots if slot_major else tokens, inner, device="cuda") * 0.2
    ).half()
    reference = torch.empty(slots, columns, dtype=torch.float16, device="cuda")
    prefix = (x, codes, scales, global_scales, permutation, groups, offsets, top_k)
    if decode_chunk is None:
        suffix = (slot_major, tokens, split, 1, 1)
        operation = torch.ops.v41sk.skinny_moe_qpn_sm70
    else:
        suffix = (slot_major, split, decode_chunk)
        operation = torch.ops.v41sk.skinny_moe_dec_sm70
    old = functools.partial(operation, *prefix, reference, *suffix)
    torch.sum(flush_buffer, dim=0, out=flush_result)
    old()
    key = f"{name}/T{tokens}/N{columns}/K{inner}/S{split}/C{decode_chunk}"
    digests[key] = digest(reference)
    cases += 1
    if args.match_groups and tokens <= 8:
        alternate_ids = (ids * 7 + 13) % distinct
        alternate_plan = native._route_plan_torch(alternate_ids)
        logits = torch.full((tokens, 128), -100.0, device="cuda")
        logits.scatter_(
            1,
            alternate_ids,
            (64.0 - torch.arange(top_k, device="cuda").float()).expand(tokens, -1),
        )
        probabilities = torch.empty(tokens, top_k, device="cuda")
        selected = torch.empty(tokens, top_k, dtype=torch.int32, device="cuda")
        matched_plan = tuple(torch.empty_like(tensor) for tensor in alternate_plan)
        torch.sum(flush_buffer, dim=0, out=flush_result)
        torch.ops.v41sk.route_plan(
            logits,
            torch.zeros(128, device="cuda"),
            probabilities,
            selected,
            *matched_plan,
            1.5,
            True,
        )
        assert torch.equal(selected, alternate_ids), key
        before, after = torch.empty_like(reference), torch.empty_like(reference)
        for plan, destination in ((alternate_plan, before), (matched_plan, after)):
            torch.sum(flush_buffer, dim=0, out=flush_result)
            operation(
                x, codes, scales, global_scales, *plan, top_k, destination, *suffix
            )
        assert torch.equal(before.view(torch.int16), after.view(torch.int16)), key
        cases += 1
    if tokens == 6 and columns in (1152, 5120):
        benchmarks[key + "/original"] = old


for tokens in (1, 6, 8, 64, 128):
    for inner, split in (
        (96, 6),
        (128, 4),
        (288, 6),
        (576, 6),
        (576, 9),
        (576, 12),
        (576, 18),
        (320, 10),
    ):
        run_case("qpn", tokens, 8, 64, inner, split)
    for split, chunk in ((4, 2), (4, 4), (4, 8)):
        run_case("decode", tokens, 8, 64, 192, split, chunk)
for tokens in (1, 6, 8, 64):
    run_case("w13", tokens, 64, 1152, 5120, 16)
    run_case("w13_decode", tokens, 64, 1152, 5120, 4, 8)
    run_case("w2", tokens, 64, 5120, 576, 6)

if args.save:
    args.reference.write_text(json.dumps(digests, indent=2) + "\n")
else:
    expected = json.loads(args.reference.read_text())
    assert expected == digests, [
        key for key in digests if expected.get(key) != digests[key]
    ]
print(
    json.dumps({"bitwise_cases": cases, "status": "saved" if args.save else "passed"}),
    flush=True,
)

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
