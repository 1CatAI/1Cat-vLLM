# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare scalar q1 native workspaces and a sixteen-layer KV working set."""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_sm70_grouped_attention_long import (
    byte_equal,
    load_operator,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference-library", type=Path, required=True)
parser.add_argument("--reference-sha256", required=True)
parser.add_argument("--candidate", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--correctness-only", action="store_true")
parser.add_argument("--sanitizer", action="store_true")
args = parser.parse_args()
library = args.reference_library.resolve()
assert hashlib.sha256(library.read_bytes()).hexdigest() == args.reference_sha256
assert torch.cuda.get_device_capability() == (7, 0)
spec = importlib.util.spec_from_file_location("flash_attn_v100_cuda", library)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
assert Path(native.__file__).resolve() == library
loaded = [load_operator(p) for p in args.candidate]
operators = [None] + [r[0] for r in loaded]
report = dict(
    scope="Scalar q1 operator screen; no full-model admission",
    frozen_library=dict(library=str(library), sha256=args.reference_sha256),
    manifests=[r[1] for r in loaded],
    checks=[],
    performance=[],
    complete=False,
)
out = args.output
assert not out.exists()


def make_case(length, padding=0):
    page = 3296
    pages = (length + page - 1) // page
    raw = torch.randn((pages, 2, page, 1, 256), dtype=torch.float16, device="cuda")
    backing = torch.empty(
        (pages, 2, page, 1, 256 + padding), dtype=torch.uint8, device="cuda"
    )
    backing[..., :256].copy_(raw.to(torch.float8_e4m3fn).view(torch.uint8))
    k, v = backing[..., :256].unbind(1)
    q = torch.randn((1, 6, 256), dtype=torch.float16, device="cuda") * 0.5
    table = torch.randperm(pages, device="cuda").int()[None].contiguous()
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    active = torch.full((1,), 256, dtype=torch.int32, device="cuda")
    arms = []
    for _ in operators:
        guard = torch.full((3, 6, 256), -777.0, dtype=torch.float16, device="cuda")
        arms.append(
            dict(
                guard=guard,
                out=guard[1:2],
                partial=torch.full((1, 6, 256, 256), -777.0, device="cuda"),
                maximum=torch.full((1, 6, 256), -777.0, device="cuda"),
                sums=torch.full((1, 6, 256), -777.0, device="cuda"),
            )
        )
    return dict(q=q, k=k, v=v, table=table, lengths=lengths, active=active, arms=arms)


def run(case, arm):
    r = case["arms"][arm]
    args = [
        case["q"],
        case["k"],
        case["v"],
        r["out"],
        case["table"],
        case["lengths"],
        r["partial"],
        r["maximum"],
        r["sums"],
        case["active"],
    ]
    if arm == 0:
        native.decode_paged_fwd(
            *args, 0.0625, 1024, 256, "fp8_e4m3", 0.5, 1.25, -1, -1, None, 0
        )
    else:
        operators[arm](*args, 0.0625, 0.5, 1.25)


torch.manual_seed(20260910)
cases_to_check = [(1024, 0), (3295, 0), (3297, 8), (131072, 0), (262144, 8)]
if args.sanitizer:
    cases_to_check = [(3297, 8), (262144, 0)]
for length, padding in cases_to_check:
    case = make_case(length, padding)
    graphs = []
    for arm in range(len(operators)):
        run(case, arm)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(case, arm)
        graphs.append(graph)
    for state in [length, 0, length]:
        case["lengths"].fill_(state)
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        for arm in range(1, len(operators)):
            exact = {
                key: byte_equal(case["arms"][0][key], case["arms"][arm][key])
                for key in ["out", "partial", "maximum", "sums"]
            }
            guards = bool((case["arms"][arm]["guard"][[0, 2]] == -777).all())
            row = dict(
                length=length,
                padding=padding,
                state=state,
                arm=arm,
                byte_exact=exact,
                guards=guards,
            )
            report["checks"].append(row)
            out.write_text(json.dumps(report, indent=2) + "\n")
            assert all(exact.values()) and guards, row
    print("SCALAR_Q1_EXACT", length, padding, flush=True)
    del graphs, case
for length in (
    []
    if args.correctness_only or args.sanitizer
    else [1024, 32768, 65536, 131072, 261888]
):
    cases = [make_case(length) for _ in range(16)]
    graphs = []
    for arm in range(len(operators)):
        for case in cases:
            run(case, arm)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for case in cases:
                run(case, arm)
        graphs.append(graph)
    for _ in range(3):
        for graph in graphs:
            graph.replay()
    torch.cuda.synchronize()
    samples = [[] for _ in operators]
    for repeat in range(5):
        for arm in (
            range(len(operators))
            if repeat % 2 == 0
            else reversed(range(len(operators)))
        ):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(4):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) / 4)
    row = dict(
        length=length,
        layers=16,
        samples_ms=samples,
        median_ms=[statistics.median(x) for x in samples],
    )
    report["performance"].append(row)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("SCALAR_Q1_PERFORMANCE", row, flush=True)
    del graphs, cases
report["complete"] = True
out.write_text(json.dumps(report, indent=2) + "\n")
