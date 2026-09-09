# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check an SM70 sparse top-k candidate against PyTorch's dense tie order.

This private operator screen uses frozen PyTorch 2.10.0, a 62080-token local
vocabulary, 64 reranked FP32 candidates, seven/eight rows and top-k 16/20/21.
Candidate IDs must be unique and in the local vocabulary. Reproduce the native
sort wrapper with build_sm70_native_sort_candidate.py. No serving route is
installed; this primitive screen does not measure a complete DFlash2 round.

The selection order follows PyTorch v2.10.0 TensorTopK.cu's multiblock gather:
all keys above the cutoff in vocabulary order, then cutoff ties in vocabulary
order. The native sort wrapper preserves the final unstable tie permutation.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def compact_gather(
    IDS,
    LOGITS,
    VALUES,
    OUT_IDS,
    K: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    VOCAB_START: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, 64)
    ids = tl.load(IDS + row * ROW_STRIDE + c)
    values = tl.load(LOGITS + row * ROW_STRIDE + c)
    bits = values.to(tl.uint32, bitcast=True)
    keys = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(
        tl.uint32
    )
    keys = tl.where(values != values, 0xFFFFFFFF, keys).to(tl.uint32)
    ordered = tl.sort(keys, descending=True)
    kth = tl.sum(tl.where(c == K - 1, ordered, 0), 0)
    above = keys > kth
    equal = keys == kth
    n_above = tl.sum(above.to(tl.int32), 0)
    before = ids[:, None] > ids[None, :]
    above_rank = tl.sum((before & above[None, :]).to(tl.int32), 1)
    equal_rank = tl.sum((before & equal[None, :]).to(tl.int32), 1)
    pos = tl.where(above, above_rank, n_above + equal_rank)
    selected = above | (equal & (pos < K))
    # If the kth value is -Inf, implicit dense background entries also tie.
    # All selected non-background values still precede those entries.
    if kth == 0x007FFFFF:
        tl.store(VALUES + row * K + pos, values, above)
        tl.store(OUT_IDS + row * K + pos, ids + VOCAB_START, above)
        low_id = c.to(tl.int64)
        prior_above = tl.sum(
            ((ids[None, :] < low_id[:, None]) & above[None, :]).to(tl.int32), 1
        )
        is_above = (
            tl.sum(((ids[None, :] == low_id[:, None]) & above[None, :]).to(tl.int32), 1)
            != 0
        )
        low_pos = n_above + c - prior_above
        keep = (c < K) & ~is_above & (low_pos < K)
        tl.store(VALUES + row * K + low_pos, -float("inf"), keep)
        tl.store(OUT_IDS + row * K + low_pos, low_id + VOCAB_START, keep)
    else:
        tl.store(VALUES + row * K + pos, values, selected)
        tl.store(OUT_IDS + row * K + pos, ids + VOCAB_START, selected)


def select(candidate_ids, logits, values, ids, vocab_start=0):
    compact_gather[(logits.shape[0],)](
        candidate_ids,
        logits,
        values,
        ids,
        K=values.shape[1],
        ROW_STRIDE=logits.stride(0),
        VOCAB_START=vocab_start,
        num_warps=4,
    )
    torch.ops.quasar_native_sort.sort_pairs(values, ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--correctness-only", action="store_true")
    a = p.parse_args()
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.ops.load_library(str(a.library))
    torch.manual_seed(20260908)
    cases = []
    width = 62080
    for rows in (7, 8):
        for k in (16, 20, 21):
            for scenario in (
                "random",
                "ties",
                "boundary_ties",
                "zeros",
                "signed_zero",
                "nan",
                "infinity",
                "negative_infinity",
                "few_finite",
            ):
                cid = torch.stack(
                    [torch.randperm(width, device="cuda")[:64] for _ in range(rows)]
                )
                logits = torch.randn(rows, 64, device="cuda")
                if scenario == "ties":
                    logits = logits.round()
                if scenario == "boundary_ties":
                    logits[:, k - 5 : k + 10] = 3.0
                if scenario in ("zeros", "signed_zero"):
                    logits.zero_()
                    if scenario == "signed_zero":
                        logits[:, ::2] = -0.0
                if scenario == "nan":
                    logits[:, ::3] = float("nan")
                if scenario == "infinity":
                    logits[:, ::3] = float("inf")
                if scenario == "negative_infinity":
                    logits.fill_(-float("inf"))
                if scenario == "few_finite":
                    logits[:, 3:] = -float("inf")
                values = [torch.empty(rows, k, device="cuda") for _ in range(2)]
                ids = [
                    torch.empty(rows, k, device="cuda", dtype=torch.int64)
                    for _ in range(2)
                ]
                dense = torch.full((rows, width), -float("inf"), device="cuda")
                dense.scatter_(1, cid, logits)
                torch.topk(dense, k, sorted=True, out=(values[0], ids[0]))
                select(cid, logits, values[1], ids[1])
                exact_values = torch.equal(
                    values[0].view(torch.uint8), values[1].view(torch.uint8)
                )
                exact_ids = torch.equal(ids[0], ids[1])
                result = dict(
                    rows=rows,
                    k=k,
                    scenario=scenario,
                    values_equal=exact_values,
                    ids_equal=exact_ids,
                )
                if not exact_values or not exact_ids:
                    result.update(
                        control_values=values[0].tolist(),
                        candidate_values=values[1].tolist(),
                        control_ids=ids[0].tolist(),
                        candidate_ids=ids[1].tolist(),
                    )
                    cases.append(result)
                    a.output.write_text(
                        json.dumps(dict(cases=cases, passed=False), indent=2)
                    )
                    raise RuntimeError(f"Dense order mismatch: {rows}, {k}, {scenario}")
                cases.append(result)
    report = dict(
        cases=cases,
        passed=True,
        library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        local_vocab_size=width,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete_round_performance=False,
    )
    if not a.correctness_only:
        rows, k = 8, 20
        cid = torch.stack(
            [torch.randperm(width, device="cuda")[:64] for _ in range(rows)]
        )
        logits = torch.randn(rows, 64, device="cuda")
        values = [torch.empty(rows, k, device="cuda") for _ in range(2)]
        ids = [torch.empty(rows, k, device="cuda", dtype=torch.int64) for _ in range(2)]
        dense = torch.empty(rows, width, device="cuda")

        def run(arm):
            if arm:
                select(cid, logits, values[1], ids[1])
            else:
                dense.fill_(-float("inf"))
                dense.scatter_(1, cid, logits)
                torch.topk(dense, k, sorted=True, out=(values[0], ids[0]))

        graphs = []
        for arm in range(2):
            run(arm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(arm)
            graphs.append(graph)
        for _ in range(3000):
            for graph in graphs:
                graph.replay()
        samples = [[], []]
        for trial in range(7):
            local = [[], []]
            for arm in (0, 1, 1, 0) if trial % 2 == 0 else (1, 0, 0, 1):
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(300):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                local[arm].append(start.elapsed_time(end) / 300)
            for arm in range(2):
                samples[arm].append(statistics.mean(local[arm]))
        assert torch.equal(values[0].view(torch.uint8), values[1].view(torch.uint8))
        assert torch.equal(ids[0], ids[1])
        report.update(
            samples_ms=samples,
            medians_ms=[statistics.median(s) for s in samples],
            paired_saved_ms=[x - y for x, y in zip(*samples)],
        )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
