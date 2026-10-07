# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-MOE sub-item 4: decode microbenchmarks of the V4.1 MoE block, per GPU, TP4-local shapes (one rank of a
PP x TP4 stage), T = 1, 2, 4, 8, measured as CUDA-graph replays (the FULL decode graphs of §1) against the
bytes-moved roofline at ``--bw`` GB/s (V100 ~800 effective).

    CUDA_VISIBLE_DEVICES=<uuid> PYTHONPATH=<worktree> python -m tests.models.deepseek_v41.test_moe_bench \\
        --out /mnt/nvme2/scratch/ds41/l-moe/bench

Collected by pytest only with DS41_MOE_BENCH=1 (it takes a few minutes and needs an idle GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

HIDDEN, INTER_TP4, N_EXPERTS, TOP_K = 5120, 576, 384, 6
EXPERT_BYTES = 4_700_160          # one TP4 expert shard incl. E8M0 (PORT_DESIGN §3.4)
GATE_BYTES = N_EXPERTS * HIDDEN * 2
SHARED_W13_BYTES = 2 * INTER_TP4 * HIDDEN * 2
SHARED_W2_BYTES = HIDDEN * INTER_TP4 * 2


def graph_us(fn: Callable[[], object], reps: int = 20, iters: int = 50) -> float:
    """Mean µs of one ``fn()`` from replays of a CUDA graph holding ``reps`` back-to-back calls."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (iters * reps)


def roofline_us(nbytes: float, bw_gbs: float) -> float:
    return nbytes / (bw_gbs * 1e3)


def unique_routes(num_tokens: int, seed: int) -> torch.Tensor:
    """Routing with all T * 6 routes on distinct experts (worst case for bytes; random 384-way routing is
    ~all-distinct at T <= 8 anyway)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    perm = torch.randperm(N_EXPERTS, device="cuda", generator=g)[: num_tokens * TOP_K]
    return perm.view(num_tokens, TOP_K).to(torch.int32).contiguous()


def run(args: argparse.Namespace) -> dict:
    from vllm.models.deepseek_v41.sm70 import gemv as v41_gemv
    from vllm.models.deepseek_v41.sm70 import moe_kernels as mk
    import vllm._custom_ops as ops

    from .test_moe_method import SyntheticCheckpoint, make_layer

    ckpt = SyntheticCheckpoint(seed=5)
    layers = {b: make_layer(N_EXPERTS, TOP_K, 4, 0, b, None, ckpt) for b in args.backends}
    # ROT weight copies rotate through each captured graph so the 6 MB L2 never serves a repeat (a real decode
    # step streams every layer's weights once)
    rot = 4
    gates = [(torch.randn(N_EXPERTS, HIDDEN, device="cuda") * 0.02).half() for _ in range(rot)]
    bias = torch.full((N_EXPERTS,), 10.8, device="cuda")
    sw13s = [(torch.randn(2 * INTER_TP4, HIDDEN, device="cuda") * 0.01).half() for _ in range(rot)]
    sw2s = [(torch.randn(HIDDEN, INTER_TP4, device="cuda") * 0.01).half() for _ in range(rot)]
    counter = iter(range(1 << 60))

    def nxt(seq: list) -> torch.Tensor:
        return seq[next(counter) % len(seq)]

    report: dict = {"bw_gbs": args.bw, "device": torch.cuda.get_device_name(), "rows": []}
    for num_tokens in args.tokens:
        x = (torch.randn(num_tokens, HIDDEN, device="cuda") * 0.5).half()
        ids = unique_routes(num_tokens, seed=num_tokens)
        id_sets = [unique_routes(num_tokens, seed=num_tokens * 100 + i) for i in range(rot)]
        w = torch.rand(num_tokens, TOP_K, device="cuda")
        logits = torch.randn(num_tokens, N_EXPERTS, device="cuda")
        routes_bytes = len(set(ids.flatten().tolist())) * EXPERT_BYTES
        tw = torch.empty(num_tokens, TOP_K, device="cuda")
        ti = torch.empty(num_tokens, TOP_K, device="cuda", dtype=torch.int32)
        te = torch.empty_like(ti)
        y_slots = (torch.randn(num_tokens * TOP_K, HIDDEN, device="cuda") * 0.1).half()
        act = (torch.randn(num_tokens, INTER_TP4, device="cuda")).half()
        ops_us = {
            "gate_gemv": (graph_us(lambda: v41_gemv.gemv(x, nxt(gates), alpha=2.0**-16)), GATE_BYTES),
            "router_topk": (graph_us(lambda: ops.topk_hash_softplus_sqrt(tw, ti, te, logits, True, 1.5, bias,
                                                                         None, None)), 0),
            "route_prep": (graph_us(lambda: mk.route_prep(ids, None, N_EXPERTS, False)), 0),
            "shared_gate_up_swiglu": (graph_us(lambda: v41_gemv.gate_up_swiglu(x, nxt(sw13s), 10.0)),
                                      SHARED_W13_BYTES),
            "shared_down_combine": (graph_us(lambda: v41_gemv.down_combine(act, nxt(sw2s), y_slots, w)),
                                    SHARED_W2_BYTES),
        }
        for backend, (layer, method) in layers.items():
            ops_us[f"experts_{backend}"] = (graph_us(lambda m=method, l=layer: m.expert_slots(l, x, nxt(id_sets))),
                                            routes_bytes)
        if "skinny" in layers:
            layer, method = layers["skinny"]
            perm, tables = mk.route_prep(ids, None, N_EXPERTS, False)
            gids, goff = tables[0]
            (w13, s13, w2, s2), (g13, g2) = method._partitions(layer)[0], layer.ds41_gscales[0]  # noqa: SLF001
            y13 = torch.empty(num_tokens * TOP_K, 2 * INTER_TP4, device="cuda", dtype=torch.float16)
            a13 = torch.randn(num_tokens * TOP_K, INTER_TP4, device="cuda").half()
            y2 = torch.empty(num_tokens * TOP_K, HIDDEN, device="cuda", dtype=torch.float16)
            w13_bytes = routes_bytes * (2 * INTER_TP4 * (HIDDEN // 2 + HIDDEN // 32)) / EXPERT_BYTES
            ops_us["skinny_w13"] = (graph_us(lambda: torch.ops._C.skinny_moe_qpn_sm70(
                x, w13, s13, g13, perm, gids, goff, TOP_K, y13, False, num_tokens, 16, 1, 1)), w13_bytes)
            ops_us["swiglu"] = (graph_us(lambda: mk.swiglu_fp32(y13, 10.0)), 0)
            for split in (4, 6, 9, 12, 18):
                ops_us[f"skinny_w2_splitk{split}"] = (graph_us(lambda s=split: torch.ops._C.skinny_moe_qpn_sm70(
                    a13, w2, s2, g2, perm, gids, goff, TOP_K, y2, True, num_tokens, s, 1, 1)),
                    routes_bytes - w13_bytes)
        for name, (us, nbytes) in ops_us.items():
            row = {"T": num_tokens, "op": name, "us": round(us, 2)}
            if nbytes:
                row.update(MB=round(nbytes / 1e6, 2), roofline_us=round(roofline_us(nbytes, args.bw), 2),
                           eff_GBs=round(nbytes / us / 1e3, 1))
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    return report


def spill_rank(rank: int, world: int, port: int, args: argparse.Namespace) -> None:
    """Phase A spill cost: T=1 expert_slots with k of the 6 routes on spilled (pinned host, UVA) experts, all
    ``world`` ranks of the board timing concurrently (they share one PCIe switch uplink, HARDWARE §A.6)."""
    import torch.distributed as dist

    from vllm.models.deepseek_v41.sm70.moe_method import make_spill_plan

    from .test_moe_method import SyntheticCheckpoint, make_layer

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    plan = make_spill_plan(args.spill, N_EXPERTS)
    layer, method = make_layer(N_EXPERTS, TOP_K, 4, rank, "skinny", plan, SyntheticCheckpoint(seed=9))
    spilled = list(plan.spilled_expert_ids)
    resident = [e for e in range(N_EXPERTS) if e not in set(spilled)]
    x = (torch.randn(1, HIDDEN, device="cuda") * 0.5).half()
    rows = []
    for k in range(TOP_K + 1):
        # rotate through many experts so neither GPU L2 nor the host side serves repeats
        batches = []
        for i in range(32):
            sp = [spilled[(i * TOP_K + j) % len(spilled)] for j in range(k)]
            rs = [resident[(i * TOP_K + j) % len(resident)] for j in range(TOP_K - k)]
            batches.append(torch.tensor([sp + rs], dtype=torch.int32, device="cuda"))
        ids = torch.empty(1, TOP_K, dtype=torch.int32, device="cuda")
        for b in batches:  # warm-up (compiles, faults in the UVA mappings)
            ids.copy_(b)
            method.expert_slots(layer, x, ids)
        torch.cuda.synchronize()
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(4):
            for b in batches:
                ids.copy_(b)
                method.expert_slots(layer, x, ids)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / (4 * len(batches))
        times = [None] * world
        dist.all_gather_object(times, us)
        if rank == 0:
            row = {"spilled_routes": k, "us_per_rank": [round(t, 1) for t in times], "ranks": world,
                   "host_MB_per_rank": round(k * EXPERT_BYTES / 1e6, 2)}
            if k:
                row["host_GBs_per_rank"] = round(k * EXPERT_BYTES / (max(times) - rows[0]["us_max"]) / 1e3, 2)
            row["us_max"] = round(max(times), 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
    if rank == 0:
        mean_spilled = TOP_K * args.spill / N_EXPERTS
        lo = int(mean_spilled)
        frac = mean_spilled - lo
        expected = rows[lo]["us_max"] * (1 - frac) + rows[min(lo + 1, TOP_K)]["us_max"] * frac
        summary = {"spill_per_layer": args.spill, "mean_spilled_routes_T1": round(mean_spilled, 3),
                   "expected_us_per_layer_T1": round(expected, 1),
                   "expected_extra_ms_per_stage_T1_20_layers": round(20 * (expected - rows[0]["us_max"]) / 1e3, 2),
                   "rows": rows}
        print(json.dumps(summary), flush=True)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / f"moe_spill_bench_ranks{world}.json").write_text(json.dumps(summary, indent=1))
    dist.barrier()
    dist.destroy_process_group()


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--backends", nargs="+", default=["skinny", "turbomind"])
    ap.add_argument("--bw", type=float, default=800.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--spill", type=int, default=0, help="Phase A spill benchmark: experts spilled per layer")
    ap.add_argument("--ranks", type=int, default=4, help="concurrent ranks (GPUs) for --spill")
    args = ap.parse_args(argv)
    if args.spill:
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        torch.multiprocessing.spawn(spill_rank, args=(args.ranks, port, args), nprocs=args.ranks, join=True)
        return {}
    report = run(args)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "moe_decode_bench.json").write_text(json.dumps(report, indent=1))
    return report


@pytest.mark.sm70
@pytest.mark.skipif(os.environ.get("DS41_MOE_BENCH") != "1", reason="benchmark: set DS41_MOE_BENCH=1")
def test_moe_decode_bench() -> None:
    report = main(["--tokens", "1", "8"])
    assert report["rows"]


if __name__ == "__main__":
    main()
    sys.exit(0)
