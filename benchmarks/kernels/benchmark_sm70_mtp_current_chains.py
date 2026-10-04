# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualify current M5 HC transport and NVFP4 chains without a profiler.

No candidate, dispatch override, parameter sweep or model admission. Checkpoint
weights and synthetic activations are used. HC requires torchrun TP4; MoE uses
one TP weight shard and a route fixture with the observed 30-group density.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from benchmarks.kernels.benchmark_sm70_moe_packed_w13 import checkpoint_weights
from benchmarks.kernels.benchmark_sm70_mtp_fp32_dense import capture, elapsed
from vllm import _sm70_ops as ops


def samples(graph):
    return [elapsed(graph) for _ in range(7)]


def hc(model):
    from benchmarks.kernels.benchmark_sm70_hc_tp4 import load_weights
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import _pack_hc_batch_weight

    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    raw = load_weights(model)
    names = [f"layer {i} {role}" for i in range(48) for role in ("attn", "mlp")]
    packed = [
        (_pack_hc_batch_weight(d, "down", rank), _pack_hc_batch_weight(u, "up", rank))
        for d, u in raw
    ]
    state = [
        (
            torch.randn(5, 10240, device="cuda", dtype=torch.half),
            torch.empty(20, 5, 96, device="cuda", dtype=torch.float32),
            torch.empty(5, 320, device="cuda", dtype=torch.half),
            torch.empty(5, 640, device="cuda", dtype=torch.half),
            torch.empty(5, 2560, device="cuda", dtype=torch.half),
            torch.empty(5, 4, device="cuda", dtype=torch.half),
        )
        for _ in packed
    ]
    comm = CustomAllreduce(dist.group.WORLD, device=rank)
    assert comm.can_sm70_qwen38_hc_batch(state[0][0])
    torch.accelerator.synchronize()
    dist.barrier()

    def chain():
        for (down, up), (x, partials, lora, local, output, injection) in zip(
            packed, state
        ):
            comm.sm70_qwen38_hc_batch(
                x, down, up, partials, lora, local, output, injection, fused_chain=True
            )

    graph = capture(chain)
    graph.replay()
    torch.accelerator.synchronize()
    assert all(torch.isfinite(t).all() for s in state for t in s[2:])
    trials = []
    for _ in range(7):
        dist.barrier()
        times = [None] * 4
        dist.all_gather_object(times, elapsed(graph))
        trials.append(times)
    down_bytes = sum(d.numel() * d.element_size() for d, _ in packed)
    up_bytes = sum(u.numel() * u.element_size() for _, u in packed)
    result = dict(
        rank_max_median_ms=statistics.median(max(t) for t in trials),
        rank_samples_ms=trials,
        layers=names,
        down_issued_weight_bytes=down_bytes,
        up_issued_weight_bytes=up_bytes,
        weight_floor_ms=(down_bytes + up_bytes) / 750e6,
        exclusions="combine/norm, final mixer, other model work",
    )
    dist.barrier()
    comm.close()
    dist.destroy_process_group()
    return result


def moe(model, rank):
    w13, s13, w2, s2 = checkpoint_weights(model, 0, rank, True)
    x = torch.randn(5, 2560, device="cuda", dtype=torch.half).mul_(0.1)
    # Each token has ten distinct choices. Twenty experts are shared between
    # two tokens; ten occur once. These are density fixtures, not captured IDs.
    ids = (torch.arange(50, device="cuda", dtype=torch.int32) % 30).contiguous()
    weights = torch.softmax(torch.randn(5, 10, device="cuda"), -1)
    mid = x.new_empty(50, 160)
    out = torch.empty_like(x)
    scratch = x.new_empty(50, 2560)
    rows = ids.new_empty(50, 8)
    experts, sizes = torch.empty_like(ids), torch.empty_like(ids)
    total = ids.new_empty(1)

    def first():
        ops.nvfp4_grouped_w13_sm70_out(
            mid, x, w13, s13, ids, rows, experts, sizes, total, 4, True
        )

    def second():
        ops.nvfp4_grouped_w2_batch_reduce_sm70_out(
            out, scratch, mid, w2, s2, weights, rows, experts, sizes, total
        )

    def chain():
        first()
        second()

    graphs = [capture(fn) for fn in (first, second, chain)]
    records = []
    for shift in (0, 240):
        ids.copy_((torch.arange(50, device="cuda", dtype=torch.int32) % 30) + shift)
        for graph in graphs:
            graph.replay()
        torch.accelerator.synchronize()
        assert int(total.item()) == 30
        assert torch.isfinite(mid).all() and torch.isfinite(out).all()
        times = [samples(graph) for graph in graphs]
        b13 = 30 * (w13[0].nbytes + s13[0].nbytes)
        b2 = 30 * (w2[0].nbytes + s2[0].nbytes)
        records.append(
            dict(
                expert_offset=shift,
                groups=30,
                samples_ms=times,
                medians_ms=[statistics.median(t) for t in times],
                timing_order=["plan+W13+SwiGLU", "W2+ordered reduce", "complete chain"],
                w13_issued_weight_bytes=b13,
                w2_issued_weight_bytes=b2,
                weight_floors_ms=[b13 / 750e6, b2 / 750e6],
            )
        )
    return dict(layer=0, tp_weight_rank=rank, captured_routes=False, fixtures=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workload", choices=("hc", "moe"), required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    assert torch.cuda.get_device_capability() == (7, 0)
    root = Path(__file__).resolve().parents[2]
    result = hc(args.model) if args.workload == "hc" else moe(args.model, rank)
    report = dict(
        complete=True,
        model_admission=False,
        source=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        native_sha256=hashlib.sha256(
            (root / "vllm/_C.abi3.so").read_bytes()
        ).hexdigest(),
        workload=args.workload,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        synthetic_activations=True,
        result=result,
    )
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
