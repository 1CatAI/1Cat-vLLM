# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""HC output sharding screen using existing SM70 collectives, no new runtime.

Unlike the M1 path in PR #481, this uses batched cuBLAS on local output columns
and disjoint-output all-reduces. Includes both communications and scatter/mix.
Real weights, synthetic activations; this is not a model-quality test.
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
from safetensors import safe_open

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu
from vllm.triton_utils import tl, triton


@triton.jit
def scatter_down(local, full, RANK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 512)
    idx = col - RANK * 80
    belongs = (idx >= 0) & (idx < 80)
    if RANK == 3:
        belongs |= (idx >= 80) & (idx < 84)
    val = tl.load(local + row * 88 + idx, belongs, 0)
    tl.store(full + row * 336 + col, val, col < 336)


@triton.jit
def mix_scatter(gate, x, out, RANK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * 256 + tl.arange(0, 256)
    local = col - RANK * 640
    valid = (local >= 0) & (local < 640)
    acc = tl.full((256,), 0, tl.float32)
    for branch in tl.static_range(4):
        g = tl.load(gate + row * 2560 + branch * 640 + local, valid, 0).to(tl.float32)
        val = tl.load(x + row * 10240 + branch * 2560 + col, valid, 0).to(tl.float32)
        acc = tl.fma(tl.sigmoid(g), val, acc)
    tl.store(out + row * 2560 + col, acc / 4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokens", default="4,8,16")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--pair", choices=("attn", "mlp"), default="attn")
    p.add_argument("--mode", choices=("full", "down"), default="full")
    p.add_argument(
        "--weight-copies",
        type=int,
        default=1,
        help="Distinct allocations of the same real weights: cache-footprint screen",
    )
    p.add_argument(
        "--profile", action="store_true", help="Capture graph nodes, skip timing sweep"
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.weight_copies <= 0:
        p.error("--weight-copies must be positive")
    calls_per_graph = max(16, args.weight_copies)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 4
    group = dist.new_group(backend="gloo")
    owned_pids = [None] * 4
    dist.all_gather_object(owned_pids, os.getpid(), group=group)

    def check_exclusive():
        other = []
        if rank == 0:
            devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
            report = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    devices,
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            for line in report.splitlines():
                pid, _, name = line.partition(",")
                if (
                    pid.strip().isdigit()
                    and int(pid) not in owned_pids
                    and "snapd-desktop-integration" not in name
                ):
                    other.append(line)
        result = [other]
        dist.broadcast_object_list(result, src=0, group=group)
        if result[0]:
            raise RuntimeError(
                f"Foreign GPU processes appeared; discard this timing: {result[0]}"
            )

    ca = CustomAllreduce(group, rank, max_size=128 * 1024)
    assert not ca.disabled
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    stem = f"model.language_model.layers.{args.layer}.{args.pair}_hyper_connection"

    def weight(suffix):
        key = stem + suffix
        with safe_open(args.model / index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key).half().cuda().contiguous()

    down = weight(".input_mix_weight_down.weight")
    injection = weight(".block_inject_weight.weight")
    down = torch.cat((down, injection, down.new_zeros(12, 10240)))
    up = weight(".input_mix_weight_up.weight")
    local_down_w = down.new_zeros(88, 10240)
    local_down_w[:80].copy_(down[rank * 80 : (rank + 1) * 80])
    if rank == 3:
        local_down_w[80:84].copy_(down[320:324])
    local_up_w = (
        up.view(4, 2560, 320)[:, rank * 640 : (rank + 1) * 640]
        .reshape(2560, 320)
        .contiguous()
    )
    weights = [(down, up, local_down_w, local_up_w)]
    for _ in range(args.weight_copies - 1):
        weights.append(tuple(w.clone() for w in weights[0]))
    results = []
    try:
        for m in map(int, args.tokens.split(",")):
            torch.manual_seed(20260905)
            x = torch.randn(m, 10240, device="cuda", dtype=torch.float16)
            sparse_down = x.new_empty(m, 336)
            full_down = torch.empty_like(sparse_down)
            sparse_output = x.new_empty(m, 2560)
            output = torch.empty_like(sparse_output)

            def baseline(index=0):
                wd, wu, _, _ = weights[index]
                d = torch.nn.functional.linear(x, wd)
                lora = hc_silu(d[:, :320], 4)
                return hc_gate_mix(x, torch.nn.functional.linear(lora, wu), 4), d[
                    :, 320:324
                ]

            def candidate(registered, index=0):
                _, wu, local_wd, local_wu = weights[index]
                d = torch.nn.functional.linear(x, local_wd)
                scatter_down[(m,)](d, sparse_down, RANK=rank)
                ca.all_reduce(sparse_down, out=full_down, registered=registered)
                lora = hc_silu(full_down[:, :320], 4)
                if args.mode == "down":
                    return hc_gate_mix(
                        x, torch.nn.functional.linear(lora, wu), 4
                    ), full_down[:, 320:324]
                gate = torch.nn.functional.linear(lora, local_wu)
                mix_scatter[(m, 10)](gate, x, sparse_output, RANK=rank)
                ca.all_reduce(sparse_output, out=output, registered=registered)
                return output, full_down[:, 320:324]

            for _ in range(3):
                baseline()
                candidate(False)
            torch.cuda.synchronize()
            dist.barrier()
            graphs, outputs = [], []
            for which in (0, 1):
                graph = torch.cuda.CUDAGraph()
                with ca.capture(), torch.cuda.graph(graph):
                    for call in range(calls_per_graph):
                        index = call % args.weight_copies
                        values = candidate(True, index) if which else baseline(index)
                graphs.append(graph)
                outputs.append(values)
            checks = []
            for scale in (0.25, 1.0, 3.0):
                x.normal_().mul_(scale)
                # Identical generators on all ranks; enforce common inputs.
                dist.broadcast(x, 0)
                expected = tuple(t.clone() for t in baseline())
                actual = tuple(t.clone() for t in candidate(False))
                sparse_down.fill_(float("nan"))
                output.fill_(float("nan"))
                graphs[1].replay()
                torch.cuda.synchronize()
                row = []
                for i, (a, b) in enumerate(zip(outputs[1], expected)):
                    torch.testing.assert_close(a, actual[i], rtol=0, atol=0)
                    delta = a.float() - b.float()
                    row.append(
                        dict(
                            exact=torch.equal(a, b),
                            max_abs=delta.abs().max().item(),
                            relative_l2=(
                                delta.norm() / b.float().norm().clamp_min(1e-12)
                            ).item(),
                            finite=torch.isfinite(a).all().item(),
                        )
                    )
                checks.append(row)
            assert all(c["finite"] for row in checks for c in row)
            if args.profile:
                for _ in range(30):
                    graphs[0].replay()
                    graphs[1].replay()
                torch.cuda.synchronize()
                check_exclusive()
                dist.barrier()
                torch.cuda.profiler.start()
                for _ in range(3):
                    for which, name in ((0, "hc.baseline"), (1, "hc.candidate")):
                        with torch.cuda.nvtx.range(name):
                            graphs[which].replay()
                            torch.cuda.synchronize()
                        dist.barrier()
                torch.cuda.profiler.stop()
                check_exclusive()
                if rank == 0:
                    print(
                        "HC profile completed; trace is composition evidence, "
                        "not a speed gate.",
                        flush=True,
                    )
                del graphs, outputs
                continue
            times = [[], []]
            for sample in range(5):
                for which in (0, 1) if sample % 2 == 0 else (1, 0):
                    for _ in range(20):
                        graphs[which].replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    check_exclusive()
                    begin, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    begin.record()
                    for _ in range(40):
                        graphs[which].replay()
                    end.record()
                    end.synchronize()
                    us = begin.elapsed_time(end) * 1000 / (40 * calls_per_graph)
                    all_us = [None] * 4
                    dist.all_gather_object(all_us, us, group=group)
                    check_exclusive()
                    times[which].append(max(all_us))
            local_result = dict(
                rank=rank,
                tokens=m,
                baseline_us=statistics.median(times[0]),
                candidate_us=statistics.median(times[1]),
                times=times,
                checks=checks,
            )
            gathered = [None] * 4
            dist.all_gather_object(gathered, local_result, group=group)
            if rank == 0:
                results.append(gathered)
                print(json.dumps(gathered), flush=True)
            del graphs, outputs
        if rank == 0:
            library = os.environ.get("VLLM_SM70_CUSTOM_AR_LIBRARY")
            runtime = {
                "custom_ar_library": library,
                "small_message_push": os.environ.get(
                    "VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES", "0"
                ),
                "calls_per_graph": calls_per_graph,
                "replicated_weight_bytes": sum(
                    w.numel() * w.element_size() for ws in weights for w in ws[:2]
                ),
                "sharded_weight_bytes": sum(
                    w.numel() * w.element_size() for ws in weights for w in ws[2:]
                ),
            }
            if library:
                runtime["custom_ar_sha256"] = hashlib.sha256(
                    Path(library).read_bytes()
                ).hexdigest()
            args.out.parent.mkdir(exist_ok=True, parents=True)
            args.out.write_text(
                json.dumps(
                    dict(args=vars(args), results=results, runtime=runtime),
                    default=str,
                    indent=2,
                )
                + "\n"
            )
    finally:
        ca.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
