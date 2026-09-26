# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""96 real HC pairs; exact pinned GEMMs + research fused IPC postops.

Includes both gathers, down/up, SiLU, mix. Excludes combine/norm and final
mixer. This is NOT model TPOT. No foreign opaque communicator pointer is used.
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
from torch.utils.cpp_extension import load

from benchmarks.benchmark_sm70_cublaslt_algorithms import _load_extension as build_lt
from benchmarks.kernels.benchmark_sm70_hc_tp4 import load_weights
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu


def build_gather():
    return load(
        name="sm70_hc_batch_gather_screen",
        sources=[str(Path(__file__).with_name("sm70_hc_batch_gather_screen.cu"))],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--model", type=Path)
    p.add_argument("--rows", default="2,4,8,16")
    p.add_argument("--pairs", type=int, default=96)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.build_only:
        build_lt()
        build_gather()
        assert not torch.cuda.is_initialized()
        return
    if a.model is None or a.out is None:
        p.error("--model and --out are required for GPU tests")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if not 1 <= a.pairs <= 96 or any(
        int(m) not in (2, 4, 8, 16) for m in a.rows.split(",")
    ):
        p.error("use 1--96 pairs and row widths from 2,4,8,16")
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        p.error("launch exactly four SM70 ranks with torchrun")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires SM70")
    torch.set_num_threads(1)
    lt = build_lt()
    gather = build_gather()
    dist.init_process_group("nccl")
    group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group, rank, max_size=128 * 1024)
    assert not comm.disabled and comm.sm70_tp4_push_buffer_ptrs is not None
    peers = comm.sm70_tp4_push_buffer_ptrs
    owned = [None] * 4
    dist.all_gather_object(owned, os.getpid(), group=group)
    sources = (
        Path(__file__),
        Path(__file__).with_name("sm70_hc_batch_gather_screen.cu"),
        Path(__file__).parents[1] / "csrc/benchmark_sm70_cublaslt_algorithms.cpp",
        Path(__file__).parents[2] / "csrc/custom_all_reduce.cuh",
    )
    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "research_only": True,
        "includes_combine_norm_or_final_mixer": False,
        "source_sha256": {
            str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        },
    }

    def exclusive():
        errors = []
        if rank == 0:
            report = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    os.environ["CUDA_VISIBLE_DEVICES"],
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            for line in report.splitlines():
                pid, mem = map(int, line.split(","))
                if pid not in owned and mem > 128:
                    errors.append(line)
        check = [errors]
        dist.broadcast_object_list(check, src=0, group=group)
        assert not check[0], check

    results = []
    try:
        exclusive()
        weights = load_weights(a.model)[: a.pairs]
        local = [
            (
                d.narrow(0, rank * 80, 88),
                u.view(4, 2560, 320)[:, rank * 640 : (rank + 1) * 640]
                .reshape(2560, 320)
                .contiguous(),
            )
            for d, u in weights
        ]
        workspace = torch.empty(32 * 1024**2, device="cuda", dtype=torch.uint8)
        for m in map(int, a.rows.split(",")):
            torch.manual_seed(20260926)
            xs = [
                torch.randn(m, 10240, device="cuda", dtype=torch.float16) * 0.1
                for _ in weights
            ]
            local_down = [
                torch.empty(m, 88, device="cuda", dtype=torch.float16) for _ in weights
            ]
            loras = [
                torch.empty(m, 320, device="cuda", dtype=torch.float16) for _ in weights
            ]
            gates = [
                torch.empty(m, 2560, device="cuda", dtype=torch.float16)
                for _ in weights
            ]
            outputs = [torch.empty_like(g) for g in gates]
            injections = [
                torch.empty(m, 4, device="cuda", dtype=torch.float16) for _ in weights
            ]
            dr = lt.LtRunner(m, 88, 10240, workspace.numel(), 1, False)
            ur = lt.LtRunner(m, 2560, 320, workspace.numel(), 1, False)
            di = dr.add_configuration(21, 5, 14, 0, 0, 22, 4)
            ui = ur.add_configuration(21, 5, 14, 0, 0, 1, 0)
            assert di >= 0 and ui >= 0

            def baseline():
                result = []
                for x, (d, u) in zip(xs, weights):
                    down = torch.nn.functional.linear(x, d)
                    lora = hc_silu(down[:, :320], 4)
                    block = hc_gate_mix(x, torch.nn.functional.linear(lora, u), 4)
                    result.append((block, down[:, 320:324]))
                return result

            def candidate():
                for i, (x, (d, u)) in enumerate(zip(xs, local)):
                    dr.run(di, local_down[i], x, d, workspace)
                    gather.down(peers, rank, local_down[i], injections[i], loras[i])
                    ur.run(ui, gates[i], loras[i], u, workspace)
                    gather.mix(peers, rank, gates[i], x, outputs[i])
                return list(zip(outputs, injections))

            for _ in range(3):
                baseline()
                candidate()
            torch.cuda.synchronize()
            dist.barrier()
            graphs = []
            captured = []
            for fn in (baseline, candidate):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    captured.append(fn())
                graphs.append(g)
            mismatches = []
            for scale in (0, 0.001, 0.03, 0.1, 1.0, 3.0):
                for x, o, j in zip(xs, outputs, injections):
                    x.normal_(0, scale)
                    o.fill_(float("nan"))
                    j.fill_(float("nan"))
                for g in graphs:
                    g.replay()
                torch.cuda.synchronize()
                errors = [
                    sum(
                        int(
                            (x.view(torch.int16) != y.view(torch.int16)).count_nonzero()
                        )
                        for b, c in zip(captured[0], captured[1])
                        for x, y in [(b[k], c[k])]
                    )
                    for k in range(2)
                ]
                mismatches.append(errors)
            all_checks = [None] * 4
            dist.all_gather_object(all_checks, mismatches, group=group)
            if any(
                any(any(z for z in case) for case in checks) for checks in all_checks
            ):
                if rank == 0:
                    a.out.write_text(
                        json.dumps(dict(m=m, quality=all_checks), indent=2)
                    )
                    print(json.dumps(dict(m=m, quality=all_checks)), flush=True)
                raise RuntimeError("HC pinned shard is not bitwise exact")
            samples = [[], []]
            for trial in range(6):
                for arm in (0, 1) if trial % 2 == 0 else (1, 0):
                    for _ in range(10):
                        graphs[arm].replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    exclusive()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(24):
                        graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    times = [None] * 4
                    dist.all_gather_object(
                        times, start.elapsed_time(end) / 24, group=group
                    )
                    samples[arm].append(max(times))
            exclusive()
            row = dict(
                environment=environment,
                m=m,
                pairs=len(weights),
                down_algorithm=dr.algorithm_info()[di],
                up_algorithm=ur.algorithm_info()[ui],
                quality=all_checks,
                samples_ms=samples,
                medians_ms=[statistics.median(v) for v in samples],
            )
            if rank == 0:
                results.append(row)
                a.out.write_text(json.dumps(results, indent=2))
                print(json.dumps(row), flush=True)
            graphs.clear()
            captured.clear()
    finally:
        comm.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
