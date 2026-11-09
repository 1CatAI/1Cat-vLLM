# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete Qwen3.8 HC workload, including norms and the final mixer.

Attention/MoE outputs are fixed external inputs; their computation and PLE
are excluded. This is a CUDA Graph microbenchmark, NOT full-model TPOT or
full-model Nsight service time. Run on four exclusively available SM70 GPUs:

CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID \
VLLM_SM70_TP4_PUSH_ALLREDUCE=1 \
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_hc_full_chain.py --model MODEL --out RESULT
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors import safe_open

import vllm.envs as envs
from benchmarks.kernels.benchmark_sm70_hc_tp4 import load_weights
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.models.qwen4_exp.nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality-inputs", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--replays", type=int, default=150)
    parser.add_argument(
        "--fused-up",
        action="store_true",
        help="Compare hidden split against fused up/mix/gather",
    )
    parser.add_argument(
        "--aux-stress-replays",
        type=int,
        default=32,
        help="Auxiliary sum2 replays per changing input with --fused-up",
    )
    args = parser.parse_args()
    if args.fused_up and not envs.VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1:
        raise RuntimeError(
            "Set VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1=1 for the aux gate"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(visible.split(",")) != 4 or int(os.environ["WORLD_SIZE"]) != 4:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES and launch exactly four ranks")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This gate is specific to SM70")
    dist.init_process_group("nccl")
    group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=group, device=local_rank, max_size=8 * 1024 * 1024)
    try:
        owned_pids = [None] * 4
        dist.all_gather_object(owned_pids, os.getpid(), group=group)

        def ensure_exclusive() -> None:
            contenders = []
            if rank == 0:
                probe = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        visible,
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                for line in probe.splitlines():
                    pid, memory = (int(part.strip()) for part in line.split(","))
                    if pid not in owned_pids and memory > 128:
                        contenders.append({"pid": pid, "MiB": memory})
            shared = [contenders]
            dist.broadcast_object_list(shared, src=0, group=group)
            if shared[0]:
                raise RuntimeError(f"GPU contention invalidates timing: {shared[0]}")

        ensure_exclusive()
        if not comm.supports_sm70_qwen38_hc_output_allgather():
            raise RuntimeError("Load a source-matched custom-AR extension")
        if args.fused_up and not comm.supports_sm70_qwen38_hc_up_mix_allgather():
            raise RuntimeError("Load an extension with fused HC up/mix/gather")
        weights = load_weights(args.model)
        mapping = json.loads((args.model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

        def get(name: str) -> torch.Tensor:
            with safe_open(
                args.model / mapping[name], framework="pt", device="cpu"
            ) as f:
                return f.get_tensor(name).half().cuda()

        prefix = "model.language_model."
        norms = [
            get(f"{prefix}layers.{layer}.{role}_hyper_connection.hc_norm.weight")
            for layer in range(48)
            for role in ("attn", "mlp")
        ]
        final_norm = get(prefix + "hyper_connection_mixer.hc_norm.weight")
        final_down = get(prefix + "hyper_connection_mixer.input_mix_weight_down.weight")
        final_up = get(prefix + "hyper_connection_mixer.input_mix_weight_up.weight")
        gen = torch.Generator(device="cuda").manual_seed(20260905)
        initial = torch.randn(
            (1, 10240), device="cuda", dtype=torch.float16, generator=gen
        )
        cores = torch.randn(
            (96, 1, 2560), device="cuda", dtype=torch.float16, generator=gen
        )
        if not comm.can_sm70_qwen38_hc_shard(initial):
            raise RuntimeError("The exact TP4 HC route is unavailable")
        tp = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=comm))
        if args.fused_up:
            sum_gen = torch.Generator(device="cuda").manual_seed(20260905 + rank)
            sum_a = torch.randn(
                96, 2560, device="cuda", dtype=torch.float16, generator=sum_gen
            )
            sum_b = torch.randn(
                96, 2560, device="cuda", dtype=torch.float16, generator=sum_gen
            )
            peer_sums = [torch.empty_like(sum_a) for _ in range(4)]
            dist.all_gather(peer_sums, sum_a + sum_b)
            expected_sum = torch.zeros_like(sum_a, dtype=torch.float32)
            for peer in peer_sums:
                expected_sum.add_(peer.float())
            expected_sum = expected_sum.half()

        def finish(state: torch.Tensor, injection: torch.Tensor):
            combined, xn = hc_combine_norm(
                state, cores[-1], injection, final_norm, 1e-6, 4
            )
            lora = hc_silu(torch.nn.functional.linear(xn, final_down), 4)
            gate = torch.nn.functional.linear(lora, final_up)
            return combined, hc_gate_mix(xn, gate, 4)

        # A model's normal warmup initializes cuBLAS before graph capture.
        finish(initial, torch.zeros((1, 4), device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()

        def capture(mode: str, overlap: bool = False):
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            outputs = []
            sums = []
            aux = torch.cuda.Stream() if overlap else None
            with (
                patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp),
                patch.object(
                    comm,
                    "supports_sm70_qwen38_hc_output_allgather",
                    return_value=mode != "gate",
                ),
                patch.object(
                    comm,
                    "supports_sm70_qwen38_hc_up_mix_allgather",
                    return_value=mode == "fused",
                ),
                comm.capture(),
                torch.cuda.graph(graph),
            ):
                main_stream = torch.cuda.current_stream()
                if aux is not None:
                    aux.wait_stream(main_stream)
                state, injection = initial, None
                for i, (down, up) in enumerate(weights):
                    # PLE at decoder layer 2 requires a materialized state.
                    # Exclude PLE computation, but retain its HC boundary.
                    if i == 2:
                        state = hc_combine(state, cores[i - 1], injection, 4)
                    if i in (0, 2):
                        xn = grouped_gemma_rmsnorm(state, norms[i], 1e-6, 4)
                    else:
                        state, xn = hc_combine_norm(
                            state, cores[i - 1], injection, norms[i], 1e-6, 4
                        )
                    block, injection = torch.ops.vllm.qwen38_sm70_fp16_fused_hc(
                        xn, down, up
                    )
                    outputs.extend((state, xn, block, injection))
                    if aux is not None:
                        with torch.cuda.stream(aux):
                            sums.append(comm.all_reduce_sum2(sum_a[i], sum_b[i]))
                outputs.extend(finish(state, injection))
                if aux is not None:
                    main_stream.wait_stream(aux)
            torch.cuda.synchronize()
            dist.barrier()
            return graph, outputs, sums

        timed_modes = ("hidden", "fused") if args.fused_up else ("gate", "hidden")
        graphs = {mode: capture(mode) for mode in timed_modes}
        if args.fused_up:
            graphs["fused_aux"] = capture("fused", overlap=True)
        mismatches = {mode: 0 for mode in list(graphs)[1:]}
        sum_mismatches = 0

        def replay_and_check(stress: bool):
            for mode, (graph, _, _) in graphs.items():
                repeats = (
                    args.aux_stress_replays if stress and mode == "fused_aux" else 1
                )
                for _ in range(repeats):
                    graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
            expected = torch.cat(
                [x.flatten().view(torch.int16) for x in graphs[timed_modes[0]][1]]
            )
            diffs = {}
            for mode in list(graphs)[1:]:
                actual = torch.cat(
                    [x.flatten().view(torch.int16) for x in graphs[mode][1]]
                )
                diffs[mode] = int(torch.count_nonzero(expected != actual))
            sum_diff = 0
            if args.fused_up:
                actual_sum = torch.stack(graphs["fused_aux"][2])
                sum_diff = int(
                    torch.count_nonzero(
                        actual_sum.view(torch.int16) != expected_sum.view(torch.int16)
                    )
                )
            return diffs, sum_diff

        for case in range(args.quality_inputs):
            initial.normal_(generator=gen)
            cores.normal_(generator=gen)
            if case == 0:
                initial.zero_()
                cores.zero_()
            elif case == 1:
                initial.mul_(0.01)
                cores.mul_(0.01)
            diffs, sum_diff = replay_and_check(stress=True)
            for mode, diff in diffs.items():
                mismatches[mode] += diff
            sum_mismatches += sum_diff
        quality = [None] * 4
        dist.all_gather_object(
            quality,
            {
                "rank": rank,
                "mismatches": sum(mismatches.values()) + sum_mismatches,
                "hc_mismatches": mismatches,
                "sum2_mismatches": sum_mismatches,
            },
            group=group,
        )
        if rank == 0:
            print({"quality": quality}, flush=True)
        if any(q["mismatches"] for q in quality):
            raise RuntimeError("Full HC outputs are not bitwise")
        ensure_exclusive()
        for mode in timed_modes:
            graph = graphs[mode][0]
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
        samples = {mode: [] for mode in timed_modes}
        for repeat in range(3):
            modes = timed_modes if repeat % 2 == 0 else timed_modes[::-1]
            for mode in modes:
                ensure_exclusive()
                graph = graphs[mode][0]
                for _ in range(20):
                    graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(args.replays):
                    graph.replay()
                end.record()
                end.synchronize()
                times = [None] * 4
                dist.all_gather_object(
                    times, start.elapsed_time(end) / args.replays, group=group
                )
                samples[mode].append(max(times))
                ensure_exclusive()
        diffs, sum_diff = replay_and_check(stress=False)
        post_quality = [None] * 4
        dist.all_gather_object(
            post_quality,
            {"rank": rank, "hc_mismatches": diffs, "sum2_mismatches": sum_diff},
            group=group,
        )
        if any(
            any(q["hc_mismatches"].values()) or q["sum2_mismatches"]
            for q in post_quality
        ):
            raise RuntimeError("Post-timing HC/sum2 output differs after epoch wrap")
        if rank == 0:
            result = {
                "source_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "scope": (
                    "full semantic HC microbenchmark; "
                    "excludes attention/MoE/PLE computation"
                ),
                "counts": {
                    "mix_pairs": 96,
                    "combine_norm": 95,
                    "separate_combine": 1,
                    "grouped_norm": 2,
                    "final_projection_pairs": 1,
                    "final_gate_mix": 1,
                },
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "visible_devices": visible,
                "quality": quality,
                "quality_inputs": args.quality_inputs,
                "post_timing_quality": post_quality,
                "aux_stress_replays": args.quality_inputs * args.aux_stress_replays
                if args.fused_up
                else 0,
                "samples_ms": samples,
                "median_ms": {mode: median(values) for mode, values in samples.items()},
            }
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
    finally:
        comm.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
