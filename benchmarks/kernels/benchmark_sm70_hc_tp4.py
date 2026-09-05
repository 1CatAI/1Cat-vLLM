# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact TP4 HC Mix gate using all 96 Qwen3.8 checkpoint weight pairs.

Run with torchrun --standalone --nproc-per-node=4 and --model /path/to/model.
Requires four peer-connected SM70 GPUs, VLLM_SM70_TP4_PUSH_ALLREDUCE=1,
and a source-matched custom-AR extension. Does not load the whole model.
Reports Mix-only CUDA Graph time, NOT full HC, TPOT, or service throughput.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors import safe_open

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc  # noqa: F401

MODULES = 96


def load_weights(model: Path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    mapping = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def get(name: str) -> torch.Tensor:
        with safe_open(model / mapping[name], framework="pt", device="cpu") as weights:
            return weights.get_tensor(name).half()

    result = []
    for layer in range(48):
        for role in ("attn", "mlp"):
            prefix = f"model.language_model.layers.{layer}.{role}_hyper_connection."
            down = torch.zeros((336, 10240), dtype=torch.float16)
            down[:320].copy_(get(prefix + "input_mix_weight_down.weight"))
            down[320:324].copy_(get(prefix + "block_inject_weight.weight"))
            up = get(prefix + "input_mix_weight_up.weight")
            result.append((down.cuda(), up.cuda()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality-inputs", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--replays", type=int, default=150)
    parser.add_argument("--stress-replays", type=int, default=32)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        raise RuntimeError("This benchmark requires exactly four SM70 GPUs")
    dist.init_process_group("nccl")
    group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=group, device=local_rank, max_size=8 * 1024 * 1024)
    try:
        if not comm.supports_sm70_qwen38_hc_output_allgather():
            raise RuntimeError("Load the source-matched custom-AR extension")
        weights = load_weights(args.model)
        generator = torch.Generator(device="cuda").manual_seed(20260905)
        xs = torch.randn(
            MODULES, 1, 10240, device="cuda", dtype=torch.float16, generator=generator
        )
        if not comm.can_sm70_qwen38_hc_shard(xs[0]):
            raise RuntimeError("The exact TP4 HC route is unavailable")
        sum_generator = torch.Generator(device="cuda").manual_seed(20260905 + rank)
        sum_a = torch.randn(
            MODULES, 2560, device="cuda", dtype=torch.float16, generator=sum_generator
        )
        sum_b = torch.randn_like(sum_a)
        peer_sums = [torch.empty_like(sum_a) for _ in range(4)]
        dist.all_gather(peer_sums, sum_a + sum_b)
        expected_sum = torch.zeros_like(sum_a, dtype=torch.float32)
        for peer in peer_sums:
            expected_sum.add_(peer.float())
        expected_sum = expected_sum.half()
        tp_group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=comm))

        def capture(hidden: bool, overlap: bool = False):
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            outputs, sums = [], []
            aux = torch.cuda.Stream() if overlap else None
            with (
                patch(
                    "vllm.distributed.parallel_state.get_tp_group",
                    return_value=tp_group,
                ),
                patch.object(
                    comm,
                    "supports_sm70_qwen38_hc_output_allgather",
                    return_value=hidden,
                ),
                comm.capture(),
                torch.cuda.graph(graph),
            ):
                main_stream = torch.cuda.current_stream()
                if aux is not None:
                    aux.wait_stream(main_stream)
                for i, (down, up) in enumerate(weights):
                    outputs.extend(
                        torch.ops.vllm.qwen38_sm70_fp16_fused_hc(xs[i], down, up)
                    )
                    if aux is not None:
                        with torch.cuda.stream(aux):
                            sums.append(comm.all_reduce_sum2(sum_a[i], sum_b[i]))
                if aux is not None:
                    main_stream.wait_stream(aux)
            torch.cuda.synchronize()
            dist.barrier()
            return graph, outputs, sums

        # Keep this legacy gate-vs-hidden benchmark on its named routes even
        # when the loaded extension also provides the newer fused up path.
        fused_override = patch.object(
            comm, "supports_sm70_qwen38_hc_up_mix_allgather", return_value=False
        )
        with fused_override:
            graphs = {
                "control": capture(False),
                "hidden": capture(True),
                "hidden_aux": capture(True, overlap=True),
            }
        mismatches = {"hidden": 0, "hidden_aux": 0, "sum2_aux": 0}
        for case in range(args.quality_inputs):
            xs.normal_(generator=generator)
            if case == 0:
                xs.zero_()
            elif case == 1:
                xs.mul_(0.01)
            for mode, (graph, _, _) in graphs.items():
                # Changing inputs plus repeated epoch wrap tests the HC/MoE
                # channels together, not just a single frozen graph replay.
                for _ in range(args.stress_replays if mode == "hidden_aux" else 1):
                    graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
            for mode in ("hidden", "hidden_aux"):
                for expected, actual in zip(
                    graphs["control"][1], graphs[mode][1], strict=True
                ):
                    mismatches[mode] += int(
                        torch.count_nonzero(
                            expected.view(torch.int16) != actual.view(torch.int16)
                        )
                    )
            mismatches["sum2_aux"] += int(
                torch.count_nonzero(
                    expected_sum.view(torch.int16)
                    != torch.stack(graphs["hidden_aux"][2]).view(torch.int16)
                )
            )
        quality = [None] * 4
        dist.all_gather_object(
            quality, {"rank": rank, "mismatches": mismatches}, group=group
        )
        if rank == 0:
            print(json.dumps({"quality": quality}), flush=True)
        if any(any(q["mismatches"].values()) for q in quality):
            if rank == 0:
                args.out.write_text(json.dumps({"quality": quality}, indent=2) + "\n")
            raise RuntimeError("Production HC or concurrent sum2 is not bitwise")

        # Warm all devices out of idle clocks before paired graph timings.
        for mode in ("control", "hidden"):
            for _ in range(args.warmup):
                graphs[mode][0].replay()
            torch.cuda.synchronize()
            dist.barrier()
        samples = {"control": [], "hidden": []}
        for repeat in range(3):
            for mode in list(samples) if repeat % 2 == 0 else list(samples)[::-1]:
                graph = graphs[mode][0]
                for _ in range(20):
                    graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
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
        if rank == 0:
            medians = {mode: median(values) for mode, values in samples.items()}
            result = {
                "modules": MODULES,
                "includes_combine_norm": False,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "quality": quality,
                "quality_inputs": args.quality_inputs,
                "aux_stress_replays": args.quality_inputs * args.stress_replays,
                "samples_ms": samples,
                "median_ms": medians,
                "saved_ms": medians["control"] - medians["hidden"],
            }
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
    finally:
        comm.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
