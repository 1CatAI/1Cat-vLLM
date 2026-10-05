# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only complete HC screen; never installs an experimental route.

Prebuild with --build-only while waiting for the GPU lease. Run with torchrun
on four owned SM70 GPUs. Each graph rotates real checkpoint weights and includes
combine, norm, both projections, SiLU, mix, and TP transport. Attention, MoE,
PLE, and the final mixer are excluded. Counts are measured from profiler events.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.utils.cpp_extension import load

from benchmarks.kernels.sm70_chain_screen_utils import graph_kernel_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--reduce-core", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).parents[1] / "csrc/sm70_hc_norm_push_screen.cu"
    extension = load(
        "sm70_hc_norm_push_screen",
        [str(source)],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        verbose=True,
    )
    if args.build_only:
        args.output.write_text(
            json.dumps(
                {
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "library": extension.__file__,
                    "research_only": True,
                },
                indent=2,
            )
            + "\n"
        )
        return

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (7, 0)
    assert int(os.environ["WORLD_SIZE"]) == 4
    dist.init_process_group("nccl")
    group = dist.new_group(backend="gloo")
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc  # noqa: F401
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine_norm

    comm = CustomAllreduce(group, rank)
    assert not comm.disabled and comm.fully_connected
    tp = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=comm))
    mapping = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def get(name):
        with safe_open(args.model / mapping[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name).half().cuda()

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    banks = []
    for i in range(args.pairs):
        role = "attn" if i % 2 == 0 else "mlp"
        prefix = f"model.language_model.layers.{i // 2}.{role}_hyper_connection."
        down = torch.zeros(336, 10240, dtype=torch.float16, device="cuda")
        down[:320].copy_(get(prefix + "input_mix_weight_down.weight"))
        down[320:324].copy_(get(prefix + "block_inject_weight.weight"))
        banks.append(
            (
                get(prefix + "hc_norm.weight"),
                down,
                get(prefix + "input_mix_weight_up.weight"),
            )
        )
    packed_banks = [
        (
            sm70_fp16_hc._pack_hc_batch_weight(down, "down", None),
            sm70_fp16_hc._pack_hc_batch_weight(up, "up", None),
        )
        for _, down, up in banks
    ]
    report = {
        "research_only": True,
        "model_admission": False,
        "pairs": args.pairs,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "excluded": "attention/MoE/PLE/final mixer",
        "widths": [],
        "includes_core_allreduce": args.reduce_core,
    }
    try:

        def screen(m):
            torch.manual_seed(4201)
            initial = torch.randn(m, 10240, device="cuda", dtype=torch.float16) * 0.1
            cores = (
                torch.randn(args.pairs, m, 2560, device="cuda", dtype=torch.float16)
                * 0.1
            )
            if args.reduce_core:
                cores.mul_(1 + rank * 0.125)
            initial_injection = torch.zeros(m, 4, device="cuda", dtype=torch.float16)
            own, handle = extension.allocate(m)
            handles = [None] * 4
            dist.all_gather_object(handles, handle, group=group)
            peers = [
                own if i == rank else extension.open_peer(h)
                for i, h in enumerate(handles)
            ]
            down_epochs = torch.zeros(m, 81, device="cuda", dtype=torch.int32)
            up_epochs = torch.zeros(m, 160, device="cuda", dtype=torch.int32)
            outputs = [
                tuple(initial.new_empty((m, n)) for n in (10240, 10240, 2560, 4))
                for _ in banks
            ]

            def candidate():
                state, injection = initial, initial_injection
                for i, (norm, down, up) in enumerate(banks):
                    combined, xn, block, new_injection = outputs[i]
                    extension.run(
                        state,
                        cores[i],
                        injection,
                        norm,
                        down,
                        up,
                        combined,
                        xn,
                        block,
                        new_injection,
                        peers,
                        down_epochs,
                        up_epochs,
                        rank,
                        1e-6,
                        args.reduce_core,
                    )
                    state, injection = combined, new_injection
                return state, outputs[-1][2], injection

            def reference():
                state, injection = initial, initial_injection
                for i, (norm, down, up) in enumerate(banks):
                    core = comm.all_reduce(cores[i]) if args.reduce_core else cores[i]
                    state, xn = hc_combine_norm(state, core, injection, norm, 1e-6, 4)
                    if m == 1:
                        block, injection = torch.ops.vllm.qwen38_sm70_fp16_fused_hc(
                            xn, down, up, None, None, False, True
                        )
                    else:
                        packed_down, packed_up = packed_banks[i]
                        block, injection = torch.ops.vllm.qwen38_sm70_fp16_fused_hc(
                            xn, down, up, packed_down, packed_up, True, False
                        )
                return state, block, injection

            with patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp):
                oracle = tuple(t.clone() for t in reference())
                candidate()
                torch.cuda.synchronize()
                dist.barrier()
                errors = [
                    dict(
                        max_abs=float((a.float() - b.float()).abs().max()),
                        rel_l2=float((a.float() - b.float()).norm() / a.float().norm()),
                    )
                    for a, b in zip(oracle, candidate())
                ]
                torch.cuda.synchronize()
                graphs = {}
                for name, fn in (("control", reference), ("candidate", candidate)):
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with comm.capture(), torch.cuda.graph(graph):
                        fn()
                    graphs[name] = graph
                samples = {name: [] for name in graphs}
                for repetition in range(7):
                    for name in list(graphs)[:: 1 if repetition % 2 == 0 else -1]:
                        dist.barrier()
                        for _ in range(10):
                            graphs[name].replay()
                        start, end = [
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        ]
                        start.record()
                        for _ in range(args.replays):
                            graphs[name].replay()
                        end.record()
                        end.synchronize()
                        times = [None] * 4
                        dist.all_gather_object(
                            times, start.elapsed_time(end) / args.replays, group=group
                        )
                        samples[name].append(max(times))
                counts = {
                    name: graph_kernel_geometry(
                        graph, args.output, f"M{m}.{name}.rank{rank}"
                    )
                    for name, graph in graphs.items()
                }
                report["widths"].append(
                    dict(
                        m=m,
                        errors=errors,
                        samples_ms=samples,
                        medians_ms={
                            n: statistics.median(v) for n, v in samples.items()
                        },
                        measured_kernel_counts=counts,
                        control_scope=(
                            "existing HC M1; native replicated FP32 MMA "
                            "verifier HC for M5"
                        ),
                    )
                )
            dist.barrier()
            for i, p in enumerate(peers):
                if i != rank:
                    extension.close_peer(p)
            extension.release(own)

        for m in (1, 5):
            screen(m)
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
    finally:
        comm.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
