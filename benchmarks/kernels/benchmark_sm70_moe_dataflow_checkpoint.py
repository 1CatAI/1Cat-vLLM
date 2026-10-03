# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import fcntl
import json
import os
import statistics
import subprocess
from pathlib import Path

os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
os.environ["MAX_JOBS"] = "2"
import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as native
from vllm.model_executor.layers.quantization.sm70_turbomind import unpack_mxfp4_weight

ROOT = Path(__file__).parent
MODEL = None
SELECTED = (0, 7, 55, 99, 120, 211, 256, 310, 401, 470)


def prepare_layer(layer, rank=0):
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    handles = {}

    def read(e, suffix):
        key = f"{prefix}.{e}.{suffix}"
        shard = index[key]
        if shard not in handles:
            handles[shard] = safe_open(MODEL / shard, framework="pt", device="cpu")
        return handles[shard].get_tensor(key)

    prepared = []
    for expert in SELECTED:
        raw13 = torch.cat(
            [
                read(expert, p + ".weight")[rank * 160 : (rank + 1) * 160]
                for p in ("gate_proj", "up_proj")
            ]
        ).cuda()
        raw2 = (
            read(expert, "down_proj.weight")[:, rank * 80 : (rank + 1) * 80]
            .contiguous()
            .cuda()
        )
        s13 = (
            torch.cat(
                [
                    read(expert, p + ".weight_scale")[
                        rank * 160 : (rank + 1) * 160
                    ].float()
                    * read(expert, p + ".weight_scale_2").float().reshape(())
                    for p in ("gate_proj", "up_proj")
                ]
            )
            .half()
            .cuda()
        )
        s2 = (
            (
                read(expert, "down_proj.weight_scale")[
                    :, rank * 10 : (rank + 1) * 10
                ].float()
                * read(expert, "down_proj.weight_scale_2").float().reshape(())
            )
            .half()
            .cuda()
        )
        w13 = native.nvfp4_sm70_prepare(
            unpack_mxfp4_weight(raw13), s13.t().contiguous(), 16, True
        )
        w2 = native.nvfp4_sm70_prepare(
            unpack_mxfp4_weight(raw2), s2.t().contiguous(), 16
        )
        prepared.append((w13[0], w13[1], w2[0], w2[1]))
    result = []
    for idx in range(4):
        first = prepared[0][idx]
        buffer = torch.empty((512, *first.shape), dtype=first.dtype, device="cuda")
        for dest, tensors in enumerate(prepared):
            buffer[dest].copy_(tensors[idx])
        result.append(buffer)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counter-only", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    global MODEL
    MODEL = args.model
    args.build.mkdir(parents=True, exist_ok=True)
    ops = load(
        name="sm70_moe_dataflow_research",
        sources=[str(ROOT / "sm70_moe_dataflow_research.cu")],
        build_directory=str(args.build),
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )
    lock_path = Path("/tmp/gpu0-3.lock")
    if not lock_path.exists():
        lock_path.touch()
    with lock_path.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        live = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0,1,2,3",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if live:
            raise SystemExit("GPU busy; screen not started")
        torch.manual_seed(9711)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        x = torch.randn((1, 2560), device="cuda", dtype=torch.float16)
        ids = torch.arange(10, device="cuda", dtype=torch.int32).reshape(1, 10)
        weights = torch.softmax(torch.randn((1, 10), device="cuda"), -1)
        inter = torch.empty((10, 160), device="cuda", dtype=torch.float16)
        out = torch.empty_like(x)
        control_out = torch.empty_like(x)
        partial = torch.empty((100, 2560), device="cuda", dtype=torch.float32)
        control = torch.zeros(2665, device="cuda", dtype=torch.int32)
        all_weights = [
            prepare_layer(layer)
            for layer in ((0,) if args.counter_only else (0, 15, 31, 47))
        ]

        def baseline(w):
            native.nvfp4_qwen38_w13_fused_swiglu_out(
                inter, x, w[0], w[1], ids.reshape(-1)
            )
            native.nvfp4_qwen38_w2_direct_reduce_out(
                control_out, inter, w[2], w[3], ids.reshape(-1), weights
            )

        def candidate(w):
            ops.nvfp4(
                x, ids, weights, w[0], w[1], w[2], w[3], partial, out, control, 160, 512
            )

        if args.counter_only:
            baseline(all_weights[0])
            candidate(all_weights[0])
            torch.cuda.synchronize()
            return
        checks = []
        for layer, w in zip((0, 15, 31, 47), all_weights):
            for scale in (0.03, 0.3, 1, 2):
                x.copy_(torch.randn_like(x) * scale)
                baseline(w)
                candidate(w)
                torch.cuda.synchronize()
                d = out.float() - control_out.float()
                check = dict(
                    layer=layer,
                    scale=scale,
                    max_abs=d.abs().max().item(),
                    relative_l2=(
                        d.norm() / control_out.float().norm().clamp_min(1e-12)
                    ).item(),
                    finite=bool(torch.isfinite(out).all()),
                )
                checks.append(check)
                print("CHECK", json.dumps(check), flush=True)
                torch.testing.assert_close(out, control_out, atol=5e-4, rtol=0.005)
        graphs = []
        for fn in (baseline, candidate):
            for _ in range(8):
                for w in all_weights:
                    fn(w)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    for w in all_weights:
                        fn(w)
            graphs.append(graph)
        timings = [[], []]
        for repeat in range(7):
            for arm in (repeat % 2, 1 - repeat % 2):
                a, b = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                a.record()
                for _ in range(50):
                    graphs[arm].replay()
                b.record()
                b.synchronize()
                timings[arm].append(
                    a.elapsed_time(b) * 1000 / (50 * 8 * len(all_weights))
                )
        report = dict(
            research_only=True,
            synthetic_activations=True,
            checkpoint_weights=True,
            tp_shard=0,
            layers=[0, 15, 31, 47],
            selected_experts=list(SELECTED),
            remapped_route_ids=list(range(10)),
            all_selected_experts_included=True,
            includes_router=False,
            includes_shared_expert=False,
            includes_communication=False,
            control="installed native W13+SiLU and W2+weighted-reduce",
            candidate="one dataflow expert kernel",
            checks=checks,
            baseline_us=timings[0],
            candidate_us=timings[1],
            baseline_median_us=statistics.median(timings[0]),
            candidate_median_us=statistics.median(timings[1]),
        )
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
