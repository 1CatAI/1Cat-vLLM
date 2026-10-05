# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole router/top-10/W13/SiLU/W2/reduction graph screen on NVFP4 weights.

TP-local expert computation excludes the shared expert and following TP sum.
M1 uses the native fused expert pair; M5 uses the direct split4 verifier path.
No runtime dispatch is installed. Prebuild on CPU with --build-only.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

from benchmarks.kernels.sm70_chain_screen_utils import graph_kernel_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--expert-only", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).parents[1] / "csrc/sm70_router_expert_chain_screen.cu"
    extension = load(
        "sm70_router_expert_chain_screen",
        [str(source)],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        verbose=True,
    )
    if args.build_only:
        args.output.write_text(json.dumps({"library": extension.__file__}) + "\n")
        return
    from benchmarks.kernels.benchmark_sm70_moe_packed_w13 import checkpoint_weights
    from vllm import _sm70_ops as ops
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
    from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
        _mtp_weighted_reduce,
    )
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
        _pack_router_batch_weight,
        _qwen38_sm70_fp16_gemv,
    )

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    mapping = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    banks = []
    for layer in range(args.layers):
        name = f"model.language_model.layers.{layer}.mlp.gate.weight"
        with safe_open(args.model / mapping[name], framework="pt", device="cpu") as f:
            router = f.get_tensor(name).half().cuda()
        banks.append(
            (
                router,
                _pack_router_batch_weight(router),
                checkpoint_weights(args.model, layer, 0, True),
            )
        )
    report = {
        "research_only": True,
        "model_admission": False,
        "scope": (
            "complete experts; precomputed routing; no shared expert or TP reduction"
            if args.expert_only
            else "router/top-10/complete experts; no shared expert or TP reduction"
        ),
        "includes_router": not args.expert_only,
        "checkpoint_weights": True,
        "synthetic_activations": True,
        "tp_shard": 0,
        "layers": args.layers,
        "widths": [],
    }

    def screen(m):
        torch.manual_seed(4201)
        inputs = [
            torch.randn(m, 2560, device="cuda", dtype=torch.float16) * 0.1
            for _ in banks
        ]
        ids = [torch.empty(m, 10, device="cuda", dtype=torch.int32) for _ in banks]
        weights = [
            torch.empty(m, 10, device="cuda", dtype=torch.float32) for _ in banks
        ]
        partials = [
            torch.empty(m, 100, 2560, device="cuda", dtype=torch.float32) for _ in banks
        ]
        flags = [torch.zeros(m, 101, device="cuda", dtype=torch.int32) for _ in banks]
        epochs = [torch.zeros(121, device="cuda", dtype=torch.int32) for _ in banks]
        outputs = [torch.empty_like(x) for x in inputs]
        intermediate = [
            torch.empty(m * 10, 160, device="cuda", dtype=torch.float16) for _ in banks
        ]
        gate_up = [
            torch.empty(m * 10, 320, device="cuda", dtype=torch.float16) for _ in banks
        ]
        down = [
            torch.empty(m * 10, 2560, device="cuda", dtype=torch.float16) for _ in banks
        ]
        controls = [torch.empty_like(x) for x in inputs]
        control_ids = []

        def control():
            control_ids.clear()
            for i, (x, (router, packed, (w13, s13, w2, s2))) in enumerate(
                zip(inputs, banks)
            ):
                if args.expert_only:
                    routing_weights, routing_ids = weights[i], ids[i]
                else:
                    logits = _qwen38_sm70_fp16_gemv(x, router, ".mlp.gate", packed)
                    routing_weights, routing_ids, _ = fused_topk(x, logits, 10, True)
                control_ids.append(routing_ids)
                route = routing_ids.view(-1)
                if m == 1:
                    ops.nvfp4_qwen38_w13_fused_swiglu_out(
                        intermediate[i], x, w13, s13, route
                    )
                    ops.nvfp4_qwen38_w2_direct_reduce_out(
                        controls[i], intermediate[i], w2, s2, route, routing_weights
                    )
                else:
                    ops.nvfp4_moe_qpn_mtp5_sm70_out(
                        gate_up[i], x, w13, s13, route, True, 4
                    )
                    ops.silu_and_mul_interleaved(intermediate[i], gate_up[i])
                    ops.nvfp4_moe_qpn_mtp5_sm70_out(
                        down[i], intermediate[i], w2, s2, route, False, 1
                    )
                    _mtp_weighted_reduce(down[i], routing_weights, controls[i])

        def candidate():
            for i, (x, (router, _, prepared)) in enumerate(zip(inputs, banks)):
                extension.run(
                    x,
                    router,
                    ids[i],
                    weights[i],
                    *prepared,
                    partials[i],
                    outputs[i],
                    flags[i],
                    epochs[i],
                    not args.expert_only,
                )

        errors = []
        for scale in (0.3, 1.0, 3.0):
            for x in inputs:
                x.normal_(0, 0.1 * scale)
            if args.expert_only:
                for i, (x, (router, packed, _)) in enumerate(zip(inputs, banks)):
                    logits = _qwen38_sm70_fp16_gemv(x, router, ".mlp.gate", packed)
                    routing_weights, routing_ids, _ = fused_topk(x, logits, 10, True)
                    ids[i].copy_(routing_ids)
                    weights[i].copy_(routing_weights)
            control()
            candidate()
            torch.cuda.synchronize()
            errors.append(
                {
                    "scale": scale,
                    "selected_ids_equal": [
                        bool(torch.equal(a, b)) for a, b in zip(control_ids, ids)
                    ],
                    "max_abs": [
                        float((a.float() - b.float()).abs().max())
                        for a, b in zip(controls, outputs)
                    ],
                    "rel_l2": [
                        float((a.float() - b.float()).norm() / a.float().norm())
                        for a, b in zip(controls, outputs)
                    ],
                }
            )
        graphs = {}
        for name, fn in (("control", control), ("candidate", candidate)):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                fn()
            graphs[name] = graph
        samples = {name: [] for name in graphs}
        for rep in range(7):
            for name in list(graphs)[:: 1 if rep % 2 == 0 else -1]:
                for _ in range(10):
                    graphs[name].replay()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(100):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 100)
        counts = {
            name: graph_kernel_geometry(graph, args.output, f"M{m}.{name}")
            for name, graph in graphs.items()
        }
        report["widths"].append(
            dict(
                m=m,
                errors=errors,
                samples_ms=samples,
                medians_ms={n: statistics.median(v) for n, v in samples.items()},
                measured_kernel_counts=counts,
            )
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for m in (1, 5):
        screen(m)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
