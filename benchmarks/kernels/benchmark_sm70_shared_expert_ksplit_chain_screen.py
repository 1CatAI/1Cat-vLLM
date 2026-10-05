# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research screen of the complete shared-expert chain, including its gate.

Run after acquiring all GPU locks; prebuild with --build-only while waiting.
No runtime route is installed. TP-local compute excludes the following TP sum.
"""

import argparse
import hashlib
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
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--widths", type=int, nargs="+", choices=(1, 5), default=(1, 5))
    parser.add_argument("--native-mtp-control", action="store_true")
    args = parser.parse_args()
    source = (
        Path(__file__).parents[1] / "csrc/sm70_shared_expert_ksplit_chain_screen.cu"
    )
    extension = load(
        "sm70_shared_expert_ksplit_chain_screen",
        [str(source)],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        verbose=True,
    )
    if args.build_only:
        args.output.write_text(json.dumps({"library": extension.__file__}) + "\n")
        return
    assert torch.cuda.get_device_capability() == (7, 0)
    import vllm._custom_ops  # noqa: F401
    from vllm import _sm70_ops as sm70_ops

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    mapping = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def get(name):
        with safe_open(args.model / mapping[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name).half()

    weights = []
    for i in range(args.layers):
        prefix = f"model.language_model.layers.{i}.mlp."
        w13 = torch.cat(
            [
                get(prefix + "shared_expert." + p + ".weight")[:160]
                for p in ("gate_proj", "up_proj")
            ]
        ).cuda()
        w2 = get(prefix + "shared_expert.down_proj.weight")[:, :160].contiguous().cuda()
        gate = get(prefix + "shared_expert_gate.weight").cuda()
        weights.append((w13, w2, gate))
    packed_up = [
        torch.stack((w13[:160], w13[160:]), dim=1)
        .reshape(320, 2560)
        .reshape(10, 32, 160, 2, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        for w13, _, _ in weights
    ]
    packed_down = [
        w2.reshape(80, 32, 10, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
        for _, w2, _ in weights
    ]
    native_packed_up = [
        w13.reshape(10, 32, 160, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
        for w13, _, _ in weights
    ]
    report = {
        "research_only": True,
        "model_admission": False,
        "cta_count": 40,
        "w13_k_splits": 4,
        "native_mtp_control": args.native_mtp_control,
        "scope": "complete TP-local shared expert; following all-reduce excluded",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "layers": args.layers,
        "widths": [],
    }

    def screen(m):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = bool(
            m == 5 and args.native_mtp_control
        )
        torch.manual_seed(4201)
        inputs = [
            torch.randn(m, 2560, device="cuda", dtype=torch.float16) * 0.1
            for _ in weights
        ]
        partials = [
            torch.empty(10, m, 2560, device="cuda", dtype=torch.float32)
            for _ in weights
        ]
        flags = [torch.zeros(80, device="cuda", dtype=torch.int32) for _ in weights]
        epochs = [torch.zeros(40, device="cuda", dtype=torch.int32) for _ in weights]
        projections = [
            torch.empty(40, m, 32, device="cuda", dtype=torch.float32) for _ in weights
        ]
        outputs = [torch.empty_like(x) for x in inputs]
        native_intermediate = [x.new_empty((m, 160)) for x in inputs]
        native_partial = [x.new_empty((8, m, 320)) for x in inputs]

        def candidate():
            for x, (_, _, gate), w13, w2, proj, p, f, e, y in zip(
                inputs,
                weights,
                packed_up,
                packed_down,
                projections,
                partials,
                flags,
                epochs,
                outputs,
            ):
                extension.run(x, w13, w2, gate, proj, p, f, e, y)

        def control():
            result = []
            for i, (x, (w13, w2, gate)) in enumerate(zip(inputs, weights)):
                if m == 5 and args.native_mtp_control:
                    intermediate = native_intermediate[i]
                    torch.ops._C.qwen38_shared_up_batch_sm70_out(
                        intermediate, native_partial[i], x, native_packed_up[i]
                    )
                else:
                    projected = torch.nn.functional.linear(x, w13)
                    intermediate = x.new_empty((m, 160))
                    torch.ops._C.silu_and_mul(intermediate, projected)
                y = torch.nn.functional.linear(intermediate, w2)
                if m == 1:
                    sm70_ops.qwen38_shared_gate_exact_out(y, x, gate)
                else:
                    g = torch.nn.functional.linear(x, gate)
                    sm70_ops.qwen38_shared_gate_sigmoid_mul_out(y, g)
                result.append(y)
            return result

        expected = control()
        candidate()
        torch.cuda.synchronize()
        errors = [
            dict(
                max_abs=float((a.float() - b.float()).abs().max()),
                rel_l2=float((a.float() - b.float()).norm() / a.float().norm()),
            )
            for a, b in zip(expected, outputs)
        ]
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

    for m in args.widths:
        screen(m)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
