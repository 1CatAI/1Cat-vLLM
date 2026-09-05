# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Benchmark-only packed FP16 HC fusion; no activation quantization.

Real checkpoint weights, synthetic activations. Timings include the complete
down/inject -> FP16 -> SiLU -> FP16 -> up -> FP16 -> gate/mix chain.
No model-quality or endpoint-throughput claim. Loop closures are synchronous.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu


def capture(fn, unroll=32):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            output = fn()
    return graph, output


def timed(graph, repeats, unroll=32):
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / (repeats * unroll)


def error(actual, expected):
    a, b = actual.float(), expected.float()
    return dict(
        exact=torch.equal(actual, expected),
        finite=torch.isfinite(a).all().item(),
        max_abs=(a - b).abs().max().item(),
        relative_l2=((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", default="0")
    parser.add_argument("--pairs", default="attn,mlp")
    parser.add_argument("--tokens", default="4,8,16")
    parser.add_argument("--splits", default="4,8,16,20")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires SM70")
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        raise RuntimeError("Set a task-owned TORCH_EXTENSIONS_DIR")
    source = Path(__file__).with_name("sm70_hc_batch_screen.cu")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    library = load(
        name="sm70_hc_batch_screen",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
        is_python_module=False,
    )
    library_sha = hashlib.sha256(Path(library).read_bytes()).hexdigest()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def weight(name):
        with safe_open(args.model / index[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name).half().cuda().contiguous()

    rows = []
    torch.manual_seed(20260905)
    for layer in map(int, args.layers.split(",")):
        for pair in args.pairs.split(","):
            stem = f"model.language_model.layers.{layer}.{pair}_hyper_connection"
            down = weight(stem + ".input_mix_weight_down.weight")
            inject = weight(stem + ".block_inject_weight.weight")
            down = torch.cat((down, inject, down.new_zeros(12, 10240)))
            up = weight(stem + ".input_mix_weight_up.weight")
            # Lossless permutation/padding, done before capture. It costs one
            # extra persistent copy in this prototype and is explicitly reported.
            down_p = torch.cat((down, down.new_zeros(16, 10240)))
            down_p = down_p.view(11, 32, 640, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
            up_p = up.view(4, 80, 32, 20, 2, 8).permute(0, 1, 3, 4, 2, 5).contiguous()
            for m in map(int, args.tokens.split(",")):
                x = torch.randn(m, 10240, dtype=torch.float16, device="cuda")
                lora = x.new_empty(m, 320)
                injection = x.new_empty(m, 4)
                out = x.new_empty(m, 2560)
                scratch = torch.empty(20, m, 352, dtype=torch.float32, device="cuda")

                def baseline():
                    d = torch.nn.functional.linear(x, down)
                    h = hc_silu(d[:, :320], 4)
                    return hc_gate_mix(x, torch.nn.functional.linear(h, up), 4), d[
                        :, 320:324
                    ]

                def up_only():
                    d = torch.nn.functional.linear(x, down)
                    h = hc_silu(d[:, :320], 4)
                    torch.ops.sm70_hc_screen.up(out, h, up_p, x)
                    return out, d[:, 320:324]

                base_graph, base_outputs = capture(baseline)
                for split in [0, *map(int, args.splits.split(","))]:

                    def full():
                        torch.ops.sm70_hc_screen.down(
                            lora, injection, x, down_p, scratch, split
                        )
                        torch.ops.sm70_hc_screen.up(out, lora, up_p, x)
                        return out, injection

                    fn = full if split else up_only
                    candidate_graph, candidate_outputs = capture(fn)
                    checks = []
                    for scale in (0.25, 1.0, 3.0):
                        x.normal_().mul_(scale)
                        expected = tuple(t.clone() for t in baseline())
                        actual = tuple(t.clone() for t in fn())
                        out.fill_(float("nan"))
                        scratch.fill_(float("nan"))
                        lora.fill_(float("nan"))
                        injection.fill_(float("nan"))
                        candidate_graph.replay()
                        torch.cuda.synchronize()
                        graph_errors = [
                            error(a, b) for a, b in zip(candidate_outputs, actual)
                        ]
                        assert all(e["exact"] and e["finite"] for e in graph_errors)
                        checks.append(
                            dict(
                                scale=scale,
                                block=error(actual[0], expected[0]),
                                injection=error(actual[1], expected[1]),
                            )
                        )
                    assert all(
                        c[k]["finite"] for c in checks for k in ("block", "injection")
                    )
                    # Alternate A/B order with fixed pointers and amortized replay
                    # submission; do not promote a single noisy minimum.
                    a, b = [], []
                    for sample in range(args.samples):
                        for graph, dest in (
                            ((base_graph, a), (candidate_graph, b))
                            if sample % 2 == 0
                            else ((candidate_graph, b), (base_graph, a))
                        ):
                            dest.append(timed(graph, args.repeats))
                    row = dict(
                        layer=layer,
                        pair=pair,
                        tokens=m,
                        split=split,
                        baseline_us=statistics.median(a),
                        candidate_us=statistics.median(b),
                        paired_saving_us=statistics.median(
                            [aa - bb for aa, bb in zip(a, b)]
                        ),
                        baseline_samples=a,
                        candidate_samples=b,
                        checks=checks,
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    del candidate_graph, candidate_outputs
                del base_graph, base_outputs
    payload = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        model=str(args.model),
        args=vars(args),
        source_sha256=source_sha,
        library_sha256=library_sha,
        extra_weight_mib_per_hc=(352 * 10240 + 10240 * 320) * 2 / 2**20,
        rows=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
