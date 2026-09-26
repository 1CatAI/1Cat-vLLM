# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same real TP4 shards and layer weights as the C1-C8 GEMM diagnostic.

Load a full normal-build _C artifact in each fresh process. Timings include
activation preparation and gate/up activation. This is not a serving benchmark.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

COUNTS = {
    "fp4_gate_up": 56,
    "fp4_down": 56,
    "fp8_out": 64,
    "fp8_in": 48,
    "fp8_qkv": 16,
    "fp8_gate_up": 8,
    "fp8_down": 8,
}


def check_exclusive():
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    pids = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    others = [int(p) for p in pids.split() if p.isdigit() and int(p) != os.getpid()]
    if others:
        raise RuntimeError(f"Benchmark GPU is also used by {others}")


def tp_weights(model, gated, layer):
    prefix = f"model.language_model.layers.{layer}.mlp."
    with safe_open(str(model), framework="pt", device="cpu") as f:
        if gated:
            packed = torch.cat(
                [
                    f.get_slice(prefix + p + ".weight_packed")[:4352]
                    for p in ("gate_proj", "up_proj")
                ]
            )
            scales = torch.cat(
                [
                    f.get_slice(prefix + p + ".weight_scale")[:4352]
                    for p in ("gate_proj", "up_proj")
                ]
            )
            scale = 1.0 / float(f.get_tensor(prefix + "gate_proj.weight_global_scale"))
        else:
            packed = f.get_slice(prefix + "down_proj.weight_packed")[:, :2176]
            scales = f.get_slice(prefix + "down_proj.weight_scale")[:, :272]
            scale = 1.0 / float(f.get_tensor(prefix + "down_proj.weight_global_scale"))
    return packed.cuda().contiguous(), scales.cuda().contiguous(), scale


def capture(call, calls):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            call()
    return graph


def fp8_weights(model, kind, layer=0):
    prefix = f"model.language_model.layers.{layer}.linear_attn."
    with safe_open(str(model), framework="pt", device="cpu") as f:
        if kind == "fp8_qkv":
            # Full attention appears every fourth layer.
            prefix = f"model.language_model.layers.{layer // 4 * 4 + 3}.self_attn."
            w = torch.cat(
                [
                    f.get_slice(prefix + p + ".weight")[:size]
                    for p, size in (("q_proj", 3072), ("k_proj", 256), ("v_proj", 256))
                ]
            )
            s = torch.cat(
                [
                    f.get_slice(prefix + p + ".weight_scale")[:size]
                    for p, size in (("q_proj", 3072), ("k_proj", 256), ("v_proj", 256))
                ]
            )
            split = 16
        elif kind == "fp8_down":
            prefix = f"model.language_model.layers.{56 + layer % 8}.mlp."
            w = f.get_slice(prefix + "down_proj.weight")[:, :4352]
            s = f.get_tensor(prefix + "down_proj.weight_scale")
            split = 16
        elif kind == "fp8_out":
            w = f.get_slice(prefix + "out_proj.weight")[:, :1536]
            s = f.get_tensor(prefix + "out_proj.weight_scale")
            split = 12
        elif kind == "fp8_in":
            w = torch.cat(
                [
                    f.get_slice(prefix + p + ".weight")[:size]
                    for p, size in (("in_proj_qkv", 2560), ("in_proj_z", 1536))
                ]
            )
            s = torch.cat(
                [
                    f.get_slice(prefix + p + ".weight_scale")[:size]
                    for p, size in (("in_proj_qkv", 2560), ("in_proj_z", 1536))
                ]
            )
            split = 16
        else:
            raise ValueError(kind)
    return w.cuda().contiguous(), s.cuda().float().contiguous(), split


def gpu_state():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def prepare(args, kind):
    gated = kind.endswith("gate_up")
    if kind.startswith("fp4"):
        packed, scales, global_scale = tp_weights(args.model, gated, args.layer)
        n, k2 = packed.shape
        k = 2 * k2
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        w, s, meta = torch.ops._C.nvfp4_sm70_prepare(
            codes.t().contiguous(),
            (scales.t().float() * global_scale).half().contiguous(),
            16,
            False,
        )
        compact = torch.ops._C.nvfp4_qpn2_prepare_scales_sm70(scales)

        def call(x, y):
            torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
                y,
                x,
                w,
                compact,
                global_scale,
                8 if gated else 16,
                2,
                s,
                16,
                int(meta[0]),
                int(meta[1]),
                gated,
                0,
            )

        return k, n, gated, call
    if gated:
        with safe_open(str(args.model), framework="pt", device="cpu") as f:
            prefix = f"model.language_model.layers.{56 + args.layer % 8}.mlp."
            weight = (
                torch.cat(
                    [
                        f.get_slice(prefix + p + ".weight")[:4352]
                        for p in ("gate_proj", "up_proj")
                    ]
                )
                .cuda()
                .contiguous()
            )
            scale = (
                torch.cat(
                    [
                        f.get_slice(prefix + p + ".weight_scale")[:4352]
                        for p in ("gate_proj", "up_proj")
                    ]
                )
                .cuda()
                .float()
                .contiguous()
            )
        split = 8
    else:
        weight, scale, split = fp8_weights(args.model, kind, args.layer)
    n, k = weight.shape
    codes, scales = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scale)
    w, s, meta = torch.ops._C.fp8_sm70_prepare(weight, scale, 128, gated)

    def call(x, y):
        if x.shape[0] <= 32:
            torch.ops._C.fp8_qpn8_dispatch_sm70_out(
                y, 0, x, codes, scales, split, 2, False, gated
            )
        else:
            torch.ops._C.fp8_gemm_sm70_out(
                y, x, w, s, 128, int(meta[0]), int(meta[1]), gated
            )

    return k, n, gated, call


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--oracles", type=Path, required=True)
    p.add_argument("--write-oracles", action="store_true")
    p.add_argument("--require-bitwise", action="store_true")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", type=int, nargs="+", default=[8, 16, 17, 24, 32, 64])
    p.add_argument("--kinds", nargs="+", default=list(COUNTS))
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(931)
    check_exclusive()
    torch.ops.load_library(str(args.extension))
    args.oracles.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        extension=str(args.extension),
        extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        layer=args.layer,
        require_bitwise=args.require_bitwise,
        gpu_before=gpu_state(),
        env={k: v for k, v in os.environ.items() if k.startswith("VLLM_SM70_")},
        note="Single GPU TP4 shards; layer-weighted estimate, no service/profiler.",
    )
    results = []
    for kind in args.kinds:
        check_exclusive()
        k, n, gated, call = prepare(args, kind)
        cases = {}
        for m in args.rows:
            x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
            y = torch.empty(m, n // 2 if gated else n, device="cuda", dtype=x.dtype)

            def one(x=x, y=y, call=call):
                call(x, y)

            graph = capture(one, 8)
            for i, amp in enumerate((0.1, 0.3, 0.01)):
                x.normal_(0, amp)
                one()
                reference = y.clone()
                y.fill_(float("nan"))
                graph.replay()
                assert torch.equal(reference.view(torch.int16), y.view(torch.int16))
                path = args.oracles / f"{kind}-m{m}-{i}.pt"
                if args.write_oracles:
                    torch.save(y.cpu(), path)
                else:
                    oracle = torch.load(path, weights_only=True).cuda()
                    if m <= 32 or args.require_bitwise:
                        assert torch.equal(
                            oracle.view(torch.int16), y.view(torch.int16)
                        ), (kind, m, i, float((oracle - y).abs().max()))
                    else:
                        torch.testing.assert_close(y, oracle, rtol=0.02, atol=0.0003)
            cases[m] = (x, y, graph, [], [])
        for _ in range(100):
            for x, y, graph, samples, eager_samples in cases.values():
                graph.replay()
        torch.cuda.synchronize()
        for r in range(7):
            order = list(cases)
            random.Random(932 + r).shuffle(order)
            for m in order:
                x, y, graph, samples, eager_samples = cases[m]
                a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                for _ in range(20):
                    graph.replay()
                b.record()
                b.synchronize()
                samples.append(a.elapsed_time(b) * 1000 / 160)
                # Include host launch gaps and temporary activation allocation
                # in the eager operator measurement, separately from replay.
                a.record()
                for _ in range(20):
                    call(x, y)
                b.record()
                b.synchronize()
                eager_samples.append(a.elapsed_time(b) * 1000 / 20)
        for m, (x, y, graph, samples, eager_samples) in cases.items():
            row = dict(
                kind=kind,
                m=m,
                n=n,
                k=k,
                count=COUNTS[kind],
                us=statistics.median(samples),
                samples_us=samples,
                eager_us=statistics.median(eager_samples),
                eager_samples_us=eager_samples,
                eager_graph_exact=True,
                oracle_check="saved"
                if args.write_oracles
                else ("bitwise" if m <= 32 or args.require_bitwise else "tolerance"),
            )
            print(json.dumps(row), flush=True)
            results.append(row)
        args.output.write_text(json.dumps(results, indent=2))
        del cases, call
    summary = []
    for m in args.rows:
        rows = [r for r in results if r["m"] == m]
        fp4 = sum(
            r["us"] * r["count"] / 1000 for r in rows if r["kind"].startswith("fp4")
        )
        fp8 = sum(
            r["us"] * r["count"] / 1000 for r in rows if r["kind"].startswith("fp8")
        )
        summary.append(dict(m=m, fp4_ms=fp4, fp8_ms=fp8, total_ms=fp4 + fp8))
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    metadata["gpu_after"] = gpu_state()
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
