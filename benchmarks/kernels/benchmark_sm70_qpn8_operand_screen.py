# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight FP8 projection screen; projection error is not a model KL gate."""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _sm70_ops as ops


def load_cases(model):
    cases = []
    with safe_open(str(model / "model.safetensors"), framework="pt", device="cpu") as f:
        prefix = "model.language_model.layers.0.linear_attn."
        rows = list(range(512)) + list(range(2048, 2560)) + list(range(4096, 5632))
        weight = torch.cat(
            (
                f.get_tensor(prefix + "in_proj_qkv.weight").view(torch.uint8)[rows],
                f.get_tensor(prefix + "in_proj_z.weight").view(torch.uint8)[:1536],
            )
        ).view(torch.float8_e4m3fn)
        scale = torch.cat(
            (
                f.get_tensor(prefix + "in_proj_qkv.weight_scale")[rows],
                f.get_tensor(prefix + "in_proj_z.weight_scale")[:1536],
            )
        ).float()
        ba = torch.cat(
            (
                f.get_tensor(prefix + "in_proj_b.weight")[:12],
                f.get_tensor(prefix + "in_proj_a.weight")[:12],
            )
        ).half()
        cases.append(("gdn_qkvz_ba", 0, weight, scale, ba))
        cases.append(
            (
                "gdn_output",
                1,
                f.get_tensor(prefix + "out_proj.weight")[:, :1536].contiguous(),
                f.get_tensor(prefix + "out_proj.weight_scale").float(),
                None,
            )
        )
        prefix = "model.language_model.layers.3.self_attn."
        weight = torch.cat(
            [
                f.get_tensor(prefix + name + ".weight").view(torch.uint8)[:n]
                for name, n in [("q_proj", 3072), ("k_proj", 256), ("v_proj", 256)]
            ]
        ).view(torch.float8_e4m3fn)
        scale = torch.cat(
            [
                f.get_tensor(prefix + name + ".weight_scale")[:n]
                for name, n in [("q_proj", 3072), ("k_proj", 256), ("v_proj", 256)]
            ]
        ).float()
        cases.append(("attention_qkv", 2, weight, scale, None))
        prefix = "model.language_model.layers.56.mlp."
        cases.append(
            (
                "fp8_mlp_down",
                3,
                f.get_tensor(prefix + "down_proj.weight")[:, :4352].contiguous(),
                f.get_tensor(prefix + "down_proj.weight_scale").float(),
                None,
            )
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--eviction-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--n16", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--unroll", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.ops.load_library(str(args.library))
    torch.ops.load_library(str(args.eviction_library))
    extension = torch.ops._qpn8_operands700
    eviction = torch.ones(33554432, device="cuda", dtype=torch.int32)
    sink = torch.empty(256, device="cuda", dtype=torch.int64)
    records = []
    for name, kind, weight, scale, ba in load_cases(args.model):
        if args.cases and name not in args.cases:
            continue
        n, k = weight.shape
        weight, scale = weight.cuda(), scale.cuda()
        codes, scales = ops.fp8_qpn8_prepare_sm70(weight, scale)
        dense = weight.float() * scale.view(n, 1)
        x = torch.randn(8, k, device="cuda", dtype=torch.float16) * 0.1
        packed = torch.empty_like(x)
        out = x.new_empty((8, 2560 if kind == 0 else n))
        z, b, a = x.new_empty((8, 1536)), x.new_empty((8, 12)), x.new_empty((8, 12))
        ba_weight = ba.cuda() if ba is not None else x.new_empty(1)
        if kind == 0:
            production = partial(
                ops.fp8_qpn8_dispatch_ba_split_sm70_out,
                out,
                z,
                b,
                a,
                x.new_empty((8, 4096)),
                x.new_empty((8, 24)),
                0,
                x,
                codes,
                scales,
                ba_weight,
            )
        else:
            production = partial(
                ops.fp8_qpn8_gemm_sm70_out,
                out,
                x,
                codes,
                scales,
                12 if kind == 1 else 16,
                2,
                True,
                False,
            )
        functions = [production] + [
            partial(
                extension.run,
                out,
                z,
                b,
                a,
                packed if v in [1, 3] else x,
                codes,
                scales,
                ba_weight,
                kind,
                v,
            )
            for v in range(4)
        ]
        names = ["production", "control", "packed", "postscale", "packed_postscale"]
        if args.unroll:
            functions.extend(
                partial(
                    extension.run, out, z, b, a, x, codes, scales, ba_weight, kind, v
                )
                for v in [5, 6]
            )
            names.extend(["loop1", "loop2"])
        if kind == 1 and args.n16:
            functions.append(
                partial(
                    extension.run,
                    out,
                    z,
                    b,
                    a,
                    packed,
                    codes,
                    scales,
                    ba_weight,
                    kind,
                    4,
                )
            )
            names.append("n16_packed_postscale")
        checks = []
        for amplitude in [0.0, 0.125, -0.125, 0.25]:
            x.copy_(torch.randn_like(x) * amplitude)
            packed.copy_(
                x.view(8, k // 16, 16).permute(1, 0, 2).contiguous().view_as(x)
            )
            reference = x.float() @ dense.t()
            production()
            expected = torch.cat((out, z), dim=1) if kind == 0 else out.clone()
            expected_ba = torch.cat((b, a), dim=1) if kind == 0 else None
            for variant, fn in zip(names, functions, strict=True):
                fn()
                actual = torch.cat((out, z), dim=1) if kind == 0 else out
                exact = torch.equal(
                    actual.view(torch.int16), expected.view(torch.int16)
                )
                if variant in ["control", "packed", "loop1", "loop2"] and not exact:
                    raise AssertionError((name, variant, "bitwise mismatch"))
                if kind == 0 and not torch.equal(torch.cat((b, a), dim=1), expected_ba):
                    raise AssertionError((name, variant, "BA changed"))
                error = actual.float() - reference
                if not torch.isfinite(error).all().item():
                    raise AssertionError((name, variant, "nonfinite"))
                checks.append(
                    {
                        "variant": variant,
                        "amplitude": amplitude,
                        "bitwise": exact,
                        "max_abs_dense_fp32": error.abs().max().item(),
                        "rms_dense_fp32": error.square().mean().sqrt().item(),
                    }
                )
        graphs = []
        for fn in functions:
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            begin = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            with torch.cuda.graph(graph):
                torch.ops._qpn_operands700.evict(eviction, sink)
                begin.record()
                fn()
                end.record()
            graphs.append((graph, begin, end))
        samples = {variant: [] for variant in names}
        if args.profile:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for graph, _, _ in [graphs[1], graphs[-1]]:
                graph.replay()
                torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        for repeat in range(45):
            for offset in range(len(graphs)):
                index = (repeat + offset) % len(graphs)
                graph, begin, end = graphs[index]
                for _ in range(10):
                    graph.replay()
                end.synchronize()
                if repeat >= 5:
                    samples[names[index]].append(begin.elapsed_time(end) * 1000)
        record = {
            "case": name,
            "shape": [8, n, k],
            "checks": checks,
            "weight_scale_bytes": codes.numel()
            + scales.numel() * 2
            + (ba.numel() * 2 if ba is not None else 0),
            "timing": {
                variant: {"mean_us": statistics.mean(values), "samples_us": values}
                for variant, values in samples.items()
            },
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "case": name,
                    "mean_us": {
                        variant: statistics.mean(values)
                        for variant, values in samples.items()
                    },
                }
            ),
            flush=True,
        )
        for graph, _, _ in graphs:
            graph.reset()
    args.output.write_text(
        json.dumps(
            {
                "research_only": True,
                "model_KL_gate": "not run",
                "packing_timed": False,
                "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
