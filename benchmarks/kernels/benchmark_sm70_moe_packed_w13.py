# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Paired graph screen, including grouping, W13/SiLU and unchanged fused W2.

Uses production native ops, or builds the same source for standalone screening. Use
--model for actual checkpoint weights; otherwise only synthetic screening is
performed. Activations are synthetic in both modes. No model quality claim.
Loop-local closures are consumed synchronously before advancing the loop.
"""

import argparse
import glob
import hashlib
import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    unpack_mxfp4_weight,
)


def checkpoint_weights(model, layer, rank, interleaved):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prepared = [[], [], [], []]
    with ExitStack() as stack:
        handles = {}

        def tensor(expert, projection, suffix):
            key = (
                f"model.language_model.layers.{layer}.mlp.experts."
                f"{expert}.{projection}.{suffix}"
            )
            shard = index[key]
            if shard not in handles:
                handles[shard] = stack.enter_context(
                    safe_open(model / shard, framework="pt", device="cpu")
                )
            return handles[shard].get_tensor(key)

        for e in range(512):
            weights, scales = [], []
            for p in ("gate_proj", "up_proj"):
                weights.append(tensor(e, p, "weight")[rank * 160 : (rank + 1) * 160])
                scales.append(
                    tensor(e, p, "weight_scale")[rank * 160 : (rank + 1) * 160].float()
                    * tensor(e, p, "weight_scale_2").float()
                )
            w13, s13, _ = ops.nvfp4_sm70_prepare(
                unpack_mxfp4_weight(torch.cat(weights).cuda()),
                torch.cat(scales).half().t().contiguous().cuda(),
                16,
                interleave_gated_silu=interleaved,
            )
            w2, s2, _ = ops.nvfp4_sm70_prepare(
                unpack_mxfp4_weight(
                    tensor(e, "down_proj", "weight")[:, rank * 80 : (rank + 1) * 80]
                    .contiguous()
                    .cuda()
                ),
                (
                    tensor(e, "down_proj", "weight_scale")[
                        :, rank * 10 : (rank + 1) * 10
                    ].float()
                    * tensor(e, "down_proj", "weight_scale_2").float()
                )
                .half()
                .t()
                .contiguous()
                .cuda(),
                16,
            )
            for dest, value in zip(prepared, (w13, s13, w2, s2)):
                dest.append(value)
    return tuple(torch.stack(values) for values in prepared)


def graph(fn, unroll=16):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    result = torch.cuda.CUDAGraph()
    with torch.cuda.graph(result):
        for _ in range(unroll):
            fn()
    return result


def latency(g, repeats, unroll=16):
    for _ in range(3):
        g.replay()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    for _ in range(repeats):
        g.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / (repeats * unroll)


def error(a, b):
    delta = a.float() - b.float()
    return dict(
        exact=torch.equal(a, b),
        max_abs=delta.abs().max().item(),
        relative_l2=(delta.norm() / b.float().norm().clamp_min(1e-12)).item(),
        finite=torch.isfinite(a).all().item(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rank", type=int, choices=range(4), default=0)
    p.add_argument("--routes", type=Path, help="Saved topk_ids tensor or {tensor: ...}")
    p.add_argument("--route-glob", help="Sweep saved routes instead of synthetic cases")
    p.add_argument("--tokens", default="4,8,16")
    p.add_argument("--splits", default="1,2,4,5,8")
    p.add_argument("--interleaved", action="store_true")
    p.add_argument("--packed-w2", action="store_true", help="Reuse grouping in W2")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--samples", type=int, default=7)
    args = p.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("SM70 only")
    if "TORCH_EXTENSIONS_DIR" not in os.environ:
        raise RuntimeError("Set a task-owned TORCH_EXTENSIONS_DIR")
    source = (
        Path(__file__).resolve().parents[2]
        / "csrc/sm70_turbomind/ops/nvfp4_grouped_decode_sm70.cu"
    )
    if not ops.has_nvfp4_grouped_decode_dispatch():
        load(
            name="sm70_moe_packed_w13_screen",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
            is_python_module=False,
            verbose=True,
        )
    torch.manual_seed(20260905)
    if args.model:
        w13, s13, w2, s2 = checkpoint_weights(
            args.model, args.layer, args.rank, args.interleaved
        )
    else:
        w13 = torch.randint(
            -(2**31), 2**31 - 1, (512, 2560, 40), device="cuda", dtype=torch.int32
        )
        w2 = torch.randint(
            -(2**31), 2**31 - 1, (512, 160, 320), device="cuda", dtype=torch.int32
        )
        s13 = torch.rand(512, 160, 320, device="cuda", dtype=torch.float16) * 2**-14
        s2 = torch.rand(512, 10, 2560, device="cuda", dtype=torch.float16) * 2**-14
    captured = {}
    route_paths = (
        sorted(map(Path, glob.glob(args.route_glob))) if args.route_glob else []
    )
    if args.route_glob and not route_paths:
        p.error("--route-glob did not match any files")
    if args.routes:
        route_paths.append(args.routes)
    for path in route_paths:
        value = torch.load(path, weights_only=True, map_location="cpu")
        if isinstance(value, dict):
            value = value["tensor"]
        if value.ndim != 2 or value.shape[1] != 10:
            p.error(f"Expected [M,10] routes: {path}")
        captured[path.name] = value.to(torch.int32)
    results = []
    for m in map(int, args.tokens.split(",")):
        if m not in (4, 8, 16):
            p.error("paired native W13 baseline supports M4/8/16")
        x = torch.randn(m, 2560, device="cuda", dtype=torch.float16) * 0.1
        ids = torch.empty(m, 10, device="cuda", dtype=torch.int32)
        topk = torch.softmax(torch.randn(m, 10, device="cuda"), -1)
        direct_mid = torch.empty(m * 10, 160, device="cuda", dtype=torch.float16)
        packed_mid = torch.empty_like(direct_mid)
        direct_out = torch.empty_like(x)
        packed_out = torch.empty_like(x)
        routed_out = torch.empty(m * 10, 2560, device="cuda", dtype=torch.float16)
        rows = torch.empty(m * 10, 8, device="cuda", dtype=torch.int32)
        experts = torch.empty(m * 10, device="cuda", dtype=torch.int32)
        sizes = torch.empty_like(experts)
        total = torch.empty(1, device="cuda", dtype=torch.int32)

        def direct():
            ops.nvfp4_moe_qpn_w13_swiglu_batch_sm70_out(
                direct_mid, x, w13, s13, ids.view(-1), args.interleaved
            )
            ops.nvfp4_moe_qpn_w2_reduce_sm70_out(
                direct_out, direct_mid, w2, s2, ids.view(-1), topk
            )

        def candidate(split):
            torch.ops._C.nvfp4_grouped_w13_sm70_out(
                packed_mid,
                x,
                w13,
                s13,
                ids.view(-1),
                rows,
                experts,
                sizes,
                total,
                split,
                args.interleaved,
            )
            if args.packed_w2:
                torch.ops._C.nvfp4_grouped_w2_sm70_out(
                    packed_out,
                    routed_out,
                    packed_mid,
                    w2,
                    s2,
                    topk,
                    rows,
                    experts,
                    sizes,
                    total,
                )
            else:
                ops.nvfp4_moe_qpn_w2_reduce_sm70_out(
                    packed_out, packed_mid, w2, s2, ids.view(-1), topk
                )

        cases = {
            "distinct": torch.arange(m * 10).view(m, 10),
            "shared10": torch.arange(10).repeat(m, 1),
        }
        if args.route_glob:
            cases = {}
        if captured:
            cases.update({name: value[:m] for name, value in captured.items()})
        else:
            cases["random"] = torch.stack([torch.randperm(512)[:10] for _ in range(m)])
        for name, case in cases.items():
            if case.shape != ids.shape:
                p.error(f"Insufficient captured rows in {name} for M{m}")
            ids.copy_(case)
            control = graph(direct)
            for split in map(int, args.splits.split(",")):
                packed = graph(lambda: candidate(split))
                # Replay uses new values and poisoned scratch, not capture-time
                # values. All source/output pointers remain fixed.
                checks = []
                replay_ids = (
                    case,
                    case.flip(0),
                    torch.arange(10).repeat(m, 1),
                    torch.arange(m * 10).view(m, 10),
                    torch.full((m, 10), -1),
                )
                for changed_ids in replay_ids:
                    x.normal_(0, 0.1)
                    ids.copy_(changed_ids)
                    for buf in (rows, experts, sizes, total):
                        buf.fill_(-12345)
                    packed_mid.fill_(float("nan"))
                    packed_out.fill_(float("nan"))
                    routed_out.fill_(float("nan"))
                    control.replay()
                    packed.replay()
                    torch.cuda.synchronize()
                    checks.append(
                        dict(
                            mid=error(packed_mid, direct_mid),
                            out=error(packed_out, direct_out),
                        )
                    )
                ids.copy_(case)
                control.replay()
                packed.replay()
                times = [[], []]
                for sample in range(args.samples):
                    for idx in (0, 1) if sample % 2 == 0 else (1, 0):
                        times[idx].append(latency((control, packed)[idx], args.repeats))
                a, b = map(statistics.median, times)
                result = dict(
                    tokens=m,
                    case=name,
                    split=split,
                    unique=case.unique().numel(),
                    control_us=a,
                    candidate_us=b,
                    saving_us=a - b,
                    paired_samples_us=times,
                    checks=checks,
                )
                if not all(
                    c[stage]["finite"] for c in checks for stage in ("mid", "out")
                ):
                    raise AssertionError(f"Non-finite output: M{m}/{name}/split{split}")
                if split == {4: 5, 8: 4, 16: 1}[m] and not all(
                    c[stage]["exact"] for c in checks for stage in ("mid", "out")
                ):
                    raise AssertionError("Packing changed same-split arithmetic")
                results.append(result)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in ("paired_samples_us", "checks")
                        }
                    ),
                    flush=True,
                )
    report = dict(
        model=str(args.model),
        layer=args.layer,
        tp4_rank=args.rank,
        synthetic_activations=True,
        graph_unroll=16,
        interleaved=args.interleaved,
        packed_w2=args.packed_w2,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        route_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in route_paths
        },
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        results=results,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
