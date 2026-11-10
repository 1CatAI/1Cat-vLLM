# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint M5 router/shared projection screen under FP32 reduction policy.

Times are operator sequences over all 48 layers, not complete model rounds.
No bitwise-equality admission: record operator errors against FP64 and defer
model-distribution/quality admission to the shared teacher-forcing tools.
"""

import argparse
import hashlib
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops


def load_weights(model, role, rank):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    suffix = (
        ".mlp.gate.weight"
        if role == "router"
        else ".mlp.shared_expert.gate_proj.weight"
    )
    names = sorted(n for n in index if "mtp" not in n and n.endswith(suffix))
    assert len(names) == 48
    result = []
    with ExitStack() as stack:
        handles = {}
        for name in names:
            file = index[name]
            if file not in handles:
                handles[file] = stack.enter_context(
                    safe_open(model / file, framework="pt")
                )
            w = handles[file].get_tensor(name)
            if role == "shared":
                up_name = name.replace(".gate_proj.weight", ".up_proj.weight")
                up_file = index[up_name]
                if up_file not in handles:
                    handles[up_file] = stack.enter_context(
                        safe_open(model / up_file, framework="pt")
                    )
                up = handles[up_file].get_tensor(up_name)
                assert tuple(w.shape) == tuple(up.shape) == (640, 2560)
                lo, hi = rank * 160, (rank + 1) * 160
                w = torch.cat((w[lo:hi], up[lo:hi]))
            result.append(w.to(device="cuda", dtype=torch.float16).contiguous())
    return names, result


def capture(fn):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def elapsed(graph):
    a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    a.record()
    for _ in range(100):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 100


def screen(model, role, rank):
    names, ws = load_weights(model, role, rank)
    xs = [torch.randn(5, 2560, device="cuda", dtype=torch.float16) for _ in ws]
    packed = [
        w.reshape(64, 8, 4, 40, 2, 8).permute(0, 3, 4, 2, 1, 5).contiguous()
        if role == "router"
        else w.reshape(10, 32, 160, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
        for w in ws
    ]
    n = 512 if role == "router" else 160
    outputs = [[x.new_empty((5, n)) for x in xs] for _ in range(2)]
    temporary = [x.new_empty((5, 320)) for x in xs] if role == "shared" else []
    partial = (
        [x.new_empty((8, 5, 320), dtype=torch.float32) for x in xs]
        if role == "shared"
        else []
    )

    def control():
        for i, (x, w) in enumerate(zip(xs, ws)):
            if role == "router":
                torch.mm(x, w.t(), out=outputs[0][i])
            else:
                torch.mm(x, w.t(), out=temporary[i])
                torch.ops._C.silu_and_mul(outputs[0][i], temporary[i])

    def candidate():
        for i, (x, w) in enumerate(zip(xs, packed)):
            if role == "router":
                torch.ops._C.qwen38_router_batch_sm70_out(outputs[1][i], x, w)
            else:
                torch.ops._C.qwen38_shared_up_batch_sm70_out(
                    outputs[1][i], partial[i], x, w
                )

    graphs = [capture(f) for f in (control, candidate)]
    errors = []
    for scale in (0.0, 0.03, 0.1, 1.0, 3.0):
        for x in xs:
            x.normal_(0, scale)
        for graph in graphs:
            graph.replay()
        torch.accelerator.synchronize()
        max_abs = [0.0, 0.0]
        max_l2 = [0.0, 0.0]
        for i, (x, w) in enumerate(zip(xs, ws)):
            ref = x.double() @ w.double().t()
            if role == "shared":
                gate, up = ref.half().chunk(2, dim=-1)
                ref = (
                    torch.nn.functional.silu(gate.double()).half().double()
                    * up.double()
                )
            for arm in (0, 1):
                actual = outputs[arm][i].double()
                assert torch.isfinite(actual).all()
                delta = actual - ref
                max_abs[arm] = max(max_abs[arm], delta.abs().max().item())
                max_l2[arm] = max(
                    max_l2[arm], (delta.norm() / ref.norm().clamp_min(1e-30)).item()
                )
        errors.append(
            dict(scale=scale, max_abs_vs_fp64=max_abs, max_relative_l2_vs_fp64=max_l2)
        )
    samples = [[], []]
    for trial in range(7):
        for arm in (0, 1) if trial % 2 == 0 else (1, 0):
            samples[arm].append(elapsed(graphs[arm]))
    medians = [statistics.median(s) for s in samples]
    record = dict(
        role=role,
        m=5,
        layers=names,
        samples_ms=samples,
        medians_ms=medians,
        saving_ms=medians[0] - medians[1],
        logical_weight_bytes=sum(w.numel() * w.element_size() for w in ws),
        operator_errors=errors,
    )
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rank", type=int, choices=range(4), default=0)
    p.add_argument("--role", choices=("router", "shared"), action="append")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005 + args.rank)
    matmul = torch.backends.cuda.matmul
    matmul.allow_fp16_reduced_precision_reduction = False
    matmul.allow_bf16_reduced_precision_reduction = False
    matmul.allow_fp16_accumulation = False
    assert torch.cuda.get_device_capability() == (7, 0)
    report = dict(
        complete=False,
        model_admission=False,
        rank=args.rank,
        results=[],
        native_sha256=hashlib.sha256(
            (Path(ops.__file__).parent / "_C.abi3.so").read_bytes()
        ).hexdigest(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for role in args.role or ("router", "shared"):
        record = screen(args.model, role, args.rank)
        report["results"].append(record)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k not in ("layers", "samples_ms", "operator_errors")
                }
            ),
            flush=True,
        )
    report["complete"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
