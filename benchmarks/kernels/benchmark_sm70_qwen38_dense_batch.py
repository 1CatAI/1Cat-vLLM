# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native router/output projection screen on every checkpoint layer.

No runtime dispatch changes, weight packing or model startup. The numerical
oracle is the unchanged high-precision Torch linear, not the C1 GEMV tree.
"""

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from statistics import median

import torch
from safetensors import safe_open

from vllm import _custom_ops as _ops  # noqa: F401


def weights(model, role, rank):
    mapping = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    suffixes = (
        (".mlp.gate.weight",)
        if role == "router"
        else (".linear_attn.out_proj.weight", ".self_attn.o_proj.weight")
    )
    names = sorted(n for n in mapping if "mtp" not in n and n.endswith(suffixes))
    assert len(names) == 48, (role, len(names))
    result = []
    with ExitStack() as stack:
        handles = {}
        for name in names:
            file = mapping[name]
            if file not in handles:
                handles[file] = stack.enter_context(
                    safe_open(model / file, framework="pt")
                )
            w = handles[file].get_tensor(name)
            if role == "output":
                assert w.shape == (2560, 6144), (name, w.shape)
                w = w[:, rank * 1536 : (rank + 1) * 1536]
            result.append(w.to(device="cuda", dtype=torch.float16).contiguous())
    return names, result


def capture(xs, ws, outs, fused):
    def run():
        for x, w, out in zip(xs, ws, outs):
            if fused:
                torch.ops._C.qwen38_dense_batch_sm70_out(out, x, w)
            else:
                torch.mm(x, w.t(), out=out)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


def elapsed(graph):
    for _ in range(10):
        graph.replay()
    a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    a.record()
    for _ in range(100):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(4), default=0)
    parser.add_argument("--rows", default="2,4,8,16")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20260927 + args.rank)
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    ):
        setattr(torch.backends.cuda.matmul, name, False)
    assert torch.cuda.get_device_capability() == (7, 0)
    root = Path(__file__).resolve().parents[2]
    native = root / "vllm/_C.abi3.so"
    report = dict(
        complete=False,
        endpoint=False,
        rank=args.rank,
        results=[],
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
    )
    for role in ("router", "output"):
        names, ws = weights(args.model, role, args.rank)
        n, k = ws[0].shape
        for m in map(int, args.rows.split(",")):
            xs = [torch.randn(m, k, device="cuda", dtype=torch.float16) for _ in ws]
            expected = [
                torch.empty(m, n, device="cuda", dtype=torch.float16) for _ in ws
            ]
            actual = [torch.empty_like(out) for out in expected]
            control = capture(xs, ws, expected, False)
            candidate = capture(xs, ws, actual, True)
            errors = []
            for scale in (0, 0.001, 0.03, 0.1, 1, 3):
                for x, out in zip(xs, actual):
                    x.normal_(0, scale)
                    out.fill_(float("nan"))
                control.replay()
                candidate.replay()
                errors.append(
                    [
                        int((a.view(torch.int16) != e.view(torch.int16)).sum())
                        for a, e in zip(actual, expected)
                    ]
                )
            exact = all(not any(v) for v in errors)
            samples = [[], []]
            if exact:
                for repeat in range(7):
                    for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                        samples[arm].append(elapsed((control, candidate)[arm]))
            record = dict(
                role=role,
                m=m,
                n=n,
                k=k,
                layers=names,
                exact=exact,
                mismatches=errors,
                samples_ms=samples,
                medians_ms=[median(v) for v in samples] if exact else None,
            )
            report["results"].append(record)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in ("layers", "samples_ms", "mismatches")
                    }
                ),
                flush=True,
            )
            del control, candidate, xs, expected, actual
        del ws
    report["complete"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    assert all(r["exact"] for r in report["results"])


if __name__ == "__main__":
    main()
