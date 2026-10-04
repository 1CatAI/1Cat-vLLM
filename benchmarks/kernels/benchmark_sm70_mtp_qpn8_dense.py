# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen existing channel-QPN8 against current native MTP projections.

Real checkpoint shards, synthetic activations, FP16 computation/FP32 reduction.
HC is excluded. No model default changes or model-quality admission are made.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.kernels.benchmark_sm70_gdn_input_batch import checkpoint_weights
from benchmarks.kernels.benchmark_sm70_mtp_fp32_dense import capture, elapsed
from benchmarks.kernels.benchmark_sm70_qwen38_dense_batch import weights
from vllm import _sm70_ops as sm70_ops
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import _pack_gdn_input_weight


def screen(model, role, rank, rows):
    if role == "head":
        index = json.loads((model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        name = next(n for n in index if n.endswith("lm_head.weight"))
        with safe_open(model / index[name], framework="pt") as f:
            view = f.get_slice(name)
            size = view.get_shape()[0] // 4
            assert view.get_shape()[0] % 4 == 0
            ws = [view[rank * size : (rank + 1) * size].half().cuda().contiguous()]
        names, bas, native = [name], [], []
    elif role == "output":
        names, ws = weights(model, role, rank)
        bas = []
        native = []
    else:
        layers = [i for i in range(48) if i % 4 != 3]
        pairs = checkpoint_weights(model, layers, rank)
        ws, bas = map(list, zip(*pairs))
        names = [f"GDN layer {i}" for i in layers]
        native = [
            (_pack_gdn_input_weight(w), _pack_gdn_input_weight(b)) for w, b in pairs
        ]
    quantized = []
    for w in ws:
        f = w.float()
        s = f.abs().amax(dim=1, keepdim=True) / 448
        s = torch.where(s == 0, torch.ones_like(s), s)
        q = (f / s).to(torch.float8_e4m3fn)
        quantized.append(sm70_ops.fp8_qpn8_prepare_sm70(q.contiguous(), s.contiguous()))
    xs = [w.new_empty((rows, w.shape[1])).normal_(0, 0.1) for w in ws]
    output = [
        [x.new_empty((rows, w.shape[0])) for x, w in zip(xs, ws)] for _ in range(2)
    ]
    separate = (
        [
            [tuple(x.new_empty((rows, n)) for n in (2560, 1536, 12, 12)) for x in xs]
            for _ in range(2)
        ]
        if bas
        else []
    )
    ba_staging = [x.new_empty((rows, 24)) for x in xs] if bas else []

    def control():
        for i, (x, w) in enumerate(zip(xs, ws)):
            if role == "head":
                torch.mm(x, w.t(), out=output[0][i])
            elif role == "output":
                torch.ops._C.qwen38_dense_batch_sm70_out(output[0][i], x, w)
            else:
                pq, pb = native[i]
                torch.ops._C.qwen38_gdn_input_batch_sm70_out(*separate[0][i], x, pq, pb)

    def candidate():
        for i, (x, (codes, scales)) in enumerate(zip(xs, quantized)):
            if not bas:
                sm70_ops.fp8_qpn8_gemm_sm70_out(
                    output[1][i], x, codes, scales, 12, 2, True, False
                )
            else:
                sm70_ops.fp8_qpn8_dispatch_ba_split_sm70_out(
                    *separate[1][i],
                    output[1][i],
                    ba_staging[i],
                    0,
                    x,
                    codes,
                    scales,
                    bas[i],
                )

    graphs = [capture(fn) for fn in (control, candidate)]
    errors = []
    for scale in (0, 0.03, 0.1, 1, 3):
        for x in xs:
            x.normal_(0, scale)
        for graph in graphs:
            graph.replay()
        torch.accelerator.synchronize()
        maximum, relative = [0.0, 0.0], [0.0, 0.0]
        for i, (x, w) in enumerate(zip(xs, ws)):
            ref = x.double() @ w.double().t()
            for arm in (0, 1):
                actual = (
                    output[arm][i] if not bas else torch.cat(separate[arm][i][:2], -1)
                )
                assert torch.isfinite(actual).all()
                delta = actual.double() - ref
                maximum[arm] = max(maximum[arm], delta.abs().max().item())
                relative[arm] = max(
                    relative[arm], (delta.norm() / ref.norm().clamp_min(1e-30)).item()
                )
            if bas:
                for arm in (0, 1):
                    assert all(torch.isfinite(a).all() for a in separate[arm][i][2:])
        errors.append(
            dict(scale=scale, max_abs_vs_fp64=maximum, max_relative_l2=relative)
        )
    samples = [[], []]
    for trial in range(7):
        for arm in (0, 1) if trial % 2 == 0 else (1, 0):
            samples[arm].append(elapsed(graphs[arm]))
    medians = [statistics.median(s) for s in samples]
    return dict(
        role=role,
        rows=rows,
        layers=names,
        samples_ms=samples,
        medians_ms=medians,
        saving_ms=medians[0] - medians[1],
        operator_errors=errors,
        fp16_weight_bytes=sum(w.numel() * w.element_size() for w in ws),
        qpn8_weight_and_scale_bytes=sum(
            t.numel() * t.element_size() for pair in quantized for t in pair
        ),
        ba_weight_bytes=sum(b.numel() * b.element_size() for b in bas),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(4), default=0)
    parser.add_argument("--rows", type=int, choices=range(1, 9), default=5)
    parser.add_argument("--role", choices=("gdn", "output", "head"), action="append")
    args = parser.parse_args()
    if args.rows == 1 and args.role != ["head"]:
        parser.error("M1 qualification is restricted to the draft head")
    torch.set_num_threads(1)
    torch.manual_seed(20261005 + args.rank)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    assert torch.cuda.get_device_capability() == (7, 0)
    native = Path(__file__).resolve().parents[2] / "vllm/_C.abi3.so"
    report = dict(
        complete=False,
        model_admission=False,
        results=[],
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for role in args.role or ("gdn", "output"):
        result = screen(args.model, role, args.rank, args.rows)
        report["results"].append(result)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("samples_ms", "layers", "operator_errors")
                }
            ),
            flush=True,
        )
    report["complete"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
