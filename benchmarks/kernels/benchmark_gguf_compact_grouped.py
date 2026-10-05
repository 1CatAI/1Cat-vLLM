# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare equal-byte IQ expert packets with canonical grouped full graphs.

Routing uses top-k distinct experts per input token. Sorted rows, original
weights and canonical controls are prepared before timing. Routing sorting and
the rest of the MoE FFN are excluded. Hold the shared GPU lock when running.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import vllm._C as core
from benchmark_gguf_turbomind import canonical_grouped_call, elapsed, prepare_projection
from benchmark_gguf_turbomind_grouped import stack_prepared

from vllm.model_executor.kernels.gguf import compact_lattice_grouped_capabilities
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-axis", type=int, choices=(0, 1), default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 5, 8, 16, 512])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.top_k <= args.experts
        or args.iterations < 1
        or any(m < 1 for m in args.m)
    ):
        parser.error("Expected positive counts and top-k no larger than experts")
    return args


def main():
    args = parse_args()
    spec = {"file": args.gguf, "tensor": args.tensor}
    report = {
        "source_core_sha256": hashlib.sha256(
            Path(core.__file__).read_bytes()
        ).hexdigest(),
        "checkpoint": Path(args.gguf).name,
        "tensor": args.tensor,
        "graph": "FULL",
        "tp": args.tp_size,
        "top_k": args.top_k,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "scope": "sorted expert projection; sorting and full FFN excluded",
        "results": [],
    }

    def save():
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    reader = GGUFReader(spec["file"])
    tensor = next(t for t in reader.tensors if t.name == spec["tensor"])
    experts = args.experts
    kind = int(tensor.tensor_type)
    if (
        kind not in (18, 21, 22)
        or tensor.data.ndim != 3
        or not 1 <= experts <= tensor.data.shape[0]
    ):
        raise ValueError("Expected stacked IQ3_XXS/IQ3_S/IQ2_S experts")
    first = RawGGUFProjection.from_rows(tensor.data[0], kind).tp_slice(
        args.tp_rank, args.tp_size, axis=args.tp_axis
    )
    n, k = first.shape
    if n % 32:
        raise ValueError("Grouped packets require N divisible by 32")
    compact_bytes = n * (k // 256) * ({18: 98, 21: 110, 22: 82}[kind])
    banks_count = max(
        1,
        int(
            np.ceil(
                2
                * getattr(
                    torch.cuda.get_device_properties(0),
                    "L2_cache_size",
                    6 * 1024 * 1024,
                )
                / (min(experts, args.top_k) * compact_bytes)
            )
        ),
    )
    report["weight_banks"] = banks_count
    compact = torch.empty((experts, compact_bytes), device="cuda", dtype=torch.uint8)
    prepared = []
    reference = []
    projections = []
    for expert in range(experts):
        raw = RawGGUFProjection.from_rows(tensor.data[expert], kind).tp_slice(
            args.tp_rank, args.tp_size, axis=args.tp_axis
        )
        payload = raw.data[:, : raw.payload_bytes_per_row]
        projection = transcode_lattice(payload, kind)
        prepared.append(prepare_projection(projection))
        projections.append(projection)
        source = torch.from_numpy(raw.data).cuda()
        torch.ops._C.gguf_lattice_compact_reorder_sm70_out(
            compact[expert], source, kind, k
        )
        reference.append(torch.from_numpy(dequantize(payload, kind)).half().cuda())
        if expert % 128 == 0:
            print("prepared", kind, expert, flush=True)
    w, stats, ptrs = stack_prepared(prepared)
    projection = projections[0]
    del prepared, projections
    banks = [compact] + [compact.clone() for _ in range(banks_count - 1)]
    control_storage = [(w, stats, ptrs)]
    meta = prepare_projection(projection)[2].tolist()
    for _ in range(banks_count - 1):
        cw = w.clone()
        cs = stats.clone()
        cp = torch.ops._C.awq_moe_build_strided_ptrs(cw, cs, *meta, experts)
        control_storage.append((cw, cs, cp))
    cap = compact_lattice_grouped_capabilities(kind, k, n, experts, torch.float16)[0]
    assert cap.reason is None, cap
    for tokens in args.m:
        rng = np.random.default_rng(20261005 + tokens)
        selected = np.stack(
            [rng.choice(experts, args.top_k, replace=False) for _ in range(tokens)]
        )
        counts = np.bincount(selected.ravel(), minlength=experts)
        boundary = np.r_[0, counts.cumsum()].tolist()
        total = int(counts.sum())
        offsets = torch.tensor(boundary, device="cuda", dtype=torch.int32)
        prefix = torch.empty_like(offsets)
        torch.manual_seed(20261005 + tokens)
        x = torch.randn((total, k), device="cuda", dtype=torch.float16)
        out = torch.empty((total, n), device="cuda", dtype=torch.float16)
        expected = torch.empty_like(out, dtype=torch.float32)
        for expert in range(experts):
            begin, end = boundary[expert : expert + 2]
            if end > begin:
                expected[begin:end] = x[begin:end].float() @ reference[expert].float().T

        def raw_call(out=out, x=x, offsets=offsets, prefix=prefix):
            for bank in banks:
                torch.ops._C.gguf_lattice_compact_grouped_sm70_out(
                    out, x, bank, offsets, prefix, kind
                )
            return out

        def canonical_call(out=out, x=x, offsets=offsets):
            for cw, cs, (wp, sp) in control_storage:
                canonical_grouped_call(projection, out, x, offsets, wp, sp, experts)()
            return out

        raw_call()
        error = float((out.float() - expected).norm() / expected.norm())
        assert bool(torch.isfinite(out).all()) and error < 0.003, error
        raw_us = elapsed(raw_call, args.iterations, capture=True) / banks_count
        canonical_call()
        canon_error = float((out.float() - expected).norm() / expected.norm())
        assert bool(torch.isfinite(out).all()) and canon_error < 0.003, canon_error
        canon_us = elapsed(canonical_call, args.iterations, capture=True) / banks_count
        row = dict(
            source_type=kind,
            tensor=spec["tensor"],
            experts=experts,
            n=n,
            k=k,
            tokens=tokens,
            routed_rows=total,
            active_experts=int((counts > 0).sum()),
            max_rows_per_expert=int(counts.max()),
            raw_us=raw_us,
            canonical_us=canon_us,
            raw_relative_l2=error,
            canonical_relative_l2=canon_error,
            persistent_raw_bytes=int(compact.numel()),
            active_raw_bytes=int((counts > 0).sum()) * compact_bytes,
            capability=cap.operator,
        )
        report["results"].append(row)
        save()
        print(row, flush=True)


if __name__ == "__main__":
    main()
