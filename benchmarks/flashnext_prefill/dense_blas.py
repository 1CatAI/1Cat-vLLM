# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare existing fused and dequantize+cuBLAS dense providers on real TP4 weights.

BLAS timing includes dequantization into its scratch on every call. This is a
provider screen, not a model qualification or a persistent FP16 weight cache.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from moe_tiles import artifact_identity, measure, relative_l2

from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize

CASES = [
    ("blk.0.attn_qkv.weight", 0),
    ("blk.1.attn_qkv.weight", 0),
    ("blk.0.ssm_out.weight", 1),
    ("blk.1.attn_gate.weight", 0),
    ("blk.5.ffn_up_shexp.weight", 0),
]


def weight_cases(reader, merged, remaining_affine=False):
    tensors = {t.name: t for t in reader.tensors}
    if remaining_affine:
        for name, axis in [
            ("blk.0.attn_gate.weight", 0),
            ("blk.5.ssm_out.weight", 1),
            ("blk.11.attn_q.weight", 0),
            ("blk.14.attn_gate.weight", 0),
            ("blk.18.ssm_out.weight", 1),
        ]:
            tensor = tensors[name]
            yield name, int(tensor.tensor_type), np.ascontiguousarray(tensor.data), axis
        arrays = [
            tensors["blk.13.attn_qkv.weight"].data[first:last]
            for first, last in [(0, 512), (2048, 2560), (4096, 5632)]
        ]
        arrays.append(tensors["blk.13.attn_gate.weight"].data[:1536])
        yield "gdn13_qkv_gate", 12, np.ascontiguousarray(np.concatenate(arrays)), -1
        return
    if not merged:
        for name, axis in CASES:
            tensor = tensors[name]
            yield name, int(tensor.tensor_type), np.ascontiguousarray(tensor.data), axis
        return
    # Physical rank-zero layouts observed after the serving merger. GDN
    # splits Q/K/V independently; QSA's two KV heads are replicated over TP4.
    cases = [
        (
            "gdn4_qkv_gate",
            [
                (4, "attn_qkv", (0, 512)),
                (4, "attn_qkv", (2048, 2560)),
                (4, "attn_qkv", (4096, 5632)),
                (4, "attn_gate", (0, 1536)),
            ],
        ),
        (
            "gdn38_qkv_gate",
            [
                (38, "attn_qkv", (0, 512)),
                (38, "attn_qkv", (2048, 2560)),
                (38, "attn_qkv", (4096, 5632)),
                (38, "attn_gate", (0, 1536)),
            ],
        ),
        (
            "qsa31_qkv",
            [
                (31, "attn_q", (0, 3072)),
                (31, "attn_k", (0, 256)),
                (31, "attn_v", (0, 256)),
            ],
        ),
        ("qsa19_qv", [(19, "attn_q", (0, 3072)), (19, "attn_v", (0, 256))]),
        ("qsa23_q", [(23, "attn_q", (0, 3072))]),
    ]
    for name, parts in cases:
        arrays, kinds = [], set()
        for layer, part, (first, last) in parts:
            tensor = tensors[f"blk.{layer}.{part}.weight"]
            arrays.append(np.asarray(tensor.data)[first:last])
            kinds.add(int(tensor.tensor_type))
        assert len(kinds) == 1, "only same-codec projections may be merged"
        yield name, kinds.pop(), np.ascontiguousarray(np.concatenate(arrays)), -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, nargs="+", default=[7344, 5347])
    parser.add_argument("--iterations", type=int, default=5)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--merged", action="store_true")
    group.add_argument("--remaining-affine", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reader = GGUFReader(args.model)
    report = {"artifact": artifact_identity(), "cases": []}
    for name, kind, raw, axis in weight_cases(
        reader, args.merged, args.remaining_affine
    ):
        canonical = transcode_affine(raw, kind)
        if axis >= 0:
            canonical = canonical.tp_slice(0, 4, axis=axis)
        n, k = canonical.codes.shape
        dense = dequantize(raw, kind)
        if axis >= 0:
            dense = dense[:n] if axis == 0 else dense[:, :k]
        reference_weight = torch.from_numpy(np.ascontiguousarray(dense)).cuda()
        w, s, meta = torch.ops._C.gguf_affine_sm70_prepare(
            torch.from_numpy(canonical.codes).cuda(),
            torch.from_numpy(canonical.scales).cuda(),
            torch.from_numpy(canonical.mins).cuda(),
            canonical.bits,
            canonical.group_size,
        )
        k_ld, s_ld = meta.tolist()
        scratch = torch.empty(k, n, dtype=torch.half, device="cuda")
        for m in args.tokens:
            torch.manual_seed(9683 + m)
            x = (torch.randn(m, k, device="cuda") * 0.125).half()
            fused_out = torch.empty(m, n, dtype=torch.half, device="cuda")
            blas_out = torch.empty_like(fused_out)

            fused = partial(
                torch.ops._C.gguf_affine_gemm_sm70_out,
                fused_out,
                x,
                w,
                s,
                canonical.bits,
                k_ld,
                s_ld,
                canonical.group_size,
            )
            blas = partial(
                torch.ops._C.gguf_affine_blas_sm70_out,
                blas_out,
                x,
                w,
                s,
                canonical.bits,
                scratch,
                canonical.group_size,
            )

            fused()
            blas()
            expected = x[:16].float() @ reference_weight.float().T
            errors = {
                "fused_fp32_max_rel_l2": relative_l2(fused_out[:16], expected),
                "blas_fp32_max_rel_l2": relative_l2(blas_out[:16], expected),
                "paired_max_rel_l2": relative_l2(blas_out, fused_out),
            }
            passed = (
                bool(torch.isfinite(fused_out).all() and torch.isfinite(blas_out).all())
                and max(errors.values()) <= 0.003
            )
            row = {
                "tensor": name,
                "source_type": kind,
                "tp_axis": axis,
                "shape_mnk": [m, n, k],
                "scratch_bytes": scratch.numel() * scratch.element_size(),
                "fused": measure(fused, args.iterations),
                "dequant_blas": measure(blas, args.iterations),
                "fused_repeat": measure(fused, args.iterations),
                **errors,
                "passed": passed,
            }
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if all(row["passed"] for row in report["cases"]) else 1)


if __name__ == "__main__":
    main()
