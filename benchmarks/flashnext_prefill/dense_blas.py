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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, nargs="+", default=[7344, 5347])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reader = GGUFReader(args.model)
    report = {"artifact": artifact_identity(), "cases": []}
    for name, axis in CASES:
        tensor = next(t for t in reader.tensors if t.name == name)
        raw = np.ascontiguousarray(tensor.data)
        kind = int(tensor.tensor_type)
        canonical = transcode_affine(raw, kind).tp_slice(0, 4, axis=axis)
        n, k = canonical.codes.shape
        dense = dequantize(raw, kind)
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
