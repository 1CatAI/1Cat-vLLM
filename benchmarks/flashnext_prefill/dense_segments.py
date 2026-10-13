# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure prefill after the serving segment-bank restoration, including DQ.

Both arms keep small-M segment storage. The control omits only the admitted
BLAS bands, reproducing the missed serving route without another checkpoint.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from dense_blas import weight_cases
from moe_tiles import artifact_identity, measure, relative_l2

from tools.sm70.native_trace import NativeDispatchTrace
from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    _prepared_gguf_mixed_projection,
    prepare_gguf_projections,
    prepared_projection_arguments,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--merged", action="store_true")
    parser.add_argument("--tokens", type=int, nargs="+", default=[7344, 5347])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = SimpleNamespace(
        kernel_config=KernelConfig(),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    report = {"artifact": artifact_identity(), "cases": []}
    reader = GGUFReader(args.model)
    with set_current_vllm_config(config):
        cases = list(weight_cases(reader, args.merged))
        if not args.merged:
            cases = cases[:4]
        for name, source, raw, axis in cases:
            if axis >= 0:
                raw = (
                    raw[: raw.shape[0] // 4]
                    if axis == 0
                    else raw[:, : raw.shape[1] // 4]
                )
            raw = np.ascontiguousarray(raw)
            reference_weight = torch.from_numpy(dequantize(raw, source).copy()).cuda()
            projection = prepare_gguf_projections(
                [(torch.from_numpy(raw).cuda(), source)], torch.float16, True, 8
            )[0]
            assert hasattr(projection, "segment_format")
            actual = prepared_projection_arguments([projection])
            assert actual[3][0] == 3 and actual[5] == [2048, -1]
            control = list(actual)
            control[3] = actual[3][:-1] + [0]
            control[5] = []
            k, n = projection.kernel.config.partition_weight_shape
            for m in args.tokens:
                torch.manual_seed(9683 + m)
                x = (torch.randn(m, k, device="cuda") * 0.125).half()

                def fused(x=x, control=control):
                    return torch.ops.vllm.prepared_gguf_mixed_projection(x, *control)

                def blas(x=x, actual=actual):
                    return torch.ops.vllm.prepared_gguf_mixed_projection(x, *actual)

                old = fused()
                with NativeDispatchTrace() as trace:
                    observed = _prepared_gguf_mixed_projection(x, *actual)
                new = blas()
                torch.testing.assert_close(new, observed, rtol=0, atol=0)
                assert any(
                    r["operator"] == "_C::gguf_affine_blas_sm70_out"
                    and r["evidence"] == "cuda_dispatch"
                    for r in trace.records
                )
                reference = x[:16].float() @ reference_weight.float().T
                errors = {
                    "control_fp32_max_rel_l2": relative_l2(old[:16], reference),
                    "candidate_fp32_max_rel_l2": relative_l2(new[:16], reference),
                    "paired_max_rel_l2": relative_l2(new, old),
                }
                row = {
                    "tensor": name,
                    "source_type": source,
                    "shape_mnk": [m, n, k],
                    "restore_fused": measure(fused, 5),
                    "restore_blas": measure(blas, 5),
                    "restore_fused_repeat": measure(fused, 5),
                    **errors,
                    "passed": bool(torch.isfinite(new).all())
                    and max(errors.values()) <= 0.003,
                }
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if all(r["passed"] for r in report["cases"]) else 1)


if __name__ == "__main__":
    main()
