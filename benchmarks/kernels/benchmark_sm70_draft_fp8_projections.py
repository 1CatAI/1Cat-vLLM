# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen draft FP8 projections without changing a model or its LM head.

Projection errors are diagnostic only. Enabling a numerical change additionally
requires full-model teacher-forcing KL and acceptance confidence intervals.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.quantization.sm70_dflash2_fp16 import (
    apply_dflash2_fp16_m8,
    prepare_dflash2_fp16_m8,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.rank)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261005)
    evict = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    rows = []
    with safe_open(args.draft / "model.safetensors", framework="pt") as f:
        for index in args.layers:

            def get(name, index=index):
                return f.get_tensor(f"layers.{index}.{name}.weight").half().cuda()

            rank = args.rank
            weights = {
                "qkv": torch.cat(
                    [
                        get("self_attn.q_proj")[rank * 1024 : (rank + 1) * 1024],
                        get("self_attn.k_proj")[rank * 256 : (rank + 1) * 256],
                        get("self_attn.v_proj")[rank * 256 : (rank + 1) * 256],
                    ]
                ),
                "o": get("self_attn.o_proj")[
                    :, rank * 1024 : (rank + 1) * 1024
                ].contiguous(),
                "gate_up": torch.cat(
                    [
                        get("mlp.gate_proj")[rank * 4352 : (rank + 1) * 4352],
                        get("mlp.up_proj")[rank * 4352 : (rank + 1) * 4352],
                    ]
                ),
                "down": get("mlp.down_proj")[
                    :, rank * 4352 : (rank + 1) * 4352
                ].contiguous(),
            }
            for name, weight in weights.items():
                n, k = weight.shape
                module = torch.nn.Module()
                module.weight = torch.nn.Parameter(weight, requires_grad=False)
                module._sm70_dflash2_fp16_m8 = True
                if not prepare_dflash2_fp16_m8(module):
                    raise RuntimeError((name, n, k))
                # Quantize the existing FP16 projection operand, not the
                # checkpoint's trained BF16 norms or the original FP8 head.
                scale = weight.float().abs().amax(1).clamp_min(1e-12) / 448.0
                quantized = (weight.float() / scale[:, None]).to(torch.float8_e4m3fn)
                codes, qscale = ops.fp8_qpn8_prepare_sm70(quantized, scale)
                x = torch.randn(8, k, device="cuda", dtype=torch.float16) * 0.1
                output = x.new_empty((8, n))
                split, chains = (8, 2) if name == "gate_up" else (16, 2)

                def candidate(
                    output=output,
                    x=x,
                    codes=codes,
                    qscale=qscale,
                    split=split,
                    chains=chains,
                ):
                    ops.fp8_qpn8_gemm_sm70_out(
                        output, x, codes, qscale, split, chains, True, False
                    )
                    return output

                def baseline(module=module, x=x):
                    return apply_dflash2_fp16_m8(module, x, None)

                reference = x.float() @ weight.float().T
                old = baseline().clone()
                new = candidate().clone()
                torch.cuda.synchronize()
                errors = {
                    "fp16_max_abs_vs_dense_fp32": float((old - reference).abs().max()),
                    "fp8_max_abs_vs_dense_fp32": float((new - reference).abs().max()),
                    "fp8_rmse_vs_dense_fp32": float(
                        (new - reference).square().mean().sqrt()
                    ),
                    "finite": bool(torch.isfinite(new).all()),
                }
                if not errors["finite"]:
                    raise AssertionError((index, name, errors))
                stream = torch.cuda.Stream()
                graphs = []
                for label, fn in [("fp16", baseline), ("fp8", candidate)]:
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            fn()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    start = torch.cuda.Event(enable_timing=True, external=True)
                    end = torch.cuda.Event(enable_timing=True, external=True)
                    with torch.cuda.graph(graph, stream=stream):
                        evict.fill_(1)
                        start.record()
                        fn()
                        end.record()
                    graphs.append((label, graph, start, end))
                samples = {"fp16": [], "fp8": []}
                for repeat in range(45):
                    for i in [repeat % 2, 1 - repeat % 2]:
                        label, graph, start, end = graphs[i]
                        for _ in range(3):
                            graph.replay()
                        end.synchronize()
                        if repeat >= 5:
                            samples[label].append(start.elapsed_time(end) * 1000)
                for _, graph, _, _ in graphs:
                    graph.reset()
                row = {
                    "layer": index,
                    "projection": name,
                    "shape": [n, k],
                    "diagnostic_errors": errors,
                    "mean_us": {key: statistics.mean(v) for key, v in samples.items()},
                    "samples_us": samples,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "research_only": True,
                "precision_change": True,
                "full_model_quality_gate_pending": True,
                "rank": args.rank,
                "cold_l2_bytes": evict.numel(),
                "records": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
