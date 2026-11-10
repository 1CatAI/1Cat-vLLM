# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Q8_0 draft MoE chunk sizes at the real TP4 shard geometry.

Synthetic weights/routing isolate memory and chunk overhead, not checkpoint
quality or whole-model speed. Paired maximum row L2 must stay <=0.001.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.models.qwen4_exp.nvidia import mtp_q8_experts as moe


def packed(shape):
    values = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    blocks = values.view(-1, 34)
    scale = torch.tensor([0.0002], dtype=torch.half, device="cuda").view(torch.uint8)
    blocks[:, :2].copy_(scale)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, default=[4096, 7344, 8192])
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(9683)
    layer = NS(
        expert_map=None,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        w13_weight=packed((512, 320, 80 * 34)),
        w2_weight=packed((512, 2560, 5 * 34)),
    )
    method = object.__new__(moe.MTPQ80MoEMethod)
    original = moe._PREFILL_CHUNK
    report = {"scope": "synthetic TP4 Q8_0 draft shard, eager CUDA events", "cases": []}
    try:
        for rows in args.tokens:
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.half) * 0.2
            ids = torch.rand(rows, 512, device="cuda").topk(10, dim=-1).indices.int()
            weights = torch.rand(rows, 10, device="cuda").softmax(-1)

            def run(x=x, weights=weights, ids=ids):
                return method.apply(layer, x, weights, ids, None, None)

            reference = None
            case = {"rows": rows, "chunks": {}}
            for chunk in (2048, 1024):
                moe._PREFILL_CHUNK = chunk
                for _ in range(2):
                    run()
                torch.accelerator.synchronize()
                base = torch.accelerator.memory_allocated()
                torch.accelerator.reset_peak_memory_stats()
                result = run()
                torch.accelerator.synchronize()
                peak = (torch.accelerator.max_memory_allocated() - base) / 2**20
                assert torch.isfinite(result).all()
                error = 0.0
                if reference is not None:
                    error = float(
                        (
                            (result.float() - reference.float()).norm(dim=-1)
                            / reference.float().norm(dim=-1).clamp_min(1e-12)
                        ).max()
                    )
                    assert error <= 0.001, error
                else:
                    reference = result.clone()
                times = []
                for _ in range(5):
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    run()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end))
                case["chunks"][str(chunk)] = dict(
                    peak_delta_mib=peak, cuda_ms=times, max_row_l2=error
                )
                del result
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
    finally:
        moe._PREFILL_CHUNK = original
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
