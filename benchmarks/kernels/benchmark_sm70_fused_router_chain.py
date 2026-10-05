# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint M5 router projection/top10/plan, all 48 layers, CUDA graphs.

Native row-major FP16 producer CTAs reuse weights across tokens and the last
producer performs selection and expert grouping. Never model-round admission.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_sm70_mtp_fp32_dense import (
    capture,
    elapsed,
    load_weights,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    _sm70_qwen38_router_topk,
)
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import _pack_router_batch_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--token-cta", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.manual_seed(20261005)
    _, weights = load_weights(args.model, "router", 0)
    packed = [_pack_router_batch_weight(w) for w in weights]
    xs = torch.randn(48, 5, 2560, device="cuda", dtype=torch.float16)

    def workspace():
        return (
            xs.new_empty(5, 512),
            torch.empty(5, 10, device="cuda", dtype=torch.float32),
            *(torch.empty(5, 10, device="cuda", dtype=torch.int32) for _ in range(2)),
            torch.empty(50, 8, device="cuda", dtype=torch.int32),
            *(torch.empty(50, device="cuda", dtype=torch.int32) for _ in range(2)),
            torch.empty(1, device="cuda", dtype=torch.int32),
            torch.zeros(1, device="cuda", dtype=torch.int32),
        )

    work = {name: [workspace() for _ in weights] for name in ("control", "fused")}

    def control():
        for i, p in enumerate(packed):
            buffers = work["control"][i]
            logits, probabilities, ids, source_rows = buffers[:4]
            torch.ops._C.qwen38_router_batch_sm70_out(logits, xs[i], p)
            _sm70_qwen38_router_topk(probabilities, ids, source_rows, logits)
            torch.ops._C.qwen38_router_fused_sm70_out(*buffers, xs[i], weights[i], True)

    def fused():
        for i, w in enumerate(weights):
            if args.token_cta:
                buffers = work["fused"][i]
                torch.ops._C.qwen38_router_token_cta_sm70_out(
                    *buffers[:4], xs[i], packed[i]
                )
                torch.ops._C.qwen38_router_fused_sm70_out(*buffers, xs[i], w, True)
            else:
                torch.ops._C.qwen38_router_fused_sm70_out(*work["fused"][i], xs[i], w)

    graphs = {"control": capture(control), "fused": capture(fused)}
    checks = []
    for scale in (0.0, 0.03, 1.0, 3.0):
        xs.normal_(0, scale)
        for graph in graphs.values():
            for _ in range(3):
                graph.replay()
        torch.cuda.synchronize()
        max_error, disagreements = 0.0, 0
        for i, (ref, candidate) in enumerate(zip(work["control"], work["fused"])):
            assert torch.isfinite(candidate[0]).all()
            assert torch.isfinite(candidate[1]).all()
            assert candidate[-1].item() == 0
            max_error = max(max_error, (candidate[0] - ref[0]).abs().max().item())
            disagreements += (candidate[2] != ref[2]).sum().item()
            torch.testing.assert_close(candidate[3], ref[3], rtol=0, atol=0)
            ids, route_rows, experts, sizes = (
                candidate[2].flatten().cpu(),
                candidate[4].cpu(),
                candidate[5].cpu(),
                candidate[6].cpu(),
            )
            total = candidate[7].item()
            assert 1 <= total <= 50
            seen = []
            for g in range(total):
                n = sizes[g].item()
                assert 1 <= n <= 8
                routed = route_rows[g, :n]
                assert torch.all(ids[routed] == experts[g])
                seen.extend(routed.tolist())
            assert sorted(seen) == list(range(50)), (scale, i)
        checks.append(
            dict(scale=scale, max_logit_error=max_error, id_differences=disagreements)
        )
    trials = {name: [] for name in graphs}
    for trial in range(7):
        for name in ("control", "fused") if trial % 2 else ("fused", "control"):
            trials[name].append(elapsed(graphs[name]))
    report = dict(
        model_admission=False,
        rows=5,
        layers=48,
        ctas=5 if args.token_cta else 80,
        warps_per_cta=16 if args.token_cta else 8,
        token_cta=args.token_cta,
        weight_bytes=48 * 512 * 2560 * 2,
        floor_ms_750GBs=48 * 512 * 2560 * 2 / 750e6,
        checks=checks,
        median_ms={name: statistics.median(s) for name, s in trials.items()},
        samples_ms=trials,
        native_sha256=hashlib.sha256(Path("vllm/_C.abi3.so").read_bytes()).hexdigest(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
