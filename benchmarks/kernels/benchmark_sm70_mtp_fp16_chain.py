# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft real-weight FP16 experts: serial-FMA stages vs two fused launches.

The snapshot contains 50 selected checkpoint TP0 experts; original IDs and
hashes are stored beside it. Unselected allocation rows are never routed.
Numerical errors are reported; full-logit/model gates remain independent.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from safetensors.torch import load_file

from benchmarks.kernels.benchmark_sm70_mtp_fp32_dense import capture, elapsed
from vllm import _custom_ops as ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    saved = load_file(str(args.weights))
    assert saved["w13"].shape == (50, 320, 2560)
    assert saved["w2"].shape == (50, 2560, 160)
    w13 = torch.zeros(512, 320, 2560, device="cuda", dtype=torch.float16)
    w2 = torch.zeros(512, 2560, 160, device="cuda", dtype=torch.float16)
    w13[:50].copy_(saved["w13"])
    w2[:50].copy_(saved["w2"])
    stages = []
    for step, m in enumerate((5, 1, 1, 1)):
        x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
        ids = (
            ((torch.arange(m * 10, device="cuda") + step * 10) % 50)
            .to(torch.int32)
            .view(m, 10)
        )
        probabilities = torch.randn(m, 10, device="cuda").softmax(dim=-1)
        buffers = {
            arm: dict(
                up=x.new_empty(m, 10, 320),
                activation=x.new_empty(m * 10, 160),
                down=x.new_empty(m, 10, 2560),
                out=x.new_empty(m, 2560),
            )
            for arm in ("control", "fused")
        }
        padded = torch.full((1,), m * 20, device="cuda", dtype=torch.int32)
        stages.append((x, ids, probabilities, padded, buffers))

    def control():
        for x, ids, probabilities, padded, b in stages:
            c = b["control"]
            torch.ops._C.sm70_mtp_moe_fp16_out(
                c["up"], x, w13, ids.flatten(), probabilities, padded, False
            )
            ops.silu_and_mul(c["activation"], c["up"].view(-1, 320))
            torch.ops._C.sm70_mtp_moe_fp16_out(
                c["down"],
                c["activation"],
                w2,
                ids.flatten(),
                probabilities,
                padded,
                True,
            )
            ops.moe_sum(c["down"], c["out"])

    def fused():
        for x, ids, probabilities, _, b in stages:
            f = b["fused"]
            torch.ops._C.sm70_mtp_moe_fp16_chain_out(
                f["out"], f["activation"], x, w13, w2, ids, probabilities
            )

    graphs = {"control": capture(control), "fused": capture(fused)}
    checks = []
    for scale in (0.0, 0.03, 1.0, 3.0):
        for x, *_ in stages:
            x.normal_(0, scale)
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        values = []
        for x, _, _, _, b in stages:
            ref, result = b["control"]["out"], b["fused"]["out"]
            assert torch.isfinite(result).all()
            if scale == 0:
                assert not torch.count_nonzero(result)
            values.append(
                dict(
                    rows=x.shape[0],
                    max_error=(ref - result).abs().max().item(),
                    relative_l2=(
                        (ref.float() - result.float()).norm()
                        / ref.float().norm().clamp_min(1e-12)
                    ).item(),
                    activation_max_error=(
                        b["control"]["activation"] - b["fused"]["activation"]
                    )
                    .abs()
                    .max()
                    .item(),
                )
            )
        checks.append(dict(scale=scale, stages=values))
    trials = {arm: [] for arm in graphs}
    for trial in range(7):
        order = ("control", "fused") if trial % 2 else ("fused", "control")
        for arm in order:
            trials[arm].append(elapsed(graphs[arm]))
    result = dict(
        model_admission=False,
        shapes=[5, 1, 1, 1],
        layers_per_step=1,
        control_launches=16,
        fused_launches=8,
        w13_ctas_per_m1=100,
        w13_warps_per_cta=8,
        w2_ctas_per_m1=80,
        w2_warps_per_cta=10,
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(Path("vllm/_C.abi3.so").read_bytes()).hexdigest(),
        checks=checks,
        samples_ms=trials,
        median_ms={arm: statistics.median(s) for arm, s in trials.items()},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
