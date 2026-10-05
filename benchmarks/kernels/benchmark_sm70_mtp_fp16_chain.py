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
    formats = parser.add_mutually_exclusive_group()
    formats.add_argument("--qpn8", action="store_true")
    formats.add_argument("--int8", action="store_true")
    formats.add_argument("--int8-block32", action="store_true")
    args = parser.parse_args()
    integer_format = args.int8 or args.int8_block32
    quantized = args.qpn8 or integer_format
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    saved = load_file(str(args.weights))
    assert saved["w13"].shape == (50, 320, 2560)
    assert saved["w2"].shape == (50, 2560, 160)
    w13 = torch.zeros(512, 320, 2560, device="cuda", dtype=torch.float16)
    w2 = torch.zeros(512, 2560, 160, device="cuda", dtype=torch.float16)
    w13[:50].copy_(saved["w13"])
    w2[:50].copy_(saved["w2"])
    if quantized:
        from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
            prepare_channel_qpn8_weight,
        )

        prepare = prepare_channel_qpn8_weight
        if integer_format:
            from vllm.models.qwen4_exp.nvidia.sm70_mtp_int8 import (
                prepare_int8_expert_weight,
            )

            prepare = lambda w: prepare_int8_expert_weight(w, block32=args.int8_block32)
        codes13, scales13 = prepare(w13.view(-1, 2560))
        codes2, scales2 = prepare(w2.view(-1, 160))
        quantized13, quantized2 = torch.empty_like(w13), torch.empty_like(w2)
        for dense, codes, scales, k in (
            (quantized13, codes13, scales13, 2560),
            (quantized2, codes2, scales2, 160),
        ):
            # Existing decoder emits [K,N], while MoE weights are [E,N,K].
            if integer_format:
                # Independent row-major quantization oracle; do not decode the
                # packed layout used by the candidate.
                source = w13 if k == 2560 else w2
                rows = source.view(-1, k).float()
                if args.int8_block32:
                    rows = rows.view(-1, k // 32, 32)
                scale = rows.float().abs().amax(-1, keepdim=True) / 127
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                integer = (rows.float() / scale).round().clamp(-127, 127).half()
                dense.copy_((integer * scale.half()).reshape_as(dense))
            else:
                decoded = torch.empty_like(codes, dtype=torch.float16)
                torch.ops._C.fp8_qpn8_dequantize_sm70_out(decoded, codes, scales)
                dense.copy_(decoded.t().reshape_as(dense))
        scales13 = (
            scales13.view(80, 512, 320)
            if args.int8_block32
            else scales13.view(512, 320)
        )
        scales2 = (
            scales2.view(5, 512, 2560) if args.int8_block32 else scales2.view(512, 2560)
        )
    arms = (
        ("control", "fused", "quantized_reference")
        if quantized
        else ("control", "fused")
    )
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
            for arm in arms
        }
        padded = torch.full((1,), m * 20, device="cuda", dtype=torch.int32)
        stages.append((x, ids, probabilities, padded, buffers))

    def control(arm="control"):
        up_weight = quantized13 if arm == "quantized_reference" else w13
        down_weight = quantized2 if arm == "quantized_reference" else w2
        for x, ids, probabilities, padded, b in stages:
            c = b[arm]
            torch.ops._C.sm70_mtp_moe_fp16_out(
                c["up"], x, up_weight, ids.flatten(), probabilities, padded, False
            )
            torch.ops._C.silu_and_mul(c["activation"], c["up"].view(-1, 320))
            torch.ops._C.sm70_mtp_moe_fp16_out(
                c["down"],
                c["activation"],
                down_weight,
                ids.flatten(),
                probabilities,
                padded,
                True,
            )
            ops.moe_sum(c["down"], c["out"])

    def fused():
        for x, ids, probabilities, _, b in stages:
            f = b["fused"]
            if quantized:
                chain = (
                    torch.ops._C.sm70_mtp_moe_int8_block32_chain_out
                    if args.int8_block32
                    else torch.ops._C.sm70_mtp_moe_int8_chain_out
                    if args.int8
                    else torch.ops._C.sm70_mtp_moe_qpn8_chain_out
                )
                chain(
                    f["out"],
                    f["activation"],
                    x,
                    codes13,
                    scales13,
                    codes2,
                    scales2,
                    ids,
                    probabilities,
                )
            else:
                torch.ops._C.sm70_mtp_moe_fp16_chain_out(
                    f["out"], f["activation"], x, w13, w2, ids, probabilities
                )

    graphs = {"control": capture(control), "fused": capture(fused)}
    if quantized:
        graphs["quantized_reference"] = capture(lambda: control("quantized_reference"))
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
            if quantized:
                oracle = b["quantized_reference"]["out"]
                relative = (
                    (oracle.float() - result.float()).norm()
                    / oracle.float().norm().clamp_min(1e-12)
                ).item()
                assert relative < 0.001, "Packed byte layout/oracle mismatch"
                values[-1]["dequantized_oracle_relative_l2"] = relative
                values[-1]["dequantized_oracle_max_error"] = (
                    (oracle - result).abs().max().item()
                )
        checks.append(dict(scale=scale, stages=values))
    trials = {arm: [] for arm in graphs}
    for trial in range(7):
        order = ("control", "fused") if trial % 2 else ("fused", "control")
        for arm in order:
            trials[arm].append(elapsed(graphs[arm]))
        if quantized:
            trials["quantized_reference"].append(elapsed(graphs["quantized_reference"]))
    result = dict(
        model_admission=False,
        candidate_format=(
            "block32_INT8"
            if args.int8_block32
            else "channel_INT8"
            if args.int8
            else "channel_QPN8"
        )
        if quantized
        else "FP16",
        shapes=[5, 1, 1, 1],
        layers_per_step=1,
        control_launches=16,
        fused_launches=8,
        w13_ctas_per_m1=50 if quantized else 100,
        w13_warps_per_cta=8,
        w2_ctas_per_m1=80,
        w2_warps_per_cta=10,
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(Path("vllm/_C.abi3.so").read_bytes()).hexdigest(),
        checks=checks,
        samples_ms=trials,
        median_ms={arm: statistics.median(s) for arm, s in trials.items()},
    )

    # Localize changes in the two stages independently; same routing/weights,
    # and the down-only comparison uses the identical control activation.
    def control_stage(stage):
        for x, ids, probabilities, padded, b in stages:
            c = b["control"]
            if stage == 1:
                torch.ops._C.sm70_mtp_moe_fp16_out(
                    c["up"], x, w13, ids.flatten(), probabilities, padded, False
                )
                torch.ops._C.silu_and_mul(c["activation"], c["up"].view(-1, 320))
            else:
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

    def fused_stage(stage):
        for x, ids, probabilities, _, b in stages:
            f = b["fused"]
            activation = f["activation"] if stage == 1 else b["control"]["activation"]
            torch.ops._C.sm70_mtp_moe_fp16_chain_out(
                f["out"], activation, x, w13, w2, ids, probabilities, stage
            )

    result["stage_medians_ms"] = {}
    for stage in () if quantized else (1, 2):
        pair = {
            "control": capture(lambda s=stage: control_stage(s)),
            "fused": capture(lambda s=stage: fused_stage(s)),
        }
        samples = {name: [] for name in pair}
        for trial in range(7):
            order = ("control", "fused") if trial % 2 else ("fused", "control")
            for name in order:
                samples[name].append(elapsed(pair[name]))
        result["stage_medians_ms"][str(stage)] = {
            name: statistics.median(s) for name, s in samples.items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
