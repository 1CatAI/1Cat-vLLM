# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the normally built router extension with the production operators."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import vllm._sm70_router_C as native
from sm70_router_benchmark_utils import load_weights, outputs, pair, topk

from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import _pack_router_batch_weight


def measure(m, weights, packed, replays):
    inputs = [torch.randn(m, 2560, device="cuda", dtype=torch.float16) for _ in weights]
    logits = [
        [torch.empty(m, 512, device="cuda", dtype=torch.float16) for _ in weights]
        for _ in range(2)
    ]
    out = [[outputs(m) for _ in weights] for _ in range(2)]
    q8 = [
        [torch.empty(m, 80, 36, device="cuda", dtype=torch.uint8) for _ in weights]
        for _ in range(2)
    ]

    def projection(i, arm):
        if m <= 16:
            torch.ops._C.qwen38_router_batch_sm70_out(
                logits[arm][i], inputs[i], packed[i]
            )
        else:
            torch.mm(inputs[i], weights[i].t(), out=logits[arm][i])

    def call(arm, variant, quantize=False):
        for i in range(48):
            projection(i, arm)
            if variant == "select_quantize":
                torch.ops.vllm_sm70_router.select_quantize(
                    *out[arm][i], q8[arm][i], logits[arm][i], inputs[i]
                )
            elif variant == "top10":
                torch.ops.vllm_sm70_router.top10(*out[arm][i], logits[arm][i])
            else:
                topk(logits[arm][i], out[arm][i])
            if quantize and variant != "select_quantize":
                torch.ops._C.gguf_quantize_q8_1_sm70_out(q8[arm][i], inputs[i])

    results = {}
    for variant in ("top10", "select_quantize"):
        quantize = variant == "select_quantize"
        call(0, "control", quantize)
        call(1, variant, quantize)
        check = dict(
            ids_exact=all(torch.equal(out[0][i][1], out[1][i][1]) for i in range(48)),
            source_exact=all(
                torch.equal(out[0][i][2], out[1][i][2]) for i in range(48)
            ),
            weights_max_abs=max(
                (out[0][i][0] - out[1][i][0]).abs().max().item() for i in range(48)
            ),
            q8_exact=(
                not quantize or all(torch.equal(q8[0][i], q8[1][i]) for i in range(48))
            ),
        )
        assert check["ids_exact"] and check["source_exact"] and check["q8_exact"], check
        assert check["weights_max_abs"] < 3e-7, check
        result = dict(
            check=check,
            timing=pair(
                lambda quantize=quantize: call(0, "control", quantize),
                lambda variant=variant, quantize=quantize: call(1, variant, quantize),
                replays,
            ),
        )
        results[variant] = result
        print(
            "native",
            m,
            variant,
            check,
            {
                k: result["timing"][k]
                for k in ("control_ms", "candidate_ms", "saved_ms")
            },
            flush=True,
        )
    return dict(m=m, variants=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replays", type=int, default=40)
    args = p.parse_args()
    torch.manual_seed(271828)
    native_path = Path(native.__file__)
    hardware_query = [
        "nvidia-smi",
        "-i",
        "0",
        "--query-gpu=name,driver_version,clocks.sm,clocks.mem",
        "--format=csv,noheader",
    ]
    result = dict(
        scope="48-layer standalone router graph chains, no model timing",
        module=str(native_path),
        sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        control_selector="M5 default Triton top-16; M20 generic CUDA topk_softmax",
        hardware=subprocess.check_output(
            hardware_query,
            text=True,
        ).strip(),
        measurements=[],
    )
    weights = load_weights(args.model)
    packed = [_pack_router_batch_weight(w) for w in weights]
    for m in (5, 20):
        measurement = measure(m, weights, packed, args.replays)
        measurement["hardware_after_timing"] = subprocess.check_output(
            hardware_query, text=True
        ).strip()
        result["measurements"].append(measurement)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
