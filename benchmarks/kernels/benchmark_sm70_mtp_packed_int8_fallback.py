# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen one packed INT8 store for draft decode and larger-batch fallback.

Includes the complete expert layer: route alignment, up, SiLU, down and route
sum. Does not include shared experts, TP communication or the model round.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from safetensors.torch import load_file

from benchmarks.kernels.benchmark_sm70_mtp_fp32_dense import capture, elapsed
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.models.qwen4_exp.nvidia.sm70_mtp_int8 import (
    packed_int8_fallback_experts,
    prepare_int8_expert_weight,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    saved = load_file(str(args.weights))
    dense, packed, scales, unsigned = [], [], [], []
    for role, n, k in (("w13", 320, 2560), ("w2", 2560, 160)):
        weight = torch.zeros(512, n, k, dtype=torch.float16, device="cuda")
        weight[:50].copy_(saved[role])
        codes, scale = prepare_int8_expert_weight(weight.view(-1, k), block32=True)
        integer = torch.full(weight.shape, 128, dtype=torch.uint8, device="cuda")
        # Independent row-major oracle, with unsigned W8A16 zero point 128.
        rows = saved[role].cuda().float().view(50, n, k // 32, 32)
        local_scale = rows.abs().amax(-1, keepdim=True) / 127
        local_scale = torch.where(local_scale == 0, 1, local_scale)
        quantized = (rows / local_scale).round().clamp(-127, 127) + 128
        integer[:50].copy_(quantized.reshape(50, n, k).to(torch.uint8))
        dense.append(weight)
        packed.append(codes)
        scales.append(scale.view(k // 32, 512, n))
        unsigned.append(integer)
        del rows, local_scale, quantized
    config = FusedMoEQuantConfig.make(
        weight_dtype=torch.int8,
        block_shape=[0, 32],
        w1_scale=scales[0].permute(1, 2, 0),
        w2_scale=scales[1].permute(1, 2, 0),
    )
    report = {
        "model_admission": False,
        "complete_round_measurement": False,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "dense_bytes": sum(w.nbytes for w in dense),
        "canonical_packed_bytes": sum(w.nbytes for w in packed + scales),
        "rows": [],
    }
    for m in (4, 20, 2048):
        x = torch.randn(m, 2560, dtype=torch.float16, device="cuda") * 0.1
        ids = torch.randn(m, 50, device="cuda").topk(10, -1).indices.to(torch.int32)
        probabilities = torch.randn(m, 10, device="cuda").softmax(-1)
        outputs = [None, None, None]

        def run(arm, outputs=outputs, x=x, probabilities=probabilities, ids=ids):
            if arm == 2:
                outputs[arm] = packed_int8_fallback_experts(
                    x, packed[0], scales[0], packed[1], scales[1], probabilities, ids
                )
            else:
                weights = dense if arm == 0 else unsigned
                outputs[arm] = fused_experts(
                    x,
                    weights[0],
                    weights[1],
                    probabilities,
                    ids,
                    quant_config=None if arm == 0 else config,
                )

        graphs = [capture(lambda arm=arm: run(arm)) for arm in range(3)]
        for graph in graphs:
            graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(outputs[1], outputs[2], rtol=0, atol=0)
        assert all(torch.isfinite(out).all() for out in outputs)
        samples = [[], [], []]
        for trial in range(5):
            for arm in (0, 1, 2) if trial % 2 else (2, 1, 0):
                samples[arm].append(elapsed(graphs[arm]))
        report["rows"].append(
            {
                "m": m,
                "arms": ["fp16", "rowmajor_uint8", "packed_int8"],
                "quantized_outputs_bitwise_equal": True,
                "median_ms": [statistics.median(values) for values in samples],
                "samples_ms": samples,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
