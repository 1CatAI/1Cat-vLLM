# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact batched shared-gate epilogue screen; not endpoint throughput.

Runs the unchanged linear in both arms, with 48 distinct checkpoint weights.
Every FP16 gate input bit pattern is checked independently of the checkpoint.
Run on one reserved idle V100 with the source-built native extension.
"""

import argparse
import json
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm import _sm70_ops as ops


def assert_bits(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )


def exhaustive_epilogue():
    logits = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32)
    logits = logits.short().view(torch.float16).reshape(-1, 1)
    source = torch.randn(16, 2560, device="cuda", dtype=torch.float16)
    # Include underflow, sign and saturation in the output multiply as well.
    specials = torch.tensor(
        [0, -32768, 1, -32767, 31743, -1025, 15360, -17408],
        device="cuda",
        dtype=torch.int16,
    ).view(torch.float16)
    source[:, : specials.numel()] = specials
    for start in range(0, len(logits), 16):
        gate = logits[start : start + 16]
        expected = source * torch.sigmoid(gate)
        actual = source.clone()
        ops.qwen38_shared_gate_sigmoid_mul_out(actual, gate)
        assert_bits(actual, expected)
    return {"logit_bit_patterns": len(logits), "finite_bits_exact": True}


def load_weights(model):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    keys = [key for key in index if key.endswith(".mlp.shared_expert_gate.weight")]
    # Exclude the speculative layer, if present.
    keys = sorted(key for key in keys if "mtp" not in key)
    assert len(keys) == 48, (len(keys), keys[:2])
    weights = []
    for key in keys:
        with safe_open(model / index[key], framework="pt", device="cpu") as handle:
            weights.append(
                handle.get_tensor(key).to(device="cuda", dtype=torch.float16)
            )
    return weights


def capture(x, weights, source, fused):
    buffers = [source.clone() for _ in weights]

    def run():
        for weight, out in zip(weights, buffers, strict=True):
            gate = F.linear(x, weight)
            out.copy_(source)  # Identical reset in both arms; not model overhead.
            if fused:
                ops.qwen38_shared_gate_sigmoid_mul_out(out, gate)
            else:
                out.mul_(torch.sigmoid(gate))

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph, buffers


def elapsed(graph, replays):
    for _ in range(10):
        graph.replay()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / replays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (7, 0)
    assert ops.has_qwen38_shared_gate_sigmoid_mul()
    torch.manual_seed(20260926)
    report = {"exhaustive": exhaustive_epilogue(), "rows": [], "endpoint": False}
    print(json.dumps(report["exhaustive"]), flush=True)
    weights = load_weights(args.model)
    for rows in (2, 4, 8, 16):
        x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
        source = torch.randn_like(x)
        control, expected = capture(x, weights, source, False)
        candidate, actual = capture(x, weights, source, True)
        for _ in range(16):
            x.normal_()
            source.normal_()
            for out in actual:
                out.fill_(float("nan"))
            control.replay()
            candidate.replay()
            for a, b in zip(actual, expected, strict=True):
                assert_bits(a, b)
        samples = [[], []]
        for repeat in range(6):
            for i in (0, 1) if repeat % 2 == 0 else (1, 0):
                samples[i].append(elapsed((control, candidate)[i], args.iterations))
        record = dict(
            rows=rows,
            layers=48,
            exact_dynamic_replays=16,
            control_ms=median(samples[0]),
            candidate_ms=median(samples[1]),
            saving_ms=median(samples[0]) - median(samples[1]),
            samples_ms=samples,
        )
        report["rows"].append(record)
        print(json.dumps(record), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
