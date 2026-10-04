# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare real GGUF FP16 projection routes with dense shard projections."""

import argparse
import json
import statistics
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
    _pack_router_batch_weight,
    _qwen38_sm70_fp16_gemv,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader


def graph_us(fn, repeats=100):
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(10):
        graph.replay()
    timings = []
    for _ in range(5):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(begin.elapsed_time(end) * 1000 / repeats)
    return statistics.median(timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--projection", choices=("ba", "router"), default="ba")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 0 <= args.rank < 4
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    reader = GGUFReader(str(args.model))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    weights = []
    for layer in range(48):
        names = (
            [f"blk.{layer}.ssm_{kind}.weight" for kind in ("beta", "alpha")]
            if args.projection == "ba"
            else [f"blk.{layer}.ffn_gate_inp.weight"]
        )
        if not all(name in tensors for name in names):
            continue
        shards = []
        for name in names:
            tensor = tensors[name]
            assert int(tensor.tensor_type) == 30
            value = torch.from_numpy(tensor.data.view(np.uint16).copy()).view(
                torch.bfloat16
            )
            if args.projection == "ba":
                assert value.shape == (48, 2560)
                shard = value[args.rank * 12 : (args.rank + 1) * 12].half()
            else:
                assert value.shape == (512, 2560)
                shard = value.half()
            assert torch.isfinite(shard).all()
            shards.append(shard.cuda())
        dense = torch.cat(shards) if len(shards) > 1 else shards[0]
        packed = (
            _pack_router_batch_weight(dense) if args.projection == "router" else None
        )
        weights.append((shards, dense, packed))
    assert weights
    results = []
    for m in (1, 5, 10, 16, 20):
        torch.manual_seed(20261004 + m)
        x = (torch.randn(m, 2560, device="cuda") * 0.125).half()
        role = (
            "model.layers.0.linear_attn.in_proj_ba"
            if args.projection == "ba"
            else "model.layers.0.mlp.gate"
        )
        fast_route = m <= 16 if args.projection == "ba" else m in (1, 5, 10)
        worst_abs = worst_l2 = 0.0
        for shards, dense, packed in weights:
            expected = (x.float() @ dense.float().T).half()
            if fast_route:
                with patch(
                    "torch.nn.functional.linear",
                    side_effect=AssertionError("Unexpected dense fallback"),
                ):
                    actual = _qwen38_sm70_fp16_gemv(x, dense, role, packed)
            else:
                actual = _qwen38_sm70_fp16_gemv(x, dense, role, packed)
            error = actual.float() - expected.float()
            worst_abs = max(worst_abs, error.abs().max().item())
            worst_l2 = max(
                worst_l2,
                (error.norm() / expected.float().norm().clamp_min(1e-20)).item(),
            )
            torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.003)

        def old_path(x=x):
            for shards, _, _ in weights:
                outputs = [torch.nn.functional.linear(x, w) for w in shards]
                if len(outputs) > 1:
                    torch.cat(outputs, dim=-1)

        def new_path(x=x, role=role):
            for _, dense, packed in weights:
                _qwen38_sm70_fp16_gemv(x, dense, role, packed)

        old_us = graph_us(old_path)
        new_us = graph_us(new_path)
        results.append(
            {
                "m": m,
                "layers": len(weights),
                "old_round_us": old_us,
                "new_round_us": new_us,
                "saved_round_us": old_us - new_us,
                "new_per_layer_us": new_us / len(weights),
                "max_abs_error": worst_abs,
                "max_relative_l2": worst_l2,
                "route": (
                    "router_batch"
                    if args.projection == "router" and m in (5, 10)
                    else "row_gemv"
                    if fast_route
                    else "dense_fallback"
                ),
            }
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
