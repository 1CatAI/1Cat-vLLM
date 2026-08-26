# SPDX-License-Identifier: Apache-2.0

"""Validate an uncached SM70 FP8 GEMM shape under CUDA Graph capture."""

import argparse
import json
import os
from pathlib import Path

import torch
import vllm._C  # noqa: F401
from safetensors import safe_open

from vllm import _sm70_ops as sm70_ops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_shard", type=Path)
    parser.add_argument("--weight-name", default="mtp.0.main_proj.weight")
    parser.add_argument("--scale-name", default="mtp.0.main_proj.scale")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 1:
        raise ValueError("--rows must exceed one so M=1 can warm the stream")
    if args.replays <= 0:
        raise ValueError("--replays must be positive")

    os.environ.setdefault("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", "1")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)

    with safe_open(args.checkpoint_shard, framework="pt", device="cpu") as f:
        weight = f.get_tensor(args.weight_name).cuda(args.device)
        scales = f.get_tensor(args.scale_name).float().cuda(args.device)

    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(weight, scales, 128, False)
    k_ld, q_ld = (int(value.item()) for value in meta)
    k = tm_weight.shape[0]
    n = tm_weight.shape[1]

    inputs = torch.randn((args.rows, k), device=args.device, dtype=torch.float16).mul_(
        16.0
    )
    graph_out = torch.empty((args.rows, n), device=args.device, dtype=torch.float16)

    capture_stream = torch.cuda.Stream(device=args.device)
    capture_stream.wait_stream(torch.cuda.current_stream(args.device))
    with torch.cuda.stream(capture_stream):
        warmup_out = torch.empty((1, n), device=args.device, dtype=torch.float16)
        sm70_ops.fp8_gemm_sm70_out(
            warmup_out,
            inputs[:1],
            tm_weight,
            tm_scales,
            128,
            k_ld,
            q_ld,
            False,
        )
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        sm70_ops.fp8_gemm_sm70_out(
            graph_out,
            inputs,
            tm_weight,
            tm_scales,
            128,
            k_ld,
            q_ld,
            False,
        )
    torch.cuda.synchronize(args.device)

    graph.replay()
    torch.cuda.synchronize(args.device)
    first_replay = graph_out.clone()
    replay_max_abs = 0.0
    for _ in range(args.replays - 1):
        graph.replay()
        torch.cuda.synchronize(args.device)
        replay_max_abs = max(
            replay_max_abs,
            float((graph_out - first_replay).abs().max().item()),
        )

    eager_out = torch.empty_like(graph_out)
    sm70_ops.fp8_gemm_sm70_out(
        eager_out,
        inputs,
        tm_weight,
        tm_scales,
        128,
        k_ld,
        q_ld,
        False,
    )
    torch.cuda.synchronize(args.device)

    graph_f32 = graph_out.float()
    eager_f32 = eager_out.float()
    diff = graph_f32 - eager_f32
    relative_l2 = float(
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(eager_f32).clamp_min(1e-12)
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            graph_f32.flatten(), eager_f32.flatten(), dim=0
        )
    )
    result = {
        "device": torch.cuda.get_device_name(args.device),
        "shape": {"m": args.rows, "n": n, "k": k},
        "weight_dtype": str(weight.dtype),
        "scale_dtype": str(scales.dtype),
        "k_ld": k_ld,
        "q_ld": q_ld,
        "replays": args.replays,
        "finite": bool(torch.isfinite(graph_out).all().item()),
        "replay_max_abs": replay_max_abs,
        "eager_max_abs": float(diff.abs().max().item()),
        "eager_relative_l2": relative_l2,
        "eager_cosine": cosine,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if not result["finite"]:
        raise RuntimeError("CUDA Graph output contains non-finite values")
    if replay_max_abs != 0.0:
        raise RuntimeError("CUDA Graph replay is not deterministic")
    if relative_l2 > 5e-3 or cosine < 0.999:
        raise RuntimeError("CUDA Graph and eager outputs diverged")


if __name__ == "__main__":
    main()
