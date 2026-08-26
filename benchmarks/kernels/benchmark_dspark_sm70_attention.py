# SPDX-License-Identifier: Apache-2.0

"""Replay a real DSpark packed-FP8 attention dump at N=7."""

import argparse
import json
from pathlib import Path

import torch

from vllm.models.deepseek_v4.sm70.sparse_kernels import (
    sm70_sparse_attention_paged_fp8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def expand_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    repeats = (rows + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))[:rows].contiguous()


def main() -> None:
    args = parse_args()
    if args.num_speculative_tokens <= 0:
        raise ValueError("--num-speculative-tokens must be positive")

    torch.cuda.set_device(args.device)
    saved = torch.load(args.dump, map_location="cpu", weights_only=True)
    q = expand_rows(saved["q"], args.num_speculative_tokens).cuda(args.device)
    main_indices = expand_rows(saved["main_indices"], args.num_speculative_tokens).cuda(
        args.device
    )
    main_lengths = expand_rows(saved["main_lengths"], args.num_speculative_tokens).cuda(
        args.device
    )
    expected = expand_rows(saved["output"], args.num_speculative_tokens).cuda(
        args.device
    )
    main_cache = saved["main_cache"].cuda(args.device)
    attn_sink = saved["attn_sink"].cuda(args.device)
    output = torch.empty_like(q)

    def launch() -> None:
        sm70_sparse_attention_paged_fp8(
            q=q,
            main_cache=main_cache,
            main_indices=main_indices,
            main_lengths=main_lengths,
            scale=float(saved["scale"]),
            attn_sink=attn_sink,
            out=output,
        )

    capture_stream = torch.cuda.Stream(device=args.device)
    capture_stream.wait_stream(torch.cuda.current_stream(args.device))
    with torch.cuda.stream(capture_stream):
        launch()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        launch()
    graph.replay()
    torch.cuda.synchronize(args.device)
    first_replay = output.clone()
    for _ in range(args.replays - 1):
        graph.replay()
    torch.cuda.synchronize(args.device)

    diff = output.float() - expected.float()
    relative_l2 = float(
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            output.float().flatten(), expected.float().flatten(), dim=0
        )
    )
    result = {
        "device": torch.cuda.get_device_name(args.device),
        "layer_prefix": saved["layer_prefix"],
        "q_shape": list(q.shape),
        "main_cache_shape": list(main_cache.shape),
        "main_lengths": main_lengths.tolist(),
        "replays": args.replays,
        "finite": bool(torch.isfinite(output).all().item()),
        "replay_max_abs": float((output - first_replay).abs().max().item()),
        "reference_max_abs": float(diff.abs().max().item()),
        "reference_relative_l2": relative_l2,
        "reference_cosine": cosine,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if not result["finite"] or result["replay_max_abs"] != 0.0:
        raise RuntimeError("DSpark N=7 attention graph is unstable")
    if relative_l2 > 1e-3 or cosine < 0.9999:
        raise RuntimeError("DSpark N=7 attention diverged from the real dump")


if __name__ == "__main__":
    main()
