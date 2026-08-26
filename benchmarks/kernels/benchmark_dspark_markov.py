# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark DSpark's sequential Markov sampling chain."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_shard", type=Path)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--timed-replays", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_block(
    input_ids: torch.Tensor,
    anchor_indices: torch.Tensor,
    base_logits: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    output_token_ids: torch.Tensor,
) -> torch.Tensor:
    prev = input_ids[anchor_indices]
    step_logits = []
    for step in range(base_logits.shape[0]):
        markov_embed = F.embedding(prev, markov_w1)
        logits = base_logits[step : step + 1] + F.linear(markov_embed, markov_w2)
        prev = logits.argmax(dim=-1)
        output_token_ids[step : step + 1].copy_(prev)
        step_logits.append(logits)
    return torch.cat(step_logits, dim=0)


def main() -> None:
    args = parse_args()
    if args.num_speculative_tokens <= 0:
        raise ValueError("--num-speculative-tokens must be positive")
    if args.timed_replays <= 0:
        raise ValueError("--timed-replays must be positive")

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    with safe_open(args.checkpoint_shard, framework="pt", device="cpu") as f:
        markov_w1 = f.get_tensor("mtp.2.markov_head.markov_w1.weight").half()
        markov_w2 = f.get_tensor("mtp.2.markov_head.markov_w2.weight").half()
    markov_w1 = markov_w1.cuda(args.device)
    markov_w2 = markov_w2.cuda(args.device)

    vocab_size = markov_w1.shape[0]
    base_logits = torch.randn(
        (args.num_speculative_tokens, vocab_size),
        device=args.device,
        dtype=torch.float16,
    ).mul_(0.05)
    input_ids = torch.full(
        (args.num_speculative_tokens,),
        128799,
        device=args.device,
        dtype=torch.long,
    )
    input_ids[0] = 0
    anchor_indices = torch.tensor([0], device=args.device, dtype=torch.long)
    graph_token_ids = torch.empty(
        args.num_speculative_tokens, device=args.device, dtype=torch.long
    )

    capture_stream = torch.cuda.Stream(device=args.device)
    capture_stream.wait_stream(torch.cuda.current_stream(args.device))
    with torch.cuda.stream(capture_stream):
        warmup_ids = torch.empty_like(graph_token_ids)
        sample_block(
            input_ids,
            anchor_indices,
            base_logits,
            markov_w1,
            markov_w2,
            warmup_ids,
        )
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_logits = sample_block(
            input_ids,
            anchor_indices,
            base_logits,
            markov_w1,
            markov_w2,
            graph_token_ids,
        )
    torch.cuda.synchronize(args.device)

    graph.replay()
    torch.cuda.synchronize(args.device)
    first_token_ids = graph_token_ids.clone()
    graph.replay()
    torch.cuda.synchronize(args.device)
    replay_stable = bool(torch.equal(first_token_ids, graph_token_ids))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.timed_replays):
        graph.replay()
    end.record()
    end.synchronize()
    replay_ms = start.elapsed_time(end) / args.timed_replays

    eager_token_ids = torch.empty_like(graph_token_ids)
    eager_logits = sample_block(
        input_ids,
        anchor_indices,
        base_logits,
        markov_w1,
        markov_w2,
        eager_token_ids,
    )
    torch.cuda.synchronize(args.device)
    tokens_match = bool(torch.equal(graph_token_ids, eager_token_ids))
    logits_match = bool(torch.equal(graph_logits, eager_logits))
    finite = bool(torch.isfinite(graph_logits).all().item())

    result = {
        "device": torch.cuda.get_device_name(args.device),
        "num_speculative_tokens": args.num_speculative_tokens,
        "vocab_size": vocab_size,
        "markov_rank": markov_w1.shape[1],
        "graph_replay_ms": replay_ms,
        "replay_stable": replay_stable,
        "tokens_match_eager": tokens_match,
        "logits_match_eager": logits_match,
        "finite": finite,
        "token_ids": graph_token_ids.tolist(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if not replay_stable or not tokens_match or not logits_match or not finite:
        raise RuntimeError("DSpark Markov graph correctness check failed")


if __name__ == "__main__":
    main()
