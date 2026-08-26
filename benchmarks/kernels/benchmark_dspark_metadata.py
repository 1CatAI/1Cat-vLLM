# SPDX-License-Identifier: Apache-2.0

"""Validate DSpark query expansion and non-causal SWA metadata."""

import argparse
import json

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.backends.mla.sparse_swa import (
    _compute_dspark_noncausal_swa_indices_kernel,
)
from vllm.v1.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    next_power_of_2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--prefix-len", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = args.num_speculative_tokens
    if n <= 0 or args.prefix_len <= 0:
        raise ValueError("token counts must be positive")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    total_len = args.prefix_len + n
    num_blocks = (total_len + args.block_size - 1) // args.block_size
    physical_blocks = torch.arange(
        2, 2 + num_blocks, device=device, dtype=torch.int32
    ).mul_(3)
    block_table = physical_blocks.unsqueeze(0).contiguous()

    next_token_ids = torch.tensor([42], device=device, dtype=torch.int32)
    target_positions = torch.arange(args.prefix_len, device=device, dtype=torch.int64)
    context_query_start = torch.tensor(
        [0, args.prefix_len], device=device, dtype=torch.int32
    )
    query_start = torch.tensor([0, n], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([total_len], device=device, dtype=torch.int32)
    token_to_req = torch.zeros(n, device=device, dtype=torch.int32)
    valid_tokens = torch.ones(n, device=device, dtype=torch.bool)

    input_ids = torch.empty(n, device=device, dtype=torch.int32)
    context_positions = torch.empty(args.prefix_len, device=device, dtype=torch.int64)
    query_positions = torch.empty(n, device=device, dtype=torch.int64)
    context_slots = torch.empty(args.prefix_len, device=device, dtype=torch.int64)
    query_slots = torch.empty(n, device=device, dtype=torch.int64)
    token_indices = torch.empty(n, device=device, dtype=torch.int32)

    index_width = ((args.window_size + n + 127) // 128) * 128
    swa_indices = torch.empty((n, 1, index_width), device=device, dtype=torch.int32)
    swa_lens = torch.empty(n, device=device, dtype=torch.int32)
    copy_block_size = min(256, next_power_of_2(total_len))
    copy_grid = (1, triton.cdiv(total_len, copy_block_size))

    def launch() -> None:
        copy_and_expand_dflash_inputs_kernel[copy_grid](
            next_token_ids,
            target_positions,
            input_ids,
            context_positions,
            query_positions,
            context_slots,
            query_slots,
            token_indices,
            block_table,
            block_table.stride(0),
            context_query_start,
            0,
            128799,
            args.block_size,
            n,
            n,
            args.prefix_len,
            BLOCK_SIZE=copy_block_size,
            HAS_NUM_REJECTED=False,
            SAMPLE_FROM_ANCHOR=True,
        )
        _compute_dspark_noncausal_swa_indices_kernel[(n,)](
            swa_indices,
            swa_indices.stride(0),
            swa_lens,
            args.window_size,
            index_width,
            query_start,
            seq_lens,
            token_to_req,
            valid_tokens,
            block_table,
            block_table.stride(0),
            args.block_size,
            TRITON_BLOCK_SIZE=1024,
        )

    capture_stream = torch.cuda.Stream(device=args.device)
    capture_stream.wait_stream(torch.cuda.current_stream(args.device))
    with torch.cuda.stream(capture_stream):
        launch()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        launch()
    for _ in range(args.replays):
        graph.replay()
    torch.cuda.synchronize(args.device)

    positions = torch.arange(total_len, dtype=torch.int64)
    physical_blocks_cpu = physical_blocks.cpu().to(torch.int64)
    expected_slots = (
        physical_blocks_cpu[positions // args.block_size] * args.block_size
        + positions % args.block_size
    )
    swa_start = max(args.prefix_len - args.window_size, 0)
    expected_swa_len = total_len - swa_start
    expected_swa = torch.full((index_width,), -1, dtype=torch.int32)
    expected_swa[:expected_swa_len] = expected_slots[swa_start:].to(torch.int32)

    expected_input_ids = torch.full((n,), 128799, dtype=torch.int32)
    expected_input_ids[0] = 42
    checks = {
        "input_ids": torch.equal(input_ids.cpu(), expected_input_ids),
        "context_positions": torch.equal(
            context_positions.cpu(), torch.arange(args.prefix_len)
        ),
        "query_positions": torch.equal(
            query_positions.cpu(), torch.arange(args.prefix_len, total_len)
        ),
        "context_slots": torch.equal(
            context_slots.cpu(), expected_slots[: args.prefix_len]
        ),
        "query_slots": torch.equal(
            query_slots.cpu(), expected_slots[args.prefix_len :]
        ),
        "token_indices": torch.equal(token_indices.cpu(), torch.arange(n)),
        "swa_lens": torch.equal(
            swa_lens.cpu(), torch.full((n,), expected_swa_len, dtype=torch.int32)
        ),
        "swa_rows": all(
            torch.equal(row, expected_swa) for row in swa_indices[:, 0].cpu()
        ),
    }
    result = {
        "device": torch.cuda.get_device_name(args.device),
        "num_speculative_tokens": n,
        "prefix_len": args.prefix_len,
        "query_positions": query_positions.tolist(),
        "swa_len": expected_swa_len,
        "index_width": index_width,
        "replays": args.replays,
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise RuntimeError("DSpark metadata validation failed")


if __name__ == "__main__":
    main()
