# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen QSA scorer CTA amortization; no production dispatch change.

Keep max-context capacity fixed while changing live lengths. Compare the
existing kernel's contiguous tile grouping against its production grouping,
including changed-length CUDA Graph replay. Timing is operator-only, not
end-to-end throughput or model-quality admission.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from benchmarks.kernels.sm70_qsa_strided_scorer import strided_qsa_mqa_paged_kernel
from vllm.models.qwen4_exp.nvidia.ops.qsa import _qsa_mqa_paged_kernel
from vllm.triton_utils import triton


def run_case(rows, length, groups, repeats):
    device = torch.device("cuda")
    # FlashNext's indexer projection is replicated, NOT sharded over TP4.
    heads, dim, ratio, page_size, table_width = 4, 128, 4, 196, 335
    columns = page_size * table_width
    block_n = 32 if rows == 1 else 64
    torch.manual_seed(20260905)
    q = torch.randn(rows, heads, dim, dtype=torch.float16, device=device)
    cache = torch.randn(
        rows * table_width, page_size, 1, dim, dtype=torch.float16, device=device
    )
    table = torch.randperm(rows * table_width, device=device).to(torch.int32)
    table = table.reshape(rows, table_width)
    request = torch.arange(rows, dtype=torch.int32, device=device)
    positions = torch.full((rows,), length - 1, dtype=torch.int32, device=device)
    lengths = torch.full_like(positions, length)
    logits = {
        group: torch.full((rows, columns), float("nan"), device=device)
        for group in groups
    }
    visible = {group: torch.empty_like(positions) for group in groups}

    def launch(group):
        out = logits[group]
        strided = group < 0
        grid = (
            min(-group, triton.cdiv(columns, block_n))
            if strided
            else triton.cdiv(columns, block_n * group)
        )
        kernel = strided_qsa_mqa_paged_kernel if strided else _qsa_mqa_paged_kernel
        kernel[(rows, grid)](
            q,
            cache,
            table,
            request,
            positions,
            lengths,
            visible[group],
            out,
            *q.stride(),
            cache.stride(0),
            cache.stride(1),
            cache.stride(3),
            *table.stride(),
            out.stride(0),
            rows,
            columns,
            cache.shape[0],
            rows,
            math.sqrt(dim),
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=table_width,
            NUM_HEADS=heads,
            HEAD_DIM=dim,
            BLOCK_N=block_n,
            BLOCK_D=dim,
            TILES_PER_PROG=1 if strided else group,
            STAGES=2,
            MAX_N=16,
            COMPRESS_RATIO=ratio,
            num_warps=2,
        )

    graphs = {}
    for group in groups:
        for _ in range(3):
            launch(group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(group)
        graphs[group] = graph

    # Shrink, grow, mixed lengths and page residues must reuse the same graph.
    for iteration in range(4):
        values = [max(1, length - (iteration * 197 + row * 31)) for row in range(rows)]
        if iteration == 3:
            values = [262144 - row * 3 for row in range(rows)]
        lengths.copy_(torch.tensor(values, dtype=torch.int32, device=device))
        positions.copy_(lengths - 1)
        q.mul_(-0.875)
        for group in groups:
            logits[group].fill_(float("nan"))
            graphs[group].replay()
        assert torch.equal(visible[1], lengths // ratio)
        valid = torch.arange(columns, device=device)[None, :] < visible[1][:, None]
        reference = logits[1][valid]
        assert torch.isfinite(reference).all()
        for group in groups:
            assert torch.equal(visible[group], visible[1])
            assert torch.equal(logits[group][valid], reference), (rows, length, group)

    lengths.fill_(length)
    positions.fill_(length - 1)
    timings = {group: [] for group in groups}
    for cycle in range(5):
        order = groups if cycle % 2 == 0 else list(reversed(groups))
        for group in order:
            graphs[group].replay()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(repeats):
                graphs[group].replay()
            end.record()
            end.synchronize()
            timings[group].append(start.elapsed_time(end) * 1000 / repeats)
    return {
        "rows": rows,
        "length": length,
        "capacity_columns": columns,
        "index_heads": heads,
        "head_dim": dim,
        "exact_changed_replays": 4,
        "groups": {
            str(group): {
                "ctas": rows
                * (
                    min(-group, triton.cdiv(columns, block_n))
                    if group < 0
                    else triton.cdiv(columns, block_n * group)
                ),
                "live_ctas": rows
                * (
                    min(-group, triton.cdiv(length // ratio, block_n))
                    if group < 0
                    else triton.cdiv(length // ratio, block_n * group)
                ),
                "median_us": statistics.median(values),
                "samples_us": values,
            }
            for group, values in timings.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[8192, 131072, 262144]
    )
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--strided-grids", type=int, nargs="*", default=[])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a previous benchmark")
    if (
        1 not in args.groups
        or min(args.groups + args.rows + args.strided_grids + [args.repeats]) < 1
    ):
        parser.error("Positive rows/groups/repeats and baseline group 1 required")
    if not all(1 <= length <= 262144 for length in args.lengths):
        parser.error("Lengths must fit the fixed 256K capacity")
    assert torch.cuda.get_device_capability() == (7, 0)
    results = []
    for rows in args.rows:
        for length in args.lengths:
            modes = args.groups + [-grid for grid in args.strided_grids]
            result = run_case(rows, length, modes, args.repeats)
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
