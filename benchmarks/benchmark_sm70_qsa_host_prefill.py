# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-card ABBA host QSA prefill; synthetic selections, not model throughput."""

import argparse
import json
import time
from pathlib import Path

import torch

from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import host_qsa_attention
from vllm.models.qwen4_exp.nvidia.ops.host_kv_prefill import host_qsa_prefill
from vllm.models.qwen4_exp.nvidia.ops.qsa import expand_qsa_block_indices_cuda


def measure(fn, iterations):
    torch.cuda.synchronize()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return {
        "wall_ms": (time.perf_counter() - start) * 1000 / iterations,
        "stream_ms": begin.elapsed_time(end) / iterations,
    }


def point(context, rows, dtype, iterations):
    torch.manual_seed(53)
    device = torch.device("cuda:0")
    page = 816
    blocks = (context + page - 1) // page
    state = HostQSAKV(blocks, page, 256, device, hot_tokens=8192, dtype=dtype)
    count = blocks * page
    key = torch.randn(count, 1, 256, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    state.write(key, value, torch.arange(count, device=device))
    table = torch.arange(blocks, device=device, dtype=torch.int32).view(1, -1)
    positions = torch.arange(context - rows, context, device=device)
    lengths = torch.tensor([context], device=device, dtype=torch.int32)
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    visible = ((positions + 1) // 4).clamp_min(1).int()
    columns = torch.arange(512, device=device, dtype=torch.int32)
    # Unique, spread selections once at least 512 complete blocks are visible.
    # Row-dependent shifts exercise overlap and CLOCK pressure.
    compressed = (
        columns[None, :] * visible[:, None] // 512
        + torch.arange(rows, device=device, dtype=torch.int32)[:, None] * 17
    ) % visible[:, None]
    short = visible[:, None] < 512
    compressed = torch.where(short, columns[None, :], compressed)
    compressed = torch.where(
        short & (columns[None, :] >= visible[:, None]), -1, compressed
    )
    indices = expand_qsa_block_indices_cuda(
        compressed, positions, lengths, requests, 4, 2048
    )
    query = torch.randn(rows, 6, 256, device=device, dtype=torch.float16)
    gate = torch.randn_like(query)
    old, new = torch.empty_like(query), torch.empty_like(query)

    def control():
        for start in range(0, rows, 32):
            stop = min(start + 32, rows)
            host_qsa_attention(
                query[start:stop],
                state,
                indices[start:stop],
                table,
                requests[start:stop],
                positions[start:stop],
                lengths,
                old[start:stop],
                gate[start:stop],
            )

    def candidate():
        host_qsa_prefill(
            query, state, indices, table, requests, positions, lengths, new, gate
        )

    control()
    candidate()
    torch.cuda.synchronize()
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    delta = new.float() - old.float()
    errors = {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / old.float().norm().clamp_min(1e-12)).item(),
        "bitwise_equal": torch.equal(new, old),
    }
    samples = []
    for name, fn in (
        ("control", control),
        ("candidate", candidate),
        ("candidate", candidate),
        ("control", control),
    ):
        samples.append({"arm": name, **measure(fn, iterations)})
    return {
        "context": context,
        "rows": rows,
        "dtype": str(dtype),
        "errors": errors,
        "ABBA": samples,
        "staged_FP16_bytes": blocks * 2 * page * 256 * 2,
        "additional_persistent_workspace_bytes": 0,
        "expected_control_attention_groups": (rows + 31) // 32,
        "expected_candidate_attention_groups": (
            (rows // 32 * 32 + 255) // 256 + bool(rows % 32)
        ),
        "counter_scope": "launch groups, not traced CUDA kernel counts",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 2048, 16384])
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--dtype", choices=["float16", "fp8_e4m3"], default="float16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.uint8
    report = {
        "complete": False,
        "scope": "synthetic single-layer prefill",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "points": [],
    }
    try:
        for context in args.contexts:
            for rows in args.rows:
                if rows > context:
                    continue
                result = point(context, rows, dtype, args.iterations)
                report["points"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        report["complete"] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
