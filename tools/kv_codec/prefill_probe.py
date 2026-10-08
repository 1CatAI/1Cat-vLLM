# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assert the installed backend's Q8192 long-prefill route with an FP64 oracle.

Synthetic operator inputs, not model/quality admission. The historical 75-TFLOP
family is labelled 79t in backend counters. Use ordinary wheels and GPU locks.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from vllm.v1.attention.backends import flash_attn_v100 as backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=128032)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=9)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (7, 0)
    assert args.context >= 8192 and args.context % 32 == 0
    assert args.heads > 0 and args.iterations > 0
    assert "site-packages" in Path(backend.__file__).parts
    assert backend._route_summary_enabled(), "Enable existing routing diagnostics"
    generator = torch.Generator(device="cuda").manual_seed(20261009)
    q = torch.randn(
        (1, 8192, 6 * args.heads, 256),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    k = torch.randn(
        (1, args.context, args.heads, 256),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    v = torch.randn(k.shape, device="cuda", dtype=k.dtype, generator=generator)
    out = torch.empty_like(q)
    q_start = torch.tensor([0, 8192], device="cuda", dtype=torch.int32)
    k_start = torch.tensor([0, args.context], device="cuda", dtype=torch.int32)
    before = dict(backend._route_counts)

    def call():
        result = backend._try_sm70_fa2_d256_prefill(
            q,
            k,
            v,
            cu_seqlens_q=q_start,
            cu_seqlens_k=k_start,
            max_seqlen_q=8192,
            max_seqlen_k=args.context,
            softmax_scale=0.0625,
            causal=True,
            window_size=(-1, -1),
            out=out,
        )
        assert result is not None, "Backend rejected Q8192 long-prefill shape"

    call()
    delta = {
        name: count - before.get(name, 0)
        for name, count in backend._route_counts.items()
        if count > before.get(name, 0)
    }
    assert delta.get("prefill_dense_d256_gqa_79t_fp32_q8192", 0) == 1, delta
    assert torch.isfinite(out).all()
    expected_bytes = out.view(torch.int16).clone()
    rows = torch.tensor([0, 63, 4095, 8191], device="cuda")
    errors = []
    for head in range(args.heads):
        query = q[0, rows, head * 6 : (head + 1) * 6].transpose(0, 1).double()
        scores = query @ k[0, :, head].double().T * 0.0625
        scores.masked_fill_(
            torch.arange(args.context, device="cuda")[None, None]
            > (args.context - 8192 + rows)[None, :, None],
            -torch.inf,
        )
        reference = (scores.softmax(-1) @ v[0, :, head].double()).transpose(0, 1)
        actual = out[0, rows, head * 6 : (head + 1) * 6].double()
        relative_l2 = float((actual - reference).norm() / reference.norm())
        assert relative_l2 < 0.006, relative_l2
        errors.append(relative_l2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    flush = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)
    timings = []
    for _ in range(args.iterations):
        flush.zero_()
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(begin.elapsed_time(end) * 1000)
    assert torch.equal(out.view(torch.int16), expected_bytes)
    args.out.write_text(
        json.dumps(
            {
                "scope": (
                    "installed backend operator route/correctness; "
                    "no model or speed parity admission"
                ),
                "context": args.context,
                "query_length": 8192,
                "kv_heads": args.heads,
                "route_delta_first_call": delta,
                "sampled_fp64_relative_l2": errors,
                "graph_replay_bitwise_equal": True,
                "cold_l2_flush_bytes": flush.numel(),
                "median_us": statistics.median(timings),
                "samples_us": timings,
                "output_sha256": hashlib.sha256(
                    out.cpu().numpy().tobytes()
                ).hexdigest(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
