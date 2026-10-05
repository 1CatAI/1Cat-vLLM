# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure pure MTP PLE metadata host work with an installed runtime."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import vllm
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
)


def fixture(n):
    builder = PleShortConvAttentionMetadataBuilder.__new__(
        PleShortConvAttentionMetadataBuilder
    )
    builder.use_spec_decode = True
    builder.use_full_cuda_graph = True
    builder.num_spec = 4
    builder.decode_cudagraph_max_bs = 4
    builder.decode_cudagraph_max_tokens = 20
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none")
    )
    builder.kv_cache_spec = SimpleNamespace(block_size=1)
    for field, shape, dtype in (
        ("spec_state_indices_tensor", (4,), torch.int32),
        ("spec_sequence_masks", (4,), torch.bool),
        ("spec_query_start_loc", (5,), torch.int32),
        ("num_accepted_tokens", (4,), torch.int32),
    ):
        setattr(builder, field, torch.empty(shape, dtype=dtype, device="cuda"))
    starts = torch.arange(0, 5 * n + 1, 5, dtype=torch.int32)
    slots = torch.arange(10, 10 + n, device="cuda", dtype=torch.int32).reshape(n, 1)
    accepted = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
    batch = SimpleNamespace(
        query_start_loc=starts.cuda(),
        query_start_loc_cpu=starts,
        num_reqs=n,
        num_actual_tokens=5 * n,
        block_table_tensor=slots,
        seq_lens=torch.full((n,), 128, dtype=torch.int32, device="cuda"),
    )
    drafts = torch.full((n,), 4, dtype=torch.int32)

    def build():
        return builder.build(
            0, batch, num_accepted_tokens=accepted, num_decode_draft_tokens_cpu=drafts
        )

    result = build()
    torch.testing.assert_close(result.spec_state_indices_tensor, slots[:, 0])
    torch.testing.assert_close(result.num_accepted_tokens, accepted)
    return build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--requests", type=int, choices=(1, 4))
    args = p.parse_args()
    torch.set_num_threads(1)
    if "site-packages" not in vllm.__file__:
        raise RuntimeError("Use an ordinary installed runtime")
    rows = []
    for n in (args.requests,) if args.requests else (1, 4):
        build = fixture(n)
        for _ in range(20):
            build()
        torch.accelerator.synchronize()
        if args.profile:
            torch.cuda.cudart().cudaProfilerStart()
        samples = []
        for _ in range(100):
            start = time.perf_counter_ns()
            build()
            samples.append((time.perf_counter_ns() - start) / 1000)
        if args.profile:
            torch.cuda.cudart().cudaProfilerStop()
        torch.accelerator.synchronize()
        rows.append({"requests": n, "median_host_us": statistics.median(samples)})
    report = {
        "version": vllm.__version__,
        "rows": rows,
        "qualification": "Metadata host wall; excludes pending GPU completion.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
