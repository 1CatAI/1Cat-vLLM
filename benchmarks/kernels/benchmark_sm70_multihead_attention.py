# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare E4M3 multi-head attention operators using CUDA Graph replay."""

import argparse
import hashlib
import json
import os
import subprocess
from functools import partial
from pathlib import Path

import torch
from flash_attn_v100 import flash_attn_interface as fa


def timed_graph(call):
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(5):
        graph.replay()
    samples = []
    for _ in range(5):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(50):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 50)
    return sorted(samples)[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["VLLM_FLASH_V100_E4M3_SCALAR_FAST"] = "0"
    torch.manual_seed(20260921)
    reports = []
    for heads in (1, 2, 4):
        for batch, length in [(b, 2048) for b in (1, 2, 4, 8, 16, 32)] + [(1, 262144)]:
            page = 800
            blocks = (length + page - 1) // page
            storage = (
                torch.randn(
                    batch * blocks,
                    2,
                    page,
                    heads,
                    256,
                    device="cuda",
                    dtype=torch.float16,
                )
                * 0.25
            )
            storage = storage.to(torch.float8_e4m3fn).view(torch.uint8)
            k, v = storage[:, 0], storage[:, 1]
            q = torch.randn(batch, heads * 6, 256, device="cuda", dtype=torch.float16)
            table = torch.arange(
                batch * blocks, device="cuda", dtype=torch.int32
            ).reshape(batch, blocks)
            lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
            scalar_out, xqa_out = torch.empty_like(q), torch.empty_like(q)
            kwargs = dict(
                kv_cache_dtype="fp8_e4m3",
                k_scale=0.75,
                v_scale=1.25,
                max_seq_len_hint=length,
                workspace_seq_capacity_hint=blocks * page,
            )

            scalar = partial(
                fa.flash_attn_decode_paged,
                q,
                k,
                v,
                table,
                lengths,
                out=scalar_out,
                **kwargs,
            )
            xqa = partial(
                fa.flash_attn_decode_paged_xqa,
                q,
                k,
                v,
                table,
                lengths,
                out=xqa_out,
                **kwargs,
                partition_size_hint=64 if batch == 1 else None,
                batch_context_routing=True,
            )

            scalar_ms = timed_graph(scalar)
            xqa_ms = timed_graph(xqa)
            assert torch.isfinite(scalar_out).all() and torch.isfinite(xqa_out).all()
            error = (
                scalar_out.float() - xqa_out.float()
            ).norm() / scalar_out.float().norm().clamp_min(1e-12)
            assert error < 0.007
            reports.append(
                dict(
                    kv_heads=heads,
                    q_heads=heads * 6,
                    batch=batch,
                    context=length,
                    scalar_ms=scalar_ms,
                    xqa_ms=xqa_ms,
                    speedup=scalar_ms / xqa_ms,
                    relative_l2=float(error),
                )
            )
            print(json.dumps(reports[-1]), flush=True)
    binary = Path(fa.flash_attn_v100_cuda.__file__)
    args.out.write_text(
        json.dumps(
            dict(
                source=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                working_tree_diff_sha256=hashlib.sha256(
                    subprocess.check_output(["git", "diff"])
                ).hexdigest(),
                native_path=str(binary),
                native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                mode="CUDA Graph replay; kernel-only, not serving TPS",
                scalar_fast=False,
                results=reports,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
