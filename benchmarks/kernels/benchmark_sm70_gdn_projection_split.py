# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse only the four GDN projection copies, never the GEMM arithmetic.

The current opaque M1 op falls back to two unchanged GEMMs followed by four
contiguous copies at M>1. Screen one bit-preserving copy kernel including
both GEMMs and rotating weight allocations. No model-quality claim.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.kernels.benchmark_sm70_moe_packed_w13 import graph, latency
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
    _qwen38_gdn_projection_split_kernel as split_kernel,
)
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
    _split_gdn_projection_outputs as split,
)


def reference(qkvz, ba):
    return (
        qkvz[:, :2560].contiguous(),
        qkvz[:, 2560:].contiguous(),
        ba[:, :12].contiguous(),
        ba[:, 12:].contiguous(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokens", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    args = p.parse_args()
    if args.out.exists() or min(args.tokens) < 2:
        p.error("Output must be new; this screen is for M>=2")
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(20260905)
    # Exercise every 16-bit payload in each of the four store branches.
    bits = torch.arange(65536, device="cuda").to(torch.int16)
    exhaustive = bits.view(torch.float16)[:, None].expand(-1, 2).contiguous()
    exhaustive_out = tuple(torch.empty_like(bits).view(torch.float16) for _ in range(4))
    split_kernel[(65536, 1)](
        exhaustive,
        exhaustive,
        *exhaustive_out,
        QKV=1,
        Z=1,
        B=1,
        A=1,
        BLOCK=256,
        num_warps=4,
        num_stages=1,
    )
    assert all(torch.equal(value.view(torch.int16), bits) for value in exhaustive_out)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights = {}
    for name in ("qkv", "z", "b", "a"):
        key = f"model.language_model.layers.0.linear_attn.in_proj_{name}.weight"
        with safe_open(args.model / index[key], framework="pt", device="cpu") as f:
            weights[name] = f.get_tensor(key).half().cuda()
    q, k, v = weights["qkv"].split((2048, 2048, 6144))
    wq = torch.cat((q[:512], k[:512], v[:1536], weights["z"][:1536])).contiguous()
    wb = torch.cat((weights["b"][:12], weights["a"][:12])).contiguous()
    assert wq.shape == (4096, 2560) and wb.shape == (24, 2560)
    # Sixteen allocations exceed L2, but contain the same actual layer's weights.
    copies = [(wq.clone(), wb.clone()) for _ in range(16)]
    rows = []
    for m in args.tokens:
        x = torch.randn(m, 2560, device="cuda", dtype=torch.float16) * 0.1
        qkvz = torch.empty(m, 4096, device="cuda", dtype=torch.float16)
        ba = torch.empty(m, 24, device="cuda", dtype=torch.float16)
        saved = [None, None]

        def run(mode, linear=False, x=x, qkvz=qkvz, ba=ba, saved=saved):
            if linear:
                for w, wg in copies:
                    q = torch.nn.functional.linear(x, w)
                    g = torch.nn.functional.linear(x, wg)
                    saved[mode] = (reference if mode == 0 else split)(q, g)
            else:
                saved[mode] = (reference if mode == 0 else split)(qkvz, ba)

        gs = [graph(lambda mode=mode: run(mode)) for mode in (0, 1)]
        # Raw payload patterns include signed zero, subnormals, infinities, NaNs.
        for shift in (0, 17, 32768, 65500):
            for value in (qkvz, ba):
                bits = (
                    (torch.arange(value.numel(), device="cuda") + shift) % 65536
                ).to(torch.int16)
                value.view(torch.int16).copy_(bits.reshape_as(value))
            for mode in (0, 1):
                for value in saved[mode]:
                    value.view(torch.int16).fill_(12345)
                gs[mode].replay()
            for actual, expected in zip(saved[1], saved[0], strict=True):
                assert actual.is_contiguous()
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        for scope in ("copies", "complete_projection_rotating_weights"):
            graphs = (
                gs
                if scope == "copies"
                else [
                    graph(lambda mode=mode: run(mode, True), unroll=1)
                    for mode in (0, 1)
                ]
            )
            for mode in (0, 1):
                graphs[mode].replay()
            assert all(
                torch.equal(a.view(torch.int16), b.view(torch.int16))
                for a, b in zip(saved[0], saved[1], strict=True)
            )
            times = [[], []]
            for sample in range(5):
                for mode in (0, 1) if sample % 2 == 0 else (1, 0):
                    times[mode].append(latency(graphs[mode], 20, 16))
            result = dict(
                m=m,
                scope=scope,
                median_us=list(map(statistics.median, times)),
                samples_us=times,
                bitwise_copy_replays=4,
                final_projection_bitwise=True,
            )
            rows.append(result)
            print(json.dumps(result), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                model=str(args.model),
                layer=0,
                tp4_rank=0,
                synthetic_activations=True,
                weight_copies=16,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                results=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
