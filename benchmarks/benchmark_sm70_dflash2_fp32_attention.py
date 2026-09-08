# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare legacy E4M3 partials with repaired FP32 attention, not FP8 loss."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from flash_attn_v100 import (
    flash_attn_grouped_e4m3_fp32_paged,
    flash_attn_grouped_verify_paged,
)
from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda


def measure(length, page, samples):
    torch.manual_seed(20260908)
    pages = (length + page - 1) // page
    capacity = pages * page
    q = torch.randn((8, 6, 256), dtype=torch.float16, device="cuda")
    raw = torch.randn((2, capacity, 1, 256), dtype=torch.float16, device="cuda")
    encoded = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    backing = torch.empty((pages, 2, page, 1, 256), dtype=torch.uint8, device="cuda")
    k, v = backing.unbind(1)
    order = torch.randperm(pages, device="cuda")
    k[order] = encoded[0].reshape_as(k)
    v[order] = encoded[1].reshape_as(v)
    table = order.int()[None].contiguous()
    seq = torch.tensor([length], dtype=torch.int32, device="cuda")
    rows = torch.arange(length - 7, length + 1, dtype=torch.int32, device="cuda")
    outputs = [torch.empty_like(q), torch.empty_like(q)]

    def legacy():
        flash_attn_grouped_verify_paged(
            q,
            k,
            v,
            table,
            seq,
            out=outputs[0],
            softmax_scale=0.0625,
            kv_cache_dtype="fp8_e4m3",
            k_scale=0.5,
            v_scale=1.25,
            one_pass=True,
        )

    def repaired():
        flash_attn_grouped_e4m3_fp32_paged(
            q,
            k,
            v,
            table,
            rows,
            out=outputs[1],
            softmax_scale=0.0625,
            k_scale=0.5,
            v_scale=1.25,
        )

    graphs = []
    for call in (legacy, repaired):
        for _ in range(20):
            call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs.append(graph)
    timings = [[], []]
    for _ in range(samples):
        for index in (0, 1, 1, 0):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[index].replay()
            end.record()
            end.synchronize()
            timings[index].append(start.elapsed_time(end))

    # FP64 resolves the FP16 output-rounding floor; also report the requested
    # PyTorch FP32 oracle on exactly the same quantized KV and causal rows.
    errors = [{}, {}]
    for dtype in (torch.float32, torch.float64):
        rk = encoded[0, :length, 0].view(torch.float8_e4m3fn).to(dtype) * 0.5
        rv = encoded[1, :length, 0].view(torch.float8_e4m3fn).to(dtype) * 1.25
        scores = q.transpose(0, 1).to(dtype) @ rk.T * 0.0625
        mask = torch.arange(length, device="cuda")[None] >= rows[:, None]
        scores.masked_fill_(mask[None], -torch.inf)
        expected = (scores.softmax(-1) @ rv).transpose(0, 1)
        denominator = expected.norm()
        floor = float((expected.half().to(dtype) - expected).norm() / denominator)
        for error, output in zip(errors, outputs):
            diff = output.to(dtype) - expected
            error[str(dtype)] = {
                "relative_l2": float(diff.norm() / denominator),
                "max_abs": float(diff.abs().max()),
                "fp16_rounding_floor": floor,
                "finite": bool(torch.isfinite(output).all()),
            }
    result = {"length": length, "page": page, "variants": {}}
    for name, times, error in zip(("legacy_half", "repaired_fp32"), timings, errors):
        values = torch.tensor(times)
        result["variants"][name] = {
            "median_ms": float(values.median()),
            "p10_ms": float(values.quantile(0.1)),
            "p90_ms": float(values.quantile(0.9)),
            "samples_ms": times,
            "error": error,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=25)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("requires an idle SM70 GPU")
    native = Path(flash_attn_v100_cuda.__file__).resolve()
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "native_path": str(native),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "precision_version": flash_attn_v100_cuda.grouped_e4m3_fp32_precision_version(),
        "contract": "B1/q8/H6/Hkv1/D256, same E4M3 KV, graph ABBA; not E2E",
        "cases": [],
    }
    for length in (8192, 65536, 131072, 262144):
        row = measure(length, 3296, args.samples)
        result["cases"].append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
