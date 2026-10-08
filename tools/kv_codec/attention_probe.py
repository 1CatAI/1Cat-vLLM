# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-shape old/new Flash-V100 operator gate, run from installed wheels.

This is direct operator evidence. Backend/model route assertion, 75T prefill,
and full C1/C4 acceptance must be measured separately with the final vLLM wheel.
Use one process per arm and hold local GPU locks for the complete A/B sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def inputs(fmt: str, heads: int, batch: int, length: int, q_len: int = 1):
    page = 784 if fmt == "auto" else 2048
    blocks = (length + page - 1) // page
    generator = torch.Generator(device="cuda").manual_seed(20261008)
    storage = (
        torch.randn(
            (batch * blocks, 2, page, heads, 256),
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        * 0.25
    )
    if fmt != "auto":
        dtype = torch.float8_e4m3fn if fmt == "fp8_e4m3" else torch.float8_e5m2
        storage = storage.to(dtype).view(torch.uint8)
    query = (
        torch.randn(
            (batch, q_len, heads * 6, 256),
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        * 0.25
    )
    table = torch.randperm(batch * blocks, device="cuda", generator=generator)
    table = table.to(torch.int32).reshape(batch, blocks)
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    return query, storage[:, 0], storage[:, 1], table, lengths


def reference(query, key, value, table, lengths, fmt):
    k_scale, v_scale = (1.0, 1.0) if fmt == "auto" else (0.75, 1.25)
    dtype = {
        "auto": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }[fmt]
    output = torch.empty_like(query, dtype=torch.float32)
    batch, q_len = query.shape[:2]
    for row in range(batch):
        length = int(lengths[row].item())
        for head in range(key.shape[2]):
            k = key[table[row].long(), :, head].reshape(-1, 256)[:length]
            v = value[table[row].long(), :, head].reshape(-1, 256)[:length]
            k = k.view(dtype).float() * k_scale
            v = v.view(dtype).float() * v_scale
            q = query[row, :, head * 6 : (head + 1) * 6].float()
            scores = torch.einsum("qhd,td->qht", q, k) * 0.0625
            positions = torch.arange(length, device="cuda")
            visible = length - q_len + torch.arange(q_len, device="cuda") + 1
            scores.masked_fill_(
                positions[None, None, :] >= visible[:, None, None], float("-inf")
            )
            output[row, :, head * 6 : (head + 1) * 6] = scores.softmax(-1) @ v
    return output


def measure(call, flush, iterations):
    for _ in range(3):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        graph.replay()
    timings = []
    for _ in range(iterations):
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
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=31)
    parser.add_argument("--contexts", default="8192,32768,131072")
    parser.add_argument("--heads", default="1,2,4")
    parser.add_argument("--formats", default="auto,fp8_e4m3,fp8_e5m2")
    parser.add_argument(
        "--routes", default="decode_xqa,decode_xqa_batch,decode_scalar,prefill_paged"
    )
    args = parser.parse_args()
    import flash_attn_v100 as fa
    from flash_attn_v100 import flash_attn_v100_cuda as extension

    assert torch.cuda.get_device_capability() == (7, 0)
    assert fa.flash_attn_decode_paged_xqa_available()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    rows = []
    for fmt in args.formats.split(","):
        scales = (1.0, 1.0) if fmt == "auto" else (0.75, 1.25)
        for heads in map(int, args.heads.split(",")):
            for length in map(int, args.contexts.split(",")):
                for route, batch, q_len in (
                    ("decode_xqa", 1, 1),
                    ("decode_xqa_batch", 2, 1),
                    ("decode_scalar", 1, 1),
                    ("prefill_paged", 1, 8),
                ):
                    if route not in args.routes.split(","):
                        continue
                    query, key, value, table, lengths = inputs(
                        fmt, heads, batch, length, q_len
                    )
                    output = torch.empty_like(query)
                    kwargs = dict(
                        kv_cache_dtype=fmt, k_scale=scales[0], v_scale=scales[1]
                    )
                    if route.startswith("decode"):
                        q3, o3 = query[:, 0], output[:, 0]
                        operation = (
                            fa.flash_attn_decode_paged
                            if route == "decode_scalar"
                            else fa.flash_attn_decode_paged_xqa
                        )
                        kwargs.update(
                            out=o3,
                            max_seq_len_hint=length,
                            workspace_seq_capacity_hint=table.shape[1] * key.shape[1],
                        )
                        if route != "decode_scalar":
                            kwargs.update(
                                partition_size_hint=64
                                if fmt == "fp8_e4m3" and batch == 1
                                else 256,
                                batch_context_routing=True,
                            )

                        call = partial(
                            operation, q3, key, value, table, lengths, **kwargs
                        )
                    else:
                        call = partial(
                            fa.flash_attn_prefill_paged,
                            query,
                            key,
                            value,
                            table,
                            lengths,
                            out=output,
                            **kwargs,
                        )

                    call()
                    expected = reference(query, key, value, table, lengths, fmt)
                    error = float((output.float() - expected).abs().max().item())
                    if not torch.isfinite(output).all() or error > 0.005:
                        raise RuntimeError(
                            f"{route}/{fmt}/H{heads}/L{length}: error={error}"
                        )
                    timings = measure(call, flush, args.iterations)
                    row = dict(
                        route=route,
                        fmt=fmt,
                        heads=heads,
                        batch=batch,
                        q_len=q_len,
                        context=length,
                        page=key.shape[1],
                        median_us=statistics.median(timings),
                        samples_us=timings,
                        max_reference_error=error,
                        output_sha256=digest(output),
                    )
                    rows.append(row)
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                        flush=True,
                    )
    result = {
        "scope": "direct operators, no backend/model admission claim",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "extension_path": extension.__file__,
        "extension_sha256": hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        "graph": True,
        "cold_l2_flush_bytes": flush.numel(),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
