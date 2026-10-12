# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost of installed shared KV algorithms, with fixed encoded inputs.

Run controls/candidates in separate processes, alternating their order. This
measures graph GPU service for an operator; it is not end-to-end throughput.
"""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import FP8_E4M3, INT8
from vllm.v1.attention.ops.sm70_grouped_long import (
    builtin_codec_attention,
    run_six_head_groups,
)


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.contiguous().cpu().view(torch.uint8).numpy()
    ).hexdigest()


def measure(function, replays: int, warmup_replays: int):
    for _ in range(20):
        function()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    for _ in range(warmup_replays):
        graph.replay()
    samples = []
    for _ in range(9):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / replays)
    return dict(samples_ms=samples, median_ms=statistics.median(samples))


def main():
    import flash_attn_v100 as package
    from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda as native

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--warmup-replays", type=int, default=2000)
    args = parser.parse_args()
    if (
        torch.cuda.get_device_capability() != (7, 0)
        or min(args.replays, args.warmup_replays) <= 0
    ):
        raise ValueError("Requires SM70 and positive replay count")
    if args.output.exists():
        raise ValueError("Preserve existing evidence; choose a new output")
    binary = Path(native.__file__)
    result = dict(
        scope="Installed operator CUDA Graph GPU service; not engine acceptance",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        protocol=dict(
            ordinary_warmups=20,
            graph_warmup_replays=args.warmup_replays,
            graph_replays_per_sample=args.replays,
            samples=9,
        ),
        flash_binary=dict(
            name=binary.name, sha256=hashlib.sha256(binary.read_bytes()).hexdigest()
        ),
        cases=[],
    )
    for codec in (FP8_E4M3, INT8):
        for length in (4096, 32768):
            torch.manual_seed(20261012)
            page, heads, dimension = 2048, 1, 256
            pages = (length + page - 1) // page + 2
            carrier = torch.zeros(
                (pages, 2, page, heads, dimension + (4 if codec.per_token_head else 0)),
                device="cuda",
                dtype=codec.storage_dtype,
            )
            view = KVCacheView.bind(carrier, codec, dimension)
            for payload, scale in (
                (view.key, view.key_scale),
                (view.value, view.value_scale),
            ):
                values = torch.randn(payload.shape, device="cuda", dtype=torch.float16)
                if codec.per_token_head:
                    raw, scales = codec.encode(values)
                    payload.copy_(raw)
                    scale.copy_(scales)
                else:
                    payload.copy_(
                        values.to(codec.element_dtype).view(codec.storage_dtype)
                    )
            table = torch.randperm(pages, device="cuda").int()[None, :]
            for algorithm, rows in (("xqa", 1), ("grouped", 8), ("grouped_long", 8)):
                query = torch.randn(
                    (rows, 6, dimension), device="cuda", dtype=torch.float16
                )
                output = torch.empty_like(query)
                lengths = torch.arange(
                    length - rows + 1, length + 1, device="cuda", dtype=torch.int32
                )
                if algorithm == "grouped_long":
                    operator = builtin_codec_attention("sm70_grouped_long", codec)
                    if operator is None:
                        raise RuntimeError(
                            "Normal FA2 artifact is missing its codec binding"
                        )
                    partial_output = torch.empty(
                        (80, 8, 6, 256), device="cuda", dtype=torch.float32
                    )
                    stats = torch.empty(
                        (80, 8, 6, 2), device="cuda", dtype=torch.float32
                    )

                    launch = partial(
                        run_six_head_groups,
                        operator,
                        query,
                        view.key,
                        view.value,
                        output,
                        table,
                        lengths,
                        partial_output,
                        stats,
                        0.0625,
                        1.0,
                        1.0,
                        codec=codec.native_id,
                        key_scales=view.key_scale,
                        value_scales=view.value_scale,
                    )
                else:
                    operation = (
                        package.flash_attn_decode_paged_xqa
                        if algorithm == "xqa"
                        else package.flash_attn_grouped_fp32_paged
                    )
                    options = (
                        dict(
                            max_seq_len_hint=length,
                            workspace_seq_capacity_hint=length,
                            partition_size_hint=256,
                        )
                        if algorithm == "xqa"
                        else {}
                    )

                    launch = partial(
                        operation,
                        query,
                        view.key,
                        view.value,
                        table,
                        lengths,
                        out=output,
                        kv_cache_dtype=codec.name,
                        softmax_scale=0.0625,
                        **view.scale_kwargs,
                        **options,
                    )
                cost = measure(launch, args.replays, args.warmup_replays)
                assert torch.isfinite(output).all()
                result["cases"].append(
                    dict(
                        codec=codec.name,
                        algorithm=algorithm,
                        context_tokens=length,
                        rows=rows,
                        input_sha256=digest(query),
                        output_sha256=digest(output),
                        **cost,
                    )
                )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [
                dict(
                    codec=c["codec"],
                    algorithm=c["algorithm"],
                    context=c["context_tokens"],
                    median_ms=c["median_ms"],
                )
                for c in result["cases"]
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
