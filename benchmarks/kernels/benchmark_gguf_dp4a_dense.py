# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual TP4 dense projections: normalized integer dots versus TurboMind."""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np
import torch
import vllm._C as core

import vllm
from vllm.model_executor.layers.quantization.gguf_dp4a_formats import (
    pack_mixed_u4_pair,
    transcode_integer_dot,
)
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    GGUFPreparedProjection,
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)
from vllm.transformers_utils.gguf_tensor_reader import (
    GGUFReader,
    dequantize,
    quant_size,
)


def cold_graph_time(operation, eviction, iterations):
    """Time graph event boundaries after eviction, excluding the eviction cost."""
    for _ in range(3):
        eviction.add_(1)
        operation()
    pairs = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in range(16)
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for start, end in pairs:
            eviction.add_(1)
            start.record()
            operation()
            end.record()
    samples = []
    for _ in range(iterations):
        graph.replay()
        pairs[-1][1].synchronize()
        samples.extend(start.elapsed_time(end) * 1000 for start, end in pairs)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("tensor")
    parser.add_argument("output", type=Path)
    parser.add_argument("--axis", type=int, choices=[0, 1], default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--join", nargs="+", default=[])
    parser.add_argument("--activated", action="store_true")
    parser.add_argument("--mixed-u4-pair", action="store_true")
    parser.add_argument("--fold-q6-base", action="store_true")
    parser.add_argument("--raw-source", action="store_true")
    parser.add_argument("--row-major", action="store_true")
    parser.add_argument("--m", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--split", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    assert "site-packages" in vllm.__file__, vllm.__file__
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    eviction = torch.zeros(32 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    reader = GGUFReader(args.model)
    tensor = next(t for t in reader.tensors if t.name == args.tensor)
    kind = int(tensor.tensor_type)
    block, size = quant_size(kind)

    def shard(tensor):
        raw = tensor.data
        if args.axis == 0:
            span = raw.shape[0] // 4
            return raw[args.rank * span : (args.rank + 1) * span].copy()
        span = raw.shape[1] // 4
        assert span % size == 0
        return raw[:, args.rank * span : (args.rank + 1) * span].copy()

    tensors = [tensor] + [
        next(t for t in reader.tensors if t.name == name) for name in args.join
    ]
    types = [int(t.tensor_type) for t in tensors]
    shards = [shard(t) for t in tensors]
    paired_type = 0
    if args.mixed_u4_pair:
        assert args.activated and not args.raw_source and not args.row_major
        assert types in ([12, 23], [23, 12])
        paired_type = types[1]
        codecs = [transcode_integer_dot(r, t) for r, t in zip(shards, types)]
        n, k = codecs[0].shape
        n *= 2
        codec = codecs[0]
        packed = [torch.from_numpy(a).cuda() for a in pack_mixed_u4_pair(*codecs)]
        projections = prepare_gguf_projections(
            [(torch.from_numpy(r).cuda(), t) for r, t in zip(shards, types)],
            torch.float16,
            True,
            8,
        )

        def control(x):
            return apply_prepared_gguf_projections(x, projections)

        reference = torch.cat(
            [torch.from_numpy(dequantize(r, t)).cuda() for r, t in zip(shards, types)]
        )
    else:
        assert all(t == kind for t in types)
        raw = np.concatenate(shards, axis=0)
        codec = transcode_integer_dot(raw, kind)
        n, k = codec.shape
        packed = [
            torch.from_numpy(t).cuda()
            for t in (
                codec.packed(fold_q6_base=True) if args.fold_q6_base else codec.packed()
            )
        ]
        control = GGUFPreparedProjection(
            torch.from_numpy(raw).cuda(), kind, torch.float16, True, 8
        )
        assert control.rejection_reason is None
        reference = torch.from_numpy(dequantize(raw, kind)).cuda()
    if args.row_major:
        assert not args.raw_source
        packed = [torch.from_numpy(t).cuda() for t in codec.row_storage()]
    if args.raw_source:
        assert kind == 14
        packed = [
            torch.from_numpy(raw).cuda(),
            torch.empty(0, device="cuda", dtype=torch.float16),
            torch.empty(0, device="cuda", dtype=torch.int8),
            torch.empty(0, device="cuda", dtype=torch.float16),
            torch.empty(0, device="cuda", dtype=torch.int8),
        ]
    result = dict(
        version=vllm.__version__,
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        accumulation="fp32",
        activation_encoding="q8_1_group32_original_half_sum",
        calls_per_graph=16,
        cache_policy="cold_L2_32MiB_read_write_eviction_before_each_projection",
        tensor=[args.tensor, *args.join],
        activated=args.activated,
        storage=(
            "row_integer_fp32_coefficients"
            if args.row_major
            else "raw_Q6_K"
            if args.raw_source
            else "N32_integer_packets"
        ),
        shape=[n, k],
        source_type=kind,
        source_types=types,
        fold_q6_base=args.fold_q6_base,
        tp_rank=args.rank,
        tp_size=4,
        axis=args.axis,
        cases=[],
    )
    for m in args.m:
        torch.manual_seed(20261005 + m)
        x = torch.randn((m, k), device="cuda", dtype=torch.float16)
        q8 = torch.empty((m, k // 32, 36), device="cuda", dtype=torch.uint8)
        width = n // 2 if args.activated else n
        out = torch.empty((m, width), device="cuda", dtype=torch.float16)
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        integer = q8[:, :, 4:].view(torch.int8).float()
        scale = q8[:, :, :2].contiguous().view(torch.float16).float()
        decoded = (integer * scale).reshape(m, k)
        expected = decoded @ reference.T
        if kind == 12 or paired_type:

            def minima(codec):
                if codec.source_type != 12:
                    return np.zeros((codec.shape[0], k // 32), dtype=np.float32)
                result = codec.dmin.astype("float32").repeat(8, axis=1)
                return result * codec.small_mins.astype("float32")

            minimum = (
                np.concatenate([minima(c) for c in codecs])
                if paired_type
                else minima(codec)
            )
            original_sum = (
                q8[:, :, 2:4].contiguous().view(torch.float16).float().squeeze(-1)
            )
            delta = original_sum - decoded.reshape(m, -1, 32).sum(-1)
            expected -= delta @ torch.from_numpy(minimum).cuda().T
        fp16_expected = x.float() @ reference.T
        if args.activated:

            def finish(values, width=width):
                values = values.half()
                return (
                    torch.nn.functional.silu(values[:, :width]) * values[:, width:]
                ).float()

            expected, fp16_expected = finish(expected), finish(fp16_expected)
        control_out = torch.empty_like(out)

        def control_operation(x=x, control_out=control_out):
            result = control(x)
            if args.activated:
                torch.ops._C.silu_and_mul(control_out, result)
                return control_out
            return result

        event_boundary_us = cold_graph_time(lambda: None, eviction, args.iterations)
        cases = []
        for split in args.split:
            scratch = torch.empty((split, m, n), device="cuda", dtype=torch.float32)
            for cooperative in (False, True):

                def operation(
                    split=split,
                    cooperative=cooperative,
                    scratch=scratch,
                    out=out,
                    q8=q8,
                ):
                    torch.ops._C.gguf_dp4a_dense_sm70_out(
                        out,
                        scratch,
                        q8,
                        *packed,
                        kind,
                        split,
                        cooperative,
                        args.activated,
                        *([paired_type] if paired_type else []),
                    )

                try:
                    operation()
                    torch.accelerator.synchronize()
                except RuntimeError as error:
                    if not any(
                        reason in str(error)
                        for reason in (
                            "resident block capacity",
                            "Raw integer dense cooperative reduction unavailable",
                        )
                    ):
                        raise
                    cases.append(
                        dict(split=split, cooperative=cooperative, rejected=str(error))
                    )
                    continue
                torch.testing.assert_close(
                    out.float(), expected, rtol=0.003, atol=0.003
                )
                samples, controls, combined_samples, clocks = [], [], [], []

                def combined(operation=operation, q8=q8, x=x):
                    torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
                    operation()

                operations = {
                    "candidate": operation,
                    "control": control_operation,
                    "combined": combined,
                }
                timing = {
                    "candidate": samples,
                    "control": controls,
                    "combined": combined_samples,
                }
                for epoch in range(4):
                    for name in list(operations)[:: 1 if epoch % 2 == 0 else -1]:
                        timing[name].append(
                            cold_graph_time(operations[name], eviction, args.iterations)
                        )
                    clocks.append(
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=clocks.sm,clocks.mem",
                                "--format=csv,noheader",
                            ],
                            text=True,
                        ).strip()
                    )
                elapsed = statistics.median(samples)
                cases.append(
                    dict(
                        split=split,
                        cooperative=cooperative,
                        epoch_us=samples,
                        median_us=elapsed,
                        control_us=controls,
                        control_median_us=statistics.median(controls),
                        encode_and_projection_us=combined_samples,
                        encode_and_projection_median_us=statistics.median(
                            combined_samples
                        ),
                        clocks=clocks,
                        storage_bytes=sum(t.numel() * t.element_size() for t in packed),
                        source_gbps=sum(t.numel() * t.element_size() for t in packed)
                        / elapsed
                        / 1000,
                        kernel_count=(
                            1
                            if cooperative or (split == 1 and not args.activated)
                            else 2
                        ),
                        official_q8_relative_l2=(
                            (out.float() - expected).norm() / expected.norm()
                        ).item(),
                    )
                )
        native_samples = [
            value for case in cases for value in case.get("control_us", [])
        ]
        result["cases"].append(
            dict(
                m=m,
                candidate=cases,
                native_us=native_samples,
                native_median_us=statistics.median(native_samples),
                event_boundary_median_us=event_boundary_us,
                timing_note=(
                    "raw event-bounded latency; "
                    "matched deltas cancel common boundary cost"
                ),
                activation_relative_l2=(
                    (expected - fp16_expected).norm() / fp16_expected.norm()
                ).item(),
                activation_max_absolute_error=(expected - fp16_expected)
                .abs()
                .max()
                .item(),
            )
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["cases"][-1]), flush=True)


if __name__ == "__main__":
    main()
