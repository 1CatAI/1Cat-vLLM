# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual TP4 dense projections: normalized integer dots versus TurboMind."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import vllm._C as core
from benchmark_gguf_dp4a_expert import graph_time

import vllm
from vllm.model_executor.layers.quantization.gguf_dp4a_formats import (
    transcode_integer_dot,
)
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    GGUFPreparedProjection,
)
from vllm.transformers_utils.gguf_tensor_reader import (
    GGUFReader,
    dequantize,
    quant_size,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("tensor")
    parser.add_argument("output", type=Path)
    parser.add_argument("--axis", type=int, choices=[0, 1], default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--split", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    assert "site-packages" in vllm.__file__, vllm.__file__
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    reader = GGUFReader(args.model)
    tensor = next(t for t in reader.tensors if t.name == args.tensor)
    kind = int(tensor.tensor_type)
    block, size = quant_size(kind)
    raw = tensor.data
    if args.axis == 0:
        span = raw.shape[0] // 4
        raw = raw[args.rank * span : (args.rank + 1) * span].copy()
    else:
        span = raw.shape[1] // 4
        assert span % size == 0
        raw = raw[:, args.rank * span : (args.rank + 1) * span].copy()
    codec = transcode_integer_dot(raw, kind)
    n, k = codec.shape
    packed = [torch.from_numpy(t).cuda() for t in codec.packed()]
    control = GGUFPreparedProjection(
        torch.from_numpy(raw).cuda(), kind, torch.float16, True, 8
    )
    assert control.rejection_reason is None
    reference = torch.from_numpy(dequantize(raw, kind)).cuda()
    result = dict(
        version=vllm.__version__,
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        tensor=args.tensor,
        shape=[n, k],
        source_type=kind,
        tp_rank=args.rank,
        tp_size=4,
        axis=args.axis,
        cases=[],
    )
    for m in args.m:
        torch.manual_seed(20261005 + m)
        x = torch.randn((m, k), device="cuda", dtype=torch.float16)
        q8 = torch.empty((m, k // 32, 36), device="cuda", dtype=torch.uint8)
        out = torch.empty((m, n), device="cuda", dtype=torch.float16)
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        integer = q8[:, :, 4:].view(torch.int8).float()
        scale = q8[:, :, :2].contiguous().view(torch.float16).float()
        decoded = (integer * scale).reshape(m, k)
        expected = decoded @ reference.T
        if kind == 12:
            minimum = codec.dmin.astype("float32").repeat(8, axis=1)
            minimum *= codec.small_mins.astype("float32")
            original_sum = (
                q8[:, :, 2:4].contiguous().view(torch.float16).float().squeeze(-1)
            )
            delta = original_sum - decoded.reshape(m, -1, 32).sum(-1)
            expected -= delta @ torch.from_numpy(minimum).cuda().T
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
                        out, scratch, q8, *packed, kind, split, cooperative, False
                    )

                try:
                    operation()
                    torch.accelerator.synchronize()
                except RuntimeError as error:
                    if "resident block capacity" not in str(error):
                        raise
                    cases.append(
                        dict(split=split, cooperative=cooperative, rejected=str(error))
                    )
                    continue
                torch.testing.assert_close(
                    out.float(), expected, rtol=0.003, atol=0.003
                )
                samples = [graph_time(operation, args.iterations) for _ in range(4)]
                elapsed = statistics.median(samples)
                cases.append(
                    dict(
                        split=split,
                        cooperative=cooperative,
                        epoch_us=samples,
                        median_us=elapsed,
                        storage_bytes=sum(t.numel() * t.element_size() for t in packed),
                        source_gbps=sum(t.numel() * t.element_size() for t in packed)
                        / elapsed
                        / 1000,
                        kernel_count=1 if split == 1 or cooperative else 2,
                        official_q8_relative_l2=(
                            (out.float() - expected).norm() / expected.norm()
                        ).item(),
                    )
                )
        native_samples = [
            graph_time(lambda x=x: control(x), args.iterations) for _ in range(4)
        ]
        result["cases"].append(
            dict(
                m=m,
                candidate=cases,
                native_us=native_samples,
                native_median_us=statistics.median(native_samples),
            )
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["cases"][-1]), flush=True)


if __name__ == "__main__":
    main()
