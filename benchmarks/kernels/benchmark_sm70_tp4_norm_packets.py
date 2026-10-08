# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare shipped TP4 norm packets with the shipped ordered reference.

Run with torchrun on four exclusively owned SM70 devices. Both paths use the
installed ordinary extension; there is no research kernel or library overlay.
Check actual FP16 model norm weights, their FP32 equivalent, changed graph
inputs and alternating metadata generations before timing a cold-L2 burst.
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--burst", type=int, default=64)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(123 + rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=90)
    )
    ca = CustomAllreduce(
        dist.new_group(backend="gloo"), torch.device("cuda", rank), max_size=1048576
    )
    assert not ca.disabled and ca.fully_connected
    with safe_open(args.model / "model.safetensors", framework="pt") as model:
        weight = (
            model.get_tensor(
                "model.language_model.layers.0.post_attention_layernorm.weight"
            )
            .half()
            .cuda()
        )
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
    residual = torch.randn(8, 5120, device="cuda", dtype=torch.float32)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    reference = torch.ops._C_custom_ar.sm70_tp4_all_reduce_gemma_rms_norm_reference
    records = []
    for norm_weight in (weight, weight.float()):
        outputs = [(torch.empty_like(x), torch.empty_like(residual)) for _ in range(2)]

        def run(arm, norm_weight=norm_weight, outputs=outputs):
            capturing = ca._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
            operation = (
                reference if arm == 0 else ops.sm70_tp4_all_reduce_gemma_rms_norm
            )
            operation(
                ca._ptr,
                x,
                residual,
                norm_weight,
                *outputs[arm],
                0 if capturing else ca.buffer_ptrs[rank],
                0 if capturing else ca.max_size,
                1e-6,
            )

        graphs = []
        for arm in range(2):
            run(arm)
            graph = torch.cuda.CUDAGraph()
            begin = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            with ca.capture(), torch.cuda.graph(graph):
                eviction.fill_(1)
                begin.record()
                for _ in range(args.burst):
                    run(arm)
                end.record()
            graphs.append((graph, begin, end))
        for amplitude in (0.01, 0.125, 1.0, 4.0):
            x.normal_(0, amplitude)
            residual.normal_(0, amplitude)
            for _ in range(4):
                for graph, _, _ in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                for old, new in zip(*outputs):
                    bits = torch.int16 if old.element_size() == 2 else torch.int32
                    assert torch.isfinite(new).all()
                    assert torch.equal(old.view(bits), new.view(bits))
        samples = [[], []]
        for iteration in range(args.iters + 20):
            for arm in (iteration % 2, 1 - iteration % 2):
                graph, begin, end = graphs[arm]
                graph.replay()
                end.synchronize()
                if iteration >= 20:
                    samples[arm].append(begin.elapsed_time(end) * 1000 / args.burst)
        records.append(
            {
                "dtype": str(norm_weight.dtype),
                "bitwise": True,
                "samples_us": samples,
                "burst": args.burst,
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"rank{rank}.json").write_text(json.dumps(records, indent=2))
    print({"rank": rank, "bitwise_fp16_and_fp32_weights": True}, flush=True)
    ca.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
