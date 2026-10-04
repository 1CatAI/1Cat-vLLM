# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GGUF HC operator comparison, including admitted TP4 ring transport."""

import argparse
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.distributed import parallel_state
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.sm70_ring import Sm70RingCommunicator
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu
from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import (
    _pack_hc_batch_weight,
    _qwen38_sm70_fp16_fused_hc,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader


def timing(fn):
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(10):
        graph.replay()
    samples = []
    for _ in range(5):
        dist.barrier()
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 10)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicated", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    assert int(os.environ["WORLD_SIZE"]) == 4
    assert torch.cuda.get_device_capability() == (7, 0)
    assert ops.supports_sm70_qwen38_hc_local()
    if args.replicated:
        assert ops.supports_sm70_qwen38_hc_replicated()
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dist.init_process_group("gloo")
    ring = Sm70RingCommunicator(
        dist.group.WORLD, torch.device(f"cuda:{rank}"), "hc-local", True
    )
    assert ring.status["enabled"], ring.status
    nccl = PyNcclCommunicator(dist.group.WORLD, device=rank)
    assert nccl.available and not nccl.disabled

    def all_reduce(x):
        output = ring.all_reduce(x)
        return output if output is not None else nccl.all_reduce(x)

    # Exercise the shipped HC op with the same collective interface as the
    # model TP group. The benchmark does not construct or load a full model.
    group = SimpleNamespace(
        world_size=4,
        rank_in_group=rank,
        device_communicator=SimpleNamespace(ca_comm=None),
        all_reduce=all_reduce,
    )
    parallel_state.get_tp_group = lambda: group
    reader = GGUFReader(str(args.model))
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    def weight(name):
        tensor = tensors[name]
        assert int(tensor.tensor_type) == 30
        return (
            torch.from_numpy(tensor.data.view(np.uint16).copy())
            .view(torch.bfloat16)
            .half()
            .cuda()
        )

    weights = []
    for layer in (0, 12, 24, 47):
        for branch in ("attn", "ffn"):
            prefix = f"blk.{layer}.hc_{branch}_"
            down = weight(prefix + "down.weight")
            injection = weight(prefix + "inject.weight")
            down = torch.cat((down, injection, down.new_zeros(12, 10240)))
            up = weight(prefix + "up.weight")
            weights.append(
                (
                    down,
                    up,
                    _pack_hc_batch_weight(
                        down, "down", None if args.replicated else rank
                    ),
                    _pack_hc_batch_weight(up, "up", None if args.replicated else rank),
                )
            )
    results = []
    try:
        for m in (5, 10):
            torch.manual_seed(20261004 + m)
            x = (torch.randn(m, 10240, device="cuda") * 0.125).half()
            worst_abs = worst_l2 = 0.0
            for down, up, packed_down, packed_up in weights:
                d = torch.nn.functional.linear(x, down)
                lora = hc_silu(d[:, :320], 4)
                reference = hc_gate_mix(x, torch.nn.functional.linear(lora, up), 4)
                actual, injection = _qwen38_sm70_fp16_fused_hc(
                    x, down, up, packed_down, packed_up, concurrent_batch=True
                )
                error = actual.float() - reference.float()
                worst_abs = max(worst_abs, error.abs().max().item())
                worst_l2 = max(
                    worst_l2, (error.norm() / reference.float().norm()).item()
                )
                torch.testing.assert_close(actual, reference, rtol=0.003, atol=0.003)
                torch.testing.assert_close(
                    injection, d[:, 320:324], rtol=0.003, atol=0.003
                )

            def old_path(x=x):
                for down, up, _, _ in weights:
                    d = torch.nn.functional.linear(x, down)
                    hc_gate_mix(
                        x, torch.nn.functional.linear(hc_silu(d[:, :320], 4), up), 4
                    )

            def new_path(x=x):
                for down, up, pd, pu in weights:
                    _qwen38_sm70_fp16_fused_hc(
                        x, down, up, pd, pu, concurrent_batch=True
                    )

            old_us = timing(old_path)
            new_us = timing(new_path)
            result = {
                "m": m,
                "pairs": len(weights),
                "old_chain_us": old_us,
                "new_chain_us": new_us,
                "max_abs_error": worst_abs,
                "relative_l2": worst_l2,
                "estimated_96_pairs_saved_us": (old_us - new_us) * 96 / len(weights),
                "replicated": args.replicated,
                "scope": (
                    "TP4 operator chain with admitted transport policy; not model round"
                ),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        args.output.with_name(args.output.stem + f".rank{rank}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
    finally:
        torch.accelerator.synchronize()
        dist.barrier()
        ring.close()
        nccl.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
