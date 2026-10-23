# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2/TP4 graph comparison of lossless compact candidate communication."""

import argparse
import gc
import importlib.util
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_gather,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Standalone module for a research screen")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.module:
        spec = importlib.util.spec_from_file_location("candidate", args.module)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from vllm.model_executor.layers import sm70_topk_gather as module
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    size = int(os.environ["WORLD_SIZE"])
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=size))
    with set_current_vllm_config(config):
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=size)
    rank = dist.get_rank()
    points = []
    for rows, k in ((8, 64), (7, 20), (32, 64), (28, 20), (1, 20)):
        for dtype in (torch.float16, torch.float32):
            values = torch.randn(rows, k, device="cuda", dtype=dtype)
            # Exercise ties, infinities, signed zero and NaNs in the transport.
            values.view(-1)[:7] = torch.tensor(
                [0.0, -0.0, 1.0, 1.0, float("inf"), -float("inf"), float("nan")],
                device="cuda",
                dtype=dtype,
            )
            ids = (
                torch.arange(rows * k, device="cuda", dtype=torch.int64).view(rows, k)
                + rank * 65536
            )
            ids[0, 0] = -1

            def canonical(values=values, ids=ids):
                return tensor_model_parallel_all_gather(
                    values, dim=-1
                ), tensor_model_parallel_all_gather(ids, dim=-1)

            def candidate(values=values, ids=ids):
                result = module.gather_topk_pairs(values, ids, vocab_size=262144)
                assert result is not None
                return result

            expected = canonical()
            actual = candidate()
            torch.cuda.synchronize()
            assert torch.equal(
                expected[0].view(
                    torch.int16 if dtype == torch.float16 else torch.int32
                ),
                actual[0].view(torch.int16 if dtype == torch.float16 else torch.int32),
            )
            assert torch.equal(expected[1], actual[1])
            graphs = {}
            outputs = {}
            for name, fn in [("canonical", canonical), ("packed", candidate)]:
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                dist.barrier()
                with graph_capture(
                    device=torch.device("cuda", torch.cuda.current_device())
                ) as context:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=context.stream):
                        outputs[name] = fn()
                graphs[name] = g
            readings = []
            for order in [
                ["canonical", "packed"],
                ["packed", "canonical"],
                ["packed", "canonical"],
                ["canonical", "packed"],
            ] * 3:
                for name in order:
                    g = graphs[name]
                    for _ in range(10):
                        g.replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    a.record()
                    for _ in range(100):
                        g.replay()
                    b.record()
                    b.synchronize()
                    duration = torch.tensor(a.elapsed_time(b) * 10, device="cuda")
                    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                    readings.append(dict(name=name, us=duration.item()))
            assert torch.equal(
                outputs["canonical"][0].view(
                    torch.int16 if dtype == torch.float16 else torch.int32
                ),
                outputs["packed"][0].view(
                    torch.int16 if dtype == torch.float16 else torch.int32
                ),
            )
            assert torch.equal(outputs["canonical"][1], outputs["packed"][1])
            # Keep the existing global TopK implementation and tie cutoff.
            for out in outputs.values():
                out[0].nan_to_num_(nan=-float("inf"))
            for name in outputs:
                top, positions = torch.topk(outputs[name][0], k=k, dim=-1)
                outputs[name] = (top, outputs[name][1].gather(-1, positions))
            assert torch.equal(outputs["canonical"][0], outputs["packed"][0])
            assert torch.equal(outputs["canonical"][1], outputs["packed"][1])
            point = dict(
                rows=rows,
                k=k,
                dtype=str(dtype),
                tp=size,
                readings=readings,
                medians={
                    name: statistics.median(
                        [r["us"] for r in readings if r["name"] == name]
                    )
                    for name in graphs
                },
            )
            points.append(point)
            if rank == 0:
                Path(args.output).write_text(json.dumps(points, indent=2))
                print(
                    point["rows"],
                    point["k"],
                    point["dtype"],
                    point["medians"],
                    flush=True,
                )
    # Captured NCCL user objects must release their communicator references.
    del g, graphs
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
