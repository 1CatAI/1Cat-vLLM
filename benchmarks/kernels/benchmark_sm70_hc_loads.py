# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-wheel TP4 HC load-policy ABBA, bitwise replay and generation checks."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import vllm._C as core
from benchmark_sm70_hc_ll import tag_wrap_case

from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.distributed.device_communicators.sm70_hc_ll import Sm70HcLLCommunicator
from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import _pack_hc_batch_weight


def main():
    parser = argparse.ArgumentParser(
        description="Same-wheel TP4 HC load-policy ABBA and replay checks"
    )
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    cfg = SimpleNamespace(kernel_config=KernelConfig(hc_ll_shard=True))
    with set_current_vllm_config(cfg):
        owner = Sm70HcLLCommunicator(
            dist.group.WORLD, torch.device("cuda", rank), "load_policy"
        )
    assert owner.status["enabled"] and owner.status["optimized_loads"], owner.status
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    packed = []
    for record in weights:
        d = torch.cat(
            (record["down"].float(), record["inj"].float(), torch.zeros(12, 10240))
        ).half()
        u = record["up"].half()
        packed.append(
            (
                _pack_hc_batch_weight(d, "down", owner.logical_rank).cuda(),
                _pack_hc_batch_weight(u, "up", owner.logical_rank).cuda(),
            )
        )
    rows = []
    for m in (1, 2, 4, 5, 8, 10, 20):
        torch.manual_seed(20261006 + m)
        x = (torch.randn(m, 10240, device="cuda") * 0.5).half()
        for w in packed:
            owner.optimized_loads = False
            reference = owner.apply(x, *w)
            torch.cuda.synchronize()
            dist.barrier()
            owner.optimized_loads = True
            actual = owner.apply(x, *w)
            torch.cuda.synchronize()
            for a, b in zip(actual, reference):
                torch.testing.assert_close(
                    a.view(torch.int16), b.view(torch.int16), rtol=0, atol=0
                )
        if m not in (5, 20):
            continue
        graphs = {}
        for arm, enabled in (("A", False), ("B", True)):
            owner.optimized_loads = enabled
            for w in packed:
                owner.apply(x, *w)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    for w in packed:
                        captured = owner.apply(x, *w)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            graphs[arm] = (graph, captured)
        # Changed-input replay checks that captured prologues use live inputs.
        x.add_(0.125)
        for arm, enabled in (("A", False), ("B", True)):
            dist.barrier()
            graphs[arm][0].replay()
            torch.cuda.synchronize()
            owner.optimized_loads = enabled
            expected = owner.apply(x, *packed[-1])
            torch.cuda.synchronize()
            for a, b in zip(graphs[arm][1], expected):
                torch.testing.assert_close(
                    a.view(torch.int16), b.view(torch.int16), rtol=0, atol=0
                )
        samples = {arm: [] for arm in graphs}
        for arm in ("A", "B", "B", "A"):
            g = graphs[arm][0]
            for _ in range(3):
                dist.barrier()
                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(40):
                    g.replay()
                b.record()
                b.synchronize()
                samples[arm].append(a.elapsed_time(b) * 1000 / 40 / 8 / len(packed))
        rows.append(dict(m=m, samples_us=samples, bitwise_equal=True))
        print(
            rank, m, {k: statistics.median(v) for k, v in samples.items()}, flush=True
        )
        del graphs
    owner.optimized_loads = True
    d = torch.cat(
        (weights[0]["down"].float(), weights[0]["inj"].float(), torch.zeros(12, 10240))
    ).half()
    full = (
        _pack_hc_batch_weight(d, "down", None).cuda(),
        _pack_hc_batch_weight(weights[0]["up"].half(), "up", None).cuda(),
        *packed[0],
    )
    wrap = tag_wrap_case(owner, full, rank)
    assert all(max(r["relative_max"]) < 0.001 for r in wrap), wrap
    records = [None] * 4
    dist.all_gather_object(records, rows)
    if rank == 0:
        args.output.write_text(
            json.dumps(
                dict(
                    scope="packaged same-wheel projection ABBA; not model latency",
                    core_sha256=hashlib.sha256(
                        Path(core.__file__).read_bytes()
                    ).hexdigest(),
                    capability=owner.status,
                    rank_records=records,
                    generation_wrap=wrap,
                    bitwise_batches=[1, 2, 4, 5, 8, 10, 20],
                    pairs=len(packed),
                ),
                indent=2,
            )
            + "\n"
        )
    owner.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
