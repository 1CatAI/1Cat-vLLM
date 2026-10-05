# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with torchrun --nproc-per-node 4; a whole installed SM70 wheel is required."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch
import torch.distributed as dist
import vllm._C as core

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.distributed.device_communicators.sm70_hc_ll import Sm70HcLLCommunicator
from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import _pack_hc_batch_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    cfg = cast(
        VllmConfig, SimpleNamespace(kernel_config=KernelConfig(hc_ll_shard=True))
    )
    with set_current_vllm_config(cfg):
        owner = Sm70HcLLCommunicator(
            dist.group.WORLD, torch.device("cuda", rank), "operator"
        )
    assert owner.status["enabled"], owner.status
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)[:2]
    packed = []
    for record in weights:
        full_down = torch.cat(
            (record["down"].float(), record["inj"].float(), torch.zeros(12, 10240))
        ).half()
        full_up = record["up"].half()
        packed.append(
            [
                _pack_hc_batch_weight(full_down, "down", None).cuda(),
                _pack_hc_batch_weight(full_up, "up", None).cuda(),
                _pack_hc_batch_weight(full_down, "down", owner.logical_rank).cuda(),
                _pack_hc_batch_weight(full_up, "up", owner.logical_rank).cuda(),
            ]
        )
    rows = []
    for m in (1, 5, 8, 20):
        torch.manual_seed(73)
        x = torch.randn(m, 10240, device="cuda", dtype=torch.float16) * 0.5
        partial = torch.empty((20, m, 352), device="cuda", dtype=torch.float32)
        lora = torch.empty((m, 320), device="cuda", dtype=torch.float16)
        reference = torch.empty((m, 2560), device="cuda", dtype=torch.float16)
        injection = torch.empty((m, 4), device="cuda", dtype=torch.float16)
        worst = [0.0, 0.0]
        graphs = []
        for pd, pu, sd, su in packed:
            if m >= 2:
                ops.sm70_qwen38_hc_replicated(
                    x, pd, pu, partial, lora, reference, injection
                )
            else:
                # The baseline operator requires at least two rows; use a
                # duplicate row solely for the numerical oracle at M1.
                xx = x.repeat(2, 1)
                pp = torch.empty((20, 2, 352), device="cuda", dtype=torch.float32)
                ll = torch.empty((2, 320), device="cuda", dtype=torch.float16)
                oo = torch.empty((2, 2560), device="cuda", dtype=torch.float16)
                ii = torch.empty((2, 4), device="cuda", dtype=torch.float16)
                ops.sm70_qwen38_hc_replicated(xx, pd, pu, pp, ll, oo, ii)
                reference.copy_(oo[:1])
                injection.copy_(ii[:1])
            for _ in range(4):
                candidate = owner.apply(x, sd, su)
            torch.accelerator.synchronize()
            assert candidate is not None
            for i, (actual, ref) in enumerate(zip(candidate, (reference, injection))):
                worst[i] = max(
                    worst[i],
                    float(
                        (actual.float() - ref.float()).abs().max()
                        / ref.float().abs().max().clamp_min(1e-6)
                    ),
                )
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                candidate = owner.apply(x, sd, su)
            for iteration in range(32):
                if rank == 0 and iteration % 7 == 0:
                    time.sleep(0.002)
                graph.replay()
            torch.accelerator.synchronize()
            dist.barrier()
            graphs.append(graph)
        assert worst[0] < 0.001 and worst[1] < 0.001, worst
        times = []
        for _ in range(3):
            dist.barrier()
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(200):
                for graph in graphs:
                    graph.replay()
            b.record()
            torch.accelerator.synchronize()
            times.append(a.elapsed_time(b) * 1000 / 200 / len(graphs))
        rows.append(dict(m=m, graph_us=times, relative_max=worst))
    records = [None] * 4
    dist.all_gather_object(records, rows)
    if rank == 0:
        args.output.write_text(
            json.dumps(
                dict(
                    rank_records=records,
                    capability=owner.status,
                    core_sha256=hashlib.sha256(
                        Path(core.__file__).read_bytes()
                    ).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
    owner.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
