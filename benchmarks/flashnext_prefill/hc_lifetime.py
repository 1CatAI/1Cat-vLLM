# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure HC temporary lifetimes with actual TP4 transport and packed weights.

Launch with torchrun on four idle V100s. No arithmetic changes are allowed:
every output must be bit-exact, including changed-input graph replay. Timing
includes unpacking, both projections, both gathers, mixing and assembly.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from collectives import measure
from hc_pipeline import sequential

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc
from vllm.models.qwen4_exp.nvidia.sm70_hcx import pack_down, pack_up


def peak(fn):
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    before = torch.accelerator.memory_allocated()
    result = fn()
    torch.accelerator.synchronize()
    growth = torch.accelerator.max_memory_allocated() - before
    del result
    return growth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    nccl = PyNcclCommunicator(dist.group.WORLD, rank)
    assert not nccl.disabled

    def gather(value, dim):
        assert dim == 0
        result = value.new_empty((4 * value.shape[0], *value.shape[1:]))
        nccl.all_gather(result, value)
        return result

    group = SimpleNamespace(
        rank_in_group=rank,
        device_communicator=SimpleNamespace(
            hc_ll_comm=SimpleNamespace(status={"enabled": True}, order=(0, 1, 2, 3))
        ),
        all_gather=gather,
    )
    torch.manual_seed(9683)
    down_weight = torch.randn(336, 10240, device="cuda", dtype=torch.half) * 0.002
    up_weight = torch.randn(10240, 320, device="cuda", dtype=torch.half) * 0.002
    down, up = pack_down(down_weight, rank), pack_up(up_weight, rank)
    del down_weight, up_weight
    report = {"cases": []}
    try:
        with patch("vllm.distributed.parallel_state.get_tp_group", lambda: group):
            for rows in (7344, 8976, 9216):
                torch.manual_seed(rows)
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.half) * 0.3

                def control(x=x):
                    local_down, local_up = hc._unpack_hc_storage(down, up)
                    return sequential(nccl, x, local_down, local_up, rank)

                def candidate(x=x):
                    return hc._sharded_hc_project(x, down, up)

                expected, actual = control(), candidate()
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
                del expected, actual
                row = {
                    "tokens": rows,
                    "control": measure(control),
                    "control_peak_bytes": peak(control),
                    "candidate": measure(candidate),
                    "candidate_peak_bytes": peak(candidate),
                    "repeat_control": measure(control),
                    "exact": True,
                }
                # A completed capture must retain valid storage after Python
                # references to intermediate tensors have been released.
                if rows == 7344:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = candidate()
                    for _ in range(3):
                        x.normal_()
                        graph.replay()
                        for got, want in zip(captured, control()):
                            torch.testing.assert_close(got, want, rtol=0, atol=0)
                    row["changed_input_graph_replays"] = 3
                    del graph, captured
                report["cases"].append(row)
                if rank == 0:
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
    finally:
        torch.accelerator.synchronize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
