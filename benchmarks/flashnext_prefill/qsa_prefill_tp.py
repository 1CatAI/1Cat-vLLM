# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 paged-selector boundary/capture checks and complete-chain timings.

The selector uses the serving PyNccl transport. All ranks compare output,
causal/request isolation, caller-owned output and changed-input graph replay.
This synthetic screen does not qualify model outputs or cache commit paths.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from collectives import measure

from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.config.sm70_sparse import Sm70SparseConfig
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_select_paged_tokens


class Group:
    world_size = 4

    def __init__(self, rank):
        self.rank_in_group = rank
        self.nccl = PyNcclCommunicator(dist.group.WORLD, rank)
        self.calls = 0

    def all_gather(self, tensor, dim):
        assert dim == 0
        self.calls += 1
        out = tensor.new_empty((4 * tensor.shape[0], *tensor.shape[1:]))
        self.nccl.all_gather(out, tensor)
        return out


def selection_error(actual, expected, positions):
    a, e = actual.sort(1).values, expected.sort(1).values
    locations = torch.searchsorted(e, a).clamp_max(e.shape[1] - 1)
    replaced = (e.gather(1, locations) != a).sum(1)
    positive = a >= 0
    valid = bool(((a <= positions[:, None]) | ~positive).all())
    distinct = bool(((a[:, 1:] > a[:, :-1]) | (a[:, 1:] == -1)).all())
    same_padding = torch.equal((a == -1).sum(1), (e == -1).sum(1))
    total = int(replaced.sum())
    maximum = int(replaced.max()) if replaced.numel() else 0
    return {
        "exact": torch.equal(actual, expected),
        "replaced_tokens": total,
        "max_replacements_per_row": maximum,
        "valid_distinct_causal": valid and distinct and same_padding,
        "passed": valid
        and distinct
        and same_padding
        and maximum <= 4
        and total * 100000 <= actual.numel(),
    }


def run_case(group, rows, columns, *, mixed=False, graph=False):
    torch.manual_seed(9683 + rows + columns)
    page = 64
    pages = (columns + page - 1) // page
    cache = (
        torch.randn(
            pages * (2 if mixed else 1), page, 1, 128, device="cuda", dtype=torch.half
        )
        * 0.125
    )
    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.half) * 0.125
    table = torch.arange(cache.shape[0], device="cuda", dtype=torch.int32).view(
        -1, pages
    )
    # Retain real paged gathers and invalid-page masking.
    table[:, 11::31] = -1
    requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
    lengths = torch.full(
        (table.shape[0],), columns * 4, device="cuda", dtype=torch.int32
    )
    positions = (
        columns * 4 - rows + torch.arange(rows, device="cuda", dtype=torch.int32)
    )
    starts = None
    if mixed:
        requests[5:] = 1
        positions[5:] -= 5
        starts = torch.tensor([0, 5, rows], dtype=torch.int32)
    expected_out = torch.empty(rows, 2051, device="cuda", dtype=torch.int32)
    storage = torch.full((rows * 2051 + 2,), -77, device="cuda", dtype=torch.int32)
    out = storage[1:-1].view(rows, 2051)

    def run(tp, output):
        return qsa_select_paged_tokens(
            q,
            cache,
            table,
            requests,
            positions,
            lengths,
            2048,
            4,
            output,
            query_start_loc_cpu=starts,
            max_seq_len=columns * 4,
            tp_group=tp,
        )

    expected = run(None, expected_out)
    group.calls = 0
    actual = run(group, out)
    expected_calls = int(not mixed and 4096 <= rows <= 9216)
    route = group.calls == expected_calls
    errors = selection_error(actual, expected, positions)
    errors["guard_exact"] = bool((storage[[0, -1]] == -77).all())
    errors["output_identity"] = actual.data_ptr() == out.data_ptr()
    errors["route_passed"] = route
    errors["all_gather_calls"] = group.calls
    errors["passed"] &= route and errors["guard_exact"] and errors["output_identity"]
    row = {"rows": rows, "columns": columns, "mixed": mixed, "ranks": [None] * 4}
    dist.all_gather_object(row["ranks"], errors)
    if rows in (7344, 7345, 9216) and not mixed:
        row["replicated"] = measure(lambda: run(None, expected_out))
        row["partitioned"] = measure(lambda: run(group, out))
        row["replicated_repeat"] = measure(lambda: run(None, expected_out))
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        group.calls = 0
        with torch.cuda.graph(captured, stream=stream):
            run(group, out)
        torch.cuda.current_stream().wait_stream(stream)
        assert group.calls == 0
        q.mul_(-0.5)
        cache.add_(0.03125)
        expected = run(None, expected_out)
        captured.replay()
        graph_errors = selection_error(out, expected, positions)
        row["graph_ranks"] = [None] * 4
        dist.all_gather_object(row["graph_ranks"], graph_errors)
    row["passed"] = all(x["passed"] for x in row["ranks"] + row.get("graph_ranks", []))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    group = Group(rank)
    policy = Sm70SparseConfig(qsa_score_tile_mb=32)
    policy.resolve()
    config = SimpleNamespace(kernel_config=KernelConfig(sm70_sparse=policy))
    report = {"cases": []}
    try:
        with set_current_vllm_config(config):
            for rows, columns in [
                (0, 1024),
                (5, 1024),
                (4095, 2048),
                (4096, 2048),
                (4097, 2048),
                (7344, 1836),
                (7345, 8672),
                (9216, 8672),
                (9217, 8672),
                (4096, 65536),
            ]:
                row = run_case(group, rows, columns, graph=rows == 4097)
                report["cases"].append(row)
                if rank == 0:
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
            report["cases"].append(run_case(group, 4101, 2048, mixed=True))
            if rank == 0:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        torch.accelerator.synchronize()
        group.nccl.destroy()
        dist.destroy_process_group()
    raise SystemExit(0 if all(r["passed"] for r in report["cases"]) else 1)


if __name__ == "__main__":
    main()
