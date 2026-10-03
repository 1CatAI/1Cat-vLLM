# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare ordinary/cooperative graph dispatch with identical kernel arguments."""

import argparse
import fcntl
import importlib.util
import json
import os
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "sm70_hc_dataflow_research", args.module
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {"nodes": 1000, "repeats": 5, "cases": []}
    # Keep the raw descriptor until process exit, including CUDA teardown.
    lock_fd = os.open("/tmp/gpu0-3.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    with torch.cuda.device(0):
        for shared in (0, 40992):
            for grid in (1, 80, 125):
                counter = torch.zeros(grid, dtype=torch.int32, device="cuda:0")
                graphs = {}
                for cooperative in (False, True):
                    module.probe(counter, cooperative, grid, shared)
                    torch.cuda.synchronize(0)
                    graph = torch.cuda.CUDAGraph()
                    stream = torch.cuda.Stream(device=0)
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(result["nodes"]):
                            module.probe(counter, cooperative, grid, shared)
                    graphs[cooperative] = graph
                times = {False: [], True: []}
                for repeat in range(result["repeats"]):
                    for mode in (False, True) if repeat % 2 == 0 else (True, False):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[mode].replay()
                        end.record()
                        end.synchronize()
                        times[mode].append(
                            start.elapsed_time(end) * 1000 / result["nodes"]
                        )
                assert (
                    counter.cpu() == 2 + 2 * result["nodes"] * result["repeats"]
                ).all()
                case = {
                    "grid": grid,
                    "threads": 128,
                    "shared_bytes": shared,
                    "ordinary_us": times[False],
                    "cooperative_us": times[True],
                    "ordinary_median_us": statistics.median(times[False]),
                    "cooperative_median_us": statistics.median(times[True]),
                }
                result["cases"].append(case)
                print(case, flush=True)
        result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
