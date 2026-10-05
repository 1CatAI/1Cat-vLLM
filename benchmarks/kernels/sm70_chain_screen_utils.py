# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retained profiler evidence for resident whole-chain graph screens."""

import json
import math
from pathlib import Path


def graph_kernel_geometry(graph, output: Path, label: str):
    import torch

    trace = output.with_name(f"{output.stem}.{label}.trace.json")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        graph.replay()
        torch.accelerator.synchronize()
    profile.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    if not kernels:
        raise RuntimeError("Profiler did not record CUDA graph kernels")
    grids = [event.get("args", {}).get("grid") for event in kernels]
    geometry = all(isinstance(grid, list) and len(grid) == 3 for grid in grids)
    return {
        "kernels": len(kernels),
        "single_cta": sum(math.prod(grid) == 1 for grid in grids) if geometry else None,
        "grid_le4": sum(math.prod(grid) <= 4 for grid in grids) if geometry else None,
        "geometry_available": geometry,
        "trace": str(trace),
    }
