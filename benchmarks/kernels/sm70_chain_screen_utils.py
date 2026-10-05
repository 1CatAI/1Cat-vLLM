# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retained profiler evidence for resident whole-chain graph screens."""

import ctypes as C
import json
import math
from pathlib import Path


def retained_graph_grids(graph):
    from benchmarks.kernels.sm70_qpn2_graph_nodes import Params, check, rt

    child = rt.cuGraphChildGraphNodeGetGraph
    child.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
    child.restype = C.c_int

    def visit(handle):
        count = C.c_size_t()
        check(rt.cuGraphGetNodes(handle, None, C.byref(count)))
        nodes = (C.c_void_p * count.value)()
        check(rt.cuGraphGetNodes(handle, nodes, C.byref(count)))
        grids = []
        for node in nodes:
            kind = C.c_int()
            check(rt.cuGraphNodeGetType(node, C.byref(kind)))
            if kind.value == 0:
                params = Params()
                check(rt.cuGraphKernelNodeGetParams_v2(node, C.byref(params)))
                grids.append((params.gridDim.x, params.gridDim.y, params.gridDim.z))
            elif kind.value == 4:
                nested = C.c_void_p()
                check(child(node, C.byref(nested)))
                grids.extend(visit(nested))
        return grids

    return visit(graph.raw_cuda_graph())


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
    grids = retained_graph_grids(graph)
    if not grids:
        raise RuntimeError("Retained CUDA graph contains no kernel nodes")
    return {
        "kernels": len(grids),
        "single_cta": sum(math.prod(grid) == 1 for grid in grids),
        "grid_le4": sum(math.prod(grid) <= 4 for grid in grids),
        "geometry_available": True,
        "geometry_source": "CUDA driver graph nodes",
        "profiler_kernel_events": len(kernels),
        "profiler_complete": len(kernels) == len(grids),
        "trace": str(trace),
    }
