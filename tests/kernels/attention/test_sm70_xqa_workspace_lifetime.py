# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise scratch ownership without requiring the native attention module."""

import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def interface(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "flash-attention-v100/flash_attn_v100/flash_attn_interface.py"
    )
    name = "_xqa_workspace_test_interface"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules, "flash_attn_v100_cuda", ModuleType("flash_attn_v100_cuda")
    )
    spec.loader.exec_module(module)
    return module


def allocate(interface, partitions, device="cpu"):
    return interface._get_xqa_staged_rescale_workspace(
        torch.empty(1, 6, 32, device=device),
        batch_capacity=1,
        num_heads=6,
        plan=interface._DecodePlan(256, partitions, partitions, partitions),
    )


@pytest.mark.parametrize("warmup", [False, True])
def test_growth_retains_captured_buffer(interface, monkeypatch, warmup):
    capturing = [not warmup]
    monkeypatch.setattr(interface, "_cuda_graph_capture_active", lambda: capturing[0])
    original = allocate(interface, 4)
    reference = weakref.ref(original)
    if warmup:
        capturing[0] = True
        assert allocate(interface, 4) is original
    del original
    capturing[0] = False
    grown = allocate(interface, 64)
    retained = reference()
    assert retained is not None
    assert grown.data_ptr() != retained.data_ptr()
    del retained
    # An intermediate eager allocation must not sever a captured ancestor.
    allocate(interface, 128)
    assert reference() is not None


def test_eager_growth_releases_uncaptured_buffer(interface, monkeypatch):
    monkeypatch.setattr(interface, "_cuda_graph_capture_active", lambda: False)
    original = allocate(interface, 4)
    reference = weakref.ref(original)
    del original
    allocate(interface, 64)
    assert reference() is None


def test_repeated_capture_reuses_storage(interface, monkeypatch):
    monkeypatch.setattr(interface, "_cuda_graph_capture_active", lambda: True)
    original = allocate(interface, 64)
    for partitions in [1, 4, 32, 64]:
        assert allocate(interface, partitions) is original
    assert len(interface._xqa_staged_rescale_workspace_cache) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("warmup", [False, True])
def test_cuda_graph_replay_survives_growth(interface, warmup):
    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle()
    results = [torch.empty((), device="cuda") for _ in range(2)]
    with torch.cuda.stream(stream):
        allocate(interface, 4, "cuda")
    if not warmup:
        interface._xqa_staged_rescale_workspace_cache.clear()
    graphs, references = [], []
    for index, partitions in enumerate([4, 64]):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=pool):
            scratch = allocate(interface, partitions, "cuda")
            scratch.fill_(index + 1)
            results[index].copy_(scratch.sum())
        references.append(weakref.ref(scratch))
        del scratch
        graphs.append(graph)
    first, second = [reference() for reference in references]
    assert first is not None and second is not None
    assert first.data_ptr() != second.data_ptr()
    for index in [0, 1, 0, 1]:
        graphs[index].replay()
        torch.accelerator.synchronize()
        assert results[index].item() == 6 * [4, 64][index] * (index + 1)
