# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact storage, range, fallback and stream contracts for host-page DMA."""

import hashlib
import importlib
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


@pytest.fixture(scope="module", autouse=True)
def native_artifact():
    if (
        not torch.accelerator.is_available()
        or not current_platform.is_device_capability(70)
    ):
        pytest.skip("requires SM70")
    module = importlib.import_module("vllm._sm70_qsa_device_C")
    assert module.__file__ is not None
    path = Path(module.__file__)
    print(
        {
            "qsa_native": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
    assert hasattr(torch.ops.vllm_sm70_qsa_device, "stage_host_run")


def history(dtype, allocation="pinned", page=16, blocks=384):
    data = (torch.arange(blocks * 2 * page * 256) % 97).to(dtype)
    data = data.view(blocks, 2, page, 1, 256)
    scales = torch.arange(blocks * page * 2, dtype=torch.float32).view(-1, 2) / 4096
    if allocation == "device":
        return data, data.cuda(), scales, scales.cuda()
    if allocation == "pinned":
        data, scales = data.pin_memory(), scales.pin_memory()
    return (
        data,
        get_accelerator_view_from_cpu_tensor(data),
        scales,
        get_accelerator_view_from_cpu_tensor(scales),
    )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("allocation,page", [("pinned", 16), ("raw_hostalloc", 4)])
def test_native_copy_preserves_bytes_scales_and_guards(dtype, allocation, page):
    expected, source, expected_scales, scales = history(dtype, allocation, page)
    # Offset source views exercise cudaPointerGetAttributes' host alias.
    source, scales = source[3:], scales[3 * page :]
    out_base = torch.full((100, 2, page, 1, 256), 117, dtype=dtype, device="cuda")
    scale_base = torch.full((100 * page, 2), -7.0, device="cuda")
    out, out_scales = out_base[1:-1], scale_base[page:-page]
    copy = torch.ops.vllm_sm70_qsa_device.stage_host_run
    assert copy(source, scales, out, out_scales, 7, 11, 65)
    torch.accelerator.synchronize()
    assert torch.equal(out[11:76].cpu(), expected[10:75])
    assert bool((out_base[:12] == 117).all())
    assert bool((out_base[77:] == 117).all())
    if dtype == torch.uint8:
        assert torch.equal(
            out_scales[11 * page : 76 * page].cpu(),
            expected_scales[10 * page : 75 * page],
        )
        assert bool((scale_base[: 12 * page] == -7).all())
        assert bool((scale_base[77 * page :] == -7).all())
    else:
        assert bool((scale_base == -7).all())


@pytest.mark.parametrize(
    "source_page,target_page,pages",
    [(-1, 0, 1), (0, -1, 1), (0, 0, -1), (380, 0, 5), (0, 380, 5)],
)
def test_invalid_range_rejected_before_either_copy(source_page, target_page, pages):
    _, source, _, scales = history(torch.uint8)
    out = torch.full_like(source, 117)
    out_scales = torch.full_like(scales, -7)
    with pytest.raises(RuntimeError, match="page range"):
        torch.ops.vllm_sm70_qsa_device.stage_host_run(
            source, scales, out, out_scales, source_page, target_page, pages
        )
    torch.accelerator.synchronize()
    assert bool((out == 117).all()) and bool((out_scales == -7).all())


def test_invalid_scales_rejected_before_history_copy():
    _, source, _, scales = history(torch.uint8)
    out, out_scales = torch.full_like(source, 117), torch.zeros(1, device="cuda")
    with pytest.raises(RuntimeError, match="scales must cover"):
        torch.ops.vllm_sm70_qsa_device.stage_host_run(
            source, scales, out, out_scales, 0, 0, 64
        )
    torch.accelerator.synchronize()
    assert bool((out == 117).all())


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_device_history_declines_without_writes_and_empty_run_is_noop(dtype):
    _, source, _, scales = history(dtype, "device")
    out, out_scales = torch.full_like(source, 117), torch.full_like(scales, -7)
    copy = torch.ops.vllm_sm70_qsa_device.stage_host_run
    assert not copy(source, scales, out, out_scales, 0, 0, 65)
    assert copy(source, scales, out, out_scales, 384, 384, 0)
    torch.accelerator.synchronize()
    assert bool((out == 117).all()) and bool((out_scales == -7).all())


class Calls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.copies = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) == "vllm_sm70_qsa_device.stage_host_run.default":
            self.copies += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype,threshold", [(torch.uint8, 64), (torch.float16, 32)])
@pytest.mark.parametrize("neighbor,direct", [(-1, False), (0, None), (1, True)])
def test_staging_threshold_and_independent_decode_admission(
    dtype, threshold, neighbor, direct
):
    expected, source, _, _ = history(dtype, "raw_hostalloc")
    state = HostQSAKV(
        384,
        16,
        256,
        torch.device("cuda", 0),
        history=source,
        hot_tokens=2048,
        rows=8,
        direct_device=direct,
    )
    used = threshold + neighbor
    table = torch.arange(used, device="cuda", dtype=torch.int32).view(1, -1)
    lengths = torch.tensor([used * 16 - 3], device="cuda", dtype=torch.int32)
    with Calls() as calls:
        result = state.stage_history(table, lengths, 131072)
    torch.accelerator.synchronize()
    assert calls.copies == int(neighbor >= 0)
    assert torch.equal(result[0].cpu(), expected[:used])


@pytest.mark.parametrize("fallback", ["fragmented", "device", "unavailable"])
def test_staging_fallbacks_keep_exact_result(fallback, monkeypatch):
    expected, source, _, _ = history(
        torch.uint8, "device" if fallback == "device" else "raw_hostalloc"
    )
    if fallback == "unavailable":
        original = importlib.import_module

        def missing(name, *args, **kwargs):
            if name == "vllm._sm70_qsa_device_C":
                raise ImportError("unavailable for this test")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", missing)
    state = HostQSAKV(
        384,
        16,
        256,
        torch.device("cuda", 0),
        history=source,
        hot_tokens=2048,
        rows=8,
        direct_device=False,
    )
    pages = torch.arange(65) * 2 if fallback == "fragmented" else torch.arange(65)
    table = pages.int().cuda().view(1, -1)
    lengths = torch.tensor([pages.numel() * 16 - 1], device="cuda", dtype=torch.int32)
    with Calls() as calls:
        result = state.stage_history(table, lengths, 131072)
    torch.accelerator.synchronize()
    assert calls.copies == int(fallback == "device")
    assert torch.equal(result[0].cpu(), expected[pages])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("runs", [3, 4, 5])
def test_few_run_limit_and_many_run_fallback(dtype, runs):
    expected, source, _, _ = history(dtype, "raw_hostalloc")
    state = HostQSAKV(
        384,
        16,
        256,
        torch.device("cuda", 0),
        history=source,
        hot_tokens=2048,
        rows=8,
        direct_device=False,
    )
    pages = (torch.arange(runs)[:, None] * 50 + torch.arange(32)).flatten()
    table = pages.int().cuda().view(1, -1)
    lengths = torch.tensor([pages.numel() * 16 - 1], device="cuda", dtype=torch.int32)
    with Calls() as calls:
        result = state.stage_history(table, lengths, 131072)
    torch.accelerator.synchronize()
    assert calls.copies == (runs if runs <= 4 else 0)
    assert torch.equal(result[0].cpu(), expected[pages])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("first,last,expected_copies", [(0, 65, 1), (5, 135, 2)])
def test_transfer_range_ignores_other_plan_runs(dtype, first, last, expected_copies):
    expected, source, _, _ = history(dtype, "raw_hostalloc", blocks=768)
    state = HostQSAKV(
        768,
        16,
        256,
        torch.device("cuda", 0),
        history=source,
        hot_tokens=2048,
        rows=8,
        direct_device=False,
    )
    pages = (torch.arange(6)[:, None] * 120 + torch.arange(70)).flatten()
    table = pages.int().cuda().view(1, -1)
    lengths = torch.tensor([pages.numel() * 16 - 1], device="cuda", dtype=torch.int32)
    plan = state.stage_plan(table, lengths, 131072)
    history_out, scales = state._stage_views(131072, slot=1)
    history_out.fill_(117)
    with Calls() as calls:
        state._stage_range(plan, history_out, scales, first, last)
    torch.accelerator.synchronize()
    assert calls.copies == expected_copies
    assert torch.equal(history_out[first:last].cpu(), expected[pages[first:last]])
    assert bool((history_out[:first] == 117).all())
    assert bool((history_out[last:] == 117).all())


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_native_copy_graph_replay_reads_updated_host_bytes(dtype):
    host, source, host_scales, scales = history(dtype)
    out, out_scales = torch.empty_like(source), torch.empty_like(scales)
    copy = torch.ops.vllm_sm70_qsa_device.stage_host_run
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        assert copy(source, scales, out, out_scales, 0, 0, 384)
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        assert copy(source, scales, out, out_scales, 0, 0, 384)
    host.fill_(53)
    host_scales.fill_(0.125)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(out.cpu(), host)
    if dtype == torch.uint8:
        assert torch.equal(out_scales.cpu(), host_scales)
