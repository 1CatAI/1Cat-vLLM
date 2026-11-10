# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned histories share one allocation without overlapping layer views."""

import contextlib
import gc

import pytest
import torch

from vllm.utils import torch_utils
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.worker.utils import allocate_host_kv_cache_pool


def config(*banks):
    return KVCacheConfig(num_blocks=1, kv_cache_tensors=list(banks), kv_cache_groups=[])


def test_host_pool_packs_aligned_disjoint_views(monkeypatch):
    allocations = []
    zeros = torch.zeros

    def allocate(size, **kwargs):
        assert kwargs.pop("pin_memory")
        result = zeros(size, **kwargs)
        allocations.append(result)
        return result

    monkeypatch.setattr(torch, "zeros", allocate)
    monkeypatch.setattr(
        torch.accelerator, "device_index", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        torch_utils, "get_accelerator_view_from_cpu_tensor", lambda t: t
    )
    banks = config(
        KVCacheTensor(513, ["target", "alias"], host_backed=True),
        KVCacheTensor(4096, ["device"]),
        KVCacheTensor(17, ["draft"], host_backed=True),
    )
    tensors = allocate_host_kv_cache_pool(banks, torch.device("cuda:0"))
    assert len(allocations) == 1 and allocations[0].numel() == 1024
    assert set(tensors) == {"target", "alias", "draft"}
    assert tensors["alias"] is tensors["target"]
    assert tensors["draft"].data_ptr() - tensors["target"].data_ptr() == 768
    tensors["target"].fill_(3)
    tensors["draft"].fill_(7)
    assert tensors["alias"].tolist() == [3] * 513
    assert allocations[0][513:768].count_nonzero() == 0
    assert tensors["draft"].tolist() == [7] * 17


def test_device_only_cache_does_not_allocate_pinned_pool(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("device-only caches must not allocate a host pool")

    monkeypatch.setattr(torch, "zeros", unexpected)
    assert (
        allocate_host_kv_cache_pool(
            config(KVCacheTensor(1024, ["device"])), torch.device("cuda:0")
        )
        == {}
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA UVA required")
def test_mapped_pool_owns_storage_and_replays_changed_input():
    tensors = allocate_host_kv_cache_pool(
        config(
            KVCacheTensor(513, ["target"], host_backed=True),
            KVCacheTensor(17, ["draft"], host_backed=True),
        ),
        torch.device("cuda:0"),
    )
    gc.collect()
    value = torch.tensor([5], dtype=torch.int8, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tensors["target"][:1].copy_(value)
        tensors["target"][-1:].copy_(value)
        tensors["draft"][:1].copy_(value)
    value.fill_(11)
    graph.replay()
    torch.accelerator.synchronize()
    target, draft = tensors["target"].cpu(), tensors["draft"].cpu()
    assert target[0] == 11 and target[-1] == 11 and draft[0] == 11
    assert target[1:-1].count_nonzero() == 0
    assert draft[1:].count_nonzero() == 0
