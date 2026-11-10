# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
from types import SimpleNamespace

import pytest
import torch

from vllm.config import KernelConfig
from vllm.v1.worker.gpu_worker import Worker


def owner(enabled, host_cache):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            kernel_config=KernelConfig(qsa_host_kv_trim_warmup_cache=enabled)
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={"attention": SimpleNamespace(host_kv=host_cache)}
        ),
    )


@pytest.mark.parametrize(
    "enabled,cache,trim",
    [
        (False, None, False),
        (False, object(), False),
        (True, None, False),
        (True, object(), True),
    ],
)
def test_trim_requires_explicit_policy_and_actual_host_consumer(
    monkeypatch, enabled, cache, trim
):
    calls = []
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: calls.append("trim"))
    assert Worker._trim_qsa_warmup_cache(owner(enabled, cache)) is trim
    assert calls == (["trim"] if trim else [])


def test_resource_policy_roundtrip_preserves_computation_hash():
    assert not KernelConfig().qsa_host_kv_trim_warmup_cache
    for host in (False, True):
        a = KernelConfig(qsa_host_kv=host)
        b = KernelConfig(qsa_host_kv=host, qsa_host_kv_trim_warmup_cache=True)
        receiver, sender = multiprocessing.Pipe(duplex=False)
        try:
            sender.send(b)
            copied = receiver.recv()
        finally:
            sender.close()
            receiver.close()
        assert copied.qsa_host_kv_trim_warmup_cache
        assert copied.compute_hash() == a.compute_hash()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_trim_preserves_active_state_and_changed_input_graph_replay():
    x = torch.ones((256, 256), device="cuda", dtype=torch.float16)
    state = torch.zeros_like(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.add_(x)
        output = state * 2
    # CUDA graph capture already trims earlier cache. Reproduce the auxiliary
    # kernel warmup that allocates unused pages after capture has finished.
    stream = torch.Stream(device="cuda")
    with torch.cuda.stream(stream):
        scratch = torch.empty(16 * 1024**2, device="cuda", dtype=torch.float32)
        scratch.fill_(3)
    stream.synchronize()
    del scratch
    before = state.clone()
    before_reserved = torch.accelerator.memory_reserved()
    assert Worker._trim_qsa_warmup_cache(owner(True, state))
    assert torch.accelerator.memory_reserved() < before_reserved
    assert torch.equal(state, before)
    for value in (2, 5, -1):
        x.fill_(value)
        graph.replay()
        before.add_(x)
        assert torch.equal(state, before)
        assert torch.equal(output, before * 2)
