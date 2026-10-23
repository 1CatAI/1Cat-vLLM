# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialized KDA tuners retain inputs and graph state without model loading."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import KernelConfig
from vllm.model_executor.layers.fla.ops.gdn_chunk_kernels import bind_kda_kernels
from vllm.model_executor.layers.fla.ops.kda import chunk_kda
from vllm.runtime_resources import release_runtime_resources


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="qualified FLA device required",
)
@torch.inference_mode()
def test_kda_owned_tuners_changed_input_replay_and_release(monkeypatch):
    torch.manual_seed(822)
    cfgs = [SimpleNamespace(kernel_config=KernelConfig()) for _ in range(2)]
    tuners = []
    for cfg in cfgs:
        cfg.kernel_config.capture_provider_inputs()
        tuners.append(bind_kda_kernels(cfg))
    assert tuners[0].recompute is not tuners[1].recompute
    assert tuners[0].delta_h is not tuners[1].delta_h
    shape = (1, 64, 4, 64)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.float16) * 0.1 for _ in range(3)
    ]
    g = -torch.rand(shape, device="cuda") * 0.1
    beta = torch.rand(1, 64, 4, device="cuda", dtype=torch.float16)
    state = torch.randn(1, 4, 64, 64, device="cuda") * 0.01

    def run(tuner):
        return chunk_kda(
            q,
            k,
            v.clone(),
            g,
            beta,
            initial_state=state.clone(),
            output_final_state=True,
            kernels=tuner,
        )

    graphs, outputs = [], []
    for tuner in tuners:
        run(tuner)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run(tuner)
        graphs.append(graph)
        outputs.append(result)
    monkeypatch.setenv("VLLM_SM70_KDA_PREFILL_SCHEDULE", "0")
    monkeypatch.setenv("VLLM_SM70_GDN_DELTA_H_BV", "bad-after-init")
    for step in range(2):
        q.add_(0.01)
        state.mul_(0.9)
        for tuner, graph, captured in zip(tuners, graphs, outputs):
            expected = run(tuner)
            graph.replay()
            assert all(torch.equal(a, b) for a, b in zip(expected, captured))
    graphs[0].reset()
    release_runtime_resources(cfgs[0])
    q.add_(0.02)
    expected = run(tuners[1])
    graphs[1].replay()
    assert all(torch.equal(a, b) for a, b in zip(expected, outputs[1]))
    graphs[1].reset()
    release_runtime_resources(cfgs[1])
