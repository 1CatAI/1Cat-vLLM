# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE rollback and cache commits across changing speculative graph inputs."""

import pytest
import torch
from torch import nn

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer


@pytest.mark.parametrize("rows", [5, 10])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("time_major", [False, True])
def test_changed_metadata_and_successive_state(
    rows, state_dtype, time_major, monkeypatch
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")
    if not hasattr(torch.ops._C, "qwen38_ple_spec_sm70_out"):
        pytest.skip("Requires source-built PLE MTP4 kernel")
    torch.manual_seed(20260927)
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.conv_state_len = 9
    layer.conv_kernel_size = 4
    layer.short_conv_dilation = 3
    shape = (3, 13, 10240) if time_major else (3, 10240, 13)
    initial = torch.randn(shape, device="cuda", dtype=state_dtype)
    states = [initial.clone(), initial.clone()]
    if time_major:
        states = [state.transpose(1, 2) for state in states]
    x = torch.randn(rows, 10240, device="cuda", dtype=torch.float16)
    weight = torch.randn(10240, 4, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.tensor([1], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 5], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([1], device="cuda", dtype=torch.int32)
    graphs, outputs = [], []
    try:
        for arm in range(2):
            monkeypatch.setenv("VLLM_SM70_MTP_PLE_CONV", str(arm))
            envs.disable_envs_cache()

            def run(arm=arm):
                return layer._short_conv_dilated_spec_batched(
                    x, states[arm], weight, ids, starts, accepted, 5
                )

            for _ in range(3):
                run()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
            graphs.append(graph)
            outputs.append(output)
        for state in states:
            state.copy_(initial.transpose(1, 2) if time_major else initial)
        for step in range(30):
            count = (0, 1, 3, 5)[step % 4]
            ids.fill_(0 if step % 7 == 0 else 1 + step % 2)
            accepted.fill_((0, 1, 3, 5, 8)[step % 5])
            starts[1] = count
            x.normal_(0, (0.001, 0.03, 0.1, 1.0, 3.0, 30.0)[step % 6])
            for graph in graphs:
                graph.replay()
            assert torch.equal(
                outputs[0][:count].view(torch.int16),
                outputs[1][:count].view(torch.int16),
            )
            # Graph padding has no live output; both paths must zero it.
            assert torch.count_nonzero(outputs[0][count:]) == 0
            assert torch.count_nonzero(outputs[1][count:]) == 0
            bits = torch.int16 if state_dtype == torch.float16 else torch.int32
            assert torch.equal(
                states[0].contiguous().view(bits), states[1].contiguous().view(bits)
            )
    finally:
        envs.disable_envs_cache()
