# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.linear.fp16_silu_down import apply


@pytest.mark.parametrize("m", [1, 5, 17, 33])
@pytest.mark.parametrize("k,n", [(17, 97), (160, 2560)])
def test_materialized_fp16_boundaries(m, k, n):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 is required")
    torch.manual_seed(4201)
    x = torch.randn(m, 2 * k, device="cuda", dtype=torch.float16) * 0.1
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.1
    activation = torch.nn.functional.silu(x[:, :k].cpu().double()).half()
    product = (activation.double() * x[:, k:].cpu().double()).half()
    expected = (product.double() @ weight.cpu().double().T).half()
    torch.testing.assert_close(apply(x, weight).cpu(), expected, atol=0.001, rtol=0.002)


def test_replay_uses_changed_inputs():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 is required")
    x = torch.ones(5, 34, device="cuda", dtype=torch.float16)
    weight = torch.ones(97, 17, device="cuda", dtype=torch.float16)
    apply(x, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = apply(x, weight)
    x.zero_()
    graph.replay()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
