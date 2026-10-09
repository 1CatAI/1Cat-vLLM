# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import prod

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts


@pytest.mark.parametrize(
    "rows,chunk,n,k,topk,activation",
    [
        (16384, 16384, 320, 2560, 10, MoEActivation.SILU),
        (16384, 4096, 320, 2560, 10, MoEActivation.SILU),
        (64, 64, 5120, 2560, 2, MoEActivation.SILU),
        (64, 64, 64, 256, 2, MoEActivation.SILU_NO_MUL),
    ],
)
def test_activation_workspace_and_output_have_independent_capacities(
    monkeypatch, rows, chunk, n, k, topk, activation
):
    class Manager:
        requested: tuple[tuple[tuple[int, ...], torch.dtype], ...] = ()

        def get_simultaneous(
            self, *requests: tuple[tuple[int, ...], torch.dtype]
        ) -> list[torch.Tensor]:
            self.requested = requests
            return [
                torch.empty(shape, device="meta", dtype=dtype)
                for shape, dtype in requests
            ]

    manager = Manager()
    monkeypatch.setattr(mk, "current_workspace_manager", lambda: manager)
    kernel = object.__new__(mk.FusedMoEKernelModularImpl)
    kernel.fused_experts = TritonExperts(
        make_dummy_moe_config(), FUSED_MOE_UNQUANTIZED_CONFIG
    )
    w13, w2, out = kernel._allocate_buffers(
        torch.float16,
        torch.device("meta"),
        chunk,
        rows,
        n,
        k,
        topk,
        512,
        512,
        None,
        activation,
    )
    activation_dim = kernel.fused_experts.adjust_N_for_activation(n, activation)
    assert w13.shape == (chunk, topk, activation_dim)
    assert w2.shape == (chunk, topk, max(n, k))
    assert out.shape == (rows, k)
    elements = [prod(shape) for shape, _ in manager.requested]
    expected = [chunk * topk * activation_dim, chunk * topk * max(n, k)]
    if chunk == rows:
        expected[0] = max(expected[0], rows * k)
    else:
        expected.append(rows * k)
    assert elements == expected
    if rows == chunk == 16384:
        assert sum(elements) * torch.float16.itemsize == 880 * 1024**2
