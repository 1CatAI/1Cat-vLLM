# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded four-expert check of the DSpark TP4 M1/M5 native TC geometry."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from tests.quantization.test_sm70_mxfp4_moe import _v4_flash_moe_config
from vllm.model_executor.layers.fused_moe import FusedMoE, MoEActivation
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import Mxfp4SM70MoEMethod
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="V100 required"
)


def _decode(packed, scale):
    lut = packed.new_tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    indices = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    return lut[indices] * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)


@pytest.mark.parametrize("tokens", [1, 5])
def test_native_v41_mxfp4_fp16_boundaries_and_replay(tokens):
    base = _v4_flash_moe_config()
    config = replace(
        base,
        num_experts=128,
        num_local_experts=4,
        num_logical_experts=128,
        experts_per_token=3,
        hidden_dim=5120,
        intermediate_size_per_partition=576,
        moe_parallel_config=replace(base.moe_parallel_config, tp_size=4),
        swiglu_limit=10.0,
    )
    generator = torch.Generator(device="cuda").manual_seed(415)

    def checkpoint(n, k):
        return (
            torch.randint(
                0,
                256,
                (4, n, k // 2),
                generator=generator,
                device="cuda",
                dtype=torch.uint8,
            ),
            torch.randint(
                120,
                123,
                (4, n, k // 32),
                generator=generator,
                device="cuda",
                dtype=torch.uint8,
            ),
        )

    gate_up, gate_scale = checkpoint(1152, 5120)
    down, down_scale = checkpoint(5120, 576)
    # Only four full-size experts are materialized in this component oracle.
    gu_reference, down_reference = (
        _decode(gate_up, gate_scale),
        _decode(down, down_scale),
    )
    layer = SimpleNamespace(
        local_num_experts=4,
        global_num_experts=128,
        top_k=3,
        moe_config=config,
        activation=MoEActivation.SILU,
        swiglu_limit=10.0,
        apply_router_weight_on_input=False,
        expert_map=None,
        w13_weight=torch.nn.Parameter(gate_up, requires_grad=False),
        w13_weight_scale=torch.nn.Parameter(gate_scale, requires_grad=False),
        w2_weight=torch.nn.Parameter(down, requires_grad=False),
        w2_weight_scale=torch.nn.Parameter(down_scale, requires_grad=False),
    )
    method = Mxfp4SM70MoEMethod(config)
    method.process_weights_after_loading(cast(FusedMoE, layer))
    x = torch.randn(tokens, 5120, generator=generator, device="cuda").half() / 4
    ids = (torch.arange(tokens * 3, device="cuda").view(tokens, 3) % 4).int()
    probabilities = torch.rand(tokens, 3, generator=generator, device="cuda")

    def run():
        return method.apply(cast(FusedMoE, layer), x, probabilities, ids, None, None)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    ids.copy_((ids + 1) % 4)
    probabilities.mul_(0.75)
    graph.replay()
    buffers = method._get_buffers(layer, tokens)
    order = ids.flatten().argsort(stable=True)
    actual_gu = buffers["gate_up"].clone()
    # Validate each GEMM at its FP16 boundary. Feeding a separately rounded
    # first GEMM into a second reference compounds that first rounding error
    # and is not a check of the second operator's own arithmetic.
    expected_gu = torch.stack(
        [
            (
                x[int(route // 3)].float() @ gu_reference[int(ids.flatten()[route])].T
            ).half()
            for route in order
        ]
    )
    torch.testing.assert_close(actual_gu, expected_gu, rtol=0.001, atol=0.00001)
    gate, up = actual_gu.float().chunk(2, dim=-1)
    expected_hidden = (
        torch.nn.functional.silu(gate.clamp(max=10))
        * up.clamp(-10, 10)
        * probabilities.flatten()[order, None]
    ).half()
    torch.testing.assert_close(buffers["intermediate"], expected_hidden, rtol=0, atol=0)
    inverse = order.argsort()
    actual_down = buffers["sorted_output"].clone()
    reference = torch.zeros_like(x, dtype=torch.float32)
    for token in range(tokens):
        for route in range(3):
            expert = int(ids[token, route])
            hidden = expected_hidden[inverse[token * 3 + route]]
            result = (hidden.float() @ down_reference[expert].T).half()
            actual_route = actual_down[inverse[token * 3 + route]]
            torch.testing.assert_close(actual_route, result, rtol=0.001, atol=0.0001)
            reference[token] += actual_route.float()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
