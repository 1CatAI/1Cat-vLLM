# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replaying a shared route plan must consume the current inputs and expert IDs."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("tokens", [64, 65, 128])
def test_shared_routes_replay_with_changed_routing_and_inputs(tokens):
    import vllm._C_gguf  # noqa: F401
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize

    from vllm.model_executor.layers.fused_moe import MoEActivation
    from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
    from vllm.model_executor.layers.quantization.gguf_native import pad_weight_tail

    # Small original Q8 banks exercise the shared executor independently of
    # model-size admission and do not need a checkpoint or multi-GB allocation.
    rng = np.random.default_rng(173)
    experts, hidden, intermediate, topk = 8, 256, 128, 2
    kind = int(GGMLQuantizationType.Q8_0)
    packed = []
    for rows, cols in ((intermediate, hidden),) * 2 + ((hidden, intermediate),):
        dense = rng.normal(0, 0.02, (experts, rows, cols)).astype(np.float32)
        raw = torch.from_numpy(quantize(dense, GGMLQuantizationType.Q8_0)).cuda()
        packed.append(pad_weight_tail(raw, kind))
    method = object.__new__(GGUFNativeMoEMethod)
    method.dp4a_admitted = False
    method.expert_trace_tokens = 0
    method.expert_cache = None
    method.hidden_size = hidden
    method.intermediate_size = intermediate
    method.num_experts = experts
    method.input_padding = {}
    method.weight_types = dict.fromkeys(("w1", "w3", "w2"), kind)
    cap = SimpleNamespace(
        operator="ggml_moe_mmq", reason=None, supports_m=lambda m: m >= 64
    )
    method.projection_capabilities = dict.fromkeys(method.weight_types, cap)
    method.prefill_capabilities = dict.fromkeys(method.weight_types, cap)
    method.prefill_block_size = 16
    layer = SimpleNamespace(
        apply_router_weight_on_input=False,
        activation=MoEActivation.SILU,
        expert_map=None,
        swiglu_limit=10.0,
        gguf_w1=packed[0],
        gguf_w3=packed[1],
        gguf_w2=packed[2],
    )
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.float16)
    ids = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    probabilities = torch.full((tokens, topk), 1 / topk, device="cuda")

    def candidate():
        return method.apply(layer, x, probabilities, ids, None, None)

    for _ in range(2):
        candidate()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate()
    for skewed in (False, True, False):
        x.normal_()
        ids.copy_(torch.rand((tokens, experts), device="cuda").argsort(-1)[:, :topk])
        if skewed:
            ids.copy_(torch.arange(experts - topk, experts, device="cuda"))
        probabilities.uniform_()
        probabilities.div_(probabilities.sum(-1, keepdim=True))
        method.prefill_block_size = 0
        reference = candidate()
        method.prefill_block_size = 16
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(captured, reference, rtol=0, atol=0)
