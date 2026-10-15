# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiled raw expert execution against independent dequantized Q8 oracles."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.kernels.quantization.test_gguf_dp4a import quantize_reference
from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
from vllm.model_executor.layers.quantization.gguf_native import packed_tp_span
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [20, 21, 24, 25, 30, 39, 40])
@pytest.mark.parametrize(
    "grouped_rows,reuse_down", [(20, False), (40, False), (40, True)]
)
def test_tiled_mtp_experts_match_q8_oracle_and_changed_graph(
    rows, grouped_rows, reuse_down
):
    experts, hidden, intermediate, top_k = 4, 256, 160, 2
    rng = np.random.default_rng(20261015)

    def weight(kind, n, k):
        block, size = quant_size(kind)
        raw = rng.integers(0, 256, (experts, n, k // block, size), dtype=np.uint8)
        raw[..., :2] = (
            rng.uniform(0.001, 0.005, raw.shape[:-1])
            .astype("<f2")[..., None]
            .view(np.uint8)
        )
        return raw.reshape(experts, n, -1)

    gate_raw, up_raw = [weight(21, intermediate, hidden) for _ in range(2)]
    gate, up = [
        torch.from_numpy(
            RawGGUFProjection.from_rows(
                raw.reshape(experts * intermediate, -1), 21
            ).data.reshape(experts, intermediate, -1)
        ).cuda()
        for raw in (gate_raw, up_raw)
    ]
    decoded_gate, decoded_up = [
        torch.from_numpy(dequantize(raw, 21)).cuda() for raw in (gate_raw, up_raw)
    ]
    down_raw = weight(42, hidden, intermediate * 4)
    first, last, left, _ = packed_tp_span(intermediate * 4, 42, 4, 1)
    down = torch.from_numpy(down_raw[..., first:last].copy()).cuda()
    decoded_down = torch.from_numpy(
        dequantize(down_raw, 42)[..., intermediate : 2 * intermediate]
    ).cuda()
    layer = SimpleNamespace(
        apply_router_weight_on_input=False,
        activation=MoEActivation.SILU,
        expert_map=None,
        gguf_w1=gate,
        gguf_w3=up,
        gguf_w2=down,
    )
    method = object.__new__(GGUFNativeMoEMethod)
    method.dp4a_admitted = True
    method.small_m_dp4a_max_m = 40
    method.expert_cache = None
    method.expert_trace_tokens = 0
    method.weight_types = {"w1": 21, "w3": 21, "w2": 42}
    method.input_padding = {"w2": (left, intermediate + left)}
    method.q8_intermediate = True
    method.grouped_gate_up = True
    method.grouped_gate_up_max_m = grouped_rows
    method.grouped_down = reuse_down
    torch.manual_seed(rows)
    x = (torch.randn(rows, hidden, device="cuda") * 0.125).half()
    ids = torch.randint(experts, (rows, top_k), dtype=torch.int32, device="cuda")
    probabilities = torch.softmax(torch.randn(rows, top_k, device="cuda"), -1)

    def run():
        return method.apply(layer, x, probabilities, ids, None, None)

    def check(actual):
        q, scales, _ = quantize_reference(x)
        decoded_x = (q.float() * scales[..., None].float()).reshape_as(x)
        a, b = [
            torch.einsum("mk,mtnk->mtn", decoded_x, w[ids.long()]).half()
            for w in (decoded_gate, decoded_up)
        ]
        activated = (torch.nn.functional.silu(a.float()) * b.float()).half()
        q, scales, _ = quantize_reference(activated.reshape(rows * top_k, intermediate))
        decoded_intermediate = (q.float() * scales[..., None].float()).reshape_as(
            activated
        )
        projected = torch.einsum(
            "mtk,mtnk->mtn", decoded_intermediate, decoded_down[ids.long()]
        ).half()
        expected = (projected.float() * probabilities[..., None]).sum(1).half()
        torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.003)
        assert torch.isfinite(actual).all()

    check(run())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for shift in (1, 3):
        x.copy_(torch.randn_like(x) * 0.125)
        ids.copy_((ids + shift) % experts)
        probabilities.copy_(torch.softmax(torch.randn_like(probabilities), -1))
        graph.replay()
        check(output)
