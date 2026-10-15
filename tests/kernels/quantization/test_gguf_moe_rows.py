# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row CTA geometry retains GGUF quantization, route order and weighted outputs."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gguf import (
    fused_moe_gguf,
    fused_moe_gguf_rows,
)
from vllm.models.qwen4_exp.nvidia.mtp_q8_experts import quantize_q8_0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 CUDA required",
)


@pytest.fixture(scope="module")
def draft_weights():
    torch.manual_seed(8015)
    # Actual TP4 draft geometry and full expert count.
    return (
        quantize_q8_0(torch.randn(512, 320, 2560, device="cuda") * 0.02),
        quantize_q8_0(torch.randn(512, 2560, 160, device="cuda") * 0.02),
    )


@pytest.mark.parametrize("tokens", [0, 1, 2, 5, 8, 10, 20, 40, 64, 65])
def test_weighted_q8_moe_and_changed_graph_are_exact(draft_weights, tokens):
    w13, w2 = draft_weights
    x = torch.randn(tokens, 2560, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, 512, (tokens, 10), device="cuda", dtype=torch.int32)
    weights = torch.randn(tokens, 10, device="cuda").softmax(-1)

    def run(op):
        return op(x, w13, w2, weights, ids, 8, 8, "silu")

    expected = run(fused_moe_gguf)
    actual = run(fused_moe_gguf_rows)
    assert torch.equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(fused_moe_gguf_rows)
    for shift in (1, 7):
        x.mul_(0.875)
        ids.add_(shift).remainder_(512)
        weights.copy_(weights.roll(1, -1))
        graph.replay()
        expected = run(fused_moe_gguf)
        assert torch.isfinite(output).all()
        assert torch.equal(output, expected)


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 319, 321])
def test_native_output_row_tails_are_exact(rows):
    x = torch.randn(2, 160, device="cuda", dtype=torch.float16)
    w = quantize_q8_0(torch.randn(4, rows, 160, device="cuda") * 0.02)
    ids = torch.tensor([[0, 3], [1, 1]], device="cuda", dtype=torch.int32)
    args = x, w, ids, 2, 8, rows, 2
    assert torch.equal(ops.ggml_moe_a8_vec_rows(*args), ops.ggml_moe_a8_vec(*args))


def test_rows_registration_fake_and_aot_dispatch():
    x = torch.randn(2, 160, device="cuda", dtype=torch.float16)
    w = quantize_q8_0(torch.randn(4, 9, 160, device="cuda") * 0.02)
    ids = torch.tensor([[0, 3], [1, 1]], device="cuda", dtype=torch.int32)
    torch.library.opcheck(
        torch.ops._C.ggml_moe_a8_vec_rows,
        (x, w, ids, 2, 8, 9, 2),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    compiled = torch.compile(fused_moe_gguf_rows, fullgraph=True)
    w13 = quantize_q8_0(torch.randn(4, 64, 160, device="cuda") * 0.02)
    w2 = quantize_q8_0(torch.randn(4, 160, 32, device="cuda") * 0.02)
    weights = torch.ones(2, 2, device="cuda") * 0.5
    args = x, w13, w2, weights, ids, 8, 8, "silu"
    assert torch.equal(compiled(*args), fused_moe_gguf(*args))
    x.mul_(0.5)
    ids.add_(1).remainder_(4)
    assert torch.equal(compiled(*args), fused_moe_gguf(*args))


def test_rows_rejects_unqualified_dtype_and_geometry():
    x = torch.ones(2, 160, device="cuda", dtype=torch.float16)
    w = quantize_q8_0(torch.ones(4, 9, 160, device="cuda"))
    ids = torch.zeros(2, 2, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="FP16 inputs and Q8_0"):
        ops.ggml_moe_a8_vec_rows(x.float(), w, ids, 2, 8, 9, 2)
    with pytest.raises(RuntimeError, match="layout"):
        ops.ggml_moe_a8_vec_rows(x, w, ids, 2, 8, 10, 2)
    with pytest.raises(RuntimeError, match="layout"):
        ops.ggml_moe_a8_vec_rows(x, w.float(), ids, 2, 8, 9, 2)
    with pytest.raises(ValueError, match="Q8_0"):
        fused_moe_gguf_rows(x, w, w, x, ids, 2, 8, "silu")


@pytest.mark.parametrize("experts", [256, 257, 512, 513])
def test_mmq_writes_valid_high_expert_ids(experts):
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    torch.manual_seed(3800 + experts)
    x = torch.randn(65, 160, device="cuda", dtype=torch.float16)
    w = quantize_q8_0(torch.randn(experts, 32, 160, device="cuda") * 0.02)
    ids = torch.full((65, 2), experts - 1, device="cuda", dtype=torch.int32)
    ids[:, 0] = min(255, experts - 1)
    ordered, expert_ids, padded = moe_align_block_size(
        ids, ops.ggml_moe_get_block_size(8), experts
    )
    reference = ops.ggml_moe_a8_vec(x, w, ids, 2, 8, 32, 65)
    output = ops.ggml_moe_a8(x, w, ordered, expert_ids, padded, 8, 32, 2, 65)
    assert torch.isfinite(output).all()
    error = (output.float() - reference.float()).norm() / reference.float().norm()
    assert error < 2e-3, error
