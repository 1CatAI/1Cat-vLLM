# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.mtp_q8_experts import (
    GGML_Q8_0,
    MTPQ80MoEMethod,
    dequantize_q8_0,
    quantize_q8_0,
)


def test_mtp_expert_loading_preserves_bytes_and_guards_k_tail():
    layer = torch.nn.Module()
    for name, shape in (
        ("w13_weight", (1, 320, 2720)),
        ("w2_weight", (1, 2560, 170)),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.zeros(shape, dtype=torch.uint8), requires_grad=False
            ),
        )
    # The checkpoint bytes themselves must survive preparation exactly.
    layer.w2_weight.flatten()[-1] = 123
    before = layer.w2_weight.clone()
    method = object.__new__(MTPQ80MoEMethod)
    method.num_experts, method.local, method.hidden = 1, 160, 2560
    method.loaded = {shard: {0} for shard in ("w1", "w3", "w2")}
    method.process_weights_after_loading(layer)
    assert torch.equal(layer.w2_weight, before)
    tail = 11 * 34  # ceil(K160/512)*512 -160, in Q8_0 blocks.
    storage = layer.w2_weight.untyped_storage()
    assert storage.nbytes() == layer.w2_weight.numel() + tail
    entire = torch.empty(0, dtype=torch.uint8).set_(storage)
    assert not entire[-tail:].any()


def test_q8_0_bytes_match_gguf_reference():
    gguf = pytest.importorskip("gguf")
    torch.manual_seed(0)
    weight = torch.randn(40, 160) * 0.02
    weight[3, :32] = 0  # all-zero block keeps d = 0
    ours = quantize_q8_0(weight).numpy()
    reference = gguf.quants.quantize(weight.numpy(), gguf.GGMLQuantizationType.Q8_0)
    np.testing.assert_array_equal(ours, reference)
    np.testing.assert_array_equal(
        dequantize_q8_0(torch.from_numpy(ours)).numpy(),
        gguf.quants.dequantize(reference, gguf.GGMLQuantizationType.Q8_0).astype(
            np.float16
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("tokens", [1, 5, 20, 65])
def test_q8_0_moe_matches_dequantized_reference(tokens):
    import vllm._custom_ops  # noqa: F401
    from vllm.model_executor.layers.quantization.gguf import fused_moe_gguf

    torch.manual_seed(tokens)
    experts, hidden, local, top_k = 512, 2560, 160, 10
    dev = torch.device("cuda")
    w13 = quantize_q8_0(torch.randn(experts, 2 * local, hidden, device=dev) * 0.02)
    w2 = quantize_q8_0(torch.randn(experts, hidden, local, device=dev) * 0.02)
    x = torch.randn(tokens, hidden, device=dev, dtype=torch.float16)
    ids = torch.stack(
        [torch.randperm(experts, device=dev)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    weights = torch.rand(tokens, top_k, device=dev).softmax(-1)

    out = fused_moe_gguf(x, w13, w2, weights, ids, GGML_Q8_0, GGML_Q8_0, "silu")

    g13 = dequantize_q8_0(w13).float()
    g2 = dequantize_q8_0(w2).float()
    reference = torch.zeros(tokens, hidden, device=dev)
    for t in range(tokens):
        for k in range(top_k):
            e = ids[t, k]
            gate, up = (g13[e] @ x[t].float()).split(local)
            act = (torch.nn.functional.silu(gate) * up).half().float()
            reference[t] += weights[t, k] * (g2[e] @ act)
    error = (out.float() - reference).norm() / reference.norm()
    assert error < 2e-2, error


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("weight_type", [2, 8])
@pytest.mark.parametrize("experts, selected", [(256, 255), (257, 256), (512, 511)])
def test_mmq_expert_capacity_graph_replay(weight_type, experts, selected):
    """Large-batch MMQ must write routes above255 and keep live graph indices."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    torch.manual_seed(experts)
    x = torch.randn(65, 256, device="cuda", dtype=torch.float16) * 0.125
    floating = torch.randn(experts, 64, 256) * 0.02
    if weight_type == GGML_Q8_0:
        packed = quantize_q8_0(floating.cuda())
    else:
        gguf = pytest.importorskip("gguf")
        quant = gguf.quants.quantize(
            floating.numpy(), gguf.GGMLQuantizationType(weight_type)
        )
        packed = torch.from_numpy(quant).cuda()
    ids = torch.full((65, 1), selected, device="cuda", dtype=torch.int32)
    sorted_ids, expert_ids, padded = moe_align_block_size(
        ids, ops.ggml_moe_get_block_size(weight_type), experts
    )

    def run():
        return ops.ggml_moe_a8(
            x, packed, sorted_ids, expert_ids, padded, weight_type, 64, 1, 65
        )

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for expert in (selected, 0):
        expert_ids.fill_(expert)
        # Every valid output must be overwritten by the captured native call.
        output.fill_(float("nan"))
        graph.replay()
        # The same physical weights as a one-expert bank must give exactly
        # the same MMQ arithmetic, including its existing A8 input policy.
        reference = ops.ggml_moe_a8(
            x,
            packed[expert : expert + 1],
            sorted_ids,
            torch.zeros_like(expert_ids),
            padded,
            weight_type,
            64,
            1,
            65,
        )
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
