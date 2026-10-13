# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.mtp_q8_experts import (
    GGML_Q8_0,
    dequantize_q8_0,
    quantize_q8_0,
)


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
@pytest.mark.parametrize("tokens", [1, 5, 20])
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
