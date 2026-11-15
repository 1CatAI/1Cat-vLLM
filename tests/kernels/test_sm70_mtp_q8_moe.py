# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 fused Q8_0 MTP draft MoE kernel parity and CUDA Graph tests."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.mtp_q8_experts import (
    GGML_Q8_0,
    dequantize_q8_0,
    quantize_q8_0,
)


def _require_sm70_mtp_q8():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")
    if not hasattr(torch.ops._C, "sm70_mtp_q8_0_moe_out"):
        pytest.skip("Requires rebuilt sm70_mtp_q8_0_moe_out")


@pytest.mark.parametrize("inter", [160, 320, 640])
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 40, 64])
def test_sm70_mtp_q8_moe_matches_gguf_and_dequant(inter, tokens):
    _require_sm70_mtp_q8()
    import vllm._custom_ops  # noqa: F401
    from vllm.model_executor.layers.quantization.gguf import fused_moe_gguf

    torch.manual_seed(1000 * inter + tokens)
    experts, hidden, top_k = 64, 2560, 10
    dev = torch.device("cuda")
    w13 = quantize_q8_0(torch.randn(experts, 2 * inter, hidden, device=dev) * 0.02)
    w2 = quantize_q8_0(torch.randn(experts, hidden, inter, device=dev) * 0.02)
    x = torch.randn(tokens, hidden, device=dev, dtype=torch.float16)
    ids = torch.stack(
        [torch.randperm(experts, device=dev)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    weights = torch.rand(tokens, top_k, device=dev, dtype=torch.float32).softmax(-1)

    actual = torch.ops.vllm.qwen4_mtp_q8_0_moe(x, w13, w2, weights, ids)
    gguf_out = fused_moe_gguf(x, w13, w2, weights, ids, GGML_Q8_0, GGML_Q8_0, "silu")

    rel_vs_gguf = (actual.float() - gguf_out.float()).norm() / gguf_out.float().norm()
    cos_vs_gguf = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), gguf_out.float().flatten(), dim=0
    )
    assert rel_vs_gguf < 2e-3, f"rel_vs_gguf={rel_vs_gguf}"
    assert cos_vs_gguf > 0.99999, f"cos_vs_gguf={cos_vs_gguf}"

    if tokens <= 8 and inter == 160:
        g13 = dequantize_q8_0(w13).float()
        g2 = dequantize_q8_0(w2).float()
        ref = torch.zeros(tokens, hidden, device=dev)
        for t in range(tokens):
            for k in range(top_k):
                e = ids[t, k]
                gate, up = (g13[e] @ x[t].float()).split(inter)
                act = (torch.nn.functional.silu(gate) * up).half().float()
                ref[t] += weights[t, k] * (g2[e] @ act)
        rel_vs_ref = (actual.float() - ref).norm() / ref.norm()
        assert rel_vs_ref < 2e-2, f"rel_vs_ref={rel_vs_ref}"


@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 40])
def test_sm70_mtp_q8_moe_cudagraph_replay_and_invalid_ids(tokens):
    _require_sm70_mtp_q8()
    import vllm._custom_ops  # noqa: F401
    from vllm.model_executor.layers.quantization.gguf import fused_moe_gguf

    torch.manual_seed(2026 + tokens)
    experts, hidden, inter, top_k = 64, 2560, 160, 10
    dev = torch.device("cuda")
    w13 = quantize_q8_0(torch.randn(experts, 2 * inter, hidden, device=dev) * 0.02)
    w2 = quantize_q8_0(torch.randn(experts, hidden, inter, device=dev) * 0.02)
    x = torch.randn(tokens, hidden, device=dev, dtype=torch.float16)
    ids = torch.randint(0, experts, (tokens, top_k), device=dev, dtype=torch.int32)
    weights = torch.rand(tokens, top_k, device=dev, dtype=torch.float32).softmax(-1)

    for _ in range(3):
        torch.ops.vllm.qwen4_mtp_q8_0_moe(x, w13, w2, weights, ids)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.vllm.qwen4_mtp_q8_0_moe(x, w13, w2, weights, ids)

    for step in range(3):
        x.normal_(0, 0.5 + 0.2 * step)
        ids.random_(0, experts)
        weights.uniform_(0.05, 1.0)
        weights.div_(weights.sum(dim=-1, keepdim=True))
        out.fill_(float("nan"))
        graph.replay()
        expected = fused_moe_gguf(
            x, w13, w2, weights, ids, GGML_Q8_0, GGML_Q8_0, "silu"
        )
        assert torch.isfinite(out).all()
        rel = (out.float() - expected.float()).norm() / expected.float().norm()
        assert rel < 2e-3, f"step={step} rel={rel}"

    # Verify padded/invalid negative expert IDs produce finite zeroed routes.
    ids[-1, :] = -1
    out.fill_(float("nan"))
    graph.replay()
    assert torch.isfinite(out).all()
    assert torch.all(out[-1] == 0)
