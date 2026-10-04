# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.quantization.gguf import (
    GGUFConfig,
    GGUFEmbeddingMethod,
    GGUFLMHeadMethod,
    _gguf_lm_head_projection,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


def test_head_and_embedding_methods_are_separate():
    config = GGUFConfig()
    head = ParallelLMHead.__new__(ParallelLMHead)
    embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    assert isinstance(config.get_quant_method(head, "lm_head"), GGUFLMHeadMethod)
    assert (
        type(config.get_quant_method(embedding, "embed_tokens")) is GGUFEmbeddingMethod
    )


def test_lm_head_actual_m_keeps_raw_outside_measured_band(monkeypatch):
    calls = []
    raw = torch.empty(4, 144, dtype=torch.uint8)
    codes = stats = torch.empty(0)

    def canonical(x, *args):
        calls.append("canonical")
        return torch.zeros(x.shape[0], 4, dtype=x.dtype)

    def legacy(x, *args):
        calls.append("raw")
        return torch.zeros(x.shape[0], 4, dtype=x.dtype)

    monkeypatch.setattr(
        torch.ops.vllm, "prepared_gguf_projection", canonical, raising=False
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.gguf.fused_mul_mat_gguf", legacy
    )
    for m in [1, 2, 4, 8, 16, 32]:
        out = _gguf_lm_head_projection(
            torch.empty(m, 256, dtype=torch.float16),
            raw,
            codes,
            stats,
            0,
            0,
            2,
            16,
            True,
            8,
        )
        assert out.shape == (m, 4)
        assert calls[-1] == ("canonical" if 2 <= m <= 16 else "raw")


def test_head_compilation_keeps_actual_m_inside_operator():
    torch._dynamo.reset()
    graphs = []

    def backend(g, inputs):
        graphs.append(g)
        return g.forward

    def f(x, raw, codes, stats):
        return torch.ops.vllm.gguf_lm_head_projection(
            x, raw, codes, stats, 163840, 62080, 2, 16, True, 8
        )

    compiled = torch.compile(f, dynamic=True, fullgraph=True, backend=backend)
    raw = torch.empty(62080, 2880, dtype=torch.uint8, device="meta")
    codes = torch.empty(5120, 7760, dtype=torch.int32, device="meta")
    stats = torch.empty(160, 62080, dtype=torch.int32, device="meta")
    for m in [8, 2, 16, 32]:
        x = torch.empty(m, 5120, dtype=torch.float16, device="meta")
        torch._dynamo.mark_dynamic(x, 0, min=2, max=8192)
        out = compiled(x, raw, codes, stats)
        assert out.shape == (m, 62080)
    assert len(graphs) == 1
    assert any(
        "gguf_lm_head_projection" in str(n.target) for n in graphs[0].graph.nodes
    )
    torch._dynamo.reset()
