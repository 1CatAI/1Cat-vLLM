# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers import vocab_parallel_embedding as embedding
from vllm.model_executor.layers.quantization.gguf import GGUFConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("kind", [8, 22])
def test_packed_embedding_shards_match_official_fp16_and_graph(monkeypatch, tp, kind):
    rows, k = 128, 5120
    block, size = gguf.GGML_QUANT_SIZES[kind]
    rng = np.random.default_rng(20261007 + kind)
    blocks = rng.integers(0, 256, (rows * k // block, size), dtype=np.uint8)
    blocks[:, :2] = np.frombuffer(np.float16(0.00390625).tobytes(), dtype=np.uint8)
    raw = blocks.reshape(rows, -1)
    oracle = (
        torch.from_numpy(gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(kind)))
        .half()
        .cuda()
    )
    monkeypatch.setattr(embedding, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(embedding, "tensor_model_parallel_all_reduce", lambda x: x)
    tokens = torch.tensor(
        [0, rows // tp - 1, rows // tp, 63, 64, 95, 96, 127], device="cuda"
    )
    for rank in range(tp):
        monkeypatch.setattr(
            embedding, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        with torch.device("cuda"):
            layer = embedding.VocabParallelEmbedding(
                rows, k, params_dtype=torch.float16, quant_config=GGUFConfig()
            )
        layer.weight_loader(layer.qweight_type, torch.tensor(kind))
        layer.weight_loader(layer.qweight, torch.from_numpy(raw))
        layer.quant_method.process_weights_after_loading(layer)
        assert layer.qweight.dtype == torch.uint8
        assert layer.qweight.numel() == raw.size // tp
        actual = layer(tokens)
        begin, end = rank * rows // tp, (rank + 1) * rows // tp
        expected = oracle[tokens].clone()
        expected[(tokens < begin) | (tokens >= end)] = 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(tokens)
        tokens.copy_(tokens.roll(1))
        graph.replay()
        expected = oracle[tokens].clone()
        expected[(tokens < begin) | (tokens >= end)] = 0
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
