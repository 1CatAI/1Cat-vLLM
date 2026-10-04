# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import gguf
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf import (
    GGUFConfig,
    GGUFEmbeddingMethod,
    GGUFLMHeadMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)


def empty_layer(cls):
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    return layer


def test_gguf_vocabulary_projection_uses_linear_preparation():
    method = GGUFConfig().get_quant_method(empty_layer(ParallelLMHead), "lm_head")
    assert type(method) is GGUFLMHeadMethod
    layer = empty_layer(ParallelLMHead)
    method.create_weights(layer, 2560, [62080], 2560, 248320, torch.float16)
    assert layer.qweight.tensor_shape == (62080, 2560)
    assert layer.qweight.input_dim == 1 and layer.qweight.output_dim == 0


def test_gguf_token_embedding_retains_lookup_method():
    method = GGUFConfig().get_quant_method(
        empty_layer(VocabParallelEmbedding), "model.embed_tokens"
    )
    assert type(method) is GGUFEmbeddingMethod


def test_unquantized_head_keeps_its_existing_method():
    method = GGUFConfig(["lm_head"]).get_quant_method(
        empty_layer(ParallelLMHead), "lm_head"
    )
    assert type(method) is UnquantizedEmbeddingMethod


@pytest.mark.parametrize("rank", range(4))
def test_quantized_head_loads_packed_vocab_rows_and_zero_padding(rank):
    layer = empty_layer(ParallelLMHead)
    layer.num_embeddings_per_partition = 4
    layer.org_vocab_size = 15
    start = rank * 4
    end = min(start + 4, layer.org_vocab_size)
    layer.shard_indices = SimpleNamespace(
        org_vocab_start_index=start, org_vocab_end_index=end
    )
    method = GGUFConfig().get_quant_method(layer, "lm_head")
    method.create_weights(layer, 256, [4], 256, 15, torch.float16)
    quant_type = gguf.GGMLQuantizationType.Q6_K
    block_bytes = gguf.GGML_QUANT_SIZES[quant_type][1]
    raw = torch.arange(15 * block_bytes).remainder(256).to(torch.uint8)
    raw = raw.reshape(15, block_bytes)
    layer.weight_loader(layer.qweight_type, torch.tensor(int(quant_type)))
    layer.weight_loader(layer.qweight, raw)
    assert layer.qweight.dtype == torch.uint8
    assert layer.qweight.shape == (4, block_bytes)
    assert layer.qweight.tensor_shape == (4, 256)
    assert layer.qweight_type.weight_type == int(quant_type)
    torch.testing.assert_close(layer.qweight[: end - start], raw[start:end])
    assert torch.count_nonzero(layer.qweight[end - start :]) == 0
