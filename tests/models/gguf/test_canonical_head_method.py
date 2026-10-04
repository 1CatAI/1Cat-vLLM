# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.gguf import (
    GGUFConfig,
    GGUFEmbeddingMethod,
    GGUFLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


def empty_layer(cls):
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    return layer


def test_gguf_vocabulary_projection_uses_linear_preparation():
    method = GGUFConfig().get_quant_method(empty_layer(ParallelLMHead), "lm_head")
    assert type(method) is GGUFLinearMethod
    layer = empty_layer(ParallelLMHead)
    method.create_weights(layer, 2560, [62080], 2560, 248320, torch.float16)
    assert layer.qweight.tensor_shape == (62080, 2560)
    assert layer.qweight.input_dim == 1 and layer.qweight.output_dim == 0


def test_gguf_token_embedding_retains_lookup_method():
    method = GGUFConfig().get_quant_method(
        empty_layer(VocabParallelEmbedding), "model.embed_tokens"
    )
    assert type(method) is GGUFEmbeddingMethod
