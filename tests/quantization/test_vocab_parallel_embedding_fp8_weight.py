# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Float8 table weights must not silently lose their quantization contract."""

import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from vllm.model_executor.layers import vocab_parallel_embedding as vocab
from vllm.model_executor.utils import set_weight_attrs


@pytest.fixture(params=[vocab.VocabParallelEmbedding, vocab.ParallelLMHead])
def layer(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> vocab.VocabParallelEmbedding:
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: 1)
    return request.param(64, 16, params_dtype=torch.bfloat16, prefix="model.table")


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ],
)
def test_rejects_float8_into_unquantized_table(
    layer: vocab.VocabParallelEmbedding, dtype: torch.dtype
) -> None:
    layer.weight.data.fill_(17)
    before = layer.weight.detach().clone()
    with pytest.raises(ValueError, match="model.table"):
        layer.weight_loader(layer.weight, torch.ones(64, 16).to(dtype))
    torch.testing.assert_close(layer.weight, before, rtol=0, atol=0)


def test_preserves_unquantized_weight_load(
    layer: vocab.VocabParallelEmbedding,
) -> None:
    weight = torch.arange(1024).reshape(64, 16).to(torch.bfloat16)
    layer.weight_loader(layer.weight, weight)
    torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)


def test_preserves_float8_destination(layer: vocab.VocabParallelEmbedding) -> None:
    weight = torch.ones(64, 16).to(torch.float8_e4m3fn)
    param = torch.nn.Parameter(torch.empty_like(weight), requires_grad=False)
    set_weight_attrs(param, {"output_dim": 0})
    layer.weight_loader(param, weight)
    assert param.dtype == weight.dtype
    torch.testing.assert_close(param.float(), weight.float(), rtol=0, atol=0)


def test_preserves_uninitialized_destination(
    layer: vocab.VocabParallelEmbedding,
) -> None:
    weight = torch.ones(64, 16).to(torch.float8_e4m3fn)
    param = UninitializedParameter(requires_grad=False)
    set_weight_attrs(param, {"output_dim": 0})
    layer.weight_loader(param, weight)
    assert param.dtype == weight.dtype
    torch.testing.assert_close(param.float(), weight.float(), rtol=0, atol=0)
