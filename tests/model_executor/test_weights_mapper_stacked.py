# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.models.utils import WeightsMapper


def test_weights_mapper_preserves_stacked_shard_id() -> None:
    mapper = WeightsMapper(
        orig_to_new_stacked={
            ".input_mix_weight_down.weight": (
                ".input_mix_weight_down_block_inject.weight",
                0,
            ),
            ".block_inject_weight.weight": (
                ".input_mix_weight_down_block_inject.weight",
                1,
            ),
        }
    )
    down = torch.ones(2, 4)
    injection = torch.full((3, 4), 2.0)

    mapped = list(
        mapper.apply(
            [
                ("layer.input_mix_weight_down.weight", down),
                ("layer.block_inject_weight.weight", injection),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        "layer.input_mix_weight_down_block_inject.weight",
        "layer.input_mix_weight_down_block_inject.weight",
    ]
    assert down.shard_id == 0
    assert injection.shard_id == 1


def test_weights_mapper_preserves_named_stacked_shard_id() -> None:
    mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q.": (".qkv.", "q"),
            ".k.": (".qkv.", "k"),
        }
    )
    q = torch.ones(2, 4)
    k = torch.full((2, 4), 2.0)

    mapped = list(
        mapper.apply(
            [
                ("layer.q.weight", q),
                ("layer.k.weight", k),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        "layer.qkv.weight",
        "layer.qkv.weight",
    ]
    assert q.shard_id == "q"
    assert k.shard_id == "k"


def test_merged_column_load_weights_forwards_stacked_shards() -> None:
    layer = object.__new__(MergedColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.output_sizes = [2, 3]
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.prefix = "hc.input_mix_weight_down_block_inject"
    weight = torch.nn.Parameter(torch.zeros(5, 4))
    calls = []

    def weight_loader(param, loaded_weight, shard_id) -> None:
        calls.append((param, loaded_weight, shard_id))

    weight.weight_loader = weight_loader
    layer.register_parameter("weight", weight)
    down = torch.ones(2, 4)
    down.shard_id = 0
    injection = torch.full((3, 4), 2.0)
    injection.shard_id = 1

    loaded = list(
        layer.load_weights(
            [
                ("weight", down),
                ("weight", injection),
            ]
        )
    )

    assert loaded == ["weight", "weight"]
    assert calls == [
        (weight, down, 0),
        (weight, injection, 1),
    ]


def test_qkv_load_weights_forwards_stacked_shards() -> None:
    layer = object.__new__(QKVParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.prefix = "attn.qkv_proj"
    weight = torch.nn.Parameter(torch.zeros(7, 4))
    calls = []

    def weight_loader(param, loaded_weight, shard_id) -> None:
        calls.append((param, loaded_weight, shard_id))

    weight.weight_loader = weight_loader
    layer.register_parameter("weight", weight)
    query = torch.ones(3, 4)
    query.shard_id = "q"
    key = torch.full((2, 4), 2.0)
    key.shard_id = "k"
    value = torch.full((2, 4), 3.0)
    value.shard_id = "v"

    loaded = list(
        layer.load_weights(
            [
                ("weight", query),
                ("weight", key),
                ("weight", value),
            ]
        )
    )

    assert loaded == ["weight", "weight", "weight"]
    assert calls == [
        (weight, query, "q"),
        (weight, key, "k"),
        (weight, value, "v"),
    ]
