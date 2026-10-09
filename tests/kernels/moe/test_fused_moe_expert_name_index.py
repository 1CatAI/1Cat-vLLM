# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexed expert-name matching must equal the linear substring scan."""

from types import SimpleNamespace

from vllm.model_executor.layers.fused_moe.layer import FusedMoE


def mapping(experts=64, base_layer=""):
    fused = [
        (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 0, "w1"),
        (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 1, "w3"),
        (f"experts.{base_layer}w2_weight", "experts.down_proj", 0, "w2"),
    ]
    per_expert = [
        (
            f"experts.{base_layer}w13_" if name != "down_proj" else "experts.w2_",
            f"experts.{expert}.{name}.{base_layer}",
            expert,
            shard,
        )
        for expert in range(experts)
        for shard, name in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"))
    ]
    return fused + per_expert


def names(experts=64):
    out = [
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}.{suffix}"
        for layer in (0, 11)
        for expert in range(experts)
        for proj in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "qweight", "qweight_type")
    ]
    return out + [
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.experts.1.gate_proj.experts.10.up_proj.weight",
        "experts.5.down_proj.experts.5.down_proj.",
        "experts.007.up_proj.x",
    ]


def test_indexed_matching_equals_linear_scan():
    for base_layer in ("", "base_layer."):
        table = mapping(base_layer=base_layer)
        layer = SimpleNamespace(
            _EXPERT_ENTRY=FusedMoE._EXPERT_ENTRY,
            _EXPERT_SUBSTRING=FusedMoE._EXPERT_SUBSTRING,
        )
        for qual_name in names():
            expected = [entry for entry in table if entry[1] in qual_name]
            actual = FusedMoE._matching_expert_entries(layer, table, qual_name)
            assert actual == expected, (base_layer, qual_name)


def test_index_is_rebuilt_for_a_new_mapping():
    layer = SimpleNamespace(
        _EXPERT_ENTRY=FusedMoE._EXPERT_ENTRY,
        _EXPERT_SUBSTRING=FusedMoE._EXPERT_SUBSTRING,
    )
    first, second = mapping(2), mapping(4)
    name = "mlp.experts.3.up_proj.weight"
    assert FusedMoE._matching_expert_entries(layer, first, name) == []
    assert FusedMoE._matching_expert_entries(layer, second, name) == [
        entry for entry in second if entry[1] in name
    ]
