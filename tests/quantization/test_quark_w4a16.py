# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused Quark W4A16 INT4/UINT4 config and packing regressions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkW4A16Int4MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes import QuarkW4A16Int4
from vllm.model_executor.layers.quantization.quark.utils import (
    canonicalize_quark_packed_int4,
    should_ignore_layer,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.platforms import current_platform

_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _quark_int4_config(
    *,
    pack_method: str = "reorder",
    symmetric: bool = True,
    exclude: list[str] | None = None,
) -> dict:
    return {
        "quant_method": "quark",
        "export": {"pack_method": pack_method, "kv_cache_group": []},
        "global_quant_config": {
            "weight": {
                "dtype": "int4" if symmetric else "uint4",
                "qscheme": "per_group",
                "is_dynamic": False,
                "group_size": 128,
                "symmetric": symmetric,
            }
        },
        "exclude": exclude or [],
    }


def _pack_nibbles(values: torch.Tensor, order: list[int]) -> torch.Tensor:
    shifts = torch.tensor(order, dtype=torch.int64, device=values.device) * 4
    return ((values.to(torch.int64) & 0xF) << shifts).sum(dim=-1).to(torch.int32)


@pytest.mark.parametrize("pack_method", ["order", "reorder"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_quark_int4_config_selects_native_scheme(pack_method, symmetric):
    config = QuarkConfig.from_config(
        _quark_int4_config(pack_method=pack_method, symmetric=symmetric)
    )
    scheme = config._get_scheme_from_config(config.quant_config["global_quant_config"])

    assert isinstance(scheme, QuarkW4A16Int4)
    assert scheme.pack_reorder is (pack_method == "reorder")
    assert scheme.is_symmetric is symmetric


@pytest.mark.parametrize("missing_field", ["group_size", "symmetric"])
def test_quark_int4_config_requires_layout_fields(missing_field):
    raw = _quark_int4_config()
    del raw["global_quant_config"]["weight"][missing_field]
    config = QuarkConfig.from_config(raw)

    with pytest.raises(ValueError, match=missing_field):
        config._get_scheme_from_config(config.quant_config["global_quant_config"])


def test_quark_mapper_preserves_calibration_metadata_and_bare_excludes():
    raw = _quark_int4_config(exclude=["lm_head"])
    raw["algo_config"] = [
        {"name": "qronos", "inside_layer_modules": ["self_attn.q_proj"]}
    ]
    config = QuarkConfig.from_config(raw)
    config.apply_vllm_mapper(WeightsMapper())

    assert config.quant_config["algo_config"] == raw["algo_config"]
    assert should_ignore_layer(
        "language_model.lm_head", ignore=config.quant_config["exclude"]
    )
    assert not should_ignore_layer(
        "language_model.model.layers.0.mlp.gate",
        ignore=config.quant_config["exclude"],
    )


@pytest.mark.parametrize("pack_reorder", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_quark_packed_int4_canonicalization_is_source_invariant(
    pack_reorder, symmetric
):
    values = torch.tensor(
        [
            [[0, 1, 7, 8, 9, 15, 2, 14], [15, 8, 0, 3, 12, 7, 1, 9]],
            [[3, 11, 5, 13, 6, 10, 4, 12], [2, 14, 1, 9, 0, 8, 7, 15]],
        ],
        dtype=torch.int32,
    )
    source_order = _REVERSE_AWQ_PACK_ORDER if pack_reorder else list(range(8))
    packed = _pack_nibbles(values, source_order)

    actual = canonicalize_quark_packed_int4(
        packed,
        pack_reorder=pack_reorder,
        is_symmetric=symmetric,
    )
    expected_values = values ^ 0x8 if symmetric else values
    expected = _pack_nibbles(expected_values, _REVERSE_AWQ_PACK_ORDER)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="Quark W4A16 dense execution requires a CUDA device",
)
@pytest.mark.parametrize("pack_method", ["order", "reorder"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_quark_int4_dense_apply_matches_dequantized_reference(pack_method, symmetric):
    torch.manual_seed(20260828)
    hidden_size = output_size = 128
    group_size = 32
    pack_reorder = pack_method == "reorder"
    layer = torch.nn.Module()
    scheme = QuarkW4A16Int4(
        group_size=group_size,
        pack_method=pack_method,
        is_symmetric=symmetric,
    )

    def weight_loader(param, loaded_weight, *args, **kwargs):
        del args, kwargs
        param.data.copy_(loaded_weight)

    with (
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        scheme.create_weights(
            layer,
            output_partition_sizes=[output_size],
            input_size_per_partition=hidden_size,
            params_dtype=torch.float16,
            weight_loader=weight_loader,
            input_size=hidden_size,
            output_size=output_size,
        )
        values = torch.randint(0, 16, (hidden_size, output_size), dtype=torch.int32)
        zeros = (
            torch.zeros(hidden_size // group_size, output_size, dtype=torch.int32)
            if symmetric
            else torch.randint(
                0,
                16,
                (hidden_size // group_size, output_size),
                dtype=torch.int32,
            )
        )
        scales = (
            torch.rand(
                hidden_size // group_size,
                output_size,
                dtype=torch.float16,
            )
            * 0.02
            + 0.002
        )
        source_order = _REVERSE_AWQ_PACK_ORDER if pack_reorder else list(range(8))
        layer.weight.data.copy_(
            _pack_nibbles(values.view(hidden_size, -1, 8), source_order)
        )
        layer.weight_zero_point.data.copy_(
            _pack_nibbles(zeros.view(hidden_size // group_size, -1, 8), source_order)
        )
        layer.weight_scale.data.copy_(scales)

        logical_values = values
        if symmetric:
            logical_values = torch.where(values >= 8, values - 16, values)
        else:
            logical_values = values - zeros.repeat_interleave(group_size, dim=0)
        reference_weight = logical_values.to(torch.float16) * scales.repeat_interleave(
            group_size, dim=0
        )

        device = torch.accelerator.current_accelerator()
        layer.to(device)
        scheme.process_weights_after_loading(layer)

    inputs = (
        torch.randn(
            7,
            hidden_size,
            device=device,
            dtype=torch.float16,
        )
        * 0.1
    )
    actual = scheme.apply_weights(layer, inputs)
    expected = inputs @ reference_weight.to(device)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)


class _FakeMoEConfig:
    has_bias = False

    def __init__(self, tp_size: int = 1, tp_rank: int = 0):
        self.tp_size = tp_size
        self.tp_rank = tp_rank


@pytest.mark.parametrize("symmetric", [False, True])
def test_quark_int4_moe_quant_config_tracks_zero_points(symmetric):
    config = QuarkConfig.from_config(_quark_int4_config(symmetric=symmetric))
    method = QuarkW4A16Int4MoEMethod(
        config.quant_config["global_quant_config"]["weight"],
        config.pack_method,
        _FakeMoEConfig(),
    )
    layer = SimpleNamespace(
        group_size=128,
        w13_weight_scale=torch.ones(2, 16, 2),
        w2_weight_scale=torch.ones(2, 8, 2),
        w13_weight_zero_point=torch.ones(2, 8, 2, dtype=torch.uint8),
        w2_weight_zero_point=torch.ones(2, 4, 2, dtype=torch.uint8),
    )

    quant_config = method.get_fused_moe_quant_config(layer)

    assert quant_config is not None
    assert quant_config.use_int4_w4a16
    assert quant_config.block_shape == [0, 128]
    assert (quant_config.w1_zp is None) is symmetric
    assert (quant_config.w2_zp is None) is symmetric


def _load_w2_zero_point(raw_zp: torch.Tensor, tp_size: int, tp_rank: int):
    config = QuarkConfig.from_config(_quark_int4_config(symmetric=False))
    moe_config = _FakeMoEConfig(tp_size, tp_rank)
    method = QuarkW4A16Int4MoEMethod(
        config.quant_config["global_quant_config"]["weight"],
        config.pack_method,
        moe_config,
    )
    layer = SimpleNamespace(
        moe_config=moe_config,
        intermediate_size_per_partition=64,
        group_size_div_factor=1,
    )
    param = torch.nn.Parameter(
        torch.zeros(1, 16, 8 // tp_size, dtype=torch.uint8),
        requires_grad=False,
    )
    loader = method.get_weight_loader(layer, weight_loader=None)
    loader(
        param,
        raw_zp.clone(),
        weight_name="w2_weight_zero_point",
        shard_id="w2",
        expert_id=0,
    )
    return param.data[0]


def test_quark_int4_moe_loader_shards_w2_zero_point_across_tp_ranks():
    raw_zp = torch.arange(8 * 16, dtype=torch.uint8).reshape(8, 16)

    full = _load_w2_zero_point(raw_zp, tp_size=1, tp_rank=0)
    shard0 = _load_w2_zero_point(raw_zp, tp_size=2, tp_rank=0)
    shard1 = _load_w2_zero_point(raw_zp, tp_size=2, tp_rank=1)

    expected = full.view(full.size(0), 2, -1)
    assert torch.equal(shard0, expected[:, 0])
    assert torch.equal(shard1, expected[:, 1])
