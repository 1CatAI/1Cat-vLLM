# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf import GGUFConfig
from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
from vllm.model_executor.layers.quantization.gguf_repack import q2_0_to_q4_1
from vllm.transformers_utils.gguf_tensor_reader import dequantize


def q2_weight(rows=16, k=640):
    random = np.random.default_rng(20261003)
    blocks = random.integers(0, 256, (rows * k // 64, 18), dtype=np.uint8)
    # Include signed zero, subnormal, signed and maximal finite scales.
    scales = np.resize(
        np.array([0, -0.0, 2**-24, -0.25, 0.125, 65504], np.float16), len(blocks)
    )
    blocks[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    return blocks.reshape(rows, -1)


def test_q2_repack_preserves_every_value_and_all_four_tp_boundaries():
    packed = q2_weight()
    converted = q2_0_to_q4_1(packed)
    expected = dequantize(packed, 42)
    actual = gguf.quants.dequantize(converted, gguf.GGMLQuantizationType.Q4_1)
    np.testing.assert_array_equal(actual, expected)
    assert converted.shape == (16, 400)
    for rank in range(4):
        shard = converted[:, rank * 100 : (rank + 1) * 100].copy()
        decoded = gguf.quants.dequantize(shard, gguf.GGMLQuantizationType.Q4_1)
        np.testing.assert_array_equal(
            decoded, expected[:, rank * 160 : (rank + 1) * 160]
        )


def layer_and_method(rank):
    layer = torch.nn.Module()
    layer.tp_size = 4
    layer.tp_rank = rank
    layer.ep_size = 1
    method = GGUFNativeMoEMethod(
        GGUFConfig(), SimpleNamespace(dp_size=1, pcp_size=1, ep_size=1)
    )
    method.create_weights(layer, 2, 256, 160, torch.float16)
    return layer, method


def test_expert_projection_storage_keeps_types_and_tp_dimensions_independent():
    full_down = q2_0_to_q4_1(q2_weight(256))
    full_gate = np.zeros((640, 144), np.uint8)  # Q4_0 K=256.
    full_up = np.zeros((640, 272), np.uint8)  # Q8_0 K=256.
    for rank in range(4):
        layer, method = layer_and_method(rank)
        for shard, value, payload, parameter in (
            ("w1", 2, full_gate, "w13"),
            ("w3", 8, full_up, "w13"),
            ("w2", 3, full_down, "w2"),
        ):
            method.load_expert(
                layer,
                getattr(layer, parameter + "_qweight_type"),
                torch.tensor(value),
                shard,
                0,
            )
            for expert in range(2):
                method.load_expert(
                    layer,
                    getattr(layer, parameter + "_qweight"),
                    torch.from_numpy(payload),
                    shard,
                    expert,
                )
        assert method.weight_types == {"w1": 2, "w3": 8, "w2": 3}
        assert layer.gguf_w1.shape == (2, 160, 144)
        assert layer.gguf_w3.shape == (2, 160, 272)
        assert layer.gguf_w2.shape == (2, 256, 100)
        np.testing.assert_array_equal(
            layer.gguf_w2[0].numpy(), full_down[:, rank * 100 : (rank + 1) * 100]
        )


@pytest.mark.parametrize("rank", range(4))
def test_original_q2_down_retains_tp_boundary_blocks(rank):
    layer, method = layer_and_method(rank)
    packed = q2_weight(256)
    method.load_expert(layer, layer.w2_qweight_type, torch.tensor(42), "w2", 0)
    method.load_expert(layer, layer.w2_qweight, torch.from_numpy(packed), "w2", 0)
    left, physical_k = method.input_padding["w2"]
    assert (left, physical_k) == (32 if rank % 2 else 0, 192)
    np.testing.assert_array_equal(
        dequantize(layer.gguf_w2[0].numpy(), 42)[:, left : left + 160],
        dequantize(packed, 42)[:, rank * 160 : (rank + 1) * 160],
    )
    assert layer.gguf_w2.numel() == 2 * 256 * 54
