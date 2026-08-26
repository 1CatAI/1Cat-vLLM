# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

import torch


def _load_adapter():
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "vllm"
        / "model_executor"
        / "layers"
        / "quantization"
        / "sm70_turbomind.py"
    )
    spec = importlib.util.spec_from_file_location("sm70_turbomind_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gptq_unpack_weight_and_zeros():
    tm = _load_adapter()
    packed = torch.tensor([[0x76543210]], dtype=torch.int32)

    weight = tm.unpack_gptq_weight(packed)
    zeros = tm.unpack_gptq_zeros(packed)

    assert weight.dtype == torch.uint8
    assert weight.tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert zeros.dtype == torch.float16
    assert zeros.tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]


def test_compressed_unpack_transposes_to_turbomind_layout():
    tm = _load_adapter()
    weight_packed = torch.tensor([[0x3210], [0x7654]], dtype=torch.int32)
    zeros_packed = torch.tensor([[0x76543210]], dtype=torch.int32)

    weight = tm.unpack_compressed_weight(weight_packed)
    zeros = tm.unpack_compressed_zeros(zeros_packed)

    expected_weight = [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert weight.dtype == torch.uint8
    assert weight.tolist() == expected_weight
    assert zeros.dtype == torch.float16
    assert zeros.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_mxfp4_unpack_uint8_blocks_transposes_to_turbomind_layout():
    tm = _load_adapter()
    packed = torch.tensor([[0x10, 0x32], [0x54, 0x76]], dtype=torch.uint8)

    weight = tm.unpack_mxfp4_weight(packed)

    assert weight.dtype == torch.uint8
    assert weight.tolist() == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_mxfp4_unpack_flattens_last_two_block_dims_like_lmdeploy():
    tm = _load_adapter()
    packed = torch.tensor([[[0x10], [0x32]], [[0x54], [0x76]]], dtype=torch.uint8)

    weight = tm.unpack_mxfp4_weight(packed)

    assert weight.dtype == torch.uint8
    assert weight.tolist() == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_symmetric_int4_zero_points_are_eight():
    tm = _load_adapter()
    scales = torch.ones((2, 3), dtype=torch.float32)

    zeros = tm.symmetric_int4_zeros_like(scales)

    assert zeros.dtype == torch.float16
    assert zeros.tolist() == [[8, 8, 8], [8, 8, 8]]
