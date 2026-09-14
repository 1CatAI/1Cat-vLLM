# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turing takes the QPN2 route of the SM70 linear path; Volta is unchanged.

CPU-only: device capabilities are mocked, no CUDA context is created.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

MODULE = "vllm.model_executor.layers.quantization.sm70_turbomind"


@pytest.fixture
def cuda_tensor():
    tensor = torch.empty(0)
    with patch.object(
        torch.Tensor, "is_cuda", new_callable=lambda: property(lambda _: True)
    ):
        yield tensor


@pytest.mark.parametrize(
    ("capability", "turbomind", "turing_qpn2"),
    [
        ((7, 0), True, False),
        ((7, 5), False, True),
        ((8, 0), False, False),
        ((8, 9), False, False),
    ],
)
def test_prepare_routes_follow_the_tensor_device(
    cuda_tensor, capability, turbomind, turing_qpn2
):
    with (
        patch(f"{MODULE}.torch.cuda.get_device_capability", return_value=capability),
        patch(f"{MODULE}.use_turbomind", return_value=True),
    ):
        assert sm70_tm.should_prepare_turbomind(cuda_tensor, True) is turbomind
        assert sm70_tm.should_prepare_turing_qpn2(cuda_tensor, True) is turing_qpn2


def test_turing_route_honours_the_backend_switch(cuda_tensor):
    with (
        patch(f"{MODULE}.torch.cuda.get_device_capability", return_value=(7, 5)),
        patch(f"{MODULE}.use_turbomind", return_value=False),
    ):
        assert sm70_tm.should_prepare_turing_qpn2(cuda_tensor, True) is False


def test_turing_route_needs_a_cuda_tensor():
    with patch(f"{MODULE}.torch.cuda.get_device_capability", return_value=(7, 5)):
        assert sm70_tm.should_prepare_turing_qpn2(torch.empty(0), True) is False


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    [
        (5120, 8704, (8, 2)),  # table entry
        (5120, 3584, (16, 2)),  # table entry
        (4096, 4096, (8, 1)),  # heuristic: 128 tiles x 8 = 1024 warps in flight
        (2048, 1024, (32, 2)),  # heuristic: needs the largest split to fill the GPU
        (256, 64, (16, 2)),  # heuristic fallback: 16 groups, split 16
    ],
)
def test_qpn2_launch_config(k, n, expected):
    assert sm70_tm.qpn2_launch_config(k, n) == expected


def test_qpn2_launch_config_rejects_unsplittable_k():
    with pytest.raises(RuntimeError):
        sm70_tm.qpn2_launch_config(16 * 3, 64)


def test_pad_qpn2_output_rows_pads_to_32():
    weight = torch.arange(40 * 8, dtype=torch.uint8).view(40, 8)
    scales = torch.arange(40 * 2, dtype=torch.uint8).view(40, 2)
    padded_weight, padded_scales, physical_n = sm70_tm.pad_qpn2_output_rows(
        weight, scales
    )
    assert physical_n == 64
    assert padded_weight.shape == (64, 8) and padded_scales.shape == (64, 2)
    assert torch.equal(padded_weight[:40], weight)
    assert torch.equal(padded_scales[:40], scales)
    assert int(padded_weight[40:].abs().sum()) == 0

    same_weight, same_scales, physical_n = sm70_tm.pad_qpn2_output_rows(
        weight[:32], scales[:32]
    )
    assert physical_n == 32
    assert same_weight is weight[:32] or torch.equal(same_weight, weight[:32])
