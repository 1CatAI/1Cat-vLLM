# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def _mock_communicator() -> CustomAllreduce:
    communicator = object.__new__(CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.world_size = 4
    communicator.fully_connected = True
    communicator.tp8_hierarchical = False
    communicator.dispatch_max_size = 1024 * 1024
    return communicator


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_should_custom_ar_accepts_supported_dtype(dtype: torch.dtype) -> None:
    communicator = _mock_communicator()

    assert communicator.should_custom_ar(torch.empty(16, dtype=dtype))


@pytest.mark.parametrize(
    "dtype",
    [torch.float64, torch.int64, torch.int32, torch.int8, torch.uint8, torch.bool],
)
def test_should_custom_ar_rejects_unsupported_dtype(dtype: torch.dtype) -> None:
    communicator = _mock_communicator()

    assert not communicator.should_custom_ar(torch.empty(16, dtype=dtype))
