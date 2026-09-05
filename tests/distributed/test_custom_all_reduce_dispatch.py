# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm._custom_ops as ops
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


@pytest.mark.parametrize("sidecar_owner", [False, True])
@pytest.mark.parametrize("owner_has_op", [False, True])
def test_hc_output_gather_stays_in_communicator_dso(
    monkeypatch: pytest.MonkeyPatch, sidecar_owner: bool, owner_has_op: bool
) -> None:
    base = SimpleNamespace(init_custom_ar=Mock())
    sidecar = SimpleNamespace()
    if sidecar_owner:
        sidecar.init_custom_ar = Mock()
    owner, other = (sidecar, base) if sidecar_owner else (base, sidecar)
    other.sm70_qwen38_hc_output_allgather = Mock()
    if owner_has_op:
        owner.sm70_qwen38_hc_output_allgather = Mock()
    monkeypatch.setattr(torch.ops, "_C_custom_ar", base)
    monkeypatch.setattr(torch.ops, "_C_custom_ar_flashnext", sidecar)
    assert ops.supports_sm70_qwen38_hc_output_allgather() == owner_has_op
    if owner_has_op:
        local = torch.empty(640)
        output = torch.empty(2560)
        ops.sm70_qwen38_hc_output_allgather(123, local, output)
        owner.sm70_qwen38_hc_output_allgather.assert_called_once_with(
            123, local, output
        )
    else:
        # An old sidecar must not borrow the new op from a rebuilt base wheel,
        # and a sidecar without init must not receive the base wheel's pointer.
        with pytest.raises(AttributeError):
            ops.sm70_qwen38_hc_output_allgather(
                123, torch.empty(640), torch.empty(2560)
            )
    other.sm70_qwen38_hc_output_allgather.assert_not_called()
