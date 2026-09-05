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
def test_tp8_push_registration_stays_in_communicator_dso(
    monkeypatch, sidecar_owner, owner_has_op
):
    base = SimpleNamespace(init_custom_ar=Mock())
    sidecar = SimpleNamespace()
    if sidecar_owner:
        sidecar.init_custom_ar = Mock()
    owner, other = (sidecar, base) if sidecar_owner else (base, sidecar)
    name = "register_sm70_tp8_hierarchical_push_allreduce_buffer"
    size_name = "sm70_tp8_hierarchical_push_allreduce_buffer_size"
    for namespace in (owner, other) if owner_has_op else (other,):
        setattr(namespace, name, Mock())
        setattr(namespace, size_name, Mock(return_value=4096))
    monkeypatch.setattr(torch.ops, "_C_custom_ar", base)
    monkeypatch.setattr(torch.ops, "_C_custom_ar_flashnext", sidecar)
    if owner_has_op:
        assert getattr(ops, size_name)() == 4096
        getattr(ops, name)(123, [1, 2])
        getattr(owner, name).assert_called_once_with(123, [1, 2])
    else:
        with pytest.raises(AttributeError):
            getattr(ops, size_name)()
        with pytest.raises(AttributeError):
            getattr(ops, name)(123, [1, 2])
    getattr(other, name).assert_not_called()
    getattr(other, size_name).assert_not_called()
