# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

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
@pytest.mark.parametrize(
    "op_name,num_inputs",
    [("sm70_qwen38_hc_output_allgather", 1), ("sm70_qwen38_hc_up_mix_allgather", 3)],
)
def test_hc_gather_stays_in_communicator_dso(
    monkeypatch: pytest.MonkeyPatch,
    sidecar_owner: bool,
    owner_has_op: bool,
    op_name: str,
    num_inputs: int,
) -> None:
    base = SimpleNamespace(init_custom_ar=Mock())
    sidecar = SimpleNamespace()
    if sidecar_owner:
        sidecar.init_custom_ar = Mock()
    owner, other = (sidecar, base) if sidecar_owner else (base, sidecar)
    setattr(other, op_name, Mock())
    if owner_has_op:
        setattr(owner, op_name, Mock())
    monkeypatch.setattr(torch.ops, "_C_custom_ar", base)
    monkeypatch.setattr(torch.ops, "_C_custom_ar_flashnext", sidecar)
    assert getattr(ops, f"supports_{op_name}")() == owner_has_op
    tensors = [torch.empty(16) for _ in range(num_inputs + 1)]
    if owner_has_op:
        getattr(ops, op_name)(123, *tensors)
        getattr(owner, op_name).assert_called_once_with(123, *tensors)
    else:
        # An old sidecar must not borrow the new op from a rebuilt base wheel,
        # and a sidecar without init must not receive the base wheel's pointer.
        with pytest.raises(AttributeError):
            getattr(ops, op_name)(123, *tensors)
    getattr(other, op_name).assert_not_called()


@pytest.mark.parametrize("fused,hidden", [(True, True), (False, True), (False, False)])
def test_hc_model_selects_available_owner_route(monkeypatch, fused, hidden) -> None:
    import vllm.distributed.parallel_state as parallel
    import vllm.models.qwen4_exp.nvidia.sm70_fp16_hc as hc

    comm = SimpleNamespace(
        rank=0,
        can_sm70_qwen38_hc_shard=Mock(return_value=True),
        supports_sm70_qwen38_hc_up_mix_allgather=Mock(return_value=fused),
        supports_sm70_qwen38_hc_output_allgather=Mock(return_value=hidden),
        sm70_qwen38_hc_down_allgather=Mock(),
        sm70_qwen38_hc_up_mix_allgather=Mock(),
        sm70_qwen38_hc_output_allgather=Mock(),
        sm70_qwen38_hc_gate_mix=Mock(),
    )
    tp = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=comm))
    monkeypatch.setattr(parallel, "get_tp_group", lambda: tp)
    monkeypatch.setattr(hc, "_runtime_ok", lambda *args: True)
    for name in (
        "_qwen38_hc_down_local_shard_kernel",
        "_qwen38_hc_up_hidden_shard_kernel",
        "_qwen38_hc_up_local_gate_kernel",
    ):
        monkeypatch.setattr(hc, name, MagicMock())
    block, injection = hc._qwen38_sm70_fp16_fused_hc(
        torch.empty(1, 10240), torch.empty(0), torch.empty(0)
    )
    assert block.shape == (1, 2560) and injection.shape == (1, 4)
    comm.sm70_qwen38_hc_down_allgather.assert_called_once()
    expected = (
        "up_mix_allgather" if fused else "output_allgather" if hidden else "gate_mix"
    )
    for name in ("up_mix_allgather", "output_allgather", "gate_mix"):
        op = getattr(comm, f"sm70_qwen38_hc_{name}")
        if name == expected:
            op.assert_called_once()
        else:
            op.assert_not_called()


@pytest.mark.parametrize("sidecar_owner", [False, True])
@pytest.mark.parametrize(
    "missing", [None, "sm70_qwen38_hc_down_allgather", "sm70_qwen38_hc_gate_mix"]
)
def test_hc_shard_admission_requires_owner_operators(
    monkeypatch, sidecar_owner, missing
):
    base = SimpleNamespace(init_custom_ar=Mock())
    sidecar = SimpleNamespace()
    if sidecar_owner:
        sidecar.init_custom_ar = Mock()
    owner, other = (sidecar, base) if sidecar_owner else (base, sidecar)
    names = ("sm70_qwen38_hc_down_allgather", "sm70_qwen38_hc_gate_mix")
    for name in names:
        setattr(other, name, Mock())
        if name != missing:
            setattr(owner, name, Mock())
    monkeypatch.setattr(torch.ops, "_C_custom_ar", base)
    monkeypatch.setattr(torch.ops, "_C_custom_ar_flashnext", sidecar)
    assert ops.supports_sm70_qwen38_hc_shard() == (missing is None)

    communicator = _mock_communicator()
    communicator.sm70_tp4_push_buffer_ptrs = [0] * 4
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    branches = torch.empty((1, 10240), dtype=torch.float16)
    assert communicator.can_sm70_qwen38_hc_shard(branches) == (missing is None)
    for name in names:
        args = (
            (123, branches, branches, branches)
            if name.endswith("gate_mix")
            else (123, branches, branches)
        )
        if name == missing:
            with pytest.raises(AttributeError):
                getattr(ops, name)(*args)
        else:
            getattr(ops, name)(*args)
            getattr(owner, name).assert_called_once_with(*args)
        getattr(other, name).assert_not_called()


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
