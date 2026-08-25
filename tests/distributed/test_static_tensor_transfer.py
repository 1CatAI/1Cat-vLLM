# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch

from vllm.distributed.parallel_state import GroupCoordinator


class _DummyWork:
    def wait(self) -> None:
        pass


def _group(rank_in_group: int, world_size: int = 2) -> GroupCoordinator:
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size = world_size
    group.rank_in_group = rank_in_group
    group.ranks = list(range(world_size))
    group.use_cpu_custom_send_recv = False
    group.device_group = None
    group.cpu_group = None
    return group


def test_static_tensor_dict_transfer_uses_caller_owned_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, torch.Tensor]] = []

    def fake_isend(tensor: torch.Tensor, *, dst: int, group: Any) -> _DummyWork:
        del group
        calls.append(("send", dst, tensor))
        return _DummyWork()

    def fake_irecv(tensor: torch.Tensor, *, src: int, group: Any) -> _DummyWork:
        del group
        tensor.fill_(7)
        calls.append(("recv", src, tensor))
        return _DummyWork()

    monkeypatch.setattr(torch.distributed, "isend", fake_isend)
    monkeypatch.setattr(torch.distributed, "irecv", fake_irecv)

    sent = torch.arange(8, dtype=torch.float32)
    send_handles = _group(0).isend_tensor_dict_static({"hidden_states": sent})

    received = torch.empty_like(sent)
    recv_handles = _group(1).irecv_tensor_dict_static({"hidden_states": received})

    assert len(send_handles) == 1
    assert len(recv_handles) == 1
    assert calls == [("send", 1, sent), ("recv", 0, received)]
    torch.testing.assert_close(received, torch.full_like(received, 7))


def test_static_tensor_dict_transfer_rejects_non_tensors() -> None:
    with pytest.raises(TypeError, match="only accepts tensors"):
        _group(0).isend_tensor_dict_static(  # type: ignore[arg-type]
            {"hidden_states": object()}
        )
