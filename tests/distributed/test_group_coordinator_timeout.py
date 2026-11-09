# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NCCL subgroups of a GroupCoordinator follow --distributed-timeout-seconds.

CPU-only: ``torch.distributed.new_group`` is replaced by a recorder, so no
process group is created. Only the arguments handed to it are checked.
"""

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import GroupCoordinator


@pytest.fixture
def new_group_calls(monkeypatch):
    calls: list[dict] = []

    def record(ranks, **kwargs):
        calls.append({"ranks": ranks, **kwargs})
        return MagicMock(name=f"pg[{kwargs.get('backend')}]")

    monkeypatch.setattr(torch.distributed, "new_group", record)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    return calls


def _build_coordinator() -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[[0]],
        local_rank=0,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
    )


def test_device_group_follows_distributed_timeout(new_group_calls):
    config = VllmConfig()
    config.parallel_config.distributed_timeout_seconds = 3600
    config.parallel_config.cpu_distributed_timeout_seconds = 120

    with set_current_vllm_config(config):
        _build_coordinator()

    device_call, cpu_call = new_group_calls
    assert device_call["backend"] == "nccl"
    assert device_call["timeout"] == timedelta(seconds=3600)
    assert cpu_call["backend"] == "gloo"
    assert cpu_call["timeout"] == timedelta(seconds=120)


def test_unset_timeout_keeps_pytorch_default(new_group_calls):
    config = VllmConfig()
    assert config.parallel_config.distributed_timeout_seconds is None

    with set_current_vllm_config(config):
        _build_coordinator()

    device_call, cpu_call = new_group_calls
    assert device_call["timeout"] is None
    assert cpu_call["timeout"] is None


def test_without_config_keeps_pytorch_default(new_group_calls):
    _build_coordinator()

    device_call, cpu_call = new_group_calls
    assert device_call["timeout"] is None
    assert cpu_call["timeout"] is None


@pytest.mark.parametrize("timeout_seconds", [3600, None])
def test_sibling_group_inherits_timeout_after_config_context_exits(
    new_group_calls, timeout_seconds
):
    config = VllmConfig()
    config.parallel_config.distributed_timeout_seconds = timeout_seconds
    config.parallel_config.cpu_distributed_timeout_seconds = 120
    with set_current_vllm_config(config):
        group = _build_coordinator()

    # PP sampled/draft transport creates its sibling after the parent group.
    # Preserve the parent's configured timeout even outside the config context.
    group.make_sibling_device_group(group_desc="pp_sampled_draft_broadcast")

    device_call, cpu_call, sibling_call = new_group_calls
    expected = (
        timedelta(seconds=timeout_seconds) if timeout_seconds is not None else None
    )
    assert device_call["timeout"] == expected
    assert cpu_call["timeout"] == timedelta(seconds=120)
    assert sibling_call == {
        "ranks": [0],
        "backend": "nccl",
        "timeout": expected,
        "group_desc": "pp_sampled_draft_broadcast",
    }
