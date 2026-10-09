# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker import cpu_worker


def _worker(monkeypatch, **overrides):
    parallel = {
        "world_size_across_dp": 1,
        "nnodes": 1,
        "distributed_executor_backend": "mp",
        "enable_elastic_ep": False,
        **overrides,
    }
    monkeypatch.setattr(cpu_worker, "CPUModelRunner", Mock())
    monkeypatch.setattr(cpu_worker, "set_random_seed", Mock())
    # init_device replaces this callable; restore it even if the test fails.
    monkeypatch.setattr(torch, "set_num_threads", torch.set_num_threads)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        distributed_init_method="tcp://127.0.0.1:0",
        vllm_config=SimpleNamespace(),
        model_config=SimpleNamespace(seed=0),
        rank=0,
        local_rank=0,
        use_v2_model_runner=False,
    )


def test_single_rank_cpu_collective_without_tcp_rendezvous(monkeypatch):
    assert not torch.distributed.is_initialized()
    worker = _worker(monkeypatch)

    def initialize_groups(*args):
        # Reproduce a failed network rendezvous unless the local CPU worker
        # already supplied its in-memory default group.
        if not torch.distributed.is_initialized():
            raise RuntimeError("TCP rendezvous unavailable")
        group = torch.distributed.new_group(backend="gloo")
        value = torch.tensor([2.5])
        torch.distributed.all_reduce(value, group=group)
        assert value.item() == 2.5
        torch.distributed.destroy_process_group(group)

    monkeypatch.setattr(
        cpu_worker, "init_worker_distributed_environment", initialize_groups
    )
    try:
        cpu_worker.CPUWorker.init_device(worker)
        assert torch.distributed.get_world_size() == 1
        assert torch.distributed.get_rank() == 0
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "overrides",
    [
        {"world_size_across_dp": 2},
        {"nnodes": 2},
        {"distributed_executor_backend": "external_launcher"},
        {"enable_elastic_ep": True},
    ],
)
def test_distributed_cpu_keeps_existing_rendezvous(monkeypatch, overrides):
    worker = _worker(monkeypatch, **overrides)
    initialize = Mock()
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)
    initialize_groups = Mock()
    monkeypatch.setattr(
        cpu_worker, "init_worker_distributed_environment", initialize_groups
    )
    cpu_worker.CPUWorker.init_device(worker)
    initialize.assert_not_called()
    initialize_groups.assert_called_once_with(
        worker.vllm_config,
        0,
        worker.distributed_init_method,
        0,
        cpu_worker.current_platform.dist_backend,
    )


def test_existing_cpu_group_is_not_reinitialized(monkeypatch):
    worker = _worker(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    initialize = Mock()
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)
    monkeypatch.setattr(cpu_worker, "init_worker_distributed_environment", Mock())
    cpu_worker.CPUWorker.init_device(worker)
    initialize.assert_not_called()
