# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fcntl
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.minimax_h3.config import H3Config
from vllm.video import gpu


def test_partial_lock_failure_releases_previously_acquired_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, "LOCK_ROOT", tmp_path)
    with (tmp_path / "1cat-vllm-v100-gpu1.lock").open("a+") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            gpu.GPUGroupLease((0, 1, 2, 3))
        with gpu.GPUGroupLease((0,)):
            pass


def test_reserved_primary_group_uses_whole_alternate_group(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, "LOCK_ROOT", tmp_path)
    groups = [(0, 1, 2, 3), (4, 5, 6, 7)]
    monkeypatch.setattr(gpu, "_available_gpu_groups", lambda tp: groups)
    with gpu.GPUGroupLease(groups[0]), gpu.acquire_gpu_group(4) as alternate:
        assert alternate.gpu_ids == groups[1]
        with pytest.raises(RuntimeError, match="unleased"):
            gpu.acquire_gpu_group(4)
    with gpu.acquire_gpu_group(4) as primary:
        assert primary.gpu_ids == groups[0]


def test_worker_startup_failure_releases_lease(tmp_path, monkeypatch):
    from vllm.video import engine

    monkeypatch.setattr(gpu, "LOCK_ROOT", tmp_path)
    monkeypatch.setattr(gpu, "_available_gpu_groups", lambda tp: [(0, 1, 2, 3)])

    def fail_pipe():
        raise RuntimeError("worker startup failed")

    monkeypatch.setattr(
        engine.mp, "get_context", lambda _: SimpleNamespace(Pipe=fail_pipe)
    )
    with pytest.raises(RuntimeError, match="worker startup failed"):
        engine.H3Engine(H3Config())
    with gpu.acquire_gpu_group(4):
        pass
