# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank NUMA binding parses its spec and degrades without risk."""

from unittest import mock

import pytest

import vllm.envs as envs
import vllm.v1.worker.gpu_worker as gpu_worker
from vllm.v1.worker.gpu_worker import Worker, _sm70_parse_rank_numa_spec


def test_parse_maps_rank_ranges_to_nodes():
    spec = "0-3:0;4-7:1"
    assert [_sm70_parse_rank_numa_spec(spec, rank) for rank in range(8)] == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]


def test_parse_accepts_single_ranks_and_unmapped_gaps():
    spec = "0:0;7:1"
    assert _sm70_parse_rank_numa_spec(spec, 0) == 0
    assert _sm70_parse_rank_numa_spec(spec, 4) is None
    assert _sm70_parse_rank_numa_spec(spec, 7) == 1


def test_parse_rejects_malformed_segments():
    with pytest.raises(ValueError, match="malformed"):
        _sm70_parse_rank_numa_spec("0-3", 1)


def test_bind_is_noop_without_spec(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_SM70_RANK_NUMA_BIND", raising=False)
    worker = object.__new__(Worker)
    worker.local_rank = 0
    with mock.patch.object(gpu_worker.os, "sched_setaffinity") as setaffinity:
        worker._sm70_bind_rank_numa()
    setaffinity.assert_not_called()


def test_bind_applies_affinity_and_policy(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_RANK_NUMA_BIND", "4-7:1")
    worker = object.__new__(Worker)
    worker.local_rank = 5
    monkeypatch.setattr(
        gpu_worker.glob,
        "glob",
        lambda pattern: ["/sys/fake/node0/cpulist", "/sys/fake/node1/cpulist"],
    )
    monkeypatch.setattr(gpu_worker, "_sm70_node_cpus", lambda node: {4, 5, 6, 7})
    monkeypatch.setattr(gpu_worker, "_sm70_set_mempolicy", lambda node: 0)
    with mock.patch.object(gpu_worker.os, "sched_setaffinity") as setaffinity:
        worker._sm70_bind_rank_numa()
    setaffinity.assert_called_once_with(0, {4, 5, 6, 7})
