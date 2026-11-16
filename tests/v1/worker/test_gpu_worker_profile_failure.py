# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core import kv_cache_utils
from vllm.v1.worker import gpu_worker


@pytest.mark.parametrize("driver_info_fails", [False, True])
@pytest.mark.parametrize("allocator_info_fails", [False, True])
def test_profile_failure_reports_memory_and_preserves_error(
    monkeypatch, driver_info_fails, allocator_info_fails
):
    failure = torch.AcceleratorError("CUDA error: out of memory")

    def profile():
        raise failure

    def driver_info(device):
        if driver_info_fails:
            raise RuntimeError("driver unavailable after accelerator failure")
        return 1000, 2000

    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: False)
    # No attention layers: the workspace bounds are not what this test checks.
    monkeypatch.setattr(gpu_worker, "configure_attention_workspace", lambda *a: None)
    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_groups", lambda *a: [])
    monkeypatch.setattr(gpu_worker.current_platform, "mem_get_info", driver_info)

    def allocator_info(device):
        if allocator_info_fails:
            raise RuntimeError("allocator unavailable after accelerator failure")
        return {
            "allocated_bytes.all.current": 123,
            "reserved_bytes.all.current": 456,
            "inactive_split_bytes.all.current": 789,
        }

    monkeypatch.setattr(torch.accelerator, "memory_stats", allocator_info)
    messages = []
    monkeypatch.setattr(
        gpu_worker.logger, "error", lambda fmt, *args: messages.append(fmt % args)
    )
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker.device = torch.device("cpu")
    worker.vllm_config = SimpleNamespace()
    worker.model_runner = SimpleNamespace(profile_run=profile, get_kv_cache_spec=dict)
    with pytest.raises(torch.AcceleratorError) as raised:
        worker.determine_available_memory()
    assert raised.value is failure
    assert len(messages) == 1
    allocator_expected = (
        "allocated=None reserved=None inactive_split=None"
        if allocator_info_fails
        else "allocated=123 reserved=456 inactive_split=789"
    )
    assert allocator_expected in messages[0]
    expected = (
        "driver_free=None driver_total=None"
        if driver_info_fails
        else ("driver_free=1000 driver_total=2000")
    )
    assert expected in messages[0]
