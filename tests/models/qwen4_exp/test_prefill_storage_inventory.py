# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.sm70_graph_observer import GraphParityWorkerExtension
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_inventory_deduplicates_device_views_and_excludes_host_aliases(monkeypatch):
    from vllm.model_executor.layers.quantization import gguf_dense_hmma

    runner = torch.nn.Module()
    value = torch.empty(32, device="cuda", dtype=torch.float16)
    runner.register_buffer("value", value)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        runner.fake_metadata = torch.empty(1024, device="cuda")
    runner.alias = value[3:]
    host = torch.empty(48, dtype=torch.uint8, pin_memory=True)
    runner.register_buffer("host_alias", get_accelerator_view_from_cpu_tensor(host))
    native_owned = torch.empty(123457, device="cuda", dtype=torch.uint8)
    worker = GraphParityWorkerExtension()
    worker.model_runner = runner
    worker.rank = 0
    before = dict(gguf_dense_hmma._workspaces)
    gguf_dense_hmma._workspaces.clear()
    try:
        report = worker.read_prefill_storages()
        assert not report["pointer_errors"]
        assert report["device_storage_bytes"] == value.untyped_storage().nbytes()
        assert report["mapped_host_storage_bytes"] == host.untyped_storage().nbytes()
        device = next(
            value for value in report["storages"] if value["memory_type"] == 2
        )
        assert len(device["owners"]) == 2
        assert any(
            block["requested_bytes"] == native_owned.numel()
            for block in report["unreachable_allocator_blocks"]
        )
        from vllm.v1.worker import workspace

        manager = workspace.WorkspaceManager(torch.device("cuda"))
        manager._current_workspaces[0] = native_owned
        monkeypatch.setattr(workspace, "_manager", manager)
        with_manager = worker.read_prefill_storages()
        assert with_manager["device_storage_bytes"] == (
            value.untyped_storage().nbytes() + native_owned.untyped_storage().nbytes()
        )
        assert any(
            "global_workspace_manager._current_workspaces.0" in record["owners"]
            for record in with_manager["storages"]
        )
        assert not any(
            block["requested_bytes"] == native_owned.numel()
            for block in with_manager["unreachable_allocator_blocks"]
        )
    finally:
        gguf_dense_hmma._workspaces.update(before)
