# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.sm70_round_cost import RouteRecorder, ordered_route_records, tensor_inventory


def test_ring_wrap_reports_lost_records():
    ids = torch.full((3, 20, 10), -1, dtype=torch.int32)
    rows = torch.tensor([5, 1, 5])
    for slot in range(3):
        ids[slot, : rows[slot]] = slot
    result = ordered_route_records(ids, rows, 5)
    assert result["dropped"] == 2
    assert [r["ordinal"] for r in result["records"]] == [2, 3, 4]
    assert [r["ids"][0][0] for r in result["records"]] == [2, 0, 1]
    assert all(r["unique_experts"] == 1 for r in result["records"])


def test_storage_aliases_are_explicit():
    module = torch.nn.Module()
    weight = torch.nn.Parameter(torch.zeros(8, 8))
    module.register_parameter("weight", weight)
    module.register_buffer("view", weight.detach()[:, :4])
    inventory = tensor_inventory(module)
    assert inventory[0]["storage_id"] == inventory[1]["storage_id"]
    assert inventory[1]["storage_alias"]
    assert inventory[1]["logical_bytes"] == 128
    assert inventory[1]["storage_bytes"] == 256


def test_unregistered_packed_operand_and_workspace_are_inventoried():
    module = torch.nn.Module()
    module.packed_down = torch.ones(2, 32, dtype=torch.float16)
    module.workspace_view = module.packed_down[:, :16]
    inventory = {record["name"]: record for record in tensor_inventory(module)}
    assert inventory["packed_down"]["logical_bytes"] == 128
    assert inventory["workspace_view"]["storage_alias"]


def test_cpu_offload_gate_does_not_create_cuda_recorder(monkeypatch):
    import vllm.config
    import vllm.sm70_round_cost
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    config = SimpleNamespace(
        kernel_config=SimpleNamespace(sm70_round_cost_diagnostics=True)
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)

    def reject_allocation(*args):
        raise AssertionError("CPU offload model must not acquire a CUDA context")

    monkeypatch.setattr(
        vllm.sm70_round_cost, "attach_route_recorder", reject_allocation
    )
    monkeypatch.setattr(MoERunner, "_select_forward", lambda self: None)
    MoERunner(
        "cpu_offload",
        object(),
        object(),
        None,
        torch.nn.Linear(4, 10),
        None,
        object(),
        False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_changed_route_graph_replays_and_reset():
    recorder = RouteRecorder("test")
    ids = torch.arange(50, device="cuda", dtype=torch.int32).reshape(5, 10)
    recorder.capture(ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        recorder.capture(ids)
    recorder.counter.zero_()
    for index in range(3):
        ids.add_(10)
        graph.replay()
    torch.accelerator.synchronize()
    result = ordered_route_records(
        recorder.ids.cpu(), recorder.rows.cpu(), int(recorder.counter.item())
    )
    assert result["total"] == 3 and result["dropped"] == 0
    assert [r["ids"][0][0] for r in result["records"]] == [10, 20, 30]
    assert all(r["unique_experts"] == 50 for r in result["records"])


def test_selection_counts_mask_causality_and_request_union():
    from vllm.sm70_round_cost import ordered_selection_records

    ids = torch.tensor(
        [[[0, 1, 1, -1], [1, 2, 3, 10]], [[0, 1, -1, -1], [0, 2, -1, -1]]]
    )
    rows = torch.tensor([2, 2])
    positions = torch.tensor([[2, 3], [1, 2]])
    requests = torch.tensor([[0, 0], [0, 1]])
    cumulative = torch.tensor([[3, 2, 0], [6, 3, 1]])
    result = ordered_selection_records(ids, rows, 2, cumulative, requests, positions)
    first, second = result["records"]
    assert first["valid_per_row"] == [3, 3]
    assert first["unique_per_row"] == [2, 3]
    assert first["logical_union_single_request"] == 4
    assert second["logical_union_single_request"] is None
    assert second["cache_hit_miss_contention_delta"] == [3, 1, 1]


def test_selection_ring_loss_has_unknown_initial_cache_delta():
    from vllm.sm70_round_cost import ordered_selection_records

    ids = torch.ones(2, 1, 4, dtype=torch.int32)
    rows = torch.ones(2, dtype=torch.int32)
    requests = torch.zeros(2, 1, dtype=torch.int32)
    positions = torch.full((2, 1), 5)
    # Ordinal 2 wraps to slot 0; only ordinals 1 and 2 remain.
    cumulative = torch.tensor([[12, 2, 0], [8, 2, 0]])
    result = ordered_selection_records(ids, rows, 3, cumulative, requests, positions)
    assert result["dropped"] == 1
    assert result["records"][0]["cache_hit_miss_contention_delta"] is None
    assert result["records"][1]["cache_hit_miss_contention_delta"] == [4, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_selection_and_cache_counter_graph_replay():
    from vllm.sm70_round_cost import SelectionRecorder, ordered_selection_records

    recorder = SelectionRecorder("test_selection", 2051, "cuda")
    ids = (
        torch.arange(5 * 2051, device="cuda", dtype=torch.int32).reshape(5, 2051) % 4096
    )
    requests = torch.zeros(5, dtype=torch.int32, device="cuda")
    positions = torch.full((5,), 8192, dtype=torch.int64, device="cuda")
    stats = torch.zeros(20 * 129, 3, dtype=torch.int64, device="cuda")
    recorder.capture(ids, requests, positions)
    recorder.capture_cache(stats, 5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        recorder.capture(ids, requests, positions)
        stats.add_(1)
        recorder.capture_cache(stats, 5)
    recorder.counter.zero_()
    stats.zero_()
    for _ in range(3):
        ids.add_(1)
        graph.replay()
    torch.accelerator.synchronize()
    result = ordered_selection_records(
        recorder.ids.cpu(),
        recorder.rows.cpu(),
        3,
        recorder.cache_totals.cpu(),
        recorder.requests.cpu(),
        recorder.positions.cpu(),
    )
    assert [r["logical_union_single_request"] for r in result["records"]] == [
        4096,
        4096,
        4096,
    ]
    assert all(
        r["cache_hit_miss_contention_delta"] == [2580, 2580, 2580]
        for r in result["records"]
    )
