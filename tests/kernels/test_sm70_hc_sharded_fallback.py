# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu
from vllm.models.qwen4_exp.nvidia.sm70_hcx import pack_down, pack_up

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("tokens", [5, 20, 21, 512])
@pytest.mark.parametrize("order", [(0, 1, 2, 3), (0, 2, 1, 3)])
@pytest.mark.parametrize("storage", ["hcx", "ll"])
def test_hcx_sharded_fallback_graph_matches_rank_reorder(
    tokens, order, storage, monkeypatch
):
    communicator = SimpleNamespace(
        status={"enabled": True}, order=order, apply=lambda *_: None
    )

    group = SimpleNamespace(
        rank_in_group=0,
        device_communicator=SimpleNamespace(hc_ll_comm=communicator),
    )
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: group)
    down_weight = torch.randn(336, 10240, device="cuda", dtype=torch.float16) * 0.002
    up_weight = torch.randn(10240, 320, device="cuda", dtype=torch.float16) * 0.002
    packs = []
    for rank in range(4):
        if storage == "hcx":
            packs.append((pack_down(down_weight, rank), pack_up(up_weight, rank)))
        else:
            packs.append(
                (
                    hc._pack_hc_batch_weight(down_weight, "down", rank),
                    hc._pack_hc_batch_weight(up_weight, "up", rank),
                )
            )
    down, up = packs[0]
    local_weights = [hc._unpack_hc_storage(*pack) for pack in packs]
    x = torch.randn(tokens, 10240, device="cuda", dtype=torch.float16)

    def reference():
        packets = [x @ weights[0].T for weights in local_weights]
        lora = hc_silu(torch.cat([packet[:, :80] for packet in packets], dim=1), 4)
        gate = torch.cat(
            [
                (lora @ weights[1].T).reshape(tokens, 4, 640)
                for weights in local_weights
            ],
            dim=2,
        ).reshape(tokens, 10240)
        return hc_gate_mix(x, gate, 4), packets[3][:, 80:84].contiguous()

    def gather(value, dim):
        assert dim == 0
        # Model each peer's projection and input columns before its nonlinear
        # mix, and keep the actual local packet as part of the collective.
        if value.shape[1] == 88:
            logical = [x @ weights[0].T for weights in local_weights]
        else:
            assert value.shape[1] == 640
            block, _ = reference()
            logical = list(block.split(640, dim=1))
        physical = [logical[order.index(rank)] for rank in range(4)]
        physical[group.rank_in_group] = value
        return torch.cat(physical, dim=0)

    group.all_gather = gather

    # Compile the unchanged arithmetic before capture, as model warmup does.
    expected = reference()
    eager = hc._sharded_hc_project(x, down, up)
    for got, want in zip(eager, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = hc._sharded_hc_project(x, down, up)
    for _ in range(3):
        x.normal_()
        graph.replay()
        for got, want in zip(captured, reference()):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_identity_hc_order_keeps_storage():
    value = torch.randn(5, 4, 88, device="cuda", dtype=torch.float16)
    assert hc._logical_hc_rank_rows(value, (0, 1, 2, 3)) is value


@pytest.mark.parametrize("tokens", [512, 8192])
def test_sharded_prefill_releases_projection_temporaries(
    tokens, monkeypatch, record_property
):
    from vllm.models.qwen4_exp.nvidia.ops.hc import (
        hc_assemble_ranks,
        hc_gate_mix_local,
        hc_packet_silu,
    )

    order = (0, 1, 2, 3)
    group = SimpleNamespace(
        rank_in_group=0,
        device_communicator=SimpleNamespace(
            hc_ll_comm=SimpleNamespace(
                status={"enabled": True}, order=order, apply=lambda *_: None
            )
        ),
        all_gather=lambda value, dim: torch.cat([value] * 4, dim=dim),
    )
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: group)
    down = pack_down(torch.randn(336, 10240, device="cuda", dtype=torch.float16), 0)
    up = pack_up(torch.randn(10240, 320, device="cuda", dtype=torch.float16), 0)
    x = torch.randn(tokens, 10240, device="cuda", dtype=torch.float16) * 0.001

    def retained_temporaries():
        # The original fallback: all Python locals survive until assembly.
        local_down, local_up = hc._unpack_hc_storage(down, up)
        local = torch.mm(x, local_down.T)
        packet = group.all_gather(local, dim=0)
        lora, injection = hc_packet_silu(packet, order, 4)
        local_gate = torch.mm(lora, local_up.T)
        mixed = hc_gate_mix_local(x, local_gate, 0, 4)
        gathered = group.all_gather(mixed, dim=0)
        return hc_assemble_ranks(gathered, order), injection

    # Compile the same kernels before measuring either allocation sequence.
    retained_temporaries()
    hc._sharded_hc_project(x, down, up)
    torch.cuda.synchronize()

    def measure(operation):
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = operation()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        return peak, tuple(value.cpu() for value in result)

    old_peak, expected = measure(retained_temporaries)
    new_peak, actual = measure(lambda: hc._sharded_hc_project(x, down, up))
    record_property("old_peak_bytes", old_peak)
    record_property("new_peak_bytes", new_peak)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert all(torch.isfinite(value).all() for value in actual)
    # At least the full local gate is dead before gather/assembly allocation.
    assert old_peak - new_peak >= tokens * 2560 * x.element_size()
