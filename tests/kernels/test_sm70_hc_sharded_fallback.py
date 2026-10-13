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


@pytest.mark.parametrize("tokens", [5, 20, 512])
@pytest.mark.parametrize("order", [(0, 1, 2, 3), (0, 2, 1, 3)])
def test_hcx_sharded_fallback_graph_matches_rank_reorder(tokens, order, monkeypatch):
    communicator = SimpleNamespace(status={"enabled": True}, order=order)

    def gather(value, dim):
        return torch.cat([value * (rank + 1) for rank in range(4)], dim=dim)

    group = SimpleNamespace(
        device_communicator=SimpleNamespace(hc_ll_comm=communicator),
        all_gather=gather,
    )
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: group)
    down_weight = torch.randn(336, 10240, device="cuda", dtype=torch.float16) * 0.002
    up_weight = torch.randn(10240, 320, device="cuda", dtype=torch.float16) * 0.002
    down, up = pack_down(down_weight, 0), pack_up(up_weight, 0)
    local_down, local_up = hc._unpack_hc_storage(down, up)
    index = torch.tensor(order, device="cuda", dtype=torch.int64)
    x = torch.randn(tokens, 10240, device="cuda", dtype=torch.float16)

    def reference():
        local = x @ local_down.T
        packet = gather(local, -1).reshape(tokens, 4, 88).index_select(1, index)
        lora = hc_silu(packet[:, :, :80].reshape(tokens, 320), 4)
        injection = packet[:, 3, 80:84].contiguous()
        local_gate = lora @ local_up.T
        gate = gather(local_gate, -1).reshape(tokens, 4, 4, 640)
        gate = gate.index_select(1, index).permute(0, 2, 1, 3).reshape(tokens, 10240)
        return hc_gate_mix(x, gate, 4), injection

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
