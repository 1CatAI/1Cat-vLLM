# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_expert_trace import ExpertTrace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def test_decode_ring_and_prefill_histogram():
    trace = ExpertTrace("cuda", num_layers=3, ring_tokens=8, top_k=2, num_experts=6)
    for layer in (0, 1, 2):
        ids = torch.tensor([[layer, 5], [1, 2]], dtype=torch.int32, device="cuda")
        trace.record(layer, ids)
        trace.record(layer, ids)
    big = torch.randint(0, 6, (40, 2), dtype=torch.int32, device="cuda")
    trace.record(1, big)
    snap = trace.snapshot()
    assert snap["counter"].tolist() == [4, 4, 4]
    for layer in (0, 1, 2):
        expected = [[layer, 5], [1, 2], [layer, 5], [1, 2]]
        assert snap["ring"][layer, :4].tolist() == expected
        assert (snap["ring"][layer, 4:] == -1).all()
    assert (
        snap["prefill"][1].tolist()
        == torch.bincount(big.flatten().cpu(), minlength=6).tolist()
    )
    assert snap["prefill"][0].sum() == 0 and snap["prefill"][2].sum() == 0


def test_ring_wraps():
    trace = ExpertTrace("cuda", num_layers=1, ring_tokens=4, top_k=1, num_experts=4)
    for value in range(6):
        trace.record(0, torch.tensor([[value % 4]], dtype=torch.int32, device="cuda"))
    snap = trace.snapshot()
    assert int(snap["counter"][0]) == 6
    assert snap["ring"][0, :, 0].tolist() == [0, 1, 2, 3][::1][:0] + [0, 1, 0, 1][
        :2
    ] + [2, 3]
