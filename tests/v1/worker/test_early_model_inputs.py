# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize(
    "mode,capability,ple,expected",
    [
        (CUDAGraphMode.FULL, True, True, ["inputs"]),
        (CUDAGraphMode.FULL, True, False, ["inputs"]),
        (CUDAGraphMode.FULL, False, True, []),
        (CUDAGraphMode.FULL, False, False, []),
        (CUDAGraphMode.NONE, True, True, ["inputs", "ple"]),
        (CUDAGraphMode.NONE, False, True, ["inputs", "ple"]),
        (CUDAGraphMode.NONE, True, False, []),
    ],
)
def test_early_phase_preserves_ple_request_ownership(mode, capability, ple, expected):
    calls = []
    inputs = {"positions": object()}

    def prepare_inputs(batch, states):
        calls.append("inputs")
        return inputs

    def prepare_forward(reqs, tokens, dummy_run):
        assert (reqs, tokens, dummy_run) == (1, 5, False)
        calls.append("ple")

    runner = SimpleNamespace(
        model_state=SimpleNamespace(
            supports_early_input_preparation=capability, prepare_inputs=prepare_inputs
        ),
        req_states=None,
        _ple_offload_connector=(
            SimpleNamespace(prepare_forward=prepare_forward) if ple else None
        ),
    )
    batch = SimpleNamespace(num_reqs=1, num_tokens_after_padding=5)
    prepared, submitted = GPUModelRunner.prepare_model_inputs_early(runner, batch, mode)
    assert calls == expected
    assert (prepared is inputs) == bool(expected)
    assert submitted == ("ple" in expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reqs", [1, 4])
def test_early_rope_keeps_stream_order_and_changed_request_positions(reqs):
    rope = RopeState(3, True, 4, 20, 64, torch.device("cuda"))
    rope.prefill_delta.np[:] = [7, 1, 2, 0]
    rope.apply_staged_writes()
    mapping = torch.tensor([2, 0, 3, 1][:reqs], device="cuda", dtype=torch.int32)
    offsets = torch.arange(reqs + 1, device="cuda", dtype=torch.int32) * 5
    lengths = torch.zeros(4, device="cuda", dtype=torch.int32)
    computed = torch.tensor([10, 12, 20, 23], device="cuda", dtype=torch.int32)

    def enqueue_positions():
        rope.prepare_positions(mapping, offsets, lengths, computed)
        return rope.get_positions(reqs * 5)

    # Control: metadata producer before positions. Candidate: positions before
    # metadata. The metadata launch touches a separate buffer on the same stream.
    marker = torch.zeros(32, device="cuda", dtype=torch.int32)
    marker.add_(1)
    control = enqueue_positions().clone()
    candidate = enqueue_positions()
    marker.add_(1)
    torch.testing.assert_close(candidate, control, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = enqueue_positions().clone()
        marker.add_(1)
    computed.add_(5)
    graph.replay()
    torch.testing.assert_close(captured, control + 5, rtol=0, atol=0)
