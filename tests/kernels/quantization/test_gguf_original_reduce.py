# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization import gguf_moe  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("m", [5, 20, 512])
def test_original_expert_reduction_matches_fp32_and_replays(m):
    torch.manual_seed(1167)
    down = torch.randn((m, 10, 2560), dtype=torch.float16, device="cuda")
    weights = torch.randn((m, 10), device="cuda").softmax(-1)
    reference = (down.float() * weights[..., None]).sum(1)
    output = torch.ops.vllm.gguf_original_expert_reduce(down, weights)
    torch.testing.assert_close(output.float(), reference, rtol=5e-4, atol=5e-4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = torch.ops.vllm.gguf_original_expert_reduce(down, weights)
    for _ in range(3):
        down.normal_()
        weights.copy_(torch.randn_like(weights).softmax(-1))
        graph.replay()
        eager = torch.ops.vllm.gguf_original_expert_reduce(down, weights)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def test_prefill_reduction_removes_two_full_fp32_temporaries():
    down = torch.randn((512, 10, 2560), dtype=torch.float16, device="cuda")
    weights = torch.full((512, 10), 0.1, device="cuda")
    # Compile before measuring live Torch allocations.
    torch.ops.vllm.gguf_original_expert_reduce(down, weights)
    torch.accelerator.synchronize()

    def peak(call):
        torch.accelerator.reset_peak_memory_stats()
        before = torch.accelerator.memory_allocated()
        output = call()
        torch.accelerator.synchronize()
        delta = torch.accelerator.max_memory_allocated() - before
        del output
        return delta

    reference_peak = peak(lambda: (down.float() * weights[..., None]).sum(1).half())
    fused_peak = peak(lambda: torch.ops.vllm.gguf_original_expert_reduce(down, weights))
    assert reference_peak - fused_peak >= 97 * 1024**2
    assert fused_peak <= 3 * 1024**2
