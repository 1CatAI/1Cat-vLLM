# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("tokens", [5, 20])
def test_capture_releases_eager_cache_without_releasing_target_graph(tokens):
    x = torch.randn(tokens, 2560, device="cuda", dtype=torch.float16)
    target_graph = torch.cuda.CUDAGraph()
    observations = {}

    class TargetManager:
        def needs_capture(self):
            return True

        def capture(self, *args, **kwargs):
            with torch.cuda.graph(target_graph):
                y = x * 2
            observations["target_output"] = y
            observations["target_pointer"] = y.data_ptr()
            # A different capture pool must not retain the target's eager
            # warmup allocation merely because the allocator cached it.
            warmup = torch.empty(128 * 1024**2, device="cuda", dtype=torch.uint8)
            warmup.zero_()
            torch.accelerator.synchronize()
            del warmup
            observations["free_after_target"] = torch.cuda.mem_get_info()[0]
            return {"target": y}

    class Draft:
        def capture(self, states):
            assert states["target"] is observations["target_output"]
            assert torch.cuda.mem_get_info()[0] >= (
                observations["free_after_target"] + 64 * 1024**2
            )
            draft_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(draft_graph):
                z = states["target"] + 1
            for seed in (7, 11, 19):
                torch.manual_seed(seed)
                x.copy_(torch.randn_like(x))
                target_graph.replay()
                draft_graph.replay()
                torch.accelerator.synchronize()
                assert states["target"].data_ptr() == observations["target_pointer"]
                torch.testing.assert_close(states["target"], x * 2, rtol=0, atol=0)
                torch.testing.assert_close(z, x * 2 + 1, rtol=0, atol=0)

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.cudagraph_manager = TargetManager()
    runner.model = SimpleNamespace()
    runner._ple_offload_connector = None
    runner.speculator = Draft()
    runner.maybe_setup_dummy_loras = lambda config: nullcontext()
    for name in (
        "model_state",
        "input_buffers",
        "intermediate_tensors",
        "block_tables",
        "attn_groups",
        "kv_cache_config",
        "lora_config",
    ):
        setattr(runner, name, None)
    runner.use_aux_hidden_state_outputs = False
    runner.capture_model()
