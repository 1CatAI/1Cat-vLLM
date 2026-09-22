# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The compact draft input must preserve the original projection exactly."""

from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM


class Projection:
    combine_hidden_states = DFlashQwen3ForCausalLM.combine_hidden_states
    combine_aux_hidden_states = DFlashQwen3ForCausalLM.combine_aux_hidden_states

    def __init__(self, device, dtype):
        fc = torch.nn.Linear(160, 32, bias=False, device=device, dtype=dtype)
        fc.input_size = 160
        self.model = SimpleNamespace(use_aux_hidden_state=True, fc=fc)


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("projection_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("rows", [8, 8192])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_projection_matches_cat_then_cast(
    monkeypatch, source_dtype, projection_dtype, rows, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    envs.disable_envs_cache()
    torch.manual_seed(17)
    model = Projection(device, projection_dtype)
    # Non-contiguous states exercise the copy path used by strided inputs too.
    aux = [
        torch.randn(rows, 64, device=device, dtype=source_dtype)[:, ::2]
        for _ in range(5)
    ]
    originals = [x.clone() for x in aux]
    monkeypatch.setenv("VLLM_DFLASH_COMPACT_AUX_HIDDEN", "0")
    expected = model.combine_aux_hidden_states(aux)
    monkeypatch.setenv("VLLM_DFLASH_COMPACT_AUX_HIDDEN", "1")
    actual = model.combine_aux_hidden_states(aux)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = model.combine_aux_hidden_states(aux)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    for original, hidden in zip(originals, aux):
        torch.testing.assert_close(original, hidden, rtol=0, atol=0)
