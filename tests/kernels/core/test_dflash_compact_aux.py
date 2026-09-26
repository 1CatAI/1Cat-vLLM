# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The compact draft input must preserve the original projection exactly."""

from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM


class Projection:
    get_aux_hidden_state_dtype = DFlashQwen3ForCausalLM.get_aux_hidden_state_dtype
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


@pytest.mark.parametrize("residual_dtype", [None, torch.float16, torch.float32])
@pytest.mark.parametrize("snapshot_dtype", [None, torch.float16, torch.float32])
@torch.inference_mode()
def test_aux_snapshot_keeps_target_arithmetic_and_storage(
    residual_dtype, snapshot_dtype
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EagleModelMixin()
    model._set_aux_hidden_state_layers((1,))
    model.aux_hidden_state_dtype = snapshot_dtype
    hidden = torch.randn(8192, 32, device=device, dtype=torch.float16)
    residual = (
        torch.randn_like(hidden, dtype=residual_dtype)
        if residual_dtype is not None
        else None
    )
    expected = hidden + residual if residual is not None else hidden
    if (
        snapshot_dtype is not None
        and torch.finfo(snapshot_dtype).bits < 8 * expected.element_size()
    ):
        expected = expected.to(snapshot_dtype)
    expected = expected.clone()
    hidden_before = hidden.clone()
    residual_before = residual.clone() if residual is not None else None

    def capture(hidden, residual):
        return model._maybe_add_hidden_state([], 1, hidden, residual)[0]

    actual = capture(hidden, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if device == "cuda":
        torch._dynamo.reset()
        compiled = torch.compile(capture, fullgraph=True)
        torch.testing.assert_close(compiled(hidden, residual), expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(hidden, residual)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)
    if residual is not None:
        torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)
        residual.zero_()
    hidden.zero_()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
