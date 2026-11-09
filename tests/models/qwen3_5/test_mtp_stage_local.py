# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.5 MTP drafter is stage-local and must not branch on the target
model's pipeline position.

``gpu_model_runner.execute_model`` returns the IntermediateTensors on every
non-final pipeline rank before speculation is reached, so the drafter only ever
runs on the last rank -- where ``get_pp_group().is_first_rank`` is False. Taking
the "receive from the previous stage" path there asserts on intermediate
tensors that nobody sends.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.models.qwen3_5_mtp import (
    Qwen3_5MTP,
    Qwen3_5MultiTokenPredictor,
)

HIDDEN = 4
TOKENS = 3


class _Layer:
    """Return the (hidden_states, residual) pair of a decoder layer."""

    def __call__(self, *, hidden_states: torch.Tensor, residual, **_kwargs):
        assert residual is None
        return hidden_states, torch.zeros_like(hidden_states)


class _Norm:
    def __call__(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        return hidden_states + residual, residual


def _predictor() -> Qwen3_5MultiTokenPredictor:
    p = object.__new__(Qwen3_5MultiTokenPredictor)
    object.__setattr__(p, "num_mtp_layers", 1)
    object.__setattr__(p, "layers", [_Layer()])
    object.__setattr__(p, "embed_tokens", lambda ids: torch.zeros(TOKENS, HIDDEN))
    object.__setattr__(p, "pre_fc_norm_embedding", nn.Identity())
    object.__setattr__(p, "pre_fc_norm_hidden", nn.Identity())
    object.__setattr__(p, "fc", lambda x: x[..., :HIDDEN])
    object.__setattr__(p, "norm", _Norm())
    return p


@pytest.mark.parametrize("is_last_rank", [True, False])
def test_drafter_runs_locally_on_any_pipeline_position(
    monkeypatch, is_last_rank: bool
) -> None:
    """Without intermediate tensors the drafter must build its own embedding
    and finalize locally instead of asserting on a hand-off that never happens
    or returning IntermediateTensors for a next stage it does not have."""
    from vllm.model_executor.models import qwen3_5_mtp as mtp_module

    group = SimpleNamespace(is_first_rank=False, is_last_rank=is_last_rank)
    monkeypatch.setattr(mtp_module, "get_pp_group", lambda: group)

    result = _predictor().forward(
        input_ids=torch.zeros(TOKENS, dtype=torch.long),
        positions=torch.arange(TOKENS),
        hidden_states=torch.zeros(TOKENS, HIDDEN),
        intermediate_tensors=None,
        spec_step_idx=0,
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == (TOKENS, HIDDEN)


def test_drafter_declares_pipeline_parallel_support() -> None:
    """The draft model config is verified against the pipeline-parallel config;
    without SupportsPP the engine refuses to start with PP > 1."""
    assert supports_pp(Qwen3_5MTP)
