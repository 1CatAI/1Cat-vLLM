# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen4Exp MTP drafter is stage-local and must not branch on the target
model's pipeline position.

``gpu_model_runner.execute_model`` returns the IntermediateTensors on every
non-final pipeline rank before speculation is reached, so the drafter only ever
runs on the last rank -- where ``get_pp_group().is_first_rank`` is False. Taking
the "receive from the previous stage" path there asserts on intermediate
tensors that nobody sends, and under the fullgraph AOT compile that is a hard
compile error rather than a runtime one: no k > 0 boots at all under pipeline
parallelism.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMultiTokenPredictor

HC_COUNT = 2
HIDDEN = 4
TOKENS = 3


class _Layer:
    """Return the three-tuple the HC decoder produces."""

    def __call__(self, *, hidden_states: torch.Tensor, **_kwargs):
        return (
            hidden_states,
            torch.ones_like(hidden_states),
            (torch.zeros_like(hidden_states)),
        )


class _Mixer:
    def combine_and_mix(self, hidden_states, block_output, injection):
        multi = hidden_states + block_output
        sample = multi[..., :HIDDEN]
        return multi, sample, injection


def _predictor() -> Qwen4ExpMultiTokenPredictor:
    p = object.__new__(Qwen4ExpMultiTokenPredictor)
    object.__setattr__(p, "hc_count", HC_COUNT)
    object.__setattr__(p, "hidden_size", HIDDEN)
    object.__setattr__(p, "num_mtp_layers", 1)
    object.__setattr__(p, "layers", [_Layer()])
    object.__setattr__(p, "hyper_connection_mixer", _Mixer())
    object.__setattr__(p, "embed_tokens", lambda ids: torch.zeros(TOKENS, HIDDEN))
    object.__setattr__(p, "pre_fc_norm_embedding", nn.Identity())
    object.__setattr__(p, "fc_embedding", nn.Identity())
    object.__setattr__(p, "pre_fc_norm_hidden", nn.Identity())
    object.__setattr__(p, "fc_hidden", nn.Identity())
    return p


@pytest.fixture
def last_pp_rank(monkeypatch):
    """A pipeline group whose final rank is the one running the drafter."""
    from vllm.models.qwen4_exp.nvidia import mtp as mtp_module

    group = SimpleNamespace(is_first_rank=False, is_last_rank=True)
    monkeypatch.setattr(mtp_module, "get_pp_group", lambda: group)
    return group


def test_drafter_embeds_on_the_last_pipeline_rank(last_pp_rank) -> None:
    """Without intermediate tensors the drafter must still build its own
    embedding instead of asserting on a hand-off that never happens."""
    predictor = _predictor()

    sample_hidden_states, multi_hidden = predictor.forward(
        input_ids=torch.zeros(TOKENS, dtype=torch.long),
        positions=torch.arange(TOKENS),
        hidden_states=torch.zeros(TOKENS, HC_COUNT * HIDDEN),
        intermediate_tensors=None,
        spec_step_idx=0,
    )

    assert sample_hidden_states.shape == (TOKENS, HIDDEN)
    assert multi_hidden.shape == (TOKENS, HC_COUNT * HIDDEN)


def test_drafter_does_not_hand_off_to_a_next_stage(monkeypatch) -> None:
    """Even with a pipeline group that reports a following stage, the drafter
    finalizes locally: it is replicated, not partitioned."""
    from vllm.models.qwen4_exp.nvidia import mtp as mtp_module

    group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    monkeypatch.setattr(mtp_module, "get_pp_group", lambda: group)
    predictor = _predictor()

    result = predictor.forward(
        input_ids=torch.zeros(TOKENS, dtype=torch.long),
        positions=torch.arange(TOKENS),
        hidden_states=torch.zeros(TOKENS, HC_COUNT * HIDDEN),
        intermediate_tensors=None,
        spec_step_idx=0,
    )

    assert isinstance(result, tuple), "the drafter must not return IntermediateTensors"
    assert len(result) == 2
