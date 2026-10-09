# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.config.observability import ObservabilityConfig
from vllm.config.sm70_runtime import SpecDecodeTraceConfig
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu.spec_decode.target_sampling import ComputedTargetLogits


@pytest.mark.parametrize("outcome", ["handled", "dense", "non_gather", "ordinary"])
@pytest.mark.parametrize("grammar", [False, True])
def test_runner_target_protocol_reuses_completed_projection(
    monkeypatch, outcome, grammar
):
    calls = []
    logits = torch.randn(2, 8)
    output = SamplerOutput(
        sampled_token_ids=torch.tensor([[3, 4]]),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=torch.tensor([2]),
    )
    result = {
        "handled": output,
        "dense": ComputedTargetLogits(logits),
        "non_gather": ComputedTargetLogits(None),
        "ordinary": None,
    }[outcome]
    runner = model_runner.GPUModelRunner.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = SimpleNamespace(kernel_config=KernelConfig())
    runner.device = torch.device("cpu")
    runner.lora_config = object()  # Graph admission is passed to the feature owner.

    def project(hidden):
        calls.append("project")
        return logits

    runner.model = SimpleNamespace(compute_logits=Mock(side_effect=project))

    def sample_target(model, sampler, hidden, batch, constraints, *, allow_graph):
        assert not allow_graph
        calls.append("feature")
        return result

    runner.speculator = SimpleNamespace(
        try_sample_target=sample_target,
        trace_target_logits=Mock(),
        trace_target_output=Mock(),
        draft_logits=object(),
    )

    def reject(actual, batch, draft_logits):
        calls.append("reject")
        assert actual is (None if outcome == "non_gather" else logits)
        assert draft_logits is runner.speculator.draft_logits
        return output

    runner.rejection_sampler = Mock(side_effect=reject)
    runner.structured_outputs_worker = SimpleNamespace(
        apply_grammar_bitmask=Mock(side_effect=lambda *args: calls.append("grammar"))
    )
    runner.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.tensor([1]))
    )
    batch = SimpleNamespace(
        num_draft_tokens=1,
        logits_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([4]),
        cu_num_logits=torch.tensor([0, 2]),
        idx_mapping=torch.tensor([0]),
    )
    constraints = (
        SimpleNamespace(structured_output_request_ids=["r"], grammar_bitmask=object())
        if grammar
        else None
    )
    counts = (torch.tensor([2]), torch.tensor([0]))
    monkeypatch.setattr(
        model_runner, "get_num_sampled_and_rejected", lambda *args: counts
    )
    actual = runner.sample(torch.randn(2, 4), batch, constraints)
    assert actual[0] is output
    assert actual[1:] == counts
    expected = ["feature"]
    if outcome != "handled":
        if outcome == "ordinary":
            expected.append("project")
        if grammar:
            expected.append("grammar")
        expected.append("reject")
    assert calls == expected
    assert runner.model.compute_logits.call_count == (outcome == "ordinary")
    assert runner.speculator.trace_target_logits.call_count == (outcome != "handled")
    runner.speculator.trace_target_output.assert_called_once()


def test_base_speculator_preserves_normal_fallback():
    assert BaseSpeculator.try_sample_target(None, None, None, None, None, None) is None


def test_trace_policy_isolated_and_does_not_change_hash(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH_DEBUG_TARGET_LOGITS", "1")
    monkeypatch.setenv("VLLM_DFLASH_DEBUG_TARGET_TRACE_MIN_POSITION", "11")
    first = ObservabilityConfig()
    monkeypatch.setenv("VLLM_DFLASH_DEBUG_TARGET_LOGITS", "0")
    monkeypatch.setenv("VLLM_DFLASH_DEBUG_TARGET_TRACE_MIN_POSITION", "invalid")
    second = ObservabilityConfig()
    assert first.spec_decode_trace.resolve().target_min_position == 11
    assert first.spec_decode_trace.target_logits
    assert not second.spec_decode_trace.target_logits
    assert second.spec_decode_trace.resolve().target_min_position == 8
    assert first.compute_hash() == second.compute_hash()
    explicit = SpecDecodeTraceConfig(
        target_logits=True, target_min_position=4
    ).resolve()
    assert explicit.target_logits and explicit.target_min_position == 4
    assert set(explicit.sources.values()) == {"typed"}
    monkeypatch.setenv("VLLM_DFLASH_DEBUG_TARGET_LOGITS", "1")
    with pytest.raises(ValueError):
        SpecDecodeTraceConfig().resolve()
