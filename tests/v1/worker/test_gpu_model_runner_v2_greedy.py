# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.sample.sampler as sampler_module
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler


def _sampler() -> Sampler:
    sampler = object.__new__(Sampler)
    sampler.compute_nans = False
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.array([0.0, 0.0], dtype=np.float32)),
        max_num_logprobs=lambda _indices: -1,
    )
    sampler.logprob_token_ids_state = SimpleNamespace(
        max_num_token_ids=lambda _indices: 0
    )
    sampler.logit_bias_state = SimpleNamespace(use_logit_bias=np.array([False, False]))
    sampler.penalties_state = SimpleNamespace(use_penalty=np.array([False, False]))
    sampler.bad_words_state = SimpleNamespace(
        num_bad_words=SimpleNamespace(np=np.array([0, 0], dtype=np.int32))
    )
    return sampler


def test_sm70_v2_greedy_fastpath_accepts_plain_temperature_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sampler_module.envs, "VLLM_SM70_GREEDY_TOKEN_FASTPATH", True)
    sampler = _sampler()
    input_batch = SimpleNamespace(idx_mapping_np=np.array([1], dtype=np.int32))

    assert sampler.can_use_sm70_greedy_token_fastpath(input_batch)


def test_sm70_v2_decode_uses_model_top_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
            assert hidden_states.shape == (1, 4)
            return torch.tensor([42], dtype=torch.int64)

        def compute_logits(self, _hidden_states: torch.Tensor) -> torch.Tensor:
            raise AssertionError("full logits must not run on the greedy fastpath")

    runner = object.__new__(GPUModelRunner)
    runner.model = Model()
    runner.lora_config = None
    runner.sampler = SimpleNamespace(
        can_use_sm70_greedy_token_fastpath=lambda _input_batch: True
    )
    runner.device = torch.device("cuda")
    runner.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=None))
    input_batch = SimpleNamespace(
        logits_indices=torch.tensor([0]),
        num_draft_tokens=0,
        num_reqs=1,
        num_tokens=1,
        is_prefilling_np=np.array([False]),
        seq_lens=torch.tensor([17], dtype=torch.int32),
        cu_num_logits=None,
        idx_mapping=None,
    )
    monkeypatch.setattr(
        model_runner_module.current_platform,
        "is_device_capability",
        lambda _capability: True,
    )
    monkeypatch.setattr(
        model_runner_module,
        "get_num_sampled_and_rejected",
        lambda *_args: (
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        ),
    )

    output, num_sampled, num_rejected = GPUModelRunner.sample(
        runner, torch.zeros(1, 4), input_batch, None
    )

    assert output.sampled_token_ids.tolist() == [[42]]
    assert num_sampled.tolist() == [1]
    assert num_rejected.tolist() == [0]

    runner.lora_config = object()
    with pytest.raises(AssertionError, match="full logits must not run"):
        GPUModelRunner.sample(runner, torch.zeros(1, 4), input_batch, None)


@pytest.mark.parametrize(
    "blocker",
    [
        "disabled",
        "nan_count",
        "random",
        "logprobs",
        "logprob_token_ids",
        "logit_bias",
        "penalty",
        "bad_words",
    ],
)
def test_sm70_v2_greedy_fastpath_rejects_non_equivalent_sampling(
    monkeypatch: pytest.MonkeyPatch, blocker: str
) -> None:
    monkeypatch.setattr(
        sampler_module.envs,
        "VLLM_SM70_GREEDY_TOKEN_FASTPATH",
        blocker != "disabled",
    )
    sampler = _sampler()
    input_batch = SimpleNamespace(idx_mapping_np=np.array([1], dtype=np.int32))

    if blocker == "nan_count":
        sampler.compute_nans = True
    elif blocker == "random":
        sampler.sampling_states.temperature.np[1] = 0.5
    elif blocker == "logprobs":
        sampler.sampling_states.max_num_logprobs = lambda _indices: 1
    elif blocker == "logprob_token_ids":
        sampler.logprob_token_ids_state.max_num_token_ids = lambda _indices: 1
    elif blocker == "logit_bias":
        sampler.logit_bias_state.use_logit_bias[1] = True
    elif blocker == "penalty":
        sampler.penalties_state.use_penalty[1] = True
    elif blocker == "bad_words":
        sampler.bad_words_state.num_bad_words.np[1] = 1

    assert not sampler.can_use_sm70_greedy_token_fastpath(input_batch)


@pytest.mark.parametrize("num_reqs", [1, 4])
@pytest.mark.parametrize(
    "blocker", [None, "disabled", "synthetic", "prefill", "grammar", "lora", "params"]
)
def test_sm70_mtp_compact_reuses_greedy_gate_and_common_bookkeeping(
    monkeypatch, num_reqs, blocker
):
    rows = num_reqs * 5
    ids = torch.arange(rows, dtype=torch.int64)
    output = SamplerOutput(
        sampled_token_ids=torch.zeros(num_reqs, 5, dtype=torch.int64),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=torch.ones(num_reqs, dtype=torch.int32),
    )
    runner = object.__new__(GPUModelRunner)
    runner.model = SimpleNamespace(
        get_top_tokens=Mock(return_value=ids),
        compute_logits=Mock(return_value=torch.zeros(rows, 8)),
    )
    runner._sm70_mtp_target_top1_probe = blocker != "disabled"
    runner._sm70_mtp_target_top1_proof = {"calls": 0, "widths": set()}
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen4_exp")
    )
    runner.speculative_config = SimpleNamespace(method="mtp")
    runner.num_speculative_steps = 4
    runner.speculator = SimpleNamespace(draft_logits=None)
    runner.lora_config = object() if blocker == "lora" else None
    runner.sampler = SimpleNamespace(
        can_use_sm70_greedy_token_fastpath=Mock(return_value=blocker != "params")
    )
    runner.rejection_sampler = Mock(return_value=output)
    runner.rejection_sampler.rejection_sample_method = (
        "synthetic" if blocker == "synthetic" else "standard"
    )
    runner.rejection_sampler.sample_from_top_tokens.return_value = output
    runner.device = torch.device("cuda")
    runner.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=None))
    runner.structured_outputs_worker = SimpleNamespace(apply_grammar_bitmask=Mock())
    cu_np = np.arange(num_reqs + 1, dtype=np.int32) * 5
    batch = SimpleNamespace(
        logits_indices=torch.arange(rows),
        num_draft_tokens=num_reqs * 4,
        num_reqs=num_reqs,
        num_tokens=rows,
        is_prefilling_np=np.array([blocker == "prefill"] * num_reqs),
        cu_num_logits_np=cu_np,
        cu_num_logits=torch.from_numpy(cu_np),
        seq_lens=torch.full((num_reqs,), 17, dtype=torch.int32),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32),
    )
    grammar = (
        SimpleNamespace(structured_output_request_ids=[], grammar_bitmask=None)
        if blocker == "grammar"
        else None
    )
    monkeypatch.setattr(
        model_runner_module, "try_dflash2_sparse_target_rejection", lambda *a, **k: None
    )
    monkeypatch.setattr(
        model_runner_module.current_platform, "is_device_capability", lambda _: True
    )
    counts = torch.arange(num_reqs, dtype=torch.int32)
    rejected = counts + 2
    bookkeeping = Mock(return_value=(counts, rejected))
    monkeypatch.setattr(
        model_runner_module, "get_num_sampled_and_rejected", bookkeeping
    )
    result, actual_counts, actual_rejected = GPUModelRunner.sample(
        runner, torch.zeros(rows, 4), batch, grammar
    )
    assert result is output
    assert actual_counts is counts and actual_rejected is rejected
    bookkeeping.assert_called_once_with(
        output.num_sampled, batch.seq_lens, batch.cu_num_logits, batch.idx_mapping, None
    )
    if blocker is None:
        runner.model.compute_logits.assert_not_called()
        runner.rejection_sampler.assert_not_called()
        runner.rejection_sampler.sample_from_top_tokens.assert_called_once_with(
            ids, batch
        )
        assert runner._sm70_mtp_target_top1_proof == {"calls": 1, "widths": {rows}}
    else:
        runner.model.get_top_tokens.assert_not_called()
        runner.rejection_sampler.sample_from_top_tokens.assert_not_called()
        runner.rejection_sampler.assert_called_once()
