# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU state/dispatch checks; GPU arithmetic is covered in runtime_contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample import sampler as sampler_module
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates


def _states():
    state = object.__new__(SamplingStates)
    state.vocab_size = 248320
    for name in ("temperature", "top_p", "top_k", "min_p", "seeds"):
        setattr(state, name, SimpleNamespace(np=np.zeros(3)))
    state.num_logprobs = np.full(3, NO_LOGPROBS, dtype=np.int32)
    return state


@pytest.mark.parametrize("count", [None, -1, 0, 1, 20])
def test_sample_logprob_request_is_normalized_and_reusable(count):
    state = _states()
    state.add_request(2, SamplingParams(logprobs=count, seed=0))
    expected = NO_LOGPROBS if count is None else 248320 if count == -1 else count
    assert state.max_num_logprobs(np.array([2])) == expected
    assert state.num_logprobs[0] == NO_LOGPROBS
    state.add_request(2, SamplingParams(logprobs=None, seed=0))
    assert state.max_num_logprobs(np.array([2])) == NO_LOGPROBS


def test_full_vocabulary_dominates_mixed_batch_count():
    state = _states()
    for slot, count in enumerate([None, 20, -1]):
        state.add_request(slot, SamplingParams(logprobs=count, seed=0))
    assert state.max_num_logprobs(np.array([1, 0])) == 20
    assert state.max_num_logprobs(np.array([1, 2])) == 248320


def test_greedy_dispatch_rejects_full_logprobs_then_recovers(monkeypatch):
    monkeypatch.setattr(sampler_module.envs, "VLLM_SM70_GREEDY_TOKEN_FASTPATH", True)
    state = _states()
    sampler = object.__new__(Sampler)
    sampler.compute_nans = False
    sampler.sampling_states = state
    sampler.logprob_token_ids_state = SimpleNamespace(max_num_token_ids=lambda _: 0)
    sampler.logit_bias_state = SimpleNamespace(use_logit_bias=np.zeros(3, dtype=bool))
    sampler.penalties_state = SimpleNamespace(use_penalty=np.zeros(3, dtype=bool))
    sampler.bad_words_state = SimpleNamespace(
        num_bad_words=SimpleNamespace(np=np.zeros(3, dtype=np.int32))
    )
    batch = SimpleNamespace(idx_mapping_np=np.array([2]))
    state.add_request(2, SamplingParams(temperature=0, logprobs=-1, seed=0))
    assert not sampler.can_use_sm70_greedy_token_fastpath(batch)
    state.add_request(2, SamplingParams(temperature=0, seed=0))
    assert sampler.can_use_sm70_greedy_token_fastpath(batch)


@pytest.mark.parametrize("vocab_size", [257, 248320])
@pytest.mark.parametrize("invalid", [-1, 0, 1024])
def test_custom_logprob_token_ids_reject_out_of_vocabulary(vocab_size, invalid):
    token_id = invalid if invalid < 0 else vocab_size + invalid
    params = SamplingParams(logprob_token_ids=[0, token_id])
    config = SimpleNamespace(max_logprobs=20, get_vocab_size=lambda: vocab_size)
    with pytest.raises(VLLMValidationError, match="logprob_token_ids"):
        params._validate_logprobs(config)


@pytest.mark.parametrize("ids", [[], [0], [256], [0, 256, 0]])
def test_custom_logprob_valid_boundaries_are_accepted(ids):
    params = SamplingParams(logprob_token_ids=ids)
    params._validate_logprobs(
        SimpleNamespace(max_logprobs=20, get_vocab_size=lambda: 257)
    )
