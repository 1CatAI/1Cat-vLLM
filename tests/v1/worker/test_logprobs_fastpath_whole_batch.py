# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM70 sampling fast paths that return no logprob rows decide per batch.

The greedy token fast path (greedy decode and greedy MTP verify) and the
DFlash2 sparse target rejection produce `logprobs_tensors=None`. That is only
correct if no request in the batch asked for logprobs or logprob token ids:
one such request anywhere in a mixed batch must turn the fast path off for
the whole batch, otherwise that request's output would carry fewer logprob
rows than tokens. These tests lock that, using the real per-batch reductions
(`max_num_logprobs`, `max_num_token_ids`).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
    _supports_sparse_sampling_contract,
)

NO_LOGPROBS = -1
BATCH = np.array([0, 1, 2, 3], dtype=np.int32)


def _states(n, temperature):
    states = object.__new__(SamplingStates)
    states.num_logprobs = np.full(n, NO_LOGPROBS, dtype=np.int32)
    states.temperature = SimpleNamespace(np=np.full(n, temperature, dtype=np.float32))
    states.top_k = SimpleNamespace(np=np.full(n, 20, dtype=np.int32))
    states.top_p = SimpleNamespace(np=np.full(n, 0.95, dtype=np.float32))
    states.min_p = SimpleNamespace(np=np.zeros(n, dtype=np.float32))
    return states


def _token_ids_state(n):
    state = object.__new__(LogprobTokenIdsState)
    state.num_token_ids = SimpleNamespace(np=np.zeros(n, dtype=np.int32))
    return state


def _sampler(n, temperature):
    sampler = object.__new__(Sampler)
    # The greedy token fast path is on in the execution policy under test.
    sampler.greedy_token_fastpath = True
    sampler.compute_nans = False
    sampler.sampling_states = _states(n, temperature)
    sampler.logprob_token_ids_state = _token_ids_state(n)
    sampler.logit_bias_state = SimpleNamespace(use_logit_bias=np.zeros(n, dtype=bool))
    sampler.penalties_state = SimpleNamespace(use_penalty=np.zeros(n, dtype=bool))
    sampler.bad_words_state = SimpleNamespace(
        num_bad_words=SimpleNamespace(np=np.zeros(n, dtype=np.int32))
    )
    return sampler


def _ask(sampler, req, kind):
    if kind == "logprobs":
        sampler.sampling_states.num_logprobs[req] = 0  # logprobs=0: sampled only
    else:
        sampler.logprob_token_ids_state.num_token_ids.np[req] = 1


def test_baseline_batches_take_the_fast_paths():
    greedy = _sampler(4, 0.0)
    assert greedy.can_use_sm70_greedy_token_fastpath(
        SimpleNamespace(idx_mapping_np=BATCH)
    )
    sampled = _sampler(4, 1.0)
    rejection = SimpleNamespace(rejection_sample_method="standard", sampler=sampled)
    batch = SimpleNamespace(
        num_reqs=4, is_prefilling_np=np.zeros(4, dtype=bool), idx_mapping_np=BATCH
    )
    assert _supports_sparse_sampling_contract(rejection, batch)


@pytest.mark.parametrize("kind", ["logprobs", "logprob_token_ids"])
@pytest.mark.parametrize("req", [0, 2, 3])
def test_one_request_disables_greedy_fastpath_for_the_whole_batch(kind, req):
    sampler = _sampler(4, 0.0)
    _ask(sampler, req, kind)
    assert not sampler.can_use_sm70_greedy_token_fastpath(
        SimpleNamespace(idx_mapping_np=BATCH)
    )
    # A batch without that request keeps the fast path.
    others = BATCH[req != BATCH]
    assert sampler.can_use_sm70_greedy_token_fastpath(
        SimpleNamespace(idx_mapping_np=others)
    )


@pytest.mark.parametrize("kind", ["logprobs", "logprob_token_ids"])
@pytest.mark.parametrize("req", [0, 2, 3])
def test_one_request_disables_sparse_rejection_for_the_whole_batch(kind, req):
    sampler = _sampler(4, 1.0)
    _ask(sampler, req, kind)
    rejection = SimpleNamespace(rejection_sample_method="standard", sampler=sampler)
    batch = SimpleNamespace(
        num_reqs=4, is_prefilling_np=np.zeros(4, dtype=bool), idx_mapping_np=BATCH
    )
    assert not _supports_sparse_sampling_contract(rejection, batch)
