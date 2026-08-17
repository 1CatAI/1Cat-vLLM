# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP probabilistic-draft guard: only reject when a non-greedy row carries drafts.

Draft tokens are proposed one step ahead with the previous batch's sampling
metadata. When that batch was all-greedy the proposer legitimately returns no
draft probabilities. If a sampled request joins on the next step it has zero
draft tokens, so the verify step must not be rejected (that rejection killed
the engine on every greedy->mixed batch transition).
"""

from types import SimpleNamespace

import torch

from vllm.v1.sample.rejection_sampler import GREEDY_TEMPERATURE
from vllm.v1.worker.gpu_model_runner import _non_greedy_rows_carry_drafts


def _meta(temperatures, num_draft_tokens):
    sampling = SimpleNamespace(
        temperature=None
        if temperatures is None
        else torch.tensor(temperatures, dtype=torch.float32)
    )
    spec = SimpleNamespace(num_draft_tokens=list(num_draft_tokens))
    return sampling, spec


def test_all_greedy_rows_with_drafts_do_not_need_probs():
    sampling, spec = _meta([GREEDY_TEMPERATURE, GREEDY_TEMPERATURE], [2, 2])
    assert _non_greedy_rows_carry_drafts(sampling, spec) is False


def test_sampled_request_joining_greedy_batch_has_no_drafts_yet():
    # Row 0: running greedy decode with 2 drafts proposed last step (argmax path).
    # Row 1: new temperature=1.0 request on its first verify step, 0 drafts.
    sampling, spec = _meta([GREEDY_TEMPERATURE, 1.0], [2, 0])
    assert _non_greedy_rows_carry_drafts(sampling, spec) is False


def test_non_greedy_row_with_drafts_requires_probs():
    sampling, spec = _meta([GREEDY_TEMPERATURE, 1.0], [2, 2])
    assert _non_greedy_rows_carry_drafts(sampling, spec) is True


def test_no_drafts_anywhere_never_requires_probs():
    sampling, spec = _meta([0.7, 1.0], [0, 0])
    assert _non_greedy_rows_carry_drafts(sampling, spec) is False


def test_missing_temperature_tensor_means_all_greedy():
    sampling, spec = _meta(None, [2, 2])
    assert _non_greedy_rows_carry_drafts(sampling, spec) is False
