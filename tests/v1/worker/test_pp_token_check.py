# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The token ids the last pipeline stage hands to the others are checked in
host memory, so an id outside the vocabulary is named instead of surfacing as
an anonymous device-side assert in the first stage's embedding lookup."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

VOCAB = 100


def _runner(discarded: list[bool]) -> SimpleNamespace:
    num_reqs = len(discarded)
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            vocab_size=VOCAB,
            req_ids=[f"req-{i}" for i in range(num_reqs)],
            num_computed_tokens_cpu=np.arange(num_reqs) + 10,
            sampling_metadata=SimpleNamespace(
                temperature=None, all_greedy=True, all_random=False
            ),
        ),
        discard_request_mask=SimpleNamespace(np=np.array(discarded)),
        input_ids=SimpleNamespace(gpu=torch.arange(8)),
        positions=torch.arange(8),
    )


def _check(runner, kind, token_ids, skip_discarded, sampler_input=None):
    GPUModelRunner._pp_check_token_ids(
        runner, kind, token_ids, skip_discarded, sampler_input
    )


def test_valid_token_ids_pass():
    _check(_runner([False, False]), "sampled", torch.tensor([[0], [99]]), True)


@pytest.mark.parametrize("bad", [VOCAB, -1])
def test_out_of_vocabulary_id_names_the_request(bad):
    with pytest.raises(RuntimeError, match=r"invalid sampled token ids.*req-1"):
        _check(_runner([False, False]), "sampled", torch.tensor([[3], [bad]]), True)


def test_discarded_request_is_not_checked_for_sampled_tokens():
    _check(_runner([False, True]), "sampled", torch.tensor([[3], [VOCAB]]), True)


def test_draft_tokens_of_a_discarded_request_are_still_checked():
    with pytest.raises(RuntimeError, match="invalid draft token ids"):
        _check(_runner([False, True]), "draft", torch.tensor([[3], [VOCAB]]), False)


def test_sampler_input_names_the_non_finite_rows():
    logits = torch.zeros(2, VOCAB)
    logits[1] = float("nan")
    with pytest.raises(
        RuntimeError, match=r"non-finite values per sampler input row \[0, 100\]"
    ):
        _check(
            _runner([False, False]),
            "sampled",
            torch.tensor([[3], [VOCAB]]),
            True,
            sampler_input=logits,
        )
