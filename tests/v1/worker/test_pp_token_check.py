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


@pytest.mark.parametrize("bad", [VOCAB, -2])
@pytest.mark.parametrize("column", [1, 2])
def test_invalid_accepted_token_in_later_columns_is_checked(bad, column):
    tokens = torch.tensor([[3, 4, 5]])
    tokens[0, column] = bad
    with pytest.raises(RuntimeError, match="invalid sampled token ids"):
        _check(_runner([False]), "sampled", tokens, True)


@pytest.mark.parametrize("tokens", [[[3, 4, -1]], [[3, -1, VOCAB]]])
def test_only_the_contiguous_accepted_prefix_is_consumed(tokens):
    _check(_runner([False]), "sampled", torch.tensor(tokens), True)


def _wire_runner(monkeypatch):
    from types import MethodType

    from vllm.v1.worker import gpu_model_runner as module

    runner = _runner([False])
    runner.input_batch.num_reqs = 1
    runner.num_spec_tokens = 2
    runner.device = torch.device("cpu")
    runner._is_all_reqs_chunked_prefill = lambda: False
    runner._pp_check_token_ids = MethodType(GPUModelRunner._pp_check_token_ids, runner)
    monkeypatch.setattr(
        module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True, rank=1, last_rank=1, cpu_group=None),
    )
    return runner


def test_sample_broadcast_checks_later_accepted_columns(monkeypatch):
    runner = _wire_runner(monkeypatch)
    broadcasts = []
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, **kwargs: broadcasts.append(tensor),
    )
    with pytest.raises(RuntimeError, match="invalid sampled token ids"):
        GPUModelRunner._pp_broadcast_prev_sampled_token_ids(
            runner, torch.tensor([[3, VOCAB]]), torch.zeros(1, VOCAB)
        )
    assert broadcasts == []


def test_sample_broadcast_keeps_the_static_padded_wire_shape(monkeypatch):
    runner = _wire_runner(monkeypatch)
    broadcasts = []
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, **kwargs: broadcasts.append(tensor),
    )
    GPUModelRunner._pp_broadcast_prev_sampled_token_ids(
        runner, torch.tensor([[3, 4]]), torch.zeros(1, VOCAB)
    )
    assert broadcasts[0].tolist() == [[3, 4, -1]]


def test_draft_broadcast_checks_all_columns_before_sending(monkeypatch):
    runner = _wire_runner(monkeypatch)
    runner._draft_token_ids = torch.tensor([[3, VOCAB]])
    broadcasts = []
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, **kwargs: broadcasts.append(tensor),
    )
    with pytest.raises(RuntimeError, match="invalid draft token ids"):
        GPUModelRunner._pp_broadcast_draft_token_ids(runner)
    assert broadcasts == []


@pytest.mark.parametrize(
    "tokens,invalid", [([3, VOCAB, -1], True), ([3, 4, -1], False)]
)
def test_receive_checks_the_matrix_before_deriving_next_tokens(
    monkeypatch, tokens, invalid
):
    runner = _wire_runner(monkeypatch)
    incoming = [torch.tensor([tokens]), torch.tensor([[7, 8]])]
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, **kwargs: tensor.copy_(incoming.pop(0)),
    )
    copies = []
    runner.valid_sampled_token_count_event = object()
    runner._copy_valid_sampled_token_count = lambda token, count: copies.append(
        (token.tolist(), count.tolist())
    )
    runner._pp_nonlast_scheduler_output = object()
    runner._update_states_after_model_execute = lambda *args: None
    if invalid:
        with pytest.raises(RuntimeError, match="invalid sampled token ids"):
            GPUModelRunner._pp_receive_spec_decode_state(runner, 1)
        assert copies == []
    else:
        GPUModelRunner._pp_receive_spec_decode_state(runner, 1)
        assert copies == [([4], [2])]
        assert runner._draft_token_ids.tolist() == [[7, 8]]
