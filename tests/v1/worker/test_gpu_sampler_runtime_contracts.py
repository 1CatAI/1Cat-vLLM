# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small GPU regressions for MRv2 sampling, without loading model weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.ops.bad_words import _apply_bad_words_single_batch
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.prompt_logprob import (
    compute_prompt_logprobs_with_chunking,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA runtime operators"
)


@pytest.mark.parametrize("prompt_len", [1, 17, 8192])
@pytest.mark.parametrize("minimum", [0, 1, 2, 17])
def test_min_tokens_uses_generated_count(prompt_len, minimum):
    state = LogitBiasState(1, torch.device("cuda"))
    params = SamplingParams(
        min_tokens=minimum, max_tokens=64, stop_token_ids=[44, 46], temperature=0
    )
    state.add_request(0, prompt_len, params)
    state.apply_staged_writes()
    counts = sorted({0, max(0, minimum - 1), minimum, minimum + 1})
    positions = torch.tensor(
        [prompt_len + count - 1 for count in counts], device="cuda"
    )
    mapping = torch.zeros(len(counts), dtype=torch.int32, device="cuda")
    logits = torch.zeros((len(counts), 128), device="cuda")
    logits[:, 44], logits[:, 46], logits[:, 42] = 10, 9, 5
    state.apply_logit_bias(logits, mapping, np.array([0]), positions)
    expected = [
        MinTokensLogitsProcessor.add_request(params, None, [42] * count) is not None
        for count in counts
    ]
    assert torch.isneginf(logits[:, 44]).tolist() == expected
    assert torch.isneginf(logits[:, 46]).tolist() == expected
    sampled = gumbel_sample(
        logits,
        mapping,
        torch.zeros(1, device="cuda"),
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        positions,
        apply_temperature=False,
        is_drafting=False,
    )
    assert sampled.tolist() == [42 if mask else 44 for mask in expected]


def _request_state(vocab_size, num_reqs=2):
    return RequestState(
        max_num_reqs=num_reqs,
        max_model_len=64,
        max_num_batched_tokens=8,
        num_speculative_steps=3,
        vocab_size=vocab_size,
        device=torch.device("cuda"),
    )


def _batch(slots):
    n = len(slots)
    mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
    return SimpleNamespace(
        idx_mapping_np=np.array(slots, dtype=np.int32),
        expanded_idx_mapping=mapping,
        expanded_local_pos=torch.zeros(n, dtype=torch.int32, device="cuda"),
        positions=torch.full((n,), 1, dtype=torch.int64, device="cuda"),
        input_ids=torch.full((n,), 2, dtype=torch.int32, device="cuda"),
        logits_indices=torch.arange(n, device="cuda"),
        cu_num_logits_np=np.arange(n + 1, dtype=np.int32),
        seq_lens=torch.full((n,), 2, dtype=torch.int32, device="cuda"),
        num_reqs=n,
    )


@pytest.mark.parametrize("vocab_size", [257, 248320])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_full_vocab_logprobs_are_not_the_none_sentinel(vocab_size, mixed, dtype):
    reqs = _request_state(vocab_size)
    sampler = Sampler(2, vocab_size, torch.device("cuda"), reqs)
    slots = []
    for i, count in enumerate([-1, 2] if mixed else [-1]):
        name = str(i)
        reqs.add_request(name, 2, [1, 2], 2, 8)
        slot = reqs.req_id_to_index[name]
        slots.append(slot)
        params = SamplingParams(temperature=0, seed=0, logprobs=count)
        # A server permitting full vocabulary logprobs accepts this request
        # without rewriting -1 into an internal count.
        params._validate_logprobs(
            SimpleNamespace(max_logprobs=-1, get_vocab_size=lambda: vocab_size)
        )
        sampler.add_request(slot, 2, params)
    reqs.apply_staged_writes()
    sampler.apply_staged_writes()
    batch = _batch(slots)
    assert not sampler.can_use_sm70_greedy_token_fastpath(batch)
    logits = (
        torch.linspace(-5, 5, vocab_size, device="cuda").to(dtype).repeat(len(slots), 1)
    )
    # Keep the winner unique even after casting a large vocabulary to FP16.
    logits[:, -1] = 6
    result = sampler(logits, batch)
    output = result.logprobs_tensors
    assert output is not None
    assert output.logprobs.shape == (len(slots), vocab_size + 1)
    assert torch.unique(output.logprob_token_ids[0]).numel() == vocab_size
    expected = torch.log_softmax(logits.float(), dim=-1).gather(
        1, output.logprob_token_ids.long()
    )
    torch.testing.assert_close(output.logprobs, expected, atol=2e-5, rtol=1e-5)
    assert result.sampled_token_ids[:, 0].tolist() == [vocab_size - 1] * len(slots)


@pytest.mark.parametrize("count", [-1, 2])
def test_prompt_logprobs_shared_helper_across_chunk_boundary(count):
    vocab_size = 257
    logits = torch.linspace(-5, 5, vocab_size, device="cuda").half().repeat(1025, 1)
    prompt_ids = torch.full((1025,), 77, dtype=torch.int64, device="cuda")
    ids, values, ranks = compute_prompt_logprobs_with_chunking(
        prompt_ids, logits, lambda chunk: chunk, count
    )
    columns = 1 + (vocab_size if count == -1 else count)
    assert values.shape == (1025, columns)
    assert values.dtype == torch.float32
    torch.testing.assert_close(ids[:, 0], prompt_ids)
    expected = torch.log_softmax(logits.float(), dim=-1).gather(1, ids.long())
    torch.testing.assert_close(values, expected, atol=2e-5, rtol=1e-5)
    expected_ranks = (logits >= logits[:, 77:78]).sum(dim=-1)
    torch.testing.assert_close(ranks, expected_ranks)
    if count == -1:
        assert torch.all(
            ids[:, 1:].sort(dim=-1).values == torch.arange(vocab_size, device="cuda")
        )


@pytest.mark.parametrize("drafts", [[], [7], [7, 8], [7, 8, 7]])
def test_bad_words_follow_current_draft_prefix(drafts):
    reqs = _request_state(128)
    reqs.add_request("request", 2, [1, 2, 5], 3, 8)
    slot = reqs.req_id_to_index["request"]
    state = BadWordsState(reqs)
    words = [[7, 9], [5, 10], [7, 8, 11], [5, 7, 12], [13]]
    params = SamplingParams()
    params._bad_words_token_ids = words
    state.add_request(slot, params)
    reqs.apply_staged_writes()
    state.apply_staged_writes()

    # Expanded row zero is the last committed token, not draft zero.
    inputs = torch.tensor([5, *drafts], dtype=torch.int32, device="cuda")
    local_pos = torch.arange(len(inputs), dtype=torch.int32, device="cuda")
    mapping = torch.full_like(local_pos, slot)
    logits = torch.arange(128, device="cuda", dtype=torch.float32).repeat(
        len(inputs), 1
    )
    expected = logits.cpu()
    for row in range(len(inputs)):
        _apply_bad_words_single_batch(expected[row], words, [5, *drafts[:row]])
    state.apply_bad_words(logits, mapping, np.array([slot]), inputs, local_pos)
    torch.testing.assert_close(logits.cpu(), expected, atol=0, rtol=0)


def test_min_tokens_graph_replay_crosses_boundary():
    state = LogitBiasState(1, torch.device("cuda"))
    state.add_request(0, 17, SamplingParams(min_tokens=2, stop_token_ids=[44]))
    state.apply_staged_writes()
    positions = torch.tensor([17], device="cuda")
    mapping = torch.zeros(1, dtype=torch.int32, device="cuda")
    template = torch.zeros(1, 128, device="cuda")
    template[:, 44] = 10
    output = template.clone()
    state.apply_logit_bias(output, mapping, np.array([0]), positions)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output.copy_(template)
        state.apply_logit_bias(output, mapping, np.array([0]), positions)
    for generated in (1, 2, 3, 1, 2):
        positions.fill_(17 + generated - 1)
        graph.replay()
        assert torch.isneginf(output[0, 44]).item() == (generated < 2)


def test_bad_words_graph_replay_and_request_slot_reuse():
    reqs = _request_state(128)
    reqs.add_request("first", 2, [1, 2, 5], 3, 8)
    slot = reqs.req_id_to_index["first"]
    state = BadWordsState(reqs)
    words = [[7, 9], [5, 10], [7, 8, 11], [8, 7, 12]]
    params = SamplingParams(_bad_words_token_ids=words)
    state.add_request(slot, params)
    reqs.apply_staged_writes()
    state.apply_staged_writes()
    inputs = torch.tensor([5, 7, 8], dtype=torch.int32, device="cuda")
    positions = torch.arange(3, dtype=torch.int32, device="cuda")
    mapping = torch.full_like(positions, slot)
    template = torch.arange(128, device="cuda", dtype=torch.float32).repeat(3, 1)
    output = template.clone()
    state.apply_bad_words(output, mapping, np.array([slot]), inputs, positions)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output.copy_(template)
        state.apply_bad_words(output, mapping, np.array([slot]), inputs, positions)
    for drafts in ([7, 8], [8, 7], [7, 8]):
        inputs.copy_(torch.tensor([5, *drafts], dtype=torch.int32, device="cuda"))
        graph.replay()
        expected = template.cpu()
        for row in range(3):
            _apply_bad_words_single_batch(expected[row], words, [5, *drafts[:row]])
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)

    reqs.remove_request("first")
    reqs.add_request("second", 2, [1, 2, 5], 3, 8)
    assert reqs.req_id_to_index["second"] == slot
    state.add_request(slot, SamplingParams())
    reqs.apply_staged_writes()
    state.apply_staged_writes()
    output.copy_(template)
    state.apply_bad_words(output, mapping, np.array([slot]), inputs, positions)
    torch.testing.assert_close(output, template, atol=0, rtol=0)
