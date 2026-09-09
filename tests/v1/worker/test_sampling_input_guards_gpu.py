# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Valid input controls through actual GPU masks; invalid IDs stay on CPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.bad_words import _apply_bad_words_single_batch
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class ByteTokenizer:
    max_token_id = 127

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("ascii"))


@pytest.mark.parametrize("word", [" ", "  ", "\t", "\n", "hello"])
def test_bad_word_preprocessing_matches_gpu_mask(word):
    device = torch.device("cuda")
    params = SamplingParams(bad_words=[word])
    params.update_from_tokenizer(ByteTokenizer())
    tokens = ByteTokenizer().encode(word)
    history = tokens[:-1]
    reqs = RequestState(1, 32, 8, 0, 128, device)
    reqs.add_request("request", 1, [1, *history], len(history), 8)
    state = BadWordsState(reqs)
    state.add_request(0, params)
    reqs.apply_staged_writes()
    state.apply_staged_writes()
    logits = torch.arange(128, dtype=torch.float32, device=device).unsqueeze(0)
    expected = logits.cpu()
    _apply_bad_words_single_batch(expected[0], params.bad_words_token_ids, history)
    state.apply_bad_words(
        logits,
        torch.zeros(1, dtype=torch.int32, device=device),
        np.array([0]),
        torch.tensor([history[-1] if history else 1], dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(logits.cpu(), expected, atol=0, rtol=0)
    assert torch.isneginf(logits[0, tokens[-1]])


@pytest.mark.parametrize("vocab_size", [128, 248320])
@pytest.mark.parametrize("generated", [1, 2, 3])
def test_valid_token_id_guards_preserve_stop_and_allow_masks(vocab_size, generated):
    eos = vocab_size - 1
    params = SamplingParams(
        min_tokens=2, allowed_token_ids=[42, eos], stop_token_ids=[eos]
    )
    config = SimpleNamespace(
        max_logprobs=20, get_vocab_size=lambda: vocab_size, logits_processors=None
    )
    params.verify(config, None, None, None)
    state = LogitBiasState(1, torch.device("cuda"))
    state.add_request(0, 8, params)
    state.apply_staged_writes()
    logits = torch.zeros(1, vocab_size, device="cuda")
    logits[0, eos], logits[0, 42] = 10, 5
    state.apply_logit_bias(
        logits,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        np.array([0]),
        torch.tensor([8 + generated - 1], device="cuda"),
    )
    expected = torch.full_like(logits, -torch.inf)
    expected[0, 42] = 5
    if generated >= 2:
        expected[0, eos] = 10
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)
