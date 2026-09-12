# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-boundary regressions: do not send invalid indices to CUDA."""

from types import SimpleNamespace

import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.bad_words import _apply_bad_words_single_batch


class Tokenizer:
    max_token_id = 127

    def __len__(self):
        return self.max_token_id + 1

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) for char in text if char != "\x00"]


def verify(params, vocab_size=128, tokenizer=None):
    config = SimpleNamespace(
        max_logprobs=20, get_vocab_size=lambda: vocab_size, logits_processors=None
    )
    params.verify(config, None, None, tokenizer)


@pytest.mark.parametrize("vocab_size", [128, 248320])
@pytest.mark.parametrize("offset", [-1, 0, 100])
@pytest.mark.parametrize("minimum", [0, 2])
def test_stop_ids_are_checked_against_model_vocabulary(vocab_size, offset, minimum):
    token = offset if offset < 0 else vocab_size + offset
    params = SamplingParams(stop_token_ids=[token], min_tokens=minimum)
    with pytest.raises(VLLMValidationError, match="stop_token_ids"):
        verify(params, vocab_size)


@pytest.mark.parametrize("vocab_size", [128, 248320])
@pytest.mark.parametrize("offset", [-1, 0, 100])
def test_allowed_ids_are_checked_without_tokenizer(vocab_size, offset):
    token = offset if offset < 0 else vocab_size + offset
    params = SamplingParams(allowed_token_ids=[0, token], detokenize=False)
    with pytest.raises(VLLMValidationError, match="allowed_token_ids"):
        verify(params, vocab_size)


def test_tokenizer_extra_ids_do_not_exceed_model_logits():
    with pytest.raises(VLLMValidationError, match="allowed_token_ids"):
        verify(
            SamplingParams(allowed_token_ids=[100]),
            vocab_size=64,
            tokenizer=Tokenizer(),
        )


@pytest.mark.parametrize("ids", [[0], [127], [0, 127, 0]])
@pytest.mark.parametrize("with_tokenizer", [False, True])
def test_valid_boundary_ids_are_preserved(ids, with_tokenizer):
    params = SamplingParams(stop_token_ids=ids, allowed_token_ids=ids, min_tokens=2)
    verify(params, tokenizer=Tokenizer() if with_tokenizer else None)
    assert params.stop_token_ids == ids
    assert params.allowed_token_ids == ids
    assert params.all_stop_token_ids == set(ids)


@pytest.mark.parametrize("ids", [[], [128]])
def test_existing_tokenizer_limit_is_preserved(ids):
    with pytest.raises(VLLMValidationError, match="allowed_token_ids"):
        verify(
            SamplingParams(allowed_token_ids=ids), vocab_size=256, tokenizer=Tokenizer()
        )


@pytest.mark.parametrize("word", [" ", "  ", "\t", "\n"])
def test_whitespace_bad_words_remain_literal_sequences(word):
    tokenizer = Tokenizer()
    params = SamplingParams(bad_words=[word])
    params.update_from_tokenizer(tokenizer)
    expected = tokenizer.encode(word)
    assert expected in params.bad_words_token_ids
    assert all(params.bad_words_token_ids)
    logits = torch.zeros(128)
    _apply_bad_words_single_batch(logits, params.bad_words_token_ids, expected[:-1])
    assert torch.isneginf(logits[expected[-1]])


@pytest.mark.parametrize("word", ["hello", " hello", "  hello"])
def test_normal_bad_word_prefix_behavior_is_unchanged(word):
    params = SamplingParams(bad_words=[word])
    params.update_from_tokenizer(Tokenizer())
    assert params.bad_words_token_ids == [Tokenizer().encode("hello")]


def test_empty_tokenization_is_a_request_error():
    params = SamplingParams(bad_words=["\x00"])
    with pytest.raises(VLLMValidationError, match="bad_words"):
        params.update_from_tokenizer(Tokenizer())


@pytest.mark.parametrize("parameter", ["stop_token_ids", "allowed_token_ids"])
@pytest.mark.parametrize("token_id", [-1, 248320])
def test_chat_request_is_rejected_at_engine_admission(parameter, token_id):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.v1.engine.input_processor import InputProcessor

    request = ChatCompletionRequest(
        model="qwen38",
        messages=[{"role": "user", "content": "hello"}],
        min_tokens=2,
        **{parameter: [token_id]},
    )
    params = request.to_sampling_params(max_tokens=8, default_sampling_params={})
    processor = object.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(
        max_logprobs=20, get_vocab_size=lambda: 248320, logits_processors=None
    )
    processor.speculative_config = None
    processor.structured_outputs_config = None
    processor.renderer = SimpleNamespace(tokenizer=None)
    with pytest.raises(VLLMValidationError) as exc:
        processor._validate_params(params, ("generate",))
    assert exc.value.parameter == parameter


def test_premerged_eos_is_validated_for_min_token_masking():
    params = SamplingParams(min_tokens=2, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=128)
    assert params.stop_token_ids == []
    with pytest.raises(VLLMValidationError, match="stop_token_ids"):
        verify(params)
