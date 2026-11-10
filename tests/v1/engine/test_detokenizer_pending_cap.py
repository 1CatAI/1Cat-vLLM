# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded pending window in incremental detokenization (upstream vLLM #63668).

Uses a small byte-level BPE tokenizer trained in the test, so no download is
needed. Byte-level BPE is the scheme of the Qwen tokenizers: CJK characters
that are not merged are split into byte tokens, and a lone UTF-8 lead byte
decodes to U+FFFD.
"""

import time

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from vllm.sampling_params import SamplingParams
from vllm.tokenizers.detokenizer_utils import MAX_PENDING_TOKENS
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import (
    FastIncrementalDetokenizer,
    SlowIncrementalDetokenizer,
)

CORPUS = [
    "Hello world, this is a simple test of the detokenizer.",
    "The quick brown fox jumps over the lazy dog.",
    "我很感谢你的热情。今天天气很好。",
    "你好，世界！",
] * 20

REPLACEMENT = "�"


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS, trainer=trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok)


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _lone_lead_byte(tokenizer) -> int:
    # 0xE2 alone: the lead byte of a 3-byte sequence.
    ids = _encode(tokenizer, "—")  # em dash, E2 80 94
    assert len(ids) == 3, "em dash must stay split into byte tokens"
    assert tokenizer.decode(ids[:1]) == REPLACEMENT
    return ids[0]


def _new(tokenizer, prompt, fast: bool, stop=None):
    params = SamplingParams(skip_special_tokens=True, stop=stop)
    request = EngineCoreRequest(
        request_id="r",
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    cls = FastIncrementalDetokenizer if fast else SlowIncrementalDetokenizer
    return cls(tokenizer, request)


def _stream(detok, ids, finish=True) -> str:
    out = ""
    for i, token_id in enumerate(ids):
        stop = detok.update([token_id], False)
        last = finish and (i == len(ids) - 1 or stop is not None)
        out += detok.get_next_output_text(last, delta=True)
        if stop is not None:
            break
    return out


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize(
    "text",
    [
        "Hello world, this is a simple test.",
        "我很感谢你的热情",
        "Mixed 中文 and English — 测试 🌶️ done",
    ],
)
def test_normal_text_matches_full_decode(tokenizer, fast, text):
    prompt = _encode(tokenizer, "Hello")
    gen = _encode(tokenizer, text)
    detok = _new(tokenizer, prompt, fast)
    out = _stream(detok, gen)
    full = tokenizer.decode(prompt + gen)
    assert out == full[len(tokenizer.decode(prompt)) :]
    assert detok.output_token_ids == gen


@pytest.mark.parametrize("fast", [True, False])
def test_multibyte_char_split_across_tokens_is_held_then_emitted(tokenizer, fast):
    gen = _encode(tokenizer, "—ok")
    detok = _new(tokenizer, _encode(tokenizer, "a"), fast)
    pieces = []
    for token_id in gen:
        detok.update([token_id], False)
        pieces.append(detok.get_next_output_text(False, delta=True))
    # The first two bytes are held, the third completes the dash.
    assert pieces[0] == "" and pieces[1] == ""
    assert "".join(pieces) + detok.get_next_output_text(True, True) == "—ok"


@pytest.mark.parametrize("fast", [True, False])
def test_malformed_run_is_streamed_and_tail_decodes(tokenizer, fast):
    prompt = _encode(tokenizer, "Hello")
    run = [_lone_lead_byte(tokenizer)] * 1024
    tail = _encode(tokenizer, " world")
    detok = _new(tokenizer, prompt, fast)
    out = _stream(detok, run, finish=False)
    held_back = len(run) - out.count(REPLACEMENT)
    assert held_back <= MAX_PENDING_TOKENS
    out += _stream(detok, tail)
    full = tokenizer.decode(prompt + run + tail)
    assert out == full[len(tokenizer.decode(prompt)) :]
    assert detok.output_token_ids == run + tail


def test_prompt_ending_in_incomplete_utf8_is_not_re_emitted(tokenizer):
    bad = _lone_lead_byte(tokenizer)
    prompt = _encode(tokenizer, "Hello") + [bad]
    run = [bad] * (2 * MAX_PENDING_TOKENS)
    tail = _encode(tokenizer, " world")
    detok = _new(tokenizer, prompt, fast=True)
    for token_id in run + tail:
        detok.update([token_id], False)
    out = detok.get_next_output_text(True, delta=True)
    assert "Hello" not in out
    assert out.endswith("world")
    assert out.count(REPLACEMENT) == len(run)


@pytest.mark.parametrize("fast", [True, False])
def test_stop_string_after_malformed_run(tokenizer, fast):
    run = [_lone_lead_byte(tokenizer)] * (3 * MAX_PENDING_TOKENS)
    tail = _encode(tokenizer, " hello world and more")
    detok = _new(tokenizer, _encode(tokenizer, "Hi"), fast, stop=["world"])
    out = _stream(detok, run + tail)
    assert out.endswith(" hello ")
    assert "world" not in out
    assert out.count(REPLACEMENT) == len(run)


@pytest.mark.parametrize("fast", [True, False])
def test_long_malformed_stream_has_bounded_cost(tokenizer, fast):
    run = [_lone_lead_byte(tokenizer)] * 8000
    detok = _new(tokenizer, _encode(tokenizer, "Hello"), fast)
    start = time.perf_counter()
    out = _stream(detok, run)
    elapsed = time.perf_counter() - start
    assert out == REPLACEMENT * len(run)
    # Unbounded, this re-decodes the growing run on every step (seconds);
    # bounded, it is linear (tens of milliseconds).
    assert elapsed < 2.0, f"{elapsed=:.2f}s"


def test_unrepresentable_ids_are_skipped_not_collected(tokenizer):
    detok = _new(tokenizer, [], fast=True)
    tail = _encode(tokenizer, " world")
    for token_id in [2**40] * (MAX_PENDING_TOKENS + 1) + tail:
        detok.update([token_id], False)
    assert detok.get_next_output_text(True, delta=True) == " world"
