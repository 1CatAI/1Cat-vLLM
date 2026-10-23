# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import copy
import pickle

import pytest
from tokenizers import pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm.tokenizers.gguf import tokenizer_from_gguf_metadata
from vllm.tokenizers.hf import get_cached_tokenizer, maybe_make_thread_pool
from vllm.utils.async_utils import AsyncMicrobatchTokenizer


def make_tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    merges = [("a", "l"), ("al", "p"), ("alp", "h"), ("alph", "a"), ("Ġ", "alpha")]
    words = [a + b for a, b in merges]
    return tokenizer_from_gguf_metadata(
        {
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "qwen35",
            "tokenizer.ggml.tokens": alphabet + words + ["<|im_end|>", "<think>"],
            "tokenizer.ggml.merges": [a + " " + b for a, b in merges],
            "tokenizer.ggml.token_type": [1] * 261 + [3, 4],
            "tokenizer.ggml.eos_token_id": 261,
        }
    )


@pytest.mark.parametrize("length", [1023, 1024, 1025, 32768])
@pytest.mark.parametrize("special", [True, False])
def test_ascii_encode_preserves_ids_and_hf_encoding_api(length, special):
    tok = make_tokenizer()
    assert tok._qwen_piece_cache is not None
    text = ("alpha<think> ALPHA 012\t\n<|im_end|> " * length)[:length]
    kwargs = {"add_special_tokens": special}
    assert tok.encode(text, **kwargs) == PreTrainedTokenizerFast.encode(
        tok, text, **kwargs
    )
    encoded = tok(text, return_offsets_mapping=True, **kwargs)
    assert encoded.encodings is not None and encoded.offset_mapping
    assert encoded.input_ids == tok.encode(text, **kwargs)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("limit", [0, 1, 21, 1024, 8192])
def test_ascii_encode_keeps_truncation(side, limit):
    tok = make_tokenizer()
    tok.truncation_side = side
    text = "alpha river 012 " * 128
    kwargs = {"truncation": True, "max_length": limit}
    assert tok.encode(text, **kwargs) == PreTrainedTokenizerFast.encode(
        tok, text, **kwargs
    )


def test_unicode_pair_tensor_and_added_token_mutation_retain_hf():
    tok = make_tokenizer()
    text = "alpha " * 200
    for source in (text + "e\u0301你好👩‍💻", text):
        assert tok.encode(source, text_pair="world") == PreTrainedTokenizerFast.encode(
            tok, source, text_pair="world"
        )
        assert tok.encode(source) == PreTrainedTokenizerFast.encode(tok, source)
        assert tok.encode(source, return_tensors="pt").equal(
            PreTrainedTokenizerFast.encode(tok, source, return_tensors="pt")
        )
    tok.add_tokens(["new_token"])
    assert tok._qwen_piece_cache is None
    assert tok.encode(text + "new_token") == PreTrainedTokenizerFast.encode(
        tok, text + "new_token"
    )


def test_copies_pickle_and_thread_pool_keep_independent_tokenizers():
    tok = get_cached_tokenizer(make_tokenizer())
    text = "alpha<think> 012\n" * 200
    expected = tok.encode(text)
    for duplicate in (copy.deepcopy(tok), pickle.loads(pickle.dumps(tok))):
        assert duplicate.encode(text) == expected
        maybe_make_thread_pool(duplicate)
        assert duplicate.encode(text) == expected
    other = make_tokenizer()
    other.add_special_tokens({"additional_special_tokens": ["<new>"]})
    assert other._qwen_piece_cache is None and tok._qwen_piece_cache is not None


def test_async_ids_use_encode_and_batch_encoding_keeps_original_api():
    asyncio.run(_async_ids_and_encoding())


async def _async_ids_and_encoding():
    tok = make_tokenizer()
    ids = []
    original = tok.encode

    def encode(*args, **kwargs):
        ids.append(1)
        return original(*args, **kwargs)

    tok.encode = encode
    asynchronous = AsyncMicrobatchTokenizer(tok)
    text = "alpha 012\n" * 200
    kwargs = {"truncation": True, "max_length": 8192}
    actual = await asynchronous.encode(text, **kwargs)
    assert actual == original(text, **kwargs) and len(ids) == 1
    encoded = await asynchronous(text, return_offsets_mapping=True, **kwargs)
    assert encoded.encodings is not None and encoded.input_ids == actual
    prompts = [text, text + "alpha", text + "alpha alpha"]
    actuals = await asyncio.gather(*(asynchronous.encode(s) for s in prompts))
    assert actuals == [original(s) for s in prompts]
