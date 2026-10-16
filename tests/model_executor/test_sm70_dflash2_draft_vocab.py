# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocabulary trimming must retain ASCII/CJK byte pieces and original IDs."""

import pytest

from vllm.model_executor.layers.sm70_dflash2_draft_vocab import (
    _byte_decoder,
    build_draft_token_ids,
)


def test_complete_cjk_and_byte_fragments_with_original_ids():
    encode = {value: character for character, value in _byte_decoder().items()}

    def token(raw):
        return "".join(encode[value] for value in raw)

    vocab = {
        token(b"common"): 3,
        token(b"rareword"): 101,
        token(b"foo_bar123()\n\t"): 104,
        token("\u03b1".encode()): 109,
        token("hello中文".encode()): 702,
        token("かな".encode()): 47,
        token("한글".encode()): 904,
        token("\U00031350".encode()): 27,
        token(b"\xe4\xb8"): 201,
        token(b"\xad"): 206,
        token("\uf900".encode()): 308,
        token("\u31c0".encode()): 403,
        token("\U0001aff0".encode()): 411,
    }
    tokenizer = {
        "model": {"type": "BPE", "vocab": vocab},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [{"id": 801, "content": "<eos>"}],
    }
    assert build_draft_token_ids(tokenizer, [3, 3, 999]) == [
        3,
        27,
        47,
        101,
        104,
        201,
        206,
        308,
        403,
        411,
        702,
        801,
        904,
    ]


def test_unsupported_tokenizer_does_not_silently_trim():
    with pytest.raises(ValueError, match="byte-level BPE"):
        build_draft_token_ids(
            {"model": {"type": "Unigram"}, "decoder": {"type": "Metaspace"}}, []
        )
