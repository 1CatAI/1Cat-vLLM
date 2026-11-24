# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native GGUF tokenizer of Inkling2 GGUFs matches the official tokenizer.

INKLING2_GGUF_Q4 and INKLING2_OFFICIAL as in the other Inkling2 GGUF tests.
"""

import os

import pytest

_GGUF = os.environ.get("INKLING2_GGUF_Q4", "")
_OFFICIAL = os.environ.get("INKLING2_OFFICIAL", "")

SAMPLES = [
    "Hello, world! It's 2026-11-24 and we've got 3.14159 reasons.",
    "def f(x):\n    return x ** 2  # square\n\n\tTabs\r\nand CRLF",
    "Ünïcödé — naïve café; 東京は晴れ。 مرحبا بالعالم. Привет, мир!",
    "   leading spaces, trailing   ",
    "emoji 🙂🚀 and 12345678901234567890 digits, URL https://example.com/a?b=c",
    "CamelCaseWords and SHOUTING and they'LL mix'D",
]


@pytest.mark.skipif(
    not (os.path.exists(_GGUF) and os.path.isdir(_OFFICIAL)),
    reason="set INKLING2_GGUF_Q4 and INKLING2_OFFICIAL",
)
def test_gguf_tokenizer_matches_official():
    from transformers import PreTrainedTokenizerFast

    from vllm.tokenizers.gguf import tokenizer_from_gguf_metadata
    from vllm.transformers_utils.gguf_config import read_gguf_metadata

    gguf_tok = tokenizer_from_gguf_metadata(read_gguf_metadata(_GGUF))
    official = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(_OFFICIAL, "tokenizer.json")
    )
    for text in SAMPLES:
        assert gguf_tok.encode(text, add_special_tokens=False) == official.encode(
            text, add_special_tokens=False
        ), text
