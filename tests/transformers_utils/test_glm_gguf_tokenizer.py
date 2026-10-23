# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from tokenizers import pre_tokenizers

from vllm.tokenizers.gguf import tokenizer_from_gguf_metadata


def metadata():
    tokens = sorted(pre_tokenizers.ByteLevel.alphabet()) + ["<sop>", "<think>"]
    return {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "glm4",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.token_type": [1] * 256 + [3, 4],
        "tokenizer.ggml.bos_token_id": 256,
        "tokenizer.chat_template": "{{ messages[0]['content'] }}",
    }


def test_glm_digit_groups_and_unicode_preservation():
    tok = tokenizer_from_gguf_metadata(metadata())
    split = tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str("1234567")
    assert split == [("123", (0, 3)), ("456", (3, 6)), ("7", (6, 7))]
    assert tok.encode("é") != tok.encode("e\u0301")
    for text in ["中文 e\u0301 １２３４", "I'm\r\n  a\t🙂", "123456789"]:
        assert tok.decode(tok.encode(text)) == text
    ids = tok.encode("<think>你好<sop>")
    assert ids[0] == 257 and ids[-1] == 256
    assert tok.decode(ids, skip_special_tokens=True) == "<think>你好"
    assert tok.encode("x", add_special_tokens=True) == tok.encode(
        "x", add_special_tokens=False
    )


@pytest.mark.parametrize("key", ["add_bos_token", "add_eos_token"])
def test_glm_rejects_unverified_special_insertion(key):
    values = metadata()
    values[f"tokenizer.ggml.{key}"] = True
    with pytest.raises(ValueError, match="BOS/EOS insertion"):
        tokenizer_from_gguf_metadata(values)
