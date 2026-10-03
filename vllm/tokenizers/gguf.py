# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizers reconstructed from GGUF vocabulary and preprocessing metadata."""

from pathlib import Path

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
)
from transformers import PreTrainedTokenizerFast

from vllm.transformers_utils.gguf_config import read_gguf_metadata
from vllm.transformers_utils.gguf_files import resolve_gguf_file

from .hf import get_cached_tokenizer

# Same segmentation as Qwen2Tokenizer and llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN2.
_QWEN2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


_QWEN35_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def tokenizer_from_gguf_metadata(metadata, **kwargs):
    model = metadata.get("tokenizer.ggml.model")
    pre = metadata.get("tokenizer.ggml.pre")
    if model != "gpt2" or pre not in ("qwen2", "qwen35"):
        raise ValueError(
            f"No native GGUF tokenizer for model={model!r}, pre={pre!r}. "
            "Provide --tokenizer with a tokenizer repository or local directory."
        )
    tokens = metadata["tokenizer.ggml.tokens"]
    vocab = {token: i for i, token in enumerate(tokens)}
    if len(vocab) != len(tokens):
        raise ValueError("Duplicate vocabulary strings in GGUF tokenizer")
    merges = [tuple(merge.split(" ")) for merge in metadata["tokenizer.ggml.merges"]]
    backend = Tokenizer(models.BPE(vocab, merges, fuse_unk=False, byte_fallback=False))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(_QWEN35_PATTERN if pre == "qwen35" else _QWEN2_PATTERN),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    if pre == "qwen35":
        backend.normalizer = normalizers.NFC()
    backend.decoder = decoders.ByteLevel()
    types = metadata.get("tokenizer.ggml.token_type", [1] * len(tokens))
    if len(types) != len(tokens):
        raise ValueError("GGUF tokenizer token_type length differs from vocabulary")
    # CONTROL tokens are special; USER_DEFINED tokens match literally, but
    # stay visible when decoding with skip_special_tokens=True.
    backend.add_tokens(
        [AddedToken(t, normalized=False) for t, ty in zip(tokens, types) if ty == 4]
    )
    special = [t for t, ty in zip(tokens, types) if ty == 3]
    backend.add_special_tokens(
        [AddedToken(t, normalized=False, special=True) for t in special]
    )
    tokenizer_kwargs = {}
    for key in ("bos", "eos", "padding", "unknown"):
        token_id = metadata.get(f"tokenizer.ggml.{key}_token_id")
        if token_id is not None:
            if not 0 <= token_id < len(tokens):
                raise ValueError(f"GGUF {key}_token_id is outside vocabulary")
            target = {"padding": "pad", "unknown": "unk"}.get(key, key)
            tokenizer_kwargs[f"{target}_token"] = tokens[token_id]
    # Qwen tokenizers do not wrap prompts in BOS/EOS. Refuse an export that
    # requests another policy until its postprocessor has a token parity gate.
    if metadata.get("tokenizer.ggml.add_bos_token", False) or metadata.get(
        "tokenizer.ggml.add_eos_token", False
    ):
        raise ValueError("GGUF Qwen tokenizer requests unsupported BOS/EOS insertion")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=special,
        chat_template=metadata.get("tokenizer.chat_template"),
        **tokenizer_kwargs,
        **kwargs,
    )


class GGUFTokenizer:
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        revision=None,
        download_dir=None,
        trust_remote_code=False,
        **kwargs,
    ):
        if args:
            raise ValueError("GGUF tokenizer takes no positional tokenizer arguments")
        path = resolve_gguf_file(
            path_or_repo_id, revision=revision, cache_dir=download_dir
        )
        tokenizer = tokenizer_from_gguf_metadata(read_gguf_metadata(path), **kwargs)
        tokenizer._vllm_gguf_metadata = True
        return get_cached_tokenizer(tokenizer)
