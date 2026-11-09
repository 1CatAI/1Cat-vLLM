# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizers reconstructed from GGUF vocabulary and preprocessing metadata."""

from pathlib import Path

import regex
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


class _QwenGGUFTokenizer(PreTrainedTokenizerFast):
    """Reuse HF's per-piece BPE IDs for repeating long ASCII ``encode`` calls.

    Unicode, pairs, tensors and unsupported options use HF. In particular,
    ``__call__`` keeps its Encoding/offset APIs; only token-ID callers accelerate.
    """

    _qwen_piece_cache = None

    def add_tokens(self, *args, **kwargs):
        self._qwen_piece_cache = None
        return super().add_tokens(*args, **kwargs)

    def add_special_tokens(self, *args, **kwargs):
        self._qwen_piece_cache = None
        return super().add_special_tokens(*args, **kwargs)

    def encode(self, text, *args, **kwargs):
        cache = self._qwen_piece_cache
        if (
            cache is None
            or args
            or not isinstance(text, str)
            or len(text) < 1024
            or not text.isascii()
            or self.split_special_tokens
            or set(kwargs) - {"add_special_tokens", "truncation", "max_length"}
            or kwargs.get("truncation", False) not in (False, True)
        ):
            return super().encode(text, *args, **kwargs)
        max_length = kwargs.get("max_length")
        if max_length is not None and (
            not isinstance(max_length, int) or max_length < 0
        ):
            return super().encode(text, *args, **kwargs)
        # Added markers split before pretokenization, exactly as HF does.
        pieces = []
        for chunk in self._qwen_marker_pattern.split(text):
            if chunk in self._qwen_markers:
                pieces.append(chunk)
            else:
                pieces.extend(self._qwen_piece_pattern.findall(chunk))
        local_ids = {piece: cache.get(piece) for piece in pieces}
        missing = [piece for piece, ids in local_ids.items() if ids is None]
        if len(missing) > 128 or sum(map(len, missing)) > 65536:
            return super().encode(text, *args, **kwargs)
        if missing:
            encoded = (
                super()
                .__call__(
                    missing,
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                .input_ids
            )
            local_ids.update(
                (piece, tuple(ids)) for piece, ids in zip(missing, encoded)
            )
        # Bound memoization independently of request length and preserve every
        # current call's local dependency when evicting old entries.
        if len(cache) + len(missing) > 4096 or sum(map(len, cache.values())) > 65536:
            cache.clear()
            cache.update(self._qwen_markers)
        for piece in missing:
            value = local_ids[piece]
            if len(cache) < 4096 and len(value) <= 512:
                cache[piece] = value
        if sum(map(len, cache.values())) > 65536:
            cache.clear()
            cache.update(self._qwen_markers)
        ids = [token_id for piece in pieces for token_id in local_ids[piece]]
        if kwargs.get("truncation", False):
            limit = self.model_max_length if max_length is None else max_length
            if len(ids) > limit:
                ids = (
                    ids[-limit:]
                    if self.truncation_side == "left" and limit
                    else ids[:limit]
                )
        return ids


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
    tokenizer = _QwenGGUFTokenizer(
        tokenizer_object=backend,
        additional_special_tokens=special,
        chat_template=metadata.get("tokenizer.chat_template"),
        **tokenizer_kwargs,
        **kwargs,
    )
    added = {
        token.content: token_id
        for token_id, token in tokenizer.added_tokens_decoder.items()
    }
    # These factories use literal, non-normalized markers and no BOS/EOS
    # postprocessor. Additional caller-supplied markers retain HF qualification.
    if all(
        token.content.isascii()
        and not (token.single_word or token.lstrip or token.rstrip or token.normalized)
        for token in tokenizer.added_tokens_decoder.values()
    ) and not any(a in b for a in added for b in added if a != b):
        tokenizer._qwen_markers = {
            name: (token_id,) for name, token_id in added.items()
        }
        tokenizer._qwen_piece_cache = tokenizer._qwen_markers.copy()
        tokenizer._qwen_marker_pattern = (
            regex.compile("(" + "|".join(map(regex.escape, added)) + ")")
            if added
            else regex.compile(r"(?!x)x")
        )
        tokenizer._qwen_piece_pattern = regex.compile(
            _QWEN35_PATTERN if pre == "qwen35" else _QWEN2_PATTERN
        )
    return tokenizer


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
