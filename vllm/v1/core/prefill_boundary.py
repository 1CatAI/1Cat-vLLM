# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proof for the one-token lookahead of an immutable MTP prompt block."""

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    hash_block_tokens,
)
from vllm.v1.request import Request


def prefill_boundary_hash(
    request: Request, boundary: int, prefix_hash: BlockHash
) -> BlockHash | None:
    # Never certify sampled/draft tokens or the final prompt boundary. The
    # next input must already be known when the original block is executed.
    # Multimodal/prompt-embedding paths retain conservative replay until their
    # shifted-embedding contracts have independent end-to-end coverage.
    if (
        not 0 < boundary < request.num_prompt_tokens
        or request.mm_features
        or request.prompt_embeds is not None
    ):
        return None
    extra_keys, _ = generate_block_hash_extra_keys(request, boundary, boundary + 1, 0)
    return hash_block_tokens(
        sha256, prefix_hash, [request.all_token_ids[boundary]], extra_keys
    )
