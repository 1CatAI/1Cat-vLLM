# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host prompt-ngram lookup for the MRV2 DFlash2 assistant."""

from __future__ import annotations

import time

import numpy as np
from numba import njit


@njit(inline="always")
def _split_token(
    prefix: np.ndarray,
    prefix_len: int,
    suffix: np.ndarray,
    index: int,
) -> int:
    if index < prefix_len:
        return int(prefix[index])
    return int(suffix[index - prefix_len])


@njit(cache=True)
def find_split_ngram_proposal(
    prefix: np.ndarray,
    prefix_len: int,
    suffix: np.ndarray,
    suffix_len: int,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    num_draft_tokens: int,
) -> np.ndarray:
    """Run the prompt-lookup KMP over ``prefix + suffix`` without copying it."""
    total_tokens = prefix_len + suffix_len
    if total_tokens < min_ngram:
        return np.empty(0, dtype=np.int32)

    num_draft_tokens = min(num_draft_tokens, max_model_len - total_tokens)
    if num_draft_tokens <= 0:
        return np.empty(0, dtype=np.int32)

    # This is the same reverse-KMP policy as the standalone vLLM ngram
    # proposer: prefer the longest suffix match and the earliest occurrence.
    lps = np.zeros(max_ngram, dtype=np.int32)
    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    while i < total_tokens:
        prefix_token = _split_token(
            prefix,
            prefix_len,
            suffix,
            total_tokens - 1 - prev_lps,
        )
        current_token = _split_token(
            prefix,
            prefix_len,
            suffix,
            total_tokens - 1 - i,
        )
        if prefix_token == current_token:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return np.empty(0, dtype=np.int32)

    start = total_tokens - 1 - position + longest_ngram
    proposal_len = min(num_draft_tokens, total_tokens - start)
    proposal = np.empty(proposal_len, dtype=np.int32)
    for j in range(proposal_len):
        proposal[j] = _split_token(prefix, prefix_len, suffix, start + j)
    return proposal


class DFlash2NgramAssist:
    """Batch adapter and counters around the split prompt lookup."""

    def __init__(
        self,
        min_ngram: int,
        max_ngram: int,
        num_draft_tokens: int,
        max_model_len: int,
    ) -> None:
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.num_draft_tokens = num_draft_tokens
        self.max_model_len = max_model_len
        self.num_eligible = 0
        self.num_full_hits = 0
        self.lookup_seconds = 0.0

        # Compile the small split-input kernel at startup, not on the first
        # user-visible decode token.
        find_split_ngram_proposal(
            np.zeros(1, dtype=np.int32),
            1,
            np.zeros(1, dtype=np.int64),
            0,
            min_ngram,
            max_ngram,
            max_model_len,
            num_draft_tokens,
        )

    def propose(
        self,
        token_ids_cpu: np.ndarray,
        req_state_indices: np.ndarray,
        prior_lengths: np.ndarray,
        sampled_token_ids: np.ndarray,
        num_sampled_tokens: np.ndarray,
        eligible: np.ndarray,
        output_tokens: np.ndarray,
        output_lengths: np.ndarray,
    ) -> int:
        """Fill fixed-width outputs and return the number of full-width hits."""
        num_reqs = len(req_state_indices)
        output_tokens[:num_reqs].fill(0)
        output_lengths[:num_reqs].fill(0)
        full_hits = 0
        started = time.perf_counter()
        for batch_idx in range(num_reqs):
            if not eligible[batch_idx]:
                continue
            self.num_eligible += 1
            req_state_idx = int(req_state_indices[batch_idx])
            proposal = find_split_ngram_proposal(
                token_ids_cpu[req_state_idx],
                int(prior_lengths[batch_idx]),
                sampled_token_ids[batch_idx],
                int(num_sampled_tokens[batch_idx]),
                self.min_ngram,
                self.max_ngram,
                self.max_model_len,
                self.num_draft_tokens,
            )
            proposal_len = len(proposal)
            output_lengths[batch_idx] = proposal_len
            if proposal_len:
                output_tokens[batch_idx, :proposal_len] = proposal
            if proposal_len == self.num_draft_tokens:
                full_hits += 1
        self.num_full_hits += full_hits
        self.lookup_seconds += time.perf_counter() - started
        return full_hits
