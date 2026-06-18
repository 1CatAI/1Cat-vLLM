# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

DraftProbTokenIds = Sequence[Sequence[int]] | torch.Tensor


def _assert_cached_tokens_match(
    cached_tokens: torch.Tensor,
    expected_tokens: torch.Tensor,
    req_id: str,
) -> None:
    if cached_tokens.shape != expected_tokens.shape:
        raise RuntimeError(
            "Cached draft probability token ids do not match verifier "
            f"draft token shape for request {req_id}: "
            f"cached={tuple(cached_tokens.shape)}, "
            f"expected={tuple(expected_tokens.shape)}."
        )
    matches = (cached_tokens == expected_tokens).all()
    if cached_tokens.device.type == "cuda":
        # Avoid a CPU synchronization in the decode hot path. A mismatch here
        # is an invariant violation that would corrupt rejection sampling.
        torch._assert_async(matches)
    elif not bool(matches.item()):
        raise RuntimeError(
            "Cached draft probability token ids do not match verifier "
            f"draft tokens for request {req_id}; refusing to use "
            "misaligned draft probabilities."
        )


def clone_draft_prob_token_ids(
    draft_token_ids: Sequence[Sequence[int]] | torch.Tensor,
) -> DraftProbTokenIds:
    if torch.is_tensor(draft_token_ids):
        return draft_token_ids.detach().clone()
    return [list(token_ids) for token_ids in draft_token_ids]


def get_aligned_draft_probs(
    *,
    req_ids: Sequence[str],
    draft_probs: torch.Tensor | None,
    draft_prob_req_ids: Sequence[str] | None,
    draft_prob_token_ids: DraftProbTokenIds | None,
    spec_decode_metadata: SpecDecodeMetadata,
) -> torch.Tensor | None:
    if draft_probs is None or draft_prob_req_ids is None:
        return None

    if draft_prob_token_ids is None:
        raise RuntimeError(
            "Cached draft probabilities are missing their draft token ids; "
            "cannot verify draft token/probability alignment."
        )
    if draft_probs.ndim != 3:
        raise RuntimeError(
            "Cached draft probabilities must have shape "
            "[num_requests, max_draft_tokens, vocab_size], got "
            f"{tuple(draft_probs.shape)}."
        )
    if draft_probs.shape[0] != len(draft_prob_req_ids):
        raise RuntimeError(
            "Cached draft probability row count does not match request ids: "
            f"rows={draft_probs.shape[0]}, req_ids={len(draft_prob_req_ids)}."
        )
    if torch.is_tensor(draft_prob_token_ids):
        if draft_prob_token_ids.ndim != 2:
            raise RuntimeError(
                "Cached draft probability token ids must have shape "
                "[num_requests, max_draft_tokens], got "
                f"{tuple(draft_prob_token_ids.shape)}."
            )
        if draft_prob_token_ids.shape[0] != len(draft_prob_req_ids):
            raise RuntimeError(
                "Cached draft probability token rows do not match request ids: "
                f"rows={draft_prob_token_ids.shape[0]}, "
                f"req_ids={len(draft_prob_req_ids)}."
            )
    elif len(draft_prob_token_ids) != len(draft_prob_req_ids):
        raise RuntimeError(
            "Cached draft probability token rows do not match request ids: "
            f"rows={len(draft_prob_token_ids)}, "
            f"req_ids={len(draft_prob_req_ids)}."
        )

    if len(req_ids) != len(spec_decode_metadata.num_draft_tokens):
        raise RuntimeError(
            "Spec decode draft metadata batch size does not match current "
            f"requests: metadata={len(spec_decode_metadata.num_draft_tokens)}, "
            f"req_ids={len(req_ids)}."
        )
    expected_num_draft_tokens = sum(spec_decode_metadata.num_draft_tokens)
    if spec_decode_metadata.draft_token_ids.numel() != expected_num_draft_tokens:
        raise RuntimeError(
            "Spec decode draft token count does not match per-request counts: "
            f"draft_token_ids={spec_decode_metadata.draft_token_ids.numel()}, "
            f"num_draft_tokens={expected_num_draft_tokens}."
        )

    row_by_req_id = {req_id: idx for idx, req_id in enumerate(draft_prob_req_ids)}
    draft_probs_rows: list[torch.Tensor] = []
    draft_token_offset = 0
    for req_id, num_draft in zip(req_ids, spec_decode_metadata.num_draft_tokens):
        if num_draft == 0:
            continue
        row_idx = row_by_req_id.get(req_id)
        if row_idx is None:
            raise RuntimeError(
                f"Missing cached draft probabilities for request {req_id}; "
                "cannot verify draft token/probability alignment."
            )
        if draft_probs.shape[1] < num_draft:
            raise RuntimeError(
                "Cached draft probabilities do not have enough draft "
                f"positions for request {req_id}: "
                f"available={draft_probs.shape[1]}, needed={num_draft}."
            )

        expected_tokens = spec_decode_metadata.draft_token_ids[
            draft_token_offset : draft_token_offset + num_draft
        ]
        if torch.is_tensor(draft_prob_token_ids):
            if draft_prob_token_ids.shape[1] < num_draft:
                raise RuntimeError(
                    "Cached draft probability token ids do not have enough "
                    f"positions for request {req_id}: "
                    f"available={draft_prob_token_ids.shape[1]}, "
                    f"needed={num_draft}."
                )
            cached_tokens = draft_prob_token_ids[row_idx, :num_draft].to(
                device=expected_tokens.device,
                dtype=expected_tokens.dtype,
            )
        else:
            cached_token_row = draft_prob_token_ids[row_idx]
            if len(cached_token_row) < num_draft:
                raise RuntimeError(
                    "Cached draft probability token ids do not have enough "
                    f"positions for request {req_id}: "
                    f"available={len(cached_token_row)}, needed={num_draft}."
                )
            cached_tokens = torch.tensor(
                cached_token_row[:num_draft],
                device=expected_tokens.device,
                dtype=expected_tokens.dtype,
            )
        _assert_cached_tokens_match(cached_tokens, expected_tokens, req_id)

        draft_probs_rows.append(draft_probs[row_idx, :num_draft])
        draft_token_offset += num_draft

    if not draft_probs_rows:
        return None
    return torch.cat(draft_probs_rows, dim=0).contiguous()
