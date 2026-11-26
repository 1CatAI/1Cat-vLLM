# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        self.copy_event = torch.cuda.Event()

        self.req_ids: list[str] = []
        self.num_draft_tokens: int = 0
        # Retain per-request proposals across interleaved non-structured or
        # pure-prefill steps until the scheduler consumes or retires each
        # request; hold the device slice alive until copy_event completes.
        self._pending_copies: list[tuple[list[str], torch.Tensor, np.ndarray]] = []
        self._latest_drafts: dict[str, list[int]] = {}
        self._removed_req_ids: set[str] = set()

    def remove_request(self, req_id: str) -> None:
        self._latest_drafts.pop(req_id, None)
        if self._pending_copies:
            self._removed_req_ids.add(req_id)

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        num_draft_tokens: int | None = None,
    ) -> None:
        batch_req_ids = list(input_batch.req_ids)
        self.req_ids = batch_req_ids
        if num_draft_tokens is None:
            num_draft_tokens = draft_tokens.shape[1]
        if not 0 <= num_draft_tokens <= draft_tokens.shape[1]:
            raise ValueError(
                "num_draft_tokens must be within the proposal tensor width"
            )
        self.num_draft_tokens = num_draft_tokens
        draft_tokens = draft_tokens[:, :num_draft_tokens]
        if not input_batch.has_structured_output_reqs:
            # Do not wipe proposals of structured-output requests that were not
            # in this batch (e.g. an interleaved pure-prefill step).
            for req_id in batch_req_ids:
                self._latest_drafts.pop(req_id, None)
            return

        for req_id in batch_req_ids:
            self._removed_req_ids.discard(req_id)

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            draft_tokens_np = async_copy_to_np(draft_tokens)
            self.copy_event.record()
        self._pending_copies.append((batch_req_ids, draft_tokens, draft_tokens_np))

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self._pending_copies:
            self.copy_event.synchronize()
            for batch_req_ids, _, batch_np in self._pending_copies:
                for req_id, row in zip(batch_req_ids, batch_np.tolist()):
                    if req_id not in self._removed_req_ids:
                        self._latest_drafts[req_id] = row
            self._pending_copies.clear()
            self._removed_req_ids.clear()
        if self._latest_drafts:
            return DraftTokenIds(
                list(self._latest_drafts.keys()),
                list(self._latest_drafts.values()),
            )
        # This case only happens when async scheduling is disabled or no
        # structured-output requests have proposed drafts yet.
        draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token used by a parallel-drafting checkpoint."""
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    raise ValueError(
        "Model config must specify dflash_config.mask_token_id, pard_token, "
        "ptd_token_id, or dspark_noise_token_id for parallel drafting."
    )
