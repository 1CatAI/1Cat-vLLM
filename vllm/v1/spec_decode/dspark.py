# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark proposer built on the existing non-causal DFlash execution path."""

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer


class DSparkProposer(DFlashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner)
        if self.use_local_argmax_reduction:
            raise ValueError(
                "DSpark cannot use local argmax reduction because its replicated "
                "Markov bias must be added to full-vocabulary logits."
            )
        self._anchor_indices = (
            torch.arange(self.max_batch_size, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logits: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del spec_step_idx

        num_rows = hidden_states.shape[0]
        batch_size, remainder = divmod(num_rows, self.num_speculative_tokens)
        if remainder:
            raise ValueError(
                "DSpark sample rows must be divisible by "
                f"num_speculative_tokens={self.num_speculative_tokens}, got {num_rows}."
            )
        if logits is None:
            logits = self.model.compute_logits(hidden_states)
        base_logits = logits.view(batch_size, self.num_speculative_tokens, -1)

        # Read anchors from the persistent expanded-query buffer. The external
        # next_token_ids tensor is not guaranteed to retain a stable address or
        # contents across asynchronous scheduling and CUDA Graph replay.
        prev = self.input_ids[self._anchor_indices[:batch_size]].to(torch.long)
        draft_tokens: list[torch.Tensor] = []
        draft_probs: list[torch.Tensor] | None = None
        for step in range(self.num_speculative_tokens):
            markov_embed = self.model.markov_embed(prev)
            step_logits = base_logits[:, step] + self.model.markov_bias(markov_embed)
            sampled, step_probs = self._sample_from_logits(
                step_logits,
                sampling_metadata,
            )
            prev = self.model.map_draft_to_target(sampled)
            draft_tokens.append(prev)
            if step_probs is not None:
                if draft_probs is None:
                    draft_probs = []
                draft_probs.append(step_probs)

        flat_tokens = torch.stack(draft_tokens, dim=1).reshape(-1)
        if draft_probs is None:
            return flat_tokens, None
        if len(draft_probs) != self.num_speculative_tokens:
            raise RuntimeError("DSpark produced incomplete draft probabilities.")
        flat_probs = torch.stack(draft_probs, dim=1).flatten(0, 1).contiguous()
        return flat_tokens, flat_probs
