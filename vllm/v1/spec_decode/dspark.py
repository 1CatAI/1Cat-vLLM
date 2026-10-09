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
        self._confidence_threshold = self.speculative_config.dspark_confidence_threshold
        self._confidence_temperatures: torch.Tensor | None = None
        if self._confidence_threshold > 0.0:
            temperatures = self.speculative_config.dspark_confidence_temperatures
            if temperatures is None:
                temperatures = [1.0] * self.num_speculative_tokens
            self._confidence_temperatures = torch.tensor(
                temperatures,
                dtype=torch.float32,
                device=device,
            ).view(1, -1)
        configured_cap = self.speculative_config.dspark_max_verification_tokens
        self._max_verification_tokens = (
            self.num_speculative_tokens if configured_cap is None else configured_cap
        )
        self.confidence_scheduling_enabled = (
            self._confidence_threshold > 0.0
            or self._max_verification_tokens < self.num_speculative_tokens
        )
        # A fixed cap needs verification lengths but not a confidence
        # projection. Collect logits only when the calibrated threshold or the
        # explicit alignment diagnostic consumes them.
        self.collect_confidence_logits = (
            self.speculative_config.diagnostic_confidence_logits
        )
        self._last_confidence_logits: torch.Tensor | None = None
        self._last_verification_lengths: torch.Tensor | None = None

    def take_last_confidence_logits(self) -> torch.Tensor | None:
        confidence_logits = self._last_confidence_logits
        self._last_confidence_logits = None
        return confidence_logits

    def take_last_verification_lengths(self) -> torch.Tensor | None:
        verification_lengths = self._last_verification_lengths
        self._last_verification_lengths = None
        return verification_lengths

    def _select_verification_lengths(
        self,
        confidence_logits: torch.Tensor,
    ) -> torch.Tensor:
        temperatures = getattr(self, "_confidence_temperatures", None)
        if temperatures is None:
            temperatures = torch.ones(
                (1, confidence_logits.shape[1]),
                dtype=torch.float32,
                device=confidence_logits.device,
            )
        threshold = float(getattr(self, "_confidence_threshold", 0.0))
        cap = int(
            getattr(
                self,
                "_max_verification_tokens",
                confidence_logits.shape[1],
            )
        )
        if threshold <= 0.0:
            return torch.full(
                (confidence_logits.shape[0],),
                cap,
                dtype=torch.int32,
                device=confidence_logits.device,
            )

        conditional_probs = torch.sigmoid(confidence_logits.float() / temperatures)
        confident_prefix = (
            (conditional_probs >= threshold).to(torch.int32).cumprod(dim=1)
        )
        return confident_prefix.sum(dim=1).clamp_max_(cap).to(torch.int32)

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
        self._last_confidence_logits = None
        self._last_verification_lengths = None
        collect_confidence_logits = bool(
            getattr(self, "collect_confidence_logits", False)
        )
        markov_embeds: list[torch.Tensor] | None = (
            [] if collect_confidence_logits else None
        )
        draft_probs: list[torch.Tensor] | None = None
        for step in range(self.num_speculative_tokens):
            markov_embed = self.model.markov_embed(prev)
            if markov_embeds is not None:
                markov_embeds.append(markov_embed)
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

        if markov_embeds is not None:
            block_hidden_states = hidden_states.view(
                batch_size, self.num_speculative_tokens, -1
            )
            self._last_confidence_logits = self.model.confidence_logits(
                block_hidden_states,
                torch.stack(markov_embeds, dim=1),
            )
        if bool(getattr(self, "confidence_scheduling_enabled", False)):
            if self._last_confidence_logits is None:
                self._last_verification_lengths = torch.full(
                    (batch_size,),
                    int(self._max_verification_tokens),
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
            else:
                self._last_verification_lengths = self._select_verification_lengths(
                    self._last_confidence_logits
                )
        flat_tokens = torch.stack(draft_tokens, dim=1).reshape(-1)
        if draft_probs is None:
            return flat_tokens, None
        if len(draft_probs) != self.num_speculative_tokens:
            raise RuntimeError("DSpark produced incomplete draft probabilities.")
        flat_probs = torch.stack(draft_probs, dim=1).flatten(0, 1).contiguous()
        return flat_tokens, flat_probs
