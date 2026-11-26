# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator for Model Runner V2: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(the DFlash machinery: context-KV precompute plus a query-block forward), then
adds intra-block dependency with a light sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly N query tokens
    (anchor + N-1 noise tokens), every query position predicts the next token
    (the ``SAMPLE_FROM_ANCHOR`` path of the shared prepare-inputs kernel).
  * Sequential Markov sampling: tokens are sampled left to right, adding a bias
    derived from the previously sampled token at each step.

The draft step, backbone forward and Markov sampling, runs inside the drafter's
FULL CUDA graph. Follows vllm-project/vllm's MRV2 DSparkSpeculator.
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.shared_weights import (
    LEGACY_DRAFT_WEIGHTS,
    share_embeddings,
    share_lm_head,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash.utils import get_parallel_draft_model

logger = init_logger(__name__)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        speculative_config = self.speculative_config
        if (
            speculative_config.dspark_confidence_threshold > 0.0
            or speculative_config.dspark_max_verification_tokens is not None
        ):
            raise ValueError(
                "DSpark confidence-scheduled verification is not supported by "
                "Model Runner V2; set VLLM_USE_V2_MODEL_RUNNER=0 to use it."
            )
        # DSpark consumes the target aux states combined to hidden_size by
        # main_proj, not DeepSeek-V4 MTP's hc_mult-wide residual for which the
        # base class widens hidden_size: size the staging buffers to the drafter.
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.inputs_embeds = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        # The anchor is the first prediction: N query positions per request.
        self.sample_from_anchor = True
        self.num_query_per_req = self.draft_block
        # Anchor (bonus) token of each request = input id at query offset 0.
        # A persistent index keeps the read graph-stable.
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        # Row s holds column s for every request: the draft_logits column each
        # sequential step writes, contiguous for the sampling kernel.
        self._step_cols = (
            torch.arange(self.draft_block, dtype=torch.int32, device=device)
            .repeat_interleave(self.max_num_reqs)
            .view(self.draft_block, self.max_num_reqs)
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = get_parallel_draft_model(self.vllm_config, "dspark_head")
        target_language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        # The DeepSeek-V4 drafter loads its own copy of the target embedding
        # from the checkpoint. With pipeline parallelism the target embedding
        # lives on the first stage, so the drafter keeps that copy: the same
        # contract as the V1 proposer.
        share_embeddings(
            model,
            target_language_model,
            LEGACY_DRAFT_WEIGHTS,
            pp_size=get_pp_group().world_size,
            log=logger,
        )
        share_lm_head(
            model,
            target_language_model,
            target_language_model,
            LEGACY_DRAFT_WEIGHTS,
            log=logger,
        )
        return model

    def _precompute_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        # The DeepSeek-V4 drafter takes its context slots per cache layer name.
        slots_by_layer: dict[str, torch.Tensor] | None = None
        if slots is not None:
            layer_names = self.model.get_draft_kv_cache_layer_names()
            per_layer = slots if isinstance(slots, list) else [slots] * len(layer_names)
            slots_by_layer = {
                name: layer_slots
                for name, layer_slots in zip(layer_names, per_layer)
                if layer_slots is not None
            }
        self.model.precompute_and_store_context_kv(
            hidden_states, positions, slots_by_layer
        )

    def _sample_step(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if self.draft_logits is None:
            return self.model.map_draft_to_target(logits.argmax(dim=-1))
        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sampled = gumbel_sample(
            logits,
            idx_mapping,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            output_processed_logits=self.draft_logits,
            output_processed_logits_col=self._step_cols[step, : logits.shape[0]],
            use_fp64=self.use_fp64_gumbel,
        )
        return self.model.map_draft_to_target(sampled)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_steps = self.draft_block
        num_sample = num_reqs * num_steps
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, num_steps, base_logits.shape[-1])
        idx_mapping = self.sample_idx_mapping[:num_sample].view(num_reqs, num_steps)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, num_steps)

        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]].to(torch.long)
        for step in range(num_steps):
            markov_embed = self.model.markov_embed(prev)
            logits = base_logits[:, step] + self.model.markov_bias(markov_embed)
            prev = self._sample_step(
                logits, idx_mapping[:, step], sample_pos[:, step], step
            )
            self.draft_tokens[:num_reqs, step] = prev
