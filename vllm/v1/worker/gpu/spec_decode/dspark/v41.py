# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 DSpark under Model Runner V2.

The V4.1 drafter (models/deepseek_v41/dspark.py) owns its CUDA graphs: the M5
query graph, the context M1..6 graphs and the B=1 greedy and coupled draft
graphs, all checked against stable input storage. This speculator keeps the
MRV2 DSpark data path (#1221) and adds what the V1 proposer
(v1/spec_decode/dspark.py) and runner give that drafter:

* the runner-level draft graph is off (the model's graphs run instead, as in
  V1), and its manager does not round the shared capture sizes;
* the target embedding is bound on the drafter's stage and the Markov shard
  path is prepared after the head is shared;
* the draft attention metadata uses the draft query slots (persistent) and a
  False prefill flag, so the query graph sees one geometry;
* draft sampling follows V1's order: greedy graph, coupled shard graph, fused
  coupled kernels, keyed per-step loop, then the MRV2 per-step loop;
* VLLM_V41_COUPLED_DRAFTS verification: keyed target samples and token
  matching (exact), from the gathered logits through the MRV2 sampler's own
  processing, or from the vocabulary shards (VLLM_V41_COUPLED_LOCAL) when only
  a temperature applies.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode import coupled_draw
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.deepseek_v41 import (
    DeepseekV41ModelState,
    V41AttnMetadata,
)
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.sm70_greedy_verify import greedy_verify
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def _language_model(model: nn.Module) -> nn.Module:
    get_language_model = getattr(model, "get_language_model", None)
    return get_language_model() if callable(get_language_model) else model


class V41DSparkSpeculator(DSparkSpeculator):
    _speculator_name = "DSpark (DeepSeek-V4.1)"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if self.draft_block != self.num_speculative_steps:
            raise ValueError(
                "DeepSeek-V4.1 DSpark under Model Runner V2 needs the draft block "
                f"({self.draft_block}) to equal num_speculative_tokens "
                f"({self.num_speculative_steps})."
            )
        # Per-round draft inputs, set by propose() for a real round:
        # (all rows greedy, [num_reqs] request state indices).
        self._round: tuple[bool, torch.Tensor] | None = None
        self._keyed_shard: Any = False  # False = not built yet
        self._local_check = [0, 0]
        self._paths_logged: set[str] = set()
        self._model_state: DeepseekV41ModelState | None = None

    def _log_path(
        self, path: str, message: str, *args: Any, warn: bool = False
    ) -> None:
        """Log a draft or verify path once. ``warn``: a requested fast path
        did not engage (the step still runs, more slowly or on MRV2's
        sampler)."""
        if path not in self._paths_logged:
            self._paths_logged.add(path)
            (logger.warning if warn else logger.info)(message, *args)

    # ------------------------------------------------------------------ setup

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # The drafter captures its own M5 query, context and draft graphs
        # against stable storage (query_graph.py), as under the V1 runner. A
        # runner-level FULL graph around them would capture those graphs inside
        # another capture. Query length 1 keeps the manager from rounding the
        # shared capture sizes in place (cudagraph_utils, legacy MRV1 path).
        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            CUDAGraphMode.NONE,
            decode_query_len=1,
        )
        logger.info(
            "DeepSeek-V4.1 DSpark under Model Runner V2: runner draft graph off "
            "(requested %s); the drafter's own M5 query, context and draft "
            "graphs run as under the V1 runner.",
            cudagraph_mode.name,
        )

    def capture(self) -> None:
        manager = self.query_cudagraph_manager
        if manager is None or not manager.needs_capture():
            return
        super().capture()

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        # The legacy sharing contract skips the embedding under pipeline
        # parallelism; V4.1 builds embed_tokens on every stage, and the drafter
        # binds the target's copy on its own (last) stage, as the V1 proposer.
        bind = getattr(model, "bind_pipeline_embedding", None)
        if bind is None:
            raise ValueError(
                "DeepSeek-V4.1 DSpark expects the V4.1 drafter "
                "(bind_pipeline_embedding)."
            )
        bind(_language_model(target_model))
        return model

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        # VLLM_V41_MARKOV_SHARD: checked against the head shared above.
        prepare = getattr(self.model, "prepare_markov_shard", None)
        if prepare is not None:
            prepare()
        # The expert storage (and any planner) exists once the target loaded.
        _raise_if_verify_truncation(self.device)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        if not isinstance(model_state, DeepseekV41ModelState):
            raise ValueError(
                "DeepSeek-V4.1 DSpark under Model Runner V2 needs the V4.1 model "
                f"state, got {type(model_state).__name__}."
            )
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._model_state = model_state

    # ------------------------------------------------------- draft metadata

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        num_query_per_req: int | None = None,
        causal: bool | Mapping[int, bool] = False,
        query_start_loc_np: np.ndarray | None = None,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        """As DraftModelSpeculator's, with the draft query slots (the V1
        proposer's slot_mapping, a persistent buffer) instead of the target's
        and a False prefill flag (only the target reads it)."""
        if not self.draft_attn_layer_names:
            return None
        if query_start_loc_np is not None or self.block_tables.cp_size > 1:
            raise ValueError(
                "DeepSeek-V4.1 DSpark under Model Runner V2 supports uniform "
                "draft queries without context parallelism."
            )
        query_per_req = self.num_query_per_req
        query_start_loc_cpu = (
            torch.clamp(self.arange[: num_reqs_padded + 1], max=num_reqs)
            * query_per_req
        )
        block_tables = [
            x[:num_reqs_padded] for x in self.block_tables.input_block_tables
        ]
        slot_mappings = self._require_query_slot_mappings()[:, :num_tokens_padded]
        draft_seq_lens_cpu_upper_bound = torch.zeros(
            num_reqs_padded, dtype=torch.int32, device="cpu"
        )
        torch.add(
            seq_lens_cpu_upper_bound[:num_reqs],
            step,
            out=draft_seq_lens_cpu_upper_bound[:num_reqs],
        )
        draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)
        return build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            query_start_loc_gpu=self.input_buffers.query_start_loc[
                : num_reqs_padded + 1
            ],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=query_per_req,
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            max_seq_len=self.draft_max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,
            model_specific_attn_metadata=V41AttnMetadata(
                torch.zeros(num_reqs_padded, dtype=torch.bool, device="cpu")
            ),
            causal=causal,
        )

    # --------------------------------------------------------------- drafts

    def propose(self, input_batch: InputBatch, *args: Any, **kwargs: Any):
        self._round = None
        if not kwargs.get("dummy_run", False) and not kwargs.get("is_profile", False):
            state = self._require_model_state()
            num_reqs = input_batch.num_reqs
            temperature = state.temperature_np[input_batch.idx_mapping_np[:num_reqs]]
            self._round = (
                bool(np.all(temperature == 0.0)),
                input_batch.idx_mapping[:num_reqs],
            )
        try:
            return super().propose(input_batch, *args, **kwargs)
        finally:
            self._round = None

    def _require_model_state(self) -> DeepseekV41ModelState:
        if self._model_state is None:
            raise RuntimeError("V4.1 DSpark attention state is not initialized.")
        return self._model_state

    def _coupled_drafts(self, all_greedy: bool) -> bool:
        return coupled_draw.ENABLED and not all_greedy and self.draft_logits is None

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        round_inputs = self._round
        if round_inputs is None or attn_metadata is None:
            # Dummy, profile and capture runs: the MRV2 DSpark path.
            super()._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            return
        all_greedy, idx_mapping = round_inputs
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        steps = self.draft_block
        num_sample = num_reqs * steps
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Anchors from the persistent query buffer (V1: self.input_ids[:1]).
        if num_reqs == 1:
            anchor = self.input_buffers.input_ids[:1]
        else:
            anchor = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        # Coupled draws: per-row temperature, stream key and the predicted
        # token's output position P (V1: proposer positions + 1), the same
        # position the target verifies the token at.
        coupled: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if self._coupled_drafts(all_greedy):
            coupled = (
                self.temperature[idx_mapping].to(torch.float32),
                self._require_model_state().keys_gpu[idx_mapping],
                self.sample_pos[:num_sample].view(num_reqs, steps),
            )

        if num_reqs == 1 and all_greedy and self.draft_logits is None:
            tokens = self.model.greedy_draft_tokens(sample_hidden, anchor)
            if tokens is not None:
                self._log_path(
                    "greedy_graph",
                    "DeepSeek-V4.1 DSpark (MRV2): greedy B=1 draft graph.",
                )
                self.draft_tokens[:1, :steps] = tokens.view(1, steps)
                return
            self._log_path(
                "greedy_graph_missed",
                "DeepSeek-V4.1 DSpark MRV2 fallback: the greedy B=1 draft graph "
                "did not run (the drafter has none); greedy drafts take the "
                "eager per-step loop (slower).",
                warn=True,
            )
        if coupled is not None:
            if num_reqs == 1:
                temperature, keys, positions = coupled
                tokens = self.model.coupled_shard_draft_tokens(
                    sample_hidden, anchor, temperature, keys, positions[0]
                )
                if tokens is not None:
                    self._log_path(
                        "coupled_shard",
                        "VLLM_V41_COUPLED_DRAFTS (MRV2): keyed drafts from the "
                        "coupled shard graph (VLLM_V41_COUPLED_SHARD).",
                    )
                    self.draft_tokens[:1, :steps] = tokens.view(1, steps)
                    return
            if _coupled_shard_enabled():
                self._log_path(
                    "coupled_shard_missed",
                    "VLLM_V41_COUPLED_SHARD MRV2 fallback: the coupled shard "
                    "draft graph did not run (%s); keyed drafts take the fused "
                    "Markov kernels or the per-step loop (slower).",
                    f"{num_reqs} rows; the graph is B=1"
                    if num_reqs != 1
                    else "no Markov shard: check VLLM_V41_MARKOV_SHARD and its "
                    "self-check line",
                    warn=True,
                )

        base_logits = self.model.compute_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, steps, base_logits.shape[-1])
        prev = anchor.to(torch.long)
        if coupled is not None:
            fused = self.model.coupled_draft_tokens(
                base_logits, prev, *coupled, coupled_draw.DRAFT_TEMP_SCALE
            )
            if fused is not None:
                self._log_path(
                    "coupled_fused",
                    "VLLM_V41_COUPLED_DRAFTS (MRV2): keyed drafts from the "
                    "fused Markov kernels (scale %.3g).",
                    coupled_draw.DRAFT_TEMP_SCALE,
                )
                for step in range(steps):
                    self.draft_tokens[:num_reqs, step] = self.model.map_draft_to_target(
                        fused[:, step]
                    )
                return
            self._log_path(
                "coupled_loop",
                "VLLM_V41_COUPLED_DRAFTS MRV2 fallback: keyed drafts, per-step "
                "loop (scale %.3g, slower): the fused Markov kernels did not "
                "run (no Markov table for this vocabulary).",
                coupled_draw.DRAFT_TEMP_SCALE,
                warn=True,
            )
        sample_idx = self.sample_idx_mapping[:num_sample].view(num_reqs, steps)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, steps)
        for step in range(steps):
            markov_embed = self.model.markov_embed(prev)
            logits = base_logits[:, step] + self.model.markov_bias(markov_embed)
            if coupled is not None:
                temperature, keys, positions = coupled
                sampled = coupled_draw.keyed_argmax(
                    logits,
                    temperature,
                    keys,
                    positions[:, step],
                    coupled_draw.DRAFT_TEMP_SCALE,
                )
                prev = self.model.map_draft_to_target(sampled)
            else:
                prev = self._sample_step(
                    logits, sample_idx[:, step], sample_pos[:, step], step
                )
            self.draft_tokens[:num_reqs, step] = prev

    # -------------------------------------------------- keyed verification

    def try_sample_target(
        self,
        model: nn.Module,
        rejection_sampler: Any,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: Any,
        *,
        allow_graph: bool = True,
    ) -> SamplerOutput | None:
        """VLLM_V41_COUPLED_DRAFTS: every verify row takes the target's keyed
        sample argmax(processed + G(key, position)); the longest draft prefix
        equal to those samples is accepted and the sample at the first
        mismatch is emitted (token matching), so the output is the target's
        own keyed samples whatever the drafts were. None (the regular
        rejection sampler) for anything the keyed path does not reproduce."""
        del allow_graph
        reason = self._keyed_verify_fallback(
            rejection_sampler, hidden_states, input_batch, grammar_output
        )
        if reason is not None:
            if reason not in _EXPECTED_UNKEYED:
                self._log_path(
                    f"verify_unkeyed_{reason}",
                    "VLLM_V41_COUPLED_DRAFTS MRV2 fallback: keyed verification "
                    "off for steps with %s; they take the MRV2 rejection "
                    "sampler, so their sampled outputs follow MRV2's sampler, "
                    "not the V1 runner's keyed streams.",
                    reason,
                    warn=True,
                )
            return None
        sampler = rejection_sampler.sampler
        state = self._require_model_state()
        expanded = input_batch.expanded_idx_mapping
        logits_indices = input_batch.logits_indices
        temperature = sampler.sampling_states.temperature.gpu[expanded]
        keys = state.keys_gpu[expanded]
        # Row i of the logits predicts the token after positions[i].
        positions = input_batch.positions[logits_indices] + 1
        draft_sampled = input_batch.input_ids[logits_indices]

        shard, local_reason = self._local_shard(model, sampler, input_batch)
        if local_reason is not None:
            self._log_path(
                f"verify_local_missed_{local_reason}",
                "VLLM_V41_COUPLED_LOCAL MRV2 fallback: keyed verification from "
                "the gathered logits for steps with %s (slower: full-vocabulary "
                "gather and the sampler's processing).",
                local_reason,
                warn=True,
            )
        if shard is not None:
            self._log_path(
                "verify_local",
                "VLLM_V41_COUPLED_DRAFTS (MRV2): keyed verification from the "
                "vocabulary shards (VLLM_V41_COUPLED_LOCAL).",
            )
            local = model.local_logits(hidden_states, shard.n).to(torch.float32)
            local.div_(temperature.unsqueeze(-1))
            sampled = shard(local, temperature, keys, positions)
            if _local_check_enabled():
                self._check_local(
                    model,
                    sampler,
                    hidden_states,
                    input_batch,
                    temperature,
                    keys,
                    positions,
                    sampled,
                )
        else:
            self._log_path(
                "verify_gathered",
                "VLLM_V41_COUPLED_DRAFTS (MRV2): keyed verification from the "
                "gathered logits after the MRV2 sampler's processing.",
            )
            processed = self._processed_logits(
                model, sampler, hidden_states, input_batch
            )
            sampled = coupled_draw.keyed_argmax(processed, temperature, keys, positions)
            if coupled_draw.TRACE:
                coupled_draw.trace_rows(
                    processed, temperature, keys, positions, sampled
                )
        sampled_ids, num_sampled = greedy_verify(
            sampled,
            draft_sampled,
            input_batch.cu_num_logits,
            rejection_sampler.num_speculative_steps,
        )
        return SamplerOutput(
            sampled_token_ids=sampled_ids,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
        )

    def _keyed_verify_fallback(
        self,
        rejection_sampler: Any,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: Any,
    ) -> str | None:
        """Why this step takes the regular rejection sampler; None when the
        keyed path gives the exact output distribution. The checks mirror
        the V1 gate (v1/sample/rejection_sampler._coupled_sampling_enabled)
        on MRV2's sampler states; all are host reads, no sync."""
        if not coupled_draw.ENABLED:
            return "off"
        if grammar_output is not None:
            return "grammar"
        if self.draft_logits is not None:
            return "probabilistic_drafts"
        if rejection_sampler is None or (
            rejection_sampler.synthetic_conditional_rates is not None
        ):
            return "synthetic"
        sampler = rejection_sampler.sampler
        if sampler.compute_nans:
            return "compute_nans"
        idx_np = input_batch.idx_mapping_np[: input_batch.num_reqs]
        if np.all(sampler.sampling_states.temperature.np[idx_np] == 0.0):
            return "all_greedy"
        if sampler.sampling_states.max_num_logprobs(idx_np) != NO_LOGPROBS:
            return "logprobs"
        if sampler.logprob_token_ids_state.max_num_token_ids(idx_np) != 0:
            return "logprob_token_ids"
        if np.any(sampler.logit_bias_state.use_logit_bias[idx_np]):
            return "logit_bias"
        if np.any(sampler.penalties_state.use_penalty[idx_np]):
            return "penalties"
        if np.any(sampler.bad_words_state.num_bad_words.np[idx_np]):
            return "bad_words"
        if (
            hidden_states.shape[0]
            != input_batch.num_reqs + input_batch.num_draft_tokens
        ):
            return "layout"
        return None

    def _processed_logits(
        self,
        model: nn.Module,
        sampler: Any,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> torch.Tensor:
        logits = model.compute_logits(hidden_states)
        logits_indices = input_batch.logits_indices
        return sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            input_batch.positions[logits_indices],
            input_batch.input_ids[logits_indices],
            input_batch.expanded_local_pos,
        )

    def _local_shard(
        self, model: nn.Module, sampler: Any, input_batch: InputBatch
    ) -> tuple[Any, str | None]:
        """(KeyedShard, None) when the step's processing is the temperature
        division alone (coupled_draw.local_fallback's conditions on MRV2
        states); (None, reason) when VLLM_V41_COUPLED_LOCAL=1 cannot apply;
        (None, None) when it is off."""
        if not _local_enabled():
            return None, None
        if coupled_draw.TRACE:
            return None, "VLLM_V41_COUPLED_TRACE"
        states = sampler.sampling_states
        idx_np = input_batch.idx_mapping_np[: input_batch.num_reqs]
        if not np.all(states.temperature.np[idx_np] > 0.0):
            return None, "greedy rows in a sampled batch"
        if np.any(states.top_k.np[idx_np] != states.vocab_size):
            return None, "top_k"
        if np.any(states.top_p.np[idx_np] != 1.0):
            return None, "top_p"
        if np.any(states.min_p.np[idx_np] != 0.0):
            return None, "min_p"
        shard = self._v41_keyed_shard(model)
        return shard, None if shard is not None else "no vocabulary shard layout"

    def _v41_keyed_shard(self, model: nn.Module):
        """coupled_draw.KeyedShard for the target's vocabulary shard, built
        once; None when the model does not lay its logits out as shards."""
        if self._keyed_shard is not False:
            return self._keyed_shard
        shard = None
        local_vocab = getattr(model, "local_vocab", None)
        layout = local_vocab() if callable(local_vocab) else None
        if layout is not None:
            from vllm.distributed import (
                get_tp_group,
                tensor_model_parallel_all_reduce,
            )

            group = get_tp_group()
            shard = coupled_draw.KeyedShard(
                *layout,
                group.rank_in_group,
                group.world_size,
                tensor_model_parallel_all_reduce,
            )
        logger.info(
            "V4.1 coupled local verify (VLLM_V41_COUPLED_LOCAL, MRV2): %s",
            f"vocabulary shard {layout[0]}+{layout[1]}"
            if layout
            else "gathered logits only",
        )
        self._keyed_shard = shard
        return shard

    def _check_local(
        self,
        model: nn.Module,
        sampler: Any,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        temperature: torch.Tensor,
        keys: torch.Tensor,
        positions: torch.Tensor,
        sampled: torch.Tensor,
    ) -> None:
        """Debug (VLLM_V41_COUPLED_LOCAL_CHECK): the gathered keyed picks for
        the same step; counts steps that differ (host sync; every rank runs it,
        so the all-gathers pair up)."""
        processed = self._processed_logits(model, sampler, hidden_states, input_batch)
        want = coupled_draw.keyed_argmax(processed, temperature, keys, positions)
        check = self._local_check
        check[0] += 1
        if not torch.equal(want, sampled):
            check[1] += 1
            logger.warning(
                "V4.1 coupled local verify (MRV2): picks differ from the "
                "gathered path (%d of %d steps)",
                check[1],
                check[0],
            )
        if check[0] % 200 == 0:
            logger.info(
                "V4.1 coupled local verify check (MRV2): %d mismatched of %d steps",
                check[1],
                check[0],
            )


def _raise_if_verify_truncation(device: torch.device) -> None:
    """The V1 runner masks drafts at verify positions the native expert
    planner did not compute (VLLM_V41_TRUNC, gguf_shared_runtime.current_trunc;
    the cutoff is set per step by VLLM_V41_TRUNC_CUTOFF). The planner
    truncates only with a cold expert set and free-slot exchange
    (VLLM_V41_SWAP_FREE=1) or a nonzero cutoff. Model Runner V2 has neither the
    mask nor the cutoff, so it refuses such a planner. A cold set of zero has
    no planner (current_trunc is None) and runs."""
    import vllm.envs as envs
    from vllm.model_executor.layers.quantization import gguf_shared_runtime

    swap_free = (envs.VLLM_V41_SWAP_FREE or "")[:1] == "1"
    cutoff = int(envs.VLLM_V41_TRUNC_CUTOFF or "0") != 0
    if not (swap_free or cutoff):
        return
    if gguf_shared_runtime.current_trunc(device) is not None:
        raise RuntimeError(
            "DeepSeek-V4.1 under Model Runner V2: verify truncation "
            "(VLLM_V41_SWAP_FREE=1 or VLLM_V41_TRUNC_CUTOFF with a cold expert "
            "set) is applied and masked only by the V1 runner. Set "
            "VLLM_V41_SWAP_FREE=0 and VLLM_V41_TRUNC_CUTOFF=0, keep every "
            "expert resident, or use VLLM_USE_V2_MODEL_RUNNER=0."
        )


# Verify fallbacks that are not a missed fast path: coupled drafts off, or an
# all-greedy step (token matching against the target argmax already).
_EXPECTED_UNKEYED = frozenset({"off", "all_greedy"})


def _local_enabled() -> bool:
    import vllm.envs as envs

    return envs.VLLM_V41_COUPLED_LOCAL == "1"


def _coupled_shard_enabled() -> bool:
    import vllm.envs as envs

    return envs.VLLM_V41_COUPLED_SHARD == "1"


def _local_check_enabled() -> bool:
    import vllm.envs as envs

    return envs.VLLM_V41_COUPLED_LOCAL_CHECK == "1"
