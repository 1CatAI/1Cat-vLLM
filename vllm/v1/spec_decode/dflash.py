# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from typing_extensions import override

from vllm import envs
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.ddtree_payload import (
    DDTreeDraftPayload,
    build_ddtree_payloads_from_logits,
    build_ddtree_payloads_from_topk,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    copy_and_expand_dflash_inputs_kernel,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    next_power_of_2,
)
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def _ddtree_debug_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_DEBUG", "0") == "1"


def _ddtree_debug_log(message: str, *args: object) -> None:
    if _ddtree_debug_enabled():
        logger.info(
            "DFlash DDTree proposer debug: %s",
            message % args if args else message,
        )


def _ddtree_worker_profile_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_WORKER_PROFILE", "0") == "1"


class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.use_dflash()
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        self.use_ddtree = vllm_config.speculative_config.use_dflash_ddtree()
        self.ddtree_budget = vllm_config.speculative_config.ddtree_budget
        self.ddtree_top_k = vllm_config.speculative_config.ddtree_top_k
        self.ddtree_chain_seed = vllm_config.speculative_config.ddtree_chain_seed
        self.ddtree_tree_mode = vllm_config.speculative_config.ddtree_tree_mode
        self.ddtree_disable_tree_verify = (
            vllm_config.speculative_config.ddtree_disable_tree_verify
        )
        self._last_ddtree_payloads: tuple[DDTreeDraftPayload, ...] | None = None
        _ddtree_debug_log(
            "init use_ddtree=%s budget=%s top_k=%s chain_seed=%s "
            "tree_mode=%s "
            "disable_tree_verify=%s num_spec=%s",
            self.use_ddtree,
            self.ddtree_budget,
            self.ddtree_top_k,
            self.ddtree_chain_seed,
            self.ddtree_tree_mode,
            self.ddtree_disable_tree_verify,
            self.num_speculative_tokens,
        )

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        # Query buffers must cover CUDA-graph padding, which can exceed the
        # actual number of DFlash query tokens.
        self._slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # For DFlash we use the input embeddings to embed the mask token
        self.parallel_drafting_hidden_state_tensor = None
        self._dumped_first_pass = False
        self._profile_rounds = 0
        self._profile_context_tokens = 0
        self._profile_query_tokens = 0
        self._profile_padded_query_tokens = 0
        self._profile_expand_elapsed_s = 0.0
        self.draft_layer_to_kv_cache_gid: dict[str, int] = {}
        self._query_slot_mapping_buffers_by_gid: dict[int, torch.Tensor] = {}
        self._context_slot_mapping_buffers_by_gid: dict[int, torch.Tensor] = {}
        self._dflash_common_attn_metadata_by_gid: (
            dict[int, CommonAttentionMetadata] | None
        ) = None
        self._dflash_new_common_attn_metadata_by_gid: dict[
            int, CommonAttentionMetadata
        ] = {}
        self._dflash_block_size_by_gid: dict[int, int] = {}

        self.dflash_causal = self.dflash_config.get("causal", False)

    def warmup_sm70_dflash_hotpath_kernels(self) -> tuple[str, ...]:
        """Warm DFlash helper kernels that otherwise JIT on first request."""
        if self.device.type != "cuda" or not current_platform.is_device_capability(70):
            return ()
        if getattr(self, "_sm70_dflash_hotpath_warmed", False):
            return ()
        self._sm70_dflash_hotpath_warmed = True

        try:
            batch_size = 1
            discard_request_mask = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=self.device,
            )
            backup_next_token_ids = torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=self.device,
            )
            next_token_ids = torch.empty(
                batch_size,
                dtype=torch.int32,
                device=self.device,
            )
            valid_sampled_tokens_count = None
            vocab_size = max(2, self.draft_model_config.get_vocab_size())
            for num_sampled_tokens in sorted({1, self.num_speculative_tokens + 1}):
                sampled_token_ids = torch.zeros(
                    (batch_size, num_sampled_tokens),
                    dtype=torch.int32,
                    device=self.device,
                )
                next_valid_count = torch.empty_like(next_token_ids)
                eagle_prepare_next_token_padded_kernel[(batch_size,)](
                    sampled_token_ids,
                    discard_request_mask,
                    backup_next_token_ids,
                    next_token_ids,
                    next_valid_count,
                    vocab_size,
                    num_sampled_tokens,
                    batch_size,
                    sampled_token_ids.stride(0),
                    BLOCK_SIZE_TOKENS=next_power_of_2(num_sampled_tokens),
                )
                if num_sampled_tokens == self.num_speculative_tokens + 1:
                    valid_sampled_tokens_count = next_valid_count
            assert valid_sampled_tokens_count is not None

            cu_num_draft_tokens = torch.full(
                (batch_size,),
                self.num_speculative_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            query_start_loc = (
                torch.arange(
                    batch_size + 1,
                    dtype=torch.int32,
                    device=self.device,
                )
                * num_sampled_tokens
            )
            token_indices_to_sample = torch.empty_like(next_token_ids)
            num_rejected_tokens_gpu = torch.empty_like(next_token_ids)
            eagle_prepare_inputs_padded_kernel[(batch_size,)](
                cu_num_draft_tokens,
                valid_sampled_tokens_count,
                query_start_loc,
                token_indices_to_sample,
                num_rejected_tokens_gpu,
                batch_size,
            )

            num_query_per_req = num_sampled_tokens
            block_size = max(1, int(self.block_size))
            block_table = torch.zeros(
                (
                    batch_size,
                    max(1, (self.max_model_len + block_size - 1) // block_size),
                ),
                dtype=torch.int32,
                device=self.device,
            )
            out_input_ids = torch.empty(
                batch_size * num_query_per_req,
                dtype=torch.int32,
                device=self.device,
            )
            out_query_positions = torch.empty(
                batch_size * num_query_per_req,
                dtype=torch.int64,
                device=self.device,
            )
            out_query_slot_mapping = torch.empty(
                batch_size * num_query_per_req,
                dtype=torch.int64,
                device=self.device,
            )
            out_token_indices = torch.empty(
                batch_size * self.num_speculative_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # Cover both chunk-aligned and first-request prompt lengths. The
            # coding smoke prompt has 63 context tokens, which pairs with
            # BLOCK_SIZE=128 and used to trigger an inference-time JIT.
            for num_context in sorted({1, self.num_speculative_tokens + 1, 63, 64}):
                max_tokens_per_req = num_context + num_query_per_req
                target_positions = torch.arange(
                    num_context,
                    dtype=torch.int64,
                    device=self.device,
                )
                out_context_positions = torch.empty(
                    num_context,
                    dtype=torch.int64,
                    device=self.device,
                )
                out_context_slot_mapping = torch.empty(
                    num_context,
                    dtype=torch.int64,
                    device=self.device,
                )
                dflash_query_start_loc = torch.tensor(
                    [0, num_context],
                    dtype=torch.int32,
                    device=self.device,
                )
                copy_block_size = min(256, next_power_of_2(max_tokens_per_req))
                grid = (
                    batch_size,
                    triton.cdiv(max_tokens_per_req, copy_block_size),
                )
                for has_num_rejected in (False, True):
                    copy_and_expand_dflash_inputs_kernel[grid](
                        next_token_ids,
                        target_positions,
                        out_input_ids,
                        out_context_positions,
                        out_query_positions,
                        out_context_slot_mapping,
                        out_query_slot_mapping,
                        out_token_indices,
                        block_table,
                        block_table.stride(0),
                        dflash_query_start_loc,
                        num_rejected_tokens_gpu if has_num_rejected else 0,
                        self.parallel_drafting_token_id,
                        block_size,
                        num_query_per_req,
                        self.num_speculative_tokens,
                        num_context,
                        BLOCK_SIZE=copy_block_size,
                        HAS_NUM_REJECTED=has_num_rejected,
                    )
            torch.accelerator.synchronize(self.device)
        except Exception as err:  # pragma: no cover - best-effort warmup
            logger.warning_once("SM70 DFlash hotpath warmup skipped: %s", err)
            return ()

        return (
            "dflash_prepare_next_token",
            "dflash_prepare_inputs",
            "dflash_expand_inputs",
        )

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        return replace(
            base,
            attention_config=replace(
                base.attention_config,
                use_non_causal=not self.dflash_causal,
            ),
        )

    @override
    def _warn_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        pass

    def _draft_kv_cache_group_ids(self) -> tuple[int, ...]:
        gids = set(self.draft_layer_to_kv_cache_gid.values())
        if not gids and self.kv_cache_gid >= 0:
            gids.add(self.kv_cache_gid)
        return tuple(sorted(gids))

    def _reset_slot_mapping_buffers_for_groups(self) -> None:
        self._query_slot_mapping_buffers_by_gid = {}
        self._context_slot_mapping_buffers_by_gid = {}
        for idx, gid in enumerate(self._draft_kv_cache_group_ids()):
            if idx == 0:
                self._query_slot_mapping_buffers_by_gid[gid] = self._slot_mapping_buffer
                self._context_slot_mapping_buffers_by_gid[gid] = (
                    self._context_slot_mapping_buffer
                )
            else:
                self._query_slot_mapping_buffers_by_gid[gid] = torch.zeros(
                    self.max_num_tokens,
                    dtype=torch.int64,
                    device=self.device,
                )
                self._context_slot_mapping_buffers_by_gid[gid] = torch.zeros(
                    self.max_num_tokens,
                    dtype=torch.int64,
                    device=self.device,
                )

    def _query_slot_mapping_buffer_for_gid(self, gid: int) -> torch.Tensor:
        if gid not in self._query_slot_mapping_buffers_by_gid:
            self._reset_slot_mapping_buffers_for_groups()
        return self._query_slot_mapping_buffers_by_gid[gid]

    def _context_slot_mapping_buffer_for_gid(self, gid: int) -> torch.Tensor:
        if gid not in self._context_slot_mapping_buffers_by_gid:
            self._reset_slot_mapping_buffers_for_groups()
        return self._context_slot_mapping_buffers_by_gid[gid]

    def _first_draft_kv_cache_group_id(self) -> int:
        gids = self._draft_kv_cache_group_ids()
        if gids:
            return gids[0]
        return self.kv_cache_gid

    def take_last_ddtree_payloads(self) -> tuple[DDTreeDraftPayload, ...] | None:
        return self._last_ddtree_payloads

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logits: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_ddtree or spec_step_idx != 0:
            _ddtree_debug_log(
                "skip build use_ddtree=%s spec_step_idx=%s hidden_shape=%s",
                self.use_ddtree,
                spec_step_idx,
                tuple(hidden_states.shape),
            )
            return super()._sample_draft_tokens(
                hidden_states,
                sampling_metadata,
                logits,
                spec_step_idx=spec_step_idx,
            )

        self._last_ddtree_payloads = None
        profile_enabled = _ddtree_worker_profile_enabled()
        budget = self.ddtree_budget or self.num_speculative_tokens
        requested_top_k = self.ddtree_top_k or budget
        model = getattr(self, "model", None)
        topk_payload_fn = getattr(model, "get_topk_tokens_and_logprobs", None)
        if (
            logits is None
            and sampling_metadata.all_greedy
            and topk_payload_fn is not None
        ):
            topk_t0 = time.perf_counter() if profile_enabled else 0.0
            topk_result = topk_payload_fn(hidden_states, requested_top_k)
            topk_extract_ms = (
                (time.perf_counter() - topk_t0) * 1000.0 if profile_enabled else 0.0
            )
            if topk_result is not None:
                topk_token_ids, topk_logprobs = topk_result
                draft_token_ids = topk_token_ids[:, 0]
                batch_size, remainder = divmod(
                    int(draft_token_ids.numel()), self.num_speculative_tokens
                )
                if remainder != 0:
                    raise ValueError(
                        "DFlash DDTree draft rows must be divisible by "
                        f"num_speculative_tokens={self.num_speculative_tokens}"
                    )
                payload_t0 = time.perf_counter() if profile_enabled else 0.0
                self._last_ddtree_payloads = build_ddtree_payloads_from_topk(
                    topk_token_ids=topk_token_ids,
                    topk_logprobs=topk_logprobs,
                    batch_size=batch_size,
                    num_speculative_tokens=self.num_speculative_tokens,
                    budget=budget,
                    chain_seed=self.ddtree_chain_seed,
                    tree_mode=self.ddtree_tree_mode,
                    flat_draft_token_ids=draft_token_ids.view(
                        batch_size,
                        self.num_speculative_tokens,
                    ),
                )
                payload_ms = (
                    (time.perf_counter() - payload_t0) * 1000.0
                    if profile_enabled
                    else 0.0
                )
                if profile_enabled:
                    logger.info(
                        "DFLASH_DDTREE_WORKER_PROFILE payload_stage "
                        "topk_extract_ms=%.3f payload_build_ms=%.3f "
                        "batch=%d spec_tokens=%d budget=%d top_k=%s "
                        "path=compact_topk",
                        topk_extract_ms,
                        payload_ms,
                        batch_size,
                        self.num_speculative_tokens,
                        budget,
                        requested_top_k,
                    )
                return draft_token_ids, None

        if logits is None:
            logits = self._compute_logits_for_step(hidden_states, spec_step_idx)
        if not hasattr(self, "_static_draft_vocab"):
            self._static_draft_vocab = None
        sample_t0 = time.perf_counter() if profile_enabled else 0.0
        draft_token_ids, draft_probs = super()._sample_draft_tokens(
            hidden_states,
            sampling_metadata,
            logits,
            spec_step_idx=spec_step_idx,
        )
        sample_ms = (
            (time.perf_counter() - sample_t0) * 1000.0 if profile_enabled else 0.0
        )

        batch_size, remainder = divmod(
            int(draft_token_ids.numel()), self.num_speculative_tokens
        )
        if remainder != 0:
            raise ValueError(
                "DFlash DDTree draft rows must be divisible by "
                f"num_speculative_tokens={self.num_speculative_tokens}"
            )
        payload_t0 = time.perf_counter() if profile_enabled else 0.0
        self._last_ddtree_payloads = build_ddtree_payloads_from_logits(
            logits=logits.detach(),
            batch_size=batch_size,
            num_speculative_tokens=self.num_speculative_tokens,
            budget=budget,
            top_k=self.ddtree_top_k,
            chain_seed=self.ddtree_chain_seed,
            tree_mode=self.ddtree_tree_mode,
            flat_draft_token_ids=draft_token_ids.view(
                batch_size,
                self.num_speculative_tokens,
            ),
        )
        payload_ms = (
            (time.perf_counter() - payload_t0) * 1000.0 if profile_enabled else 0.0
        )
        if profile_enabled:
            logger.info(
                "DFLASH_DDTREE_WORKER_PROFILE payload_stage "
                "flat_sample_ms=%.3f payload_build_ms=%.3f "
                "batch=%d spec_tokens=%d budget=%d top_k=%s "
                "logits_rows=%d vocab=%d",
                sample_ms,
                payload_ms,
                batch_size,
                self.num_speculative_tokens,
                budget,
                self.ddtree_top_k,
                int(logits.shape[0]),
                int(logits.shape[1]),
            )
        if self._last_ddtree_payloads:
            first_payload = self._last_ddtree_payloads[0]
            _ddtree_debug_log(
                "built payloads batch=%s flat_len=%s tree_len=%s has_extra_nodes=%s",
                len(self._last_ddtree_payloads),
                len(first_payload.flat_draft_token_ids),
                len(first_payload.tree_token_ids),
                len(first_payload.tree_token_ids)
                > len(first_payload.flat_draft_token_ids),
            )
        else:
            _ddtree_debug_log("built no payloads batch=%s", batch_size)
        return draft_token_ids, draft_probs

    @override
    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        kv_cache_layers = {
            layer_name
            for kv_cache_group in kv_cache_config.kv_cache_groups
            for layer_name in kv_cache_group.layer_names
        }
        missing = self._draft_attn_layer_names - kv_cache_layers
        assert not missing, f"DFlash draft layers missing from KV cache: {missing}"

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        self.validate_same_kv_cache_group(kv_cache_config)
        self.draft_layer_to_kv_cache_gid = {}
        self.kv_cache_gid = -1
        layer_to_gid: dict[str, int] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                layer_to_gid[layer_name] = gid
            if self.kv_cache_gid < 0 and self._draft_attn_layer_names & set(
                group.layer_names
            ):
                self.kv_cache_gid = gid

        attention_groups: dict[tuple[int, tuple[str, str]], AttentionGroup] = {}
        for layer_name in self._draft_attn_layer_names:
            kv_cache_gid = layer_to_gid[layer_name]
            self.draft_layer_to_kv_cache_gid[layer_name] = kv_cache_gid
            kv_cache_spec = kv_cache_config.kv_cache_groups[kv_cache_gid].kv_cache_spec
            layer_kv_cache_spec = kv_cache_spec
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]

            attn_backend = all_attn_layers[layer_name].get_attn_backend()
            group_key = (kv_cache_gid, attn_backend.full_cls_name())
            if group_key not in attention_groups:
                kernel_block_size = (
                    kernel_block_sizes[kv_cache_gid]
                    if kernel_block_sizes is not None
                    and kv_cache_gid < len(kernel_block_sizes)
                    else None
                )
                attn_group = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=layer_kv_cache_spec,
                    kv_cache_group_id=kv_cache_gid,
                )
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                attention_groups[group_key] = attn_group
            else:
                attention_groups[group_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        if self.draft_attn_groups:
            self.block_size = (
                self.draft_attn_groups[0]
                .get_metadata_builder()
                .kv_cache_spec.block_size
            )
        self._reset_slot_mapping_buffers_for_groups()
        self._dflash_block_size_by_gid = {
            group.kv_cache_group_id: (
                group.get_metadata_builder().kv_cache_spec.block_size
            )
            for group in self.draft_attn_groups
        }
        logger.debug("Using block size %d for drafting layers", self.block_size)

    def set_common_attn_metadata_by_kv_cache_group(
        self,
        metadata_by_gid: dict[int, CommonAttentionMetadata],
    ) -> None:
        self._dflash_common_attn_metadata_by_gid = metadata_by_gid

    @override
    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # DFlash writes query slot mappings directly in set_inputs_first_pass.
        # Re-copying common metadata here would overwrite the query-only layout.
        return {
            name: self._query_slot_mapping_buffer_for_gid(
                self.draft_layer_to_kv_cache_gid.get(name, self.kv_cache_gid)
            )[:num_tokens]
            for name in self._draft_attn_layer_names
        }

    def _pad_query_buffers(
        self,
        *,
        num_query_tokens: int,
        num_input_tokens: int,
    ) -> None:
        if num_input_tokens <= num_query_tokens:
            return
        self.input_ids[num_query_tokens:num_input_tokens].zero_()
        self.positions[num_query_tokens:num_input_tokens].zero_()
        buffers: Iterable[torch.Tensor] = (
            self._query_slot_mapping_buffers_by_gid.values()
        )
        if not self._query_slot_mapping_buffers_by_gid:
            buffers = [self._slot_mapping_buffer]
        for buffer in buffers:
            buffer[num_query_tokens:num_input_tokens].fill_(PADDING_SLOT_ID)

    def _maybe_dump_first_pass(
        self,
        *,
        cad: CommonAttentionMetadata,
        new_cad: CommonAttentionMetadata,
        target_positions: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        num_context: int,
        num_query_total: int,
        effective_seq_lens: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> None:
        if self._dumped_first_pass or not envs.VLLM_DFLASH_DUMP_FIRST_PASS:
            return

        dump_path = f"/tmp/dflash_first_pass_pid{os.getpid()}.pt"
        torch.save(
            {
                "block_size": self.block_size,
                "num_context": num_context,
                "num_query_total": num_query_total,
                "num_speculative_tokens": self.num_speculative_tokens,
                "cad_query_start_loc": cad.query_start_loc.detach().cpu(),
                "cad_seq_lens": cad.seq_lens.detach().cpu(),
                "cad_block_table": cad.block_table_tensor.detach().cpu(),
                "cad_slot_mapping": cad.slot_mapping.detach().cpu(),
                "target_positions": target_positions.detach().cpu(),
                "next_token_ids": next_token_ids.detach().cpu(),
                "context_positions": self._context_positions_buffer[:num_context]
                .detach()
                .cpu(),
                "context_slot_mapping": self._context_slot_mapping_buffer[:num_context]
                .detach()
                .cpu(),
                "query_positions": self.positions[:num_query_total].detach().cpu(),
                "query_slot_mapping": self._slot_mapping_buffer[:num_query_total]
                .detach()
                .cpu(),
                "token_indices_to_sample": token_indices_to_sample.detach().cpu(),
                "effective_seq_lens": effective_seq_lens.detach().cpu(),
                "new_cad_query_start_loc": new_cad.query_start_loc.detach().cpu(),
                "new_cad_seq_lens": new_cad.seq_lens.detach().cpu(),
                "new_cad_block_table": new_cad.block_table_tensor.detach().cpu(),
                "new_cad_slot_mapping": new_cad.slot_mapping.detach().cpu(),
                "num_rejected_tokens_gpu": None
                if num_rejected_tokens_gpu is None
                else num_rejected_tokens_gpu.detach().cpu(),
            },
            dump_path,
        )
        self._dumped_first_pass = True
        logger.warning("Saved DFlash first-pass metadata to %s", dump_path)

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # Store for build_model_inputs_first_pass to use
        self._dflash_num_context = num_context
        self._dflash_num_query_tokens = num_query_total

        # We don't need to copy into a buffer here since the context preprocessing
        # does not run in a CUDA graph
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Launch fused triton kernel for input_ids, positions, slot_mapping,
        # and token_indices_to_sample
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        start_event = None
        end_event = None
        if envs.VLLM_DFLASH_PROFILE and self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        # In padded mode, cad.seq_lens includes rejected tokens. Subtract
        # them so attention only sees the valid prefix of context states.
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        first_gid = self._first_draft_kv_cache_group_id()
        common_metadata_by_gid = self._dflash_common_attn_metadata_by_gid
        if common_metadata_by_gid is None:
            common_metadata_by_gid = {first_gid: cad}
        elif first_gid not in common_metadata_by_gid:
            common_metadata_by_gid = {**common_metadata_by_gid, first_gid: cad}

        new_cad_by_gid: dict[int, CommonAttentionMetadata] = {}
        for gid in self._draft_kv_cache_group_ids():
            group_cad = common_metadata_by_gid.get(gid, cad)
            context_slot_mapping_buffer = self._context_slot_mapping_buffer_for_gid(gid)
            query_slot_mapping_buffer = self._query_slot_mapping_buffer_for_gid(gid)
            block_size = self._dflash_block_size_by_gid.get(gid, self.block_size)
            copy_and_expand_dflash_inputs_kernel[grid](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=context_slot_mapping_buffer,
                out_query_slot_mapping_ptr=query_slot_mapping_buffer,
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=group_cad.block_table_tensor,
                block_table_stride=group_cad.block_table_tensor.stride(0),
                # Metadata
                query_start_loc_ptr=group_cad.query_start_loc,
                num_rejected_tokens_ptr=(
                    num_rejected_tokens_gpu if has_num_rejected else 0
                ),
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=block_size,
                num_query_per_req=num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=num_context,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_NUM_REJECTED=has_num_rejected,
            )
            group_effective_seq_lens = group_cad.seq_lens
            if has_num_rejected:
                group_effective_seq_lens = (
                    group_effective_seq_lens - num_rejected_tokens_gpu
                )
            new_seq_lens_cpu_upper_bound = (
                group_cad.seq_lens_cpu_upper_bound + num_query_per_req
                if group_cad.seq_lens_cpu_upper_bound is not None
                else None
            )
            new_cad_by_gid[gid] = CommonAttentionMetadata(
                query_start_loc=new_query_start_loc,
                seq_lens=group_effective_seq_lens + num_query_per_req,
                query_start_loc_cpu=(
                    torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                    * num_query_per_req
                ),
                _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
                seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
                num_reqs=group_cad.num_reqs,
                num_actual_tokens=num_query_total,
                max_query_len=num_query_per_req,
                max_seq_len=group_cad.max_seq_len + num_query_per_req,
                block_table_tensor=group_cad.block_table_tensor,
                slot_mapping=query_slot_mapping_buffer[:num_query_total],
                causal=self.dflash_causal,
            )
        expand_elapsed_s = 0.0
        if start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            expand_elapsed_s = start_event.elapsed_time(end_event) / 1000.0

        self._dflash_new_common_attn_metadata_by_gid = new_cad_by_gid
        new_cad = new_cad_by_gid[first_gid]
        self._maybe_dump_first_pass(
            cad=cad,
            new_cad=new_cad,
            target_positions=target_positions,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            num_context=num_context,
            num_query_total=num_query_total,
            effective_seq_lens=effective_seq_lens,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        if envs.VLLM_DFLASH_PROFILE:
            self._profile_rounds += 1
            self._profile_context_tokens += num_context
            self._profile_query_tokens += num_query_total
            self._profile_expand_elapsed_s += expand_elapsed_s

        return num_query_total, token_indices_to_sample, new_cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Key differences to default dummy_run:
        - Only one forward pass due to parallel drafting
        - DFlash uses context states as unpadded metadata, so hidden_states will
        use the unpadded num_tokens instead of num_input_tokens
        - max_query_tokens is quite small, DFlash only sees spec tokens as queries
        - Multimodal inputs are not currently supported
        """
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        (
            cudagraph_runtime_mode,
            num_input_tokens,
            num_tokens_across_dp,
            batch_descriptor,
        ) = self._determine_batch_execution_and_padding(
            num_query_tokens, use_cudagraphs=use_cudagraphs
        )

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}
        self._pad_query_buffers(
            num_query_tokens=num_query_tokens,
            num_input_tokens=num_input_tokens,
        )

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        # Context states will be passed directly to the precomputation without
        # going through the buffer, since no CUDA graph is used for the precomputation.
        # For the dummy run, we use the dummy buffer.
        context_states = self.hidden_states[:num_tokens]

        # Run the KV projection (GEMM + norms + RoPE) for memory profiling,
        self.model.precompute_and_store_context_kv(context_states, context_positions)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        # Context and query positions/slots were written to separate
        # buffers by the kernel — no copy needed.
        num_context = self._dflash_num_context
        num_query_tokens = self._dflash_num_query_tokens
        profile_enabled = _ddtree_worker_profile_enabled()
        profile_t0 = time.perf_counter() if profile_enabled else 0.0
        self._pad_query_buffers(
            num_query_tokens=num_query_tokens,
            num_input_tokens=num_input_tokens,
        )
        pad_ms = (time.perf_counter() - profile_t0) * 1000.0 if profile_enabled else 0.0
        if envs.VLLM_DFLASH_PROFILE:
            self._profile_padded_query_tokens += num_input_tokens
            rounds = self._profile_rounds
            if rounds > 0 and rounds % envs.VLLM_DFLASH_PROFILE_LOG_INTERVAL == 0:
                avg_context_tokens = self._profile_context_tokens / rounds
                avg_query_tokens = self._profile_query_tokens / rounds
                avg_padded_query_tokens = self._profile_padded_query_tokens / rounds
                avg_expand_ms = self._profile_expand_elapsed_s * 1000.0 / rounds
                padding_ratio = (
                    avg_padded_query_tokens / avg_query_tokens
                    if avg_query_tokens > 0
                    else 0.0
                )
                logger.info(
                    "DFlash proposer profile: rounds=%d avg_context_tokens=%.1f "
                    "avg_query_tokens=%.1f avg_padded_query_tokens=%.1f "
                    "avg_padding_ratio=%.2f avg_expand_ms=%.3f",
                    rounds,
                    avg_context_tokens,
                    avg_query_tokens,
                    avg_padded_query_tokens,
                    padding_ratio,
                    avg_expand_ms,
                )

        # Pre-insert context KVs directly into cache
        profile_t0 = time.perf_counter() if profile_enabled else 0.0
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # Shape is already [num_context, hidden_size]
            self._context_positions_buffer[:num_context],
            {
                layer_name: self._context_slot_mapping_buffer_for_gid(gid)[:num_context]
                for layer_name, gid in self.draft_layer_to_kv_cache_gid.items()
            },
        )
        precompute_ms = (
            (time.perf_counter() - profile_t0) * 1000.0 if profile_enabled else 0.0
        )
        if profile_enabled:
            logger.info(
                "DFLASH_DDTREE_WORKER_PROFILE build_inputs_split "
                "pad_ms=%.3f precompute_context_kv_ms=%.3f "
                "context_tokens=%d query_tokens=%d padded_query_tokens=%d",
                pad_ms,
                precompute_ms,
                num_context,
                num_query_tokens,
                num_input_tokens,
            )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group: list[object] = []
        per_layer: dict[str, object] = {}
        metadata_by_gid = self._dflash_new_common_attn_metadata_by_gid
        profile_enabled = _ddtree_worker_profile_enabled()
        for attn_group in self.draft_attn_groups:
            group_cad = metadata_by_gid.get(attn_group.kv_cache_group_id, cad)
            profile_t0 = time.perf_counter() if profile_enabled else 0.0
            builder = attn_group.get_metadata_builder()
            attn_metadata = builder.build_for_drafting(
                common_attn_metadata=group_cad,
                draft_index=draft_index,
            )
            if profile_enabled:
                logger.info(
                    "DFLASH_DDTREE_WORKER_PROFILE attn_metadata_group "
                    "gid=%s builder=%s layers=%d ms=%.3f max_query_len=%s "
                    "num_actual_tokens=%s",
                    attn_group.kv_cache_group_id,
                    type(builder).__name__,
                    len(attn_group.layer_names),
                    (time.perf_counter() - profile_t0) * 1000.0,
                    getattr(group_cad, "max_query_len", None),
                    getattr(group_cad, "num_actual_tokens", None),
                )
            per_group.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = attn_metadata
        if not self.dflash_causal:
            # Require all layers to support non-causal attention when required by DFlash
            for layer_name, attn_metadata in per_layer.items():
                assert getattr(attn_metadata, "causal", None) is False, (
                    f"Attention metadata for layer {layer_name} does not have"
                    " non-causal support, which is required for DFlash."
                    " Consider using a different attention backend, e.g FlashAttention."
                )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        return self.dflash_config.get("use_aux_hidden_state", True)

    @property
    def dflash_config(self):
        return getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}
