# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    copy_and_expand_dflash_inputs_kernel,
)

logger = init_logger(__name__)


def _dflash_profile_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_PROFILE", "0") == "1"


def _dflash_profile_log_interval() -> int:
    return max(1, int(os.getenv("VLLM_DFLASH_PROFILE_LOG_INTERVAL", "32")))


def _dflash_debug_corruption_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DEBUG_CORRUPTION", "0") == "1"


class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs.
        # The query-side buffers must also be able to cover cudagraph padding,
        # which can exceed the actual number of DFlash query tokens.
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
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
        self._debug_last_first_pass: dict[str, Any] | None = None
        self._query_slot_mapping_buffers_by_gid: dict[int, torch.Tensor] = {}
        self._context_slot_mapping_buffers_by_gid: dict[int, torch.Tensor] = {}
        self._dflash_common_attn_metadata_by_gid: (
            dict[int, CommonAttentionMetadata] | None
        ) = None
        self._dflash_new_common_attn_metadata_by_gid: dict[
            int, CommonAttentionMetadata
        ] = {}
        self._dflash_block_size_by_gid: dict[int, int] = {}

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
                self._query_slot_mapping_buffers_by_gid[gid] = (
                    self._slot_mapping_buffer
                )
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
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        self._reset_slot_mapping_buffers_for_groups()
        self._dflash_block_size_by_gid = {
            group.kv_cache_group_id: (
                group.get_metadata_builder().kv_cache_spec.block_size
            )
            for group in self.draft_attn_groups
        }

    def set_common_attn_metadata_by_kv_cache_group(
        self,
        metadata_by_gid: dict[int, CommonAttentionMetadata],
    ) -> None:
        self._dflash_common_attn_metadata_by_gid = metadata_by_gid

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
        buffers = self._query_slot_mapping_buffers_by_gid.values()
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
        if self._dumped_first_pass:
            return
        if os.getenv("VLLM_DFLASH_DUMP_FIRST_PASS", "0") != "1":
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
                "num_rejected_tokens_gpu": (
                    None
                    if num_rejected_tokens_gpu is None
                    else num_rejected_tokens_gpu.detach().cpu()
                ),
            },
            dump_path,
        )
        self._dumped_first_pass = True
        logger.warning("Saved DFlash first-pass metadata to %s", dump_path)

    @override
    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # DFlash writes query slot mappings directly in set_inputs_first_pass.
        # Re-copying common_attn_metadata.slot_mapping here would overwrite the
        # query-only layout with target-model metadata.
        return {
            name: self._query_slot_mapping_buffer_for_gid(
                self.draft_layer_to_kv_cache_gid.get(name, self.kv_cache_gid)
            )[:num_tokens]
            for name in self._draft_attn_layer_names
        }

    @override
    def _raise_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        # Support for multimodal inputs has not been tested.
        pass

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

        incoming_token_indices_to_sample = (
            None
            if token_indices_to_sample is None
            else token_indices_to_sample.detach().cpu()
        )
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
        if _dflash_profile_enabled() and self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        first_gid = self._first_draft_kv_cache_group_id()
        common_metadata_by_gid = self._dflash_common_attn_metadata_by_gid
        if common_metadata_by_gid is None:
            common_metadata_by_gid = {first_gid: cad}
        elif first_gid not in common_metadata_by_gid:
            common_metadata_by_gid = {**common_metadata_by_gid, first_gid: cad}

        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        new_cad_by_gid: dict[int, CommonAttentionMetadata] = {}
        for gid in self._draft_kv_cache_group_ids():
            group_cad = common_metadata_by_gid.get(gid, cad)
            context_slot_mapping_buffer = self._context_slot_mapping_buffer_for_gid(
                gid
            )
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
            new_cad_by_gid[gid] = CommonAttentionMetadata(
                query_start_loc=new_query_start_loc,
                seq_lens=group_effective_seq_lens + num_query_per_req,
                query_start_loc_cpu=(
                    torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                    * num_query_per_req
                ),
                _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
                num_reqs=group_cad.num_reqs,
                num_actual_tokens=num_query_total,
                max_query_len=num_query_per_req,
                max_seq_len=group_cad.max_seq_len + num_query_per_req,
                block_table_tensor=group_cad.block_table_tensor,
                slot_mapping=query_slot_mapping_buffer[:num_query_total],
                causal=False,  # Non-causal attention is required for DFlash
            )
        expand_elapsed_s = 0.0
        if start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            expand_elapsed_s = start_event.elapsed_time(end_event) / 1000.0

        self._dflash_new_common_attn_metadata_by_gid = new_cad_by_gid
        new_cad = new_cad_by_gid[first_gid]
        query_slot_mapping = new_cad.slot_mapping

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
        if _dflash_profile_enabled():
            self._profile_rounds += 1
            self._profile_context_tokens += num_context
            self._profile_query_tokens += num_query_total
            self._profile_expand_elapsed_s += expand_elapsed_s
        if _dflash_debug_corruption_enabled():
            self._debug_last_first_pass = {
                "incoming_token_indices_to_sample": incoming_token_indices_to_sample,
                "outgoing_token_indices_to_sample": (
                    token_indices_to_sample.detach().cpu()
                ),
                "next_token_ids": next_token_ids.detach().cpu(),
                "query_positions": self.positions[:num_query_total].detach().cpu(),
                "query_slot_mapping": query_slot_mapping.detach().cpu(),
                "num_context": num_context,
                "num_query_total": num_query_total,
                "effective_seq_lens": effective_seq_lens.detach().cpu(),
                "num_rejected_tokens_gpu": (
                    None
                    if num_rejected_tokens_gpu is None
                    else num_rejected_tokens_gpu.detach().cpu()
                ),
            }

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
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
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
        self._pad_query_buffers(
            num_query_tokens=num_query_tokens,
            num_input_tokens=num_input_tokens,
        )
        if _dflash_profile_enabled():
            self._profile_padded_query_tokens += num_input_tokens
            rounds = self._profile_rounds
            if rounds > 0 and rounds % _dflash_profile_log_interval() == 0:
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
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # Shape is already [num_context, hidden_size]
            self._context_positions_buffer[:num_context],
            {
                layer_name: self._context_slot_mapping_buffer_for_gid(gid)[
                    :num_context
                ]
                for layer_name, gid in self.draft_layer_to_kv_cache_gid.items()
            },
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
        for attn_group in self.draft_attn_groups:
            group_cad = metadata_by_gid.get(attn_group.kv_cache_group_id, cad)
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=group_cad, draft_index=draft_index
            )
            per_group.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = attn_metadata
        for layer_name, attn_metadata in per_layer.items():
            assert getattr(attn_metadata, "causal", None) is False, (
                f"Attention metadata for layer {layer_name} does not have"
                " non-causal support, which is required for DFlash."
                " Consider using a different attention backend, such as FlashAttention."
            )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state
