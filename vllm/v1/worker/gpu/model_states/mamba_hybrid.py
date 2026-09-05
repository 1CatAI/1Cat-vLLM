# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    is_conv_state_dim_first,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.flash_attn_v100 import (
    DFlash2SmallQGroupDescriptor,
    DFlash2SmallQPreparedMetadata,
    FlashAttnV100MetadataBuilder,
    prepare_dflash2_smallq_group_metadata,
)
from vllm.v1.attention.backends.gdn_attn import (
    DFlash2GDNGroupDescriptor,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
    prepare_dflash2_gdn_group_metadata,
)
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.attn_utils import (
    CommonGDNSpecMetadata,
    build_attn_metadata,
    compute_common_gdn_attn_metadata,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mamba_align import (
    preprocess_mamba_align_fused_kernel,
    run_mamba_align_postprocess,
    run_mamba_align_precopy,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.spec_decode import uses_dflash_selector_engine
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    get_mamba_groups,
    get_mamba_types,
)
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None
    common_gdn_metadata: CommonGDNSpecMetadata | None = None
    prepared_dflash2_gdn_metadata: dict[int, GDNAttentionMetadata] | None = None
    prepared_dflash2_smallq_metadata: (
        dict[int, DFlash2SmallQPreparedMetadata] | None
    ) = None

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        if isinstance(attn_metadata_builder, FlashAttnV100MetadataBuilder):
            return {
                "prepared_dflash2_smallq_metadata": (
                    None
                    if self.prepared_dflash2_smallq_metadata is None
                    else self.prepared_dflash2_smallq_metadata.get(
                        id(attn_metadata_builder)
                    )
                )
            }
        if not isinstance(
            attn_metadata_builder,
            (
                Mamba2AttentionMetadataBuilder,
                GDNAttentionMetadataBuilder,
                PleShortConvAttentionMetadataBuilder,
            ),
        ):
            return {}
        kwargs = {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
        }
        if isinstance(attn_metadata_builder, GDNAttentionMetadataBuilder):
            kwargs["common_gdn_metadata"] = self.common_gdn_metadata
            if self.prepared_dflash2_gdn_metadata is not None:
                kwargs["prepared_dflash2_metadata"] = (
                    self.prepared_dflash2_gdn_metadata.get(id(attn_metadata_builder))
                )
        return kwargs


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention + Mamba / linear-attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self._use_dflash2_common_gdn_metadata = bool(
            envs.VLLM_SM70_DFLASH2_VERIFY_FASTPATH
            and uses_dflash_selector_engine(vllm_config)
        )
        if self._use_dflash2_common_gdn_metadata:
            logger.info_once("DFlash2 shared GDN batch metadata fast path enabled.")
        self._use_dflash2_fused_gdn_metadata = bool(
            self._use_dflash2_common_gdn_metadata
            and envs.VLLM_SM70_DFLASH2_FUSED_GDN_METADATA
            and self.cache_config.mamba_cache_mode in ("none", "align")
            and device.type == "cuda"
            and current_platform.is_device_capability(70)
        )
        self._dflash2_gdn_builders: (
            list[tuple[int, GDNAttentionMetadataBuilder]] | None
        ) = None
        self._dflash2_gdn_group_descriptor: DFlash2GDNGroupDescriptor | None = None
        self._dflash2_fused_gdn_metadata_logged = False
        self._use_dflash2_grouped_smallq_metadata = bool(
            self._use_dflash2_common_gdn_metadata
            and envs.VLLM_SM70_DFLASH2_FUSED_SMALLQ_METADATA
            and envs.VLLM_SM70_DFLASH2_GROUPED_SMALLQ_METADATA
            and device.type == "cuda"
            and current_platform.is_device_capability(70)
        )
        self._dflash2_smallq_builders: (
            list[tuple[int, FlashAttnV100MetadataBuilder]] | None
        ) = None
        self._dflash2_smallq_group_descriptor: DFlash2SmallQGroupDescriptor | None = (
            None
        )
        self._dflash2_grouped_smallq_metadata_logged = False
        self._align_mode = self.cache_config.mamba_cache_mode == "align"
        if self._align_mode:
            self._mamba_state_idx_gpu = torch.full(
                (self.max_num_reqs,), -2, dtype=torch.int32, device=self.device
            )
            self._mamba_src_col_gpu = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
            )
            self._mamba_token_bias_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # A recycled request slot may contain the previous request's accepted
        # count. The neutral value is required in both align and non-align mode.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Delay state-column seeding until preprocess_state has the
            # resolved MambaSpec block size. -2 is distinct from -1 (fresh
            # request with no computed state).
            self._mamba_state_idx_gpu[req_index].fill_(-2)

    def _get_mamba_group_info(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            self._mamba_group_ids, self._mamba_spec = get_mamba_groups(kv_cache_config)
        return self._mamba_group_ids, self._mamba_spec

    def _ensure_align_ctx(
        self,
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: list[int],
        block_tables: tuple[torch.Tensor, ...],
    ) -> MambaSpecDecodeGPUContext:
        copy_funcs = None
        if self._mamba_ctx is None:
            copy_funcs = self.model.get_mamba_state_copy_funcs(
                get_mamba_types(kv_cache_config)
            )
            # This V100 closure intentionally retains the tree's default SD
            # layout. The official DS-row extension is unrelated to Qwen3.8's
            # configured path and would expand the DDTree-sensitive closure.
            if (
                any(get_conv_copy_spec in funcs for funcs in copy_funcs.values())
                and is_conv_state_dim_first()
            ):
                raise ValueError(
                    "MRV2 align prefix caching with speculative decoding requires "
                    "the default SD Mamba conv-state layout on this V100 path."
                )
            self._mamba_ctx = MambaSpecDecodeGPUContext.create(
                max_num_reqs=self.max_num_reqs,
                kv_cache_config=kv_cache_config,
                mamba_state_copy_funcs=copy_funcs,
                device=self.device,
                make_buffer=lambda n, dtype: CpuGpuBuffer(
                    n,
                    dtype=dtype,
                    device=self.device,
                    pin_memory=is_pin_memory_available(),
                ),
            )
        ctx = self._mamba_ctx
        if not ctx.is_initialized:
            if copy_funcs is None:
                copy_funcs = self.model.get_mamba_state_copy_funcs(
                    get_mamba_types(kv_cache_config)
                )
            forward_context = self.vllm_config.compilation_config.static_forward_context
            ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                copy_funcs,
                [block_tables[group_id] for group_id in mamba_group_ids],
            )
        return ctx

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Move align-mode recurrent state before a real MRV2 forward."""
        if not self._align_mode or input_batch.num_reqs == 0:
            return
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._ensure_align_ctx(kv_cache_config, mamba_group_ids, block_tables)

        block = 256
        preprocess_mamba_align_fused_kernel[
            (triton.cdiv(input_batch.num_reqs, block),)
        ](
            input_batch.idx_mapping,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_token_bias_gpu,
            input_batch.num_reqs,
            BLOCK_SIZE=block,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )
        run_mamba_align_precopy(
            ctx,
            input_batch.num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_token_bias_gpu,
            input_batch.idx_mapping,
        )

    def _get_dflash2_gdn_builders(
        self,
        attn_groups: list[list[AttentionGroup]],
    ) -> list[tuple[int, GDNAttentionMetadataBuilder]]:
        if self._dflash2_gdn_builders is None:
            builders: list[tuple[int, GDNAttentionMetadataBuilder]] = []
            for kv_cache_group_id, groups in enumerate(attn_groups):
                for group in groups:
                    builder = group.get_metadata_builder(0)
                    if isinstance(builder, GDNAttentionMetadataBuilder):
                        builders.append((kv_cache_group_id, builder))
            self._dflash2_gdn_builders = builders
        return self._dflash2_gdn_builders

    def _get_dflash2_smallq_builders(
        self,
        attn_groups: list[list[AttentionGroup]],
    ) -> list[tuple[int, FlashAttnV100MetadataBuilder]]:
        if self._dflash2_smallq_builders is None:
            builders: list[tuple[int, FlashAttnV100MetadataBuilder]] = []
            for kv_cache_group_id, groups in enumerate(attn_groups):
                for group in groups:
                    builder = group.get_metadata_builder(0)
                    if isinstance(builder, FlashAttnV100MetadataBuilder):
                        builders.append((kv_cache_group_id, builder))
            self._dflash2_smallq_builders = builders
        return self._dflash2_smallq_builders

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        common_gdn_metadata = None
        prepared_dflash2_gdn_metadata = None
        prepared_dflash2_smallq_metadata = None
        if not for_capture:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel rather than a raw zero draft count.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            if input_batch.num_draft_tokens_per_req is not None:
                spec_decode_mask = (
                    input_batch.num_draft_tokens_per_req > 0
                ) & ~input_batch.is_prefilling_np
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    input_batch.num_draft_tokens_per_req,
                    -1,
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)
            if self._use_dflash2_common_gdn_metadata:
                speculative_config = self.vllm_config.speculative_config
                assert speculative_config is not None
                common_gdn_metadata = compute_common_gdn_attn_metadata(
                    num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                    query_start_loc=input_batch.query_start_loc,
                    query_start_loc_cpu=query_start_loc_cpu,
                    num_spec_state_tokens=(
                        speculative_config.num_speculative_state_tokens()
                    ),
                    legacy_mixed_decode_routing=(
                        envs.VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING
                    ),
                )

            if (
                self._use_dflash2_fused_gdn_metadata
                and cudagraph_mode == CUDAGraphMode.FULL
                and common_gdn_metadata is not None
            ):
                prepared_result = prepare_dflash2_gdn_group_metadata(
                    builders_by_group=self._get_dflash2_gdn_builders(attn_groups),
                    block_tables=block_tables,
                    common_gdn_metadata=common_gdn_metadata,
                    num_accepted_tokens=num_accepted_tokens,
                    num_actual_tokens=num_tokens,
                    descriptor=self._dflash2_gdn_group_descriptor,
                    state_start_indices=(
                        self._mamba_state_idx_gpu if self._align_mode else None
                    ),
                    req_index_mapping=(
                        input_batch.idx_mapping if self._align_mode else None
                    ),
                )
                if prepared_result is not None:
                    (
                        prepared_dflash2_gdn_metadata,
                        self._dflash2_gdn_group_descriptor,
                    ) = prepared_result
                    if not self._dflash2_fused_gdn_metadata_logged:
                        logger.info(
                            "DFlash2 fused GDN metadata active for %d cache groups.",
                            len(prepared_dflash2_gdn_metadata),
                        )
                        self._dflash2_fused_gdn_metadata_logged = True

            if (
                self._use_dflash2_grouped_smallq_metadata
                and cudagraph_mode == CUDAGraphMode.FULL
                and common_gdn_metadata is not None
            ):
                seq_lens_cpu = input_batch.seq_lens_cpu_upper_bound[:num_reqs]
                max_seq_len_hint = int(seq_lens_cpu.max().item())
                prepared_smallq_result = prepare_dflash2_smallq_group_metadata(
                    builders_by_group=self._get_dflash2_smallq_builders(attn_groups),
                    block_tables=block_tables,
                    seq_lens=input_batch.seq_lens,
                    query_start_loc=input_batch.query_start_loc,
                    query_start_loc_cpu=query_start_loc_cpu,
                    num_reqs=num_reqs,
                    num_query_tokens=num_tokens,
                    max_seq_len_hint=max_seq_len_hint,
                    workspace_seq_capacity_cap=self.max_model_len,
                    descriptor=self._dflash2_smallq_group_descriptor,
                )
                if prepared_smallq_result is not None:
                    (
                        prepared_dflash2_smallq_metadata,
                        self._dflash2_smallq_group_descriptor,
                    ) = prepared_smallq_result
                    if not self._dflash2_grouped_smallq_metadata_logged:
                        logger.info(
                            "DFlash2 grouped small-query metadata active for %d "
                            "full-attention cache groups.",
                            len(prepared_dflash2_smallq_metadata),
                        )
                        self._dflash2_grouped_smallq_metadata_logged = True

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            common_gdn_metadata=common_gdn_metadata,
            prepared_dflash2_gdn_metadata=prepared_dflash2_gdn_metadata,
            prepared_dflash2_smallq_metadata=prepared_dflash2_smallq_metadata,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            prefix_anchor_lens=input_batch.prefix_anchor_lens,
        )
        if common_gdn_metadata is not None:
            assert common_gdn_metadata.spec_query_start_loc.numel() == (
                common_gdn_metadata.num_spec_decodes + 1
            )
            # Metadata kernels/copies and FULL CUDA Graph replay are enqueued on
            # the same current stream, which already supplies the required
            # device ordering. Retain the old value check as an opt-in debug
            # fence while the no-sync route is quality- and trace-gated.
            if envs.VLLM_SM70_DFLASH2_GDN_SYNC_ASSERT:
                assert (
                    common_gdn_metadata.spec_query_start_loc[-1].item()
                    == common_gdn_metadata.num_spec_decode_tokens
                )
        return attn_metadata

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        if isinstance(num_sampled, int):
            num_reqs = idx_mapping.shape[0]
            if num_reqs:
                _fill_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    self.num_accepted_tokens_gpu,
                    VALUE=max(num_sampled, 1),
                )
        else:
            num_reqs = idx_mapping.shape[0]
            if num_reqs:
                _scatter_num_accepted_kernel[(num_reqs,)](
                    idx_mapping, num_sampled, self.num_accepted_tokens_gpu
                )
        if (
            self._align_mode
            and num_computed_tokens is not None
            and self._mamba_ctx is not None
            and idx_mapping.shape[0] > 0
        ):
            run_mamba_align_postprocess(
                self._mamba_ctx,
                idx_mapping.shape[0],
                self.num_accepted_tokens_gpu,
                self._mamba_state_idx_gpu,
                num_computed_tokens,
                idx_mapping,
            )


@triton.jit
def _fill_num_accepted_kernel(
    idx_mapping_ptr,
    num_accepted_ptr,
    VALUE: tl.constexpr,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx >= 0:
        tl.store(num_accepted_ptr + req_state_idx, VALUE)


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    num_accepted_ptr,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))
