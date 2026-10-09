# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import math
import os
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    _sm70_dump_qwen_layer_tensor,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import (
    QSAForwardMetadata,
    build_qsa_metadata,
    qsa_dcp_block_geometry,
)
from .indexer_qsa import QSAIndexer

logger = init_logger(__name__)


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # The replicated draft's slot map depends on logical positions as well
    # as the block table; FlashAttention's update hook only receives a table.
    supports_update_block_table: bool = False
    # QSA's DCP attention localizes its own selections and never reads the
    # per-rank context lengths, which cost a dozen small kernels per build.
    builds_dcp_context_lens: ClassVar[bool] = False

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.replicated_draft = (
            vllm_config.parallel_config.decode_context_parallel_size == 2
            and all("mtp" in name.split(".") for name in layer_names)
        )
        if self.replicated_draft:
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self.draft_block_table_buffer: torch.Tensor | None = None
            self.draft_token_to_req = torch.empty(
                max_tokens, dtype=torch.int32, device=device
            )
            self.draft_logical_positions = torch.empty(
                max_tokens, dtype=torch.int64, device=device
            )
            self.draft_slot_mapping = torch.empty(
                max_tokens, dtype=torch.int64, device=device
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if not self.replicated_draft:
            return metadata
        # Draft K/V owns the full global span on every DCP rank. The mixed
        # target/draft group has already expanded each scheduler page into
        # kernel blocks using the target's rank-local page size, so only
        # expand the remaining draft-to-target ratio.
        draft_page_size, _, _ = qsa_dcp_block_geometry(
            self.vllm_config, self.layer_names[0]
        )
        if draft_page_size % self.block_size:
            raise RuntimeError("QSA draft kernel block must divide its physical page")
        group_page_size = (
            self.vllm_config.cache_config.block_size
            // self.vllm_config.parallel_config.decode_context_parallel_size
        )
        if group_page_size >= self.block_size and group_page_size % self.block_size:
            raise RuntimeError("QSA group page must divide into kernel blocks")
        group_blocks_per_page = max(1, group_page_size // self.block_size)
        draft_blocks_per_page = draft_page_size // self.block_size
        if draft_blocks_per_page % group_blocks_per_page:
            raise RuntimeError("QSA draft blocks must divide the group page")
        expansion = draft_blocks_per_page // group_blocks_per_page
        if expansion > 1:
            table = common_attn_metadata.block_table_tensor
            rows, columns = table.shape
            if self.draft_block_table_buffer is None:
                self.draft_block_table_buffer = torch.empty(
                    (
                        self.vllm_config.scheduler_config.max_num_seqs,
                        columns * expansion,
                    ),
                    dtype=table.dtype,
                    device=table.device,
                )
            if (
                rows > self.draft_block_table_buffer.shape[0]
                or columns * expansion > self.draft_block_table_buffer.shape[1]
            ):
                raise RuntimeError("QSA draft block-table buffer is too small")
            expanded = self.draft_block_table_buffer[:rows, : columns * expansion]
            expanded_view = expanded.view(rows, columns, expansion)
            for sub_block in range(expansion):
                torch.mul(table, expansion, out=expanded_view[:, :, sub_block])
                expanded_view[:, :, sub_block].add_(sub_block)
            common_attn_metadata = common_attn_metadata.replace(
                block_table_tensor=expanded
            )
            metadata.block_table = expanded
        _, _, slot_mapping = build_qsa_metadata(
            common_attn_metadata,
            self.draft_token_to_req,
            self.draft_logical_positions,
            self.draft_slot_mapping,
            storage_block_size=self.block_size,
            compress_ratio=1,
            map_plain_slot=True,
        )
        if common_attn_metadata.is_dummy_batch:
            slot_mapping.fill_(PAD_SLOT_ID)
        metadata.slot_mapping = slot_mapping
        return metadata


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        # QSA chooses its split-K reduction depth from the runtime batch
        # shape, so it cannot inherit FlashAttention's stronger guarantee.
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = True
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size not in (1, 2):
            raise NotImplementedError("Qwen4Exp QSA supports DCP1 or DCP2")
        if self.kv_cache_dtype not in (
            "auto",
            "float16",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires FP16/BF16 or E4M3 main KV cache"
            )
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_gate: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        dense_short_context: bool = False,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        if num_tokens and getattr(layer, "host_kv_enabled", False):
            # Direct host QSA writes every active row, including empty
            # selections. Only graph padding requires explicit initialization.
            output[num_tokens:].zero_()
        else:
            output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        if getattr(layer, "host_kv_enabled", False):
            if query_positions is None or sequence_lengths is None:
                raise RuntimeError(
                    "Host QSA requires exact positions and sequence lengths"
                )
            layer.host_kv_forward(
                query[:num_tokens],
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                query_positions[:num_tokens],
                sequence_lengths,
                output[:num_tokens],
                output_gate,
            )
            return output
        # This tree's FlashAttention cache ABI keeps K/V on dimension 1:
        # [num_blocks, 2, block_size, num_kv_heads, head_size].
        key_cache, value_cache = kv_cache.unbind(1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError("Qwen4Exp QSA requires FP16/BF16 queries")
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            if key_cache.dtype != torch.uint8 or value_cache.dtype != torch.uint8:
                raise RuntimeError("Qwen4Exp QSA E4M3 cache must use uint8 storage")
        elif key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
            raise RuntimeError("Qwen4Exp QSA FP16/BF16 cache must match query dtype")

        from .ops.qsa import qsa_sparse_paged_attention

        if dense_short_context:
            from .ops.qsa_dense import qsa_dense_decode

            gate = None
            if output_gate is not None:
                gate = output_gate[:num_tokens].reshape(
                    num_tokens, query.shape[1], query.shape[2]
                )
            assert sequence_lengths is not None and query_positions is not None
            qsa_dense_decode(
                query[:num_tokens],
                key_cache,
                value_cache,
                attn_metadata.block_table,
                token_to_req,
                query_positions[:num_tokens],
                output[:num_tokens],
                gate,
                sequence_lengths.shape[0],
            )
            return output

        if getattr(layer, "qsa_dcp_sharded", False):
            if output_gate is None:
                raise RuntimeError("QSA DCP requires its output gate")
            self._forward_qsa_dcp(
                layer,
                query[:num_tokens],
                key_cache,
                value_cache,
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                output[:num_tokens],
                output_gate[:num_tokens],
            )
            return output

        qsa_metadata: dict[str, torch.Tensor] = {}
        if query_positions is not None:
            qsa_metadata["query_positions"] = query_positions[:num_tokens]
        if sequence_lengths is not None:
            qsa_metadata["sequence_lengths"] = sequence_lengths
        if output_gate is not None:
            qsa_metadata["output_gate"] = output_gate[:num_tokens]
        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            **qsa_metadata,
        )
        return output

    def _forward_qsa_dcp(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        logical_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> None:
        from vllm.distributed import get_dcp_group

        from .ops.qsa import _qsa_output_gate, qsa_sparse_paged_attention
        from .ops.qsa_dcp import (
            qsa_dcp_local_selection_width,
            qsa_localize_dcp_indices,
        )

        local_indices_buffer = layer.dcp_local_indices_buffer
        partial_output_buffer = layer.dcp_partial_output_buffer
        partial_lse_buffer = layer.dcp_partial_lse_buffer
        if any(
            buffer is None
            for buffer in (
                local_indices_buffer,
                partial_output_buffer,
                partial_lse_buffer,
            )
        ):
            raise RuntimeError("QSA DCP target workspaces are not initialized")
        local_indices = qsa_localize_dcp_indices(
            logical_indices,
            local_indices_buffer[: query.shape[0]],
            dcp_world_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            interleave_size=layer.cp_kv_cache_interleave_size,
            local_block_size=key_cache.shape[1],
        )
        group = get_dcp_group()
        gathered_query = group.all_gather(query.contiguous(), dim=1)
        partial_output = partial_output_buffer[: query.shape[0]]
        partial_lse = partial_lse_buffer[: query.shape[0]]
        if partial_output.shape != gathered_query.shape:
            raise RuntimeError("QSA DCP partial output has wrong head geometry")
        # Only this prefix of the compacted columns can hold an owned token.
        local_width = qsa_dcp_local_selection_width(
            layer.indexer.token_topk,
            layer.indexer.compress_ratio,
            self.dcp_world_size,
            layer.cp_kv_cache_interleave_size,
            local_indices.shape[1],
        )
        qsa_sparse_paged_attention(
            gathered_query,
            key_cache,
            value_cache,
            local_indices[:, :local_width],
            block_table,
            token_to_req,
            partial_output,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            lse=partial_lse,
        )
        merged = cast(
            torch.Tensor,
            self.dcp_combine(
                partial_output,
                partial_lse,
                group,
                is_lse_base_on_e=False,
            ),
        )
        # Round to the output dtype first, as DCP1's attention output is, then
        # apply the Triton gate DCP1's page4 route uses. This replaces a chain
        # of five elementwise kernels (12.2 us per decode layer on V100).
        output.copy_(merged)
        _qsa_output_gate(output, output_gate.view_as(output))


def _verify_e4m3_kv_requirements(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> None:
    """Admit calibrated E4M3 from kernel capabilities, independent of TP/MTP."""
    if cache_config.cache_dtype not in ("fp8", "fp8_e4m3"):
        return
    from .ops.qsa import qsa_e4m3_capability_reason

    if reason := qsa_e4m3_capability_reason(model_config.dtype):
        raise NotImplementedError(f"QSA E4M3 cache unavailable: {reason}")


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = True
    # The paged indexer and sparse attention switch launch profiles after 32
    # query rows. Advertise the first row count in the wider profile so the
    # generic MRV2 warmup can compile it before serving traffic.
    kernel_warmup_prefill_token_counts = (33,)

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        dcp_local_indices_buffer: torch.Tensor | None = None,
        dcp_partial_output_buffer: torch.Tensor | None = None,
        dcp_partial_lse_buffer: torch.Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError("Qwen4Exp QSA requires FP16 or BF16")
        if cache_config.cache_dtype not in (
            "auto",
            "float16",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires FP16/BF16 or E4M3 main KV cache"
            )
        _verify_e4m3_kv_requirements(vllm_config, model_config, cache_config)
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support prefill context parallelism"
            )
        if parallel_config.decode_context_parallel_size not in (1, 2):
            raise NotImplementedError("Qwen4Exp QSA supports DCP1 or DCP2")
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True
        qkv_quant_config = quant_config
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            qkv_quant_config = None

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=qkv_quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        _, _, self.qsa_dcp_sharded = qsa_dcp_block_geometry(
            vllm_config, self.layer_name
        )
        self.qsa_dcp_sharded &= parallel_config.decode_context_parallel_size > 1
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.dcp_local_indices_buffer = dcp_local_indices_buffer
        self.dcp_partial_output_buffer = dcp_partial_output_buffer
        self.dcp_partial_lse_buffer = dcp_partial_lse_buffer
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3") and (
            cache_config.calculate_kv_scales
        ):
            raise ValueError(
                "QSA calibrated E4M3 forbids calculate_kv_scales; "
                "load an offline scale overlay instead"
            )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        self.host_kv_enabled = vllm_config.kernel_config.qsa_host_kv_active
        self.host_kv_hot_tokens = vllm_config.kernel_config.qsa_host_kv_hot_tokens
        self.host_kv_device_reference = (
            vllm_config.kernel_config.qsa_host_kv_device_reference
        )
        self.host_kv_dtype = (
            vllm_config.kernel_config.qsa_host_kv_draft_dtype
            if getattr(vllm_config, "is_speculative_draft", False)
            else vllm_config.kernel_config.qsa_host_kv_dtype
        )
        if self.kv_cache_dtype not in ("fp8", "fp8_e4m3") and (
            self.kv_cache_torch_dtype != model_config.dtype
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA main cache dtype must match the model dtype"
            )
        if self.host_kv_enabled:
            self.kv_cache_torch_dtype = (
                torch.uint8 if self.host_kv_dtype == "fp8_e4m3" else torch.float16
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)
        self._qsa_kv_scales_finalized = self.kv_cache_dtype not in (
            "fp8",
            "fp8_e4m3",
        )
        if not self._qsa_kv_scales_finalized:
            # Keep checkpoint loading state separate from runtime scales.
            # Negative is deliberately invalid and the slots are deleted once
            # validation copies them into the runtime buffers.
            self.k_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
            self.v_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        self._dense_short_context = bool(
            vllm_config.kernel_config.qsa_dense_short_context
        )
        self._sm70_qsa_prep = bool(
            vllm_config.kernel_config.sm70_qsa_prep
            and not self.host_kv_enabled
            and getattr(self.rotary_emb, "is_neox_style", False)
            and self.head_dim == 256
            and self.num_kv_heads == 1
            and cache_config.cache_dtype in ("auto", "float16")
            and model_config.dtype == torch.float16
            and self.attn_output_gate
        )
        speculative = vllm_config.speculative_config
        # Bucketed graphs are only replayed for uniform decode batches, whose
        # requests each hold exactly this many query tokens.
        self._decode_query_len = 1 + (
            speculative.num_speculative_tokens if speculative is not None else 0
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._set_topk_indices_buffer(
            max_tokens=max_tokens,
            topk_indices_buffer=topk_indices_buffer,
        )

        self._round_cost_selection = None
        if (
            vllm_config.kernel_config.sm70_round_cost_diagnostics
            and self.topk_indices_buffer.is_cuda
        ):
            from vllm.sm70_round_cost import attach_selection_recorder

            self._round_cost_selection = attach_selection_recorder(
                self.layer_name,
                self.indexer.output_width,
                self.topk_indices_buffer.device,
                self,
            )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def _set_topk_indices_buffer(
        self,
        *,
        max_tokens: int,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        if topk_indices_buffer is None:
            self.register_buffer(
                "topk_indices_buffer",
                torch.empty(
                    max_tokens,
                    self.indexer.output_width,
                    dtype=torch.int32,
                ),
                persistent=False,
            )
            return

        expected_width = self.indexer.output_width
        if (
            topk_indices_buffer.dtype != torch.int32
            or topk_indices_buffer.ndim != 2
            or topk_indices_buffer.shape[0] < max_tokens
            or topk_indices_buffer.shape[1] != expected_width
        ):
            raise ValueError(
                "QSA shared top-k buffer must have dtype int32 and shape "
                f"[{max_tokens} or more, {expected_width}], got "
                f"dtype={topk_indices_buffer.dtype}, "
                f"shape={tuple(topk_indices_buffer.shape)}"
            )
        self.topk_indices_buffer = topk_indices_buffer

    def adopt_default_kv_scales(self) -> None:
        """Use the module's own unit scales when the checkpoint has none.

        The calibrated overlay exists to keep FP8 E4M3 K/V inside range; a
        checkpoint that was never calibrated has no such overlay, so the layer
        keeps the 1.0 defaults set at construction instead of the -1.0 loading
        sentinel and is marked finalized so nothing re-validates it.
        """
        set_default_quant_scales(self, register_buffer=False)
        if hasattr(self, "k_scale"):
            del self.k_scale
        if hasattr(self, "v_scale"):
            del self.v_scale
        self._qsa_kv_scales_finalized = True

    def validate_loaded_kv_scales(self) -> None:
        if self.kv_cache_dtype not in ("fp8", "fp8_e4m3"):
            return
        if self._qsa_kv_scales_finalized:
            raise RuntimeError(
                f"QSA E4M3 scales already finalized for {self.layer_name}"
            )
        if not hasattr(self, "k_scale") or not hasattr(self, "v_scale"):
            raise RuntimeError(
                f"QSA E4M3 loading slots are unavailable for {self.layer_name}"
            )

        scales = {
            "K": float(self.k_scale.item()),
            "V": float(self.v_scale.item()),
        }
        invalid = [
            name
            for name, value in scales.items()
            if not math.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(
                f"QSA E4M3 calibrated scales are required for {self.layer_name} "
                f"(invalid: {', '.join(invalid)}). Refusing to start."
            )
        self._k_scale.copy_(scales["K"])
        self._v_scale.copy_(scales["V"])
        self._k_scale_float = scales["K"]
        self._v_scale_float = scales["V"]
        del self.k_scale
        del self.v_scale
        self._qsa_kv_scales_finalized = True

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        if self.host_kv_enabled:
            from .ops.host_kv import HostQSAKV

            self.host_kv = HostQSAKV(
                kv_cache.shape[0],
                kv_cache.shape[2],
                self.head_dim,
                kv_cache.device,
                hot_tokens=self.host_kv_hot_tokens,
                history=kv_cache,
                width=self.indexer.output_width,
                device_reference=self.host_kv_device_reference,
            )
            logger.info_once(
                "QSA encoded history initialized: storage=%s, dtype=%s, hot_tokens=%d; "
                "attention reads FP16 hot pages; only unresolved misses are staged.",
                "device_reference" if self.host_kv_device_reference else "host",
                self.host_kv_dtype,
                self.host_kv_hot_tokens,
            )

    def host_kv_forward(
        self, query, indices, table, requests, positions, lengths, output, gate
    ) -> None:
        from .ops.host_kv_attention import host_qsa_attention

        state = self.host_kv
        for start in range(0, query.shape[0], state.rows):
            stop = min(start + state.rows, query.shape[0])
            host_qsa_attention(
                query[start:stop],
                state,
                indices[start:stop],
                table,
                requests[start:stop],
                positions[start:stop],
                lengths,
                output[start:stop],
                gate[start:stop] if gate is not None else None,
            )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size, _, dcp_sharded = qsa_dcp_block_geometry(
            vllm_config, self.layer_name
        )
        return FullAttentionSpec(
            block_size=block_size,
            host_backed=self.host_kv_enabled,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            dcp_sharded=dcp_sharded,
        )

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split, normalize, and rotate Q/K using this tree's Qwen3 API."""
        q_gate, key, value = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
        )
        token_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*token_shape, self.num_heads, 2 * self.head_dim)
        query, gate = torch.chunk(q_gate, 2, dim=-1)
        query = self.q_norm(query).reshape(*token_shape, self.q_size)
        key = self.k_norm(
            key.view(*token_shape, self.num_kv_heads, self.head_dim)
        ).reshape(*token_shape, self.kv_size)
        query, key = self.rotary_emb(positions, query, key)
        return query, key, value, gate.reshape(*token_shape, self.q_size)

    def _use_dense_short_context(
        self, num_tokens: int, side_metadata: QSAForwardMetadata
    ) -> bool:
        """True in context-bucketed graphs where QSA would select every token."""
        if not self._dense_short_context or self.indexer.skip_topk:
            return False
        if getattr(self, "qsa_dcp_sharded", False):
            return False
        descriptor = get_forward_context().batch_descriptor
        bucket = getattr(descriptor, "attention_context_bucket", None)
        if bucket is None or bucket > self.indexer.token_topk:
            logger.info_once(
                "QSA dense short-context skipped: tokens=%d bucket=%s budget=%d",
                num_tokens,
                bucket,
                self.indexer.token_topk,
            )
            return False
        from .ops.qsa_dense import qsa_dense_supported

        supported = qsa_dense_supported(
            self._decode_query_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.kv_cache.dtype,
            side_metadata.seq_lens.shape[0],
        )
        logger.info_once(
            "QSA dense short-context %s: tokens=%d requests=%d bucket=%d",
            "enabled" if supported else "unsupported",
            num_tokens,
            side_metadata.seq_lens.shape[0],
            bucket,
        )
        return supported

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor | None = None,
    ) -> None:
        if not self._qsa_kv_scales_finalized:
            raise RuntimeError(
                f"QSA E4M3 scales were not finalized for {self.layer_name}"
            )
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        if os.getenv("VLLM_QSA_KV_CALIBRATION_DIR"):
            from .ops.qsa_kv_calibration import observe_qsa_kv

            observe_qsa_kv(
                self.indexer.layer_id,
                key[:num_tokens],
                value[:num_tokens],
            )
        dense = self._use_dense_short_context(num_tokens, side_metadata)
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
            select=not dense,
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        selected = _sm70_dump_qwen_layer_tensor(
            "qsa_selected_indices",
            self.indexer.layer_id,
            "qsa",
            selected,
        )
        recorder = self._round_cost_selection
        if recorder is not None and not dense:
            recorder.capture(
                selected,
                side_metadata.token_to_req[:num_tokens],
                side_metadata.logical_positions[:num_tokens],
            )
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if self.host_kv_enabled:
            self.host_kv.write(key, value, main_metadata.slot_mapping)
        elif query.dim() == 2:
            # SM70 prep: raw qkv rows -> normalized/rotated query + cache write.
            key_cache, value_cache = self.kv_cache.unbind(1)
            prepared = query.new_empty((num_tokens, self.num_heads, self.head_dim))
            text_positions = positions[0] if positions.dim() == 2 else positions
            torch.ops._C.qsa_prep_sm70_out(
                query[:num_tokens],
                text_positions[:num_tokens].to(torch.int64),
                self.rotary_emb.cos_sin_cache,
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.variance_epsilon,
                prepared,
                key_cache,
                value_cache,
                main_metadata.slot_mapping[:num_tokens].to(torch.int64),
            )
            query = key = value = prepared
        else:
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            output_gate=output_gate,
            query_positions=side_metadata.logical_positions,
            sequence_lengths=side_metadata.seq_lens,
            dense_short_context=dense,
        )
        if recorder is not None and self.host_kv_enabled and not dense:
            recorder.capture_cache(self.host_kv._stats, num_tokens)
        _sm70_dump_qwen_layer_tensor(
            "qsa_core_out",
            self.indexer.layer_id,
            "qsa",
            output[:num_tokens],
        )

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        side = getattr(self, "sm70_side_projection", None)
        indexer_input = hidden_states
        if side is not None and 1 <= num_tokens <= 8:
            # The indexer receives its projected q/k instead of hidden states.
            qkv, indexer_input = side(hidden_states)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        if self._sm70_qsa_prep and num_tokens <= 32:
            # Raw rows go to the attention op, which normalizes, rotates and
            # caches them in one launch; the gate stays strided in q_gate.
            q_gate = qkv[:, : self.q_size * 2].view(
                num_tokens, self.num_heads, 2 * self.head_dim
            )
            gate = q_gate[..., self.head_dim :].reshape(num_tokens, self.q_size)
            query = qkv
            key = value = qkv
            attn_output = qkv.new_empty((num_tokens, self.num_heads, self.head_dim))
        else:
            q, k, v, gate = self._project_qkv_gate(qkv, positions)
            query = q.view(num_tokens, self.num_heads, self.head_dim)
            key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
            attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                indexer_input,
                positions,
                query,
                key,
                value,
                attn_output,
                gate,
                encoded_layer_name,
            )
        else:
            qwen4_exp_qsa_with_output(
                indexer_input,
                positions,
                query,
                key,
                value,
                attn_output,
                gate,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        hcx_projection = getattr(self, "sm70_hcx_projection_name", None)
        if hcx_projection is not None:
            projected_output = torch.ops.vllm.qwen38_sm70_hcx_output_projection(
                flat_output, hcx_projection
            )
            if output is not None:
                output.copy_(projected_output)
            return projected_output
        projected_output, _ = self.o_proj(flat_output)
        if output is not None:
            output.copy_(projected_output)
        return projected_output


def qwen4_exp_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_gate: torch.Tensor | None,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
        output_gate,
    )


def qwen4_exp_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_gate: torch.Tensor | None,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, output_gate, layer_name


direct_register_custom_op(
    op_name="qwen4_exp_qsa_with_output",
    op_func=qwen4_exp_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen4_exp_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "qwen4_exp_qsa_with_output",
]
