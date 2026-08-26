# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DeepSeek V4 sparse MLA implementation for exact SM70 CUDA devices."""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLASparseBackend,
    DeepseekV4SparseMLAAttentionImpl,
)
from vllm.models.deepseek_v4.sm70.sparse_kernels import (
    sm70_sparse_attention_gathered,
    sm70_sparse_attention_paged_fp8,
    sm70_sparse_attention_paged_fp8_splitk,
    sm70_sparse_attention_paged_fp8_splitk_qk_dsplit,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.attention import DeepseekV4MLAAttention
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

logger = init_logger(__name__)


class DeepseekV4SM70SparseBackend(DeepseekV4FlashMLASparseBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]

    @staticmethod
    def get_name() -> str:
        return "V4_SM70_TRITON_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["DeepseekV4SM70SparseImpl"]:
        return DeepseekV4SM70SparseImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 7 and capability.minor == 0


class DeepseekV4SM70SparseImpl(DeepseekV4SparseMLAAttentionImpl):
    """FP16 Triton attention with direct packed-FP8 paged decode."""

    backend_cls: ClassVar[type[AttentionBackend]] = DeepseekV4SM70SparseBackend
    # The upstream value of 32 reserves about 8.3 GiB at the checkpoint's
    # 1M context. V100-32GB needs a smaller model-wide eager workspace budget.
    PREFILL_CHUNK_SIZE = 8

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    @classmethod
    def forward_mqa(  # type: ignore[override]
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert q.dtype == output.dtype == torch.float16
        assert q.shape == output.shape
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            swa_only = layer.compress_ratio <= 1
            compressed_tokens = (
                0
                if swa_only
                else (layer.max_model_len + layer.compress_ratio - 1)
                // layer.compress_ratio
            )
            workspace_tokens = (
                compressed_tokens + layer.window_size + layer.max_num_batched_tokens
            )
            assert layer.topk_indices_buffer is not None
            if swa_only:
                top_k = 0
            elif layer.compress_ratio == 128:
                top_k = compressed_tokens
            else:
                top_k = layer.topk_indices_buffer.shape[-1]
            combined_topk = round_up(top_k + layer.window_size, 128)
            current_workspace_manager().get_simultaneous(
                (
                    (cls.PREFILL_CHUNK_SIZE, workspace_tokens, q.shape[-1]),
                    torch.float16,
                ),
                ((layer.max_num_batched_tokens, combined_topk), torch.int32),
                ((layer.max_num_batched_tokens,), torch.int32),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        sparse_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(layer.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(layer.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = layer.compress_ratio <= 1
        compressed_cache = layer.kv_cache if not swa_only else None
        num_decode_tokens = swa_metadata.num_decode_tokens

        if swa_metadata.num_prefills > 0:
            cls._forward_prefill(
                layer,
                q[num_decode_tokens:],
                compressed_cache,
                layer.swa_cache_layer.kv_cache,
                output[num_decode_tokens:],
                sparse_metadata,
                swa_metadata,
            )
        if swa_metadata.num_decodes > 0:
            cls._forward_decode(
                layer,
                q[:num_decode_tokens],
                compressed_cache,
                output[:num_decode_tokens],
                sparse_metadata,
                swa_metadata,
                swa_only,
            )

    @classmethod
    def _forward_decode(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        compressed_cache: torch.Tensor | None,
        output: torch.Tensor,
        sparse_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        swa_only: bool,
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert sparse_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = sparse_metadata.block_size // layer.compress_ratio
            valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if layer.compress_ratio == 4:
                assert layer.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    layer.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    sparse_metadata.block_table[: swa_metadata.num_decodes],
                    block_size,
                    valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            elif layer.compress_ratio == 128:
                topk_indices = sparse_metadata.c128a_global_decode_topk_indices
                topk_lens = sparse_metadata.c128a_decode_topk_lens
            else:
                raise ValueError(
                    f"Unsupported DeepSeek V4 compress_ratio={layer.compress_ratio}"
                )

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None and swa_lens is not None
        use_splitk = (
            (swa_only and envs.VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_SWA)
            or (layer.compress_ratio == 4 and envs.VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_C4)
            or (
                layer.compress_ratio == 128
                and envs.VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_C128
            )
        )
        if use_splitk:
            main_width = swa_indices.reshape(num_decode_tokens, -1).shape[1]
            extra_width = (
                0
                if topk_indices is None
                else topk_indices.reshape(num_decode_tokens, -1).shape[1]
            )
            num_partials = (main_width + 15) // 16 + (extra_width + 15) // 16
            use_qk_dsplit = envs.VLLM_SM70_DSV4_SPARSE_MLA_QK_DSPLIT
            workspace_specs = [
                ((num_decode_tokens, q.shape[1], num_partials), torch.float32),
                ((num_decode_tokens, q.shape[1], num_partials), torch.float32),
                (
                    (num_decode_tokens, q.shape[1], num_partials, q.shape[2]),
                    torch.float32,
                ),
            ]
            if use_qk_dsplit:
                workspace_specs.extend(
                    (
                        (
                            (
                                num_decode_tokens,
                                q.shape[1],
                                num_partials,
                                q.shape[2] // 64,
                                16,
                            ),
                            torch.float32,
                        ),
                        (
                            (num_decode_tokens, q.shape[1], num_partials, 16),
                            torch.float16,
                        ),
                    )
                )
            workspaces = current_workspace_manager().get_simultaneous(*workspace_specs)
            partial_max, partial_sum, partial_acc = workspaces[:3]
            logger.info_once(
                "DeepSeek V4 SM70 %s sparse MLA split-K%s enabled.",
                "SWA-only" if swa_only else f"C{layer.compress_ratio}",
                " QK-D split" if use_qk_dsplit else "",
            )
            common_kwargs = dict(
                q=q,
                main_cache=layer.swa_cache_layer.kv_cache,
                main_indices=swa_indices,
                main_lengths=swa_lens,
                scale=layer.scale,
                attn_sink=layer.attn_sink[: q.shape[1]],
                out=output,
                extra_cache=compressed_cache,
                extra_indices=topk_indices,
                extra_lengths=topk_lens,
                partial_max=partial_max,
                partial_sum=partial_sum,
                partial_acc=partial_acc,
            )
            if use_qk_dsplit:
                partial_qk, partial_probs = workspaces[3:]
                sm70_sparse_attention_paged_fp8_splitk_qk_dsplit(
                    partial_qk=partial_qk,
                    partial_probs=partial_probs,
                    **common_kwargs,
                )
            else:
                sm70_sparse_attention_paged_fp8_splitk(**common_kwargs)
        else:
            sm70_sparse_attention_paged_fp8(
                q=q,
                main_cache=layer.swa_cache_layer.kv_cache,
                main_indices=swa_indices,
                main_lengths=swa_lens,
                scale=layer.scale,
                attn_sink=layer.attn_sink[: q.shape[1]],
                out=output,
                extra_cache=compressed_cache,
                extra_indices=topk_indices,
                extra_lengths=topk_lens,
            )

    @classmethod
    def _forward_prefill(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        compressed_cache: torch.Tensor | None,
        swa_cache: torch.Tensor,
        output: torch.Tensor,
        sparse_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = sparse_metadata is None
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc_cpu is not None and query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if layer.compress_ratio == 4:
                assert layer.topk_indices_buffer is not None
                topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            elif layer.compress_ratio == 128:
                assert sparse_metadata is not None
                topk_indices = sparse_metadata.c128a_prefill_topk_indices
                assert topk_indices is not None
            else:
                raise ValueError(
                    f"Unsupported DeepSeek V4 compress_ratio={layer.compress_ratio}"
                )
            top_k = topk_indices.shape[-1]
        else:
            assert layer.topk_indices_buffer is not None
            topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
            top_k = 0

        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=layer.compress_ratio,
            prefill_chunk_size=cls.PREFILL_CHUNK_SIZE,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        combined_topk = round_up(top_k + layer.window_size, 128)
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            current_chunk = chunk_end - chunk_start
            workspace = workspace_manager.get_simultaneous(
                ((current_chunk, chunk_M, q.shape[-1]), torch.float16),
                ((layer.max_num_batched_tokens, combined_topk), torch.int32),
                ((layer.max_num_batched_tokens,), torch.int32),
            )
            kv_workspace, combined_indices_out, combined_lens_out = workspace
            if not swa_only:
                assert sparse_metadata is not None and compressed_cache is not None
                dequantize_and_gather_k_cache(
                    kv_workspace[:current_chunk],
                    compressed_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // layer.compress_ratio,
                    gather_lens=None,
                    block_table=sparse_metadata.block_table[num_decodes:][
                        chunk_start:chunk_end
                    ],
                    block_size=sparse_metadata.block_size // layer.compress_ratio,
                    offset=0,
                )

            dequantize_and_gather_k_cache(
                kv_workspace[:current_chunk],
                swa_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_metadata.block_table[num_decodes:][
                    chunk_start:chunk_end
                ],
                block_size=swa_metadata.block_size,
                offset=chunk_N,
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices_out = combined_indices_out[: query_end - query_start]
            combined_lens_out = combined_lens_out[: query_end - query_start]
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                layer.window_size,
                layer.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
                out=(combined_indices_out, combined_lens_out),
            )
            sm70_sparse_attention_gathered(
                q[query_start:query_end],
                kv_workspace[:current_chunk],
                combined_indices,
                combined_lens,
                layer.scale,
                layer.attn_sink[: q.shape[1]],
                output[query_start:query_end],
            )

        logger.debug_once("DeepSeek V4 SM70 FP16 sparse attention route active.")
