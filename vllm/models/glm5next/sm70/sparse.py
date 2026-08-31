# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GLM-5.3-Flash sparse MLA backend for exact SM70 CUDA devices."""

import os
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.models.deepseek_v4.sm70.sparse_kernels import (
    sm70_sparse_attention_gathered,
)
from vllm.models.glm5next.sm70.fp8_kv import (
    GLM5_FP8_KV_SLOT_BYTES,
    sm70_glm5_fp8_kv_insert,
    sm70_glm5_sparse_attention_paged_fp8,
    sm70_glm5_sparse_attention_paged_fp8_batched_gemm,
    sm70_glm5_sparse_attention_paged_fp8_gemm,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionLayer, SparseMLAAttentionImpl
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

_DEBUG_DFLASH_SPARSE_INDICES = bool(
    int(os.getenv("VLLM_DFLASH_DEBUG_COORD_TRACE", "0"))
)
_DFLASH_SPARSE_INDICES_SEEN = False
_FP8_GEMM_MAX_TOKENS = 8


class Glm5NextSM70SparseBackend(FlashMLASparseBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "GLM5_SM70_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["Glm5NextSM70SparseImpl"]:
        return Glm5NextSM70SparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del num_kv_heads
        if cache_dtype_str in {"fp8", "fp8_e4m3"}:
            return (num_blocks, block_size, GLM5_FP8_KV_SLOT_BYTES)
        return (num_blocks, block_size, head_size)

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 7 and capability.minor == 0

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        del head_size, dtype, kv_cache_dtype, block_size, use_mla, has_sink
        del use_sparse, device_capability
        config = get_current_vllm_config_or_none()
        if config is None or config.model_config is None:
            return None
        text_config = config.model_config.hf_text_config
        if (
            getattr(text_config, "qk_rope_head_dim", None) != 0
            or getattr(text_config, "kv_lora_rank", None) != 512
        ):
            return "GLM5_SM70_SPARSE requires NoPE MLA with kv_lora_rank=512"
        return None


class Glm5NextSM70SparseImpl(SparseMLAAttentionImpl[FlashMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        del alibi_slopes, sliding_window, logits_soft_cap, attn_type
        del kv_sharing_target_layer_name, topk_indices_buffer
        if num_kv_heads != 1 or head_size != 512:
            raise NotImplementedError(
                "GLM-5.3 SM70 sparse MLA requires one 512-wide latent KV head."
            )
        if kv_cache_dtype not in {"auto", "float16", "fp8", "fp8_e4m3"}:
            raise NotImplementedError(
                "GLM-5.3 SM70 sparse MLA supports FP16 or E4M3 KV cache."
            )
        if mla_args["qk_rope_head_dim"] != 0 or mla_args["kv_lora_rank"] != 512:
            raise NotImplementedError(
                "GLM-5.3 SM70 sparse MLA requires NoPE and kv_lora_rank=512."
            )
        if indexer is None:
            raise ValueError("GLM-5.3 sparse MLA requires its KPool indexer.")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.softmax_scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.use_fp8_cache = kv_cache_dtype in {"fp8", "fp8_e4m3"}
        self.kv_lora_rank = 512
        topk_indices_buffer = indexer.topk_indices_buffer
        if topk_indices_buffer is None:
            raise ValueError("GLM-5.3 sparse MLA requires an index buffer.")
        self.topk_indices_buffer = topk_indices_buffer
        self.index_width = self.topk_indices_buffer.shape[1]

        config = get_current_vllm_config_or_none()
        if config is None:
            raise RuntimeError("GLM-5.3 SM70 sparse MLA requires VllmConfig.")
        max_tokens = config.scheduler_config.max_num_batched_tokens
        self.fp8_gemm_max_tokens = min(max_tokens, _FP8_GEMM_MAX_TOKENS)
        workspace_specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            ((max_tokens, num_heads, self.kv_lora_rank), torch.float16),
        ]
        if self.use_fp8_cache:
            workspace_specs.extend(
                (
                    (
                        (
                            self.fp8_gemm_max_tokens,
                            self.index_width,
                            self.kv_lora_rank,
                        ),
                        torch.float16,
                    ),
                    (
                        (self.fp8_gemm_max_tokens, num_heads, self.index_width),
                        torch.float16,
                    ),
                    (
                        (self.fp8_gemm_max_tokens, num_heads, self.index_width),
                        torch.float16,
                    ),
                )
            )
        current_workspace_manager().get_simultaneous(*workspace_specs)
        logger.info_once(
            "GLM-5.3-Flash route: SM70 FP16 sparse MLA with %s KV%s.",
            "packed E4M3FN" if self.use_fp8_cache else "FP16",
            (
                " and B1/M2-M8 dequant + tensor-core GEMM decode"
                if self.use_fp8_cache
                else ""
            ),
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if not self.use_fp8_cache:
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
        del k_scale
        if kv_cache.numel() == 0 or slot_mapping is None:
            return
        if k_pe.shape[-1] != 0:
            raise NotImplementedError("GLM-5.3 packed FP8 KV is NoPE-only.")
        sm70_glm5_fp8_kv_insert(
            kv_c_normed,
            kv_cache.view(torch.uint8),
            slot_mapping.flatten(),
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del layer
        global _DFLASH_SPARSE_INDICES_SEEN
        if isinstance(q, tuple):
            q_nope, q_pe = q
            if q_pe.shape[-1] != 0:
                raise NotImplementedError("GLM-5.3 SM70 sparse MLA is NoPE-only.")
            q = q_nope

        if q.dtype != torch.float16:
            raise TypeError("GLM-5.3 SM70 sparse MLA requires an FP16 query.")
        if q.shape[-1] != self.kv_lora_rank:
            raise ValueError(f"Expected latent query width 512, got {q.shape[-1]}.")

        num_tokens = q.shape[0]
        topk_indices = self.topk_indices_buffer[:num_tokens]
        global_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        if (
            _DEBUG_DFLASH_SPARSE_INDICES
            and not _DFLASH_SPARSE_INDICES_SEEN
            and 1 < num_tokens <= 8
        ):
            first_slot = int(attn_metadata.slot_mapping[0].item())
            first_logical_position = (
                first_slot % attn_metadata.block_size if first_slot >= 0 else -1
            )
            min_position = int(
                os.getenv("VLLM_DFLASH_DEBUG_TARGET_TRACE_MIN_POSITION", "8")
            )
            if first_logical_position >= min_position:
                valid = topk_indices.ge(0)
                logical_min = torch.where(
                    valid, topk_indices, torch.iinfo(topk_indices.dtype).max
                ).amin(dim=1)
                logical_max = torch.where(valid, topk_indices, -1).amax(dim=1)
                logger.warning(
                    "DFLASH_TARGET_SPARSE_INDICES req_ids=%s valid_counts=%s "
                    "slot_mapping=%s logical_min=%s logical_max=%s logical_sum=%s "
                    "logical_prefix=%s block_table_prefix=%s global_prefix=%s",
                    attn_metadata.req_id_per_token[:num_tokens].detach().cpu().tolist(),
                    valid.sum(dim=1).detach().cpu().tolist(),
                    attn_metadata.slot_mapping[:num_tokens].detach().cpu().tolist(),
                    logical_min.detach().cpu().tolist(),
                    logical_max.detach().cpu().tolist(),
                    torch.where(valid, topk_indices, 0)
                    .sum(dim=1)
                    .detach()
                    .cpu()
                    .tolist(),
                    topk_indices[:, :24].detach().cpu().tolist(),
                    attn_metadata.block_table[:1, :4].detach().cpu().tolist(),
                    global_indices[:num_tokens, :24].detach().cpu().tolist(),
                )
                _DFLASH_SPARSE_INDICES_SEEN = True
        workspace_manager = current_workspace_manager()
        if self.use_fp8_cache:
            if num_tokens == 1:
                out, gathered_kv, scores, probs = workspace_manager.get_simultaneous(
                    ((1, self.num_heads, self.kv_lora_rank), torch.float16),
                    ((self.index_width, self.kv_lora_rank), torch.float16),
                    ((self.num_heads, self.index_width), torch.float16),
                    ((self.num_heads, self.index_width), torch.float16),
                )
                sm70_glm5_sparse_attention_paged_fp8_gemm(
                    q,
                    kv_c_and_k_pe_cache.view(torch.uint8),
                    global_indices,
                    valid_counts,
                    self.softmax_scale,
                    out,
                    gathered_kv,
                    scores,
                    probs,
                )
            elif num_tokens <= self.fp8_gemm_max_tokens:
                out, gathered_kv, scores, probs = workspace_manager.get_simultaneous(
                    (
                        (num_tokens, self.num_heads, self.kv_lora_rank),
                        torch.float16,
                    ),
                    (
                        (num_tokens, self.index_width, self.kv_lora_rank),
                        torch.float16,
                    ),
                    (
                        (num_tokens, self.num_heads, self.index_width),
                        torch.float16,
                    ),
                    (
                        (num_tokens, self.num_heads, self.index_width),
                        torch.float16,
                    ),
                )
                sm70_glm5_sparse_attention_paged_fp8_batched_gemm(
                    q,
                    kv_c_and_k_pe_cache.view(torch.uint8),
                    global_indices,
                    valid_counts,
                    self.softmax_scale,
                    out,
                    gathered_kv,
                    scores,
                    probs,
                )
            else:
                (out,) = workspace_manager.get_simultaneous(
                    (
                        (num_tokens, self.num_heads, self.kv_lora_rank),
                        torch.float16,
                    ),
                )
                sm70_glm5_sparse_attention_paged_fp8(
                    q,
                    kv_c_and_k_pe_cache.view(torch.uint8),
                    global_indices,
                    valid_counts,
                    self.softmax_scale,
                    out,
                )
        else:
            if kv_c_and_k_pe_cache.dtype != torch.float16:
                raise TypeError("GLM-5.3 FP16 KV route received a non-FP16 cache.")
            (out,) = workspace_manager.get_simultaneous(
                (
                    (num_tokens, self.num_heads, self.kv_lora_rank),
                    torch.float16,
                ),
            )
            sm70_sparse_attention_gathered(
                q,
                kv_c_and_k_pe_cache,
                global_indices,
                valid_counts,
                self.softmax_scale,
                None,
                out,
            )
        return out, None
