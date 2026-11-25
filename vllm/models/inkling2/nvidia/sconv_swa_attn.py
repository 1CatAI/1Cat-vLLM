# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling short-conv state managed as a sliding-window KV cache (inlined)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from .ops.sconv import sconv_seq_metadata


@dataclass
class InklingSconvMetadata(AttentionMetadata):
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_idx: torch.Tensor
    query_start: torch.Tensor


class InklingSconvMetadataBuilder(AttentionMetadataBuilder[InklingSconvMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, AttentionSpec)
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.seq_idx_buffer = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )
        self.query_start_buffer = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InklingSconvMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        num_padded_tokens = common_attn_metadata.slot_mapping.shape[0]
        assert num_padded_tokens >= num_actual_tokens

        sconv_seq_metadata(
            common_attn_metadata.query_start_loc,
            num_reqs,
            num_actual_tokens,
            self.seq_idx_buffer,
            self.query_start_buffer,
            num_padded_tokens,
        )

        return InklingSconvMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:num_padded_tokens],
            seq_idx=self.seq_idx_buffer[:num_padded_tokens],
            query_start=self.query_start_buffer[:num_padded_tokens],
        )


class InklingSconvBackend(AttentionBackend):
    """Custom dummy backend for the sconv sliding-window cache management."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "INKLING_SCONV_SWA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "InklingSconvBackend has no attention impl; the conv runs out-of-band."
        )

    @staticmethod
    def get_builder_cls() -> type[InklingSconvMetadataBuilder]:
        return InklingSconvMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, num_kv_heads, block_size, head_size)


class InklingConvState(nn.Module, AttentionLayerBase):
    """Per-decoder-layer owner emitting one sliding-window conv-state spec."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        kernel_size: int,
        prefix: str,
        replicate_hidden: bool = False,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= num_kv_heads, (
            f"sconv SWA cache supports tp_size <= num_kv_heads ({num_kv_heads}), "
            f"got {tp_size}"
        )
        self.num_kv_heads = num_kv_heads // tp_size
        hidden_per_head = hidden_size // (
            self.num_kv_heads if replicate_hidden else num_kv_heads
        )
        raw_head_size = 2 * head_dim + 2 * hidden_per_head
        self.head_size = 1 << (raw_head_size - 1).bit_length()
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        self.stream_ranges: tuple[tuple[int, int], ...] = (
            (0, head_dim),
            (head_dim, head_dim),
            (2 * head_dim, hidden_per_head),
            (2 * head_dim + hidden_per_head, hidden_per_head),
        )
        vllm_config = get_current_vllm_config()
        self._dtype = vllm_config.model_config.dtype
        assert self._dtype in (torch.float16, torch.bfloat16), (
            f"sconv SWA cache supports float16/bfloat16 only, got {self._dtype}"
        )
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self): ...

    @property
    def cache_block_size(self) -> int:
        if self.kv_cache.numel() > 0:
            return self.kv_cache.shape[2]
        return self.block_size

    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingSconvBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self._dtype,
            sliding_window=self.sliding_window,
        )
