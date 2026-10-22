# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 FP16 latent cache registration, with physical compression geometry."""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v4.compressor import CompressorBackend
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


@dataclass
class V41CacheMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    is_prefilling: torch.Tensor
    block_size: int
    causal: bool = True


class V41CacheBuilder(AttentionMetadataBuilder):
    # The model admits one request, and exact M1..6 graphs retain the same
    # block/slot buffers. The device flag also supports a small prefill at
    # the same shape; no mixed-request geometry is admitted by build().
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    uses_physical_block_table = True

    _shared_flags: ClassVar[dict] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Every V4.1 group reads the same single-request prefill state. One
        # device flag (and its last host value) is shared by all builders.
        shared = V41CacheBuilder._shared_flags.get(self.device)
        if shared is None:
            shared = [torch.ones(1, dtype=torch.bool, device=self.device), None, None]
            V41CacheBuilder._shared_flags[self.device] = shared
        self._shared = shared
        self.prefill_flag = shared[0]

    def _set_flag(self, value, token):
        shared = self._shared
        if shared[1] is not None and shared[2] == value:
            return
        shared[0].fill_(value)
        shared[1], shared[2] = token, value

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        del common_prefix_len, fast_build
        common = common_attn_metadata
        if common.num_reqs != 1:
            raise ValueError("V4.1 FP16 currently supports exactly one request")
        # The common flag is normally CPU-backed. A graph must read one
        # stable device address that the builder updates before every replay,
        # rather than capturing a host copy or a dummy batch's constant flag.
        source = common.is_prefilling
        if common.is_dummy_batch:
            self._set_flag(False, common)
        elif source is None:
            self._set_flag(True, common)
        elif source.device.type == "cpu":
            self._set_flag(bool(source[0]), common)
        else:
            self.prefill_flag.copy_(source[:1])
            self._shared[1] = None
        return V41CacheMetadata(
            common.block_table_tensor,
            common.slot_mapping,
            self.prefill_flag,
            self.kv_cache_spec.block_size,
            causal=common.causal,
        )


class V41CacheBackend(CompressorBackend):
    supported_dtypes = [torch.float16]

    @staticmethod
    def get_name():
        return "DEEPSEEK_V41_FP16_LATENTS"

    @staticmethod
    def get_builder_cls():
        return V41CacheBuilder

    @classmethod
    def get_supported_head_sizes(cls):
        return [128, 512]


class V41LatentCache(nn.Module, AttentionLayerBase):
    """One K=V vector per physical row, after the official FP4/FP8 QDQ."""

    def __init__(self, prefix, head_dim=512, ratio=1, sliding_window=None):
        super().__init__()
        self.prefix, self.head_dim, self.ratio = prefix, head_dim, ratio
        self.sliding_window = sliding_window
        self.kv_cache = torch.empty(0, dtype=torch.float16)
        context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in context or head_dim not in (128, 512) or ratio not in (1, 2):
            raise ValueError("Invalid or duplicate V4.1 latent cache")
        context[prefix] = self

    def get_attn_backend(self):
        return V41CacheBackend

    def get_kv_cache_spec(self, vllm_config):
        block = vllm_config.cache_config.block_size
        if block % self.ratio:
            raise ValueError("V4.1 cache block must cover complete compression groups")
        kwargs = dict(
            block_size=block,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.float16,
            compress_ratio=self.ratio,
            # One page size across window, compressed and index pools. The
            # engine exposes padded physical-page strides to the gather ops.
            alignment=block * 512 * 2,
        )
        if self.sliding_window is not None:
            return SlidingWindowMLASpec(sliding_window=self.sliding_window, **kwargs)
        return MLAAttentionSpec(**kwargs)

    def metadata(self):
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return None
        return metadata[self.prefix]
