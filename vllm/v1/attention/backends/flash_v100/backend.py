# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 attention backend registration."""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100 import impl as _impl
from vllm.v1.attention.backends.flash_v100 import metadata as _metadata
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


class FlashAttnV100Backend(TritonAttentionBackend):
    """Flash Attention V100 Backend."""

    # Keep vLLM unified KV cache update path.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_impl_cls():
        return _impl.FlashAttnV100Impl

    @staticmethod
    def get_builder_cls():
        return _metadata.FlashAttnV100MetadataBuilder

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_V100"

    @staticmethod
    def get_supported_kernel_block_sizes():
        if _config.options().value("kernel_block_size16"):
            return [16]
        return TritonAttentionBackend.get_supported_kernel_block_sizes()

    @staticmethod
    def get_hybrid_block_multiple() -> int | None:
        # _compact_sm70_hybrid_sliding_window_pages divides a higher-precision
        # draft block by its page ratio (2x FP16 over 8-bit, 4x over FP32) and
        # rounds attention alone up when the result is not a multiple of 16:
        # 1648/1664 then leaves Mamba on another grid and nothing hits (#1422).
        return 64

    @classmethod
    def supports_non_causal(cls) -> bool:
        # D-Flash uses non-causal decoder attention over the draft query
        # tokens. The V100 backend handles this in the prefill paths by
        # forwarding attn_metadata.causal to FA2/Triton-compatible kernels.
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # Keep this aligned with the dense prefill kernel dispatch table.
        return [64, 128, 256]
