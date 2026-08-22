# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    PrefixAnchoredSWASpec,
    get_kv_quant_mode,
)


class PrefixAnchoredSWAAttention(Attention):
    """Attention layer that reports ``PrefixAnchoredSWASpec`` as its KV spec.

    Drop-in replacement for the standard ``Attention`` layer when the model
    is configured with prefix-anchored sliding-window attention
    (``decode_sliding_window > 0``): the prompt/prefix stays globally
    attended while generated tokens additionally attend only a fixed window
    of recent tokens. The actual masking logic lives in the attention
    backend; this layer only overrides ``get_kv_cache_spec`` so the KV cache
    manager instantiates ``PrefixAnchoredSWAManager`` (instead of
    ``FullAttentionManager``) and can therefore evict "gap" blocks to keep
    per-request KV memory bounded at O(prefix + window).
    """

    def __init__(self, *args, decode_sliding_window: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._decode_sliding_window = decode_sliding_window

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return None
        assert self.sliding_window is None, (
            "Prefix-anchored SWA cannot be combined with a uniform "
            "per-layer sliding window."
        )
        return PrefixAnchoredSWASpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            decode_sliding_window=self._decode_sliding_window,
        )
