# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 attention implementation (forward and route bodies)."""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100 import debug as _debug
from vllm.v1.attention.backends.flash_v100 import debug_compare as _debug_compare
from vllm.v1.attention.backends.flash_v100 import decode as _decode
from vllm.v1.attention.backends.flash_v100 import dense_prefill as _dense_prefill
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import ops as _ops
from vllm.v1.attention.backends.flash_v100 import prefill as _prefill
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100 import state as _state
from vllm.v1.attention.backends.flash_v100 import verify as _verify
from vllm.v1.attention.backends.flash_v100.spec.attention import (
    ATTENTION_HOOKS,
    SpecAttentionMethods,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    KVCodec,
    resolve_kv_codec,
)
from vllm.v1.attention.ops.sm70_grouped import (
    load_grouped_e4m3_fp32,
    load_grouped_fp16_fp32,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


class FlashAttnV100Impl(SpecAttentionMethods, TritonAttentionImpl):
    """Flash Attention V100 implementation with explicit fallback policy."""

    allow_triton_fallback = _config.ConfigField[bool]("allow_triton_fallback")
    compare_bhmd_out_dir = _config.ConfigField[str | None]("compare_bhmd_out_dir")
    compare_bhmd_out_max_calls = _config.ConfigField[int]("compare_bhmd_out_max_calls")
    compare_triton_out_dir = _config.ConfigField[str | None]("compare_triton_out_dir")
    compare_triton_out_max_calls = _config.ConfigField[int](
        "compare_triton_out_max_calls"
    )
    compare_triton_tensor_dump_dir = _config.ConfigField[str | None](
        "compare_triton_tensor_dump_dir"
    )
    compare_triton_tensor_dump_max_tokens = _config.ConfigField[int](
        "compare_triton_tensor_dump_max_tokens"
    )
    decode_strategy = _config.ConfigField[str]("decode_strategy")
    prefill_bfla_mask_block_n = _config.ConfigField[int]("prefill_bfla_mask_block_n")
    prefill_bfla_min_kv = _config.ConfigField[int]("prefill_bfla_min_kv")
    prefill_bfla_min_q = _config.ConfigField[int]("prefill_bfla_min_q")
    prefill_contig_dense_allow_copy = _config.ConfigField[bool](
        "prefill_contig_dense_allow_copy"
    )
    prefill_contig_dense_min_kv = _config.ConfigField[int](
        "prefill_contig_dense_min_kv"
    )
    prefill_contig_dense_min_q = _config.ConfigField[int]("prefill_contig_dense_min_q")
    prefill_gather_dense_min_kv = _config.ConfigField[int](
        "prefill_gather_dense_min_kv"
    )
    prefill_gather_dense_min_q = _config.ConfigField[int]("prefill_gather_dense_min_q")
    prefill_split_kv_max_q = _config.ConfigField[int]("prefill_split_kv_max_q")
    prefill_split_kv_min_kv = _config.ConfigField[int]("prefill_split_kv_min_kv")
    prefill_split_kv_min_q = _config.ConfigField[int]("prefill_split_kv_min_q")
    prefill_split_kv_tokens = _config.ConfigField[int]("prefill_split_kv_tokens")
    prefix_anchored_decode_window = _config.ConfigField[int | None](
        "prefix_anchored_decode_window"
    )
    smallq_decode_max_model_len = _config.ConfigField[int](
        "smallq_decode_max_model_len"
    )
    smallq_decode_max_query_len = _config.ConfigField[int](
        "smallq_decode_max_query_len"
    )
    use_decode_dense_cache = _config.ConfigField[bool]("use_decode_dense_cache")
    use_decode_dense_reference = _config.ConfigField[bool]("use_decode_dense_reference")
    use_decode_paged_prefill = _config.ConfigField[bool]("use_decode_paged_prefill")
    use_decode_paged_prefill_bhmd_out = _config.ConfigField[bool](
        "use_decode_paged_prefill_bhmd_out"
    )
    use_decode_scalar_paged = _config.ConfigField[bool]("use_decode_scalar_paged")
    use_decode_wmma_wrapper = _config.ConfigField[bool]("use_decode_wmma_wrapper")
    use_decode_xqa = _config.ConfigField[bool]("use_decode_xqa")
    use_flash_v100 = _config.ConfigField[bool]("use_flash_v100")
    use_flash_v100_decode = _config.ConfigField[bool]("use_flash_v100_decode")
    use_flash_v100_prefill_bfla = _config.ConfigField[bool](
        "use_flash_v100_prefill_bfla"
    )
    use_flash_v100_prefill_contig_dense = _config.ConfigField[bool](
        "use_flash_v100_prefill_contig_dense"
    )
    use_flash_v100_prefill_gather_dense = _config.ConfigField[bool](
        "use_flash_v100_prefill_gather_dense"
    )
    use_flash_v100_prefill_paged = _config.ConfigField[bool](
        "use_flash_v100_prefill_paged"
    )
    use_flash_v100_prefill_splitkv = _config.ConfigField[bool](
        "use_flash_v100_prefill_splitkv"
    )
    use_fp8_prefill_bridge = _config.ConfigField[bool]("use_fp8_prefill_bridge")
    use_prefill_paged_cache = _config.ConfigField[bool]("use_prefill_paged_cache")
    use_smallq_decode_xqa = _config.ConfigField[bool]("use_smallq_decode_xqa")
    use_triton_prefill = _config.ConfigField[bool]("use_triton_prefill")

    def __init__(self, *args, **kwargs):
        self.prefix_anchored_decode_window = kwargs.pop(
            "prefix_anchored_decode_window", None
        )
        super().__init__(*args, **kwargs)
        _routing._log_kv_dtype_contract(self.kv_cache_dtype)
        self.kv_cache_dtype = _routing._normalize_flash_v100_kv_cache_dtype(
            self.kv_cache_dtype
        )
        (
            self.flash_attn_func,
            self.flash_attn_bhmd_func,
            self.flash_attn_decode_paged,
            self.flash_attn_decode_paged_xqa,
            self.flash_attn_decode_paged_wmma,
            self.flash_attn_prefill_paged,
            self.flash_attn_prefill_paged_bhmd,
            self.flash_attn_prefill_paged_bfla,
            self.flash_attn_prefill_paged_splitkv,
        ) = _ops._get_flash_ops()
        self.flash_attn_grouped_verify_paged = _ops._get_flash_grouped_verify_op()
        use_e4m3_fp32 = (
            _config.registered("VLLM_FLASH_V100_E4M3_GROUPED_FP32")
            and self.kv_codec is FP8_E4M3
            and current_platform.is_device_capability(70)
        )
        self.flash_attn_grouped_e4m3_fp32_paged = (
            load_grouped_e4m3_fp32() if use_e4m3_fp32 else None
        )
        self.flash_attn_grouped_fp16_fp32_paged = (
            load_grouped_fp16_fp32()
            if self.kv_codec is FP16 and current_platform.is_device_capability(70)
            else None
        )
        ATTENTION_HOOKS.initialize_scalar_tail(self, use_e4m3_fp32)
        if use_e4m3_fp32 and self.flash_attn_grouped_e4m3_fp32_paged is None:
            logger.warning_once(
                "E4M3 grouped FP32 requires Flash-V100 precision revision 4; "
                "the E4M3 scalar fallback also requires this revision for "
                "FP32 partial storage. Rebuild the extension and restart workers.",
                scope="process",
            )
        ATTENTION_HOOKS.initialize_verify_abi(self)
        self.fp8_e5m2_paged_kv_to_fp16 = _ops._get_fp8_e5m2_paged_kv_bridge_op()
        self.fp8_e4m3_paged_kv_to_fp16 = (
            _ops._get_sm70_v37_e4m3_bridge_op() if self.kv_codec is FP8_E4M3 else None
        )
        # V100 FA2 kernels consume fp16 Q. FP8 KV cache support is implemented
        # as storage compression only, with K/V dequantized inside FA2 kernels.
        self.supports_quant_query_input = False
        self.use_flash_v100 = self.flash_attn_func is not None
        self.use_flash_v100_decode = self.flash_attn_decode_paged is not None
        self._flash_decode_paged_kwargs = {
            name
            for name in (
                "window_size",
                "max_seq_len_hint",
                "workspace_seq_capacity_hint",
                "active_num_partitions",
                "partition_size_hint",
                "anchor_lens",
                "anchored_window",
            )
            if self.flash_attn_decode_paged is not None
            and _ops._callable_accepts_keyword(self.flash_attn_decode_paged, name)
        }
        self._flash_prefill_paged_supports_anchor = (
            self.flash_attn_prefill_paged is not None
            and _ops._callable_accepts_keyword(
                self.flash_attn_prefill_paged, "anchor_lens"
            )
        )
        ATTENTION_HOOKS.configure_prefill(self)
        paged_prefill_enable = _config.raw("VLLM_FLASH_V100_ENABLE_PAGED_PREFILL")
        paged_prefill_disable = (
            _config.raw("VLLM_FLASH_V100_DISABLE_PAGED_PREFILL", "0") == "1"
        )
        self.use_flash_v100_prefill_paged = (
            self.flash_attn_prefill_paged is not None
            and paged_prefill_enable != "0"
            and not paged_prefill_disable
        )
        self.use_fp8_prefill_bridge = (
            self.fp8_e4m3_paged_kv_to_fp16 is not None
            if self.kv_codec is FP8_E4M3
            else self.fp8_e5m2_paged_kv_to_fp16 is not None
        ) and _config.raw("VLLM_FLASH_V100_FP8_PREFILL_BRIDGE", "1") != "0"
        self.use_flash_v100_prefill_splitkv = (
            self.flash_attn_prefill_paged_splitkv is not None
            and _config.registered("VLLM_FLASH_V100_PREFILL_SPLIT_KV")
            and self.use_flash_v100_prefill_paged
        )
        self.use_flash_v100_prefill_bfla = (
            self.flash_attn_prefill_paged_bfla is not None
            and _config.registered("VLLM_FLASH_V100_BFLA_PREFILL")
            and self.use_flash_v100_prefill_paged
        )
        self.use_flash_v100_prefill_contig_dense = (
            self.flash_attn_func is not None
            and self.use_flash_v100_prefill_paged
            and _config.registered("VLLM_FLASH_V100_PREFILL_CONTIG_DENSE")
        )
        self.prefill_contig_dense_min_q = _config.registered(
            "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_Q"
        )
        self.prefill_contig_dense_min_kv = _config.registered(
            "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_KV"
        )
        self.prefill_contig_dense_allow_copy = _config.registered(
            "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_ALLOW_COPY"
        )
        self.use_flash_v100_prefill_gather_dense = (
            self.use_flash_v100_prefill_paged
            and _config.registered("VLLM_FLASH_V100_PREFILL_GATHER_DENSE")
        )
        self.prefill_gather_dense_min_q = _config.registered(
            "VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_Q"
        )
        self.prefill_gather_dense_min_kv = _config.registered(
            "VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_KV"
        )
        self.prefill_split_kv_tokens = _config.registered(
            "VLLM_FLASH_V100_PREFILL_SPLIT_KV_TOKENS"
        )
        self.prefill_split_kv_min_q = _config.registered(
            "VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_Q"
        )
        self.prefill_split_kv_max_q = _config.registered(
            "VLLM_FLASH_V100_PREFILL_SPLIT_KV_MAX_Q"
        )
        self.prefill_split_kv_min_kv = _config.registered(
            "VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_KV"
        )
        self.prefill_bfla_min_q = _config.registered("VLLM_FLASH_V100_BFLA_MIN_Q")
        self.prefill_bfla_min_kv = _config.registered("VLLM_FLASH_V100_BFLA_MIN_KV")
        self.prefill_bfla_mask_block_n = _config.registered(
            "VLLM_FLASH_V100_BFLA_MASK_BLOCK_N"
        )
        self.use_prefill_paged_cache = (
            _config.raw("VLLM_FLASH_V100_PREFILL_USE_PAGED_CACHE", "0") == "1"
        )
        # Explicit diagnostic fallback only. The production migration target is
        # a complete Flash-V100 backend, so selected Flash routes should not
        # hide Flash prefill issues behind Triton by default.
        self.use_triton_prefill = (
            _config.raw("VLLM_FLASH_V100_PREFILL_USE_TRITON", "0") != "0"
        )
        self.allow_triton_fallback = (
            _config.raw("VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK", "0") == "1"
        )
        self.smallq_decode_max_query_len = int(
            _config.raw("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16")
        )
        self.smallq_decode_max_model_len = int(
            _config.raw("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN", "0")
        )
        self.use_decode_dense_reference = (
            _config.raw("VLLM_FLASH_V100_DECODE_DENSE_REFERENCE", "0") == "1"
        )
        self.use_decode_dense_cache = (
            _config.raw("VLLM_FLASH_V100_DECODE_DENSE_CACHE", "0") == "1"
        )
        # Classified quality rule: long q=1 scalar paged decode is a Type-B
        # reduction-order path, not a Type-A layout bug. Keep it as the
        # production Flash decode default so an explicit FLASH_ATTN_V100
        # selection does not silently become Triton during CUDA graph capture.
        decode_paged_prefill_env = _config.raw(
            "VLLM_FLASH_V100_DECODE_USE_PAGED_PREFILL"
        )
        self.use_decode_paged_prefill = decode_paged_prefill_env == "1"
        decode_bhmd_out_env = _config.raw("VLLM_FLASH_V100_DECODE_USE_BHMD_OUT")
        self.use_decode_paged_prefill_bhmd_out = decode_bhmd_out_env != "0"
        self.use_decode_wmma_wrapper = (
            _config.raw("VLLM_FLASH_V100_DECODE_USE_WMMA_WRAPPER", "0") == "1"
        )
        self.use_decode_xqa = _config.raw("VLLM_FLASH_V100_DECODE_USE_XQA", "1") == "1"
        self.use_smallq_decode_xqa = (
            self.use_decode_xqa
            and _config.raw("VLLM_FLASH_V100_SMALLQ_DECODE_USE_XQA", "1") == "1"
        )
        self.decode_strategy = _routing.resolve_decode_strategy(
            self.kv_codec,
            self.flash_attn_decode_paged_xqa,
            enabled=self.use_decode_xqa
            and self.head_size == 256
            and self.num_heads == 6 * self.num_kv_heads,
        )
        ATTENTION_HOOKS.configure_verifier(self)
        decode_scalar_paged_env = _config.raw("VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED")
        self.use_decode_scalar_paged = decode_scalar_paged_env != "0"
        self.compare_bhmd_out_dir = _config.raw("VLLM_FLASH_V100_COMPARE_BHMD_OUT_DIR")
        self.compare_bhmd_out_max_calls = int(
            _config.raw("VLLM_FLASH_V100_COMPARE_BHMD_OUT_MAX_CALLS", "0")
        )
        self._compare_bhmd_out_calls = 0
        self.compare_triton_out_dir = _config.raw(
            "VLLM_FLASH_V100_COMPARE_TRITON_OUT_DIR"
        )
        self.compare_triton_out_max_calls = int(
            _config.raw("VLLM_FLASH_V100_COMPARE_TRITON_OUT_MAX_CALLS", "0")
        )
        self.compare_triton_tensor_dump_dir = _config.raw(
            "VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_DIR"
        )
        self.compare_triton_tensor_dump_max_tokens = int(
            _config.raw("VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_MAX_TOKENS", "64")
        )
        self._compare_triton_out_calls = 0
        self.workspace = _decode.V100Workspace()

        if self.prefix_anchored_decode_window is not None:
            if (
                self.prefix_anchored_decode_window <= 0
                or self.attn_type != AttentionType.DECODER
                or self.kv_codec is not FP16
            ):
                raise ValueError(
                    "prefix-anchored SWA requires a positive window, causal "
                    "decoder attention, and an fp16 KV cache"
                )
            if self.use_triton_prefill:
                raise ValueError(
                    "prefix-anchored SWA cannot use the Triton prefill fallback"
                )
            if (
                not self.use_flash_v100_decode
                or not self.use_decode_scalar_paged
                or not {"anchor_lens", "anchored_window"}
                <= self._flash_decode_paged_kwargs
            ):
                raise RuntimeError(
                    "prefix-anchored SWA requires the masked scalar paged "
                    "decode extension"
                )
            if (
                not self.use_flash_v100_prefill_paged
                or not self._flash_prefill_paged_supports_anchor
            ):
                raise RuntimeError(
                    "prefix-anchored SWA requires the masked paged prefill extension"
                )

            # Select the only two routes that carry the anchored mask once at
            # construction time. The default-off hot path therefore retains
            # its existing route predicates without extra metadata parsing.
            self.smallq_decode_max_query_len = 0
            self.use_decode_paged_prefill = False
            self.use_decode_dense_cache = False
            self.use_decode_dense_reference = False
            self.use_decode_xqa = False
            self.use_smallq_decode_xqa = False
            self.use_flash_v100_prefill_splitkv = False
            self.use_flash_v100_prefill_bfla = False
            self.use_flash_v100_prefill_contig_dense = False
            self.use_flash_v100_prefill_gather_dense = False

        self.config = _config.V100AttnConfig.take_legacy_attributes(vars(self))

    _maybe_compare_bhmd_out = _debug_compare._maybe_compare_bhmd_out

    _reserve_bhmd_compare_call = _debug_compare._reserve_bhmd_compare_call

    _reserve_triton_compare_call = _debug_compare._reserve_triton_compare_call

    _write_bhmd_compare_report = _debug_compare._write_bhmd_compare_report

    _write_triton_compare_report = _debug_compare._write_triton_compare_report

    _maybe_write_triton_tensor_dump = _debug_compare._maybe_write_triton_tensor_dump

    _small_tensor_list = staticmethod(_debug_compare._small_tensor_list)

    _layer_debug_info = staticmethod(_debug_compare._layer_debug_info)

    _tensor_compare_stats = staticmethod(_debug_compare._tensor_compare_stats)

    _prefill_raw_kv_cache_compare_stats = (
        _debug_compare._prefill_raw_kv_cache_compare_stats
    )

    _maybe_compare_triton_output = _debug_compare._maybe_compare_triton_output

    @property
    def kv_codec(self) -> KVCodec | None:
        """Storage codec of this layer's KV cache."""
        return resolve_kv_codec(self.kv_cache_dtype)

    def _xqa_kv_codec(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> KVCodec | None:
        """The codec when the XQA operator reads this cache natively."""
        codec = self.kv_codec
        if codec not in (FP16, FP8_E4M3, FP8_E5M2) or not codec.stores(
            key_cache, value_cache
        ):
            return None
        # Feature contracts can require a separate native FP32 verifier route.
        if ATTENTION_HOOKS.reject_xqa(codec, attn_metadata):
            return None
        return codec

    def _supports_flash_v100_path(self) -> bool:
        """Check whether current layer/config can run Flash V100 safely."""
        supported_kv_dtype = not _routing._uses_fp8_kv_cache(
            self.kv_cache_dtype
        ) or self.kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2")
        return (
            self.use_flash_v100
            and self.attn_type == AttentionType.DECODER
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and supported_kv_dtype
        )

    def _flash_v100_has_sliding_window(self) -> bool:
        sliding_window = self.sliding_window
        if sliding_window is None:
            return False
        return tuple(sliding_window) != (-1, -1)

    def _flash_v100_window_size(self, causal: bool) -> tuple[int, int]:
        if not self._flash_v100_has_sliding_window():
            return (-1, -1)
        left, right = tuple(self.sliding_window)
        left = int(left)
        right = int(right)
        if not causal and left >= 0 and right == 0:
            right = left
        return (left, right)

    _call_flash_attn_decode_paged = _decode._call_flash_attn_decode_paged

    _smallq_decode_xqa_allowed = _verify._smallq_decode_xqa_allowed

    _call_flash_attn_smallq_decode_paged = _verify._call_flash_attn_smallq_decode_paged

    _anchored_swa_params = _decode._anchored_swa_params

    _small_query_decode_enabled = _verify._small_query_decode_enabled

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward path.

        - Prefill: use Flash-V100 by default. Triton prefill is an explicit
          diagnostic fallback only.
        - Decode: use scalar paged Flash-V100 by default, including CUDA graph
          capture/replay, so selecting this backend is not a no-op in
          production decode. Mixed Triton/Flash routes are never silent.
        """

        if attn_metadata is None:
            assert output is not None
            if (
                self.attn_type == AttentionType.DECODER
                and self.sliding_window == (-1, -1)
                and self.alibi_slopes is None
                and not self.logits_soft_cap
                and self.sinks is None
                and abs(self.scale - 0.0625) <= 1.0e-8
            ):
                _dense_prefill._profile_sm70_prefill_workspace(query, self.num_kv_heads)
            _routing._record_route(
                _routing.ROUTE_SPECS["metadata_none_zero_output"].name
            )
            return output.fill_(0)

        ATTENTION_HOOKS.validate_contract(self, layer, attn_metadata)

        if not self._supports_flash_v100_path():
            layer_info = self._layer_debug_info(layer)
            feature_fallback = ATTENTION_HOOKS.fallback_kind(layer_info)
            message = (
                "FLASH_ATTN_V100 cannot run this layer/config because a required "
                "Flash op is unavailable or the attention features/KV cache dtype "
                "are unsupported. Select TRITON_ATTN for a full Triton route, or "
                "set VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1 for explicit "
                "diagnostic fallback. "
                f"Details: layer={layer_info.get('layer_name')!r}, "
                f"flash_ops_available={self.use_flash_v100}, "
                f"attn_type={self.attn_type!r}, "
                f"has_alibi={self.alibi_slopes is not None}, "
                f"logits_soft_cap={self.logits_soft_cap!r}, "
                f"has_sinks={self.sinks is not None}, "
                f"kv_cache_dtype={self.kv_cache_dtype!r}."
            )
            ATTENTION_HOOKS.unsupported(self, layer_info, message, feature_fallback)
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        assert output is not None
        is_prefill = attn_metadata.max_query_len > 1
        is_capturing = _routing._is_cuda_graph_capturing(query)
        layer_name = self._layer_debug_info(layer).get("layer_name")
        if _debug._draft_graph_debug_enabled():
            _debug._draft_graph_debug_log(
                "forward:enter",
                "layer=%s is_prefill=%s is_capturing=%s max_query_len=%s "
                "max_seq_len=%s num_actual_tokens=%s %s %s %s %s %s %s",
                layer_name,
                is_prefill,
                is_capturing,
                int(attn_metadata.max_query_len),
                int(attn_metadata.max_seq_len),
                int(attn_metadata.num_actual_tokens),
                _debug._format_tensor_debug(query, "query"),
                _debug._format_tensor_debug(output, "output"),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "query_start_loc", None),
                    "attn_qsl",
                ),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "seq_lens", None),
                    "attn_seq",
                ),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "block_table", None),
                    "attn_bt",
                ),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "smallq_decode_seq_lens", None),
                    "smallq_seq",
                ),
            )
        _debug._sm70_profile_trace(
            "forward enter layer=%s q_shape=%s k_shape=%s v_shape=%s "
            "kv_shape=%s is_prefill=%s is_capturing=%s max_query_len=%s "
            "max_seq_len=%s num_actual_tokens=%s use_decode_scalar=%s "
            "use_decode_paged_prefill=%s use_prefill_paged=%s "
            "use_triton_prefill=%s",
            layer_name,
            tuple(query.shape),
            tuple(key.shape),
            tuple(value.shape),
            tuple(kv_cache.shape) if hasattr(kv_cache, "shape") else None,
            is_prefill,
            is_capturing,
            int(attn_metadata.max_query_len),
            int(attn_metadata.max_seq_len),
            int(attn_metadata.num_actual_tokens),
            self.use_decode_scalar_paged,
            self.use_decode_paged_prefill,
            self.use_flash_v100_prefill_paged,
            self.use_triton_prefill,
        )

        if is_prefill:
            available_query_tokens = min(
                int(query.shape[0]),
                int(key.shape[0]),
                int(value.shape[0]),
                int(output.shape[0]),
            )
            metadata_live_token_mismatch = (
                _kv_layout._metadata_expects_more_query_tokens_than_available(
                    attn_metadata,
                    available_query_tokens,
                )
            )
            if self.use_triton_prefill:
                if not _state._logged_prefill_triton_safe:
                    logger.info(
                        "FLASH_ATTN_V100 prefill uses explicit Triton diagnostic "
                        "fallback because VLLM_FLASH_V100_PREFILL_USE_TRITON=1; "
                        "this mixed route is not a final performance path."
                    )
                    _state._logged_prefill_triton_safe = True
                _debug._sm70_profile_trace(
                    "forward branch=prefill_triton_safe layer=%s",
                    layer_name,
                )
                self.workspace.decode_cache.invalidate()
                _routing._record_route(_routing.ROUTE_SPECS["prefill_triton_safe"].name)
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            if is_capturing:
                # CUDA graph capture uses dummy metadata whose seq_lens can
                # look like no-prefix prefill, while replayed MTP verification
                # is a uniform small-query decode over an existing KV prefix.
                # Capture the same small-query kernel branch that replay needs.
                capture_prefix = ATTENTION_HOOKS.capture_prefix_kind(
                    layer, attn_metadata
                )
                if capture_prefix:
                    ATTENTION_HOOKS.record_capture_prefix()
                    return self._flash_v100_prefill_with_prefix(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                    )
                smallq_decode = self._small_query_decode_enabled(attn_metadata)
                if smallq_decode:
                    if _debug._draft_graph_debug_enabled():
                        _debug._draft_graph_debug_log(
                            "forward:prefill_capture_smallq",
                            "layer=%s %s %s %s",
                            layer_name,
                            _debug._format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_decode_block_table",
                                    None,
                                ),
                                "smallq_bt",
                            ),
                            _debug._format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_decode_seq_lens",
                                    None,
                                ),
                                "smallq_seq",
                            ),
                            _debug._format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_query_start_loc",
                                    None,
                                ),
                                "smallq_qsl",
                            ),
                        )
                    _debug._sm70_profile_trace(
                        "forward branch=prefill_capture_smallq layer=%s",
                        layer_name,
                    )
                    _routing._record_route(
                        _routing.ROUTE_SPECS["prefill_capture_smallq"].name
                    )
                    ATTENTION_HOOKS.record_capture_layout(attn_metadata)
                    return self._flash_v100_prefill_with_prefix(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                    )
                _debug._sm70_profile_trace(
                    "forward branch=prefill_capture_full_flash layer=%s",
                    layer_name,
                )
            has_prefix_context = (
                metadata_live_token_mismatch
                or _kv_layout._has_prefix_context(attn_metadata)
            )
            smallq_decode = has_prefix_context and self._small_query_decode_enabled(
                attn_metadata
            )
            if has_prefix_context:
                if _debug._draft_graph_debug_enabled():
                    _debug._draft_graph_debug_log(
                        "forward:prefill_prefix",
                        "layer=%s smallq=%s metadata_live_token_mismatch=%s %s %s %s",
                        layer_name,
                        smallq_decode,
                        metadata_live_token_mismatch,
                        _debug._format_tensor_debug(
                            getattr(attn_metadata, "smallq_decode_block_table", None),
                            "smallq_bt",
                        ),
                        _debug._format_tensor_debug(
                            getattr(attn_metadata, "smallq_decode_seq_lens", None),
                            "smallq_seq",
                        ),
                        _debug._format_tensor_debug(
                            getattr(attn_metadata, "smallq_query_start_loc", None),
                            "smallq_qsl",
                        ),
                    )
                _debug._sm70_profile_trace(
                    "forward branch=prefill_prefix layer=%s smallq=%s",
                    layer_name,
                    smallq_decode,
                )
                if not _state._logged_prefill_prefix_flash:
                    if smallq_decode:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via small-query paged decode)."
                        )
                    elif self.use_flash_v100_prefill_paged:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via direct paged prefill kernel)."
                        )
                    else:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via paged-KV gather)."
                        )
                    _state._logged_prefill_prefix_flash = True
                if metadata_live_token_mismatch:
                    logger.info(
                        "FLASH_ATTN_V100 prefill switched to prefix/live-token "
                        "path because layer QKV tokens (%d) are shorter than "
                        "query metadata span.",
                        available_query_tokens,
                    )
                _routing._log_fp8_kv_cache_route(
                    "prefill", self.kv_cache_dtype, "prefix"
                )
                self.workspace.decode_cache.invalidate()
                result = self._flash_v100_prefill_with_prefix(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                )
                self._maybe_compare_triton_output(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                    "prefill_prefix",
                )
                _routing._record_route(
                    _routing.ROUTE_SPECS["prefill_prefix_flash"].name
                )
                return result
            if not _state._logged_prefill_flash:
                logger.info(
                    "FLASH_ATTN_V100 prefill path active (no prefix/chunked context)."
                )
                _state._logged_prefill_flash = True
            self.workspace.decode_cache.invalidate()
            if self.use_prefill_paged_cache and self.use_flash_v100_prefill_paged:
                _debug._sm70_profile_trace(
                    "forward branch=prefill_no_prefix_paged_cache layer=%s",
                    layer_name,
                )
                if not _state._logged_prefill_paged_cache:
                    logger.warning(
                        "FLASH_ATTN_V100 no-prefix prefill is reading paged "
                        "KV cache for strict input-source diagnostics. This "
                        "may be slower than dense raw-KV prefill."
                    )
                    _state._logged_prefill_paged_cache = True
                _routing._log_fp8_kv_cache_route(
                    "prefill", self.kv_cache_dtype, "no_prefix_paged_cache"
                )
                result = self._flash_v100_prefill_with_prefix(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                )
                self._maybe_compare_triton_output(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                    "prefill_no_prefix_paged_cache",
                )
                _routing._record_route(
                    _routing.ROUTE_SPECS["prefill_no_prefix_paged_cache_flash"].name
                )
                return result
            _debug._sm70_profile_trace(
                "forward branch=prefill_no_prefix_dense layer=%s",
                layer_name,
            )
            result = self._flash_v100_prefill(query, key, value, attn_metadata, output)
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "prefill_no_prefix",
            )
            _routing._record_route(
                _routing.ROUTE_SPECS["prefill_no_prefix_dense_flash"].name
            )
            return result

        if not self.use_flash_v100_decode:
            message = (
                "FLASH_ATTN_V100 decode cannot run because the paged decode op "
                "is unavailable. Select TRITON_ATTN for a full Triton route, or "
                "set VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1 for explicit "
                "diagnostic fallback."
            )
            if not self.allow_triton_fallback:
                raise RuntimeError(message)
            if self.use_flash_v100 and not _state._warned_decode_fallback:
                logger.warning("%s", message)
                _state._warned_decode_fallback = True
            _debug._sm70_profile_trace(
                "forward branch=decode_triton_no_flash_decode layer=%s",
                layer_name,
            )
            _routing._record_route(
                _routing.ROUTE_SPECS["decode_triton_no_flash_decode"].name
            )
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if (
            self.use_decode_paged_prefill
            and self.use_flash_v100_prefill_paged
            and not is_capturing
        ):
            _routing._log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "decode_as_paged_prefill"
            )
            _debug._sm70_profile_trace(
                "forward branch=decode_paged_prefill layer=%s",
                layer_name,
            )
            result = self._flash_v100_decode_as_paged_prefill(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "decode_paged_prefill",
            )
            _routing._record_route(_routing.ROUTE_SPECS["decode_paged_prefill"].name)
            return result
        if self.use_decode_dense_cache and not is_capturing:
            _routing._log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "dense_cache_bridge"
            )
            _debug._sm70_profile_trace(
                "forward branch=decode_dense_cache layer=%s",
                layer_name,
            )
            result = self._flash_v100_decode_dense_cache(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
            )
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "decode_dense_cache",
            )
            _routing._record_route(_routing.ROUTE_SPECS["decode_dense_cache"].name)
            return result
        if self.use_decode_dense_reference and not is_capturing:
            _routing._log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "dense_reference_bridge"
            )
            _debug._sm70_profile_trace(
                "forward branch=decode_dense_reference layer=%s",
                layer_name,
            )
            result = self._flash_v100_decode_dense_reference(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "decode_dense_reference",
            )
            _routing._record_route(_routing.ROUTE_SPECS["decode_dense_reference"].name)
            return result
        if not self.use_decode_scalar_paged:
            message = (
                "FLASH_ATTN_V100 decode has no enabled Flash route: scalar "
                "paged decode is disabled and the strict paged-prefill bridge "
                "is unavailable or CUDA graph capture is active. Re-enable "
                "VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED=1, select TRITON_ATTN "
                "for a full Triton route, or set "
                "VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1 for explicit "
                "diagnostic fallback."
            )
            if not self.allow_triton_fallback:
                raise RuntimeError(message)
            if not _state._warned_decode_strict_fallback:
                logger.warning("%s", message)
                _state._warned_decode_strict_fallback = True
            _debug._sm70_profile_trace(
                "forward branch=decode_triton_scalar_disabled layer=%s",
                layer_name,
            )
            _routing._record_route(
                _routing.ROUTE_SPECS["decode_triton_scalar_disabled"].name
            )
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if not _state._logged_decode_flash:
            logger.info(
                "FLASH_ATTN_V100 decode path active (paged KV, "
                "CUDA-graph safe; selected route is reported separately)."
            )
            _state._logged_decode_flash = True
        if _debug._draft_graph_debug_enabled():
            _debug._draft_graph_debug_log(
                "forward:decode",
                "layer=%s %s %s %s",
                layer_name,
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "query_start_loc", None),
                    "attn_qsl",
                ),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "seq_lens", None),
                    "attn_seq",
                ),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "block_table", None),
                    "attn_bt",
                ),
            )
        _debug._sm70_profile_trace(
            "forward branch=decode_scalar_paged layer=%s",
            layer_name,
        )
        result = self._flash_v100_decode(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )
        self._maybe_compare_triton_output(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
            "decode_scalar_paged",
        )
        return result

    _flash_v100_decode_as_paged_prefill = _decode._flash_v100_decode_as_paged_prefill

    _flash_v100_decode_dense_cache = _decode._flash_v100_decode_dense_cache

    _flash_v100_decode_dense_reference = _decode._flash_v100_decode_dense_reference

    _flash_v100_prefill = _prefill._flash_v100_prefill

    _flash_v100_decode = _decode._flash_v100_decode

    _flash_v100_small_query_prefill_as_decode = (
        _verify._flash_v100_small_query_prefill_as_decode
    )

    _should_use_fp8_prefill_bridge = _prefill._should_use_fp8_prefill_bridge

    _run_fp8_prefill_bridge = _prefill._run_fp8_prefill_bridge

    _should_use_prefill_splitkv = _prefill._should_use_prefill_splitkv

    _should_use_prefill_bfla = _prefill._should_use_prefill_bfla

    _should_use_prefill_contig_dense = _prefill._should_use_prefill_contig_dense

    _should_use_prefill_gather_dense = _prefill._should_use_prefill_gather_dense

    _prefill_prefix_decode_rows_allowed = _prefill._prefill_prefix_decode_rows_allowed

    _run_mixed_rows_grouped_e4m3 = _prefill._run_mixed_rows_grouped_e4m3

    _run_prefill_prefix_decode_rows = _prefill._run_prefill_prefix_decode_rows

    _run_prefill_paged_call = _prefill._run_prefill_paged_call

    _flash_v100_prefill_with_prefix = _prefill._flash_v100_prefill_with_prefix


# Preserve the original __class__ cell semantics of the extracted super call.
_super_owner = FlashAttnV100Impl
