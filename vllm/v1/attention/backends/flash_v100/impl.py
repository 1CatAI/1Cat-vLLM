# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 attention implementation (forward and route bodies)."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Any, cast

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100 import debug as _debug
from vllm.v1.attention.backends.flash_v100 import debug_compare as _debug_compare
from vllm.v1.attention.backends.flash_v100 import decode as _decode
from vllm.v1.attention.backends.flash_v100 import dense_prefill as _dense_prefill
from vllm.v1.attention.backends.flash_v100 import ops as _ops
from vllm.v1.attention.backends.flash_v100 import prefill as _prefill
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100 import verify as _verify
from vllm.v1.attention.backends.flash_v100.spec import attention as _feature
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


class FlashAttnV100Impl(TritonAttentionImpl):
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

    @property
    def spec_attention(self) -> _feature.SpecAttentionState:
        attributes = vars(self)
        if "_spec_attention" not in attributes:
            attributes["_spec_attention"] = _feature.SpecAttentionState(
                _ops._callable_accepts_keyword
            )
        return attributes["_spec_attention"]

    @property
    def comparison_state(self):
        attributes = vars(self)
        if "_comparison_state" not in attributes:
            state = _debug_compare.ComparisonState()
            for name in _debug_compare.COUNTER_FIELDS:
                if name in attributes:
                    setattr(state, name, attributes.pop(name))
            attributes["_comparison_state"] = state
        return attributes["_comparison_state"]

    def __getattr__(self, name: str) -> Any:
        if name in _feature.POLICY_FIELDS:
            return getattr(self.spec_attention, name)
        if name in _debug_compare.COUNTER_FIELDS:
            return getattr(self.comparison_state, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _feature.POLICY_FIELDS:
            setattr(self.spec_attention, name, value)
        elif name in _debug_compare.COUNTER_FIELDS:
            setattr(self.comparison_state, name, value)
        else:
            super().__setattr__(name, value)

    def _policy(self):
        policy = getattr(self, "config", None)
        if policy is None:
            attributes = vars(self)
            policy = SimpleNamespace(
                **{
                    field.name: attributes[field.name]
                    for field in fields(_config.V100AttnConfig)
                    if field.name in attributes
                }
            )
        return policy

    def _contract_validator(self):
        return getattr(self, _feature.VALIDATION_METHOD)

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
        self.spec_attention.initialize_scalar_tail(use_e4m3_fp32)
        if use_e4m3_fp32 and self.flash_attn_grouped_e4m3_fp32_paged is None:
            logger.warning_once(
                "E4M3 grouped FP32 requires Flash-V100 precision revision 4; "
                "the E4M3 scalar fallback also requires this revision for "
                "FP32 partial storage. Rebuild the extension and restart workers.",
                scope="process",
            )
        self.spec_attention.initialize_verify_abi(
            _ops._flash_attn_grouped_verify_max_query_tokens,
            _ops._flash_attn_grouped_verify_request_major_abi_version,
        )
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
        self.flash_attn_prefill_paged = self.spec_attention.configure_prefill(
            self.flash_attn_prefill_paged
        )
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
        self.spec_attention.configure_verifier(self.flash_attn_grouped_verify_paged)
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

    _small_tensor_list = staticmethod(_debug_compare._small_tensor_list)

    _layer_debug_info = staticmethod(_debug_compare._layer_debug_info)

    _tensor_compare_stats = staticmethod(_debug_compare._tensor_compare_stats)

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
        if _feature.reject_xqa(codec, attn_metadata):
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

    def _new_decode_executor(self) -> _decode.DecodeExecutor:
        policy = self._policy()
        config = _decode.DecodeConfig(
            cast(_config.V100AttnConfig, policy),
            getattr(self, "scale", 1.0),
            self.kv_cache_dtype,
            getattr(self, "attn_type", AttentionType.DECODER),
            getattr(self, "sliding_window", None),
        )
        ops = _decode.DecodeOps(
            dense=getattr(self, "flash_attn_func", None),
            paged=getattr(self, "flash_attn_decode_paged", None),
            xqa=getattr(self, "flash_attn_decode_paged_xqa", None),
            wmma=getattr(self, "flash_attn_decode_paged_wmma", None),
            prefill=getattr(self, "flash_attn_prefill_paged", None),
            prefill_bhmd=getattr(self, "flash_attn_prefill_paged_bhmd", None),
            paged_keywords=getattr(self, "_flash_decode_paged_kwargs", set()),
            scalar_tail=getattr(self, "_sm70_scalar_tail_attention", None),
            reject_xqa=_feature.reject_xqa,
            reserve_bhmd_compare=self._reserve_bhmd_compare_call,
            write_bhmd_compare=self._write_bhmd_compare_report,
            compare_bhmd=self._maybe_compare_bhmd_out,
            compare_triton=self._maybe_compare_triton_output,
            triton_forward=super().forward,
            profile_trace=_debug._sm70_profile_trace,
            draft_debug_enabled=_debug._draft_graph_debug_enabled,
            draft_debug_log=_debug._draft_graph_debug_log,
            format_debug=_debug._format_tensor_debug,
            scalar_override=vars(self).get("_call_flash_attn_decode_paged"),
        )
        workspace = getattr(self, "workspace", None)
        if workspace is None:
            workspace = _decode.V100Workspace()
        return _decode.DecodeExecutor(config, ops, workspace)

    def _call_flash_attn_decode_paged(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        softmax_scale: float,
        out: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        window_size: tuple[int, int] = (-1, -1),
        max_seq_len_hint: int | None = None,
        workspace_seq_capacity_hint: int | None = None,
        active_num_partitions: int | None = None,
        partition_size_hint: int | None = None,
        anchor_lens: torch.Tensor | None = None,
        anchored_window: int = 0,
    ) -> None:
        return self._new_decode_executor()._call_flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            out=out,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            window_size=window_size,
            max_seq_len_hint=max_seq_len_hint,
            workspace_seq_capacity_hint=workspace_seq_capacity_hint,
            active_num_partitions=active_num_partitions,
            partition_size_hint=partition_size_hint,
            anchor_lens=anchor_lens,
            anchored_window=anchored_window,
        )

    def _new_comparison_executor(self):
        return _debug_compare.ComparisonExecutor(
            self._policy(),
            getattr(self, "scale", 1.0),
            getattr(self, "kv_cache_dtype", "auto"),
            _debug_compare.ComparisonOps(
                triton_forward=super().forward,
                paged_bhmd=getattr(self, "flash_attn_prefill_paged_bhmd", None),
            ),
            self.comparison_state,
            {
                name: vars(self)[name]
                for name in (
                    *_debug_compare.LEGACY_METHODS,
                    *_debug_compare.STATIC_METHODS,
                )
                if name in vars(self)
            },
        )

    def _new_prefill_executor(self):
        policy = self._policy()
        settings = _prefill.PrefillConfig(
            cast(_config.V100AttnConfig, policy),
            getattr(self, "scale", 1.0),
            getattr(self, "kv_cache_dtype", "auto"),
        )
        native: dict[str, Any] = {
            field: getattr(self, legacy_name, None)
            for legacy_name, field in _prefill.OPERATOR_FIELDS.items()
            if field != "supports_anchor"
        }
        callbacks: dict[str, Any] = {
            field: getattr(self, legacy_name, None)
            for field, legacy_name in _feature.PREFILL_CALLBACK_FIELDS.items()
        }
        ops = _prefill.PrefillDriverOps(
            **native,
            supports_anchor=getattr(
                self, "_flash_prefill_paged_supports_anchor", False
            ),
            **_feature.prefill_dependencies(self.spec_attention),
            **callbacks,
            triton_forward=super().forward,
            compare_triton=self._maybe_compare_triton_output,
            small_query_enabled=self._small_query_decode_enabled,
            capture_prefix_kind=_feature.capture_prefix_kind,
            record_capture_prefix=_feature.record_capture_prefix,
            record_capture_layout=_feature.record_capture_layout,
        )
        return _prefill.PrefillExecutor(
            settings,
            ops,
            getattr(self, "workspace", None) or _decode.V100Workspace(),
            {
                name: vars(self)[name]
                for name in _prefill.LEGACY_METHODS
                if name in vars(self)
            },
        )

    def _new_verification_executor(self) -> _verify.VerificationExecutor:
        policy = self._policy()
        limits: dict[str, Any] = {
            name: getattr(self, legacy_name, default)
            for name, (
                legacy_name,
                default,
            ) in _feature.VERIFICATION_CONFIG_FIELDS.items()
        }
        config = _verify.VerificationConfig(
            policy=cast(_config.V100AttnConfig, policy),
            scale=getattr(self, "scale", 1.0),
            kv_cache_dtype=getattr(self, "kv_cache_dtype", "auto"),
            alibi_slopes=getattr(self, "alibi_slopes", None),
            logits_soft_cap=getattr(self, "logits_soft_cap", 0.0),
            **limits,
        )
        ops = _verify.VerificationOps(
            grouped=getattr(self, "flash_attn_grouped_verify_paged", None),
            fp16_grouped=getattr(self, "flash_attn_grouped_fp16_fp32_paged", None),
            e4m3_grouped=getattr(self, "flash_attn_grouped_e4m3_fp32_paged", None),
            xqa=getattr(self, "flash_attn_decode_paged_xqa", None),
            window_size=getattr(self, "_flash_v100_window_size", None),
            layer_info=getattr(self, "_layer_debug_info", None),
            xqa_codec=getattr(self, "_xqa_kv_codec", None),
            decode=getattr(self, "_call_flash_attn_decode_paged", None),
            **{
                name: vars(self).get(legacy_name)
                for name, legacy_name in _feature.VERIFICATION_OVERRIDES.items()
            },
        )
        return _verify.VerificationExecutor(config, ops)

    def _anchored_swa_params(
        self, attn_metadata: TritonAttentionMetadata
    ) -> tuple[torch.Tensor | None, int]:
        return self._new_decode_executor()._anchored_swa_params(attn_metadata)

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

        _feature.validate_contract(self._contract_validator(), layer, attn_metadata)

        if not self._supports_flash_v100_path():
            layer_info = self._layer_debug_info(layer)
            feature_fallback = _feature.fallback_kind(layer_info)
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
            _feature.unsupported(self._policy(), layer_info, message, feature_fallback)
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
            return self._forward_prefill(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                is_capturing,
                layer_name,
            )

        return self._forward_decode(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
            is_capturing,
            layer_name,
        )

    def _forward_prefill(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
        is_capturing: bool,
        layer_name: object,
    ) -> torch.Tensor:
        return self._new_prefill_executor().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
            is_capturing,
            layer_name,
        )

    def _forward_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
        is_capturing: bool,
        layer_name: object,
    ) -> torch.Tensor:
        return self._new_decode_executor().forward(
            _decode.DecodeRequest(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                is_capturing,
                layer_name,
            )
        )

    def _flash_v100_decode_as_paged_prefill(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._new_decode_executor()._flash_v100_decode_as_paged_prefill(
            layer, query, kv_cache, attn_metadata, output
        )

    def _flash_v100_decode_dense_cache(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._new_decode_executor()._flash_v100_decode_dense_cache(
            layer, query, key, value, kv_cache, attn_metadata, output
        )

    def _flash_v100_decode_dense_reference(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._new_decode_executor()._flash_v100_decode_dense_reference(
            layer, query, kv_cache, attn_metadata, output
        )

    def _flash_v100_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._new_decode_executor()._flash_v100_decode(
            layer, query, key, value, kv_cache, attn_metadata, output
        )


# Preserve the original __class__ cell semantics of the extracted super call.
_super_owner = FlashAttnV100Impl


# External method compatibility lives at the assembly boundary, not in an
# executor receiving a backend. Instance-level overrides retain Python binding.
def _verification_method(method):
    if method == "validate_contract":

        def validate(instance, layer, metadata):
            return _verify.validate_contract(
                layer, metadata, instance._flash_v100_window_size
            )

        return validate

    def call(instance, *args, **kwargs):
        return getattr(instance._new_verification_executor(), method)(*args, **kwargs)

    return call


for _legacy_name, _method in _feature.VERIFICATION_METHODS.items():
    _compatibility_method = _verification_method(_method)
    setattr(FlashAttnV100Impl, _legacy_name, _compatibility_method)
    globals()[_legacy_name] = _compatibility_method


def _prefill_method(method):
    def call(instance, *args, **kwargs):
        return getattr(instance._new_prefill_executor(), method)(*args, **kwargs)

    return call


for _legacy_name in _prefill.LEGACY_METHODS:
    _compatibility_method = _prefill_method(_legacy_name)
    setattr(FlashAttnV100Impl, _legacy_name, _compatibility_method)


def _comparison_method(method):
    def call(instance, *args, **kwargs):
        return getattr(instance._new_comparison_executor(), method)(*args, **kwargs)

    return call


for _legacy_name in _debug_compare.LEGACY_METHODS:
    _compatibility_method = _comparison_method(_legacy_name)
    setattr(FlashAttnV100Impl, _legacy_name, _compatibility_method)
