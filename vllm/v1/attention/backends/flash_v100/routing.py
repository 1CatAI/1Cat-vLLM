# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route accounting, tracing and decode partition/XQA admission policy."""

from __future__ import annotations

import atexit
import json
import os
from dataclasses import dataclass
from typing import Literal

import regex as re
import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import CUDAGRAPH_VARIANT_LONG_CONTEXT
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)
from vllm.v1.attention.kv_codecs import (
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    KVCodec,
    canonical_kv_cache_dtype,
    resolve_kv_codec,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")

RouteStage = Literal["decode", "verify", "mixed_decode", "prefill", "metadata", "any"]
_NATIVE_CODECS = frozenset((FP16, FP8_E4M3, FP8_E5M2))
_ALL_CODECS = _NATIVE_CODECS | {BF16}


@dataclass(frozen=True)
class RouteShape:
    """Host-visible dimensions only; never read device sequence lengths."""

    rows: int
    heads_q: int
    heads_kv: int
    head_dim: int
    page_size: int = 0
    chunk_size: int = 0

    @property
    def gqa(self) -> int:
        if self.heads_kv <= 0 or self.heads_q % self.heads_kv:
            return 0
        return self.heads_q // self.heads_kv


@dataclass(frozen=True)
class RouteSpec:
    """Structural admission and the unchanged compatibility counter name.

    Operator availability, metadata/stride/ABI guards and legacy policy still
    have to admit a candidate. A declaration alone never claims a native op
    exists. Bridge declarations describe their input storage codec; dense
    declarations describe the already-converted tensors consumed by the op.
    """

    name: str
    stage: RouteStage
    codecs: frozenset[KVCodec] = _NATIVE_CODECS
    head_dims: tuple[int, ...] = ()
    gqa_ratios: tuple[int, ...] = ()
    page_alignment: int = 1
    page_sizes: tuple[int, ...] = ()
    chunk_alignment: int = 1
    min_chunk: int = 0
    max_chunk: int | None = None
    chunk_sizes: tuple[int, ...] = ()
    fallback: bool = False
    observer: bool = False
    xqa: bool = False

    def shape_reason(self, codec: KVCodec | None, shape: RouteShape) -> str | None:
        if codec not in self.codecs:
            return "codec"
        if self.head_dims and shape.head_dim not in self.head_dims:
            return "head_dim"
        if self.gqa_ratios and shape.gqa not in self.gqa_ratios:
            return "gqa"
        if self.page_alignment > 1 and (
            shape.page_size <= 0 or shape.page_size % self.page_alignment
        ):
            return "page_alignment"
        if self.page_sizes and shape.page_size not in self.page_sizes:
            return "page_size"
        if shape.chunk_size < self.min_chunk:
            return "min_chunk"
        if self.max_chunk is not None and shape.chunk_size > self.max_chunk:
            return "max_chunk"
        if shape.chunk_size % self.chunk_alignment:
            return "chunk_alignment"
        if self.chunk_sizes and shape.chunk_size not in self.chunk_sizes:
            return "chunk_size"
        return None


# All 44 literal counters, plus concrete names from the old dynamic sites.
# Format-bearing counter names are retained until a separate compatibility
# change; implementation admission is expressed by codecs instead.
ROUTE_SPECS: dict[str, RouteSpec] = {}


def _declare(names: str, stage: RouteStage, **kwargs) -> None:
    for name in names.split():
        if name in ROUTE_SPECS:
            raise ValueError(f"Duplicate Flash-V100 route: {name}")
        ROUTE_SPECS[name] = RouteSpec(name, stage, **kwargs)


_declare("decode_xqa_paged", "decode", head_dims=(256,), gqa_ratios=(4, 6, 8), xqa=True)
_declare(
    "prefill_prefix_decode_rows_xqa",
    "mixed_decode",
    head_dims=(256,),
    gqa_ratios=(4, 6, 8),
    xqa=True,
)
_declare(
    "prefill_smallq_decode_xqa", "verify", head_dims=(256,), gqa_ratios=(6, 8), xqa=True
)
_declare("decode_scalar_paged", "decode", fallback=True)
_declare("prefill_smallq_decode_scalar", "verify", fallback=True)
_declare("prefill_prefix_decode_rows_scalar", "mixed_decode", fallback=True)
_declare("decode_dense_cache decode_paged_prefill", "decode")
_declare("decode_dense_reference", "decode", codecs=_ALL_CODECS)
_declare(
    "decode_e4m3_compact_scalar_tail",
    "decode",
    codecs=frozenset((FP8_E4M3,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    page_alignment=16,
)
_declare(
    "decode_triton_no_flash_decode decode_triton_scalar_disabled",
    "decode",
    codecs=_ALL_CODECS,
    fallback=True,
)
_declare(
    "unsupported_triton_fallback dflash_draft_triton_fallback",
    "any",
    codecs=_ALL_CODECS,
    fallback=True,
)
_declare("prefill_triton_safe", "prefill", codecs=_ALL_CODECS, fallback=True)
_declare("prefill_ddtree_triton", "verify", codecs=_ALL_CODECS)
_declare(
    "metadata_none_zero_output prefill_capture_dflash_noncausal_paged "
    "prefill_capture_smallq prefill_capture_smallq_ddtree_metadata "
    "prefill_capture_smallq_no_ddtree_metadata",
    "metadata",
    codecs=_ALL_CODECS,
    observer=True,
)
_declare(
    "prefill_smallq_fp16_grouped_fp32",
    "verify",
    codecs=frozenset((FP16,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    page_alignment=16,
)
_declare(
    "prefill_smallq_e4m3_grouped_fp32",
    "verify",
    codecs=frozenset((FP8_E4M3,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    page_alignment=16,
)
_declare(
    "prefill_prefix_decode_rows_e4m3_grouped_fp32",
    "mixed_decode",
    codecs=frozenset((FP8_E4M3,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    page_alignment=16,
)
_declare("prefill_smallq_dflash2_grouped_verify prefill_ddtree_dense", "verify")
_declare(
    "prefill_dense_splitd_d256 prefill_prefix_splitd_d256 "
    "prefill_prefix_paged_splitd_d256 prefill_prefix_gather_splitd_d256 "
    "prefill_prefix_contig_splitd_d256",
    "prefill",
    codecs=frozenset((FP16,)),
    head_dims=(256,),
)
_declare(
    "prefill_dense_splitd_d256_splitkv3_kernel",
    "prefill",
    codecs=frozenset((FP16,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    chunk_sizes=(4096, 8000),
)
_declare(
    "prefill_dense_d256_gqa_arch_long",
    "prefill",
    codecs=frozenset((FP16,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    min_chunk=64,
    max_chunk=8192,
)
_declare(
    "prefill_dense_d256_gqa_v37",
    "prefill",
    codecs=frozenset((FP16,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    chunk_alignment=64,
    min_chunk=64,
    max_chunk=8192,
)
_declare(
    "prefill_dense_d256_gqa_79t_fp32",
    "prefill",
    codecs=frozenset((FP16,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    min_chunk=8000,
    max_chunk=8192,
)
_declare(
    "prefill_dense_d256_gqa_79t_fp32_fringe_fallback",
    "prefill",
    codecs=frozenset((FP16,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    min_chunk=8001,
    max_chunk=8192,
)
_declare(
    "prefill_dense_d256_gqa_79t_fp32_q8192",
    "prefill",
    codecs=frozenset((FP16,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    min_chunk=8001,
    max_chunk=8192,
)
_declare(
    "prefill_dense_d256_gqa_79t_fp32_q8192_pad",
    "prefill",
    codecs=frozenset((FP16,)),
    head_dims=(256,),
    gqa_ratios=(6,),
    min_chunk=8001,
    max_chunk=8191,
)
_declare(
    "prefill_prefix_fp8_bridge_exact_d256 "
    "prefill_prefix_fp8_bridge_exact_d256_tailpad "
    "prefill_prefix_fp8_bridge_exact_dense_d256 "
    "prefill_prefix_fp8_bridge_exact_dense_d256_tailpad",
    "prefill",
    codecs=frozenset((FP8_E4M3, FP8_E5M2)),
    head_dims=(256,),
)
_declare("prefill_prefix_fp8_e4m3_bridge", "prefill", codecs=frozenset((FP8_E4M3,)))
_declare(
    "decode_strategy_legacy_revision",
    "decode",
    codecs=frozenset((FP8_E4M3,)),
    fallback=True,
    observer=True,
)
_declare("prefill_prefix_fp8_e5m2_bridge", "prefill", codecs=frozenset((FP8_E5M2,)))
_declare(
    "prefill_no_prefix_dense_flash prefill_no_prefix_paged_cache_flash "
    "prefill_prefix_bfla prefill_prefix_contig_dense "
    "prefill_prefix_contig_dense_bhmd prefill_prefix_contig_dense_fa2_d256 "
    "prefill_prefix_dflash_noncausal_batch prefill_prefix_flash "
    "prefill_prefix_paged_anchored prefill_prefix_splitkv",
    "prefill",
)

# The sibling FlashInfer backend uses this shared accounting owner too.
_declare(
    "flashinfer_sm70_fixed_entry flashinfer_sm70_splitkv3_fast_visible",
    "prefill",
    codecs=frozenset((FP16,)),
    head_dims=(256,),
    page_alignment=16,
    page_sizes=(784,),
    observer=True,
)

# Diagnostic families are observers, not additional dispatch implementations.
_ROUTE_FAMILIES = (
    (
        re.compile(r"decode_xqa_e4m3_dynamic_page[0-9]+"),
        RouteSpec(
            "decode_xqa_e4m3_dynamic_page{page}",
            "decode",
            frozenset((FP8_E4M3,)),
            observer=True,
        ),
    ),
    (
        re.compile(r"decode_xqa_p[0-9]+_page[0-9]+"),
        RouteSpec("decode_xqa_p{partition}_page{page}", "decode", observer=True),
    ),
    (
        re.compile(r"fp8_kv_decode(?:_.+)?"),
        RouteSpec(
            "fp8_kv_decode", "decode", frozenset((FP8_E4M3, FP8_E5M2)), observer=True
        ),
    ),
    (
        re.compile(r"fp8_kv_prefill(?:_.+)?"),
        RouteSpec(
            "fp8_kv_prefill", "prefill", frozenset((FP8_E4M3, FP8_E5M2)), observer=True
        ),
    ),
)


def route_spec(name: str) -> RouteSpec:
    """Resolve literal and dynamic compatibility counters; reject typos."""
    spec = ROUTE_SPECS.get(name)
    if spec is not None:
        return spec
    for pattern, spec in _ROUTE_FAMILIES:
        if pattern.fullmatch(name):
            return spec
    raise ValueError(f"Undeclared Flash-V100 route: {name}")


@dataclass(frozen=True)
class RouteContext:
    stage: RouteStage
    codec: KVCodec | None
    shape: RouteShape
    enabled: bool = True
    available: bool = True
    query: torch.Tensor | None = None
    metadata: TritonAttentionMetadata | None = None
    seq_rows: int | None = None
    max_seq_len_hint: int | None = None
    workspace_seq_capacity_hint: int | None = None
    partition_size_hint: int | None = None
    window_size: tuple[int, int] = (-1, -1)


def _xqa_reason(spec: RouteSpec, context: RouteContext) -> str | None:
    """One codec/shape decision for uniform, mixed and small-query decode."""
    shape = context.shape
    smallq = context.stage == "verify"
    if smallq and (
        context.partition_size_hint is not None or context.window_size != (-1, -1)
    ):
        return "partition_or_window"
    if context.seq_rows is not None and shape.rows != context.seq_rows:
        return "sequence_rows"
    if smallq and shape.gqa not in (6, 8):
        return "gqa"
    if not smallq and shape.gqa == 4:
        if context.stage == "decode":
            assert context.metadata is not None
            if not _decode_xqa_allowed_for_q_per_kv(shape.gqa, context.metadata):
                return "short_gqa4"
        elif int(context.max_seq_len_hint or 0) < _decode_xqa_q4_min_seq_len():
            return "short_gqa4"
    if context.codec is FP8_E4M3:
        if shape.gqa != 6:
            return "e4m3_gqa"
        # The legacy small-Q guard accepts zero rows, whereas uniform and
        # mixed decode require one row or an explicitly enabled batch.
        if (shape.rows > 1 if smallq else shape.rows != 1) and not (
            shape.rows > 1 and _config.registered("VLLM_FLASH_V100_E4M3_BATCH_XQA")
        ):
            return "e4m3_batch"
    if smallq:
        assert context.query is not None
        capture = bool(getattr(context.metadata, "flash_v100_cudagraph_capture", False))
        capture = capture or _is_cuda_graph_capturing(context.query)
        hint = max(
            int(context.max_seq_len_hint or 0),
            int(context.workspace_seq_capacity_hint or 0) if capture else 0,
        )
        minimum = int(
            _config.raw("VLLM_FLASH_V100_SMALLQ_DECODE_XQA_MIN_SEQ_LEN", "4096")
        )
        if context.codec is FP8_E5M2:
            minimum = max(minimum, _decode_fp8_xqa_min_seq_len())
        if hint < max(1, minimum):
            return "short_verify"
    elif context.codec is FP8_E5M2:
        if shape.gqa == 4:
            return "e5m2_gqa4"
        if context.stage == "decode":
            assert context.metadata is not None and context.query is not None
            if not _decode_fp8_xqa_allowed(context.metadata, context.query):
                return "short_e5m2"
        elif int(context.max_seq_len_hint or 0) < _decode_fp8_xqa_min_seq_len():
            return "short_e5m2"
    if context.stage == "decode" and context.window_size != (-1, -1):
        return "window"
    return None


def route_reason(spec: RouteSpec, context: RouteContext) -> str | None:
    if spec.stage not in (context.stage, "any"):
        return "stage"
    if not context.enabled:
        return "disabled"
    if not context.available:
        return "unavailable"
    reason = spec.shape_reason(context.codec, context.shape)
    if reason is not None:
        return reason
    return _xqa_reason(spec, context) if spec.xqa else None


def select_route(
    context: RouteContext, candidates: tuple[str, ...], *, fallback: str | None = None
) -> RouteSpec | None:
    """Select in declared priority order; caller supplies operator availability.

    A missing optional route can return None so another native implementation
    can be tried. Final dispatch supplies an explicit fallback counter.
    """
    for name in candidates:
        spec = route_spec(name)
        if route_reason(spec, context) is None:
            return spec
    if fallback is None:
        return None
    spec = route_spec(fallback)
    if not spec.fallback:
        raise ValueError(f"Route is not a declared fallback: {fallback}")
    return spec


def _batch_context_routing_for_graph_variant(
    routing_enabled: bool,
    graph_variant: int | None,
) -> bool:
    if not routing_enabled:
        return False
    if graph_variant is None:
        # Eager execution can route directly from the live batch and context.
        return True
    return graph_variant == CUDAGRAPH_VARIANT_LONG_CONTEXT


def _batch_context_routing_cache_dtype_supported(cache_dtype: str | None) -> bool:
    """Admit the exact FP8 XQA formats implemented by Flash-V100."""
    codec = resolve_kv_codec(cache_dtype)
    return codec is FP8_E5M2 or (
        codec is FP8_E4M3 and _config.registered("VLLM_FLASH_V100_E4M3_BATCH_XQA")
    )


_logged_fp8_kv_prefill = False
_logged_fp8_kv_decode = False
_logged_kv_dtype_contracts: set[str] = set()
_route_summary_registered = False
_route_counts: dict[str, int] = {}
_fallback_counts: dict[str, int] = {}
_decode_active_trace_signatures: set[tuple[object, ...]] = set()
_DEFAULT_DECODE_PARTITION_SIZE = 256
_VALID_DECODE_PARTITION_SIZES = (256, 512, 1024)
_DEFAULT_Q4_XQA_MIN_SEQ_LEN = 32768
_DEFAULT_FP8_XQA_MIN_SEQ_LEN = 16384


def _normalize_flash_v100_kv_cache_dtype(kv_cache_dtype: str) -> str:
    # One spelling per codec: "float16" is the extension's "auto" layout and
    # the `fp8` shorthand is E4M3, so every route sees the same format name.
    return canonical_kv_cache_dtype(kv_cache_dtype)


def _decode_dynamic_partitions_enabled() -> bool:
    return _config.raw("VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS", "1") != "0"


def _decode_partition_size_for_metadata(
    max_seq_len_hint: int | None = None,
) -> int:
    raw = _config.raw("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
    if raw is None:
        return _select_default_decode_partition_size(max_seq_len_hint)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_FLASH_V100_DECODE_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {raw!r}"
        ) from exc
    if value not in _VALID_DECODE_PARTITION_SIZES:
        raise ValueError(
            "VLLM_FLASH_V100_DECODE_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {value}"
        )
    return value


def resolve_decode_strategy(
    codec: KVCodec | None, operator, *, enabled: bool
) -> Literal["shared", "legacy"]:
    """Capture the requested/effective policy once, before graph construction."""
    cfg = get_current_vllm_config_or_none()
    requested = cfg.kernel_config.sm70_decode_strategy if cfg is not None else "shared"
    if requested == "legacy":
        return "legacy"
    if codec is not FP8_E4M3:
        return "shared"
    if not enabled or operator is None:
        return "legacy"
    revision = getattr(operator, "shared_decode_strategy_revision", 0)
    if isinstance(revision, int) and revision >= 1:
        return "shared"
    _record_route(ROUTE_SPECS["decode_strategy_legacy_revision"].name)
    return "legacy"


def _g6_aligned_page_partition_size_hint(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_cache_dtype: str,
    *,
    strategy: Literal["shared", "legacy"] = "legacy",
) -> int | None:
    if _config.raw("VLLM_FLASH_V100_DECODE_PARTITION_SIZE") is not None:
        return None
    if _config.raw("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", "1") == "0":
        return None
    if not (
        query.ndim == 3
        and query.shape[0] == 1
        and query.shape[2] == 256
        and key_cache.ndim == 4
        and key_cache.shape[1] >= _DEFAULT_DECODE_PARTITION_SIZE
        and key_cache.shape[1] % 16 == 0
        and key_cache.shape[2] > 0
        and key_cache.shape[3] == 256
        and query.shape[1] == 6 * key_cache.shape[2]
        and value_cache.shape == key_cache.shape
        and value_cache.dtype == key_cache.dtype
    ):
        return None
    codec = resolve_kv_codec(kv_cache_dtype)
    if codec is None or not codec.stores(key_cache, value_cache):
        return None
    if codec is FP16 or (codec is FP8_E4M3 and strategy == "shared"):
        # Share shape planning, while retaining each codec's actual storage and
        # partial precision. The native launch/reducer accepts PARTIAL_T;
        # E4M3 must keep its mandatory FP32 workspace.
        # The exact FP16 page-784 graph contains p256 and p1024 nodes and
        # selects between them from device seq_lens. Plan the p256 workspace
        # envelope once.
        return 256 if key_cache.shape[1] == 784 else None
    if codec is FP8_E4M3:
        # Plan a p64 workspace envelope. The native G6 path keeps this captured
        # shape while selecting p64/p256 and long wave partitions from device
        # sequence lengths.
        return 64
    if codec is FP8_E5M2:
        # Plan the largest p256 workspace once. The extension selects p256 or
        # p1024 from device seq_lens, so CUDA graph replay keeps one
        # captured shape while short and long contexts use different kernels.
        # Keep this layout-driven rather than using model-name allowlists.
        return 256
    return None


def _log_kv_dtype_contract(kv_cache_dtype: str) -> None:
    if kv_cache_dtype in _logged_kv_dtype_contracts:
        return
    _logged_kv_dtype_contracts.add(kv_cache_dtype)
    if kv_cache_dtype == "fp8":
        logger.warning(
            "SM70 Flash-V100 received an unresolved `fp8` KV-cache dtype and "
            "will interpret it as upstream E4M3. Normal EngineArgs processing "
            "rewrites the SM70 `fp8` shorthand to `fp8_e4m3`; this warning "
            "usually means the backend was constructed directly. KV-cache "
            "dtype is independent of model weight quantization."
        )
    elif kv_cache_dtype == "fp8_e4m3":
        logger.info(
            "SM70 Flash-V100 is using explicitly requested E4M3 KV cache. "
            "The decode route depends on the native extension and tensor "
            "layout. KV-cache dtype is independent of model weight quantization."
        )
    elif kv_cache_dtype == "fp8_e5m2":
        logger.info(
            "SM70 Flash-V100 is using explicit E5M2 KV cache. This controls "
            "KV storage only; model weight quantization is configured "
            "separately."
        )


def _mtp_context_bucket_partition_size_hint() -> int | None:
    raw = _config.raw("VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {raw!r}"
        ) from exc
    if value not in _VALID_DECODE_PARTITION_SIZES:
        raise ValueError(
            "VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {value}"
        )
    return value


def _mtp5_xqa_dual_cta_partition_size_hint() -> int | None:
    if _config.raw("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "1") != "1":
        return None
    raw = _config.raw("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "1024")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {raw!r}"
        ) from exc
    if value not in _VALID_DECODE_PARTITION_SIZES:
        raise ValueError(
            "VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {value}"
        )
    return value


def _select_default_decode_partition_size(
    max_seq_len_hint: int | None,
) -> int:
    if max_seq_len_hint is None:
        return _DEFAULT_DECODE_PARTITION_SIZE

    seq_len = max(1, int(max_seq_len_hint))
    if seq_len >= 32768:
        return 1024
    return _DEFAULT_DECODE_PARTITION_SIZE


def _decode_xqa_q4_min_seq_len() -> int:
    raw = _config.raw("VLLM_FLASH_V100_DECODE_XQA_Q4_MIN_SEQ_LEN")
    if raw is None:
        return _DEFAULT_Q4_XQA_MIN_SEQ_LEN
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(
            f"VLLM_FLASH_V100_DECODE_XQA_Q4_MIN_SEQ_LEN must be an integer, got {raw!r}"
        ) from exc


def _decode_fp8_xqa_min_seq_len() -> int:
    raw = _config.raw("VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN")
    if raw is None:
        return _DEFAULT_FP8_XQA_MIN_SEQ_LEN
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(
            "VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN must be an integer, "
            f"got {raw!r}"
        ) from exc


def _decode_fp8_xqa_allowed(
    attn_metadata: TritonAttentionMetadata,
    query: torch.Tensor,
) -> bool:
    graph_capture = bool(
        getattr(attn_metadata, "flash_v100_cudagraph_capture", False)
    ) or _is_cuda_graph_capturing(query)
    if graph_capture:
        hint_names = (
            "flash_v100_static_decode_seq_hint",
            "flash_v100_decode_workspace_seq_capacity_hint",
            "flash_v100_decode_max_seq_len_hint",
        )
    else:
        hint_names = (
            "flash_v100_decode_max_seq_len_hint",
            "flash_v100_static_decode_seq_hint",
            "flash_v100_decode_workspace_seq_capacity_hint",
        )
    for name in hint_names:
        seq_hint = getattr(attn_metadata, name, None)
        if seq_hint is not None:
            return int(seq_hint) >= _decode_fp8_xqa_min_seq_len()
    return False


def _decode_xqa_allowed_for_q_per_kv(
    q_per_kv: int,
    attn_metadata: TritonAttentionMetadata,
) -> bool:
    if q_per_kv in (6, 8):
        return True
    if q_per_kv != 4:
        return False

    seq_hint = getattr(
        attn_metadata,
        "flash_v100_decode_workspace_seq_capacity_hint",
        None,
    )
    if seq_hint is None:
        seq_hint = getattr(
            attn_metadata,
            "flash_v100_static_decode_seq_hint",
            None,
        )
    if seq_hint is None:
        seq_hint = getattr(
            attn_metadata,
            "flash_v100_decode_max_seq_len_hint",
            None,
        )
    if seq_hint is None:
        return False
    return int(seq_hint) >= _decode_xqa_q4_min_seq_len()


def _e4m3_batch_xqa_allowed(query: torch.Tensor) -> bool:
    """Admit GQA6 batches independently of the number of local KV heads."""
    return (
        _config.registered("VLLM_FLASH_V100_E4M3_BATCH_XQA")
        and query.ndim == 3
        and query.shape[0] > 1
        and query.shape[1] > 0
        and query.shape[1] % 6 == 0
        and query.shape[2] == 256
    )


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())


def _route_summary_enabled() -> bool:
    if _config.env_is_set("VLLM_SM70_DEBUG"):
        return "routing" in _config.registered("VLLM_SM70_DEBUG")
    return (
        _config.raw("VLLM_FLASH_V100_ROUTE_SUMMARY", "0") == "1"
        or _config.raw("VLLM_FLASH_V100_DEBUG_ROUTE_SUMMARY", "0") == "1"
    )


def _log_route_summary() -> None:
    if _route_counts:
        logger.info(
            "FLASH_ATTN_V100 route summary: %s",
            json.dumps(_route_counts, sort_keys=True),
        )
        _route_counts.clear()
    if _fallback_counts:
        logger.warning(
            "FLASH_ATTN_V100 fallback summary: %s",
            json.dumps(_fallback_counts, sort_keys=True),
        )
        _fallback_counts.clear()


def _record_route(route: str) -> None:
    global _route_summary_registered
    spec = route_spec(route)
    if spec.fallback:
        _fallback_counts[route] = _fallback_counts.get(route, 0) + 1
        logger.warning_once(
            "FLASH_ATTN_V100 explicit fallback selected: %s",
            route,
            scope="process",
        )
    enabled = _route_summary_enabled()
    if not enabled and not spec.fallback:
        return
    if enabled:
        _route_counts[route] = _route_counts.get(route, 0) + 1
    if not _route_summary_registered:
        atexit.register(_log_route_summary)
        _route_summary_registered = True


def _ddtree_trace_event(event: str, payload: dict[str, object]) -> None:
    trace_path = _config.raw("VLLM_DFLASH_DDTREE_TRACE_JSONL")
    if not trace_path:
        return
    record = {"event": event, "pid": os.getpid(), **payload}
    try:
        with open(trace_path, "a", encoding="utf-8") as trace_file:
            json.dump(record, trace_file, ensure_ascii=True, sort_keys=True)
            trace_file.write("\n")
    except OSError:
        logger.exception("Failed to write DDTree trace event to %s", trace_path)


def _ddtree_trace_enabled() -> bool:
    return bool(_config.raw("VLLM_DFLASH_DDTREE_TRACE_JSONL"))


def _decode_active_trace_enabled() -> bool:
    return _config.raw("VLLM_FLASH_V100_TRACE_DECODE_ACTIVE", "0") == "1"


def _decode_active_value(active_num_partitions: object) -> int | None:
    if not isinstance(active_num_partitions, torch.Tensor):
        return None
    if active_num_partitions.numel() == 0:
        return None
    return int(active_num_partitions.detach().reshape(-1)[0].item())


def _trace_decode_active(
    *,
    route: str,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    window_size: tuple[int, int],
) -> None:
    if not _decode_active_trace_enabled():
        return
    if torch.cuda.is_current_stream_capturing():
        return

    active_value = _decode_active_value(
        getattr(attn_metadata, "flash_v100_decode_active_num_partitions", None)
    )
    seq_len = int(seq_lens[: query.shape[0]].max().item())
    partition_size = _decode_partition_size_for_metadata(seq_len)
    expected_active = max(1, (seq_len + partition_size - 1) // partition_size)
    max_seq_hint = getattr(
        attn_metadata,
        "flash_v100_decode_max_seq_len_hint",
        None,
    )
    workspace_hint = getattr(
        attn_metadata,
        "flash_v100_decode_workspace_seq_capacity_hint",
        None,
    )
    static_hint = getattr(
        attn_metadata,
        "flash_v100_static_decode_seq_hint",
        None,
    )
    workspace_partitions = (
        max(1, (int(workspace_hint) + partition_size - 1) // partition_size)
        if workspace_hint is not None
        else None
    )
    signature = (
        route,
        int(query.shape[0]),
        int(query.shape[1]),
        int(key_cache.shape[2]),
        int(query.shape[2]),
        int(key_cache.shape[1]),
        seq_len,
        partition_size,
        active_value,
        expected_active,
        workspace_partitions,
        window_size,
    )
    if signature in _decode_active_trace_signatures:
        return
    _decode_active_trace_signatures.add(signature)
    logger.info(
        "FLASH_ATTN_V100 decode active trace: route=%s q=%d heads_q=%d "
        "heads_kv=%d head_dim=%d page_size=%d seq_len=%d partition=%d "
        "active=%s expected_active=%d workspace_partitions=%s "
        "max_seq_hint=%s workspace_hint=%s static_hint=%s window=%s",
        route,
        query.shape[0],
        query.shape[1],
        key_cache.shape[2],
        query.shape[2],
        key_cache.shape[1],
        seq_len,
        partition_size,
        active_value,
        expected_active,
        workspace_partitions,
        max_seq_hint,
        workspace_hint,
        static_hint,
        window_size,
    )


def _trace_decode_active_metadata(
    *,
    stage: str,
    max_seq_len_hint: int,
    workspace_seq_capacity_hint: int | None,
    static_decode_seq_hint: int | None,
    active: int,
    partition_size: int,
) -> None:
    if not _decode_active_trace_enabled():
        return

    expected_active = max(
        1,
        (int(max_seq_len_hint) + partition_size - 1) // partition_size,
    )
    workspace_partitions = (
        max(
            1,
            (int(workspace_seq_capacity_hint) + partition_size - 1) // partition_size,
        )
        if workspace_seq_capacity_hint is not None
        else None
    )
    signature = (
        "metadata",
        stage,
        active,
        partition_size,
        workspace_partitions,
    )
    if signature in _decode_active_trace_signatures:
        return
    _decode_active_trace_signatures.add(signature)
    logger.info(
        "FLASH_ATTN_V100 decode active metadata: stage=%s seq_len_hint=%d "
        "partition=%d active=%d expected_active=%d workspace_partitions=%s "
        "workspace_hint=%s static_hint=%s",
        stage,
        max_seq_len_hint,
        partition_size,
        active,
        expected_active,
        workspace_partitions,
        workspace_seq_capacity_hint,
        static_decode_seq_hint,
    )


def _uses_fp8_kv_cache(kv_cache_dtype: str) -> bool:
    return isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8")


def _log_fp8_kv_cache_route(stage: str, kv_cache_dtype: str, route: str) -> None:
    global _logged_fp8_kv_decode, _logged_fp8_kv_prefill

    if not _uses_fp8_kv_cache(kv_cache_dtype):
        return
    if stage not in ("prefill", "decode"):
        raise ValueError(f"Unsupported FP8 KV cache route stage: {stage}")
    _record_route(f"fp8_kv_{stage}")
    _record_route(f"fp8_kv_{stage}_{route}")
    if stage == "prefill":
        if _logged_fp8_kv_prefill:
            return
        logger.info(
            "FLASH_ATTN_V100 FP8 KV cache prefill path active "
            "(kv_cache_dtype=%s, route=%s).",
            kv_cache_dtype,
            route,
        )
        _logged_fp8_kv_prefill = True
        return
    if stage == "decode":
        if _logged_fp8_kv_decode:
            return
        logger.info(
            "FLASH_ATTN_V100 FP8 KV cache decode path active "
            "(kv_cache_dtype=%s, route=%s).",
            kv_cache_dtype,
            route,
        )
        _logged_fp8_kv_decode = True
        return
