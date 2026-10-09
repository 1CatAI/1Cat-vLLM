# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialization adapter for explicit native linear/MoE/collective policy inputs."""

import os
from typing import Any, ClassVar

from pydantic import Field

from vllm.config.utils import config, hash_factors

# Order is the versioned native vector ABI. Append fields; never reorder.
NATIVE_FIELDS = (
    ("tm_gemm_trace", "TM_GEMM_TRACE", ("awq", "fp8", "mxfp4", "nvfp4", "gguf"), True),
    (
        "tm_gemm_trace_filter",
        "TM_GEMM_TRACE_FILTER",
        ("awq", "fp8", "mxfp4", "nvfp4", "gguf"),
        True,
    ),
    (
        "tm_gemm_trace_limit",
        "TM_GEMM_TRACE_LIMIT",
        ("awq", "fp8", "mxfp4", "nvfp4", "gguf"),
        True,
    ),
    ("tm_gemm_tune", "TM_GEMM_TUNE", ("awq", "fp8", "mxfp4", "nvfp4", "gguf"), False),
    (
        "tm_gemm_warn_cache_miss",
        "TM_GEMM_WARN_CACHE_MISS",
        ("awq", "fp8", "mxfp4", "nvfp4", "gguf"),
        True,
    ),
    ("awq_dense_tune_max_m", "VLLM_SM70_AWQ_DENSE_TUNE_MAX_M", ("awq",), False),
    (
        "awq_moe_batched_active_exact_w2",
        "VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2",
        ("awq",),
        False,
    ),
    ("awq_moe_dispatch_policy", "VLLM_SM70_AWQ_MOE_DISPATCH_POLICY", ("awq",), False),
    ("awq_moe_tune_max_tokens", "VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS", ("awq",), False),
    ("awq_mtp_m5_fast_selector", "VLLM_SM70_AWQ_MTP_M5_FAST_SELECTOR", ("awq",), False),
    (
        "awq_preserve_default_splits",
        "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS",
        ("awq",),
        False,
    ),
    (
        "awq_preserve_default_splits_only",
        "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY",
        ("awq",),
        False,
    ),
    (
        "awq_qwen38_moe_compact_grouped_decode",
        "VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE",
        ("awq",),
        False,
    ),
    ("awq_reuse_imported_cache", "VLLM_SM70_AWQ_REUSE_IMPORTED_CACHE", ("awq",), False),
    ("awq_tp2_fast_selector", "VLLM_SM70_AWQ_TP2_FAST_SELECTOR", ("awq",), False),
    ("awq_tp2_fast_targets", "VLLM_SM70_AWQ_TP2_FAST_TARGETS", ("awq",), False),
    ("awq_tp4_qkv_cta64", "VLLM_SM70_AWQ_TP4_QKV_CTA64", ("awq",), False),
    ("awq_tune_small_shapes", "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES", ("awq", "fp8"), False),
    ("dflash2_qpn8_rerank", "VLLM_SM70_DFLASH2_QPN8_RERANK", ("f16",), False),
    (
        "dflash2_qpn8_rerank_shadow",
        "VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW",
        ("f16",),
        False,
    ),
    ("f16_dense_max_m", "VLLM_SM70_F16_DENSE_MAX_M", ("f16",), False),
    ("f16_dense_tune_max_m", "VLLM_SM70_F16_DENSE_TUNE_MAX_M", ("f16",), False),
    ("fp8_0dot3_dense_selector", "VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR", ("fp8",), False),
    ("fp8_dense_tune_max_m", "VLLM_SM70_FP8_DENSE_TUNE_MAX_M", ("fp8",), False),
    ("fp8_grouped_bmm_decode", "VLLM_SM70_FP8_GROUPED_BMM_DECODE", ("fp8",), False),
    ("fp8_moe_prepare_vec", "VLLM_SM70_FP8_MOE_PREPARE_VEC", ("fp8",), False),
    (
        "fp8_moe_single_token_per_expert_dispatch",
        "VLLM_SM70_FP8_MOE_SINGLE_TOKEN_PER_EXPERT_DISPATCH",
        ("fp8",),
        False,
    ),
    ("fp8_prefill_cutlass", "VLLM_SM70_FP8_PREFILL_CUTLASS", ("fp8",), False),
    (
        "fp8_prefill_fast_selector",
        "VLLM_SM70_FP8_PREFILL_FAST_SELECTOR",
        ("fp8",),
        False,
    ),
    (
        "fp8_preserve_default_splits",
        "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS",
        ("fp8",),
        False,
    ),
    (
        "fp8_preserve_default_splits_only",
        "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY",
        ("fp8",),
        False,
    ),
    ("fp8_reuse_imported_cache", "VLLM_SM70_FP8_REUSE_IMPORTED_CACHE", ("fp8",), False),
    ("fp8_safe_fast_selector", "VLLM_SM70_FP8_SAFE_FAST_SELECTOR", ("fp8",), False),
    ("fp8_tune_small_shapes", "VLLM_SM70_FP8_TUNE_SMALL_SHAPES", ("fp8",), False),
    (
        "glm53_moe_shuffle_sort_q8",
        "VLLM_SM70_GLM53_MOE_SHUFFLE_SORT_Q8",
        ("nvfp4",),
        False,
    ),
    (
        "moe_single_token_fastpath",
        "VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH",
        ("awq", "fp8", "mxfp4", "nvfp4", "gguf"),
        False,
    ),
    (
        "moe_single_token_permute_fastpath",
        "VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH",
        ("awq", "fp8", "mxfp4", "nvfp4", "gguf"),
        False,
    ),
    (
        "moe_single_token_unpermute_fastpath",
        "VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH",
        ("awq", "fp8", "mxfp4", "nvfp4", "gguf"),
        False,
    ),
    ("mxfp4_dense_tune_max_m", "VLLM_SM70_MXFP4_DENSE_TUNE_MAX_M", ("mxfp4",), False),
    (
        "mxfp4_moe_broadcast_input_decode",
        "VLLM_SM70_MXFP4_MOE_BROADCAST_INPUT_DECODE",
        ("mxfp4",),
        False,
    ),
    (
        "mxfp4_moe_compact_grouped_decode",
        "VLLM_SM70_MXFP4_MOE_COMPACT_GROUPED_DECODE",
        ("mxfp4",),
        False,
    ),
    ("mxfp4_moe_grouped_m8", "VLLM_SM70_MXFP4_MOE_GROUPED_M8", ("mxfp4",), False),
    (
        "mxfp4_moe_grouped_m8_expert_rows",
        "VLLM_SM70_MXFP4_MOE_GROUPED_M8_EXPERT_ROWS",
        ("mxfp4",),
        False,
    ),
    (
        "mxfp4_moe_grouped_m8_fast_selector",
        "VLLM_SM70_MXFP4_MOE_GROUPED_M8_FAST_SELECTOR",
        ("mxfp4",),
        False,
    ),
    (
        "mxfp4_moe_grouped_verifier",
        "VLLM_SM70_MXFP4_MOE_GROUPED_VERIFIER",
        ("mxfp4",),
        False,
    ),
    ("mxfp4_tune_small_shapes", "VLLM_SM70_MXFP4_TUNE_SMALL_SHAPES", ("mxfp4",), False),
    ("nvfp4_dense_tune_max_m", "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M", ("nvfp4",), False),
    (
        "nvfp4_moe_grouped_expert_rows",
        "VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS",
        ("nvfp4",),
        False,
    ),
    (
        "nvfp4_moe_grouped_prefill",
        "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL",
        ("nvfp4",),
        False,
    ),
    (
        "nvfp4_moe_tune_max_tokens",
        "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS",
        ("nvfp4",),
        False,
    ),
    ("nvfp4_qpn2_m16_native", "VLLM_SM70_NVFP4_QPN2_M16_NATIVE", ("nvfp4",), False),
    (
        "nvfp4_qwen38_moe_fast_prefill",
        "VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL",
        ("nvfp4",),
        False,
    ),
    (
        "nvfp4_qwen38_tp4_m1_fast_selector",
        "VLLM_SM70_NVFP4_QWEN38_TP4_M1_FAST_SELECTOR",
        ("nvfp4",),
        False,
    ),
    ("nvfp4_tune_small_shapes", "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES", ("nvfp4",), False),
    (
        "profile_trace",
        "VLLM_SM70_PROFILE_TRACE",
        ("awq", "fp8", "mxfp4", "nvfp4", "gguf"),
        True,
    ),
    ("fp8_qpn8_m16", "VLLM_SM70_FP8_QPN8_M16", ("fp8",), False),
    ("fp8_qpn8_m32_chunked", "VLLM_SM70_FP8_QPN8_M32_CHUNKED", ("fp8",), False),
    ("fp8_qpn8_m32_native", "VLLM_SM70_FP8_QPN8_M32_NATIVE", ("fp8",), False),
    (
        "glm_exact_kda_half2_rows",
        "VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS",
        ("f16",),
        False,
    ),
    (
        "dflash_sharded_context_fc",
        "VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC",
        ("f16",),
        False,
    ),
    (
        "tm_gemm_cache_summary",
        "TM_GEMM_CACHE_SUMMARY",
        ("awq", "fp8", "mxfp4", "nvfp4", "gguf", "f16"),
        True,
    ),
)

UNSET = "\x1f"


@config
class Sm70NativeConfig:
    """Per-engine native tuning; explicit values precede legacy aliases.

    Unset legacy values stay unset, preserving each original native default.
    Diagnostics are captured but excluded from calculation fingerprints.
    """

    tm_gemm_trace: bool | None = None
    """Native compatibility input TM_GEMM_TRACE."""
    tm_gemm_trace_filter: str | None = None
    """Native compatibility input TM_GEMM_TRACE_FILTER."""
    tm_gemm_trace_limit: int | None = None
    """Native compatibility input TM_GEMM_TRACE_LIMIT."""
    tm_gemm_tune: str | None = None
    """Native compatibility input TM_GEMM_TUNE."""
    tm_gemm_warn_cache_miss: bool | None = None
    """Native compatibility input TM_GEMM_WARN_CACHE_MISS."""
    awq_dense_tune_max_m: int | None = None
    """Native compatibility input VLLM_SM70_AWQ_DENSE_TUNE_MAX_M."""
    awq_moe_batched_active_exact_w2: bool | None = None
    """Native compatibility input VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2."""
    awq_moe_dispatch_policy: str | None = None
    """Native compatibility input VLLM_SM70_AWQ_MOE_DISPATCH_POLICY."""
    awq_moe_tune_max_tokens: int | None = None
    """Native compatibility input VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS."""
    awq_mtp_m5_fast_selector: bool | None = None
    """Native compatibility input VLLM_SM70_AWQ_MTP_M5_FAST_SELECTOR."""
    awq_preserve_default_splits: bool | None = None
    """Native compatibility input VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS."""
    awq_preserve_default_splits_only: bool | None = None
    """Native compatibility input VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY."""
    awq_qwen38_moe_compact_grouped_decode: bool | None = None
    """Native compatibility input VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE."""
    awq_reuse_imported_cache: bool | None = None
    """Native compatibility input VLLM_SM70_AWQ_REUSE_IMPORTED_CACHE."""
    awq_tp2_fast_selector: bool | None = None
    """Native compatibility input VLLM_SM70_AWQ_TP2_FAST_SELECTOR."""
    awq_tp2_fast_targets: str | None = None
    """Native compatibility input VLLM_SM70_AWQ_TP2_FAST_TARGETS."""
    awq_tp4_qkv_cta64: bool | None = None
    """Native compatibility input VLLM_SM70_AWQ_TP4_QKV_CTA64."""
    awq_tune_small_shapes: bool | None = None
    """Native compatibility input VLLM_SM70_AWQ_TUNE_SMALL_SHAPES."""
    dflash2_qpn8_rerank: bool | None = None
    """Native compatibility input VLLM_SM70_DFLASH2_QPN8_RERANK."""
    dflash2_qpn8_rerank_shadow: bool | None = None
    """Native compatibility input VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW."""
    f16_dense_max_m: int | None = None
    """Native compatibility input VLLM_SM70_F16_DENSE_MAX_M."""
    f16_dense_tune_max_m: int | None = None
    """Native compatibility input VLLM_SM70_F16_DENSE_TUNE_MAX_M."""
    fp8_0dot3_dense_selector: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR."""
    fp8_dense_tune_max_m: int | None = None
    """Native compatibility input VLLM_SM70_FP8_DENSE_TUNE_MAX_M."""
    fp8_grouped_bmm_decode: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_GROUPED_BMM_DECODE."""
    fp8_moe_prepare_vec: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_MOE_PREPARE_VEC."""
    fp8_moe_single_token_per_expert_dispatch: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_MOE_SINGLE_TOKEN_PER_EXPERT_DISPATCH."""
    fp8_prefill_cutlass: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_PREFILL_CUTLASS."""
    fp8_prefill_fast_selector: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_PREFILL_FAST_SELECTOR."""
    fp8_preserve_default_splits: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS."""
    fp8_preserve_default_splits_only: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY."""
    fp8_reuse_imported_cache: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_REUSE_IMPORTED_CACHE."""
    fp8_safe_fast_selector: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_SAFE_FAST_SELECTOR."""
    fp8_tune_small_shapes: bool | None = None
    """Native compatibility input VLLM_SM70_FP8_TUNE_SMALL_SHAPES."""
    glm53_moe_shuffle_sort_q8: bool | None = None
    """Native compatibility input VLLM_SM70_GLM53_MOE_SHUFFLE_SORT_Q8."""
    moe_single_token_fastpath: bool | None = None
    """Native compatibility input VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH."""
    moe_single_token_permute_fastpath: bool | None = None
    """Native compatibility input VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH."""
    moe_single_token_unpermute_fastpath: bool | None = None
    """Native compatibility input VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH."""
    mxfp4_dense_tune_max_m: int | None = None
    """Native compatibility input VLLM_SM70_MXFP4_DENSE_TUNE_MAX_M."""
    mxfp4_moe_broadcast_input_decode: bool | None = None
    """Native compatibility input VLLM_SM70_MXFP4_MOE_BROADCAST_INPUT_DECODE."""
    mxfp4_moe_compact_grouped_decode: bool | None = None
    """Native compatibility input VLLM_SM70_MXFP4_MOE_COMPACT_GROUPED_DECODE."""
    mxfp4_moe_grouped_m8: bool | None = None
    """Native compatibility input VLLM_SM70_MXFP4_MOE_GROUPED_M8."""
    mxfp4_moe_grouped_m8_expert_rows: bool | None = None
    """Native compatibility input VLLM_SM70_MXFP4_MOE_GROUPED_M8_EXPERT_ROWS."""
    mxfp4_moe_grouped_m8_fast_selector: bool | None = None
    """Native compatibility input VLLM_SM70_MXFP4_MOE_GROUPED_M8_FAST_SELECTOR."""
    mxfp4_moe_grouped_verifier: bool | None = None
    """Native compatibility input VLLM_SM70_MXFP4_MOE_GROUPED_VERIFIER."""
    mxfp4_tune_small_shapes: bool | None = None
    """Native compatibility input VLLM_SM70_MXFP4_TUNE_SMALL_SHAPES."""
    nvfp4_dense_tune_max_m: int | None = None
    """Native compatibility input VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M."""
    nvfp4_moe_grouped_expert_rows: bool | None = None
    """Native compatibility input VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS."""
    nvfp4_moe_grouped_prefill: bool | None = None
    """Native compatibility input VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL."""
    nvfp4_moe_tune_max_tokens: int | None = None
    """Native compatibility input VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS."""
    nvfp4_qpn2_m16_native: bool | None = None
    """Native compatibility input VLLM_SM70_NVFP4_QPN2_M16_NATIVE."""
    nvfp4_qwen38_moe_fast_prefill: bool | None = None
    """Native compatibility input VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL."""
    nvfp4_qwen38_tp4_m1_fast_selector: bool | None = None
    """Native compatibility input VLLM_SM70_NVFP4_QWEN38_TP4_M1_FAST_SELECTOR."""
    nvfp4_tune_small_shapes: bool | None = None
    """Native compatibility input VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES."""
    profile_trace: bool | None = None
    """Native compatibility input VLLM_SM70_PROFILE_TRACE."""
    fp8_qpn8_m16: bool | None = None
    """Native initialization input VLLM_SM70_FP8_QPN8_M16."""
    fp8_qpn8_m32_chunked: bool | None = None
    """Native initialization input VLLM_SM70_FP8_QPN8_M32_CHUNKED."""
    fp8_qpn8_m32_native: bool | None = None
    """Native initialization input VLLM_SM70_FP8_QPN8_M32_NATIVE."""
    glm_exact_kda_half2_rows: int | None = None
    """Native initialization input VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS."""
    dflash_sharded_context_fc: bool | None = None
    """Native initialization input VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC."""
    tm_gemm_cache_summary: bool | None = None
    """Native initialization input TM_GEMM_CACHE_SUMMARY."""
    values: tuple[str, ...] = Field(default=(), init=False)
    """Frozen native ABI values, prepared once for this format."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Provenance of the captured values; excluded from graph fingerprints."""

    def resolve(self, family: str, overrides: dict[str, Any] | None = None) -> None:
        if self.values:
            return
        overrides = overrides or {}
        values = []
        for field, alias, families, diagnostic in NATIVE_FIELDS:
            if family not in families:
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"Native option {field} does not apply to {family}"
                    )
                values.append(UNSET)
                continue
            value = getattr(self, field)
            source = "configuration" if value is not None else "default"
            if alias in overrides:
                if value is not None and value != overrides[alias]:
                    raise ValueError(f"Conflicting typed requests for {alias}")
                value = overrides[alias]
                source = "configuration"
            if value is None:
                value = os.getenv(alias, UNSET)
                source = alias if value != UNSET else "default"
            elif isinstance(value, bool):
                value = "1" if value else "0"
            else:
                value = str(value)
            values.append(value)
            self.sources[field] = source
        # Legacy combined FASTPATH is an OR, including when a stage alias is 0.
        # An explicit typed stage replaces that stage only; preserve the other
        # stage's legacy OR before removing the combined alias.
        indices = {field: i for i, (field, *_rest) in enumerate(NATIVE_FIELDS)}
        stages = (
            "moe_single_token_permute_fastpath",
            "moe_single_token_unpermute_fastpath",
        )
        combined = "moe_single_token_fastpath"
        if any(
            self.sources.get(field) == "configuration" for field in (*stages, combined)
        ):

            def enabled(value):
                import regex as re

                match = re.match(r"\s*([+-]?\d+)", value)
                return bool(match and int(match[1]))

            for field in stages:
                idx = indices[field]
                if self.sources.get(field) != "configuration":
                    values[idx] = str(
                        int(enabled(values[idx]) or enabled(values[indices[combined]]))
                    )
                    self.sources[field] = "resolved_single_token_fastpath"
            values[indices[combined]] = "0"
        self.values = tuple(values)

    def hash_options(self) -> dict[str, str]:
        return {
            field: value
            for (field, alias, families, diagnostic), value in zip(
                NATIVE_FIELDS, self.values
            )
            if not diagnostic and value != UNSET
        }


def capture_linear_native_config(family: str) -> Sm70NativeConfig:
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    if cfg is None:
        native = Sm70NativeConfig()
    else:
        policy = getattr(cfg.kernel_config, "sm70_" + family)
        native = policy if family == "mxfp4" else policy.native
    native.resolve(family)
    return native


def compile_ignored_aliases(kernel) -> set[str]:
    """Migrated values are hashed by their effective engine config instead.

    Generic FP16/auxiliary controls and legacy modular permutation remain
    independently hashed until those consumers have a prepared policy owner.
    """
    from vllm.config.kernel import (
        SM70_AWQ_LINEAR_ALIASES,
        SM70_FP8_LINEAR_ALIASES,
        SM70_NVFP4_LINEAR_ALIASES,
    )
    from vllm.config.sm70_moe import (
        ALIASES,
        AWQ_COMPARE_ALIASES,
        AWQ_DUMP_ALIASES,
        COMMON_ALIASES,
        FP8_COMPARE_ALIASES,
        MXFP4_ALIASES,
        NVFP4_ALIASES,
    )

    ignored = {
        alias
        for field, alias, families, diagnostic in NATIVE_FIELDS
        if "f16" not in families
        and field not in {"awq_tune_small_shapes"}
        and not field.startswith("moe_single_token_")
    }
    ignored.update(SM70_NVFP4_LINEAR_ALIASES.values())
    ignored.update(
        name for field, name in SM70_AWQ_LINEAR_ALIASES.items() if field != "enabled"
    )
    # These three FP8 aliases still participate in import-time optional library
    # admission or shared backend selection. Their raw ABI choice stays hashed.
    ignored.update(
        name
        for field, name in SM70_FP8_LINEAR_ALIASES.items()
        if field not in {"enabled", "qpn8", "qpn8_pp2_tp4"}
    )
    for aliases in (*ALIASES.values(), NVFP4_ALIASES, MXFP4_ALIASES):
        ignored.update(aliases.values())
    for names in COMMON_ALIASES.values():
        ignored.update(names)
    ignored.update(alias for alias, _ in AWQ_DUMP_ALIASES.values())
    ignored.update(AWQ_COMPARE_ALIASES.values())
    ignored.update(alias for alias, _ in FP8_COMPARE_ALIASES.values())
    # Shared switches may also serve a generic modular MoE in the same engine.
    # Keep the legacy key until every such consumer has a prepared owner.
    ignored.difference_update(
        alias
        for field, alias, _, _ in NATIVE_FIELDS
        if field.startswith("moe_single_token_")
    )
    ignored.add("VLLM_SM70_QWEN38_QPN_ROUTE_DEBUG")
    return ignored


@config
class CollectiveNativeConfig:
    """Typed overrides are encoded once for the existing native selectors.

    Raw strings retain historical atoi/strtol/exact-string differences. Native
    validation remains at the original operation checkpoint, including errors
    from options that are irrelevant to another topology or operator.
    """

    sm70_tp4_push_allreduce_small_messages: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES."""

    sm70_tp4_push_allreduce_concurrency: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY."""

    sm70_tp4_push_allreduce_qwen38_batch: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH."""

    sm70_qwen38_batch_fastpath: str | int | bool | None = None
    """Initialization override for VLLM_SM70_QWEN38_BATCH_FASTPATH."""

    sm70_tp4_push_allreduce_qwen38_batch_blocks: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH_BLOCKS."""

    sm70_tp4_push_allreduce_mtp5: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5."""

    sm70_tp2_ar_gemma_rms_threads: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP2_AR_GEMMA_RMS_THREADS."""

    sm70_tp4_long_fused_norm_threads: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_LONG_FUSED_NORM_THREADS."""

    sm70_tp4_long_fused_norm_blocks: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_LONG_FUSED_NORM_BLOCKS."""

    sm70_tp8_hierarchical_custom_ar: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR."""

    sm70_tp8_hierarchical_push_blocks: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP8_HIERARCHICAL_PUSH_BLOCKS."""

    custom_allreduce_block_limit: str | int | bool | None = None
    """Initialization override for VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT."""

    sm70_tp4_mtp_ar_block_tuning: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_MTP_AR_BLOCK_TUNING."""

    sm70_tp4_m5_ar_threads: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_M5_AR_THREADS."""

    sm70_tp4_small_ar_pack32: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_SMALL_AR_PACK32."""

    custom_allreduce_algo: str | int | bool | None = None
    """Initialization override for VLLM_CUSTOM_ALLREDUCE_ALGO."""

    sm70_tp4_push_allreduce_sum2_m1: str | int | bool | None = None
    """Initialization override for VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1."""

    sm70_profile_trace: str | int | bool | None = None
    """Initialization override for VLLM_SM70_PROFILE_TRACE."""

    aliases: ClassVar[dict[str, str]] = {
        "sm70_tp4_push_allreduce_small_messages": (
            "VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES"
        ),
        "sm70_tp4_push_allreduce_concurrency": (
            "VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY"
        ),
        "sm70_tp4_push_allreduce_qwen38_batch": (
            "VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH"
        ),
        "sm70_qwen38_batch_fastpath": "VLLM_SM70_QWEN38_BATCH_FASTPATH",
        "sm70_tp4_push_allreduce_qwen38_batch_blocks": (
            "VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH_BLOCKS"
        ),
        "sm70_tp4_push_allreduce_mtp5": "VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5",
        "sm70_tp2_ar_gemma_rms_threads": "VLLM_SM70_TP2_AR_GEMMA_RMS_THREADS",
        "sm70_tp4_long_fused_norm_threads": "VLLM_SM70_TP4_LONG_FUSED_NORM_THREADS",
        "sm70_tp4_long_fused_norm_blocks": "VLLM_SM70_TP4_LONG_FUSED_NORM_BLOCKS",
        "sm70_tp8_hierarchical_custom_ar": "VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR",
        "sm70_tp8_hierarchical_push_blocks": "VLLM_SM70_TP8_HIERARCHICAL_PUSH_BLOCKS",
        "custom_allreduce_block_limit": "VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT",
        "sm70_tp4_mtp_ar_block_tuning": "VLLM_SM70_TP4_MTP_AR_BLOCK_TUNING",
        "sm70_tp4_m5_ar_threads": "VLLM_SM70_TP4_M5_AR_THREADS",
        "sm70_tp4_small_ar_pack32": "VLLM_SM70_TP4_SMALL_AR_PACK32",
        "custom_allreduce_algo": "VLLM_CUSTOM_ALLREDUCE_ALGO",
        "sm70_tp4_push_allreduce_sum2_m1": "VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1",
        "sm70_profile_trace": "VLLM_SM70_PROFILE_TRACE",
    }

    values: tuple[str, ...] = Field(default=(), init=False)
    """Versioned immutable input vector; sentinel preserves unset versus empty."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Parameter provenance, excluded from computation hashes."""
    active: bool = Field(default=True, init=False)
    """False when native communication cannot affect the engine."""

    hash_fields: tuple[str, ...] | None = Field(default=None, init=False)
    """Fields admitted by the engine topology; None preserves standalone behavior."""

    def resolve(self, overrides=None):
        if self.values:
            return
        overrides = overrides or {}
        values = []
        for field, alias in self.aliases.items():
            value = getattr(self, field)
            if value is not None:
                if field in overrides and str(
                    int(value) if isinstance(value, bool) else value
                ) != str(overrides[field]):
                    raise ValueError(f"Conflicting typed requests for {alias}")
                self.sources[field] = "typed"
            elif field in overrides:
                value = overrides[field]
                self.sources[field] = "owner"
            else:
                # These defaults used to be written into os.environ on import.
                default = (
                    "1"
                    if field
                    in (
                        "sm70_tp4_push_allreduce_small_messages",
                        "sm70_tp4_push_allreduce_concurrency",
                    )
                    else "\x1f"
                )
                value = os.environ.get(alias, default)
                self.sources[field] = alias if alias in os.environ else "default"
            values.append(str(int(value) if isinstance(value, bool) else value))
        self.values = tuple(values)

    def resolve_for_owners(self, overrides, *, layers=None, trace=None):
        if trace is not None and trace.sources.get("profile_trace") == "typed":
            overrides["sm70_profile_trace"] = int(trace.profile_trace)
        if layers is not None and layers.sources.get("batch_fastpath") == "typed":
            overrides["sm70_qwen38_batch_fastpath"] = int(layers.batch_fastpath)
        self.resolve(overrides)
        if (
            layers is not None
            and self.sources.get("sm70_qwen38_batch_fastpath") == "typed"
        ):
            layers.batch_fastpath = self.registry_bool(
                "sm70_qwen38_batch_fastpath", "1"
            )
            layers.sources["batch_fastpath"] = "typed:collective_native"

    def finalize_hash(self, tp, pre_ampere):
        self.hash_fields = tuple(
            field
            for field in self.aliases
            if field.startswith("custom_allreduce_")
            or (
                pre_ampere
                and (
                    (field.startswith("sm70_tp2_") and tp == 2)
                    or (field.startswith("sm70_tp4_") and tp == 4)
                    or (field == "sm70_qwen38_batch_fastpath" and tp == 4)
                    or (field.startswith("sm70_tp8_") and tp == 8)
                )
            )
        )

    def raw(self, field, default=None):
        value = self.values[tuple(self.aliases).index(field)]
        return default if value == "\x1f" else value

    def registry_bool(self, field, default):
        return bool(int(self.raw(field, default)))

    def compute_hash(self):
        return hash_factors(
            {
                field: value
                for field, value in zip(self.aliases, self.values)
                if field != "sm70_profile_trace"
                and (self.hash_fields is None or field in self.hash_fields)
            }
            if self.active
            else {}
        )
