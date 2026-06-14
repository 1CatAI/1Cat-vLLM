# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request decode benchmark for SM70 migration validation.

This is intentionally narrower than the standard serving benchmark. It fixes
the prompt token length, generation length, sampling parameters, model args,
and SM70 gates so old-vs-latest no-MTP decode comparisons are reproducible.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    if value.startswith(("{", "[")):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_extra_engine_args(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE for --engine-arg, got {value!r}")
        key, raw = value.split("=", 1)
        parsed[key.replace("-", "_")] = _parse_scalar(raw)
    return parsed


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _serialize_logprobs(logprobs: Any) -> Any:
    if logprobs is None:
        return None

    serialized = []
    for step in logprobs:
        if step is None:
            serialized.append(None)
            continue
        serialized_step = {}
        for token_id, value in step.items():
            entry = {
                "logprob": float(value.logprob),
            }
            rank = getattr(value, "rank", None)
            if rank is not None:
                entry["rank"] = rank
            decoded_token = getattr(value, "decoded_token", None)
            if decoded_token is not None:
                entry["decoded_token"] = decoded_token
            serialized_step[str(token_id)] = entry
        serialized.append(serialized_step)
    return serialized


def _tracked_env() -> dict[str, str]:
    prefixes = (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "VLLM_SM70_",
        "VLLM_MQ_",
        "VLLM_USE_",
        "VLLM_DISABLE_",
        "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE",
        "VLLM_QWEN3NEXT_",
        "VLLM_QWEN3_NEXT_",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_FLASH_",
        "FLASH_ATTN",
        "FLA_",
        "CUDA_MODULE_LOADING",
        "TORCH_CUDA_ARCH_LIST",
        "TORCHINDUCTOR_",
        "TRITON_",
        "MAX_JOBS",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "CMAKE_BUILD_TYPE",
        "NVCC_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "MALLOC_ARENA_MAX",
    )
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes)
    }


def _atoi_nonzero(raw: str | None) -> bool:
    if raw is None:
        return False
    try:
        return int(raw) != 0
    except ValueError:
        return False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _atoi_nonzero(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _is_fp8_kv_cache_dtype(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("fp8")


def _fp8_tune_enabled(fp8_tune_raw: str | None, awq_tune_raw: str | None) -> bool:
    if fp8_tune_raw is not None:
        return _atoi_nonzero(fp8_tune_raw)
    if awq_tune_raw is not None:
        return _atoi_nonzero(awq_tune_raw)
    return True


def _sm70_tune_policy() -> dict[str, Any]:
    awq_tune_raw = os.environ.get("VLLM_SM70_AWQ_TUNE_SMALL_SHAPES")
    awq_tune0_pinned = awq_tune_raw == "0"
    awq_moe_safe_default_selector = awq_tune_raw is None or awq_tune0_pinned
    fp8_tune_raw = os.environ.get("VLLM_SM70_FP8_TUNE_SMALL_SHAPES")
    fp8_dynamic_measure_enabled = _fp8_tune_enabled(
        fp8_tune_raw,
        awq_tune_raw,
    )
    return {
        "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": awq_tune_raw,
        "VLLM_SM70_FP8_TUNE_SMALL_SHAPES": fp8_tune_raw,
        "awq_tune0_pinned_effective": awq_tune0_pinned,
        "awq_moe_safe_default_selector_effective": (
            awq_moe_safe_default_selector
        ),
        "fp8_safe_default_selector_effective": (
            not fp8_dynamic_measure_enabled
        ),
        "awq_dense_dynamic_measure_enabled": (
            awq_tune_raw is not None and _atoi_nonzero(awq_tune_raw)
        ),
        "awq_moe_dynamic_measure_enabled": (
            awq_tune_raw is not None and _atoi_nonzero(awq_tune_raw)
        ),
        "fp8_dynamic_measure_enabled": fp8_dynamic_measure_enabled,
        "generic_f16_dynamic_measure_enabled": (
            awq_tune_raw is None or _atoi_nonzero(awq_tune_raw)
        ),
        "note": (
            "C++ SM70 selectors read getenv() directly: unset keeps generic "
            "F16 dynamic-measure tuning enabled; AWQ dense/MoE stays "
            "fixed-dispatch unless VLLM_SM70_AWQ_TUNE_SMALL_SHAPES is set "
            "nonzero. FP8 dense defaults to dynamic small-shape selection "
            "unless VLLM_SM70_FP8_TUNE_SMALL_SHAPES=0, or the legacy AWQ "
            "tune env explicitly overrides it while the FP8 env is unset. "
            "Accepted MoE safe-route evidence records these selector fields; "
            "older artifacts still need explicit tune0 pins."
        ),
    }


def _sm70_turbomind_policy() -> dict[str, Any]:
    awq_turbomind = _env_bool("VLLM_SM70_AWQ_TURBOMIND", True)
    awq_tune_raw = os.environ.get("VLLM_SM70_AWQ_TUNE_SMALL_SHAPES")
    awq_tune0_pinned = awq_tune_raw == "0"
    awq_moe_safe_default_selector = awq_tune_raw is None or awq_tune0_pinned
    fp8_tune_raw = os.environ.get("VLLM_SM70_FP8_TUNE_SMALL_SHAPES")
    fp8_safe_default_selector = not _fp8_tune_enabled(
        fp8_tune_raw,
        awq_tune_raw,
    )
    fp8_turbomind = _env_bool("VLLM_SM70_FP8_TURBOMIND", True)
    fp8_dequant_fallback = _env_bool("VLLM_SM70_FP8_DEQUANT_FALLBACK", True)
    fp8_moe_dequant_fallback = _env_bool(
        "VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK",
        False,
    )
    unquantized_moe_0dot3_config = _env_bool(
        "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_CONFIG",
        True,
    )
    awq_warmup = _env_bool("VLLM_SM70_AWQ_WARMUP", True)
    awq_moe_disable = _env_bool("VLLM_SM70_AWQ_MOE_DISABLE", False)
    awq_moe_batched = _env_bool("VLLM_SM70_AWQ_MOE_BATCHED_GEMM", True)
    awq_moe_legacy_single_token = _env_bool(
        "VLLM_SM70_AWQ_MOE_LEGACY_SINGLE_TOKEN_COMPACT", True
    )
    fp8_moe_batched = _env_bool("VLLM_SM70_FP8_MOE_BATCHED_GEMM", True)
    fp8_moe_batched_w13_dispatch = _env_bool(
        "VLLM_SM70_FP8_MOE_BATCHED_W13_PER_EXPERT_DISPATCH",
        False,
    )
    fp8_moe_batched_w2_dispatch = _env_bool(
        "VLLM_SM70_FP8_MOE_BATCHED_W2_PER_EXPERT_DISPATCH",
        False,
    )
    fp8_moe_permute_with_scratch = _env_bool(
        "VLLM_SM70_FP8_MOE_PERMUTE_WITH_SCRATCH",
        True,
    )
    fp8_moe_legacy_single_token = _env_bool(
        "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT",
        False,
    )
    f16_dense = _env_bool("VLLM_SM70_ENABLE_DENSE_F16_FASTPATH", False)
    dense_cudagraph = _env_bool("VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE", False)
    return {
        "VLLM_SM70_AWQ_TURBOMIND": os.environ.get(
            "VLLM_SM70_AWQ_TURBOMIND"
        ),
        "awq_turbomind_effective": awq_turbomind,
        "VLLM_SM70_FP8_TURBOMIND": os.environ.get(
            "VLLM_SM70_FP8_TURBOMIND"
        ),
        "fp8_turbomind_effective": fp8_turbomind,
        "VLLM_SM70_FP8_DEQUANT_FALLBACK": os.environ.get(
            "VLLM_SM70_FP8_DEQUANT_FALLBACK"
        ),
        "fp8_dequant_fallback_effective": fp8_dequant_fallback,
        "VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK": os.environ.get(
            "VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK"
        ),
        "fp8_moe_dequant_fallback_effective": fp8_moe_dequant_fallback,
        "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_CONFIG": os.environ.get(
            "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_CONFIG"
        ),
        "unquantized_moe_0dot3_config_effective": (
            unquantized_moe_0dot3_config
        ),
        "VLLM_SM70_AWQ_WARMUP": os.environ.get("VLLM_SM70_AWQ_WARMUP"),
        "awq_warmup_effective": awq_warmup,
        "VLLM_SM70_AWQ_WARMUP_MAX_M": os.environ.get(
            "VLLM_SM70_AWQ_WARMUP_MAX_M"
        ),
        "awq_warmup_max_m_effective": _env_int(
            "VLLM_SM70_AWQ_WARMUP_MAX_M",
            8,
        ),
        "VLLM_SM70_AWQ_WARMUP_MAX_MOE_TOKENS": os.environ.get(
            "VLLM_SM70_AWQ_WARMUP_MAX_MOE_TOKENS"
        ),
        "awq_warmup_max_moe_tokens_effective": _env_int(
            "VLLM_SM70_AWQ_WARMUP_MAX_MOE_TOKENS",
            8,
        ),
        "VLLM_SM70_GEMM_LUT_PATH": os.environ.get("VLLM_SM70_GEMM_LUT_PATH"),
        "VLLM_SM70_AWQ_REUSE_IMPORTED_CACHE": os.environ.get(
            "VLLM_SM70_AWQ_REUSE_IMPORTED_CACHE"
        ),
        "awq_reuse_imported_cache_effective": _env_bool(
            "VLLM_SM70_AWQ_REUSE_IMPORTED_CACHE",
            False,
        ),
        "VLLM_SM70_AWQ_DENSE_TUNE_MAX_M": os.environ.get(
            "VLLM_SM70_AWQ_DENSE_TUNE_MAX_M"
        ),
        "awq_dense_tune_max_m_effective": _env_int(
            "VLLM_SM70_AWQ_DENSE_TUNE_MAX_M",
            16,
        ),
        "VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS": os.environ.get(
            "VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS"
        ),
        "awq_moe_tune_max_tokens_effective": _env_int(
            "VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS",
            128,
        ),
        "awq_moe_safe_default_selector_effective": (
            awq_moe_safe_default_selector
        ),
        "VLLM_SM70_FP8_TUNE_SMALL_SHAPES": fp8_tune_raw,
        "fp8_safe_default_selector_effective": fp8_safe_default_selector,
        "VLLM_SM70_AWQ_MOE_DISABLE": os.environ.get(
            "VLLM_SM70_AWQ_MOE_DISABLE"
        ),
        "awq_moe_disable_effective": awq_moe_disable,
        "VLLM_SM70_AWQ_MOE_BATCHED_GEMM": os.environ.get(
            "VLLM_SM70_AWQ_MOE_BATCHED_GEMM"
        ),
        "awq_moe_batched_gemm_effective": awq_moe_batched,
        "VLLM_SM70_AWQ_MOE_LEGACY_SINGLE_TOKEN_COMPACT": os.environ.get(
            "VLLM_SM70_AWQ_MOE_LEGACY_SINGLE_TOKEN_COMPACT"
        ),
        "awq_moe_legacy_single_token_compact_effective": (
            awq_moe_legacy_single_token
        ),
        "VLLM_SM70_FP8_MOE_BATCHED_GEMM": os.environ.get(
            "VLLM_SM70_FP8_MOE_BATCHED_GEMM"
        ),
        "fp8_moe_batched_gemm_effective": fp8_moe_batched,
        "VLLM_SM70_FP8_MOE_BATCHED_W13_PER_EXPERT_DISPATCH": os.environ.get(
            "VLLM_SM70_FP8_MOE_BATCHED_W13_PER_EXPERT_DISPATCH"
        ),
        "fp8_moe_batched_w13_per_expert_dispatch_effective": (
            fp8_moe_batched_w13_dispatch
        ),
        "VLLM_SM70_FP8_MOE_BATCHED_W2_PER_EXPERT_DISPATCH": os.environ.get(
            "VLLM_SM70_FP8_MOE_BATCHED_W2_PER_EXPERT_DISPATCH"
        ),
        "fp8_moe_batched_w2_per_expert_dispatch_effective": (
            fp8_moe_batched_w2_dispatch
        ),
        "VLLM_SM70_FP8_MOE_PERMUTE_WITH_SCRATCH": os.environ.get(
            "VLLM_SM70_FP8_MOE_PERMUTE_WITH_SCRATCH"
        ),
        "fp8_moe_permute_with_scratch_effective": (
            fp8_moe_permute_with_scratch
        ),
        "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT": os.environ.get(
            "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT"
        ),
        "fp8_moe_legacy_single_token_compact_effective": (
            fp8_moe_legacy_single_token
        ),
        "VLLM_SM70_ENABLE_DENSE_F16_FASTPATH": os.environ.get(
            "VLLM_SM70_ENABLE_DENSE_F16_FASTPATH"
        ),
        "f16_dense_fastpath_effective": f16_dense,
        "VLLM_SM70_F16_DENSE_ALLOWLIST": os.environ.get(
            "VLLM_SM70_F16_DENSE_ALLOWLIST"
        ),
        "VLLM_SM70_MOE_DENSE_ALLOWLIST": os.environ.get(
            "VLLM_SM70_MOE_DENSE_ALLOWLIST"
        ),
        "VLLM_SM70_F16_DENSE_MAX_M": os.environ.get(
            "VLLM_SM70_F16_DENSE_MAX_M"
        ),
        "f16_dense_max_m_effective": _env_int("VLLM_SM70_F16_DENSE_MAX_M", 64),
        "VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE": os.environ.get(
            "VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE"
        ),
        "dense_cudagraph_capture_effective": dense_cudagraph,
        "accepted_dense_default_policy": (
            awq_turbomind
            and fp8_turbomind
            and fp8_dequant_fallback
            and awq_warmup
            and not awq_moe_disable
            and not awq_moe_batched
            and not fp8_moe_batched
            and not fp8_moe_batched_w13_dispatch
            and not fp8_moe_batched_w2_dispatch
            and not f16_dense
        ),
        "accepted_awq_moe_default_policy": (
            awq_turbomind
            and awq_moe_safe_default_selector
            and not awq_moe_disable
            and awq_moe_batched
            and awq_moe_legacy_single_token
        ),
        "awq_moe_0dot3_baseline_policy": (
            awq_turbomind
            and awq_moe_safe_default_selector
            and not awq_moe_disable
            and awq_moe_batched
            and awq_moe_legacy_single_token
        ),
        "awq_moe_diagnostic_indexed_policy": (
            awq_turbomind
            and awq_moe_safe_default_selector
            and not awq_moe_disable
            and not awq_moe_legacy_single_token
        ),
        "accepted_fp8_moe_default_policy": (
            fp8_turbomind
            and fp8_dequant_fallback
            and not fp8_moe_dequant_fallback
            and fp8_safe_default_selector
            and fp8_moe_batched
            and not fp8_moe_batched_w13_dispatch
            and not fp8_moe_batched_w2_dispatch
            and fp8_moe_permute_with_scratch
            and not fp8_moe_legacy_single_token
        ),
        "fp8_moe_diagnostic_per_expert_dense_policy": (
            fp8_turbomind
            and fp8_dequant_fallback
            and not fp8_moe_dequant_fallback
            and fp8_safe_default_selector
            and not fp8_moe_batched
            and not fp8_moe_batched_w13_dispatch
            and not fp8_moe_batched_w2_dispatch
        ),
        "fp8_moe_0dot3_dequant_fallback_policy": (
            fp8_dequant_fallback
            and fp8_moe_dequant_fallback
            and unquantized_moe_0dot3_config
            and fp8_safe_default_selector
            and not fp8_moe_batched
            and not fp8_moe_batched_w13_dispatch
            and not fp8_moe_batched_w2_dispatch
        ),
        "route_hit_oracle": (
            "Accepted dense route-hit requires logs such as "
            "`SM70 AWQ TurboMind dense path enabled`, "
            "`SM70 FP8 TurboMind W8A16 dense path enabled`, and when warmup "
            "is relevant `SM70 AWQ warmup finished`. AWQ MoE production "
            "throughput evidence must keep the default fast route: batched "
            "GEMM enabled, legacy single-token compact enabled, and no "
            "decode-token cap that disables batched MoE during prefill. "
            "Strict indexed/dense-stage variants are diagnostics only; do "
            "not use them as accepted speed baselines. FP8 MoE production "
            "throughput evidence must use the native batched route and log "
            "`SM70 FP8 MoE TurboMind batched path enabled`; the per-expert "
            "dense-stage native route is a diagnostic fallback. FP8 MoE also "
            "keeps the 0.0.3 fallback lane, which requires "
            "`SM70 FP8 MoE fallback enabled` and dequantized fp16 expert "
            "weights plus `Using SM70 0.0.3 unquantized MoE default config`, "
            "and remains separate. FP8 batched W13/W2 per-expert dispatch "
            "flags are stage-local diagnostic lanes and must be recorded "
            "separately. MoE evidence additionally records fixed-dispatch "
            "tune policy: either explicit tune0 pins or source-level "
            "unset-env safe-default selector fields."
        ),
    }


def _sm70_attention_policy(kv_cache_dtype: Any) -> dict[str, Any]:
    selector_enabled = _env_bool("VLLM_SM70_FLASH_ATTN_V100", True)
    prefill_use_triton = _env_bool("VLLM_FLASH_V100_PREFILL_USE_TRITON", False)
    allow_triton_fallback = _env_bool(
        "VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK",
        False,
    )
    decode_scalar_paged = _env_bool(
        "VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED",
        True,
    )
    if selector_enabled:
        expected_sm70_priority = [
            "FLASH_ATTN_V100",
            "TRITON_ATTN",
            "FLEX_ATTENTION",
            "TURBOQUANT",
        ]
    else:
        expected_sm70_priority = [
            "FLASH_ATTN",
            "FLASHINFER",
            "TRITON_ATTN",
            "FLEX_ATTENTION",
            "TURBOQUANT",
        ]
    kv_cache_dtype_str = kv_cache_dtype if isinstance(kv_cache_dtype, str) else None
    fp8_kv_cache_requested = _is_fp8_kv_cache_dtype(kv_cache_dtype_str)
    full_flash_default_policy = (
        selector_enabled
        and not prefill_use_triton
        and not allow_triton_fallback
        and decode_scalar_paged
    )
    return {
        "VLLM_SM70_FLASH_ATTN_V100": os.environ.get(
            "VLLM_SM70_FLASH_ATTN_V100"
        ),
        "selector_enabled_effective": selector_enabled,
        "expected_sm70_priority": expected_sm70_priority,
        "VLLM_FLASH_V100_PREFILL_USE_TRITON": os.environ.get(
            "VLLM_FLASH_V100_PREFILL_USE_TRITON"
        ),
        "prefill_use_triton_effective": prefill_use_triton,
        "VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK": os.environ.get(
            "VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK"
        ),
        "allow_triton_fallback_effective": allow_triton_fallback,
        "VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED": os.environ.get(
            "VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED"
        ),
        "decode_scalar_paged_effective": decode_scalar_paged,
        "full_flash_default_policy": full_flash_default_policy,
        "kv_cache_dtype": kv_cache_dtype_str,
        "fp8_kv_cache_requested_effective": fp8_kv_cache_requested,
        "fp8_kv_cache_full_flash_policy": (
            full_flash_default_policy and fp8_kv_cache_requested
        ),
        "note": (
            "This records effective defaults even when env vars are unset; "
            "route-hit still requires backend logs or route_summary evidence. "
            "FP8 KV cache route-hit requires kv_cache_dtype=fp8* plus runtime "
            "Flash-V100 prefill, FP8 cache write, and FP8 KV scalar-paged "
            "decode logs; FP8 weight quantization alone is not FP8 KV cache "
            "evidence."
        ),
    }


def _sm70_graph_policy() -> dict[str, Any]:
    sm70_breakable = _env_bool("VLLM_SM70_USE_BREAKABLE_CUDAGRAPH", False)
    generic_breakable = _env_bool("VLLM_USE_BREAKABLE_CUDAGRAPH", False)
    dense_capture = _env_bool("VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE", False)
    flash_no_compile = _env_bool(
        "VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE",
        True,
    )
    flash_0dot3_compile = _env_bool(
        "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH",
        False,
    )
    flash_0dot3_eager_profile = _env_bool(
        "VLLM_SM70_FLASH_V100_0DOT3_EAGER_PROFILE_RUN",
        True,
    )
    flash_0dot3_benchmark_combo = _env_bool(
        "VLLM_SM70_FLASH_V100_0DOT3_BENCHMARK_COMBO_KERNEL",
        False,
    )
    flash_0dot3_decode_only_capture = _env_bool(
        "VLLM_SM70_FLASH_V100_0DOT3_DECODE_ONLY_CAPTURE",
        False,
    )
    use_aot_compile = _env_bool("VLLM_USE_AOT_COMPILE", flash_0dot3_compile)
    disable_compile_cache = _env_bool(
        "VLLM_DISABLE_COMPILE_CACHE",
        flash_0dot3_compile,
    )
    flash_capture_size = _env_int(
        "VLLM_SM70_FLASH_V100_DECODE_GRAPH_CAPTURE_SIZE",
        1,
    )
    return {
        "VLLM_SM70_USE_BREAKABLE_CUDAGRAPH": os.environ.get(
            "VLLM_SM70_USE_BREAKABLE_CUDAGRAPH"
        ),
        "sm70_breakable_requested_effective": sm70_breakable,
        "VLLM_USE_BREAKABLE_CUDAGRAPH": os.environ.get(
            "VLLM_USE_BREAKABLE_CUDAGRAPH"
        ),
        "breakable_cudagraph_effective": generic_breakable,
        "sm70_breakable_mapping_effective": sm70_breakable and generic_breakable,
        "VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE": os.environ.get(
            "VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE"
        ),
        "dense_cudagraph_capture_effective": dense_capture,
        "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH": os.environ.get(
            "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH"
        ),
        "sm70_flash_v100_0dot3_compile_graph_effective": flash_0dot3_compile,
        "VLLM_SM70_FLASH_V100_0DOT3_EAGER_PROFILE_RUN": os.environ.get(
            "VLLM_SM70_FLASH_V100_0DOT3_EAGER_PROFILE_RUN"
        ),
        "sm70_flash_v100_0dot3_eager_profile_effective": (
            flash_0dot3_compile and flash_0dot3_eager_profile
        ),
        "VLLM_SM70_FLASH_V100_0DOT3_BENCHMARK_COMBO_KERNEL": os.environ.get(
            "VLLM_SM70_FLASH_V100_0DOT3_BENCHMARK_COMBO_KERNEL"
        ),
        "sm70_flash_v100_0dot3_benchmark_combo_kernel_requested": (
            flash_0dot3_benchmark_combo
        ),
        "sm70_flash_v100_0dot3_benchmark_combo_kernel_effective": (
            flash_0dot3_compile
        ),
        "VLLM_SM70_FLASH_V100_0DOT3_DECODE_ONLY_CAPTURE": os.environ.get(
            "VLLM_SM70_FLASH_V100_0DOT3_DECODE_ONLY_CAPTURE"
        ),
        "sm70_flash_v100_0dot3_decode_only_capture_effective": (
            flash_0dot3_compile and flash_0dot3_decode_only_capture
        ),
        "VLLM_USE_AOT_COMPILE": os.environ.get("VLLM_USE_AOT_COMPILE"),
        "use_aot_compile_effective": use_aot_compile,
        "VLLM_DISABLE_COMPILE_CACHE": os.environ.get(
            "VLLM_DISABLE_COMPILE_CACHE"
        ),
        "disable_compile_cache_effective": disable_compile_cache,
        "sm70_flash_v100_0dot3_in_memory_aot_compile_effective": (
            flash_0dot3_compile and use_aot_compile and disable_compile_cache
        ),
        "VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE": os.environ.get(
            "VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE"
        ),
        "flash_v100_decode_graph_no_compile_effective": (
            flash_no_compile and not flash_0dot3_compile
        ),
        "VLLM_SM70_FLASH_V100_DECODE_GRAPH_CAPTURE_SIZE": os.environ.get(
            "VLLM_SM70_FLASH_V100_DECODE_GRAPH_CAPTURE_SIZE"
        ),
        "flash_v100_decode_graph_capture_size_effective": flash_capture_size,
        "default_policy_unchanged": (
            not sm70_breakable
            and not generic_breakable
            and not dense_capture
            and not flash_no_compile
            and not flash_0dot3_compile
        ),
        "old_0dot3_compile_graph_policy": (
            flash_0dot3_compile
            and not sm70_breakable
            and not generic_breakable
            and not dense_capture
        ),
        "route_hit_oracle": (
            "SM70 breakable CUDA graph route-hit requires "
            "VLLM_SM70_USE_BREAKABLE_CUDAGRAPH=1, automatic mapping to "
            "VLLM_USE_BREAKABLE_CUDAGRAPH=1, `Breakable CUDA graph enabled`, "
            "graph capture completion, and backend/kernel route logs. The "
            "0.0.3 Flash-V100 compile graph route requires "
            "sm70_flash_v100_0dot3_compile_graph_effective=true, "
            "mode=VLLM_COMPILE, cudagraph_mode=FULL_AND_PIECEWISE, "
            "small capture sizes, graph capture completion, eager-profile "
            "startup policy status, combo-kernel benchmark policy status, "
            "decode-only capture policy status, in-memory AOT compile status, "
            "and full Flash-V100 route logs. "
            "The Flash-V100 no-compile decode graph route requires "
            "flash_v100_decode_graph_no_compile_effective=true, "
            "mode=NONE, cudagraph_mode=FULL_DECODE_ONLY, graph capture "
            "completion, and full Flash-V100 route logs."
        ),
    }


def _sm70_comm_policy(engine_kwargs: dict[str, Any]) -> dict[str, Any]:
    disable_custom_all_reduce = bool(
        engine_kwargs.get("disable_custom_all_reduce", False)
    )
    moe_sum2 = _env_bool("VLLM_SM70_MOE_ADD_ALLREDUCE", False)
    top1_custom_ar = _env_bool("VLLM_SM70_TOP1_CUSTOM_AR", False)
    top1_only_custom_ar = top1_custom_ar and disable_custom_all_reduce
    return {
        "disable_custom_all_reduce": disable_custom_all_reduce,
        "custom_all_reduce_enabled_effective": not disable_custom_all_reduce,
        "hidden_state_custom_allreduce_effective": not disable_custom_all_reduce,
        "production_custom_allreduce_default_policy": (
            not disable_custom_all_reduce
        ),
        "VLLM_SM70_MOE_ADD_ALLREDUCE": os.environ.get(
            "VLLM_SM70_MOE_ADD_ALLREDUCE"
        ),
        "all_reduce_sum2_requested_effective": moe_sum2,
        "VLLM_SM70_TOP1_CUSTOM_AR": os.environ.get("VLLM_SM70_TOP1_CUSTOM_AR"),
        "top1_custom_allreduce_effective": top1_custom_ar,
        "top1_only_custom_allreduce_effective": top1_only_custom_ar,
        "route_hit_oracle": (
            "Production TP communication evidence must show "
            "`disable_custom_all_reduce=false` in engine kwargs and runtime "
            "logs selecting `['CUSTOM', 'PYNCCL']` for group `tp:0`. "
            "Top1-only custom allreduce evidence is separate: it may keep "
            "`disable_custom_all_reduce=true`, must keep hidden-state "
            "all-reduce on PYNCCL, and must show `SM70 custom top1 argmax "
            "resources enabled ... hidden-state custom all-reduce dispatch "
            "remains disabled` plus `SM70 custom top1 argmax path enabled`. "
            "Accepted all_reduce_sum2 evidence additionally requires "
            "`VLLM_SM70_MOE_ADD_ALLREDUCE=1` plus the C++ trace "
            "`SM70 custom all_reduce_sum2 op reached ... capture=active`; "
            "the Python candidate log alone is not sufficient."
        ),
    }


def _sm70_gdn_fla_policy() -> dict[str, Any]:
    packed_recurrent = _env_bool(
        "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE",
        True,
    )
    fla_recurrent = _env_bool("VLLM_SM70_FLA_RECURRENT_SCHEDULE", True)
    gdn_kkt = _env_bool("VLLM_SM70_GDN_KKT_SCHEDULE", True)
    gdn_delta_h = _env_bool("VLLM_SM70_GDN_DELTA_H_SCHEDULE", True)
    gdn_chunk_o = _env_bool("VLLM_SM70_GDN_CHUNK_O_SCHEDULE", True)
    fused_sigmoid_sched = _env_bool(
        "VLLM_SM70_FUSED_SIGMOID_GATING_SCHED",
        True,
    )
    mixed_qkv = _env_bool("VLLM_SM70_FUSED_SIGMOID_MIXED_QKV", False)
    mixed_qkv_compare = _env_bool(
        "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV_COMPARE",
        False,
    )
    flashqla_decode = _env_bool("VLLM_SM70_GDN_DECODE_FLASHQLA", True)
    flashqla_decode_route_debug = _env_bool(
        "VLLM_SM70_GDN_DECODE_FLASHQLA_ROUTE_DEBUG",
        False,
    )
    empty_core_out = _env_bool("VLLM_SM70_GDN_EMPTY_CORE_OUT", False)
    gdn_z_contiguous = _env_bool("VLLM_SM70_GDN_Z_CONTIGUOUS", False)
    gemma_rms_compile_native = _env_bool(
        "VLLM_SM70_GEMMA_RMS_NORM_COMPILE_NATIVE",
        False,
    )
    qwen3next_shared_moe_overlap = _env_bool(
        "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP",
        False,
    )
    qwen3next_disable_shared_moe_overlap = _env_bool(
        "VLLM_SM70_DISABLE_QWEN3NEXT_SHARED_MOE_OVERLAP",
        False,
    )
    return {
        "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": os.environ.get(
            "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"
        ),
        "packed_recurrent_decode_effective": packed_recurrent,
        "VLLM_SM70_FLA_RECURRENT_SCHEDULE": os.environ.get(
            "VLLM_SM70_FLA_RECURRENT_SCHEDULE"
        ),
        "fla_recurrent_schedule_effective": fla_recurrent,
        "VLLM_SM70_FLA_BV": os.environ.get("VLLM_SM70_FLA_BV"),
        "VLLM_SM70_FLA_WARPS": os.environ.get("VLLM_SM70_FLA_WARPS"),
        "VLLM_SM70_FLA_STAGES": os.environ.get("VLLM_SM70_FLA_STAGES"),
        "VLLM_SM70_FLA_TARGET_WAVES": os.environ.get(
            "VLLM_SM70_FLA_TARGET_WAVES"
        ),
        "VLLM_SM70_FLA_BV_CANDIDATES": os.environ.get(
            "VLLM_SM70_FLA_BV_CANDIDATES"
        ),
        "VLLM_SM70_GDN_KKT_SCHEDULE": os.environ.get(
            "VLLM_SM70_GDN_KKT_SCHEDULE"
        ),
        "gdn_kkt_schedule_effective": gdn_kkt,
        "VLLM_SM70_GDN_KKT_BK": os.environ.get("VLLM_SM70_GDN_KKT_BK"),
        "VLLM_SM70_GDN_KKT_WARPS": os.environ.get("VLLM_SM70_GDN_KKT_WARPS"),
        "VLLM_SM70_GDN_KKT_STAGES": os.environ.get("VLLM_SM70_GDN_KKT_STAGES"),
        "VLLM_SM70_GDN_DELTA_H_SCHEDULE": os.environ.get(
            "VLLM_SM70_GDN_DELTA_H_SCHEDULE"
        ),
        "gdn_delta_h_schedule_effective": gdn_delta_h,
        "VLLM_SM70_GDN_DELTA_H_BV": os.environ.get(
            "VLLM_SM70_GDN_DELTA_H_BV"
        ),
        "VLLM_SM70_GDN_DELTA_H_WARPS": os.environ.get(
            "VLLM_SM70_GDN_DELTA_H_WARPS"
        ),
        "VLLM_SM70_GDN_DELTA_H_STAGES": os.environ.get(
            "VLLM_SM70_GDN_DELTA_H_STAGES"
        ),
        "VLLM_SM70_GDN_CHUNK_O_SCHEDULE": os.environ.get(
            "VLLM_SM70_GDN_CHUNK_O_SCHEDULE"
        ),
        "gdn_chunk_o_schedule_effective": gdn_chunk_o,
        "VLLM_SM70_GDN_CHUNK_O_BK": os.environ.get("VLLM_SM70_GDN_CHUNK_O_BK"),
        "VLLM_SM70_GDN_CHUNK_O_BV": os.environ.get("VLLM_SM70_GDN_CHUNK_O_BV"),
        "VLLM_SM70_GDN_CHUNK_O_WARPS": os.environ.get(
            "VLLM_SM70_GDN_CHUNK_O_WARPS"
        ),
        "VLLM_SM70_GDN_CHUNK_O_STAGES": os.environ.get(
            "VLLM_SM70_GDN_CHUNK_O_STAGES"
        ),
        "VLLM_SM70_FUSED_SIGMOID_GATING_SCHED": os.environ.get(
            "VLLM_SM70_FUSED_SIGMOID_GATING_SCHED"
        ),
        "fused_sigmoid_gating_schedule_effective": fused_sigmoid_sched,
        "VLLM_SM70_FUSED_SIGMOID_GATING_BV": os.environ.get(
            "VLLM_SM70_FUSED_SIGMOID_GATING_BV"
        ),
        "VLLM_SM70_FUSED_SIGMOID_GATING_WARPS": os.environ.get(
            "VLLM_SM70_FUSED_SIGMOID_GATING_WARPS"
        ),
        "VLLM_SM70_FUSED_SIGMOID_GATING_STAGES": os.environ.get(
            "VLLM_SM70_FUSED_SIGMOID_GATING_STAGES"
        ),
        "VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING": os.environ.get(
            "VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING"
        ),
        "qwen3_next_legacy_fused_sigmoid_effective": _env_bool(
            "VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING",
            True,
        ),
        "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV": os.environ.get(
            "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV"
        ),
        "fused_sigmoid_mixed_qkv_effective": mixed_qkv,
        "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV_COMPARE": os.environ.get(
            "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV_COMPARE"
        ),
        "fused_sigmoid_mixed_qkv_compare_effective": mixed_qkv_compare,
        "VLLM_SM70_GDN_DECODE_FLASHQLA": os.environ.get(
            "VLLM_SM70_GDN_DECODE_FLASHQLA"
        ),
        "gdn_decode_flashqla_effective": flashqla_decode,
        "VLLM_SM70_GDN_DECODE_FLASHQLA_ROUTE_DEBUG": os.environ.get(
            "VLLM_SM70_GDN_DECODE_FLASHQLA_ROUTE_DEBUG"
        ),
        "gdn_decode_flashqla_route_debug_effective": (
            flashqla_decode_route_debug
        ),
        "VLLM_SM70_GDN_EMPTY_CORE_OUT": os.environ.get(
            "VLLM_SM70_GDN_EMPTY_CORE_OUT"
        ),
        "gdn_empty_core_out_effective": empty_core_out,
        "VLLM_SM70_GDN_Z_CONTIGUOUS": os.environ.get(
            "VLLM_SM70_GDN_Z_CONTIGUOUS"
        ),
        "gdn_z_contiguous_effective": gdn_z_contiguous,
        "VLLM_SM70_GEMMA_RMS_NORM_COMPILE_NATIVE": os.environ.get(
            "VLLM_SM70_GEMMA_RMS_NORM_COMPILE_NATIVE"
        ),
        "gemma_rms_norm_compile_native_effective": gemma_rms_compile_native,
        "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP": os.environ.get(
            "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP"
        ),
        "qwen3next_shared_moe_overlap_effective": qwen3next_shared_moe_overlap,
        "VLLM_SM70_DISABLE_QWEN3NEXT_SHARED_MOE_OVERLAP": os.environ.get(
            "VLLM_SM70_DISABLE_QWEN3NEXT_SHARED_MOE_OVERLAP"
        ),
        "qwen3next_disable_shared_moe_overlap_effective": (
            qwen3next_disable_shared_moe_overlap
        ),
        "FLA_USE_FAST_OPS": os.environ.get("FLA_USE_FAST_OPS"),
        "fla_use_fast_ops_effective": _env_bool("FLA_USE_FAST_OPS", False),
        "FLA_COMPILER_MODE": os.environ.get("FLA_COMPILER_MODE"),
        "fla_compiler_mode_effective": _env_bool("FLA_COMPILER_MODE", False),
        "FLA_USE_CUDA_GRAPH": os.environ.get("FLA_USE_CUDA_GRAPH"),
        "fla_use_cuda_graph_effective": _env_bool("FLA_USE_CUDA_GRAPH", False),
        "FLA_USE_TMA": os.environ.get("FLA_USE_TMA"),
        "fla_use_tma_effective": _env_bool("FLA_USE_TMA", False),
        "accepted_gdn_fla_default_policy": (
            packed_recurrent
            and not fla_recurrent
            and not gdn_kkt
            and not gdn_delta_h
            and not gdn_chunk_o
            and not fused_sigmoid_sched
            and not mixed_qkv
            and not mixed_qkv_compare
            and not empty_core_out
            and not gdn_z_contiguous
            and not gemma_rms_compile_native
            and not qwen3next_shared_moe_overlap
            and not qwen3next_disable_shared_moe_overlap
        ),
        "route_hit_oracle": (
            "Accepted GDN/FLA route-hit must record whether packed recurrent "
            "decode was enabled and whether any default-off SM70 schedules or "
            "mixed-QKV routes were explicitly enabled. GDN z materialization "
            "is diagnostic-only until it restores model-level tokens and "
            "passes decode speed checks. Mixed-QKV route-hit logs include "
            "`SM70 fused sigmoid GDN mixed-QKV decode route enabled`; compare "
            "mode must log an exact match before quality evidence can count."
        ),
    }


def _sm70_moe_policy() -> dict[str, Any]:
    add_allreduce = _env_bool("VLLM_SM70_MOE_ADD_ALLREDUCE", False)
    single_token_fastpath = _env_bool("VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH", False)
    single_token_permute = _env_bool(
        "VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH", False
    )
    single_token_unpermute = _env_bool(
        "VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH", True
    )
    single_token_indexed_stage = _env_bool(
        "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH", False
    )
    single_token_indexed_w13 = _env_bool(
        "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W13_FASTPATH", False
    )
    single_token_indexed_w2 = _env_bool(
        "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH", False
    )
    unquantized_moe_inplace_disabled = _env_bool(
        "VLLM_SM70_DISABLE_UNQUANTIZED_MOE_INPLACE",
        False,
    )
    unquantized_moe_0dot3_functional = _env_bool(
        "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_FUNCTIONAL",
        False,
    )
    profile_trace = _env_bool("VLLM_SM70_PROFILE_TRACE", False) or _env_bool(
        "VLLM_SM70_DECODE_TILE_PROFILE",
        False,
    )
    return {
        "VLLM_SM70_MOE_ADD_ALLREDUCE": os.environ.get(
            "VLLM_SM70_MOE_ADD_ALLREDUCE"
        ),
        "moe_add_allreduce_effective": add_allreduce,
        "VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH": os.environ.get(
            "VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH"
        ),
        "single_token_fastpath_effective": single_token_fastpath,
        "VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH": os.environ.get(
            "VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH"
        ),
        "single_token_permute_fastpath_effective": single_token_permute,
        "VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH": os.environ.get(
            "VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH"
        ),
        "single_token_unpermute_fastpath_effective": single_token_unpermute,
        "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH": os.environ.get(
            "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH"
        ),
        "single_token_indexed_stage_fastpath_effective": (
            single_token_indexed_stage
        ),
        "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W13_FASTPATH": os.environ.get(
            "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W13_FASTPATH"
        ),
        "single_token_indexed_w13_fastpath_effective": (
            single_token_indexed_stage or single_token_indexed_w13
        ),
        "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH": os.environ.get(
            "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH"
        ),
        "single_token_indexed_w2_fastpath_effective": (
            single_token_indexed_stage or single_token_indexed_w2
        ),
        "VLLM_SM70_DISABLE_UNQUANTIZED_MOE_INPLACE": os.environ.get(
            "VLLM_SM70_DISABLE_UNQUANTIZED_MOE_INPLACE"
        ),
        "unquantized_moe_inplace_output_env_allowed": (
            not unquantized_moe_inplace_disabled
        ),
        "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_FUNCTIONAL": os.environ.get(
            "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_FUNCTIONAL"
        ),
        "unquantized_moe_0dot3_functional_env_allowed": (
            unquantized_moe_0dot3_functional
        ),
        "VLLM_SM70_PROFILE_TRACE": os.environ.get("VLLM_SM70_PROFILE_TRACE"),
        "VLLM_SM70_DECODE_TILE_PROFILE": os.environ.get(
            "VLLM_SM70_DECODE_TILE_PROFILE"
        ),
        "profile_trace_effective": profile_trace,
        "expected_default": False,
        "single_token_unpermute_expected_default": True,
        "route_hit_oracle": (
            "For CUDA-graph fast-path acceptance, require the C++ trace "
            "`SM70 custom all_reduce_sum2 op reached ... capture=active`; "
            "the Python MoERunner candidate log alone is not a custom-op hit. "
            "For AWQ/FP8 safe MoE decode, require the active-expert dense log "
            "and `single-token weighted-reduce path enabled` to prove the "
            "default unpermute fast path is hit. Indexed dense-stage is a "
            "separate default-off launch-reduction candidate and requires "
            "`single-token indexed dense-stage path enabled` plus exactness "
            "evidence before it can count as accepted. The 0.0.3 FP8 MoE "
            "fallback lane also records whether legacy unquantized MoE "
            "inplace output and functional fused_experts are allowed."
        ),
    }


def _sm70_sampling_policy() -> dict[str, Any]:
    flash_0dot3_compile = _env_bool(
        "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH",
        False,
    )
    greedy_fastpath = _env_bool("VLLM_SM70_GREEDY_TOKEN_FASTPATH", True)
    greedy_trace = _env_bool("VLLM_SM70_GREEDY_TOKEN_FASTPATH_TRACE", False)
    lm_head_top1 = _env_bool(
        "VLLM_SM70_LM_HEAD_TOP1", not flash_0dot3_compile
    )
    lm_head_top1_tc = _env_bool("VLLM_SM70_LM_HEAD_TOP1_TC", False)
    custom_top1_ar = _env_bool("VLLM_SM70_TOP1_CUSTOM_AR", False)
    full_lm_head = _env_bool("VLLM_SM70_ENABLE_LM_HEAD_FASTPATH", False)
    dense_f16 = _env_bool("VLLM_SM70_ENABLE_DENSE_F16_FASTPATH", False)
    return {
        "VLLM_SM70_GREEDY_TOKEN_FASTPATH": os.environ.get(
            "VLLM_SM70_GREEDY_TOKEN_FASTPATH"
        ),
        "greedy_token_fastpath_effective": greedy_fastpath,
        "VLLM_SM70_GREEDY_TOKEN_FASTPATH_TRACE": os.environ.get(
            "VLLM_SM70_GREEDY_TOKEN_FASTPATH_TRACE"
        ),
        "greedy_token_fastpath_trace_effective": greedy_trace,
        "VLLM_SM70_LM_HEAD_TOP1": os.environ.get("VLLM_SM70_LM_HEAD_TOP1"),
        "lm_head_top1_effective": lm_head_top1,
        "VLLM_SM70_LM_HEAD_TOP1_TC": os.environ.get(
            "VLLM_SM70_LM_HEAD_TOP1_TC"
        ),
        "lm_head_top1_tc_effective": lm_head_top1_tc,
        "VLLM_SM70_TOP1_CUSTOM_AR": os.environ.get("VLLM_SM70_TOP1_CUSTOM_AR"),
        "top1_custom_ar_effective": custom_top1_ar,
        "VLLM_SM70_ENABLE_LM_HEAD_FASTPATH": os.environ.get(
            "VLLM_SM70_ENABLE_LM_HEAD_FASTPATH"
        ),
        "full_lm_head_fastpath_effective": full_lm_head,
        "VLLM_SM70_ENABLE_DENSE_F16_FASTPATH": os.environ.get(
            "VLLM_SM70_ENABLE_DENSE_F16_FASTPATH"
        ),
        "dense_f16_fastpath_effective": dense_f16,
        "pure_greedy_top1_default_policy": (
            greedy_fastpath
            and lm_head_top1
            and not lm_head_top1_tc
            and not custom_top1_ar
            and not full_lm_head
        ),
        "lm_head_0dot3_full_fastpath_policy": (
            full_lm_head and not lm_head_top1 and not lm_head_top1_tc
        ),
        "compile_graph_local_logits_top1_policy": (
            flash_0dot3_compile
            and greedy_fastpath
            and not lm_head_top1
            and not lm_head_top1_tc
            and not full_lm_head
            and not custom_top1_ar
        ),
        "route_hit_oracle": (
            "Accepted default greedy decode requires "
            "`SM70 LM head top1 layout prepared` plus "
            "`SM70 fused LM head top1 path enabled`; "
            "The SM70 Flash-V100 compile-graph quality lane instead requires "
            "VLLM_SM70_LM_HEAD_TOP1=0 and "
            "compile_graph_local_logits_top1_policy=true, preserving greedy "
            "pair-gather while avoiding the fused top1 epilogue. "
            "`SM70 dense fp16 fast path enabled for LM head` is expected only "
            "when the explicit full LM-head fast path gate is enabled. "
            "The 0.0.3 LM-head baseline lane requires full LM-head fast path "
            "on and top1 LM-head lanes off. "
            "Custom top1 allreduce remains experimental/default-off."
        ),
    }


def _enable_diagnostics_after_load() -> None:
    enable_files = (
        os.environ.get("VLLM_SM70_DUMP_SAMPLER_LOGITS_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_DUMP_SAMPLE_TENSORS_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_DUMP_GDN_CORE_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_DUMP_GDN_PROJ_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_COMPARE_GDN_PACKED_DECODE_ENABLE_FILE"),
    )
    for enable_file in enable_files:
        if not enable_file:
            continue
        path = Path(enable_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _make_prompt_token_ids(
    tokenizer: Any,
    prompt_base: str,
    input_len: int,
) -> list[int]:
    if input_len <= 0:
        raise ValueError("--input-len must be positive")

    chunk = tokenizer.encode(prompt_base, add_special_tokens=False)
    if not chunk:
        raise ValueError("--prompt-base produced no tokens")

    repeated: list[int] = []
    while len(repeated) < input_len:
        repeated.extend(chunk)
    return repeated[:input_len]


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.case_json is None:
        return [
            {
                "name": "default",
                "input_len": args.input_len,
                "output_len": args.output_len,
                "prompt_base": args.prompt_base,
            }
        ]

    raw_cases = json.loads(args.case_json.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("--case-json must contain a non-empty JSON list")

    cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case {index} must be an object")
        input_len = int(raw_case.get("input_len", args.input_len))
        output_len = int(raw_case.get("output_len", args.output_len))
        name = str(raw_case.get("name", f"case_{index}"))
        prompt_base = str(raw_case.get("prompt_base", args.prompt_base))
        cases.append(
            {
                "name": name,
                "input_len": input_len,
                "output_len": output_len,
                "prompt_base": prompt_base,
            }
        )
    return cases


def _hash_ids(token_ids: list[int]) -> str:
    encoded = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_delta(end: float, start: float) -> float | None:
    if end <= 0.0 or start <= 0.0:
        return None
    return end - start


def _request_metrics_dict(metrics: Any, output_tokens: int) -> dict[str, Any] | None:
    if metrics is None:
        return None

    queued_time = _safe_delta(metrics.scheduled_ts, metrics.queued_ts)
    prefill_time = _safe_delta(metrics.first_token_ts, metrics.scheduled_ts)
    decode_time = _safe_delta(metrics.last_token_ts, metrics.first_token_ts)
    inference_time = _safe_delta(metrics.last_token_ts, metrics.scheduled_ts)
    steady_decode_tokens = max(output_tokens - 1, 0)
    steady_decode_tps = (
        steady_decode_tokens / decode_time
        if decode_time and steady_decode_tokens > 0
        else None
    )
    tpot_seconds = (
        decode_time / steady_decode_tokens
        if decode_time and steady_decode_tokens > 0
        else None
    )

    return {
        "num_generation_tokens": metrics.num_generation_tokens,
        "first_token_latency": metrics.first_token_latency,
        "queued_time": queued_time,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "inference_time": inference_time,
        "steady_decode_tokens": steady_decode_tokens,
        "steady_decode_tps": steady_decode_tps,
        "tpot_seconds": tpot_seconds,
        "raw": {
            "arrival_time": metrics.arrival_time,
            "queued_ts": metrics.queued_ts,
            "scheduled_ts": metrics.scheduled_ts,
            "first_token_ts": metrics.first_token_ts,
            "last_token_ts": metrics.last_token_ts,
            "is_corrupted": metrics.is_corrupted,
        },
    }


def _compute_spec_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    num_drafts = int(raw.get("num_drafts") or 0)
    num_draft_tokens = int(raw.get("num_draft_tokens") or 0)
    num_accepted_tokens = int(raw.get("num_accepted_tokens") or 0)
    per_pos_accepted = [int(v) for v in raw.get("per_pos_accepted", [])]
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "acceptance_length": (
            1.0 + (num_accepted_tokens / num_drafts)
            if num_drafts > 0
            else None
        ),
        "draft_tokens_per_step": (
            num_draft_tokens / num_drafts if num_drafts > 0 else None
        ),
        "overall_acceptance_rate": (
            num_accepted_tokens / num_draft_tokens
            if num_draft_tokens > 0
            else None
        ),
        "per_pos_accepted": per_pos_accepted,
        "per_pos_acceptance_rates": (
            [v / num_drafts for v in per_pos_accepted]
            if num_drafts > 0
            else []
        ),
    }


def _spec_metrics_snapshot(llm: Any) -> dict[str, Any] | None:
    try:
        metrics = llm.get_metrics()
    except Exception:
        return None

    raw: dict[str, Any] = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
        "per_pos_accepted": [],
    }
    seen = False
    for metric in metrics:
        name = getattr(metric, "name", "")
        if name == "vllm:spec_decode_num_drafts" and hasattr(metric, "value"):
            raw["num_drafts"] += int(metric.value)
            seen = True
        elif (
            name == "vllm:spec_decode_num_draft_tokens"
            and hasattr(metric, "value")
        ):
            raw["num_draft_tokens"] += int(metric.value)
            seen = True
        elif (
            name == "vllm:spec_decode_num_accepted_tokens"
            and hasattr(metric, "value")
        ):
            raw["num_accepted_tokens"] += int(metric.value)
            seen = True
        elif (
            name == "vllm:spec_decode_num_accepted_tokens_per_pos"
            and hasattr(metric, "values")
        ):
            values = [int(v) for v in metric.values]
            if len(raw["per_pos_accepted"]) < len(values):
                raw["per_pos_accepted"].extend(
                    [0] * (len(values) - len(raw["per_pos_accepted"]))
                )
            for index, value in enumerate(values):
                raw["per_pos_accepted"][index] += value
            seen = True

    if not seen:
        return None
    return _compute_spec_metrics(raw)


def _diff_spec_metrics(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if after is None:
        return None
    before = before or {}
    per_pos_after = list(after.get("per_pos_accepted", []))
    per_pos_before = list(before.get("per_pos_accepted", []))
    per_pos_len = max(len(per_pos_after), len(per_pos_before))
    per_pos = []
    for index in range(per_pos_len):
        value_after = per_pos_after[index] if index < len(per_pos_after) else 0
        value_before = per_pos_before[index] if index < len(per_pos_before) else 0
        per_pos.append(value_after - value_before)
    return _compute_spec_metrics(
        {
            "num_drafts": int(after.get("num_drafts") or 0)
            - int(before.get("num_drafts") or 0),
            "num_draft_tokens": int(after.get("num_draft_tokens") or 0)
            - int(before.get("num_draft_tokens") or 0),
            "num_accepted_tokens": int(after.get("num_accepted_tokens") or 0)
            - int(before.get("num_accepted_tokens") or 0),
            "per_pos_accepted": per_pos,
        }
    )


def _run_once(llm: Any, prompt: dict[str, list[int]], sampling_params: Any) -> dict:
    request_prompt = {"prompt_token_ids": list(prompt["prompt_token_ids"])}
    spec_metrics_before = _spec_metrics_snapshot(llm)
    start = time.perf_counter()
    outputs = llm.generate([request_prompt], sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    spec_metrics_after = _spec_metrics_snapshot(llm)

    output_ids = list(outputs[0].outputs[0].token_ids)
    output_tokens = len(output_ids)
    logprobs = outputs[0].outputs[0].logprobs
    request_metrics = _request_metrics_dict(outputs[0].metrics, output_tokens)
    return {
        "elapsed_seconds": elapsed,
        "output_tokens": output_tokens,
        "output_tps": output_tokens / elapsed if elapsed > 0 else None,
        "request_metrics": request_metrics,
        "spec_decode_metrics": _diff_spec_metrics(
            spec_metrics_before,
            spec_metrics_after,
        ),
        "text": outputs[0].outputs[0].text,
        "token_ids": output_ids,
        "token_hash": _hash_ids(output_ids),
        "logprobs": _serialize_logprobs(logprobs),
        "finish_reason": outputs[0].outputs[0].finish_reason,
        "stop_reason": outputs[0].outputs[0].stop_reason,
    }


def _metric_values(repeats: list[dict], metric_name: str) -> list[float]:
    values = []
    for item in repeats:
        request_metrics = item.get("request_metrics")
        if not request_metrics:
            continue
        value = request_metrics.get(metric_name)
        if value is not None:
            values.append(value)
    return values


def _summarize(repeats: list[dict]) -> dict[str, Any]:
    values = [item["output_tps"] for item in repeats if item["output_tps"]]
    metric_summary = {}
    for metric_name in (
        "first_token_latency",
        "prefill_time",
        "decode_time",
        "steady_decode_tps",
        "tpot_seconds",
    ):
        metric_values = _metric_values(repeats, metric_name)
        if metric_values:
            metric_summary[f"{metric_name}_values"] = metric_values
            metric_summary[f"{metric_name}_min"] = min(metric_values)
            metric_summary[f"{metric_name}_max"] = max(metric_values)
            metric_summary[f"{metric_name}_mean"] = (
                sum(metric_values) / len(metric_values)
            )
    if not values:
        return {"output_tps_values": [], **metric_summary}
    return {
        "output_tps_values": values,
        "output_tps_min": min(values),
        "output_tps_max": max(values),
        "output_tps_mean": sum(values) / len(values),
        **metric_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument(
        "--case-json",
        type=Path,
        help=(
            "Optional JSON list of cases with name/input_len/output_len/"
            "prompt_base. Loads the model once and runs every case."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--prompt-base",
        default=(
            "This fixed benchmark prompt is used to create a deterministic "
            "tokenized input for single-request decode measurement. "
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization")
    parser.add_argument(
        "--kv-cache-dtype",
        help=(
            "Optional vLLM KV cache dtype, e.g. fp8, fp8_e4m3, or fp8_e5m2. "
            "This is separate from model weight quantization."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        help=(
            "Optional scheduler cap. Useful for route-hit checks where the "
            "default LLM_CLASS cap would otherwise compile up to 8192 tokens."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        help="Optional scheduler sequence cap for single-request route checks.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument(
        "--disable-log-stats",
        action="store_true",
        help=(
            "Disable vLLM request metrics. By default this SM70 harness keeps "
            "request metrics enabled so prefill/TTFT and steady decode TPS are "
            "recorded separately."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attention-backend",
        help="Optional vLLM attention backend enum name, e.g. TRITON_ATTN.",
    )
    parser.add_argument(
        "--cuda-profile-repeat",
        action="store_true",
        help="Wrap only measured repeats in cudaProfilerStart/Stop.",
    )
    parser.add_argument("--engine-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    import torch
    from transformers import AutoTokenizer

    import vllm
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model),
        trust_remote_code=args.trust_remote_code,
    )
    raw_cases = _load_cases(args)
    cases = []
    for raw_case in raw_cases:
        prompt_token_ids = _make_prompt_token_ids(
            tokenizer,
            raw_case["prompt_base"],
            raw_case["input_len"],
        )
        cases.append(
            {
                **raw_case,
                "prompt": {"prompt_token_ids": prompt_token_ids},
                "prompt_token_hash": _hash_ids(prompt_token_ids),
                "sampling_params": SamplingParams(
                    max_tokens=raw_case["output_len"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    presence_penalty=args.presence_penalty,
                    ignore_eos=args.ignore_eos,
                    skip_special_tokens=False,
                    logprobs=args.logprobs if args.logprobs != 0 else None,
                ),
            }
        )

    engine_kwargs = _parse_extra_engine_args(args.engine_arg)
    llm_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "disable_log_stats": args.disable_log_stats,
        "seed": args.seed,
        "attention_backend": args.attention_backend,
    }
    llm_kwargs.update(engine_kwargs)
    llm_kwargs = {key: value for key, value in llm_kwargs.items() if value is not None}

    load_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - load_start
    _enable_diagnostics_after_load()

    case_results = []
    for case in cases:
        warmups = [
            _run_once(llm, case["prompt"], case["sampling_params"])
            for _ in range(args.warmup)
        ]
        case_results.append(
            {
                "name": case["name"],
                "prompt": {
                    "input_len": len(case["prompt"]["prompt_token_ids"]),
                    "token_hash": case["prompt_token_hash"],
                },
                "sampling_params": {
                    "max_tokens": case["output_len"],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "min_p": args.min_p,
                    "presence_penalty": args.presence_penalty,
                    "ignore_eos": args.ignore_eos,
                    "skip_special_tokens": False,
                    "logprobs": args.logprobs if args.logprobs != 0 else None,
                },
                "warmups": warmups,
                "repeats": [],
                "summary": {},
            }
        )

    if args.cuda_profile_repeat:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for case, result in zip(cases, case_results):
            repeats = [
                _run_once(llm, case["prompt"], case["sampling_params"])
                for _ in range(args.repeat)
            ]
            result["repeats"] = repeats
            result["summary"] = _summarize(repeats)
    finally:
        if args.cuda_profile_repeat:
            torch.cuda.cudart().cudaProfilerStop()

    first_case = case_results[0]

    payload = {
        "model": str(args.model),
        "vllm": {
            "version": getattr(vllm, "__version__", None),
            "file": getattr(vllm, "__file__", None),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cuda_device_count": torch.cuda.device_count(),
        "device_capabilities": [
            list(torch.cuda.get_device_capability(i))
            for i in range(torch.cuda.device_count())
        ],
        "env": _tracked_env(),
        "sm70_tune_policy": _sm70_tune_policy(),
        "sm70_turbomind_policy": _sm70_turbomind_policy(),
        "sm70_attention_policy": _sm70_attention_policy(
            llm_kwargs.get("kv_cache_dtype")
        ),
        "sm70_graph_policy": _sm70_graph_policy(),
        "sm70_comm_policy": _sm70_comm_policy(llm_kwargs),
        "sm70_gdn_fla_policy": _sm70_gdn_fla_policy(),
        "sm70_moe_policy": _sm70_moe_policy(),
        "sm70_sampling_policy": _sm70_sampling_policy(),
        "engine_kwargs": llm_kwargs,
        "sampling_params": first_case["sampling_params"],
        "cuda_profile_repeat": args.cuda_profile_repeat,
        "prompt": first_case["prompt"],
        "load_seconds": load_seconds,
        "warmups": first_case["warmups"],
        "repeats": first_case["repeats"],
        "summary": first_case["summary"],
        "cases": case_results,
    }

    payload = _json_safe(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    printable = {
        key: value
        for key, value in payload.items()
        if key not in ("warmups", "repeats")
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
