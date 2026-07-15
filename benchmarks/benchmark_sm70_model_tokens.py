# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump and compare deterministic full-model token outputs for SM70 migration.

This harness is intentionally small and strict. It is used after op-level
SM70 TurboMind parity to verify that a full vLLM model load/generate path still
produces identical deterministic token IDs.
"""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = [
    "Write a concise explanation of why deterministic validation matters.",
]


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


def _hash_ids(token_ids: list[int]) -> str:
    encoded = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value if item is not None]
    return []


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


def _prompt_generation_metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "input_len": args.input_len,
        "uses_repeated_prompt_base": args.input_len is not None,
    }
    if args.input_len is None:
        return metadata

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model),
        trust_remote_code=args.trust_remote_code,
    )
    prompt_base_token_ids = tokenizer.encode(
        args.prompt_base,
        add_special_tokens=False,
    )
    metadata.update(
        {
            "prompt_base": args.prompt_base,
            "prompt_base_token_count": len(prompt_base_token_ids),
            "prompt_base_token_hash": _hash_ids(prompt_base_token_ids),
        }
    )
    return metadata


def _eos_token_ids_for_model(args: argparse.Namespace) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model),
        trust_remote_code=args.trust_remote_code,
    )
    eos_token_ids = set(_normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    generation_config = getattr(tokenizer, "generation_config", None)
    if generation_config is not None:
        eos_token_ids.update(
            _normalize_token_ids(getattr(generation_config, "eos_token_id", None))
        )
    return sorted(eos_token_ids)


def _first_token_index(token_ids: list[int], candidates: set[int]) -> int | None:
    if not candidates:
        return None
    for index, token_id in enumerate(token_ids):
        if token_id in candidates:
            return index
    return None


def _metric_snapshot(llm: Any) -> list[dict[str, Any]]:
    try:
        metrics = llm.get_metrics()
    except (AssertionError, AttributeError):
        return []
    records: list[dict[str, Any]] = []
    for metric in metrics:
        if is_dataclass(metric):
            records.append(asdict(metric))
        else:
            records.append({"repr": repr(metric)})
    return records


def _spec_decoding_summary(
    metrics: list[dict[str, Any]],
) -> dict[str, Any] | None:
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    accepted_tokens_per_pos: list[int] = []

    for metric in metrics:
        name = metric.get("name")
        if name == "vllm:spec_decode_num_drafts":
            num_drafts += int(metric.get("value") or 0)
        elif name == "vllm:spec_decode_num_draft_tokens":
            num_draft_tokens += int(metric.get("value") or 0)
        elif name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted_tokens += int(metric.get("value") or 0)
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values = metric.get("values")
            if isinstance(values, list):
                if len(values) > len(accepted_tokens_per_pos):
                    accepted_tokens_per_pos.extend(
                        [0] * (len(values) - len(accepted_tokens_per_pos))
                    )
                for idx, value in enumerate(values):
                    accepted_tokens_per_pos[idx] += int(value)

    if num_drafts == 0:
        return None

    avg_accepted_tokens = num_accepted_tokens / num_drafts
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "avg_accepted_tokens_no_bonus": avg_accepted_tokens,
        "mean_acceptance_length": 1 + avg_accepted_tokens,
        "draft_acceptance_rate": (
            num_accepted_tokens / num_draft_tokens
            if num_draft_tokens
            else None
        ),
        "accepted_tokens_per_pos": accepted_tokens_per_pos,
        "per_position_acceptance_rate": [
            value / num_drafts for value in accepted_tokens_per_pos
        ],
    }


def _load_prompts(args: argparse.Namespace) -> list[Any]:
    if args.input_len is not None:
        if args.prompt or args.prompts_json is not None:
            raise ValueError(
                "--input-len/--prompt-base cannot be mixed with text prompts"
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model),
            trust_remote_code=args.trust_remote_code,
        )
        prompt_token_ids = _make_prompt_token_ids(
            tokenizer,
            args.prompt_base,
            args.input_len,
        )
        return [{"prompt_token_ids": prompt_token_ids}]

    prompts = list(args.prompt)
    if args.prompts_json is not None:
        loaded = json.loads(args.prompts_json.read_text())
        if not isinstance(loaded, list) or not all(isinstance(p, str) for p in loaded):
            raise TypeError("--prompts-json must contain a JSON list of strings")
        prompts.extend(loaded)
    return prompts or DEFAULT_PROMPTS


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


def _actual_prompt_logprobs(
    prompt_token_ids: list[int],
    prompt_logprobs: Any,
) -> list[float]:
    if prompt_logprobs is None:
        return []

    values: list[float] = []
    for pos, step in enumerate(prompt_logprobs):
        if pos == 0 or step is None or pos >= len(prompt_token_ids):
            continue
        value = step.get(prompt_token_ids[pos])
        if value is not None:
            values.append(float(value.logprob))
    return values


def _prompt_metrics(
    prompt_token_ids: list[int],
    prompt_logprobs: Any,
) -> dict[str, Any]:
    values = _actual_prompt_logprobs(prompt_token_ids, prompt_logprobs)
    if not values:
        return {
            "token_count": 0,
            "avg_nll": None,
            "perplexity": None,
        }
    avg_nll = -sum(values) / len(values)
    return {
        "token_count": len(values),
        "avg_nll": avg_nll,
        "perplexity": math.exp(avg_nll),
    }


def _tracked_env() -> dict[str, str]:
    prefixes = (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "VLLM_SM70_",
        "VLLM_DFLASH_",
        "VLLM_MAMBA_",
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


def _is_fp8_kv_cache_dtype(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("fp8")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _tune_enabled_default_true(raw: str | None) -> bool:
    if raw is not None:
        return _atoi_nonzero(raw)
    return True


def _sm70_tune_policy() -> dict[str, Any]:
    awq_tune_raw = os.environ.get("VLLM_SM70_AWQ_TUNE_SMALL_SHAPES")
    awq_preserve_splits_raw = os.environ.get(
        "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS"
    )
    awq_tune0_pinned = awq_tune_raw == "0"
    awq_moe_safe_default_selector = awq_tune_raw is None or awq_tune0_pinned
    fp8_tune_raw = os.environ.get("VLLM_SM70_FP8_TUNE_SMALL_SHAPES")
    fp8_safe_fast_selector_raw = os.environ.get(
        "VLLM_SM70_FP8_SAFE_FAST_SELECTOR"
    )
    fp8_preserve_splits_raw = os.environ.get(
        "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS"
    )
    fp8_preserve_splits_only_raw = os.environ.get(
        "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY"
    )
    mxfp4_tune_raw = os.environ.get("VLLM_SM70_MXFP4_TUNE_SMALL_SHAPES")
    nvfp4_tune_raw = os.environ.get("VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES")
    fp8_dynamic_measure_enabled = _tune_enabled_default_true(fp8_tune_raw)
    mxfp4_dynamic_measure_enabled = _tune_enabled_default_true(mxfp4_tune_raw)
    nvfp4_dynamic_measure_enabled = _tune_enabled_default_true(nvfp4_tune_raw)
    return {
        "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": awq_tune_raw,
        "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS": awq_preserve_splits_raw,
        "VLLM_SM70_FP8_TUNE_SMALL_SHAPES": fp8_tune_raw,
        "VLLM_SM70_FP8_SAFE_FAST_SELECTOR": fp8_safe_fast_selector_raw,
        "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS": fp8_preserve_splits_raw,
        "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY": (
            fp8_preserve_splits_only_raw
        ),
        "VLLM_SM70_FP8_DENSE_TUNE_MAX_M": os.environ.get(
            "VLLM_SM70_FP8_DENSE_TUNE_MAX_M"
        ),
        "VLLM_SM70_MXFP4_TUNE_SMALL_SHAPES": mxfp4_tune_raw,
        "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES": nvfp4_tune_raw,
        "awq_tune0_pinned_effective": awq_tune0_pinned,
        "awq_preserve_default_splits_effective": (
            awq_preserve_splits_raw is None
            or _atoi_nonzero(awq_preserve_splits_raw)
        ),
        "awq_moe_safe_default_selector_effective": (
            awq_moe_safe_default_selector
        ),
        "fp8_safe_default_selector_effective": (
            not fp8_dynamic_measure_enabled
        ),
        "fp8_safe_fast_selector_effective": _atoi_nonzero(
            fp8_safe_fast_selector_raw
        ),
        "fp8_preserve_default_splits_effective": (
            fp8_preserve_splits_raw is None
            or _atoi_nonzero(fp8_preserve_splits_raw)
        ),
        "fp8_preserve_default_splits_only_effective": _atoi_nonzero(
            fp8_preserve_splits_only_raw
        ),
        "fp8_dense_tune_max_m_effective": _env_int(
            "VLLM_SM70_FP8_DENSE_TUNE_MAX_M",
            16,
        ),
        "mxfp4_safe_default_selector_effective": (
            not mxfp4_dynamic_measure_enabled
        ),
        "nvfp4_safe_default_selector_effective": (
            not nvfp4_dynamic_measure_enabled
        ),
        "awq_dense_dynamic_measure_enabled": (
            awq_tune_raw is not None and _atoi_nonzero(awq_tune_raw)
        ),
        "awq_moe_dynamic_measure_enabled": (
            awq_tune_raw is not None and _atoi_nonzero(awq_tune_raw)
        ),
        "fp8_dynamic_measure_enabled": fp8_dynamic_measure_enabled,
        "mxfp4_dynamic_measure_enabled": mxfp4_dynamic_measure_enabled,
        "nvfp4_dynamic_measure_enabled": nvfp4_dynamic_measure_enabled,
        "generic_f16_dynamic_measure_enabled": (
            awq_tune_raw is None or _atoi_nonzero(awq_tune_raw)
        ),
        "note": (
            "C++ SM70 selectors read getenv() directly: unset keeps generic "
            "F16 dynamic-measure tuning enabled; AWQ dense/MoE stays "
            "fixed-dispatch unless VLLM_SM70_AWQ_TUNE_SMALL_SHAPES is set "
            "nonzero. FP8 dense defaults to dynamic small-shape selection "
            "unless VLLM_SM70_FP8_TUNE_SMALL_SHAPES=0. MXFP4/NVFP4 "
            "TurboMind dense defaults to dynamic small-shape selection unless "
            "their VLLM_SM70_*FP4_TUNE_SMALL_SHAPES knobs are set to 0. "
            "AWQ dynamic dense tuning preserves the heuristic/default split-K "
            "count by default so measured kernels do not change fp16 reduction "
            "order. FP8 safe-fast selector is an explicit diagnostic lane; it "
            "uses dynamic dense selection with FP8-specific preserve-default "
            "split controls and must pass output/hash gates before becoming a "
            "default route. "
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
    fp8_safe_fast_selector = _env_bool(
        "VLLM_SM70_FP8_SAFE_FAST_SELECTOR", False
    )
    mxfp4_tune_raw = os.environ.get("VLLM_SM70_MXFP4_TUNE_SMALL_SHAPES")
    nvfp4_tune_raw = os.environ.get("VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES")
    fp8_safe_default_selector = not _tune_enabled_default_true(fp8_tune_raw)
    mxfp4_safe_default_selector = not _tune_enabled_default_true(mxfp4_tune_raw)
    nvfp4_safe_default_selector = not _tune_enabled_default_true(nvfp4_tune_raw)
    fp8_turbomind = _env_bool("VLLM_SM70_FP8_TURBOMIND", True)
    fp8_dense_gated_silu = _env_bool(
        "VLLM_SM70_FP8_DENSE_GATED_SILU",
        True,
    )
    nvfp4_turbomind = _env_bool("VLLM_SM70_NVFP4_TURBOMIND", False)
    mxfp4_turbomind = _env_bool("VLLM_SM70_MXFP4_TURBOMIND", False)
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
    awq_moe_batched_exact_w2 = _env_bool(
        "VLLM_SM70_AWQ_MOE_BATCHED_EXACT_W2", False
    )
    awq_moe_batched_active_exact_w2 = _env_bool(
        "VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2", False
    )
    awq_moe_batched_decode_max_tokens = _env_int(
        "VLLM_SM70_AWQ_MOE_BATCHED_DECODE_MAX_TOKENS", 0
    )
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
        True,
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
        "VLLM_SM70_FP8_DENSE_GATED_SILU": os.environ.get(
            "VLLM_SM70_FP8_DENSE_GATED_SILU"
        ),
        "fp8_dense_gated_silu_effective": fp8_dense_gated_silu,
        "VLLM_SM70_NVFP4_TURBOMIND": os.environ.get(
            "VLLM_SM70_NVFP4_TURBOMIND"
        ),
        "nvfp4_turbomind_effective": nvfp4_turbomind,
        "VLLM_SM70_MXFP4_TURBOMIND": os.environ.get(
            "VLLM_SM70_MXFP4_TURBOMIND"
        ),
        "mxfp4_turbomind_effective": mxfp4_turbomind,
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
            16,
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
        "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS": os.environ.get(
            "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS"
        ),
        "awq_preserve_default_splits_effective": _env_bool(
            "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS",
            True,
        ),
        "VLLM_SM70_AWQ_DENSE_TUNE_MAX_M": os.environ.get(
            "VLLM_SM70_AWQ_DENSE_TUNE_MAX_M"
        ),
        "awq_dense_tune_max_m_effective": _env_int(
            "VLLM_SM70_AWQ_DENSE_TUNE_MAX_M",
            16,
        ),
        "VLLM_SM70_MXFP4_DENSE_TUNE_MAX_M": os.environ.get(
            "VLLM_SM70_MXFP4_DENSE_TUNE_MAX_M"
        ),
        "mxfp4_dense_tune_max_m_effective": _env_int(
            "VLLM_SM70_MXFP4_DENSE_TUNE_MAX_M",
            16,
        ),
        "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M": os.environ.get(
            "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M"
        ),
        "nvfp4_dense_tune_max_m_effective": _env_int(
            "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M",
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
        "VLLM_SM70_FP8_SAFE_FAST_SELECTOR": os.environ.get(
            "VLLM_SM70_FP8_SAFE_FAST_SELECTOR"
        ),
        "fp8_safe_fast_selector_effective": fp8_safe_fast_selector,
        "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS": os.environ.get(
            "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS"
        ),
        "fp8_preserve_default_splits_effective": _env_bool(
            "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS",
            True,
        ),
        "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY": os.environ.get(
            "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY"
        ),
        "fp8_preserve_default_splits_only_effective": _env_bool(
            "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY",
            False,
        ),
        "VLLM_SM70_FP8_DENSE_TUNE_MAX_M": os.environ.get(
            "VLLM_SM70_FP8_DENSE_TUNE_MAX_M"
        ),
        "fp8_dense_tune_max_m_effective": _env_int(
            "VLLM_SM70_FP8_DENSE_TUNE_MAX_M",
            16,
        ),
        "VLLM_SM70_MXFP4_TUNE_SMALL_SHAPES": mxfp4_tune_raw,
        "mxfp4_safe_default_selector_effective": (
            mxfp4_safe_default_selector
        ),
        "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES": nvfp4_tune_raw,
        "nvfp4_safe_default_selector_effective": (
            nvfp4_safe_default_selector
        ),
        "VLLM_SM70_AWQ_MOE_DISABLE": os.environ.get(
            "VLLM_SM70_AWQ_MOE_DISABLE"
        ),
        "awq_moe_disable_effective": awq_moe_disable,
        "VLLM_SM70_AWQ_MOE_BATCHED_GEMM": os.environ.get(
            "VLLM_SM70_AWQ_MOE_BATCHED_GEMM"
        ),
        "awq_moe_batched_gemm_effective": awq_moe_batched,
        "VLLM_SM70_AWQ_MOE_BATCHED_EXACT_W2": os.environ.get(
            "VLLM_SM70_AWQ_MOE_BATCHED_EXACT_W2"
        ),
        "awq_moe_batched_exact_w2_effective": awq_moe_batched_exact_w2,
        "VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2": os.environ.get(
            "VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2"
        ),
        "awq_moe_batched_active_exact_w2_effective": (
            awq_moe_batched_active_exact_w2
        ),
        "VLLM_SM70_AWQ_MOE_BATCHED_DECODE_MAX_TOKENS": os.environ.get(
            "VLLM_SM70_AWQ_MOE_BATCHED_DECODE_MAX_TOKENS"
        ),
        "awq_moe_batched_decode_max_tokens_effective": (
            awq_moe_batched_decode_max_tokens
        ),
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
        "accepted_fp8_moe_default_policy": (
            fp8_turbomind
            and fp8_dequant_fallback
            and not fp8_moe_dequant_fallback
            and fp8_moe_batched
            and not fp8_moe_batched_w13_dispatch
            and not fp8_moe_batched_w2_dispatch
            and fp8_moe_permute_with_scratch
            and fp8_moe_legacy_single_token
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
            "throughput evidence must use the native batched route, keep "
            "`VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK=0`, keep scratch permute "
            "enabled, keep legacy single-token exact-layout compact enabled, "
            "and log `SM70 FP8 MoE TurboMind batched path enabled`; "
            "the per-expert dense-stage native route is a diagnostic fallback. "
            "FP8 MoE also keeps the 0.0.3 fallback lane, which requires "
            "`SM70 FP8 MoE fallback enabled` and dequantized fp16 expert "
            "weights plus `Using SM70 0.0.3 unquantized MoE default config`, "
            "and remains separate. FP8 batched W13/W2 per-expert dispatch "
            "flags are stage-local diagnostic lanes and must be recorded "
            "separately. FP8 legacy single-token compact evidence must show "
            "the exact-layout active source-group route, not the old top-k "
            "descriptor compact route."
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
    prefill_d256_output_stride_268 = _env_bool(
        "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268",
        True,
    )
    prefill_d256_sw_pipeline_qk = _env_bool(
        "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK",
        True,
    )
    prefill_d256_sw_pipeline_pv = _env_bool(
        "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV",
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
        "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268": os.environ.get(
            "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268"
        ),
        "prefill_d256_output_stride_268_effective": (
            prefill_d256_output_stride_268
        ),
        "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK": os.environ.get(
            "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK"
        ),
        "prefill_d256_sw_pipeline_qk_effective": prefill_d256_sw_pipeline_qk,
        "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV": os.environ.get(
            "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV"
        ),
        "prefill_d256_sw_pipeline_pv_effective": prefill_d256_sw_pipeline_pv,
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
        False,
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
            "small capture sizes, graph capture completion, decode-only "
            "capture policy status, in-memory AOT compile status, and full "
            "Flash-V100 route logs. "
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
        "VLLM_SM70_TP4_M5_AR_THREADS": os.environ.get(
            "VLLM_SM70_TP4_M5_AR_THREADS"
        ),
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
    fla_recurrent = _env_bool("VLLM_SM70_FLA_RECURRENT_SCHEDULE", False)
    gdn_kkt = _env_bool("VLLM_SM70_GDN_KKT_SCHEDULE", False)
    gdn_delta_h = _env_bool("VLLM_SM70_GDN_DELTA_H_SCHEDULE", False)
    gdn_chunk_o = _env_bool("VLLM_SM70_GDN_CHUNK_O_SCHEDULE", False)
    fused_sigmoid_sched = _env_bool(
        "VLLM_SM70_FUSED_SIGMOID_GATING_SCHED",
        False,
    )
    mixed_qkv = _env_bool("VLLM_SM70_FUSED_SIGMOID_MIXED_QKV", False)
    mixed_qkv_compare = _env_bool(
        "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV_COMPARE",
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


def _enable_sampler_logits_dump_after_load() -> None:
    enable_files = (
        os.environ.get("VLLM_SM70_DUMP_SAMPLER_LOGITS_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_DUMP_SAMPLE_TENSORS_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_DUMP_GDN_CORE_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_DUMP_GDN_PROJ_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_DUMP_QWEN_LAYER_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_COMPARE_GDN_PACKED_DECODE_ENABLE_FILE"),
        os.environ.get("VLLM_SM70_AWQ_MOE_COMPARE_DENSE_ENABLE_FILE"),
    )
    for enable_file in enable_files:
        if not enable_file:
            continue
        path = Path(enable_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _dump(args: argparse.Namespace) -> int:
    if args.out is None:
        raise ValueError("--out is required in dump mode")

    from vllm import LLM, SamplingParams

    capture_decode_after_prefix_warmup = (
        args.cuda_profiler_capture_decode_after_prefix_warmup
    )
    if capture_decode_after_prefix_warmup and args.sequential_prompts:
        raise ValueError(
            "--cuda-profiler-capture-decode-after-prefix-warmup does not "
            "support --sequential-prompts"
        )
    if capture_decode_after_prefix_warmup and args.prompt_logprobs != 0:
        raise ValueError(
            "--cuda-profiler-capture-decode-after-prefix-warmup does not "
            "support --prompt-logprobs because it bypasses prefix-cache reads"
        )

    prompts = _load_prompts(args)
    if capture_decode_after_prefix_warmup and len(prompts) != 1:
        raise ValueError(
            "--cuda-profiler-capture-decode-after-prefix-warmup requires "
            "exactly one prompt"
        )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=args.stop or None,
        seed=args.sampling_seed,
        ignore_eos=args.ignore_eos,
        skip_special_tokens=False,
        logprobs=args.logprobs if args.logprobs != 0 else None,
        prompt_logprobs=args.prompt_logprobs
        if args.prompt_logprobs != 0
        else None,
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
    if capture_decode_after_prefix_warmup:
        if llm_kwargs.get("enforce_eager"):
            raise ValueError(
                "--cuda-profiler-capture-decode-after-prefix-warmup requires "
                "CUDA graphs; do not set --enforce-eager or "
                "--engine-arg enforce_eager=true"
            )
        llm_kwargs["enable_prefix_caching"] = True
    llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}

    start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - start
    _enable_sampler_logits_dump_after_load()

    prime_generate_seconds: float | None = None
    if capture_decode_after_prefix_warmup:
        import torch

        prime_start = time.perf_counter()
        llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        prime_generate_seconds = time.perf_counter() - prime_start
        torch.cuda.cudart().cudaProfilerStart()
    elif args.cuda_profiler_capture_generate:
        import torch

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()
    try:
        if args.sequential_prompts:
            outputs = []
            for prompt in prompts:
                outputs.extend(llm.generate([prompt], sampling_params))
        else:
            outputs = llm.generate(prompts, sampling_params)
    finally:
        if args.cuda_profiler_capture_generate or capture_decode_after_prefix_warmup:
            import torch

            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
    generate_seconds = time.perf_counter() - start
    metrics_snapshot = _metric_snapshot(llm)

    import torch

    import vllm

    prompt_generation = _prompt_generation_metadata(args)
    eos_token_ids = _eos_token_ids_for_model(args)
    eos_token_id_set = set(eos_token_ids)

    records = []
    total_output_tokens = 0
    for request_output in outputs:
        prompt_token_ids = list(request_output.prompt_token_ids or [])
        completions = []
        for completion in request_output.outputs:
            token_ids = list(completion.token_ids)
            first_eos_index = _first_token_index(token_ids, eos_token_id_set)
            forced_tokens_after_eos = (
                max(len(token_ids) - first_eos_index - 1, 0)
                if args.ignore_eos and first_eos_index is not None
                else 0
            )
            total_output_tokens += len(token_ids)
            completions.append({
                "index": completion.index,
                "token_ids": token_ids,
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "contains_eos": first_eos_index is not None,
                "first_eos_token_index": first_eos_index,
                "forced_tokens_after_eos": forced_tokens_after_eos,
                "cumulative_logprob": completion.cumulative_logprob,
                "logprobs": _serialize_logprobs(completion.logprobs),
            })
        records.append({
            "request_id": request_output.request_id,
            "prompt": request_output.prompt,
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_hash": _hash_ids(prompt_token_ids),
            "prompt_logprobs": _serialize_logprobs(request_output.prompt_logprobs),
            "prompt_metrics": _prompt_metrics(
                prompt_token_ids, request_output.prompt_logprobs
            ),
            "request_metrics": _request_metrics_dict(
                request_output.metrics,
                sum(len(completion["token_ids"]) for completion in completions),
            ),
            "outputs": completions,
        })

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
        "prompt_generation": prompt_generation,
        "eos_token_ids": eos_token_ids,
        "metrics_snapshot": metrics_snapshot,
        "spec_decoding_metrics": _spec_decoding_summary(metrics_snapshot),
        "sampling_params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "stop": args.stop,
            "seed": args.sampling_seed,
            "ignore_eos": args.ignore_eos,
            "skip_special_tokens": False,
            "logprobs": args.logprobs if args.logprobs != 0 else None,
            "prompt_logprobs": args.prompt_logprobs
            if args.prompt_logprobs != 0
            else None,
        },
        "sequential_prompts": args.sequential_prompts,
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "total_output_tokens": total_output_tokens,
        "records": records,
    }
    if prime_generate_seconds is not None:
        payload["cuda_profiler_capture"] = {
            "mode": "decode_after_prefix_warmup",
            "enable_prefix_caching": llm_kwargs["enable_prefix_caching"],
            "enforce_eager": llm_kwargs["enforce_eager"],
            "prime": {
                "prompt_count": len(prompts),
                "same_prompt_as_capture": True,
                "cuda_profiler_enabled": False,
                "generate_seconds": prime_generate_seconds,
                "excluded_from": [
                    "generate_seconds",
                    "request_metrics",
                    "outputs",
                ],
            },
            "capture": {
                "prompt_count": len(prompts),
                "cuda_profiler_enabled": True,
                "gpu_synchronized_before_start": True,
                "generate_seconds": generate_seconds,
            },
        }
    payload = _json_safe(payload)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"},
                     indent=2, sort_keys=True))
    return 0


def _compare(args: argparse.Namespace) -> int:
    if args.json_out is None:
        raise ValueError("--json-out is required in compare mode")
    left = json.loads(args.compare[0].read_text())
    right = json.loads(args.compare[1].read_text())

    pairs = []
    equal = True
    for idx, (left_record, right_record) in enumerate(
            zip(left["records"], right["records"], strict=False)):
        left_prompt_ids = left_record["prompt_token_ids"]
        right_prompt_ids = right_record["prompt_token_ids"]
        prompt_equal = left_prompt_ids == right_prompt_ids
        output_pairs = []
        for out_idx, (left_out, right_out) in enumerate(
                zip(left_record["outputs"], right_record["outputs"],
                    strict=False)):
            left_ids = left_out["token_ids"]
            right_ids = right_out["token_ids"]
            output_equal = left_ids == right_ids
            output_pairs.append({
                "index": out_idx,
                "equal": output_equal,
                "left_len": len(left_ids),
                "right_len": len(right_ids),
                "first_mismatch": _first_mismatch(left_ids, right_ids),
            })
            equal = equal and output_equal
        same_output_count = len(left_record["outputs"]) == len(right_record["outputs"])
        pairs.append({
            "index": idx,
            "prompt_equal": prompt_equal,
            "left_prompt_len": len(left_prompt_ids),
            "right_prompt_len": len(right_prompt_ids),
            "same_output_count": same_output_count,
            "outputs": output_pairs,
        })
        equal = equal and prompt_equal and same_output_count

    same_request_count = len(left["records"]) == len(right["records"])
    equal = equal and same_request_count
    prompt_logprob_diff = _compare_prompt_logprobs(left, right)
    output_logprob_diff = _compare_output_logprobs(left, right)
    output_top_logprob_diff = _compare_output_top_logprobs(left, right)
    output_top_logprob_common_prefix_diff = (
        _compare_output_top_logprobs_common_prefix(left, right)
    )
    prompt_perplexity = _compare_prompt_perplexity(left, right)
    sampler_logits_diff = (
        _compare_logits_dirs(args.left_logits_dir, args.right_logits_dir)
        if args.left_logits_dir is not None and args.right_logits_dir is not None
        else None
    )
    model_quality_gate = _model_quality_gate(
        token_equal=equal,
        prompt_logprob_diff=prompt_logprob_diff,
        output_logprob_diff=output_logprob_diff,
        output_top_logprob_diff=output_top_logprob_diff,
        prompt_perplexity=prompt_perplexity,
        sampler_logits_diff=sampler_logits_diff,
        args=args,
    )
    result = {
        "left": str(args.compare[0]),
        "right": str(args.compare[1]),
        "equal": equal,
        "same_request_count": same_request_count,
        "left_request_count": len(left["records"]),
        "right_request_count": len(right["records"]),
        "pairs": pairs,
        "prompt_logprob_diff": prompt_logprob_diff,
        "output_logprob_diff": output_logprob_diff,
        "output_top_logprob_diff": output_top_logprob_diff,
        "output_top_logprob_common_prefix_diff": (
            output_top_logprob_common_prefix_diff
        ),
        "prompt_perplexity": prompt_perplexity,
        "sampler_logits_diff": sampler_logits_diff,
        "model_quality_gate": model_quality_gate,
        "left_meta": _without_records(left),
        "right_meta": _without_records(right),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    args.json_out.write_text(text + "\n")
    print(text)
    if args.require_model_quality_gate:
        return 0 if model_quality_gate["label"] == "model-pass" else 1
    return 0 if equal else 1


def _within_bound(value: float | None, bound: float | None) -> bool | None:
    if value is None:
        return None
    if bound is None:
        return None
    return value <= bound


def _max_prompt_perplexity_abs_diff(
    prompt_perplexity: list[dict[str, Any]],
) -> float | None:
    values = [
        float(entry["abs_diff"])
        for entry in prompt_perplexity
        if entry.get("abs_diff") is not None
    ]
    return max(values) if values else None


def _model_quality_gate(
    *,
    token_equal: bool,
    prompt_logprob_diff: dict[str, Any] | None,
    output_logprob_diff: dict[str, Any] | None,
    output_top_logprob_diff: dict[str, Any] | None,
    prompt_perplexity: list[dict[str, Any]],
    sampler_logits_diff: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    checks = {
        "token_equal": token_equal,
        "prompt_logprob_max_abs_diff": None
        if prompt_logprob_diff is None
        else prompt_logprob_diff.get("max_abs_diff"),
        "output_logprob_max_abs_diff": None
        if output_logprob_diff is None
        else output_logprob_diff.get("max_abs_diff"),
        "output_top_logprob_max_abs_diff": None
        if output_top_logprob_diff is None
        else output_top_logprob_diff.get("max_abs_diff"),
        "prompt_perplexity_max_abs_diff": _max_prompt_perplexity_abs_diff(
            prompt_perplexity
        ),
        "sampler_logits_max_abs_diff": None
        if sampler_logits_diff is None
        else sampler_logits_diff.get("max_abs_diff"),
        "sampler_logits_all_argmax_equal": None
        if sampler_logits_diff is None
        else sampler_logits_diff.get("all_argmax_equal"),
    }
    bounds = {
        "prompt_logprob_max_abs_diff": args.max_prompt_logprob_diff_for_accept,
        "output_logprob_max_abs_diff": args.max_output_logprob_diff_for_accept,
        "output_top_logprob_max_abs_diff": (
            args.max_output_top_logprob_diff_for_accept
        ),
        "prompt_perplexity_max_abs_diff": (
            args.max_prompt_perplexity_abs_diff_for_accept
        ),
        "sampler_logits_max_abs_diff": args.max_sampler_logits_diff_for_accept,
    }

    failed = []
    pending = []
    warning = []
    if not token_equal:
        message = "deterministic token IDs differ"
        if args.allow_token_diff_for_model_quality_gate:
            warning.append(message)
        else:
            failed.append(message)
    sampler_argmax_equal = checks["sampler_logits_all_argmax_equal"]
    if sampler_argmax_equal is False:
        message = "sampler logits argmax differs"
        if args.allow_sampler_argmax_diff_for_model_quality_gate:
            warning.append(message)
        else:
            failed.append(message)
    elif sampler_argmax_equal is None:
        message = "sampler logits argmax evidence missing"
        if args.allow_missing_sampler_logits_for_model_quality_gate:
            warning.append(message)
        else:
            pending.append(message)
    for name, bound in bounds.items():
        value = checks[name]
        if (
            name == "sampler_logits_max_abs_diff"
            and value is None
            and args.allow_missing_sampler_logits_for_model_quality_gate
        ):
            warning.append(f"{name} evidence missing")
            continue
        within = _within_bound(value, bound)
        if within is False:
            failed.append(f"{name}={value} exceeds bound {bound}")
        elif value is not None and bound is None:
            pending.append(f"{name}={value} has no configured acceptance bound")
        elif value is None:
            pending.append(f"{name} evidence missing")

    if failed:
        label = "model-fail"
        default_acceptance = "not default-accepted"
    elif pending:
        label = "B-pending"
        default_acceptance = "not default-accepted"
    else:
        label = "model-pass"
        default_acceptance = "model-level gate passed"

    return {
        "label": label,
        "default_acceptance": default_acceptance,
        "checks": checks,
        "bounds": bounds,
        "failed_evidence": failed,
        "pending_evidence": pending,
        "warning_evidence": warning,
    }


def _logprob_value(entry: Any) -> float | None:
    if isinstance(entry, dict):
        value = entry.get("logprob")
        return None if value is None else float(value)
    return None


def _step_logprob_for_token(step: Any, token_id: int) -> float | None:
    if not isinstance(step, dict):
        return None
    entry = step.get(str(token_id))
    return _logprob_value(entry)


def _record_actual_output_logprobs(record: dict[str, Any]) -> list[float | None]:
    values: list[float | None] = []
    for output in record.get("outputs") or []:
        token_ids = output.get("token_ids") or []
        logprobs = output.get("logprobs") or []
        for index, token_id in enumerate(token_ids):
            step = logprobs[index] if index < len(logprobs) else None
            values.append(_step_logprob_for_token(step, int(token_id)))
    return values


def _compare_optional_values(
    left_values: list[float | None],
    right_values: list[float | None],
) -> dict[str, Any]:
    diffs: list[float] = []
    same_count = len(left_values) == len(right_values)
    same_none_mask = same_count
    for left_value, right_value in zip(left_values, right_values, strict=False):
        if left_value is None or right_value is None:
            same_none_mask = same_none_mask and left_value is right_value
            continue
        diffs.append(abs(left_value - right_value))
    return {
        "same_count": same_count,
        "same_none_mask": same_none_mask,
        "count": len(diffs),
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
    }


def _compare_output_logprobs(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    diffs: list[float] = []
    same_count = len(left["records"]) == len(right["records"])
    same_none_mask = same_count
    for left_record, right_record in zip(
        left["records"], right["records"], strict=False
    ):
        comparison = _compare_optional_values(
            _record_actual_output_logprobs(left_record),
            _record_actual_output_logprobs(right_record),
        )
        same_count = same_count and bool(comparison["same_count"])
        same_none_mask = same_none_mask and bool(comparison["same_none_mask"])
        if comparison["max_abs_diff"] is not None:
            diffs.append(float(comparison["max_abs_diff"]))
    if not diffs:
        return None
    return {
        "same_count": same_count,
        "same_none_mask": same_none_mask,
        "record_count": len(left["records"]),
        "max_abs_diff": max(diffs),
    }


def _top_logprob_maps(logprobs: Any) -> list[dict[str, float]]:
    if not isinstance(logprobs, list):
        return []
    maps: list[dict[str, float]] = []
    for step in logprobs:
        if not isinstance(step, dict):
            maps.append({})
            continue
        values: dict[str, float] = {}
        for key, entry in step.items():
            logprob = _logprob_value(entry)
            if logprob is not None:
                values[str(key)] = logprob
        maps.append(values)
    return maps


def _record_output_top_logprob_maps(record: dict[str, Any]) -> list[dict[str, float]]:
    maps: list[dict[str, float]] = []
    for output in record.get("outputs") or []:
        maps.extend(_top_logprob_maps(output.get("logprobs") or []))
    return maps


def _compare_top_logprob_maps(
    left_maps: list[dict[str, float]],
    right_maps: list[dict[str, float]],
) -> dict[str, Any]:
    diffs: list[float] = []
    missing_left = 0
    missing_right = 0
    same_step_count = len(left_maps) == len(right_maps)
    for left_step, right_step in zip(left_maps, right_maps, strict=False):
        left_keys = set(left_step)
        right_keys = set(right_step)
        missing_left += len(right_keys - left_keys)
        missing_right += len(left_keys - right_keys)
        for key in left_keys & right_keys:
            diffs.append(abs(left_step[key] - right_step[key]))
    return {
        "same_step_count": same_step_count,
        "step_count": min(len(left_maps), len(right_maps)),
        "common_value_count": len(diffs),
        "missing_left": missing_left,
        "missing_right": missing_right,
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
    }


def _compare_output_top_logprobs(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    max_diffs: list[float] = []
    same_step_count = len(left["records"]) == len(right["records"])
    missing_left = 0
    missing_right = 0
    common_value_count = 0
    for left_record, right_record in zip(
        left["records"], right["records"], strict=False
    ):
        comparison = _compare_top_logprob_maps(
            _record_output_top_logprob_maps(left_record),
            _record_output_top_logprob_maps(right_record),
        )
        same_step_count = same_step_count and bool(comparison["same_step_count"])
        missing_left += int(comparison["missing_left"])
        missing_right += int(comparison["missing_right"])
        common_value_count += int(comparison["common_value_count"])
        if comparison["max_abs_diff"] is not None:
            max_diffs.append(float(comparison["max_abs_diff"]))
    if not max_diffs and common_value_count == 0:
        return None
    return {
        "same_step_count": same_step_count,
        "common_value_count": common_value_count,
        "missing_left": missing_left,
        "missing_right": missing_right,
        "max_abs_diff": max(max_diffs) if max_diffs else None,
    }


def _common_prefix_len(left_ids: list[int], right_ids: list[int]) -> int:
    for idx, (left_id, right_id) in enumerate(zip(left_ids, right_ids, strict=False)):
        if left_id != right_id:
            return idx
    return min(len(left_ids), len(right_ids))


def _compare_output_top_logprobs_common_prefix(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    max_diffs: list[float] = []
    total_sum = 0.0
    common_value_count = 0
    missing_left = 0
    missing_right = 0
    compared_steps = 0
    prefix_lengths: list[dict[str, int]] = []
    for record_idx, (left_record, right_record) in enumerate(
        zip(left["records"], right["records"], strict=False)
    ):
        for output_idx, (left_output, right_output) in enumerate(
            zip(left_record.get("outputs") or [],
                right_record.get("outputs") or [],
                strict=False)
        ):
            left_ids = left_output.get("token_ids") or []
            right_ids = right_output.get("token_ids") or []
            prefix_len = _common_prefix_len(left_ids, right_ids)
            prefix_lengths.append({
                "record_index": record_idx,
                "output_index": output_idx,
                "common_prefix_len": prefix_len,
            })
            if prefix_len <= 0:
                continue
            left_maps = _top_logprob_maps(left_output.get("logprobs") or [])
            right_maps = _top_logprob_maps(right_output.get("logprobs") or [])
            for left_step, right_step in zip(
                left_maps[:prefix_len],
                right_maps[:prefix_len],
                strict=False,
            ):
                compared_steps += 1
                left_keys = set(left_step)
                right_keys = set(right_step)
                missing_left += len(right_keys - left_keys)
                missing_right += len(left_keys - right_keys)
                for key in left_keys & right_keys:
                    diff = abs(left_step[key] - right_step[key])
                    max_diffs.append(diff)
                    total_sum += diff
                    common_value_count += 1
    if not max_diffs and common_value_count == 0:
        return {
            "prefix_lengths": prefix_lengths,
            "compared_steps": compared_steps,
            "common_value_count": 0,
            "missing_left": missing_left,
            "missing_right": missing_right,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
    return {
        "prefix_lengths": prefix_lengths,
        "compared_steps": compared_steps,
        "common_value_count": common_value_count,
        "missing_left": missing_left,
        "missing_right": missing_right,
        "max_abs_diff": max(max_diffs),
        "mean_abs_diff": total_sum / common_value_count
        if common_value_count
        else None,
    }


def _record_actual_prompt_logprobs(record: dict[str, Any]) -> list[float]:
    prompt_token_ids = record.get("prompt_token_ids") or []
    prompt_logprobs = record.get("prompt_logprobs") or []
    values: list[float] = []
    for pos, step in enumerate(prompt_logprobs):
        if pos == 0 or step is None or pos >= len(prompt_token_ids):
            continue
        entry = step.get(str(prompt_token_ids[pos]))
        if entry is not None:
            values.append(float(entry["logprob"]))
    return values


def _compare_prompt_logprobs(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    diffs: list[float] = []
    same_count = len(left["records"]) == len(right["records"])
    for left_record, right_record in zip(
        left["records"], right["records"], strict=False
    ):
        left_values = _record_actual_prompt_logprobs(left_record)
        right_values = _record_actual_prompt_logprobs(right_record)
        same_count = same_count and len(left_values) == len(right_values)
        for left_value, right_value in zip(left_values, right_values, strict=False):
            diffs.append(abs(left_value - right_value))
    if not diffs:
        return None
    return {
        "same_count": same_count,
        "count": len(diffs),
        "max_abs_diff": max(diffs),
        "mean_abs_diff": sum(diffs) / len(diffs),
    }


def _compare_prompt_perplexity(
    left: dict[str, Any],
    right: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for idx, (left_record, right_record) in enumerate(
        zip(left["records"], right["records"], strict=False)
    ):
        left_ppl = (left_record.get("prompt_metrics") or {}).get("perplexity")
        right_ppl = (right_record.get("prompt_metrics") or {}).get("perplexity")
        result.append({
            "index": idx,
            "left_perplexity": left_ppl,
            "right_perplexity": right_ppl,
            "abs_diff": None
            if left_ppl is None or right_ppl is None
            else abs(float(left_ppl) - float(right_ppl)),
        })
    return result


def _load_logits_dumps(path: Path) -> list[dict[str, Any]]:
    import torch

    entries = []
    for file_path in sorted(path.glob("sampler_logits_*.pt")):
        payload = torch.load(file_path, map_location="cpu", weights_only=False)
        entries.append({
            "path": str(file_path),
            "step": int(payload["step"]),
            "stage": payload.get("stage"),
            "shape": list(payload.get("shape", tuple(payload["logits"].shape))),
            "dtype": payload.get("dtype"),
            "all_greedy": bool(payload.get("all_greedy")),
            "logits": payload["logits"].float(),
        })
    entries.sort(key=lambda item: item["step"])
    return entries


def _compare_logits_dirs(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    import torch

    left_entries = _load_logits_dumps(left_dir)
    right_entries = _load_logits_dumps(right_dir)
    comparisons = []
    global_max = 0.0
    total_sum = 0.0
    total_count = 0
    same_dump_count = len(left_entries) == len(right_entries)
    all_shape_equal = same_dump_count
    all_argmax_equal = same_dump_count
    num_argmax_mismatch = 0
    first_argmax_mismatch = None
    for idx, (left, right) in enumerate(zip(left_entries, right_entries, strict=False)):
        shape_equal = tuple(left["logits"].shape) == tuple(right["logits"].shape)
        all_shape_equal = all_shape_equal and shape_equal
        payload = {
            "index": idx,
            "left_step": left["step"],
            "right_step": right["step"],
            "left_stage": left["stage"],
            "right_stage": right["stage"],
            "shape_equal": shape_equal,
            "shape": left["shape"],
        }
        if shape_equal:
            diff = (left["logits"] - right["logits"]).abs()
            max_diff = float(diff.max().item()) if diff.numel() else 0.0
            mean_diff = float(diff.mean().item()) if diff.numel() else 0.0
            left_argmax = left["logits"].argmax(dim=-1)
            right_argmax = right["logits"].argmax(dim=-1)
            argmax_equal = bool(torch.equal(left_argmax, right_argmax))
            all_argmax_equal = all_argmax_equal and argmax_equal
            if not argmax_equal:
                num_argmax_mismatch += 1
                if first_argmax_mismatch is None:
                    first_argmax_mismatch = idx
            global_max = max(global_max, max_diff)
            total_sum += float(diff.sum().item())
            total_count += int(diff.numel())
            payload.update({
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
                "argmax_equal": argmax_equal,
            })
        else:
            all_argmax_equal = False
        comparisons.append(payload)
    return {
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "same_dump_count": same_dump_count,
        "left_dump_count": len(left_entries),
        "right_dump_count": len(right_entries),
        "all_shape_equal": all_shape_equal,
        "all_argmax_equal": all_argmax_equal,
        "num_argmax_mismatch": num_argmax_mismatch,
        "first_argmax_mismatch": first_argmax_mismatch,
        "max_abs_diff": global_max,
        "mean_abs_diff": total_sum / total_count if total_count else None,
        "comparisons": comparisons,
    }


def _first_mismatch(left: list[int], right: list[int]) -> dict[str, Any] | None:
    for idx, (left_id, right_id) in enumerate(zip(left, right, strict=False)):
        if left_id != right_id:
            return {"index": idx, "left": left_id, "right": right_id}
    if len(left) != len(right):
        idx = min(len(left), len(right))
        return {
            "index": idx,
            "left": None if len(left) <= len(right) else left[idx],
            "right": None
            if len(right) <= len(left)
            else right[idx],
        }
    return None


def _without_records(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "records"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--left-logits-dir", type=Path)
    parser.add_argument("--right-logits-dir", type=Path)
    parser.add_argument(
        "--require-model-quality-gate",
        action="store_true",
        help=(
            "Fail compare mode unless model_quality_gate.label is model-pass. "
            "Use this for default-enable acceptance, not for exploratory token "
            "equality checks."
        ),
    )
    parser.add_argument(
        "--allow-token-diff-for-model-quality-gate",
        action="store_true",
        help=(
            "Record deterministic token differences as warning evidence instead "
            "of failing the model quality gate. Use only with explicit numeric "
            "or dataset-level acceptance bounds."
        ),
    )
    parser.add_argument(
        "--allow-sampler-argmax-diff-for-model-quality-gate",
        action="store_true",
        help=(
            "Record sampler-logits argmax differences as warning evidence "
            "instead of failing the model quality gate. Use only when low-margin "
            "token flips are covered by separate numeric/dataset gates."
        ),
    )
    parser.add_argument(
        "--allow-missing-sampler-logits-for-model-quality-gate",
        action="store_true",
        help=(
            "Record missing sampler-logits dumps as warning evidence instead "
            "of keeping the model quality gate pending. Use for dataset or "
            "serving quality gates where logits dumps are intentionally absent."
        ),
    )
    parser.add_argument("--max-prompt-logprob-diff-for-accept", type=float)
    parser.add_argument("--max-output-logprob-diff-for-accept", type=float)
    parser.add_argument("--max-output-top-logprob-diff-for-accept", type=float)
    parser.add_argument("--max-prompt-perplexity-abs-diff-for-accept", type=float)
    parser.add_argument("--max-sampler-logits-diff-for-accept", type=float)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument(
        "--input-len",
        type=int,
        help=(
            "Create a tokenized prompt by repeating --prompt-base to exactly "
            "this many tokens. This matches benchmark_sm70_decode.py."
        ),
    )
    parser.add_argument(
        "--prompt-base",
        default=(
            "This fixed benchmark prompt is used to create a deterministic "
            "tokenized input for single-request decode measurement. "
        ),
    )
    parser.add_argument(
        "--sequential-prompts",
        action="store_true",
        help="Generate each prompt in a separate llm.generate call.",
    )
    cuda_profiler_group = parser.add_mutually_exclusive_group()
    cuda_profiler_group.add_argument(
        "--cuda-profiler-capture-generate",
        action="store_true",
        help=(
            "Call cudaProfilerStart/Stop around llm.generate so nsys can "
            "capture only the measured generation window."
        ),
    )
    cuda_profiler_group.add_argument(
        "--cuda-profiler-capture-decode-after-prefix-warmup",
        action="store_true",
        help=(
            "Prime one prompt with prefix caching while the CUDA profiler is "
            "off, then capture only the second matching llm.generate call. "
            "Requires one non-sequential prompt and CUDA graphs."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help=(
            "Stop string for generation. May be passed multiple times. "
            "Useful for HumanEval-style code-completion probes."
        ),
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        help="Optional per-request SamplingParams seed for deterministic sampling.",
    )
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--prompt-logprobs", type=int, default=0)
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
            "default LLM_CLASS cap would otherwise compile a wider token range."
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
    parser.add_argument("--engine-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.compare is not None:
        return _compare(args)
    if args.model is None:
        raise ValueError("--model is required in dump mode")
    return _dump(args)


if __name__ == "__main__":
    raise SystemExit(main())
