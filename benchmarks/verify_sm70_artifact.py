# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify SM70 migration benchmark artifacts carry required evidence.

This tool is intentionally dependency-free. It checks benchmark JSON metadata
and optional runtime logs without importing vLLM, loading models, or touching
CUDA. Use it as a lightweight guard before treating an SM70 benchmark artifact
as route-hit, quality, or throughput evidence.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_REQUIRED_POLICIES = (
    "sm70_tune_policy",
    "sm70_turbomind_policy",
    "sm70_attention_policy",
    "sm70_gdn_fla_policy",
    "sm70_moe_policy",
    "sm70_sampling_policy",
)

FULL_FLASH_V100_JSON_EXPECTATIONS = (
    ("sm70_attention_policy.selector_enabled_effective", True),
    ("sm70_attention_policy.prefill_use_triton_effective", False),
    ("sm70_attention_policy.allow_triton_fallback_effective", False),
    ("sm70_attention_policy.decode_scalar_paged_effective", True),
    ("sm70_attention_policy.full_flash_default_policy", True),
)

FULL_FLASH_V100_REQUIRED_LOGS = (
    "FLASH_ATTN_V100 prefill path active",
    "FLASH_ATTN_V100 decode path active",
)

FULL_FLASH_V100_REJECTED_LOGS = (
    "FLASH_ATTN_V100 prefill uses explicit Triton diagnostic fallback",
    "FLASH_ATTN_V100 decode fallback to Triton",
    "set VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1",
)

SM70_FP8_KV_CACHE_ROUTE_JSON_EXPECTATIONS = (
    ("sm70_attention_policy.fp8_kv_cache_requested_effective", True),
    ("sm70_attention_policy.fp8_kv_cache_full_flash_policy", True),
)

SM70_FP8_KV_CACHE_ROUTE_REQUIRED_LOGS = (
    "SM70 FP8 KV cache C++ reshape_and_cache_flash path enabled",
    "FLASH_ATTN_V100 FP8 KV cache decode path active",
    "route=scalar_paged",
)

SM70_FP8_KV_CACHE_ROUTE_REJECTED_LOGS = (
    "FLASH_ATTN_V100 no-prefix prefill is reading paged KV cache for strict "
    "input-source diagnostics",
    "FLASH_ATTN_V100 decode-as-paged-prefill path active",
    "FLASH_ATTN_V100 decode dense-cache path active",
    "FLASH_ATTN_V100 decode dense-reference path active",
    "route=no_prefix_paged_cache",
    "route=decode_as_paged_prefill",
    "route=dense_cache_bridge",
    "route=dense_reference_bridge",
)

TURBOMIND_DEFAULT_JSON_EXPECTATIONS = (
    ("sm70_turbomind_policy.accepted_dense_default_policy", True),
)

AWQ_TURBOMIND_DENSE_JSON_EXPECTATIONS = (
    ("sm70_turbomind_policy.awq_turbomind_effective", True),
)

AWQ_TURBOMIND_DENSE_REQUIRED_LOGS = (
    "SM70 AWQ TurboMind dense path enabled",
)

FP8_TURBOMIND_DENSE_JSON_EXPECTATIONS = (
    ("sm70_turbomind_policy.fp8_turbomind_effective", True),
    ("sm70_turbomind_policy.fp8_dequant_fallback_effective", True),
)

FP8_TURBOMIND_DENSE_REQUIRED_LOGS = (
    "SM70 FP8 TurboMind W8A16 dense path enabled",
)

FP8_0DOT3_DENSE_DEQUANT_JSON_EXPECTATIONS = (
    ("sm70_turbomind_policy.fp8_turbomind_effective", False),
    ("sm70_turbomind_policy.fp8_dequant_fallback_effective", True),
)

FP8_0DOT3_DENSE_DEQUANT_REQUIRED_LOGS = (
    "SM70 FP8 fallback enabled: FP8 block weights are dequantized",
)

FP8_0DOT3_DENSE_DEQUANT_REJECTED_REGEXES = (
    r"\[fp8\.py:\d+\].*SM70 FP8 TurboMind W8A16 dense path enabled",
)

AWQ_MOE_SAFE_ROUTE_JSON_EXPECTATIONS = (
    ("sm70_turbomind_policy.accepted_awq_moe_default_policy", True),
    ("sm70_turbomind_policy.awq_moe_disable_effective", False),
    ("sm70_turbomind_policy.awq_moe_batched_gemm_effective", False),
    ("sm70_turbomind_policy.awq_moe_legacy_single_token_compact_effective",
     False),
    ("sm70_moe_policy.single_token_unpermute_fastpath_effective", True),
)

AWQ_MOE_SAFE_ROUTE_REQUIRED_LOGS = (
    "SM70 AWQ MoE TurboMind per-expert dense path enabled",
    "SM70 AWQ MoE CUDA-graph-safe dense-stage path enabled",
    "SM70 AWQ MoE single-token active-expert dense path enabled",
    "SM70 AWQ MoE single-token weighted-reduce path enabled",
)

AWQ_MOE_BATCHED_SAFE_ROUTE_JSON_EXPECTATIONS = (
    ("sm70_turbomind_policy.accepted_awq_moe_default_policy", True),
    ("sm70_turbomind_policy.awq_moe_disable_effective", False),
    ("sm70_turbomind_policy.awq_moe_batched_gemm_effective", True),
    ("sm70_turbomind_policy.awq_moe_legacy_single_token_compact_effective",
     False),
)

AWQ_MOE_BATCHED_SAFE_ROUTE_REQUIRED_LOGS = (
    "SM70 AWQ MoE TurboMind batched path enabled",
    "SM70 AWQ MoE batched W13 using per-expert dispatch selection",
)

AWQ_MOE_BATCHED_SAFE_ROUTE_REJECTED_LOGS = (
    "SM70 AWQ MoE 0.0.3 legacy single-token compact path enabled",
)

AWQ_MOE_0DOT3_BASELINE_ROUTE_JSON_EXPECTATIONS = (
    ("sm70_turbomind_policy.awq_moe_0dot3_baseline_policy", True),
    ("sm70_turbomind_policy.awq_moe_disable_effective", False),
    ("sm70_turbomind_policy.awq_moe_batched_gemm_effective", True),
    ("sm70_turbomind_policy.awq_moe_legacy_single_token_compact_effective",
     True),
)

AWQ_MOE_0DOT3_BASELINE_ROUTE_REQUIRED_LOGS = (
    "SM70 AWQ MoE TurboMind batched path enabled",
    "SM70 AWQ MoE 0.0.3 legacy single-token compact path enabled",
)

FP8_MOE_SAFE_ROUTE_JSON_EXPECTATIONS = (
    ("sm70_turbomind_policy.accepted_fp8_moe_default_policy", True),
    ("sm70_turbomind_policy.fp8_moe_dequant_fallback_effective", False),
    ("sm70_turbomind_policy.fp8_moe_batched_gemm_effective", True),
)

FP8_MOE_SAFE_ROUTE_REQUIRED_LOGS = (
    "SM70 FP8 MoE TurboMind batched path enabled",
)

FP8_MOE_0DOT3_FALLBACK_ROUTE_JSON_EXPECTATIONS = (
    ("sm70_turbomind_policy.fp8_moe_0dot3_dequant_fallback_policy", True),
    ("sm70_turbomind_policy.fp8_dequant_fallback_effective", True),
    ("sm70_turbomind_policy.fp8_moe_dequant_fallback_effective", True),
    ("sm70_turbomind_policy.fp8_moe_batched_gemm_effective", False),
)

FP8_MOE_0DOT3_FALLBACK_ROUTE_REQUIRED_LOGS = (
    "SM70 FP8 MoE fallback enabled",
    "unquantized Triton MoE path",
    "Using SM70 0.0.3 unquantized MoE default config",
)

FP8_MOE_0DOT3_FALLBACK_ROUTE_REJECTED_LOGS = (
    "Using SM70 0.0.3 unquantized MoE functional fused_experts path",
)

SM70_BREAKABLE_GRAPH_ROUTE_JSON_EXPECTATIONS = (
    ("sm70_graph_policy.sm70_breakable_requested_effective", True),
    ("sm70_graph_policy.breakable_cudagraph_effective", True),
    ("sm70_graph_policy.sm70_breakable_mapping_effective", True),
)

SM70_BREAKABLE_GRAPH_ROUTE_REQUIRED_LOGS = (
    "Breakable CUDA graph enabled",
    "Graph capturing finished",
)

SM70_FLASH_V100_0DOT3_COMPILE_GRAPH_JSON_EXPECTATIONS = (
    ("sm70_graph_policy.sm70_flash_v100_0dot3_compile_graph_effective", True),
    ("sm70_graph_policy.old_0dot3_compile_graph_policy", True),
    ("sm70_graph_policy.flash_v100_decode_graph_no_compile_effective", False),
)

SM70_FLASH_V100_0DOT3_COMPILE_GRAPH_REQUIRED_LOGS = (
    "Using SM70 Flash-V100 0.0.3 compile CUDA graph policy",
    "cudagraph_mode=FULL_AND_PIECEWISE",
    "Graph capturing finished",
)

SM70_FLASH_V100_0DOT3_COMPILE_GRAPH_REJECTED_LOGS = (
    "Using SM70 Flash-V100 no-compile decode CUDA graph policy",
    "Using SM70 Flash-V100 no-compile decode CUDA graph policy: "
    "mode=NONE, cudagraph_mode=FULL_DECODE_ONLY",
)

SM70_CUSTOM_ALLREDUCE_ROUTE_JSON_EXPECTATIONS = (
    ("sm70_comm_policy.custom_all_reduce_enabled_effective", True),
    ("sm70_comm_policy.production_custom_allreduce_default_policy", True),
)

SM70_CUSTOM_ALLREDUCE_ROUTE_REQUIRED_LOGS = (
    "Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) "
    "for group 'tp:0'",
)

SM70_CUSTOM_ALLREDUCE_ROUTE_REJECTED_LOGS = (
    "Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'",
    "disable_custom_all_reduce=True",
    '"disable_custom_all_reduce": true',
)

SM70_ALL_REDUCE_SUM2_ROUTE_JSON_EXPECTATIONS = (
    ("sm70_comm_policy.custom_all_reduce_enabled_effective", True),
    ("sm70_comm_policy.all_reduce_sum2_requested_effective", True),
    ("sm70_moe_policy.moe_add_allreduce_effective", True),
)

SM70_ALL_REDUCE_SUM2_ROUTE_REQUIRED_LOGS = (
    "SM70 MoE shared+routed all_reduce_sum2 candidate selected",
)

SM70_ALL_REDUCE_SUM2_ROUTE_REQUIRED_REGEXES = (
    r"SM70 custom all_reduce_sum2 op reached[^\n]*capture=active",
)

DECODE_TIMING_SUMMARY_FIELDS = (
    "output_tps",
    "first_token_latency",
    "prefill_time",
    "decode_time",
    "steady_decode_tps",
    "tpot_seconds",
)

DECODE_TIMING_REQUEST_METRIC_FIELDS = (
    "first_token_latency",
    "prefill_time",
    "decode_time",
    "steady_decode_tps",
    "tpot_seconds",
)

INVENTORY_CLASSIFICATION_SECTIONS = (
    "old_only_runtime_env",
    "old_only_keyword_files",
    "old_only_torch_ops",
)


def _parse_expected_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _get_json_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index < len(current):
                current = current[index]
                continue
        raise KeyError(path)
    return current


def _parse_expectation(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(
            f"Expected PATH=VALUE for --expect-json, got {raw!r}"
        )
    path, expected = raw.split("=", 1)
    if not path:
        raise ValueError(f"Expected non-empty PATH in {raw!r}")
    return path, _parse_expected_value(expected)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_logs(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _policy_location(payload: Any, policy: str) -> str | None:
    if isinstance(payload, dict) and isinstance(payload.get(policy), dict):
        return "top-level"
    if not isinstance(payload, dict):
        return None
    left_meta = payload.get("left_meta")
    right_meta = payload.get("right_meta")
    if (
        isinstance(left_meta, dict)
        and isinstance(right_meta, dict)
        and isinstance(left_meta.get(policy), dict)
        and isinstance(right_meta.get(policy), dict)
    ):
        return "compare-meta"
    return None


def _awq_moe_safe_tune_gate_for_meta(meta: Any) -> dict[str, Any]:
    if not isinstance(meta, dict):
        return {
            "passed": False,
            "reason": "metadata is missing or is not an object",
        }
    tune_policy = meta.get("sm70_tune_policy")
    turbomind_policy = meta.get("sm70_turbomind_policy")
    if not isinstance(tune_policy, dict):
        return {
            "passed": False,
            "reason": "sm70_tune_policy is missing or is not an object",
        }
    if not isinstance(turbomind_policy, dict):
        return {
            "passed": False,
            "reason": "sm70_turbomind_policy is missing or is not an object",
        }

    raw = tune_policy.get("VLLM_SM70_AWQ_TUNE_SMALL_SHAPES")
    tune_safe_default = tune_policy.get(
        "awq_moe_safe_default_selector_effective"
    )
    turbomind_safe_default = turbomind_policy.get(
        "awq_moe_safe_default_selector_effective"
    )
    pinned = raw == "0"
    source_safe_unset = (
        raw is None
        and tune_safe_default is True
        and turbomind_safe_default is True
    )
    return {
        "passed": pinned or source_safe_unset,
        "raw": raw,
        "tune_safe_default": tune_safe_default,
        "turbomind_safe_default": turbomind_safe_default,
        "reason": (
            "accepted raw tune0 pin"
            if pinned
            else (
                "accepted source-level unset-env safe default"
                if source_safe_unset
                else (
                    "requires raw '0' or raw null plus both "
                    "awq_moe_safe_default_selector_effective fields"
                )
            )
        ),
    }


def _awq_moe_safe_tune_gate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "passed": False,
            "location": None,
            "checks": [],
        }

    if isinstance(payload.get("sm70_tune_policy"), dict):
        check = _awq_moe_safe_tune_gate_for_meta(payload)
        return {
            "passed": check["passed"],
            "location": "top-level",
            "checks": [check],
        }

    left_meta = payload.get("left_meta")
    right_meta = payload.get("right_meta")
    if isinstance(left_meta, dict) and isinstance(right_meta, dict):
        checks = [
            {
                "side": "left_meta",
                **_awq_moe_safe_tune_gate_for_meta(left_meta),
            },
            {
                "side": "right_meta",
                **_awq_moe_safe_tune_gate_for_meta(right_meta),
            },
        ]
        return {
            "passed": all(check["passed"] for check in checks),
            "location": "compare-meta",
            "checks": checks,
        }

    return {
        "passed": False,
        "location": None,
        "checks": [{
            "passed": False,
            "reason": "no top-level or compare metadata policies found",
        }],
    }


def _evaluate_json_expectation(
    payload: Any,
    path: str,
    expected: Any,
) -> tuple[Any, bool, bool, str | None]:
    try:
        actual = _get_json_path(payload, path)
        return actual, True, actual == expected, "top-level"
    except KeyError:
        pass

    if not isinstance(payload, dict):
        return None, False, False, None
    left_meta = payload.get("left_meta")
    right_meta = payload.get("right_meta")
    if not isinstance(left_meta, dict) or not isinstance(right_meta, dict):
        return None, False, False, None
    try:
        left_actual = _get_json_path(left_meta, path)
        right_actual = _get_json_path(right_meta, path)
    except KeyError:
        return None, False, False, None

    actual = {
        "left_meta": left_actual,
        "right_meta": right_actual,
    }
    return (
        actual,
        True,
        left_actual == expected and right_actual == expected,
        "compare-meta",
    )


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _check_summary_timing(summary: Any, name: str) -> list[str]:
    if not isinstance(summary, dict):
        return [f"{name}.summary is missing or is not an object"]

    failures = []
    for field in DECODE_TIMING_SUMMARY_FIELDS:
        values = summary.get(f"{field}_values")
        mean = summary.get(f"{field}_mean")
        if not isinstance(values, list) or not values:
            failures.append(f"{name}.summary.{field}_values missing or empty")
            continue
        if not all(_is_positive_number(value) for value in values):
            failures.append(f"{name}.summary.{field}_values contains non-positive")
        if not _is_positive_number(mean):
            failures.append(f"{name}.summary.{field}_mean missing or non-positive")
    return failures


def _check_repeat_timing(repeats: Any, name: str) -> list[str]:
    if not isinstance(repeats, list) or not repeats:
        return [f"{name}.repeats missing or empty"]

    failures = []
    for index, repeat in enumerate(repeats):
        if not isinstance(repeat, dict):
            failures.append(f"{name}.repeats[{index}] is not an object")
            continue
        metrics = repeat.get("request_metrics")
        if not isinstance(metrics, dict):
            failures.append(f"{name}.repeats[{index}].request_metrics missing")
            continue
        for field in DECODE_TIMING_REQUEST_METRIC_FIELDS:
            if not _is_positive_number(metrics.get(field)):
                failures.append(
                    f"{name}.repeats[{index}].request_metrics.{field} "
                    "missing or non-positive"
                )
    return failures


def _decode_timing_failures(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["artifact JSON root is not an object"]

    failures = []
    failures.extend(_check_summary_timing(payload.get("summary"), "root"))
    failures.extend(_check_repeat_timing(payload.get("repeats"), "root"))

    cases = payload.get("cases")
    if cases is None:
        return failures
    if not isinstance(cases, list) or not cases:
        return failures + ["cases is present but missing or empty"]

    for index, case in enumerate(cases):
        name = f"cases[{index}]"
        if not isinstance(case, dict):
            failures.append(f"{name} is not an object")
            continue
        failures.extend(_check_summary_timing(case.get("summary"), name))
        failures.extend(_check_repeat_timing(case.get("repeats"), name))
    return failures


def _model_quality_failures(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["artifact JSON root is not an object"]
    gate = payload.get("model_quality_gate")
    if not isinstance(gate, dict):
        return ["model_quality_gate is missing or is not an object"]

    failures = []
    if gate.get("label") != "model-pass":
        failures.append(
            "model_quality_gate.label is not model-pass "
            f"({gate.get('label')!r})"
        )
    if gate.get("default_acceptance") != "model-level gate passed":
        failures.append(
            "model_quality_gate.default_acceptance is not model-level gate "
            f"passed ({gate.get('default_acceptance')!r})"
        )

    failed_evidence = gate.get("failed_evidence")
    if failed_evidence:
        failures.append("model_quality_gate.failed_evidence is not empty")
    pending_evidence = gate.get("pending_evidence")
    if pending_evidence:
        failures.append("model_quality_gate.pending_evidence is not empty")

    checks = gate.get("checks")
    bounds = gate.get("bounds")
    if not isinstance(checks, dict):
        failures.append("model_quality_gate.checks is missing or is not an object")
        checks = {}
    if not isinstance(bounds, dict):
        failures.append("model_quality_gate.bounds is missing or is not an object")
        bounds = {}

    if checks.get("token_equal") is not True:
        failures.append("model_quality_gate.checks.token_equal is not true")
    if checks.get("sampler_logits_all_argmax_equal") is not True:
        failures.append(
            "model_quality_gate.checks.sampler_logits_all_argmax_equal is not true"
        )

    for name, bound in bounds.items():
        value = checks.get(name)
        if bound is None:
            failures.append(f"model_quality_gate.bounds.{name} is missing")
            continue
        if value is None:
            failures.append(f"model_quality_gate.checks.{name} is missing")
            continue
        if value > bound:
            failures.append(
                f"model_quality_gate.checks.{name}={value} exceeds bound {bound}"
            )
    return failures


def _inventory_complete_failures(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["artifact JSON root is not an object"]
    gate = payload.get("inventory_gate")
    if not isinstance(gate, dict):
        return ["inventory_gate is missing or is not an object"]

    failures = []
    if gate.get("label") != "inventory-complete":
        failures.append(
            "inventory_gate.label is not inventory-complete "
            f"({gate.get('label')!r})"
        )
    if gate.get("complete") is not True:
        failures.append("inventory_gate.complete is not true")

    old_only_env_vars = gate.get("old_only_env_vars")
    if old_only_env_vars:
        failures.append(
            "inventory_gate.old_only_env_vars is not empty: "
            f"{old_only_env_vars!r}"
        )
    if gate.get("old_only_env_vars_count") != 0:
        failures.append(
            "inventory_gate.old_only_env_vars_count is not 0 "
            f"({gate.get('old_only_env_vars_count')!r})"
        )

    for section_name in INVENTORY_CLASSIFICATION_SECTIONS:
        section = gate.get(section_name)
        if not isinstance(section, dict):
            failures.append(
                f"inventory_gate.{section_name} is missing or is not an object"
            )
            continue
        unclassified = section.get("unclassified")
        if unclassified:
            failures.append(
                f"inventory_gate.{section_name}.unclassified is not empty: "
                f"{unclassified!r}"
            )
        if section.get("unclassified_count") != 0:
            failures.append(
                f"inventory_gate.{section_name}.unclassified_count is not 0 "
                f"({section.get('unclassified_count')!r})"
            )
    return failures


def _verify(
    payload: Any,
    *,
    json_path: Path,
    log_paths: list[Path],
    required_policies: list[str],
    json_expectations: list[tuple[str, Any]],
    log_expectations: list[str],
    log_regex_expectations: list[str],
    rejected_log_expectations: list[str],
    rejected_log_regex_expectations: list[str],
    require_decode_timing: bool,
    require_model_quality_pass: bool,
    require_inventory_complete: bool,
) -> dict[str, Any]:
    policy_results = []
    missing_policies = []
    for policy in required_policies:
        location = _policy_location(payload, policy)
        present = location is not None
        if not present:
            missing_policies.append(policy)
        policy_results.append({
            "policy": policy,
            "present": present,
            "location": location,
        })

    json_results = []
    failed_json_expectations = []
    for path, expected in json_expectations:
        actual, found, matched, location = _evaluate_json_expectation(
            payload, path, expected
        )
        result = {
            "path": path,
            "expected": expected,
            "actual": actual,
            "found": found,
            "matched": matched,
            "location": location,
        }
        json_results.append(result)
        if not matched:
            failed_json_expectations.append(result)

    logs = _read_logs(log_paths) if log_paths else ""
    log_results = []
    missing_log_texts = []
    for text in log_expectations:
        present = text in logs
        if not present:
            missing_log_texts.append(text)
        log_results.append({
            "text": text,
            "present": present,
        })

    log_regex_results = []
    missing_log_regexes = []
    for pattern in log_regex_expectations:
        present = re.search(pattern, logs) is not None
        if not present:
            missing_log_regexes.append(pattern)
        log_regex_results.append({
            "pattern": pattern,
            "present": present,
        })

    rejected_log_results = []
    rejected_log_texts = []
    for text in rejected_log_expectations:
        present = text in logs
        if present:
            rejected_log_texts.append(text)
        rejected_log_results.append({
            "text": text,
            "present": present,
        })

    rejected_log_regex_results = []
    rejected_log_regexes = []
    for pattern in rejected_log_regex_expectations:
        present = re.search(pattern, logs) is not None
        if present:
            rejected_log_regexes.append(pattern)
        rejected_log_regex_results.append({
            "pattern": pattern,
            "present": present,
        })

    decode_timing_failures = (
        _decode_timing_failures(payload) if require_decode_timing else []
    )
    model_quality_failures = (
        _model_quality_failures(payload) if require_model_quality_pass else []
    )
    inventory_complete_failures = (
        _inventory_complete_failures(payload)
        if require_inventory_complete else []
    )

    passed = not (
        missing_policies
        or failed_json_expectations
        or missing_log_texts
        or missing_log_regexes
        or rejected_log_texts
        or rejected_log_regexes
        or decode_timing_failures
        or model_quality_failures
        or inventory_complete_failures
    )
    return {
        "passed": passed,
        "json_path": str(json_path),
        "log_paths": [str(path) for path in log_paths],
        "policy_results": policy_results,
        "missing_policies": missing_policies,
        "json_expectations": json_results,
        "failed_json_expectations": failed_json_expectations,
        "log_expectations": log_results,
        "missing_log_texts": missing_log_texts,
        "log_regex_expectations": log_regex_results,
        "missing_log_regexes": missing_log_regexes,
        "rejected_log_expectations": rejected_log_results,
        "rejected_log_texts": rejected_log_texts,
        "rejected_log_regex_expectations": rejected_log_regex_results,
        "rejected_log_regexes": rejected_log_regexes,
        "decode_timing_required": require_decode_timing,
        "decode_timing_failures": decode_timing_failures,
        "model_quality_pass_required": require_model_quality_pass,
        "model_quality_failures": model_quality_failures,
        "inventory_complete_required": require_inventory_complete,
        "inventory_complete_failures": inventory_complete_failures,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify SM70 benchmark JSON policy fields and route logs."
    )
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Benchmark result JSON to verify.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        action="append",
        default=[],
        help="Runtime log to scan. May be passed more than once.",
    )
    parser.add_argument(
        "--require-policy",
        action="append",
        default=[],
        help=(
            "Additional top-level JSON policy object that must be present. "
            "Default SM70 policies are always required."
        ),
    )
    parser.add_argument(
        "--expect-json",
        action="append",
        default=[],
        help=(
            "Require an exact JSON value match, written as PATH=VALUE. "
            "PATH is dot-separated and VALUE is parsed as JSON when possible."
        ),
    )
    parser.add_argument(
        "--expect-log",
        action="append",
        default=[],
        help="Require literal text to appear in one of the supplied logs.",
    )
    parser.add_argument(
        "--expect-log-regex",
        action="append",
        default=[],
        help="Require a Python regular expression to match supplied logs.",
    )
    parser.add_argument(
        "--reject-log",
        action="append",
        default=[],
        help="Fail if literal text appears in one of the supplied logs.",
    )
    parser.add_argument(
        "--reject-log-regex",
        action="append",
        default=[],
        help="Fail if a Python regular expression matches supplied logs.",
    )
    parser.add_argument(
        "--require-full-flash-v100",
        action="store_true",
        help=(
            "Require the artifact to prove full FlashAttention-V100 policy and "
            "logs: Flash selector enabled, Triton prefill/fallback disabled, "
            "Flash prefill/decode logs present, and mixed fallback logs absent."
        ),
    )
    parser.add_argument(
        "--require-sm70-fp8-kv-cache-route",
        action="store_true",
        help=(
            "Require an explicit SM70 FP8 KV cache route on full Flash-V100: "
            "JSON must show kv_cache_dtype=fp8* and full Flash policy, and "
            "logs must show Flash-V100 prefill, FP8 cache write, and FP8 KV "
            "scalar-paged decode. No-prefix paged-cache diagnostics, "
            "dense/reference/debug bridges, and FP8 model weight quantization "
            "alone are not accepted."
        ),
    )
    parser.add_argument(
        "--require-decode-timing",
        action="store_true",
        help=(
            "Require benchmark_sm70_decode-style timing evidence: per-repeat "
            "request metrics plus summary values for TTFT/prefill and pure "
            "steady decode TPS/TPOT. This rejects artifacts where request "
            "metrics were disabled or output_tps is the only speed signal."
        ),
    )
    parser.add_argument(
        "--require-model-quality-pass",
        action="store_true",
        help=(
            "Require benchmark_sm70_model_tokens.py compare-mode output to "
            "have model_quality_gate.label=model-pass, model-level default "
            "acceptance, exact deterministic token equality, sampler argmax "
            "equality, configured numeric bounds, and no failed/pending "
            "quality evidence. Compare artifacts may carry SM70 policy blocks "
            "inside left_meta/right_meta instead of at the top level."
        ),
    )
    parser.add_argument(
        "--require-inventory-complete",
        action="store_true",
        help=(
            "Require benchmark_sm70_source_inventory.py output to prove that "
            "old-only SM70/V100 env vars, runtime env vars, keyword files, and "
            "custom torch ops are fully classified. This source-inventory "
            "artifact is not a benchmark route artifact, so default SM70 "
            "policy-block checks are skipped when this flag is used."
        ),
    )
    parser.add_argument(
        "--require-turbomind-default-policy",
        action="store_true",
        help=(
            "Require the accepted SM70 TurboMind dense default policy. This "
            "proves only policy state; pair it with AWQ/FP8 dense route gates "
            "and runtime logs for actual route-hit evidence."
        ),
    )
    parser.add_argument(
        "--require-awq-turbomind-dense",
        action="store_true",
        help=(
            "Require SM70 AWQ TurboMind dense policy and its runtime route log."
        ),
    )
    parser.add_argument(
        "--require-fp8-turbomind-dense",
        action="store_true",
        help=(
            "Require SM70 FP8 TurboMind W8A16 dense policy, dequant fallback "
            "policy, and runtime route log."
        ),
    )
    parser.add_argument(
        "--require-fp8-0dot3-dense-dequant-route",
        action="store_true",
        help=(
            "Require the 0.0.3 SM70 FP8 dense fallback lane: dense FP8 block "
            "weights are dequantized to fp16 at load time, and the dense "
            "TurboMind W8A16 route is not active. This is the old 35B-A3B-FP8 "
            "baseline lane and is intentionally separate from the 27B-FP8 "
            "TurboMind dense lane."
        ),
    )
    parser.add_argument(
        "--require-awq-moe-safe-route",
        action="store_true",
        help=(
            "Require the accepted SM70 AWQ MoE dense-safe route: batched MoE "
            "disabled, "
            "AWQ MoE enabled, either VLLM_SM70_AWQ_TUNE_SMALL_SHAPES=0 "
            "or source-level unset-env safe-default evidence, and per-expert "
            "dense, graph-safe dense-stage, and single-token active-expert "
            "runtime logs present."
        ),
    )
    parser.add_argument(
        "--require-awq-moe-batched-safe-route",
        action="store_true",
        help=(
            "Require the accepted SM70 AWQ MoE batched-safe route: batched MoE "
            "enabled, AWQ MoE enabled, legacy compact disabled, either "
            "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES=0 or source-level unset-env "
            "safe-default evidence, and logs proving the W13 per-expert "
            "dispatch-selection fix is active."
        ),
    )
    parser.add_argument(
        "--require-awq-moe-0dot3-baseline-route",
        action="store_true",
        help=(
            "Require the 0.0.3 SM70 AWQ MoE throughput baseline route: AWQ "
            "MoE enabled, batched MoE enabled, and the runtime batched "
            "TurboMind MoE log present. This is separate from the strict "
            "per-expert dense exactness route."
        ),
    )
    parser.add_argument(
        "--require-fp8-moe-safe-route",
        action="store_true",
        help=(
            "Require the accepted SM70 FP8 MoE throughput route: native "
            "batched MoE enabled, FP8 MoE dequant fallback disabled, and the "
            "runtime batched TurboMind MoE log present."
        ),
    )
    parser.add_argument(
        "--require-fp8-moe-0dot3-fallback-route",
        action="store_true",
        help=(
            "Require the 0.0.3 SM70 FP8 MoE throughput baseline route: FP8 "
            "MoE expert weights dequantized to fp16 after loading and "
            "executed by the unquantized Triton MoE path. This is separate "
            "from the native SM70 FP8 MoE dense-stage diagnostic route."
        ),
    )
    parser.add_argument(
        "--require-sm70-breakable-graph-route",
        action="store_true",
        help=(
            "Require the explicit SM70 breakable CUDA graph lane: the JSON "
            "must show VLLM_SM70_USE_BREAKABLE_CUDAGRAPH mapped to the generic "
            "breakable graph env, and logs must show breakable wrapper "
            "activation and CUDA graph capture completion."
        ),
    )
    parser.add_argument(
        "--require-sm70-flash-v100-0dot3-compile-graph",
        action="store_true",
        help=(
            "Require the 0.0.3 SM70 Flash-V100 graph policy: "
            "VLLM_COMPILE + FULL_AND_PIECEWISE, no no-compile graph policy, "
            "and graph capture completion. Use this for old-tree 35B speed "
            "baseline recovery artifacts."
        ),
    )
    parser.add_argument(
        "--require-sm70-custom-allreduce-route",
        action="store_true",
        help=(
            "Require production TP custom allreduce evidence: JSON must show "
            "custom allreduce was not disabled, and logs must show the tp:0 "
            "communicator selected ['CUSTOM', 'PYNCCL'] rather than PYNCCL "
            "only."
        ),
    )
    parser.add_argument(
        "--require-sm70-all-reduce-sum2-route",
        action="store_true",
        help=(
            "Require the SM70 MoE shared+routed all_reduce_sum2 route: JSON "
            "must show VLLM_SM70_MOE_ADD_ALLREDUCE enabled, logs must show the "
            "MoE candidate, and the C++ all_reduce_sum2 trace must match "
            "capture=active on the same line."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path for the verification summary JSON.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        payload = _read_json(args.json)
        expectations = [
            _parse_expectation(raw) for raw in args.expect_json
        ]
        log_expectations = list(args.expect_log)
        log_regex_expectations = list(args.expect_log_regex)
        rejected_log_expectations = list(args.reject_log)
        rejected_log_regex_expectations = list(args.reject_log_regex)
        if args.require_full_flash_v100:
            expectations.extend(FULL_FLASH_V100_JSON_EXPECTATIONS)
            log_expectations.extend(FULL_FLASH_V100_REQUIRED_LOGS)
            rejected_log_expectations.extend(FULL_FLASH_V100_REJECTED_LOGS)
        if args.require_sm70_fp8_kv_cache_route:
            expectations.extend(FULL_FLASH_V100_JSON_EXPECTATIONS)
            expectations.extend(SM70_FP8_KV_CACHE_ROUTE_JSON_EXPECTATIONS)
            log_expectations.extend(FULL_FLASH_V100_REQUIRED_LOGS)
            log_expectations.extend(SM70_FP8_KV_CACHE_ROUTE_REQUIRED_LOGS)
            rejected_log_expectations.extend(FULL_FLASH_V100_REJECTED_LOGS)
            rejected_log_expectations.extend(
                SM70_FP8_KV_CACHE_ROUTE_REJECTED_LOGS
            )
        if args.require_turbomind_default_policy:
            expectations.extend(TURBOMIND_DEFAULT_JSON_EXPECTATIONS)
        if args.require_awq_turbomind_dense:
            expectations.extend(AWQ_TURBOMIND_DENSE_JSON_EXPECTATIONS)
            log_expectations.extend(AWQ_TURBOMIND_DENSE_REQUIRED_LOGS)
        if args.require_fp8_turbomind_dense:
            expectations.extend(FP8_TURBOMIND_DENSE_JSON_EXPECTATIONS)
            log_expectations.extend(FP8_TURBOMIND_DENSE_REQUIRED_LOGS)
        if args.require_fp8_0dot3_dense_dequant_route:
            expectations.extend(FP8_0DOT3_DENSE_DEQUANT_JSON_EXPECTATIONS)
            log_expectations.extend(FP8_0DOT3_DENSE_DEQUANT_REQUIRED_LOGS)
            rejected_log_regex_expectations.extend(
                FP8_0DOT3_DENSE_DEQUANT_REJECTED_REGEXES
            )
        if args.require_awq_moe_safe_route:
            expectations.extend(AWQ_MOE_SAFE_ROUTE_JSON_EXPECTATIONS)
            log_expectations.extend(AWQ_MOE_SAFE_ROUTE_REQUIRED_LOGS)
        if args.require_awq_moe_batched_safe_route:
            expectations.extend(AWQ_MOE_BATCHED_SAFE_ROUTE_JSON_EXPECTATIONS)
            log_expectations.extend(AWQ_MOE_BATCHED_SAFE_ROUTE_REQUIRED_LOGS)
            rejected_log_expectations.extend(
                AWQ_MOE_BATCHED_SAFE_ROUTE_REJECTED_LOGS
            )
        if args.require_awq_moe_0dot3_baseline_route:
            expectations.extend(AWQ_MOE_0DOT3_BASELINE_ROUTE_JSON_EXPECTATIONS)
            log_expectations.extend(
                AWQ_MOE_0DOT3_BASELINE_ROUTE_REQUIRED_LOGS
            )
        if args.require_fp8_moe_safe_route:
            expectations.extend(FP8_MOE_SAFE_ROUTE_JSON_EXPECTATIONS)
            log_expectations.extend(FP8_MOE_SAFE_ROUTE_REQUIRED_LOGS)
        if args.require_fp8_moe_0dot3_fallback_route:
            expectations.extend(
                FP8_MOE_0DOT3_FALLBACK_ROUTE_JSON_EXPECTATIONS
            )
            log_expectations.extend(
                FP8_MOE_0DOT3_FALLBACK_ROUTE_REQUIRED_LOGS
            )
            rejected_log_expectations.extend(
                FP8_MOE_0DOT3_FALLBACK_ROUTE_REJECTED_LOGS
            )
        extra_required_policies = list(args.require_policy)
        if args.require_sm70_breakable_graph_route:
            extra_required_policies.append("sm70_graph_policy")
            expectations.extend(SM70_BREAKABLE_GRAPH_ROUTE_JSON_EXPECTATIONS)
            log_expectations.extend(SM70_BREAKABLE_GRAPH_ROUTE_REQUIRED_LOGS)
        if args.require_sm70_flash_v100_0dot3_compile_graph:
            extra_required_policies.append("sm70_graph_policy")
            expectations.extend(
                SM70_FLASH_V100_0DOT3_COMPILE_GRAPH_JSON_EXPECTATIONS
            )
            log_expectations.extend(
                SM70_FLASH_V100_0DOT3_COMPILE_GRAPH_REQUIRED_LOGS
            )
            rejected_log_expectations.extend(
                SM70_FLASH_V100_0DOT3_COMPILE_GRAPH_REJECTED_LOGS
            )
        if args.require_sm70_custom_allreduce_route:
            extra_required_policies.append("sm70_comm_policy")
            expectations.extend(SM70_CUSTOM_ALLREDUCE_ROUTE_JSON_EXPECTATIONS)
            log_expectations.extend(SM70_CUSTOM_ALLREDUCE_ROUTE_REQUIRED_LOGS)
            rejected_log_expectations.extend(
                SM70_CUSTOM_ALLREDUCE_ROUTE_REJECTED_LOGS
            )
        if args.require_sm70_all_reduce_sum2_route:
            extra_required_policies.append("sm70_comm_policy")
            expectations.extend(SM70_ALL_REDUCE_SUM2_ROUTE_JSON_EXPECTATIONS)
            log_expectations.extend(SM70_ALL_REDUCE_SUM2_ROUTE_REQUIRED_LOGS)
            log_regex_expectations.extend(
                SM70_ALL_REDUCE_SUM2_ROUTE_REQUIRED_REGEXES
            )
        required_policies = [] if args.require_inventory_complete else list(
            dict.fromkeys(
                list(DEFAULT_REQUIRED_POLICIES) + extra_required_policies
            )
        )
        result = _verify(
            payload,
            json_path=args.json,
            log_paths=args.log,
            required_policies=required_policies,
            json_expectations=expectations,
            log_expectations=log_expectations,
            log_regex_expectations=log_regex_expectations,
            rejected_log_expectations=rejected_log_expectations,
            rejected_log_regex_expectations=rejected_log_regex_expectations,
            require_decode_timing=args.require_decode_timing,
            require_model_quality_pass=args.require_model_quality_pass,
            require_inventory_complete=args.require_inventory_complete,
        )
        if (
            args.require_awq_moe_safe_route
            or args.require_awq_moe_batched_safe_route
        ):
            awq_moe_safe_tune_gate = _awq_moe_safe_tune_gate(payload)
            result["awq_moe_safe_tune_gate"] = awq_moe_safe_tune_gate
            if not awq_moe_safe_tune_gate["passed"]:
                result["passed"] = False
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result = {
            "passed": False,
            "json_path": str(args.json),
            "log_paths": [str(path) for path in args.log],
            "error": str(error),
        }

    summary = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
