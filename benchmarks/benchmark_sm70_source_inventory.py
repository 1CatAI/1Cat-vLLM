# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source inventory helper for the SM70/V100 migration.

This script compares the old 0.0.3 source tree with the latest migration tree.
It is intentionally static: it does not import vLLM, start CUDA, build
extensions, or run models. The output is a JSON ledger that helps avoid
silently dropping old V100-specific routes during upstream migration.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_OLD_ROOT = Path("/home/ymzx/桌面/1cat-vllm/1Cat-vLLM-0.0.3/vllm")
DEFAULT_LATEST_ROOT = Path("/home/ymzx/桌面/1cat-vllm/vllm")

SOURCE_TOP_LEVELS = (
    "vllm",
    "csrc",
    "flash-attention-v100",
    "lmdeploy",
    "benchmarks",
    "docs/design",
)

RUNTIME_ENV_TOP_LEVELS = (
    "vllm",
    "csrc",
    "flash-attention-v100",
    "lmdeploy",
)

IGNORE_PARTS = {
    ".git",
    ".deps",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "backend_logs",
    "bench_results",
    "build",
    "dist",
}

TEXT_SUFFIXES = {
    ".cc",
    ".cmake",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".jinja",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

KEYWORDS = (
    "SM70",
    "V100",
    "FLASH_ATTN_V100",
    "TURBOMIND",
    "TurboMind",
    "all_reduce_sum2",
    "top1_argmax",
    "DFlash",
    "DFLASH",
    "MTP",
    "FP8_KV",
    "kv_cache_dtype",
    "GDN",
    "FLA",
    "mixed_qkv",
    "gated_silu",
    "broadcast_input",
    "indexed_expert",
    "direct_reduce",
    "weighted_reduce",
)

ENV_RE = re.compile(r"\bVLLM_[A-Z0-9_]+\b")
IGNORED_ENV_NAMES = {
    # CMake cache variable, not a runtime environment gate.
    "VLLM_PYTHON_EXECUTABLE",
}
TORCH_OP_RE = re.compile(r"\b(?:torch::)?ops\.def\(\s*(?:R)?\"([A-Za-z0-9_]+)")
CUSTOM_AR_RE = re.compile(r"\bcustom_ar\.def\(\s*(?:R)?\"([A-Za-z0-9_]+)")
CACHE_OP_RE = re.compile(r"\bcache_ops\.def\(\s*(?:R)?\"([A-Za-z0-9_]+)")
PY_TORCH_OP_RE = re.compile(r"\btorch\.ops\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)")
IGNORED_PY_TORCH_OP_NAMESPACES = {
    # PyTorch builtins are not vLLM migration items. Keep this inventory
    # focused on extension/custom-op schemas that can be dropped during an
    # upstream rebase.
    "aten",
}

OLD_ONLY_TORCH_OP_CLASSIFICATIONS: dict[str, dict[str, str]] = {
    "_C::awq_moe_gemm_sm70": {
        "status": "closed-replaced-api-cleanup",
        "doc_item": "160",
    },
    "_C::awq_moe_single_token_compact_prepare": {
        "status": "closed-replaced",
        "doc_item": "160",
    },
    "_C::awq_moe_single_token_exact_layout_prepare": {
        "status": "closed-replaced",
        "doc_item": "160",
    },
    "_C::awq_moe_single_token_sm70_out": {
        "status": "closed-replaced",
        "doc_item": "160",
    },
    "_C::awq_moe_single_token_weighted_reduce_out": {
        "status": "closed-not-default",
        "doc_item": "160",
    },
    "_C::convert_vertical_slash_indexes": {
        "status": "closed-not-current-attention-target",
        "doc_item": "185",
    },
    "_C::convert_vertical_slash_indexes_mergehead": {
        "status": "closed-not-current-attention-target",
        "doc_item": "185",
    },
    "_C::cutlass_scaled_sparse_mm": {
        "status": "closed-non-sm70-target",
        "doc_item": "185",
    },
    "_C::cutlass_sparse_compress": {
        "status": "closed-non-sm70-target",
        "doc_item": "185",
    },
    "_C::cutlass_sparse_scaled_mm_supported": {
        "status": "closed-non-sm70-target",
        "doc_item": "185",
    },
    "_C::fp8_moe_single_token_router_sm70_out": {
        "status": "paused-unsafe",
        "doc_item": "161",
    },
    "_C::fp8_moe_single_token_sm70_out": {
        "status": "closed-replaced",
        "doc_item": "161",
    },
    "_C::get_cutlass_pplx_moe_mm_data": {
        "status": "closed-non-sm70-target",
        "doc_item": "185",
    },
    "_C_cpu::mla_decode_kvcache": {
        "status": "closed-cpu-only-non-v100-target",
        "doc_item": "186",
    },
    "_C_utils::init_cpu_threads_env": {
        "status": "closed-cpu-only-non-v100-target",
        "doc_item": "186",
    },
    "_qutlass_C::matmul_ada_mxf4_bf16_tn": {
        "status": "closed-upstream-qutlass-replaced-non-awq-fp8-v100-target",
        "doc_item": "186",
    },
    "vllm::fi_trtllm_fp8_per_tensor_moe": {
        "status": "closed-upstream-modular-replaced",
        "doc_item": "186",
    },
    "vllm::flashinfer_fused_moe_bf16": {
        "status": "closed-upstream-modular-replaced",
        "doc_item": "186",
    },
    "vllm::flashinfer_fused_moe_blockscale_fp8": {
        "status": "closed-upstream-modular-replaced",
        "doc_item": "186",
    },
    "vllm::gdn_attention_core": {
        "status": "closed-replaced",
        "doc_item": "184",
    },
    "vllm::inplace_fused_experts": {
        "status": "closed-upstream-schema-replaced",
        "doc_item": "186",
    },
    "vllm::outplace_fused_experts": {
        "status": "closed-upstream-schema-replaced",
        "doc_item": "186",
    },
    "vllm::rocm_aiter_gemm_a8w8": {
        "status": "closed-rocm-only-non-v100-target",
        "doc_item": "186",
    },
    "vllm::rocm_aiter_rms_norm": {
        "status": "closed-rocm-only-non-v100-target",
        "doc_item": "186",
    },
    "vllm::rocm_aiter_rmsnorm2d_fwd_with_add": {
        "status": "closed-rocm-only-non-v100-target",
        "doc_item": "186",
    },
    "vllm::sm70_unquantized_gemm": {
        "status": "closed-replaced",
        "doc_item": "188",
    },
    "vllm::unified_attention": {
        "status": "closed-upstream-schema-replaced",
        "doc_item": "184",
    },
    "vllm::unified_mla_attention": {
        "status": "closed-upstream-schema-replaced",
        "doc_item": "184",
    },
}

OLD_ONLY_RUNTIME_ENV_CLASSIFICATIONS: dict[str, dict[str, str]] = {
    "VLLM_1CAT_DISABLE_QWEN35_MTP_DEFAULTS": {
        "status": "paused-default-policy-not-restored",
        "doc_item": "198",
    },
    "VLLM_ATTENTION_BACKEND": {
        "status": "closed-upstream-config-replaced",
        "doc_item": "196",
    },
    "VLLM_MTP_STOCHASTIC_TOKEN_MATCHING": {
        "status": "paused-behavior-changing",
        "doc_item": "138",
    },
    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": {
        "status": "closed-non-sm70-runtime",
        "doc_item": "196",
    },
    "VLLM_QWEN35_MTP_KEEP_QUANT": {
        "status": "closed-upstream-replaced-default",
        "doc_item": "195",
    },
    "VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS": {
        "status": "closed-upstream-replaced-default",
        "doc_item": "195",
    },
    "VLLM_SM70_AWQ_COMPACT_COMPARE": {
        "status": "closed-replaced-diagnostic",
        "doc_item": "160",
    },
    "VLLM_SM70_AWQ_ENABLE_SINGLE_TOKEN_COMPACT": {
        "status": "closed-replaced",
        "doc_item": "160",
    },
    "VLLM_SM70_FP8_SHARED_GATE_UP_GATED_SILU": {
        "status": "paused-quality-unproven-gated-silu-route",
        "doc_item": "197",
    },
    "VLLM_SM70_GATE_UP_GATED_SILU": {
        "status": "paused-quality-unproven-gated-silu-route",
        "doc_item": "197",
    },
    "VLLM_TRITON_ATTN_SM70_QHEAD_SPLIT": {
        "status": "closed-not-current-target",
        "doc_item": "162",
    },
}

_PAUSED_UNSAFE_FP8_MOE_RUNTIME_ENVS = {
    "VLLM_SM70_FP8_MOE_ALLOW_UNSAFE_BROADCAST_INPUT",
    "VLLM_SM70_FP8_MOE_BROADCAST_INPUT",
    "VLLM_SM70_FP8_MOE_BROADCAST_INPUT_IDXS",
    "VLLM_SM70_FP8_MOE_COMPACT_COMPARE",
    "VLLM_SM70_FP8_MOE_COMPACT_STRICT_COMPARE",
    "VLLM_SM70_FP8_MOE_EXACT_LAYOUT",
    "VLLM_SM70_FP8_MOE_EXACT_REDUCE",
    "VLLM_SM70_FP8_MOE_GATED_SILU_EPILOGUE",
    "VLLM_SM70_FP8_MOE_INDEXED_EXPERT_PTRS",
    "VLLM_SM70_FP8_MOE_ROUTER_TOPK_COMPARE",
    "VLLM_SM70_FP8_MOE_ROUTER_TOPK_CPP",
    "VLLM_SM70_FP8_MOE_ROUTER_TOPK_PERSISTENT",
    "VLLM_SM70_FP8_MOE_SINGLE_TOKEN_COMPACT",
    "VLLM_SM70_FP8_MOE_SINGLE_TOKEN_CPP",
    "VLLM_SM70_FP8_MOE_W2_DIRECT_REDUCE",
    "VLLM_SM70_FP8_MOE_WEIGHTED_REDUCE_EPILOGUE",
}

for _env_name in _PAUSED_UNSAFE_FP8_MOE_RUNTIME_ENVS:
    OLD_ONLY_RUNTIME_ENV_CLASSIFICATIONS[_env_name] = {
        "status": "paused-unsafe-old-fp8-compact-family",
        "doc_item": "161",
    }

OLD_ONLY_KEYWORD_FILE_PREFIX_CLASSIFICATIONS: tuple[tuple[str, str, str], ...] = (
    ("benchmarks/", "closed-historical-benchmark", "199"),
    ("docs/design/", "closed-historical-doc", "199"),
    ("lmdeploy/", "closed-vendored-reference-extracted", "199"),
    ("csrc/attention/", "closed-upstream-moved-libtorch-stable", "199"),
    ("csrc/quantization/awq/", "closed-replaced-sm70-turbomind", "199"),
)

OLD_ONLY_KEYWORD_FILE_CLASSIFICATIONS: dict[str, dict[str, str]] = {
    "csrc/cache_kernels.cu": {
        "status": "closed-upstream-moved-libtorch-stable",
        "doc_item": "199",
    },
    "csrc/cache_kernels_fused.cu": {
        "status": "closed-upstream-moved-libtorch-stable",
        "doc_item": "199",
    },
    "vllm/compilation/matcher_utils.py": {
        "status": "closed-keyword-only-upstream-changed",
        "doc_item": "199",
    },
    "vllm/model_executor/layers/batch_invariant.py": {
        "status": "closed-keyword-only-upstream-changed",
        "doc_item": "199",
    },
    "vllm/model_executor/layers/fla/ops/cumsum.py": {
        "status": "closed-by-gdn-fla-source-items",
        "doc_item": "148",
    },
    "vllm/model_executor/layers/fla/ops/wy_fast.py": {
        "status": "closed-by-gdn-fla-source-items",
        "doc_item": "148",
    },
    "vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py": {
        "status": "closed-non-sm70-upstream-fused-moe",
        "doc_item": "199",
    },
    "vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py": {
        "status": "closed-non-sm70-upstream-fused-moe",
        "doc_item": "199",
    },
    "vllm/model_executor/layers/fused_moe/layer.py": {
        "status": "closed-by-latest-fused-moe-items",
        "doc_item": "142",
    },
    "vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py": {
        "status": "closed-by-latest-fused-moe-items",
        "doc_item": "161",
    },
    "vllm/model_executor/layers/kda.py": {
        "status": "closed-non-current-sm70-target",
        "doc_item": "199",
    },
    "vllm/model_executor/layers/mamba/mamba_utils.py": {
        "status": "closed-by-mamba-align-items",
        "doc_item": "176",
    },
    "vllm/model_executor/layers/quantization/compressed_tensors/"
    "compressed_tensors_moe.py": {
        "status": "closed-non-sm70-upstream-quantization",
        "doc_item": "199",
    },
    "vllm/model_executor/layers/quantization/modelopt.py": {
        "status": "closed-non-sm70-upstream-quantization",
        "doc_item": "199",
    },
    "vllm/model_executor/layers/quantization/utils/nvfp4_utils.py": {
        "status": "closed-non-sm70-upstream-quantization",
        "doc_item": "199",
    },
    "vllm/model_executor/models/qwen2_moe.py": {
        "status": "closed-by-shared-gate-source-item",
        "doc_item": "197",
    },
    "vllm/model_executor/models/qwen3_vl.py": {
        "status": "closed-non-current-sm70-target",
        "doc_item": "199",
    },
    "vllm/v1/attention/backends/tree_attn.py": {
        "status": "closed-non-current-attention-target",
        "doc_item": "184",
    },
    "vllm/v1/attention/ops/triton_decode_attention.py": {
        "status": "closed-by-attention-source-items",
        "doc_item": "162",
    },
    "vllm/v1/sample/rejection_sampler.py": {
        "status": "closed-by-spec-diagnostic-items",
        "doc_item": "138",
    },
    "vllm/v1/spec_decode/eagle.py": {
        "status": "closed-by-dflash-diagnostic-items",
        "doc_item": "138",
    },
    "vllm/v1/worker/gpu/attn_utils.py": {
        "status": "closed-by-attention-source-items",
        "doc_item": "162",
    },
}


def _is_under_top_level(relative_path: Path, top_levels: tuple[str, ...]) -> bool:
    path = relative_path.as_posix()
    return any(path == top or path.startswith(f"{top}/") for top in top_levels)


def _is_under_source_top_level(relative_path: Path) -> bool:
    return _is_under_top_level(relative_path, SOURCE_TOP_LEVELS)


def _is_under_runtime_env_top_level(relative_path: Path) -> bool:
    return _is_under_top_level(relative_path, RUNTIME_ENV_TOP_LEVELS)


def _should_scan(relative_path: Path) -> bool:
    if any(part in IGNORE_PARTS for part in relative_path.parts):
        return False
    if not _is_under_source_top_level(relative_path):
        return False
    return relative_path.suffix in TEXT_SUFFIXES


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_source_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for top_level in SOURCE_TOP_LEVELS:
        top_path = root / top_level
        if not top_path.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(top_path):
            dirnames[:] = [
                dirname for dirname in dirnames if dirname not in IGNORE_PARTS
            ]
            base = Path(dirpath)
            for filename in filenames:
                path = base / filename
                relative_path = path.relative_to(root)
                if _should_scan(relative_path):
                    paths.append(path)
    return sorted(paths)


def _keyword_hits(text: str) -> list[str]:
    return sorted({keyword for keyword in KEYWORDS if keyword in text})


def _env_names(text: str) -> list[str]:
    names: list[str] = []
    for env_name in ENV_RE.findall(text):
        # Documentation often refers to whole env families such as
        # VLLM_QWEN3_NEXT_*. The regex intentionally stops before the '*',
        # leaving a trailing underscore. Treat these as wildcard prefixes, not
        # concrete env gates that need one-by-one migration.
        if env_name.endswith("_"):
            continue
        if env_name in IGNORED_ENV_NAMES:
            continue
        names.append(env_name)
    return names


def _scan_tree(root: Path) -> dict[str, Any]:
    env_to_files: dict[str, set[str]] = defaultdict(set)
    runtime_env_to_files: dict[str, set[str]] = defaultdict(set)
    keyword_files: dict[str, list[str]] = {}
    torch_ops: dict[str, set[str]] = defaultdict(set)
    file_count = 0
    scanned_bytes = 0

    for path in _iter_source_files(root):
        relative_path = path.relative_to(root)
        text = _read_text(path)
        rel = relative_path.as_posix()
        file_count += 1
        scanned_bytes += len(text)

        env_names = _env_names(text)
        for env_name in env_names:
            env_to_files[env_name].add(rel)
        if _is_under_runtime_env_top_level(relative_path):
            for env_name in env_names:
                runtime_env_to_files[env_name].add(rel)

        hits = _keyword_hits(text)
        if hits:
            keyword_files[rel] = hits

        for op_name in TORCH_OP_RE.findall(text):
            torch_ops[f"_C::{op_name}"].add(rel)
        for op_name in CUSTOM_AR_RE.findall(text):
            torch_ops[f"_C_custom_ar::{op_name}"].add(rel)
        for op_name in CACHE_OP_RE.findall(text):
            torch_ops[f"_C_cache_ops::{op_name}"].add(rel)
        for namespace, op_name in PY_TORCH_OP_RE.findall(text):
            if namespace in IGNORED_PY_TORCH_OP_NAMESPACES:
                continue
            torch_ops[f"{namespace}::{op_name}"].add(rel)

    env_counts = Counter()
    for env_name, files in env_to_files.items():
        env_counts[env_name] = len(files)

    return {
        "root": str(root),
        "file_count": file_count,
        "scanned_bytes": scanned_bytes,
        "env_vars": {
            name: sorted(files)
            for name, files in sorted(env_to_files.items())
        },
        "runtime_env_vars": {
            name: sorted(files)
            for name, files in sorted(runtime_env_to_files.items())
        },
        "env_file_counts": dict(sorted(env_counts.items())),
        "keyword_files": dict(sorted(keyword_files.items())),
        "torch_ops": {
            name: sorted(files)
            for name, files in sorted(torch_ops.items())
        },
    }


def _diff_keys(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_keys = set(left)
    right_keys = set(right)
    return {
        "old_only": sorted(left_keys - right_keys),
        "latest_only": sorted(right_keys - left_keys),
        "common": sorted(left_keys & right_keys),
    }


def _classify_old_only_torch_ops(old_only: list[str]) -> dict[str, Any]:
    classified = {
        op_name: OLD_ONLY_TORCH_OP_CLASSIFICATIONS[op_name]
        for op_name in old_only
        if op_name in OLD_ONLY_TORCH_OP_CLASSIFICATIONS
    }
    unclassified = [
        op_name
        for op_name in old_only
        if op_name not in OLD_ONLY_TORCH_OP_CLASSIFICATIONS
    ]
    return {
        "classified": dict(sorted(classified.items())),
        "unclassified": sorted(unclassified),
    }


def _classify_old_only_runtime_envs(old_only: list[str]) -> dict[str, Any]:
    classified = {
        env_name: OLD_ONLY_RUNTIME_ENV_CLASSIFICATIONS[env_name]
        for env_name in old_only
        if env_name in OLD_ONLY_RUNTIME_ENV_CLASSIFICATIONS
    }
    unclassified = [
        env_name
        for env_name in old_only
        if env_name not in OLD_ONLY_RUNTIME_ENV_CLASSIFICATIONS
    ]
    return {
        "classified": dict(sorted(classified.items())),
        "unclassified": sorted(unclassified),
    }


def _classify_old_only_keyword_files(old_only: list[str]) -> dict[str, Any]:
    classified: dict[str, dict[str, str]] = {}
    unclassified: list[str] = []
    for path in old_only:
        if path in OLD_ONLY_KEYWORD_FILE_CLASSIFICATIONS:
            classified[path] = OLD_ONLY_KEYWORD_FILE_CLASSIFICATIONS[path]
            continue
        for prefix, status, doc_item in OLD_ONLY_KEYWORD_FILE_PREFIX_CLASSIFICATIONS:
            if path.startswith(prefix):
                classified[path] = {
                    "status": status,
                    "doc_item": doc_item,
                    "prefix": prefix,
                }
                break
        else:
            unclassified.append(path)
    return {
        "classified": dict(sorted(classified.items())),
        "unclassified": sorted(unclassified),
    }


def _compact_diff(old: dict[str, Any], latest: dict[str, Any]) -> dict[str, Any]:
    diff = {
        "env_vars": _diff_keys(old["env_vars"], latest["env_vars"]),
        "runtime_env_vars": _diff_keys(
            old["runtime_env_vars"], latest["runtime_env_vars"]
        ),
        "keyword_files": _diff_keys(old["keyword_files"], latest["keyword_files"]),
        "torch_ops": _diff_keys(old["torch_ops"], latest["torch_ops"]),
    }
    diff["old_only_torch_op_classification"] = _classify_old_only_torch_ops(
        diff["torch_ops"]["old_only"]
    )
    diff["old_only_runtime_env_classification"] = (
        _classify_old_only_runtime_envs(diff["runtime_env_vars"]["old_only"])
    )
    diff["old_only_keyword_file_classification"] = (
        _classify_old_only_keyword_files(diff["keyword_files"]["old_only"])
    )
    return diff


def _classification_counts(section: dict[str, Any]) -> dict[str, Any]:
    classified = section["classified"]
    unclassified = section["unclassified"]
    return {
        "classified_count": len(classified),
        "unclassified_count": len(unclassified),
        "unclassified": unclassified,
    }


def _inventory_gate(diff: dict[str, Any]) -> dict[str, Any]:
    runtime_env = _classification_counts(
        diff["old_only_runtime_env_classification"]
    )
    keyword_files = _classification_counts(
        diff["old_only_keyword_file_classification"]
    )
    torch_ops = _classification_counts(
        diff["old_only_torch_op_classification"]
    )
    old_only_env_vars = diff["env_vars"]["old_only"]
    complete = (
        not old_only_env_vars
        and runtime_env["unclassified_count"] == 0
        and keyword_files["unclassified_count"] == 0
        and torch_ops["unclassified_count"] == 0
    )
    return {
        "label": "inventory-complete" if complete else "inventory-incomplete",
        "complete": complete,
        "old_only_env_vars_count": len(old_only_env_vars),
        "old_only_env_vars": old_only_env_vars,
        "old_only_runtime_env": runtime_env,
        "old_only_keyword_files": keyword_files,
        "old_only_torch_ops": torch_ops,
    }


def _build_payload(old_root: Path, latest_root: Path) -> dict[str, Any]:
    old = _scan_tree(old_root)
    latest = _scan_tree(latest_root)
    diff = _compact_diff(old, latest)
    return {
        "old_root": str(old_root),
        "latest_root": str(latest_root),
        "source_top_levels": list(SOURCE_TOP_LEVELS),
        "ignored_parts": sorted(IGNORE_PARTS),
        "keywords": list(KEYWORDS),
        "old": old,
        "latest": latest,
        "diff": diff,
        "inventory_gate": _inventory_gate(diff),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--latest-root", type=Path, default=DEFAULT_LATEST_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    payload = _build_payload(args.old_root.resolve(), args.latest_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=args.indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out),
        "old_files": payload["old"]["file_count"],
        "latest_files": payload["latest"]["file_count"],
        "old_env_vars": len(payload["old"]["env_vars"]),
        "latest_env_vars": len(payload["latest"]["env_vars"]),
        "old_runtime_env_vars": len(payload["old"]["runtime_env_vars"]),
        "latest_runtime_env_vars": len(payload["latest"]["runtime_env_vars"]),
            "old_keyword_files": len(payload["old"]["keyword_files"]),
            "latest_keyword_files": len(payload["latest"]["keyword_files"]),
            "old_torch_ops": len(payload["old"]["torch_ops"]),
            "latest_torch_ops": len(payload["latest"]["torch_ops"]),
            "inventory_complete": payload["inventory_gate"]["complete"],
            "old_only_env_vars": payload["inventory_gate"][
                "old_only_env_vars_count"
            ],
            "old_only_runtime_env_unclassified": payload["inventory_gate"][
                "old_only_runtime_env"
            ]["unclassified_count"],
            "old_only_keyword_files_unclassified": payload["inventory_gate"][
                "old_only_keyword_files"
            ]["unclassified_count"],
            "old_only_torch_ops_unclassified": payload["inventory_gate"][
                "old_only_torch_ops"
            ]["unclassified_count"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
