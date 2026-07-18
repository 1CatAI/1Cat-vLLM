# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strict operator-only A/B benchmark for the SM70 split-KV exact shape.

This compares the accepted fixed paged BM32 pair-scratch pybind entry with
its three-way split-KV variant at B=1, Hq=6, Hkv=1, M=8096, N=121440,
D=256, and page_size=784 by default. It does not create an engine, load a
model, or run an end-to-end workload. Default acceptance may use a different
host KV length through --accept-kv-len, while --screen may use one through
--screen-kv-len; both retain the remaining fixed shape.

Each normal/reverse/tail page-table case receives 20 alternating warmup rounds
and 100 alternating, paired CUDA-event measurements. A GPU run requires an
exclusive SM70 GPU at exactly 1200 MHz. Any observed clock deviation or an
external GPU PID is a hard failure.

Use --static-check for a CPU-only source-contract check. That mode neither
imports torch nor initializes CUDA.

Use --correctness-only to retain the strict GPU clock/PID gate and execute the
three exact numerical comparisons without warmup or timing.

Use --screen for a non-acceptance normal-page-table screen with correctness,
five warmups, and twenty paired rounds.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BATCH_SIZE = 1
M = 8096
N = 121440
H_Q = 6
H_KV = 1
HEAD_DIM = 256
PAGE_SIZE = 784
SPLIT_PARTS = 3
CACHE_PAGES = (N + PAGE_SIZE - 1) // PAGE_SIZE
ALLOCATED_CACHE_TOKENS = CACHE_PAGES * PAGE_SIZE
TAIL_PAGE_VALID_TOKENS = N - (CACHE_PAGES - 1) * PAGE_SIZE
TAIL_PAGE_UNUSED_TOKENS = PAGE_SIZE - TAIL_PAGE_VALID_TOKENS
SOFTMAX_SCALE = HEAD_DIM**-0.5

REQUIRED_GRAPHICS_CLOCK_MHZ = 1200
WARMUP_ROUNDS = 20
PAIRED_ROUNDS = 100
SCREEN_WARMUP_ROUNDS = 5
SCREEN_PAIRED_ROUNDS = 20
ACCEPTANCE_P50_PCT_MAX = -2.0
ACCEPTANCE_CANDIDATE_WIN_COUNT_MIN = 95
SCREEN_P50_PCT_MAX = -2.0
SCREEN_CANDIDATE_WIN_COUNT_MIN = 19
LOAD_MONITOR_INTERVAL_MS = 250
LOAD_MONITOR_COMMAND_TIMEOUT_SECONDS = 5.0
OUTPUT_MAX_ABS_THRESHOLD = 0.001953125
LSE_MAX_ABS_THRESHOLD = 0.001
INERT_GRAPHICS_PROCESS_NAME = "/usr/lib/xorg/Xorg"
INERT_GRAPHICS_PROCESS_TYPE = "G"
SPLITKV3_TOTAL_TILES = 949
SPLITKV3_LAST_SPLIT_BEGIN = 80896
SPLITKV3_EARLIEST_CAUSAL_CUTOFF = N - M
SPLITKV3_FAST_VISIBLE_CONDITION = True
SPLITKV3_EXPECTED_CHECK_SPLIT_EMPTY = False
SPLITKV3_BLOCK_N = 128

SOURCE_ROOT = Path(__file__).resolve().parents[1]
FLASH_V100_ROOT = SOURCE_ROOT / "flash-attention-v100"
PYBIND_SOURCE = FLASH_V100_ROOT / "kernel" / "fused_mha_api.cpp"
PAGED_SOURCE = FLASH_V100_ROOT / "kernel" / "fused_mha_forward_paged.cu"

UNSPLIT_ROUTE = "accepted_unsplit_fixed_pybind"
SPLITKV3_ROUTE = "splitkv3_fixed_pybind"
ROUTE_NAMES = (UNSPLIT_ROUTE, SPLITKV3_ROUTE)
UNSPLIT_ENTRYPOINT = "prefill_paged_d256_bm32_allp_pair_scratch_fwd"
SPLITKV3_ENTRYPOINT = "prefill_paged_d256_bm32_allp_pair_scratch_splitkv3_fwd"

# Reverse this schedule each round so neither entry always gets the first launch.
ORDER_SCHEDULE = (
    (UNSPLIT_ROUTE, SPLITKV3_ROUTE),
    (SPLITKV3_ROUTE, UNSPLIT_ROUTE),
)
ACCEPTANCE_PAGE_TABLES = ("normal", "reverse", "tail")
SCREEN_PAGE_TABLES = ("normal",)

# Kept lazy so --static-check cannot initialize CUDA as an import side effect.
torch: Any = None


@dataclass(frozen=True)
class _Route:
    name: str
    launch: Callable[[], Any]


@dataclass(frozen=True)
class _SplitWorkspace:
    output: Any
    row_max: Any
    row_sum: Any


@dataclass(frozen=True)
class _KvShape:
    n: int
    cache_pages: int
    allocated_cache_tokens: int
    tail_page_valid_tokens: int
    tail_page_unused_tokens: int
    total_tiles: int
    last_split_begin: int
    earliest_causal_cutoff: int
    fast_visible_condition: bool
    expected_check_split_empty: bool


def _kv_shape(n: int) -> _KvShape:
    if n < M:
        raise ValueError(f"KV length must be at least M={M}, got {n}")
    cache_pages = (n + PAGE_SIZE - 1) // PAGE_SIZE
    allocated_cache_tokens = cache_pages * PAGE_SIZE
    total_tiles = (n + SPLITKV3_BLOCK_N - 1) // SPLITKV3_BLOCK_N
    last_split_begin = (SPLIT_PARTS - 1) * total_tiles // SPLIT_PARTS * SPLITKV3_BLOCK_N
    earliest_causal_cutoff = n - M
    fast_visible_condition = last_split_begin <= earliest_causal_cutoff
    return _KvShape(
        n=n,
        cache_pages=cache_pages,
        allocated_cache_tokens=allocated_cache_tokens,
        tail_page_valid_tokens=n - (cache_pages - 1) * PAGE_SIZE,
        tail_page_unused_tokens=allocated_cache_tokens - n,
        total_tiles=total_tiles,
        last_split_begin=last_split_begin,
        earliest_causal_cutoff=earliest_causal_cutoff,
        fast_visible_condition=fast_visible_condition,
        expected_check_split_empty=not fast_visible_condition,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-gpu",
        type=int,
        default=0,
        help="Physical GPU index to expose as logical cuda:0.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional JSON destination. Otherwise emit the result to stdout.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--static-check",
        action="store_true",
        help="Validate the exact contract without importing torch or using a GPU.",
    )
    mode_group.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run strict numerical gates but skip all warmup and timing.",
    )
    mode_group.add_argument(
        "--screen",
        action="store_true",
        help=(
            "Run the non-acceptance normal-only screen: correctness, then "
            "5 warmups and 20 paired rounds."
        ),
    )
    parser.add_argument(
        "--screen-kv-len",
        type=int,
        metavar="N",
        help="Screen-only host KV length; defaults to 121440 when --screen is set.",
    )
    parser.add_argument(
        "--accept-kv-len",
        type=int,
        metavar="N",
        help=(
            "Default-acceptance-only host KV length; defaults to 121440 and "
            "cannot be combined with another execution mode."
        ),
    )
    args = parser.parse_args()
    if args.physical_gpu < 0:
        parser.error("--physical-gpu must be non-negative")
    if args.accept_kv_len is not None and (
        args.screen
        or args.static_check
        or args.correctness_only
        or args.screen_kv_len is not None
    ):
        parser.error(
            "--accept-kv-len is only valid in default acceptance mode; it cannot "
            "be combined with --screen, --static-check, --correctness-only, or "
            "--screen-kv-len"
        )
    if args.screen_kv_len is not None and not args.screen:
        parser.error("--screen-kv-len requires --screen")
    if args.accept_kv_len is not None and args.accept_kv_len < M:
        parser.error(f"--accept-kv-len must be at least M={M}")
    if args.screen:
        args.screen_kv_len = N if args.screen_kv_len is None else args.screen_kv_len
        if args.screen_kv_len < M:
            parser.error(f"--screen-kv-len must be at least M={M}")
    else:
        args.screen_kv_len = N
    args.accept_kv_len = N if args.accept_kv_len is None else args.accept_kv_len
    return args


def _shape_for_args(args: argparse.Namespace) -> _KvShape:
    return _kv_shape(args.screen_kv_len if args.screen else args.accept_kv_len)


def _split_visibility_metadata(shape: _KvShape) -> dict[str, Any]:
    return {
        "host_actual_n": shape.n,
        "total_tiles": shape.total_tiles,
        "last_split_begin": shape.last_split_begin,
        "earliest_causal_cutoff": shape.earliest_causal_cutoff,
        "fast_visible_predicate": {
            "expression": "last_split_begin <= earliest_causal_cutoff",
            "value": shape.fast_visible_condition,
        },
        "fast_visible_condition": shape.fast_visible_condition,
        "expected_kernel": {
            "CHECK_SPLIT_EMPTY": shape.expected_check_split_empty,
        },
    }


def _page_specs(shape: _KvShape | None = None) -> dict[str, dict[str, Any]]:
    shape = _kv_shape(N) if shape is None else shape
    normal = tuple(range(shape.cache_pages))
    return {
        "normal": {
            "page_ids": normal,
            "description": "Logical pages map to identical physical page IDs.",
        },
        "reverse": {
            "page_ids": tuple(reversed(normal)),
            "description": "Logical pages traverse physical pages in reverse order.",
        },
        "tail": {
            "page_ids": (normal[-1], *normal[:-1]),
            "description": (
                "The physical page containing the partial tail is moved to logical "
                "page zero."
            ),
        },
    }


def _geometry(shape: _KvShape | None = None) -> dict[str, Any]:
    shape = _kv_shape(N) if shape is None else shape
    return {
        "batch_size": BATCH_SIZE,
        "m": M,
        "n": shape.n,
        "heads_q": H_Q,
        "heads_kv": H_KV,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "cache_pages": shape.cache_pages,
        "allocated_cache_tokens": shape.allocated_cache_tokens,
        "tail_page_valid_tokens": shape.tail_page_valid_tokens,
        "tail_page_unused_tokens": shape.tail_page_unused_tokens,
        "split_parts": SPLIT_PARTS,
        "q_tiles_bm32": M // 32,
        "unsplit_ctas": (M // 32) * H_Q,
        "split_workspace_output_elements": (
            BATCH_SIZE * H_Q * SPLIT_PARTS * M * HEAD_DIM
        ),
        **_split_visibility_metadata(shape),
    }


def _numerical_thresholds() -> dict[str, dict[str, Any]]:
    return {
        "output": {
            "max_abs_lte": OUTPUT_MAX_ABS_THRESHOLD,
            "p99_abs_recorded": True,
        },
        "lse": {
            "max_abs_lte": LSE_MAX_ABS_THRESHOLD,
            "p99_abs_recorded": True,
        },
    }


def _performance_gate_contracts() -> dict[str, dict[str, Any]]:
    return {
        "acceptance": {
            "candidate_vs_reference_p50_pct_lte": ACCEPTANCE_P50_PCT_MAX,
            "candidate_faster_win_count_gte": ACCEPTANCE_CANDIDATE_WIN_COUNT_MIN,
            "paired_rounds": PAIRED_ROUNDS,
            "page_tables": list(ACCEPTANCE_PAGE_TABLES),
            "enforced": True,
        },
        "screen": {
            "candidate_vs_reference_p50_pct_lte": SCREEN_P50_PCT_MAX,
            "candidate_faster_win_count_gte": SCREEN_CANDIDATE_WIN_COUNT_MIN,
            "paired_rounds": SCREEN_PAIRED_ROUNDS,
            "page_tables": list(SCREEN_PAGE_TABLES),
            "enforced": False,
            "screening_not_acceptance": True,
        },
    }


def _source_contract_failures() -> list[str]:
    failures: list[str] = []
    for path in (PYBIND_SOURCE, PAGED_SOURCE):
        if not path.is_file():
            failures.append(f"required source file is missing: {path}")

    if failures:
        return failures

    pybind_text = PYBIND_SOURCE.read_text(encoding="utf-8")
    paged_text = PAGED_SOURCE.read_text(encoding="utf-8")
    for entrypoint in (UNSPLIT_ENTRYPOINT, SPLITKV3_ENTRYPOINT):
        if f'"{entrypoint}"' not in pybind_text:
            failures.append(f"pybind entrypoint is missing: {entrypoint}")
    required_split_tokens = (
        "flash_attention_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3",
        "D256_BM32_PHASE_SPLIT_PARTS = 3",
        "split_tmp_out must have shape [B, H, 3, M, D]",
        "split_tmp_row_max must have shape [B, H, 3, M]",
        "split_tmp_row_sum must have shape [B, H, 3, M]",
    )
    for token in required_split_tokens:
        if token not in paged_text:
            failures.append(f"splitkv3 source contract token is missing: {token}")

    split_function = (
        "flash_attention_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3("
    )
    split_start = paged_text.find(split_function)
    split_end = paged_text.find("return {out_fp16, softmax_lse};", split_start)
    if split_start < 0 or split_end < 0:
        failures.append("could not locate the splitkv3 source contract body")
        return failures
    split_contract_body = paged_text[split_start:split_end]
    host_length_tokens = (
        "const int64_t actual_n",
        "TORCH_CHECK(B == 1",
        "host-length path requires B=1",
        "actual_n >= M",
        "block_table.size(1) >= required_pages",
        "block_table does not cover actual_n host length",
    )
    for token in host_length_tokens:
        if token not in split_contract_body:
            failures.append(f"splitkv3 host-length token is missing: {token}")
    if "seq_lens" in split_contract_body:
        failures.append("splitkv3 host-length body must not accept seq_lens")
    return failures


def _static_contract() -> dict[str, Any]:
    failures: list[str] = []
    acceptance_shape = _kv_shape(N)
    if (BATCH_SIZE, M, N, H_Q, H_KV, HEAD_DIM, PAGE_SIZE) != (
        1,
        8096,
        121440,
        6,
        1,
        256,
        784,
    ):
        failures.append("exact real-shape constants changed")
    if CACHE_PAGES != 155:
        failures.append(f"cache_pages={CACHE_PAGES}, expected 155")
    if ALLOCATED_CACHE_TOKENS != 121520:
        failures.append(
            f"allocated_cache_tokens={ALLOCATED_CACHE_TOKENS}, expected 121520"
        )
    if TAIL_PAGE_VALID_TOKENS != 704:
        failures.append(
            f"tail_page_valid_tokens={TAIL_PAGE_VALID_TOKENS}, expected 704"
        )
    if M % 32 != 0:
        failures.append("M must be divisible by the BM32 query tile")
    if H_Q % H_KV != 0:
        failures.append("Hq must be divisible by Hkv")
    if SPLIT_PARTS != 3:
        failures.append("splitkv3 requires exactly three partitions")
    if SPLITKV3_BLOCK_N != 128:
        failures.append("splitkv3 block-N must remain 128")
    if WARMUP_ROUNDS != 20 or PAIRED_ROUNDS != 100:
        failures.append(
            "production benchmark requires exactly 20 warmups and 100 pairs"
        )
    if LOAD_MONITOR_INTERVAL_MS <= 0:
        failures.append("load monitor interval must be positive")
    if OUTPUT_MAX_ABS_THRESHOLD != 0.001953125:
        failures.append("output max-abs threshold must remain 0.001953125")
    if LSE_MAX_ABS_THRESHOLD != 0.001:
        failures.append("LSE max-abs threshold must remain 0.001")
    if INERT_GRAPHICS_PROCESS_NAME != "/usr/lib/xorg/Xorg":
        failures.append("inert graphics process name must remain /usr/lib/xorg/Xorg")
    if INERT_GRAPHICS_PROCESS_TYPE != "G":
        failures.append("inert graphics process type must remain G")
    if set(ORDER_SCHEDULE[0]) != set(ROUTE_NAMES):
        failures.append("first timing order does not contain each route once")
    if set(ORDER_SCHEDULE[1]) != set(ROUTE_NAMES):
        failures.append("second timing order does not contain each route once")

    expected_pages = tuple(range(CACHE_PAGES))
    page_specs = _page_specs(acceptance_shape)
    if tuple(page_specs) != ACCEPTANCE_PAGE_TABLES:
        failures.append("page-table cases must be normal, reverse, tail")
    if SCREEN_PAGE_TABLES != ("normal",):
        failures.append("screen mode must be restricted to the normal page table")
    if SCREEN_WARMUP_ROUNDS != 5 or SCREEN_PAIRED_ROUNDS != 20:
        failures.append("screen mode requires exactly 5 warmups and 20 pairs")
    if (
        ACCEPTANCE_P50_PCT_MAX,
        ACCEPTANCE_CANDIDATE_WIN_COUNT_MIN,
    ) != (-2.0, 95):
        failures.append("acceptance performance gate must remain p50<=-2% and wins>=95")
    if (SCREEN_P50_PCT_MAX, SCREEN_CANDIDATE_WIN_COUNT_MIN) != (-2.0, 19):
        failures.append("screen performance gate must remain p50<=-2% and wins>=19")
    if (
        SPLITKV3_TOTAL_TILES,
        SPLITKV3_LAST_SPLIT_BEGIN,
        SPLITKV3_EARLIEST_CAUSAL_CUTOFF,
    ) != (949, 80896, 113344):
        failures.append("splitkv3 visibility geometry changed")
    fast_visible_condition = (
        SPLITKV3_LAST_SPLIT_BEGIN <= SPLITKV3_EARLIEST_CAUSAL_CUTOFF
    )
    if fast_visible_condition != SPLITKV3_FAST_VISIBLE_CONDITION:
        failures.append("splitkv3 fast-visible condition is inconsistent")
    if SPLITKV3_EXPECTED_CHECK_SPLIT_EMPTY:
        failures.append("splitkv3 expected CHECK_SPLIT_EMPTY must remain false")
    if (
        acceptance_shape.cache_pages,
        acceptance_shape.allocated_cache_tokens,
        acceptance_shape.tail_page_valid_tokens,
        acceptance_shape.tail_page_unused_tokens,
    ) != (
        CACHE_PAGES,
        ALLOCATED_CACHE_TOKENS,
        TAIL_PAGE_VALID_TOKENS,
        TAIL_PAGE_UNUSED_TOKENS,
    ):
        failures.append("acceptance KV page geometry is inconsistent")
    if (
        acceptance_shape.total_tiles,
        acceptance_shape.last_split_begin,
        acceptance_shape.earliest_causal_cutoff,
        acceptance_shape.fast_visible_condition,
        acceptance_shape.expected_check_split_empty,
    ) != (
        SPLITKV3_TOTAL_TILES,
        SPLITKV3_LAST_SPLIT_BEGIN,
        SPLITKV3_EARLIEST_CAUSAL_CUTOFF,
        SPLITKV3_FAST_VISIBLE_CONDITION,
        SPLITKV3_EXPECTED_CHECK_SPLIT_EMPTY,
    ):
        failures.append("acceptance splitkv3 visibility metadata is inconsistent")
    for name, specification in page_specs.items():
        page_ids = specification["page_ids"]
        if len(page_ids) != CACHE_PAGES:
            failures.append(
                f"{name} block table has {len(page_ids)} pages, expected 155"
            )
        elif tuple(sorted(page_ids)) != expected_pages:
            failures.append(f"{name} block table is not a page permutation")
    if page_specs["tail"]["page_ids"][0] != CACHE_PAGES - 1:
        failures.append("tail block table must put the partial physical page first")

    failures.extend(_source_contract_failures())
    if failures:
        raise RuntimeError("static contract failed: " + "; ".join(failures))
    return {
        "passed": True,
        "geometry": _geometry(acceptance_shape),
        "warmup_rounds": WARMUP_ROUNDS,
        "paired_rounds": PAIRED_ROUNDS,
        "screen_warmup_rounds": SCREEN_WARMUP_ROUNDS,
        "screen_paired_rounds": SCREEN_PAIRED_ROUNDS,
        "acceptance_page_tables": list(ACCEPTANCE_PAGE_TABLES),
        "screen_page_tables": list(SCREEN_PAGE_TABLES),
        "order_schedule": [list(order) for order in ORDER_SCHEDULE],
        "entrypoints": {
            "unsplit": UNSPLIT_ENTRYPOINT,
            "splitkv3": SPLITKV3_ENTRYPOINT,
        },
        "splitkv3_host_length": {
            "actual_n": N,
            "batch_size": BATCH_SIZE,
            "sequence_lengths_tensor": "not accepted by splitkv3",
        },
        "acceptance_length_override": {
            "argument": "--accept-kv-len N",
            "allowed_execution_mode": "acceptance",
            "default_host_actual_n": N,
            "minimum_host_actual_n": M,
            "forbidden_with": [
                "--screen",
                "--static-check",
                "--correctness-only",
                "--screen-kv-len",
            ],
        },
        "numerical_thresholds": _numerical_thresholds(),
        "performance_gate_contracts": _performance_gate_contracts(),
        "splitkv3_visibility": _split_visibility_metadata(acceptance_shape),
        "inert_graphics_policy": {
            "required_name": INERT_GRAPHICS_PROCESS_NAME,
            "required_type": INERT_GRAPHICS_PROCESS_TYPE,
            "identity_fields": ["pid", "name", "type"],
            "must_exist_at_startup_and_remain_unchanged": True,
        },
        "gpu_not_touched": True,
    }


def _configure_cuda_visibility(physical_gpu: int) -> None:
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch was imported before CUDA_VISIBLE_DEVICES could be fixed"
        )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)


def _load_torch() -> None:
    global torch
    if torch is None:
        torch = importlib.import_module("torch")


def _require_cuda_runtime() -> None:
    _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark must expose exactly one GPU through CUDA_VISIBLE_DEVICES"
        )
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError(
            "fixed splitkv3 entry requires SM70; got "
            f"{torch.cuda.get_device_capability(0)} on {torch.cuda.get_device_name(0)}"
        )


def _nvidia_smi() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required for the strict occupancy gate")
    return executable


def _nvidia_smi_xml_gpu(
    executable: str,
    physical_gpu: int,
    *,
    timeout_seconds: float | None = None,
) -> ElementTree.Element:
    result = subprocess.run(
        [executable, "-q", "-x", "-i", str(physical_gpu)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    try:
        root = ElementTree.fromstring(result.stdout)
    except ElementTree.ParseError as exc:
        raise RuntimeError("could not parse nvidia-smi XML") from exc

    gpus = root.findall("gpu")
    if len(gpus) != 1:
        raise RuntimeError(f"expected one GPU in nvidia-smi XML, got {len(gpus)}")
    return gpus[0]


def _xml_gpu_processes(gpu: ElementTree.Element) -> list[dict[str, Any]]:
    process_root = gpu.find("processes")
    if process_root is None:
        raise RuntimeError("nvidia-smi XML omitted the process list")

    processes: list[dict[str, Any]] = []
    for process in process_root.findall("process_info"):
        pid_text = process.findtext("pid")
        if pid_text is None or not pid_text.isdigit():
            raise RuntimeError(f"unexpected GPU process pid: {pid_text!r}")
        processes.append(
            {
                "pid": int(pid_text),
                "name": process.findtext("process_name") or "unknown",
                "type": process.findtext("type") or "unknown",
                "used_memory_mib": process.findtext("used_memory") or "unknown",
            }
        )
    return processes


def _xml_graphics_clock_mhz(gpu: ElementTree.Element) -> int:
    clock_text = gpu.findtext("clocks/graphics_clock")
    match = re.search(r"\d+", clock_text or "")
    if match is None:
        raise RuntimeError(
            f"nvidia-smi XML omitted a parseable graphics clock: {clock_text!r}"
        )
    return int(match.group())


def _process_identity(process: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pid": int(process["pid"]),
        "name": str(process["name"]),
        "type": str(process["type"]),
    }


def _is_inert_graphics_process(process: Mapping[str, Any]) -> bool:
    return (
        process["name"] == INERT_GRAPHICS_PROCESS_NAME
        and process["type"] == INERT_GRAPHICS_PROCESS_TYPE
    )


def _startup_inert_graphics_baseline(
    processes: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    baseline: dict[int, dict[str, Any]] = {}
    for process in processes:
        if not _is_inert_graphics_process(process):
            continue
        identity = _process_identity(process)
        if identity["pid"] in baseline:
            raise RuntimeError(
                "nvidia-smi reported duplicate startup inert graphics PID "
                f"{identity['pid']}"
            )
        baseline[identity["pid"]] = identity
    return baseline


def _inert_graphics_policy_metadata(
    baseline: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "policy": (
            "Only startup-existing /usr/lib/xorg/Xorg processes with type=G are "
            "allowed as inert external graphics contexts."
        ),
        "required_name": INERT_GRAPHICS_PROCESS_NAME,
        "required_type": INERT_GRAPHICS_PROCESS_TYPE,
        "identity_fields": ["pid", "name", "type"],
        "baseline_identities": [dict(baseline[pid]) for pid in sorted(baseline)],
        "baseline_entries_must_remain_present": True,
        "used_memory_mib": "recorded per observation; not part of identity",
    }


def _startup_inert_graphics_metadata(
    baseline: Mapping[int, Mapping[str, Any]],
    processes: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = _inert_graphics_policy_metadata(baseline)
    startup_inert = [
        process for process in processes if _is_inert_graphics_process(process)
    ]
    startup_non_inert = [
        process for process in processes if not _is_inert_graphics_process(process)
    ]
    metadata.update(
        {
            "status": "accepted" if not startup_non_inert else "rejected",
            "startup_observations": startup_inert,
            "startup_non_inert_processes": startup_non_inert,
        }
    )
    return metadata


def _process_policy(
    processes: list[dict[str, Any]],
    *,
    allowed_pids: set[int],
    inert_graphics_baseline: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    observed_inert: list[dict[str, Any]] = []
    changed_inert: list[dict[str, Any]] = []
    unexpected_external: list[dict[str, Any]] = []
    benchmark_processes: list[dict[str, Any]] = []
    observed_baseline_pids: set[int] = set()

    for process in processes:
        pid = int(process["pid"])
        if pid in allowed_pids:
            benchmark_processes.append(process)
            continue

        expected_identity = inert_graphics_baseline.get(pid)
        if expected_identity is None:
            unexpected_external.append(process)
            continue

        if _process_identity(process) == dict(expected_identity):
            observed_inert.append(process)
            observed_baseline_pids.add(pid)
        else:
            changed_inert.append(
                {
                    "expected": dict(expected_identity),
                    "observed": process,
                }
            )

    missing_inert = [
        dict(inert_graphics_baseline[pid])
        for pid in sorted(inert_graphics_baseline)
        if pid not in observed_baseline_pids
        and not any(change["expected"]["pid"] == pid for change in changed_inert)
    ]
    external_compute_or_c_plus_g = [
        process
        for process in processes
        if process["pid"] not in allowed_pids and process["type"] in {"C", "C+G"}
    ]
    passed = not (missing_inert or changed_inert or unexpected_external)
    return {
        "allowed_benchmark_pids": sorted(allowed_pids),
        "benchmark_processes": benchmark_processes,
        "inert_graphics_baseline": [
            dict(inert_graphics_baseline[pid])
            for pid in sorted(inert_graphics_baseline)
        ],
        "observed_inert_graphics": observed_inert,
        "missing_inert_graphics": missing_inert,
        "changed_inert_graphics": changed_inert,
        "external_compute_or_c_plus_g": external_compute_or_c_plus_g,
        "unexpected_external_processes": unexpected_external,
        "passed": passed,
    }


def _gpu_state(physical_gpu: int) -> dict[str, Any]:
    executable = _nvidia_smi()
    query = (
        "index,uuid,name,compute_cap,clocks.current.graphics,clocks.max.graphics,pstate"
    )
    result = subprocess.run(
        [
            executable,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "--id",
            str(physical_gpu),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"unexpected nvidia-smi GPU output: {result.stdout!r}")
    index, uuid, name, capability, current_clock, max_clock, pstate = fields
    gpu = _nvidia_smi_xml_gpu(executable, physical_gpu)
    return {
        "physical_gpu": int(index),
        "uuid": uuid,
        "name": name,
        "compute_capability": capability,
        "graphics_clock_mhz": int(current_clock),
        "graphics_clock_max_mhz": int(max_clock),
        "pstate": pstate,
        "gpu_processes": sorted(
            _xml_gpu_processes(gpu),
            key=lambda process: (process["pid"], process["name"]),
        ),
    }


def _require_exclusive_snapshot(
    state: dict[str, Any],
    *,
    physical_gpu: int,
    phase: str,
    allowed_pids: set[int],
    inert_graphics_baseline: Mapping[int, Mapping[str, Any]],
) -> None:
    process_policy = _process_policy(
        state["gpu_processes"],
        allowed_pids=allowed_pids,
        inert_graphics_baseline=inert_graphics_baseline,
    )
    state["process_policy"] = process_policy
    if state["physical_gpu"] != physical_gpu:
        raise RuntimeError(
            f"{phase}: nvidia-smi returned GPU {state['physical_gpu']}, "
            f"not requested GPU {physical_gpu}"
        )
    if state["compute_capability"] != "7.0":
        raise RuntimeError(f"{phase}: expected SM70, got {state}")
    if state["graphics_clock_mhz"] != REQUIRED_GRAPHICS_CLOCK_MHZ:
        raise RuntimeError(
            f"{phase}: expected {REQUIRED_GRAPHICS_CLOCK_MHZ} MHz, got "
            f"{state['graphics_clock_mhz']} MHz"
        )
    if not process_policy["passed"]:
        raise RuntimeError(f"{phase}: GPU process policy failed: {process_policy}")


def _record_gpu_snapshot(
    result: dict[str, Any],
    *,
    physical_gpu: int,
    phase: str,
    allowed_pids: set[int],
    inert_graphics_baseline: Mapping[int, Mapping[str, Any]],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if state is None:
        state = _gpu_state(physical_gpu)
    result["gpu"]["snapshots"][phase] = state
    _require_exclusive_snapshot(
        state,
        physical_gpu=physical_gpu,
        phase=phase,
        allowed_pids=allowed_pids,
        inert_graphics_baseline=inert_graphics_baseline,
    )
    return state


def _load_monitor_sample(physical_gpu: int) -> dict[str, Any]:
    executable = _nvidia_smi()
    gpu = _nvidia_smi_xml_gpu(
        executable,
        physical_gpu,
        timeout_seconds=LOAD_MONITOR_COMMAND_TIMEOUT_SECONDS,
    )
    return {
        "physical_gpu": physical_gpu,
        "graphics_clock_mhz": _xml_graphics_clock_mhz(gpu),
        "gpu_processes": _xml_gpu_processes(gpu),
    }


class _LoadMonitor:
    """Strictly sample clock and GPU PIDs throughout correctness and timing."""

    def __init__(
        self,
        *,
        physical_gpu: int,
        allowed_pids: set[int],
        inert_graphics_baseline: Mapping[int, Mapping[str, Any]],
        interval_ms: int,
    ) -> None:
        self._physical_gpu = physical_gpu
        self._allowed_pids = frozenset(allowed_pids)
        self._inert_graphics_baseline = {
            int(pid): dict(identity)
            for pid, identity in inert_graphics_baseline.items()
        }
        self._interval_seconds = interval_ms / 1000.0
        self._started_at = time.monotonic()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._violations: list[dict[str, Any]] = []
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name="sm70-splitkv3-load-monitor",
            daemon=True,
        )

    def _record_sample(self, sample: dict[str, Any]) -> None:
        reasons: list[str] = []
        if "error" in sample:
            reasons.append(str(sample["error"]))
        else:
            if sample["physical_gpu"] != self._physical_gpu:
                reasons.append(
                    f"sampled GPU {sample['physical_gpu']}, expected "
                    f"{self._physical_gpu}"
                )
            if sample["graphics_clock_mhz"] != REQUIRED_GRAPHICS_CLOCK_MHZ:
                reasons.append(
                    f"graphics clock={sample['graphics_clock_mhz']} MHz, "
                    f"requires {REQUIRED_GRAPHICS_CLOCK_MHZ} MHz"
                )
            process_policy = _process_policy(
                sample["gpu_processes"],
                allowed_pids=set(self._allowed_pids),
                inert_graphics_baseline=self._inert_graphics_baseline,
            )
            sample["process_policy"] = process_policy
            if not process_policy["passed"]:
                reasons.append(f"GPU process policy failed: {process_policy}")

        with self._lock:
            sample["sample_index"] = len(self._samples)
            sample["valid"] = not reasons
            self._samples.append(sample)
            if reasons:
                self._violations.append(
                    {
                        "sample_index": sample["sample_index"],
                        "elapsed_ms": sample["elapsed_ms"],
                        "reasons": reasons,
                        "sample": sample,
                    }
                )

    def _capture_once(self) -> None:
        started_at = time.monotonic()
        try:
            sample = _load_monitor_sample(self._physical_gpu)
        except Exception as exc:
            sample = {"error": f"{type(exc).__name__}: {exc}"}
        completed_at = time.monotonic()
        sample["elapsed_ms"] = (started_at - self._started_at) * 1000.0
        sample["sample_duration_ms"] = (completed_at - started_at) * 1000.0
        self._record_sample(sample)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._capture_once()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("load monitor was started twice")
        self._started = True
        self._capture_once()
        self._thread.start()

    def assert_healthy(self) -> None:
        with self._lock:
            violations = list(self._violations)
        if violations:
            raise RuntimeError(
                "load monitor detected invalid GPU state "
                f"({len(violations)} sample(s)): {violations[0]}"
            )

    def stop(self) -> dict[str, Any]:
        if self._started:
            self._stop_event.set()
            self._thread.join(timeout=LOAD_MONITOR_COMMAND_TIMEOUT_SECONDS + 1.0)
            if self._thread.is_alive():
                self._record_sample(
                    {
                        "elapsed_ms": (time.monotonic() - self._started_at) * 1000.0,
                        "sample_duration_ms": 0.0,
                        "error": "load monitor thread did not stop",
                    }
                )
        with self._lock:
            samples = list(self._samples)
            violations = list(self._violations)
        return {
            "sampler": "background nvidia-smi XML",
            "interval_ms": LOAD_MONITOR_INTERVAL_MS,
            "expected_graphics_clock_mhz": REQUIRED_GRAPHICS_CLOCK_MHZ,
            "allowed_pids": sorted(self._allowed_pids),
            "allowed_benchmark_pids": sorted(self._allowed_pids),
            "inert_graphics_context": _inert_graphics_policy_metadata(
                self._inert_graphics_baseline
            ),
            "passed": not violations,
            "samples": samples,
            "violations": violations,
        }


def _load_apis() -> tuple[Any, Any, dict[str, str]]:
    if not FLASH_V100_ROOT.is_dir():
        raise RuntimeError(f"Flash-V100 source tree not found: {FLASH_V100_ROOT}")
    if "flash_attn_v100" in sys.modules:
        raise RuntimeError("flash_attn_v100 was imported before this benchmark")
    sys.path.insert(0, str(FLASH_V100_ROOT))

    flash_attn_v100 = importlib.import_module("flash_attn_v100")
    interface = importlib.import_module("flash_attn_v100.flash_attn_interface")
    native_extension = getattr(interface, "flash_attn_v100_cuda", None)
    unsplit = getattr(native_extension, UNSPLIT_ENTRYPOINT, None)
    splitkv3 = getattr(native_extension, SPLITKV3_ENTRYPOINT, None)
    if not all(callable(entry) for entry in (unsplit, splitkv3)):
        raise RuntimeError(
            "Flash-V100 build lacks the accepted unsplit or splitkv3 pybind entry"
        )
    metadata = {
        "flash_attn_v100_package": str(Path(flash_attn_v100.__file__).resolve()),
        "flash_attn_interface": str(Path(interface.__file__).resolve()),
        "native_extension": str(Path(native_extension.__file__).resolve()),
    }
    return unsplit, splitkv3, metadata


def _runtime_metadata(api_metadata: Mapping[str, str]) -> dict[str, Any]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
        "logical_device_name": torch.cuda.get_device_name(0),
        "logical_device_capability": list(torch.cuda.get_device_capability(0)),
        **api_metadata,
    }


def _make_inputs(seed: int, shape: _KvShape) -> tuple[Any, Any, Any, Any]:
    torch.manual_seed(seed)
    query = torch.randn(
        (BATCH_SIZE, H_Q, M, HEAD_DIM),
        device="cuda",
        dtype=torch.float16,
    )
    key_cache = torch.randn(
        (shape.cache_pages, PAGE_SIZE, H_KV, HEAD_DIM),
        device="cuda",
        dtype=torch.float16,
    )
    value_cache = torch.randn_like(key_cache)
    sequence_lengths = torch.full(
        (BATCH_SIZE,),
        shape.n,
        device="cuda",
        dtype=torch.int32,
    )
    return query, key_cache, value_cache, sequence_lengths


def _make_workspace(query: Any) -> _SplitWorkspace:
    workspace = _SplitWorkspace(
        output=torch.empty(
            (BATCH_SIZE, H_Q, SPLIT_PARTS, M, HEAD_DIM),
            device=query.device,
            dtype=torch.float32,
        ),
        row_max=torch.empty(
            (BATCH_SIZE, H_Q, SPLIT_PARTS, M),
            device=query.device,
            dtype=torch.float32,
        ),
        row_sum=torch.empty(
            (BATCH_SIZE, H_Q, SPLIT_PARTS, M),
            device=query.device,
            dtype=torch.float32,
        ),
    )
    expected_shapes = {
        "output": (BATCH_SIZE, H_Q, SPLIT_PARTS, M, HEAD_DIM),
        "row_max": (BATCH_SIZE, H_Q, SPLIT_PARTS, M),
        "row_sum": (BATCH_SIZE, H_Q, SPLIT_PARTS, M),
    }
    for name, tensor in (
        ("output", workspace.output),
        ("row_max", workspace.row_max),
        ("row_sum", workspace.row_sum),
    ):
        if tuple(tensor.shape) != expected_shapes[name]:
            raise RuntimeError(
                f"split workspace {name} has wrong shape: {tensor.shape}"
            )
        if tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise RuntimeError(f"split workspace {name} must be contiguous fp32")
    return workspace


def _workspace_metadata(workspace: _SplitWorkspace) -> dict[str, Any]:
    return {
        "output": {
            "shape": list(workspace.output.shape),
            "dtype": str(workspace.output.dtype),
            "contiguous": bool(workspace.output.is_contiguous()),
            "data_ptr": workspace.output.data_ptr(),
        },
        "row_max": {
            "shape": list(workspace.row_max.shape),
            "dtype": str(workspace.row_max.dtype),
            "contiguous": bool(workspace.row_max.is_contiguous()),
            "data_ptr": workspace.row_max.data_ptr(),
        },
        "row_sum": {
            "shape": list(workspace.row_sum.shape),
            "dtype": str(workspace.row_sum.dtype),
            "contiguous": bool(workspace.row_sum.is_contiguous()),
            "data_ptr": workspace.row_sum.data_ptr(),
        },
    }


def _make_cases(query: Any, shape: _KvShape) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for name, specification in _page_specs(shape).items():
        page_ids = torch.tensor(
            specification["page_ids"],
            device=query.device,
            dtype=torch.int32,
        )
        cases[name] = {
            "block_table": page_ids.view(BATCH_SIZE, shape.cache_pages),
            "unsplit_out": torch.empty_like(query),
            "unsplit_lse": torch.empty(
                query.shape[:-1],
                device=query.device,
                dtype=torch.float32,
            ),
            "splitkv3_out": torch.empty_like(query),
            "splitkv3_lse": torch.empty(
                query.shape[:-1],
                device=query.device,
                dtype=torch.float32,
            ),
            "specification": specification,
        }
    return cases


def _assert_alias(returned: Any, preallocated: Any, label: str) -> None:
    if returned.data_ptr() != preallocated.data_ptr():
        raise RuntimeError(f"{label} did not honor its preallocated buffer")


def _assert_finite(tensor: Any, label: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{label} contains non-finite values")


def _absolute_error(
    reference: Any,
    candidate: Any,
    label: str,
    *,
    max_abs_threshold: float,
) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise RuntimeError(
            f"{label} shape mismatch: {tuple(reference.shape)} != "
            f"{tuple(candidate.shape)}"
        )
    if reference.dtype != candidate.dtype:
        raise RuntimeError(
            f"{label} dtype mismatch: {reference.dtype} != {candidate.dtype}"
        )
    _assert_finite(reference, f"{label} reference")
    _assert_finite(candidate, f"{label} candidate")
    difference = (reference.float() - candidate.float()).abs().flatten()
    max_abs = float(difference.max().item())
    return {
        "max_abs": max_abs,
        "p99_abs": float(torch.quantile(difference, 0.99).item()),
        "max_abs_lte": max_abs_threshold,
        "max_abs_passed": max_abs <= max_abs_threshold,
        "bitwise_equal": bool(
            torch.equal(reference.view(torch.uint8), candidate.view(torch.uint8))
        ),
        "elements": difference.numel(),
    }


def _numerical_gate_failures(
    name: str,
    output_abs: Mapping[str, Any],
    lse_abs: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    for label, evidence in (("output", output_abs), ("LSE", lse_abs)):
        if not evidence["max_abs_passed"]:
            failures.append(
                f"{name} {label} max_abs={evidence['max_abs']:.9g} exceeds "
                f"{evidence['max_abs_lte']:.9g}"
            )
    return failures


def _launch_unsplit(
    entry: Any,
    *,
    query: Any,
    key_cache: Any,
    value_cache: Any,
    sequence_lengths: Any,
    case: Mapping[str, Any],
) -> tuple[Any, Any]:
    return entry(
        query,
        key_cache,
        value_cache,
        case["unsplit_out"],
        case["unsplit_lse"],
        case["block_table"],
        sequence_lengths,
        SOFTMAX_SCALE,
    )


def _launch_splitkv3(
    entry: Any,
    *,
    query: Any,
    key_cache: Any,
    value_cache: Any,
    workspace: _SplitWorkspace,
    case: Mapping[str, Any],
    actual_n: int,
) -> tuple[Any, Any]:
    return entry(
        query,
        key_cache,
        value_cache,
        case["splitkv3_out"],
        case["splitkv3_lse"],
        workspace.output,
        workspace.row_max,
        workspace.row_sum,
        case["block_table"],
        actual_n,
        SOFTMAX_SCALE,
    )


def _run_correctness(
    *,
    cases: Mapping[str, Mapping[str, Any]],
    query: Any,
    key_cache: Any,
    value_cache: Any,
    sequence_lengths: Any,
    workspace: _SplitWorkspace,
    unsplit_entry: Any,
    splitkv3_entry: Any,
    correctness: dict[str, Any],
    shape: _KvShape,
    health_check: Callable[[], None],
) -> None:
    gate_failures: list[str] = []
    for name, case in cases.items():
        health_check()
        unsplit_out, unsplit_lse = _launch_unsplit(
            unsplit_entry,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            sequence_lengths=sequence_lengths,
            case=case,
        )
        splitkv3_out, splitkv3_lse = _launch_splitkv3(
            splitkv3_entry,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            workspace=workspace,
            case=case,
            actual_n=shape.n,
        )
        torch.cuda.synchronize()
        _assert_alias(unsplit_out, case["unsplit_out"], f"{name} unsplit output")
        _assert_alias(unsplit_lse, case["unsplit_lse"], f"{name} unsplit LSE")
        _assert_alias(splitkv3_out, case["splitkv3_out"], f"{name} splitkv3 output")
        _assert_alias(splitkv3_lse, case["splitkv3_lse"], f"{name} splitkv3 LSE")
        output_abs = _absolute_error(
            unsplit_out,
            splitkv3_out,
            f"{name} output",
            max_abs_threshold=OUTPUT_MAX_ABS_THRESHOLD,
        )
        lse_abs = _absolute_error(
            unsplit_lse,
            splitkv3_lse,
            f"{name} LSE",
            max_abs_threshold=LSE_MAX_ABS_THRESHOLD,
        )
        correctness[name] = {
            "reference": UNSPLIT_ROUTE,
            "candidate": SPLITKV3_ROUTE,
            "output_abs": output_abs,
            "lse_abs": lse_abs,
            "buffers_preallocated": {
                "unsplit_output": True,
                "unsplit_lse": True,
                "splitkv3_output": True,
                "splitkv3_lse": True,
                "split_workspace": True,
            },
        }
        gate_failures.extend(_numerical_gate_failures(name, output_abs, lse_abs))
        health_check()
    if gate_failures:
        raise RuntimeError("numerical gate failed: " + "; ".join(gate_failures))


def _percentile(samples: list[float], percentile: float) -> float:
    if not samples:
        raise RuntimeError("cannot calculate a percentile without samples")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"invalid percentile: {percentile}")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _timing_stats(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": samples,
        "mean_ms": float(statistics.mean(samples)),
        "min_ms": float(min(samples)),
        "p50_ms": _percentile(samples, 50.0),
        "p90_ms": _percentile(samples, 90.0),
        "p99_ms": _percentile(samples, 99.0),
        "max_ms": float(max(samples)),
    }


def _paired_stats(raw_rounds: list[dict[str, Any]]) -> dict[str, Any]:
    paired_rounds = len(raw_rounds)
    if paired_rounds == 0:
        raise RuntimeError("paired comparison requires at least one round")
    unsplit_samples = [
        float(round_data["samples_ms"][UNSPLIT_ROUTE]) for round_data in raw_rounds
    ]
    splitkv3_samples = [
        float(round_data["samples_ms"][SPLITKV3_ROUTE]) for round_data in raw_rounds
    ]
    deltas = [
        splitkv3_ms - unsplit_ms
        for unsplit_ms, splitkv3_ms in zip(unsplit_samples, splitkv3_samples)
    ]
    splitkv3_wins = sum(
        splitkv3_ms < unsplit_ms
        for unsplit_ms, splitkv3_ms in zip(unsplit_samples, splitkv3_samples)
    )
    unsplit_wins = sum(
        unsplit_ms < splitkv3_ms
        for unsplit_ms, splitkv3_ms in zip(unsplit_samples, splitkv3_samples)
    )
    ties = paired_rounds - splitkv3_wins - unsplit_wins
    unsplit_p50 = _percentile(unsplit_samples, 50.0)
    splitkv3_p50 = _percentile(splitkv3_samples, 50.0)
    unsplit_p90 = _percentile(unsplit_samples, 90.0)
    splitkv3_p90 = _percentile(splitkv3_samples, 90.0)
    unsplit_p99 = _percentile(unsplit_samples, 99.0)
    splitkv3_p99 = _percentile(splitkv3_samples, 99.0)
    return {
        "reference": UNSPLIT_ROUTE,
        "candidate": SPLITKV3_ROUTE,
        "candidate_minus_reference_mean_ms": float(statistics.mean(deltas)),
        "candidate_minus_reference_p50_ms": _percentile(deltas, 50.0),
        "candidate_vs_reference_mean_pct": (
            (statistics.mean(splitkv3_samples) / statistics.mean(unsplit_samples) - 1.0)
            * 100.0
        ),
        "candidate_vs_reference_p50_pct": (splitkv3_p50 / unsplit_p50 - 1.0) * 100.0,
        "candidate_vs_reference_p90_pct": (splitkv3_p90 / unsplit_p90 - 1.0) * 100.0,
        "candidate_vs_reference_p99_pct": (splitkv3_p99 / unsplit_p99 - 1.0) * 100.0,
        "candidate_faster_win_count": splitkv3_wins,
        "candidate_faster_win_percent": (splitkv3_wins / paired_rounds) * 100.0,
        "reference_faster_win_count": unsplit_wins,
        "reference_faster_win_percent": (unsplit_wins / paired_rounds) * 100.0,
        "tie_count": ties,
        "tie_percent": (ties / paired_rounds) * 100.0,
        "raw_delta_ms": deltas,
    }


def _time_route(route: _Route, events: tuple[Any, Any]) -> float:
    start, end = events
    start.record()
    route.launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _run_interleaved_timing(
    routes: Mapping[str, _Route],
    *,
    warmup_rounds: int,
    paired_rounds: int,
    health_check: Callable[[], None],
) -> dict[str, Any]:
    if tuple(sorted(routes)) != tuple(sorted(ROUTE_NAMES)):
        raise RuntimeError(f"unexpected route set: {sorted(routes)}")
    if warmup_rounds < 0:
        raise RuntimeError("warmup rounds must be non-negative")
    if paired_rounds <= 0:
        raise RuntimeError("paired rounds must be positive")

    for warmup_round in range(warmup_rounds):
        health_check()
        for name in ORDER_SCHEDULE[warmup_round % len(ORDER_SCHEDULE)]:
            routes[name].launch()
        health_check()
    torch.cuda.synchronize()
    health_check()

    events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in ROUTE_NAMES
    }
    raw_rounds: list[dict[str, Any]] = []
    for round_index in range(paired_rounds):
        health_check()
        order = ORDER_SCHEDULE[round_index % len(ORDER_SCHEDULE)]
        samples = {name: _time_route(routes[name], events[name]) for name in order}
        raw_rounds.append(
            {
                "round": round_index,
                "route_order": list(order),
                "samples_ms": samples,
            }
        )
        health_check()

    route_samples = {
        name: [float(round_data["samples_ms"][name]) for round_data in raw_rounds]
        for name in ROUTE_NAMES
    }
    return {
        "warmup_rounds": warmup_rounds,
        "paired_rounds": paired_rounds,
        "order_schedule": [list(order) for order in ORDER_SCHEDULE],
        "raw_rounds": raw_rounds,
        "route_stats": {
            name: _timing_stats(route_samples[name]) for name in ROUTE_NAMES
        },
        "paired_comparison": _paired_stats(raw_rounds),
    }


def _evaluate_performance_gate(
    timing: Mapping[str, Any],
    *,
    gate_name: str,
    page_tables: tuple[str, ...],
    p50_pct_max: float,
    candidate_win_count_min: int,
    expected_paired_rounds: int,
    enforced: bool,
) -> dict[str, Any]:
    case_results: dict[str, Any] = {}
    failure_reasons: list[str] = []
    for name in page_tables:
        timing_case = timing.get(name)
        if not isinstance(timing_case, Mapping):
            reason = "timing record is missing"
            case_results[name] = {"passed": False, "reasons": [reason]}
            failure_reasons.append(f"{name}: {reason}")
            continue
        comparison = timing_case.get("paired_comparison")
        if not isinstance(comparison, Mapping):
            reason = "paired comparison is missing"
            case_results[name] = {"passed": False, "reasons": [reason]}
            failure_reasons.append(f"{name}: {reason}")
            continue

        p50_pct = float(comparison["candidate_vs_reference_p50_pct"])
        candidate_wins = int(comparison["candidate_faster_win_count"])
        observed_rounds = int(timing_case["paired_rounds"])
        p50_passed = p50_pct <= p50_pct_max
        wins_passed = candidate_wins >= candidate_win_count_min
        rounds_passed = observed_rounds == expected_paired_rounds
        reasons: list[str] = []
        if not p50_passed:
            reasons.append(
                f"candidate_vs_reference_p50_pct={p50_pct:.6f} exceeds "
                f"{p50_pct_max:.6f}"
            )
        if not wins_passed:
            reasons.append(
                f"candidate_faster_win_count={candidate_wins} is below "
                f"{candidate_win_count_min}/{expected_paired_rounds}"
            )
        if not rounds_passed:
            reasons.append(
                f"paired_rounds={observed_rounds}, expected {expected_paired_rounds}"
            )
        case_results[name] = {
            "candidate_vs_reference_p50_pct": p50_pct,
            "candidate_vs_reference_p50_pct_lte": p50_pct_max,
            "candidate_vs_reference_p50_pct_passed": p50_passed,
            "candidate_faster_win_count": candidate_wins,
            "candidate_faster_win_count_gte": candidate_win_count_min,
            "candidate_faster_win_count_passed": wins_passed,
            "paired_rounds": observed_rounds,
            "paired_rounds_expected": expected_paired_rounds,
            "paired_rounds_passed": rounds_passed,
            "passed": not reasons,
            "reasons": reasons,
        }
        failure_reasons.extend(f"{name}: {reason}" for reason in reasons)

    return {
        "name": gate_name,
        "enforced": enforced,
        "screening_not_acceptance": gate_name == "screen",
        "page_tables": list(page_tables),
        "candidate_vs_reference_p50_pct_lte": p50_pct_max,
        "candidate_faster_win_count_gte": candidate_win_count_min,
        "paired_rounds_expected": expected_paired_rounds,
        "status": "passed" if not failure_reasons else "failed",
        "passed": not failure_reasons,
        "case_results": case_results,
        "failure_reasons": failure_reasons,
    }


def _case_routes(
    *,
    case: Mapping[str, Any],
    query: Any,
    key_cache: Any,
    value_cache: Any,
    sequence_lengths: Any,
    workspace: _SplitWorkspace,
    unsplit_entry: Any,
    splitkv3_entry: Any,
    shape: _KvShape,
) -> dict[str, _Route]:
    def unsplit() -> Any:
        return _launch_unsplit(
            unsplit_entry,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            sequence_lengths=sequence_lengths,
            case=case,
        )

    def splitkv3() -> Any:
        return _launch_splitkv3(
            splitkv3_entry,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            workspace=workspace,
            case=case,
            actual_n=shape.n,
        )

    return {
        UNSPLIT_ROUTE: _Route(UNSPLIT_ROUTE, unsplit),
        SPLITKV3_ROUTE: _Route(SPLITKV3_ROUTE, splitkv3),
    }


def _route_metadata(shape: _KvShape) -> dict[str, Any]:
    return {
        UNSPLIT_ROUTE: {
            "entrypoint": f"flash_attn_v100_cuda.{UNSPLIT_ENTRYPOINT}",
            "causal": True,
            "out_preallocated": True,
            "softmax_lse_preallocated": True,
            "sequence_length_transport": "GPU int32 sequence_lengths tensor",
            "generic_dispatch": "not used; direct fixed pybind entry",
        },
        SPLITKV3_ROUTE: {
            "entrypoint": f"flash_attn_v100_cuda.{SPLITKV3_ENTRYPOINT}",
            "causal": True,
            "out_preallocated": True,
            "softmax_lse_preallocated": True,
            "workspace_preallocated": True,
            "workspace_output_shape": [BATCH_SIZE, H_Q, SPLIT_PARTS, M, HEAD_DIM],
            "workspace_row_shape": [BATCH_SIZE, H_Q, SPLIT_PARTS, M],
            "workspace_dtype": "torch.float32",
            "sequence_length_transport": {
                "kind": "host actual_n",
                "value": shape.n,
                "gpu_sequence_lengths_tensor": "not accepted",
            },
            "host_actual_n": shape.n,
            "split_visibility": _split_visibility_metadata(shape),
            "expected_kernel": {
                "CHECK_SPLIT_EMPTY": shape.expected_check_split_empty,
            },
            "generic_dispatch": "not used; direct fixed pybind entry",
        },
    }


def _execution_mode(args: argparse.Namespace) -> str:
    if args.static_check:
        return "static_check"
    if args.correctness_only:
        return "correctness_only"
    if args.screen:
        return "screening_not_acceptance"
    return "acceptance"


def _mode_metadata(args: argparse.Namespace, shape: _KvShape) -> dict[str, Any]:
    execution_mode = _execution_mode(args)
    acceptance_host_n = shape.n if execution_mode == "acceptance" else N
    return {
        "execution_mode": execution_mode,
        "screening_not_acceptance": args.screen,
        "acceptance_contract": {
            "page_tables": list(ACCEPTANCE_PAGE_TABLES),
            "warmup_rounds_per_page_table": WARMUP_ROUNDS,
            "paired_rounds_per_page_table": PAIRED_ROUNDS,
            "executed": execution_mode == "acceptance",
            "host_actual_n": acceptance_host_n,
            "acceptance_host_n": acceptance_host_n,
            "length_override_requested": (
                execution_mode == "acceptance" and args.accept_kv_len != N
            ),
            "performance_gate": _performance_gate_contracts()["acceptance"],
        },
        "screen_contract": {
            "page_tables": list(SCREEN_PAGE_TABLES),
            "correctness_before_timing": True,
            "warmup_rounds_per_page_table": SCREEN_WARMUP_ROUNDS,
            "paired_rounds_per_page_table": SCREEN_PAIRED_ROUNDS,
            "executed": args.screen,
            "host_actual_n": shape.n,
            "fast_visible_predicate": _split_visibility_metadata(shape)[
                "fast_visible_predicate"
            ],
            "result_must_not_be_used_as_acceptance": True,
            "performance_gate": _performance_gate_contracts()["screen"],
        },
    }


def _result_template(args: argparse.Namespace, shape: _KvShape) -> dict[str, Any]:
    mode_metadata = _mode_metadata(args, shape)
    is_acceptance = mode_metadata["execution_mode"] == "acceptance"
    page_tables = {
        name: {
            "page_ids": list(specification["page_ids"]),
            "description": specification["description"],
        }
        for name, specification in _page_specs(shape).items()
    }
    return {
        "schema_version": 1,
        "benchmark": "sm70_flashinfer_splitkv3_realshape_ab",
        "scope": (
            "CUDA-event operator microbenchmark only; no model, engine, HTTP, "
            "or end-to-end execution is performed."
        ),
        "status": "running",
        "geometry": _geometry(shape),
        "configuration": {
            "physical_gpu": args.physical_gpu,
            "execution_mode": mode_metadata["execution_mode"],
            "correctness_only": args.correctness_only,
            "screen": args.screen,
            "screen_kv_len": shape.n if args.screen else None,
            "accept_kv_len": shape.n if is_acceptance else None,
            "acceptance_host_n": shape.n if is_acceptance else None,
            "host_actual_n": shape.n,
            "fast_visible_predicate": _split_visibility_metadata(shape)[
                "fast_visible_predicate"
            ],
            "execution_page_tables": list(
                SCREEN_PAGE_TABLES if args.screen else ACCEPTANCE_PAGE_TABLES
            ),
            "warmup_rounds_per_page_table": (
                SCREEN_WARMUP_ROUNDS if args.screen else WARMUP_ROUNDS
            ),
            "paired_rounds_per_page_table": (
                SCREEN_PAIRED_ROUNDS if args.screen else PAIRED_ROUNDS
            ),
            "load_monitor_interval_ms": LOAD_MONITOR_INTERVAL_MS,
            "seed": args.seed,
        },
        "mode_metadata": mode_metadata,
        "routes": _route_metadata(shape),
        "page_tables": page_tables,
        "numerical_thresholds": _numerical_thresholds(),
        "numerical_gate": {
            "status": "not_run",
            "thresholds": _numerical_thresholds(),
        },
        "performance_gate": {
            "status": "not_run",
            "contracts": _performance_gate_contracts(),
        },
        "gpu": {
            "required_graphics_clock_mhz": REQUIRED_GRAPHICS_CLOCK_MHZ,
            "external_gpu_pid_pollution": "fail",
            "frequency_deviation": "fail",
            "inert_graphics_context": {
                "status": "not_captured",
                **_inert_graphics_policy_metadata({}),
                "startup_observations": [],
                "startup_non_inert_processes": [],
            },
            "load_monitor": {"status": "not_started"},
            "snapshots": {},
        },
    }


def _run_gpu_benchmark(
    args: argparse.Namespace,
    result: dict[str, Any],
    shape: _KvShape,
) -> None:
    _configure_cuda_visibility(args.physical_gpu)
    startup_state = _gpu_state(args.physical_gpu)
    inert_graphics_baseline = _startup_inert_graphics_baseline(
        startup_state["gpu_processes"]
    )
    result["gpu"]["inert_graphics_context"] = _startup_inert_graphics_metadata(
        inert_graphics_baseline,
        startup_state["gpu_processes"],
    )
    before = _record_gpu_snapshot(
        result,
        physical_gpu=args.physical_gpu,
        phase="before_cuda_initialization",
        allowed_pids=set(),
        inert_graphics_baseline=inert_graphics_baseline,
        state=startup_state,
    )
    result["gpu"]["processes_before"] = before["gpu_processes"]
    _require_cuda_runtime()
    unsplit_entry, splitkv3_entry, api_metadata = _load_apis()
    result["runtime"] = _runtime_metadata(api_metadata)

    query, key_cache, value_cache, sequence_lengths = _make_inputs(args.seed, shape)
    workspace = _make_workspace(query)
    cases = _make_cases(query, shape)
    correctness_cases = (
        {name: cases[name] for name in SCREEN_PAGE_TABLES} if args.screen else cases
    )
    result["correctness_scope"] = {
        "page_tables": list(correctness_cases),
        "screening_not_acceptance": args.screen,
    }
    result["split_workspace"] = _workspace_metadata(workspace)
    torch.cuda.synchronize()
    _record_gpu_snapshot(
        result,
        physical_gpu=args.physical_gpu,
        phase="after_preallocation",
        allowed_pids={os.getpid()},
        inert_graphics_baseline=inert_graphics_baseline,
    )

    monitor = _LoadMonitor(
        physical_gpu=args.physical_gpu,
        allowed_pids={os.getpid()},
        inert_graphics_baseline=inert_graphics_baseline,
        interval_ms=LOAD_MONITOR_INTERVAL_MS,
    )
    with torch.inference_mode():
        try:
            monitor.start()
            monitor.assert_healthy()
            result["correctness"] = {}
            result["numerical_gate"]["status"] = "running"
            try:
                _run_correctness(
                    cases=correctness_cases,
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    sequence_lengths=sequence_lengths,
                    workspace=workspace,
                    unsplit_entry=unsplit_entry,
                    splitkv3_entry=splitkv3_entry,
                    correctness=result["correctness"],
                    shape=shape,
                    health_check=monitor.assert_healthy,
                )
            except BaseException:
                result["numerical_gate"]["status"] = "failed"
                raise
            result["numerical_gate"]["status"] = "passed"
            torch.cuda.synchronize()
            monitor.assert_healthy()
            _record_gpu_snapshot(
                result,
                physical_gpu=args.physical_gpu,
                phase="after_correctness",
                allowed_pids={os.getpid()},
                inert_graphics_baseline=inert_graphics_baseline,
            )

            if args.correctness_only:
                result["timing"] = {
                    "status": "skipped",
                    "reason": "--correctness-only",
                    "warmup_rounds_executed_per_page_table": 0,
                    "paired_rounds_executed_per_page_table": 0,
                }
                result["timing_scope"] = {
                    "page_tables": [],
                    "screening_not_acceptance": False,
                }
            else:
                timing_cases = (
                    {name: cases[name] for name in SCREEN_PAGE_TABLES}
                    if args.screen
                    else cases
                )
                timing_warmup_rounds = (
                    SCREEN_WARMUP_ROUNDS if args.screen else WARMUP_ROUNDS
                )
                timing_paired_rounds = (
                    SCREEN_PAIRED_ROUNDS if args.screen else PAIRED_ROUNDS
                )
                timing: dict[str, Any] = {}
                for name, case in timing_cases.items():
                    _record_gpu_snapshot(
                        result,
                        physical_gpu=args.physical_gpu,
                        phase=f"before_timing_{name}",
                        allowed_pids={os.getpid()},
                        inert_graphics_baseline=inert_graphics_baseline,
                    )
                    timing[name] = _run_interleaved_timing(
                        _case_routes(
                            case=case,
                            query=query,
                            key_cache=key_cache,
                            value_cache=value_cache,
                            sequence_lengths=sequence_lengths,
                            workspace=workspace,
                            unsplit_entry=unsplit_entry,
                            splitkv3_entry=splitkv3_entry,
                            shape=shape,
                        ),
                        warmup_rounds=timing_warmup_rounds,
                        paired_rounds=timing_paired_rounds,
                        health_check=monitor.assert_healthy,
                    )
                    torch.cuda.synchronize()
                    monitor.assert_healthy()
                    _record_gpu_snapshot(
                        result,
                        physical_gpu=args.physical_gpu,
                        phase=f"after_timing_{name}",
                        allowed_pids={os.getpid()},
                        inert_graphics_baseline=inert_graphics_baseline,
                    )
                result["timing"] = timing
                result["timing_scope"] = {
                    "page_tables": list(timing_cases),
                    "warmup_rounds_per_page_table": timing_warmup_rounds,
                    "paired_rounds_per_page_table": timing_paired_rounds,
                    "screening_not_acceptance": args.screen,
                }
        finally:
            monitor_record = monitor.stop()
            result["gpu"]["load_monitor"] = monitor_record

    monitor.assert_healthy()
    final_phase = (
        "after_correctness_only"
        if args.correctness_only
        else "after_screen"
        if args.screen
        else "after_timing"
    )
    after = _record_gpu_snapshot(
        result,
        physical_gpu=args.physical_gpu,
        phase=final_phase,
        allowed_pids={os.getpid()},
        inert_graphics_baseline=inert_graphics_baseline,
    )
    result["gpu"]["processes_after"] = after["gpu_processes"]

    if args.correctness_only:
        result["performance_gate"] = {
            "status": "not_run",
            "reason": "--correctness-only skips performance timing",
            "contracts": _performance_gate_contracts(),
        }
        return

    if args.screen:
        result["performance_gate"] = _evaluate_performance_gate(
            result["timing"],
            gate_name="screen",
            page_tables=SCREEN_PAGE_TABLES,
            p50_pct_max=SCREEN_P50_PCT_MAX,
            candidate_win_count_min=SCREEN_CANDIDATE_WIN_COUNT_MIN,
            expected_paired_rounds=SCREEN_PAIRED_ROUNDS,
            enforced=False,
        )
        return

    performance_gate = _evaluate_performance_gate(
        result["timing"],
        gate_name="acceptance",
        page_tables=ACCEPTANCE_PAGE_TABLES,
        p50_pct_max=ACCEPTANCE_P50_PCT_MAX,
        candidate_win_count_min=ACCEPTANCE_CANDIDATE_WIN_COUNT_MIN,
        expected_paired_rounds=PAIRED_ROUNDS,
        enforced=True,
    )
    result["performance_gate"] = performance_gate
    if not performance_gate["passed"]:
        raise RuntimeError(
            "acceptance performance gate failed: "
            + "; ".join(performance_gate["failure_reasons"])
        )


def _emit_result(args: argparse.Namespace, result: Mapping[str, Any]) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out is None:
        print(payload)
        return
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "json_out": str(args.json_out)}))


def main() -> int:
    args = _parse_args()
    shape = _shape_for_args(args)
    result = _result_template(args, shape)
    exit_code = 0
    try:
        result["static_contract"] = _static_contract()
        if args.static_check:
            result["status"] = "static_check_passed"
        else:
            _run_gpu_benchmark(args, result, shape)
            result["status"] = (
                "correctness_only_passed"
                if args.correctness_only
                else "screening_not_acceptance"
                if args.screen
                else "passed"
            )
    except BaseException as exc:
        exit_code = 1
        result["status"] = (
            "screening_not_acceptance_failed" if args.screen else "failed"
        )
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    _emit_result(args, result)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
