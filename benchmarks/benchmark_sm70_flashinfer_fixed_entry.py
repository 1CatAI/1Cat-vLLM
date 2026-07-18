# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Operator-only SM70 long-context paged/fixed/dense A/B microbenchmark.

The exact geometry is M=8096, N=121440, Hq=6, Hkv=1, D=256, page=784.
It compares the accepted generic paged BM32 route, the fixed BM32 pybind
entry, and, only for an identity page table, a zero-copy contiguous dense
Flash-V100 view. This is not an end-to-end or full-backend benchmark.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# These must precede importing torch so logical cuda:0 is physical GPU0.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


SOURCE_ROOT = Path(__file__).resolve().parents[1]
FLASH_V100_ROOT = SOURCE_ROOT / "flash-attention-v100"
EXPECTED_PYTHON = Path("/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python")
DEFAULT_EXTENSION = (
    FLASH_V100_ROOT / "flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so"
)
OUTPUT_DEFAULT = (
    SOURCE_ROOT
    / "bench_results"
    / "flashinfer_sm70_20260716"
    / "fixed_entry_dense_operator_ab100_m8096_n121440_clock1200.json"
)
REQUIRED_GRAPHICS_CLOCK_MHZ = 1200

BATCH_SIZE = 1
QUERY_LEN = 8096
KV_LEN = 121440
Q_HEADS = 6
KV_HEADS = 1
HEAD_DIM = 256
PAGE_SIZE = 784
CACHE_PAGES = (KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE
ALLOCATED_CACHE_TOKENS = CACHE_PAGES * PAGE_SIZE
TAIL_PAGE_VALID_TOKENS = KV_LEN - (CACHE_PAGES - 1) * PAGE_SIZE

PAGED_BM32_ENV = {
    "VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM": "1",
    "VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE": "1",
    "VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P": "1",
    "VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH": "1",
}
DENSE_ENV = {
    "VLLM_FLASH_V100_DENSE_D256_WMMA_QK": "1",
    "VLLM_FLASH_V100_DENSE_D256_LOW_SMEM": "0",
    "VLLM_FLASH_V100_PREFILL_SCALAR_PV": "0",
}
CONTROL_ENV = tuple((*PAGED_BM32_ENV, *DENSE_ENV))

PAGED_SYMBOL = "flash_attention_forward_paged_d256_bm32_phase_kernelILb1ELb1ELb1EE"
DENSE_SYMBOL = "flash_attention_forward_kernelILi256ELb0ELb1ELb1EE"
SM70_REGISTERS_PER_SM = 65536
SM70_SHARED_BYTES_PER_SM = 98304
SM70_THREADS_PER_SM = 2048
SM70_MAX_CTA_PER_SM = 32
PAGED_THREADS_PER_CTA = 512
PAGED_MAX_SHARED_BYTES = 49152
DENSE_THREADS_PER_CTA = 512
# KernelConfig<256, false>::TOTAL_SMEM in fused_mha_forward.cu.
DENSE_DYNAMIC_SHARED_BYTES = 96512
REQUIRED_P50_SPEEDUP_PCT = 2.0
REQUIRED_FASTER_WINS = 95
REQUIRED_PAIRED_ROUNDS = 100

# Delayed so --inspect-resources is genuinely CPU-only.
torch: Any = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--json-out", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--extension-path", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument(
        "--inspect-resources",
        action="store_true",
        help=(
            "Inspect the extension with cuobjdump without importing torch "
            "or using CUDA."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow fewer rounds for an explicitly non-acceptance smoke run.",
    )
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.warmup_rounds < 0:
        parser.error("--warmup-rounds must be non-negative")
    if not args.inspect_resources and not args.smoke:
        if args.rounds != REQUIRED_PAIRED_ROUNDS:
            parser.error(f"formal runs require --rounds={REQUIRED_PAIRED_ROUNDS}")
        if args.warmup_rounds < 20:
            parser.error("formal runs require --warmup-rounds >= 20")
    return args


def _require_expected_python() -> None:
    if not EXPECTED_PYTHON.exists() or not Path(sys.executable).samefile(
        EXPECTED_PYTHON
    ):
        raise RuntimeError(
            "accepted evidence requires "
            f"{EXPECTED_PYTHON}; got {Path(sys.executable).resolve()}"
        )


def _load_torch() -> None:
    global torch
    if torch is None:
        torch = importlib.import_module("torch")


def _require_accepted_runtime() -> None:
    _require_expected_python()
    _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark must expose only physical GPU0 through CUDA_VISIBLE_DEVICES=0"
        )
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError(
            "fixed entry requires SM70; got "
            f"{torch.cuda.get_device_capability(0)} on {torch.cuda.get_device_name(0)}"
        )


def _gpu_state() -> dict[str, Any]:
    query = (
        "index,name,uuid,compute_cap,clocks.current.graphics,"
        "clocks.max.graphics,pstate,driver_version"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 8:
        raise RuntimeError(f"unexpected nvidia-smi output: {completed.stdout!r}")
    return {
        "physical_index": int(fields[0]),
        "name": fields[1],
        "uuid": fields[2],
        "compute_capability": fields[3],
        "graphics_clock_mhz": int(fields[4]),
        "graphics_clock_max_mhz": int(fields[5]),
        "pstate": fields[6],
        "driver_version": fields[7],
    }


def _require_fixed_clock(state: Mapping[str, Any], stage: str) -> None:
    if state["physical_index"] != 0:
        raise RuntimeError(f"{stage}: expected physical GPU0, got {state}")
    if state["compute_capability"] != "7.0":
        raise RuntimeError(f"{stage}: expected SM70 GPU0, got {state}")
    if state["graphics_clock_mhz"] != REQUIRED_GRAPHICS_CLOCK_MHZ:
        raise RuntimeError(
            f"{stage}: expected {REQUIRED_GRAPHICS_CLOCK_MHZ}MHz graphics clock, "
            f"got {state['graphics_clock_mhz']}MHz"
        )


@contextmanager
def _restore_control_env():
    saved = {name: os.environ.get(name) for name in CONTROL_ENV}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _set_env(values: Mapping[str, str]) -> None:
    for name, value in values.items():
        os.environ[name] = value


def _function_sections(text: str, symbol: str) -> list[str]:
    sections: list[str] = []
    name: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if name is not None and symbol in name:
            sections.append("".join(lines))

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("Function"):
            flush()
            name = stripped[len("Function") :].strip()
            if name.startswith(":"):
                name = name[1:].strip()
            name = name.rstrip(":").strip()
            lines = [line]
        elif name is not None:
            lines.append(line)
    flush()
    return sections


def _instruction_count(sass: str, mnemonic: str) -> int:
    return len(re.findall(rf"\b{re.escape(mnemonic)}(?:\.|\b)", sass))


def _parse_resource_section(section: str) -> dict[str, int] | None:
    values: dict[str, int] = {}
    for name in ("REG", "STACK", "SHARED", "LOCAL"):
        match = re.search(rf"\b{name}\s*:\s*(\d+)", section)
        if match is None:
            return None
        values[name.lower()] = int(match.group(1))
    return values


def _active_ctas_per_sm(
    *,
    registers_per_thread: int,
    threads_per_cta: int,
    shared_bytes_per_cta: int,
) -> int:
    register_limited = SM70_REGISTERS_PER_SM // (
        registers_per_thread * threads_per_cta
    )
    shared_limited = SM70_SHARED_BYTES_PER_SM // shared_bytes_per_cta
    thread_limited = SM70_THREADS_PER_SM // threads_per_cta
    return min(register_limited, shared_limited, thread_limited, SM70_MAX_CTA_PER_SM)


def _inspect_kernel(
    *,
    resource_dump: str,
    sass_dump: str,
    symbol: str,
    threads_per_cta: int,
    dynamic_shared_bytes: int,
) -> dict[str, Any]:
    resource_sections = _function_sections(resource_dump, symbol)
    sass_sections = _function_sections(sass_dump, symbol)
    if len(resource_sections) != 1 or len(sass_sections) != 1:
        return {
            "available": False,
            "symbol": symbol,
            "resource_section_count": len(resource_sections),
            "sass_section_count": len(sass_sections),
        }

    resource = _parse_resource_section(resource_sections[0])
    if resource is None:
        return {
            "available": False,
            "symbol": symbol,
            "reason": "cuobjdump resource fields were incomplete",
        }
    sass = sass_sections[0]
    instructions = {
        name: _instruction_count(sass, mnemonic)
        for name, mnemonic in (
            ("hmma", "HMMA"),
            ("ldg", "LDG"),
            ("lds", "LDS"),
            ("sts", "STS"),
            ("ldl", "LDL"),
            ("stl", "STL"),
            ("bar", "BAR"),
        )
    }
    active_ctas = _active_ctas_per_sm(
        registers_per_thread=resource["reg"],
        threads_per_cta=threads_per_cta,
        shared_bytes_per_cta=dynamic_shared_bytes,
    )
    return {
        "available": True,
        "symbol": symbol,
        "registers_per_thread": resource["reg"],
        "stack_bytes_per_thread": resource["stack"],
        "static_shared_bytes": resource["shared"],
        "local_bytes_per_thread": resource["local"],
        "dynamic_shared_bytes": dynamic_shared_bytes,
        "threads_per_cta": threads_per_cta,
        "active_ctas_per_sm": active_ctas,
        "resident_warps": active_ctas * (threads_per_cta // 32),
        "instructions": instructions,
    }


def _paged_resource_gate(kernel: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not kernel.get("available"):
        reasons.append("paged true,true,true kernel was absent from cuobjdump")
        return {"passed": False, "reasons": reasons}
    if kernel["registers_per_thread"] > 128:
        reasons.append(f"REG={kernel['registers_per_thread']}, requires <=128")
    if kernel["stack_bytes_per_thread"] != 0:
        reasons.append(
            f"STACK={kernel['stack_bytes_per_thread']}, requires 0 for paged gate"
        )
    if kernel["local_bytes_per_thread"] != 0:
        reasons.append(
            f"LOCAL={kernel['local_bytes_per_thread']}, requires 0 for paged gate"
        )
    if kernel["dynamic_shared_bytes"] > PAGED_MAX_SHARED_BYTES:
        reasons.append(
            f"shared={kernel['dynamic_shared_bytes']}, "
            f"requires <= {PAGED_MAX_SHARED_BYTES}"
        )
    if kernel["active_ctas_per_sm"] != 2:
        reasons.append(
            f"active_ctas_per_sm={kernel['active_ctas_per_sm']}, requires 2"
        )
    instructions = kernel["instructions"]
    if instructions["hmma"] == 0:
        reasons.append("paged target has no HMMA")
    if instructions["ldl"] or instructions["stl"]:
        reasons.append(
            "paged target has local-memory instructions "
            f"LDL={instructions['ldl']} STL={instructions['stl']}"
        )
    return {"passed": not reasons, "reasons": reasons}


def _dense_resource_audit(kernel: Mapping[str, Any]) -> dict[str, Any]:
    comparison_reasons: list[str] = []
    strict_paged_reasons: list[str] = []
    if not kernel.get("available"):
        comparison_reasons.append("dense D256 WMMA target was absent from cuobjdump")
        return {
            "comparison_ready": False,
            "comparison_reasons": comparison_reasons,
            "strict_paged_gate_passed": False,
            "strict_paged_gate_reasons": comparison_reasons,
        }
    if kernel["registers_per_thread"] > 128:
        comparison_reasons.append(
            f"REG={kernel['registers_per_thread']}, requires <=128"
        )
    if kernel["local_bytes_per_thread"] != 0:
        comparison_reasons.append(
            f"LOCAL={kernel['local_bytes_per_thread']}, requires 0"
        )
    if kernel["dynamic_shared_bytes"] > SM70_SHARED_BYTES_PER_SM:
        comparison_reasons.append(
            f"shared={kernel['dynamic_shared_bytes']} exceeds SM70 limit"
        )
    if kernel["active_ctas_per_sm"] != 1:
        comparison_reasons.append(
            f"active_ctas_per_sm={kernel['active_ctas_per_sm']}, expected 1"
        )
    if kernel["instructions"]["hmma"] == 0:
        comparison_reasons.append("dense target has no HMMA")

    if kernel["stack_bytes_per_thread"] != 0:
        strict_paged_reasons.append(
            f"STACK={kernel['stack_bytes_per_thread']} rather than 0"
        )
    if kernel["instructions"]["ldl"] or kernel["instructions"]["stl"]:
        strict_paged_reasons.append(
            "local-memory instructions "
            f"LDL={kernel['instructions']['ldl']} "
            f"STL={kernel['instructions']['stl']}"
        )
    if kernel["dynamic_shared_bytes"] > PAGED_MAX_SHARED_BYTES:
        strict_paged_reasons.append(
            f"shared={kernel['dynamic_shared_bytes']} exceeds paged 48KiB gate"
        )
    if kernel["active_ctas_per_sm"] != 2:
        strict_paged_reasons.append(
            f"active_ctas_per_sm={kernel['active_ctas_per_sm']} rather than 2"
        )
    return {
        "comparison_ready": not comparison_reasons,
        "comparison_reasons": comparison_reasons,
        "strict_paged_gate_passed": not strict_paged_reasons,
        "strict_paged_gate_reasons": strict_paged_reasons,
    }


def _inspect_resources(extension_path: Path) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return {"available": False, "reason": "cuobjdump was not found"}
    if not extension_path.is_file():
        return {
            "available": False,
            "reason": f"extension does not exist: {extension_path}",
        }
    resource_result = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(extension_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    sass_result = subprocess.run(
        [cuobjdump, "--dump-sass", str(extension_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if resource_result.returncode or sass_result.returncode:
        return {
            "available": False,
            "extension_path": str(extension_path),
            "resource_returncode": resource_result.returncode,
            "sass_returncode": sass_result.returncode,
            "resource_stderr": resource_result.stderr[-1000:],
            "sass_stderr": sass_result.stderr[-1000:],
        }

    paged = _inspect_kernel(
        resource_dump=resource_result.stdout,
        sass_dump=sass_result.stdout,
        symbol=PAGED_SYMBOL,
        threads_per_cta=PAGED_THREADS_PER_CTA,
        dynamic_shared_bytes=41936,
    )
    dense = _inspect_kernel(
        resource_dump=resource_result.stdout,
        sass_dump=sass_result.stdout,
        symbol=DENSE_SYMBOL,
        threads_per_cta=DENSE_THREADS_PER_CTA,
        dynamic_shared_bytes=DENSE_DYNAMIC_SHARED_BYTES,
    )
    return {
        "available": True,
        "extension_path": str(extension_path.resolve()),
        "sm70_limits": {
            "registers_per_sm": SM70_REGISTERS_PER_SM,
            "shared_bytes_per_sm": SM70_SHARED_BYTES_PER_SM,
            "threads_per_sm": SM70_THREADS_PER_SM,
        },
        "generic_and_fixed_paged_bm32": paged,
        "generic_and_fixed_strict_gate": _paged_resource_gate(paged),
        "dense_d256_wmma": dense,
        "dense_comparator_audit": _dense_resource_audit(dense),
    }


def _make_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    query = torch.randn(
        (BATCH_SIZE, Q_HEADS, QUERY_LEN, HEAD_DIM),
        dtype=torch.float16,
        device="cuda",
    )
    key_cache = torch.randn(
        (CACHE_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM),
        dtype=torch.float16,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    seq_lens = torch.tensor([KV_LEN], dtype=torch.int32, device="cuda")
    normal = torch.arange(CACHE_PAGES, dtype=torch.int32, device="cuda")
    reverse = torch.flip(normal, dims=(0,))
    return query, key_cache, value_cache, seq_lens, normal, reverse


def _make_zero_copy_dense_view(
    cache: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if KV_HEADS != 1 or cache.shape[2] != 1:
        raise RuntimeError("zero-copy dense view is only valid for Hkv=1")
    view = cache.view(BATCH_SIZE, KV_HEADS, ALLOCATED_CACHE_TOKENS, HEAD_DIM)[
        :, :, :KV_LEN, :
    ]
    info = {
        "eligible": True,
        "layout": "cache[P,page,1,D].view(1,1,P*page,D)[..., :N, :]",
        "is_contiguous": bool(view.is_contiguous()),
        "aliases_cache_data_ptr": view.data_ptr() == cache.data_ptr(),
        "logical_tokens": KV_LEN,
        "allocated_tokens": ALLOCATED_CACHE_TOKENS,
    }
    if not info["is_contiguous"] or not info["aliases_cache_data_ptr"]:
        raise RuntimeError(f"dense identity view unexpectedly copied: {info}")
    return view, info


def _make_case(
    name: str,
    table_ids: torch.Tensor,
    query: torch.Tensor,
) -> dict[str, Any]:
    return {
        "name": name,
        "block_table": table_ids.view(1, -1).contiguous(),
        "generic_out": torch.empty_like(query),
        "fixed_out": torch.empty_like(query),
        "fixed_lse": torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        ),
    }


def _byte_exact_evidence(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, Any]:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {
            "bitwise_equal": False,
            "shape_match": list(expected.shape) == list(actual.shape),
            "dtype_match": expected.dtype == actual.dtype,
        }
    expected_bytes = expected.view(torch.uint8)
    actual_bytes = actual.view(torch.uint8)
    equal = bool(torch.equal(expected_bytes, actual_bytes))
    evidence: dict[str, Any] = {
        "bitwise_equal": equal,
        "shape_match": True,
        "dtype_match": True,
    }
    if not equal:
        evidence["different_bytes"] = int(
            torch.count_nonzero(expected_bytes != actual_bytes).item()
        )
    return evidence


def _dense_diagnostic_evidence(
    reference: torch.Tensor,
    dense: torch.Tensor,
) -> dict[str, Any]:
    evidence = _byte_exact_evidence(reference, dense)
    diff = (reference.float() - dense.float()).abs()
    evidence.update(
        {
            "comparison_policy": (
                "diagnostic only: no tolerance is used as an exactness gate; "
                "dense D256 has different 64-wide online-softmax panels than "
                "paged BM32's 16-wide FP16-P panels"
            ),
            "finite": bool(torch.isfinite(dense).all().item()),
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
        }
    )
    return evidence


def _time_once(fn: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _sample_stats(samples: Iterable[float]) -> dict[str, float]:
    values = list(samples)
    return {
        "mean_ms": float(statistics.mean(values)),
        "p50_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _pairwise_stats(
    samples: Mapping[str, list[float]],
    *,
    left: str,
    right: str,
) -> dict[str, Any]:
    left_samples = samples[left]
    right_samples = samples[right]
    paired = list(zip(left_samples, right_samples))
    deltas = [right_ms - left_ms for left_ms, right_ms in paired]
    left_stats = _sample_stats(left_samples)
    right_stats = _sample_stats(right_samples)
    return {
        "right_minus_left_mean_ms": float(statistics.mean(deltas)),
        "right_minus_left_p50_ms": float(statistics.median(deltas)),
        "right_faster_wins": sum(
            right_ms < left_ms for left_ms, right_ms in paired
        ),
        "left_faster_wins": sum(
            left_ms < right_ms for left_ms, right_ms in paired
        ),
        "ties": sum(
            left_ms == right_ms for left_ms, right_ms in paired
        ),
        "right_vs_left_mean_pct": (
            (right_stats["mean_ms"] / left_stats["mean_ms"] - 1.0) * 100.0
        ),
        "right_vs_left_p50_pct": (
            (right_stats["p50_ms"] / left_stats["p50_ms"] - 1.0) * 100.0
        ),
        "deltas_ms": deltas,
    }


def _performance_gate(
    pair: Mapping[str, Any],
    *,
    rounds: int,
    comparison: str,
) -> dict[str, Any]:
    speedup_p50_pct = -float(pair["right_vs_left_p50_pct"])
    wins = int(pair["right_faster_wins"])
    reasons: list[str] = []
    if rounds != REQUIRED_PAIRED_ROUNDS:
        reasons.append(
            f"rounds={rounds}, requires {REQUIRED_PAIRED_ROUNDS} paired rounds"
        )
    if speedup_p50_pct < REQUIRED_P50_SPEEDUP_PCT:
        reasons.append(
            f"p50 speedup={speedup_p50_pct:.3f}%, "
            f"requires >= {REQUIRED_P50_SPEEDUP_PCT:.1f}%"
        )
    if wins < REQUIRED_FASTER_WINS:
        reasons.append(f"faster wins={wins}, requires >= {REQUIRED_FASTER_WINS}")
    return {
        "comparison": comparison,
        "passed": not reasons,
        "p50_speedup_pct": speedup_p50_pct,
        "faster_wins": wins,
        "required_p50_speedup_pct": REQUIRED_P50_SPEEDUP_PCT,
        "required_faster_wins": REQUIRED_FASTER_WINS,
        "required_rounds": REQUIRED_PAIRED_ROUNDS,
        "reasons": reasons,
    }


def _interleaved_time(
    routes: Mapping[str, Callable[[], Any]],
    *,
    warmup_rounds: int,
    rounds: int,
) -> dict[str, Any]:
    names = tuple(routes)
    orders = tuple(itertools.permutations(names))
    for warmup_index in range(warmup_rounds):
        for name in orders[warmup_index % len(orders)]:
            routes[name]()
    torch.cuda.synchronize()

    samples = {name: [] for name in names}
    for round_index in range(rounds):
        for name in orders[round_index % len(orders)]:
            samples[name].append(_time_once(routes[name]))

    pairs = {
        "fixed_vs_generic": _pairwise_stats(samples, left="generic", right="fixed")
    }
    if "dense" in routes:
        pairs["dense_vs_generic"] = _pairwise_stats(
            samples, left="generic", right="dense"
        )
    return {
        "rounds": rounds,
        "warmup_rounds": warmup_rounds,
        "route_order_schedule": "round-robin over every route permutation",
        "routes": {
            name: {**_sample_stats(samples[name]), "samples_ms": samples[name]}
            for name in names
        },
        "pairs": pairs,
    }


def _runtime_metadata(flash_attn_v100: Any, interface: Any) -> dict[str, Any]:
    extension = interface.flash_attn_v100_cuda
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
        "flash_attn_v100_package_path": str(Path(flash_attn_v100.__file__).resolve()),
        "flash_attn_interface_path": str(Path(interface.__file__).resolve()),
        "extension_path": str(Path(extension.__file__).resolve()),
    }


def _write_json(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = _parse_args()
    result: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "sm70_flashinfer_fixed_entry_dense_operator_ab",
        "scope": (
            "CUDA-event operator microbenchmark only; this is not an end-to-end "
            "model or full-backend performance claim."
        ),
        "status": "running",
        "geometry": {
            "batch_size": BATCH_SIZE,
            "query_len_m": QUERY_LEN,
            "kv_len_n": KV_LEN,
            "heads_q": Q_HEADS,
            "heads_kv": KV_HEADS,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
            "cache_pages": CACHE_PAGES,
            "allocated_cache_tokens": ALLOCATED_CACHE_TOKENS,
            "tail_page_valid_tokens": TAIL_PAGE_VALID_TOKENS,
            "paged_grid_x": QUERY_LEN // 32,
            "ctas": (QUERY_LEN // 32) * Q_HEADS,
        },
        "generic": {
            "entrypoint": "flash_attn_prefill_paged_bhmd",
            "causal": True,
            "dispatch_env": PAGED_BM32_ENV,
            "out_preallocated": True,
            "softmax_lse_preallocated": False,
        },
        "fixed": {
            "entrypoint": "flash_attn_prefill_paged_d256_bm32_allp_pair_scratch",
            "fixed_kernel_specialization": "causal=true, all_p=true, pair_scratch=true",
            "out_preallocated": True,
            "softmax_lse_preallocated": True,
            "cuda_graph_capture": (
                "not measured; native entry rejects unvalidated capture"
            ),
        },
        "dense": {
            "entrypoint": "flash_attn_bhmd_func",
            "causal": True,
            "dispatch_env": DENSE_ENV,
            "out_preallocated": True,
            "softmax_lse_preallocated": False,
            "role": (
                "zero-copy contiguous dense comparator, not a fixed-entry candidate"
            ),
        },
    }

    try:
        _require_expected_python()
        resources = _inspect_resources(args.extension_path)
        result["resource_audit"] = resources
        if not resources.get("available"):
            raise RuntimeError(f"resource inspection unavailable: {resources}")
        strict_gate = resources["generic_and_fixed_strict_gate"]
        if not strict_gate["passed"]:
            raise RuntimeError(
                f"paged generic/fixed resource gate failed: {strict_gate}"
            )
        dense_audit = resources["dense_comparator_audit"]
        if not dense_audit["comparison_ready"]:
            raise RuntimeError(f"dense comparator resource audit failed: {dense_audit}")
        if args.inspect_resources:
            result["status"] = "resource_ready_cpu_only"
            _write_json(args.json_out, result)
            print(json.dumps({"status": result["status"], "json": str(args.json_out)}))
            return

        _require_accepted_runtime()
        if str(FLASH_V100_ROOT) not in sys.path:
            sys.path.insert(0, str(FLASH_V100_ROOT))
        import flash_attn_v100
        from flash_attn_v100 import flash_attn_interface

        runtime = _runtime_metadata(flash_attn_v100, flash_attn_interface)
        if Path(runtime["extension_path"]) != args.extension_path.resolve():
            raise RuntimeError(
                "loaded extension differs from resource-audited extension: "
                f"{runtime['extension_path']} vs {args.extension_path.resolve()}"
            )
        fixed_api = getattr(
            flash_attn_v100,
            "flash_attn_prefill_paged_d256_bm32_allp_pair_scratch",
            None,
        )
        dense_api = getattr(flash_attn_v100, "flash_attn_bhmd_func", None)
        if fixed_api is None or dense_api is None:
            raise RuntimeError(
                "loaded extension is missing fixed or dense Flash-V100 API"
            )

        clock_before = _gpu_state()
        _require_fixed_clock(clock_before, "before allocation")
        result["runtime"] = runtime
        result["gpu_clock_mhz"] = {"before": clock_before}

        (
            query,
            key_cache,
            value_cache,
            seq_lens,
            normal_ids,
            reverse_ids,
        ) = _make_inputs(args.seed)
        dense_k, dense_k_info = _make_zero_copy_dense_view(key_cache)
        dense_v, dense_v_info = _make_zero_copy_dense_view(value_cache)
        normal = _make_case("normal", normal_ids, query)
        normal["dense_out"] = torch.empty_like(query)
        normal["dense_k"] = dense_k
        normal["dense_v"] = dense_v
        reverse = _make_case("reverse", reverse_ids, query)
        cases = [normal, reverse]
        result["page_tables"] = {
            "normal": {
                "description": "identity physical-page order",
                "dense_zero_copy": {"k": dense_k_info, "v": dense_v_info},
            },
            "reverse": {
                "description": "reverse physical-page order",
                "dense_zero_copy": {
                    "eligible": False,
                    "reason": (
                        "logical token order crosses physical pages in reverse; "
                        "a contiguous [B,H,N,D] view would change semantics and "
                        "materialization would not be zero-copy"
                    ),
                },
            },
        }

        with _restore_control_env():
            _set_env(DENSE_ENV)
            fixed_exactness: dict[str, Any] = {}
            dense_exactness: dict[str, Any] = {}
            for case in cases:
                _set_env(PAGED_BM32_ENV)
                generic_out = flash_attn_v100.flash_attn_prefill_paged_bhmd(
                    query,
                    key_cache,
                    value_cache,
                    case["block_table"],
                    seq_lens,
                    out=case["generic_out"],
                    causal=True,
                )
                if generic_out.data_ptr() != case["generic_out"].data_ptr():
                    raise RuntimeError(
                        "generic paged entry did not honor preallocated out"
                    )

                _set_env({name: "0" for name in PAGED_BM32_ENV})
                fixed_out, fixed_lse = fixed_api(
                    query,
                    key_cache,
                    value_cache,
                    case["block_table"],
                    seq_lens,
                    out=case["fixed_out"],
                    softmax_lse=case["fixed_lse"],
                )
                torch.cuda.synchronize()
                if fixed_out.data_ptr() != case["fixed_out"].data_ptr():
                    raise RuntimeError("fixed entry allocated replacement output")
                if fixed_lse.data_ptr() != case["fixed_lse"].data_ptr():
                    raise RuntimeError("fixed entry allocated replacement softmax_lse")
                fixed_evidence = _byte_exact_evidence(generic_out, fixed_out)
                fixed_evidence["fixed_lse_finite"] = bool(
                    torch.isfinite(fixed_lse).all().item()
                )
                fixed_evidence["fixed_out_preallocated"] = True
                fixed_evidence["fixed_lse_preallocated"] = True
                fixed_exactness[case["name"]] = fixed_evidence
                if not fixed_evidence["bitwise_equal"]:
                    raise RuntimeError(
                        f"fixed entry bitwise mismatch for {case['name']} page table"
                    )
                if not fixed_evidence["fixed_lse_finite"]:
                    raise RuntimeError(
                        f"fixed entry produced non-finite LSE for {case['name']}"
                    )

                if case["name"] == "normal":
                    _set_env(PAGED_BM32_ENV)
                    dense_out = dense_api(
                        query,
                        case["dense_k"],
                        case["dense_v"],
                        causal=True,
                        out=case["dense_out"],
                    )
                    torch.cuda.synchronize()
                    if dense_out.data_ptr() != case["dense_out"].data_ptr():
                        raise RuntimeError("dense entry allocated replacement output")
                    dense_evidence = _dense_diagnostic_evidence(generic_out, dense_out)
                    dense_evidence["out_preallocated"] = True
                    dense_exactness[case["name"]] = dense_evidence
                    if not dense_evidence["finite"]:
                        raise RuntimeError("dense zero-copy output is non-finite")
                else:
                    dense_exactness[case["name"]] = {
                        "status": "skipped",
                        "reason": result["page_tables"]["reverse"]["dense_zero_copy"][
                            "reason"
                        ],
                    }

            _set_env(PAGED_BM32_ENV)
            _set_env(DENSE_ENV)
            result["fixed_env_independence_probe"] = {
                "env_values_during_fixed_exactness": {
                    name: "0" for name in PAGED_BM32_ENV
                },
                "bitwise_against_generic_env_1": fixed_exactness,
            }
            result["dense_correctness_diagnostic"] = dense_exactness

            for case in cases:

                def generic(case: dict[str, Any] = case) -> torch.Tensor:
                    return flash_attn_v100.flash_attn_prefill_paged_bhmd(
                        query,
                        key_cache,
                        value_cache,
                        case["block_table"],
                        seq_lens,
                        out=case["generic_out"],
                        causal=True,
                    )

                def fixed(
                    case: dict[str, Any] = case,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    return fixed_api(
                        query,
                        key_cache,
                        value_cache,
                        case["block_table"],
                        seq_lens,
                        out=case["fixed_out"],
                        softmax_lse=case["fixed_lse"],
                    )

                routes: dict[str, Callable[[], Any]] = {
                    "generic": generic,
                    "fixed": fixed,
                }
                if case["name"] == "normal":

                    def dense(case: dict[str, Any] = case) -> torch.Tensor:
                        return dense_api(
                            query,
                            case["dense_k"],
                            case["dense_v"],
                            causal=True,
                            out=case["dense_out"],
                        )

                    routes["dense"] = dense
                case["timing"] = _interleaved_time(
                    routes,
                    warmup_rounds=args.warmup_rounds,
                    rounds=args.rounds,
                )

        clock_after = _gpu_state()
        _require_fixed_clock(clock_after, "after timing")
        result["gpu_clock_mhz"]["after"] = clock_after
        result["results"] = {case["name"]: case["timing"] for case in cases}
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        _write_json(args.json_out, result)
        raise
    else:
        _write_json(args.json_out, result)
        print(json.dumps({"status": result["status"], "json": str(args.json_out)}))


if __name__ == "__main__":
    main()
