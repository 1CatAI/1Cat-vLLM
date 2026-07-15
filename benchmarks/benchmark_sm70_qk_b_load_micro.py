# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark exact SM70 WMMA QK matrix-B load paths at BM16/BN128/K256."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

PATHS = ("native", "direct", "shuffle")
K_BLOCK_M = 16
K_BLOCK_N = 128
K_HEAD_DIM = 256
K_REQUIRED_VISIBLE_DEVICES = "2"
K_REQUIRED_CLOCK_MHZ = 1200
K_ARTIFACT_RELATIVE_PATH = Path(
    "bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/"
    "prefill_pipeline_20260714/wmma_matrix_ab_compact_b_fragment_probe.json"
)


def _load_extension() -> object:
    source = Path(__file__).resolve().parent / "csrc" / "sm70_qk_b_load_micro.cu"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    return load(
        name="sm70_qk_b_load_micro_v1",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"],
        verbose=os.getenv("VLLM_SM70_QK_B_LOAD_MICRO_VERBOSE") == "1",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--panels", type=int, default=1024)
    parser.add_argument("--token-stride", type=int, default=K_HEAD_DIM)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--launches-per-sample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--clock-mhz", type=int, default=K_REQUIRED_CLOCK_MHZ)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Record the gated NCU command; it is never run before wall-time passes.",
    )
    return parser.parse_args()


def _require_environment(args: argparse.Namespace, device: torch.device) -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != K_REQUIRED_VISIBLE_DEVICES:
        raise RuntimeError(
            "This benchmark requires CUDA_VISIBLE_DEVICES=2 so cuda:0 is "
            "physical GPU 2."
        )
    if args.physical_gpu != 2:
        raise RuntimeError("This benchmark is fixed to physical GPU 2.")
    if args.clock_mhz != K_REQUIRED_CLOCK_MHZ:
        raise RuntimeError("This benchmark is fixed to a 1200 MHz graphics clock.")
    if device.type != "cuda" or device.index not in (None, 0):
        raise RuntimeError("Use logical cuda:0 after CUDA_VISIBLE_DEVICES=2.")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This benchmark requires an SM70 CUDA device.")
    if args.panels < 1 or args.warmup < 0:
        raise ValueError("panels must be positive and warmup cannot be negative.")
    if not args.profile_only and args.rounds < 100:
        raise ValueError("rounds must be at least 100.")
    if args.launches_per_sample < 1:
        raise ValueError("launches-per-sample must be positive.")
    if args.token_stride < K_HEAD_DIM or args.token_stride % 8:
        raise ValueError("token-stride must be at least 256 and divisible by 8.")


def _graphics_clock_mhz(physical_gpu: int) -> int:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required to verify the locked clock.")
    command = [
        nvidia_smi,
        "--query-gpu=clocks.current.graphics",
        "--format=csv,noheader,nounits",
        "--id",
        str(physical_gpu),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    match = re.search(r"(\d+)", result.stdout)
    if match is None:
        raise RuntimeError(f"Could not parse graphics clock: {result.stdout!r}")
    return int(match.group(1))


def _verify_clock(args: argparse.Namespace) -> int:
    clock_mhz = _graphics_clock_mhz(args.physical_gpu)
    if clock_mhz != args.clock_mhz:
        raise RuntimeError(
            f"GPU {args.physical_gpu} graphics clock is {clock_mhz} MHz, "
            f"not the required {args.clock_mhz} MHz."
        )
    return clock_mhz


def _mapping_evidence() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    artifact = root / K_ARTIFACT_RELATIVE_PATH
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    matrix_b = payload["matrix_b"]
    lanes = matrix_b["lane_elements"]
    paired_lanes_equal = len(lanes) == 32 and all(
        lanes[lane] == lanes[lane ^ 4] for lane in range(32)
    )
    compact_exact = bool(matrix_b["compact_col_fragment_words_equal"])
    compact_xor = int(matrix_b["compact_col_fragment_max_word_xor"])
    if not compact_exact or compact_xor != 0 or not paired_lanes_equal:
        raise RuntimeError("The authoritative matrix-B fragment mapping is not valid.")
    return {
        "artifact": str(K_ARTIFACT_RELATIVE_PATH),
        "probe_source": "benchmarks/csrc/sm70_wmma_fragment_probe.cu",
        "compact_col_fragment_words_equal": compact_exact,
        "compact_col_fragment_max_word_xor": compact_xor,
        "lane_xor_4_words_equal": paired_lanes_equal,
    }


def _time_batch(fn: Callable[[], None], launches_per_sample: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches_per_sample):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / launches_per_sample


def _timing_summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "min_us": min(samples),
        "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
        "max_us": max(samples),
    }


def _uint32_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.view(torch.uint32), right.view(torch.uint32)))


def _uint32_mismatch_words(left: torch.Tensor, right: torch.Tensor) -> int:
    different = torch.ne(left.view(torch.uint32), right.view(torch.uint32))
    return int(torch.count_nonzero(different).item())


def _function_sections(text: str, symbol: str) -> str | None:
    pattern = re.compile(r"(?m)^\s*Function\s*(?::)?\s*(.+?)\s*$")
    headers = list(pattern.finditer(text))
    for index, header in enumerate(headers):
        if symbol not in header.group(1):
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        return text[header.start() : end]
    return None


def _instruction_count(sass: str, mnemonic: str) -> int:
    return len(re.findall(rf"\b{re.escape(mnemonic)}(?:\.|\b)", sass))


def _global_load_summary(sass: str) -> dict[str, int]:
    lines = [
        line for line in sass.splitlines() if "LDG" in line and ".CONSTANT" not in line
    ]
    return {
        "instructions": len(lines),
        "vector_128_instructions": sum(".128" in line for line in lines),
        "predicated_instructions": sum(
            "@" in line.split("LDG", 1)[0] for line in lines
        ),
    }


def _inspect_binary(extension: object) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return {"available": False, "reason": "cuobjdump was not found"}

    extension_path = Path(extension.__file__).resolve()
    sass_result = subprocess.run(
        [cuobjdump, "--dump-sass", str(extension_path)],
        text=True,
        capture_output=True,
    )
    resource_result = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(extension_path)],
        text=True,
        capture_output=True,
    )
    if sass_result.returncode or resource_result.returncode:
        return {
            "available": False,
            "extension": str(extension_path),
            "sass_returncode": sass_result.returncode,
            "resource_returncode": resource_result.returncode,
            "sass_stderr": sass_result.stderr[-1000:],
            "resource_stderr": resource_result.stderr[-1000:],
        }

    symbols = {
        "native": "qk_b_load_native_kernel",
        "direct": "qk_b_load_direct_kernel",
        "shuffle": "qk_b_load_shuffle_kernel",
    }
    kernels: dict[str, Any] = {}
    for path, symbol in symbols.items():
        sass = _function_sections(sass_result.stdout, symbol)
        resources = _function_sections(resource_result.stdout, symbol)
        if sass is None or resources is None:
            kernels[path] = {"available": False, "symbol": symbol}
            continue
        registers = re.search(r"\bREG\s*:\s*(\d+)", resources)
        local_bytes = re.search(r"\bLOCAL\s*:\s*(\d+)", resources)
        kernels[path] = {
            "available": True,
            "symbol": symbol,
            "registers": int(registers.group(1)) if registers else None,
            "local_bytes": int(local_bytes.group(1)) if local_bytes else None,
            "instructions": {
                "hmma": _instruction_count(sass, "HMMA"),
                "ldg": _instruction_count(sass, "LDG"),
                "shfl": _instruction_count(sass, "SHFL"),
                "ldl": _instruction_count(sass, "LDL"),
                "stl": _instruction_count(sass, "STL"),
            },
            "global_loads": _global_load_summary(sass),
        }
    return {
        "available": True,
        "extension": str(extension_path),
        "kernels": kernels,
    }


def _resource_gate(binary: dict[str, Any]) -> tuple[bool, list[str]]:
    if not binary.get("available"):
        return False, ["SASS/resource inspection unavailable"]
    shuffle = binary["kernels"].get("shuffle", {})
    if not shuffle.get("available"):
        return False, ["shuffle kernel was absent from cuobjdump output"]
    reasons: list[str] = []
    registers = shuffle.get("registers")
    local_bytes = shuffle.get("local_bytes")
    instructions = shuffle.get("instructions", {})
    global_loads = shuffle.get("global_loads", {})
    if registers is None or registers > 64:
        reasons.append(f"shuffle REG={registers}, requires <=64")
    if local_bytes is None or local_bytes != 0:
        reasons.append(f"shuffle LOCAL={local_bytes}, requires 0")
    if instructions.get("ldl", 0) or instructions.get("stl", 0):
        reasons.append(
            "shuffle SASS has local-memory instructions "
            f"LDL={instructions.get('ldl', 0)} STL={instructions.get('stl', 0)}"
        )
    if instructions.get("hmma", 0) == 0:
        reasons.append("shuffle SASS has no HMMA instruction")
    if instructions.get("shfl", 0) == 0:
        reasons.append("shuffle SASS has no SHFL instruction")
    if global_loads.get("predicated_instructions", 0) == 0:
        reasons.append("shuffle SASS has no predicated global B-load instruction")
    return not reasons, reasons


def _ncu_payload(args: argparse.Namespace, wall_time_passed: bool) -> dict[str, Any]:
    metrics = [
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
    ]
    command = [
        "sudo",
        "-E",
        "env",
        "CUDA_VISIBLE_DEVICES=2",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "ncu",
        "--target-processes",
        "all",
        "--metrics",
        ",".join(metrics),
        "--kernel-name",
        "regex:.*qk_b_load_(native|shuffle)_kernel.*",
        "--launch-count",
        "2",
        "--force-overwrite",
        "--csv",
        "-o",
        "/tmp/sm70_qk_b_load_micro_ncu",
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile-only",
        "--warmup",
        "0",
        "--panels",
        str(args.panels),
        "--token-stride",
        str(args.token_stride),
    ]
    if not wall_time_passed:
        return {
            "requested": args.ncu,
            "status": "skipped_before_wall_time_gate",
            "metrics": metrics,
            "command_after_wall_time_pass": command,
        }
    return {
        "requested": args.ncu,
        "status": "manual_command_required",
        "metrics": metrics,
        "command": command,
        "note": "NCU is intentionally not run by the benchmark process; "
        "the host must grant GPU performance-counter permission.",
    }


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    _require_environment(args, device)
    mapping_evidence = _mapping_evidence()
    clock_before_mhz = _verify_clock(args)
    extension = _load_extension()

    torch.manual_seed(args.seed)
    query = torch.empty(
        (args.panels, K_BLOCK_M, K_HEAD_DIM),
        device=device,
        dtype=torch.float16,
    ).uniform_(-1.0, 1.0)
    key = torch.empty(
        (args.panels, K_BLOCK_N, args.token_stride),
        device=device,
        dtype=torch.float16,
    ).uniform_(-1.0, 1.0)
    outputs = {
        path: torch.empty(
            (args.panels, K_BLOCK_M, K_BLOCK_N),
            device=device,
            dtype=torch.float32,
        )
        for path in PATHS
    }
    launchers: dict[str, Callable[[], None]] = {
        "native": lambda: extension.native(query, key, outputs["native"]),
        "direct": lambda: extension.direct(query, key, outputs["direct"]),
        "shuffle": lambda: extension.shuffle(query, key, outputs["shuffle"]),
    }

    for _ in range(args.warmup):
        for path in PATHS:
            launchers[path]()
    torch.cuda.synchronize(device)

    for path in PATHS:
        launchers[path]()
    torch.cuda.synchronize(device)
    if args.profile_only:
        return 0

    exactness = {
        "word_dtype": "uint32",
        "word_count": outputs["native"].numel(),
        "native_direct_equal": _uint32_equal(outputs["native"], outputs["direct"]),
        "native_shuffle_equal": _uint32_equal(outputs["native"], outputs["shuffle"]),
        "direct_shuffle_equal": _uint32_equal(outputs["direct"], outputs["shuffle"]),
        "native_direct_mismatch_words": _uint32_mismatch_words(
            outputs["native"], outputs["direct"]
        ),
        "native_shuffle_mismatch_words": _uint32_mismatch_words(
            outputs["native"], outputs["shuffle"]
        ),
        "direct_shuffle_mismatch_words": _uint32_mismatch_words(
            outputs["direct"], outputs["shuffle"]
        ),
    }
    if not all(
        exactness[key]
        for key in (
            "native_direct_equal",
            "native_shuffle_equal",
            "direct_shuffle_equal",
        )
    ):
        raise RuntimeError("At least one QK route changed C-store uint32 words.")

    samples = {path: [] for path in PATHS}
    orders: list[list[str]] = []
    for round_index in range(args.rounds):
        native_direct = (
            ["native", "direct"] if round_index % 2 == 0 else ["direct", "native"]
        )
        order = native_direct.copy()
        order.insert(round_index % 3, "shuffle")
        orders.append(order)
        for path in order:
            samples[path].append(_time_batch(launchers[path], args.launches_per_sample))
    clock_after_mhz = _verify_clock(args)

    timings = {path: _timing_summary(samples[path]) for path in PATHS}
    native_median = timings["native"]["median_us"]
    shuffle_median = timings["shuffle"]["median_us"]
    shuffle_speedup_pct = 100.0 * (native_median - shuffle_median) / native_median
    direct_speedup_pct = (
        100.0 * (native_median - timings["direct"]["median_us"]) / native_median
    )
    paired_wins = {
        "native_vs_direct": sum(
            left < right for left, right in zip(samples["native"], samples["direct"])
        ),
        "direct_vs_native": sum(
            right < left for left, right in zip(samples["native"], samples["direct"])
        ),
        "native_vs_shuffle": sum(
            left < right for left, right in zip(samples["native"], samples["shuffle"])
        ),
        "shuffle_vs_native": sum(
            left < right for left, right in zip(samples["shuffle"], samples["native"])
        ),
        "shuffle_vs_direct": sum(
            left < right for left, right in zip(samples["shuffle"], samples["direct"])
        ),
    }
    winner = min(PATHS, key=lambda path: timings[path]["median_us"])
    binary = _inspect_binary(extension)
    resource_passed, resource_reasons = _resource_gate(binary)
    wall_time_passed = shuffle_speedup_pct >= 1.0
    correctness_passed = all(
        exactness[key]
        for key in (
            "native_direct_equal",
            "native_shuffle_equal",
            "direct_shuffle_equal",
        )
    )
    integration_allowed = correctness_passed and wall_time_passed and resource_passed
    rejection_reasons: list[str] = []
    if not wall_time_passed:
        rejection_reasons.append(
            f"shuffle speedup {shuffle_speedup_pct:.3f}% is below the 1.000% gate"
        )
    rejection_reasons.extend(resource_reasons)

    payload = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": args.physical_gpu,
        "clock_mhz": {
            "required": args.clock_mhz,
            "before": clock_before_mhz,
            "after": clock_after_mhz,
        },
        "mapping_evidence": mapping_evidence,
        "shape": {
            "BM": K_BLOCK_M,
            "BN": K_BLOCK_N,
            "K": K_HEAD_DIM,
            "panels": args.panels,
            "token_stride": args.token_stride,
            "key_layout": "[panel, token, token_major_K_stride]",
        },
        "paths": {
            "native": "wmma::load_matrix_sync(matrix_b, col_major)",
            "direct": "two ld.global.v4.u32 per lane into the B fragment",
            "shuffle": "lanes with (lane & 4) == 0 load; all lanes use "
            "__shfl_sync(0xffffffff, word, lane & ~4)",
        },
        "warmup": args.warmup,
        "rounds": args.rounds,
        "launches_per_sample": args.launches_per_sample,
        "seed": args.seed,
        "interleaving": {
            "native_direct_order": "alternates every round",
            "shuffle_position": "rotates through first, middle, and last",
            "first_six_orders": orders[:6],
        },
        "exactness": exactness,
        "timing": timings,
        "speedup_vs_native_pct": {
            "direct": direct_speedup_pct,
            "shuffle": shuffle_speedup_pct,
        },
        "paired_wins": paired_wins,
        "winner_by_median": winner,
        "sass_resource_check": binary,
        "gates": {
            "correctness_passed": correctness_passed,
            "shuffle_wall_time_passed": wall_time_passed,
            "shuffle_resource_passed": resource_passed,
            "integration_allowed": integration_allowed,
            "decision": (
                "eligible_for_ncu_before_any_separate_integration"
                if integration_allowed
                else "rejected_do_not_integrate"
            ),
            "rejection_reasons": rejection_reasons,
        },
        "ncu": _ncu_payload(args, wall_time_passed),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
