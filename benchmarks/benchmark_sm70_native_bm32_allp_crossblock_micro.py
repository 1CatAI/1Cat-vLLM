# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict gate for the SM70 BM32 all-P direct-fragment cross-block micro."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

K_REQUIRED_VISIBLE_DEVICES = "2"
K_REQUIRED_PHYSICAL_GPU = 2
K_REQUIRED_CLOCK_MHZ = 1200
K_DEFAULT_GROUPS = 144
K_DEFAULT_WARMUP = 20
K_DEFAULT_ROUNDS = 100
K_DEFAULT_LAUNCHES = 8
K_M = 32
K_D = 256
K_BLOCK_N = 128
K_ALLP_SHARED_BYTES = 41744
K_REQUIRED_SPEEDUP_PCT = 2.0
K_REQUIRED_PAIR_WIN_RATE_PCT = 95
K_CASES = (
    ("random", 2),
    ("random", 4),
    ("alternating", 2),
    ("alternating", 4),
)
K_BASELINE_SYMBOL = "sm70_native_bm32_allp_crossblock_baseline"
K_CANDIDATE_SYMBOL = "sm70_native_bm32_allp_crossblock_candidate"
K_NCU_METRICS = (
    "gpu__time_duration.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_barrier_per_warp_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active",
    "smsp__inst_executed_pipe_tensor.sum",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--groups", type=int, default=K_DEFAULT_GROUPS)
    parser.add_argument("--warmup", type=int, default=K_DEFAULT_WARMUP)
    parser.add_argument("--rounds", type=int, default=K_DEFAULT_ROUNDS)
    parser.add_argument(
        "--launches",
        "--launches-per-sample",
        dest="launches_per_sample",
        type=int,
        default=K_DEFAULT_LAUNCHES,
    )
    parser.add_argument("--physical-gpu", type=int, default=K_REQUIRED_PHYSICAL_GPU)
    parser.add_argument("--clock-mhz", type=int, default=K_REQUIRED_CLOCK_MHZ)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Permit a short resource and exactness preflight before formal timing.",
    )
    parser.add_argument("--skip-ncu", action="store_true")
    parser.add_argument("--ncu-timeout-seconds", type=int, default=180)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _require_environment(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != K_REQUIRED_VISIBLE_DEVICES:
        raise RuntimeError(
            "This benchmark requires CUDA_VISIBLE_DEVICES=2 so cuda:0 is "
            "physical GPU 2."
        )
    if args.device != 0:
        raise RuntimeError("Use logical device 0 after CUDA_VISIBLE_DEVICES=2.")
    if args.physical_gpu != K_REQUIRED_PHYSICAL_GPU:
        raise RuntimeError("This benchmark is fixed to physical GPU 2.")
    if args.clock_mhz != K_REQUIRED_CLOCK_MHZ:
        raise RuntimeError("This benchmark is fixed to a 1200 MHz graphics clock.")
    if args.groups != K_DEFAULT_GROUPS:
        raise ValueError(f"groups must be exactly {K_DEFAULT_GROUPS}.")
    if args.smoke:
        if args.warmup < 0 or args.rounds < 1 or args.launches_per_sample < 1:
            raise ValueError("smoke warmup, rounds, and launches must be valid.")
        return
    if args.warmup < K_DEFAULT_WARMUP:
        raise ValueError(f"warmup must be at least {K_DEFAULT_WARMUP}.")
    if args.rounds != K_DEFAULT_ROUNDS:
        raise ValueError(f"formal runs require exactly {K_DEFAULT_ROUNDS} rounds.")
    if args.launches_per_sample != K_DEFAULT_LAUNCHES:
        raise ValueError(
            "formal runs require exactly "
            f"{K_DEFAULT_LAUNCHES} launches per sample."
        )


def _graphics_clock_mhz(physical_gpu: int) -> int:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required to verify the locked clock.")
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=clocks.current.graphics",
            "--format=csv,noheader,nounits",
            "--id",
            str(physical_gpu),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
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


def _build_binary(verbose: bool) -> tuple[Path, list[str], str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required to build the SM70 cross-block micro.")
    source = (
        Path(__file__).resolve().parent
        / "csrc"
        / "sm70_native_bm32_allp_crossblock_micro.cu"
    )
    if not source.is_file():
        raise RuntimeError(f"Microbenchmark source does not exist: {source}")
    build_dir = (
        Path(tempfile.gettempdir())
        / f"vllm-sm70-native-bm32-allp-crossblock-micro-{os.getuid()}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_native_bm32_allp_crossblock_micro_sm70"
    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "--generate-code=arch=compute_70,code=sm_70",
        "--ptxas-options=-v",
        "-o",
        str(binary),
        str(source),
    ]
    if verbose:
        print("build:", " ".join(command), file=sys.stderr)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return binary, command, result.stderr


def _ptxas_function_properties(ptxas_log: str, symbol: str) -> dict[str, int] | None:
    header = re.compile(
        rf"Compiling entry function '{re.escape(symbol)}'.*?(?="
        r"Compiling entry function '|\Z)",
        re.DOTALL,
    )
    match = header.search(ptxas_log)
    if match is None:
        return None
    section = match.group(0)
    stack = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
        r"(\d+) bytes spill loads",
        section,
    )
    registers = re.search(r"Used (\d+) registers", section)
    shared = re.search(r"(\d+) bytes smem", section)
    if stack is None or registers is None or shared is None:
        return None
    return {
        "stack_frame_bytes": int(stack.group(1)),
        "spill_store_bytes": int(stack.group(2)),
        "spill_load_bytes": int(stack.group(3)),
        "registers_per_thread": int(registers.group(1)),
        "static_shared_bytes": int(shared.group(1)),
    }


def _function_section(text: str, symbol: str) -> str | None:
    pattern = re.compile(r"(?m)^\s*Function\s*:\s*(.+?)\s*$")
    headers = list(pattern.finditer(text))
    for index, header in enumerate(headers):
        if symbol not in header.group(1):
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        return text[header.start() : end]
    return None


def _instruction_count(sass: str, mnemonic: str) -> int:
    return len(re.findall(rf"\b{re.escape(mnemonic)}(?:\.|\b)", sass))


def _inspect_sass(binary: Path) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return {"available": False, "reason": "cuobjdump was not found"}
    result = subprocess.run(
        [cuobjdump, "--dump-sass", str(binary)], text=True, capture_output=True
    )
    if result.returncode:
        return {
            "available": False,
            "returncode": result.returncode,
            "stderr": result.stderr[-2000:],
        }
    kernels: dict[str, Any] = {}
    for path, symbol in (
        ("baseline", K_BASELINE_SYMBOL),
        ("candidate", K_CANDIDATE_SYMBOL),
    ):
        section = _function_section(result.stdout, symbol)
        if section is None:
            kernels[path] = {"available": False, "symbol": symbol}
            continue
        instructions = {
            "hmma": _instruction_count(section, "HMMA"),
            "ldg": _instruction_count(section, "LDG"),
            "lds": _instruction_count(section, "LDS"),
            "sts": _instruction_count(section, "STS"),
            "ldl": _instruction_count(section, "LDL"),
            "stl": _instruction_count(section, "STL"),
            "bar": _instruction_count(section, "BAR"),
        }
        kernels[path] = {
            "available": True,
            "symbol": symbol,
            "instructions": instructions,
        }
    return {
        "available": True,
        "binary": str(binary),
        "kernels": kernels,
        "static_instruction_delta": {
            mnemonic: (
                kernels.get("candidate", {})
                .get("instructions", {})
                .get(mnemonic, 0)
                - kernels.get("baseline", {})
                .get("instructions", {})
                .get(mnemonic, 0)
            )
            for mnemonic in ("hmma", "ldg", "lds", "sts", "ldl", "stl", "bar")
        },
    }


def _resource_gate(
    payload: dict[str, Any],
    sass: dict[str, Any],
    ptxas_baseline: dict[str, int] | None,
    ptxas_candidate: dict[str, int] | None,
    build_command: list[str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    resources = payload.get("resources") or {}
    paths = payload.get("paths") or {}
    execution = payload.get("execution") or {}
    runtime_gate = payload.get("resource_gate") or {}

    if execution.get("resource_gate_pass") is not True:
        reasons.append("C++ runtime resource gate did not pass")
    if runtime_gate.get("runtime_pass") is not True:
        reasons.append("C++ resource gate did not pass")
    if any("maxrregcount" in argument for argument in build_command):
        reasons.append("build command must not use maxrregcount")

    for path in ("baseline", "candidate"):
        resource = resources.get(path, {})
        if resource.get("threads_per_cta") != 512:
            reasons.append(
                f"{path} threads_per_cta={resource.get('threads_per_cta')}, "
                "requires 512"
            )
        if resource.get("registers_per_thread", 65) > 64:
            reasons.append(
                f"{path} REG={resource.get('registers_per_thread')}, "
                "requires <=64"
            )
        if resource.get("local_bytes_per_thread") != 0:
            reasons.append(
                f"{path} LOCAL={resource.get('local_bytes_per_thread')}, "
                "requires 0"
            )
        if resource.get("static_shared_bytes") != K_ALLP_SHARED_BYTES:
            reasons.append(
                f"{path} shared={resource.get('static_shared_bytes')}, "
                f"requires {K_ALLP_SHARED_BYTES}"
            )
        if resource.get("active_ctas_per_sm") != 2:
            reasons.append(
                f"{path} active_ctas_per_sm="
                f"{resource.get('active_ctas_per_sm')}, requires exactly 2"
            )
        description = str(paths.get(path, ""))
        if "1 CTA/group" not in description or "BM32" not in description:
            reasons.append(f"{path} is not declared as the BM32 one-CTA path")

    for path, ptxas in (
        ("baseline", ptxas_baseline),
        ("candidate", ptxas_candidate),
    ):
        if ptxas is None:
            reasons.append(f"{path} PTXAS properties were not found")
            continue
        if ptxas["registers_per_thread"] > 64:
            reasons.append(
                f"PTXAS {path} REG={ptxas['registers_per_thread']}, "
                "requires <=64"
            )
        for key in ("stack_frame_bytes", "spill_store_bytes", "spill_load_bytes"):
            if ptxas[key] != 0:
                reasons.append(f"PTXAS {path} {key}={ptxas[key]}, requires 0")
        if ptxas["static_shared_bytes"] != K_ALLP_SHARED_BYTES:
            reasons.append(
                f"PTXAS {path} shared={ptxas['static_shared_bytes']}, "
                f"requires {K_ALLP_SHARED_BYTES}"
            )

    if not sass.get("available"):
        reasons.append("SASS inspection unavailable")
    else:
        for path in ("baseline", "candidate"):
            path_sass = sass.get("kernels", {}).get(path, {})
            if not path_sass.get("available"):
                reasons.append(f"{path} SASS function was not found")
                continue
            instructions = path_sass.get("instructions", {})
            for mnemonic in ("hmma", "ldg", "lds", "sts"):
                if instructions.get(mnemonic, 0) == 0:
                    reasons.append(
                        f"{path} SASS has no {mnemonic.upper()} instruction"
                    )
            if instructions.get("ldl", 0) or instructions.get("stl", 0):
                reasons.append(
                    f"{path} SASS has local instructions "
                    f"LDL={instructions.get('ldl', 0)} "
                    f"STL={instructions.get('stl', 0)}"
                )

    scratch = payload.get("scratch") or {}
    expected_scratch = {
        "shared_v2_spills_per_fragment": 4,
        "shared_v2_reloads_per_fragment": 4,
        "shared_v2_spills_per_cta": 64,
        "shared_v2_reloads_per_cta": 64,
    }
    for key, expected in expected_scratch.items():
        if scratch.get(key) != expected:
            reasons.append(f"scratch {key}={scratch.get(key)}, requires {expected}")
    if scratch.get("handoff") != "one 64-thread named barrier per QK warp pair":
        reasons.append("pair-slab named-barrier contract changed")

    hmma = payload.get("hmma_contract") or {}
    nblocks = (payload.get("shape") or {}).get("nblocks")
    expected_qk = 8 * (K_D // 16) * 2
    expected_pv = 16 * 4 * 2
    expected_total = (expected_qk + expected_pv) * nblocks if nblocks else None
    if hmma.get("qk_per_block") != expected_qk:
        reasons.append("QK HMMA semantic count changed")
    if hmma.get("pv_per_block") != expected_pv:
        reasons.append("PV HMMA semantic count changed")
    if hmma.get("baseline_total_per_group") != expected_total:
        reasons.append("baseline HMMA total per group changed")
    if hmma.get("candidate_total_per_group") != expected_total:
        reasons.append("candidate HMMA total per group changed")
    if hmma.get("per_group_equal") is not True:
        reasons.append("baseline/candidate HMMA total per group differs")
    return not reasons, reasons


def _correctness_gate(
    payload: dict[str, Any], groups: int
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    exactness = payload.get("exactness") or {}
    expected_words = groups * K_M * K_D // 2
    if exactness.get("word_dtype") != "uint32 packed fp16":
        reasons.append(
            "word_dtype="
            f"{exactness.get('word_dtype')}, requires uint32 packed fp16"
        )
    if exactness.get("word_count") != expected_words:
        reasons.append(
            "word_count="
            f"{exactness.get('word_count')}, requires {expected_words}"
        )
    if exactness.get("full_output") is not True:
        reasons.append("C++ probe did not compare the full output")
    if exactness.get("bitwise_equal") is not True:
        reasons.append("candidate output is not bitwise equal to baseline")
    if exactness.get("mismatch_words") != 0:
        reasons.append(
            "mismatch_words="
            f"{exactness.get('mismatch_words')}, requires 0"
        )
    xor = exactness.get("xor", {})
    if xor.get("max_word") != 0 or xor.get("reduction") != 0:
        reasons.append("packed uint32 XOR result is nonzero")
    return not reasons, reasons


def _performance_gate(
    payload: dict[str, Any], rounds: int
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    pairs = payload.get("pairs") or {}
    if pairs.get("count") != rounds:
        reasons.append(f"pair count={pairs.get('count')}, requires {rounds}")
    required_wins = (rounds * K_REQUIRED_PAIR_WIN_RATE_PCT + 99) // 100
    if pairs.get("candidate_faster", 0) < required_wins:
        reasons.append(
            f"candidate_faster={pairs.get('candidate_faster')}, "
            f"requires >= {required_wins}/{rounds}"
        )
    speedup_pct = (payload.get("timing") or {}).get(
        "candidate_speedup_vs_baseline_pct"
    )
    if speedup_pct is None or speedup_pct < K_REQUIRED_SPEEDUP_PCT:
        reasons.append(
            f"median speedup={speedup_pct}, requires "
            f">= {K_REQUIRED_SPEEDUP_PCT}%"
        )
    return not reasons, reasons


def _binary_command(
    binary: Path, args: argparse.Namespace, pattern: str, nblocks: int
) -> list[str]:
    return [
        str(binary),
        "--device",
        str(args.device),
        "--groups",
        str(args.groups),
        "--nblocks",
        str(nblocks),
        "--warmup",
        str(args.warmup),
        "--rounds",
        str(args.rounds),
        "--launches-per-sample",
        str(args.launches_per_sample),
        "--pattern",
        pattern,
    ]


def _run_case(
    binary: Path,
    args: argparse.Namespace,
    pattern: str,
    nblocks: int,
    sass: dict[str, Any],
    ptxas_baseline: dict[str, int] | None,
    ptxas_candidate: dict[str, int] | None,
    build_command: list[str],
) -> dict[str, Any]:
    command = _binary_command(binary, args, pattern, nblocks)
    if args.verbose:
        print("run:", " ".join(command), file=sys.stderr)
    completed = subprocess.run(command, text=True, capture_output=True)
    result: dict[str, Any] = {
        "name": f"{pattern}_nblocks{nblocks}",
        "pattern": pattern,
        "nblocks": nblocks,
        "command": command,
        "returncode": completed.returncode,
    }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        result["cpp_json"] = None
        result["gates"] = {
            "correctness_passed": False,
            "resource_passed": False,
            "performance_evaluated": not args.smoke,
            "performance_passed": False if not args.smoke else None,
            "reasons": [f"C++ JSON parse failure: {error}"],
        }
        result["stdout_tail"] = completed.stdout[-3000:]
        result["stderr_tail"] = completed.stderr[-3000:]
        return result

    correctness_passed, correctness_reasons = _correctness_gate(payload, args.groups)
    resource_passed, resource_reasons = _resource_gate(
        payload, sass, ptxas_baseline, ptxas_candidate, build_command
    )
    if args.smoke:
        performance_passed: bool | None = None
        performance_reasons: list[str] = []
    else:
        performance_passed, performance_reasons = _performance_gate(
            payload, args.rounds
        )
    reasons = [
        *([] if completed.returncode == 0 else ["C++ benchmark returned nonzero"]),
        *correctness_reasons,
        *resource_reasons,
        *performance_reasons,
    ]
    result["cpp_json"] = payload
    result["gates"] = {
        "correctness_passed": correctness_passed,
        "resource_passed": resource_passed,
        "performance_evaluated": not args.smoke,
        "performance_passed": performance_passed,
        "preflight_passed": (
            completed.returncode == 0 and correctness_passed and resource_passed
        ),
        "full_case_passed": (
            completed.returncode == 0
            and correctness_passed
            and resource_passed
            and (performance_passed is True if not args.smoke else True)
        ),
        "reasons": reasons,
    }
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-3000:]
    return result


def _number(value: str) -> float | None:
    try:
        return float(value.replace(",", "").strip())
    except ValueError:
        return None


def _parse_ncu_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in csv.reader(io.StringIO(text)):
        for metric in K_NCU_METRICS:
            if metric not in row:
                continue
            for cell in reversed(row):
                parsed = _number(cell)
                if parsed is not None:
                    values[metric] = parsed
                    break
    return values


def _run_ncu_kernel(
    binary: Path, args: argparse.Namespace, profile_kernel: str
) -> dict[str, Any]:
    ncu = shutil.which("ncu")
    if ncu is None:
        return {"status": "unavailable", "reason": "ncu was not found"}
    symbol = K_BASELINE_SYMBOL if profile_kernel == "baseline" else K_CANDIDATE_SYMBOL
    command = [
        ncu,
        "--target-processes",
        "all",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(K_NCU_METRICS),
        "--kernel-name",
        f"regex:.*{symbol}.*",
        "--launch-count",
        "1",
        str(binary),
        "--device",
        "0",
        "--groups",
        str(args.groups),
        "--nblocks",
        "4",
        "--warmup",
        "0",
        "--rounds",
        "1",
        "--launches-per-sample",
        "1",
        "--pattern",
        "random",
        "--profile-only",
        "--profile-kernel",
        profile_kernel,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = K_REQUIRED_VISIBLE_DEVICES
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=args.ncu_timeout_seconds,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "command": command,
            "timeout_seconds": args.ncu_timeout_seconds,
            "stdout_tail": (error.stdout or "")[-3000:],
            "stderr_tail": (error.stderr or "")[-3000:],
        }
    metrics = _parse_ncu_metrics(completed.stdout)
    active = metrics.get("smsp__warps_active.avg.per_cycle_active")
    eligible = metrics.get("smsp__warps_eligible.avg.per_cycle_active")
    shared_load = metrics.get(
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
    )
    shared_store = metrics.get(
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
    )
    bank_conflicts = (
        shared_load + shared_store
        if shared_load is not None and shared_store is not None
        else None
    )
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "metrics": metrics,
        "derived": {
            "duration_ns": metrics.get("gpu__time_duration.sum"),
            "shared_bank_conflicts_total": bank_conflicts,
            "no_eligible_warps_per_cycle": (
                max(active - eligible, 0.0)
                if active is not None and eligible is not None
                else None
            ),
            "barrier_stall_per_warp_active": metrics.get(
                "smsp__warp_issue_stalled_barrier_per_warp_active"
            ),
            "long_scoreboard_per_warp_active": metrics.get(
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active"
            ),
            "short_scoreboard_per_warp_active": metrics.get(
                "smsp__warp_issue_stalled_short_scoreboard_per_warp_active"
            ),
            "mio_throttle_per_warp_active": metrics.get(
                "smsp__warp_issue_stalled_mio_throttle_per_warp_active"
            ),
            "tensor_instructions": metrics.get(
                "smsp__inst_executed_pipe_tensor.sum"
            ),
        },
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
        "permission_error": "ERR_NVGPUCTRPERM"
        in (completed.stdout + completed.stderr),
    }


def _ncu_payload(
    binary: Path, args: argparse.Namespace, aggregate_gate_passed: bool
) -> dict[str, Any]:
    if args.smoke:
        return {
            "status": "skipped_smoke_preflight",
            "metrics_requested_after_formal_gate": list(K_NCU_METRICS),
        }
    if not aggregate_gate_passed:
        return {
            "status": "skipped_before_formal_gate",
            "metrics_requested_after_gate": list(K_NCU_METRICS),
        }
    if args.skip_ncu:
        return {
            "status": "skipped_by_user_after_formal_gate",
            "metrics_requested": list(K_NCU_METRICS),
        }
    baseline = _run_ncu_kernel(binary, args, "baseline")
    candidate = _run_ncu_kernel(binary, args, "candidate")
    baseline_tensor = baseline.get("derived", {}).get("tensor_instructions")
    candidate_tensor = candidate.get("derived", {}).get("tensor_instructions")
    return {
        "status": "attempted_after_formal_gate",
        "case": {"pattern": "random", "nblocks": 4},
        "metrics_requested": list(K_NCU_METRICS),
        "baseline": baseline,
        "candidate": candidate,
        "tensor_instruction_match": (
            baseline_tensor == candidate_tensor
            if baseline_tensor is not None and candidate_tensor is not None
            else None
        ),
    }


def _aggregate_gates(cases: list[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    expected_case_count = len(K_CASES)
    preflight_all_cases = len(cases) == expected_case_count and all(
        case["gates"]["preflight_passed"] for case in cases
    )
    correctness_all_cases = len(cases) == expected_case_count and all(
        case["gates"]["correctness_passed"] for case in cases
    )
    resource_all_cases = len(cases) == expected_case_count and all(
        case["gates"]["resource_passed"] for case in cases
    )
    if smoke:
        reasons: list[str] = []
        if not preflight_all_cases:
            reasons.append(
                "at least one smoke case failed resource or bitwise exactness"
            )
        return {
            "mode": "smoke",
            "cases_completed": len(cases),
            "expected_cases": expected_case_count,
            "correctness_all_cases_passed": correctness_all_cases,
            "resource_all_cases_passed": resource_all_cases,
            "aggregate_passed": preflight_all_cases,
            "decision": (
                "smoke_passed_run_formal_100_round_gate"
                if preflight_all_cases
                else "stop_before_formal_timing"
            ),
            "rejection_reasons": reasons,
        }

    performance_all_cases = len(cases) == expected_case_count and all(
        case["gates"]["performance_passed"] is True for case in cases
    )
    aggregate_passed = (
        correctness_all_cases and resource_all_cases and performance_all_cases
    )
    reasons = []
    if not correctness_all_cases:
        reasons.append("at least one case failed full-output bitwise exactness")
    if not resource_all_cases:
        reasons.append("at least one case failed the resource/SASS gate")
    if not performance_all_cases:
        reasons.append("at least one case failed the 2%/95-win performance gate")
    return {
        "mode": "formal",
        "cases_completed": len(cases),
        "expected_cases": expected_case_count,
        "correctness_all_cases_passed": correctness_all_cases,
        "resource_all_cases_passed": resource_all_cases,
        "all_cases_performance_passed": performance_all_cases,
        "aggregate_passed": aggregate_passed,
        "decision": (
            "formal_gate_passed_recommend_ncu"
            if aggregate_passed
            else "rejected_do_not_profile_or_integrate"
        ),
        "rejection_reasons": reasons,
        "performance_gate": {
            "required_median_speedup_pct": K_REQUIRED_SPEEDUP_PCT,
            "required_pair_win_rate_pct": K_REQUIRED_PAIR_WIN_RATE_PCT,
            "required_wins_at_100_rounds": 95,
        },
    }


def _run_cases(
    binary: Path,
    args: argparse.Namespace,
    sass: dict[str, Any],
    ptxas_baseline: dict[str, int] | None,
    ptxas_candidate: dict[str, int] | None,
    build_command: list[str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for pattern, nblocks in K_CASES:
        case = _run_case(
            binary,
            args,
            pattern,
            nblocks,
            sass,
            ptxas_baseline,
            ptxas_candidate,
            build_command,
        )
        cases.append(case)
        if not case["gates"].get("preflight_passed"):
            break
    return cases


def _main() -> int:
    args = _parse_args()
    _require_environment(args)
    clock_before_mhz = _verify_clock(args)
    binary, build_command, ptxas_log = _build_binary(args.verbose)
    ptxas_baseline = _ptxas_function_properties(ptxas_log, K_BASELINE_SYMBOL)
    ptxas_candidate = _ptxas_function_properties(ptxas_log, K_CANDIDATE_SYMBOL)
    sass = _inspect_sass(binary)
    cases = _run_cases(
        binary,
        args,
        sass,
        ptxas_baseline,
        ptxas_candidate,
        build_command,
    )
    aggregate = _aggregate_gates(cases, args.smoke)
    clock_after_timing_mhz = _verify_clock(args)
    ncu = _ncu_payload(binary, args, aggregate["aggregate_passed"])
    clock_after_ncu_mhz = _verify_clock(args)
    payload: dict[str, Any] = {
        "benchmark": "sm70_native_bm32_allp_crossblock_micro",
        "target": "sm_70",
        "matrix": [
            {"pattern": pattern, "nblocks": nblocks}
            for pattern, nblocks in K_CASES
        ],
        "configuration": {
            "groups": args.groups,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "launches_per_sample": args.launches_per_sample,
            "smoke": args.smoke,
        },
        "build": {
            "command": build_command,
            "target": "sm_70",
            "ptxas_optimization_level": "production_default",
            "maxrregcount_present": any(
                "maxrregcount" in argument for argument in build_command
            ),
            "ptxas_log": ptxas_log[-5000:],
            "ptxas": {
                "baseline": ptxas_baseline,
                "candidate": ptxas_candidate,
            },
        },
        "runtime": {
            "python": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpu": args.physical_gpu,
            "clock_mhz": {
                "required": args.clock_mhz,
                "before": clock_before_mhz,
                "after_timing": clock_after_timing_mhz,
                "after_ncu": clock_after_ncu_mhz,
            },
        },
        "sass": sass,
        "cases": cases,
        "gates": aggregate,
        "ncu": ncu,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if aggregate["aggregate_passed"] else 1


def main() -> int:
    try:
        return _main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "benchmark": "sm70_native_bm32_allp_crossblock_micro",
                    "fatal_error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
