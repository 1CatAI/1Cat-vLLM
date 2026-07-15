# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate gate for the SM70 BM32 all-P shared-scratch microbenchmark."""

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
K_ALLP_SHARED_BYTES = 41744
K_REQUIRED_SPEEDUP_PCT = 1.0
K_REQUIRED_PAIR_WIN_RATE_PCT = 95
K_CASES = (
    ("random", 1),
    ("random", 2),
    ("random", 4),
    ("alternating", 1),
    ("alternating", 2),
    ("alternating", 4),
)
K_BASELINE_SYMBOL = "sm70_native_bm32_allp_scratch_baseline"
K_CANDIDATE_SYMBOL = "sm70_native_bm32_allp_scratch_candidate"
K_NCU_METRICS = (
    "gpu__time_duration.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
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
    parser.add_argument("--json-out", type=Path)
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
    if args.groups < K_DEFAULT_GROUPS:
        raise ValueError(f"groups must be at least {K_DEFAULT_GROUPS}.")
    if args.warmup < K_DEFAULT_WARMUP:
        raise ValueError(f"warmup must be at least {K_DEFAULT_WARMUP}.")
    if args.rounds < K_DEFAULT_ROUNDS:
        raise ValueError(f"rounds must be at least {K_DEFAULT_ROUNDS}.")
    if args.launches_per_sample < K_DEFAULT_LAUNCHES:
        raise ValueError(
            "launches-per-sample must be at least "
            f"{K_DEFAULT_LAUNCHES}."
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
        raise RuntimeError("nvcc is required to build the SM70 phase microbenchmark.")
    source = (
        Path(__file__).resolve().parent
        / "csrc"
        / "sm70_native_bm32_allp_scratch_micro.cu"
    )
    if not source.is_file():
        raise RuntimeError(f"Microbenchmark source does not exist: {source}")
    build_dir = (
        Path(tempfile.gettempdir())
        / f"vllm-sm70-native-bm32-allp-scratch-micro-{os.getuid()}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_native_bm32_allp_scratch_micro_sm70"
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
    for path, symbol, ctas_per_group in (
        ("baseline", K_BASELINE_SYMBOL, 1),
        ("candidate", K_CANDIDATE_SYMBOL, 1),
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
            "ctas_per_group": ctas_per_group,
            "instructions": instructions,
            "instructions_per_group": {
                name: count * ctas_per_group
                for name, count in instructions.items()
            },
        }
    baseline_instructions = kernels.get("baseline", {}).get("instructions", {})
    candidate_instructions = kernels.get("candidate", {}).get("instructions", {})
    return {
        "available": True,
        "binary": str(binary),
        "kernels": kernels,
        "instruction_deltas": {
            name: candidate_instructions.get(name, 0)
            - baseline_instructions.get(name, 0)
            for name in ("hmma", "ldg", "lds", "sts", "ldl", "stl", "bar")
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
    resources = payload.get("resources", {})
    paths = payload.get("paths", {})
    execution = payload.get("execution", {})
    runtime_gate = payload.get("resource_gate", {})

    if execution.get("resource_gate_pass") is not True:
        reasons.append("C++ all-P resource gate did not pass")
    if runtime_gate.get("runtime_pass") is not True:
        reasons.append("C++ runtime resource gate did not pass")

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
        if resource.get("active_ctas_per_sm", 0) < 2:
            reasons.append(
                f"{path} active_ctas_per_sm="
                f"{resource.get('active_ctas_per_sm')}, requires >=2"
            )
        description = str(paths.get(path, ""))
        if "1 CTA/group" not in description or "BM32" not in description:
            reasons.append(f"{path} is not declared as the BM32 all-P path")
    if any("maxrregcount" in argument for argument in build_command):
        reasons.append("build command must not use maxrregcount")

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
            for mnemonic in ("hmma", "ldg", "lds"):
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

        deltas = sass.get("instruction_deltas", {})
        for mnemonic in ("hmma", "ldg", "lds", "sts", "ldl", "stl"):
            if deltas.get(mnemonic) != 0:
                reasons.append(
                    f"SASS {mnemonic.upper()} delta={deltas.get(mnemonic)}, "
                    "requires 0 for a layout-only A/B"
                )
        if deltas.get("bar") != 1:
            reasons.append(
                f"SASS BAR delta={deltas.get('bar')}, requires exactly 1 "
                "pair-handoff barrier"
            )

    scratch = payload.get("scratch", {})
    expected_scratch = {
        "shared_v2_spills_per_fragment": 4,
        "shared_v2_reloads_per_fragment": 4,
        "shared_v2_spills_per_cta": 64,
        "shared_v2_reloads_per_cta": 64,
    }
    for key, expected in expected_scratch.items():
        if scratch.get(key) != expected:
            reasons.append(
                f"scratch {key}={scratch.get(key)}, requires {expected}"
            )
    return not reasons, reasons


def _correctness_gate(
    payload: dict[str, Any], groups: int
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    exactness = payload.get("exactness", {})
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
    pairs = payload.get("pairs", {})
    if pairs.get("count") != rounds:
        reasons.append(f"pair count={pairs.get('count')}, requires {rounds}")
    required_wins = (rounds * K_REQUIRED_PAIR_WIN_RATE_PCT + 99) // 100
    if pairs.get("candidate_faster", 0) < required_wins:
        reasons.append(
            f"candidate_faster={pairs.get('candidate_faster')}, "
            f"requires >= {required_wins}/{rounds}"
        )
    speedup_pct = payload.get("timing", {}).get(
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
            "performance_passed": False,
            "reasons": [f"C++ JSON parse failure: {error}"],
        }
        result["stdout_tail"] = completed.stdout[-3000:]
        result["stderr_tail"] = completed.stderr[-3000:]
        return result

    correctness_passed, correctness_reasons = _correctness_gate(payload, args.groups)
    resource_passed, resource_reasons = _resource_gate(
        payload, sass, ptxas_baseline, ptxas_candidate, build_command
    )
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
        "performance_passed": performance_passed,
        "full_case_passed": (
            completed.returncode == 0
            and correctness_passed
            and resource_passed
            and performance_passed
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
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "metrics": metrics,
        "derived": {
            "no_eligible_warps_per_cycle": (
                max(active - eligible, 0.0)
                if active is not None and eligible is not None
                else None
            ),
            "long_scoreboard_per_warp_active": metrics.get(
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
            ),
            "barrier_stall_pct": metrics.get(
                "smsp__warp_issue_stalled_barrier_per_warp_active.pct"
            ),
            "shared_bank_conflicts": metrics.get(
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
            ),
            "duration_ns": metrics.get("gpu__time_duration.sum"),
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
    if not aggregate_gate_passed:
        return {
            "status": "skipped_before_correctness_resource_gate",
            "metrics_requested_after_gate": list(K_NCU_METRICS),
        }
    if args.skip_ncu:
        return {
            "status": "skipped_by_user_after_correctness_resource_gate",
            "metrics_requested": list(K_NCU_METRICS),
        }
    return {
        "status": "attempted_after_correctness_resource_gate",
        "case": {"pattern": "random", "nblocks": 4},
        "metrics_requested": list(K_NCU_METRICS),
        "baseline": _run_ncu_kernel(binary, args, "baseline"),
        "candidate": _run_ncu_kernel(binary, args, "candidate"),
    }


def _aggregate_gates(cases: list[dict[str, Any]]) -> dict[str, Any]:
    correctness_all_cases = all(
        case["gates"]["correctness_passed"] for case in cases
    )
    resource_all_cases = all(case["gates"]["resource_passed"] for case in cases)
    all_case_performance_passed = all(
        case["gates"]["performance_passed"] for case in cases
    )
    aggregate_passed = (
        correctness_all_cases
        and resource_all_cases
        and all_case_performance_passed
    )
    reasons: list[str] = []
    if not correctness_all_cases:
        reasons.append("at least one case failed full-output bitwise correctness")
    if not resource_all_cases:
        reasons.append("at least one case failed the all-P resource gate")
    if not all_case_performance_passed:
        reasons.append("at least one case failed the performance gate")
    return {
        "correctness_all_cases_passed": correctness_all_cases,
        "resource_all_cases_passed": resource_all_cases,
        "all_cases_performance_passed": all_case_performance_passed,
        "aggregate_passed": aggregate_passed,
        "decision": (
            "all_cases_correctness_resource_and_performance_gate_passed"
            if aggregate_passed
            else "rejected_do_not_integrate"
        ),
        "rejection_reasons": reasons,
        "performance_gate": {
            "required_median_speedup_pct": K_REQUIRED_SPEEDUP_PCT,
            "required_pair_win_rate_pct": K_REQUIRED_PAIR_WIN_RATE_PCT,
        },
    }


def _write_payload(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    _require_environment(args)
    clock_before_mhz = _verify_clock(args)
    binary, build_command, ptxas_log = _build_binary(args.verbose)
    ptxas_candidate = _ptxas_function_properties(ptxas_log, K_CANDIDATE_SYMBOL)
    ptxas_baseline = _ptxas_function_properties(ptxas_log, K_BASELINE_SYMBOL)
    sass = _inspect_sass(binary)

    cases = [
        _run_case(
            binary,
            args,
            pattern,
            nblocks,
            sass,
            ptxas_baseline,
            ptxas_candidate,
            build_command,
        )
        for pattern, nblocks in K_CASES
    ]
    aggregate = _aggregate_gates(cases)
    clock_after_timing_mhz = _verify_clock(args)
    ncu = _ncu_payload(binary, args, aggregate["aggregate_passed"])
    clock_after_ncu_mhz = _verify_clock(args)

    payload: dict[str, Any] = {
        "benchmark": "sm70_native_bm32_allp_scratch_micro",
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
        },
        "build": {
            "command": build_command,
            "target": "sm_70",
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
    _write_payload(payload, args.json_out)
    return 0 if aggregate["aggregate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
