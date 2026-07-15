# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and measure the SM70 M64 all-P microbenchmark against 2x BM32."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

K_GROUPS = 72
K_M = 64
K_D = 256
K_BLOCK_N = 128
K_BASELINE_THREADS = 512
K_CANDIDATE_THREADS = 384
K_MAX_DYNAMIC_SHARED_BYTES = 48 * 1024
K_MAX_CANDIDATE_REGISTERS = 85
K_REQUIRED_SPEEDUP_PCT = 10.0
K_DEFAULT_WARMUP = 20
K_DEFAULT_ROUNDS = 100
K_DEFAULT_LAUNCHES = 8
K_SMOKE_CASES = (
    ("random", 1),
    ("alternating", 1),
    ("random", 2),
    ("alternating", 2),
    ("random", 4),
    ("alternating", 4),
)
K_TIMING_CASES = (
    ("random", 1),
    ("alternating", 1),
    ("random", 2),
    ("alternating", 2),
    ("random", 4),
    ("alternating", 4),
    ("random", 64),
    ("alternating", 64),
)
K_BASELINE_SYMBOL = "sm70_native_bm64_allp_bm32_pair_scratch_baseline"
K_CANDIDATE_SYMBOL = "sm70_native_bm64_allp_m64_candidate"
K_BASELINE_SHARED_BYTES = 41744


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--clock-mhz", type=int, default=1200)
    parser.add_argument("--groups", type=int, default=K_GROUPS)
    parser.add_argument("--warmup", type=int, default=K_DEFAULT_WARMUP)
    parser.add_argument("--rounds", type=int, default=K_DEFAULT_ROUNDS)
    parser.add_argument(
        "--launches",
        "--launches-per-sample",
        dest="launches_per_sample",
        type=int,
        default=K_DEFAULT_LAUNCHES,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run build/resource/bitwise cases without timing rounds.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.device != 0:
        raise ValueError("The child process exposes one GPU; use --device 0.")
    if args.physical_gpu < 0:
        raise ValueError("--physical-gpu must be non-negative.")
    if args.clock_mhz <= 0:
        raise ValueError("--clock-mhz must be positive.")
    if args.groups < 1:
        raise ValueError("--groups must be positive.")
    if args.warmup < 0 or args.rounds < 1 or args.launches_per_sample < 1:
        raise ValueError("warmup must be non-negative; rounds and launches positive.")


def _child_cuda_configuration(physical_gpu: int) -> dict[str, str]:
    # CUDA otherwise enumerates the desktop GPU before the V100s on this host.
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(physical_gpu),
    }


def _child_environment(physical_gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(_child_cuda_configuration(physical_gpu))
    return environment


def _nvidia_smi() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required to select an idle V100.")
    return executable


def _selected_gpu_state(physical_gpu: int) -> dict[str, Any]:
    executable = _nvidia_smi()
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index,uuid,name,compute_cap,memory.free,"
            "clocks.current.graphics",
            "--format=csv,noheader,nounits",
            "--id",
            str(physical_gpu),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"Could not parse selected GPU: {result.stdout!r}")
    index, uuid, name, capability, free_mib, graphics_clock_mhz = fields
    if capability != "7.0":
        raise RuntimeError(
            f"Physical GPU {physical_gpu} is {name} (SM{capability}), not SM70."
        )

    applications = subprocess.run(
        [
            executable,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    active = [
        line for line in applications.stdout.splitlines() if line.startswith(uuid + ",")
    ]
    if active:
        raise RuntimeError(
            f"Physical GPU {physical_gpu} is not idle: {'; '.join(active)}"
        )
    return {
        "index": int(index),
        "uuid": uuid,
        "name": name,
        "compute_capability": capability,
        "memory_free_mib": int(free_mib),
        "graphics_clock_mhz": int(graphics_clock_mhz),
        "compute_processes": active,
    }


def _require_idle_clock(
    physical_gpu: int, required_clock_mhz: int, phase: str
) -> dict[str, Any]:
    state = _selected_gpu_state(physical_gpu)
    actual_clock_mhz = state["graphics_clock_mhz"]
    if actual_clock_mhz != required_clock_mhz:
        raise RuntimeError(
            f"GPU {physical_gpu} graphics clock is {actual_clock_mhz} MHz "
            f"{phase}; requires exactly {required_clock_mhz} MHz."
        )
    return state


def _build_binary(verbose: bool) -> tuple[Path, list[str], str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required to build the SM70 microbenchmark.")
    source = Path(__file__).resolve().parent / "csrc" / "sm70_native_bm64_allp_micro.cu"
    if not source.is_file():
        raise RuntimeError(f"Microbenchmark source does not exist: {source}")
    build_dir = (
        Path(tempfile.gettempdir()) / f"vllm-sm70-native-bm64-allp-micro-{os.getuid()}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_native_bm64_allp_micro_sm70"
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
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return binary, command, completed.stderr


def _ptxas_function_properties(ptxas_log: str, symbol: str) -> dict[str, int] | None:
    section_match = re.search(
        rf"Compiling entry function '{re.escape(symbol)}'.*?(?="
        r"Compiling entry function '|\Z)",
        ptxas_log,
        re.DOTALL,
    )
    if section_match is None:
        return None
    section = section_match.group(0)
    stack = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
        r"(\d+) bytes spill loads",
        section,
    )
    registers = re.search(r"Used (\d+) registers", section)
    shared = re.search(r"(\d+) bytes smem", section)
    if stack is None or registers is None:
        return None
    return {
        "stack_frame_bytes": int(stack.group(1)),
        "spill_store_bytes": int(stack.group(2)),
        "spill_load_bytes": int(stack.group(3)),
        "registers_per_thread": int(registers.group(1)),
        "static_shared_bytes": int(shared.group(1)) if shared else 0,
    }


def _function_section(sass: str, symbol: str) -> str | None:
    headers = list(re.finditer(r"(?m)^\s*Function\s*:\s*(.+?)\s*$", sass))
    for index, header in enumerate(headers):
        if symbol not in header.group(1):
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(sass)
        return sass[header.start() : end]
    return None


def _instruction_count(sass: str, mnemonic: str) -> int:
    return len(re.findall(rf"\b{re.escape(mnemonic)}(?:\.|\b)", sass))


def _inspect_sass(binary: Path) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return {"available": False, "reason": "cuobjdump was not found"}
    completed = subprocess.run(
        [cuobjdump, "--dump-sass", str(binary)], text=True, capture_output=True
    )
    if completed.returncode:
        return {
            "available": False,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-3000:],
        }
    kernels: dict[str, Any] = {}
    for name, symbol, ctas_per_logical_group in (
        ("baseline", K_BASELINE_SYMBOL, 2),
        ("candidate", K_CANDIDATE_SYMBOL, 1),
    ):
        section = _function_section(completed.stdout, symbol)
        if section is None:
            kernels[name] = {"available": False, "symbol": symbol}
            continue
        instructions = {
            mnemonic.lower(): _instruction_count(section, mnemonic)
            for mnemonic in ("HMMA", "LDG", "LDS", "LDL", "STL")
        }
        kernels[name] = {
            "available": True,
            "symbol": symbol,
            "ctas_per_logical_group": ctas_per_logical_group,
            "instructions": instructions,
        }
    return {"available": True, "binary": str(binary), "kernels": kernels}


def _build_gate(
    command: list[str],
    ptxas_baseline: dict[str, int] | None,
    ptxas_candidate: dict[str, int] | None,
    sass: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if any("maxrregcount" in argument for argument in command):
        reasons.append("build command must not force maxrregcount")
    expected = (
        ("baseline", ptxas_baseline, 64, K_BASELINE_SHARED_BYTES),
        ("candidate", ptxas_candidate, K_MAX_CANDIDATE_REGISTERS, 0),
    )
    for name, properties, max_registers, expected_static_shared in expected:
        if properties is None:
            reasons.append(f"PTXAS properties missing for {name}")
            continue
        if properties["registers_per_thread"] > max_registers:
            reasons.append(
                f"PTXAS {name} REG={properties['registers_per_thread']}, "
                f"requires <= {max_registers}"
            )
        if properties["static_shared_bytes"] != expected_static_shared:
            reasons.append(
                f"PTXAS {name} static shared="
                f"{properties['static_shared_bytes']}, requires "
                f"{expected_static_shared}"
            )
        for key in ("stack_frame_bytes", "spill_store_bytes", "spill_load_bytes"):
            if properties[key] != 0:
                reasons.append(f"PTXAS {name} {key}={properties[key]}, requires 0")
    if not sass.get("available"):
        return reasons
    for name in ("baseline", "candidate"):
        instructions = sass.get("kernels", {}).get(name, {}).get("instructions", {})
        if instructions.get("ldl", 0) or instructions.get("stl", 0):
            reasons.append(
                f"{name} SASS local instructions: "
                f"LDL={instructions.get('ldl', 0)} STL={instructions.get('stl', 0)}"
            )
    return reasons


def _correctness_gate(payload: dict[str, Any], groups: int) -> list[str]:
    reasons: list[str] = []
    exactness = payload.get("exactness") or {}
    expected_words = groups * K_M * K_D // 2
    if exactness.get("word_dtype") != "uint32 packed fp16":
        reasons.append("bitwise probe did not use packed uint32 fp16 words")
    if exactness.get("word_count") != expected_words:
        reasons.append(
            f"word_count={exactness.get('word_count')}, requires {expected_words}"
        )
    if exactness.get("full_output") is not True:
        reasons.append("bitwise probe did not compare full output")
    if exactness.get("bitwise_equal") is not True:
        reasons.append(
            "bitwise mismatch: " + json.dumps(exactness.get("first_difference"))
        )
    if exactness.get("mismatch_words") != 0:
        reasons.append(f"mismatch_words={exactness.get('mismatch_words')}, requires 0")
    xor = exactness.get("xor") or {}
    if xor.get("max_word") != 0 or xor.get("reduction") != 0:
        reasons.append("packed uint32 XOR is nonzero")
    return reasons


def _runtime_resource_gate(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    execution = payload.get("execution") or {}
    resources = payload.get("resources") or {}
    baseline = resources.get("baseline") or {}
    candidate = resources.get("candidate") or {}
    paths = payload.get("paths") or {}
    storage = payload.get("candidate_storage") or {}
    if execution.get("resource_gate_pass") is not True:
        reasons.append("C++ runtime resource gate did not pass")
    if baseline.get("threads_per_cta") != K_BASELINE_THREADS:
        reasons.append("baseline does not use 512 threads")
    if baseline.get("registers_per_thread") != 64:
        reasons.append("baseline does not retain the accepted 64-register shape")
    if baseline.get("static_shared_bytes") != K_BASELINE_SHARED_BYTES:
        reasons.append("baseline does not retain the accepted 41,744B shared layout")
    if baseline.get("dynamic_shared_bytes") != 0:
        reasons.append("baseline unexpectedly uses dynamic shared memory")
    if baseline.get("local_bytes_per_thread") != 0:
        reasons.append("baseline runtime local bytes are nonzero")
    if baseline.get("active_ctas_per_sm") != 2:
        reasons.append("baseline must retain two CTAs per SM")
    if candidate.get("threads_per_cta") != K_CANDIDATE_THREADS:
        reasons.append("candidate does not use 384 threads")
    if (
        candidate.get("registers_per_thread", K_MAX_CANDIDATE_REGISTERS + 1)
        > K_MAX_CANDIDATE_REGISTERS
    ):
        reasons.append("candidate exceeds 85 registers per thread")
    if candidate.get("dynamic_shared_bytes", K_MAX_DYNAMIC_SHARED_BYTES + 1) > (
        K_MAX_DYNAMIC_SHARED_BYTES
    ):
        reasons.append("candidate dynamic shared memory exceeds 48 KiB")
    if candidate.get("local_bytes_per_thread") != 0:
        reasons.append("candidate runtime local bytes are nonzero")
    if candidate.get("active_ctas_per_sm") != 2:
        reasons.append("candidate must retain two CTAs per SM")
    if "2 CTA/M64" not in str(paths.get("baseline", "")):
        reasons.append("baseline path does not declare 2 CTA per M64 group")
    if "PAIR_SCRATCH=true" not in str(paths.get("baseline", "")):
        reasons.append("baseline path is not the accepted PAIR_SCRATCH path")
    if "1 CTA/M64" not in str(paths.get("candidate", "")):
        reasons.append("candidate path does not declare one M64 CTA per group")
    if storage.get("q_persistent_across_kv_blocks") is not True:
        reasons.append("candidate does not declare persistent Q across KV blocks")
    if storage.get("qk_scores_in_registers") is not True:
        reasons.append("candidate does not keep QK scores in registers")
    if storage.get("cta_barriers") != 0:
        reasons.append("candidate still contains CTA-wide barriers")
    return reasons


def _performance_gate(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    timing = payload.get("timing") or {}
    speedup = timing.get("candidate_speedup_vs_baseline_pct")
    if not isinstance(speedup, (int, float)):
        return False, ["candidate speedup was not reported"]
    if speedup < K_REQUIRED_SPEEDUP_PCT:
        return False, [
            f"candidate speedup {speedup:.3f}% is below {K_REQUIRED_SPEEDUP_PCT:.3f}%"
        ]
    return True, []


def _binary_command(
    binary: Path,
    args: argparse.Namespace,
    pattern: str,
    nblocks: int,
    groups: int,
    smoke_only: bool,
) -> list[str]:
    command = [
        str(binary),
        "--device",
        "0",
        "--groups",
        str(groups),
        "--nblocks",
        str(nblocks),
        "--warmup",
        str(0 if smoke_only else args.warmup),
        "--rounds",
        str(1 if smoke_only else args.rounds),
        "--launches-per-sample",
        str(1 if smoke_only else args.launches_per_sample),
        "--pattern",
        pattern,
    ]
    if smoke_only:
        command.append("--smoke-only")
    return command


def _run_case(
    binary: Path,
    args: argparse.Namespace,
    pattern: str,
    nblocks: int,
    groups: int,
    smoke_only: bool,
) -> dict[str, Any]:
    command = _binary_command(binary, args, pattern, nblocks, groups, smoke_only)
    if args.verbose:
        print("run:", " ".join(command), file=sys.stderr)
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=_child_environment(args.physical_gpu),
    )
    result: dict[str, Any] = {
        "name": f"{pattern}_nblocks{nblocks}",
        "pattern": pattern,
        "nblocks": nblocks,
        "groups": groups,
        "smoke_only": smoke_only,
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
            "performance_passed": False if not smoke_only else None,
            "reasons": [f"C++ JSON parse failure: {error}"],
        }
        result["stdout_tail"] = completed.stdout[-3000:]
        result["stderr_tail"] = completed.stderr[-3000:]
        return result

    correctness_reasons = _correctness_gate(payload, groups)
    resource_reasons = _runtime_resource_gate(payload)
    performance_passed, performance_reasons = (
        (None, []) if smoke_only else _performance_gate(payload)
    )
    reasons = [
        *([] if completed.returncode == 0 else ["C++ benchmark returned nonzero"]),
        *correctness_reasons,
        *resource_reasons,
        *performance_reasons,
    ]
    result["cpp_json"] = payload
    result["gates"] = {
        "correctness_passed": not correctness_reasons,
        "resource_passed": not resource_reasons,
        "performance_passed": performance_passed,
        "hard_gates_passed": (
            completed.returncode == 0
            and not correctness_reasons
            and not resource_reasons
        ),
        "reasons": reasons,
    }
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-3000:]
    return result


def _required_device_bytes(groups: int, nblocks: int) -> int:
    query = groups * K_M * K_D * 2
    kv = groups * nblocks * K_BLOCK_N * K_D * 2
    outputs = 2 * groups * K_M * K_D * 2
    return query + 2 * kv + outputs


def _long_context_allowed(state: dict[str, Any], groups: int) -> tuple[bool, int]:
    required = _required_device_bytes(groups, 64)
    available = state["memory_free_mib"] * 1024 * 1024
    return available >= required * 2, required


def _aggregate(
    smoke_cases: list[dict[str, Any]], timing_cases: list[dict[str, Any]]
) -> dict[str, Any]:
    smoke_hard_passed = all(case["gates"]["hard_gates_passed"] for case in smoke_cases)
    timing_hard_passed = all(
        case["gates"].get("hard_gates_passed", False) for case in timing_cases
    )
    speedup_cases = [
        case
        for case in timing_cases
        if case.get("gates", {}).get("performance_passed") is not None
    ]
    all_speedup_passed = bool(speedup_cases) and all(
        case["gates"]["performance_passed"] for case in speedup_cases
    )
    primary_cases = [case for case in speedup_cases if case["nblocks"] in (4, 64)]
    primary_speedup_passed = len(primary_cases) == 4 and all(
        case["gates"]["performance_passed"] for case in primary_cases
    )
    return {
        "smoke_hard_gates_passed": smoke_hard_passed,
        "timing_hard_gates_passed": timing_hard_passed,
        "hard_gates_passed": smoke_hard_passed and timing_hard_passed,
        "speedup_10pct_all_timing_cases": all_speedup_passed,
        "speedup_10pct_nblocks4_and_64": primary_speedup_passed,
        "required_candidate_speedup_pct": K_REQUIRED_SPEEDUP_PCT,
        "decision": (
            "measurement_complete"
            if smoke_hard_passed and timing_hard_passed
            else "rejected_correctness_or_resource_gate"
        ),
    }


def _write_payload(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    binary, build_command, ptxas_log = _build_binary(args.verbose)
    ptxas_baseline = _ptxas_function_properties(ptxas_log, K_BASELINE_SYMBOL)
    ptxas_candidate = _ptxas_function_properties(ptxas_log, K_CANDIDATE_SYMBOL)
    sass = _inspect_sass(binary)
    build_reasons = _build_gate(build_command, ptxas_baseline, ptxas_candidate, sass)
    if build_reasons:
        _write_payload(
            {
                "benchmark": "sm70_native_bm64_allp_micro",
                "mode": "build_preflight_rejected_before_gpu_launch",
                "build": {
                    "command": build_command,
                    "ptxas": {
                        "baseline": ptxas_baseline,
                        "candidate": ptxas_candidate,
                    },
                    "ptxas_log": ptxas_log[-6000:],
                },
                "sass": sass,
                "gates": {"hard_gates_passed": False, "reasons": build_reasons},
            },
            args.json_out,
        )
        return 1

    smoke_cases: list[dict[str, Any]] = []
    gpu_states: list[dict[str, Any]] = []
    for pattern, nblocks in K_SMOKE_CASES:
        before = _require_idle_clock(
            args.physical_gpu, args.clock_mhz, f"before smoke {pattern}/N{nblocks}"
        )
        case = _run_case(binary, args, pattern, nblocks, 1, True)
        after = _require_idle_clock(
            args.physical_gpu, args.clock_mhz, f"after smoke {pattern}/N{nblocks}"
        )
        case["gpu_state_before"] = before
        case["gpu_state_after"] = after
        gpu_states.extend((before, after))
        smoke_cases.append(case)

    smoke_hard_passed = all(case["gates"]["hard_gates_passed"] for case in smoke_cases)
    if args.smoke or not smoke_hard_passed:
        payload = {
            "benchmark": "sm70_native_bm64_allp_micro",
            "mode": "resource_and_bitwise_smoke_no_timing",
            "target": "sm_70",
            "build": {
                "command": build_command,
                "ptxas": {
                    "baseline": ptxas_baseline,
                    "candidate": ptxas_candidate,
                },
                "ptxas_log": ptxas_log[-6000:],
            },
            "runtime": {
                "physical_gpu": args.physical_gpu,
                "child_cuda_environment": _child_cuda_configuration(args.physical_gpu),
                "selected_gpu_states": gpu_states,
                "clock_mhz": {"required": args.clock_mhz, "strict": True},
                "frequency_control": "not modified",
            },
            "sass": sass,
            "smoke_cases": smoke_cases,
            "gates": {"hard_gates_passed": smoke_hard_passed},
        }
        _write_payload(payload, args.json_out)
        return 0 if smoke_hard_passed else 1

    timing_cases: list[dict[str, Any]] = []
    long_context: dict[str, Any] = {"nblocks": 64, "patterns": []}
    for pattern, nblocks in K_TIMING_CASES:
        before = _require_idle_clock(
            args.physical_gpu, args.clock_mhz, f"before timing {pattern}/N{nblocks}"
        )
        gpu_states.append(before)
        if nblocks == 64:
            allowed, required_bytes = _long_context_allowed(before, args.groups)
            long_context["required_device_bytes"] = required_bytes
            long_context["allowed_by_free_memory_check"] = allowed
            long_context["patterns"].append(pattern)
            if not allowed:
                timing_cases.append(
                    {
                        "name": "random_nblocks64",
                        "pattern": pattern,
                        "nblocks": nblocks,
                        "groups": args.groups,
                        "skipped": "insufficient_free_memory_before_launch",
                        "gates": {
                            "hard_gates_passed": False,
                            "performance_passed": None,
                            "reasons": ["long-context memory precheck failed"],
                        },
                    }
                )
                continue
        case = _run_case(binary, args, pattern, nblocks, args.groups, False)
        after = _require_idle_clock(
            args.physical_gpu, args.clock_mhz, f"after timing {pattern}/N{nblocks}"
        )
        case["gpu_state_before"] = before
        case["gpu_state_after"] = after
        gpu_states.append(after)
        timing_cases.append(case)

    aggregate = _aggregate(smoke_cases, timing_cases)
    payload = {
        "benchmark": "sm70_native_bm64_allp_micro",
        "target": "sm_70",
        "matrix": [
            {"pattern": pattern, "nblocks": nblocks, "groups": args.groups}
            for pattern, nblocks in K_TIMING_CASES
        ],
        "configuration": {
            "logical_m64_groups": args.groups,
            "baseline_ctas_per_logical_group": 2,
            "candidate_ctas_per_logical_group": 1,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "launches_per_sample": args.launches_per_sample,
            "clock_mhz": args.clock_mhz,
        },
        "build": {
            "command": build_command,
            "ptxas": {
                "baseline": ptxas_baseline,
                "candidate": ptxas_candidate,
            },
            "ptxas_log": ptxas_log[-6000:],
        },
        "runtime": {
            "physical_gpu": args.physical_gpu,
            "child_cuda_environment": _child_cuda_configuration(args.physical_gpu),
            "selected_gpu_states": gpu_states,
            "clock_mhz": {"required": args.clock_mhz, "strict": True},
            "frequency_control": "not modified",
        },
        "sass": sass,
        "smoke_cases": smoke_cases,
        "timing_cases": timing_cases,
        "long_context": long_context,
        "gates": aggregate,
    }
    _write_payload(payload, args.json_out)
    return 0 if aggregate["hard_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
