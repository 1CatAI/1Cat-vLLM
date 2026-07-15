# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and measure the standalone SM70 BM32 3-way split-KV microbenchmark."""

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

K_DEFAULT_GROUPS = 96
K_DEFAULT_NBLOCKS = 96
K_DEFAULT_WARMUP = 20
K_DEFAULT_ROUNDS = 100
K_DEFAULT_LAUNCHES = 8
K_SPLIT_PARTS = 3
K_M64 = 64
K_D = 256
K_BN = 128
K_BM32_THREADS = 512
K_ALLP_SHARED_BYTES = 41744
K_BASELINE_SYMBOL = "sm70_native_bm32_splitkv_unsplit_baseline"
K_PARTIAL_SYMBOL = "sm70_native_bm32_splitkv_partial"
K_MERGE_SYMBOL = "sm70_native_bm32_splitkv_merge"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--clock-mhz", type=int, default=1200)
    parser.add_argument("--groups", type=int, default=K_DEFAULT_GROUPS)
    parser.add_argument("--nblocks", type=int, default=K_DEFAULT_NBLOCKS)
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
        "--pattern", choices=("random", "alternating"), default="random"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run resource and numerical checks without timing rounds.",
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
    if args.nblocks < K_SPLIT_PARTS:
        raise ValueError(
            f"--nblocks must be at least {K_SPLIT_PARTS} for 3-way split-KV."
        )
    if args.warmup < 0 or args.rounds < 1 or args.launches_per_sample < 1:
        raise ValueError("warmup must be non-negative; rounds and launches positive.")


def _child_cuda_configuration(physical_gpu: int) -> dict[str, str]:
    # Use PCI order before masking so logical cuda:0 is the requested GPU.
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
        raise RuntimeError("nvidia-smi is required to select an idle SM70 GPU.")
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
    source = (
        Path(__file__).resolve().parent / "csrc" / "sm70_native_bm32_splitkv_micro.cu"
    )
    if not source.is_file():
        raise RuntimeError(f"Microbenchmark source does not exist: {source}")
    build_dir = (
        Path(tempfile.gettempdir())
        / f"vllm-sm70-native-bm32-splitkv-micro-{os.getuid()}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_native_bm32_splitkv_micro_sm70"
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
    for name, symbol, ctas_per_m64_group in (
        ("baseline", K_BASELINE_SYMBOL, 2),
        ("partial", K_PARTIAL_SYMBOL, 6),
        ("merge", K_MERGE_SYMBOL, 2),
    ):
        section = _function_section(completed.stdout, symbol)
        if section is None:
            kernels[name] = {"available": False, "symbol": symbol}
            continue
        instructions = {
            mnemonic.lower(): _instruction_count(section, mnemonic)
            for mnemonic in ("HMMA", "LDG", "LDS", "STS", "LDL", "STL", "BAR")
        }
        kernels[name] = {
            "available": True,
            "symbol": symbol,
            "ctas_per_m64_group": ctas_per_m64_group,
            "instructions": instructions,
        }
    return {"available": True, "binary": str(binary), "kernels": kernels}


def _strict_bm32_ptxas_gate(name: str, properties: dict[str, int] | None) -> list[str]:
    if properties is None:
        return [f"PTXAS properties missing for {name}"]
    reasons: list[str] = []
    expected = {
        "registers_per_thread": 64,
        "static_shared_bytes": K_ALLP_SHARED_BYTES,
        "stack_frame_bytes": 0,
        "spill_store_bytes": 0,
        "spill_load_bytes": 0,
    }
    for field, required in expected.items():
        if properties.get(field) != required:
            reasons.append(
                f"PTXAS {name} {field}={properties.get(field)}, requires {required}"
            )
    return reasons


def _build_gate(
    command: list[str], ptxas: dict[str, dict[str, int] | None], sass: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if any("maxrregcount" in argument for argument in command):
        reasons.append("build command must not force maxrregcount")
    for name in ("baseline", "partial"):
        reasons.extend(_strict_bm32_ptxas_gate(name, ptxas[name]))

    merge_ptxas = ptxas["merge"]
    if merge_ptxas is None:
        reasons.append("PTXAS properties missing for merge")
    else:
        for field in ("stack_frame_bytes", "spill_store_bytes", "spill_load_bytes"):
            if merge_ptxas[field] != 0:
                reasons.append(f"PTXAS merge {field}={merge_ptxas[field]}, requires 0")

    if not sass.get("available"):
        reasons.append("SASS inspection unavailable")
        return reasons
    kernels = sass.get("kernels", {})
    for name in ("baseline", "partial", "merge"):
        kernel = kernels.get(name, {})
        if not kernel.get("available"):
            reasons.append(f"SASS function missing for {name}")
            continue
        instructions = kernel.get("instructions", {})
        if instructions.get("ldl", 0) or instructions.get("stl", 0):
            reasons.append(
                f"{name} SASS local instructions: LDL={instructions.get('ldl', 0)} "
                f"STL={instructions.get('stl', 0)}"
            )
    for name in ("baseline", "partial"):
        instructions = kernels.get(name, {}).get("instructions", {})
        for mnemonic in ("hmma", "ldg", "lds"):
            if instructions.get(mnemonic, 0) == 0:
                reasons.append(f"{name} SASS has no {mnemonic.upper()} instruction")
    if kernels.get("merge", {}).get("instructions", {}).get("ldg", 0) == 0:
        reasons.append("merge SASS has no LDG instruction")
    return reasons


def _runtime_resource_gate(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    execution = payload.get("execution", {})
    resources = payload.get("resources", {})
    resource_gate = payload.get("resource_gate", {})
    if execution.get("resource_gate_pass") is not True:
        reasons.append("C++ runtime BM32 resource gate did not pass")
    if resource_gate.get("runtime_pass") is not True:
        reasons.append("C++ runtime resource_gate.runtime_pass is false")
    for name in ("baseline", "partial"):
        resource = resources.get(name, {})
        expected = {
            "threads_per_cta": K_BM32_THREADS,
            "registers_per_thread": 64,
            "static_shared_bytes": K_ALLP_SHARED_BYTES,
            "dynamic_shared_bytes": 0,
            "local_bytes_per_thread": 0,
            "active_ctas_per_sm": 2,
        }
        for field, required in expected.items():
            if resource.get(field) != required:
                reasons.append(
                    f"runtime {name} {field}={resource.get(field)}, requires {required}"
                )
    merge = resources.get("merge", {})
    if merge.get("threads_per_cta") is None:
        reasons.append("runtime merge resources were not reported")
    paths = payload.get("paths", {})
    if "ALL_P=true PAIR_SCRATCH=true" not in str(paths.get("baseline", "")):
        reasons.append("baseline path does not declare accepted ALL_P+PAIR_SCRATCH")
    if "3-way split-KV" not in str(paths.get("partial", "")):
        reasons.append("partial path does not declare 3-way split-KV")
    return reasons


def _correctness_gate(payload: dict[str, Any], groups: int) -> list[str]:
    reasons: list[str] = []
    comparison = payload.get("comparison") or {}
    expected_elements = groups * K_M64 * K_D
    if comparison.get("full_output") is not True:
        reasons.append("comparison did not cover the full output")
    if comparison.get("all_finite") is not True:
        reasons.append("split output contains non-finite values")
    max_abs_error = comparison.get("max_abs_error")
    tolerance = comparison.get("abs_tolerance")
    if not isinstance(max_abs_error, (int, float)) or not isinstance(
        tolerance, (int, float)
    ):
        reasons.append("comparison did not report numeric max_abs_error/tolerance")
    elif max_abs_error > tolerance:
        reasons.append(
            f"max_abs_error={max_abs_error} exceeds declared tolerance={tolerance}"
        )
    mismatches = comparison.get("bitwise_mismatch_elements")
    if (
        not isinstance(mismatches, int)
        or mismatches < 0
        or mismatches > expected_elements
    ):
        reasons.append("comparison reported an invalid bitwise mismatch count")
    # Split-KV changes the online-softmax reduction order; bitwise equality is reported,
    # but deliberately is not a correctness gate.
    return reasons


def _timing_gate(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    timing = payload.get("timing") or {}
    for name in ("baseline", "partial", "merge", "combined"):
        summary = timing.get(name)
        if not isinstance(summary, dict):
            reasons.append(f"timing did not report {name}")
            continue
        for field in ("median_us", "p90_us"):
            value = summary.get(field)
            if not isinstance(value, (int, float)) or value < 0:
                reasons.append(f"timing {name}.{field} is not a non-negative number")
    speedup = timing.get("combined_speedup_vs_baseline_pct")
    if not isinstance(speedup, (int, float)):
        reasons.append("combined speedup versus baseline was not reported")
    return reasons


def _binary_command(
    binary: Path, args: argparse.Namespace, smoke_only: bool
) -> list[str]:
    command = [
        str(binary),
        "--device",
        "0",
        "--groups",
        str(args.groups),
        "--nblocks",
        str(args.nblocks),
        "--warmup",
        str(0 if smoke_only else args.warmup),
        "--rounds",
        str(1 if smoke_only else args.rounds),
        "--launches-per-sample",
        str(1 if smoke_only else args.launches_per_sample),
        "--pattern",
        args.pattern,
    ]
    if smoke_only:
        command.append("--smoke-only")
    return command


def _run_case(binary: Path, args: argparse.Namespace) -> dict[str, Any]:
    command = _binary_command(binary, args, args.smoke)
    if args.verbose:
        print("run:", " ".join(command), file=sys.stderr)
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=_child_environment(args.physical_gpu),
    )
    result: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
    }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        result["cpp_json"] = None
        result["gates"] = {
            "runtime_resource_passed": False,
            "correctness_passed": False,
            "timing_reported": False if not args.smoke else None,
            "reasons": [f"C++ JSON parse failure: {error}"],
        }
        result["stdout_tail"] = completed.stdout[-3000:]
        result["stderr_tail"] = completed.stderr[-3000:]
        return result

    runtime_resource_reasons = _runtime_resource_gate(payload)
    correctness_reasons = _correctness_gate(payload, args.groups)
    timing_reasons = [] if args.smoke else _timing_gate(payload)
    reasons = [
        *([] if completed.returncode == 0 else ["C++ benchmark returned nonzero"]),
        *runtime_resource_reasons,
        *correctness_reasons,
        *timing_reasons,
    ]
    result["cpp_json"] = payload
    result["gates"] = {
        "runtime_resource_passed": not runtime_resource_reasons,
        "correctness_passed": not correctness_reasons,
        "timing_reported": None if args.smoke else not timing_reasons,
        "hard_gates_passed": (
            completed.returncode == 0
            and not runtime_resource_reasons
            and not correctness_reasons
            and (args.smoke or not timing_reasons)
        ),
        "reasons": reasons,
    }
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-3000:]
    return result


def _write_payload(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    try:
        binary, build_command, ptxas_log = _build_binary(args.verbose)
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = (
            error.stderr if isinstance(error, subprocess.CalledProcessError) else ""
        )
        _write_payload(
            {
                "benchmark": "sm70_native_bm32_splitkv_micro",
                "mode": "build_failed_before_gpu_launch",
                "error": str(error),
                "stderr_tail": (stderr or "")[-6000:],
            },
            args.json_out,
        )
        return 1

    ptxas = {
        "baseline": _ptxas_function_properties(ptxas_log, K_BASELINE_SYMBOL),
        "partial": _ptxas_function_properties(ptxas_log, K_PARTIAL_SYMBOL),
        "merge": _ptxas_function_properties(ptxas_log, K_MERGE_SYMBOL),
    }
    sass = _inspect_sass(binary)
    build_reasons = _build_gate(build_command, ptxas, sass)
    if build_reasons:
        _write_payload(
            {
                "benchmark": "sm70_native_bm32_splitkv_micro",
                "mode": "build_preflight_rejected_before_gpu_launch",
                "target": "sm_70",
                "build": {
                    "command": build_command,
                    "ptxas": ptxas,
                    "ptxas_log_tail": ptxas_log[-6000:],
                },
                "sass": sass,
                "gates": {"build_passed": False, "reasons": build_reasons},
            },
            args.json_out,
        )
        return 1

    try:
        before = _require_idle_clock(
            args.physical_gpu, args.clock_mhz, "before benchmark"
        )
        case = _run_case(binary, args)
        after = _require_idle_clock(
            args.physical_gpu, args.clock_mhz, "after benchmark"
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        _write_payload(
            {
                "benchmark": "sm70_native_bm32_splitkv_micro",
                "mode": "runtime_precondition_failed",
                "target": "sm_70",
                "error": str(error),
                "build": {
                    "command": build_command,
                    "ptxas": ptxas,
                    "ptxas_log_tail": ptxas_log[-6000:],
                },
                "sass": sass,
            },
            args.json_out,
        )
        return 1

    hard_gates_passed = case["gates"].get("hard_gates_passed") is True
    payload: dict[str, Any] = {
        "benchmark": "sm70_native_bm32_splitkv_micro",
        "mode": "smoke" if args.smoke else "measurement",
        "target": "sm_70",
        "configuration": {
            "logical_m64_groups": args.groups,
            "nblocks": args.nblocks,
            "tokens": args.nblocks * K_BN,
            "split_parts": K_SPLIT_PARTS,
            "baseline_ctas": args.groups * 2,
            "partial_ctas": args.groups * 2 * K_SPLIT_PARTS,
            "warmup": 0 if args.smoke else args.warmup,
            "rounds": 1 if args.smoke else args.rounds,
            "launches_per_sample": 1 if args.smoke else args.launches_per_sample,
            "input_pattern": args.pattern,
        },
        "build": {
            "command": build_command,
            "target": "sm_70",
            "maxrregcount_present": any(
                "maxrregcount" in argument for argument in build_command
            ),
            "ptxas": ptxas,
            "ptxas_log_tail": ptxas_log[-6000:],
        },
        "runtime": {
            "physical_gpu": args.physical_gpu,
            "child_cuda_environment": _child_cuda_configuration(args.physical_gpu),
            "selected_gpu_state_before": before,
            "selected_gpu_state_after": after,
            "clock_mhz": {"required": args.clock_mhz, "strict": True},
            "frequency_control": "not modified",
        },
        "sass": sass,
        "case": case,
        "gates": {
            "build_passed": True,
            "runtime_resource_passed": case["gates"].get("runtime_resource_passed"),
            "correctness_passed": case["gates"].get("correctness_passed"),
            "timing_reported": case["gates"].get("timing_reported"),
            "hard_gates_passed": hard_gates_passed,
            "reasons": case["gates"].get("reasons", []),
        },
    }
    _write_payload(payload, args.json_out)
    return 0 if hard_gates_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
