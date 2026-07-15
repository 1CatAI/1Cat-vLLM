# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and inspect the FlashInfer-shaped SM70 Volta MMA primitive probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

K_TARGET = "sm_70"
K_DEFAULT_GROUPS = 4096
K_DEFAULT_WARMUP = 20
K_DEFAULT_ROUNDS = 100
K_DEFAULT_LAUNCHES_PER_SAMPLE = 8
K_REDUCTION_K = 256
K_K_TILES = 16
K_EXPECTED_HMMA_PER_KERNEL = 256
K_KERNELS = {
    "qk_row_col": {
        "reference": "sm70_flashinfer_volta_mma_qk_reference_kernel",
        "compatibility": "sm70_flashinfer_volta_mma_qk_compat_kernel",
    },
    "pv_row_row": {
        "reference": "sm70_flashinfer_volta_mma_pv_reference_kernel",
        "compatibility": "sm70_flashinfer_volta_mma_pv_compat_kernel",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Logical CUDA device after CUDA_VISIBLE_DEVICES is constrained.",
    )
    parser.add_argument(
        "--physical-gpu",
        type=int,
        default=0,
        help="Physical SM70 GPU to inspect and expose to the child process.",
    )
    parser.add_argument(
        "--clock-mhz",
        type=int,
        default=1200,
        help="Required graphics clock before the probe is allowed to run.",
    )
    parser.add_argument("--groups", type=int, default=K_DEFAULT_GROUPS)
    parser.add_argument("--warmup", type=int, default=K_DEFAULT_WARMUP)
    parser.add_argument("--rounds", type=int, default=K_DEFAULT_ROUNDS)
    parser.add_argument(
        "--launches-per-sample",
        type=int,
        default=K_DEFAULT_LAUNCHES_PER_SAMPLE,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one group and one reference/compatibility timing pair.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.device != 0:
        raise ValueError("The child exposes exactly one GPU; --device must be 0.")
    if args.physical_gpu < 0:
        raise ValueError("--physical-gpu must be non-negative.")
    if args.clock_mhz <= 0:
        raise ValueError("--clock-mhz must be positive.")
    if args.groups < 1 or args.rounds < 1 or args.launches_per_sample < 1:
        raise ValueError("groups, rounds, and launches-per-sample must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")


def _nvidia_smi() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required for the SM70 preflight check.")
    return executable


def _selected_gpu_state(physical_gpu: int) -> dict[str, Any]:
    executable = _nvidia_smi()
    query = subprocess.run(
        [
            executable,
            "--query-gpu=index,uuid,name,compute_cap,utilization.gpu,"
            "memory.used,clocks.current.graphics",
            "--format=csv,noheader,nounits",
            "--id",
            str(physical_gpu),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    fields = [field.strip() for field in query.stdout.strip().split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"Could not parse GPU state: {query.stdout!r}")
    index, uuid, name, capability, utilization, memory_used, graphics_clock = fields
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
        line
        for line in applications.stdout.splitlines()
        if line.strip().startswith(uuid + ",")
    ]
    return {
        "index": int(index),
        "uuid": uuid,
        "name": name,
        "compute_capability": capability,
        "utilization_gpu_pct": int(utilization),
        "memory_used_mib": int(memory_used),
        "graphics_clock_mhz": int(graphics_clock),
        "compute_processes": active,
    }


def _require_idle_clock(physical_gpu: int, required_clock_mhz: int) -> dict[str, Any]:
    state = _selected_gpu_state(physical_gpu)
    if state["compute_processes"]:
        processes = "; ".join(state["compute_processes"])
        raise RuntimeError(f"Physical GPU {physical_gpu} is not idle: {processes}")
    if state["utilization_gpu_pct"] != 0:
        raise RuntimeError(
            f"Physical GPU {physical_gpu} utilization is "
            f"{state['utilization_gpu_pct']}%, not idle."
        )
    if state["graphics_clock_mhz"] != required_clock_mhz:
        raise RuntimeError(
            f"Physical GPU {physical_gpu} graphics clock is "
            f"{state['graphics_clock_mhz']} MHz; requires exactly "
            f"{required_clock_mhz} MHz before timing."
        )
    return state


def _child_environment(physical_gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
        }
    )
    return environment


def _build_binary(verbose: bool) -> tuple[Path, list[str], str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required to build the SM70 primitive probe.")
    repository_root = Path(__file__).resolve().parents[1]
    source = repository_root / "benchmarks/csrc/sm70_flashinfer_volta_mma_micro.cu"
    include_dir = repository_root / "flashinfer-sm70/include"
    if not source.is_file() or not include_dir.is_dir():
        raise RuntimeError(
            "The SM70 primitive source or compatibility include is missing."
        )
    build_dir = (
        Path(tempfile.gettempdir())
        / f"vllm-sm70-flashinfer-volta-mma-micro-{os.getuid()}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_flashinfer_volta_mma_micro_sm70"
    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        "-lineinfo",
        f"-I{include_dir}",
        "--generate-code=arch=compute_70,code=sm_70",
        "--ptxas-options=-v",
        "-o",
        str(binary),
        str(source),
    ]
    if verbose:
        print("build:", " ".join(command), file=sys.stderr)
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    ptxas_log = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if ptxas_log:
        print(ptxas_log, end="" if ptxas_log.endswith("\n") else "\n", file=sys.stderr)
    return binary, command, ptxas_log


def _ptxas_function_properties(ptxas_log: str, symbol: str) -> dict[str, int] | None:
    entries = list(re.finditer(r"Compiling entry function '([^']+)'", ptxas_log))
    for index, entry in enumerate(entries):
        if entry.group(1) != symbol:
            continue
        end = entries[index + 1].start() if index + 1 < len(entries) else len(ptxas_log)
        section = ptxas_log[entry.start() : end]
        stack = re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads",
            section,
        )
        registers = re.search(r"Used (\d+) registers", section)
        if stack is None or registers is None:
            return None
        return {
            "registers_per_thread": int(registers.group(1)),
            "stack_frame_bytes": int(stack.group(1)),
            "spill_store_bytes": int(stack.group(2)),
            "spill_load_bytes": int(stack.group(3)),
        }
    return None


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
    for operation, implementations in K_KERNELS.items():
        for implementation, symbol in implementations.items():
            section = _function_section(completed.stdout, symbol)
            key = f"{operation}.{implementation}"
            if section is None:
                kernels[key] = {"available": False, "symbol": symbol}
                continue
            kernels[key] = {
                "available": True,
                "symbol": symbol,
                "instructions": {
                    mnemonic.lower(): _instruction_count(section, mnemonic)
                    for mnemonic in ("HMMA", "LDL", "STL")
                },
            }
    return {"available": True, "binary": str(binary), "kernels": kernels}


def _runtime_command(binary: Path, args: argparse.Namespace) -> list[str]:
    command = [
        str(binary),
        "--device",
        str(args.device),
        "--groups",
        str(args.groups),
        "--warmup",
        str(args.warmup),
        "--rounds",
        str(args.rounds),
        "--launches-per-sample",
        str(args.launches_per_sample),
    ]
    if args.smoke:
        command.append("--smoke")
    return command


def _effective_measurement(args: argparse.Namespace) -> tuple[int, int]:
    if args.smoke:
        return 1, 1
    return args.groups, args.rounds


def _check_payload(payload: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    groups, rounds = _effective_measurement(args)
    if payload.get("target") != K_TARGET:
        failures.append(f"target={payload.get('target')!r}, requires {K_TARGET!r}")
    if payload.get("scope") != "primitive compatibility only; no attention speed claim":
        failures.append("result did not retain the primitive-only scope declaration")
    if payload.get("all_bitwise_equal") is not True:
        failures.append("C++ probe reported a non-bitwise-equal output")
    if payload.get("shape", {}).get("groups") != groups:
        failures.append("C++ probe reported an unexpected group count")
    shape = payload.get("shape", {})
    if shape.get("a") != ["groups", 16, K_REDUCTION_K]:
        failures.append("C++ probe did not use an M16xK256 A operand")
    if shape.get("qk_b_physical") != ["groups", 16, K_REDUCTION_K]:
        failures.append("C++ probe did not use a physical N16xK256 key")
    if shape.get("pv_b") != ["groups", K_REDUCTION_K, 16]:
        failures.append("C++ probe did not use a K256xN16 value operand")
    work = payload.get("work", {})
    if work.get("k_tiles") != K_K_TILES:
        failures.append("C++ probe did not execute 16 ordered K16 tiles")
    if work.get("wmma_m16n16k16_updates_per_warp") != K_K_TILES:
        failures.append("C++ probe did not report 16 accumulator updates")
    if work.get("compatibility_pointer_control_used") is not False:
        failures.append("compatibility kernel still uses the pointer control")

    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 4:
        return failures + ["expected random/alternating QK/PV results"]
    seen = {(item.get("input_pattern"), item.get("operation")) for item in results}
    expected = {
        ("random", "qk_row_col"),
        ("random", "pv_row_row"),
        ("alternating", "qk_row_col"),
        ("alternating", "pv_row_row"),
    }
    if seen != expected:
        failures.append(f"unexpected pattern/operation coverage: {sorted(seen)!r}")

    expected_words = groups * 16 * 16
    for item in results:
        exactness = item.get("exactness", {})
        if item.get("reference_implementation") != (
            "independent direct nvcuda::wmma K=256 loop"
        ):
            failures.append(
                f"{item.get('operation')} lacks an independent WMMA reference"
            )
        if item.get("accumulator_lifetime") != (
            "one load, 16 register updates, one store"
        ):
            failures.append(
                f"{item.get('operation')} did not keep one accumulator lifetime"
            )
        if exactness.get("word_dtype") != "uint32(fp32)":
            failures.append(
                f"{item.get('operation')} did not compare FP32 bit patterns"
            )
        if exactness.get("word_count") != expected_words:
            failures.append(f"{item.get('operation')} compared an incomplete output")
        if exactness.get("full_output") is not True:
            failures.append(f"{item.get('operation')} did not mark the output complete")
        if exactness.get("bitwise_equal") is not True:
            failures.append(f"{item.get('operation')} is not bitwise equal")
        if exactness.get("mismatch_words") != 0:
            failures.append(f"{item.get('operation')} has mismatched FP32 words")
        if exactness.get("xor_reduction") != 0 or exactness.get("max_word_xor") != 0:
            failures.append(f"{item.get('operation')} has a nonzero XOR diagnostic")
        pairs = item.get("pairs", {})
        if pairs.get("count") != rounds:
            failures.append(f"{item.get('operation')} pair count does not match rounds")
        if pairs.get("compatibility_first_count") != rounds // 2:
            failures.append(f"{item.get('operation')} did not alternate pair order")
        for timing_name in ("reference", "compatibility"):
            median = item.get("timing", {}).get(timing_name, {}).get("median_us")
            if not isinstance(median, (int, float)) or not math.isfinite(median):
                failures.append(
                    f"{item.get('operation')} did not report finite "
                    f"{timing_name} timing"
                )
    return failures


def _check_build(
    command: list[str], ptxas: dict[str, dict[str, int] | None], sass: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    if "--generate-code=arch=compute_70,code=sm_70" not in command:
        failures.append("build command is not fixed to sm_70")
    if any("maxrregcount" in argument for argument in command):
        failures.append("build command must not force maxrregcount")
    for operation, implementations in K_KERNELS.items():
        for implementation, symbol in implementations.items():
            properties = ptxas.get(symbol)
            label = f"{operation}.{implementation}"
            if properties is None:
                failures.append(f"PTXAS registers/stack/spill data missing for {label}")
                continue
            for field in (
                "registers_per_thread",
                "stack_frame_bytes",
                "spill_store_bytes",
                "spill_load_bytes",
            ):
                if field not in properties:
                    failures.append(f"PTXAS {field} missing for {label}")
            for field in ("spill_store_bytes", "spill_load_bytes"):
                if properties.get(field) != 0:
                    failures.append(
                        f"PTXAS {label} {field}={properties.get(field)}, requires 0"
                    )
    if not sass.get("available"):
        failures.append("SASS inspection is unavailable")
        return failures
    for operation in K_KERNELS:
        for implementation in ("reference", "compatibility"):
            label = f"{operation}.{implementation}"
            kernel = sass.get("kernels", {}).get(label, {})
            if not kernel.get("available"):
                failures.append(f"SASS function missing for {label}")
                continue
            instructions = kernel.get("instructions", {})
            if instructions.get("hmma") != K_EXPECTED_HMMA_PER_KERNEL:
                failures.append(
                    f"SASS {label} HMMA={instructions.get('hmma')}, "
                    f"requires {K_EXPECTED_HMMA_PER_KERNEL}"
                )
            for mnemonic in ("ldl", "stl"):
                if instructions.get(mnemonic) != 0:
                    failures.append(
                        f"SASS {label} {mnemonic.upper()}="
                        f"{instructions.get(mnemonic)}, requires 0"
                    )
    return failures


def _write_report(report: dict[str, Any], destination: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    _validate_args(args)

    # This happens before compilation so an unlocked or busy GPU cannot produce
    # a timing result that looks comparable to the fixed-clock microbenchmark.
    gpu_before = _require_idle_clock(args.physical_gpu, args.clock_mhz)
    binary, build_command, ptxas_log = _build_binary(args.verbose)
    ptxas = {
        symbol: _ptxas_function_properties(ptxas_log, symbol)
        for implementations in K_KERNELS.values()
        for symbol in implementations.values()
    }
    sass = _inspect_sass(binary)

    command = _runtime_command(binary, args)
    if args.verbose:
        print("run:", " ".join(command), file=sys.stderr)
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=_child_environment(args.physical_gpu),
    )
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    payload: dict[str, Any]
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The CUDA microbenchmark did not emit JSON: "
            f"{error}; stdout tail={completed.stdout[-2000:]!r}"
        ) from error

    failures = _check_build(build_command, ptxas, sass)
    failures.extend(_check_payload(payload, args))
    if completed.returncode:
        failures.append(f"CUDA microbenchmark exited with {completed.returncode}")
    report = {
        "preflight": {
            "physical_gpu": args.physical_gpu,
            "required_clock_mhz": args.clock_mhz,
            "state_before_build": gpu_before,
        },
        "build": {
            "target": K_TARGET,
            "command": build_command,
            "ptxas": ptxas,
        },
        "sass": sass,
        "result": payload,
        "checks": {
            "compatibility_passed": not failures,
            "failures": failures,
            "required_hmma_per_kernel": K_EXPECTED_HMMA_PER_KERNEL,
            "required_spill_ldl_stl": 0,
            "timing_is_observational_only": True,
        },
    }
    _write_report(report, args.json_out)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
