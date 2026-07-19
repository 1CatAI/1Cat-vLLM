# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate and measure the structural SM70 FlashInfer BM64 pipeline microbenchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import regex as re

K_GROUPS = 72
K_M = 64
K_D = 256
K_BLOCK_N = 128
K_BASELINE_THREADS = 512
K_CANDIDATE_THREADS = 256
K_BASELINE_SHARED_BYTES = 41744
K_CANDIDATE_SHARED_BYTES = 43776
K_MAX_CANDIDATE_REGISTERS = 128
K_MAX_SHARED_BYTES = 48 * 1024
K_SMOKE_CASES = (
    ("random", 1),
    ("alternating", 1),
    ("random", 2),
    ("alternating", 2),
    ("random", 4),
    ("alternating", 4),
)
K_BASELINE_SYMBOL = "sm70_native_bm64_allp_bm32_pair_scratch_baseline"
K_CANDIDATE_SYMBOL = "sm70_flashinfer_bm64_pipeline_candidate"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--groups", type=int, default=K_GROUPS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument(
        "--launches",
        "--launches-per-sample",
        dest="launches_per_sample",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--require-clock-mhz",
        type=int,
        help="Reject execution unless the selected GPU reports this graphics clock.",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Permit a GPU with active compute processes.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run resource and bitwise smoke cases without timing.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.device != 0:
        raise ValueError("The child process exposes one GPU; use --device 0.")
    if args.physical_gpu < 0:
        raise ValueError("--physical-gpu must be non-negative.")
    if args.groups < 1:
        raise ValueError("--groups must be positive.")
    if args.warmup < 0 or args.rounds < 1 or args.launches_per_sample < 1:
        raise ValueError("warmup must be non-negative; rounds and launches positive.")
    if args.require_clock_mhz is not None and args.require_clock_mhz <= 0:
        raise ValueError("--require-clock-mhz must be positive.")


def _child_cuda_configuration(physical_gpu: int) -> dict[str, str]:
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
        raise RuntimeError("nvidia-smi is required to select an SM70 GPU.")
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
    return {
        "index": int(index),
        "uuid": uuid,
        "name": name,
        "compute_capability": capability,
        "memory_free_mib": int(free_mib),
        "graphics_clock_mhz": int(graphics_clock_mhz),
        "compute_processes": active,
    }


def _require_execution_device(args: argparse.Namespace, phase: str) -> dict[str, Any]:
    state = _selected_gpu_state(args.physical_gpu)
    if state["compute_capability"] != "7.0":
        raise RuntimeError(
            f"Physical GPU {args.physical_gpu} is {state['name']} "
            f"(SM{state['compute_capability']}), not SM70, {phase}."
        )
    if state["compute_processes"] and not args.allow_busy_gpu:
        raise RuntimeError(
            f"Physical GPU {args.physical_gpu} is busy {phase}: "
            + "; ".join(state["compute_processes"])
        )
    if (
        args.require_clock_mhz is not None
        and state["graphics_clock_mhz"] != args.require_clock_mhz
    ):
        raise RuntimeError(
            f"Physical GPU {args.physical_gpu} graphics clock is "
            f"{state['graphics_clock_mhz']} MHz {phase}; requires "
            f"{args.require_clock_mhz} MHz."
        )
    return state


def _source_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "csrc"
        / "sm70_flashinfer_bm64_pipeline_micro.cu"
    )


def _build_binary(verbose: bool) -> tuple[Path, list[str], str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required to build the SM70 microbenchmark.")
    source = _source_path()
    if not source.is_file():
        raise RuntimeError(f"Microbenchmark source does not exist: {source}")
    build_dir = (
        Path(tempfile.gettempdir())
        / f"vllm-sm70-flashinfer-bm64-pipeline-micro-{os.getuid()}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_flashinfer_bm64_pipeline_micro_sm70"
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
    for name, symbol in (
        ("baseline", K_BASELINE_SYMBOL),
        ("candidate", K_CANDIDATE_SYMBOL),
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
            "instructions": instructions,
        }
    return {"available": True, "binary": str(binary), "kernels": kernels}


def _source_gate() -> list[str]:
    source = _source_path().read_text(encoding="utf-8")
    reasons: list[str] = []
    if '#include "sm70_native_bm64_allp_micro.cu"' not in source:
        reasons.append("candidate does not directly reuse the accepted BM32 baseline")
    if "sm70_native_bm64_allp_bm32_pair_scratch_baseline" not in source:
        reasons.append("candidate source does not launch the pair-scratch baseline")
    if "__launch_bounds__(kPipelineThreads, 2)" not in source:
        reasons.append("candidate launch bounds do not require two CTAs per SM")
    if "maxrregcount" in source.lower():
        reasons.append("candidate source must not force maxrregcount")
    if re.search(r"asm[^\n]*hmma", source, re.IGNORECASE):
        reasons.append("candidate contains raw HMMA assembly instead of WMMA")
    return reasons


def _build_gate(
    command: list[str],
    ptxas_baseline: dict[str, int] | None,
    ptxas_candidate: dict[str, int] | None,
    sass: dict[str, Any],
) -> list[str]:
    reasons = _source_gate()
    if any("maxrregcount" in argument.lower() for argument in command):
        reasons.append("build command must not force maxrregcount")
    expected = (
        ("baseline", ptxas_baseline, 64, K_BASELINE_SHARED_BYTES),
        (
            "candidate",
            ptxas_candidate,
            K_MAX_CANDIDATE_REGISTERS,
            K_CANDIDATE_SHARED_BYTES,
        ),
    )
    for name, properties, max_registers, expected_shared in expected:
        if properties is None:
            reasons.append(f"PTXAS properties missing for {name}")
            continue
        if properties["registers_per_thread"] > max_registers:
            reasons.append(
                f"PTXAS {name} REG={properties['registers_per_thread']}, "
                f"requires <= {max_registers}"
            )
        if properties["static_shared_bytes"] != expected_shared:
            reasons.append(
                f"PTXAS {name} static shared="
                f"{properties['static_shared_bytes']}, requires {expected_shared}"
            )
        for key in ("stack_frame_bytes", "spill_store_bytes", "spill_load_bytes"):
            if properties[key] != 0:
                reasons.append(f"PTXAS {name} {key}={properties[key]}, requires 0")

    if not sass.get("available"):
        return [*reasons, "SASS inspection is required for the hard gate"]
    for name in ("baseline", "candidate"):
        kernel = sass.get("kernels", {}).get(name, {})
        if not kernel.get("available"):
            reasons.append(f"SASS section missing for {name}")
            continue
        instructions = kernel.get("instructions", {})
        if instructions.get("ldl", 0) or instructions.get("stl", 0):
            reasons.append(
                f"{name} SASS local instructions: LDL={instructions.get('ldl', 0)} "
                f"STL={instructions.get('stl', 0)}"
            )
        if instructions.get("hmma", 0) == 0:
            reasons.append(f"{name} SASS has no Volta HMMA from native WMMA")
    candidate_instructions = (
        sass.get("kernels", {}).get("candidate", {}).get("instructions", {})
    )
    for mnemonic in ("ldg", "sts", "bar"):
        if candidate_instructions.get(mnemonic, 0) == 0:
            reasons.append(f"candidate SASS has no {mnemonic.upper()} instruction")
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
        reasons.append("bitwise probe did not compare the full output")
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
    contract = payload.get("candidate_contract") or {}
    paths = payload.get("paths") or {}
    if execution.get("resource_gate_pass") is not True:
        reasons.append("C++ runtime resource gate did not pass")
    if baseline.get("threads_per_cta") != K_BASELINE_THREADS:
        reasons.append("baseline does not retain 512 threads per CTA")
    if baseline.get("registers_per_thread") != 64:
        reasons.append("baseline does not retain the accepted 64-register shape")
    if baseline.get("static_shared_bytes") != K_BASELINE_SHARED_BYTES:
        reasons.append("baseline does not retain the accepted 41,744B layout")
    if baseline.get("dynamic_shared_bytes") != 0:
        reasons.append("baseline unexpectedly uses dynamic shared memory")
    if baseline.get("local_bytes_per_thread") != 0:
        reasons.append("baseline runtime local bytes are nonzero")
    if baseline.get("active_ctas_per_sm") != 2:
        reasons.append("baseline must retain two CTAs per SM")
    if candidate.get("threads_per_cta") != K_CANDIDATE_THREADS:
        reasons.append("candidate does not use eight 32-thread warps")
    if candidate.get("registers_per_thread", K_MAX_CANDIDATE_REGISTERS + 1) > (
        K_MAX_CANDIDATE_REGISTERS
    ):
        reasons.append("candidate exceeds 128 registers per thread")
    if candidate.get("static_shared_bytes") != K_CANDIDATE_SHARED_BYTES:
        reasons.append("candidate shared layout is not the specified 43,776B layout")
    if (
        candidate.get("static_shared_bytes", K_MAX_SHARED_BYTES + 1)
        > K_MAX_SHARED_BYTES
    ):
        reasons.append("candidate shared memory exceeds 48 KiB")
    if candidate.get("dynamic_shared_bytes") != 0:
        reasons.append("candidate unexpectedly uses dynamic shared memory")
    if candidate.get("local_bytes_per_thread") != 0:
        reasons.append("candidate runtime local bytes are nonzero")
    if candidate.get("active_ctas_per_sm") != 2:
        reasons.append("candidate must retain two CTAs per SM")
    if "PAIR_SCRATCH=true" not in str(paths.get("baseline", "")):
        reasons.append("baseline path is not the accepted pair-scratch path")
    if "reused from sm70_native_bm64_allp_micro.cu" not in str(
        paths.get("baseline", "")
    ):
        reasons.append("baseline is not declared as the existing BM32 reference")
    required_contract = {
        "warps": 8,
        "m16_owners": 4,
        "d16_fp32_output_fragments_per_warp": 8,
        "fp32_output_values_per_warp": 64,
        "physical_bn": 16,
        "logical_softmax_bn": 32,
        "value_payload_gprs_per_partner": 16,
        "q_shared_bytes": 32768,
        "reusable_kv_stage_bytes": 8192,
        "probability_bytes": 2048,
        "row_state_bytes": 768,
        "total_shared_bytes": K_CANDIDATE_SHARED_BYTES,
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            reasons.append(
                f"candidate contract {key}={contract.get(key)}, requires {expected}"
            )
    proof = str(contract.get("bn16_bitwise_incompatibility_proof", ""))
    if "max(N0,N1)" not in proof or "FP32 sum" not in proof:
        reasons.append("candidate does not declare the BN16-to-BN32 proof")
    barriers = contract.get("barriers") or {}
    if barriers.get("q_ready") != "4x bar.sync count=64":
        reasons.append("candidate owner-Q barriers do not have 64 participants")
    if barriers.get("stage_epochs") != "bar.sync id=5 count=256":
        reasons.append("candidate stage barriers do not have 256 participants")
    return reasons


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
            "hard_gates_passed": False,
            "reasons": [f"C++ JSON parse failure: {error}"],
        }
        result["stdout_tail"] = completed.stdout[-3000:]
        result["stderr_tail"] = completed.stderr[-3000:]
        return result

    correctness_reasons = _correctness_gate(payload, groups)
    resource_reasons = _runtime_resource_gate(payload)
    reasons = [
        *([] if completed.returncode == 0 else ["C++ benchmark returned nonzero"]),
        *correctness_reasons,
        *resource_reasons,
    ]
    result["cpp_json"] = payload
    result["gates"] = {
        "correctness_passed": not correctness_reasons,
        "resource_passed": not resource_reasons,
        "hard_gates_passed": completed.returncode == 0 and not reasons,
        "reasons": reasons,
    }
    if not smoke_only and payload.get("timing") is None:
        result["gates"]["hard_gates_passed"] = False
        result["gates"]["reasons"].append("timing was not reported after gates")
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-3000:]
    return result


def _write_payload(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def _case_state(args: argparse.Namespace, phase: str) -> dict[str, Any]:
    return _require_execution_device(args, phase)


def _preflight_payload(
    build_command: list[str],
    ptxas_baseline: dict[str, int] | None,
    ptxas_candidate: dict[str, int] | None,
    ptxas_log: str,
    sass: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "benchmark": "sm70_flashinfer_bm64_pipeline_micro",
        "mode": "build_preflight_rejected_before_gpu_launch",
        "build": {
            "source": str(_source_path()),
            "command": build_command,
            "ptxas": {
                "baseline": ptxas_baseline,
                "candidate": ptxas_candidate,
            },
            "ptxas_log": ptxas_log[-6000:],
        },
        "sass": sass,
        "gates": {"hard_gates_passed": False, "reasons": reasons},
    }


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
            _preflight_payload(
                build_command,
                ptxas_baseline,
                ptxas_candidate,
                ptxas_log,
                sass,
                build_reasons,
            ),
            args.json_out,
        )
        return 1

    smoke_cases: list[dict[str, Any]] = []
    gpu_states: list[dict[str, Any]] = []
    for pattern, nblocks in K_SMOKE_CASES:
        before = _case_state(args, f"before smoke {pattern}/N{nblocks}")
        case = _run_case(binary, args, pattern, nblocks, 1, True)
        after = _case_state(args, f"after smoke {pattern}/N{nblocks}")
        case["gpu_state_before"] = before
        case["gpu_state_after"] = after
        smoke_cases.append(case)
        gpu_states.extend((before, after))

    smoke_hard_passed = all(case["gates"]["hard_gates_passed"] for case in smoke_cases)
    if args.smoke or not smoke_hard_passed:
        payload = {
            "benchmark": "sm70_flashinfer_bm64_pipeline_micro",
            "mode": "resource_and_bitwise_smoke_no_timing",
            "target": "sm_70",
            "build": {
                "source": str(_source_path()),
                "command": build_command,
                "ptxas": {
                    "baseline": ptxas_baseline,
                    "candidate": ptxas_candidate,
                },
                "ptxas_log": ptxas_log[-6000:],
            },
            "sass": sass,
            "runtime": {
                "physical_gpu": args.physical_gpu,
                "child_cuda_environment": _child_cuda_configuration(args.physical_gpu),
                "selected_gpu_states": gpu_states,
                "required_clock_mhz": args.require_clock_mhz,
                "allow_busy_gpu": args.allow_busy_gpu,
            },
            "smoke_cases": smoke_cases,
            "gates": {
                "hard_gates_passed": smoke_hard_passed,
                "timing_withheld_until_resource_and_correctness_pass": True,
            },
        }
        _write_payload(payload, args.json_out)
        return 0 if smoke_hard_passed else 1

    timing_cases: list[dict[str, Any]] = []
    for pattern, nblocks in K_SMOKE_CASES:
        before = _case_state(args, f"before timing {pattern}/N{nblocks}")
        case = _run_case(binary, args, pattern, nblocks, args.groups, False)
        after = _case_state(args, f"after timing {pattern}/N{nblocks}")
        case["gpu_state_before"] = before
        case["gpu_state_after"] = after
        timing_cases.append(case)
        gpu_states.extend((before, after))

    timing_hard_passed = all(
        case["gates"]["hard_gates_passed"] for case in timing_cases
    )
    payload = {
        "benchmark": "sm70_flashinfer_bm64_pipeline_micro",
        "target": "sm_70",
        "matrix": [
            {"pattern": pattern, "nblocks": nblocks, "groups": args.groups}
            for pattern, nblocks in K_SMOKE_CASES
        ],
        "configuration": {
            "logical_m64_groups": args.groups,
            "baseline_ctas_per_logical_group": 2,
            "candidate_ctas_per_logical_group": 1,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "launches_per_sample": args.launches_per_sample,
        },
        "build": {
            "source": str(_source_path()),
            "command": build_command,
            "ptxas": {
                "baseline": ptxas_baseline,
                "candidate": ptxas_candidate,
            },
            "ptxas_log": ptxas_log[-6000:],
        },
        "sass": sass,
        "runtime": {
            "physical_gpu": args.physical_gpu,
            "child_cuda_environment": _child_cuda_configuration(args.physical_gpu),
            "selected_gpu_states": gpu_states,
            "required_clock_mhz": args.require_clock_mhz,
            "allow_busy_gpu": args.allow_busy_gpu,
        },
        "smoke_cases": smoke_cases,
        "timing_cases": timing_cases,
        "gates": {
            "smoke_hard_gates_passed": smoke_hard_passed,
            "timing_hard_gates_passed": timing_hard_passed,
            "hard_gates_passed": smoke_hard_passed and timing_hard_passed,
            "timing_started_after_smoke_pass": True,
        },
    }
    _write_payload(payload, args.json_out)
    return 0 if payload["gates"]["hard_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
