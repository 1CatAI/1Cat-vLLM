# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the SM70 native-WMMA BM32 QK panel candidate."""

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
K_MIN_GROUPS = 512
K_DEFAULT_GROUPS = 1024
K_MIN_ROUNDS = 100
K_REQUIRED_SPEEDUP_PCT = 2.0
K_EXPECTED_CANDIDATE_HMMA = 512
K_NCU_METRICS = (
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
    "smsp__inst_executed_pipe_tensor.sum",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--groups", type=int, default=K_DEFAULT_GROUPS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=K_MIN_ROUNDS)
    parser.add_argument("--launches-per-sample", type=int, default=8)
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
    if args.groups < K_MIN_GROUPS:
        raise ValueError(f"groups must be at least {K_MIN_GROUPS}.")
    if args.warmup < 20:
        raise ValueError("warmup must be at least 20.")
    if args.rounds < K_MIN_ROUNDS:
        raise ValueError("rounds must be at least 100.")
    if args.launches_per_sample < 2:
        raise ValueError("launches-per-sample must be at least 2.")


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
        raise RuntimeError("nvcc is required to build the SM70 BM32 microbenchmark.")
    source = (
        Path(__file__).resolve().parent
        / "csrc"
        / "sm70_native_bm32_qk_panel_micro.cu"
    )
    build_dir = (
        Path(tempfile.gettempdir())
        / f"vllm-sm70-native-bm32-qk-panel-micro-{os.getuid()}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_native_bm32_qk_panel_micro_sm70"
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


def _binary_command(binary: Path, args: argparse.Namespace) -> list[str]:
    return [
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


def _ldg_width_count(sass: str, width: int) -> int:
    return len(
        re.findall(rf"\bLDG(?:\.[A-Z0-9_]+)*\.{width}(?:\.|\b)", sass)
    )


def _inspect_sass(binary: Path) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return {"available": False, "reason": "cuobjdump was not found"}
    result = subprocess.run(
        [cuobjdump, "--dump-sass", str(binary)],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        return {
            "available": False,
            "returncode": result.returncode,
            "stderr": result.stderr[-2000:],
        }
    kernels: dict[str, Any] = {}
    for path, symbol in (
        ("baseline", "native_bm32_baseline_kernel"),
        ("candidate", "native_bm32_candidate_kernel"),
    ):
        section = _function_section(result.stdout, symbol)
        if section is None:
            kernels[path] = {"available": False, "symbol": symbol}
            continue
        kernels[path] = {
            "available": True,
            "symbol": symbol,
            "instructions": {
                "hmma": _instruction_count(section, "HMMA"),
                "ldg": _instruction_count(section, "LDG"),
                "ldg_64": _ldg_width_count(section, 64),
                "ldg_128": _ldg_width_count(section, 128),
                "lds": _instruction_count(section, "LDS"),
                "ldl": _instruction_count(section, "LDL"),
                "stl": _instruction_count(section, "STL"),
            },
        }
    return {"available": True, "binary": str(binary), "kernels": kernels}


def _resource_gate(payload: dict[str, Any], sass: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    resources = payload.get("resources", {})
    baseline = resources.get("baseline", {})
    candidate = resources.get("candidate", {})
    paths = payload.get("paths", {})
    if baseline.get("active_ctas_per_sm") != 2:
        reasons.append(
            "baseline active_ctas_per_sm="
            f"{baseline.get('active_ctas_per_sm')}, requires 2"
        )
    if baseline.get("resident_qk_warps") != 16:
        reasons.append(
            "baseline resident_qk_warps="
            f"{baseline.get('resident_qk_warps')}, requires 16"
        )
    if candidate.get("threads_per_cta") != 256:
        reasons.append(
            "candidate threads_per_cta="
            f"{candidate.get('threads_per_cta')}, requires 256"
        )
    if candidate.get("active_ctas_per_sm") != 4:
        reasons.append(
            "candidate active_ctas_per_sm="
            f"{candidate.get('active_ctas_per_sm')}, requires 4"
        )
    if candidate.get("resident_qk_warps") != 32:
        reasons.append(
            "candidate resident_qk_warps="
            f"{candidate.get('resident_qk_warps')}, requires 32"
        )
    if candidate.get("registers_per_thread", 65) > 64:
        reasons.append(
            "candidate REG="
            f"{candidate.get('registers_per_thread')}, requires <=64"
        )
    if candidate.get("local_bytes_per_thread") != 0:
        reasons.append(
            "candidate LOCAL="
            f"{candidate.get('local_bytes_per_thread')}, requires 0"
        )
    if paths.get("baseline_ctas_per_group") != 2:
        reasons.append(
            "baseline_ctas_per_group="
            f"{paths.get('baseline_ctas_per_group')}, requires 2"
        )
    if paths.get("candidate_ctas_per_group") != 1:
        reasons.append(
            "candidate_ctas_per_group="
            f"{paths.get('candidate_ctas_per_group')}, requires 1"
        )
    if paths.get("candidate_b_fragment_loads_per_k16") != 1:
        reasons.append(
            "candidate_b_fragment_loads_per_k16="
            f"{paths.get('candidate_b_fragment_loads_per_k16')}, requires 1"
        )
    if not sass.get("available"):
        reasons.append("SASS inspection unavailable")
        return False, reasons
    baseline_sass = sass.get("kernels", {}).get("baseline", {})
    candidate_sass = sass.get("kernels", {}).get("candidate", {})
    if not baseline_sass.get("available") or not candidate_sass.get("available"):
        reasons.append("baseline or candidate SASS function was not found")
        return False, reasons
    instructions = candidate_sass.get("instructions", {})
    if instructions.get("ldl", 0) or instructions.get("stl", 0):
        reasons.append(
            "candidate SASS has local instructions "
            f"LDL={instructions.get('ldl', 0)} STL={instructions.get('stl', 0)}"
        )
    if instructions.get("hmma", 0) < K_EXPECTED_CANDIDATE_HMMA:
        reasons.append(
            "candidate HMMA="
            f"{instructions.get('hmma', 0)}, requires at least "
            f"{K_EXPECTED_CANDIDATE_HMMA}"
        )
    if instructions.get("ldg", 0) == 0 or instructions.get("lds", 0) == 0:
        reasons.append("candidate SASS is missing LDG or LDS")
    return not reasons, reasons


def _number(value: str) -> float | None:
    normalized = value.replace(",", "").strip()
    try:
        return float(normalized)
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
    kernel_symbol = (
        "native_bm32_baseline_kernel"
        if profile_kernel == "baseline"
        else "native_bm32_candidate_kernel"
    )
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
        f"regex:.*{kernel_symbol}.*",
        "--launch-count",
        "1",
        str(binary),
        "--device",
        "0",
        "--groups",
        str(args.groups),
        "--profile-only",
        "--profile-kernel",
        profile_kernel,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = K_REQUIRED_VISIBLE_DEVICES
    try:
        result = subprocess.run(
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
            "stdout": (error.stdout or "")[-2000:],
            "stderr": (error.stderr or "")[-2000:],
        }
    metrics = _parse_ncu_metrics(result.stdout)
    active = metrics.get("smsp__warps_active.avg.per_cycle_active")
    eligible = metrics.get("smsp__warps_eligible.avg.per_cycle_active")
    no_eligible = max(active - eligible, 0.0) if active is not None and eligible is not None else None
    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "command": command,
        "returncode": result.returncode,
        "metrics": metrics,
        "derived": {
            "no_eligible_warps_per_cycle": no_eligible,
            "long_scoreboard_per_warp_active": metrics.get(
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active"
            ),
            "tensor_instructions": metrics.get(
                "smsp__inst_executed_pipe_tensor.sum"
            ),
        },
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
        "permission_error": "ERR_NVGPUCTRPERM" in (result.stdout + result.stderr),
    }


def _ncu_payload(
    binary: Path, args: argparse.Namespace, eligible_for_ncu: bool
) -> dict[str, Any]:
    if not eligible_for_ncu:
        return {
            "status": "skipped_before_full_gate",
            "metrics_requested_after_gate": list(K_NCU_METRICS),
        }
    if args.skip_ncu:
        return {
            "status": "skipped_by_user_after_full_gate",
            "metrics_requested": list(K_NCU_METRICS),
        }
    return {
        "status": "attempted_after_full_gate",
        "metrics_requested": list(K_NCU_METRICS),
        "baseline": _run_ncu_kernel(binary, args, "baseline"),
        "candidate": _run_ncu_kernel(binary, args, "candidate"),
    }


def main() -> int:
    args = _parse_args()
    _require_environment(args)
    clock_before_mhz = _verify_clock(args)
    binary, build_command, ptxas_log = _build_binary(args.verbose)
    command = _binary_command(binary, args)
    if args.verbose:
        print("run:", " ".join(command), file=sys.stderr)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout, end="", file=sys.stderr)
        return completed.returncode

    payload = json.loads(completed.stdout)
    exactness = payload.get("exactness", {})
    expected_words = args.groups * 32 * 128
    if exactness.get("word_count") != expected_words or not exactness.get(
        "full_32x128"
    ):
        raise RuntimeError("Probe did not compare the complete [groups, 32, 128] output.")
    sass = _inspect_sass(binary)
    resource_passed, resource_reasons = _resource_gate(payload, sass)
    candidate_speedup_pct = payload["timing"]["candidate_speedup_vs_baseline_pct"]
    wall_time_passed = candidate_speedup_pct >= K_REQUIRED_SPEEDUP_PCT
    correctness_passed = bool(exactness.get("bitwise_equal"))
    eligible_for_ncu = correctness_passed and resource_passed and wall_time_passed
    rejection_reasons: list[str] = []
    if not correctness_passed:
        rejection_reasons.append("candidate output differs from baseline uint32 words")
    if not wall_time_passed:
        rejection_reasons.append(
            f"candidate speedup {candidate_speedup_pct:.3f}% is below the 2.000% gate"
        )
    if resource_passed and not wall_time_passed:
        rejection_reasons.append(
            "BM32 serializes top and bottom native M16 accumulator chains in one "
            "CTA; B reuse did not overcome the doubled per-warp HMMA chain."
        )
    rejection_reasons.extend(resource_reasons)
    clock_after_timing_mhz = _verify_clock(args)
    ncu = _ncu_payload(binary, args, eligible_for_ncu)
    clock_after_ncu_mhz = _verify_clock(args)

    baseline_sass = sass.get("kernels", {}).get("baseline", {}).get(
        "instructions", {}
    )
    candidate_sass = sass.get("kernels", {}).get("candidate", {}).get(
        "instructions", {}
    )
    payload["build"] = {
        "command": build_command,
        "target": "sm_70",
        "ptxas_log": ptxas_log[-5000:],
    }
    payload["runtime"] = {
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": args.physical_gpu,
        "clock_mhz": {
            "required": args.clock_mhz,
            "before": clock_before_mhz,
            "after_timing": clock_after_timing_mhz,
            "after_ncu": clock_after_ncu_mhz,
        },
    }
    payload["sass"] = sass
    payload["candidate_b_reuse"] = {
        "native_b_fragment_loads_per_k16": {
            "baseline_two_ctas": 2,
            "candidate_one_cta": 1,
            "ratio": 0.5,
        },
        "static_instruction_counts": {
            "baseline_one_cta": baseline_sass,
            "baseline_two_ctas_aggregate": {
                key: value * 2 for key, value in baseline_sass.items()
            },
            "candidate_one_cta": candidate_sass,
        },
        "candidate_mma_order": "top then bottom for each K16",
    }
    payload["gates"] = {
        "correctness_passed": correctness_passed,
        "wall_time_passed": wall_time_passed,
        "resource_passed": resource_passed,
        "eligible_for_ncu": eligible_for_ncu,
        "integration_allowed": eligible_for_ncu,
        "decision": (
            "eligible_for_minimal_ncu_before_any_main_kernel_integration"
            if eligible_for_ncu
            else "rejected_do_not_integrate"
        ),
        "rejection_reasons": rejection_reasons,
        "required_candidate_speedup_pct": K_REQUIRED_SPEEDUP_PCT,
    }
    payload["ncu"] = ncu
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
