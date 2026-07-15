# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark SM70 native and raw-HMMA D256 QK panel candidates."""

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
K_SCALAR_RAW_LDG_REFERENCE = 261
K_EXPECTED_RAW_B_LDG64 = 64
K_EXPECTED_RAW_K16_B_LDG128 = 32
K_EXPECTED_REUSE_B_HMMA = 512
K_K4_VECTOR_LDG_REFERENCE = 69
K_CANDIDATE_VARIANTS = (
    "raw_k4_vector",
    "raw_k16_stage",
    "raw_k16_double",
    "raw_m16n256_reuse_b",
)
K_CANDIDATE_SYMBOLS = {
    "raw_k4_vector": "qk_panel_raw_k4_vector_kernel",
    "raw_k16_stage": "qk_panel_raw_k16_stage_kernel",
    "raw_k16_double": "qk_panel_raw_k16_double_kernel",
    "raw_m16n256_reuse_b": "qk_panel_raw_m16n256_reuse_b_kernel",
}
K_BM8_STRUCTURAL_REJECTION = (
    "BM8 splits M16 into two independent M8 CTAs, so each M half rereads K "
    "and doubles B bytes per useful output."
)
K_M16_REUSE_B_STRUCTURAL_REJECTION = (
    "B reuse removes the M-half duplicate load, but each warp serializes top "
    "and bottom m8n8k4 accumulator chains while the grid has half as many CTAs "
    "as the baseline."
)
K_NCU_METRICS = (
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
    "smsp__inst_executed_pipe_tensor.sum",
)
K_SCALAR_PREVIOUS_ARTIFACT = {
    "status": "historical_reference_not_remeasured_in_multivariant_run",
    "implementation": "four __ldg(__half) loads plus pack_half2 per token/K4",
    "loads_per_token_k4": 4,
    "measurement": {
        "physical_gpu": 2,
        "clock_mhz": 1200,
        "groups": 1024,
        "warmup_pairs": 20,
        "rounds": 100,
        "launches_per_sample": 8,
        "timing": {
            "baseline_median_us": 212.479994,
            "raw_median_us": 1781.63195,
            "raw_speedup_vs_baseline_pct": -738.493976,
        },
        "sass": {"hmma": 256, "ldg": 261, "lds": 32, "ldl": 0, "stl": 0},
        "resources": {
            "registers_per_thread": 64,
            "local_bytes_per_thread": 0,
            "active_ctas_per_sm": 4,
            "resident_qk_warps": 32,
        },
    },
}


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


def _build_binary(verbose: bool) -> tuple[Path, list[str], str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required to build the SM70 panel microbenchmark.")
    source = (
        Path(__file__).resolve().parent
        / "csrc"
        / "sm70_raw_hmma_qk_panel_micro.cu"
    )
    build_dir = (
        Path(tempfile.gettempdir())
        / f"vllm-sm70-raw-hmma-qk-panel-micro-{os.getuid()}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_raw_hmma_qk_panel_micro_sm70"
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


def _binary_command(
    binary: Path, args: argparse.Namespace, variant: str
) -> list[str]:
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
        "--raw-variant",
        variant,
    ]


def _run_variant(
    binary: Path, args: argparse.Namespace, variant: str
) -> tuple[dict[str, Any], list[str]]:
    command = _binary_command(binary, args, variant)
    if args.verbose:
        print("run:", " ".join(command), file=sys.stderr)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(
            f"{variant} exited with {completed.returncode}: "
            f"{completed.stdout[-2000:]}{completed.stderr[-2000:]}"
        )
    payload = json.loads(completed.stdout)
    if payload.get("paths", {}).get("raw_variant") != variant:
        raise RuntimeError(f"Probe reported the wrong raw variant for {variant}.")
    expected_words = args.groups * 16 * 256
    if payload.get("exactness", {}).get("word_count") != expected_words:
        raise RuntimeError("Probe did not compare the complete [groups, 16, 256] output.")
    return payload, command


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
    symbols = {"baseline": "qk_panel_baseline_kernel", **K_CANDIDATE_SYMBOLS}
    kernels: dict[str, Any] = {}
    for path, symbol in symbols.items():
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


def _resource_gate(
    payload: dict[str, Any], sass: dict[str, Any], variant: str
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    resources = payload.get("resources", {})
    baseline = resources.get("baseline", {})
    raw = resources.get("raw", {})
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
    if raw.get("active_ctas_per_sm") != 4:
        reasons.append(
            f"{variant} active_ctas_per_sm={raw.get('active_ctas_per_sm')}, requires 4"
        )
    if raw.get("resident_qk_warps") != 32:
        reasons.append(
            f"{variant} resident_qk_warps={raw.get('resident_qk_warps')}, requires 32"
        )
    if raw.get("registers_per_thread", 65) > 64:
        reasons.append(
            f"{variant} REG={raw.get('registers_per_thread')}, requires <=64"
        )
    if raw.get("local_bytes_per_thread") != 0:
        reasons.append(
            f"{variant} LOCAL={raw.get('local_bytes_per_thread')}, requires 0"
        )
    if raw.get("threads_per_cta") != 256:
        reasons.append(
            f"{variant} threads_per_cta={raw.get('threads_per_cta')}, requires 256"
        )
    if variant == "raw_m16n256_reuse_b":
        paths = payload.get("paths", {})
        if paths.get("baseline_ctas_per_group") != 2:
            reasons.append(
                "reuse-B baseline_ctas_per_group="
                f"{paths.get('baseline_ctas_per_group')}, requires 2"
            )
        if paths.get("raw_ctas_per_group") != 1:
            reasons.append(
                "reuse-B raw_ctas_per_group="
                f"{paths.get('raw_ctas_per_group')}, requires 1"
            )
    if not sass.get("available"):
        reasons.append("SASS inspection unavailable")
        return False, reasons
    baseline_sass = sass.get("kernels", {}).get("baseline", {})
    raw_sass = sass.get("kernels", {}).get(variant, {})
    if not baseline_sass.get("available") or not raw_sass.get("available"):
        reasons.append(f"baseline or {variant} SASS function was not found")
        return False, reasons
    instructions = raw_sass.get("instructions", {})
    if instructions.get("ldl", 0) or instructions.get("stl", 0):
        reasons.append(
            f"{variant} SASS has local instructions LDL={instructions.get('ldl', 0)} "
            f"STL={instructions.get('stl', 0)}"
        )
    if instructions.get("hmma", 0) == 0:
        reasons.append(f"{variant} SASS has no HMMA instruction")
    if instructions.get("ldg", 0) == 0:
        reasons.append(f"{variant} SASS has no LDG instruction")
    if instructions.get("lds", 0) == 0:
        reasons.append(f"{variant} SASS has no LDS instruction")
    return not reasons, reasons


def _load_gate(sass: dict[str, Any], variant: str) -> tuple[bool, list[str]]:
    if not sass.get("available"):
        return False, ["SASS inspection unavailable for B-load proof"]
    raw_sass = sass.get("kernels", {}).get(variant, {})
    if not raw_sass.get("available"):
        return False, [f"{variant} SASS function was not found for B-load proof"]
    instructions = raw_sass.get("instructions", {})
    reasons: list[str] = []
    if variant == "raw_k4_vector":
        if instructions.get("ldg_64", 0) < K_EXPECTED_RAW_B_LDG64:
            reasons.append(
                "raw_k4_vector LDG.64="
                f"{instructions.get('ldg_64', 0)}, requires at least "
                f"{K_EXPECTED_RAW_B_LDG64} for one vector B load per K4"
            )
        if instructions.get("ldg", K_SCALAR_RAW_LDG_REFERENCE) >= K_SCALAR_RAW_LDG_REFERENCE:
            reasons.append(
                "raw_k4_vector total LDG="
                f"{instructions.get('ldg')}, must be below scalar reference "
                f"{K_SCALAR_RAW_LDG_REFERENCE}"
            )
        return not reasons, reasons
    if instructions.get("ldg_128", 0) < K_EXPECTED_RAW_K16_B_LDG128:
        reasons.append(
            f"{variant} LDG.128={instructions.get('ldg_128', 0)}, requires at "
            f"least {K_EXPECTED_RAW_K16_B_LDG128} for two B loads per K16"
        )
    if instructions.get("ldg", K_K4_VECTOR_LDG_REFERENCE) >= K_K4_VECTOR_LDG_REFERENCE:
        reasons.append(
            f"{variant} total LDG={instructions.get('ldg')}, must be below "
            f"raw_k4_vector reference {K_K4_VECTOR_LDG_REFERENCE}"
        )
    if instructions.get("ldg_64", 0):
        reasons.append(
            f"{variant} has unexpected LDG.64={instructions.get('ldg_64')}"
        )
    if variant == "raw_m16n256_reuse_b" and instructions.get("hmma", 0) < K_EXPECTED_REUSE_B_HMMA:
        reasons.append(
            f"reuse-B HMMA={instructions.get('hmma', 0)}, requires at least "
            f"{K_EXPECTED_REUSE_B_HMMA}"
        )
    return not reasons, reasons


def _candidate_artifact(
    payload: dict[str, Any], sass: dict[str, Any], variant: str, command: list[str]
) -> dict[str, Any]:
    exactness = payload["exactness"]
    timing = payload["timing"]
    raw_speedup_pct = timing["raw_speedup_vs_baseline_pct"]
    correctness_passed = bool(exactness.get("bitwise_equal"))
    resource_passed, resource_reasons = _resource_gate(payload, sass, variant)
    load_passed, load_reasons = _load_gate(sass, variant)
    wall_time_passed = raw_speedup_pct >= K_REQUIRED_SPEEDUP_PCT
    eligible_for_ncu = (
        correctness_passed and resource_passed and load_passed and wall_time_passed
    )
    rejection_reasons: list[str] = []
    if not correctness_passed:
        rejection_reasons.append("candidate output differs from baseline uint32 words")
    if not wall_time_passed:
        rejection_reasons.append(
            f"raw speedup {raw_speedup_pct:.3f}% is below the 2.000% gate"
        )
    structural_rejection = None
    if load_passed and not wall_time_passed:
        structural_rejection = (
            K_M16_REUSE_B_STRUCTURAL_REJECTION
            if variant == "raw_m16n256_reuse_b"
            else K_BM8_STRUCTURAL_REJECTION
        )
        rejection_reasons.append(structural_rejection)
    rejection_reasons.extend(resource_reasons)
    rejection_reasons.extend(load_reasons)
    instructions = sass.get("kernels", {}).get(variant, {}).get(
        "instructions", {}
    )
    if variant == "raw_k4_vector":
        load_artifact = {
            "implementation": "aligned inline PTX ld.global.v2.u32",
            "loads_per_token_k4": 1,
            "bytes_per_load": 8,
            "expected_b_ldg_64": K_EXPECTED_RAW_B_LDG64,
            "observed_total_ldg_64": instructions.get("ldg_64", 0),
            "observed_total_ldg": instructions.get("ldg", 0),
        }
    elif variant == "raw_m16n256_reuse_b":
        raw_ctas_per_group = payload.get("paths", {}).get("raw_ctas_per_group")
        baseline_ctas_per_group = payload.get("paths", {}).get(
            "baseline_ctas_per_group"
        )
        load_artifact = {
            "implementation": "aligned inline PTX ld.global.v4.u32",
            "loads_per_token_k16": 2,
            "bytes_per_load": 16,
            "b_reused_across_m8_halves": True,
            "expected_b_main_loop_ldg_128": K_EXPECTED_RAW_K16_B_LDG128,
            "observed_total_ldg_128": instructions.get("ldg_128", 0),
            "estimated_non_b_ldg_128": max(
                instructions.get("ldg_128", 0) - K_EXPECTED_RAW_K16_B_LDG128,
                0,
            ),
            "expected_hmma": K_EXPECTED_REUSE_B_HMMA,
            "observed_hmma": instructions.get("hmma", 0),
            "shared_q_stages_per_cta": 1,
            "a_loads": "top then bottom ld.shared.v2.u32 per K4",
            "baseline_ctas_per_group": baseline_ctas_per_group,
            "raw_ctas_per_group": raw_ctas_per_group,
            "grid_cta_ratio_vs_baseline": (
                raw_ctas_per_group / baseline_ctas_per_group
                if baseline_ctas_per_group
                else None
            ),
        }
    else:
        load_artifact = {
            "implementation": "aligned inline PTX ld.global.v4.u32",
            "loads_per_token_k16": 2,
            "bytes_per_load": 16,
            "expected_b_main_loop_ldg_128": K_EXPECTED_RAW_K16_B_LDG128,
            "observed_total_ldg_128": instructions.get("ldg_128", 0),
            "estimated_non_b_ldg_128": max(
                instructions.get("ldg_128", 0) - K_EXPECTED_RAW_K16_B_LDG128,
                0,
            ),
            "observed_total_ldg": instructions.get("ldg", 0),
            "a_loads_per_k16": "two ld.shared.v4.u32",
        }
        if variant == "raw_k16_double":
            load_artifact["prefetch_schedule"] = (
                "next K16 B halves follow current K4 0 and K4 1 HMMA"
            )
    return {
        "status": "measured",
        "command": command,
        "paths": payload["paths"],
        "exactness": exactness,
        "timing": timing,
        "pairs": payload["pairs"],
        "resources": payload["resources"]["raw"],
        "sass": instructions,
        "load_artifact": load_artifact,
        "gates": {
            "correctness_passed": correctness_passed,
            "wall_time_passed": wall_time_passed,
            "resource_passed": resource_passed,
            "load_passed": load_passed,
            "structural_rejection": structural_rejection,
            "eligible_for_ncu": eligible_for_ncu,
            "integration_allowed": eligible_for_ncu,
            "rejection_reasons": rejection_reasons,
            "required_raw_speedup_pct": K_REQUIRED_SPEEDUP_PCT,
        },
    }


def _stage_resources_allow_double(artifact: dict[str, Any]) -> bool:
    resources = artifact.get("resources", {})
    return (
        artifact.get("gates", {}).get("correctness_passed", False)
        and resources.get("registers_per_thread", 65) <= 64
        and resources.get("local_bytes_per_thread") == 0
        and resources.get("active_ctas_per_sm") == 4
        and resources.get("resident_qk_warps") == 32
    )


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
    binary: Path, args: argparse.Namespace, candidate: str, profile_kernel: str
) -> dict[str, Any]:
    ncu = shutil.which("ncu")
    if ncu is None:
        return {"status": "unavailable", "reason": "ncu was not found"}
    kernel_symbol = (
        "qk_panel_baseline_kernel"
        if profile_kernel == "baseline"
        else K_CANDIDATE_SYMBOLS[candidate]
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
        "--raw-variant",
        candidate,
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
        "permission_error": "ERR_NVGPUCTRPERM" in result.stderr,
    }


def _ncu_payload(
    binary: Path, args: argparse.Namespace, eligible_candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    if not eligible_candidates:
        return {
            "status": "skipped_before_full_gate",
            "metrics_requested_after_gate": list(K_NCU_METRICS),
        }
    candidate = max(
        eligible_candidates,
        key=lambda item: item["timing"]["raw_speedup_vs_baseline_pct"],
    )
    candidate_name = candidate["name"]
    if args.skip_ncu:
        return {
            "status": "skipped_by_user_after_full_gate",
            "metrics_requested": list(K_NCU_METRICS),
            "candidate": candidate_name,
        }
    return {
        "status": "attempted_after_full_gate",
        "metrics_requested": list(K_NCU_METRICS),
        "candidate": candidate_name,
        "baseline": _run_ncu_kernel(binary, args, candidate_name, "baseline"),
        "raw": _run_ncu_kernel(binary, args, candidate_name, "raw"),
    }


def main() -> int:
    args = _parse_args()
    _require_environment(args)
    clock_before_mhz = _verify_clock(args)
    binary, build_command, ptxas_log = _build_binary(args.verbose)
    sass = _inspect_sass(binary)

    raw_payloads: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    for variant in ("raw_k4_vector", "raw_k16_stage"):
        raw_payload, command = _run_variant(binary, args, variant)
        raw_payloads[variant] = raw_payload
        candidates[variant] = _candidate_artifact(
            raw_payload, sass, variant, command
        )

    stage = candidates["raw_k16_stage"]
    if _stage_resources_allow_double(stage):
        raw_payload, command = _run_variant(binary, args, "raw_k16_double")
        raw_payloads["raw_k16_double"] = raw_payload
        candidates["raw_k16_double"] = _candidate_artifact(
            raw_payload, sass, "raw_k16_double", command
        )
    else:
        candidates["raw_k16_double"] = {
            "status": "skipped_stage_resource_gate",
            "requires": "raw_k16_stage REG<=64, LOCAL=0, 4 CTA/SM, 32 QK warps, and bitwise equality",
            "stage_resources": stage.get("resources", {}),
            "stage_gates": stage.get("gates", {}),
        }

    raw_payload, command = _run_variant(binary, args, "raw_m16n256_reuse_b")
    raw_payloads["raw_m16n256_reuse_b"] = raw_payload
    candidates["raw_m16n256_reuse_b"] = _candidate_artifact(
        raw_payload, sass, "raw_m16n256_reuse_b", command
    )

    clock_after_timing_mhz = _verify_clock(args)
    eligible_candidates = [
        {"name": name, **artifact}
        for name, artifact in candidates.items()
        if artifact.get("status") == "measured"
        and artifact.get("gates", {}).get("eligible_for_ncu")
    ]
    ncu = _ncu_payload(binary, args, eligible_candidates)
    clock_after_ncu_mhz = _verify_clock(args)

    reference = raw_payloads["raw_k4_vector"]
    measured_candidates = {
        name: artifact
        for name, artifact in candidates.items()
        if artifact.get("status") == "measured"
    }
    raw_b_load_artifacts: dict[str, Any] = {
        "scalar_previous": K_SCALAR_PREVIOUS_ARTIFACT,
    }
    for name, artifact in measured_candidates.items():
        raw_b_load_artifacts[name] = {
            "load_artifact": artifact["load_artifact"],
            "sass": artifact["sass"],
            "resources": artifact["resources"],
            "timing": artifact["timing"],
            "exactness": artifact["exactness"],
        }

    payload = {
        "device": reference["device"],
        "target": "sm_70",
        "shape": reference["shape"],
        "comparison_contract": {
            "same_query_key_input": True,
            "input_seed": "0x6d2b79f5 per variant process",
            "output": "[groups, M16, N256]",
            "compared_output_words_per_variant": args.groups * 16 * 256,
            "full_16x256": True,
            "comparison": "baseline versus each raw candidate",
            "historical_candidate_artifacts_retained": True,
        },
        "paths": {
            "baseline": reference["paths"]["baseline"],
            "raw_geometry": reference["paths"]["raw"],
            "shared_q": reference["paths"]["shared_q"],
            "raw_k4_order": reference["paths"]["raw_k4_order"],
        },
        "measurement": reference["measurement"],
        "baseline": {
            "resources": reference["resources"]["baseline"],
            "sass": sass.get("kernels", {}).get("baseline", {}).get(
                "instructions", {}
            ),
            "timing_per_candidate": {
                name: artifact["timing"]["baseline"]
                for name, artifact in measured_candidates.items()
            },
        },
        "candidates": candidates,
        "raw_b_load_artifacts": raw_b_load_artifacts,
        "build": {
            "command": build_command,
            "target": "sm_70",
            "ptxas_log": ptxas_log[-6000:],
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
        "gates": {
            "eligible_candidates": [item["name"] for item in eligible_candidates],
            "integration_allowed": bool(eligible_candidates),
            "decision": (
                "eligible_for_minimal_ncu_before_any_main_kernel_integration"
                if eligible_candidates
                else "all_candidates_rejected_do_not_run_ncu_or_integrate"
            ),
            "required_raw_speedup_pct": K_REQUIRED_SPEEDUP_PCT,
        },
        "ncu": ncu,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
