# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark exact SM70 raw-HMMA PV D256 panel candidates."""

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
K_NCU_METRICS = (
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
    "smsp__inst_executed_pipe_tensor.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
)
K_LEGACY_SPLIT_VARIANTS = ("raw_single", "raw_pipelined")
K_REUSE_V_VARIANTS = (
    "raw_m16d256_reuse_v",
    "raw_m16d256_reuse_v_k16_double",
)
K_RAW_VARIANTS = K_LEGACY_SPLIT_VARIANTS + K_REUSE_V_VARIANTS
K_VARIANTS = ("baseline",) + K_RAW_VARIANTS
K_LOADER_CLASS_BY_VARIANT = {
    "raw_single": "raw_vector",
    "raw_pipelined": "k16_stage_double",
    "raw_m16d256_reuse_v": "raw_m16_reuse_v",
    "raw_m16d256_reuse_v_k16_double": "raw_m16_reuse_v_k16_double",
}
K_CTAS_PER_GROUP = {
    "baseline": 1,
    "raw_single": 2,
    "raw_pipelined": 2,
    "raw_m16d256_reuse_v": 1,
    "raw_m16d256_reuse_v_k16_double": 1,
}
K_EXACTNESS_KEY_BY_VARIANT = {
    "raw_single": "baseline_vs_raw_single",
    "raw_pipelined": "baseline_vs_raw_pipelined",
    "raw_m16d256_reuse_v": "baseline_vs_raw_m16d256_reuse_v",
    "raw_m16d256_reuse_v_k16_double": ("baseline_vs_raw_m16d256_reuse_v_k16_double"),
}
K_SASS_EXPECTATIONS = {
    "raw_single": {"hmma": 32, "ldg_e64": 8, "lds_64": 8},
    "raw_pipelined": {"hmma": 32, "ldg_e64": 8, "lds_64": 8},
    "raw_m16d256_reuse_v": {"hmma": 64, "ldg_e64": 8, "lds_64": 16},
    "raw_m16d256_reuse_v_k16_double": {
        "hmma": 64,
        "ldg_e64": 8,
        "lds_64": 16,
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
        raise RuntimeError("nvcc is required to build the SM70 PV microbenchmark.")
    source = (
        Path(__file__).resolve().parent / "csrc" / "sm70_raw_hmma_pv_panel_micro.cu"
    )
    build_dir = (
        Path(tempfile.gettempdir()) / f"vllm-sm70-raw-hmma-pv-panel-micro-{os.getuid()}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_raw_hmma_pv_panel_micro_sm70"
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


def _sized_instruction_count(sass: str, mnemonic: str, width: int) -> int:
    pattern = rf"\b{re.escape(mnemonic)}(?:\.[A-Z0-9]+)*\.{width}(?:\.[A-Z0-9]+)*\b"
    return len(re.findall(pattern, sass))


def _hmma_call_structure(sass: str) -> dict[str, Any]:
    pattern = re.compile(
        r"\bHMMA(?:\.[A-Z0-9]+)*\.STEP([0-3])\s+"
        r"(R\d+),\s+(R\d+)[^,]*,\s+(R\d+)"
    )
    instructions = [
        (int(step), destination, operand_a, operand_b)
        for step, destination, operand_a, operand_b in pattern.findall(sass)
    ]
    valid_step_groups = len(instructions) % 4 == 0
    calls: list[dict[str, Any]] = []
    for index in range(0, len(instructions), 4):
        group = instructions[index : index + 4]
        if len(group) != 4 or [entry[0] for entry in group] != [0, 1, 2, 3]:
            valid_step_groups = False
            continue
        b_registers = {entry[3] for entry in group}
        calls.append(
            {
                "destination_registers": tuple(entry[1] for entry in group),
                "b_register": next(iter(b_registers))
                if len(b_registers) == 1
                else None,
            }
        )
    pair_count = len(calls) // 2
    top_signature = calls[0]["destination_registers"] if calls else None
    bottom_signature = calls[1]["destination_registers"] if len(calls) > 1 else None
    alternating_accumulator_chains = bool(calls) and all(
        call["destination_registers"]
        == (top_signature if index % 2 == 0 else bottom_signature)
        for index, call in enumerate(calls)
    )
    reused_b_operand_per_pair = len(calls) % 2 == 0 and all(
        calls[index]["b_register"] is not None
        and calls[index]["b_register"] == calls[index + 1]["b_register"]
        for index in range(0, len(calls), 2)
    )
    return {
        "ptx_m8n8k4_calls": len(calls),
        "top_bottom_pairs": pair_count,
        "valid_step_groups": valid_step_groups,
        "alternating_accumulator_chains": alternating_accumulator_chains,
        "reused_b_operand_per_pair": reused_b_operand_per_pair,
    }


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
    for variant in K_VARIANTS:
        symbol = f"pv_panel_{variant}_kernel"
        section = _function_section(result.stdout, symbol)
        if section is None:
            kernels[variant] = {"available": False, "symbol": symbol}
            continue
        instructions = {
            "hmma": _instruction_count(section, "HMMA"),
            "ldg": _instruction_count(section, "LDG"),
            "lds": _instruction_count(section, "LDS"),
            "ldl": _instruction_count(section, "LDL"),
            "stl": _instruction_count(section, "STL"),
            "ldg_e64": _sized_instruction_count(section, "LDG", 64),
            "lds_64": _sized_instruction_count(section, "LDS", 64),
            "ldg_e16": _sized_instruction_count(section, "LDG", 16),
            "lds_16": _sized_instruction_count(section, "LDS", 16),
        }
        ctas_per_group = K_CTAS_PER_GROUP[variant]
        kernels[variant] = {
            "available": True,
            "symbol": symbol,
            "loader_class": K_LOADER_CLASS_BY_VARIANT.get(variant, "native"),
            "ctas_per_group": ctas_per_group,
            "instructions": instructions,
            "hmma_call_structure": _hmma_call_structure(section),
            "instructions_per_group": {
                "hmma": instructions["hmma"] * ctas_per_group,
                "ldg_e64": instructions["ldg_e64"] * ctas_per_group,
                "lds_64": instructions["lds_64"] * ctas_per_group,
            },
        }
    return {
        "available": True,
        "binary": str(binary),
        "kernels": kernels,
        "loader_classes": {
            "raw_scalar": {
                "present": False,
                "structural_gate": "not_used",
                "reason": "No scalar raw PV kernel is built in this binary.",
            },
            "raw_vector": {
                "variant": "raw_single",
                "criterion": "8 LDG.E.64 V and 8 LDS.64 P instructions; no "
                "scalar half operand loads",
            },
            "k16_stage_double": {
                "variant": "raw_pipelined",
                "criterion": "two named K16 banks, each comprising four K4 "
                "vector P/V operands before its four canonical HMMAs",
            },
            "raw_m16_reuse_v": {
                "variant": "raw_m16d256_reuse_v",
                "criterion": "1 CTA/group; 8 LDG.E.64 V operands feed 16 raw "
                "top-then-bottom m8n8k4 calls",
            },
            "raw_m16_reuse_v_k16_double": {
                "variant": "raw_m16d256_reuse_v_k16_double",
                "criterion": "same M16 V reuse with complete K16 operand banks",
            },
        },
    }


def _resource_reasons(
    payload: dict[str, Any], sass: dict[str, Any], variant: str
) -> list[str]:
    reasons: list[str] = []
    resources = payload.get("resources", {})
    baseline = resources.get("baseline", {})
    candidate = resources.get(variant, {})
    topology = payload.get("launch_topology", {})
    baseline_topology = topology.get("baseline", {})
    candidate_topology = topology.get(variant, {})
    if baseline.get("active_ctas_per_sm") != 2:
        reasons.append(
            "baseline active_ctas_per_sm="
            f"{baseline.get('active_ctas_per_sm')}, requires 2"
        )
    if baseline.get("resident_active_warps") != 32:
        reasons.append(
            "baseline resident_active_warps="
            f"{baseline.get('resident_active_warps')}, requires 32"
        )
    if baseline_topology.get("ctas_per_group") != 1:
        reasons.append(
            "baseline ctas_per_group="
            f"{baseline_topology.get('ctas_per_group')}, requires 1"
        )
    if baseline_topology.get("threads_per_cta") != 512:
        reasons.append(
            "baseline threads_per_cta="
            f"{baseline_topology.get('threads_per_cta')}, requires 512"
        )
    if candidate.get("active_ctas_per_sm") != 4:
        reasons.append(
            f"{variant} active_ctas_per_sm="
            f"{candidate.get('active_ctas_per_sm')}, requires 4"
        )
    if candidate.get("resident_active_warps") != 32:
        reasons.append(
            f"{variant} resident_active_warps="
            f"{candidate.get('resident_active_warps')}, requires 32"
        )
    expected_ctas_per_group = K_CTAS_PER_GROUP[variant]
    if candidate_topology.get("ctas_per_group") != expected_ctas_per_group:
        reasons.append(
            f"{variant} ctas_per_group="
            f"{candidate_topology.get('ctas_per_group')}, requires "
            f"{expected_ctas_per_group}"
        )
    if candidate_topology.get("threads_per_cta") != 256:
        reasons.append(
            f"{variant} threads_per_cta="
            f"{candidate_topology.get('threads_per_cta')}, requires 256"
        )
    if (
        candidate.get("registers_per_thread") is None
        or candidate.get("registers_per_thread", 65) > 64
    ):
        reasons.append(
            f"{variant} REG={candidate.get('registers_per_thread')}, requires <=64"
        )
    if candidate.get("local_bytes_per_thread") != 0:
        reasons.append(
            f"{variant} LOCAL={candidate.get('local_bytes_per_thread')}, requires 0"
        )
    if not sass.get("available"):
        reasons.append("SASS inspection unavailable")
        return reasons
    baseline_sass = sass.get("kernels", {}).get("baseline", {})
    candidate_sass = sass.get("kernels", {}).get(variant, {})
    if not baseline_sass.get("available") or not candidate_sass.get("available"):
        reasons.append("baseline or candidate SASS function was not found")
        return reasons
    instructions = candidate_sass.get("instructions", {})
    instructions_per_group = candidate_sass.get("instructions_per_group", {})
    hmma_structure = candidate_sass.get("hmma_call_structure", {})
    expected_instructions = K_SASS_EXPECTATIONS[variant]
    expected_loader_class = K_LOADER_CLASS_BY_VARIANT[variant]
    if candidate_sass.get("loader_class") != expected_loader_class:
        reasons.append(
            f"{variant} SASS loader_class={candidate_sass.get('loader_class')}, "
            f"requires {expected_loader_class}"
        )
    if instructions.get("ldl", 0) or instructions.get("stl", 0):
        reasons.append(
            f"{variant} SASS has local instructions LDL={instructions.get('ldl', 0)} "
            f"STL={instructions.get('stl', 0)}"
        )
    for mnemonic in ("hmma", "ldg", "lds"):
        if instructions.get(mnemonic, 0) == 0:
            reasons.append(f"{variant} SASS has no {mnemonic.upper()} instruction")
    for mnemonic, expected_count in expected_instructions.items():
        if instructions.get(mnemonic) != expected_count:
            reasons.append(
                f"{variant} {mnemonic} count={instructions.get(mnemonic)}, "
                f"requires {expected_count}"
            )
    if candidate_sass.get("ctas_per_group") != expected_ctas_per_group:
        reasons.append(
            f"{variant} SASS ctas_per_group="
            f"{candidate_sass.get('ctas_per_group')}, requires "
            f"{expected_ctas_per_group}"
        )
    if variant in K_REUSE_V_VARIANTS:
        if instructions_per_group.get("ldg_e64") != 8:
            reasons.append(
                f"{variant} LDG.E.64/group="
                f"{instructions_per_group.get('ldg_e64')}, requires 8"
            )
        if instructions_per_group.get("hmma") != 64:
            reasons.append(
                f"{variant} HMMA/group={instructions_per_group.get('hmma')}, "
                "requires 64"
            )
        if hmma_structure.get("ptx_m8n8k4_calls") != 16:
            reasons.append(
                f"{variant} raw HMMA calls="
                f"{hmma_structure.get('ptx_m8n8k4_calls')}, requires 16"
            )
        if hmma_structure.get("top_bottom_pairs") != 8:
            reasons.append(
                f"{variant} top/bottom HMMA pairs="
                f"{hmma_structure.get('top_bottom_pairs')}, requires 8"
            )
        if not hmma_structure.get("valid_step_groups"):
            reasons.append(f"{variant} has an invalid HMMA STEP0..3 grouping")
        if not hmma_structure.get("alternating_accumulator_chains"):
            reasons.append(
                f"{variant} does not alternate top/bottom accumulator chains"
            )
        if not hmma_structure.get("reused_b_operand_per_pair"):
            reasons.append(f"{variant} does not reuse one B/V operand per HMMA pair")
        if candidate_topology.get("ctas_per_group") != baseline_topology.get(
            "ctas_per_group"
        ):
            reasons.append(f"{variant} CTA count does not match baseline CTA count")
    if instructions.get("ldg_e16", 0) or instructions.get("lds_16", 0):
        reasons.append(
            f"{variant} has scalar half operand loads LDG.16="
            f"{instructions.get('ldg_e16', 0)} LDS.16="
            f"{instructions.get('lds_16', 0)}"
        )
    return reasons


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


def _run_ncu_variant(
    binary: Path, args: argparse.Namespace, variant: str
) -> dict[str, Any]:
    ncu = shutil.which("ncu")
    if ncu is None:
        return {"status": "unavailable", "reason": "ncu was not found"}
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
        f"regex:.*pv_panel_{variant}_kernel.*",
        "--launch-count",
        "1",
        str(binary),
        "--device",
        "0",
        "--groups",
        str(args.groups),
        "--profile-only",
        "--profile-kernel",
        variant,
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
    no_eligible = (
        max(active - eligible, 0.0)
        if active is not None and eligible is not None
        else None
    )
    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "command": command,
        "returncode": result.returncode,
        "metrics": metrics,
        "derived": {
            "eligible_warps_per_cycle": eligible,
            "no_eligible_warps_per_cycle": no_eligible,
            "long_scoreboard_per_warp_active": metrics.get(
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active"
            ),
            "tensor_instructions": metrics.get("smsp__inst_executed_pipe_tensor.sum"),
            "l1_global_load_sectors": metrics.get(
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
            ),
            "l1_global_store_sectors": metrics.get(
                "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
            ),
            "l2_read_sectors": metrics.get("lts__t_sectors_op_read.sum"),
            "l2_write_sectors": metrics.get("lts__t_sectors_op_write.sum"),
        },
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
        "permission_error": "ERR_NVGPUCTRPERM" in result.stderr,
    }


def _ncu_payload(
    binary: Path, args: argparse.Namespace, candidate_gates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    wall_passed = [
        variant
        for variant in K_REUSE_V_VARIANTS
        if candidate_gates[variant]["wall_time_passed"]
    ]
    eligible = [
        variant
        for variant in K_REUSE_V_VARIANTS
        if candidate_gates[variant]["eligible_for_ncu"]
    ]
    result: dict[str, Any] = {
        "metrics_requested_after_wall_gate": list(K_NCU_METRICS),
        "wall_gate_passed_variants": wall_passed,
        "eligible_variants": eligible,
    }
    if not wall_passed:
        result["status"] = "not_attempted_before_wall_gate"
        return result
    if not eligible:
        result["status"] = "not_eligible_after_wall_gate"
        return result
    if args.skip_ncu:
        result["status"] = "skipped_by_user_after_wall_gate"
        return result
    result["status"] = "attempted_after_wall_gate"
    result["baseline"] = _run_ncu_variant(binary, args, "baseline")
    for variant in eligible:
        result[variant] = _run_ncu_variant(binary, args, variant)
    return result


def _candidate_gate(
    payload: dict[str, Any], variant: str, resource_reasons: list[str]
) -> dict[str, Any]:
    exactness = payload.get("exactness", {})
    exactness_key = K_EXACTNESS_KEY_BY_VARIANT[variant]
    speedup = payload["timing"][f"{variant}_speedup_vs_baseline_pct"]
    correctness_passed = bool(exactness.get(exactness_key, {}).get("bitwise_equal"))
    wall_time_passed = speedup >= K_REQUIRED_SPEEDUP_PCT
    resource_passed = not resource_reasons
    is_reuse_v_candidate = variant in K_REUSE_V_VARIANTS
    rejection_reasons: list[str] = []
    if not correctness_passed:
        rejection_reasons.append(
            f"{exactness_key} differs at full-output uint32 granularity"
        )
    if not wall_time_passed:
        rejection_reasons.append(
            f"{variant} speedup {speedup:.3f}% is below the 2.000% gate"
        )
    rejection_reasons.extend(resource_reasons)
    if not is_reuse_v_candidate:
        rejection_reasons.append(
            "legacy BM8 split uses 2 CTA/group and rereads V for both M halves"
        )
    eligible_for_ncu = (
        is_reuse_v_candidate
        and correctness_passed
        and wall_time_passed
        and resource_passed
    )
    if not is_reuse_v_candidate:
        decision = "legacy_bm8_split_retained_for_comparison_only"
    elif eligible_for_ncu:
        decision = "eligible_for_minimal_ncu_before_any_main_kernel_integration"
    else:
        decision = "rejected_do_not_integrate"
    return {
        "candidate_family": (
            "m16_reuse_v" if is_reuse_v_candidate else "legacy_bm8_split"
        ),
        "exactness_key": exactness_key,
        "correctness_passed": correctness_passed,
        "wall_time_passed": wall_time_passed,
        "resource_passed": resource_passed,
        "eligible_for_ncu": eligible_for_ncu,
        "integration_allowed": eligible_for_ncu,
        "speedup_vs_baseline_pct": speedup,
        "required_raw_speedup_pct": K_REQUIRED_SPEEDUP_PCT,
        "decision": decision,
        "rejection_reasons": rejection_reasons,
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
    expected_words = args.groups * 16 * 256
    if exactness.get("word_count_per_comparison") != expected_words:
        raise RuntimeError(
            "Probe did not compare the complete [groups, 16, 256] output."
        )
    sass = _inspect_sass(binary)
    candidate_gates = {
        variant: _candidate_gate(
            payload, variant, _resource_reasons(payload, sass, variant)
        )
        for variant in K_RAW_VARIANTS
    }
    clock_after_timing_mhz = _verify_clock(args)
    ncu = _ncu_payload(binary, args, candidate_gates)
    clock_after_ncu_mhz = _verify_clock(args)

    payload["build"] = {
        "command": build_command,
        "target": "sm_70",
        "ptxas_log": ptxas_log[-4000:],
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
    payload["gates"] = candidate_gates
    payload["result_groups"] = {
        "legacy_bm8_split": list(K_LEGACY_SPLIT_VARIANTS),
        "m16_reuse_v": list(K_REUSE_V_VARIANTS),
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
