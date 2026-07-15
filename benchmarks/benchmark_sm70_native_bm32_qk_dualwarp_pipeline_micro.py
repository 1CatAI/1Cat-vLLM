# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Strict go/no-go harness for the SM70 BM32 dual-warp QK pipeline experiment."""

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

VISIBLE_DEVICES = "2"
PHYSICAL_GPU = 2
CLOCK_MHZ = 1200
MAX_SHARED_BYTES = 41936
GROUPS = 144
WARMUP = 20
ROUNDS = 100
LAUNCHES = 8
CASES = tuple(
    (pattern, blocks) for pattern in ("random", "alternating") for blocks in (1, 2, 4)
)

BASELINE = "sm70_native_bm32_qk_dualwarp_pipeline_baseline"
CANDIDATE = "sm70_native_bm32_qk_dualwarp_pipeline_candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def clock() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=clocks.current.graphics",
            "--format=csv,noheader,nounits",
            "--id",
            str(PHYSICAL_GPU),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    found = re.search(r"\d+", result.stdout)
    if found is None:
        raise RuntimeError(f"could not parse GPU clock: {result.stdout!r}")
    return int(found.group())


def require_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE_DEVICES:
        raise RuntimeError(
            "requires CUDA_VISIBLE_DEVICES=2 (logical cuda:0 is physical GPU 2)"
        )
    if clock() != CLOCK_MHZ:
        raise RuntimeError(
            f"physical GPU 2 must already be locked at {CLOCK_MHZ} MHz; harness never changes clocks"
        )


def build(verbose: bool) -> tuple[Path, list[str], str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required")
    source = (
        Path(__file__).with_name("csrc")
        / "sm70_native_bm32_qk_dualwarp_pipeline_micro.cu"
    )
    binary = (
        Path(tempfile.gettempdir())
        / f"vllm-sm70-dualwarp-pipeline-{os.getuid()}"
        / "micro"
    )
    binary.parent.mkdir(parents=True, exist_ok=True)
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
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    if verbose:
        print("build:", " ".join(command), file=sys.stderr)
        print(completed.stderr, end="", file=sys.stderr)
    return binary, command, completed.stderr


def ptxas(log: str, symbol: str) -> dict[str, int] | None:
    match = re.search(
        rf"Compiling entry function '{re.escape(symbol)}'.*?(?=Compiling entry function '|\Z)",
        log,
        re.S,
    )
    if match is None:
        return None
    text = match.group()
    stack = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
        text,
    )
    regs = re.search(r"Used (\d+) registers", text)
    smem = re.search(r"(\d+) bytes smem", text)
    if stack is None or regs is None or smem is None:
        return None
    return {
        "stack_frame_bytes": int(stack.group(1)),
        "spill_store_bytes": int(stack.group(2)),
        "spill_load_bytes": int(stack.group(3)),
        "registers_per_thread": int(regs.group(1)),
        "static_shared_bytes": int(smem.group(1)),
    }


def run_case(
    binary: Path, pattern: str, nblocks: int, smoke: bool
) -> tuple[dict[str, Any] | None, str]:
    command = [
        str(binary),
        "--device",
        "0",
        "--groups",
        str(GROUPS),
        "--nblocks",
        str(nblocks),
        "--warmup",
        str(WARMUP),
        "--rounds",
        str(ROUNDS),
        "--launches",
        str(LAUNCHES),
        "--pattern",
        pattern,
    ]
    if smoke:
        command.append("--smoke")
    completed = subprocess.run(command, text=True, capture_output=True)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return (
            None,
            f"benchmark JSON parse failure (returncode={completed.returncode}): {completed.stderr[-1000:]}",
        )
    if completed.returncode:
        return payload, f"benchmark returned {completed.returncode}"
    return payload, ""


def resource_reasons(
    payload: dict[str, Any], properties: dict[str, dict[str, int] | None]
) -> list[str]:
    reasons: list[str] = []
    for path in ("baseline", "candidate"):
        runtime = payload.get("resources", {}).get(path, {})
        if runtime.get("threads_per_cta") != 512:
            reasons.append(
                f"{path}: threads_per_cta={runtime.get('threads_per_cta')}, requires 512"
            )
        if runtime.get("registers_per_thread", 65) > 64:
            reasons.append(f"{path}: runtime registers >64")
        if runtime.get("local_bytes_per_thread") != 0:
            reasons.append(f"{path}: runtime local bytes must be 0")
        if runtime.get("static_shared_bytes", MAX_SHARED_BYTES + 1) > MAX_SHARED_BYTES:
            reasons.append(
                f"{path}: runtime shared={runtime.get('static_shared_bytes')}, requires <={MAX_SHARED_BYTES}"
            )
        if runtime.get("active_ctas_per_sm", 0) < 2:
            reasons.append(f"{path}: requires >=2 CTA/SM")
        prop = properties[path]
        if prop is None:
            reasons.append(f"{path}: PTXAS properties unavailable")
            continue
        if prop["registers_per_thread"] > 64:
            reasons.append(
                f"{path}: PTXAS registers={prop['registers_per_thread']}, requires <=64"
            )
        if prop["static_shared_bytes"] > MAX_SHARED_BYTES:
            reasons.append(
                f"{path}: PTXAS shared={prop['static_shared_bytes']}, requires <={MAX_SHARED_BYTES}"
            )
        for key in ("stack_frame_bytes", "spill_store_bytes", "spill_load_bytes"):
            if prop[key]:
                reasons.append(f"{path}: PTXAS {key}={prop[key]}, requires 0")
    return reasons


def ptxas_build_reasons(properties: dict[str, dict[str, int] | None]) -> list[str]:
    reasons: list[str] = []
    for path in ("baseline", "candidate"):
        prop = properties[path]
        if prop is None:
            reasons.append(f"{path}: PTXAS properties unavailable")
            continue
        if prop["registers_per_thread"] > 64:
            reasons.append(
                f"{path}: PTXAS registers={prop['registers_per_thread']}, requires <=64"
            )
        if prop["static_shared_bytes"] > MAX_SHARED_BYTES:
            reasons.append(
                f"{path}: PTXAS shared={prop['static_shared_bytes']}, requires <={MAX_SHARED_BYTES}"
            )
        for key in ("stack_frame_bytes", "spill_store_bytes", "spill_load_bytes"):
            if prop[key]:
                reasons.append(f"{path}: PTXAS {key}={prop[key]}, requires 0")
    return reasons


def case_reasons(
    payload: dict[str, Any], properties: dict[str, dict[str, int] | None]
) -> list[str]:
    reasons = resource_reasons(payload, properties)
    dataflow = payload.get("dataflow", {})
    if dataflow.get("equivalence_check_passed") is not True:
        reasons.append("dataflow equivalence check did not pass")
    if dataflow.get("all_16_warps_execute_qk_hmma") is not True:
        reasons.append("candidate does not establish 16 QK HMMA warps")
    if dataflow.get("global_k_loads_per_pair_k16") != 1:
        reasons.append("candidate does not establish one global K load per pair/K16")
    exact = payload.get("exactness", {})
    if not exact.get("bitwise_equal") or exact.get("mismatch_words") != 0:
        reasons.append("full packed-uint32 bitwise exactness failed")
    xor = exact.get("xor", {})
    if xor.get("reduction") != 0 or xor.get("max_word") != 0:
        reasons.append("packed-uint32 XOR is nonzero")
    pairs = payload.get("pairs", {})
    if pairs.get("count") != ROUNDS:
        reasons.append(f"paired rounds={pairs.get('count')}, requires {ROUNDS}")
    if pairs.get("candidate_faster", 0) < 95:
        reasons.append(
            f"candidate wins={pairs.get('candidate_faster')}, requires >=95/100"
        )
    speedup = payload.get("timing", {}).get("candidate_speedup_vs_baseline_pct")
    if speedup is None or speedup < 2.0:
        reasons.append(f"median wall improvement={speedup}%, requires >=2.0%")
    return reasons


def emit(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    require_environment()
    binary, command, log = build(args.verbose)
    properties = {"baseline": ptxas(log, BASELINE), "candidate": ptxas(log, CANDIDATE)}
    smoke_cases: list[dict[str, Any]] = []
    rejection = ptxas_build_reasons(properties)
    rejection.append(
        "candidate aliases persistent query storage for QK results; "
        "multi-block execution would require restaging 16 KiB of Q and an "
        "additional CTA barrier for every KV block"
    )
    if rejection:
        emit(
            {
                "benchmark": "sm70_native_bm32_qk_dualwarp_pipeline_micro",
                "dataflow": {
                    "ledger_closed_schedule": "8 dedicated K-staging producers plus 8 non-QK consumers",
                    "candidate": "producer loads once then computes top QK; consumer computes bottom QK; all 16 warps execute HMMA",
                    "equivalent_to_closed_schedule": False,
                    "multi_block_query_alias_valid": False,
                },
                "build": {
                    "command": command,
                    "optimization": "-O3",
                    "ptxas": properties,
                },
                "smoke": smoke_cases,
                "formal": [],
                "decision": "rejected_do_not_integrate",
                "rejection_reasons": rejection,
                "ncu": "not_run_ptxas_resource_gate_failed",
            },
            args.json_out,
        )
        return 1
    # Smoke is resource + exact only. It deliberately does not launch timing rounds.
    for pattern, nblocks in CASES:
        payload, error = run_case(binary, pattern, nblocks, smoke=True)
        record: dict[str, Any] = {
            "pattern": pattern,
            "nblocks": nblocks,
            "payload": payload,
        }
        if error:
            record["rejection_reasons"] = [error]
            rejection.append(f"{pattern}/nblocks={nblocks}: {error}")
        elif resource_reasons(payload, properties):
            record["rejection_reasons"] = resource_reasons(payload, properties)
            rejection.extend(
                f"{pattern}/nblocks={nblocks}: {reason}"
                for reason in record["rejection_reasons"]
            )
        elif not payload.get("dataflow", {}).get("equivalence_check_passed"):
            record["rejection_reasons"] = [
                "candidate is equivalent to a closed dedicated-producer/consumer schedule"
            ]
            rejection.append(
                f"{pattern}/nblocks={nblocks}: dataflow equivalence check failed"
            )
        elif not payload.get("exactness", {}).get("bitwise_equal"):
            record["rejection_reasons"] = ["full-output exactness failed"]
            rejection.append(f"{pattern}/nblocks={nblocks}: exactness failed")
        else:
            record["rejection_reasons"] = []
        smoke_cases.append(record)
        if rejection:
            break
    if rejection:
        emit(
            {
                "benchmark": "sm70_native_bm32_qk_dualwarp_pipeline_micro",
                "dataflow": {
                    "ledger_closed_schedule": "8 dedicated K-staging producers plus 8 non-QK consumers",
                    "candidate": "producer loads once then computes top QK; consumer computes bottom QK; all 16 warps execute HMMA",
                    "equivalent_to_closed_schedule": False,
                    "multi_block_query_alias_valid": False,
                },
                "build": {
                    "command": command,
                    "optimization": "-O3",
                    "ptxas": properties,
                },
                "smoke": smoke_cases,
                "decision": "rejected_do_not_integrate",
                "rejection_reasons": rejection,
                "ncu": "not_run_gate_failed",
            },
            args.json_out,
        )
        return 1
    formal_cases: list[dict[str, Any]] = []
    for pattern, nblocks in CASES:
        payload, error = run_case(binary, pattern, nblocks, smoke=False)
        reasons = [error] if error else case_reasons(payload, properties)
        formal_cases.append(
            {
                "pattern": pattern,
                "nblocks": nblocks,
                "payload": payload,
                "rejection_reasons": reasons,
            }
        )
        if reasons:
            rejection.extend(
                f"{pattern}/nblocks={nblocks}: {reason}" for reason in reasons
            )
            break
    require_environment()
    emit(
        {
            "benchmark": "sm70_native_bm32_qk_dualwarp_pipeline_micro",
            "target": "sm70",
            "dataflow": {
                "ledger_closed_schedule": "8 dedicated K-staging producers plus 8 non-QK consumers",
                "candidate": "producer loads once then computes top QK; consumer computes bottom QK; all 16 warps execute HMMA",
                "equivalent_to_closed_schedule": False,
                "multi_block_query_alias_valid": False,
            },
            "build": {"command": command, "optimization": "-O3", "ptxas": properties},
            "configuration": {
                "physical_gpu": PHYSICAL_GPU,
                "cuda_visible_devices": VISIBLE_DEVICES,
                "clock_mhz": CLOCK_MHZ,
                "rounds": ROUNDS,
                "launches_per_round": LAUNCHES,
            },
            "smoke": smoke_cases,
            "formal": formal_cases,
            "decision": "accepted_for_follow_up_only"
            if not rejection
            else "rejected_do_not_integrate",
            "rejection_reasons": rejection,
            "ncu": "not_run_gate_failed"
            if rejection
            else "not_run_microbenchmark_decision_complete",
        },
        args.json_out,
    )
    return 0 if not rejection else 1


if __name__ == "__main__":
    raise SystemExit(main())
