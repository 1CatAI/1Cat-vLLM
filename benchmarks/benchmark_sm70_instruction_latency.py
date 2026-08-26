# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure SM70 instruction dependency and throughput costs in GPU cycles."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _build(iterations: int, verbose: bool) -> tuple[Path, list[str], str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required")
    source = (
        Path(__file__).resolve().parent / "csrc" / "sm70_instruction_latency_probe.cu"
    )
    build_dir = Path(tempfile.gettempdir()) / f"vllm-sm70-latency-probe-{os.getuid()}"
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_instruction_latency_probe_sm70"
    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "--generate-code=arch=compute_70,code=sm_70",
        "--ptxas-options=-v",
        f"-DSM70_LATENCY_PROBE_ITERATIONS={iterations}",
        "-o",
        str(binary),
        str(source),
    ]
    if verbose:
        print("build:", " ".join(command))
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return binary, command, completed.stderr


def _inspect_sass(binary: Path) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is required")
    completed = subprocess.run(
        [cuobjdump, "--dump-sass", str(binary)],
        check=True,
        text=True,
        capture_output=True,
    )
    sections = re.split(r"(?=\s*Function\s*:)", completed.stdout)
    payload: dict[str, Any] = {}
    for symbol in (
        "dependent_hmma_probe",
        "independent_hmma_probe",
        "dependent_shared_load_probe",
        "dependent_global_load_probe",
        "shared_roundtrip_probe",
    ):
        section = next(
            part
            for part in sections
            if re.search(rf"\d+{re.escape(symbol)}E", part.splitlines()[0])
        )
        payload[symbol] = {
            mnemonic.lower(): len(
                re.findall(rf"\b{re.escape(mnemonic)}(?:\.|\b)", section)
            )
            for mnemonic in ("HMMA", "LDG", "LDS", "STS", "BAR", "BRA")
        }
    return payload


def _derive_measured_costs(payload: dict[str, Any]) -> dict[str, float]:
    probes = payload["probes"]
    return {
        "dependent_integer_loop_cycles": probes["empty_loop"][
            "median_cycles_per_operation"
        ],
        "dependent_hmma_m8n8k4_cycles": probes["dependent_hmma_m8n8k4"][
            "median_cycles_per_operation"
        ],
        "four_chain_hmma_m8n8k4_issue_cycles": probes["four_chain_hmma_m8n8k4"][
            "median_cycles_per_operation"
        ],
        "dependent_shared_load_cycles": probes["dependent_shared_load"][
            "median_cycles_per_operation"
        ],
        "dependent_global_cg_load_l2_hit_cycles": probes[
            "dependent_global_cg_load_l2_hit"
        ]["median_cycles_per_operation"],
        "shared_store_load_warp_roundtrip_cycles": probes[
            "shared_store_load_warp_roundtrip"
        ]["median_cycles_per_operation"],
    }


def main() -> int:
    args = _parse_args()
    binary, build_command, ptxas_log = _build(args.iterations, args.verbose)
    command = [
        str(binary),
        "--device",
        str(args.device),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
    ]
    if args.verbose:
        print("run:", " ".join(command))
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)
    payload["measured_costs"] = _derive_measured_costs(payload)
    payload["sass"] = _inspect_sass(binary)
    payload["build"] = {
        "binary": str(binary),
        "command": build_command,
        "ptxas_log": ptxas_log,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
