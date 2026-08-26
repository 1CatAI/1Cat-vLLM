# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and run the SM70 minimum-communication C-to-A microbenchmark."""

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

SYMBOLS = {
    "reference": "sm70_hmma_c2a_reference",
    "candidate": "sm70_hmma_c2a_mincomm",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=144)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--launches", type=int, default=8)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _build(verbose: bool) -> tuple[Path, list[str], str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required")
    source = Path(__file__).resolve().parent / "csrc" / "sm70_hmma_c2a_mincomm_micro.cu"
    build_dir = Path(tempfile.gettempdir()) / f"vllm-sm70-c2a-mincomm-{os.getuid()}"
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_hmma_c2a_mincomm_micro_sm70"
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
        print("build:", " ".join(command))
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return binary, command, result.stderr


def _function_section(sass: str, symbol: str) -> str:
    headers = list(re.finditer(r"(?m)^\s*Function\s*:\s*(.+?)\s*$", sass))
    for index, header in enumerate(headers):
        if symbol not in header.group(1):
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(sass)
        return sass[header.start() : end]
    raise RuntimeError(f"cuobjdump did not find {symbol}")


def _inspect_sass(binary: Path) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is required")
    result = subprocess.run(
        [cuobjdump, "--dump-sass", str(binary)],
        check=True,
        text=True,
        capture_output=True,
    )
    payload: dict[str, Any] = {}
    for name, symbol in SYMBOLS.items():
        section = _function_section(result.stdout, symbol)
        payload[name] = {
            "symbol": symbol,
            "instructions": {
                mnemonic.lower(): len(
                    re.findall(rf"\b{re.escape(mnemonic)}(?:\.|\b)", section)
                )
                for mnemonic in ("SHFL", "F2F", "PRMT", "LDL", "STL")
            },
        }
    return payload


def main() -> int:
    args = _parse_args()
    binary, build_command, ptxas_log = _build(args.verbose)
    command = [
        str(binary),
        "--device",
        str(args.device),
        "--blocks",
        str(args.blocks),
        "--warmup",
        str(args.warmup),
        "--rounds",
        str(args.rounds),
        "--launches",
        str(args.launches),
    ]
    if args.verbose:
        print("run:", " ".join(command))
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)
    payload["build"] = {
        "command": build_command,
        "binary": str(binary),
        "ptxas_log": ptxas_log,
    }
    payload["sass"] = _inspect_sass(binary)
    reference_shfl = payload["sass"]["reference"]["instructions"]["shfl"]
    candidate_shfl = payload["sass"]["candidate"]["instructions"]["shfl"]
    payload["structural_gate"] = {
        "reference_shfl": reference_shfl,
        "candidate_shfl": candidate_shfl,
        "expected_shfl_reduction": 16,
        "passed": reference_shfl - candidate_shfl == 16,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if not payload["structural_gate"]["passed"]:
        return 1
    return 0 if payload["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
