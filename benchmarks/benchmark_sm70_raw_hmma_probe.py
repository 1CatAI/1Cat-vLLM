# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and run the isolated full-16x32 SM70 raw m8n8k4 bitwise probe."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Logical CUDA device after CUDA_VISIBLE_DEVICES filtering.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _build_binary(source: Path, verbose: bool) -> tuple[Path, list[str]]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required to build the SM70 raw HMMA probe.")

    build_dir = (
        Path(tempfile.gettempdir())
        / f"vllm-sm70-raw-hmma-probe-{os.getuid()}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "sm70_raw_hmma_probe_sm70"
    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "--generate-code=arch=compute_70,code=sm_70",
        "-o",
        str(binary),
        str(source),
    ]
    if verbose:
        print("build:", " ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)
    return binary, command


def main() -> int:
    args = _parse_args()
    source = Path(__file__).resolve().parent / "csrc" / "sm70_raw_hmma_probe.cu"
    binary, build_command = _build_binary(source, args.verbose)
    command = [str(binary), "--device", str(args.device)]
    if args.verbose:
        print("run:", " ".join(command), file=sys.stderr)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout, end="", file=sys.stderr)
        return completed.returncode

    payload = json.loads(completed.stdout)
    gate = payload.get("gate", {})
    if gate.get("compared_output_words") != 512 or not gate.get("full_16x32"):
        raise RuntimeError("Probe did not compare the required full 16x32 output.")
    payload["build"] = {
        "command": build_command,
        "target": "sm_70",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
