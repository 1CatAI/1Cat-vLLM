# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the extracted reader with an immutable baseline on SM70.

Run under the local GPU locks. This is a reader gate, not an attention/model
performance gate. No vLLM install, borrowed extension or private DSO is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import regex as re

BASE_SHA = "c4f6245f841466782752a8c3283e4727565cf17a"


def normalize_ptx(path: Path) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", path.read_text(), flags=re.DOTALL).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = args.out / "baseline"
    baseline.mkdir(exist_ok=True)
    header = subprocess.check_output(
        ["git", "show", f"{BASE_SHA}:flash-attention-v100/kernel/fp8_kv_utils.cuh"],
        cwd=root,
        text=True,
    )
    (baseline / "fp8_kv_utils.cuh").write_text(header)
    includes = {
        "baseline": baseline,
        "candidate": root / "flash-attention-v100/kernel",
    }
    source = Path(__file__).with_name("reader_probe.cu")
    commands = []
    for arm, include in includes.items():
        command = [
            str(args.nvcc),
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-arch=sm_70",
            "-I",
            str(include),
            str(source),
        ]
        ptx_command = command + ["--ptx", "-o", str(args.out / f"{arm}.ptx")]
        commands.append(ptx_command)
        subprocess.run(ptx_command, check=True)
        if not args.compile_only:
            executable = (
                args.out / arm / "probe" if arm == "baseline" else args.out / "probe"
            )
            build_command = command + ["-o", str(executable)]
            commands.append(build_command)
            subprocess.run(build_command, check=True)
            subprocess.run([str(executable), str(args.out / f"{arm}.bin")], check=True)
    ptx_equal = normalize_ptx(args.out / "baseline.ptx") == normalize_ptx(
        args.out / "candidate.ptx"
    )
    outputs = {}
    if not args.compile_only:
        for arm in includes:
            data = (args.out / f"{arm}.bin").read_bytes()
            outputs[arm] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    result = {
        "base_sha": BASE_SHA,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=root)
        ).hexdigest(),
        "candidate_reader_sha256": hashlib.sha256(
            (includes["candidate"] / "kv_codec.cuh").read_bytes()
        ).hexdigest(),
        "nvcc": subprocess.check_output([str(args.nvcc), "--version"], text=True),
        "normalized_ptx_equal": ptx_equal,
        "outputs": outputs,
        "bitwise_equal": outputs.get("baseline") == outputs.get("candidate")
        if outputs
        else None,
        "commands": commands,
        "scope": (
            "FP16 all 65536 encodings; E4M3/E5M2 all 256 encodings; "
            "four scales; float/half reads"
        ),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("normalized_ptx_equal", "bitwise_equal", "scope")
            }
        )
    )
    if not ptx_equal or (outputs and not result["bitwise_equal"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
