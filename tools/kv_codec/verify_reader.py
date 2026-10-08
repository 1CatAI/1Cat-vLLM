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
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", path.read_text(), flags=re.DOTALL)
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--packed", action="store_true")
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
    if args.packed:
        original = subprocess.check_output(
            [
                "git",
                "show",
                f"{BASE_SHA}:flash-attention-v100/kernel/flash_decode_paged.cu",
            ],
            cwd=root,
            text=True,
        )
        start = original.index(
            "__device__ __forceinline__ uint32_t\nfp8_e5m2_pair_to_half2_bits"
        )
        end = original.index(
            "\ntemplate <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT", start
        )
        with (baseline / "fp8_kv_utils.cuh").open("a") as file:
            file.write("\nnamespace flash_v100 {\n" + original[start:end] + "\n}\n")
            # Exercise the original global-loader format branch, including
            # its LUT parameter, rather than changing the probe's alias/escape
            # behavior by calling only the lowest-level converters.
            start = original.index(
                "  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16)",
                original.index(
                    "__device__ __forceinline__ uint4 load_xqa_tc_kv_vector("
                ),
            )
            end = original.index("\n}\n", start)
            file.write(
                "\nnamespace flash_v100 {\n"
                "template <int KV_DTYPE, bool E4M3_SHARED_LUT>\n"
                "__device__ __forceinline__ uint4 legacy_load_half8(\n"
                "const void* __restrict__ kv_cache, const int64_t physical_offset,\n"
                "const int vec_col, const uint16_t* __restrict__ e4m3_lut) {\n"
                + original[start:end]
                + "\n}\n}\n"
            )
    includes = {
        "baseline": baseline,
        "candidate": root / "flash-attention-v100/kernel",
    }
    source = Path(__file__).with_name(
        "packed_reader_probe.cu" if args.packed else "reader_probe.cu"
    )
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
        if args.packed and arm == "candidate":
            command.append("-DKV_CODEC_CANDIDATE=1")
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
            "All FP16 storage encodings and 65536 FP8 byte pairs; "
            "standard/fast/LUT packed reads and base/column global addressing"
            if args.packed
            else "FP16 all 65536 encodings; E4M3/E5M2 all 256 encodings; "
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
