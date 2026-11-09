# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Literal CUDA anchors retain their production source spelling.
# ruff: noqa: E501
"""Build private q8 QPN2 layout experiments without changing their arithmetic.

The input's physical layout is [K/16, 8, 16]. With --pack-gated-output,
the fused gate/up projection also writes [hidden/16, 8, 16]. These tensors
must only be passed to consumers of the corresponding private layout.
This does not install a serving operator or enable a model route.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def replace_once(source: str, old: str, new: str, count: int = 1) -> str:
    if source.count(old) != count:
        raise ValueError(f"Expected {count} occurrences of source anchor: {old}")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pack-gated-output", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    original = root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    source = replace_once(
        original.read_text(),
        """const half* input_row = input + static_cast<size_t>(row) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);""",
        """const half* input_row = input + static_cast<size_t>(group) * 128 + row * 16;
        input01 = *reinterpret_cast<const uint4*>(input_row);
        input23 = *reinterpret_cast<const uint4*>(input_row + 8);""",
        count=2,
    )
    source = replace_once(
        source,
        "m >= 1 && m <= kQpn2MaxRows && out.size(0) == m",
        "m == 8 && out.size(0) == m",
    )
    source = replace_once(
        source,
        '"NVFP4 QPN2 requires M in [1, ", kQpn2MaxRows, "]"',
        '"Private packed-input NVFP4 QPN2 requires M=8"',
    )
    if args.pack_gated_output:
        source = replace_once(
            source,
            """output[static_cast<size_t>(output_row) * hidden + blockIdx.x * 32 +
             output_col] = __hmul(silu, up_half);""",
            """const int logical_col = blockIdx.x * 32 + output_col;
      output[static_cast<size_t>(logical_col / 16) * 128 +
             output_row * 16 + logical_col % 16] = __hmul(silu, up_half);""",
        )
    namespace = "_qpn2_packed_mlp" if args.pack_gated_output else "_qpn2_packed_input"
    prefix = "packedmlp_" if args.pack_gated_output else "packedinput_"
    source = source.replace("_qpn2_candidate", namespace)
    source = source.replace("nvfp4_qpn2_", prefix + "nvfp4_qpn2_")
    path = sources / "qpn2-packed-input.cu"
    path.write_text(source)
    shutil.copy2(original.parent / "LICENSE.v100-skinny", sources)
    flags = [
        "-O3",
        "-lineinfo",
        "-gencode=arch=compute_70,code=sm_70",
        "-DVLLM_NVFP4_QPN2_STANDALONE",
        "-DVLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE",
        "-Xptxas=-v",
    ]
    manifest = {
        "input_source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "namespace": namespace,
        "input_layout": "[K/16, 8, 16]",
        "gated_output_layout": "[hidden/16, 8, 16]"
        if args.pack_gated_output
        else "[8, hidden]",
        "extra_cuda_cflags": flags,
        "math_mode": "default",
        "scope": "Private q8 operator; producer/consumer pairing required",
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir(exist_ok=True)
        library = Path(
            load(
                name=namespace.removeprefix("_") + "_candidate",
                sources=[str(path)],
                build_directory=str(build),
                extra_cuda_cflags=flags,
                is_python_module=False,
                verbose=True,
            )
        )
        manifest["library"] = str(library)
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
