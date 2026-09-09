# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a private fixed-q8 QPN2 candidate with a 64-register cap.

Only row-bound handling and compiler register allocation differ from the
production source. Keep the reduction chains, weight decoding and ordinary
CUDA math. The host entry points reject every row count except eight.
This builder does not install a serving route or enable a default.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def replace(source: str, old: str, new: str, count: int) -> str:
    if source.count(old) != count:
        raise ValueError(f"Expected {count} occurrences of source anchor: {old}")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    original = root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    source = replace(
        original.read_text(),
        "const int row_base = blockIdx.y * kQpn2RowsPerCta * RowTiles;",
        "const int row_base = 0;",
        2,
    )
    source = replace(source, "if (row < m) {", "{", 2)
    source = replace(source, "if (output_row < m) {", "{", 2)
    source = replace(
        source,
        "m >= 1 && m <= kQpn2MaxRows && out.size(0) == m",
        "m == 8 && out.size(0) == m",
        1,
    )
    source = replace(
        source,
        '"NVFP4 QPN2 requires M in [1, ", kQpn2MaxRows, "]"',
        '"Private fixed-q8 QPN2 requires M=8"',
        1,
    )
    for gated in ("false", "true"):
        anchor = f"  check_qpn2_tensors(out, input, codes, scales, {gated});"
        source = replace(
            source,
            anchor,
            '  TORCH_CHECK(input.size(0) == 8, "benchmark candidate is q8-only");'
            "\n" + anchor,
            1,
        )
    source = source.replace("_qpn2_candidate", "_qpn2_capped")
    source = source.replace("nvfp4_qpn2_", "q8capped_nvfp4_qpn2_")
    path = sources / "qpn2-fixed-cap64-sidecar.cu"
    path.write_text(source)
    shutil.copy2(original.parent / "LICENSE.v100-skinny", sources)
    flags = [
        "-O3",
        "-lineinfo",
        "-gencode=arch=compute_70,code=sm_70",
        "-DVLLM_NVFP4_QPN2_STANDALONE",
        "-DVLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE",
        "-Xptxas=-v",
        "--maxrregcount=64",
    ]
    manifest = {
        "input_source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "namespace": "_qpn2_capped",
        "extra_cuda_cflags": flags,
        "math_mode": "default",
        "row_count": 8,
        "scope": "Private operator candidate; full-model admission required",
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir(exist_ok=True)
        library = Path(
            load(
                name="qpn2_fixed_cap64_sidecar",
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
