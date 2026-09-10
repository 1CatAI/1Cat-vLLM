# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a private E4M3 grouped-attention scheduling candidate.

The original per-head arithmetic, FP32 numerator/max/sum workspace and
native input validation remain intact. Only the number of heads per CTA
changes. This builder installs no serving route or default.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Expected one source anchor: {old}")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--head-groups", type=int, choices=(1, 3), default=3)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "flash-attention-v100"
    original = root / "kernel/flash_decode_paged.cu"
    source = original.read_text()
    barrier = """        __syncwarp();
        if (lane_id == 0) {
          if (tile_sum > 0.0f) {"""
    if source.count(barrier) != 1:
        raise ValueError("The grouped online-softmax warp-state fix is required")
    if args.head_groups == 3:
        source = replace_once(
            source,
            "constexpr int kGroupedVerifyRows = 48;",
            "constexpr int kGroupedVerifyRows = 16;",
        )
        source = replace_once(
            source,
            "constexpr int kGroupedVerifyThreads = 512;",
            "constexpr int kGroupedVerifyThreads = 256;",
        )
        source = replace_once(
            source,
            "kernel<<<dim3(1, 80), kGroupedVerifyThreads, "
            "kCompensatedSmemBytes, stream>>>",
            "kernel<<<dim3(3, 80), kGroupedVerifyThreads, "
            "kCompensatedSmemBytes, stream>>>",
        )
    source = replace_once(
        source,
        "flash_attention_grouped_e4m3_fp32_paged(",
        "private_grouped_e4m3_fp32_paged(",
    )
    source += (
        "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        '  m.def("run", &private_grouped_e4m3_fp32_paged);\n}\n'
    )
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    for name in ("include", "kernel"):
        target = sources / name
        target.mkdir(exist_ok=True)
        for pattern in ("*.h", "*.cuh"):
            for header in (root / name).glob(pattern):
                shutil.copy2(header, target)
    shutil.copy2(root / "LICENSE", sources)
    path = sources / "kernel/grouped-attention.cu"
    path.write_text(source)
    # Retain Flash-V100's existing math flags; this is a scheduling candidate.
    flags = [
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_70,code=sm_70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
        "-Xptxas=-v",
    ]
    manifest = {
        "input_source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_files": {
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
        "head_groups": args.head_groups,
        "extra_cuda_cflags": flags,
        "scope": "Private operator candidate; full-model admission required",
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir(exist_ok=True)
        library = Path(
            load(
                name="sm70_grouped_attention_candidate",
                sources=[str(path)],
                build_directory=str(build),
                extra_cuda_cflags=flags,
                extra_include_paths=[str(sources / "kernel"), str(sources / "include")],
                verbose=True,
            ).__file__
        )
        manifest["library"] = str(library)
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
