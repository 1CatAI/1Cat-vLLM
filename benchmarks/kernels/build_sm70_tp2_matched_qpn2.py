# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build an isolated TP2 QPN2 candidate with TurboMind's rounding/split order.

This reproduces the retained TP2 projection experiment. It does not install a
library or enable a serving route. The caller must use the split count observed
for the matching TurboMind projection; this is not an autotuning interface.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def replace_exact(source: str, old: str, new: str, count: int) -> str:
    if source.count(old) != count:
        raise ValueError(f"QPN2 source anchor changed: {old!r}")
    return source.replace(old, new)


def generate(source: str) -> str:
    source = source.replace("nvfp4_qpn2_", "tp2_qpn2_matched_").replace(
        "TORCH_LIBRARY_FRAGMENT(_qpn2_candidate,",
        "TORCH_LIBRARY_FRAGMENT(_tp2_qpn2_matched,",
    )
    source = replace_exact(
        source,
        "  const half2 global_scale2 = __float2half2_rn(global_scale * 16384.0f);\n",
        "",
        2,
    )
    source = replace_exact(
        source,
        """    const half2 scale = __hmul2(
        fp8e4m3_to_half2(__ldg(scale_ptr + static_cast<size_t>(group) * 32)),
        global_scale2);""",
        """    const half2 raw_scale = fp8e4m3_to_half2(
        __ldg(scale_ptr + static_cast<size_t>(group) * 32));
    // Match TurboMind's effective FP16 scale before E2M1 multiplication.
    // Do not round the global factor to FP16 before multiplying the group.
    const half effective = __float2half_rn(__low2float(raw_scale) * global_scale);
    const half2 scale = __hmul2(__halves2half2(effective, effective),
                                __float2half2_rn(16384.0f));""",
        2,
    )
    source = replace_exact(
        source,
        "  const int groups_per_warp = groups_k16 / SplitK;\n"
        "  const int group_begin = warp * groups_per_warp;",
        """  const int chunks = k / 64;
  const int chunks_per_warp = chunks / SplitK;
  const int extra_begin = SplitK - chunks % SplitK;
  const int group_begin = (warp * chunks_per_warp + max(warp - extra_begin, 0)) * 4;
  const int groups_per_warp = (chunks_per_warp + (warp >= extra_begin)) * 4;""",
        2,
    )
    source = replace_exact(
        source,
        "split_k == 8 || split_k == 16 || split_k == 32",
        "(split_k >= 1 && split_k <= 16) || split_k == 32",
        1,
    )
    source = replace_exact(
        source,
        "(input.size(1) / 16) % split_k == 0",
        "input.size(1) / 64 >= split_k",
        2,
    )
    pieces = []
    for split in range(1, 17):
        if split in (5, 7, 8, 9, 16):
            continue
        condition = "if" if not pieces else "else if"
        pieces.append(
            f"  {condition} (split_k == {split}) {{\n"
            f"    VLLM_LAUNCH_QPN2(1, {split}, 1);\n  }}"
        )
    dispatch = (
        "\n".join(pieces)
        + """ else if (split_k == 5) {
    VLLM_LAUNCH_QPN2(1, 5, 1);
  } else if (split_k == 7) {
    VLLM_LAUNCH_QPN2(1, 7, 1);
  } else if (split_k == 9) {
    VLLM_LAUNCH_QPN2(1, 9, 1);
  } else if (native_two_tile && split_k == 8 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN2(2, 8, 1);"""
    )
    source = replace_exact(
        source,
        "  if (native_two_tile && split_k == 8 && accumulator_chains == 1) {\n"
        "    VLLM_LAUNCH_QPN2(2, 8, 1);",
        dispatch,
        1,
    )
    return source.replace("qpn2_matched", "qpn2_matched_all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    parent = root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    output = args.output_dir.resolve()
    sources = output / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    path = sources / "qpn2-matched-all.cu"
    path.write_text(generate(parent.read_text()))
    shutil.copy2(parent.parent / "LICENSE.v100-skinny", sources)
    flags = [
        "-O3",
        "-std=c++17",
        "-DVLLM_NVFP4_QPN2_STANDALONE",
        "-DVLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    report = {
        "parent_source_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cuda_flags": flags,
        "supported_experiment": "TP2 q8, one accumulator, observed TM split",
        "serving_route_enabled": False,
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = output / "build"
        build.mkdir(exist_ok=True)
        library = Path(
            load(
                name="tp2_qpn2_matched_all",
                sources=[str(path)],
                build_directory=str(build),
                extra_cflags=["-O3"],
                extra_cuda_cflags=flags,
                is_python_module=False,
                verbose=True,
            )
        )
        report.update(
            library=str(library),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
