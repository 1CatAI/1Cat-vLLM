# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build isolated activation-reuse controls without changing serving dispatch.

The original warp K partitions, HMMA sequence and final reduction are retained.
Only the native gated projection's activation source changes. Each CTA stages a
small panel once; gate and up warps read the same panel from shared memory. This
uses the same group/row/K layout as the M32 N128 experiment, without its external
split-K or output-layout changes.
"""

import argparse
import os
from pathlib import Path


def candidate_source(source: str, panel: int) -> str:
    start = source.index("__global__ void nvfp4_qpn2_gated_sm70_kernel(")
    end = source.index("\ntemplate <int SplitK", start)
    kernel = source[start:end]
    shared = "  __shared__ float partials[2][SplitK][RowTiles * 256];"
    assert kernel.count(shared) == 1
    kernel = kernel.replace(
        shared,
        shared
        + f"\n  constexpr int Panel = SplitK * RowTiles == 8 ? {panel} : 4;"
        + "\n  __shared__ __align__(16) half activation[SplitK][Panel]"
        + "[RowTiles * 8][16];",
    )
    loop = """#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {"""
    assert kernel.count(loop) == 1
    kernel = kernel.replace(
        loop,
        """  for (int panel = 0; panel < groups_per_warp; panel += Panel) {
    // The first projection loads each unique activation vector once.
    // Both projections participate in the barriers, including padded rows.
    if (projection == 0) {
      for (int vector = lane; vector < Panel * RowTiles * 8 * 2;
           vector += 32) {
        const int pg = vector / (RowTiles * 8 * 2);
        const int row = (vector / 2) % (RowTiles * 8);
        const int kk = (vector & 1) * 8;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (row_base + row < m && panel + pg < groups_per_warp) {
          value = *reinterpret_cast<const uint4*>(
              input + static_cast<size_t>(row_base + row) * k
              + (group_begin + panel + pg) * 16 + kk);
        }
        *reinterpret_cast<uint4*>(&activation[warp][pg][row][kk]) = value;
      }
    }
    __syncthreads();
#pragma unroll 4
    for (int pg = 0; pg < Panel && panel + pg < groups_per_warp; ++pg) {
      const int group = group_begin + panel + pg;""",
    )
    old_load = (
        "      const int row = row_base + row_tile * kQpn2RowsPerCta + local_row;"
        """
      if (row < m) {
        const half* input_row = input + static_cast<size_t>(row) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }"""
    )
    assert kernel.count(old_load) == 1
    kernel = kernel.replace(
        old_load,
        """      const half* staged =
          &activation[warp][pg][row_tile * kQpn2RowsPerCta + local_row][0];
      input01 = *reinterpret_cast<const uint4*>(staged);
      input23 = *reinterpret_cast<const uint4*>(staged + 8);""",
    )
    boundary = """  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {"""
    # Close the added panel loop after the original group loop.
    pos = kernel.index(boundary, kernel.index("const int group ="))
    kernel = kernel[:pos] + kernel[pos:].replace(
        boundary,
        """    }
    __syncthreads();
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {""",
        1,
    )
    result = source[:start] + kernel + source[end:]
    return result.replace("_qpn2_candidate", f"_qpn2_activation_panel{panel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panels", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--n16", action="store_true")
    args = parser.parse_args()
    from torch.utils.cpp_extension import load

    os.environ.setdefault("MAX_JOBS", "1")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    source_dir = args.source_root / "csrc/sm70_turbomind/ops"
    original = (source_dir / "nvfp4_qpn2_sm70.cu").read_text()
    variants = [
        (f"panel{panel}", candidate_source(original, panel)) for panel in args.panels
    ]
    if args.n16:
        extra = Path(__file__).with_name("qpn2_native_n16_research.cuh").read_text()
        variants.append(("native_n16", original + "\n" + extra))
    for variant, text in variants:
        build = args.output / variant
        build.mkdir(parents=True, exist_ok=True)
        generated = build / "candidate.cu"
        generated.write_text(text)
        load(
            name=f"qpn2_activation_{variant}",
            sources=[str(generated)],
            extra_include_paths=[str(source_dir)],
            extra_cuda_cflags=[
                "-O3",
                "-DVLLM_NVFP4_QPN2_STANDALONE",
                "-DVLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE",
                "--ptxas-options=-v",
            ],
            extra_ldflags=["-Wl,-Bsymbolic"],
            build_directory=str(build),
            is_python_module=False,
            verbose=True,
        )


if __name__ == "__main__":
    main()
