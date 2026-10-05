# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a research-only M8 control that computes BA after QKV in the same CTA.

The 96 BA tasks reuse the first 96 of the 128 QKV CTAs. Both reductions retain
all original arithmetic; only the placement of the BA tasks changes. M1 and
ordinary projection dispatch are unchanged. No serving operator is replaced.
"""

import argparse
import os
from pathlib import Path


def candidate_source(source: str) -> str:
    start = source.index("__global__ void fp8_qpn8_sm70_kernel(")
    end = source.index("\ntemplate <int SplitK", start)
    kernel = source[start:end]
    begin = kernel.index("  if constexpr (FusedBA) {")
    finish = kernel.index("  const int quadpair", begin)
    early = kernel[begin:finish]
    body_start = early.index("      constexpr int kBARowsPerBlock")
    body_end = early.index("      return;")
    body = early[body_start:body_end]
    body = body.replace("M1Only ? 0 : (tile - qpn_tiles)", "tile")
    body = body.replace("(tile - qpn_tiles) %", "tile %")
    kernel = (
        kernel[:begin]
        + early.replace("if constexpr (FusedBA)", "if constexpr (FusedBA && M1Only)")
        + kernel[finish:]
    )
    tail = (
        """  if constexpr (FusedBA && !M1Only) {
    // All QKV consumers finish reading the partials before BA reuses them.
    __syncthreads();
    if (tile < m * ((ba_n + 1) / 2)) {
"""
        + body
        + """    }
  }
"""
    )
    kernel = kernel.rstrip()[:-1] + tail + "}\n"
    source = source[:start] + kernel + source[end:]
    grid = "<<<(n / 32 + m * ba_n / 2), 512, 0, at::cuda::getCurrentCUDAStream()>>>"
    assert source.count(grid) == 1
    source = source.replace(
        grid, "<<<(n / 32), 512, 0, at::cuda::getCurrentCUDAStream()>>>"
    )
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from torch.utils.cpp_extension import load

    os.environ.setdefault("MAX_JOBS", "1")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    source_dir = args.source_root / "csrc/sm70_turbomind/ops"
    source = (source_dir / "fp8_qpn8_sm70.cu").read_text()
    for name, text in [("control", source), ("late_ba", candidate_source(source))]:
        build = args.output / name
        build.mkdir(parents=True, exist_ok=True)
        generated = build / "candidate.cu"
        generated.write_text(
            text.replace(
                "TORCH_LIBRARY_FRAGMENT(_C_qwen38,",
                f"TORCH_LIBRARY_FRAGMENT(_qpn8_ba_{name},",
            )
        )
        load(
            name=f"qpn8_ba_{name}",
            sources=[str(generated)],
            extra_include_paths=[str(source_dir)],
            extra_cuda_cflags=[
                "-O3",
                "-DVLLM_QPN8_STANDALONE",
                "-DVLLM_QPN8_STANDALONE_QWEN38_NAMESPACE",
                "--ptxas-options=-v",
            ],
            extra_ldflags=["-Wl,-Bsymbolic"],
            build_directory=str(build),
            is_python_module=False,
            verbose=True,
        )


if __name__ == "__main__":
    main()
