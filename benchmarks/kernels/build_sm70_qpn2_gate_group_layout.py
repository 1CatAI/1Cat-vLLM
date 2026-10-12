# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen adjacent gate/up K16 bundles, unchanged bytes and arithmetic.

The current paired reader loads gate and up panels about 12.5 MB apart.
This layout puts their 288-byte bundles adjacent, retaining each lane's
contiguous eight-byte codes. Repacking is outside capture; no serving layout
or persistent memory scheme is installed by the independent extension.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_paired_gate import generate as paired_source
from torch.utils.cpp_extension import load

READER = r"""
struct GateGroupReader {
  const uint8_t* base;
  int lane;
  float global;
  __device__ GateGroupReader(const uint8_t* w, const void*, int tile,
                            int groups, int thread_lane, float scale)
      : base(w + static_cast<size_t>(tile % 136) * groups * 576 +
             (tile / 136) * 288), lane(thread_lane), global(scale) {}
  __device__ __forceinline__ void load(int group, half2* weights) const {
    const uint8_t* panel = base + static_cast<size_t>(group) * 576;
    const uint2 q = __ldcs(reinterpret_cast<const uint2*>(panel + lane * 8));
    const half2 scale = nvfp4_effective_scale(__ldg(panel + 256 + lane),global);
    dequant_e2m1x8(q.x, scale, weights);
    dequant_e2m1x8(q.y, scale, weights + 4);
  }
};
"""


def generate(source):
    text = paired_source(source)
    begin = text.index("template <bool Interleave>")
    end = text.index("void launch_pair(", begin)
    kernel = text[begin:end].replace("Nvfp4PairReader<false, true>", "GateGroupReader")
    return text[:begin] + READER + kernel + text[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "gate_group_layout.cu"
    source = args.source_root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    text = generate(source)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    load(
        name="qpn2_gate_group_layout_screen",
        sources=[str(path)],
        extra_include_paths=[str(source.parent)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
