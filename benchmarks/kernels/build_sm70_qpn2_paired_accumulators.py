# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen independent HMMA chains after paired activation reuse.

Two or four chains change the accumulation grouping and need numerical
admission. Existing NCU evidence puts a long wait at the first HMMA operand
use. This screen keeps weights, grids and bytes unchanged and measures a
whole layer, rather than treating instruction count as a speed result.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_paired_gate import generate as paired_source
from torch.utils.cpp_extension import load


def generate(source):
    text = paired_source(source)
    begin = text.index("template <bool Interleave>")
    end = text.index("void launch_pair(", begin)
    body = text[begin:end]
    body = body.replace(
        "template <bool Interleave>", "template <bool Interleave, int NAcc>"
    ).replace("__launch_bounds__(256, 3)", "__launch_bounds__(256, 2)")
    body = body.replace("float accum[2][8]", "float accum[2][NAcc][8]")
    loop_end = body.index("#pragma unroll\n  for (int p = 0;", 1)
    loop = body[:loop_end]
    for projection in ("0", "1", "p"):
        loop = loop.replace(
            "accum[" + projection + "]",
            "accum[" + projection + "][chain]",
        )
    old_loop = (
        "#pragma unroll 4\n"
        "  for (int group = warp * per_warp; group < (warp + 1) * per_warp; ++group) {"
    )
    assert loop.count(old_loop) == 1
    loop = loop.replace(
        old_loop,
        "#pragma unroll 1\n"
        "  for (int chunk = warp * per_warp; chunk < (warp+1)*per_warp; "
        "chunk += NAcc) {\n"
        "#pragma unroll\n"
        "    for (int chain = 0; chain < NAcc; ++chain) {\n"
        "      const int group = chunk + chain;",
    )
    # Static unrolling keeps every accumulator in registers. Dynamic array
    # indexing from the first attempt allocated thread-local memory instead.
    body = loop + "  }\n" + body[loop_end:]
    old = "      partials[p][warp][r * 32 + qp * 8 + c] = accum[p][i];"
    assert body.count(old) == 1
    body = body.replace(
        old,
        """      float sum = accum[p][0][i];
#pragma unroll
      for (int acc = 1; acc < NAcc; ++acc) sum += accum[p][acc][i];
      partials[p][warp][r * 32 + qp * 8 + c] = sum;""",
    )
    text = text[:begin] + body + text[end:]
    text = text.replace("paired_gate_kernel<false>", "paired_gate_kernel<false, 2>")
    text = text.replace("paired_gate_kernel<true>", "paired_gate_kernel<false, 4>")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "paired_accumulators.cu"
    source = args.source_root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    text = generate(source)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    load(
        name="qpn2_paired_static_acc_screen",
        sources=[str(path)],
        extra_include_paths=[str(source.parent)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
