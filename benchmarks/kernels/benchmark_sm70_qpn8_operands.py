# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Literal source anchors retain their production spelling.
# ruff: noqa: E501
"""Build isolated FP8 activation-layout and post-accumulation scale probes.

The latter changes rounding and requires model-level teacher-forcing admission.
It does not touch draft projections or LM heads, or install a serving route.
"""

import argparse
import os
from pathlib import Path

VARIANTS = ["control", "packed", "postscale", "packed_postscale"]


def generate(root):
    source = (root / "csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu").read_text()
    helpers = source[
        source.index("__device__ __forceinline__ void fp8x8_to_half2x4") : source.index(
            "__global__ void fp8_qpn8_prepack_sm70_kernel"
        )
    ]
    macro = source[
        source.index("#define VLLM_SM70_MMA_8N8K4") : source.index(
            "__global__ void fp8_qpn8_ba_split_copy_sm70_kernel"
        )
    ]
    start = source.index("__global__ void fp8_qpn8_sm70_kernel(")
    start = source.rfind("template <", 0, start)
    end = source.index("\nvoid launch_fp8_qpn8_sm70", start)
    end = source.rfind("\ntemplate <", start, end)
    original = source[start:end]
    assert original.count("__global__") == 1
    copies = []
    for variant in VARIANTS:
        kernel = original.replace("fp8_qpn8_sm70_kernel", "qpn8_" + variant)
        if "packed" in variant:
            old = (
                "const half* input_row = input + static_cast<size_t>(input_row_idx) * k;\n"
                "        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);\n"
                "        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);"
            )
            assert kernel.count(old) == 1
            kernel = kernel.replace(
                old,
                (
                    "const half* input_row = input + static_cast<size_t>(group) * 128 + input_row_idx * 16;\n"
                    "        input01 = *reinterpret_cast<const uint4*>(input_row);\n"
                    "        input23 = *reinterpret_cast<const uint4*>(input_row + 8);"
                ),
            )
            old_ba = "const float2 x = __half22float2(__ldg(input2 + pair));"
            assert kernel.count(old_ba) == 1
            kernel = kernel.replace(
                old_ba,
                (
                    "const float2 x = __half22float2(__ldg(reinterpret_cast<const half2*>(input) + "
                    "static_cast<size_t>(pair / 8) * 64 + token * 8 + pair % 8));"
                ),
            )
        if "postscale" in variant:
            multiply = "weights[i] = __hmul2(weights[i], scale2);"
            assert kernel.count(multiply) == 1
            kernel = kernel.replace(
                multiply, "// Per-channel scale follows the FP32 reduction."
            )
            marker = "    if constexpr (M1Only) {\n      const int output_col = tile * 32 + element;"
            assert kernel.count(marker) == 1
            kernel = kernel.replace(
                marker,
                "    value *= __half2float(__ldg(group_scales + tile * 32 + (element & 31)));\n"
                + marker,
            )
        copies.append(kernel)
    launches = []
    for index, variant in enumerate(VARIANTS):
        launches.append(f"""    case {index}:
      if (kind == 0) qpn8_{variant}<16,2,true,false,false,true,true>
        <<<224,512,0,stream>>>(w,s,x,y,z,bw,nullptr,b,a,24,2560,4096,k,8,true);
      else if (kind == 1) qpn8_{variant}<12,2,true,false>
        <<<160,384,0,stream>>>(w,s,x,y,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,n,k,8,true);
      else qpn8_{variant}<16,2,true,false>
        <<<n/32,512,0,stream>>>(w,s,x,y,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,n,k,8,true);
      break;""")
    wrapper = Path(__file__).with_suffix(".cuh").read_text()
    return (
        "#include <torch/all.h>\n#include <torch/library.h>\n"
        "#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAGuard.h>\n"
        "#include <c10/cuda/CUDAException.h>\n#include <cuda_fp16.h>\n"
        "namespace {\n"
        + helpers
        + macro
        + "\n".join(copies)
        + wrapper.replace("// GENERATED_LAUNCHES", "\n".join(launches))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generated = args.output / "fp8_operands.cu"
    generated.write_text(generate(args.source_root))
    if args.generate_only:
        return
    from torch.utils.cpp_extension import load

    os.environ.setdefault("MAX_JOBS", "1")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    load(
        name="qpn8_operands700",
        sources=[str(generated)],
        extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
        extra_ldflags=["-Wl,-Bsymbolic"],
        build_directory=str(args.output),
        is_python_module=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
