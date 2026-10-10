# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source anchors and CUDA strings retain their original spelling.
# ruff: noqa: E501
"""Screen an explicit two-group M8 operand pipeline against wheel kernels.

Raw codes and scales for the next iteration issue before current decoding and
MMA. Volatile loads make that ordering visible in SASS. No arithmetic, split,
layout, accumulation chain or output rounding is changed. Research-only.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_product_lookup import function_end
from torch.utils.cpp_extension import load


def kernel(text, name):
    begin = text.rindex("template <", 0, text.index("void " + name + "("))
    return text[begin : function_end(text, begin)]


def generate(root):
    source = (root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu").read_text()
    helpers = source[
        source.index("__device__ __forceinline__ half2 fp8e4m3") : source.index(
            "// Batch kernels share"
        )
    ]
    macro = source[
        source.index("#define VLLM_SM70_QPN2_MMA") : source.index(
            "// Four row tiles reuse"
        )
    ]
    gate = kernel(source, "nvfp4_qpn2_m8_paired_gated_sm70_kernel")
    gate = gate.replace(
        "nvfp4_qpn2_m8_paired_gated_sm70_kernel", "qpn2_m8_gate_pipeline"
    )
    gate = gate.replace("256, 3)", "256, 2)")
    begin = gate.index("  Nvfp4PairReader<")
    end = gate.index("  float accum", begin)
    gate = (
        gate[:begin]
        + r"""
  const uint8_t* base[2]={
      codes+static_cast<size_t>(blockIdx.x)*groups*288+lane*8,
      codes+static_cast<size_t>(blockIdx.x+hidden/32)*groups*288+lane*8};
  const int first=warp*per_warp,last=(warp+1)*per_warp;
  RawOperand current[2]={load_raw(base[0],lane,first),load_raw(base[1],lane,first)};
  RawOperand next[2]={load_raw(base[0],lane,first+1),load_raw(base[1],lane,first+1)};
"""
        + gate[end:]
    )
    gate = gate.replace(
        "#pragma unroll 4\n  for (int group", "#pragma unroll 1\n  for (int group"
    )
    marker = "    const half* a = input +"
    begin = gate.index(marker)
    gate = (
        gate[:begin]
        + r"""
    RawOperand future[2]{};
    if(group+2<last) {
      future[0]=load_raw(base[0],lane,group+2);
      future[1]=load_raw(base[1],lane,group+2);
    }
"""
        + gate[begin:]
    )
    gate = gate.replace(
        "readers[p].load(group, weights[p]);",
        "decode_raw(current[p],global,weights[p]);",
    )
    gate = gate.replace(
        "readers[p].load(group, weights);", "decode_raw(current[p],global,weights);"
    )
    marker = "\n  }\n#pragma unroll\n  for (int p ="
    assert gate.count(marker) == 1
    gate = gate.replace(
        marker,
        "\n    current[0]=next[0];current[1]=next[1];\n"
        "    next[0]=future[0];next[1]=future[1];" + marker,
    )

    down = kernel(source, "nvfp4_qpn2_sm70_kernel")
    down = down.replace("nvfp4_qpn2_sm70_kernel", "qpn2_m8_down_pipeline")
    down = down.replace("__global__ void", "__global__ __launch_bounds__(512,2) void")
    begin = down.index("  const Nvfp4Qpn2CodeReader<")
    end = down.index("\n\n  float accum", begin)
    down = (
        down[:begin]
        + r"""
  const uint8_t* base=codes+static_cast<size_t>(tile)*groups_k16*288+lane*8;
  RawOperand current=load_raw(base,lane,group_begin);
  RawOperand next=load_raw(base,lane,group_begin+1);
"""
        + down[end:]
    )
    down = down.replace(
        "#pragma unroll 4\n  for (int group", "#pragma unroll 1\n  for (int group"
    )
    begin = down.index("    const uint2 packed =")
    end = down.index("\n    const unsigned* b =", begin)
    down = (
        down[:begin]
        + r"""
    RawOperand future{};
    if(group+2<group_begin+groups_per_warp) future=load_raw(base,lane,group+2);
    half2 weights[8];
    decode_raw(current,global_scale,weights);
"""
        + down[end:]
    )
    end = function_end(down, down.index("  for (int group")) - 1
    down = down[:end] + "    current=next;next=future;\n  " + down[end:]
    return (
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
namespace {
constexpr int kQpn2RowsPerCta=8;
"""
        + helpers
        + macro
        + r"""
struct RawOperand {uint2 codes;unsigned scale;};
__device__ __forceinline__ RawOperand load_raw(const uint8_t* base,int lane,int group) {
  RawOperand value;
  const uint8_t* address=base+static_cast<size_t>(group)*288;
  asm volatile("ld.global.cs.v2.u32 {%0,%1},[%2];" :
      "=r"(value.codes.x),"=r"(value.codes.y) : "l"(address) : "memory");
  const uint8_t* sf=address-lane*8+256+lane;
  asm volatile("ld.global.nc.u8 %0,[%1];" : "=r"(value.scale) : "l"(sf) : "memory");
  return value;
}
__device__ __forceinline__ void decode_raw(RawOperand raw,float global,half2* weights) {
  const half2 scale=nvfp4_effective_scale(static_cast<uint8_t>(raw.scale),global);
  dequant_e2m1x8(raw.codes.x,scale,weights);
  dequant_e2m1x8(raw.codes.y,scale,weights+4);
}
"""
        + gate
        + down
        + r"""
}
void launch_pair(torch::Tensor output,torch::Tensor input,torch::Tensor codes,
    torch::Tensor scales,double global,int mode) {
  TORCH_CHECK(input.sizes()==torch::IntArrayRef({8,5120}) && input.is_contiguous());
  TORCH_CHECK(output.sizes()==torch::IntArrayRef({8,4352}) && codes.is_contiguous());
  qpn2_m8_gate_pipeline<false><<<136,256,0,at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(),scales.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),4352,5120,global);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void launch_down(torch::Tensor output,torch::Tensor input,torch::Tensor codes,
    torch::Tensor scales,double global) {
  TORCH_CHECK(input.sizes()==torch::IntArrayRef({8,4352}) && input.is_contiguous());
  TORCH_CHECK(output.sizes()==torch::IntArrayRef({8,5120}) && codes.is_contiguous());
  qpn2_m8_down_pipeline<16,2,1,false,false,true><<<160,512,0,at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(),scales.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),5120,4352,8,global);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("pair",&launch_pair);m.def("down",&launch_down);}
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "m8_pipeline.cu"
    output.write_text(generate(args.source_root))
    load(
        name="sm70_qpn2_m8_pipeline_screen",
        sources=[str(output)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
