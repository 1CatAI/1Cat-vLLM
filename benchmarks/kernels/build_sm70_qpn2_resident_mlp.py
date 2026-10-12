# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source anchors and CUDA strings retain their original spelling.
# ruff: noqa: E501
"""Screen chunk-ready gate/down tasks within one resident SM70 MLP kernel.

Every producer is cooperatively admitted. A down warp waits only for its next
32-column gate chunk; there is no whole-grid gate/down barrier. Split order,
MMA chains, gate FP16 rounding and final output rounding remain unchanged.
This extension is research-only, not a production runtime dependency.
"""

import argparse
from pathlib import Path

from build_sm70_qpn2_m8_pipeline import kernel
from torch.utils.cpp_extension import load


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
    gate = kernel(source, "nvfp4_qpn2_gated_sm70_kernel")
    gate = gate.replace("nvfp4_qpn2_gated_sm70_kernel", "gate_task")
    gate = gate.replace("__global__ void", "__device__ __forceinline__ void")
    gate = gate.replace(
        "int k, int m, float global_scale) {",
        "int k, int m, float global_scale,float* scratch) {",
    )
    gate = gate.replace(
        "__shared__ float partials[2][SplitK][RowTiles * 256];",
        "auto partials=reinterpret_cast<float (*)[SplitK][RowTiles*256]>(scratch);",
    )
    down = kernel(source, "nvfp4_qpn2_sm70_kernel")
    down = down.replace("nvfp4_qpn2_sm70_kernel", "down_task")
    down = down.replace("__global__ void", "__device__ __forceinline__ void")
    down = down.replace(
        "int m, float global_scale) {",
        "int m, float global_scale,float* scratch,const int* ready,int generation) {",
    )
    down = down.replace(
        "__shared__ float partials[SplitK][RowTiles * 256];",
        "auto partials=reinterpret_cast<float (*)[RowTiles*256]>(scratch);",
    )
    down = down.replace(
        "    const unsigned* b = reinterpret_cast<const unsigned*>(weights);",
        "    wait_chunk(ready,group/2,generation);\n"
        "    const unsigned* b = reinterpret_cast<const unsigned*>(weights);",
    )
    down = down.replace(
        "*reinterpret_cast<const uint4*>(input_row + group * 16)",
        "load_activation(input_row+group*16)",
    ).replace(
        "*reinterpret_cast<const uint4*>(input_row + group * 16 + 8)",
        "load_activation(input_row+group*16+8)",
    )
    assert "__global__" not in gate + down
    assert "wait_chunk(ready,group/2,generation)" in down
    return (
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include "nvfp4_qpn2_layout.cuh"
namespace {
constexpr int kQpn2RowsPerCta=8;
__device__ __forceinline__ void wait_chunk(const int* ready,int chunk,int generation) {
  int value;
  do {
    asm volatile("ld.global.cg.u32 %0,[%1];" : "=r"(value):"l"(ready+chunk):"memory");
  } while(value!=generation);
}
__device__ __forceinline__ uint4 load_activation(const half* address) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];" :
      "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(address):"memory");
  return v;
}
"""
        + helpers
        + macro
        + gate
        + down
        + r"""
__global__ __launch_bounds__(512,2) void resident_mlp(
    const uint8_t* gate_codes,const uint8_t* down_codes,const half* input,
    half* intermediate,half* output,int* ready,float gate_scale,float down_scale) {
  __shared__ float scratch[4096];
  __shared__ int generation;
  if(threadIdx.x==0) {
    generation=ready[136+blockIdx.x]+1;
    ready[136+blockIdx.x]=generation;
  }
  __syncthreads();
  if(blockIdx.x<136) {
    gate_task<8,1,1,false,true>(gate_codes,nullptr,input,intermediate,
        4352,5120,8,gate_scale,scratch);
    __threadfence();
    __syncthreads();
    if(threadIdx.x==0) {
      asm volatile("st.global.wb.u32 [%0],%1;" ::"l"(ready+blockIdx.x),"r"(generation):"memory");
    }
  }
  // Distinct CTAs can enter down as soon as their own gate task completes.
  // All producers are already resident, so chunk polling cannot starve them.
  down_task<16,2,1,false,false,true>(down_codes,nullptr,intermediate,output,
      5120,4352,8,down_scale,scratch,ready,generation);
}
}
void mlp(torch::Tensor output,torch::Tensor intermediate,torch::Tensor input,
    torch::Tensor gate_codes,torch::Tensor down_codes,torch::Tensor ready,
    double gate_scale,double down_scale) {
  TORCH_CHECK(input.sizes()==torch::IntArrayRef({8,5120}) && input.is_contiguous());
  TORCH_CHECK(intermediate.sizes()==torch::IntArrayRef({8,4352}) && intermediate.is_contiguous());
  TORCH_CHECK(output.sizes()==torch::IntArrayRef({8,5120}) && output.is_contiguous());
  TORCH_CHECK(gate_codes.is_contiguous() && down_codes.is_contiguous());
  TORCH_CHECK(ready.numel()==296 && ready.scalar_type()==torch::kInt32);
  int device,resident;cudaDeviceProp properties;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,resident_mlp,512,0));
  TORCH_CHECK(properties.major==7 && properties.minor==0 && properties.cooperativeLaunch &&
              resident*properties.multiProcessorCount>=160,
              "Every gate producer must be admitted resident");
  auto* g=gate_codes.data_ptr<uint8_t>();auto* d=down_codes.data_ptr<uint8_t>();
  auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* mid=reinterpret_cast<half*>(intermediate.data_ptr<at::Half>());
  auto* y=reinterpret_cast<half*>(output.data_ptr<at::Half>());
  auto* flags=ready.data_ptr<int>();float gs=gate_scale,ds=down_scale;
  void* args[]={&g,&d,&x,&mid,&y,&flags,&gs,&ds};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(resident_mlp),dim3(160),
      dim3(512),args,0,at::cuda::getCurrentCUDAStream()));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("mlp",&mlp);}
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "resident_mlp.cu"
    output.write_text(generate(args.source_root))
    load(
        name="sm70_qpn2_resident_mlp_screen",
        sources=[str(output)],
        extra_include_paths=[str(args.source_root / "csrc/sm70_turbomind/ops")],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
