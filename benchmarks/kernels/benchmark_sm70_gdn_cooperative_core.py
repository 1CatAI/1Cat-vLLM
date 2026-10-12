# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a research BV2 warp-local delta/norm kernel with a resident grid.

Retain 768 independent two-column warp tasks. Group them into 80 resident
CTAs, cache normalized Q/K per CTA, and parallelize 96 output-head rows after
one cooperative grid barrier. This does not install a serving implementation.
"""

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load

SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset=16; offset; offset/=2)
    value += __shfl_xor_sync(0xffffffff,value,offset);
  return value;
}
__device__ __forceinline__ float four_sum(const float* values) {
  return (values[0]+values[1])+(values[2]+values[3]);
}
template <bool CacheQK>
__global__ __launch_bounds__(512,1) void cooperative_core(
    const half* qkv, const half* z, const float* g, const float* beta,
    const half* weight, float* states, const int* indices,
    const int* accepted, const int* cu, half* raw, half* output, int stride) {
  __shared__ __align__(16) float qk_cache[48][128];
  __shared__ float decay_cache[24], beta_cache[24];
  const int lane=threadIdx.x%32, warp=threadIdx.x/32;
  const int missing=3-(blockIdx.x%4);
  const int tokens=cu[1]-cu[0];
  if constexpr (CacheQK) {
    for(int row=warp;row<48;row+=16) {
      const int local_group=row/16, token=(row%16)/2, which=row%2;
      const int group=local_group+(local_group>=missing);
      float value[4], squares[4];
#pragma unroll
      for(int i=0;i<4;++i) {
        value[i]=token<tokens ? __half2float(
            qkv[token*2560+which*512+group*128+lane*4+i]) : 0.f;
        squares[i]=value[i]*value[i];
      }
      const float length=sqrtf(warp_sum(four_sum(squares))+1e-6f);
#pragma unroll
      for(int i=0;i<4;++i) {
        float normalized=__fdividef(value[i],length);
        if(which==0) normalized*=0.08838834764831845f;
        qk_cache[row][lane*4+i]=normalized;
      }
    }
    if(threadIdx.x<24) {
      const int t=threadIdx.x/3, slot=threadIdx.x%3;
      const int head=blockIdx.x%4+slot*4;
      decay_cache[threadIdx.x]=__expf(g[t*12+head]);
      beta_cache[threadIdx.x]=beta[t*12+head];
    }
  }
  __syncthreads();
  const int task=blockIdx.x+warp*80;
  const int start_slot=indices[accepted[0]-1];
  if(task<768 && start_slot>=0 && tokens>0) {
    const int head=task%12, vbase=(task/12)*2;
    const int group=head/3, local_group=group-(group>missing);
    float matrix[2][4];
#pragma unroll
    for(int v=0;v<2;++v) {
#pragma unroll
      for(int i=0;i<4;++i)
        matrix[v][i]=states[static_cast<size_t>(start_slot)*stride+
            head*16384+(vbase+v)*128+lane*4+i];
    }
    for(int t=0;t<tokens;++t) {
      float q[4], k[4];
      float decay, multiplier;
      if constexpr (CacheQK) {
        const int row=(local_group*8+t)*2;
#pragma unroll
        for(int i=0;i<4;++i) {
          q[i]=qk_cache[row][lane*4+i];
          k[i]=qk_cache[row+1][lane*4+i];
        }
        decay=decay_cache[t*3+head/4];
        multiplier=beta_cache[t*3+head/4];
      } else {
        float qs[4],ks[4];
#pragma unroll
        for(int i=0;i<4;++i) {
          q[i]=__half2float(qkv[t*2560+group*128+lane*4+i]);
          k[i]=__half2float(qkv[t*2560+512+group*128+lane*4+i]);
          qs[i]=q[i]*q[i]; ks[i]=k[i]*k[i];
        }
        const float qlength=sqrtf(warp_sum(four_sum(qs))+1e-6f);
        const float klength=sqrtf(warp_sum(four_sum(ks))+1e-6f);
#pragma unroll
        for(int i=0;i<4;++i) {
          q[i]=__fdividef(q[i],qlength)*0.08838834764831845f;
          k[i]=__fdividef(k[i],klength);
        }
        decay=__expf(g[t*12+head]); multiplier=beta[t*12+head];
      }
      const int target=indices[t];
#pragma unroll
      for(int v=0;v<2;++v) {
        float products[4];
#pragma unroll
        for(int i=0;i<4;++i) {
          matrix[v][i]=__fmul_rn(matrix[v][i],decay);
          products[i]=matrix[v][i]*k[i];
        }
        const float dot=warp_sum(four_sum(products));
        float delta=__half2float(qkv[t*2560+1024+head*128+vbase+v])-dot;
        delta*=multiplier;
#pragma unroll
        for(int i=0;i<4;++i) {
          matrix[v][i]=fmaf(delta,k[i],matrix[v][i]);
          products[i]=matrix[v][i]*q[i];
          if(target>=0)
            states[static_cast<size_t>(target)*stride+head*16384+
                (vbase+v)*128+lane*4+i]=matrix[v][i];
        }
        const float value=warp_sum(four_sum(products));
        if(lane==0) raw[t*1536+head*128+vbase+v]=__float2half_rn(value);
      }
    }
  }
  // Every CTA is admitted resident before launch. This is one phase boundary,
  // not polling a producer grid that may not be scheduled yet.
  cooperative_groups::this_grid().sync();
  const int row=blockIdx.x+warp*80;
  if(row<96) {
    float values[4],squares[4];
#pragma unroll
    for(int i=0;i<4;++i) {
      values[i]=__half2float(raw[row*128+lane*4+i]);
      squares[i]=values[i]*values[i];
    }
    const float inverse=rsqrtf(warp_sum(four_sum(squares))/128.f+1e-6f);
#pragma unroll
    for(int i=0;i<4;++i) {
      const int col=lane*4+i;
      const float gate=__half2float(z[row*128+col]);
      const float sigmoid=__fdividef(1.f,1.f+__expf(-gate));
      float value=values[i]*inverse*__half2float(weight[col]);
      value*=gate*sigmoid;
      output[row*128+col]=__float2half_rn(value);
    }
  }
}
void launch(torch::Tensor output, torch::Tensor raw, torch::Tensor qkv,
    torch::Tensor z, torch::Tensor g, torch::Tensor beta, torch::Tensor weight,
    torch::Tensor state, torch::Tensor indices, torch::Tensor accepted,
    torch::Tensor cu, bool cache) {
  TORCH_CHECK(qkv.sizes()==torch::IntArrayRef({8,2560}) && qkv.is_contiguous());
  TORCH_CHECK(output.sizes()==torch::IntArrayRef({8,1536}) && output.is_contiguous());
  TORCH_CHECK(state.dim()==4 && state.size(1)==12 && state.size(2)==128 &&
              state.size(3)==128 && state.is_contiguous() &&
              state.scalar_type()==torch::kFloat32);
  TORCH_CHECK(indices.numel()==8 && accepted.numel()==1 && cu.numel()==2);
  const half* q=reinterpret_cast<const half*>(qkv.data_ptr<at::Half>());
  const half* zz=reinterpret_cast<const half*>(z.data_ptr<at::Half>());
  const half* w=reinterpret_cast<const half*>(weight.data_ptr<at::Half>());
  const float* gg=g.data_ptr<float>(); const float* bb=beta.data_ptr<float>();
  float* s=state.data_ptr<float>(); const int* ix=indices.data_ptr<int>();
  const int* acc=accepted.data_ptr<int>(); const int* lengths=cu.data_ptr<int>();
  half* r=reinterpret_cast<half*>(raw.data_ptr<at::Half>());
  half* y=reinterpret_cast<half*>(output.data_ptr<at::Half>());
  int stride=state.stride(0), device;
  C10_CUDA_CHECK(cudaGetDevice(&device)); cudaDeviceProp properties;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
  auto kernel=cache ? cooperative_core<true> : cooperative_core<false>;
  int resident=0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident,kernel,512,0));
  TORCH_CHECK(properties.major==7 && properties.minor==0 &&
              properties.cooperativeLaunch &&
              resident*properties.multiProcessorCount>=80,
              "Requires a resident cooperative SM70 grid");
  void* args[]={&q,&zz,&gg,&bb,&w,&s,&ix,&acc,&lengths,&r,&y,&stride};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel),
      dim3(80),dim3(512),args,0,at::cuda::getCurrentCUDAStream()));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("launch",&launch); }
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "cooperative_core.cu"
    if not source.exists() or source.read_text() != SOURCE:
        source.write_text(SOURCE)
    load(
        name="gdn_cooperative_core_screen",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
