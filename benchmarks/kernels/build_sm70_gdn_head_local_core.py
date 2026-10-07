# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen BV2 delta/gated norm with head-local partial packets.

Keep the 768 independent two-column warp tasks in 96 eight-warp CTAs.
Eight CTAs cooperate per head without a whole-grid barrier. Cooperative
launch admission guarantees residency; a normal-grid spin is not assumed safe.
Convolution stays unchanged in this first screen.
"""

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load

SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float head_warp_sum(float value) {
#pragma unroll
  for(int offset=16;offset;offset/=2) {
    unsigned bits;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
      : "=r"(bits) : "r"(__float_as_uint(value)),"r"(offset));
    value += __uint_as_float(bits);
  }
  return value;
}
__device__ __forceinline__ float head_four_sum(const float* values) {
  return (values[0]+values[1])+(values[2]+values[3]);
}
__global__ __launch_bounds__(256,1) void head_local_core(
    const half* qkv,const half* z,const float* g,const float* beta,
    const half* weight,float* states,const int* indices,const int* accepted,
    const int* cu,half* raw,half* output,int stride,
    uint64_t* packets,uint32_t* generations) {
  __shared__ __align__(16) float qk[16][128];
  __shared__ half values[8][16];
  __shared__ float decay[8],multiplier[8];
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  const int head=blockIdx.x/8,tile=blockIdx.x%8,group=head/3;
  const int vbase=tile*16+warp*2;
  const int tokens=cu[1]-cu[0];
  const uint32_t generation=generations[head]+1;
  for(int which=0;which<2;++which) {
    float vector[4],squares[4];
#pragma unroll
    for(int i=0;i<4;++i) {
      vector[i]=warp<tokens ? __half2float(
        qkv[warp*2560+which*512+group*128+lane*4+i]) : 0.f;
      squares[i]=vector[i]*vector[i];
    }
    const float length=sqrtf(head_warp_sum(head_four_sum(squares))+1e-6f);
#pragma unroll
    for(int i=0;i<4;++i) {
      float normalized=__fdividef(vector[i],length);
      if(which==0) normalized*=0.08838834764831845f;
      qk[warp*2+which][lane*4+i]=normalized;
    }
  }
  if(threadIdx.x<8) {
    decay[threadIdx.x]=__expf(g[threadIdx.x*12+head]);
    multiplier[threadIdx.x]=beta[threadIdx.x*12+head];
  }
  if(lane==0) {
#pragma unroll
    for(int t=0;t<8;++t) {
      values[t][warp*2]=__float2half(0.f);
      values[t][warp*2+1]=__float2half(0.f);
      raw[t*1536+head*128+vbase]=__float2half(0.f);
      raw[t*1536+head*128+vbase+1]=__float2half(0.f);
    }
  }
  __syncthreads();
  const int start=indices[accepted[0]-1];
  if(start>=0 && tokens>0) {
    float matrix[2][4];
#pragma unroll
    for(int v=0;v<2;++v) {
#pragma unroll
      for(int i=0;i<4;++i) matrix[v][i]=
        states[static_cast<size_t>(start)*stride+head*16384+
               (vbase+v)*128+lane*4+i];
    }
    for(int t=0;t<tokens;++t) {
      float q[4],k[4];
#pragma unroll
      for(int i=0;i<4;++i) {
        q[i]=qk[t*2][lane*4+i]; k[i]=qk[t*2+1][lane*4+i];
      }
      const int target=indices[t];
#pragma unroll
      for(int v=0;v<2;++v) {
        float products[4];
#pragma unroll
        for(int i=0;i<4;++i) {
          matrix[v][i]=__fmul_rn(matrix[v][i],decay[t]);
          products[i]=matrix[v][i]*k[i];
        }
        const float dot=head_warp_sum(head_four_sum(products));
        float delta=__half2float(qkv[t*2560+1024+head*128+vbase+v])-dot;
        delta*=multiplier[t];
#pragma unroll
        for(int i=0;i<4;++i) {
          matrix[v][i]=fmaf(delta,k[i],matrix[v][i]);
          products[i]=matrix[v][i]*q[i];
          if(target>=0) states[static_cast<size_t>(target)*stride+head*16384+
            (vbase+v)*128+lane*4+i]=matrix[v][i];
        }
        const half value=__float2half_rn(head_warp_sum(head_four_sum(products)));
        if(lane==0) {
          values[t][warp*2+v]=value;
          raw[t*1536+head*128+vbase+v]=value;
        }
      }
    }
  }
  __syncthreads();
  // A warp owns one output-token row; only its head's eight CTAs cooperate.
  const float value=lane<16 ? __half2float(values[warp][lane]) : 0.f;
  const float variance=head_warp_sum(value*value);
  uint64_t* row=packets+(head*8+warp)*8;
  if(lane==0) {
    const uint64_t packet=(uint64_t(generation)<<32)|__float_as_uint(variance);
    asm volatile("st.volatile.global.u64 [%0], %1;" ::
      "l"(row+tile),"l"(packet):"memory");
  }
  float partial=0.f;
  if(lane<8) {
    uint64_t packet;
    do {
      asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(packet) :
        "l"(row+lane):"memory");
    } while(uint32_t(packet>>32)!=generation);
    partial=__uint_as_float(uint32_t(packet));
  }
  __syncwarp();
  float total=0.f;
#pragma unroll
  for(int part=0;part<8;++part) {
    unsigned bits;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;"
      : "=r"(bits) : "r"(__float_as_uint(partial)),"r"(part));
    total+=__uint_as_float(bits);
  }
  const float inverse=rsqrtf(total/128.f+1e-6f);
  if(lane<16) {
    const int col=tile*16+lane;
    const float gate=__half2float(z[warp*1536+head*128+col]);
    const float sigmoid=__fdividef(1.f,1.f+__expf(-gate));
    float normalized=value*inverse*__half2float(weight[col]);
    normalized*=gate*sigmoid;
    output[warp*1536+head*128+col]=__float2half_rn(normalized);
  }
  __syncthreads();
  if(tile==0 && threadIdx.x==0) generations[head]=generation;
}
void launch(torch::Tensor output,torch::Tensor raw,torch::Tensor qkv,
    torch::Tensor z,torch::Tensor g,torch::Tensor beta,torch::Tensor weight,
    torch::Tensor state,torch::Tensor indices,torch::Tensor accepted,
    torch::Tensor cu,torch::Tensor packets,torch::Tensor generations) {
  TORCH_CHECK(qkv.sizes()==torch::IntArrayRef({8,2560}) && qkv.is_contiguous());
  TORCH_CHECK(output.sizes()==torch::IntArrayRef({8,1536}) && output.is_contiguous());
  TORCH_CHECK(state.dim()==4 && state.size(1)==12 && state.size(2)==128 &&
              state.size(3)==128 && state.is_contiguous() &&
              state.scalar_type()==torch::kFloat32);
  TORCH_CHECK(indices.numel()==8 && accepted.numel()==1 && cu.numel()==2);
  TORCH_CHECK(packets.numel()==12*8*8 && generations.numel()==12);
  const half* q=reinterpret_cast<const half*>(qkv.data_ptr<at::Half>());
  const half* zz=reinterpret_cast<const half*>(z.data_ptr<at::Half>());
  const half* w=reinterpret_cast<const half*>(weight.data_ptr<at::Half>());
  const float* gg=g.data_ptr<float>();const float* bb=beta.data_ptr<float>();
  float* s=state.data_ptr<float>();const int* ix=indices.data_ptr<int>();
  const int* acc=accepted.data_ptr<int>();const int* lengths=cu.data_ptr<int>();
  half* r=reinterpret_cast<half*>(raw.data_ptr<at::Half>());
  half* y=reinterpret_cast<half*>(output.data_ptr<at::Half>());
  auto* flags=reinterpret_cast<uint64_t*>(packets.data_ptr<int64_t>());
  auto* counter=reinterpret_cast<uint32_t*>(generations.data_ptr<int>());
  int stride=state.stride(0),device,resident;
  C10_CUDA_CHECK(cudaGetDevice(&device));cudaDeviceProp properties;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &resident,head_local_core,256,0));
  TORCH_CHECK(properties.major==7 && properties.minor==0 &&
    properties.cooperativeLaunch && resident*properties.multiProcessorCount>=96,
    "Requires resident head-local SM70 tasks");
  void* args[]={&q,&zz,&gg,&bb,&w,&s,&ix,&acc,&lengths,&r,&y,&stride,&flags,&counter};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
    reinterpret_cast<void*>(head_local_core),dim3(96),dim3(256),args,0,
    at::cuda::getCurrentCUDAStream()));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("launch",&launch); }
"""


def vector_state_source():
    text = SOURCE
    old = """    for(int v=0;v<2;++v) {
#pragma unroll
      for(int i=0;i<4;++i) matrix[v][i]=
        states[static_cast<size_t>(start)*stride+head*16384+
               (vbase+v)*128+lane*4+i];
    }"""
    new = """    for(int v=0;v<2;++v) {
      const float4 packed=*reinterpret_cast<const float4*>(
        states+static_cast<size_t>(start)*stride+head*16384+
               (vbase+v)*128+lane*4);
      matrix[v][0]=packed.x;matrix[v][1]=packed.y;
      matrix[v][2]=packed.z;matrix[v][3]=packed.w;
    }"""
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = """\
          if(target>=0) states[static_cast<size_t>(target)*stride+head*16384+
            (vbase+v)*128+lane*4+i]=matrix[v][i];
        }
        const half value="""
    new = """        }
        if(target>=0) *reinterpret_cast<float4*>(
          states+static_cast<size_t>(target)*stride+head*16384+
                 (vbase+v)*128+lane*4)=make_float4(
          matrix[v][0],matrix[v][1],matrix[v][2],matrix[v][3]);
        const half value="""
    assert text.count(old) == 1
    return text.replace(old, new)


CONV_HELPERS = r"""
__device__ __forceinline__ half head_conv_value(const half* input,
    const half* old, const half* weights, int t, int feature, int old_feature) {
  float sum=0.f;
#pragma unroll
  for(int tap=0;tap<4;++tap) {
    const int token=t+tap-3;
    const half value=token<0 ? old[(token+3)*128+old_feature] :
      input[token*2560+feature];
    sum=__fadd_rn(sum,__half2float(__hmul(value,weights[feature*4+tap])));
  }
  return __float2half_rn(__fdividef(sum,1.f+__expf(-sum)));
}
"""


CONV_PRELOAD = r"""
  const int history_slot=history_indices[0],previous=accepted[0]-1;
  // History copies precede the generation packet, so the Q/K owner can
  // overwrite the original history only after every group reader copied it.
  for(int i=threadIdx.x;i<2*3*128;i+=256) {
    const int which=i/(3*128),tap=(i/128)%3,col=i%128;
    const int feature=which*512+group*128+col;
    old_qk[which][tap][col]=history[history_slot*hs+feature*hd+
      (previous+tap)*ht];
  }
  if(lane<3) {
#pragma unroll
    for(int v=0;v<2;++v) old_v[lane][warp*2+v]=
      history[history_slot*hs+(1024+head*128+vbase+v)*hd+(previous+lane)*ht];
  }
  __syncthreads();
  uint32_t* readers=conv_ready+group*24;
  const uint32_t history_generation=generations[group*3]+1;
  if(threadIdx.x==0) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::
      "l"(readers+(head%3)*8+tile),"r"(history_generation):"memory");
  }
  if(head%3==0 && tile==0 && threadIdx.x<24) {
    uint32_t ready;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(ready):
        "l"(readers+threadIdx.x):"memory");
    } while(ready!=history_generation);
  }
  __syncthreads();
  if(head%3==0 && tile==0) {
#pragma unroll
    for(int which=0;which<2;++which) {
#pragma unroll
      for(int i=0;i<4;++i) {
        const int col=lane*4+i,feature=which*512+group*128+col;
        if(warp==0) {
          history[history_slot*hs+feature*hd]=old_qk[which][1][col];
          history[history_slot*hs+feature*hd+ht]=old_qk[which][2][col];
        }
        if(warp<tokens) history[history_slot*hs+feature*hd+(warp+2)*ht]=
          qkv[warp*2560+feature];
      }
    }
  }
  if(lane==0) {
#pragma unroll
    for(int v=0;v<2;++v) {
      const int feature=1024+head*128+vbase+v;
      history[history_slot*hs+feature*hd]=old_v[1][warp*2+v];
      history[history_slot*hs+feature*hd+ht]=old_v[2][warp*2+v];
      for(int t=0;t<tokens;++t)
        history[history_slot*hs+feature*hd+(t+2)*ht]=qkv[t*2560+feature];
    }
  }
"""


def convolution_source():
    text = vector_state_source()
    marker = "__global__ __launch_bounds__(256,1)"
    assert text.count(marker) == 1
    text = text.replace(marker, CONV_HELPERS + marker)
    text = text.replace(
        "uint64_t* packets,uint32_t* generations) {",
        "uint64_t* packets,uint32_t* generations, const half* conv,\n"
        "    half* history,const int* history_indices,const half* a,\n"
        "    const half* b,const float* a_log,const half* bias,\n"
        "    uint32_t* conv_ready,int hs,int hd,int ht) {",
        1,
    )
    text = text.replace(
        "  __shared__ float decay[8],multiplier[8];",
        "  __shared__ float decay[8],multiplier[8];\n"
        "  __shared__ half old_qk[2][3][128],old_v[3][16];",
        1,
    )
    marker = "  for(int which=0;which<2;++which) {"
    text = text.replace(marker, CONV_PRELOAD + marker, 1)
    old = """      vector[i]=warp<tokens ? __half2float(
        qkv[warp*2560+which*512+group*128+lane*4+i]) : 0.f;"""
    new = """      vector[i]=warp<tokens ? __half2float(head_conv_value(
        qkv,&old_qk[which][0][0],conv,warp,
        which*512+group*128+lane*4+i,lane*4+i)) : 0.f;"""
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = """    decay[threadIdx.x]=__expf(g[threadIdx.x*12+head]);
    multiplier[threadIdx.x]=beta[threadIdx.x*12+head];"""
    new = """    const float activation=__half2float(a[threadIdx.x*12+head])+
      __half2float(bias[head]);
    const float soft=activation<=20.f ? __logf(1.f+__expf(activation)):activation;
    decay[threadIdx.x]=__expf(-__expf(a_log[head])*soft);
    const float bv=__half2float(b[threadIdx.x*12+head]);
    multiplier[threadIdx.x]=__fdividef(1.f,1.f+__expf(-bv));"""
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = """        float delta=__half2float(qkv[t*2560+1024+head*128+vbase+v])-dot;"""
    new = """        float convolved=0.f;
        if(lane==0) {
          float sum=0.f;
#pragma unroll
          for(int tap=0;tap<4;++tap) {
            const int token=t+tap-3,feature=1024+head*128+vbase+v;
            const half vv=token<0 ? old_v[token+3][warp*2+v]:
              qkv[token*2560+feature];
            sum=__fadd_rn(sum,__half2float(__hmul(vv,conv[feature*4+tap])));
          }
          convolved=__half2float(__float2half_rn(
            __fdividef(sum,1.f+__expf(-sum))));
        }
        convolved=__shfl_sync(0xffffffff,convolved,0);
        float delta=convolved-dot;"""
    assert text.count(old) == 1
    text = text.replace(old, new)
    text = text.replace(
        "torch::Tensor cu,torch::Tensor packets,torch::Tensor generations) {",
        "torch::Tensor cu,torch::Tensor packets,torch::Tensor generations,\n"
        "    torch::Tensor conv,torch::Tensor history,torch::Tensor history_indices,\n"
        "    torch::Tensor a,torch::Tensor b,torch::Tensor a_log,torch::Tensor bias,\n"
        "    torch::Tensor conv_ready) {",
        1,
    )
    old = (
        "  void* args[]={&q,&zz,&gg,&bb,&w,&s,&ix,&acc,&lengths,&r,&y,&stride,"
        "&flags,&counter};"
    )
    new = """  const half* cw=reinterpret_cast<const half*>(conv.data_ptr<at::Half>());
  half* hist=reinterpret_cast<half*>(history.data_ptr<at::Half>());
  const int* hi=history_indices.data_ptr<int>();
  const half* aa=reinterpret_cast<const half*>(a.data_ptr<at::Half>());
  const half* bbb=reinterpret_cast<const half*>(b.data_ptr<at::Half>());
  const float* al=a_log.data_ptr<float>();
  const half* bi=reinterpret_cast<const half*>(bias.data_ptr<at::Half>());
  auto* ready=reinterpret_cast<uint32_t*>(conv_ready.data_ptr<int>());
  int hs=history.stride(0),hd=history.stride(1),ht=history.stride(2);
  void* args[]={&q,&zz,&gg,&bb,&w,&s,&ix,&acc,&lengths,&r,&y,&stride,&flags,&counter,
    &cw,&hist,&hi,&aa,&bbb,&al,&bi,&ready,&hs,&hd,&ht};"""
    assert text.count(old) == 1
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vector-state", action="store_true")
    parser.add_argument("--convolution", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "head_local_core.cu"
    text = (
        convolution_source()
        if args.convolution
        else vector_state_source()
        if args.vector_state
        else SOURCE
    )
    if not source.exists() or source.read_text() != text:
        source.write_text(text)
    load(
        name="gdn_head_local_convolution_screen"
        if args.convolution
        else "gdn_head_local_vector_state_screen"
        if args.vector_state
        else "gdn_head_local_core_screen",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
