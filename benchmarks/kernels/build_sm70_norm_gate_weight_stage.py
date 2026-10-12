# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source anchors and CUDA strings retain their original spelling.
# ruff: noqa: E501
"""Stage raw gate weights while separate warps finish peer reduction/norm.

Four warps publish and normalize the input; four stage a raw weight prefix.
Other resident CTAs only stage weights. Gate warps wait for their next input
part, consume the staged prefix, then continue the unchanged ordered MMA.
Shared storage is reused for partial sums only after every warp consumes it.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_paired_gate import generate as pair_source
from benchmark_sm70_qpn2_product_lookup import function_end
from build_sm70_resident_projection_norm import generate as support_source
from torch.utils.cpp_extension import load


def generate(root):
    text = support_source(root, native_protocol=True)
    text = text[: text.index("PYBIND11_MODULE(")]
    qpn2 = (root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu").read_text()
    reader = qpn2[
        qpn2.index(
            "template <bool TurboMindLayout, bool BundledScales = false>"
        ) : qpn2.index("#define VLLM_SM70_QPN2_MMA")
    ]
    native = (root / "csrc/custom_all_reduce.cuh").read_text()
    begin = native.index("template <typename WeightT, bool Reference = false>")
    prefix = native[begin : function_end(native, begin)]
    prefix = prefix.replace(
        "sm70_push_allreduce_gemma_rms_norm", "weight_stage_norm"
    ).replace("__global__ __launch_bounds__(128, 1)", "__device__ __forceinline__")
    prefix = prefix.replace(
        "constexpr int Threads = 128", "constexpr int Threads = 256"
    )
    prefix = prefix.replace(
        "float epsilon) {",
        "float epsilon,const uint8_t* next_codes,unsigned char* raw_prefix,"
        "uint32_t own_generation,uint32_t* done) {",
    )
    prefix = prefix.replace(
        "    reinterpret_cast<P*>(output)[pack] = normalized;",
        "    vllm::sm70_push_store_volatile_16b(normalized,reinterpret_cast<char*>(output),pack);",
    )
    assert "reinterpret_cast<P*>(output)[pack] = normalized" not in prefix
    marker = "  using Reduce = cub::BlockReduce<float, Threads>;"
    assert prefix.count(marker) == 1
    prefix = prefix.replace(
        marker,
        "  else { stage_weight_prefix<128>(next_codes,raw_prefix,tid-128); }\n"
        + marker,
    )
    end = prefix.rindex("}")
    prefix = (
        prefix[:end]
        + r"""
  __threadfence();
  __syncthreads();
  if(tid==0) {
    asm volatile("st.release.gpu.global.u32 [%0],%1;" ::
        "l"(done+blockIdx.x),"r"(own_generation):"memory");
  }
"""
        + prefix[end:]
    )
    pair = pair_source(root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu")
    begin = pair.index("template <bool Interleave>")
    end = pair.index("\n}\nvoid launch_pair", begin)
    gate = pair[begin:end].replace("paired_gate_kernel", "staged_norm_gate")
    gate = gate.replace("const half* input", "half* input")
    gate = gate.replace(
        "template <bool Interleave>",
        "template <bool Interleave,bool NormPrefix=true,bool Debug=false>",
    )
    gate = gate.replace("__launch_bounds__(256, 3)", "__launch_bounds__(256, 2)")
    gate = gate.replace(
        "half* output, int hidden, int k, float global) {",
        "half* output,int hidden,int k,float global,vllm::RankData buffers,"
        "const void* local_pointer,const half* projected,const float* residual,"
        "const float* weight,float* residual_out,int rank,uint32_t* epochs,"
        "uint8_t* debug_prefix,half* debug_input) {",
    )
    gate = gate.replace(
        "  __shared__ float partials[2][Split][256];",
        r"""
  __shared__ __align__(16) unsigned char raw_prefix[2*Split*10*288];
  auto partials=reinterpret_cast<float (*)[Split][256]>(raw_prefix);
  __shared__ uint32_t generation;
  if constexpr(NormPrefix) {
  if(threadIdx.x==0) generation=epochs[blockIdx.x]+1;
  __syncthreads();
  if(blockIdx.x<40) {
    weight_stage_norm<float>(buffers,projected,residual,weight,
        const_cast<half*>(input),residual_out,rank,local_pointer,1e-6f,
        codes,raw_prefix,generation,epochs+136);
  } else {
    stage_weight_prefix<256>(codes,raw_prefix,threadIdx.x);
    __syncthreads();
  }
  } else {
    stage_weight_prefix<256>(codes,raw_prefix,threadIdx.x);
    __syncthreads();
  }
""",
    )
    marker = "    const half* a = input +"
    begin = gate.index(marker)
    marker = "  const int lane = threadIdx.x & 31;"
    assert gate.count(marker) == 1
    gate = gate.replace(
        marker,
        r"""
  if constexpr(NormPrefix) {
    if(threadIdx.x<40) wait_norm_part(epochs+136+threadIdx.x,generation);
    __syncthreads();
  }
"""
        + marker,
    )
    gate = gate.replace(
        marker,
        r"""
  if constexpr(Debug) {
    if(blockIdx.x==0 || blockIdx.x==40) {
      const int slot=blockIdx.x==0 ? 0 : 1;
      for(int byte=threadIdx.x;byte<2*8*10*288;byte+=blockDim.x)
        debug_prefix[slot*2*8*10*288+byte]=raw_prefix[byte];
    }
    __syncthreads();
  }
"""
        + marker,
    )
    gate = gate.replace(
        "*reinterpret_cast<const uint4*>(a)", "load_current_activation(a)"
    ).replace("*reinterpret_cast<const uint4*>(a + 8)", "load_current_activation(a+8)")
    marker = "    const unsigned* a0 ="
    index = gate.index(marker)
    gate = (
        gate[:index]
        + r"""
    if constexpr(Debug) {
      if(blockIdx.x==0 || blockIdx.x==40) {
        const int slot=blockIdx.x==0 ? 0 : 1;
        half* destination=debug_input+(((slot*8+warp)*40+group%40)*32+lane)*16;
        *reinterpret_cast<uint4*>(destination)=a01;
        *reinterpret_cast<uint4*>(destination+8)=a23;
      }
    }
"""
        + gate[index:]
    )
    gate = gate.replace(
        "readers[p].load(group, weights[p]);",
        "load_gate_operand(readers[p],raw_prefix,p,warp,lane,group,global,weights[p]);",
    ).replace(
        "readers[p].load(group, weights);",
        "load_gate_operand(readers[p],raw_prefix,p,warp,lane,group,global,weights);",
    )
    marker = (
        "load_gate_operand(readers[p],raw_prefix,p,warp,lane,group,global,weights);"
    )
    gate = gate.replace(
        marker,
        marker
        + r"""
        if constexpr(Debug) {
          if(blockIdx.x==0 || blockIdx.x==40) {
            const int slot=blockIdx.x==0 ? 0 : 1;
            const size_t word=((((slot*2+p)*8+warp)*40+group%40)*32+lane)*32;
            auto* destination=reinterpret_cast<uint4*>(debug_prefix+2*2*8*10*288+word);
            destination[0]=reinterpret_cast<const uint4*>(weights)[0];
            destination[1]=reinterpret_cast<const uint4*>(weights)[1];
          }
        }
""",
    )
    marker = "\n  }\n#pragma unroll\n  for (int p ="
    assert gate.count(marker) == 1
    gate = gate.replace(marker, "\n    if(group%40==9) __syncthreads();" + marker)
    end = gate.rindex("}")
    gate = (
        gate[:end]
        + "  if constexpr(NormPrefix) if(threadIdx.x==0) epochs[blockIdx.x]=generation;\n"
        + gate[end:]
    )
    helpers = r"""
template<int Workers>
__device__ __forceinline__ void stage_weight_prefix(
    const uint8_t* codes,unsigned char* shared,int tid) {
  constexpr int Jobs=2*8*10*16;
  for(int batch=0;batch<Jobs;batch+=8*Workers) {
    uint4 raw[8];uint16_t sf[8];
#pragma unroll
    for(int stage=0;stage<8;++stage) {
      const int job=batch+tid+stage*Workers;
      if(job<Jobs) {
        const int lane=(job%16)*2,step=(job/16)%10;
        const int warp=((job/16)/10)%8,projection=((job/16)/10)/8;
        const uint8_t* block=codes+
            (static_cast<size_t>(blockIdx.x+projection*136)*320+warp*40+step)*288;
        raw[stage]=__ldg(reinterpret_cast<const uint4*>(block+lane*8));
        sf[stage]=__ldg(reinterpret_cast<const uint16_t*>(block+256+lane));
      }
    }
#pragma unroll
    for(int stage=0;stage<8;++stage) {
      const int job=batch+tid+stage*Workers;
      if(job<Jobs) {
        const int lane=(job%16)*2;
        unsigned char* block=shared+(job/16)*288;
        *reinterpret_cast<uint4*>(block+lane*8)=raw[stage];
        *reinterpret_cast<uint16_t*>(block+256+lane)=sf[stage];
      }
    }
  }
}
__device__ __forceinline__ void wait_norm_part(const uint32_t* pointer,uint32_t generation) {
  uint32_t value;
  do {
    asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(value):"l"(pointer):"memory");
  } while(value!=generation);
}
__device__ __forceinline__ uint4 load_current_activation(const half* address) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3},[%4];" :
      "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(address):"memory");
  return v;
}
template<typename Reader>
__device__ __forceinline__ void load_gate_operand(
    const Reader& reader,const unsigned char* shared,int p,int warp,int lane,
    int group,float global,half2* weights) {
  if(group%40<10) {
    const unsigned char* block=shared+((p*8+warp)*10+group%40)*288;
    const uint2 raw=*reinterpret_cast<const uint2*>(block+lane*8);
    const half2 sf=nvfp4_effective_scale(block[256+lane],global);
    dequant_e2m1x8(raw.x,sf,weights);dequant_e2m1x8(raw.y,sf,weights+4);
  } else reader.load(group,weights);
}
"""
    wrapper = r"""
void fused(torch::Tensor up,torch::Tensor normalized,torch::Tensor residual_out,
    torch::Tensor projected,torch::Tensor residual,torch::Tensor weight,
    torch::Tensor codes,torch::Tensor scales,double scale,
    const std::vector<int64_t>& pointers,int rank,torch::Tensor epochs) {
  TORCH_CHECK(projected.sizes()==torch::IntArrayRef({8,5120}));
  TORCH_CHECK(projected.scalar_type()==torch::kFloat16 && projected.is_contiguous());
  TORCH_CHECK(residual.scalar_type()==torch::kFloat32 && residual.is_contiguous());
  TORCH_CHECK(weight.scalar_type()==torch::kFloat32 && weight.numel()==5120);
  TORCH_CHECK(epochs.numel()==176 && epochs.scalar_type()==torch::kInt32);
  auto kernel=staged_norm_gate<false>;
  int device,resident;cudaDeviceProp properties;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,100));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,256,0));
  TORCH_CHECK(properties.major==7 && properties.minor==0 && properties.cooperativeLaunch &&
      resident*properties.multiProcessorCount>=136,"Requires all gate producers resident");
  auto data=peers(pointers);const void* local=data.ptrs[rank];
  const auto* c=codes.data_ptr<uint8_t>();const auto* s=scales.data_ptr<uint8_t>();
  auto* x=reinterpret_cast<half*>(normalized.data_ptr<at::Half>());
  auto* y=reinterpret_cast<half*>(up.data_ptr<at::Half>());
  const auto* p=reinterpret_cast<const half*>(projected.data_ptr<at::Half>());
  const auto* r=residual.data_ptr<float>();const auto* w=weight.data_ptr<float>();
  auto* ro=residual_out.data_ptr<float>();auto* signals=reinterpret_cast<uint32_t*>(epochs.data_ptr<int>());
  int hidden=4352,k=5120;float global=scale;
  uint8_t* dp=nullptr;half* dx=nullptr;
  void* args[]={&c,&s,&x,&y,&hidden,&k,&global,&data,&local,&p,&r,&w,&ro,&rank,&signals,&dp,&dx};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel),dim3(136),
      dim3(256),args,0,at::cuda::getCurrentCUDAStream()));
}
template<int Workers>
__global__ void dump_weight_stage(const uint8_t* codes,uint8_t* output) {
  __shared__ __align__(16) unsigned char raw_prefix[2*8*10*288];
  if(threadIdx.x<Workers) stage_weight_prefix<Workers>(codes,raw_prefix,threadIdx.x);
  __syncthreads();
  for(int byte=threadIdx.x;byte<2*8*10*288;byte+=blockDim.x)
    output[static_cast<size_t>(blockIdx.x)*2*8*10*288+byte]=raw_prefix[byte];
}
void dump_stage(torch::Tensor output,torch::Tensor codes,int workers) {
  TORCH_CHECK(output.numel()==136*2*8*10*288 && output.is_contiguous());
  if(workers==128) dump_weight_stage<128><<<136,256,0,at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(),output.data_ptr<uint8_t>());
  else dump_weight_stage<256><<<136,256,0,at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(),output.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gate_only(torch::Tensor output,torch::Tensor input,torch::Tensor codes,
    torch::Tensor scales,double scale) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(staged_norm_gate<false,false>,
      cudaFuncAttributePreferredSharedMemoryCarveout,100));
  vllm::RankData peers{};
  staged_norm_gate<false,false><<<136,256,0,at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(),scales.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),4352,5120,scale,
      peers,nullptr,nullptr,nullptr,nullptr,nullptr,0,nullptr,nullptr,nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("fused",&fused);m.def("dump_stage",&dump_stage);m.def("gate",&gate_only);
}
"""
    pure_debug = wrapper[
        wrapper.index("void gate_only(") : function_end(
            wrapper, wrapper.index("void gate_only(")
        )
    ]
    pure_debug = pure_debug.replace("void gate_only(", "void gate_debug(")
    pure_debug = pure_debug.replace(
        "torch::Tensor scales,double scale)",
        "torch::Tensor scales,double scale,torch::Tensor prefix_dump,torch::Tensor input_dump)",
    )
    pure_debug = pure_debug.replace(
        "staged_norm_gate<false,false>", "staged_norm_gate<false,false,true>"
    )
    pure_debug = pure_debug.replace(
        "0,nullptr,nullptr,nullptr);",
        "0,nullptr,prefix_dump.data_ptr<uint8_t>(),reinterpret_cast<half*>(input_dump.data_ptr<at::Half>()));",
    )
    wrapper = wrapper.replace("PYBIND11_MODULE(", pure_debug + "\nPYBIND11_MODULE(")
    wrapper = wrapper.replace(
        'm.def("gate",&gate_only);',
        'm.def("gate",&gate_only);m.def("gate_debug",&gate_debug);',
    )
    begin = wrapper.index("void fused(")
    end = function_end(wrapper, begin)
    debug_wrapper = wrapper[begin:end].replace("void fused(", "void fused_debug(")
    debug_wrapper = debug_wrapper.replace(
        "int rank,torch::Tensor epochs) {",
        "int rank,torch::Tensor epochs,torch::Tensor prefix_dump,torch::Tensor input_dump) {",
    ).replace("staged_norm_gate<false>", "staged_norm_gate<false,true,true>")
    debug_wrapper = debug_wrapper.replace(
        "uint8_t* dp=nullptr;half* dx=nullptr;",
        "uint8_t* dp=prefix_dump.data_ptr<uint8_t>();"
        "half* dx=reinterpret_cast<half*>(input_dump.data_ptr<at::Half>());",
    )
    wrapper = wrapper.replace("PYBIND11_MODULE(", debug_wrapper + "\nPYBIND11_MODULE(")
    wrapper = wrapper.replace(
        'm.def("fused",&fused);',
        'm.def("fused",&fused);m.def("fused_debug",&fused_debug);',
    )
    # The support module already declares a different fused wrapper.
    text = text.replace("void fused(", "void support_fused(")
    return (
        text
        + "\nnamespace {\n"
        + reader
        + "\n}\n"
        + helpers
        + prefix
        + "\nnamespace {\n"
        + gate
        + "\n}\n"
        + wrapper
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "norm_gate_weight_stage.cu"
    output.write_text(generate(args.source_root))
    load(
        name="sm70_norm_gate_weight_stage_screen",
        sources=[str(output)],
        extra_include_paths=[
            str(args.source_root / "csrc"),
            str(args.source_root / "csrc/sm70_turbomind/ops"),
        ],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
