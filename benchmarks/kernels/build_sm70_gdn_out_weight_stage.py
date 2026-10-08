# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Screen a resident GDN/norm/FP8-out/collective chain with weight staging.

The original 768 two-column state tasks use eight warps in 96 CTAs. Other
warps stage half of the following FP8 projection while these tasks run.
Each projection warp waits only for the eight state tiles of its K head.
This is a numerical research screen, not an installed model dispatch.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_product_lookup import function_end
from build_sm70_gdn_head_local_core import vector_state_source
from build_sm70_resident_projection_norm import generate as resident_source
from torch.utils.cpp_extension import load


def generate(root):
    support = resident_source(root, native_protocol=True)
    support = support[: support.index("PYBIND11_MODULE(")]
    begin = support.rindex("template <", 0, support.index("void resident_out_norm("))
    end = function_end(support, begin)
    producer = support[begin:end]
    producer = producer.replace("resident_out_norm", "gdn_out_weight_stage")
    producer = producer.replace("const half* __restrict__ input", "half* input")
    producer = producer.replace(
        "float* residual_out) {",
        "float* residual_out,const half* qkv,const half* z,const float* g,"
        "const float* beta,const half* gate_weight,float* state,const int* indices,"
        "const int* accepted,const int* cu,half* raw,int stride,"
        "uint64_t* packets,uint32_t* head_epochs,uint32_t* ready) {",
    )
    producer = producer.replace(
        "  __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];",
        "  static_assert(SplitK==12 && RowTiles==1 && !M1Only);\n"
        "  __shared__ __align__(16) unsigned char weight_prefix[12*4*512];\n"
        "  auto partials=reinterpret_cast<float (*)[256]>(weight_prefix);\n"
        "  __shared__ uint32_t generation;\n"
        "  if(threadIdx.x==0) generation=ready[96+blockIdx.x]+1;\n"
        "  __syncthreads();\n"
        "  if(blockIdx.x<96) {\n"
        "    if(threadIdx.x<256) {\n"
        "      gdn_produce(qkv,z,g,beta,gate_weight,state,indices,accepted,cu,"
        "raw,input,stride,packets,head_epochs);\n"
        "      if(threadIdx.x==0) stage_release(ready+blockIdx.x,generation);\n"
        "    } else stage_out_weights<256>(codes,weight_prefix,threadIdx.x-256);\n"
        "  } else stage_out_weights<512>(codes,weight_prefix,threadIdx.x);\n"
        "  __syncthreads();",
    )
    marker = "  if(warp<SplitK) {\n"
    assert producer.count(marker) == 1
    producer = producer.replace(
        marker,
        marker
        + "  if(lane<8) stage_wait(ready+warp*8+lane,generation);\n  __syncwarp();\n",
    )
    old = "__ldcs(code_ptr + static_cast<size_t>(group) * 32)"
    assert producer.count(old) == 1
    producer = producer.replace(
        old,
        "(group-group_begin<4 ? *reinterpret_cast<const uint4*>("
        "weight_prefix+((warp*4+group-group_begin)*32+lane)*16) : " + old + ")",
    )
    for suffix in ("", " + 8"):
        old = "*reinterpret_cast<const uint4*>(input_row + group * 16" + suffix + ")"
        assert producer.count(old) == 1
        producer = producer.replace(
            old, "stage_activation(input_row + group * 16" + suffix + ")"
        )
    marker = "    if constexpr (PrefetchCodes) {\n      prefetched = next;\n    }\n  }"
    assert producer.count(marker) == 1
    # Every producer warp consumes all four staged groups before the shared
    # prefix is reused for the FP32 projection partial sums.
    producer = producer.replace(marker, marker)
    marker = "  if constexpr (M1Only) {\n"
    # The inactive four warps must join this CTA barrier too.
    position = producer.index(marker, producer.index("#pragma unroll 4\n"))
    producer = (
        producer[:position]
        + "  }\n  __syncthreads();\n  if(warp<SplitK) {\n"
        + producer[position:]
    )
    producer = (
        producer[: producer.rindex("}")]
        + "  if(threadIdx.x==0) ready[96+blockIdx.x]=generation;\n}\n"
    )

    core_source = vector_state_source()
    begin = core_source.index(
        "__global__ __launch_bounds__(256,1) void head_local_core("
    )
    core = core_source[begin : function_end(core_source, begin)]
    core = core.replace(
        "__global__ __launch_bounds__(256,1) void head_local_core(",
        "__device__ __forceinline__ void gdn_produce(",
    )
    core = core.replace(
        "__syncthreads();", 'asm volatile("bar.sync 1,256;" ::: "memory");'
    )
    core = core.replace(
        "    output[warp*1536+head*128+col]=__float2half_rn(normalized);",
        "    const half result=__float2half_rn(normalized);\n"
        '    asm volatile("st.volatile.global.u16 [%0],%1;" ::'
        '"l"(output+warp*1536+head*128+col),"h"(__half_as_ushort(result)):"memory");',
    )
    core = (
        core[: core.rindex("}")]
        + '  __threadfence();\n  asm volatile("bar.sync 1,256;" ::: "memory");\n}\n'
    )
    helpers = core_source[
        core_source.index("__device__ __forceinline__ float head_warp_sum") : begin
    ]
    helpers += r"""
__device__ __forceinline__ void stage_release(uint32_t* p,uint32_t value) {
  asm volatile("st.release.gpu.global.u32 [%0],%1;" ::"l"(p),"r"(value):"memory");
}
__device__ __forceinline__ void stage_wait(const uint32_t* p,uint32_t expected) {
  uint32_t value;
  do { asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" :"=r"(value):"l"(p):"memory"); }
  while(value!=expected);
}
__device__ __forceinline__ uint4 stage_activation(const half* p) {
  uint4 value;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3},[%4];" :
    "=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w):"l"(p):"memory");
  return value;
}
template<int Workers>
__device__ __forceinline__ void stage_out_weights(
    const uint8_t* codes,unsigned char* prefix,int tid) {
  constexpr int Jobs=12*4*32;
  for(int batch=0;batch<Jobs;batch+=4*Workers) {
    uint4 raw[4];
#pragma unroll
    for(int slot=0;slot<4;++slot) {
      const int job=batch+tid+slot*Workers;
      if(job<Jobs) {
        const int lane=job%32,local=job/32;
        const int group=(local/4)*8+local%4;
        raw[slot]=__ldg(reinterpret_cast<const uint4*>(codes)+
            (static_cast<size_t>(blockIdx.x)*96+group)*32+lane);
      }
    }
#pragma unroll
    for(int slot=0;slot<4;++slot) {
      const int job=batch+tid+slot*Workers;
      if(job<Jobs) reinterpret_cast<uint4*>(prefix)[job]=raw[slot];
    }
  }
}
"""
    # Core shared storage is separate from the weight prefix. Only the first
    # eight warps join its named 256-thread barrier.
    wrapper_begin = support.index("void fused_out(")
    wrapper_end = function_end(support, wrapper_begin)
    wrapper = support[wrapper_begin:wrapper_end].replace(
        "void fused_out(", "void gdn_out_launch("
    )
    wrapper = wrapper.replace(
        "const std::vector<int64_t>& buffers,int rank) {",
        "const std::vector<int64_t>& buffers,int rank,torch::Tensor qkv,"
        "torch::Tensor z,torch::Tensor g,torch::Tensor beta,torch::Tensor gate_weight,"
        "torch::Tensor state,torch::Tensor indices,torch::Tensor accepted,"
        "torch::Tensor cu,torch::Tensor raw_state_output,torch::Tensor packets,"
        "torch::Tensor head_epochs,torch::Tensor ready) {",
    )
    wrapper = wrapper.replace("resident_out_norm<", "gdn_out_weight_stage<")
    wrapper = wrapper.replace(
        "const auto* x=reinterpret_cast<const half*>", "auto* x=reinterpret_cast<half*>"
    )
    wrapper = wrapper.replace(
        "  void* args[]=",
        r"""
  const auto* q=reinterpret_cast<const half*>(qkv.data_ptr<at::Half>());
  const auto* zz=reinterpret_cast<const half*>(z.data_ptr<at::Half>());
  const auto* gg=g.data_ptr<float>();const auto* bb=beta.data_ptr<float>();
  const auto* gw=reinterpret_cast<const half*>(gate_weight.data_ptr<at::Half>());
  auto* st=state.data_ptr<float>();const auto* ix=indices.data_ptr<int>();
  const auto* acc=accepted.data_ptr<int>();const auto* lengths=cu.data_ptr<int>();
  auto* raw_state=reinterpret_cast<half*>(raw_state_output.data_ptr<at::Half>());
  int stride=state.stride(0);
  auto* pk=reinterpret_cast<uint64_t*>(packets.data_ptr<int64_t>());
  auto* he=reinterpret_cast<uint32_t*>(head_epochs.data_ptr<int>());
  auto* flags=reinterpret_cast<uint32_t*>(ready.data_ptr<int>());
  TORCH_CHECK(ready.numel()==256 && packets.numel()==768 && head_epochs.numel()==12);
  void* args[]=""",
    ).replace(
        "&addresses,&rank,&local,&r,&w,&y,&ro};",
        "&addresses,&rank,&local,&r,&w,&y,&ro,&q,&zz,&gg,&bb,&gw,&st,&ix,&acc,&lengths,&raw_state,&stride,&pk,&he,&flags};",
    )
    return (
        support
        + helpers
        + core
        + producer
        + wrapper
        + '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("launch",&gdn_out_launch);}\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "gdn_out_weight_stage.cu"
    path.write_text(generate(args.source_root))
    load(
        name="sm70_gdn_out_weight_stage_screen",
        sources=[str(path)],
        extra_include_paths=[
            str(args.source_root / "csrc"),
            str(args.source_root / "csrc/sm70_turbomind/ops"),
        ],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
