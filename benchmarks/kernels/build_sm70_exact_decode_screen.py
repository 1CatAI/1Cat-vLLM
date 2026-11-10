# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile owned production translation units for a complete-layer screen.

The private namespace is for research only. It does not install an operator
into a serving runtime or replace the source-complete wheel admission.
"""

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load

WRAPPER = r"""
#include <torch/extension.h>
#define VLLM_NVFP4_QPN2_STANDALONE
#define VLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE
#include "sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
#include "custom_all_reduce.cuh"

void launch_mlp(torch::Tensor out,torch::Tensor input,torch::Tensor bundle,
    torch::Tensor scales,torch::Tensor table,double global,bool gated,int mode) {
  TORCH_CHECK(mode==0 && input.size(0)==8);
  auto stream=at::cuda::getCurrentCUDAStream();
  auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y=reinterpret_cast<half*>(out.data_ptr<at::Half>());
  if(gated) launch_qpn2_gated<8,1,1,false,true>(
    bundle.data_ptr<uint8_t>(),scales.data_ptr<uint8_t>(),x,y,
    out.size(1),input.size(1),8,global,stream);
  else launch_qpn2<16,2,1,false,true>(bundle.data_ptr<uint8_t>(),
    scales.data_ptr<uint8_t>(),x,y,out.size(1),input.size(1),8,global,stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void pair(torch::Tensor out,torch::Tensor input,torch::Tensor bundle,
    torch::Tensor scales,double global,int mode) {
  TORCH_CHECK(mode==1);
  nvfp4_qpn2_gated_sm70_out(out,input,bundle.narrow(2,0,256),
    bundle.narrow(2,256,32),global,8,1);
}
vllm::RankData peers(const std::vector<int64_t>& pointers) {
  TORCH_CHECK(pointers.size()==4);
  vllm::RankData data{};
  for(int i=0;i<4;++i) data.ptrs[i]=reinterpret_cast<void*>(pointers[i]);
  return data;
}
torch::Tensor alias(int64_t pointer,int device) {
  return torch::from_blob(reinterpret_cast<void*>(pointer),{8,5120},
    [](void*){},torch::TensorOptions().device(torch::kCUDA,device)
    .dtype(torch::kFloat16));
}
size_t buffer_bytes() { return vllm::kSm70Tp4PushAllreduceBufferBytes; }
void initialize(const std::vector<int64_t>& pointers,int rank) {
  char* base=reinterpret_cast<char*>(pointers.at(rank));
  auto stream=at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(base,0,buffer_bytes(),stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
    base+vllm::kSm70PushNormOffset+vllm::kSm70PushNormMetaBytes,0x7f,
    vllm::kSm70PushNormPacketOffset-vllm::kSm70PushNormOffset-
    vllm::kSm70PushNormMetaBytes,stream));
}
void launch_norm(torch::Tensor out,torch::Tensor rout,torch::Tensor input,
    torch::Tensor residual,torch::Tensor weight,
    const std::vector<int64_t>& buffers,const std::vector<int64_t>& inputs,
    int rank,int mode) {
  TORCH_CHECK(mode==1 && input.sizes()==torch::IntArrayRef({8,5120}));
  TORCH_CHECK(input.data_ptr()==reinterpret_cast<void*>(inputs.at(rank)));
  vllm::sm70_push_allreduce_gemma_rms_norm<float><<<40,128,0,
    at::cuda::getCurrentCUDAStream()>>>(peers(buffers),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      residual.data_ptr<float>(),weight.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),rout.data_ptr<float>(),
      rank,reinterpret_cast<void*>(buffers.at(rank)),1e-6f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("launch",&launch_mlp);m.def("pair",&pair);
  m.def("launch",&launch_norm);m.def("alias",&alias);
  m.def("initialize",&initialize);m.def("buffer_bytes",&buffer_bytes);
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "exact_decode.cu"
    source.write_text(WRAPPER)
    load(
        name="qwen38_exact_decode_screen",
        sources=[str(source)],
        extra_include_paths=[str(args.source_root / "csrc")],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
