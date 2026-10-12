# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# CUDA source anchors retain their original spelling.
# ruff: noqa: E501
"""Screen a resident down publisher and consume-only five-part norm suffix.

All 160 producer CTAs are admitted resident. Forty also consume disjoint norm
parts after publishing their own outputs. Each payload independently protects
its readiness; no whole-grid phase barrier or raw-output copy is needed.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_product_lookup import function_end
from sm70_tp4_local_buffer_pointer import generate as publisher_source
from torch.utils.cpp_extension import load


def generate(root):
    text = publisher_source(root, publication=True)
    text = text[: text.index("PYBIND11_MODULE(")]
    begin = text.rindex("template <", 0, text.index("void consume_packet_norm("))
    prefix = text[begin : function_end(text, begin)]
    prefix = prefix.replace("consume_packet_norm", "resident_down_norm_suffix")
    prefix = prefix.replace(
        "__global__ __launch_bounds__(32, 1)", "__device__ __forceinline__"
    )
    assert "__global__" not in prefix
    prefix = prefix.replace(
        "constexpr int Threads = 32, Parts = 20;",
        "constexpr int Threads = 512, Parts = 5;",
    )
    assert "Threads = 512" in prefix
    begin = text.rindex("template <", 0, text.index("void warp_publish_down("))
    producer = text[begin : function_end(text, begin)]
    producer = producer.replace("warp_publish_down", "resident_down_norm")
    producer = producer.replace(
        "__global__ void", "__global__ __launch_bounds__(512,2) void"
    )
    old = "const void* local_buffer) {"
    assert producer.count(old) == 1
    producer = producer.replace(
        old,
        "const void* local_buffer,const float* residual,"
        "const float* norm_weight,half* normalized,"
        "float* residual_out) {",
    )
    producer = (
        producer[: producer.rindex("}")]
        + r"""
  __syncthreads();
  if(blockIdx.x<40) {
    resident_down_norm_suffix<float>(buffers,nullptr,residual,norm_weight,
        normalized,residual_out,rank,local_buffer,1e-6f);
  }
}
"""
    )
    wrapper = r"""
void fused(torch::Tensor normalized,torch::Tensor residual_out,
    torch::Tensor input,torch::Tensor residual,torch::Tensor norm_weight,
    torch::Tensor codes,torch::Tensor scales,double global,
    const std::vector<int64_t>& buffers,int rank) {
  TORCH_CHECK(input.sizes()==torch::IntArrayRef({8,4352}) && input.is_contiguous());
  TORCH_CHECK(normalized.sizes()==torch::IntArrayRef({8,5120}) && normalized.is_contiguous());
  TORCH_CHECK(residual.sizes()==normalized.sizes() && residual.is_contiguous());
  TORCH_CHECK(residual.scalar_type()==torch::kFloat32 &&
              residual_out.scalar_type()==torch::kFloat32 &&
              norm_weight.scalar_type()==torch::kFloat32);
  auto kernel=resident_down_norm<16,2,1,false,false,true>;
  int device,resident;cudaDeviceProp properties;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,512,0));
  TORCH_CHECK(properties.major==7 && properties.minor==0 && properties.cooperativeLaunch &&
              resident*properties.multiProcessorCount>=160,
              "Every down producer must be admitted resident");
  auto* c=codes.data_ptr<uint8_t>();auto* s=scales.data_ptr<uint8_t>();
  const auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  half* raw=nullptr;
  auto* y=reinterpret_cast<half*>(normalized.data_ptr<at::Half>());
  auto* ro=residual_out.data_ptr<float>();const auto* r=residual.data_ptr<float>();
  const auto* w=norm_weight.data_ptr<float>();
  auto addresses=peers(buffers);const void* local=reinterpret_cast<void*>(buffers.at(rank));
  int n=5120,k=4352,m=8;float scale=global;
  void* args[]={&c,&s,&x,&raw,&n,&k,&m,&scale,&addresses,&rank,&local,&r,&w,&y,&ro};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel),dim3(160),
      dim3(512),args,0,at::cuda::getCurrentCUDAStream()));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("buffer_bytes",&buffer_bytes);m.def("initialize",&initialize);
  m.def("fused",&fused);
}
"""
    return text + prefix + producer + wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "resident_down_norm.cu"
    source.write_text(generate(args.source_root))
    load(
        name="sm70_resident_down_norm_screen",
        sources=[str(source)],
        extra_include_paths=[
            str(args.source_root / "csrc"),
            str(args.source_root / "csrc/sm70_turbomind/ops"),
        ],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
