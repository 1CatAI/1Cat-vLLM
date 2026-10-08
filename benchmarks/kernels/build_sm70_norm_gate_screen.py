# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen a resident M8 allreduce/norm prefix inside paired NVFP4 gate/up.

Forty CTAs normalize disjoint input parts, then publish completion flags. All
136 gate CTAs are admitted with a cooperative launch and checked occupancy.
This removes one graph kernel; it changes no weight layout or GDN recurrence.
The normalizer's padded 256-thread reduction must be checked numerically.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_paired_gate import generate as paired_source
from sm70_tp4_norm_partial_packets import generate as packet_source
from torch.utils.cpp_extension import load


def generate(root, single_ready=False):
    text = packet_source(root)
    begin = text.rindex("template <", 0, text.index("void partial_packet_norm("))
    end = text.index("size_t buffer_bytes()", begin)
    prefix = text[begin:end]
    prefix = prefix.replace(
        "__global__ __launch_bounds__(128, 1) void partial_packet_norm(",
        "__device__ __forceinline__ void resident_norm_prefix(",
    )
    assert "__global__" not in prefix
    prefix = prefix.replace(
        "float epsilon) {",
        "float epsilon, const void* local_pointer, uint32_t generation) {",
    )
    prefix = prefix.replace(
        "constexpr int Threads = 128, Parts = kSm70PushNormParts;",
        "constexpr int Threads = 256, Parts = kSm70PushNormParts;",
    )
    prefix = prefix.replace(
        "reinterpret_cast<const char*>(buffers.ptrs[rank])",
        "reinterpret_cast<const char*>(local_pointer)",
    )
    begin = prefix.index("  auto* generation_meta =")
    end = prefix.index("  const int epoch_offset", begin)
    prefix = prefix[:begin] + prefix[end:]
    prefix = prefix.replace(
        "      if (part==0) generation_meta->generation[row]=generation;", ""
    )
    end = prefix.rindex("}")
    prefix = (
        prefix[:end]
        + r"""
  __syncthreads();
  if (tid==0) {
    __threadfence();
    auto* signal=reinterpret_cast<volatile GateSignals*>(
        const_cast<char*>(reinterpret_cast<const char*>(local_pointer))+
        kSm70Tp4PushAllreduceBufferBytes+sizeof(PullSignals)+sizeof(PartialPacketMeta));
    signal->done[row][part]=generation;
  }
"""
        + prefix[end:]
    )
    pair = paired_source(root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu")
    pair = pair[: pair.index("PYBIND11_MODULE(")]
    pair = pair.replace("void launch(", "void launch_projection_control(")
    begin = pair.index("template <bool Interleave>")
    end = pair.index("\n}\nvoid launch_pair", begin)
    gate = pair[begin:end].replace("paired_gate_kernel", "resident_norm_gate")
    gate = gate.replace("__launch_bounds__(256, 3)", "__launch_bounds__(256, 1)")
    gate = gate.replace(
        "half* output, int hidden, int k, float global) {",
        "half* output, int hidden, int k, float global, vllm::RankData buffers, "
        "const void* local_pointer, const half* projected, const float* residual, "
        "const float* weight, float* residual_out) {",
    )
    marker = "  constexpr int Split = 8;"
    assert gate.count(marker) == 1
    gate = gate.replace(
        marker,
        r"""
  auto* signal=reinterpret_cast<volatile GateSignals*>(
      const_cast<char*>(reinterpret_cast<const char*>(local_pointer))+
      kSm70Tp4PushAllreduceBufferBytes+sizeof(PullSignals)+sizeof(PartialPacketMeta));
  const uint32_t generation=signal->generation[blockIdx.x]+1;
  if (blockIdx.x<40) {
    resident_norm_prefix<float>(buffers,projected,residual,weight,
        const_cast<half*>(input),residual_out,rank_argument,1e-6f,
        local_pointer,generation);
  }
  if (threadIdx.x<40) {
    while (signal->done[threadIdx.x/5][threadIdx.x%5]!=generation) {}
  }
  __syncthreads();
  constexpr int Split = 8;
""".replace("rank_argument", "rank"),
    )
    gate = gate.replace(
        "const float* weight, float* residual_out)",
        "const float* weight, float* residual_out, int rank)",
    )
    end = gate.rindex("}")
    gate = (
        gate[:end]
        + "  if (threadIdx.x==0) signal->generation[blockIdx.x]=generation;\n"
        + gate[end:]
    )
    signal = r"""
struct GateSignals {
  uint32_t generation[136];
  uint32_t done[8][5];
};
"""
    if single_ready:
        signal = signal.replace(
            "  uint32_t done[8][5];", "  uint32_t done[8][5];\n  uint32_t ready;"
        )
        old = """  if (threadIdx.x<40) {
    while (signal->done[threadIdx.x/5][threadIdx.x%5]!=generation) {}
  }
  __syncthreads();"""
        new = """  if (blockIdx.x==0) {
    if (threadIdx.x<40) {
      while (signal->done[threadIdx.x/5][threadIdx.x%5]!=generation) {}
    }
    __syncthreads();
    if (threadIdx.x==0) {
      __threadfence();
      signal->ready=generation;
    }
  } else if (threadIdx.x==0) {
    while (signal->ready!=generation) {}
  }
  __syncthreads();"""
        assert gate.count(old) == 1
        gate = gate.replace(old, new)
    marker = "size_t buffer_bytes() {"
    text = text.replace(
        marker, signal + prefix + pair + "\nnamespace {\n" + gate + "\n}\n" + marker
    )
    text = text.replace(
        "sizeof(PartialPacketMeta);", "sizeof(PartialPacketMeta)+sizeof(GateSignals);"
    )
    wrapper = r"""
void fused(torch::Tensor up,torch::Tensor normalized,torch::Tensor residual_out,
           torch::Tensor projected,torch::Tensor residual,torch::Tensor weight,
           torch::Tensor codes,torch::Tensor scales,double scale,
           const std::vector<int64_t>& pointers,int rank) {
  TORCH_CHECK(projected.sizes()==torch::IntArrayRef({8,5120}));
  TORCH_CHECK(projected.scalar_type()==torch::kFloat16 && projected.is_contiguous());
  TORCH_CHECK(residual.scalar_type()==torch::kFloat32 && residual.is_contiguous());
  TORCH_CHECK(weight.scalar_type()==torch::kFloat32 && weight.numel()==5120);
  int resident=0,sm=0,cooperative=0,device=0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident,resident_norm_gate<false>,256,0));
  TORCH_CHECK(cooperative && resident*sm>=136,"resident prefix admission failed");
  auto data=peers(pointers); const void* local=data.ptrs[rank];
  const auto* c=codes.data_ptr<uint8_t>(); const auto* s=scales.data_ptr<uint8_t>();
  auto* x=reinterpret_cast<const half*>(normalized.data_ptr<at::Half>());
  auto* y=reinterpret_cast<half*>(up.data_ptr<at::Half>());
  auto* p=reinterpret_cast<const half*>(projected.data_ptr<at::Half>());
  auto* r=residual.data_ptr<float>();auto* w=weight.data_ptr<float>();
  auto* ro=residual_out.data_ptr<float>();int hidden=4352,k=5120;float global=scale;
  void* arguments[]={&c,&s,&x,&y,&hidden,&k,&global,&data,&local,&p,&r,&w,&ro,&rank};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(
      resident_norm_gate<false>),dim3(136),dim3(256),arguments,0,
      at::cuda::getCurrentCUDAStream()));
}
"""
    text = text.replace("PYBIND11_MODULE(", wrapper + "\nPYBIND11_MODULE(")
    return text.replace(
        'm.def("alias",&alias);',
        f'm.attr("single_ready")={str(single_ready).lower()}; '
        'm.def("fused",&fused); m.def("alias",&alias);',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--single-ready", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "resident_norm_gate.cu"
    source.write_text(generate(args.source_root, args.single_ready))
    load(
        name=(
            "sm70_resident_norm_gate_join_screen"
            if args.single_ready
            else "sm70_resident_norm_gate_screen"
        ),
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
