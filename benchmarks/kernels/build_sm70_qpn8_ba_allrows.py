# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse each BF16-derived b/a weight pair across all eight verify rows.

Retain each row's original FMA and warp/CTA reduction order. The fused launch
uses twelve b/a CTAs instead of ninety-six without widening its weight format.
Only an independent screen extension is built.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_effective_scale import extract_kernel
from benchmark_sm70_qpn8_two_phase import generate as qpn_source
from torch.utils.cpp_extension import load


def generate(root):
    source = root / "csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu"
    text = qpn_source(source)
    text = text[: text.index("PYBIND11_MODULE(")]
    original = extract_kernel(source.read_text(), "fp8_qpn8_sm70_kernel")
    candidate = original.replace("fp8_qpn8_sm70_kernel", "ba_allrows")
    begin = candidate.index("      constexpr int kBARowsPerBlock")
    end = candidate.index("      return;", begin)
    candidate = (
        candidate[:begin]
        + r"""
      constexpr int kBAThreadsPerRow=256;
      const int ba_group=threadIdx.x/kBAThreadsPerRow;
      const int ba_thread=threadIdx.x%kBAThreadsPerRow;
      const int ba_warp=ba_thread>>5;
      const int ba_row=(tile-qpn_tiles)*2+ba_group;
      float values[8]={};
      const half2* weight2=reinterpret_cast<const half2*>(
          ba_weight+static_cast<size_t>(ba_row)*k);
      for(int pair=ba_thread;pair<k/2;pair+=kBAThreadsPerRow) {
        const float2 weight=__half22float2(__ldg(weight2+pair));
#pragma unroll
        for(int token=0;token<8;++token) {
          const half2* input2=reinterpret_cast<const half2*>(
              input+static_cast<size_t>(token)*k);
          const float2 x=__half22float2(__ldg(input2+pair));
          values[token]=fmaf(x.x,weight.x,values[token]);
          values[token]=fmaf(x.y,weight.y,values[token]);
        }
      }
#pragma unroll
      for(int token=0;token<8;++token) {
#pragma unroll
        for(int offset=16;offset>0;offset>>=1)
          values[token]+=__shfl_down_sync(0xffffffffU,values[token],offset);
        if(lane==0) partials[ba_group][token*8+ba_warp]=values[token];
      }
      __syncthreads();
      if(ba_warp==0) {
#pragma unroll
        for(int token=0;token<8;++token) {
          float value=lane<8 ? partials[ba_group][token*8+lane] : 0.f;
#pragma unroll
          for(int offset=16;offset>0;offset>>=1)
            value+=__shfl_down_sync(0xffffffffU,value,offset);
          if(lane==0) {
            if(ba_row<ba_n/2)
              b_output[static_cast<size_t>(token)*(ba_n/2)+ba_row]=
                  __float2half(value);
            else
              a_output[static_cast<size_t>(token)*(ba_n/2)+ba_row-ba_n/2]=
                  __float2half(value);
          }
        }
      }
"""
        + candidate[end:]
    )
    wrapper = r"""
void launch_allrows(torch::Tensor q,torch::Tensor z,torch::Tensor b,torch::Tensor a,
    torch::Tensor x,torch::Tensor codes,torch::Tensor scales,torch::Tensor ba,int) {
  TORCH_CHECK(x.sizes()==torch::IntArrayRef({8,5120}) && x.is_contiguous());
  TORCH_CHECK(q.sizes()==torch::IntArrayRef({8,2560}) && q.is_contiguous());
  TORCH_CHECK(z.sizes()==torch::IntArrayRef({8,1536}) && z.is_contiguous());
  TORCH_CHECK(ba.sizes()==torch::IntArrayRef({24,5120}) && ba.is_contiguous());
  ba_allrows<16,2,true,false,false,true,true><<<140,512,0,
      at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<half*>(z.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(ba.data_ptr<at::Half>()),nullptr,
      reinterpret_cast<half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<half*>(a.data_ptr<at::Half>()),24,2560,4096,5120,8,true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("launch",&launch_allrows); }
"""
    return text + candidate + wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "ba_allrows.cu"
    source.write_text(generate(args.source_root))
    load(
        name="qpn8_ba_allrows_screen",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
