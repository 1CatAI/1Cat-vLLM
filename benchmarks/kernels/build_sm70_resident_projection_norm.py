# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# CUDA source anchors retain their original spelling.
# ruff: noqa: E501
"""Extend the resident norm suffix to the original FP8 GDN out projection.

The twelve original split warps retain their sums. Four additional warps are
inactive during GEMM and join the norm suffix's padded reduction. No precision
change, global grid barrier or second consumer kernel is introduced.
"""

import argparse
from pathlib import Path

import regex as re
from benchmark_sm70_qpn2_effective_scale import extract_kernel
from benchmark_sm70_qpn2_product_lookup import function_end
from benchmark_sm70_qpn2_warp_input import WARP_INPUT
from benchmark_sm70_qpn8_two_phase import generate as fp8_source
from build_sm70_qpn2_resident_down_norm import generate as down_source
from torch.utils.cpp_extension import load


def generate(root, native_protocol=False, packed_mlp=False):
    text = down_source(root)
    text = text[: text.index("PYBIND11_MODULE(")]
    fp8 = fp8_source(root / "csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu")
    fp8 = fp8[: fp8.index("PYBIND11_MODULE(")]
    fp8 = fp8.replace("void launch(", "void launch_fp8_projection_control(")
    producer = extract_kernel(
        (root / "csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu").read_text(),
        "fp8_qpn8_sm70_kernel",
    )
    producer = producer.replace("fp8_qpn8_sm70_kernel", "resident_out_norm")
    producer = producer.replace(
        "__global__ void", "__global__ __launch_bounds__(512,2) void"
    )
    producer = producer.replace(
        "bool channel_scales) {",
        "bool channel_scales, vllm::RankData buffers,int rank,const void* local_buffer,"
        "const float* residual,const float* norm_weight,half* normalized,float* residual_out) {",
    )
    # Only the first twelve warps own original K slices. Keep every thread at
    # the subsequent CTA barrier and at the uniform block-local norm suffix.
    begin = producer.index("#pragma unroll 4\n  for (int group =")
    end = producer.index("  __syncthreads();", begin)
    producer = (
        producer[:begin]
        + "  if(warp<SplitK) {\n"
        + producer[begin:end]
        + "  }\n"
        + producer[end:]
    )
    begin = text.rindex("template <", 0, text.index("void resident_down_norm("))
    down = text[begin : function_end(text, begin)]
    begin = down.index("      using P =")
    # The publication epilogue has no blank lines: copy its complete payload
    # block up to the close of the output-row predicate.
    marker = "          vllm::sm70_push_store_volatile_16b(payload,destination,pack);"
    assert down.count(marker) == 1
    end = down.index(marker) + len(marker)
    end = down.index("\n      }", end) + len("\n      }")
    payload = down[begin:end]
    old = "          output[static_cast<size_t>(output_row) * n + col] =\n              __float2half(value);"
    assert producer.count(old) == 1
    producer = producer.replace(old, payload)
    producer = (
        producer[: producer.rindex("}")]
        + r"""
  __syncthreads();
  if(blockIdx.x<40)
    resident_down_norm_suffix<float>(buffers,nullptr,residual,norm_weight,
        normalized,residual_out,rank,local_buffer,1e-6f);
}
"""
    )
    wrapper = r"""
void fused_out(torch::Tensor normalized,torch::Tensor residual_out,torch::Tensor input,
    torch::Tensor residual,torch::Tensor norm_weight,torch::Tensor codes,
    torch::Tensor scales,const std::vector<int64_t>& buffers,int rank) {
  TORCH_CHECK(input.sizes()==torch::IntArrayRef({8,1536}) && input.is_contiguous());
  TORCH_CHECK(normalized.sizes()==torch::IntArrayRef({8,5120}) && normalized.is_contiguous());
  TORCH_CHECK(residual.scalar_type()==torch::kFloat32 && norm_weight.scalar_type()==torch::kFloat32);
  auto kernel=resident_out_norm<12,2,true,false,false,false,false>;
  int device,resident;cudaDeviceProp properties;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,512,0));
  TORCH_CHECK(properties.major==7 && properties.minor==0 && properties.cooperativeLaunch &&
              resident*properties.multiProcessorCount>=160,"Requires all out producers resident");
  auto* c=codes.data_ptr<uint8_t>();const auto* s=reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  half* raw=nullptr;const half* ba=nullptr;
  int ba_n=0,n=5120,k=1536,m=8,qkv_n=5120;bool channel=true;
  auto addresses=peers(buffers);const void* local=reinterpret_cast<void*>(buffers.at(rank));
  const auto* r=residual.data_ptr<float>();const auto* w=norm_weight.data_ptr<float>();
  auto* y=reinterpret_cast<half*>(normalized.data_ptr<at::Half>());
  auto* ro=residual_out.data_ptr<float>();
  void* args[]={&c,&s,&x,&raw,&raw,&ba,&raw,&raw,&raw,&ba_n,&qkv_n,&n,&k,&m,&channel,
                &addresses,&rank,&local,&r,&w,&y,&ro};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel),dim3(160),
      dim3(512),args,0,at::cuda::getCurrentCUDAStream()));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("buffer_bytes",&buffer_bytes);m.def("initialize",&initialize);
  m.def("fused",&fused);m.def("out",&fused_out);
}
"""
    generated = text + fp8 + producer + wrapper
    if native_protocol:
        native = (root / "csrc/custom_all_reduce.cuh").read_text()
        begin = native.index("template <typename WeightT, bool Reference = false>")
        suffix = native[begin : function_end(native, begin)]
        suffix = suffix.replace(
            "sm70_push_allreduce_gemma_rms_norm", "resident_down_norm_suffix"
        )
        suffix = suffix.replace(
            "__global__ __launch_bounds__(128, 1)", "__device__ __forceinline__"
        )
        suffix = suffix.replace(
            "constexpr int Threads = 128", "constexpr int Threads = 512"
        )
        begin = suffix.index("    P value = reinterpret_cast<const P*>(input)[pack];")
        end = suffix.index("    P peers[4];", begin)
        suffix = suffix[:begin] + suffix[end:]
        begin = generated.rindex(
            "template <", 0, generated.index("void resident_down_norm_suffix(")
        )
        end = function_end(generated, begin)
        generated = generated[:begin] + suffix + generated[end:]
        old = "const auto* meta=reinterpret_cast<const volatile Sm70PushNormMeta*>(local);"
        assert generated.count(old) == 3
        generated = generated.replace(
            old,
            "const auto* meta=reinterpret_cast<const volatile Sm70PushNormPacketMeta*>("
            "local+kSm70PushNormPacketOffset-kSm70PushNormOffset);",
        )
        marker = "  initialize_partials<<<1,128,0,stream>>>(base);"
        assert generated.count(marker) == 1
        generated = generated.replace(
            marker,
            "  C10_CUDA_CHECK(cudaMemsetAsync(base+kSm70PushNormPacketOffset,0,"
            "kSm70PushNormMetaBytes,stream));\n" + marker,
        )
    if packed_mlp:
        assert native_protocol
        begin = generated.rindex(
            "template <", 0, generated.index("void resident_down_norm_suffix(")
        )
        end = function_end(generated, begin)
        suffix = generated[begin:end].replace(
            "bool Reference = false>", "bool Reference = false,bool Packed = false>"
        )
        suffix = suffix.replace(
            "reinterpret_cast<P*>(output) + pack",
            "reinterpret_cast<P*>(output)+(Packed ? "
            "((part*PacksPerPart+tid)/2*16+row*2+(part*PacksPerPart+tid)%2) : pack)",
        )
        generated = generated[:begin] + suffix + generated[end:]
        begin = generated.rindex(
            "template <", 0, generated.index("void resident_down_norm(")
        )
        end = function_end(generated, begin)
        down = generated[begin:end]
        start = down.index("      uint4 input01 =")
        stop = down.index("      VLLM_SM70_QPN2_MMA", start)
        down = (
            down[:start]
            + "      constexpr bool WarpLoad=false;\n"
            + WARP_INPUT
            + down[stop:]
        )
        generated = generated[:begin] + down + generated[end:]
        begin = generated.index("void resident_out_norm(")
        end = function_end(generated, begin)
        out = generated[begin:end].replace(
            "resident_down_norm_suffix<float>(",
            "resident_down_norm_suffix<float,false,true>(",
        )
        generated = generated[:begin] + out + generated[end:]
        # Two research modules can coexist in one TP worker. Distinct kernel
        # and wrapper symbols prevent dynamic-linker interposition between
        # the ordinary and packed protocol arms.
        for symbol in (
            "resident_down_norm_suffix",
            "resident_down_norm",
            "resident_out_norm",
            "fused_out",
            "fused",
        ):
            generated = re.sub(r"\b" + symbol + r"\b", symbol + "_packed", generated)
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-protocol", action="store_true")
    parser.add_argument("--packed-mlp", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "resident_projection_norm.cu"
    source.write_text(generate(args.source_root, args.native_protocol, args.packed_mlp))
    load(
        name=(
            "sm70_resident_projection_packed_screen"
            if args.packed_mlp
            else "sm70_resident_projection_norm_screen"
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
