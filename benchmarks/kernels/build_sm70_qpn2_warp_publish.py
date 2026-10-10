# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build down publication without an extra shared-memory phase boundary.

Gather the final FP16 values within each output-row warp. The existing split
reduction and its single CTA barrier are unchanged. Pair this with consume-only
partial-packet norm; the independent extension does not register a serving op.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_effective_scale import extract_kernel
from benchmark_sm70_qpn2_effective_scale import generate as projection_source
from sm70_tp4_norm_partial_packets import generate as norm_source
from torch.utils.cpp_extension import load


def generate(root):
    norm = norm_source(root, parts=20)
    begin = norm.rindex("template <", 0, norm.index("void partial_packet_norm("))
    end = norm.index("size_t buffer_bytes()", begin)
    consume = norm[begin:end].replace("partial_packet_norm", "consume_packet_norm")
    begin = consume.index("    P value = reinterpret_cast<const P*>(input)[pack];")
    end = consume.index("    P peers[4];", begin)
    consume = consume[:begin] + consume[end:]
    projection = projection_source(root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu")
    projection = projection[: projection.index("PYBIND11_MODULE(")]
    projection = projection.replace("void launch(", "void launch_projection_base(")
    producer = extract_kernel(projection, "nvfp4_qpn2_sm70_kernel_scale0")
    producer = producer.replace("nvfp4_qpn2_sm70_kernel_scale0", "warp_publish_down")
    producer = producer.replace(
        "int m, float global_scale) {",
        "int m, float global_scale, vllm::RankData buffers, int rank) {",
    )
    store = (
        "      output[static_cast<size_t>(output_row) * n + tile * 32 + output_col] =\n"
        "          __float2half(value);"
    )
    assert producer.count(store) == 1
    producer = producer.replace(
        store,
        r"""
      using P = typename vllm::packed_t<half>::P;
      half rounded=__float2half(value);
      vllm::sm70_push_escape_sentinel(rounded);
      const unsigned bits=__half_as_ushort(rounded);
      const unsigned pair=bits|(__shfl_down_sync(0xffffffff,bits,1)<<16);
      uint4 words;
      words.x=__shfl_sync(0xffffffff,pair,(lane&~7)+0);
      words.y=__shfl_sync(0xffffffff,pair,(lane&~7)+2);
      words.z=__shfl_sync(0xffffffff,pair,(lane&~7)+4);
      words.w=__shfl_sync(0xffffffff,pair,(lane&~7)+6);
      if ((lane&7)==0) {
        P payload=*reinterpret_cast<P*>(&words);
        const int pack=output_row*(5120/8)+tile*4+lane/8;
        const char* local=reinterpret_cast<const char*>(buffers.ptrs[rank])+
                          kSm70PushNormOffset;
        const auto* meta=reinterpret_cast<const volatile Sm70PushNormMeta*>(local);
        const uint32_t generation=meta->generation[output_row]+1;
        const int epoch_offset=(generation&1)*4*(8*5120/8);
#pragma unroll
        for (int peer=0;peer<4;++peer) {
          char* destination=const_cast<char*>(reinterpret_cast<const char*>(
              buffers.ptrs[peer]))+kSm70PushNormOffset+kSm70PushNormMetaBytes+
              (epoch_offset+rank*(8*5120/8))*sizeof(P);
          vllm::sm70_push_store_volatile_16b(payload,destination,pack);
        }
      }
""",
    )
    wrapper_begin = norm.index("void launch(torch::Tensor")
    wrapper_end = norm.index("PYBIND11_MODULE(", wrapper_begin)
    wrapper = norm[wrapper_begin:wrapper_end].replace("void launch(", "void consume(")
    wrapper = wrapper.replace(
        "partial_packet_norm<float>", "consume_packet_norm<float>"
    )
    project_wrapper = r"""
void project(torch::Tensor output, torch::Tensor input, torch::Tensor codes,
             torch::Tensor scales, double global,
             const std::vector<int64_t>& buffers, int rank) {
  TORCH_CHECK(input.sizes()==torch::IntArrayRef({8,4352}) && input.is_contiguous());
  TORCH_CHECK(output.sizes()==torch::IntArrayRef({8,5120}) && output.is_contiguous());
  auto stream=at::cuda::getCurrentCUDAStream();
  warp_publish_down<16,2,1,false,false,true><<<160,512,0,stream>>>(
      codes.data_ptr<uint8_t>(),scales.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      5120,4352,8,global,peers(buffers),rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
    marker = "PYBIND11_MODULE("
    norm = norm.replace(
        marker,
        projection + consume + producer + wrapper + project_wrapper + "\n" + marker,
    )
    return norm.replace(
        'm.def("alias",&alias);',
        'm.def("project",&project); m.def("consume",&consume);\n'
        '  m.def("alias",&alias);',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "warp_publish.cu"
    text = generate(args.source_root)
    if not source.exists() or source.read_text() != text:
        source.write_text(text)
    load(
        name="qpn2_warp_publish_norm_screen",
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
