# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a research AR+norm that issues L2 hints after publishing peer packets."""

import argparse
import os
from pathlib import Path


def generate(root):
    header = (root / "csrc/custom_all_reduce.cuh").read_text()
    start = header.index("template <typename WeightT, bool Reference = false>")
    end = header.index("\nclass CustomAllreduce {", start)
    kernel = header[start:end].replace(
        "sm70_push_allreduce_gemma_rms_norm", "qpn_norm_overlap"
    )
    kernel = kernel.replace(
        "template <typename WeightT, bool Reference = false>",
        "template <typename WeightT, int Prefix, bool Reference = false>",
    )
    signature = "float epsilon) {"
    assert kernel.count(signature) == 1
    kernel = kernel.replace(signature, "float epsilon, const uint8_t* codes) {")
    marker = "    P peers[4];"
    assert kernel.count(marker) == 1
    kernel = kernel.replace(
        marker,
        """    if constexpr (Prefix > 0) {
      constexpr int Sectors = 272 * 8 * Prefix * 8;
      for (int i = blockIdx.x * 128 + tid; i < Sectors; i += 40 * 128) {
        const int sector = i & 7;
        const int group = i >> 3;
        const int offset = group % Prefix;
        const int partition = (group / Prefix) & 7;
        const int tile = group / (Prefix * 8);
        const size_t address =
            (static_cast<size_t>(tile) * 320 + partition * 40 + offset) * 256 +
            sector * 32;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(codes + address));
      }
    }
"""
        + marker,
    )
    return (
        "#include <torch/all.h>\n#include <torch/library.h>\n"
        "#include <ATen/cuda/CUDAContext.h>\n"
        "#include <c10/cuda/CUDAGuard.h>\n"
        "#include <c10/cuda/CUDAException.h>\n"
        '#include "custom_all_reduce.cuh"\nnamespace vllm {\n'
        + kernel
        + "\n}\n"
        + r"""
void run(torch::Tensor output, torch::Tensor residual_out, torch::Tensor input,
         torch::Tensor residual, torch::Tensor weight, torch::Tensor codes,
         const std::vector<int64_t>& pointers, int64_t rank,
         int64_t buffer_bytes, int64_t prefix) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
              input.sizes() == at::IntArrayRef({8, 5120}));
  TORCH_CHECK(pointers.size() == 4 && rank >= 0 && rank < 4 &&
              buffer_bytes == vllm::kSm70Tp4PushAllreduceBufferBytes);
  TORCH_CHECK(output.sizes() == input.sizes() &&
              residual.sizes() == input.sizes() &&
              residual_out.sizes() == input.sizes() &&
              weight.sizes() == at::IntArrayRef({5120}));
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 &&
              weight.scalar_type() == torch::kFloat16 &&
              residual.scalar_type() == torch::kFloat32 &&
              residual_out.scalar_type() == torch::kFloat32 &&
              codes.scalar_type() == torch::kUInt8 &&
              codes.numel() == 272 * 320 * 256);
  for (const auto& t : {input, output, residual, residual_out, weight, codes})
    TORCH_CHECK(t.is_contiguous() && t.device() == input.device());
  c10::cuda::CUDAGuard guard(input.device());
  vllm::RankData peers{};
  for (int peer = 0; peer < 4; ++peer)
    peers.ptrs[peer] = reinterpret_cast<void*>(pointers[peer]);
  auto kernel = vllm::qpn_norm_overlap<half, 0>;
  switch (prefix) {
    case 0: break;
    case 1: kernel = vllm::qpn_norm_overlap<half, 1>; break;
    case 2: kernel = vllm::qpn_norm_overlap<half, 2>; break;
    case 4: kernel = vllm::qpn_norm_overlap<half, 4>; break;
    default: TORCH_CHECK(false, "Unsupported warm prefix");
  }
  kernel<<<40, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      peers, reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      residual.data_ptr<float>(),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      residual_out.data_ptr<float>(), rank, 1e-6f, codes.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY(_qpn_norm_overlap700, library) {
  library.def("run(Tensor(a!) output, Tensor(b!) residual_out, Tensor input, "
              "Tensor residual, Tensor weight, Tensor codes, int[] pointers, "
              "int rank, int buffer_bytes, int prefix) -> ()");
}
TORCH_LIBRARY_IMPL(_qpn_norm_overlap700, CUDA, library) {
  library.impl("run", &run);
}
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "norm_overlap.cu"
    source.write_text(generate(args.source_root))
    if args.generate_only:
        return
    from torch.utils.cpp_extension import load

    os.environ.setdefault("MAX_JOBS", "1")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    load(
        name="qpn_norm_overlap700",
        sources=[str(source)],
        extra_include_paths=[str((args.source_root / "csrc").resolve())],
        extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
        extra_ldflags=["-Wl,-Bsymbolic"],
        build_directory=str(args.output),
        is_python_module=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
