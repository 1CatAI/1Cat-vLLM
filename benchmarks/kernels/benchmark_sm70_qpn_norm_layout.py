# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a research-only TP4 norm producer with a K16 output layout.

Only the normalized-output address changes. Residuals, reductions, peer
packets, precision and synchronization retain their production semantics.
"""

import argparse
import os
from pathlib import Path


def generate(root):
    header = (root / "csrc/custom_all_reduce.cuh").read_text()
    start = header.index("template <typename WeightT, bool Reference = false>")
    end = header.index("\nclass CustomAllreduce {", start)
    kernel = header[start:end].replace(
        "sm70_push_allreduce_gemma_rms_norm", "qpn_norm_packed"
    )
    old = '"l"(reinterpret_cast<P*>(output) + pack)'
    assert kernel.count(old) == 1
    kernel = kernel.replace(
        old,
        '"l"(reinterpret_cast<P*>(output) + '
        "(column / 16) * 16 + row * 2 + (column / 8) % 2)",
    )
    wrapper = r"""
void run(torch::Tensor output, torch::Tensor residual_out, torch::Tensor input,
         torch::Tensor residual, torch::Tensor weight,
         const std::vector<int64_t>& pointers, int64_t rank,
         int64_t buffer_bytes, bool packed) {
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
              residual_out.scalar_type() == torch::kFloat32);
  for (const auto& tensor : {input, output, residual, residual_out, weight})
    TORCH_CHECK(tensor.is_contiguous() && tensor.device() == input.device());
  c10::cuda::CUDAGuard guard(input.device());
  vllm::RankData peers{};
  for (int peer = 0; peer < 4; ++peer)
    peers.ptrs[peer] = reinterpret_cast<void*>(pointers[peer]);
  auto kernel = packed ? vllm::qpn_norm_packed<half>
                      : vllm::sm70_push_allreduce_gemma_rms_norm<half>;
  kernel<<<40,128,0,at::cuda::getCurrentCUDAStream()>>>(
      peers, reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      residual.data_ptr<float>(),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      residual_out.data_ptr<float>(), rank, 1e-6f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY(_qpn_norm_layout700, library) {
  library.def("run(Tensor(a!) output, Tensor(b!) residual_out, Tensor input, "
              "Tensor residual, Tensor weight, int[] pointers, int rank, "
              "int buffer_bytes, bool packed) -> ()");
}
TORCH_LIBRARY_IMPL(_qpn_norm_layout700, CUDA, library) {
  library.impl("run", &run);
}
"""
    return (
        "#include <torch/all.h>\n#include <torch/library.h>\n"
        "#include <ATen/cuda/CUDAContext.h>\n"
        "#include <c10/cuda/CUDAGuard.h>\n"
        "#include <c10/cuda/CUDAException.h>\n"
        '#include "custom_all_reduce.cuh"\nnamespace vllm {\n'
        + kernel
        + "\n}\n"
        + wrapper
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generated = args.output / "norm_layout.cu"
    generated.write_text(generate(args.source_root))
    if args.generate_only:
        return
    from torch.utils.cpp_extension import load

    os.environ.setdefault("MAX_JOBS", "1")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    load(
        name="qpn_norm_layout700",
        sources=[str(generated)],
        extra_include_paths=[str((args.source_root / "csrc").resolve())],
        extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
        extra_ldflags=["-Wl,-Bsymbolic"],
        build_directory=str(args.output),
        is_python_module=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
