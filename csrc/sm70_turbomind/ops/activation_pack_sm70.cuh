// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

namespace vllm::sm70 {

// Half index of (row, column) in the [K/16, M, 16] layout of pack_k16_input.
__host__ __device__ __forceinline__ size_t k16_half_index(int row, int column,
                                                          int m) {
  return (static_cast<size_t>(column >> 4) * m + row) * 16 + (column & 15);
}

// [M, K] -> [K/16, M, 16]. This packs only activations; quantized weights
// and their numerics remain unchanged. Callers admit contiguous K%16==0.
static __global__ void pack_k16_input(const half* __restrict__ input,
                                      half* __restrict__ packed, int m, int k) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * k / 2) return;
  const int pair = index & 7;
  const int row = (index >> 3) % m;
  const int group = (index >> 3) / m;
  reinterpret_cast<half2*>(packed)[index] = reinterpret_cast<const half2*>(
      input)[static_cast<size_t>(row) * k / 2 + group * 8 + pair];
}

// A producer that also writes its output in K16 layout hands that twin to the
// next projection, which then skips its pack launch. The slot keeps the output
// alive, so an equal data pointer always names the same allocation.
struct K16Handoff {
  torch::Tensor raw;
  torch::Tensor packed;
};

inline K16Handoff& k16_handoff() {
  static thread_local K16Handoff* slot = new K16Handoff();
  return *slot;
}

// A twin for a producer's [M, K] output, or nullptr when K16 does not apply.
inline half* k16_twin_for(const torch::Tensor& output, torch::Tensor& twin) {
  if (output.size(1) % 16 != 0) return nullptr;
  twin = torch::empty_like(output);
  return reinterpret_cast<half*>(twin.data_ptr<at::Half>());
}

inline void k16_publish(const torch::Tensor& raw, const torch::Tensor& twin) {
  auto& slot = k16_handoff();
  slot.raw = raw;
  slot.packed = twin;
}

// The K16 activations of `input`, from the handoff slot or from a pack launch.
// `holder` keeps a freshly packed buffer alive for the caller.
inline half* k16_packed_input(const torch::Tensor& input, torch::Tensor& holder,
                              cudaStream_t stream) {
  constexpr int kThreads = 256;
  const auto launch = [&](const torch::Tensor& into) {
    pack_k16_input<<<(input.numel() / 2 + kThreads - 1) / kThreads, kThreads, 0,
                     stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(into.data_ptr<at::Half>()),
        static_cast<int>(input.size(0)), static_cast<int>(input.size(1)));
  };
  auto& slot = k16_handoff();
  if (slot.raw.defined() && slot.raw.data_ptr() == input.data_ptr() &&
      slot.raw.sizes() == input.sizes()) {
    return reinterpret_cast<half*>(slot.packed.data_ptr<at::Half>());
  }
  holder = torch::empty_like(input);
  launch(holder);
  return reinterpret_cast<half*>(holder.data_ptr<at::Half>());
}

}  // namespace vllm::sm70
