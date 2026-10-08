// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Exhaust the storage bit patterns, including signed zeros, subnormals and
// NaNs.
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fp8_kv_utils.cuh"

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(status));
    exit(1);
  }
}

template <int Format, bool Bits>
__global__ void probe(const void* cache, uint32_t* result, int count,
                      float scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  result[3 * index] = __float_as_uint(
      flash_v100::load_kv_cache_float_unscaled<Format, Bits>(cache, index));
  result[3 * index + 1] = __float_as_uint(
      flash_v100::load_kv_cache_float<Format>(cache, index, scale));
  result[3 * index + 2] = __half_as_ushort(
      flash_v100::load_kv_cache_half<Format>(cache, index, scale));
}

template <int Format, bool Bits>
void run(FILE* file) {
  constexpr int count = Format == flash_v100::KV_CACHE_DTYPE_FP16 ? 65536 : 256;
  constexpr int bytes = Format == flash_v100::KV_CACHE_DTYPE_FP16 ? 2 : 1;
  std::vector<uint16_t> input(count);
  for (int index = 0; index < count; ++index) input[index] = index;
  if (bytes == 1) {
    auto* raw = reinterpret_cast<uint8_t*>(input.data());
    for (int index = 0; index < count; ++index) raw[index] = index;
  }
  void* cache = nullptr;
  uint32_t* output = nullptr;
  check(cudaMalloc(&cache, count * bytes));
  check(cudaMalloc(&output, count * 3 * sizeof(uint32_t)));
  check(cudaMemcpy(cache, input.data(), count * bytes, cudaMemcpyHostToDevice));
  std::vector<uint32_t> host(count * 3);
  for (const float scale : {1.0f, 0.125f, 0.003142f, 128.0f}) {
    probe<Format, Bits>
        <<<(count + 255) / 256, 256>>>(cache, output, count, scale);
    check(cudaGetLastError());
    check(cudaMemcpy(host.data(), output, host.size() * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    if (fwrite(host.data(), sizeof(uint32_t), host.size(), file) !=
        host.size()) {
      fprintf(stderr, "Failed to write probe output\n");
      exit(1);
    }
  }
  check(cudaFree(output));
  check(cudaFree(cache));
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  FILE* file = fopen(argv[1], "wb");
  if (!file) return 2;
  run<flash_v100::KV_CACHE_DTYPE_FP16, false>(file);
  run<flash_v100::KV_CACHE_DTYPE_FP8_E4M3, false>(file);
  run<flash_v100::KV_CACHE_DTYPE_FP8_E4M3, true>(file);
  run<flash_v100::KV_CACHE_DTYPE_FP8_E5M2, false>(file);
  return fclose(file) == 0 ? 0 : 1;
}
