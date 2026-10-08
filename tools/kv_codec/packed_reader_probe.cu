// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Exhaust byte pairs, including mixed normal/subnormal/NaN lanes, and compare
// packed conversion and aligned global loads against the original helpers.
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fp8_kv_utils.cuh"

#ifndef KV_CODEC_CANDIDATE
template <int Format, bool Fast, bool Lut>
__device__ __forceinline__ uint4 legacy_packed(uint64_t raw,
                                               const uint16_t* table) {
  if constexpr (Format == 1) {
    if constexpr (Lut) {
      return flash_v100::fp8_e4m3fn_vector_to_half8_lut(raw, table);
    } else if constexpr (Fast) {
      return flash_v100::fp8_e4m3fn_vector_to_half8_fast(raw);
    } else {
      return flash_v100::fp8_e4m3fn_vector_to_half8(raw);
    }
  } else {
    return flash_v100::fp8_e5m2_vector_to_half8(raw);
  }
}
#endif

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(status));
    exit(1);
  }
}

template <int Format, bool Fast, bool Lut>
__global__ void probe(const void* cache, uint4* result, int count) {
  __shared__ uint16_t table[256];
  if constexpr (Lut && Format == 1) {
    table[threadIdx.x] = flash_v100::fp8_e4m3fn_to_half_bits(threadIdx.x);
    __syncthreads();
  }
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int64_t base = static_cast<int64_t>(index - (index & 1)) * 8;
  const int col = index & 1;
  uint4 decoded;
  uint4 loaded;
#ifdef KV_CODEC_CANDIDATE
  using Reader = flash_v100::KVReader<Format>;
  loaded = Reader::template load_half8 < Lut &&
           Format == 1 > (cache, base, col, table);
  if constexpr (Format == 0) {
    decoded = loaded;
  } else {
    const uint64_t raw = static_cast<const uint64_t*>(cache)[index];
    decoded = Reader::template half8_from_packed<Fast, Lut>(raw, table);
  }
#else
  loaded = flash_v100::legacy_load_half8 < Format,
  Lut && Format == 1 > (cache, base, col, table);
  if constexpr (Format == 0) {
    decoded = loaded;
  } else {
    const uint64_t raw = static_cast<const uint64_t*>(cache)[index];
    decoded = legacy_packed<Format, Fast, Lut>(raw, table);
  }
#endif
  result[2 * index] = decoded;
  result[2 * index + 1] = loaded;
}

template <int Format, bool Fast = false, bool Lut = false>
void run(FILE* file) {
  constexpr int count = Format == 0 ? 8192 : 65536;
  constexpr int bytes = Format == 0 ? 16 : 8;
  std::vector<uint64_t> input(count * bytes / 8);
  if constexpr (Format == 0) {
    auto* raw = reinterpret_cast<uint16_t*>(input.data());
    for (int index = 0; index < 65536; ++index) raw[index] = index;
  } else {
    for (int index = 0; index < count; ++index) {
      input[index] = static_cast<uint64_t>(index) |
                     (static_cast<uint64_t>(index ^ 0x5a5a) << 16) |
                     (static_cast<uint64_t>(index ^ 0xff00) << 32) |
                     (static_cast<uint64_t>(index ^ 0x00ff) << 48);
    }
  }
  void* cache = nullptr;
  uint4* output = nullptr;
  check(cudaMalloc(&cache, count * bytes));
  check(cudaMalloc(&output, 2 * count * sizeof(uint4)));
  check(cudaMemcpy(cache, input.data(), count * bytes, cudaMemcpyHostToDevice));
  probe<Format, Fast, Lut><<<(count + 255) / 256, 256>>>(cache, output, count);
  check(cudaGetLastError());
  std::vector<uint4> host(2 * count);
  check(cudaMemcpy(host.data(), output, host.size() * sizeof(uint4),
                   cudaMemcpyDeviceToHost));
  if (fwrite(host.data(), sizeof(uint4), host.size(), file) != host.size())
    exit(1);
  check(cudaFree(output));
  check(cudaFree(cache));
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  FILE* file = fopen(argv[1], "wb");
  if (!file) return 2;
  run<0>(file);
  run<1>(file);
  run<1, true>(file);
  run<1, false, true>(file);
  run<2>(file);
  run<2, true, true>(file);  // Existing packed E5M2 ignores the LUT hint.
  return fclose(file) == 0 ? 0 : 1;
}
