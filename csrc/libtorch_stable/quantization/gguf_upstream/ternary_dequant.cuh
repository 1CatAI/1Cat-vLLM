// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Storage/element-order oracle: llama.cpp ggml-quants.c, pinned in the
// adjacent manifest (MIT). Newly written CUDA decoder for TQ1_0/TQ2_0.
#pragma once

template <typename scalar_t, int type>
__global__ void dequantize_ternary(const uint8_t* input, scalar_t* output,
                                   int64_t blocks) {
  const int64_t block = blockIdx.x;
  const int j = threadIdx.x;
  if (block >= blocks) return;
  constexpr int bytes =
      type == GGML_TYPE_TQ1_0 ? sizeof(block_tq1_0) : sizeof(block_tq2_0);
  const uint8_t* packed = input + block * bytes;
  const float scale =
      __half2float(*reinterpret_cast<const half*>(packed + bytes - 2));
  int code;
  if constexpr (type == GGML_TYPE_TQ2_0) {
    const int byte = (j / 128) * 32 + j % 32;
    code = ((packed[byte] >> (2 * ((j % 128) / 32))) & 3) - 1;
  } else {
    int byte, digit;
    if (j < 160) {
      byte = j % 32;
      digit = j / 32;
    } else if (j < 240) {
      byte = 32 + (j - 160) % 16;
      digit = (j - 160) / 16;
    } else {
      byte = 48 + (j - 240) % 4;
      digit = (j - 240) / 4;
    }
    int power = 1;
    for (int n = 0; n < digit; ++n) power *= 3;
    const uint8_t shifted = static_cast<uint8_t>(packed[byte] * power);
    code = ((static_cast<uint16_t>(shifted) * 3) >> 8) - 1;
  }
  output[block * 256 + j] = static_cast<scalar_t>(scale * code);
}
