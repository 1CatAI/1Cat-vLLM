// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Register-level building blocks shared by the packed SM70-family GEMMs: the
// e2m1 code decoder, and Turing's m16n8k8 tensor-core MMA with the ldmatrix
// load that feeds it. Volta runs mma.m8n8k4 at full rate; Turing executes it
// at half its tensor rate and gets its full rate only from m16n8k8, which
// every later generation also runs. The m16n8k8 helpers trap below SM75, so
// the SM70 build of a file that also holds an SM75 kernel compiles and fails
// loudly if that kernel ever runs without its sm_75 code.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace vllm::sm70 {

// Derived from TurboMind's cvt_f16x8_e2m1 (lmdeploy, Apache-2.0;
// csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/attention/quantization.h).
// Eight e2m1 codes (one 32-bit word, low nibble first) to fp16 bit patterns:
// sign and exponent/mantissa bits move into fp16 positions, and the 2^14
// exponent re-bias is left to the caller's scale. out[i] pairs codes
// (i, i + 4) of the word; the weight layouts place their nibbles so that
// these pairs are the operand registers the MMA reads.
__device__ __forceinline__ void fp4x8_to_half2x4(unsigned x, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000U;
  constexpr unsigned kExponentMantissa = 0x0E000E00U;
  unsigned values[4];
  values[0] = ((x << 12) & kSign) | ((x << 9) & kExponentMantissa);
  values[1] = ((x << 8) & kSign) | ((x << 5) & kExponentMantissa);
  values[2] = ((x << 4) & kSign) | ((x << 1) & kExponentMantissa);
  values[3] = ((x << 0) & kSign) | ((x >> 3) & kExponentMantissa);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = *reinterpret_cast<const half2*>(&values[i]);
  }
}

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4],
                                            uint32_t address) {
#if __CUDA_ARCH__ >= 750
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
#else
  // Below SM75 a missing sm_75 cubin would otherwise return garbage silently.
  __trap();
#endif
}

// The transposing variant: lane l receives rows 2 * (l % 4) and 2 * (l % 4) + 1
// of column l / 4 of each matrix, which is the B fragment of a row-major
// [k][n] tile.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4],
                                                  uint32_t address) {
#if __CUDA_ARCH__ >= 750
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(address));
#endif
}

// D (16x8, fp32) += A (16x8, row) * B (8x8, col). Lane l holds A rows l / 4
// and l / 4 + 8 at k (l % 4) * 2 + {0, 1} in a0 / a1, B column l / 4 at the
// same k in b0, and D rows l / 4 (d[0], d[1]) and l / 4 + 8 (d[2], d[3]) at
// columns (l % 4) * 2 + {0, 1}.
__device__ __forceinline__ void mma_m16n8k8(float (&d)[4], uint32_t a0,
                                            uint32_t a1, uint32_t b0) {
#if __CUDA_ARCH__ >= 750
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b0));
#else
  __trap();
#endif
}

// m16n8k8 fragment order of a packed FP4 weight tile (32 output columns x 16
// K per code group, 8 bytes per lane, the same tile shape as the m8n8k4
// order). The weights are the 16-row A operand and the activation rows the
// 8-column B operand, so a skinny batch fills whole MMAs: eight rows are one
// B tile. Inside a code group the K order is permuted identically on both
// operands, which leaves the dot product unchanged: lane l (row group
// l / 4, quad position t = l % 4) holds K 4t .. 4t + 3, the first pair for
// MMA step 0 and the second for step 1, so one 8-byte load of an activation
// row feeds both steps.
//
// Code word w (w = 0, 1) of a lane covers A tile w (columns 16w .. 16w + 15);
// nibble n of the word is the weight at column m16n8k8_fp4_column(l, w, n)
// and K 4t + m16n8k8_fp4_k(n). After fp4x8_to_half2x4, out[2s] and
// out[2s + 1] are A registers a0 / a1 of MMA step s. The scale table holds
// four bytes per row group and scale group, for columns l / 4 + 8j
// (j = 0 .. 3); qpn_prepack in
// vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py writes
// this order.
__host__ __device__ constexpr int m16n8k8_fp4_column(int lane, int word,
                                                     int nibble) {
  return 16 * word + (lane >> 2) + 8 * (nibble & 1);
}

__host__ __device__ constexpr int m16n8k8_fp4_k(int nibble) {
  return ((nibble >> 1) & 1) * 2 + (nibble >> 2);
}

}  // namespace vllm::sm70
