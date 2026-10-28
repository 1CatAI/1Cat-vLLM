// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// QPN8 code layout shared by the SM70 packed GEMMs and the SM75 MMA GEMM.
// Each 32-column tile and 16-deep K group is stored as 32 lanes x 16 bytes:
// lane l holds column qpn8_col_from_lane(l), its 16 K values reordered so the
// fast decoder yields the half2 pairs (k, k + 1) for k = 0, 2, ..., 14.
// The layout derives from dnv2003/v100-skinny (MIT); see LICENSE.v100-skinny.
#pragma once

#include <cuda_fp16.h>

namespace vllm::sm70 {

// E4M3 bits placed into FP16 positions; the 2^8 exponent offset lives in the
// prescaled group scales.
__device__ __forceinline__ void fp8x8_to_half2x4_fast(uint2 q, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x3f803f80u;
  unsigned permuted[4];
  permuted[0] = __byte_perm(q.x, q.y, 0x0400);
  permuted[1] = __byte_perm(q.x, q.y, 0x0501);
  permuted[2] = __byte_perm(q.x, q.y, 0x0602);
  permuted[3] = __byte_perm(q.x, q.y, 0x0703);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned value =
        ((permuted[i] << 8) & kSign) | ((permuted[i] << 7) & kExponentMantissa);
    out[i] = *reinterpret_cast<const half2*>(&value);
  }
}

__device__ __forceinline__ int qpn8_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

}  // namespace vllm::sm70
