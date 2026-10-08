// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// IQ block layouts follow ggml-org/llama.cpp (MIT; packaged LICENSE).
#pragma once
#include <cstdint>

#ifdef __CUDACC__
  #define GGUF_GROUP_INLINE __host__ __device__ __forceinline__
#else
  #define GGUF_GROUP_INLINE inline
#endif

namespace vllm::sm70_gguf {

// Signed integer words, source FP16 scale bits and integer subscales. Keeping
// this independent of the activation lets multiple tokens reuse one decode.
struct LatticeGroup32 {
  int32_t words[8];
  uint16_t scale_bits;
  uint8_t scale0;
  uint8_t scale1;
};

GGUF_GROUP_INLINE uint32_t group_load_u32(const uint8_t* p) {
#ifdef __CUDA_ARCH__
  return uint32_t(*reinterpret_cast<const uint16_t*>(p)) |
         (uint32_t(*reinterpret_cast<const uint16_t*>(p + 2)) << 16);
#else
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
#endif
}

GGUF_GROUP_INLINE uint32_t group_sign_mask(int nibble) {
#ifdef __CUDA_ARCH__
  const uint32_t bits = uint32_t(nibble) * 0x10204080U;
  uint32_t mask;
  asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(mask) : "r"(bits));
  return mask;
#else
  return ((nibble & 1) ? 0x000000ffU : 0) | ((nibble & 2) ? 0x0000ff00U : 0) |
         ((nibble & 4) ? 0x00ff0000U : 0) | ((nibble & 8) ? 0xff000000U : 0);
#endif
}

GGUF_GROUP_INLINE uint32_t group_restore_sign(uint32_t value, uint32_t mask) {
#ifdef __CUDA_ARCH__
  return __vsub4(value ^ mask, mask);
#else
  uint32_t result = 0;
  for (int i = 0; i < 4; ++i) {
    const uint8_t v = (value >> (i * 8)) & 255;
    const uint8_t m = (mask >> (i * 8)) & 255;
    result |= uint32_t(uint8_t((v ^ m) - m)) << (i * 8);
  }
  return result;
#endif
}

template <int Type, bool BankAware = false>
GGUF_GROUP_INLINE LatticeGroup32 load_lattice_group(const uint8_t* block,
                                                    int sub,
                                                    const uint32_t* book,
                                                    const uint32_t* masks) {
  static_assert(Type == 18 || Type == 21 || Type == 22);
  LatticeGroup32 result{};
  result.scale_bits = uint16_t(block[0]) | (uint16_t(block[1]) << 8);
  const uint32_t low = group_load_u32(block + 2 + sub * (Type == 22 ? 4 : 8));
  uint32_t high = 0;
  if constexpr (Type != 22) high = group_load_u32(block + 6 + sub * 8);
  const uint32_t signs = group_load_u32(block +
                                        (Type == 18   ? 66
                                         : Type == 21 ? 74
                                                      : 34) +
                                        sub * 4);
  const int qh = Type == 18 ? 0 : block[66 + sub];
  if constexpr (Type == 21) {
    result.scale0 = 1 + 2 * ((block[106 + sub / 2] >> (4 * (sub % 2))) & 15);
  } else if constexpr (Type == 18) {
    result.scale0 = 1 + 2 * (signs >> 28);
  } else {
    result.scale0 = 1 + 2 * (block[74 + sub] & 15);
    result.scale1 = 1 + 2 * (block[74 + sub] >> 4);
  }
#pragma unroll
  for (int octet = 0; octet < 4; ++octet) {
    int first, second, sign;
    if constexpr (Type == 22) {
      first =
          (((low >> (octet * 8)) & 255) | (((qh >> (2 * octet)) & 3) << 8)) * 2;
      second = first + 1;
    } else {
      const uint32_t codes = octet < 2 ? low : high;
      const int shift = (octet % 2) * 16;
      first = (codes >> shift) & 255;
      second = (codes >> (shift + 8)) & 255;
      if constexpr (Type == 21) {
        first |= ((qh >> (2 * octet)) & 1) << 8;
        second |= ((qh >> (2 * octet + 1)) & 1) << 8;
      }
    }
    if constexpr (Type == 18) {
      sign = (signs >> (7 * octet)) & 127;
#ifdef __CUDA_ARCH__
      sign |= (__popc(sign) & 1) << 7;
#else
      unsigned parity = unsigned(sign);
      parity ^= parity >> 4;
      parity ^= parity >> 2;
      parity ^= parity >> 1;
      sign |= (parity & 1) << 7;
#endif
    } else {
      sign = (signs >> (8 * octet)) & 255;
    }
    const uint32_t s0 =
        BankAware ? group_sign_mask(sign & 15) : masks[sign & 15];
    const uint32_t s1 =
        BankAware ? group_sign_mask(sign >> 4) : masks[sign >> 4];
    if constexpr (BankAware && Type == 22) {
      first = first / 2 + (first % 2) * 1024;
      second = second / 2 + (second % 2) * 1024;
    }
    result.words[2 * octet] = int32_t(group_restore_sign(book[first], s0));
    result.words[2 * octet + 1] = int32_t(group_restore_sign(book[second], s1));
  }
  return result;
}
}  // namespace vllm::sm70_gguf

#undef GGUF_GROUP_INLINE
