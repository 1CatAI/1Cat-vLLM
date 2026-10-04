// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "gguf_lattice_raw.cuh"

namespace vllm::sm70_gguf {

// A permutation of source bits: no expanded scale, sign, or index storage.
// A 32-column tile stores 32 K octets of tightly packed index/sign packets,
// followed by the original FP16 d plane and original small-scale byte plane.
template <int Type>
struct LatticeCompactDecoder : LatticeRawDecoder<Type> {
  static constexpr int kPacketBits = Type == 21 ? 26 : 18;
  static constexpr int kPacketBytesPerRow = kPacketBits * 32 / 8;
  static constexpr int kScaleBytes = Type == 21 ? 4 : 8;
  static constexpr int kBlockBytes = Type == 21 ? 110 : 82;
  static_assert(kPacketBytesPerRow + 2 + kScaleBytes == kBlockBytes);

  struct Parameters {
    float d;
    uint64_t scales;
  };

  __device__ static uint32_t source_packet(const uint8_t* block, int octet) {
    const uint32_t high = block[66 + octet / 4];
    if constexpr (Type == 21) {
      const uint32_t first =
          block[2 + 2 * octet] | (((high >> (2 * (octet % 4))) & 1) << 8);
      const uint32_t second =
          block[3 + 2 * octet] | (((high >> (2 * (octet % 4) + 1)) & 1) << 8);
      return first | (second << 9) | (uint32_t{block[74 + octet]} << 18);
    } else {
      const uint32_t index =
          block[2 + octet] | (((high >> (2 * (octet % 4))) & 3) << 8);
      return index | (uint32_t{block[34 + octet]} << 10);
    }
  }

  template <bool FullWidth = false>
  __device__ static Parameters parameters(const uint8_t* tile, int width,
                                          int col) {
    if constexpr (FullWidth) width = 32;
    const auto* metadata = tile + width * kPacketBytesPerRow;
    Parameters result{};
    result.d = __half2float(reinterpret_cast<const half*>(metadata)[col]);
    const auto* scales = metadata + width * 2 + col * kScaleBytes;
    if constexpr (FullWidth) {
      if constexpr (Type == 21)
        result.scales = *reinterpret_cast<const uint32_t*>(scales);
      else
        result.scales = *reinterpret_cast<const uint64_t*>(scales);
    } else {
#pragma unroll
      for (int i = 0; i < kScaleBytes; ++i)
        result.scales |= uint64_t{scales[i]} << (8 * i);
    }
    return result;
  }

  struct PacketWindow {
    uint64_t loaded;
    int bias;
  };

  // Fetch is separate from shuffling so a caller can prefetch the next
  // original-bit packet window while decoding and multiplying the current one.
  template <bool FullWidth = false>
  __device__ static PacketWindow fetch(const uint8_t* tile, int width,
                                       int octet) {
    const int lane = threadIdx.x % 32;
    if constexpr (FullWidth) {
      const auto* words =
          reinterpret_cast<const uint64_t*>(tile) + octet * (kPacketBits / 2);
      return {lane < kPacketBits / 2 ? words[lane] : 0, 0};
    } else {
      const auto address = reinterpret_cast<uintptr_t>(tile);
      const int64_t bit =
          (address & 7) * 8 + int64_t{octet} * width * kPacketBits;
      const auto* words =
          reinterpret_cast<const uint64_t*>(address & ~uintptr_t{7});
      const int bias = bit % 64, count = (bias + width * kPacketBits + 63) / 64;
      return {lane < count ? words[bit / 64 + lane] : 0, bias};
    }
  }

  __device__ static uint32_t extract(PacketWindow window, int col) {
    const int bit = window.bias + col * kPacketBits;
    const int source = bit / 64, shift = bit % 64;
    const uint64_t low = __shfl_sync(0xffffffffU, window.loaded, source);
    const uint64_t high = __shfl_sync(0xffffffffU, window.loaded, source + 1);
    const uint64_t value =
        shift ? (low >> shift) | (high << (64 - shift)) : low;
    return value & ((uint32_t{1} << kPacketBits) - 1);
  }

  // Every lane participates, including masked output columns. N tails keep
  // one continuous bitstream, with no per-octet alignment padding.
  template <bool FullWidth = false>
  __device__ static uint32_t packet(const uint8_t* tile, int width, int octet,
                                    int col) {
    return extract(fetch<FullWidth>(tile, width, octet), col);
  }

  struct ScalarWindow {
    uint64_t low;
    uint64_t high;
    int shift;
  };

  // Warp-parallel K for vector decode: lanes read different K octets.
  __device__ static ScalarWindow scalar_fetch(const uint8_t* tile, int width,
                                              int octet, int col) {
    const auto address = reinterpret_cast<uintptr_t>(tile);
    const int64_t bit =
        (address & 7) * 8 + (int64_t{octet} * width + col) * kPacketBits;
    const auto* words =
        reinterpret_cast<const uint64_t*>(address & ~uintptr_t{7});
    return {words[bit / 64], words[bit / 64 + 1], int(bit % 64)};
  }

  __device__ static uint32_t scalar_extract(ScalarWindow window) {
    const auto [low, high, shift] = window;
    const uint64_t value =
        shift ? (low >> shift) | (high << (64 - shift)) : low;
    return value & ((uint32_t{1} << kPacketBits) - 1);
  }

  __device__ static uint32_t scalar_packet(const uint8_t* tile, int width,
                                           int octet, int col) {
    return scalar_extract(scalar_fetch(tile, width, octet, col));
  }

  __device__ static turbomind::Array<float, 8> fragment(Parameters parameters,
                                                        uint32_t packet,
                                                        int octet,
                                                        const uint8_t* grid) {
    uint64_t packed;
    uint32_t signs;
    float scale;
    if constexpr (Type == 21) {
      const int nibble = (parameters.scales >> (4 * (octet / 4))) & 15;
      scale = parameters.d * (1 + 2 * nibble);
      const uint32_t first = packet & 511, second = (packet >> 9) & 511;
      packed = *reinterpret_cast<const uint32_t*>(grid + first * 4) |
               (uint64_t{*reinterpret_cast<const uint32_t*>(grid + second * 4)}
                << 32);
      signs = packet >> 18;
    } else {
      const int nibble = (parameters.scales >> (4 * (octet / 2))) & 15;
      scale = (parameters.d * (0.5f + nibble)) * 0.25f;
      packed = *reinterpret_cast<const uint64_t*>(grid + (packet & 1023) * 8);
      signs = packet >> 10;
    }
    turbomind::Array<float, 8> result;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float value = static_cast<int>((packed >> (8 * i)) & 255) - 128;
      result[i] = scale * value * ((signs >> i) & 1 ? -1.f : 1.f);
    }
    return result;
  }
};
}  // namespace vllm::sm70_gguf
