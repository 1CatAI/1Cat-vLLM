// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "gguf_iq3_nibble_book.cuh"
#include "gguf_iq4_xs_native.cuh"
#include "gguf_lattice_raw.cuh"

namespace vllm::sm70_gguf {
// Each reader consumes one current K128 record. It owns only its original
// source metadata and advances pointers; it never stages the following record.
template <int Type>
struct NativePairReader;

template <>
struct NativePairReader<18> {
  using Decoder = LatticeRawDecoder<18>;
  static constexpr int kBlockBytes = 98;
  static constexpr int kBookId = 18;
  static constexpr int kBookBytes = Decoder::kCodebookBytes;
  struct Record {
    uint32_t indices[8];
    uint32_t sign_scale[4];
    float d;
  };
  const uint8_t* payload;
  const half* original_d;
  float cached_d;
  int half_block;
  bool first;

  __device__ NativePairReader(const uint8_t* source, int tile, int blocks_k,
                              int first_part, int col) {
    const uint8_t* macro = source + int64_t{tile} * blocks_k * 32 * kBlockBytes;
    payload = macro + first_part * 1536 + col * 16;
    original_d = reinterpret_cast<const half*>(macro + blocks_k * 3072 +
                                               (first_part / 2) * 64 + col * 2);
    cached_d = 0;
    half_block = first_part & 1;
    first = true;
  }

  __device__ static void initialize(uint8_t* book) {
    Decoder::initialize(book);
  }

  __device__ Record load() {
    if (first || half_block == 0) cached_d = __half2float(*original_d);
    const uint4 a = *reinterpret_cast<const uint4*>(payload);
    const uint4 b = *reinterpret_cast<const uint4*>(payload + 512);
    const uint4 c = *reinterpret_cast<const uint4*>(payload + 1024);
    Record record{{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w},
                  {c.x, c.y, c.z, c.w},
                  cached_d};
    payload += 1536;
    original_d += half_block * 32;
    half_block ^= 1;
    first = false;
    return record;
  }

  template <int Segment, int Fragment>
  __device__ static turbomind::Array<half, 8> fragment(const Record& record,
                                                       const uint8_t* book) {
    static_assert(Segment >= 0 && Segment < 8 && Fragment >= 0 && Fragment < 2);
    constexpr int octet = 2 * Segment + Fragment;
    const uint32_t indices = record.indices[octet / 2] >> ((octet & 1) * 16);
    const uint32_t aux = record.sign_scale[octet / 4];
    const uint32_t sign_index = (aux >> (7 * (octet & 3))) & 127;
    const uint32_t signs = sign_index | ((__popc(sign_index) & 1) << 7);
    const uint32_t a =
        *reinterpret_cast<const uint32_t*>(book + (indices & 255) * 4);
    const uint32_t b =
        *reinterpret_cast<const uint32_t*>(book + ((indices >> 8) & 255) * 4);
    // The existing original-block decoder owns operand formation and its
    // exact FP16 proof; this reader only reconstructs original fields.
    return Decoder::table_fragment<half>(a | (uint64_t{b} << 32), signs,
                                         record.d, aux >> 28);
  }
};

template <>
struct NativePairReader<21> {
  using Decoder = Iq3NibbleBookDecoder;
  static constexpr int kBlockBytes = 110;
  static constexpr int kBookId = 21;
  static constexpr int kBookBytes = Decoder::kSignedCodebookBytes;
  struct Record {
    uint32_t words[13];
    Decoder::Parameters params;
    int half_block;
  };
  const uint8_t* payload;
  const uint32_t* tail;
  const half* original_d;
  const uint32_t* original_scales;
  Decoder::Parameters cache;
  int half_block;
  bool first;

  __device__ NativePairReader(const uint8_t* source, int tile, int blocks_k,
                              int first_part, int col) {
    const uint8_t* macro = source + int64_t{tile} * blocks_k * 32 * kBlockBytes;
    payload = macro + first_part * 1536 + col * 16;
    tail = reinterpret_cast<const uint32_t*>(macro + blocks_k * 3072 +
                                             first_part * 128 + col * 4);
    const uint8_t* metadata = macro + blocks_k * 3328 + (first_part / 2) * 192;
    original_d = reinterpret_cast<const half*>(metadata + col * 2);
    original_scales =
        reinterpret_cast<const uint32_t*>(metadata + 64 + col * 4);
    cache = {};
    half_block = first_part & 1;
    first = true;
  }

  __device__ static void initialize(uint8_t* book) {
    Decoder::initialize<true>(book);
  }

  __device__ Record load() {
    if (first || half_block == 0) {
      const half d = *original_d;
      cache.d = __half2float(d);
      cache.base = __halves2half2(d, d);
      cache.scales = *original_scales;
    }
    const uint4 a = *reinterpret_cast<const uint4*>(payload);
    const uint4 b = *reinterpret_cast<const uint4*>(payload + 512);
    const uint4 c = *reinterpret_cast<const uint4*>(payload + 1024);
    Record record{
        {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, *tail},
        cache,
        half_block};
    payload += 1536;
    tail += 32;
    original_d += half_block * 96;
    original_scales += half_block * 48;
    half_block ^= 1;
    first = false;
    return record;
  }

  template <int Segment, int Fragment>
  __device__ static turbomind::Array<half, 8> fragment(const Record& record,
                                                       const uint8_t* book) {
    static_assert(Segment >= 0 && Segment < 8 && Fragment >= 0 && Fragment < 2);
    constexpr int bit = (Segment * 4 + Fragment * 2) * 13;
    const int nibble =
        (record.params.scales >> (record.half_block * 16 + (Segment / 2) * 4)) &
        15;
    return Decoder::fragment_signed<half, true>(
        record.params, nibble, Decoder::signed_index<bit>(record.words),
        Decoder::signed_index<bit + 13>(record.words), book);
  }
};

template <>
struct NativePairReader<23> {
  using Decoder = Iq4XsNativeDecoder;
  static constexpr int kBlockBytes = 136;
  static constexpr int kBookId = -1;
  static constexpr int kBookBytes = 0;
  struct Record {
    uint4 packets[4];
    Decoder::Parameters params;
    int half_block;
  };
  const uint8_t* payload;
  const half* original_d;
  const uint16_t* original_hi;
  const uint32_t* original_lo;
  Decoder::Parameters cache;
  int half_block;
  bool first;

  __device__ NativePairReader(const uint8_t* source, int tile, int blocks_k,
                              int first_part, int col) {
    const uint8_t* macro = source + int64_t{tile} * blocks_k * 32 * kBlockBytes;
    const int block = first_part / 2;
    payload = macro + first_part * 2048 + col * 16;
    original_d = reinterpret_cast<const half*>(macro + blocks_k * 4096 +
                                               block * 64 + col * 2);
    original_hi = reinterpret_cast<const uint16_t*>(macro + blocks_k * 4160 +
                                                    block * 64 + col * 2);
    original_lo = reinterpret_cast<const uint32_t*>(macro + blocks_k * 4224 +
                                                    block * 128 + col * 4);
    cache = {};
    half_block = first_part & 1;
    first = true;
  }

  __device__ static void initialize(uint8_t*) {}

  __device__ Record load() {
    if (first || half_block == 0) {
      cache.d = __half2float(*original_d);
      cache.scales_hi = *original_hi;
      cache.scales_lo = *original_lo;
    }
    Record record{{*reinterpret_cast<const uint4*>(payload),
                   *reinterpret_cast<const uint4*>(payload + 512),
                   *reinterpret_cast<const uint4*>(payload + 1024),
                   *reinterpret_cast<const uint4*>(payload + 1536)},
                  cache,
                  half_block};
    payload += 2048;
    original_d += half_block * 32;
    original_hi += half_block * 32;
    original_lo += half_block * 32;
    half_block ^= 1;
    first = false;
    return record;
  }

  template <int Segment, int Fragment>
  __device__ static turbomind::Array<half, 8> fragment(const Record& record,
                                                       const uint8_t*) {
    static_assert(Segment >= 0 && Segment < 8 && Fragment >= 0 && Fragment < 2);
    const uint4 data = record.packets[Segment / 2];
    uint32_t packet;
    if constexpr (Segment % 2 == 0)
      packet = Fragment == 0 ? data.x : data.y;
    else
      packet = Fragment == 0 ? data.z : data.w;
    return Decoder::fragment<half>(record.params,
                                   record.half_block * 4 + Segment / 2, packet);
  }
};
}  // namespace vllm::sm70_gguf
