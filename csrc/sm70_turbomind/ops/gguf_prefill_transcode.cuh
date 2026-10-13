// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Original GGUF expert rows -> TurboMind canonical grouped-GEMM storage, in
// one pass and without temporaries, for prefill of experts kept in their
// original encoding (decode keeps using the original bytes).
//
// Produces exactly what gguf_device_transcode.py's codecs followed by the
// gguf_{lattice,lut4,affine}_sm70_prepare operators produce:
//   weight packet (n, k8) at ((n / 32) * (K / 8) + k8) * 32 + n % 32
//   stats (g, n) at g * N + n
// Lattice packets are the codec's 16-bit packet verbatim; LUT4 packets hold
// nibbles c0,c2,c4,c6,c1,c3,c5,c7; U2 packets hold code j at bit 2 * p(j),
// p(j) = j / 2 + (j % 2) * 4.
//
// One CTA converts one pack of 32 output columns of one expert. Its source
// rows are contiguous, so the CTA first copies them into shared memory with
// 8-byte loads (coalesced, also for zero-copy reads of pinned host memory),
// then decodes from shared memory. With `interleave`, output column 2r is
// row r of the first source and 2r + 1 row r of the second (gate, up), the
// order TurboMind's gated-SiLU epilogue expects.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace vllm::sm70_gguf::transcode {

constexpr int kPack = 32;
constexpr int kThreads = 256;

template <int Type>
struct Format;

// Lattice formats: 256-value blocks; U2 carrier, 16-bit packet per 8 values.
template <>
struct Format<21> {  // IQ3_S
  static constexpr int kBlock = 256, kBlockBytes = 110, kGroup = 32;
  using Stat = uint64_t;
};
template <>
struct Format<18> {  // IQ3_XXS
  static constexpr int kBlock = 256, kBlockBytes = 98, kGroup = 32;
  using Stat = uint64_t;
};
template <>
struct Format<22> {  // IQ2_S
  static constexpr int kBlock = 256, kBlockBytes = 82, kGroup = 16;
  using Stat = uint32_t;
};
// LUT4: U4 carrier, 32-bit packet per 8 values, FP16 scale per 32 values.
template <>
struct Format<20> {  // IQ4_NL
  static constexpr int kBlock = 32, kBlockBytes = 18, kGroup = 32;
  using Stat = uint16_t;
};
template <>
struct Format<23> {  // IQ4_XS
  static constexpr int kBlock = 256, kBlockBytes = 136, kGroup = 32;
  using Stat = uint16_t;
};
// Q2_0: U2 carrier; stats are FP16 (scale, min = -scale) pairs.
template <>
struct Format<42> {
  static constexpr int kBlock = 64, kBlockBytes = 18, kGroup = 32;
  using Stat = uint32_t;
};

__device__ __forceinline__ float f16(const uint8_t* p) {
  return __half2float(__ushort_as_half(uint16_t(p[0]) | (uint16_t(p[1]) << 8)));
}

// Round to FP16 like torch's .half(); flag finite values that overflow.
__device__ __forceinline__ uint16_t to_f16(float value, int* error) {
  const __half h = __float2half_rn(value);
  if (isfinite(value) && !isfinite(__half2float(h))) atomicOr(error, 1);
  return __half_as_ushort(h);
}

// Decodes one group (kGroup values) of one row: writes the packets of the
// group's octets to `packets` and returns the stats word. `row` is the
// row's source bytes, `k0` the group's first logical column within the
// source row.
template <int Type>
struct Group;

// Packets of one group: uint16 per octet for U2 carriers (lattice, Q2_0),
// uint32 for U4 (LUT4).
template <int Type>
using PacketOf =
    std::conditional_t<Type == 20 || Type == 23, uint32_t, uint16_t>;

template <>
struct Group<21> {
  __device__ static uint64_t decode(const uint8_t* row, int k0,
                                    uint16_t (&packets)[4], int* error) {
    const uint8_t* b = row + (k0 / 256) * 110;
    const int sub = (k0 % 256) / 32;
    const int qh = b[66 + sub];
#pragma unroll
    for (int q = 0; q < 4; ++q)
      packets[q] =
          uint16_t(b[2 + sub * 8 + 2 * q] | (b[2 + sub * 8 + 2 * q + 1] << 8));
    const int nib = (b[106 + sub / 2] >> (4 * (sub % 2))) & 15;
    const uint16_t scale = to_f16(f16(b) * float(1 + 2 * nib), error);
    uint64_t signs = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s)
      signs |= uint64_t(b[74 + sub * 4 + s]) << (8 * s);
    return uint64_t(scale) | (signs << 16) | (uint64_t(qh & 255) << 48);
  }
};

template <>
struct Group<18> {
  __device__ static uint64_t decode(const uint8_t* row, int k0,
                                    uint16_t (&packets)[4], int* error) {
    const uint8_t* b = row + (k0 / 256) * 98;
    const int sub = (k0 % 256) / 32;
#pragma unroll
    for (int q = 0; q < 4; ++q)
      packets[q] =
          uint16_t(b[2 + sub * 8 + 2 * q] | (b[2 + sub * 8 + 2 * q + 1] << 8));
    const uint8_t* m = b + 66 + sub * 4;
    const uint32_t meta = uint32_t(m[0]) | (uint32_t(m[1]) << 8) |
                          (uint32_t(m[2]) << 16) | (uint32_t(m[3]) << 24);
    // Same FP32 operation order as the codec: d * (0.5 + s) * 0.5.
    const float scale_f = f16(b) * (0.5f + float(meta >> 28)) * 0.5f;
    const uint16_t scale = to_f16(scale_f, error);
    uint64_t signs = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t index = (meta >> (7 * s)) & 127;
      index |= uint32_t(__popc(index) & 1) << 7;  // ksigns_iq2xs
      signs |= uint64_t(index) << (8 * s);
    }
    return uint64_t(scale) | (signs << 16);
  }
};

template <>
struct Group<22> {
  // Group of 16 values: two octets, each one grid index and one sign byte.
  __device__ static uint32_t decode(const uint8_t* row, int k0,
                                    uint16_t (&packets)[2], int* error) {
    const uint8_t* b = row + (k0 / 256) * 82;
    const int sub = (k0 % 256) / 16;  // 0..15
    uint32_t high = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int t = sub * 2 + i;  // octet within the block, 0..31
      packets[i] = uint16_t(b[2 + t] | (b[34 + t] << 8));
      high |= uint32_t((b[66 + t / 4] >> (2 * (t % 4))) & 3) << (2 * i);
    }
    const int nib = (b[74 + sub / 2] >> (4 * (sub % 2))) & 15;
    const float scale_f = f16(b) * (0.5f + float(nib)) * 0.25f;
    return uint32_t(to_f16(scale_f, error)) | (high << 16);
  }
};

// LUT4: 32 codes of one group, low nibbles of bytes 0..15 then high nibbles.
__device__ __forceinline__ void lut4_packets(const uint8_t* q,
                                             uint32_t (&packets)[4]) {
#pragma unroll
  for (int o = 0; o < 4; ++o) {
    uint32_t packet = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int v = o * 8 + j;
      const uint32_t code = v < 16 ? q[v] & 15 : q[v - 16] >> 4;
      packet |= code << (4 * (j / 2 + (j % 2) * 4));
    }
    packets[o] = packet;
  }
}

// FP16 LUT4 scale with the codec's checks: non-finite, overflow, underflow.
__device__ __forceinline__ uint16_t lut4_scale(float value, int* error) {
  const __half h = __float2half_rn(value);
  if (!isfinite(value) || !isfinite(__half2float(h)) ||
      (value != 0.f && __half2float(h) == 0.f))
    atomicOr(error, 1);
  return __half_as_ushort(h);
}

template <>
struct Group<20> {  // IQ4_NL
  __device__ static uint16_t decode(const uint8_t* row, int k0,
                                    uint32_t (&packets)[4], int* error) {
    const uint8_t* b = row + (k0 / 32) * 18;
    lut4_packets(b + 2, packets);
    return lut4_scale(f16(b), error);
  }
};

template <>
struct Group<23> {  // IQ4_XS
  __device__ static uint16_t decode(const uint8_t* row, int k0,
                                    uint32_t (&packets)[4], int* error) {
    const uint8_t* b = row + (k0 / 256) * 136;
    const int sub = (k0 % 256) / 32;
    lut4_packets(b + 8 + sub * 16, packets);
    const int high = (int(b[2]) | (int(b[3]) << 8)) >> (2 * sub) & 3;
    const int low = (b[4 + sub / 2] >> (4 * (sub % 2))) & 15;
    return lut4_scale(f16(b) * float((low | (high << 4)) - 32), error);
  }
};

template <>
struct Group<42> {  // Q2_0: 64-value blocks, value = code * d - d
  __device__ static uint32_t decode(const uint8_t* row, int k0,
                                    uint16_t (&packets)[4], int* error) {
    const uint8_t* b = row + (k0 / 64) * 18;
    const int v0 = k0 % 64;
#pragma unroll
    for (int o = 0; o < 4; ++o) {
      uint32_t packet = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int v = v0 + o * 8 + j;
        const uint32_t code = (b[2 + v / 4] >> (2 * (v % 4))) & 3;
        packet |= code << (2 * (j / 2 + (j % 2) * 4));
      }
      packets[o] = uint16_t(packet);
    }
    const uint16_t scale = uint16_t(b[0]) | (uint16_t(b[1]) << 8);
    (void)error;
    return uint32_t(scale) | (uint32_t(scale ^ 0x8000u) << 16);
  }
};

// Persistent: the CTAs walk (expert, pack) items for the first `*count`
// entries of `experts`; the count stays on the device so a launch can cover
// a residency-dependent subset (e.g. only cold experts, with few CTAs that
// stream pinned host memory over PCIe without taking SMs from the GEMM).
// src0/src1: per expert id, address of the expert's first source row (device
// or host-mapped); rows are contiguous with `src_stride` bytes. src1 selects
// gate/up interleaving. Output columns cover source columns
// [k_offset, k_offset + k). dst_w/dst_s: stacked canonical banks.
template <int Type>
__global__ void __launch_bounds__(kThreads)
    transcode_kernel(uint8_t* __restrict__ dst_w, uint8_t* __restrict__ dst_s,
                     int64_t w_stride, int64_t s_stride,
                     const int64_t* __restrict__ src0,
                     const int64_t* __restrict__ src1,
                     const int32_t* __restrict__ experts,
                     const int32_t* __restrict__ count, int packs, int n_out,
                     int k, int k_offset, int src_stride, int* error) {
  using F = Format<Type>;
  using Packet = PacketOf<Type>;
  constexpr int octets = F::kGroup / 8;
  extern __shared__ __align__(16) uint8_t smem[];
  const bool interleave = src1 != nullptr;
  const int sources = interleave ? 2 : 1;
  const int rows = kPack / sources;
  const int groups = k / F::kGroup;
  const int items = __ldg(count) * packs;
  for (int work = blockIdx.x; work < items; work += gridDim.x) {
    const int expert = __ldg(experts + work / packs);
    const int pack = work % packs;
    __syncthreads();  // the previous item is done with the staged rows
    // Stage the pack's source rows (contiguous) with 8-byte loads.
    for (int s = 0; s < sources; ++s) {
      const auto* base = reinterpret_cast<const uint64_t*>(
          reinterpret_cast<const uint8_t*>((s ? src1 : src0)[expert]) +
          int64_t(pack) * rows * src_stride);
      auto* stage = reinterpret_cast<uint64_t*>(smem + s * rows * src_stride);
      const int words = rows * src_stride / 8;
      for (int i = threadIdx.x; i < words; i += kThreads) stage[i] = base[i];
    }
    __syncthreads();
    auto* weight = reinterpret_cast<Packet*>(dst_w + expert * w_stride);
    auto* stats =
        reinterpret_cast<typename F::Stat*>(dst_s + expert * s_stride);
    for (int item = threadIdx.x; item < groups * kPack; item += kThreads) {
      const int lane = item % kPack, g = item / kPack;
      const int source = interleave ? lane % 2 : 0;
      const int r = interleave ? lane / 2 : lane;
      const uint8_t* row = smem + (source * rows + r) * src_stride;
      Packet packets[octets];
      const auto stat =
          Group<Type>::decode(row, k_offset + g * F::kGroup, packets, error);
#pragma unroll
      for (int q = 0; q < octets; ++q)
        weight[(int64_t(pack) * (k / 8) + g * octets + q) * kPack + lane] =
            packets[q];
      stats[int64_t(g) * n_out + pack * kPack + lane] = stat;
    }
  }
}

}  // namespace vllm::sm70_gguf::transcode
