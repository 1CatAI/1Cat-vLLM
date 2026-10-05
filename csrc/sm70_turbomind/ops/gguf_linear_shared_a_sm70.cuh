// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "gguf_pair_shared_a_sm70.cuh"
namespace vllm::sm70_gguf {
// Consume the already prepared canonical stream. No source superblock or
// nested scale unpacking is repeated in the main loop. Operand conversion is
// the same transform used by canonical GEMM/dequantization.
template <int Bits, int Group>
struct CanonicalAffineReader {
  static constexpr int kBookBytes = 0;
  static constexpr int kBookId = 0;
  using Packet = std::conditional_t<Bits == 2, uint16_t, uint32_t>;
  using Code =
      std::conditional_t<Bits == 2, turbomind::uint2_t, turbomind::uint4_t>;
  struct Record {
    Packet packets[16];
    uint32_t coefficients[128 / Group];
  };
  const Packet* packets;
  const uint32_t* coefficients;
  int stat_stride;
  __device__ CanonicalAffineReader(const uint8_t* source, const uint32_t* stats,
                                   int n, int k, int tile, int first, int col) {
    packets = reinterpret_cast<const Packet*>(source) +
              int64_t{tile} * (k / 8) * 32 + first * 16 * 32 + col;
    coefficients = stats + int64_t{first} * (128 / Group) * n + tile * 32 + col;
    stat_stride = n;
  }
  __device__ static void initialize(uint8_t*) {}
  __device__ Record load() {
    Record record;
#pragma unroll
    for (int i = 0; i < 16; ++i) record.packets[i] = packets[i * 32];
#pragma unroll
    for (int i = 0; i < 128 / Group; ++i)
      record.coefficients[i] = coefficients[i * stat_stride];
    packets += 16 * 32;
    coefficients += (128 / Group) * stat_stride;
    return record;
  }
  template <int Segment, int Fragment>
  __device__ static turbomind::Array<half, 8> fragment(const Record& record,
                                                       const uint8_t*) {
    constexpr int octet = Segment * 2 + Fragment;
    turbomind::Array<Code, 8> data[1][1];
    reinterpret_cast<Packet&>(data[0][0]) = record.packets[octet];
    turbomind::Array<uint32_t, 1> stats[1][1];
    stats[0][0][0] = record.coefficients[(octet * 8) / Group];
    turbomind::Array<half, 8> decoded[1][1];
    turbomind::gemm::Transform_HMMA_SIMT_B::apply(decoded, 0, data, stats, 1);
    return decoded[0][0];
  }
};

// Read the existing LUT4 converter stream without expanding source scales
// again. Packet lanes and half multiplication match canonical GEMM exactly.
struct CanonicalIq4Reader {
  static constexpr int kBookBytes = 0;
  static constexpr int kBookId = 0;
  struct Record {
    uint32_t packets[16];
    uint16_t coefficients[4];
  };
  const uint32_t* packets;
  const uint16_t* coefficients;
  int stat_stride;
  __device__ CanonicalIq4Reader(const uint8_t* source, const uint32_t* stats,
                                int n, int k, int tile, int first, int col) {
    packets = reinterpret_cast<const uint32_t*>(source) +
              int64_t{tile} * (k / 8) * 32 + first * 16 * 32 + col;
    coefficients = reinterpret_cast<const uint16_t*>(stats) +
                   int64_t{first} * 4 * n + tile * 32 + col;
    stat_stride = n;
  }
  __device__ static void initialize(uint8_t*) {}
  __device__ Record load() {
    Record record;
#pragma unroll
    for (int i = 0; i < 16; ++i) record.packets[i] = packets[i * 32];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      record.coefficients[i] = coefficients[i * stat_stride];
    packets += 16 * 32;
    coefficients += 4 * stat_stride;
    return record;
  }
  template <int Segment, int Fragment>
  __device__ static turbomind::Array<half, 8> fragment(const Record& record,
                                                       const uint8_t*) {
    constexpr int octet = Segment * 2 + Fragment;
    turbomind::Array<turbomind::uint4_t, 8> data[1][1];
    reinterpret_cast<uint32_t&>(data[0][0]) = record.packets[octet];
    turbomind::Array<uint16_t, 1> stats[1][1];
    stats[0][0][0] = record.coefficients[octet / 4];
    turbomind::Array<half, 8> decoded[1][1];
    turbomind::gemm::Transform_HMMA_SM70_Lut4<0>::apply(decoded, 0, data, stats,
                                                        1);
    return decoded[0][0];
  }
};

// Two N32 subtiles share the pair's activation staging and codebook. Two
// global K partitions expose 160 CTAs for N5120, with no separate reduction
// launch. Partial sums remain FP32 until the last CTA writes the output.
template <class Reader, bool Canonical = false, int GlobalSplitK = 2,
          bool HeadTiledInput = false>
__device__ __forceinline__ void native_linear_n64_body(
    half* __restrict__ output, const half* __restrict__ input,
    const uint8_t* __restrict__ weight, const uint32_t* __restrict__ stats,
    float* __restrict__ partials, int* __restrict__ counters, int n, int k,
    int tile, int scratch_tile, int output_stride, int output_offset,
    int stats_stride, uint8_t* shared) {
  constexpr int SplitK = 8;
  uint8_t* book = shared;
  auto* reductions = reinterpret_cast<float (*)[SplitK][256]>(shared);
  auto* staged_a = reinterpret_cast<half(*)[8][136]>(shared + 16384);
  auto* last_cta =
      reinterpret_cast<bool*>(shared + 16384 + sizeof(half) * 8 * 8 * 136);
  static_assert(Reader::kBookBytes <= 16384);
  Reader::initialize(book);
  const int lane = threadIdx.x & 31;
  const int subtile = threadIdx.x >> 8;
  const int warp = (threadIdx.x >> 5) & 7;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quadpair * 8 + row;
  const int parts = k / 128;
  const int global_first = blockIdx.y * parts / GlobalSplitK;
  const int local_parts = parts / GlobalSplitK;
  const int first = global_first + warp * local_parts / SplitK;
  const int last = global_first + (warp + 1) * local_parts / SplitK;
  Reader reader = [&]() {
    if constexpr (Canonical)
      return Reader(weight, stats, stats_stride, k, tile * 2 + subtile, first,
                    col);
    else
      return Reader(weight, tile * 2 + subtile, k / 256, first, col);
  }();
  float accum[8] = {};
  for (int part = 0; part < (local_parts + SplitK - 1) / SplitK; ++part) {
    for (int vector = threadIdx.x; vector < SplitK * 8 * 16;
         vector += blockDim.x) {
      const int k_warp = vector / 128;
      const int input_row = (vector / 16) & 7;
      const int k_vector = vector & 15;
      const int input_part =
          global_first + k_warp * local_parts / SplitK + part;
      const int end_part = global_first + (k_warp + 1) * local_parts / SplitK;
      if (input_part < end_part) {
        int logical_part = input_part;
        if constexpr (HeadTiledInput) {
          // The TP4 GDN shard contains four groups of three 128-wide heads.
          // Restore the adapter's head order while staging A, without a copy.
          logical_part = (input_part % 4) * 3 + input_part / 4;
        }
        const uint4 value = *reinterpret_cast<const uint4*>(
            input + int64_t{input_row} * k + logical_part * 128 + k_vector * 8);
        *reinterpret_cast<uint4*>(&staged_a[k_warp][input_row][k_vector * 8]) =
            value;
      }
    }
    __syncthreads();
    if (first + part < last) {
      const auto record = reader.load();
      native_pair_segments<0, Reader>(accum, record, &staged_a[warp][row][0],
                                      book);
    }
    __syncthreads();
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int output_row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    reductions[subtile][warp][output_row * 32 + quadpair * 8 + output_col] =
        accum[i];
  }
  __syncthreads();
  const int element = threadIdx.x & 255;
  float sum = 0.f;
#pragma unroll
  for (int part = 0; part < SplitK; ++part)
    sum += reductions[subtile][part][element];
  if constexpr (GlobalSplitK == 1) {
    const int column = tile * 64 + subtile * 32 + element % 32;
    if (column < n)
      output[int64_t{element / 32} * output_stride + output_offset + column] =
          __float2half_rn(sum);
    return;
  }
  const int partial_base = (scratch_tile * GlobalSplitK + blockIdx.y) * 512;
  volatile float* published = partials;
  published[partial_base + threadIdx.x] = sum;
  // All writers publish their own words before one thread publishes the
  // completion ticket. The last CTA reads each partition in fixed order.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    *last_cta = atomicAdd(counters + scratch_tile, 1) == GlobalSplitK - 1;
  __syncthreads();
  if (*last_cta) {
    sum = 0.f;
#pragma unroll
    for (int part = 0; part < GlobalSplitK; ++part)
      sum +=
          published[(scratch_tile * GlobalSplitK + part) * 512 + threadIdx.x];
    const int column = tile * 64 + subtile * 32 + element % 32;
    if (column < n)
      output[int64_t{element / 32} * output_stride + output_offset + column] =
          __float2half_rn(sum);
    __syncthreads();
    if (threadIdx.x == 0) atomicExch(counters + scratch_tile, 0);
  }
}

}  // namespace vllm::sm70_gguf
