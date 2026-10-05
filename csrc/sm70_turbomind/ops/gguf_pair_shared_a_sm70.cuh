// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <new>
#include "gguf_native_pair_readers_sm70.cuh"

namespace vllm::sm70_gguf {
__device__ __forceinline__ void native_pair_mma(float (&accum)[8], uint32_t a0,
                                                uint32_t a1, uint32_t b0,
                                                uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]),
        "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

template <int Segment, class Reader>
__device__ __forceinline__ void native_pair_segments(
    float (&accum)[8], const typename Reader::Record& record,
    const half* activation, const uint8_t* book) {
  const auto first = Reader::template fragment<Segment, 0>(record, book);
  const auto second = Reader::template fragment<Segment, 1>(record, book);
  const uint4 a0 = *reinterpret_cast<const uint4*>(activation + Segment * 16);
  const uint4 a1 =
      *reinterpret_cast<const uint4*>(activation + Segment * 16 + 8);
  const auto* b = reinterpret_cast<const uint32_t*>(&first);
  const auto* c = reinterpret_cast<const uint32_t*>(&second);
  native_pair_mma(accum, a0.x, a0.y, b[0], b[1]);
  native_pair_mma(accum, a0.z, a0.w, b[2], b[3]);
  native_pair_mma(accum, a1.x, a1.y, c[0], c[1]);
  native_pair_mma(accum, a1.z, a1.w, c[2], c[3]);
  if constexpr (Segment < 7)
    native_pair_segments<Segment + 1, Reader>(accum, record, activation, book);
}

// The tested N32/M8 staging/reduction schedule, independent of weight format.
// Each projection has its own reader state. Only the active projection's
// state is constructed, so unrelated record/metadata pointers stay disjoint.
template <class GateReader, class UpReader>
__global__ __launch_bounds__(512, 2) void native_pair_shared_a_kernel(
    half* __restrict__ output, const half* __restrict__ input,
    const uint8_t* __restrict__ gate, const uint8_t* __restrict__ up,
    int hidden, int k) {
  constexpr int SplitK = 8;
  constexpr bool SameBook =
      GateReader::kBookBytes > 0 && GateReader::kBookId == UpReader::kBookId;
  constexpr int BookBytes =
      GateReader::kBookBytes + (SameBook ? 0 : UpReader::kBookBytes);
  union alignas(16) Storage {
    uint8_t books[BookBytes > 0 ? BookBytes : 1];
    float partials[2][SplitK][256];
  };
  __shared__ Storage storage;
  __shared__ half staged_a[SplitK][8][136];
  uint8_t* gate_book = storage.books;
  uint8_t* up_book = storage.books + (SameBook ? 0 : GateReader::kBookBytes);
  GateReader::initialize(gate_book);
  if constexpr (!SameBook) UpReader::initialize(up_book);

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int projection = warp_id / SplitK;
  const int warp = warp_id % SplitK;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quadpair * 8 + row;
  const int groups_per_warp = (k / 16) / SplitK;
  const int first_part = warp * groups_per_warp / 8;
  union ReaderState {
    GateReader gate;
    UpReader up;
    __device__ ReaderState() {}
  } readers;
  if (projection == 0)
    new (&readers.gate) GateReader(gate, blockIdx.x, k / 256, first_part, col);
  else
    new (&readers.up) UpReader(up, blockIdx.x, k / 256, first_part, col);
  float accum[8] = {};
  for (int part = 0; part < groups_per_warp / 8; ++part) {
    for (int vector = threadIdx.x; vector < SplitK * 8 * 16;
         vector += blockDim.x) {
      const int k_warp = vector / 128;
      const int input_row = (vector / 16) & 7;
      const int k_vector = vector & 15;
      const int input_k =
          k_warp * groups_per_warp * 16 + part * 128 + k_vector * 8;
      const uint4 value = *reinterpret_cast<const uint4*>(
          input + int64_t{input_row} * k + input_k);
      *reinterpret_cast<uint4*>(&staged_a[k_warp][input_row][k_vector * 8]) =
          value;
    }
    __syncthreads();
    const half* activation = &staged_a[warp][row][0];
    if (projection == 0) {
      const auto record = readers.gate.load();
      native_pair_segments<0, GateReader>(accum, record, activation, gate_book);
    } else {
      const auto record = readers.up.load();
      native_pair_segments<0, UpReader>(accum, record, activation, up_book);
    }
    __syncthreads();
  }
  // Both codebooks are dead. All warps finish before the union is reused.
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int output_row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    storage.partials[projection][warp]
                    [output_row * 32 + quadpair * 8 + output_col] = accum[i];
  }
  __syncthreads();
  if (threadIdx.x < 256) {
    const int element = threadIdx.x;
    float gate_sum = 0.f, up_sum = 0.f;
#pragma unroll
    for (int part = 0; part < SplitK; ++part) {
      gate_sum += storage.partials[0][part][element];
      up_sum += storage.partials[1][part][element];
    }
    const float g = __half2float(__float2half(gate_sum));
    const half silu = __float2half(g / (1.f + expf(-g)));
    output[int64_t{element / 32} * hidden + blockIdx.x * 32 + element % 32] =
        __hmul(silu, __float2half(up_sum));
  }
}
}  // namespace vllm::sm70_gguf
