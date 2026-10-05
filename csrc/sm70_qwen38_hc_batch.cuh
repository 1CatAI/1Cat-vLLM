// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cooperative_groups.h>

// Selected arithmetic from the exact HC batch screen. Included after
// custom_all_reduce.cuh; runtime ownership stays with that communicator.
namespace vllm::qwen38_hc_batch {
__device__ __forceinline__ void mma(float (&d)[8], uint32_t a0, uint32_t a1,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// Match the Triton HC post-op, not CUDA's --use_fast_math division.
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0,%1,%2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float sigmoid(float x) {
  float e, d;
  asm("mul.f32 %0,%1,0fBFB8AA3B;" : "=f"(e) : "f"(x));
  asm("ex2.approx.f32 %0,%1;" : "=f"(e) : "f"(e));
  asm("add.f32 %0,%1,0f3F800000;" : "=f"(d) : "f"(e));
  return div_full(1.0f, d);
}

// Each quad pair computes one branch's 8x8 output. All four branch values
// for a hidden column live at identical lane offsets in the four quad pairs.
// PairRows shares the same 16-byte weight load across two independent M8
// accumulators. The K sequence in each accumulator is unchanged.
template <bool PairRows, int Warps, int Unroll, bool FuseMix,
          bool FullOutput = false>
__device__ __forceinline__ void hc_up_batch_body(
    const half* __restrict__ lora, const half* __restrict__ packed,
    const half* __restrict__ branches, half* __restrict__ out, int rows,
    int hidden, int hidden_offset, int block_x, int block_y) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int tile = block_x * Warps + warp;
  if (tile >= hidden / 8) return;
  const int group = PairRows ? 0 : block_y;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int branch = (lane >> 2) & 3;
  const int col = branch * 8 + r;
  const half* w = packed + static_cast<size_t>(tile) * 320 * 32;
  float accum[PairRows ? 2 : 1][8] = {};
#pragma unroll Unroll
  for (int g = 0; g < 20; ++g) {
    const uint4 lo = *reinterpret_cast<const uint4*>(w + (g * 64 + col) * 8);
    const uint4 hi =
        *reinterpret_cast<const uint4*>(w + (g * 64 + 32 + col) * 8);
#pragma unroll
    for (int p = 0; p < (PairRows ? 2 : 1); ++p) {
      const int row = (group + p) * 8 + r;
      uint4 a = make_uint4(0, 0, 0, 0), b = a;
      if (row < rows) {
        const half* x = lora + row * 320 + g * 16;
        a = *reinterpret_cast<const uint4*>(x);
        b = *reinterpret_cast<const uint4*>(x + 8);
      }
      mma(accum[p], a.x, a.y, lo.x, lo.y);
      mma(accum[p], a.z, a.w, lo.z, lo.w);
      mma(accum[p], b.x, b.y, hi.x, hi.y);
      mma(accum[p], b.z, b.w, hi.z, hi.w);
    }
  }
#pragma unroll
  for (int p = 0; p < (PairRows ? 2 : 1); ++p) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row =
          (group + p) * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      const int h =
          tile * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const half gate = __float2half_rn(accum[p][i]);
      if constexpr (FuseMix) {
        const float gate_f = __half2float(gate);
        float v = 0.0f;
        if (row < rows) {
          v = __half2float(
              branches[row * 10240 + branch * 2560 + hidden_offset + h]);
        }
        const float s = sigmoid(gate_f);
        float mixed = 0.0f;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int src = (lane & ~12) | (b << 2);
          const float gs = __shfl_sync(0xffffffff, s, src);
          const float x = __shfl_sync(0xffffffff, v, src);
          mixed = fmaf(gs, x, mixed);
        }
        if (branch == 0 && row < rows) {
          const int offset =
              FullOutput ? row * 2560 + hidden_offset + h : row * hidden + h;
          out[offset] = __float2half_rn(div_full(mixed, 4.0f));
        }
      } else if (row < rows) {
        out[row * 4 * hidden + branch * hidden + h] = gate;
      }
    }
  }
}

template <bool PairRows, int Warps, int Unroll, bool FuseMix,
          bool FullOutput = false>
__global__ __launch_bounds__(32 * Warps, 4) void hc_up_batch(
    const half* lora, const half* packed, const half* branches, half* out,
    int rows, int hidden, int hidden_offset) {
  hc_up_batch_body<PairRows, Warps, Unroll, FuseMix, FullOutput>(
      lora, packed, branches, out, rows, hidden, hidden_offset, blockIdx.x,
      blockIdx.y);
}

template <bool PairRows, int Warps, bool WarpM16, int N = 352,
          bool RoundPartials = false, int Unroll = 4>
__device__ __forceinline__ void hc_down_partials_body(
    const half* __restrict__ x, const half* __restrict__ packed,
    float* __restrict__ partials, int rows, int block_x, int block_y,
    int block_z) {
  constexpr int TileN = WarpM16 ? 16 : 32;
  const int lane = threadIdx.x % 32;
  const int tile = block_x * Warps + threadIdx.x / 32;
  if (tile >= N / TileN) return;
  const int group = (PairRows || WarpM16) ? 0 : block_y;
  const int split = block_z;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int output_quad = WarpM16 ? quad % 2 : quad;
  const int col = output_quad * 8 + r;
  const int row_base = WarpM16 ? (quad / 2) * 8 : group * 8;
  const half* w = packed + static_cast<size_t>(tile) * 10240 * TileN;
  float accum[PairRows ? 2 : 1][8] = {};
#pragma unroll Unroll
  for (int g = 0; g < 32; ++g) {
    const int kg = split * 32 + g;
    const uint4 lo =
        *reinterpret_cast<const uint4*>(w + (kg * 2 * TileN + col) * 8);
    const uint4 hi =
        *reinterpret_cast<const uint4*>(w + (kg * 2 * TileN + TileN + col) * 8);
#pragma unroll
    for (int p = 0; p < (PairRows ? 2 : 1); ++p) {
      const int row = row_base + p * 8 + r;
      uint4 a = make_uint4(0, 0, 0, 0), b = a;
      if (row < rows) {
        const half* input = x + row * 10240 + kg * 16;
        a = *reinterpret_cast<const uint4*>(input);
        b = *reinterpret_cast<const uint4*>(input + 8);
      }
      mma(accum[p], a.x, a.y, lo.x, lo.y);
      mma(accum[p], a.z, a.w, lo.z, lo.w);
      mma(accum[p], b.x, b.y, hi.x, hi.y);
      mma(accum[p], b.z, b.w, hi.z, hi.w);
    }
  }
#pragma unroll
  for (int p = 0; p < (PairRows ? 2 : 1); ++p) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row =
          row_base + p * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      const int n = tile * TileN + output_quad * 8 +
                    ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      if (row < rows) {
        // The established MTP cuBLAS path materializes each K512 partial
        // in FP16 before the ordered FP32 split reduction. Keep that boundary
        // independently of the FP32-partial concurrent-decode contract.
        const float value = RoundPartials
                                ? __half2float(__float2half_rn(accum[p][i]))
                                : accum[p][i];
        partials[(split * rows + row) * N + n] = value;
      }
    }
  }
}

template <bool PairRows, int Warps, bool WarpM16, int N = 352,
          bool RoundPartials = false>
__global__ __launch_bounds__(32 * Warps, 4) void hc_down_partials(
    const half* x, const half* packed, float* partials, int rows) {
  hc_down_partials_body<PairRows, Warps, WarpM16, N, RoundPartials>(
      x, packed, partials, rows, blockIdx.x, blockIdx.y, blockIdx.z);
}

// Disjoint dense packets can use any graph-safe TP collective. Keep the
// arithmetic identical to gather_body<true>, including its Half boundaries.
__global__ void down_local_packet(const float* partials, half* output, int rank,
                                  int rows) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * 336) return;
  const int row = index / 336, column = index % 336;
  const int local = column - rank * 80;
  const bool lora = local >= 0 && local < 80;
  const bool injection = rank == 3 && column >= 320 && column < 324;
  half value = __float2half_rn(0.0f);
  if (lora || injection) {
    float acc = 0.0f;
#pragma unroll
    for (int split = 0; split < 20; ++split)
      acc = __fadd_rn(acc, partials[(split * rows + row) * 96 + local]);
    value = __float2half_rn(acc);
    if (lora) {
      const float x = div_full(__half2float(value), 4.0f);
      value = __float2half_rn(__fmul_rn(x, sigmoid(x)));
    }
  }
  output[index] = value;
}

__global__ void down_replicated_reduce(const float* partials, half* lora,
                                       half* injection, int rows) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * 324) return;
  const int row = index / 324, col = index % 324;
  float acc = 0.0f;
#pragma unroll
  for (int split = 0; split < 20; ++split)
    acc = __fadd_rn(acc, partials[(split * rows + row) * 352 + col]);
  const half value = __float2half_rn(acc);
  if (col < 320) {
    const float x = div_full(__half2float(value), 4.0f);
    lora[row * 320 + col] = __float2half_rn(__fmul_rn(x, sigmoid(x)));
  } else {
    injection[row * 4 + col - 320] = value;
  }
}

template <bool Down>
__device__ __forceinline__ void gather_body(RankData buffers, const void* input,
                                            half* output, half* injection,
                                            int rank, int rows, int block) {
  constexpr int cols = Down ? 88 : 640;
  constexpr int packs_per_row = cols / 8;
  constexpr int stride = 16 * cols;
  constexpr size_t channel =
      Down ? kSm70Qwen38HcBatchDownOffset : kSm70Qwen38HcBatchOutputOffset;
  auto* local =
      const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[rank])) +
      channel;
  auto* counters = reinterpret_cast<uint32_t*>(local);
  const uint32_t epoch = counters[block];
  const uint32_t tag = epoch + 1u;  // Zero is always an empty packet.
  const int slot = epoch * 4 * stride;
  const int offset = block * blockDim.x + threadIdx.x;
  if (offset < rows * packs_per_row) {
    const int row = offset / packs_per_row;
    const int col = (offset % packs_per_row) * 8;
    // Each half travels with a tag in the SAME 32-bit word. Unlike sentinel
    // escaping, transport preserves all half bits, including NaN payloads.
    uint32_t words[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      half value;
      if constexpr (Down) {
        float acc = 0.0f;
#pragma unroll
        for (int split = 0; split < 20; ++split)
          acc = __fadd_rn(acc, static_cast<const float*>(
                                   input)[(split * rows + row) * 96 + col + i]);
        value = __float2half_rn(acc);
        if (col < 80) {
          const float x = div_full(__half2float(value), 4.0f);
          value = __float2half_rn(__fmul_rn(x, sigmoid(x)));
        }
      } else {
        value = static_cast<const half*>(input)[offset * 8 + i];
      }
      words[i] = (tag << 16) | __half_as_ushort(value);
    }
    const uint4 lo = make_uint4(words[0], words[1], words[2], words[3]);
    const uint4 hi = make_uint4(words[4], words[5], words[6], words[7]);
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      auto* dest =
          const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[peer])) +
          channel + kSm70Qwen38HcBatchCounterBytes;
      dest += (slot + rank * stride) * sizeof(uint32_t);
      sm70_push_store_volatile_16b(lo, dest, offset * 2);
      sm70_push_store_volatile_16b(hi, dest, offset * 2 + 1);
    }
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      const void* source = local + kSm70Qwen38HcBatchCounterBytes +
                           (slot + peer * stride) * sizeof(uint32_t);
      uint4 a, b;
      do {
        sm70_push_load_volatile_16b(a, source, offset * 2);
        sm70_push_load_volatile_16b(b, source, offset * 2 + 1);
      } while ((a.x >> 16) != tag || (a.y >> 16) != tag || (a.z >> 16) != tag ||
               (a.w >> 16) != tag || (b.x >> 16) != tag || (b.y >> 16) != tag ||
               (b.z >> 16) != tag || (b.w >> 16) != tag);
      const uint32_t received[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const half value = __ushort_as_half(received[i] & 0xffffu);
        if constexpr (Down) {
          if (col < 80)
            output[row * 320 + peer * 80 + col + i] = value;
          else if (peer == 3 && i < 4)
            injection[row * 4 + i] = value;
        } else {
          output[row * 2560 + peer * 640 + col + i] = value;
        }
      }
      // Clear every consumed packet before reusing its epoch. A shrinking
      // batch leaves some lanes inactive; those lanes must not retain a tag
      // that could be mistaken for a new packet when the batch grows again.
      const uint4 empty = make_uint4(0, 0, 0, 0);
      sm70_push_store_volatile_16b(empty, const_cast<void*>(source),
                                   offset * 2);
      sm70_push_store_volatile_16b(empty, const_cast<void*>(source),
                                   offset * 2 + 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) counters[block] = (epoch + 1u) & 1u;
}

template <bool Down>
__global__ void gather(RankData buffers, const void* input, half* output,
                       half* injection, int rank, int rows) {
  gather_body<Down>(buffers, input, output, injection, rank, rows, blockIdx.x);
}

// The same arithmetic and half+tag transport as the four-launch path.
// All CTAs reside together; grid barriers close only local data dependencies.
template <bool FullUnroll>
__global__ __launch_bounds__(32, 4) void hc_cooperative(
    RankData buffers, int rank, const half* input, const half* packed_down,
    const half* packed_up, float* partials, half* lora, half* local_output,
    half* output, half* injection, int rows) {
  const int block = blockIdx.x, groups = (rows + 7) / 8;
  auto grid = cooperative_groups::this_grid();
  if (block < 60 * groups)
    hc_down_partials_body<false, 1, false, 96, true, FullUnroll ? 32 : 4>(
        input, packed_down, partials, rows, block % 3, (block / 3) % groups,
        block / (3 * groups));
  grid.sync();
  if (block < (rows * 11 + 31) / 32)
    gather_body<true>(buffers, partials, lora, injection, rank, rows, block);
  grid.sync();
  hc_up_batch_body<false, 1, FullUnroll ? 20 : 4, true>(
      lora, packed_up, input, local_output, rows, 640, rank * 640, block % 80,
      block / 80);
  grid.sync();
  if (block < (rows * 80 + 31) / 32)
    gather_body<false>(buffers, local_output, output, injection, rank, rows,
                       block);
}

__device__ __forceinline__ void hc_publish_packets(RankData buffers, int rank,
                                                   int group, int tile,
                                                   const uint32_t* packets,
                                                   half* gathered, int rows) {
  const int lane = threadIdx.x % 32;
  constexpr int stride = 16 * 640;
  constexpr int header = kSm70Qwen38HcBatchFusedBlocks * sizeof(uint32_t);
  constexpr size_t channel = kSm70Qwen38HcBatchFusedOffset;
  const int block = group * 80 + tile;
  auto* local =
      const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[rank])) +
      channel;
  auto* counters = reinterpret_cast<uint32_t*>(local);
  const uint32_t epoch = counters[block], tag = epoch + 1u;
  const int slot = epoch * 4 * stride;
  // Retile the MMA fragments into contiguous, lossless half+tag packets.
  // Only valid rows publish/read/clear. Every active block owns its own
  // epoch across odd-count shrinking/growing graph replays.
  __syncthreads();
  const int row = group * 8 + lane / 2;
  if (threadIdx.x < 16 && row < rows) {
    const int pack = block * 16 + lane;
    const auto payload = reinterpret_cast<const uint4*>(packets)[lane];
    const uint32_t tags = tag << 16;
    const uint4 words = make_uint4(payload.x | tags, payload.y | tags,
                                   payload.z | tags, payload.w | tags);
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      auto* dest =
          const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[peer])) +
          channel + header + (slot + rank * stride) * sizeof(uint32_t);
      sm70_push_store_volatile_16b(words, dest, pack);
    }
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      void* source = local + header + (slot + peer * stride) * sizeof(uint32_t);
      uint4 received;
      do {
        sm70_push_load_volatile_16b(received, source, pack);
      } while ((received.x >> 16) != tag || (received.y >> 16) != tag ||
               (received.z >> 16) != tag || (received.w >> 16) != tag);
      half* dest =
          gathered + row * 2560 + peer * 640 + tile * 8 + (lane % 2) * 4;
      dest[0] = __ushort_as_half(received.x & 0xffffu);
      dest[1] = __ushort_as_half(received.y & 0xffffu);
      dest[2] = __ushort_as_half(received.z & 0xffffu);
      dest[3] = __ushort_as_half(received.w & 0xffffu);
      sm70_push_store_volatile_16b(make_uint4(0, 0, 0, 0), source, pack);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) counters[block] = (epoch + 1u) & 1u;
}

template <bool PairRows, int Warps, int Unroll, bool FuseMix,
          bool FuseGather = false>
__global__ __launch_bounds__(32 * Warps, 4) void hc_up_batch_fused_gather(
    const half* __restrict__ lora, const half* __restrict__ packed,
    const half* __restrict__ branches, half* __restrict__ out, int rows,
    int hidden, int hidden_offset, RankData buffers, int rank, half* gathered) {
  static_assert(!FuseGather || (!PairRows && Warps == 1 && FuseMix));
  __shared__ __align__(16) uint32_t packets[64];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int tile = blockIdx.x * Warps + warp;
  if (tile >= hidden / 8) return;
  const int group = PairRows ? 0 : blockIdx.y;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int branch = (lane >> 2) & 3;
  const int col = branch * 8 + r;
  const half* w = packed + static_cast<size_t>(tile) * 320 * 32;
  float accum[PairRows ? 2 : 1][8] = {};
#pragma unroll Unroll
  for (int g = 0; g < 20; ++g) {
    const uint4 lo = *reinterpret_cast<const uint4*>(w + (g * 64 + col) * 8);
    const uint4 hi =
        *reinterpret_cast<const uint4*>(w + (g * 64 + 32 + col) * 8);
#pragma unroll
    for (int p = 0; p < (PairRows ? 2 : 1); ++p) {
      const int row = (group + p) * 8 + r;
      uint4 a = make_uint4(0, 0, 0, 0), b = a;
      if (row < rows) {
        const half* x = lora + row * 320 + g * 16;
        a = *reinterpret_cast<const uint4*>(x);
        b = *reinterpret_cast<const uint4*>(x + 8);
      }
      mma(accum[p], a.x, a.y, lo.x, lo.y);
      mma(accum[p], a.z, a.w, lo.z, lo.w);
      mma(accum[p], b.x, b.y, hi.x, hi.y);
      mma(accum[p], b.z, b.w, hi.z, hi.w);
    }
  }
#pragma unroll
  for (int p = 0; p < (PairRows ? 2 : 1); ++p) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row =
          (group + p) * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      const int h =
          tile * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const half gate = __float2half_rn(accum[p][i]);
      if constexpr (FuseMix) {
        const float gate_f = __half2float(gate);
        float v = 0.0f;
        if (row < rows) {
          v = __half2float(
              branches[row * 10240 + branch * 2560 + hidden_offset + h]);
        }
        const float s = sigmoid(gate_f);
        float mixed = 0.0f;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int src = (lane & ~12) | (b << 2);
          const float gs = __shfl_sync(0xffffffff, s, src);
          const float x = __shfl_sync(0xffffffff, v, src);
          mixed = fmaf(gs, x, mixed);
        }
        if (branch == 0 && row < rows) {
          const half value = __float2half_rn(div_full(mixed, 4.0f));
          out[row * hidden + h] = value;
          if constexpr (FuseGather)
            packets[(row - group * 8) * 8 + (h - tile * 8)] =
                __half_as_ushort(value);
        }
      } else if (row < rows) {
        out[row * 4 * hidden + branch * hidden + h] = gate;
      }
    }
  }
  if constexpr (FuseGather) {
    hc_publish_packets(buffers, rank, group, tile, packets, gathered, rows);
  }
}

// Assign adjacent columns to adjacent lanes for the twenty FP32 partial
// reads. The old pack-per-thread assignment reads eight interleaved columns
// serially, and leaves few active lanes at M2. Arithmetic/order is unchanged;
// shared memory only retile words for the existing half+tag transport.
__global__ void down_gather_coalesced(RankData buffers, const float* partials,
                                      half* lora, half* injection, int rank,
                                      int rows) {
  constexpr int cols = 88, stride = 16 * cols;
  constexpr size_t channel = kSm70Qwen38HcBatchDownOffset;
  __shared__ __align__(16) uint32_t packets[128];
  auto* local =
      const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[rank])) +
      channel;
  auto* counters = reinterpret_cast<uint32_t*>(local);
  const uint32_t epoch = counters[blockIdx.x], tag = epoch + 1u;
  const int slot = epoch * 4 * stride;
  const int index = blockIdx.x * 128 + threadIdx.x;
  const int row = index / cols, col = index % cols;
  if (index < rows * cols) {
    float acc = 0.0f;
#pragma unroll
    for (int split = 0; split < 20; ++split)
      acc = __fadd_rn(acc, partials[(split * rows + row) * 96 + col]);
    half value = __float2half_rn(acc);
    if (col < 80) {
      const float x = div_full(__half2float(value), 4.0f);
      value = __float2half_rn(__fmul_rn(x, sigmoid(x)));
    }
    packets[threadIdx.x] = (tag << 16) | __half_as_ushort(value);
  }
  __syncthreads();
  // 88 columns and the 128-thread block are both multiples of four, so
  // no transport vector spans a row or a partially valid final packet.
  if (threadIdx.x % 4 == 0 && index < rows * cols) {
    const int pack = index / 4;
    const uint4 words =
        reinterpret_cast<const uint4*>(packets)[threadIdx.x / 4];
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      auto* dest =
          const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[peer])) +
          channel + kSm70Qwen38HcBatchCounterBytes +
          (slot + rank * stride) * sizeof(uint32_t);
      sm70_push_store_volatile_16b(words, dest, pack);
    }
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      void* source = local + kSm70Qwen38HcBatchCounterBytes +
                     (slot + peer * stride) * sizeof(uint32_t);
      uint4 received;
      do {
        sm70_push_load_volatile_16b(received, source, pack);
      } while ((received.x >> 16) != tag || (received.y >> 16) != tag ||
               (received.z >> 16) != tag || (received.w >> 16) != tag);
      const uint32_t values[4] = {received.x, received.y, received.z,
                                  received.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const half value = __ushort_as_half(values[i] & 0xffffu);
        if (col < 80)
          lora[row * 320 + peer * 80 + col + i] = value;
        else if (col == 80 && peer == 3)
          injection[row * 4 + i] = value;
      }
      sm70_push_store_volatile_16b(make_uint4(0, 0, 0, 0), source, pack);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) counters[blockIdx.x] = (epoch + 1u) & 1u;
}

// Split K among warps inside one CTA. Each weight element is still read once
// for the M8 row tile; FP32 partials are reduced through shared memory.
template <int Warps>
__global__ __launch_bounds__(32 * Warps) void hc_down_cta_split(
    const half* x, const half* packed, float* partials, int rows) {
  __shared__ float sums[Warps][8][32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int tile = blockIdx.x, group = blockIdx.y, split = blockIdx.z;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3, col = quad * 8 + r;
  const int row = group * 8 + r;
  const half* w = packed + static_cast<size_t>(tile) * 10240 * 32;
  float accum[8] = {};
  for (int g = warp * (32 / Warps); g < (warp + 1) * (32 / Warps); ++g) {
    const int kg = split * 32 + g;
    const uint4 lo = *reinterpret_cast<const uint4*>(w + (kg * 64 + col) * 8);
    const uint4 hi =
        *reinterpret_cast<const uint4*>(w + (kg * 64 + 32 + col) * 8);
    uint4 a = {}, b = {};
    if (row < rows) {
      const half* input = x + row * 10240 + kg * 16;
      a = *reinterpret_cast<const uint4*>(input);
      b = *reinterpret_cast<const uint4*>(input + 8);
    }
    mma(accum, a.x, a.y, lo.x, lo.y);
    mma(accum, a.z, a.w, lo.z, lo.w);
    mma(accum, b.x, b.y, hi.x, hi.y);
    mma(accum, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    sums[warp][rr][quad * 8 + cc] = accum[i];
  }
  __syncthreads();
  const int t = threadIdx.x;
  if (t < 256 && group * 8 + t / 32 < rows) {
    float value = 0.0f;
#pragma unroll
    for (int w = 0; w < Warps; ++w)
      value = __fadd_rn(value, sums[w][t / 32][t % 32]);
    partials[(split * rows + group * 8 + t / 32) * 96 + tile * 32 + t % 32] =
        value;
  }
}

template <int Warps>
__global__ __launch_bounds__(32 * Warps) void hc_up_cta_split_gather(
    const half* lora, const half* packed, const half* branches, half* out,
    int rows, RankData buffers, int rank, half* gathered) {
  __shared__ float sums[Warps][8][32];
  __shared__ half gates[8][32];
  __shared__ __align__(16) uint32_t packets[64];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int tile = blockIdx.x, group = blockIdx.y;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int branch = (lane >> 2) & 3, col = branch * 8 + r;
  const int row = group * 8 + r;
  const half* w = packed + static_cast<size_t>(tile) * 320 * 32;
  float accum[8] = {};
  for (int g = warp * 20 / Warps; g < (warp + 1) * 20 / Warps; ++g) {
    const uint4 lo = *reinterpret_cast<const uint4*>(w + (g * 64 + col) * 8);
    const uint4 hi =
        *reinterpret_cast<const uint4*>(w + (g * 64 + 32 + col) * 8);
    uint4 a = {}, b = {};
    if (row < rows) {
      const half* input = lora + row * 320 + g * 16;
      a = *reinterpret_cast<const uint4*>(input);
      b = *reinterpret_cast<const uint4*>(input + 8);
    }
    mma(accum, a.x, a.y, lo.x, lo.y);
    mma(accum, a.z, a.w, lo.z, lo.w);
    mma(accum, b.x, b.y, hi.x, hi.y);
    mma(accum, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    sums[warp][rr][branch * 8 + cc] = accum[i];
  }
  __syncthreads();
  const int t = threadIdx.x;
  if (t < 256) {
    float value = 0.0f;
#pragma unroll
    for (int w = 0; w < Warps; ++w)
      value = __fadd_rn(value, sums[w][t / 32][t % 32]);
    gates[t / 32][t % 32] = __float2half_rn(value);
  }
  __syncthreads();
  if (t < 64 && group * 8 + t / 8 < rows) {
    const int rr = t / 8, h = tile * 8 + t % 8;
    const int row = group * 8 + rr;
    float mixed = 0.0f;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float gate = __half2float(gates[rr][b * 8 + t % 8]);
      const float value =
          __half2float(branches[row * 10240 + b * 2560 + rank * 640 + h]);
      mixed = fmaf(sigmoid(gate), value, mixed);
    }
    const half value = __float2half_rn(div_full(mixed, 4.0f));
    out[row * 640 + h] = value;
    packets[t] = __half_as_ushort(value);
  }
  hc_publish_packets(buffers, rank, group, tile, packets, gathered, rows);
}

// Complete column-pair ownership fuses projection, SiLU and disjoint TP
// packets without a cross-CTA barrier. Quad K slices only load live columns.
template <int Warps>
__global__ __launch_bounds__(32 * Warps) void hc_down_cta_finish(
    const half* x, const half* packed, half* lora, half* injection, int rows,
    RankData buffers, int rank) {
  __shared__ float sums[Warps][8][2];
  __shared__ __align__(16) uint32_t packets[16];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int group = blockIdx.y, column = blockIdx.x * 2;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0), quad = (lane >> 2) & 3;
  const int row = group * 8 + r;
  const half* w = packed + static_cast<size_t>(column / 32) * 10240 * 32;
  float accum[8] = {};
  for (int g = 0; g < 640 / (Warps * 4); ++g) {
    const int kg = (warp * 4 + quad) * (640 / (Warps * 4)) + g;
    uint4 lo = {}, hi = {}, a = {}, b = {};
    if (r < 2 && (column < 80 || rank == 3)) {
      const int col = column % 32 + r;
      lo = *reinterpret_cast<const uint4*>(w + (kg * 64 + col) * 8);
      hi = *reinterpret_cast<const uint4*>(w + (kg * 64 + 32 + col) * 8);
    }
    if (row < rows) {
      const half* input = x + row * 10240 + kg * 16;
      a = *reinterpret_cast<const uint4*>(input);
      b = *reinterpret_cast<const uint4*>(input + 8);
    }
    mma(accum, a.x, a.y, lo.x, lo.y);
    mma(accum, a.z, a.w, lo.z, lo.w);
    mma(accum, b.x, b.y, hi.x, hi.y);
    mma(accum, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    float value = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q)
      value = __fadd_rn(
          value, __shfl_sync(0xffffffff, accum[i], (lane & ~12) | (q << 2)));
    if (quad == 0 && cc < 2) sums[warp][rr][cc] = value;
  }
  __syncthreads();
  constexpr int stride = 16 * 88;
  constexpr size_t channel = kSm70Qwen38HcBatchDownOffset;
  const int block = group * 44 + blockIdx.x;
  auto* local =
      const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[rank])) +
      channel;
  auto* counters = reinterpret_cast<uint32_t*>(local);
  const uint32_t epoch = counters[block], tag = epoch + 1u;
  const int slot = epoch * 4 * stride, t = threadIdx.x;
  if (t < 16) {
    float projected = 0.0f;
#pragma unroll
    for (int w = 0; w < Warps; ++w)
      projected = __fadd_rn(projected, sums[w][t / 2][t % 2]);
    half value = __float2half_rn(projected);
    if (column < 80) {
      const float scaled = div_full(__half2float(value), 4.0f);
      value = __float2half_rn(__fmul_rn(scaled, sigmoid(scaled)));
    }
    packets[t] = (tag << 16) | __half_as_ushort(value);
  }
  __syncthreads();
  if (t < 4) {
    const int pack = block * 4 + t;
    const uint4 words = reinterpret_cast<const uint4*>(packets)[t];
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      auto* dest =
          const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[peer])) +
          channel + kSm70Qwen38HcBatchCounterBytes +
          (slot + rank * stride) * sizeof(uint32_t);
      sm70_push_store_volatile_16b(words, dest, pack);
    }
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      void* source = local + kSm70Qwen38HcBatchCounterBytes +
                     (slot + peer * stride) * sizeof(uint32_t);
      uint4 received;
      do {
        sm70_push_load_volatile_16b(received, source, pack);
      } while ((received.x >> 16) != tag || (received.y >> 16) != tag ||
               (received.z >> 16) != tag || (received.w >> 16) != tag);
      const uint32_t values[4] = {received.x, received.y, received.z,
                                  received.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = group * 8 + (t * 4 + i) / 2, col = column + i % 2;
        if (row < rows) {
          const half value = __ushort_as_half(values[i] & 0xffffu);
          if (col < 80)
            lora[row * 320 + peer * 80 + col] = value;
          else if (peer == 3 && col < 84)
            injection[row * 4 + col - 80] = value;
        }
      }
      sm70_push_store_volatile_16b(make_uint4(0, 0, 0, 0), source, pack);
    }
  }
  __syncthreads();
  if (t == 0) counters[block] = (epoch + 1u) & 1u;
}

inline void launch(RankData buffers, int rank, const half* input,
                   const half* packed_down, const half* packed_up,
                   float* partials, half* lora, half* local_output,
                   half* output, half* injection, int rows, bool round_partials,
                   bool cooperative, bool full_unroll, cudaStream_t stream,
                   bool fused_chain, int cta_split_warps = 0) {
  TORCH_CHECK(!fused_chain || (!round_partials && !cooperative),
              "Fused concurrent HC requires FP32 partials");
  TORCH_CHECK(cta_split_warps == 0 ||
                  (fused_chain &&
                   (abs(cta_split_warps) == 8 || abs(cta_split_warps) == 16)),
              "HC CTA split requires 8/16 warps and the FP32 fused chain");
  if (cta_split_warps) {
    const int warps = abs(cta_split_warps);
    if (cta_split_warps == -8)
      hc_down_cta_finish<8><<<dim3(44, (rows + 7) / 8), 256, 0, stream>>>(
          input, packed_down, lora, injection, rows, buffers, rank);
    else if (cta_split_warps == -16)
      hc_down_cta_finish<16><<<dim3(44, (rows + 7) / 8), 512, 0, stream>>>(
          input, packed_down, lora, injection, rows, buffers, rank);
    else if (warps == 8)
      hc_down_cta_split<8><<<dim3(3, (rows + 7) / 8, 20), 256, 0, stream>>>(
          input, packed_down, partials, rows);
    else
      hc_down_cta_split<16><<<dim3(3, (rows + 7) / 8, 20), 512, 0, stream>>>(
          input, packed_down, partials, rows);
    if (cta_split_warps > 0)
      down_gather_coalesced<<<(rows * 88 + 127) / 128, 128, 0, stream>>>(
          buffers, partials, lora, injection, rank, rows);
    if (warps == 8)
      hc_up_cta_split_gather<8><<<dim3(80, (rows + 7) / 8), 256, 0, stream>>>(
          lora, packed_up, input, local_output, rows, buffers, rank, output);
    else
      hc_up_cta_split_gather<16><<<dim3(80, (rows + 7) / 8), 512, 0, stream>>>(
          lora, packed_up, input, local_output, rows, buffers, rank, output);
    return;
  }
  if (fused_chain) {
    hc_down_partials<false, 1, false, 96>
        <<<dim3(3, (rows + 7) / 8, 20), 32, 0, stream>>>(input, packed_down,
                                                         partials, rows);
    down_gather_coalesced<<<(rows * 88 + 127) / 128, 128, 0, stream>>>(
        buffers, partials, lora, injection, rank, rows);
    hc_up_batch_fused_gather<false, 1, 4, true, true>
        <<<dim3(80, (rows + 7) / 8), 32, 0, stream>>>(
            lora, packed_up, input, local_output, rows, 640, rank * 640,
            buffers, rank, output);
    return;
  }
  if (cooperative) {
    TORCH_CHECK(round_partials,
                "Cooperative HC requires the MTP FP16 contract");
    void* args[] = {&buffers,   &rank,      &input, &packed_down,
                    &packed_up, &partials,  &lora,  &local_output,
                    &output,    &injection, &rows};
    CUDACHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(full_unroll ? hc_cooperative<true>
                                            : hc_cooperative<false>),
        dim3(80 * ((rows + 7) / 8)), dim3(32), args, 0, stream));
    return;
  }
  if (round_partials) {
    hc_down_partials<false, 1, false, 96, true>
        <<<dim3(3, (rows + 7) / 8, 20), 32, 0, stream>>>(input, packed_down,
                                                         partials, rows);
  } else {
    hc_down_partials<false, 1, false, 96>
        <<<dim3(3, (rows + 7) / 8, 20), 32, 0, stream>>>(input, packed_down,
                                                         partials, rows);
  }
  gather<true><<<(rows * 11 + 127) / 128, 128, 0, stream>>>(
      buffers, partials, lora, injection, rank, rows);
  hc_up_batch<false, 1, 4, true><<<dim3(80, (rows + 7) / 8), 32, 0, stream>>>(
      lora, packed_up, input, local_output, rows, 640, rank * 640);
  gather<false><<<(rows * 80 + 127) / 128, 128, 0, stream>>>(
      buffers, local_output, output, injection, rank, rows);
}
}  // namespace vllm::qwen38_hc_batch
