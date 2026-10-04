// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include "gguf_lattice_compact.cuh"
#include "src/turbomind/kernels/gemm/arch/mma_sm70.h"

void gguf_workspace_f16_gemm_sm70_out(torch::Tensor out, torch::Tensor input,
                                      torch::Tensor offsets,
                                      torch::Tensor weight_ptrs);

namespace {
// One warp cooperatively stages one original block. Row padding is the only
// persistent storage overhead; unaligned block starts are stitched in shared.
template <int Type>
__device__ const uint8_t* stage_block(uint8_t* shared, const uint8_t* row,
                                      int block, int row_stride) {
  constexpr int bytes = vllm::sm70_gguf::LatticeRawDecoder<Type>::kBlockBytes;
  const int lane = threadIdx.x % 32;
  const int start = block * bytes, aligned = start & ~7, bias = start & 7;
  const int words = (bias + bytes + 7) / 8;
  if (lane < words && aligned + lane * 8 < row_stride)
    reinterpret_cast<uint64_t*>(shared)[lane] =
        *reinterpret_cast<const uint64_t*>(row + aligned + lane * 8);
  __syncwarp();
  return shared + bias;
}

template <int Type, class Output>
__global__ void raw_dequant_kernel(Output* out, const uint8_t* weight, int n,
                                   int k, int stride) {
  using Decode = vllm::sm70_gguf::LatticeRawDecoder<Type>;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  __shared__ __align__(16) uint8_t raw[4][120];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.x * 4 + warp, block = blockIdx.y;
  if (row >= n) return;
  const uint8_t* data = stage_block<Type>(
      raw[warp], weight + (int64_t)row * stride, block, stride);
  const auto values = Decode::template fragment<Output>(data, lane * 8, grid);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    out[(int64_t)row * k + block * 256 + lane * 8 + i] =
        static_cast<Output>(values[i]);
}

template <int Type>
__global__ void raw_dequant_transpose_kernel(half* out, const uint8_t* weight,
                                             int n, int k, int stride) {
  using Decode = vllm::sm70_gguf::LatticeRawDecoder<Type>;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  __shared__ __align__(16) uint8_t raw[4][120];
  // One bank step per row in the transposed gather, without changing storage.
  __shared__ half decoded[32][258];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int first = blockIdx.x * 32, block = blockIdx.y;
  for (int row = warp; row < 32; row += 4) {
    if (first + row < n) {
      const uint8_t* data = stage_block<Type>(
          raw[warp], weight + (int64_t)(first + row) * stride, block, stride);
      const auto values = Decode::template fragment<half>(data, lane * 8, grid);
#pragma unroll
      for (int j = 0; j < 8; ++j) decoded[row][lane * 8 + j] = values[j];
      __syncwarp();
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 32 * 256; i += blockDim.x) {
    const int local_k = i / 32, row = i % 32;
    if (first + row < n)
      out[(int64_t)(block * 256 + local_k) * n + first + row] =
          decoded[row][local_k];
  }
}

template <int Type, bool Split, bool Prefetch = false, bool FactorScale = false>
__global__ void raw_vec_kernel(half* out, float* partial, const half* x,
                               const uint8_t* weight, int n, int k, int stride,
                               int splits) {
  using Decode = vllm::sm70_gguf::LatticeRawDecoder<Type>;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  __shared__ __align__(16) uint8_t raw[4][120];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.x * 4 + warp;
  if (row >= n) return;
  float sum = 0.f;
  const int blocks = k / 256;
  const int begin = blocks * blockIdx.y / splits;
  const int end = blocks * (blockIdx.y + 1) / splits;
  const auto* row_data = weight + int64_t{row} * stride;
  if constexpr (Prefetch) stage_block<Type>(raw[warp], row_data, begin, stride);
  for (int block = begin; block < end; ++block) {
    constexpr int bytes = Decode::kBlockBytes;
    const int next_start = (block + 1) * bytes;
    const int aligned = next_start & ~7, bias = next_start & 7;
    const int words = (bias + bytes + 7) / 8;
    const bool load_next =
        block + 1 < end && lane < words && aligned + lane * 8 < stride;
    uint64_t next_word = 0;
    if constexpr (Prefetch) {
      if (load_next)
        next_word =
            *reinterpret_cast<const uint64_t*>(row_data + aligned + lane * 8);
    }
    const uint8_t* data;
    if constexpr (Prefetch)
      data = raw[warp] + (block * bytes & 7);
    else
      data = stage_block<Type>(raw[warp], row_data, block, stride);
    const auto values =
        Decode::template fragment<float, !FactorScale>(data, lane * 8, grid);
    // K is block aligned and each lane owns eight adjacent half values.
    // One 128-bit load replaces eight strided 16-bit memory instructions.
    const uint4 loaded =
        *reinterpret_cast<const uint4*>(x + block * 256 + lane * 8);
    const auto& activation =
        reinterpret_cast<const turbomind::Array<half, 8>&>(loaded);
    if constexpr (FactorScale) {
      float local = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        local = fmaf(__half2float(activation[i]), values[i], local);
      sum = fmaf(Decode::block_scale(data, lane * 8), local, sum);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        sum = fmaf(__half2float(activation[i]), values[i], sum);
    }
    __syncwarp();
    if constexpr (Prefetch) {
      if (load_next) reinterpret_cast<uint64_t*>(raw[warp])[lane] = next_word;
      __syncwarp();
    }
  }
#pragma unroll
  for (int distance = 16; distance > 0; distance /= 2)
    sum += __shfl_down_sync(0xffffffffU, sum, distance);
  if (lane == 0) {
    if constexpr (Split)
      partial[(int64_t)blockIdx.y * n + row] = sum;
    else
      out[row] = __float2half_rn(sum);
  }
}

__global__ void reduce_vec(half* out, const float* partial, int n, int splits) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n) return;
  float sum = 0.f;
  for (int p = 0; p < splits; ++p) sum += partial[(int64_t)p * n + col];
  out[col] = __float2half_rn(sum);
}

template <int Type, int NT, int MT, bool Compact = false,
          bool FullWidth = false, bool Prefetch = false, bool Staged = false,
          bool StageActivation = false>
__device__ __forceinline__ void raw_mma_body(half* out, float* partial,
                                             const half* x,
                                             const uint8_t* weight, int m,
                                             int n, int k, int stride,
                                             int splits) {
  using Decode = vllm::sm70_gguf::LatticeRawDecoder<Type>;
  using Packed = vllm::sm70_gguf::LatticeCompactDecoder<Type>;
  using MMA = turbomind::gemm::SM70_MMA_884;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  __shared__ __align__(16) uint8_t raw[NT][120];
  __shared__ float sums[4][MT][NT];
  __shared__ __align__(16) half activations[StageActivation ? MT : 1][264];
  static_assert(!StageActivation ||
                (Compact && NT == 32 && MT == 16 && !Staged));
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int col_begin = blockIdx.x * NT, row_begin = blockIdx.y * MT;
  const int bcol = lane / 16 * 4 + (lane & 12) * 2 + lane % 4;
  const int arow = lane / 16 * 4 + lane % 4;
  typename MMA::FragC accum[MT / 8]{};
  const int blocks = k / 256;
  const int begin = blocks * blockIdx.z / splits;
  const int end = blocks * (blockIdx.z + 1) / splits;
  constexpr int kStageWords = Packed::kBlockBytes * 32 / 8;
  constexpr int kWordsPerThread = (kStageWords + 127) / 128;
  __shared__ uint64_t packets[Staged ? kStageWords : 1];
  if constexpr (Staged) {
    static_assert(Compact && FullWidth && Prefetch);
    const auto* first_tile = reinterpret_cast<const uint64_t*>(
        weight + int64_t{col_begin} * blocks * Packed::kBlockBytes +
        int64_t{begin} * 32 * Packed::kBlockBytes);
#pragma unroll
    for (int i = 0; i < kWordsPerThread; ++i) {
      const int word = threadIdx.x + i * 128;
      if (word < kStageWords) packets[word] = first_tile[word];
    }
  }
  if constexpr (StageActivation) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int index = threadIdx.x + i * 128;
      const int row = index / 32, octet = index % 32;
      uint4 value{};
      if (row_begin + row < m)
        value = *reinterpret_cast<const uint4*>(
            x + int64_t{row_begin + row} * k + begin * 256 + octet * 8);
      *reinterpret_cast<uint4*>(&activations[row][octet * 8]) = value;
    }
  }
  for (int block = begin; block < end; ++block) {
    uint64_t next_words[kWordsPerThread]{};
    if constexpr (Staged) {
      __syncthreads();
      // SM70 has no cp.async. Hold the following block in registers while
      // every warp decodes and multiplies the current shared block. Include
      // original d/scales so reconstruction needs no serial metadata LDG.
      if (block + 1 < end) {
        const auto* next_tile = reinterpret_cast<const uint64_t*>(
            weight + int64_t{col_begin} * blocks * Packed::kBlockBytes +
            int64_t{block + 1} * 32 * Packed::kBlockBytes);
#pragma unroll
        for (int i = 0; i < kWordsPerThread; ++i) {
          const int word = threadIdx.x + i * 128;
          if (word < kStageWords) next_words[i] = next_tile[word];
        }
      }
    }
    uint4 next_activations[StageActivation ? 4 : 1]{};
    if constexpr (StageActivation) {
      __syncthreads();
      // Keep the following contiguous activation tile in registers while
      // the current shared tile feeds decoding/MMA. SM70 has no cp.async.
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int index = threadIdx.x + i * 128;
        const int row = index / 32, octet = index % 32;
        if (block + 1 < end && row_begin + row < m)
          next_activations[i] = *reinterpret_cast<const uint4*>(
              x + int64_t{row_begin + row} * k + (block + 1) * 256 + octet * 8);
      }
    }
    const int width = FullWidth ? 32 : min(NT, n - col_begin);
    const uint8_t* tile = nullptr;
    typename Packed::Parameters parameters{};
    if constexpr (Compact) {
      static_assert(NT == 32);
      tile = weight + int64_t{col_begin} * blocks * Packed::kBlockBytes +
             int64_t{block} * width * Packed::kBlockBytes;
      if (FullWidth || col_begin + bcol < n)
        parameters = Packed::template parameters<FullWidth>(
            Staged ? reinterpret_cast<const uint8_t*>(packets) : tile, width,
            bcol);
    } else {
      // All loaders in a warp read consecutive uint64 words of one source row.
      for (int row = warp; row < NT; row += 4) {
        if (col_begin + row < n)
          stage_block<Type>(raw[row],
                            weight + (int64_t)(col_begin + row) * stride, block,
                            stride);
        else {
          if (lane < 15) reinterpret_cast<uint64_t*>(raw[row])[lane] = 0;
          __syncwarp();
        }
      }
      __syncthreads();
    }
    if constexpr (Compact && Prefetch) {
      const bool valid = FullWidth || col_begin + bcol < n;
      const auto* packet_tile =
          Staged ? reinterpret_cast<const uint8_t*>(packets) : tile;
      auto current =
          Packed::template fetch<FullWidth>(packet_tile, width, warp * 8);
      typename MMA::FragA activation[MT / 8]{};
#pragma unroll
      for (int index = 0; index < MT / 8; ++index) {
        const int row = row_begin + index * 8 + arow;
        if (row < m)
          *reinterpret_cast<uint4*>(&activation[index]) =
              *reinterpret_cast<const uint4*>(
                  StageActivation
                      ? &activations[index * 8 + arow][warp * 64]
                      : x + int64_t{row} * k + block * 256 + warp * 64);
      }
#pragma unroll
      for (int step = 0; step < 64; step += 8) {
        typename Packed::PacketWindow next{};
        typename MMA::FragA next_activation[MT / 8]{};
        if (step < 56) {
          next = Packed::template fetch<FullWidth>(packet_tile, width,
                                                   warp * 8 + step / 8 + 1);
#pragma unroll
          for (int index = 0; index < MT / 8; ++index) {
            const int row = row_begin + index * 8 + arow;
            if (row < m)
              *reinterpret_cast<uint4*>(&next_activation[index]) =
                  *reinterpret_cast<const uint4*>(
                      StageActivation
                          ? &activations[index * 8 + arow][warp * 64 + step + 8]
                          : x + int64_t{row} * k + block * 256 + warp * 64 +
                                step + 8);
          }
        }
        const uint32_t packet = Packed::extract(current, valid ? bcol : 0);
        typename MMA::FragB b{};
        if (valid) {
          const auto values = Packed::template fragment<half>(
              parameters, packet, warp * 8 + step / 8, grid);
#pragma unroll
          for (int j = 0; j < 8; ++j) b[j] = values[j];
        }
#pragma unroll
        for (int index = 0; index < MT / 8; ++index) {
          MMA::fma(accum[index], activation[index], b, accum[index]);
          activation[index] = next_activation[index];
        }
        current = next;
      }
    } else {
#pragma unroll
      for (int step = 0; step < 64; step += 8) {
        const int base = warp * 64 + step;
        typename MMA::FragB b{};
        if constexpr (Compact) {
          const bool valid = FullWidth || col_begin + bcol < n;
          const uint32_t packet = Packed::template packet<FullWidth>(
              tile, width, base / 8, valid ? bcol : 0);
          if (valid) {
            const auto values = Packed::template fragment<half>(
                parameters, packet, base / 8, grid);
#pragma unroll
            for (int i = 0; i < 8; ++i) b[i] = values[i];
          }
        } else if (bcol < NT && col_begin + bcol < n) {
          const auto values = Decode::template fragment<half>(
              raw[bcol] + (block * Decode::kBlockBytes & 7), base, grid);
#pragma unroll
          for (int i = 0; i < 8; ++i) b[i] = values[i];
        }
#pragma unroll
        for (int tile = 0; tile < MT / 8; ++tile) {
          typename MMA::FragA a{};
          const int row = row_begin + tile * 8 + arow;
          if (row < m) {
            *reinterpret_cast<uint4*>(&a) = *reinterpret_cast<const uint4*>(
                StageActivation ? &activations[tile * 8 + arow][base]
                                : x + (int64_t)row * k + block * 256 + base);
          }
          MMA::fma(accum[tile], a, b, accum[tile]);
        }
      }
    }
    if constexpr (Staged) {
      __syncthreads();
      if (block + 1 < end) {
#pragma unroll
        for (int i = 0; i < kWordsPerThread; ++i) {
          const int word = threadIdx.x + i * 128;
          if (word < kStageWords) packets[word] = next_words[i];
        }
      }
    } else if constexpr (StageActivation) {
      __syncthreads();
      if (block + 1 < end) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int index = threadIdx.x + i * 128;
          const int row = index / 32, octet = index % 32;
          *reinterpret_cast<uint4*>(&activations[row][octet * 8]) =
              next_activations[i];
        }
      }
    } else if constexpr (!Compact) {
      __syncthreads();
    }
  }
  const auto origin = MMA::thread_offset_C();
  constexpr auto offsets = MMA::static_offset_C();
#pragma unroll
  for (int tile = 0; tile < MT / 8; ++tile)
#pragma unroll
    for (int pair = 0; pair < 4; ++pair)
#pragma unroll
      for (int item = 0; item < 2; ++item) {
        const int row = tile * 8 + origin.x + offsets[pair].x;
        const int col = origin.y + offsets[pair].y + item;
        if (col < NT) sums[warp][row][col] = accum[tile][pair * 2 + item];
      }
  __syncthreads();
  for (int i = threadIdx.x; i < MT * NT; i += blockDim.x) {
    const int row = i / NT, col = i % NT;
    if (row_begin + row < m && col_begin + col < n) {
      float value = 0.f;
#pragma unroll
      for (int w = 0; w < 4; ++w) value += sums[w][row][col];
      const int64_t index = (int64_t)(row_begin + row) * n + col_begin + col;
      if (splits == 1)
        out[index] = __float2half_rn(value);
      else
        partial[(int64_t)blockIdx.z * m * n + index] = value;
    }
  }
}

template <int Type, int NT, int MT, bool Compact = false,
          bool FullWidth = false, bool Prefetch = false, bool Staged = false,
          bool StageActivation = false>
__global__ void raw_mma_kernel(half* out, float* partial, const half* x,
                               const uint8_t* weight, int m, int n, int k,
                               int stride, int splits) {
  raw_mma_body<Type, NT, MT, Compact, FullWidth, Prefetch, Staged,
               StageActivation>(out, partial, x, weight, m, n, k, stride,
                                splits);
}

template <int Type, bool FullWidth, bool Prefetch, bool Staged>
__global__ __launch_bounds__(128, 7) void bounded_mma_kernel(
    half* out, float* partial, const half* x, const uint8_t* weight, int m,
    int n, int k, int stride, int splits) {
  raw_mma_body<Type, 32, 16, true, FullWidth, Prefetch, Staged>(
      out, partial, x, weight, m, n, k, stride, splits);
}

template <int Type>
__global__ void compact_reorder_kernel(uint8_t* out, const uint8_t* source,
                                       int n, int blocks, int stride,
                                       int64_t storage_bytes) {
  using Decode = vllm::sm70_gguf::LatticeCompactDecoder<Type>;
  const int first = blockIdx.x * 32, width = min(32, n - first);
  const int block = blockIdx.y;
  uint8_t* tile = out + int64_t{first} * blocks * Decode::kBlockBytes +
                  int64_t{block} * width * Decode::kBlockBytes;
  const int packet_bytes = width * Decode::kPacketBytesPerRow;
  for (int word = threadIdx.x; word < packet_bytes / 8; word += blockDim.x) {
    const int bit = word * 64;
    int index = bit / Decode::kPacketBits;
    int shift = -(bit % Decode::kPacketBits);
    uint64_t packed = 0;
    while (shift < 64 && index < width * 32) {
      const int col = index % width, octet = index / width;
      const auto* raw =
          source + int64_t{first + col} * stride + block * Decode::kBlockBytes;
      const uint64_t packet = Decode::source_packet(raw, octet);
      packed |= shift < 0 ? packet >> -shift : packet << shift;
      shift += Decode::kPacketBits;
      ++index;
    }
    // Tail macro-tiles are only two-byte aligned. This is a loading-time
    // permutation; four natural uint16 writes avoid padding every macro-tile.
    auto* destination = reinterpret_cast<uint16_t*>(tile + word * 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) destination[j] = packed >> (j * 16);
  }
  if (threadIdx.x < width) {
    const auto* raw = source + int64_t{first + threadIdx.x} * stride +
                      block * Decode::kBlockBytes;
    reinterpret_cast<uint16_t*>(tile + packet_bytes)[threadIdx.x] =
        *reinterpret_cast<const uint16_t*>(raw);
  }
  for (int i = threadIdx.x; i < width * Decode::kScaleBytes; i += blockDim.x) {
    const int col = i / Decode::kScaleBytes,
              component = i % Decode::kScaleBytes;
    const auto* raw =
        source + int64_t{first + col} * stride + block * Decode::kBlockBytes;
    tile[packet_bytes + width * 2 + i] =
        raw[(Type == 21 ? 106 : 74) + component];
  }
  if (blockIdx.x == (n - 1) / 32 && block == blocks - 1 && threadIdx.x == 0)
    for (int64_t i = int64_t{n} * blocks * Decode::kBlockBytes;
         i < storage_bytes; ++i)
      out[i] = 0;
}

// Decode one compressed tile for all row groups in a CTA. Temporary FP16
// fragments match official FP32 dequantization followed by the final cast.
template <int Type, int TileM>
__global__ void compact_prefill_kernel(half* out, float* partial,
                                       const half* input, const uint8_t* weight,
                                       int m, int n, int k, int splits) {
  using Decode = vllm::sm70_gguf::LatticeCompactDecoder<Type>;
  using MMA = turbomind::gemm::SM70_MMA_884;
  constexpr int WarpRows = TileM / 4;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  __shared__ __align__(16) half activations[TileM][72];
  __shared__ __align__(16) half weights[128][72];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int col_begin = blockIdx.x * 128, row_begin = blockIdx.y * TileM;
  const int decode_group = warp / 2, first_col = col_begin + decode_group * 32;
  const int physical_col =
      (lane & 3) | ((lane & 4) << 2) | ((lane & 8) >> 1) | ((lane & 16) >> 1);
  const int row_group = warp / 2, col_group = warp % 2;
  const int arow = lane / 16 * 4 + lane % 4;
  typename MMA::FragC accum[WarpRows / 8][2]{};
  const int blocks = k / 256;
  const int begin = blocks * blockIdx.z / splits;
  const int end = blocks * (blockIdx.z + 1) / splits;
  for (int block = begin; block < end; ++block) {
    const uint8_t* tile = nullptr;
    typename Decode::Parameters parameters{};
    if (first_col < n) {
      tile = weight + int64_t{first_col} * blocks * Decode::kBlockBytes +
             int64_t{block} * 32 * Decode::kBlockBytes;
      parameters = Decode::template parameters<true>(tile, 32, lane);
    }
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      for (int index = threadIdx.x; index < TileM * 8; index += 256) {
        const int row = index / 8, octet = index % 8;
        uint4 value{};
        if (row_begin + row < m)
          value = *reinterpret_cast<const uint4*>(
              input + int64_t{row_begin + row} * k + block * 256 + chunk * 64 +
              octet * 8);
        *reinterpret_cast<uint4*>(&activations[row][octet * 8]) = value;
      }
#pragma unroll
      for (int octet = warp % 2; octet < 8; octet += 2) {
        __align__(16) turbomind::Array<half, 8> values{};
        if (first_col < n) {
          const uint32_t packet =
              Decode::template packet<true>(tile, 32, chunk * 8 + octet, lane);
          values = Decode::template fragment<half>(parameters, packet,
                                                   chunk * 8 + octet, grid);
        }
        *reinterpret_cast<uint4*>(
            &weights[decode_group * 32 + physical_col][octet * 8]) =
            *reinterpret_cast<const uint4*>(&values);
      }
      __syncthreads();
#pragma unroll
      for (int step = 0; step < 64; step += 8) {
        typename MMA::FragA a[WarpRows / 8];
        typename MMA::FragB b[2];
#pragma unroll
        for (int i = 0; i < WarpRows / 8; ++i)
          *reinterpret_cast<uint4*>(&a[i]) = *reinterpret_cast<const uint4*>(
              &activations[row_group * WarpRows + i * 8 + arow][step]);
#pragma unroll
        for (int j = 0; j < 2; ++j)
          *reinterpret_cast<uint4*>(&b[j]) = *reinterpret_cast<const uint4*>(
              &weights[(col_group * 2 + j) * 32 + lane][step]);
#pragma unroll
        for (int i = 0; i < WarpRows / 8; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            MMA::fma(accum[i][j], a[i], b[j], accum[i][j]);
      }
      __syncthreads();
    }
  }
  const auto origin = MMA::thread_offset_C();
  constexpr auto offsets = MMA::static_offset_C();
#pragma unroll
  for (int i = 0; i < WarpRows / 8; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int pair = 0; pair < 4; ++pair)
#pragma unroll
        for (int item = 0; item < 2; ++item) {
          const int row = row_begin + row_group * WarpRows + i * 8 + origin.x +
                          offsets[pair].x;
          const int col = col_begin + col_group * 64 + j * 32 + origin.y +
                          offsets[pair].y + item;
          if (row < m && col < n) {
            const int64_t index = int64_t{row} * n + col;
            const float value = accum[i][j][pair * 2 + item];
            if (splits == 1)
              out[index] = __float2half_rn(value);
            else
              partial[int64_t{blockIdx.z} * m * n + index] = value;
          }
        }
}

template <int Type, class Output, bool Transpose, bool FullWidth = false,
          bool PackedOutput = false>
__global__ void compact_dequant_kernel(Output* out, const uint8_t* weight,
                                       int n, int k) {
  using Decode = vllm::sm70_gguf::LatticeCompactDecoder<Type>;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int first = blockIdx.x * 32,
            width = FullWidth ? 32 : min(32, n - first);
  const int block = blockIdx.y;
  const auto* tile = weight + int64_t{first} * (k / 256) * Decode::kBlockBytes +
                     int64_t{block} * width * Decode::kBlockBytes;
  typename Decode::Parameters parameters{};
  if (lane < width)
    parameters = Decode::template parameters<FullWidth>(tile, width, lane);
  const int warps = blockDim.x / 32;
  const int first_octet = warp + warps * blockIdx.z;
  const int step = warps * gridDim.z;
  auto current = Decode::template fetch<FullWidth>(tile, width, first_octet);
  for (int octet = first_octet; octet < 32; octet += step) {
    typename Decode::PacketWindow next{};
    if (octet + step < 32)
      next = Decode::template fetch<FullWidth>(tile, width, octet + step);
    const auto packet = Decode::extract(current, lane < width ? lane : 0);
    if (lane < width) {
      const auto values =
          Decode::template fragment<Output>(parameters, packet, octet, grid);
      if constexpr (std::is_same_v<Output, half> && !Transpose) {
        static_assert(!PackedOutput || FullWidth);
        const int64_t index =
            PackedOutput
                ? ((int64_t{first / 32} * (k / 8) + block * 32 + octet) * 32 +
                   lane) *
                      8
                : int64_t{first + lane} * k + block * 256 + octet * 8;
        *reinterpret_cast<uint4*>(out + index) =
            *reinterpret_cast<const uint4*>(&values);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int logical_k = block * 256 + octet * 8 + j, row = first + lane;
          const int64_t index = Transpose ? int64_t{logical_k} * n + row
                                          : int64_t{row} * k + logical_k;
          out[index] = static_cast<Output>(values[j]);
        }
      }
    }
    current = next;
  }
}

template <int Type, bool FullWidth = false>
__global__ void compact_dequant_natural_kernel(half* out, const uint8_t* weight,
                                               int n, int k) {
  using Decode = vllm::sm70_gguf::LatticeCompactDecoder<Type>;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  // Eight-half row padding keeps every vector aligned. Shared copy swaps
  // warp ownership from N columns to contiguous K without a strided store.
  __shared__ __align__(16) half decoded[32][264];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int first = blockIdx.x * 32,
            width = FullWidth ? 32 : min(32, n - first);
  const int block = blockIdx.y;
  const auto* tile = weight + int64_t{first} * (k / 256) * Decode::kBlockBytes +
                     int64_t{block} * width * Decode::kBlockBytes;
  typename Decode::Parameters parameters{};
  if (lane < width)
    parameters = Decode::template parameters<FullWidth>(tile, width, lane);
  for (int octet = warp; octet < 32; octet += 4) {
    const auto packet = Decode::template packet<FullWidth>(
        tile, width, octet, lane < width ? lane : 0);
    if (lane < width) {
      const auto values =
          Decode::template fragment<half>(parameters, packet, octet, grid);
      *reinterpret_cast<uint4*>(&decoded[lane][octet * 8]) =
          *reinterpret_cast<const uint4*>(&values);
    }
  }
  __syncthreads();
  for (int index = threadIdx.x; index < 32 * 32; index += blockDim.x) {
    const int row = index / 32, octet = index % 32;
    if (row < width) {
      const int64_t destination =
          int64_t{first + row} * k + block * 256 + octet * 8;
      *reinterpret_cast<uint4*>(out + destination) =
          *reinterpret_cast<const uint4*>(&decoded[row][octet * 8]);
    }
  }
}

template <int Type, bool Split>
__global__ void compact_vec_kernel(half* out, float* partial, const half* x,
                                   const uint8_t* weight, int n, int k,
                                   int splits) {
  using Decode = vllm::sm70_gguf::LatticeCompactDecoder<Type>;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int first = blockIdx.x * 128 + warp * 32;
  if (first >= n) return;
  const int width = min(32, n - first), blocks = k / 256;
  const int begin = blocks * blockIdx.y / splits;
  const int end = blocks * (blockIdx.y + 1) / splits;
  float sum = 0.f;
  for (int block = begin; block < end; ++block) {
    const auto* tile = weight + int64_t{first} * blocks * Decode::kBlockBytes +
                       int64_t{block} * width * Decode::kBlockBytes;
    typename Decode::Parameters parameters{};
    if (lane < width) parameters = Decode::parameters(tile, width, lane);
    for (int octet = 0; octet < 32; ++octet) {
      const auto packet =
          Decode::packet(tile, width, octet, lane < width ? lane : 0);
      const auto values = Decode::fragment(parameters, packet, octet, grid);
      const uint4 loaded =
          *reinterpret_cast<const uint4*>(x + block * 256 + octet * 8);
      const auto& activation =
          reinterpret_cast<const turbomind::Array<half, 8>&>(loaded);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        sum = fmaf(__half2float(activation[j]), values[j], sum);
    }
  }
  if (lane < width) {
    if constexpr (Split)
      partial[int64_t{blockIdx.y} * n + first + lane] = sum;
    else
      out[first + lane] = __float2half_rn(sum);
  }
}

template <int Type, bool Split, bool FullWidth = false>
__global__ void compact_row_vec_kernel(half* out, float* partial, const half* x,
                                       const uint8_t* weight, int n, int k,
                                       int splits) {
  using Decode = vllm::sm70_gguf::LatticeCompactDecoder<Type>;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.x * 4 + warp;
  if (row >= n) return;
  const int first = row / 32 * 32;
  const int width = FullWidth ? 32 : min(32, n - first), col = row - first;
  const int blocks = k / 256, begin = blocks * blockIdx.y / splits;
  const int end = blocks * (blockIdx.y + 1) / splits;
  float sum = 0.f;
  for (int block = begin; block < end; ++block) {
    const auto* tile = weight + int64_t{first} * blocks * Decode::kBlockBytes +
                       int64_t{block} * width * Decode::kBlockBytes;
    const auto parameters =
        Decode::template parameters<FullWidth>(tile, width, col);
    const auto packet = Decode::scalar_packet(tile, width, lane, col);
    const auto values = Decode::fragment(parameters, packet, lane, grid);
    const uint4 loaded =
        *reinterpret_cast<const uint4*>(x + block * 256 + lane * 8);
    const auto& activation =
        reinterpret_cast<const turbomind::Array<half, 8>&>(loaded);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      sum = fmaf(__half2float(activation[j]), values[j], sum);
  }
#pragma unroll
  for (int distance = 16; distance > 0; distance /= 2)
    sum += __shfl_down_sync(0xffffffffU, sum, distance);
  if (lane == 0) {
    if constexpr (Split)
      partial[int64_t{blockIdx.y} * n + row] = sum;
    else
      out[row] = __float2half_rn(sum);
  }
}

template <int Type, int NT, int MT, bool Compact = false,
          bool FullWidth = false, bool Prefetch = false, bool Staged = false,
          bool Bounded = false, bool StageActivation = false>
void launch_mma(torch::Tensor out, torch::Tensor input, torch::Tensor weight,
                torch::Tensor partial, int splits, cudaStream_t stream) {
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  const dim3 grid((n + NT - 1) / NT, (m + MT - 1) / MT, splits);
  if constexpr (Bounded && MT == 16) {
    static_assert(Compact && NT == 32);
    bounded_mma_kernel<Type, FullWidth, Prefetch, Staged>
        <<<grid, 128, 0, stream>>>(
            reinterpret_cast<half*>(out.data_ptr()),
            splits > 1 ? partial.data_ptr<float>() : nullptr,
            reinterpret_cast<const half*>(input.data_ptr()),
            weight.data_ptr<uint8_t>(), m, n, k, 0, splits);
  } else {
    raw_mma_kernel<Type, NT, MT, Compact, FullWidth, Prefetch, Staged,
                   StageActivation><<<grid, 128, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()),
        splits > 1 ? partial.data_ptr<float>() : nullptr,
        reinterpret_cast<const half*>(input.data_ptr()),
        weight.data_ptr<uint8_t>(), m, n, k, Compact ? 0 : weight.size(1),
        splits);
  }
}
void validate_raw(torch::Tensor w, int type, int64_t n, int64_t k,
                  bool allow_xxs = false) {
  TORCH_CHECK(type == 21 || type == 22 || (allow_xxs && type == 18),
              "Unsupported raw GGUF lattice type");
  const int block_bytes = type == 18 ? 98 : type == 21 ? 110 : 82;
  const int64_t row_bytes = k / 256 * block_bytes;
  TORCH_CHECK(
      w.is_cuda() && w.scalar_type() == torch::kUInt8 && w.dim() == 2 &&
          w.is_contiguous() && w.size(0) == n && n > 0 && n <= INT_MAX &&
          k > 0 && k <= INT_MAX && k % 256 == 0 &&
          w.size(1) == (row_bytes + 7) / 8 * 8,
      "Raw GGUF storage must contain original rows with only 8-byte alignment");
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 0, "Raw GGUF requires SM70");
}

template <int Type>
void launch_dequant(torch::Tensor out, torch::Tensor weight,
                    cudaStream_t stream) {
  const dim3 grid((out.size(0) + 3) / 4, out.size(1) / 256);
  if (out.scalar_type() == torch::kFloat32)
    raw_dequant_kernel<Type><<<grid, 128, 0, stream>>>(
        out.data_ptr<float>(), weight.data_ptr<uint8_t>(), out.size(0),
        out.size(1), weight.size(1));
  else
    raw_dequant_kernel<Type><<<grid, 128, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()), weight.data_ptr<uint8_t>(),
        out.size(0), out.size(1), weight.size(1));
}
void lattice_blas_accumulate(torch::Tensor out, torch::Tensor input,
                             torch::Tensor scratch, bool natural_layout = false,
                             int64_t algorithm = 99) {
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  cublasMath_t saved;
  TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle, &saved));
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(
      handle, static_cast<cublasMath_t>(
                  CUBLAS_TENSOR_OP_MATH |
                  CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  const float alpha = 1.f, beta = 0.f;
  const auto status = cublasGemmEx(
      handle, natural_layout ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
      &alpha, scratch.data_ptr(), CUDA_R_16F, natural_layout ? k : n,
      input.data_ptr(), CUDA_R_16F, k, &beta, out.data_ptr(),
      out.scalar_type() == torch::kFloat32 ? CUDA_R_32F : CUDA_R_16F, n,
      CUBLAS_COMPUTE_32F, static_cast<cublasGemmAlgo_t>(algorithm));
  const auto restore = cublasSetMathMode(handle, saved);
  TORCH_CUDABLAS_CHECK(status);
  TORCH_CUDABLAS_CHECK(restore);
}
}  // namespace

void gguf_lattice_raw_dequantize_sm70_out(torch::Tensor out,
                                          torch::Tensor weight,
                                          int64_t source_type) {
  TORCH_CHECK(out.device() == weight.device() && out.dim() == 2 &&
                  out.is_contiguous() &&
                  (out.scalar_type() == torch::kFloat16 ||
                   out.scalar_type() == torch::kFloat32),
              "Raw GGUF dequant requires a contiguous FP16/FP32 [N,K] output");
  const c10::cuda::CUDAGuard guard(weight.device());
  validate_raw(weight, source_type, out.size(0), out.size(1), true);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (source_type == 18)
    launch_dequant<18>(out, weight, stream);
  else if (source_type == 21)
    launch_dequant<21>(out, weight, stream);
  else
    launch_dequant<22>(out, weight, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_lattice_raw_vec_sm70_out(torch::Tensor out, torch::Tensor input,
                                   torch::Tensor weight, int64_t source_type,
                                   torch::Tensor partial, int64_t splits,
                                   bool prefetch, bool factor_scale) {
  TORCH_CHECK(source_type != 18 || !factor_scale,
              "IQ3_XXS retains original FP32 vector accumulation");
  TORCH_CHECK(input.device() == weight.device() &&
                  out.device() == weight.device() &&
                  input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
                  input.size(0) == 1 && out.dim() == 2 && out.size(0) == 1 &&
                  input.is_contiguous() && out.is_contiguous(),
              "Raw GGUF vector requires FP16 M=1");
  const c10::cuda::CUDAGuard guard(weight.device());
  const int n = out.size(1), k = input.size(1);
  validate_raw(weight, source_type, n, k, true);
  TORCH_CHECK(splits >= 1 && splits <= k / 256,
              "Invalid raw GGUF split-K count");
  if (splits > 1)
    TORCH_CHECK(partial.device() == weight.device() &&
                    partial.scalar_type() == torch::kFloat32 &&
                    partial.is_contiguous() && partial.numel() >= splits * n,
                "Raw GGUF split-K requires FP32 partial storage");
  const dim3 grid((n + 3) / 4, splits);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define RAW_VEC(TYPE, PREFETCH, FACTOR)                                      \
  if (splits == 1)                                                           \
    raw_vec_kernel<TYPE, false, PREFETCH, FACTOR><<<grid, 128, 0, stream>>>( \
        reinterpret_cast<half*>(out.data_ptr()), nullptr,                    \
        reinterpret_cast<const half*>(input.data_ptr()),                     \
        weight.data_ptr<uint8_t>(), n, k, weight.size(1), splits);           \
  else                                                                       \
    raw_vec_kernel<TYPE, true, PREFETCH, FACTOR><<<grid, 128, 0, stream>>>(  \
        reinterpret_cast<half*>(out.data_ptr()), partial.data_ptr<float>(),  \
        reinterpret_cast<const half*>(input.data_ptr()),                     \
        weight.data_ptr<uint8_t>(), n, k, weight.size(1), splits)
#define SELECT_RAW_VEC(TYPE, FACTOR) \
  if (prefetch) {                    \
    RAW_VEC(TYPE, true, FACTOR);     \
  } else {                           \
    RAW_VEC(TYPE, false, FACTOR);    \
  }
  if (source_type == 18) {
    SELECT_RAW_VEC(18, false);
  } else if (source_type == 21) {
    if (factor_scale) {
      SELECT_RAW_VEC(21, true);
    } else {
      SELECT_RAW_VEC(21, false);
    }
  } else {
    if (factor_scale) {
      SELECT_RAW_VEC(22, true);
    } else {
      SELECT_RAW_VEC(22, false);
    }
  }
#undef SELECT_RAW_VEC
#undef RAW_VEC
  if (splits > 1)
    reduce_vec<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()), partial.data_ptr<float>(), n,
        splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_lattice_raw_blas_sm70_out(torch::Tensor out, torch::Tensor input,
                                    torch::Tensor weight, int64_t source_type,
                                    torch::Tensor scratch) {
  TORCH_CHECK(
      input.device() == weight.device() && out.device() == weight.device() &&
          scratch.device() == weight.device() &&
          input.scalar_type() == torch::kFloat16 &&
          out.scalar_type() == torch::kFloat16 &&
          scratch.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
          out.dim() == 2 && scratch.dim() == 2 && input.is_contiguous() &&
          out.is_contiguous() && scratch.is_contiguous() &&
          out.size(0) == input.size(0) && scratch.size(0) == input.size(1) &&
          scratch.size(1) == out.size(1),
      "Raw GGUF BLAS requires FP16 [K,N] scratch");
  const c10::cuda::CUDAGuard guard(weight.device());
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  validate_raw(weight, source_type, n, k);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((n + 31) / 32, k / 256);
  if (source_type == 21)
    raw_dequant_transpose_kernel<21><<<grid, 128, 0, stream>>>(
        reinterpret_cast<half*>(scratch.data_ptr()), weight.data_ptr<uint8_t>(),
        n, k, weight.size(1));
  else
    raw_dequant_transpose_kernel<22><<<grid, 128, 0, stream>>>(
        reinterpret_cast<half*>(scratch.data_ptr()), weight.data_ptr<uint8_t>(),
        n, k, weight.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  lattice_blas_accumulate(out, input, scratch);
}

void gguf_lattice_raw_mma_sm70_out(torch::Tensor out, torch::Tensor input,
                                   torch::Tensor weight, int64_t source_type,
                                   torch::Tensor partial, int64_t splits,
                                   int64_t tile_n) {
  TORCH_CHECK(
      input.device() == weight.device() && out.device() == weight.device() &&
          input.scalar_type() == torch::kFloat16 &&
          out.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
          out.dim() == 2 && input.size(0) == out.size(0) && input.size(0) > 0 &&
          input.size(0) <= 64 && input.is_contiguous() && out.is_contiguous(),
      "Raw GGUF MMA requires contiguous FP16 matrices and M <= 64");
  const c10::cuda::CUDAGuard guard(weight.device());
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  validate_raw(weight, source_type, n, k);
  TORCH_CHECK(splits >= 1 && splits <= k / 256 && (tile_n == 8 || tile_n == 32),
              "Invalid raw GGUF MMA tile or split-K count");
  if (splits > 1)
    TORCH_CHECK(partial.device() == weight.device() &&
                    partial.scalar_type() == torch::kFloat32 &&
                    partial.is_contiguous() &&
                    partial.numel() >= splits * m * n,
                "Raw GGUF MMA split-K requires FP32 partial storage");
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_RAW_MMA(TYPE, NT)                                           \
  if (m <= 8)                                                              \
    launch_mma<TYPE, NT, 8>(out, input, weight, partial, splits, stream);  \
  else if (m <= 16)                                                        \
    launch_mma<TYPE, NT, 16>(out, input, weight, partial, splits, stream); \
  else                                                                     \
    launch_mma<TYPE, NT, 32>(out, input, weight, partial, splits, stream)
  if (source_type == 21) {
    if (tile_n == 8) {
      LAUNCH_RAW_MMA(21, 8);
    } else {
      LAUNCH_RAW_MMA(21, 32);
    }
  } else {
    if (tile_n == 8) {
      LAUNCH_RAW_MMA(22, 8);
    } else {
      LAUNCH_RAW_MMA(22, 32);
    }
  }
#undef LAUNCH_RAW_MMA
  if (splits > 1)
    reduce_vec<<<(m * n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()), partial.data_ptr<float>(),
        m * n, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
void validate_compact(torch::Tensor weight, int type, int64_t n, int64_t k) {
  TORCH_CHECK(type == 21 || type == 22, "Unsupported compact GGUF type");
  const int bytes = type == 21 ? 110 : 82;
  TORCH_CHECK(n > 0 && n <= INT_MAX && k > 0 && k <= INT_MAX && k % 256 == 0 &&
                  weight.is_cuda() && weight.scalar_type() == torch::kUInt8 &&
                  weight.dim() == 1 && weight.is_contiguous() &&
                  weight.numel() == (n * (k / 256) * bytes + 7) / 8 * 8,
              "Compact GGUF must retain the exact source bit count");
  const auto* device = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(device->major == 7 && device->minor == 0,
              "Compact GGUF requires SM70");
}
void validate_compact_io(torch::Tensor out, torch::Tensor input,
                         torch::Tensor weight, bool allow_fp32_output = false) {
  TORCH_CHECK(
      input.device() == weight.device() && out.device() == weight.device() &&
          input.scalar_type() == torch::kFloat16 &&
          (out.scalar_type() == torch::kFloat16 ||
           (allow_fp32_output && out.scalar_type() == torch::kFloat32)) &&
          input.dim() == 2 && out.dim() == 2 && input.size(0) == out.size(0) &&
          input.size(0) > 0 && input.is_contiguous() && out.is_contiguous(),
      "Compact GGUF requires contiguous FP16 activations and admitted output "
      "dtype");
}
}  // namespace

void gguf_lattice_compact_reorder_sm70_out(torch::Tensor out, torch::Tensor raw,
                                           int64_t source_type,
                                           int64_t logical_k) {
  TORCH_CHECK(out.device() == raw.device(), "Compact reorder device mismatch");
  const c10::cuda::CUDAGuard guard(raw.device());
  TORCH_CHECK(raw.dim() == 2, "Compact reorder requires original GGUF rows");
  const int n = raw.size(0), k = logical_k;
  validate_raw(raw, source_type, n, k);
  validate_compact(out, source_type, n, k);
  const dim3 grid((n + 31) / 32, k / 256);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (source_type == 21)
    compact_reorder_kernel<21><<<grid, 128, 0, stream>>>(
        out.data_ptr<uint8_t>(), raw.data_ptr<uint8_t>(), n, k / 256,
        raw.size(1), out.numel());
  else
    compact_reorder_kernel<22><<<grid, 128, 0, stream>>>(
        out.data_ptr<uint8_t>(), raw.data_ptr<uint8_t>(), n, k / 256,
        raw.size(1), out.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_lattice_compact_dequantize_sm70_out(torch::Tensor out,
                                              torch::Tensor weight,
                                              int64_t source_type) {
  TORCH_CHECK(out.device() == weight.device() && out.dim() == 2 &&
                  out.is_contiguous() &&
                  (out.scalar_type() == torch::kFloat32 ||
                   out.scalar_type() == torch::kFloat16),
              "Compact GGUF dequant requires FP16/FP32 [N,K]");
  const c10::cuda::CUDAGuard guard(weight.device());
  const int n = out.size(0), k = out.size(1);
  validate_compact(weight, source_type, n, k);
  const dim3 grid((n + 31) / 32, k / 256);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define COMPACT_DQ(TYPE, FULL)                                              \
  if (out.scalar_type() == torch::kFloat32)                                 \
    compact_dequant_kernel<TYPE, float, false, FULL>                        \
        <<<grid, 256, 0, stream>>>(out.data_ptr<float>(),                   \
                                   weight.data_ptr<uint8_t>(), n, k);       \
  else                                                                      \
    compact_dequant_kernel<TYPE, half, false, FULL>                         \
        <<<grid, 256, 0, stream>>>(reinterpret_cast<half*>(out.data_ptr()), \
                                   weight.data_ptr<uint8_t>(), n, k)
#define COMPACT_DQ_SELECT(TYPE) \
  if (n % 32 == 0) {            \
    COMPACT_DQ(TYPE, true);     \
  } else {                      \
    COMPACT_DQ(TYPE, false);    \
  }
  if (source_type == 21) {
    COMPACT_DQ_SELECT(21);
  } else {
    COMPACT_DQ_SELECT(22);
  }
#undef COMPACT_DQ_SELECT
#undef COMPACT_DQ
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_lattice_compact_vec_sm70_out(torch::Tensor out, torch::Tensor input,
                                       torch::Tensor weight,
                                       int64_t source_type,
                                       torch::Tensor partial, int64_t splits,
                                       bool row_wise) {
  validate_compact_io(out, input, weight);
  const c10::cuda::CUDAGuard guard(weight.device());
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  validate_compact(weight, source_type, n, k);
  TORCH_CHECK(m == 1 && splits >= 1 && splits <= k / 256,
              "Invalid compact vector split-K");
  if (splits > 1)
    TORCH_CHECK(partial.device() == weight.device() &&
                    partial.scalar_type() == torch::kFloat32 &&
                    partial.is_contiguous() && partial.numel() >= splits * n,
                "Compact vector requires FP32 partial storage");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(row_wise ? (n + 3) / 4 : (n + 127) / 128, splits);
#define COMPACT_ROW_VEC(TYPE, FULL)                                         \
  if (splits == 1)                                                          \
    compact_row_vec_kernel<TYPE, false, FULL><<<grid, 128, 0, stream>>>(    \
        reinterpret_cast<half*>(out.data_ptr()), nullptr,                   \
        reinterpret_cast<const half*>(input.data_ptr()),                    \
        weight.data_ptr<uint8_t>(), n, k, splits);                          \
  else                                                                      \
    compact_row_vec_kernel<TYPE, true, FULL><<<grid, 128, 0, stream>>>(     \
        reinterpret_cast<half*>(out.data_ptr()), partial.data_ptr<float>(), \
        reinterpret_cast<const half*>(input.data_ptr()),                    \
        weight.data_ptr<uint8_t>(), n, k, splits)
  if (row_wise) {
    if (source_type == 21) {
      if (n % 32 == 0) {
        COMPACT_ROW_VEC(21, true);
      } else {
        COMPACT_ROW_VEC(21, false);
      }
    } else {
      if (n % 32 == 0) {
        COMPACT_ROW_VEC(22, true);
      } else {
        COMPACT_ROW_VEC(22, false);
      }
    }
  } else {
#define COMPACT_VEC(TYPE)                                                   \
  if (splits == 1)                                                          \
    compact_vec_kernel<TYPE, false><<<grid, 128, 0, stream>>>(              \
        reinterpret_cast<half*>(out.data_ptr()), nullptr,                   \
        reinterpret_cast<const half*>(input.data_ptr()),                    \
        weight.data_ptr<uint8_t>(), n, k, splits);                          \
  else                                                                      \
    compact_vec_kernel<TYPE, true><<<grid, 128, 0, stream>>>(               \
        reinterpret_cast<half*>(out.data_ptr()), partial.data_ptr<float>(), \
        reinterpret_cast<const half*>(input.data_ptr()),                    \
        weight.data_ptr<uint8_t>(), n, k, splits)
    if (source_type == 21) {
      COMPACT_VEC(21);
    } else {
      COMPACT_VEC(22);
    }
#undef COMPACT_VEC
  }
#undef COMPACT_ROW_VEC
  if (splits > 1)
    reduce_vec<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()), partial.data_ptr<float>(), n,
        splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_lattice_compact_mma_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor weight,
    int64_t source_type, torch::Tensor partial, int64_t splits, bool prefetch,
    bool staged, int64_t row_tile, bool occupancy7, bool stage_activation) {
  validate_compact_io(out, input, weight);
  const c10::cuda::CUDAGuard guard(weight.device());
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  validate_compact(weight, source_type, n, k);
  TORCH_CHECK(m <= 64 && splits >= 1 && splits <= k / 256,
              "Invalid compact MMA split-K");
  if (splits > 1)
    TORCH_CHECK(partial.device() == weight.device() &&
                    partial.scalar_type() == torch::kFloat32 &&
                    partial.is_contiguous() &&
                    partial.numel() >= splits * m * n,
                "Compact MMA requires FP32 partial storage");
  TORCH_CHECK(
      row_tile == 0 || row_tile == 8 || row_tile == 16 || row_tile == 32,
      "Compact MMA row tile must be auto, 8, 16, or 32");
  const int mt = row_tile ? row_tile : m <= 8 ? 8 : m <= 16 ? 16 : 32;
  TORCH_CHECK(!occupancy7 || mt == 16,
              "Seven-CTA calibration requires a 16-row tile");
  TORCH_CHECK(!staged || n % 32 == 0,
              "Compact shared staging requires complete N32 tiles");
  const auto stream = at::cuda::getCurrentCUDAStream();
#define COMPACT_MMA(TYPE, FULL, PREFETCH, STAGED, BOUNDED)           \
  if (mt == 8)                                                       \
    launch_mma<TYPE, 32, 8, true, FULL, PREFETCH, STAGED, BOUNDED>(  \
        out, input, weight, partial, splits, stream);                \
  else if (mt == 16)                                                 \
    launch_mma<TYPE, 32, 16, true, FULL, PREFETCH, STAGED, BOUNDED>( \
        out, input, weight, partial, splits, stream);                \
  else                                                               \
    launch_mma<TYPE, 32, 32, true, FULL, PREFETCH, STAGED, BOUNDED>( \
        out, input, weight, partial, splits, stream)
#define COMPACT_SELECT(TYPE, FULL, BOUNDED)         \
  if (prefetch) {                                   \
    COMPACT_MMA(TYPE, FULL, true, false, BOUNDED);  \
  } else {                                          \
    COMPACT_MMA(TYPE, FULL, false, false, BOUNDED); \
  }
#define COMPACT_DISPATCH(BOUNDED)                 \
  if (staged) {                                   \
    if (source_type == 21) {                      \
      COMPACT_MMA(21, true, true, true, BOUNDED); \
    } else {                                      \
      COMPACT_MMA(22, true, true, true, BOUNDED); \
    }                                             \
  } else if (source_type == 21) {                 \
    if (n % 32 == 0) {                            \
      COMPACT_SELECT(21, true, BOUNDED);          \
    } else {                                      \
      COMPACT_SELECT(21, false, BOUNDED);         \
    }                                             \
  } else {                                        \
    if (n % 32 == 0) {                            \
      COMPACT_SELECT(22, true, BOUNDED);          \
    } else {                                      \
      COMPACT_SELECT(22, false, BOUNDED);         \
    }                                             \
  }
  if (stage_activation) {
    TORCH_CHECK(mt == 16 && !staged && !occupancy7,
                "Activation staging requires an unbounded M16 tile without "
                "weight staging");
#define COMPACT_ACT(TYPE, FULL, PREF)                             \
  launch_mma<TYPE, 32, 16, true, FULL, PREF, false, false, true>( \
      out, input, weight, partial, splits, stream)
#define COMPACT_ACT_PREF(TYPE, FULL) \
  if (prefetch) {                    \
    COMPACT_ACT(TYPE, FULL, true);   \
  } else {                           \
    COMPACT_ACT(TYPE, FULL, false);  \
  }
#define COMPACT_ACT_SELECT(TYPE)   \
  if (n % 32 == 0) {               \
    COMPACT_ACT_PREF(TYPE, true);  \
  } else {                         \
    COMPACT_ACT_PREF(TYPE, false); \
  }
    if (source_type == 21) {
      COMPACT_ACT_SELECT(21);
    } else {
      COMPACT_ACT_SELECT(22);
    }
#undef COMPACT_ACT_SELECT
#undef COMPACT_ACT_PREF
#undef COMPACT_ACT
  } else if (occupancy7) {
    COMPACT_DISPATCH(true);
  } else {
    COMPACT_DISPATCH(false);
  }
#undef COMPACT_DISPATCH
#undef COMPACT_SELECT
#undef COMPACT_MMA
  if (splits > 1)
    reduce_vec<<<(m * n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()), partial.data_ptr<float>(),
        m * n, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_lattice_compact_blas_sm70_out(torch::Tensor out, torch::Tensor input,
                                        torch::Tensor weight,
                                        int64_t source_type,
                                        torch::Tensor scratch,
                                        bool natural_layout, int64_t algorithm,
                                        int64_t dq_partitions) {
  TORCH_CHECK(
      algorithm == 99 || algorithm == 102,
      "Compact GGUF BLAS accepts only default or calibrated algorithm 2");
  TORCH_CHECK(dq_partitions == 1 || dq_partitions == 2 || dq_partitions == 4,
              "Compact GGUF DQ partitions must be 1, 2 or 4");
  TORCH_CHECK(!natural_layout || dq_partitions == 1,
              "Natural-layout DQ does not support partitions");
  validate_compact_io(out, input, weight, true);
  const c10::cuda::CUDAGuard guard(weight.device());
  const int n = out.size(1), k = input.size(1);
  validate_compact(weight, source_type, n, k);
  TORCH_CHECK(scratch.device() == weight.device() &&
                  scratch.scalar_type() == torch::kFloat16 &&
                  scratch.dim() == 2 &&
                  scratch.size(0) == (natural_layout ? n : k) &&
                  scratch.size(1) == (natural_layout ? k : n) &&
                  scratch.is_contiguous(),
              "Compact GGUF BLAS scratch must match its FP16 weight layout");
  const dim3 grid((n + 31) / 32, k / 256, dq_partitions);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define COMPACT_BLAS_DQ(TYPE, FULL)                                           \
  if (natural_layout)                                                         \
    compact_dequant_natural_kernel<TYPE, FULL><<<grid, 128, 0, stream>>>(     \
        reinterpret_cast<half*>(scratch.data_ptr()),                          \
        weight.data_ptr<uint8_t>(), n, k);                                    \
  else                                                                        \
    compact_dequant_kernel<TYPE, half, true, FULL><<<grid, 256, 0, stream>>>( \
        reinterpret_cast<half*>(scratch.data_ptr()),                          \
        weight.data_ptr<uint8_t>(), n, k)
#define COMPACT_BLAS_DQ_SELECT(TYPE) \
  if (n % 32 == 0) {                 \
    COMPACT_BLAS_DQ(TYPE, true);     \
  } else {                           \
    COMPACT_BLAS_DQ(TYPE, false);    \
  }
  if (source_type == 21) {
    COMPACT_BLAS_DQ_SELECT(21);
  } else {
    COMPACT_BLAS_DQ_SELECT(22);
  }
#undef COMPACT_BLAS_DQ_SELECT
#undef COMPACT_BLAS_DQ
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  lattice_blas_accumulate(out, input, scratch, natural_layout, algorithm);
}

void gguf_lattice_compact_tm_f16_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor weight,
    int64_t source_type, torch::Tensor scratch, torch::Tensor offsets,
    torch::Tensor weight_ptrs, int64_t dq_partitions) {
  TORCH_CHECK(dq_partitions == 1 || dq_partitions == 2 || dq_partitions == 4,
              "Compact GGUF DQ partitions must be 1, 2 or 4");
  validate_compact_io(out, input, weight);
  const c10::cuda::CUDAGuard guard(weight.device());
  const int n = out.size(1), k = input.size(1);
  validate_compact(weight, source_type, n, k);
  TORCH_CHECK(n % 32 == 0 && scratch.device() == weight.device() &&
                  scratch.scalar_type() == torch::kFloat16 &&
                  scratch.dim() == 2 && scratch.size(0) == n &&
                  scratch.size(1) == k && scratch.is_contiguous(),
              "Compact GGUF TurboMind requires an aligned FP16 workspace");
  const dim3 grid(n / 32, k / 256, dq_partitions);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (source_type == 21)
    compact_dequant_kernel<21, half, false, true, true>
        <<<grid, 256, 0, stream>>>(reinterpret_cast<half*>(scratch.data_ptr()),
                                   weight.data_ptr<uint8_t>(), n, k);
  else
    compact_dequant_kernel<22, half, false, true, true>
        <<<grid, 256, 0, stream>>>(reinterpret_cast<half*>(scratch.data_ptr()),
                                   weight.data_ptr<uint8_t>(), n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gguf_workspace_f16_gemm_sm70_out(out, input, offsets, weight_ptrs);
}

namespace {
// One CTA owns two output rows from each projection of an active expert.
// Repeated route slots return before initializing the shared codebook. Every
// original block is decoded once per token pair and output row, with no
// tensor-core M tile padding. Larger expert intervals use additional pairs.
template <int Type, int MaxTokens>
__global__ void raw_grouped_gate_up_kernel(
    half* gate, half* up, const half* input, const uint8_t* gate_weights,
    const uint8_t* up_weights, const int* offsets, const int64_t* ids, int n,
    int k, int stride) {
  const int slot = blockIdx.y;
  const int expert = ids[slot];
  if (slot && ids[slot - 1] == expert) return;
  const int begin = offsets[expert], end = offsets[expert + 1];
  using Decode = vllm::sm70_gguf::LatticeRawDecoder<Type>;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  __shared__ __align__(16) uint8_t raw[4][120];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.x * 2 + warp % 2;
  if (row >= n) return;
  const uint8_t* weights = warp < 2 ? gate_weights : up_weights;
  half* output = warp < 2 ? gate : up;
  const uint8_t* row_data = weights + (int64_t{expert} * n + row) * stride;
  for (int chunk = begin; chunk < end; chunk += MaxTokens) {
    float sums[MaxTokens] = {};
    stage_block<Type>(raw[warp], row_data, 0, stride);
    for (int block = 0; block < k / 256; ++block) {
      // Use the existing raw vector's software prefetch: keep the next
      // original block in registers while decoding and multiplying this one.
      // SM70 has no cp.async. Per-token FMA and reduction order is unchanged.
      constexpr int bytes = Decode::kBlockBytes;
      const int next_start = (block + 1) * bytes;
      const int aligned = next_start & ~7, bias = next_start & 7;
      const int words = (bias + bytes + 7) / 8;
      const bool load_next =
          block + 1 < k / 256 && lane < words && aligned + lane * 8 < stride;
      uint64_t next_word = 0;
      if (load_next)
        next_word =
            *reinterpret_cast<const uint64_t*>(row_data + aligned + lane * 8);
      const auto* data = raw[warp] + (block * bytes & 7);
      const auto values = Decode::fragment(data, lane * 8, grid);
#pragma unroll
      for (int token = 0; token < MaxTokens; ++token) {
        if (chunk + token < end) {
          const auto* x =
              input + int64_t{chunk + token} * k + block * 256 + lane * 8;
          const uint4 loaded = *reinterpret_cast<const uint4*>(x);
          const auto& activation =
              reinterpret_cast<const turbomind::Array<half, 8>&>(loaded);
#pragma unroll
          for (int i = 0; i < 8; ++i)
            sums[token] =
                fmaf(__half2float(activation[i]), values[i], sums[token]);
        }
      }
      __syncwarp();
      if (load_next) reinterpret_cast<uint64_t*>(raw[warp])[lane] = next_word;
      __syncwarp();
    }
#pragma unroll
    for (int token = 0; token < MaxTokens; ++token) {
      if (chunk + token < end) {
#pragma unroll
        for (int distance = 16; distance > 0; distance /= 2)
          sums[token] += __shfl_down_sync(0xffffffffU, sums[token], distance);
        if (lane == 0)
          output[int64_t{chunk + token} * n + row] =
              __float2half_rn(sums[token]);
      }
    }
  }
}

template <int Type, int MaxTokens>
void launch_grouped_gate_up(torch::Tensor gate, torch::Tensor up,
                            torch::Tensor input, torch::Tensor gate_weights,
                            torch::Tensor up_weights, torch::Tensor offsets,
                            torch::Tensor ids, cudaStream_t stream) {
  raw_grouped_gate_up_kernel<Type, MaxTokens>
      <<<dim3((gate.size(1) + 1) / 2, input.size(0)), 128, 0, stream>>>(
          reinterpret_cast<half*>(gate.data_ptr()),
          reinterpret_cast<half*>(up.data_ptr()),
          reinterpret_cast<const half*>(input.data_ptr()),
          gate_weights.data_ptr<uint8_t>(), up_weights.data_ptr<uint8_t>(),
          offsets.data_ptr<int>(), ids.data_ptr<int64_t>(), gate.size(1),
          input.size(1), gate_weights.size(2));
}

template <int Type>
void dispatch_grouped_gate_up(torch::Tensor gate, torch::Tensor up,
                              torch::Tensor input, torch::Tensor gate_weights,
                              torch::Tensor up_weights, torch::Tensor offsets,
                              torch::Tensor ids, int tokens,
                              cudaStream_t stream) {
#define GROUPED_CASE(MAX)                                                      \
  launch_grouped_gate_up<Type, MAX>(gate, up, input, gate_weights, up_weights, \
                                    offsets, ids, stream)
  if (tokens <= 1) {
    GROUPED_CASE(1);
  } else {
    GROUPED_CASE(2);
  }
#undef GROUPED_CASE
}
}  // namespace

// IDs and offsets describe expert-sorted top-k routing with distinct experts
// per original token. Consequently an expert receives at most R / top_k
// tokens. This is the same prepared-routing contract as grouped GEMM.
void gguf_lattice_raw_grouped_gate_up_sm70_out(
    torch::Tensor gate, torch::Tensor up, torch::Tensor input,
    torch::Tensor gate_weights, torch::Tensor up_weights, torch::Tensor offsets,
    torch::Tensor ids, int64_t source_type, int64_t top_k) {
  TORCH_CHECK(source_type == 18 || source_type == 21 || source_type == 22,
              "Raw grouped gate/up supports IQ3_XXS, IQ3_S and IQ2_S");
  TORCH_CHECK(
      input.is_cuda() && input.dim() == 2 &&
          input.scalar_type() == torch::kFloat16 && input.is_contiguous(),
      "Raw grouped gate/up requires a contiguous FP16 activation matrix");
  for (const auto& tensor : {gate, up}) {
    TORCH_CHECK(tensor.device() == input.device() && tensor.dim() == 2 &&
                    tensor.scalar_type() == torch::kFloat16 &&
                    tensor.is_contiguous() && tensor.size(0) == input.size(0),
                "Raw grouped gate/up output descriptor mismatch");
  }
  TORCH_CHECK(gate.sizes() == up.sizes(), "Gate/up output shapes differ");
  for (const auto& tensor : {gate_weights, up_weights}) {
    TORCH_CHECK(tensor.device() == input.device() && tensor.dim() == 3 &&
                    tensor.scalar_type() == torch::kUInt8 &&
                    tensor.is_contiguous() && tensor.size(1) == gate.size(1),
                "Raw grouped gate/up requires prepared [E,N,bytes] weights");
  }
  TORCH_CHECK(gate_weights.sizes() == up_weights.sizes() &&
                  offsets.device() == input.device() && offsets.dim() == 1 &&
                  offsets.scalar_type() == torch::kInt32 &&
                  offsets.is_contiguous() &&
                  offsets.numel() == gate_weights.size(0) + 1 &&
                  ids.device() == input.device() && ids.dim() == 1 &&
                  ids.scalar_type() == torch::kInt64 && ids.is_contiguous() &&
                  ids.numel() == input.size(0),
              "Raw grouped gate/up route descriptor mismatch");
  const int64_t routes = input.size(0), k = input.size(1), n = gate.size(1);
  const int bytes = source_type == 18 ? 98 : source_type == 21 ? 110 : 82;
  TORCH_CHECK(top_k > 0 && top_k <= 16 && routes % top_k == 0 &&
                  routes / top_k <= 32 && n > 0 && n <= INT_MAX && k > 0 &&
                  k <= INT_MAX && k % 256 == 0 && gate_weights.size(0) > 0 &&
                  gate_weights.size(2) == (k / 256 * bytes + 7) / 8 * 8,
              "Raw grouped gate/up dimensions exceed the small-batch contract");
  if (!routes) return;
  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (source_type == 18)
    dispatch_grouped_gate_up<18>(gate, up, input, gate_weights, up_weights,
                                 offsets, ids, routes / top_k, stream);
  else if (source_type == 21)
    dispatch_grouped_gate_up<21>(gate, up, input, gate_weights, up_weights,
                                 offsets, ids, routes / top_k, stream);
  else
    dispatch_grouped_gate_up<22>(gate, up, input, gate_weights, up_weights,
                                 offsets, ids, routes / top_k, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_lattice_compact_prefill_sm70_out(torch::Tensor out,
                                           torch::Tensor input,
                                           torch::Tensor weight,
                                           int64_t source_type,
                                           torch::Tensor partial,
                                           int64_t splits, int64_t row_tile) {
  validate_compact_io(out, input, weight);
  const c10::cuda::CUDAGuard guard(weight.device());
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  validate_compact(weight, source_type, n, k);
  TORCH_CHECK(m >= 512 && n % 32 == 0 && (row_tile == 64 || row_tile == 128) &&
                  splits >= 1 && splits <= k / 256 && int64_t{m} * n <= INT_MAX,
              "Compact prefill requires aligned output rows and valid split-K");
  if (splits > 1)
    TORCH_CHECK(partial.device() == weight.device() &&
                    partial.scalar_type() == torch::kFloat32 &&
                    partial.is_contiguous() &&
                    partial.numel() >= splits * int64_t{m} * n,
                "Compact prefill requires FP32 partial storage");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((n + 127) / 128, (m + row_tile - 1) / row_tile, splits);
#define COMPACT_PREFILL(TYPE, ROWS)                             \
  compact_prefill_kernel<TYPE, ROWS><<<grid, 256, 0, stream>>>( \
      reinterpret_cast<half*>(out.data_ptr()),                  \
      splits > 1 ? partial.data_ptr<float>() : nullptr,         \
      reinterpret_cast<const half*>(input.data_ptr()),          \
      weight.data_ptr<uint8_t>(), m, n, k, splits)
  if (source_type == 21) {
    if (row_tile == 64) {
      COMPACT_PREFILL(21, 64);
    } else {
      COMPACT_PREFILL(21, 128);
    }
  } else {
    if (row_tile == 64) {
      COMPACT_PREFILL(22, 64);
    } else {
      COMPACT_PREFILL(22, 128);
    }
  }
#undef COMPACT_PREFILL
  if (splits > 1)
    reduce_vec<<<(m * n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()), partial.data_ptr<float>(),
        m * n, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
