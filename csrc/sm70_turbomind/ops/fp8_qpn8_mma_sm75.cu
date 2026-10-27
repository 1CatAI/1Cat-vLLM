// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Block-FP8 (or MXFP8 row-group) GEMM on the QPN8 packed codes with Turing's
// m16n8k8 tensor-core MMA. Turing executes the Volta m8n8k4 MMA of the packed
// GEMMs at half its tensor rate, and above M=8 the block-FP8 route otherwise
// reconstructs the whole FP16 weight on every call. Here each CTA decodes and
// block-scales its codes once while staging them into shared memory, exactly as
// the dense reconstruction rounds them, and feeds both operands through
// ldmatrix. No second weight layout is held. Split-K across CTAs writes FP32
// partials that a second kernel sums in fixed split order.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "fp8_qpn8_codes_sm70.cuh"
#include "mma_fragments_sm70.cuh"

namespace {

using vllm::sm70::fp8x8_to_half2x4_fast;
using vllm::sm70::ldmatrix_x4;
using vllm::sm70::mma_m16n8k8;
using vllm::sm70::qpn8_col_from_lane;
using vllm::sm70::shared_address;

constexpr int kBlockN = 128;
constexpr int kBlockK = 32;
constexpr int kThreads = 256;
constexpr int kChunks = kBlockK / 8;  // 16-byte chunks per shared-memory row
// Row tiles per raster group: concurrent CTAs share these A rows and a few B
// column tiles, so both operands are reused from L2 instead of re-reading
// every weight once per row tile.
constexpr int kGroupRowTiles = 16;

// 64-byte rows: XOR the chunk with (row / 2) so the eight rows one ldmatrix
// phase reads land in distinct bank quads.
__device__ __forceinline__ int swizzled(int row, int chunk) {
  return row * kChunks + (chunk ^ ((row >> 1) & 3));
}

// RowGroupScales: MXFP8 scales laid out [K/32, N], one per column and K step
// (kBlockK == 32); off, the 128 x 128 block scales [K/128, N/32].
template <int BlockM, int WarpsM, int MinBlocks, bool RowGroupScales>
__global__ __launch_bounds__(kThreads, MinBlocks) void fp8_qpn8_mma_sm75_kernel(
    const half* __restrict__ input, const uint4* __restrict__ codes,
    const half* __restrict__ group_scales, half* __restrict__ output,
    float* __restrict__ partials, int m, int n, int k, int splits) {
  constexpr int WarpsN = kThreads / 32 / WarpsM;
  constexpr int WarpM = BlockM / WarpsM;
  constexpr int WarpN = kBlockN / WarpsN;
  constexpr int RowFrags = WarpM / 16;
  constexpr int ColFrags = WarpN / 8;
  constexpr int AChunks = BlockM * kChunks;
  constexpr int ALoads = (AChunks + kThreads - 1) / kThreads;
  static_assert(RowFrags >= 1 && ColFrags >= 2 && ColFrags % 2 == 0,
                "warp tile must hold whole m16 rows and n8 column pairs");

  __shared__ __align__(16) uint4 a_shared[2][BlockM * kChunks];
  __shared__ __align__(16) uint4 b_shared[2][kBlockN * kChunks];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp % WarpsM;
  const int warp_n = warp / WarpsM;

  const int row_tiles = (m + BlockM - 1) / BlockM;
  const int per_group = kGroupRowTiles * (n / kBlockN);
  const int first_row_tile = (blockIdx.x / per_group) * kGroupRowTiles;
  const int group_rows = min(row_tiles - first_row_tile, kGroupRowTiles);
  const int in_group = blockIdx.x % per_group;
  const int m0 = (first_row_tile + in_group % group_rows) * BlockM;
  const int n0 = (in_group / group_rows) * kBlockN;

  // Split-K boundaries stay on 128-deep scale blocks.
  const int k_blocks = k / 128;
  const int split = blockIdx.z;
  const int block_begin = split * k_blocks / splits;
  const int block_end = (split + 1) * k_blocks / splits;
  const int k_begin = block_begin * 128;
  const int steps = (block_end - block_begin) * (128 / kBlockK);

  // B staging: one 16-byte code word (16 K of one column) per thread.
  const int b_tile = tid >> 6;
  const int b_group = (tid >> 5) & 1;
  const int b_row = b_tile * 32 + qpn8_col_from_lane(lane);
  const uint4* b_source = codes +
                          (static_cast<size_t>(n0 / 32 + b_tile) * (k >> 4) +
                           k_begin / 16 + b_group) *
                              32 +
                          lane;
  const half* scale_source = RowGroupScales ? group_scales + n0 + b_row
                                            : group_scales + n0 / 32 + b_tile;
  const int tiles_n32 = n >> 5;

  // One K step held in registers on its way into shared memory. The scale is
  // read together with the codes, so no global load sits between the MMAs and
  // the shared-memory store. Two steps in flight measured slower for every
  // tile height (more registers, fewer resident blocks).
  struct Staged {
    uint4 a[ALoads];
    uint4 b;
    half scale;
  };

  auto load_global = [&](int step, Staged& staged) {
    const int k0 = k_begin + step * kBlockK;
#pragma unroll
    for (int i = 0; i < ALoads; ++i) {
      const int index = tid + i * kThreads;
      staged.a[i] = make_uint4(0, 0, 0, 0);
      if (index < AChunks && m0 + index / kChunks < m) {
        staged.a[i] = *reinterpret_cast<const uint4*>(
            input + static_cast<size_t>(m0 + index / kChunks) * k + k0 +
            (index % kChunks) * 8);
      }
    }
    staged.b =
        __ldcs(b_source + static_cast<size_t>(step) * (kBlockK / 16) * 32);
    staged.scale =
        RowGroupScales
            ? scale_source[static_cast<size_t>(k0 / kBlockK) * n]
            : scale_source[static_cast<size_t>(k0 / 128) * tiles_n32];
  };

  auto store_shared = [&](const Staged& staged, int stage) {
#pragma unroll
    for (int i = 0; i < ALoads; ++i) {
      const int index = tid + i * kThreads;
      if (index < AChunks) {
        a_shared[stage][swizzled(index / kChunks, index % kChunks)] =
            staged.a[i];
      }
    }
    const half2 scale2 = __halves2half2(staged.scale, staged.scale);
    half2 weights[8];
    fp8x8_to_half2x4_fast(make_uint2(staged.b.x, staged.b.y), weights);
    fp8x8_to_half2x4_fast(make_uint2(staged.b.z, staged.b.w), weights + 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      weights[i] = __hmul2(weights[i], scale2);
    }
    const uint4* packed = reinterpret_cast<const uint4*>(weights);
    b_shared[stage][swizzled(b_row, b_group * 2)] = packed[0];
    b_shared[stage][swizzled(b_row, b_group * 2 + 1)] = packed[1];
  };

  float accum[RowFrags][ColFrags][4];
#pragma unroll
  for (int i = 0; i < RowFrags; ++i) {
#pragma unroll
    for (int j = 0; j < ColFrags; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        accum[i][j][e] = 0.0f;
      }
    }
  }

  Staged next;
  load_global(0, next);
  store_shared(next, 0);
  __syncthreads();

  for (int step = 0; step < steps; ++step) {
    const int stage = step & 1;
    if (step + 1 < steps) {
      load_global(step + 1, next);
    }
#pragma unroll
    for (int half_k = 0; half_k < kBlockK / 16; ++half_k) {
      uint32_t a[RowFrags][4];
#pragma unroll
      for (int i = 0; i < RowFrags; ++i) {
        const int row = warp_m * WarpM + i * 16 + (lane & 15);
        const int chunk = half_k * 2 + (lane >> 4);
        ldmatrix_x4(a[i],
                    shared_address(&a_shared[stage][swizzled(row, chunk)]));
      }
      uint32_t b[ColFrags][2];
#pragma unroll
      for (int j = 0; j < ColFrags; j += 2) {
        const int row =
            warp_n * WarpN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
        const int chunk = half_k * 2 + ((lane >> 3) & 1);
        uint32_t r[4];
        ldmatrix_x4(r, shared_address(&b_shared[stage][swizzled(row, chunk)]));
        b[j][0] = r[0];
        b[j][1] = r[1];
        b[j + 1][0] = r[2];
        b[j + 1][1] = r[3];
      }
#pragma unroll
      for (int i = 0; i < RowFrags; ++i) {
#pragma unroll
        for (int j = 0; j < ColFrags; ++j) {
          mma_m16n8k8(accum[i][j], a[i][0], a[i][1], b[j][0]);
          mma_m16n8k8(accum[i][j], a[i][2], a[i][3], b[j][1]);
        }
      }
    }
    if (step + 1 < steps) {
      store_shared(next, stage ^ 1);
    }
    __syncthreads();
  }

  const int group = lane >> 2;
  const int pair = lane & 3;
#pragma unroll
  for (int i = 0; i < RowFrags; ++i) {
#pragma unroll
    for (int half_row = 0; half_row < 2; ++half_row) {
      const int row = m0 + warp_m * WarpM + i * 16 + group + half_row * 8;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (int j = 0; j < ColFrags; ++j) {
        const int col = n0 + warp_n * WarpN + j * 8 + pair * 2;
        const float x = accum[i][j][half_row * 2];
        const float y = accum[i][j][half_row * 2 + 1];
        if (splits == 1) {
          *reinterpret_cast<half2*>(output + static_cast<size_t>(row) * n +
                                    col) = __floats2half2_rn(x, y);
        } else {
          *reinterpret_cast<float2*>(
              partials + (static_cast<size_t>(split) * m + row) * n + col) =
              make_float2(x, y);
        }
      }
    }
  }
}

__global__ void fp8_qpn8_split_reduce_sm75_kernel(
    const float2* __restrict__ partials, half2* __restrict__ output, int pairs,
    int splits) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pairs) {
    return;
  }
  float2 sum = partials[index];
  for (int s = 1; s < splits; ++s) {
    const float2 value = partials[static_cast<size_t>(s) * pairs + index];
    sum.x += value.x;
    sum.y += value.y;
  }
  output[index] = __floats2half2_rn(sum.x, sum.y);
}

template <int BlockM, int WarpsM, int MinBlocks, bool RowGroupScales>
void launch_fp8_qpn8_mma_sm75(torch::Tensor& out, const torch::Tensor& input,
                              const torch::Tensor& codes,
                              const torch::Tensor& group_scales, int splits,
                              cudaStream_t stream) {
  const int m = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1));
  const int n = static_cast<int>(codes.size(1));
  torch::Tensor partials;
  if (splits > 1) {
    partials =
        torch::empty({splits, m, n}, input.options().dtype(torch::kFloat32));
  }
  const dim3 grid((n / kBlockN) * ((m + BlockM - 1) / BlockM), 1, splits);
  fp8_qpn8_mma_sm75_kernel<BlockM, WarpsM, MinBlocks, RowGroupScales>
      <<<grid, kThreads, 0, stream>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint4*>(codes.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(group_scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          splits > 1 ? partials.data_ptr<float>() : nullptr, m, n, k, splits);
  if (splits > 1) {
    const int pairs = m * n / 2;
    constexpr int kReduceThreads = 256;
    fp8_qpn8_split_reduce_sm75_kernel<<<(pairs + kReduceThreads - 1) /
                                            kReduceThreads,
                                        kReduceThreads, 0, stream>>>(
        reinterpret_cast<const float2*>(partials.data_ptr<float>()),
        reinterpret_cast<half2*>(out.data_ptr<at::Half>()), pairs, splits);
  }
}

// Resident CTAs per SM of one kernel instance. Every SM75 device has the same
// register file, shared memory and thread limits, so one answer per process
// holds for all of them.
template <int BlockM, int WarpsM, int MinBlocks>
int resident_ctas_per_sm() {
  static const int ctas = [] {
    int blocks = 0;
    AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, fp8_qpn8_mma_sm75_kernel<BlockM, WarpsM, MinBlocks, false>,
        kThreads, 0));
    return blocks;
  }();
  return ctas;
}

struct TileChoice {
  int block_m;
  int splits;
};

// Wave model: resident waves x K blocks per CTA x per-slot cost of one 128-deep
// K block, plus the split-K partials' traffic. Per-tile costs (relative to
// 16 rows) and the reduction weight were fitted on a Quadro RTX 8000 over the
// Qwen3.8-27B TP2 projections at M=1..512 (2026-10-20): the choice is 1.9 %
// behind the measured best tile and split on average, 15.7 % at worst. Ties
// keep fewer splits, then fewer rows.
TileChoice choose_tile(int64_t m, int64_t n, int64_t k, int sms) {
  constexpr int kBlockMs[] = {16, 32, 64, 128};
  constexpr double kSlotBlockCost[] = {1.0, 1.1, 1.5, 2.6};
  constexpr double kReductionCost = 2.4;
  const int resident[] = {
      resident_ctas_per_sm<16, 1, 2>(), resident_ctas_per_sm<32, 2, 2>(),
      resident_ctas_per_sm<64, 2, 2>(), resident_ctas_per_sm<128, 2, 2>()};
  const int64_t k_blocks = k / 128;
  const int64_t max_splits = std::min<int64_t>(16, k_blocks);
  TileChoice best{16, 1};
  double best_cost = -1.0;
  for (int splits = 1; splits <= max_splits; ++splits) {
    for (int i = 0; i < 4; ++i) {
      const int block_m = kBlockMs[i];
      if (block_m > std::max<int64_t>(16, 2 * m)) {
        continue;
      }
      const int64_t slots = int64_t{resident[i]} * sms;
      const int64_t ctas =
          (n / kBlockN) * ((m + block_m - 1) / block_m) * splits;
      const int64_t waves = (ctas + slots - 1) / slots;
      const int64_t blocks_per_cta = (k_blocks + splits - 1) / splits;
      double cost = static_cast<double>(waves * blocks_per_cta) *
                    kSlotBlockCost[i] * static_cast<double>(slots) /
                    resident[0];
      if (splits > 1) {
        cost += kReductionCost * (splits + 1) * static_cast<double>(m * n) /
                (128.0 * 128.0);
      }
      if (best_cost < 0.0 || cost < best_cost) {
        best_cost = cost;
        best = TileChoice{block_m, splits};
      }
    }
  }
  return best;
}

}  // namespace

void fp8_qpn8_mma_sm75_out(torch::Tensor out, torch::Tensor input,
                           torch::Tensor codes, torch::Tensor group_scales) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda() && codes.is_cuda() &&
                  group_scales.is_cuda(),
              "fp8_qpn8_mma_sm75_out: tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "fp8_qpn8_mma_sm75_out: input and output must be float16");
  TORCH_CHECK(codes.scalar_type() == torch::kUInt8 &&
                  group_scales.scalar_type() == torch::kFloat16,
              "fp8_qpn8_mma_sm75_out: expected uint8 codes and float16 "
              "group scales");
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2 && codes.dim() == 2 &&
                  group_scales.dim() == 2,
              "fp8_qpn8_mma_sm75_out: expected 2D tensors");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous() &&
                  codes.is_contiguous() && group_scales.is_contiguous(),
              "fp8_qpn8_mma_sm75_out: tensors must be contiguous");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = codes.size(1);
  TORCH_CHECK(codes.size(0) == k && out.size(0) == m && out.size(1) == n,
              "fp8_qpn8_mma_sm75_out: shape mismatch");
  TORCH_CHECK(n > 0 && n % kBlockN == 0 && k > 0 && k % 128 == 0,
              "fp8_qpn8_mma_sm75_out: requires N%128=K%128=0");
  const bool row_group_scales =
      group_scales.size(0) == k / 32 && group_scales.size(1) == n;
  TORCH_CHECK(row_group_scales || (group_scales.size(0) == k / 128 &&
                                   group_scales.size(1) == n / 32),
              "fp8_qpn8_mma_sm75_out: expected 128 x 128 block scales "
              "[K/128, N/32] or row-group scales [K/32, N]");
  TORCH_CHECK(m * n < (int64_t{1} << 31),
              "fp8_qpn8_mma_sm75_out: output too large");
  if (m == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 5,
              "fp8_qpn8_mma_sm75_out: requires an SM75 device");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const TileChoice tile = choose_tile(m, n, k, properties->multiProcessorCount);
  const int splits = tile.splits;

#define VLLM_LAUNCH_QPN8_MMA(BLOCK_M, WARPS_M, MIN_BLOCKS)           \
  do {                                                               \
    if (row_group_scales) {                                          \
      launch_fp8_qpn8_mma_sm75<BLOCK_M, WARPS_M, MIN_BLOCKS, true>(  \
          out, input, codes, group_scales, splits, stream);          \
    } else {                                                         \
      launch_fp8_qpn8_mma_sm75<BLOCK_M, WARPS_M, MIN_BLOCKS, false>( \
          out, input, codes, group_scales, splits, stream);          \
    }                                                                \
  } while (0)

  switch (tile.block_m) {
    case 16:
      VLLM_LAUNCH_QPN8_MMA(16, 1, 2);
      break;
    case 32:
      VLLM_LAUNCH_QPN8_MMA(32, 2, 2);
      break;
    case 64:
      VLLM_LAUNCH_QPN8_MMA(64, 2, 2);
      break;
    default:
      VLLM_LAUNCH_QPN8_MMA(128, 2, 2);
      break;
  }
#undef VLLM_LAUNCH_QPN8_MMA
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
