// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// The M8 execution layout follows the in-tree QPN8 kernels, derived from
// dnv2003/v100-skinny (MIT). See LICENSE.v100-skinny for the retained notice.
// Weights stay FP16; the independent K partitions accumulate in FP32.
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/where.h>
#include <limits>
#include <climits>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#define MMA(C, A0, A1, B0, B1)                                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

namespace {

template <int Warps, int Tile>
__global__ void dflash2_fp16_m8_kernel(half* __restrict__ output,
                                       const half* __restrict__ input,
                                       const half* __restrict__ weight, int n,
                                       int k) {
  constexpr int Split = Warps * (Tile == 16 ? 2 : 1);
  __shared__ float partials[Split][8 * Tile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quad = (lane >> 2) & 3;
  const int split = warp * (Tile == 16 ? 2 : 1) + (Tile == 16 ? quad / 2 : 0);
  const int slot = Tile == 16 ? (lane & 7) | ((lane & 16) >> 1) : lane;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups = k / 16;
  const int count = groups / Split;
  const int begin = split * count;
  float accum[2][8] = {};
  uint4 w0, w1;
  auto load_weight = [&](int group, uint4& first, uint4& second) {
    const half* ptr =
        weight +
        ((static_cast<size_t>(blockIdx.x) * groups + group) * Tile + slot) * 16;
    first = __ldcs(reinterpret_cast<const uint4*>(ptr));
    second = __ldcs(reinterpret_cast<const uint4*>(ptr + 8));
  };
#pragma unroll 2
  for (int group = begin; group < begin + count; ++group) {
    load_weight(group, w0, w1);
    const half* x = input + static_cast<size_t>(row) * k + group * 16;
    const uint4 a0 = *reinterpret_cast<const uint4*>(x);
    const uint4 a1 = *reinterpret_cast<const uint4*>(x + 8);
    MMA(accum[0], a0.x, a0.y, w0.x, w0.y);
    MMA(accum[1], a0.z, a0.w, w0.z, w0.w);
    MMA(accum[0], a1.x, a1.y, w1.x, w1.y);
    MMA(accum[1], a1.z, a1.w, w1.z, w1.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int out_row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int out_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    const int tile_quad = Tile == 16 ? quad & 1 : quad;
    partials[split][out_row * Tile + tile_quad * 8 + out_col] =
        accum[0][i] + accum[1][i];
  }
  __syncthreads();
  for (int element = threadIdx.x; element < 8 * Tile; element += blockDim.x) {
    float value = 0;
#pragma unroll
    for (int part = 0; part < Split; ++part) value += partials[part][element];
    const int out_row = element / Tile;
    const int out_col = element % Tile;
    output[static_cast<size_t>(out_row) * n + blockIdx.x * Tile + out_col] =
        __float2half_rn(value);
  }
}

}  // namespace

namespace {

// Retain the dense head's FP32 partitions, including its final FP32 sum.
// Changing that schedule can change FP16 rounding at close logits.
template <int Partitions>
__global__ void fp16_candidate_scores_sm70_kernel(
    half* __restrict__ scores, const half* __restrict__ input,
    const half* __restrict__ weight, const int* __restrict__ ids,
    int vocabulary, int candidates) {
  constexpr int Width = 5120;
  constexpr int PrefetchGroups = 4;
  const int row = blockIdx.x;
  const int candidate = blockIdx.y;
  const int lane = threadIdx.x;
  const int token = ids[row * candidates + candidate];
  if (token < 0 || token >= vocabulary) {
    if (lane == 0) {
      scores[row * candidates + candidate] = __ushort_as_half(0x7e00);
    }
    return;
  }

  __shared__ __align__(16) half activation[Width];
  __shared__ __align__(16) half candidate_weight[Width];
  const half* weight_row = weight + static_cast<int64_t>(token) * Width;
  for (int vector = lane; vector < Width / 8; vector += 32) {
    reinterpret_cast<uint4*>(activation)[vector] =
        __ldg(reinterpret_cast<const uint4*>(input + row * Width) + vector);
    reinterpret_cast<uint4*>(candidate_weight)[vector] =
        __ldg(reinterpret_cast<const uint4*>(weight_row) + vector);
  }
  __syncwarp();

  float accumulator[Partitions][8] = {};
#pragma unroll
  for (int part = 0; part < Partitions; ++part) {
    for (int first = part * (Width / 16 / Partitions);
         first < (part + 1) * (Width / 16 / Partitions);
         first += PrefetchGroups) {
      uint4 weight_first[PrefetchGroups];
      uint4 weight_second[PrefetchGroups];
#pragma unroll
      for (int group = 0; group < PrefetchGroups; ++group) {
        weight_first[group] = make_uint4(0, 0, 0, 0);
        weight_second[group] = make_uint4(0, 0, 0, 0);
        if (lane == 0) {
          const half* address = candidate_weight + (first + group) * 16;
          weight_first[group] = *reinterpret_cast<const uint4*>(address);
          weight_second[group] = *reinterpret_cast<const uint4*>(address + 8);
        }
      }
#pragma unroll
      for (int group = 0; group < PrefetchGroups; ++group) {
        uint4 input_first = make_uint4(0, 0, 0, 0);
        uint4 input_second = make_uint4(0, 0, 0, 0);
        if ((lane & 19) == 0) {
          const half* address = activation + (first + group) * 16;
          input_first = *reinterpret_cast<const uint4*>(address);
          input_second = *reinterpret_cast<const uint4*>(address + 8);
        }
        MMA(accumulator[part], input_first.x, input_first.y,
            weight_first[group].x, weight_first[group].y);
        MMA(accumulator[part], input_first.z, input_first.w,
            weight_first[group].z, weight_first[group].w);
        MMA(accumulator[part], input_second.x, input_second.y,
            weight_second[group].x, weight_second[group].y);
        MMA(accumulator[part], input_second.z, input_second.w,
            weight_second[group].z, weight_second[group].w);
      }
    }
  }
  if (lane == 0) {
    float value = accumulator[0][0];
#pragma unroll
    for (int part = 1; part < Partitions; ++part) value += accumulator[part][0];
    scores[row * candidates + candidate] = __float2half_rn(value);
  }
}

}  // namespace

void sm70_fp16_candidate_scores_out(torch::Tensor scores, torch::Tensor input,
                                    torch::Tensor weight, torch::Tensor ids,
                                    bool tiled_multirow) {
  TORCH_CHECK(scores.is_cuda() && input.is_cuda() && weight.is_cuda() &&
              ids.is_cuda());
  TORCH_CHECK(
      scores.scalar_type() == at::kHalf && input.scalar_type() == at::kHalf &&
      weight.scalar_type() == at::kHalf && ids.scalar_type() == at::kInt);
  TORCH_CHECK(scores.is_contiguous() && input.is_contiguous() &&
              weight.is_contiguous() && ids.is_contiguous());
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && ids.dim() == 2 &&
              scores.dim() == 2);
  TORCH_CHECK(input.size(1) == 5120 && weight.size(1) == 5120 &&
              weight.size(0) > 0 && weight.size(0) <= INT_MAX);
  TORCH_CHECK(input.size(0) >= 1 && ids.size(0) == input.size(0) &&
              scores.sizes() == ids.sizes() && ids.size(1) >= 1 &&
              ids.size(1) <= 256);
  TORCH_CHECK(input.device() == weight.device() &&
              input.device() == scores.device() &&
              input.device() == ids.device());
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0);
  TORCH_CHECK(input.size(0) <= properties->maxGridSize[0]);
  if (!tiled_multirow && input.size(0) > 8) {
    // cuBLAS can select a different accumulator schedule for larger batches.
    // Keep the candidate scores identical to the caller's dense projection.
    auto dense = at::mm(input, weight.t());
    auto bounded_ids = ids.clamp(0, weight.size(0) - 1).to(torch::kInt64);
    scores.copy_(dense.gather(1, bounded_ids));
    scores.masked_fill_(ids.lt(0).logical_or(ids.ge(weight.size(0))),
                        std::numeric_limits<float>::quiet_NaN());
    return;
  }
  // The qualified wider shard's multirow cuBLAS projection uses two K
  // partitions; a single row and the narrower shard retain one chain.
  auto kernel = input.size(0) > 1 && weight.size(0) > 65536
                    ? fp16_candidate_scores_sm70_kernel<2>
                    : fp16_candidate_scores_sm70_kernel<1>;
  kernel<<<dim3(input.size(0), ids.size(1)), 32, 0,
           at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(scores.data_ptr()),
      reinterpret_cast<const half*>(input.data_ptr()),
      reinterpret_cast<const half*>(weight.data_ptr()), ids.data_ptr<int>(),
      weight.size(0), ids.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_dflash2_fp16_m8_out(torch::Tensor output, torch::Tensor input,
                              torch::Tensor packed, int64_t tile,
                              int64_t warps) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda() && packed.is_cuda());
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
              output.scalar_type() == at::kHalf &&
              packed.scalar_type() == at::kHalf);
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous() &&
              packed.is_contiguous());
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 8 && output.dim() == 2 &&
              output.size(0) == 8);
  TORCH_CHECK(input.device() == output.device() &&
              input.device() == packed.device());
  const int n = output.size(1), k = input.size(1);
  TORCH_CHECK((tile == 16 || tile == 32) && (warps == 8 || warps == 16));
  TORCH_CHECK(n % tile == 0 && k % (16 * warps * (tile == 16 ? 2 : 1)) == 0);
  TORCH_CHECK(packed.numel() == static_cast<int64_t>(n) * k);
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
#define LAUNCH(W, T)                                          \
  dflash2_fp16_m8_kernel<W, T><<<n / T, W * 32, 0, stream>>>( \
      reinterpret_cast<half*>(output.data_ptr()),             \
      reinterpret_cast<const half*>(input.data_ptr()),        \
      reinterpret_cast<const half*>(packed.data_ptr()), n, k)
  if (tile == 16 && warps == 16) {
    LAUNCH(16, 16);
  } else if (tile == 32 && warps == 8) {
    LAUNCH(8, 32);
  } else if (tile == 32 && warps == 16) {
    LAUNCH(16, 32);
  } else {
    TORCH_CHECK(false, "Unsupported DFlash2 FP16 M8 geometry");
  }
#undef LAUNCH
  AT_CUDA_CHECK(cudaGetLastError());
}

void sm70_dflash2_fp16_dispatch_out(torch::Tensor output, torch::Tensor input,
                                    torch::Tensor packed, torch::Tensor weight,
                                    int64_t tile, int64_t warps) {
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && output.dim() == 2);
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda());
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
              weight.scalar_type() == at::kHalf &&
              output.scalar_type() == at::kHalf);
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
              output.is_contiguous());
  TORCH_CHECK(input.device() == weight.device() &&
              input.device() == output.device());
  TORCH_CHECK(input.size(1) == weight.size(1) &&
              output.size(0) == input.size(0) &&
              output.size(1) == weight.size(0));
  if (input.size(0) == 8) {
    sm70_dflash2_fp16_m8_out(output, input, packed, tile, warps);
    return;
  }
  // F.linear with a 2D input and no bias uses this same matrix product.
  // Keep runtime M dispatch inside the op so a dynamic AOT prefill trace
  // does not erase the M8 decode route.
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  at::mm_out(output, input, weight.t());
}

namespace {
// Certified rows preserve their output storage. Other rows execute the
// complete original-weight projection with the qualified FP32 partitions.

template <int Prefetch, int Partitions>
__global__ __launch_bounds__(32, 16) void fp16_masked_head_sm70_kernel(
    half* scores, const half* input, const half* weight, const int* certified,
    int rows, int vocab) {
  constexpr int kRowsPerTile = 8;
  const int row_base = blockIdx.x * kRowsPerTile;
  input += static_cast<int64_t>(row_base) * 5120;
  scores += static_cast<int64_t>(row_base) * vocab;
  certified += row_base;
  rows = min(rows - row_base, kRowsPerTile);
  bool needed = false;
#pragma unroll
  for (int row = 0; row < 8; ++row) {
    if (row < rows && certified[row] == 0) needed = true;
  }
  if (!needed) return;
  int lane = threadIdx.x;
  int column = blockIdx.y * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) +
               ((lane & 16) ? 4 : 0);
  int input_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  bool active = input_row < rows && certified[input_row] == 0;
  const half* weight_row = weight + static_cast<int64_t>(column) * 5120;
  float accum[Partitions][8] = {};
#pragma unroll
  for (int part = 0; part < Partitions; ++part) {
    for (int first = part * (320 / Partitions);
         first < (part + 1) * (320 / Partitions); first += Prefetch) {
      uint4 first_weight[Prefetch], second_weight[Prefetch];
#pragma unroll
      for (int group = 0; group < Prefetch; ++group) {
        const half* address = weight_row + (first + group) * 16;
        first_weight[group] = __ldcs(reinterpret_cast<const uint4*>(address));
        second_weight[group] =
            __ldcs(reinterpret_cast<const uint4*>(address + 8));
      }
#pragma unroll
      for (int group = 0; group < Prefetch; ++group) {
        uint4 a0 = make_uint4(0, 0, 0, 0), a1 = make_uint4(0, 0, 0, 0);
        if (active) {
          const half* address = input + input_row * 5120 + (first + group) * 16;
          a0 = __ldg(reinterpret_cast<const uint4*>(address));
          a1 = __ldg(reinterpret_cast<const uint4*>(address + 8));
        }
        MMA(accum[part], a0.x, a0.y, first_weight[group].x,
            first_weight[group].y);
        MMA(accum[part], a0.z, a0.w, first_weight[group].z,
            first_weight[group].w);
        MMA(accum[part], a1.x, a1.y, second_weight[group].x,
            second_weight[group].y);
        MMA(accum[part], a1.z, a1.w, second_weight[group].z,
            second_weight[group].w);
      }
    }
  }
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    int row = (index & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    int column = blockIdx.y * 32 + ((lane >> 2) & 3) * 8 + (index & 1) +
                 (((lane >> 1) & 1) << 1) + ((index >> 2) << 2);
    if (row < rows && certified[row] == 0) {
      float value = accum[0][index];
#pragma unroll
      for (int part = 1; part < Partitions; ++part) value += accum[part][index];
      scores[static_cast<int64_t>(row) * vocab + column] =
          __float2half_rn(value);
    }
  }
}

}  // namespace

void sm70_fp16_masked_head_out(torch::Tensor scores, torch::Tensor input,
                               torch::Tensor weight, torch::Tensor certified,
                               bool tiled_multirow) {
  TORCH_CHECK(scores.is_cuda() && input.is_cuda() && weight.is_cuda() &&
              certified.is_cuda());
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 && input.size(1) == 5120);
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == 5120 &&
              weight.size(0) > 0 && weight.size(0) % 32 == 0);
  const int64_t vocab = weight.size(0);
  TORCH_CHECK(scores.dim() == 2 && scores.size(0) == input.size(0) &&
              scores.size(1) == vocab);
  TORCH_CHECK(certified.dim() == 1 && certified.numel() == input.size(0));
  TORCH_CHECK(
      input.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf &&
      scores.scalar_type() == at::kHalf && certified.scalar_type() == at::kInt);
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
              scores.is_contiguous() && certified.is_contiguous());
  TORCH_CHECK(input.device() == weight.device() &&
              input.device() == scores.device() &&
              input.device() == certified.device());
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0);
  if (!tiled_multirow && input.size(0) > 8) {
    auto dense = at::mm(input, weight.t());
    at::where_out(scores, certified.eq(0).unsqueeze(1), dense, scores);
    return;
  }
  const int64_t row_tiles = (input.size(0) + 7) / 8;
  TORCH_CHECK(row_tiles <= properties->maxGridSize[0]);
  TORCH_CHECK(vocab / 32 <= properties->maxGridSize[1]);
  auto kernel = input.size(0) > 1 && vocab > 65536
                    ? fp16_masked_head_sm70_kernel<8, 2>
                    : fp16_masked_head_sm70_kernel<8, 1>;
  kernel<<<dim3(row_tiles, vocab / 32), 32, 0,
           at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(scores.data_ptr()),
      reinterpret_cast<const half*>(input.data_ptr()),
      reinterpret_cast<const half*>(weight.data_ptr()),
      certified.data_ptr<int>(), input.size(0), vocab);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
