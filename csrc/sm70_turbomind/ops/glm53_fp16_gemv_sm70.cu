// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kGlm53K = 4096;
constexpr int kGlm53N = 6416;
constexpr int kLanesPerRow = 16;
constexpr int kChunkK = 512;
constexpr int kChunkCount = kGlm53K / kChunkK;
constexpr int kThreads = kLanesPerRow * kChunkCount;

template <int kBatch>
__global__ void glm53_fp16_gemv_sm70_kernel(half* __restrict__ output,
                                            const half* __restrict__ input,
                                            const half* __restrict__ weight) {
  const int thread = threadIdx.x;
  const int chunk = thread / kLanesPerRow;
  const int lane = thread & (kLanesPerRow - 1);
  const int row = blockIdx.x;

  // Each (chunk, lane) pair preserves cuBLAS's 32-element ascending FMA
  // chain. Shared memory then joins chunk 0..7 followed by lane 0..15.
  float chunk_sum[kBatch] = {};
#pragma unroll 4
  for (int tile = 0; tile < kChunkK / kLanesPerRow; ++tile) {
    const int k = chunk * kChunkK + tile * kLanesPerRow + lane;
    const half w = __ldg(weight + static_cast<size_t>(row) * kGlm53K + k);
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      const half x = __ldg(input + static_cast<size_t>(batch) * kGlm53K + k);
      chunk_sum[batch] =
          __fmaf_rn(__half2float(x), __half2float(w), chunk_sum[batch]);
    }
  }

  __shared__ float chunk_partials[kBatch][kThreads];
#pragma unroll
  for (int batch = 0; batch < kBatch; ++batch) {
    chunk_partials[batch][chunk * kLanesPerRow + lane] = chunk_sum[batch];
  }
  __syncthreads();

  if (chunk == 0) {
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      float lane_sum = chunk_partials[batch][lane];
#pragma unroll
      for (int source_chunk = 1; source_chunk < kChunkCount; ++source_chunk) {
        lane_sum = __fadd_rn(
            lane_sum,
            chunk_partials[batch][source_chunk * kLanesPerRow + lane]);
      }

      float row_sum = 0.0f;
#pragma unroll
      for (int source_lane = 0; source_lane < kLanesPerRow; ++source_lane) {
        const float value =
            __shfl_sync(0x0000ffffu, lane_sum, source_lane, kLanesPerRow);
        if (lane == 0) {
          row_sum = source_lane == 0 ? value : __fadd_rn(row_sum, value);
        }
      }
      if (lane == 0) {
        output[static_cast<size_t>(batch) * kGlm53N + row] =
            __float2half_rn(row_sum);
      }
    }
  }
}

__device__ __forceinline__ half2 glm53_load_weight_half2(const half2* address) {
  unsigned packed;
  asm("ld.global.cg.u32 %0, [%1];" : "=r"(packed) : "l"(address));
  return *reinterpret_cast<half2*>(&packed);
}

template <int kBatch, int kRowsPerBlock>
__global__ void glm53_fp16_gemv_half2_sm70_kernel(
    half* __restrict__ output, const half* __restrict__ input,
    const half* __restrict__ weight) {
  constexpr int kLanePairs = kLanesPerRow / 2;
  constexpr int kThreadsPerRow = kChunkCount * kLanePairs;
  constexpr int kThreads = kRowsPerBlock * kThreadsPerRow;
  static_assert(kThreads <= 1024);
  static_assert(kGlm53N % kRowsPerBlock == 0);

  const int row_in_block = threadIdx.x / kThreadsPerRow;
  const int row_thread = threadIdx.x % kThreadsPerRow;
  const int chunk = row_thread / kLanePairs;
  const int lane_pair = row_thread % kLanePairs;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;

  float2 chunk_sum[kBatch] = {};
#pragma unroll 1
  for (int tile = 0; tile < kChunkK / kLanesPerRow; ++tile) {
    const int k = chunk * kChunkK + tile * kLanesPerRow + lane_pair * 2;
    const half2 w = glm53_load_weight_half2(reinterpret_cast<const half2*>(
        weight + static_cast<size_t>(row) * kGlm53K + k));
    const float2 weight_value = __half22float2(w);
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      const half2 x = __ldg(reinterpret_cast<const half2*>(
          input + static_cast<size_t>(batch) * kGlm53K + k));
      const float2 input_value = __half22float2(x);
      chunk_sum[batch].x =
          __fmaf_rn(input_value.x, weight_value.x, chunk_sum[batch].x);
      chunk_sum[batch].y =
          __fmaf_rn(input_value.y, weight_value.y, chunk_sum[batch].y);
    }
  }

  __shared__ float tail_partials[kRowsPerBlock][kBatch]
                                [kChunkCount / 2 * kLanesPerRow];
  __shared__ float lane_partials[kRowsPerBlock][kBatch][kLanesPerRow];
  float2 head_sum[kBatch] = {};
#pragma unroll
  for (int batch = 0; batch < kBatch; ++batch) {
    if (chunk < kChunkCount / 2) {
      head_sum[batch] = chunk_sum[batch];
#pragma unroll
      for (int source_chunk = 1; source_chunk < kChunkCount / 2;
           ++source_chunk) {
        head_sum[batch].x =
            __fadd_rn(head_sum[batch].x,
                      __shfl_sync(0xffffffffu, chunk_sum[batch].x,
                                  source_chunk * kLanePairs + lane_pair));
        head_sum[batch].y =
            __fadd_rn(head_sum[batch].y,
                      __shfl_sync(0xffffffffu, chunk_sum[batch].y,
                                  source_chunk * kLanePairs + lane_pair));
      }
    } else {
      const int tail = (chunk - kChunkCount / 2) * kLanesPerRow + lane_pair * 2;
      tail_partials[row_in_block][batch][tail] = chunk_sum[batch].x;
      tail_partials[row_in_block][batch][tail + 1] = chunk_sum[batch].y;
    }
  }
  __syncthreads();

  if (chunk == 0) {
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      float even_sum = head_sum[batch].x;
      float odd_sum = head_sum[batch].y;
#pragma unroll
      for (int source_chunk = kChunkCount / 2; source_chunk < kChunkCount;
           ++source_chunk) {
        const int tail_chunk = source_chunk - kChunkCount / 2;
        even_sum = __fadd_rn(
            even_sum, tail_partials[row_in_block][batch]
                                   [tail_chunk * kLanesPerRow + lane_pair * 2]);
        odd_sum = __fadd_rn(
            odd_sum,
            tail_partials[row_in_block][batch]
                         [tail_chunk * kLanesPerRow + lane_pair * 2 + 1]);
      }

      lane_partials[row_in_block][batch][lane_pair * 2] = even_sum;
      lane_partials[row_in_block][batch][lane_pair * 2 + 1] = odd_sum;
    }
  }
  __syncthreads();

  if (row_thread == 0) {
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      float row_sum = lane_partials[row_in_block][batch][0];
#pragma unroll
      for (int source_lane = 1; source_lane < kLanesPerRow; ++source_lane) {
        row_sum =
            __fadd_rn(row_sum, lane_partials[row_in_block][batch][source_lane]);
      }
      output[static_cast<size_t>(batch) * kGlm53N + row] =
          __float2half_rn(row_sum);
    }
  }
}

template <int kBatch, int kRowsPerBlock>
__global__ void glm53_fp16_gemv_half2_broadcast_sm70_kernel(
    half* __restrict__ output, const half* __restrict__ input,
    const half* __restrict__ weight) {
  constexpr int kLanePairs = kLanesPerRow / 2;
  constexpr int kThreads = kRowsPerBlock * kChunkCount * kLanePairs;
  static_assert(kRowsPerBlock == 4);
  static_assert(kThreads == 256);

  // One warp owns one K chunk for four output rows. The first eight lanes
  // load each input half2 once, then broadcast it to the other three rows.
  const int warp_lane = threadIdx.x & 31;
  const int chunk = threadIdx.x / 32;
  const int row_in_block = warp_lane / kLanePairs;
  const int lane_pair = warp_lane % kLanePairs;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;

  float2 chunk_sum[kBatch] = {};
#pragma unroll 1
  for (int tile = 0; tile < kChunkK / kLanesPerRow; ++tile) {
    const int k = chunk * kChunkK + tile * kLanesPerRow + lane_pair * 2;
    const half2 w = glm53_load_weight_half2(reinterpret_cast<const half2*>(
        weight + static_cast<size_t>(row) * kGlm53K + k));
    const float2 weight_value = __half22float2(w);
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      unsigned input_packed = 0;
      if (row_in_block == 0) {
        input_packed = __ldg(reinterpret_cast<const unsigned*>(
            input + static_cast<size_t>(batch) * kGlm53K + k));
      }
      input_packed = __shfl_sync(0xffffffffu, input_packed, lane_pair);
      const half2 x = *reinterpret_cast<half2*>(&input_packed);
      const float2 input_value = __half22float2(x);
      chunk_sum[batch].x =
          __fmaf_rn(input_value.x, weight_value.x, chunk_sum[batch].x);
      chunk_sum[batch].y =
          __fmaf_rn(input_value.y, weight_value.y, chunk_sum[batch].y);
    }
  }

  __shared__ float chunk_partials[kRowsPerBlock][kBatch][kChunkCount]
                                 [kLanesPerRow];
#pragma unroll
  for (int batch = 0; batch < kBatch; ++batch) {
    chunk_partials[row_in_block][batch][chunk][lane_pair * 2] =
        chunk_sum[batch].x;
    chunk_partials[row_in_block][batch][chunk][lane_pair * 2 + 1] =
        chunk_sum[batch].y;
  }
  __syncthreads();

  if (chunk == 0) {
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      float even_sum = chunk_partials[row_in_block][batch][0][lane_pair * 2];
      float odd_sum = chunk_partials[row_in_block][batch][0][lane_pair * 2 + 1];
#pragma unroll
      for (int source_chunk = 1; source_chunk < kChunkCount; ++source_chunk) {
        even_sum = __fadd_rn(
            even_sum,
            chunk_partials[row_in_block][batch][source_chunk][lane_pair * 2]);
        odd_sum =
            __fadd_rn(odd_sum, chunk_partials[row_in_block][batch][source_chunk]
                                             [lane_pair * 2 + 1]);
      }

      float row_sum = 0.0f;
#pragma unroll
      for (int source_pair = 0; source_pair < kLanePairs; ++source_pair) {
        const float even_value =
            __shfl_sync(0xffffffffu, even_sum, source_pair, kLanePairs);
        const float odd_value =
            __shfl_sync(0xffffffffu, odd_sum, source_pair, kLanePairs);
        if (lane_pair == 0) {
          row_sum =
              source_pair == 0 ? even_value : __fadd_rn(row_sum, even_value);
          row_sum = __fadd_rn(row_sum, odd_value);
        }
      }
      if (lane_pair == 0) {
        output[static_cast<size_t>(batch) * kGlm53N + row] =
            __float2half_rn(row_sum);
      }
    }
  }
}

template <int kRowsPerBlock, bool kSwizzled>
__device__ __forceinline__ int glm53_staged_partial_index(int row, int chunk,
                                                          int lane) {
  if constexpr (kSwizzled) {
    return ((chunk * kRowsPerBlock + row) * kLanesPerRow + lane);
  }
  return ((row * kChunkCount + chunk) * kLanesPerRow + lane);
}

template <int kBatch, int kRowsPerBlock, bool kSwizzled,
          bool kBroadcastInput = true>
__global__ void glm53_fp16_gemv_half2_broadcast_staged_sm70_kernel(
    half* __restrict__ output, const half* __restrict__ input,
    const half* __restrict__ weight) {
  constexpr int kLanePairs = kLanesPerRow / 2;
  constexpr int kThreads = kRowsPerBlock * kChunkCount * kLanePairs;
  static_assert(kRowsPerBlock == 4);
  static_assert(kThreads == 256);

  const int warp_lane = threadIdx.x & 31;
  const int chunk = threadIdx.x / 32;
  const int row_in_block = warp_lane / kLanePairs;
  const int lane_pair = warp_lane % kLanePairs;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;

  float2 chunk_sum[kBatch] = {};
#pragma unroll 1
  for (int tile = 0; tile < kChunkK / kLanesPerRow; ++tile) {
    const int k = chunk * kChunkK + tile * kLanesPerRow + lane_pair * 2;
    const half2 w = glm53_load_weight_half2(reinterpret_cast<const half2*>(
        weight + static_cast<size_t>(row) * kGlm53K + k));
    const float2 weight_value = __half22float2(w);
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      unsigned input_packed = 0;
      if constexpr (!kBroadcastInput) {
        input_packed = __ldg(reinterpret_cast<const unsigned*>(
            input + static_cast<size_t>(batch) * kGlm53K + k));
      } else if (row_in_block == 0) {
        input_packed = __ldg(reinterpret_cast<const unsigned*>(
            input + static_cast<size_t>(batch) * kGlm53K + k));
      }
      if constexpr (kBroadcastInput) {
        input_packed = __shfl_sync(0xffffffffu, input_packed, lane_pair);
      }
      const half2 x = *reinterpret_cast<half2*>(&input_packed);
      const float2 input_value = __half22float2(x);
      chunk_sum[batch].x =
          __fmaf_rn(input_value.x, weight_value.x, chunk_sum[batch].x);
      chunk_sum[batch].y =
          __fmaf_rn(input_value.y, weight_value.y, chunk_sum[batch].y);
    }
  }

  __shared__ float chunk_partials[kRowsPerBlock * kChunkCount * kLanesPerRow];
#pragma unroll
  for (int batch = 0; batch < kBatch; ++batch) {
    chunk_partials[glm53_staged_partial_index<kRowsPerBlock, kSwizzled>(
        row_in_block, chunk, lane_pair * 2)] = chunk_sum[batch].x;
    chunk_partials[glm53_staged_partial_index<kRowsPerBlock, kSwizzled>(
        row_in_block, chunk, lane_pair * 2 + 1)] = chunk_sum[batch].y;
    __syncthreads();

    if (chunk == 0) {
      float even_sum =
          chunk_partials[glm53_staged_partial_index<kRowsPerBlock, kSwizzled>(
              row_in_block, 0, lane_pair * 2)];
      float odd_sum =
          chunk_partials[glm53_staged_partial_index<kRowsPerBlock, kSwizzled>(
              row_in_block, 0, lane_pair * 2 + 1)];
#pragma unroll
      for (int source_chunk = 1; source_chunk < kChunkCount; ++source_chunk) {
        even_sum = __fadd_rn(
            even_sum,
            chunk_partials[glm53_staged_partial_index<kRowsPerBlock, kSwizzled>(
                row_in_block, source_chunk, lane_pair * 2)]);
        odd_sum = __fadd_rn(
            odd_sum,
            chunk_partials[glm53_staged_partial_index<kRowsPerBlock, kSwizzled>(
                row_in_block, source_chunk, lane_pair * 2 + 1)]);
      }

      float row_sum = 0.0f;
#pragma unroll
      for (int source_pair = 0; source_pair < kLanePairs; ++source_pair) {
        const float even_value =
            __shfl_sync(0xffffffffu, even_sum, source_pair, kLanePairs);
        const float odd_value =
            __shfl_sync(0xffffffffu, odd_sum, source_pair, kLanePairs);
        if (lane_pair == 0) {
          row_sum =
              source_pair == 0 ? even_value : __fadd_rn(row_sum, even_value);
          row_sum = __fadd_rn(row_sum, odd_value);
        }
      }
      if (lane_pair == 0) {
        output[static_cast<size_t>(batch) * kGlm53N + row] =
            __float2half_rn(row_sum);
      }
    }
    if (batch + 1 < kBatch) {
      __syncthreads();
    }
  }
}

void validate_glm53_fp16_gemv_tensors(const torch::Tensor& output,
                                      const torch::Tensor& input,
                                      const torch::Tensor& weight) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda(),
              "sm70_glm53_fp16_gemv_out: tensors must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half &&
                  weight.scalar_type() == at::ScalarType::Half &&
                  output.scalar_type() == at::ScalarType::Half,
              "sm70_glm53_fp16_gemv_out: tensors must be float16");
  TORCH_CHECK(
      input.is_contiguous() && weight.is_contiguous() && output.is_contiguous(),
      "sm70_glm53_fp16_gemv_out: tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 8 &&
                  input.size(1) == kGlm53K,
              "sm70_glm53_fp16_gemv_out: input must be [M, 4096] with "
              "1 <= M <= 8");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kGlm53N &&
                  weight.size(1) == kGlm53K,
              "sm70_glm53_fp16_gemv_out: weight must be [6416, 4096]");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == input.size(0) &&
                  output.size(1) == kGlm53N,
              "sm70_glm53_fp16_gemv_out: output must be [M, 6416]");
  TORCH_CHECK(
      input.device() == weight.device() && input.device() == output.device(),
      "sm70_glm53_fp16_gemv_out: tensors must share a device");
}

}  // namespace

void sm70_glm53_fp16_gemv_out(torch::Tensor output, torch::Tensor input,
                              torch::Tensor weight) {
  validate_glm53_fp16_gemv_tensors(output, input, weight);
  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "sm70_glm53_fp16_gemv_out: requires SM70");
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(input.device().index());
  const char* variant_raw = std::getenv("VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS");
  const int variant = variant_raw == nullptr ? -5 : std::atoi(variant_raw);
  if (input.size(0) == 8 && variant != 0) {
    if (variant == -5) {
      glm53_fp16_gemv_half2_broadcast_staged_sm70_kernel<8, 4, true, false>
          <<<kGlm53N / 4, 256, 0, stream>>>(
              reinterpret_cast<half*>(output.data_ptr<at::Half>()),
              reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
              reinterpret_cast<const half*>(weight.data_ptr<at::Half>()));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    if (variant == -3) {
      glm53_fp16_gemv_half2_broadcast_staged_sm70_kernel<8, 4, true>
          <<<kGlm53N / 4, 256, 0, stream>>>(
              reinterpret_cast<half*>(output.data_ptr<at::Half>()),
              reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
              reinterpret_cast<const half*>(weight.data_ptr<at::Half>()));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    if (variant == -2) {
      glm53_fp16_gemv_half2_broadcast_staged_sm70_kernel<8, 4, false>
          <<<kGlm53N / 4, 256, 0, stream>>>(
              reinterpret_cast<half*>(output.data_ptr<at::Half>()),
              reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
              reinterpret_cast<const half*>(weight.data_ptr<at::Half>()));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    if (variant == -4) {
      glm53_fp16_gemv_half2_broadcast_sm70_kernel<8, 4>
          <<<kGlm53N / 4, 256, 0, stream>>>(
              reinterpret_cast<half*>(output.data_ptr<at::Half>()),
              reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
              reinterpret_cast<const half*>(weight.data_ptr<at::Half>()));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
#define VLLM_LAUNCH_GLM53_GEMV_HALF2(rows)                           \
  glm53_fp16_gemv_half2_sm70_kernel<8, rows>                         \
      <<<kGlm53N / rows, rows * 64, 0, stream>>>(                    \
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),      \
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()), \
          reinterpret_cast<const half*>(weight.data_ptr<at::Half>()))
    switch (variant) {
      case 1:
        VLLM_LAUNCH_GLM53_GEMV_HALF2(1);
        break;
      case 2:
        VLLM_LAUNCH_GLM53_GEMV_HALF2(2);
        break;
      case 4:
        VLLM_LAUNCH_GLM53_GEMV_HALF2(4);
        break;
      case 8:
        VLLM_LAUNCH_GLM53_GEMV_HALF2(8);
        break;
      case 16:
        VLLM_LAUNCH_GLM53_GEMV_HALF2(16);
        break;
      default:
        TORCH_CHECK(false,
                    "VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS must be one of "
                    "-5, -4, -3, -2, 0, 1, 2, 4, 8, or 16");
    }
#undef VLLM_LAUNCH_GLM53_GEMV_HALF2
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#define VLLM_LAUNCH_GLM53_GEMV(batch)                                     \
  case batch:                                                             \
    glm53_fp16_gemv_sm70_kernel<batch><<<kGlm53N, kThreads, 0, stream>>>( \
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),             \
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),        \
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()));      \
    break
  switch (input.size(0)) {
    VLLM_LAUNCH_GLM53_GEMV(1);
    VLLM_LAUNCH_GLM53_GEMV(2);
    VLLM_LAUNCH_GLM53_GEMV(3);
    VLLM_LAUNCH_GLM53_GEMV(4);
    VLLM_LAUNCH_GLM53_GEMV(5);
    VLLM_LAUNCH_GLM53_GEMV(6);
    VLLM_LAUNCH_GLM53_GEMV(7);
    VLLM_LAUNCH_GLM53_GEMV(8);
  }
#undef VLLM_LAUNCH_GLM53_GEMV
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef VLLM_GLM53_GEMV_STANDALONE
TORCH_LIBRARY_FRAGMENT(_C_glm53_gemv_bench, ops) {
  ops.def(
      "sm70_glm53_fp16_gemv_out(Tensor(a!) output, Tensor input, Tensor "
      "weight) -> ()");
  ops.impl("sm70_glm53_fp16_gemv_out", torch::kCUDA, &sm70_glm53_fp16_gemv_out);
}
#elif defined(VLLM_GLM53_GEMV_SIDECAR)
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "sm70_glm53_fp16_gemv_out(Tensor(a!) output, Tensor input, Tensor "
      "weight) -> ()");
  ops.impl("sm70_glm53_fp16_gemv_out", torch::kCUDA, &sm70_glm53_fp16_gemv_out);
}
#endif
