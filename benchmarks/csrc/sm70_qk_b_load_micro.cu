// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

namespace {

constexpr int kBlockM = 16;
constexpr int kBlockN = 128;
constexpr int kHeadDim = 256;
constexpr int kWmmaK = 16;
constexpr int kWarpsPerBlock = kBlockN / 16;
constexpr int kThreads = kWarpsPerBlock * 32;
constexpr uint32_t kFullWarpMask = 0xffffffff;

using AFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                                         16, 16, 16, __half,
                                         nvcuda::wmma::row_major>;
using BFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                         16, 16, 16, __half,
                                         nvcuda::wmma::col_major>;
using CFragment = nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                                         16, 16, 16, float>;

enum class BLoadPath : int { kNative, kDirect, kShuffle };

__device__ __forceinline__ void load_global_v4_u32(
    const __half* __restrict__ address, uint32_t* __restrict__ words) {
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]),
                 "=r"(words[3])
               : "l"(address)
               : "memory");
}

__device__ __forceinline__ int matrix_b_column_for_lane(int lane) {
  // The runtime fragment probe proves that lane ^ 4 owns the same native
  // col-major B fragment. Every unique lane owns this complete logical column.
  return (lane & 3) + 4 * ((lane >> 4) & 1) + 8 * ((lane >> 3) & 1);
}

template<BLoadPath kPath>
__device__ __forceinline__ void load_matrix_b(
    BFragment& fragment, const __half* __restrict__ tile, int token_stride,
    int lane) {
  if constexpr (kPath == BLoadPath::kNative) {
    nvcuda::wmma::load_matrix_sync(fragment, tile, token_stride);
    return;
  }

  const int column = matrix_b_column_for_lane(lane);
  const __half* column_base = tile + column * token_stride;
  uint32_t* words = reinterpret_cast<uint32_t*>(&fragment);

  if constexpr (kPath == BLoadPath::kDirect) {
    load_global_v4_u32(column_base, words);
    load_global_v4_u32(column_base + 8, words + 4);
    return;
  }

  if ((lane & 4) == 0) {
    load_global_v4_u32(column_base, words);
    load_global_v4_u32(column_base + 8, words + 4);
  } else {
#pragma unroll
    for (int word = 0; word < 8; ++word) {
      words[word] = 0;
    }
  }

  const int source_lane = lane & ~4;
#pragma unroll
  for (int word = 0; word < 8; ++word) {
    words[word] = __shfl_sync(kFullWarpMask, words[word], source_lane);
  }
}

template<BLoadPath kPath>
__device__ __forceinline__ void qk_b_load_body(
    const __half* __restrict__ query, const __half* __restrict__ key,
    float* __restrict__ output, __half* __restrict__ shared_query, int panels,
    int token_stride) {
  const int panel = blockIdx.x;
  if (panel >= panels) {
    return;
  }

  const int query_panel_offset = panel * kBlockM * kHeadDim;
  for (int index = threadIdx.x; index < kBlockM * kHeadDim;
       index += blockDim.x) {
    shared_query[index] = query[query_panel_offset + index];
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int tile_n = (threadIdx.x >> 5) * 16;
  const __half* key_tile =
      key + static_cast<int64_t>(panel) * kBlockN * token_stride
      + tile_n * token_stride;

  AFragment a_fragment;
  BFragment b_fragment;
  CFragment c_fragment;
  nvcuda::wmma::fill_fragment(c_fragment, 0.0f);

#pragma unroll
  for (int k_offset = 0; k_offset < kHeadDim; k_offset += kWmmaK) {
    nvcuda::wmma::load_matrix_sync(a_fragment, shared_query + k_offset,
                                   kHeadDim);
    load_matrix_b<kPath>(b_fragment, key_tile + k_offset, token_stride, lane);
    nvcuda::wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
  }

  float* output_tile =
      output + static_cast<int64_t>(panel) * kBlockM * kBlockN + tile_n;
  nvcuda::wmma::store_matrix_sync(output_tile, c_fragment, kBlockN,
                                  nvcuda::wmma::mem_row_major);
}

__global__ void qk_b_load_native_kernel(const __half* __restrict__ query,
                                         const __half* __restrict__ key,
                                         float* __restrict__ output,
                                         int panels,
                                         int token_stride) {
  __shared__ __align__(16) __half shared_query[kBlockM * kHeadDim];
  qk_b_load_body<BLoadPath::kNative>(query, key, output, shared_query, panels,
                                     token_stride);
}

__global__ void qk_b_load_direct_kernel(const __half* __restrict__ query,
                                         const __half* __restrict__ key,
                                         float* __restrict__ output,
                                         int panels,
                                         int token_stride) {
  __shared__ __align__(16) __half shared_query[kBlockM * kHeadDim];
  qk_b_load_body<BLoadPath::kDirect>(query, key, output, shared_query, panels,
                                     token_stride);
}

__global__ void qk_b_load_shuffle_kernel(const __half* __restrict__ query,
                                          const __half* __restrict__ key,
                                          float* __restrict__ output,
                                          int panels,
                                          int token_stride) {
  __shared__ __align__(16) __half shared_query[kBlockM * kHeadDim];
  qk_b_load_body<BLoadPath::kShuffle>(query, key, output, shared_query, panels,
                                      token_stride);
}

void check_inputs(const torch::Tensor& query, const torch::Tensor& key,
                  const torch::Tensor& output) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(query.scalar_type() == at::ScalarType::Half,
              "query must be fp16");
  TORCH_CHECK(key.scalar_type() == at::ScalarType::Half, "key must be fp16");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Float,
              "output must be fp32");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(query.dim() == 3 && query.size(1) == kBlockM
                  && query.size(2) == kHeadDim,
              "query must have shape [panels, 16, 256]");
  TORCH_CHECK(key.dim() == 3 && key.size(0) == query.size(0)
                  && key.size(1) == kBlockN && key.size(2) >= kHeadDim,
              "key must have shape [panels, 128, token_stride >= 256]");
  TORCH_CHECK(key.size(2) % 8 == 0,
              "token_stride must preserve 16-byte vector alignment");
  TORCH_CHECK(output.dim() == 3 && output.size(0) == query.size(0)
                  && output.size(1) == kBlockM && output.size(2) == kBlockN,
              "output must have shape [panels, 16, 128]");
  TORCH_CHECK(query.get_device() == key.get_device()
                  && query.get_device() == output.get_device(),
              "query, key, and output must use the same device");
}

template<BLoadPath kPath>
void launch(torch::Tensor query, torch::Tensor key, torch::Tensor output) {
  check_inputs(query, key, output);
  const c10::cuda::CUDAGuard device_guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
  const dim3 grid(query.size(0));
  const int token_stride = static_cast<int>(key.size(2));

  if constexpr (kPath == BLoadPath::kNative) {
    qk_b_load_native_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        output.data_ptr<float>(), query.size(0), token_stride);
  } else if constexpr (kPath == BLoadPath::kDirect) {
    qk_b_load_direct_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        output.data_ptr<float>(), query.size(0), token_stride);
  } else {
    qk_b_load_shuffle_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        output.data_ptr<float>(), query.size(0), token_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_native(torch::Tensor query, torch::Tensor key, torch::Tensor output) {
  launch<BLoadPath::kNative>(query, key, output);
}

void launch_direct(torch::Tensor query, torch::Tensor key, torch::Tensor output) {
  launch<BLoadPath::kDirect>(query, key, output);
}

void launch_shuffle(torch::Tensor query, torch::Tensor key, torch::Tensor output) {
  launch<BLoadPath::kShuffle>(query, key, output);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("native", &launch_native, "Native WMMA B-load panel");
  module.def("direct", &launch_direct, "Direct vector B-load panel");
  module.def("shuffle", &launch_shuffle, "Shuffled compact B-load panel");
}
