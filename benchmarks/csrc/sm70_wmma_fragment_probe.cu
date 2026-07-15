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

constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr int kMaxStride = 32;
constexpr int kFragmentWords = 8;

__device__ __forceinline__ int swizzled_row_slot(int row) {
  return (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1);
}

__global__ void dump_matrix_a_fragment_kernel(
    const __half* __restrict__ input,
    int32_t* __restrict__ output,
    int stride) {
  __shared__ __align__(16) __half smem[kRows * kMaxStride];

  for (int index = threadIdx.x; index < kRows * kCols;
       index += blockDim.x) {
    const int row = index / kCols;
    const int col = index - row * kCols;
    smem[row * stride + col] = input[index];
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                           __half, nvcuda::wmma::row_major>
        fragment;
    nvcuda::wmma::load_matrix_sync(fragment, smem, stride);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&fragment);
#pragma unroll
    for (int word = 0; word < kFragmentWords; ++word) {
      output[threadIdx.x * kFragmentWords + word] =
          static_cast<int32_t>(words[word]);
    }
  }
}

__global__ void compare_swizzled_matrix_a_fragment_kernel(
    const __half* __restrict__ input,
    int32_t* __restrict__ reference_output,
    int32_t* __restrict__ swizzled_output) {
  __shared__ __align__(16) __half reference[kRows * kCols];
  __shared__ __align__(16) __half swizzled[kRows * kCols];

  for (int index = threadIdx.x; index < kRows * kCols;
       index += blockDim.x) {
    const int row = index / kCols;
    const int col = index - row * kCols;
    reference[index] = input[index];
    const int plane = col >> 3;
    const int inner = col & 7;
    const int slot = swizzled_row_slot(row);
    swizzled[plane * 128 + slot * 8 + inner] = input[index];
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    using Fragment =
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __half, nvcuda::wmma::row_major>;
    Fragment reference_fragment;
    Fragment swizzled_fragment;
    nvcuda::wmma::load_matrix_sync(reference_fragment, reference, kCols);

    const int lane = threadIdx.x;
    const int row =
        (lane & 3) + ((lane >> 4) & 1) * 4 + ((lane >> 2) & 1) * 8;
    const int slot = swizzled_row_slot(row);
    const uint4 low =
        *reinterpret_cast<const uint4*>(swizzled + slot * 8);
    const uint4 high =
        *reinterpret_cast<const uint4*>(swizzled + 128 + slot * 8);
    uint32_t* swizzled_words =
        reinterpret_cast<uint32_t*>(&swizzled_fragment);
    swizzled_words[0] = low.x;
    swizzled_words[1] = low.y;
    swizzled_words[2] = low.z;
    swizzled_words[3] = low.w;
    swizzled_words[4] = high.x;
    swizzled_words[5] = high.y;
    swizzled_words[6] = high.z;
    swizzled_words[7] = high.w;

    const uint32_t* reference_words =
        reinterpret_cast<const uint32_t*>(&reference_fragment);
#pragma unroll
    for (int word = 0; word < kFragmentWords; ++word) {
      const int output_index = threadIdx.x * kFragmentWords + word;
      reference_output[output_index] =
          static_cast<int32_t>(reference_words[word]);
      swizzled_output[output_index] =
          static_cast<int32_t>(swizzled_words[word]);
    }
  }
}

template<typename Layout>
__global__ void dump_matrix_b_fragment_kernel(
    const __half* __restrict__ input,
    int32_t* __restrict__ output) {
  if (threadIdx.x < 32) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                           __half, Layout>
        fragment;
    nvcuda::wmma::load_matrix_sync(fragment, input, kCols);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&fragment);
#pragma unroll
    for (int word = 0; word < kFragmentWords; ++word) {
      output[threadIdx.x * kFragmentWords + word] =
          static_cast<int32_t>(words[word]);
    }
  }
}

__device__ __forceinline__ void load_global_v4_u32(
    const __half* __restrict__ address,
    uint32_t* __restrict__ words) {
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[0]), "=r"(words[1]),
                 "=r"(words[2]), "=r"(words[3])
               : "l"(address)
               : "memory");
}

__global__ void compare_compact_matrix_b_col_fragment_kernel(
    const __half* __restrict__ input,
    int32_t* __restrict__ reference_output,
    int32_t* __restrict__ compact_output) {
  if (threadIdx.x < 32) {
    using Fragment =
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __half, nvcuda::wmma::col_major>;
    Fragment reference_fragment;
    Fragment compact_fragment;
    nvcuda::wmma::load_matrix_sync(reference_fragment, input, kCols);

    const int lane = threadIdx.x;
    const int column =
        (lane & 3) + 4 * ((lane >> 4) & 1) + 8 * ((lane >> 3) & 1);
    const __half* column_base = input + column * kRows;
    uint32_t* compact_words = reinterpret_cast<uint32_t*>(&compact_fragment);
    load_global_v4_u32(column_base, compact_words);
    load_global_v4_u32(column_base + 8, compact_words + 4);

    const uint32_t* reference_words =
        reinterpret_cast<const uint32_t*>(&reference_fragment);
#pragma unroll
    for (int word = 0; word < kFragmentWords; ++word) {
      const int output_index = threadIdx.x * kFragmentWords + word;
      reference_output[output_index] =
          static_cast<int32_t>(reference_words[word]);
      compact_output[output_index] =
          static_cast<int32_t>(compact_words[word]);
    }
  }
}

torch::Tensor dump_matrix_a_fragment(torch::Tensor input, int64_t stride) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half,
              "input must be fp16");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.numel() == kRows * kCols,
              "input must contain one 16x16 tile");
  TORCH_CHECK(stride >= kCols && stride <= kMaxStride && stride % 8 == 0,
              "stride must be 16, 24, or 32");

  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty(
      {32, kFragmentWords},
      torch::TensorOptions().device(input.device()).dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  dump_matrix_a_fragment_kernel<<<1, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      output.data_ptr<int32_t>(), static_cast<int>(stride));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> compare_swizzled_matrix_a_fragment(
    torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half,
              "input must be fp16");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.numel() == kRows * kCols,
              "input must contain one 16x16 tile");

  const c10::cuda::CUDAGuard device_guard(input.device());
  auto options =
      torch::TensorOptions().device(input.device()).dtype(torch::kInt32);
  auto reference_output = torch::empty({32, kFragmentWords}, options);
  auto swizzled_output = torch::empty({32, kFragmentWords}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  compare_swizzled_matrix_a_fragment_kernel<<<1, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reference_output.data_ptr<int32_t>(),
      swizzled_output.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {reference_output, swizzled_output};
}

torch::Tensor dump_matrix_b_fragment(torch::Tensor input, bool col_major) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half,
              "input must be fp16");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.numel() == kRows * kCols,
              "input must contain one 16x16 tile");

  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty(
      {32, kFragmentWords},
      torch::TensorOptions().device(input.device()).dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (col_major) {
    dump_matrix_b_fragment_kernel<nvcuda::wmma::col_major><<<1, 32, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        output.data_ptr<int32_t>());
  } else {
    dump_matrix_b_fragment_kernel<nvcuda::wmma::row_major><<<1, 32, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        output.data_ptr<int32_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> compare_compact_matrix_b_col_fragment(
    torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half,
              "input must be fp16");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.numel() == kRows * kCols,
              "input must contain one 16x16 tile");

  const c10::cuda::CUDAGuard device_guard(input.device());
  auto options =
      torch::TensorOptions().device(input.device()).dtype(torch::kInt32);
  auto reference_output = torch::empty({32, kFragmentWords}, options);
  auto compact_output = torch::empty({32, kFragmentWords}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  compare_compact_matrix_b_col_fragment_kernel<<<1, 32, 0, stream>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reference_output.data_ptr<int32_t>(), compact_output.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {reference_output, compact_output};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("dump_matrix_a_fragment", &dump_matrix_a_fragment);
  module.def("compare_swizzled_matrix_a_fragment",
             &compare_swizzled_matrix_a_fragment);
  module.def("dump_matrix_b_fragment", &dump_matrix_b_fragment);
  module.def("compare_compact_matrix_b_col_fragment",
             &compare_compact_matrix_b_col_fragment);
}
