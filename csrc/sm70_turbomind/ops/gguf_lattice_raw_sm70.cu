// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include "gguf_lattice_raw.cuh"
#include "src/turbomind/kernels/gemm/arch/mma_sm70.h"

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
  const auto values = Decode::fragment(data, lane * 8, grid);
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
      const auto values = Decode::fragment(data, lane * 8, grid);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        decoded[row][lane * 8 + j] = __float2half_rn(values[j]);
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

template <int Type, bool Split>
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
  for (int block = begin; block < end; ++block) {
    const uint8_t* data = stage_block<Type>(
        raw[warp], weight + (int64_t)row * stride, block, stride);
    const auto values = Decode::fragment(data, lane * 8, grid);
    // K is block aligned and each lane owns eight adjacent half values.
    // One 128-bit load replaces eight strided 16-bit memory instructions.
    const uint4 loaded =
        *reinterpret_cast<const uint4*>(x + block * 256 + lane * 8);
    const auto& activation =
        reinterpret_cast<const turbomind::Array<half, 8>&>(loaded);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sum = fmaf(__half2float(activation[i]), values[i], sum);
    __syncwarp();
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

template <int Type, int NT, int MT>
__global__ void raw_mma_kernel(half* out, float* partial, const half* x,
                               const uint8_t* weight, int m, int n, int k,
                               int stride, int splits) {
  using Decode = vllm::sm70_gguf::LatticeRawDecoder<Type>;
  using MMA = turbomind::gemm::SM70_MMA_884;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  __shared__ __align__(16) uint8_t raw[NT][120];
  __shared__ float sums[4][MT][NT];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int col_begin = blockIdx.x * NT, row_begin = blockIdx.y * MT;
  const int bcol = lane / 16 * 4 + (lane & 12) * 2 + lane % 4;
  const int arow = lane / 16 * 4 + lane % 4;
  typename MMA::FragC accum[MT / 8]{};
  const int blocks = k / 256;
  const int begin = blocks * blockIdx.z / splits;
  const int end = blocks * (blockIdx.z + 1) / splits;
  for (int block = begin; block < end; ++block) {
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
#pragma unroll
    for (int step = 0; step < 64; step += 8) {
      const int base = warp * 64 + step;
      typename MMA::FragB b{};
      if (bcol < NT && col_begin + bcol < n) {
        const auto values = Decode::fragment(
            raw[bcol] + (block * Decode::kBlockBytes & 7), base, grid);
#pragma unroll
        for (int i = 0; i < 8; ++i) b[i] = __float2half_rn(values[i]);
      }
#pragma unroll
      for (int tile = 0; tile < MT / 8; ++tile) {
        typename MMA::FragA a{};
        const int row = row_begin + tile * 8 + arow;
        if (row < m) {
          *reinterpret_cast<uint4*>(&a) = *reinterpret_cast<const uint4*>(
              x + (int64_t)row * k + block * 256 + base);
        }
        MMA::fma(accum[tile], a, b, accum[tile]);
      }
    }
    __syncthreads();
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

template <int Type, int NT, int MT>
void launch_mma(torch::Tensor out, torch::Tensor input, torch::Tensor weight,
                torch::Tensor partial, int splits, cudaStream_t stream) {
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  const dim3 grid((n + NT - 1) / NT, (m + MT - 1) / MT, splits);
  raw_mma_kernel<Type, NT, MT><<<grid, 128, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      splits > 1 ? partial.data_ptr<float>() : nullptr,
      reinterpret_cast<const half*>(input.data_ptr()),
      weight.data_ptr<uint8_t>(), m, n, k, weight.size(1), splits);
}
void validate_raw(torch::Tensor w, int type, int64_t n, int64_t k) {
  TORCH_CHECK(type == 21 || type == 22, "Unsupported raw GGUF lattice type");
  const int block_bytes = type == 21 ? 110 : 82;
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
  validate_raw(weight, source_type, out.size(0), out.size(1));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (source_type == 21)
    launch_dequant<21>(out, weight, stream);
  else
    launch_dequant<22>(out, weight, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_lattice_raw_vec_sm70_out(torch::Tensor out, torch::Tensor input,
                                   torch::Tensor weight, int64_t source_type,
                                   torch::Tensor partial, int64_t splits) {
  TORCH_CHECK(input.device() == weight.device() &&
                  out.device() == weight.device() &&
                  input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
                  input.size(0) == 1 && out.dim() == 2 && out.size(0) == 1 &&
                  input.is_contiguous() && out.is_contiguous(),
              "Raw GGUF vector requires FP16 M=1");
  const c10::cuda::CUDAGuard guard(weight.device());
  const int n = out.size(1), k = input.size(1);
  validate_raw(weight, source_type, n, k);
  TORCH_CHECK(splits >= 1 && splits <= k / 256,
              "Invalid raw GGUF split-K count");
  if (splits > 1)
    TORCH_CHECK(partial.device() == weight.device() &&
                    partial.scalar_type() == torch::kFloat32 &&
                    partial.is_contiguous() && partial.numel() >= splits * n,
                "Raw GGUF split-K requires FP32 partial storage");
  const dim3 grid((n + 3) / 4, splits);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define RAW_VEC(TYPE)                                                       \
  if (splits == 1)                                                          \
    raw_vec_kernel<TYPE, false><<<grid, 128, 0, stream>>>(                  \
        reinterpret_cast<half*>(out.data_ptr()), nullptr,                   \
        reinterpret_cast<const half*>(input.data_ptr()),                    \
        weight.data_ptr<uint8_t>(), n, k, weight.size(1), splits);          \
  else                                                                      \
    raw_vec_kernel<TYPE, true><<<grid, 128, 0, stream>>>(                   \
        reinterpret_cast<half*>(out.data_ptr()), partial.data_ptr<float>(), \
        reinterpret_cast<const half*>(input.data_ptr()),                    \
        weight.data_ptr<uint8_t>(), n, k, weight.size(1), splits)
  if (source_type == 21) {
    RAW_VEC(21);
  } else {
    RAW_VEC(22);
  }
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
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  cublasMath_t saved;
  TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle, &saved));
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(
      handle, static_cast<cublasMath_t>(
                  CUBLAS_TENSOR_OP_MATH |
                  CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  const float alpha = 1.f, beta = 0.f;
  const auto status = cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, scratch.data_ptr(),
      CUDA_R_16F, n, input.data_ptr(), CUDA_R_16F, k, &beta, out.data_ptr(),
      CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  const auto restore = cublasSetMathMode(handle, saved);
  TORCH_CUDABLAS_CHECK(status);
  TORCH_CUDABLAS_CHECK(restore);
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
