// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include "gguf_dp4a.cuh"

namespace {
using vllm::sm70_gguf::Q8_1;

__device__ __forceinline__ uint32_t u4_word(uint32_t nibbles) {
  return __byte_perm(nibbles & 0x0f0fU, (nibbles >> 4) & 0x0f0fU, 0x5140);
}

// Three normalized decoder families: signed i8/group16, unsigned u4 affine,
// and u4 integer LUT. Original Half d and integer small scales stay separate.
template <int Kind>
struct IntegerGroup {
  int words[8];
  float scale0, scale1, minimum;

  __device__ static IntegerGroup load(const int* codes, const half* d,
                                      const int8_t* scales, const half* dmin,
                                      const int8_t* mins, int n, int k, int col,
                                      int group) {
    IntegerGroup result;
    constexpr int Words = Kind == 0 ? 8 : 4;
    const int64_t packet =
        (int64_t{col / 32} * (k / 32 * Words) + group * Words) * 32 + col % 32;
#pragma unroll
    for (int i = 0; i < Words; ++i) {
      const uint32_t value = codes[packet + i * 32];
      if constexpr (Kind == 0)
        result.words[i] = value;
      else if constexpr (Kind == 1) {
        result.words[2 * i] = u4_word(value);
        result.words[2 * i + 1] = u4_word(value >> 16);
      } else {
        using Lut = turbomind::gemm::Transform_HMMA_SM70_Lut4<0>;
        result.words[2 * i] = Lut::iq_values(value) ^ 0x80808080U;
        result.words[2 * i + 1] = Lut::iq_values(value >> 16) ^ 0x80808080U;
      }
    }
    if constexpr (Kind == 0) {
      const int64_t index = int64_t{group * 2} * n + col;
      result.scale0 = __half2float(d[index]) * float(scales[index]);
      result.scale1 = __half2float(d[index + n]) * float(scales[index + n]);
      result.minimum = 0.f;
    } else {
      const int64_t index = int64_t{group} * n + col;
      result.scale0 = __half2float(d[index]) * float(scales[index]);
      result.scale1 = result.scale0;
      result.minimum =
          Kind == 1 ? __half2float(dmin[index]) * float(mins[index]) : 0.f;
    }
    return result;
  }

  __device__ float dot(const Q8_1& x) const {
    const int* values = reinterpret_cast<const int*>(x.qs);
    int a = 0, b = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) a = __dp4a(words[i], values[i], a);
#pragma unroll
    for (int i = 4; i < 8; ++i) b = __dp4a(words[i], values[i], b);
    float result = (float(a) * scale0 + float(b) * scale1) * __low2float(x.ds);
    if constexpr (Kind == 1) result -= minimum * __high2float(x.ds);
    return result;
  }
};

__device__ half finish(float gate, float up) {
  const float g = __half2float(__float2half_rn(gate));
  const half silu = __float2half_rn(g / (1.f + expf(-g)));
  return __hmul(silu, __float2half_rn(up));
}

template <bool Activated>
__global__ void reduce_dense(half* out, const float* partial, int m, int n,
                             int split, int out_stride) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = Activated ? n / 2 : n;
  if (col >= width) return;
  const int token = blockIdx.y;
  float gate = 0.f, up = 0.f;
  for (int p = 0; p < split; ++p) {
    gate += partial[(int64_t{p} * m + token) * n + col];
    if constexpr (Activated)
      up += partial[(int64_t{p} * m + token) * n + col + width];
  }
  out[int64_t{token} * out_stride + col] =
      Activated ? finish(gate, up) : __float2half_rn(gate);
}

template <int Kind, int TileM, bool Cooperative, bool Activated>
__global__ void dense(half* out, float* partial, const Q8_1* x,
                      const int* codes, const half* d, const int8_t* scales,
                      const half* dmin, const int8_t* mins, int m, int n, int k,
                      int split, int out_stride) {
  __shared__ float sums[4][TileM][32];
  extern __shared__ Q8_1 activations[];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int tiles_m = (m + TileM - 1) / TileM;
  const int tile_n = blockIdx.x / tiles_m, tile_m = blockIdx.x % tiles_m;
  const int col = tile_n * 32 + lane, begin_m = tile_m * TileM;
  const int groups = k / 32;
  const int begin = groups * blockIdx.y / split,
            end = groups * (blockIdx.y + 1) / split;
  // Coalesce the activation transfer once per CTA. Broadcast shared loads
  // feed every output column, avoiding repeated long-latency global loads
  // inside the five-token integer-dot loop.
  const int span = end - begin;
  const int words = TileM * span * (sizeof(Q8_1) / sizeof(int));
  auto* cached = reinterpret_cast<int*>(activations);
  const auto* source = reinterpret_cast<const int*>(x);
  for (int index = threadIdx.x; index < words; index += blockDim.x) {
    const int token = index / (span * 9), offset = index % (span * 9);
    cached[index] =
        begin_m + token < m
            ? source[((begin_m + token) * groups + begin) * 9 + offset]
            : 0;
  }
  __syncthreads();
  float values[TileM] = {};
  for (int group = begin + warp; group < end; group += 4) {
    const auto weight = IntegerGroup<Kind>::load(codes, d, scales, dmin, mins,
                                                 n, k, col, group);
#pragma unroll
    for (int t = 0; t < TileM; ++t)
      if (begin_m + t < m)
        values[t] += weight.dot(activations[t * span + group - begin]);
  }
#pragma unroll
  for (int t = 0; t < TileM; ++t) sums[warp][t][lane] = values[t];
  __syncthreads();
  if (!warp) {
#pragma unroll
    for (int t = 0; t < TileM; ++t) {
      if (begin_m + t < m) {
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) sum += sums[w][t][lane];
        if (!Activated && split == 1)
          out[int64_t{begin_m + t} * out_stride + col] = __float2half_rn(sum);
        else
          partial[(int64_t{blockIdx.y} * m + begin_m + t) * n + col] = sum;
      }
    }
  }
  if constexpr (Cooperative) {
    cooperative_groups::this_grid().sync();
    if (!blockIdx.y && !warp && (!Activated || col < n / 2)) {
#pragma unroll
      for (int t = 0; t < TileM; ++t) {
        if (begin_m + t < m) {
          float gate = 0.f, up = 0.f;
          for (int p = 0; p < split; ++p) {
            gate += partial[(int64_t{p} * m + begin_m + t) * n + col];
            if constexpr (Activated)
              up += partial[(int64_t{p} * m + begin_m + t) * n + col + n / 2];
          }
          out[int64_t{begin_m + t} * out_stride + col] =
              Activated ? finish(gate, up) : __float2half_rn(gate);
        }
      }
    }
  }
}

template <int Kind, int TileM, bool Activated>
void launch(torch::Tensor out, torch::Tensor partial, torch::Tensor x,
            torch::Tensor codes, torch::Tensor d, torch::Tensor scales,
            torch::Tensor dmin, torch::Tensor mins, int split,
            bool cooperative) {
  int m = x.size(0), n = codes.size(0) * 32, k = x.size(1) * 32,
      stride = out.stride(0);
  dim3 grid(n / 32 * ((m + TileM - 1) / TileM), split);
  const int shared_bytes =
      TileM * ((k / 32 + split - 1) / split) * sizeof(Q8_1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto output = reinterpret_cast<half*>(out.data_ptr());
  auto scratch = partial.data_ptr<float>();
  auto activation = reinterpret_cast<const Q8_1*>(x.data_ptr());
  auto weight = codes.data_ptr<int>();
  auto ds = reinterpret_cast<const half*>(d.data_ptr());
  auto sc = scales.data_ptr<int8_t>();
  auto dm = reinterpret_cast<const half*>(dmin.data_ptr());
  auto mn = mins.data_ptr<int8_t>();
  if (cooperative && (split > 1 || Activated)) {
    auto kernel = dense<Kind, TileM, true, Activated>;
    int blocks;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, 128, shared_bytes));
    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        props->cooperativeLaunch &&
            grid.x * grid.y <= blocks * props->multiProcessorCount,
        "Cooperative integer GEMM grid exceeds resident block capacity");
    void* arguments[] = {&output, &scratch, &activation, &weight, &ds,
                         &sc,     &dm,      &mn,         &m,      &n,
                         &k,      &split,   &stride};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel),
                                               grid, dim3(128), arguments,
                                               shared_bytes, stream));
  } else {
    dense<Kind, TileM, false, Activated><<<grid, 128, shared_bytes, stream>>>(
        output, scratch, activation, weight, ds, sc, dm, mn, m, n, k, split,
        stride);
    if (split > 1 || Activated)
      reduce_dense<Activated>
          <<<dim3(((Activated ? n / 2 : n) + 127) / 128, m), 128, 0, stream>>>(
              output, scratch, m, n, split, stride);
  }
}

template <int Kind>
void dispatch(torch::Tensor out, torch::Tensor partial, torch::Tensor x,
              torch::Tensor codes, torch::Tensor d, torch::Tensor scales,
              torch::Tensor dmin, torch::Tensor mins, int split,
              bool cooperative, bool activated) {
  if (x.size(0) == 1) {
    if (activated)
      launch<Kind, 1, true>(out, partial, x, codes, d, scales, dmin, mins,
                            split, cooperative);
    else
      launch<Kind, 1, false>(out, partial, x, codes, d, scales, dmin, mins,
                             split, cooperative);
  } else {
    if (activated)
      launch<Kind, 5, true>(out, partial, x, codes, d, scales, dmin, mins,
                            split, cooperative);
    else
      launch<Kind, 5, false>(out, partial, x, codes, d, scales, dmin, mins,
                             split, cooperative);
  }
}
}  // namespace

void gguf_dp4a_dense_sm70_out(torch::Tensor out, torch::Tensor partial,
                              torch::Tensor x, torch::Tensor codes,
                              torch::Tensor d, torch::Tensor scales,
                              torch::Tensor dmin, torch::Tensor mins,
                              int64_t source_type, int64_t split,
                              bool cooperative, bool activated) {
  TORCH_CHECK(source_type == 12 || source_type == 14 || source_type == 23,
              "Unsupported integer-dot storage family");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kUInt8 && x.dim() == 3 &&
                  x.size(2) == sizeof(Q8_1) && x.is_contiguous() &&
                  (x.size(0) == 1 || x.size(0) == 5 || x.size(0) == 20) &&
                  x.size(1) > 0 && x.size(1) % 8 == 0,
              "Expected calibrated Q8_1 [M,K/32,36]");
  const int m = x.size(0), k = x.size(1) * 32;
  for (const auto& t : {out, partial, codes, d, scales, dmin, mins})
    TORCH_CHECK(t.device() == x.device(),
                "Integer GEMM tensors must share a CUDA device");
  const int group = source_type == 14 ? 16 : 32;
  TORCH_CHECK(codes.scalar_type() == torch::kInt32 && codes.dim() == 3 &&
                  codes.is_contiguous() && codes.size(0) > 0 &&
                  codes.size(2) == 32 &&
                  codes.size(1) == k / (source_type == 14 ? 4 : 8),
              "Invalid N32 integer packets");
  const int n = codes.size(0) * 32;
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.dim() == 2 &&
                  out.size(0) == m && out.size(1) == (activated ? n / 2 : n) &&
                  out.stride(1) == 1 && out.stride(0) >= out.size(1) &&
                  (!activated || n % 64 == 0),
              "Invalid integer GEMM output view");
  TORCH_CHECK(d.scalar_type() == torch::kFloat16 && d.is_contiguous() &&
                  d.dim() == 2 && d.size(0) == k / group && d.size(1) == n &&
                  scales.scalar_type() == torch::kInt8 &&
                  scales.is_contiguous() && scales.sizes() == d.sizes(),
              "Invalid original scale levels");
  TORCH_CHECK(dmin.scalar_type() == torch::kFloat16 && dmin.is_contiguous() &&
                  mins.scalar_type() == torch::kInt8 && mins.is_contiguous() &&
                  (source_type != 12 ||
                   (dmin.sizes() == d.sizes() && mins.sizes() == d.sizes())),
              "Invalid affine minimum levels");
  TORCH_CHECK(split >= 1 && split <= 16 && split <= x.size(1) &&
                  partial.scalar_type() == torch::kFloat32 &&
                  partial.is_contiguous() && partial.numel() == split * m * n,
              "Invalid FP32 split-K scratch");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0,
              "Integer GEMM requires SM70");
  if (source_type == 14)
    dispatch<0>(out, partial, x, codes, d, scales, dmin, mins, split,
                cooperative, activated);
  else if (source_type == 12)
    dispatch<1>(out, partial, x, codes, d, scales, dmin, mins, split,
                cooperative, activated);
  else
    dispatch<2>(out, partial, x, codes, d, scales, dmin, mins, split,
                cooperative, activated);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
