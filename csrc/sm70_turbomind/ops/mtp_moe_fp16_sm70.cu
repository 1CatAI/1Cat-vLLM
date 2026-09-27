// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
// Preserve the tuned BM2 Triton kernel's sequential FP32 FMA order and
// FP16 projection boundaries. In particular, W2 applies the router weight
// before the FP16 store; moe_sum still reduces the ten stored FP16 routes.
template <int N, int K, int BN, int BK, bool Weighted>
__global__ void mtp_moe_fp16_tile_kernel(const half* x, const half* w,
                                         const int32_t* ids,
                                         const float* weights,
                                         const int32_t* padded, half* y) {
  __shared__ half tile[BN][BK + 2];
  __shared__ float input[BK];
  const int t = threadIdx.x, route = blockIdx.y;
  if (route * 2 >= *padded) return;
  const int row = Weighted ? route : route / 10;
  const int expert = ids[route], col = blockIdx.x * BN + t;
  if (expert < 0 || expert >= 512) {
    if (t < BN && col < N) y[route * N + col] = __float2half(0.0f);
    return;
  }
  float acc = 0.0f;
  for (int base = 0; base < K; base += BK) {
    for (int i = t; i < BK; i += 128)
      input[i] = base + i < K ? __half2float(x[row * K + base + i]) : 0.0f;
    // Coalesced 128-bit loads followed by a bank-padded shared layout.
    for (int i = t; i < BN * BK / 8; i += 128) {
      const int n = i / (BK / 8), k = i % (BK / 8) * 8;
      union {
        uint4 v;
        half h[8];
      } data;
      data.v = make_uint4(0, 0, 0, 0);
      if (blockIdx.x * BN + n < N && base + k < K)
        data.v = *reinterpret_cast<const uint4*>(
            w + (int64_t(expert) * N + blockIdx.x * BN + n) * K + base + k);
#pragma unroll
      for (int j = 0; j < 8; ++j) tile[n][k + j] = data.h[j];
    }
    __syncthreads();
    if (t < BN) {
#pragma unroll
      for (int k = 0; k < BK; ++k)
        acc = __fmaf_rn(input[k], __half2float(tile[t][k]), acc);
    }
    __syncthreads();
  }
  if (t < BN && col < N) {
    if (Weighted) acc = __fmul_rn(acc, weights[route]);
    y[route * N + col] = __float2half_rn(acc);
  }
}

__global__ void mtp_moe_fp16_m1_w13_kernel(const half* x, const half* w,
                                           const int32_t* ids,
                                           const int32_t* padded, half* y) {
  constexpr int kN = 320, kK = 2560, kThreads = 64;
  __shared__ float input[kK];
  const int t = threadIdx.x, route = blockIdx.y;
  if (route * 2 >= *padded) return;
  const int expert = ids[route], col = blockIdx.x * kThreads + t;
  for (int k = t; k < kK; k += kThreads) input[k] = __half2float(x[k]);
  __syncthreads();
  float acc = 0.0f;
  if (expert >= 0 && expert < 512) {
    const half* p = w + (int64_t(expert) * kN + col) * kK;
    for (int k = 0; k < kK; k += 8) {
      union {
        uint4 v;
        half h[8];
      } data;
      data.v = *reinterpret_cast<const uint4*>(p + k);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc = __fmaf_rn(input[k + j], __half2float(data.h[j]), acc);
    }
  }
  y[route * kN + col] = __float2half_rn(acc);
}

void mtp_moe_fp16_out(torch::Tensor out, torch::Tensor x, torch::Tensor w,
                      torch::Tensor ids, torch::Tensor weights,
                      torch::Tensor padded, bool down) {
  TORCH_CHECK(x.is_cuda(), "SM70 MTP MoE requires CUDA tensors");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "MTP FP16 MoE is SM70 only");
  for (const auto& tensor : {out, x, w, ids, weights, padded}) {
    TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(),
                "MTP FP16 MoE requires same-device contiguous tensors");
  }
  TORCH_CHECK(out.scalar_type() == at::kHalf && x.scalar_type() == at::kHalf &&
                  w.scalar_type() == at::kHalf &&
                  ids.scalar_type() == at::kInt &&
                  weights.scalar_type() == at::kFloat &&
                  padded.scalar_type() == at::kInt,
              "Invalid MTP FP16 MoE dtypes");
  TORCH_CHECK(weights.dim() == 2 && weights.size(1) == 10 &&
                  (weights.size(0) == 1 || weights.size(0) == 5),
              "MTP FP16 MoE requires M1/M5, topk=10");
  const int m = weights.size(0), n = down ? 2560 : 320, k = down ? 160 : 2560;
  TORCH_CHECK(w.sizes() == at::IntArrayRef({512, n, k}) &&
                  x.sizes() == at::IntArrayRef({down ? m * 10 : m, k}) &&
                  out.sizes() == at::IntArrayRef({m, 10, n}) &&
                  ids.sizes() == at::IntArrayRef({m * 10}) &&
                  padded.numel() == 1,
              "Invalid MTP FP16 MoE geometry");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "MTP FP16 MoE requires 16-byte aligned weights");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const half*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const half*>(w.data_ptr());
  const auto* ip = ids.data_ptr<int32_t>();
  const auto* tp = weights.data_ptr<float>();
  const auto* pp = padded.data_ptr<int32_t>();
  auto* op = reinterpret_cast<half*>(out.data_ptr());
  if (down) {
    mtp_moe_fp16_tile_kernel<2560, 160, 64, 64, true>
        <<<dim3(40, m * 10), 128, 0, stream>>>(xp, wp, ip, tp, pp, op);
  } else if (m == 1) {
    mtp_moe_fp16_m1_w13_kernel<<<dim3(5, 10), 64, 0, stream>>>(xp, wp, ip, pp,
                                                               op);
  } else {
    mtp_moe_fp16_tile_kernel<320, 2560, 32, 128, false>
        <<<dim3(10, m * 10), 128, 0, stream>>>(xp, wp, ip, tp, pp, op);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm70_mtp_moe_fp16_out(Tensor(a!) out, Tensor x, Tensor w, Tensor ids, "
      "Tensor weights, Tensor padded, bool down) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm70_mtp_moe_fp16_out", &mtp_moe_fp16_out);
}
