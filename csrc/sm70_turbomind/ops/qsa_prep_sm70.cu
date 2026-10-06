// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// QSA attention prepare for SM70 decode/verify: one launch replaces the compiled q/k GemmaRMSNorm +
// RoPE kernels, reshape_and_cache and the copy around them. Per token (CTA): warps 0..HQ-1 take the
// query heads, warp HQ takes the key head (+ value copy). head_dim 256 (8 dims per lane), partial
// rotary over the first ROT dims, NeoX pairing (i, i + ROT/2); text positions (MRoPE sections equal).
// Gate stays in place (q_gate rows are [q | g] per head) for the attention kernel to read strided.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace qp {
constexpr int D = 256;
__device__ __forceinline__ float tof(float v) { return v; }
__device__ __forceinline__ float tof(half v) { return __half2float(v); }

template <typename CS>
__global__ void prep(const half* qkv, int ld_qkv, const int64_t* pos, const CS* cos_sin, int rot, const half* qw,
                     const half* kw, float eps, half* query, half* kc, half* vc, const int64_t* slot, int BS,
                     long long sk_block, long long sk_tok, long long sv_block, long long sv_tok, int HQ) {
  const int t = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp > HQ) return;
  const bool is_k = warp == HQ;
  const half* src = qkv + static_cast<long long>(t) * ld_qkv + (is_k ? HQ * 2 * D : warp * 2 * D) + lane * 8;
  const uint4 raw = *reinterpret_cast<const uint4*>(src);
  const half* xh = reinterpret_cast<const half*>(&raw);
  const half* w = (is_k ? kw : qw) + lane * 8;
  const uint4 wraw = *reinterpret_cast<const uint4*>(w);
  const half* wh = reinterpret_cast<const half*>(&wraw);
  float x[8], ss = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = __half2float(xh[j]);
    ss += x[j] * x[j];
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss / D + eps);
  // GemmaRMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w), rounded to the activation dtype
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = __half2float(__float2half_rn(x[j] * inv * (1.f + __half2float(wh[j]))));
  // NeoX RoPE on dims [0, rot): pair (i, i + rot/2), partner 8-dim chunk is rot/16 lanes away
  const int half_rot = rot / 2, d0 = lane * 8;
  float px[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int off = half_rot / 8;
    const float up = __shfl_down_sync(0xffffffffu, x[j], off);
    const float dn = __shfl_up_sync(0xffffffffu, x[j], off);
    px[j] = (d0 < half_rot) ? up : dn;
  }
  if (d0 < rot) {
    const long long p = pos[t];
    const CS* cs = cos_sin + p * rot;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = d0 + j;
      const bool first = d < half_rot;
      const int i = first ? d : d - half_rot;
      const float c = tof(cs[i]), s = tof(cs[half_rot + i]);
      x[j] = first ? x[j] * c - px[j] * s : x[j] * c + px[j] * s;
    }
  }
  uint4 outv;
  half* oh = reinterpret_cast<half*>(&outv);
#pragma unroll
  for (int j = 0; j < 8; ++j) oh[j] = __float2half_rn(x[j]);
  if (!is_k) {
    *reinterpret_cast<uint4*>(query + (static_cast<long long>(t) * HQ + warp) * D + lane * 8) = outv;
  } else {
    const long long sl = slot[t];
    if (sl >= 0) {
      const long long blk = sl / BS, off = sl % BS;
      *reinterpret_cast<uint4*>(kc + blk * sk_block + off * sk_tok + lane * 8) = outv;
      const uint4 v = *reinterpret_cast<const uint4*>(qkv + static_cast<long long>(t) * ld_qkv + HQ * 2 * D + D + lane * 8);
      *reinterpret_cast<uint4*>(vc + blk * sv_block + off * sv_tok + lane * 8) = v;
    }
  }
}
}  // namespace qp

void qsa_prep_sm70_out(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cos_sin, torch::Tensor qw, torch::Tensor kw, double eps,
              torch::Tensor query, torch::Tensor kc, torch::Tensor vc, torch::Tensor slot) {
  const c10::cuda::CUDAGuard guard(qkv.device());
  const int M = qkv.size(0), HQ = query.size(1), rot = cos_sin.size(1);
  TORCH_CHECK(query.size(2) == qp::D && qkv.stride(1) == 1 && rot % 16 == 0 && rot <= qp::D && kc.size(2) == 1);
  TORCH_CHECK(pos.dtype() == torch::kInt64 && slot.dtype() == torch::kInt64);
  auto st = at::cuda::getCurrentCUDAStream();
#define GO(T, P)                                                                                                   \
  qp::prep<T><<<M, 32 * (HQ + 1), 0, st>>>(                                                                        \
      reinterpret_cast<const half*>(qkv.data_ptr()), qkv.stride(0), pos.data_ptr<int64_t>(), P, rot,                 \
      reinterpret_cast<const half*>(qw.data_ptr()), reinterpret_cast<const half*>(kw.data_ptr()),                   \
      static_cast<float>(eps), reinterpret_cast<half*>(query.data_ptr()), reinterpret_cast<half*>(kc.data_ptr()),    \
      reinterpret_cast<half*>(vc.data_ptr()), slot.data_ptr<int64_t>(), kc.size(1), kc.stride(0), kc.stride(1),     \
      vc.stride(0), vc.stride(1), HQ)
  if (cos_sin.dtype() == torch::kFloat32)
    GO(float, cos_sin.data_ptr<float>());
  else
    GO(half, reinterpret_cast<const half*>(cos_sin.data_ptr()));
#undef GO
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

