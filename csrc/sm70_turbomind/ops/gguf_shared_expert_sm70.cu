// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Based on the supplied dense_mv2/shexp2 SM70 projection implementation.
// TP4 shared expert for M<=8 in two launches, no grid barrier:
//   launch 1 (swiglu_mv): h = fp16(silu(fp16(x Wg^T)) * fp16(x Wu^T)); one
//   extra
//            CTA computes sg = fp16(x . w_shared_gate) in FP32.
//   launch 2 (dense_mv):  out = fp16(fp16(h Wd^T) * sigmoid(sg)).
// A CTA owns gate tile p and up tile p (half its warps each), so SiLU*mul is
// applied in the epilogue. Split-K across CTAs reduces deterministically.
#define DMV_NO_MODULE
#include "gguf_dense_segments_sm70.cuh"

namespace {

struct SwArgs {
  Seg gate, up;
  const half* x;
  int ldx;
  const void* wg;
  bool gate_half;
  half* h;
  half* sg;
  float* ws;
  int* cnt;
  int M, K, I, split;
};

template <int FMT, int HW>
__device__ __forceinline__ void half_body(const SwArgs& a, const Seg& s, int p,
                                          int g0, int g1, uint4* xs, half2* lut,
                                          float (&acc)[8], int hw) {
  const int S = (a.K + 31) / 32, G = (S + 3) / 4;
  body<FMT, HW, false>(s, p, g0, g1, S, G, a.x, a.ldx, a.M, a.K, xs, lut, acc,
                       hw);
}

template <int HW>
__device__ __forceinline__ void run_half(const SwArgs& a, const Seg& s, int p,
                                         int g0, int g1, uint4* xs, half2* lut,
                                         float (&acc)[8], int hw) {
  switch (s.fmt) {
    case Q4K:
      half_body<Q4K, HW>(a, s, p, g0, g1, xs, lut, acc, hw);
      break;
    case Q5K:
      half_body<Q5K, HW>(a, s, p, g0, g1, xs, lut, acc, hw);
      break;
    case Q6K:
      half_body<Q6K, HW>(a, s, p, g0, g1, xs, lut, acc, hw);
      break;
    case LUT4:
      half_body<LUT4, HW>(a, s, p, g0, g1, xs, lut, acc, hw);
      break;
    default:
      half_body<Q8, HW>(a, s, p, g0, g1, xs, lut, acc, hw);
      break;
  }
}

__device__ __forceinline__ void write_h(const SwArgs& a, int p, int v, float g,
                                        float u) {
  const int lv = v >> 3, i = v & 7;
  const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
  const int col = (i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2);
  const int row = p * 32 + ((lv >> 2) & 3) * 8 + col;
  if (token < a.M && row < a.I) {
    const float gh = __half2float(__float2half_rn(g));
    const float uh = __half2float(__float2half_rn(u));
    a.h[token * a.I + row] = __float2half_rn(gh / (1.0f + __expf(-gh)) * uh);
  }
}

template <int W, bool Batch>
__global__ void __launch_bounds__(32 * W) swiglu_mv(SwArgs a) {
  constexpr int HW = W / 2;
  extern __shared__ uint4 smem[];
  __shared__ int last;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int npair = (a.I + 31) / 32;
  const int p = blockIdx.x, sp = blockIdx.y;
  if constexpr (Batch) {
    const int first_row = blockIdx.z * 8;
    a.M = min(8, a.M - first_row);
    a.x += first_row * a.ldx;
    a.h += first_row * a.I;
    a.sg += first_row;
    a.ws += blockIdx.z * (npair + 1) * a.split * 512;
    a.cnt += blockIdx.z * (npair + 1);
  }
  half2* lut = reinterpret_cast<half2*>(smem + W * 256);
  if (p == npair) {  // shared gate dot (only split 0)
    if (sp != 0) return;
    float* part = reinterpret_cast<float*>(smem);
    float acc[8] = {};
    for (int k = threadIdx.x; k < a.K; k += 32 * W) {
      const float w = a.gate_half
                          ? __half2float(static_cast<const half*>(a.wg)[k])
                          : static_cast<const float*>(a.wg)[k];
#pragma unroll
      for (int m = 0; m < 8; ++m)
        if (m < a.M) acc[m] = fmaf(__half2float(a.x[m * a.ldx + k]), w, acc[m]);
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      float v = acc[m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) part[m * W + warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 8 && static_cast<int>(threadIdx.x) < a.M) {
      float v = 0.f;
      for (int w = 0; w < W; ++w) v += part[threadIdx.x * W + w];
      a.sg[threadIdx.x] = __float2half_rn(v);
    }
    return;
  }
  for (int i = threadIdx.x; i < 256; i += 32 * W)
    lut[i] = __halves2half2(__int2half_rn(kIQ4[i & 15]),
                            __int2half_rn(kIQ4[i >> 4]));
  __syncthreads();
  const int S = (a.K + 31) / 32, G = (S + 3) / 4;
  const int gps = (G + a.split - 1) / a.split;
  const int g0 = min(G, sp * gps), g1 = min(G, g0 + gps);
  float acc[8] = {};
  const bool isup = warp >= HW;
  run_half<HW>(a, isup ? a.up : a.gate, p, g0, g1, smem + (isup ? HW * 256 : 0),
               lut, acc, warp % HW);
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int i = 0; i < 8; ++i) red[warp * 256 + lane * 8 + i] = acc[i];
  __syncthreads();
  for (int v = threadIdx.x; v < 256; v += 32 * W) {
    float g = 0.f, u = 0.f;
#pragma unroll
    for (int w = 0; w < HW; ++w) {
      g += red[w * 256 + v];
      u += red[(w + HW) * 256 + v];
    }
    if (a.split == 1) {
      write_h(a, p, v, g, u);
    } else {
      float* dst = a.ws + (static_cast<size_t>(p) * a.split + sp) * 512;
      dst[v] = g;
      dst[256 + v] = u;
    }
  }
  if (a.split == 1) return;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    last = atomicAdd(a.cnt + p, 1) == a.split - 1;
    if (last) __threadfence();
  }
  __syncthreads();
  if (!last) return;
  for (int v = threadIdx.x; v < 256; v += 32 * W) {
    float g = 0.f, u = 0.f;
    for (int q = 0; q < a.split; ++q) {
      const float* src = a.ws + (static_cast<size_t>(p) * a.split + q) * 512;
      g += __ldcg(src + v);
      u += __ldcg(src + 256 + v);
    }
    write_h(a, p, v, g, u);
  }
  if (threadIdx.x == 0) a.cnt[p] = 0;
}

Seg make_seg(const torch::Tensor& codes, const torch::Tensor& high,
             const torch::Tensor& scale, int64_t fmt, int64_t n) {
  Seg s{};
  s.codes = reinterpret_cast<const uint4*>(codes.data_ptr());
  s.high =
      high.numel() ? reinterpret_cast<const uint4*>(high.data_ptr()) : nullptr;
  s.scale = reinterpret_cast<const uint4*>(scale.data_ptr());
  s.n = static_cast<int>(n);
  s.fmt = static_cast<int>(fmt);
  return s;
}

template <int W>
void launch_sw(const SwArgs& a) {
  const size_t smem = static_cast<size_t>(W) * 256 * 16 + 1024;
  if (smem > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        swiglu_mv<W, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
  }
  const int npair = (a.I + 31) / 32;
  if (a.M <= 8) {
    swiglu_mv<W, false><<<dim3(npair + 1, a.split), 32 * W, smem,
                          at::cuda::getCurrentCUDAStream()>>>(a);
  } else {
    if (smem > 48 * 1024)
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          swiglu_mv<W, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem));
    swiglu_mv<W, true><<<dim3(npair + 1, a.split, (a.M + 7) / 8), 32 * W, smem,
                         at::cuda::getCurrentCUDAStream()>>>(a);
  }
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "swiglu_mv launch: ", cudaGetErrorString(e));
}
}  // namespace

void gguf_shared_gate_up_sm70_out(torch::Tensor x, std::vector<torch::Tensor> g,
                                  std::vector<torch::Tensor> u,
                                  std::vector<int64_t> fmts, torch::Tensor wg,
                                  torch::Tensor h, torch::Tensor sg,
                                  torch::Tensor ws, torch::Tensor cnt,
                                  int64_t split, int64_t warps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.stride(1) == 1 && x.size(0) > 0 && x.size(0) <= 32,
              "input must be CUDA FP16 [1..32, K]");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "SM70 required");
  TORCH_CHECK(g.size() == 3 && u.size() == 3 && fmts.size() == 2 &&
                  h.dim() == 2 && h.size(0) == x.size(0) && h.size(1) > 0,
              "invalid shared-expert storage");
  TORCH_CHECK(x.size(1) > 0 && x.size(1) % 32 == 0 && split > 0 &&
                  split <= (x.size(1) + 127) / 128,
              "invalid shared-expert K or split count");
  TORCH_CHECK(warps == 4 || warps == 8 || warps == 16,
              "unsupported shared-expert warp count");
  auto same = [&](const torch::Tensor& t, at::ScalarType dtype) {
    TORCH_CHECK(
        t.device() == x.device() && t.scalar_type() == dtype &&
            t.is_contiguous(),
        "shared-expert storage must be contiguous and on the input device");
  };
  TORCH_CHECK(wg.scalar_type() == at::kFloat || wg.scalar_type() == at::kHalf,
              "shared gate requires FP16 or FP32 dense weights");
  same(wg, wg.scalar_type());
  same(h, at::kHalf);
  same(sg, at::kHalf);
  same(ws, at::kFloat);
  same(cnt, at::kInt);
  TORCH_CHECK(wg.numel() == x.size(1) && sg.numel() >= x.size(0),
              "invalid shared gate size");
  const int64_t tiles = (h.size(1) + 31) / 32, groups = (x.size(1) + 127) / 128;
  TORCH_CHECK(ws.numel() >= ((x.size(0) + 7) / 8) * (tiles + 1) * split * 512 &&
                  cnt.numel() >= ((x.size(0) + 7) / 8) * (tiles + 1),
              "shared-expert workspace or counters too small");
  for (int i = 0; i < 2; ++i) {
    auto& packed = i == 0 ? g : u;
    TORCH_CHECK(fmts[i] >= Q4K && fmts[i] <= Q8,
                "invalid shared-expert format");
    same(packed[0], at::kByte);
    same(packed[2], at::kByte);
    if (packed[1].numel()) same(packed[1], at::kByte);
    TORCH_CHECK(packed[0].numel() ==
                        tiles * groups * 4 * 512 * (fmts[i] == Q8 ? 2 : 1) &&
                    packed[2].numel() == tiles * groups * 512 &&
                    packed[1].numel() == tiles * groups * 512 *
                                             (fmts[i] == Q5K   ? 1
                                              : fmts[i] == Q6K ? 2
                                                               : 0),
                "shared-expert packed size mismatch");
  }
  SwArgs a{};
  const int64_t I = h.size(1);
  a.gate = make_seg(g[0], g[1], g[2], fmts[0], I);
  a.up = make_seg(u[0], u[1], u[2], fmts[1], I);
  a.x = reinterpret_cast<const half*>(x.data_ptr());
  a.ldx = static_cast<int>(x.stride(0));
  a.wg = wg.data_ptr();
  a.gate_half = wg.scalar_type() == at::kHalf;
  a.h = reinterpret_cast<half*>(h.data_ptr());
  a.sg = reinterpret_cast<half*>(sg.data_ptr());
  a.ws = ws.data_ptr<float>();
  a.cnt = cnt.data_ptr<int>();
  a.M = static_cast<int>(x.size(0));
  a.K = static_cast<int>(x.size(1));
  a.I = static_cast<int>(I);
  a.split = static_cast<int>(split);
  TORCH_CHECK(a.M <= 32);
  switch (warps) {
    case 4:
      launch_sw<4>(a);
      break;
    case 8:
      launch_sw<8>(a);
      break;
    default:
      launch_sw<16>(a);
      break;
  }
}

namespace {
// Reconstruct only the old packed integer layout in transient shared scratch.
// The resident bank remains the coalesced segment layout for every M.
__device__ uint32_t spread_two(uint32_t v) {
  v &= 65535u;
  v = (v | (v << 8)) & 0x00ff00ffu;
  v = (v | (v << 4)) & 0x0f0f0f0fu;
  return (v | (v << 2)) & 0x33333333u;
}
__device__ uint32_t spread_one(uint32_t v) {
  v &= 65535u;
  v = (v | (v << 8)) & 0x00ff00ffu;
  v = (v | (v << 4)) & 0x0f0f0f0fu;
  v = (v | (v << 2)) & 0x33333333u;
  return (v | (v << 1)) & 0x55555555u;
}
__device__ uint32_t lut_native_word(uint32_t v) {
  uint32_t result = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    result |= ((v >> (4 * j)) & 15) << (4 * (j / 2 + (j % 2) * 4));
  return result;
}
__global__ void segment_native_restore(Seg s, int k, int n, int groups,
                                       uint32_t* weight, void* stats) {
  const int64_t index = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (index >= int64_t{k} * n / 32) return;
  const int col = (index / (k / 32 * 32)) * 32 + index % 32;
  const int kb = ((index / 32) % (k / 32)) * 32;
  const int lane = (col % 4) | (((col % 32) / 8) << 2) | ((col & 4) << 2);
  const int t = col / 32, step = (kb / 32) % 4, g = kb / 128;
  const int64_t packet = (int64_t{t} * groups * 4 + kb / 32) * 32 + lane;
  const int64_t dst = (int64_t{t} * (k / 8) + kb / 8) * 32 + col % 32;
  uint4 values =
      s.codes[s.fmt == Q8
                  ? ((int64_t{t} * groups * 4 + kb / 32) * 2) * 32 + lane
                  : packet];
  if (s.fmt == Q8) {
    uint4 other =
        s.codes[((int64_t{t} * groups * 4 + kb / 32) * 2 + 1) * 32 + lane];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t value = word(i < 4 ? values : other, i % 4) ^ 0x80808080u;
      weight[(dst + (i / 2) * 32) * 2 + i % 2] =
          __byte_perm(value, value, 0x3120);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t value = word(values, i);
      weight[dst + i * 32] = s.fmt == LUT4 ? lut_native_word(value) : value;
    }
  }
  const uint32_t coef =
      word(s.scale[(int64_t{t} * groups + g) * 32 + lane], step);
  const int64_t stat_index = int64_t{kb / 32} * n + col;
  if (s.fmt == LUT4)
    static_cast<uint16_t*>(stats)[stat_index] = coef & 65535u;
  else if (s.fmt == Q5K) {
    const uint32_t high =
        word(s.high[(int64_t{t} * groups + g) * 32 + lane], step);
    const uint32_t natural = spread_one(high) | (spread_one(high >> 16) << 1);
    static_cast<uint64_t*>(stats)[stat_index] =
        coef | (uint64_t{natural} << 32);
  } else if (s.fmt == Q6K) {
    const uint4 high =
        s.high[((int64_t{t} * groups + g) * 2 + step / 2) * 32 + lane];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint32_t h = word(high, (step % 2) * 2 + i);
      uint32_t natural = spread_two(h) | (spread_two(h >> 16) << 2);
      uint16_t scale = i ? coef >> 16 : coef & 65535u;
      uint16_t minimum = __half_as_ushort(
          __hmul(__ushort_as_half(scale), __float2half_rn(-32.f)));
      static_cast<uint64_t*>(stats)[int64_t{kb / 16 + i} * n + col] =
          scale | (uint32_t{minimum} << 16) | (uint64_t{natural} << 32);
    }
  } else {
    uint32_t affine = coef;
    if (s.fmt == Q8) {
      uint16_t minimum = __half_as_ushort(
          __hmul(__ushort_as_half(coef & 65535u), __float2half_rn(-128.f)));
      affine = (coef & 65535u) | (uint32_t{minimum} << 16);
    }
    static_cast<uint32_t*>(stats)[stat_index] = affine;
  }
}
}  // namespace

void gguf_dense_restore_canonical_sm70_out(torch::Tensor weight,
                                           torch::Tensor stats,
                                           torch::Tensor codes,
                                           torch::Tensor high,
                                           torch::Tensor scale, int64_t fmt,
                                           int64_t k, int64_t n) {
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == at::kByte &&
                  codes.is_contiguous(),
              "invalid resident packed codes");
  const c10::cuda::CUDAGuard guard(codes.device());
  TORCH_CHECK(
      fmt >= Q4K && fmt <= Q8 && k > 0 && k % 32 == 0 && n > 0 && n % 32 == 0,
      "invalid restore geometry");
  const int64_t groups = (k + 127) / 128, tiles = n / 32;
  TORCH_CHECK(
      scale.device() == codes.device() && scale.scalar_type() == at::kByte &&
          scale.is_contiguous() && scale.numel() == tiles * groups * 512 &&
          codes.numel() == tiles * groups * 4 * 512 * (fmt == Q8 ? 2 : 1),
      "resident storage size mismatch");
  TORCH_CHECK(high.numel() == tiles * groups * 512 *
                                  (fmt == Q5K   ? 1
                                   : fmt == Q6K ? 2
                                                : 0),
              "resident high plane mismatch");
  if (high.numel())
    TORCH_CHECK(high.device() == codes.device() &&
                    high.scalar_type() == at::kByte && high.is_contiguous(),
                "invalid resident high plane");
  TORCH_CHECK(weight.device() == codes.device() &&
                  weight.scalar_type() == at::kInt && weight.is_contiguous() &&
                  weight.numel() == n * k / (fmt == Q8 ? 4 : 8),
              "invalid transient packed weight");
  TORCH_CHECK(
      stats.device() == codes.device() && stats.is_contiguous() &&
          stats.scalar_type() == (fmt == LUT4                ? at::kShort
                                  : fmt == Q5K || fmt == Q6K ? at::kLong
                                                             : at::kInt) &&
          stats.numel() == k / (fmt == Q6K ? 16 : 32) * n,
      "invalid transient coefficients");
  Seg s = make_seg(codes, high, scale, fmt, n);
  const int64_t count = k * n / 32;
  segment_native_restore<<<(count + 255) / 256, 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      s, k, n, groups, reinterpret_cast<uint32_t*>(weight.data_ptr()),
      stats.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
