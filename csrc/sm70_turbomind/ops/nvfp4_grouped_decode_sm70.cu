// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Experimental multi-row native-NVFP4 decode. Python dispatch defaults off.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
constexpr int kMaxTokens = 40;
constexpr int kTopK = 10;
constexpr int kMaxRoutes = kMaxTokens * kTopK;
constexpr int kPack = 8;
constexpr int kExperts = 512;

// Integer atomics only: route order within a pack is immaterial because both
// projections scatter back to the original route before the unchanged W2.
__global__ void plan_kernel(const int32_t* ids, int32_t* rows, int32_t* experts,
                            int32_t* sizes, int32_t* total, int routes) {
  __shared__ int counts[kExperts + 1];
  __shared__ int group_base[kExperts + 1];
  const int t = threadIdx.x;
  for (int e = t; e <= kExperts; e += blockDim.x) counts[e] = 0;
  if (t == 0) *total = 0;
  __syncthreads();
  int expert = 0, ordinal = 0;
  if (t < routes) {
    expert = ids[t];
    if (expert < 0 || expert >= kExperts) expert = kExperts;
    ordinal = atomicAdd(counts + expert, 1);
  }
  __syncthreads();
  for (int e = t; e <= kExperts; e += blockDim.x) {
    const int n_groups = (counts[e] + kPack - 1) / kPack;
    group_base[e] = n_groups > 0 ? atomicAdd(total, n_groups) : 0;
  }
  __syncthreads();
  if (t < routes) {
    const int group = group_base[expert] + ordinal / kPack;
    rows[group * kPack + ordinal % kPack] = t;
    if (ordinal % kPack == 0) {
      experts[group] = expert;
      sizes[group] = min(kPack, counts[expert] - ordinal);
    }
  }
}

__device__ __forceinline__ void decode(unsigned packed, half2 scale,
                                       half2* out) {
  constexpr unsigned sign = 0x80008000u, em = 0x0e000e00u;
  unsigned v[4] = {((packed << 12) & sign) | ((packed << 9) & em),
                   ((packed << 8) & sign) | ((packed << 5) & em),
                   ((packed << 4) & sign) | ((packed << 1) & em),
                   (packed & sign) | ((packed >> 3) & em)};
#pragma unroll
  for (int i = 0; i < 4; ++i)
    out[i] = __hmul2(*reinterpret_cast<half2*>(v + i), scale);
}

__device__ __forceinline__ half nvfp4_strict_hmma_scale(uint8_t scale_code,
                                                        float global_scale) {
  const unsigned code = scale_code;
  const unsigned mag = code & 0x7fU;
  const unsigned sub_hi =
      static_cast<unsigned>((0x232221201e1c1800ULL >> (mag * 8U)) & 0xffU)
      << 8U;
  unsigned half_bits =
      mag < 8U ? sub_hi : (mag == 0x7fU ? 0x7e00U : (mag << 7U) + 0x2000U);
  half_bits |= (code & 0x80U) << 8U;
  const half raw_scale = *reinterpret_cast<const half*>(&half_bits);
  const half prepared = __float2half_rn(__half2float(raw_scale) * global_scale);
  return __hmul(prepared, __float2half_rn(16384.0f));
}

#define PACKED_MMA(C, A0, A1, B0, B1)                               \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int Inter, int Split, bool Interleaved, int MinBlocks = 2,
          bool RawScale = false>
__global__ __launch_bounds__(64 * Split, MinBlocks) void w13_kernel(
    const half* __restrict__ x, const uint32_t* __restrict__ weights,
    const void* __restrict__ scales_raw,
    const float* __restrict__ global_scales, const int32_t* __restrict__ rows,
    const int32_t* __restrict__ experts, const int32_t* __restrict__ sizes,
    const int32_t* __restrict__ total, half* __restrict__ out) {
  constexpr int kTiles = Inter / 32;
  // Split within the CTA: union partial and projected into 16 KB so Split=8
  // achieves 3 CTAs/SM (48 expert groups/wave) on SM70 with zero register
  // spill.
  union SmemStorage {
    float partial[2][Split][kPack][32];
    half projected[2][kPack][32];
  };
  __shared__ SmemStorage smem;
  const int total_groups = *total;
  if (blockIdx.y >= total_groups) return;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int projection = warp / Split, split = warp % Split;
  const int tile = Interleaved ? blockIdx.x * 2 + projection
                               : blockIdx.x + projection * kTiles;
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + mma_row;
  for (int group_id = blockIdx.y; group_id < total_groups;
       group_id += gridDim.y) {
    const int count = sizes[group_id], expert = experts[group_id];
    const bool repeated =
        (group_id > 0 && experts[group_id - 1] == expert) ||
        (group_id + 1 < total_groups && experts[group_id + 1] == expert);
    const int route = mma_row < count ? rows[group_id * kPack + mma_row] : 0;
    float accum[8] = {};
    if (expert < kExperts) {
      const uint32_t* w =
          weights + static_cast<size_t>(expert) * 2560 * (Inter / 4);
      const size_t scale_base = static_cast<size_t>(expert) * 160 * (2 * Inter);
      const half* s = RawScale
                          ? nullptr
                          : static_cast<const half*>(scales_raw) + scale_base;
      const uint8_t* sc =
          RawScale ? static_cast<const uint8_t*>(scales_raw) + scale_base
                   : nullptr;
      float gs = 0.0f;
      if constexpr (RawScale) {
        gs =
            __ldg(global_scales + static_cast<size_t>(expert) * 2 + projection);
      }
      const half* input = x + static_cast<size_t>(route / kTopK) * 2560;
#pragma unroll 4
      for (int g = split * (160 / Split); g < (split + 1) * (160 / Split);
           ++g) {
        const size_t offset =
            (static_cast<size_t>(tile) * 320 + g * 2) * 32 + col;
        const size_t s_idx = (g * (2 * kTiles) + tile) * 32 + col;
        half scalar;
        if constexpr (RawScale) {
          scalar = nvfp4_strict_hmma_scale(__ldg(sc + s_idx), gs);
        } else {
          scalar = __hmul(__ldg(s + s_idx), __float2half_rn(16384.0f));
        }
        const half2 scale = __halves2half2(scalar, scalar);
        half2 decoded[4];
        decode(repeated ? __ldg(w + offset) : __ldcs(w + offset), scale,
               decoded);
        const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
        uint4 lo = mma_row < count
                       ? __ldg(reinterpret_cast<const uint4*>(input + g * 16))
                       : make_uint4(0, 0, 0, 0);
        PACKED_MMA(accum, lo.x, lo.y, b[0], b[1]);
        PACKED_MMA(accum, lo.z, lo.w, b[2], b[3]);
        decode(repeated ? __ldg(w + offset + 32) : __ldcs(w + offset + 32),
               scale, decoded);
        uint4 hi =
            mma_row < count
                ? __ldg(reinterpret_cast<const uint4*>(input + g * 16 + 8))
                : make_uint4(0, 0, 0, 0);
        PACKED_MMA(accum, hi.x, hi.y, b[0], b[1]);
        PACKED_MMA(accum, hi.z, hi.w, b[2], b[3]);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      smem.partial[projection][split][r][quad * 8 + c] = accum[i];
    }
    __syncthreads();
    // 2 * kPack * 32 == 512; for Split < 8 loop covers all 512 elements.
    half reduced_vals[Split < 8 ? (8 / Split) : 1];
#pragma unroll
    for (int it = 0; it < (Split < 8 ? (8 / Split) : 1); ++it) {
      const int idx = threadIdx.x + it * (64 * Split);
      if (idx < 2 * kPack * 32) {
        const int p = idx / (kPack * 32), r = idx / 32 % kPack, c = idx % 32;
        float value = 0;
#pragma unroll
        for (int sp = 0; sp < Split; ++sp) value += smem.partial[p][sp][r][c];
        reduced_vals[it] = __float2half_rn(value);
      }
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < (Split < 8 ? (8 / Split) : 1); ++it) {
      const int idx = threadIdx.x + it * (64 * Split);
      if (idx < 2 * kPack * 32) {
        const int p = idx / (kPack * 32), r = idx / 32 % kPack, c = idx % 32;
        smem.projected[p][r][c] = reduced_vals[it];
      }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < count * 32; idx += blockDim.x) {
      const int r = idx / 32, c = idx % 32;
      const int p = Interleaved ? c / 16 : 0;
      const int pc = Interleaved ? c % 16 * 2 : c;
      const half gate = smem.projected[p][r][pc];
      const half up =
          Interleaved ? smem.projected[p][r][pc + 1] : smem.projected[1][r][c];
      const float gf = __half2float(gate);
      const half activated = __float2half_rn(gf / (1.0f + expf(-gf)));
      out[static_cast<size_t>(rows[group_id * kPack + r]) * Inter +
          blockIdx.x * 32 + c] = __hmul(activated, up);
    }
    if (group_id + gridDim.y < total_groups) __syncthreads();
  }
}

template <bool RawScale>
void run_w13_impl(torch::Tensor out, torch::Tensor x, torch::Tensor w,
                  torch::Tensor s, const torch::Tensor* global_scales,
                  torch::Tensor ids, torch::Tensor rows, torch::Tensor experts,
                  torch::Tensor sizes, torch::Tensor total, int64_t split,
                  bool interleaved) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int routes = x.size(0) * kTopK;
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 2560 && routes > 0 &&
              routes <= kMaxRoutes);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == routes);
  const int inter = out.size(1);
  TORCH_CHECK(inter == 160 || inter == 320 || inter == 640);
  for (const auto& t : {out, x, w, s, ids, rows, experts, sizes, total}) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  }
  TORCH_CHECK(x.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf &&
              w.scalar_type() == at::kInt);
  if constexpr (RawScale) {
    TORCH_CHECK(s.scalar_type() == at::kByte && global_scales != nullptr &&
                global_scales->is_cuda() &&
                global_scales->device() == x.device() &&
                global_scales->is_contiguous() &&
                global_scales->scalar_type() == at::kFloat &&
                global_scales->numel() == 2 * kExperts);
  } else {
    TORCH_CHECK(s.scalar_type() == at::kHalf);
  }
  for (const auto& t : {ids, rows, experts, sizes, total})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(ids.numel() == routes && rows.numel() >= routes * kPack &&
              experts.numel() >= routes && sizes.numel() >= routes &&
              total.numel() == 1 && out.numel() == routes * inter &&
              w.numel() ==
                  static_cast<int64_t>(kExperts) * 2560 * (inter / 4) &&
              s.numel() == static_cast<int64_t>(kExperts) * 160 * (2 * inter));
  const float* gs_ptr = RawScale ? global_scales->data_ptr<float>() : nullptr;
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  plan_kernel<<<1, 512, 0, stream>>>(
      ids.data_ptr<int32_t>(), rows.data_ptr<int32_t>(),
      experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
      total.data_ptr<int32_t>(), routes);
  if (split == 4 && inter == 160) {
    const int gy = routes > 200 ? routes : std::min(routes, 48);
    if (interleaved) {
      w13_kernel<160, 4, true, 3, RawScale><<<dim3(5, gy), 256, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr()),
          reinterpret_cast<const uint32_t*>(w.data_ptr()), s.data_ptr(), gs_ptr,
          rows.data_ptr<int32_t>(), experts.data_ptr<int32_t>(),
          sizes.data_ptr<int32_t>(), total.data_ptr<int32_t>(),
          reinterpret_cast<half*>(out.data_ptr()));
    } else {
      w13_kernel<160, 4, false, 3, RawScale><<<dim3(5, gy), 256, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr()),
          reinterpret_cast<const uint32_t*>(w.data_ptr()), s.data_ptr(), gs_ptr,
          rows.data_ptr<int32_t>(), experts.data_ptr<int32_t>(),
          sizes.data_ptr<int32_t>(), total.data_ptr<int32_t>(),
          reinterpret_cast<half*>(out.data_ptr()));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const int grid_y = std::min(routes, routes <= 60 ? 48 : 192);
#define LAUNCH(INTER, S, I)                                              \
  w13_kernel<INTER, S, I, 2, RawScale>                                   \
      <<<dim3((INTER) / 32, grid_y), 64 * S, 0, stream>>>(               \
          reinterpret_cast<const half*>(x.data_ptr()),                   \
          reinterpret_cast<const uint32_t*>(w.data_ptr()), s.data_ptr(), \
          gs_ptr, rows.data_ptr<int32_t>(), experts.data_ptr<int32_t>(), \
          sizes.data_ptr<int32_t>(), total.data_ptr<int32_t>(),          \
          reinterpret_cast<half*>(out.data_ptr()))
#define LAUNCH_INTER(S, I)   \
  if (inter == 160) {        \
    LAUNCH(160, S, I);       \
  } else if (inter == 320) { \
    LAUNCH(320, S, I);       \
  } else {                   \
    LAUNCH(640, S, I);       \
  }
#define CASE(S)               \
  case S:                     \
    if (interleaved) {        \
      LAUNCH_INTER(S, true);  \
    } else {                  \
      LAUNCH_INTER(S, false); \
    }                         \
    break
  switch (split) {
    CASE(1);
    CASE(2);
    CASE(4);
    CASE(5);
    CASE(8);
    default:
      TORCH_CHECK(false, "Unsupported split");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef CASE
#undef LAUNCH_INTER
#undef LAUNCH
}

void run(torch::Tensor out, torch::Tensor x, torch::Tensor w, torch::Tensor s,
         torch::Tensor ids, torch::Tensor rows, torch::Tensor experts,
         torch::Tensor sizes, torch::Tensor total, int64_t split,
         bool interleaved) {
  run_w13_impl<false>(out, x, w, s, nullptr, ids, rows, experts, sizes, total,
                      split, interleaved);
}

void run_raw(torch::Tensor out, torch::Tensor x, torch::Tensor w,
             torch::Tensor scale_codes, torch::Tensor global_scales,
             torch::Tensor ids, torch::Tensor rows, torch::Tensor experts,
             torch::Tensor sizes, torch::Tensor total, int64_t split,
             bool interleaved) {
  run_w13_impl<true>(out, x, w, scale_codes, &global_scales, ids, rows, experts,
                     sizes, total, split, interleaved);
}

// All groups, including singletons, use one kernel. The earlier prototype
// launched separate repeated/singleton kernels; their overhead erased reuse.
template <int Inter, bool RawScale = false>
__global__ void w2_kernel(const half* x, const uint32_t* weights,
                          const void* scales_raw, const float* global_scales,
                          const int32_t* rows, const int32_t* experts,
                          const int32_t* sizes, const int32_t* total,
                          half* out) {
  constexpr int kW2Groups = Inter / 16;
  const int group = blockIdx.y * 4 + threadIdx.x / 32;
  if (group >= *total) return;
  const int count = sizes[group], expert = experts[group];
  const int lane = threadIdx.x % 32, quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0), col = quad * 8 + r;
  const int route = r < count ? rows[group * kPack + r] : 0;
  float accum[8] = {};
  if (expert < kExperts) {
    const uint32_t* w = weights + static_cast<size_t>(expert) * Inter * 320;
    const size_t scale_base = static_cast<size_t>(expert) * kW2Groups * 2560;
    const half* s =
        RawScale ? nullptr : static_cast<const half*>(scales_raw) + scale_base;
    const uint8_t* sc =
        RawScale ? static_cast<const uint8_t*>(scales_raw) + scale_base
                 : nullptr;
    float gs = 0.0f;
    if constexpr (RawScale) {
      gs = __ldg(global_scales + expert);
    }
    const half* input = x + static_cast<size_t>(route) * Inter;
#pragma unroll
    for (int g = 0; g < kW2Groups; ++g) {
      const int offset = (blockIdx.x * (2 * kW2Groups) + g * 2) * 32 + col;
      const int s_idx = (g * 80 + blockIdx.x) * 32 + col;
      half scalar;
      if constexpr (RawScale) {
        scalar = nvfp4_strict_hmma_scale(__ldg(sc + s_idx), gs);
      } else {
        scalar = __hmul(__ldg(s + s_idx), __float2half_rn(16384.0f));
      }
      const half2 scale = __halves2half2(scalar, scalar);
      half2 decoded[8];
      decode(__ldcs(w + offset), scale, decoded);
      decode(__ldcs(w + offset + 32), scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
      uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
      if (r < count) {
        lo = *reinterpret_cast<const uint4*>(input + g * 16);
        hi = *reinterpret_cast<const uint4*>(input + g * 16 + 8);
      }
      PACKED_MMA(accum, lo.x, lo.y, b[0], b[1]);
      PACKED_MMA(accum, lo.z, lo.w, b[2], b[3]);
      PACKED_MMA(accum, hi.x, hi.y, b[4], b[5]);
      PACKED_MMA(accum, hi.z, hi.w, b[6], b[7]);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    if (row < count) {
      const int dst = rows[group * kPack + row];
      out[static_cast<size_t>(dst) * 2560 + blockIdx.x * 32 + quad * 8 + c] =
          __float2half_rn(accum[i]);
    }
  }
}

__global__ void reduce_kernel(const half* routed, const float* weights,
                              half* out) {
  const int token = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
  float result = 0;
#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot)
    result = fmaf(__half2float(routed[(token * kTopK + slot) * 2560 + col]),
                  weights[token * kTopK + slot], result);
  out[token * 2560 + col] = __float2half_rn(result);
}

// Each CTA owns 32 output columns for the entire token batch. A warp reuses
// one expert's weights across up to eight routed rows, and loops over actual
// groups rather than token slots. Keep the FP16 W2 boundary in CTA memory;
// after one barrier, reduce each token in the original top-k order. No global
// scatter tensor, floating-point atomics, or inter-CTA synchronization.
template <int Inter, int MaxRoutes, bool RawScale = false>
__global__ __launch_bounds__(1024) void w2_batch_reduce_kernel(
    const half* x, const uint32_t* weights, const void* scales_raw,
    const float* global_scales, const float* topk, const int32_t* rows,
    const int32_t* experts, const int32_t* sizes, const int32_t* total,
    half* out, int tokens) {
  constexpr int kW2Groups = Inter / 16;
  __shared__ half values[MaxRoutes][32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quad * 8 + r, tile = blockIdx.x;
  const int groups = *total;
  for (int group = warp; group < groups; group += 32) {
    const int count = sizes[group], expert = experts[group];
    const bool repeated = (group > 0 && experts[group - 1] == expert) ||
                          (group + 1 < groups && experts[group + 1] == expert);
    const int route = r < count ? rows[group * kPack + r] : 0;
    float accum[8] = {};
    if (expert < kExperts) {
      const uint32_t* w = weights + static_cast<size_t>(expert) * Inter * 320;
      const size_t scale_base = static_cast<size_t>(expert) * kW2Groups * 2560;
      const half* s = RawScale
                          ? nullptr
                          : static_cast<const half*>(scales_raw) + scale_base;
      const uint8_t* sc =
          RawScale ? static_cast<const uint8_t*>(scales_raw) + scale_base
                   : nullptr;
      float gs = 0.0f;
      if constexpr (RawScale) {
        gs = __ldg(global_scales + expert);
      }
      const half* input = x + static_cast<size_t>(route) * Inter;
#pragma unroll
      for (int g = 0; g < kW2Groups; ++g) {
        const int offset = (tile * (2 * kW2Groups) + g * 2) * 32 + col;
        const int s_idx = (g * 80 + tile) * 32 + col;
        half scalar;
        if constexpr (RawScale) {
          scalar = nvfp4_strict_hmma_scale(__ldg(sc + s_idx), gs);
        } else {
          scalar = __hmul(__ldg(s + s_idx), __float2half_rn(16384.0f));
        }
        const half2 scale = __halves2half2(scalar, scalar);
        half2 decoded[8];
        decode(repeated ? __ldg(w + offset) : __ldcs(w + offset), scale,
               decoded);
        decode(repeated ? __ldg(w + offset + 32) : __ldcs(w + offset + 32),
               scale, decoded + 4);
        const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
        uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
        if (r < count) {
          lo = __ldg(reinterpret_cast<const uint4*>(input + g * 16));
          hi = __ldg(reinterpret_cast<const uint4*>(input + g * 16 + 8));
        }
        PACKED_MMA(accum, lo.x, lo.y, b[0], b[1]);
        PACKED_MMA(accum, lo.z, lo.w, b[2], b[3]);
        PACKED_MMA(accum, hi.x, hi.y, b[4], b[5]);
        PACKED_MMA(accum, hi.z, hi.w, b[6], b[7]);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (row < count) {
        values[rows[group * kPack + row]][quad * 8 + c] =
            __float2half_rn(accum[i]);
      }
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < tokens * 32; idx += blockDim.x) {
    const int token = idx / 32, c = idx % 32;
    float weighted = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      weighted = fmaf(__half2float(values[token * kTopK + slot][c]),
                      __ldg(topk + token * kTopK + slot), weighted);
    }
    out[token * 2560 + tile * 32 + c] = __float2half_rn(weighted);
  }
}

template <bool RawScale>
void w2_dispatch(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
                 torch::Tensor w, torch::Tensor s,
                 const torch::Tensor* global_scales, torch::Tensor topk,
                 torch::Tensor rows, torch::Tensor experts, torch::Tensor sizes,
                 torch::Tensor total, bool batch_reduce) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int inter = x.dim() == 2 ? x.size(1) : 0;
  TORCH_CHECK(x.dim() == 2 && (inter == 160 || inter == 320 || inter == 640) &&
              x.size(0) % kTopK == 0);
  const int routes = x.size(0), tokens = routes / kTopK;
  TORCH_CHECK(tokens >= 1 && tokens <= kMaxTokens);
  for (const auto& t :
       {out, routed, x, w, s, topk, rows, experts, sizes, total})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  for (const auto& t : {out, routed, x})
    TORCH_CHECK(t.scalar_type() == at::kHalf);
  if constexpr (RawScale) {
    TORCH_CHECK(s.scalar_type() == at::kByte && global_scales != nullptr &&
                global_scales->is_cuda() &&
                global_scales->device() == x.device() &&
                global_scales->is_contiguous() &&
                global_scales->scalar_type() == at::kFloat &&
                global_scales->numel() == kExperts);
  } else {
    TORCH_CHECK(s.scalar_type() == at::kHalf);
  }
  for (const auto& t : {w, rows, experts, sizes, total})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(topk.scalar_type() == at::kFloat && topk.numel() == routes &&
              out.numel() == tokens * 2560 &&
              (batch_reduce || routed.numel() == routes * 2560) &&
              rows.numel() >= routes * kPack && experts.numel() >= routes &&
              sizes.numel() >= routes && total.numel() == 1 &&
              w.numel() == static_cast<int64_t>(kExperts) * inter * 320 &&
              s.numel() ==
                  static_cast<int64_t>(kExperts) * (inter / 16) * 2560);
  const float* gs_ptr = RawScale ? global_scales->data_ptr<float>() : nullptr;
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (batch_reduce) {
    auto launch_br = [&](auto inter_tag, auto max_routes_tag) {
      constexpr int Inter = decltype(inter_tag)::value;
      constexpr int MaxR = decltype(max_routes_tag)::value;
      w2_batch_reduce_kernel<Inter, MaxR, RawScale><<<80, 1024, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr()),
          reinterpret_cast<const uint32_t*>(w.data_ptr()), s.data_ptr(), gs_ptr,
          topk.data_ptr<float>(), rows.data_ptr<int32_t>(),
          experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
          total.data_ptr<int32_t>(), reinterpret_cast<half*>(out.data_ptr()),
          tokens);
    };
    auto dispatch_inter = [&](auto max_routes_tag) {
      if (inter == 160) {
        launch_br(std::integral_constant<int, 160>{}, max_routes_tag);
      } else if (inter == 320) {
        launch_br(std::integral_constant<int, 320>{}, max_routes_tag);
      } else {
        launch_br(std::integral_constant<int, 640>{}, max_routes_tag);
      }
    };
    if (tokens <= 10) {
      dispatch_inter(std::integral_constant<int, 100>{});
    } else if (tokens <= 16) {
      dispatch_inter(std::integral_constant<int, 160>{});
    } else if (tokens <= 20) {
      dispatch_inter(std::integral_constant<int, 200>{});
    } else {
      dispatch_inter(std::integral_constant<int, kMaxRoutes>{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  auto w2_fn = inter == 160   ? w2_kernel<160, RawScale>
               : inter == 320 ? w2_kernel<320, RawScale>
                              : w2_kernel<640, RawScale>;
  w2_fn<<<dim3(80, (routes + 3) / 4), 128, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const uint32_t*>(w.data_ptr()), s.data_ptr(), gs_ptr,
      rows.data_ptr<int32_t>(), experts.data_ptr<int32_t>(),
      sizes.data_ptr<int32_t>(), total.data_ptr<int32_t>(),
      reinterpret_cast<half*>(routed.data_ptr()));
  reduce_kernel<<<dim3(10, tokens), 256, 0, stream>>>(
      reinterpret_cast<const half*>(routed.data_ptr()), topk.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void w2(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
        torch::Tensor w, torch::Tensor s, torch::Tensor topk,
        torch::Tensor rows, torch::Tensor experts, torch::Tensor sizes,
        torch::Tensor total) {
  w2_dispatch<false>(out, routed, x, w, s, nullptr, topk, rows, experts, sizes,
                     total, false);
}

void w2_raw(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
            torch::Tensor w, torch::Tensor scale_codes,
            torch::Tensor global_scales, torch::Tensor topk, torch::Tensor rows,
            torch::Tensor experts, torch::Tensor sizes, torch::Tensor total) {
  w2_dispatch<true>(out, routed, x, w, scale_codes, &global_scales, topk, rows,
                    experts, sizes, total, false);
}

void w2_batch_reduce(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
                     torch::Tensor w, torch::Tensor s, torch::Tensor topk,
                     torch::Tensor rows, torch::Tensor experts,
                     torch::Tensor sizes, torch::Tensor total) {
  w2_dispatch<false>(out, routed, x, w, s, nullptr, topk, rows, experts, sizes,
                     total, true);
}

void w2_batch_reduce_raw(torch::Tensor out, torch::Tensor routed,
                         torch::Tensor x, torch::Tensor w,
                         torch::Tensor scale_codes, torch::Tensor global_scales,
                         torch::Tensor topk, torch::Tensor rows,
                         torch::Tensor experts, torch::Tensor sizes,
                         torch::Tensor total) {
  w2_dispatch<true>(out, routed, x, w, scale_codes, &global_scales, topk, rows,
                    experts, sizes, total, true);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "nvfp4_grouped_w13_sm70_out(Tensor(a!) out, Tensor x, Tensor w, Tensor "
      "s, Tensor ids, "
      "Tensor(b!) rows, Tensor(c!) experts, Tensor(d!) sizes, Tensor(e!) "
      "total, "
      "int split, bool interleaved) -> ()");
  m.def(
      "nvfp4_grouped_w13_raw_sm70_out(Tensor(a!) out, Tensor x, Tensor w, "
      "Tensor scale_codes, Tensor global_scales, Tensor ids, "
      "Tensor(b!) rows, Tensor(c!) experts, Tensor(d!) sizes, Tensor(e!) "
      "total, "
      "int split, bool interleaved) -> ()");
  m.def(
      "nvfp4_grouped_w2_sm70_out(Tensor(a!) out, Tensor(b!) routed, Tensor x, "
      "Tensor w, Tensor s, "
      "Tensor topk, Tensor rows, Tensor experts, Tensor sizes, Tensor total) "
      "-> ()");
  m.def(
      "nvfp4_grouped_w2_raw_sm70_out(Tensor(a!) out, Tensor(b!) routed, Tensor "
      "x, Tensor w, Tensor scale_codes, Tensor global_scales, "
      "Tensor topk, Tensor rows, Tensor experts, Tensor sizes, Tensor total) "
      "-> ()");
  m.def(
      "nvfp4_grouped_w2_batch_reduce_sm70_out(Tensor(a!) out, Tensor routed, "
      "Tensor x, Tensor w, Tensor s, Tensor topk, Tensor rows, Tensor experts, "
      "Tensor sizes, Tensor total) -> ()");
  m.def(
      "nvfp4_grouped_w2_batch_reduce_raw_sm70_out(Tensor(a!) out, Tensor "
      "routed, Tensor x, Tensor w, Tensor scale_codes, Tensor global_scales, "
      "Tensor topk, Tensor rows, Tensor experts, Tensor sizes, Tensor total) "
      "-> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("nvfp4_grouped_w13_sm70_out", &run);
  m.impl("nvfp4_grouped_w13_raw_sm70_out", &run_raw);
  m.impl("nvfp4_grouped_w2_sm70_out", &w2);
  m.impl("nvfp4_grouped_w2_raw_sm70_out", &w2_raw);
  m.impl("nvfp4_grouped_w2_batch_reduce_sm70_out", &w2_batch_reduce);
  m.impl("nvfp4_grouped_w2_batch_reduce_raw_sm70_out", &w2_batch_reduce_raw);
}
