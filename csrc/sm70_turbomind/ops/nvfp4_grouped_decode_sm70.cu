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
constexpr int kMaxRoutes = 160;
constexpr int kPack = 8;
constexpr int kExperts = 512;
constexpr int kChunks = kMaxRoutes / kPack;

// Integer atomics only: route order within a pack is immaterial because both
// projections scatter back to the original route before the unchanged W2.
__global__ void plan_kernel(const int32_t* ids, int32_t* rows, int32_t* experts,
                            int32_t* sizes, int32_t* total, int routes) {
  __shared__ int counts[kExperts + 1];
  __shared__ int groups[kExperts + 1][kChunks];
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
  if (t < routes && ordinal % kPack == 0) {
    const int group = atomicAdd(total, 1);
    groups[expert][ordinal / kPack] = group;
    experts[group] = expert;
    sizes[group] = min(kPack, counts[expert] - ordinal);
  }
  __syncthreads();
  if (t < routes) {
    const int group = groups[expert][ordinal / kPack];
    rows[group * kPack + ordinal % kPack] = t;
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

#define PACKED_MMA(C, A0, A1, B0, B1)                               \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int Split, bool Interleaved>
__global__ void w13_kernel(const half* x, const uint32_t* weights,
                           const half* scales, const int32_t* rows,
                           const int32_t* experts, const int32_t* sizes,
                           const int32_t* total, half* out) {
  // Split within the CTA: no floating-point atomics or global partial tensor.
  __shared__ float partial[2][Split][kPack][32];
  __shared__ half projected[2][kPack][32];
  const int group_id = blockIdx.y;
  if (group_id >= *total) return;
  const int count = sizes[group_id], expert = experts[group_id];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int projection = warp / Split, split = warp % Split;
  const int tile =
      Interleaved ? blockIdx.x * 2 + projection : blockIdx.x + projection * 5;
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + mma_row;
  const int route = mma_row < count ? rows[group_id * kPack + mma_row] : 0;
  float accum[8] = {};
  if (expert < kExperts) {
    const uint32_t* w = weights + static_cast<size_t>(expert) * 2560 * 40;
    const half* s = scales + static_cast<size_t>(expert) * 160 * 320;
    const half* input = x + static_cast<size_t>(route / 10) * 2560;
#pragma unroll 4
    for (int g = split * (160 / Split); g < (split + 1) * (160 / Split); ++g) {
      const size_t offset =
          (static_cast<size_t>(tile) * 320 + g * 2) * 32 + col;
      const half scalar = __hmul(__ldg(s + (g * 10 + tile) * 32 + col),
                                 __float2half_rn(16384.0f));
      const half2 scale = __halves2half2(scalar, scalar);
      half2 decoded[8];
      decode(__ldcs(w + offset), scale, decoded);
      decode(__ldcs(w + offset + 32), scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
      uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
      if (mma_row < count) {
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
    const int r = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partial[projection][split][r][quad * 8 + c] = accum[i];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < 2 * kPack * 32; idx += blockDim.x) {
    const int p = idx / (kPack * 32), r = idx / 32 % kPack, c = idx % 32;
    // FP16 materialization is retained before SiLU, then again before the
    // multiplication. Split>1 changes FP32 association, not quantization.
    float value = 0;
#pragma unroll
    for (int s = 0; s < Split; ++s) value += partial[p][s][r][c];
    projected[p][r][c] = __float2half_rn(value);
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < count * 32; idx += blockDim.x) {
    const int r = idx / 32, c = idx % 32;
    const int p = Interleaved ? c / 16 : 0;
    const int pc = Interleaved ? c % 16 * 2 : c;
    const half gate = projected[p][r][pc];
    const half up = Interleaved ? projected[p][r][pc + 1] : projected[1][r][c];
    const float gf = __half2float(gate);
    const half activated = __float2half_rn(gf / (1.0f + expf(-gf)));
    out[static_cast<size_t>(rows[group_id * kPack + r]) * 160 +
        blockIdx.x * 32 + c] = __hmul(activated, up);
  }
}

void run(torch::Tensor out, torch::Tensor x, torch::Tensor w, torch::Tensor s,
         torch::Tensor ids, torch::Tensor rows, torch::Tensor experts,
         torch::Tensor sizes, torch::Tensor total, int64_t split,
         bool interleaved) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int routes = x.size(0) * 10;
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 2560 && routes > 0 && routes <= 160);
  for (const auto& t : {out, x, w, s, ids, rows, experts, sizes, total}) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  }
  TORCH_CHECK(x.scalar_type() == at::kHalf && s.scalar_type() == at::kHalf &&
              out.scalar_type() == at::kHalf && w.scalar_type() == at::kInt);
  for (const auto& t : {ids, rows, experts, sizes, total})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(ids.numel() == routes && rows.numel() >= routes * kPack &&
              experts.numel() >= routes && sizes.numel() >= routes &&
              total.numel() == 1 && out.numel() == routes * 160 &&
              w.numel() == 512 * 2560 * 40 && s.numel() == 512 * 160 * 320);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  plan_kernel<<<1, 256, 0, stream>>>(
      ids.data_ptr<int32_t>(), rows.data_ptr<int32_t>(),
      experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
      total.data_ptr<int32_t>(), routes);
#define LAUNCH(S, I)                                                         \
  w13_kernel<S, I><<<dim3(5, routes), 64 * S, 0, stream>>>(                  \
      reinterpret_cast<const half*>(x.data_ptr()),                           \
      reinterpret_cast<const uint32_t*>(w.data_ptr()),                       \
      reinterpret_cast<const half*>(s.data_ptr()), rows.data_ptr<int32_t>(), \
      experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),                \
      total.data_ptr<int32_t>(), reinterpret_cast<half*>(out.data_ptr()))
#define CASE(S)         \
  case S:               \
    if (interleaved) {  \
      LAUNCH(S, true);  \
    } else {            \
      LAUNCH(S, false); \
    }                   \
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
#undef LAUNCH
}

// All groups, including singletons, use one kernel. The earlier prototype
// launched separate repeated/singleton kernels; their overhead erased reuse.
__global__ void w2_kernel(const half* x, const uint32_t* weights,
                          const half* scales, const int32_t* rows,
                          const int32_t* experts, const int32_t* sizes,
                          const int32_t* total, half* out) {
  const int group = blockIdx.y * 4 + threadIdx.x / 32;
  if (group >= *total) return;
  const int count = sizes[group], expert = experts[group];
  const int lane = threadIdx.x % 32, quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0), col = quad * 8 + r;
  const int route = r < count ? rows[group * kPack + r] : 0;
  float accum[8] = {};
  if (expert < kExperts) {
    const uint32_t* w = weights + static_cast<size_t>(expert) * 160 * 320;
    const half* s = scales + static_cast<size_t>(expert) * 10 * 2560;
    const half* input = x + static_cast<size_t>(route) * 160;
#pragma unroll
    for (int g = 0; g < 10; ++g) {
      const int offset = (blockIdx.x * 20 + g * 2) * 32 + col;
      const half scalar = __hmul(__ldg(s + (g * 80 + blockIdx.x) * 32 + col),
                                 __float2half_rn(16384.0f));
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
  for (int slot = 0; slot < 10; ++slot)
    result = fmaf(__half2float(routed[(token * 10 + slot) * 2560 + col]),
                  weights[token * 10 + slot], result);
  out[token * 2560 + col] = __float2half_rn(result);
}

void w2(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
        torch::Tensor w, torch::Tensor s, torch::Tensor topk,
        torch::Tensor rows, torch::Tensor experts, torch::Tensor sizes,
        torch::Tensor total) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 160 && x.size(0) % 10 == 0);
  const int routes = x.size(0), tokens = routes / 10;
  TORCH_CHECK(tokens >= 1 && tokens <= 16);
  for (const auto& t :
       {out, routed, x, w, s, topk, rows, experts, sizes, total})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  for (const auto& t : {out, routed, x, s})
    TORCH_CHECK(t.scalar_type() == at::kHalf);
  for (const auto& t : {w, rows, experts, sizes, total})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(topk.scalar_type() == at::kFloat && topk.numel() == routes &&
              out.numel() == tokens * 2560 && routed.numel() == routes * 2560 &&
              rows.numel() >= routes * 8 && experts.numel() >= routes &&
              sizes.numel() >= routes && total.numel() == 1 &&
              w.numel() == 512 * 160 * 320 && s.numel() == 512 * 10 * 2560);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  w2_kernel<<<dim3(80, (routes + 3) / 4), 128, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const uint32_t*>(w.data_ptr()),
      reinterpret_cast<const half*>(s.data_ptr()), rows.data_ptr<int32_t>(),
      experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
      total.data_ptr<int32_t>(), reinterpret_cast<half*>(routed.data_ptr()));
  reduce_kernel<<<dim3(10, tokens), 256, 0, stream>>>(
      reinterpret_cast<const half*>(routed.data_ptr()), topk.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

constexpr int kEp4Experts = 128;
constexpr int kEp4Intermediate = 640;
constexpr int kEp4W13Tiles = kEp4Intermediate / 32;
constexpr int kEp4W2Tiles = 2560 / 32;
constexpr int kEp4ChunksPerExpert = kChunks;

// Build only rank-local groups. Keep the full route-count bound instead of
// relying on top-k uniqueness: malformed or alternative routers must not
// overflow shared metadata. Remote routes remain absent from the local
// projection and are supplied by the owning rank through the final all-reduce.
__global__ void ep4_plan_kernel(const int32_t* ids, int32_t* rows,
                                int32_t* experts, int32_t* sizes,
                                int32_t* total, int routes, int expert_start) {
  __shared__ int counts[kEp4Experts];
  __shared__ int groups[kEp4Experts][kEp4ChunksPerExpert];
  const int t = threadIdx.x;
  if (t < kEp4Experts) counts[t] = 0;
  if (t == 0) *total = 0;
  __syncthreads();

  int expert = -1;
  int ordinal = -1;
  if (t < routes) {
    expert = ids[t] - expert_start;
    if (0 <= expert && expert < kEp4Experts) {
      ordinal = atomicAdd(counts + expert, 1);
    }
  }
  __syncthreads();

  if (ordinal >= 0 && ordinal % kPack == 0) {
    const int group = atomicAdd(total, 1);
    groups[expert][ordinal / kPack] = group;
    experts[group] = expert;
    sizes[group] = min(kPack, counts[expert] - ordinal);
  }
  __syncthreads();
  if (ordinal >= 0) {
    const int group = groups[expert][ordinal / kPack];
    rows[group * kPack + ordinal % kPack] = t;
  }
}

template <int Split, bool Interleaved>
__global__ void ep4_w13_kernel(const half* x, const uint32_t* weights,
                               const half* scales, const int32_t* rows,
                               const int32_t* experts, const int32_t* sizes,
                               const int32_t* total, half* out) {
  __shared__ float partial[2][Split][kPack][32];
  __shared__ half projected[2][kPack][32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int projection = warp / Split;
  const int split = warp % Split;
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + mma_row;
  const int task_count = *total * kEp4W13Tiles;

  for (int task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int group_id = task / kEp4W13Tiles;
    const int output_tile = task % kEp4W13Tiles;
    const int count = sizes[group_id];
    const int expert = experts[group_id];
    const int tile = Interleaved ? output_tile * 2 + projection
                                 : output_tile + projection * kEp4W13Tiles;
    const int route = mma_row < count ? rows[group_id * kPack + mma_row] : 0;
    float accum[8] = {};
    const uint32_t* w = weights + static_cast<size_t>(expert) * 2560 * 160;
    const half* scale_base = scales + static_cast<size_t>(expert) * 160 * 1280;
    const half* input = x + static_cast<size_t>(route / 10) * 2560;
#pragma unroll 4
    for (int g = split * (160 / Split); g < (split + 1) * (160 / Split); ++g) {
      const size_t offset =
          (static_cast<size_t>(tile) * 320 + g * 2) * 32 + col;
      const half scalar = __hmul(__ldg(scale_base + (g * 40 + tile) * 32 + col),
                                 __float2half_rn(16384.0f));
      const half2 scale = __halves2half2(scalar, scalar);
      half2 decoded[8];
      decode(__ldcs(w + offset), scale, decoded);
      decode(__ldcs(w + offset + 32), scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
      uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
      if (mma_row < count) {
        lo = *reinterpret_cast<const uint4*>(input + g * 16);
        hi = *reinterpret_cast<const uint4*>(input + g * 16 + 8);
      }
      PACKED_MMA(accum, lo.x, lo.y, b[0], b[1]);
      PACKED_MMA(accum, lo.z, lo.w, b[2], b[3]);
      PACKED_MMA(accum, hi.x, hi.y, b[4], b[5]);
      PACKED_MMA(accum, hi.z, hi.w, b[6], b[7]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partial[projection][split][r][quad * 8 + c] = accum[i];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 2 * kPack * 32; idx += blockDim.x) {
      const int p = idx / (kPack * 32);
      const int r = idx / 32 % kPack;
      const int c = idx % 32;
      float value = 0;
#pragma unroll
      for (int s = 0; s < Split; ++s) value += partial[p][s][r][c];
      projected[p][r][c] = __float2half_rn(value);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < count * 32; idx += blockDim.x) {
      const int r = idx / 32;
      const int c = idx % 32;
      const int p = Interleaved ? c / 16 : 0;
      const int pc = Interleaved ? c % 16 * 2 : c;
      const half gate = projected[p][r][pc];
      const half up =
          Interleaved ? projected[p][r][pc + 1] : projected[1][r][c];
      const float gf = __half2float(gate);
      const half activated = __float2half_rn(gf / (1.0f + expf(-gf)));
      out[static_cast<size_t>(rows[group_id * kPack + r]) * kEp4Intermediate +
          output_tile * 32 + c] = __hmul(activated, up);
    }
    __syncthreads();
  }
}

__global__ void ep4_w2_kernel(const half* x, const uint32_t* weights,
                              const half* scales, const int32_t* rows,
                              const int32_t* experts, const int32_t* sizes,
                              const int32_t* total, half* routed) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quad * 8 + r;
  const int group_batches = (*total + 3) / 4;
  const int task_count = group_batches * kEp4W2Tiles;

  for (int task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int output_tile = task % kEp4W2Tiles;
    const int group = task / kEp4W2Tiles * 4 + warp;
    if (group < *total) {
      const int count = sizes[group];
      const int expert = experts[group];
      const int route = r < count ? rows[group * kPack + r] : 0;
      float accum[8] = {};
      const uint32_t* w =
          weights + static_cast<size_t>(expert) * kEp4Intermediate * 320;
      const half* scale_base = scales + static_cast<size_t>(expert) * 40 * 2560;
      const half* input = x + static_cast<size_t>(route) * kEp4Intermediate;
#pragma unroll
      for (int g = 0; g < 40; ++g) {
        const int offset = (output_tile * 80 + g * 2) * 32 + col;
        const half scalar =
            __hmul(__ldg(scale_base + (g * 80 + output_tile) * 32 + col),
                   __float2half_rn(16384.0f));
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
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        if (row < count) {
          const int dst = rows[group * kPack + row];
          routed[static_cast<size_t>(dst) * 2560 + output_tile * 32 + quad * 8 +
                 c] = __float2half_rn(accum[i]);
        }
      }
    }
  }
}

__global__ void ep4_reduce_kernel(const half* routed, const float* weights,
                                  const int32_t* ids, half* out, int tokens,
                                  int expert_start) {
  const int token = blockIdx.y;
  const int col = blockIdx.x * 256 + threadIdx.x;
  float result = 0;
#pragma unroll
  for (int slot = 0; slot < 10; ++slot) {
    const int route = token * 10 + slot;
    const int expert = ids[route] - expert_start;
    if (0 <= expert && expert < kEp4Experts) {
      result = fmaf(__half2float(routed[route * 2560 + col]), weights[route],
                    result);
    }
  }
  out[token * 2560 + col] = __float2half_rn(result);
}

void ep4_run(torch::Tensor out, torch::Tensor x, torch::Tensor w,
             torch::Tensor s, torch::Tensor ids, torch::Tensor rows,
             torch::Tensor experts, torch::Tensor sizes, torch::Tensor total,
             int64_t expert_start, int64_t split, bool interleaved,
             int64_t blocks) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int routes = x.size(0) * 10;
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 2560 && routes > 0 &&
              routes <= kMaxRoutes);
  TORCH_CHECK(expert_start >= 0 && expert_start % kEp4Experts == 0 &&
              expert_start + kEp4Experts <= kExperts);
  TORCH_CHECK(blocks > 0 && blocks <= 640);
  for (const auto& t : {out, x, w, s, ids, rows, experts, sizes, total}) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  }
  TORCH_CHECK(x.scalar_type() == at::kHalf && s.scalar_type() == at::kHalf &&
              out.scalar_type() == at::kHalf && w.scalar_type() == at::kInt);
  for (const auto& t : {ids, rows, experts, sizes, total})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(ids.numel() == routes && rows.numel() >= routes * kPack &&
              experts.numel() >= routes && sizes.numel() >= routes &&
              total.numel() == 1 && out.numel() == routes * kEp4Intermediate &&
              w.numel() == kEp4Experts * 2560 * 160 &&
              s.numel() == kEp4Experts * 160 * 1280);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  ep4_plan_kernel<<<1, 256, 0, stream>>>(
      ids.data_ptr<int32_t>(), rows.data_ptr<int32_t>(),
      experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
      total.data_ptr<int32_t>(), routes, static_cast<int>(expert_start));
#define EP4_LAUNCH(S, I)                                                     \
  ep4_w13_kernel<S, I><<<static_cast<int>(blocks), 64 * S, 0, stream>>>(     \
      reinterpret_cast<const half*>(x.data_ptr()),                           \
      reinterpret_cast<const uint32_t*>(w.data_ptr()),                       \
      reinterpret_cast<const half*>(s.data_ptr()), rows.data_ptr<int32_t>(), \
      experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),                \
      total.data_ptr<int32_t>(), reinterpret_cast<half*>(out.data_ptr()))
#define EP4_CASE(S)         \
  case S:                   \
    if (interleaved) {      \
      EP4_LAUNCH(S, true);  \
    } else {                \
      EP4_LAUNCH(S, false); \
    }                       \
    break
  switch (split) {
    EP4_CASE(1);
    EP4_CASE(2);
    EP4_CASE(4);
    EP4_CASE(5);
    EP4_CASE(8);
    default:
      TORCH_CHECK(false, "Unsupported EP4 W13 split");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef EP4_CASE
#undef EP4_LAUNCH
}

void ep4_w2(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
            torch::Tensor w, torch::Tensor s, torch::Tensor topk,
            torch::Tensor ids, torch::Tensor rows, torch::Tensor experts,
            torch::Tensor sizes, torch::Tensor total, int64_t expert_start,
            int64_t blocks) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kEp4Intermediate &&
              x.size(0) % 10 == 0);
  const int routes = x.size(0);
  const int tokens = routes / 10;
  TORCH_CHECK(tokens >= 1 && tokens <= 16 && blocks > 0 && blocks <= 640);
  TORCH_CHECK(expert_start >= 0 && expert_start % kEp4Experts == 0 &&
              expert_start + kEp4Experts <= kExperts);
  for (const auto& t :
       {out, routed, x, w, s, topk, ids, rows, experts, sizes, total})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  for (const auto& t : {out, routed, x, s})
    TORCH_CHECK(t.scalar_type() == at::kHalf);
  for (const auto& t : {w, ids, rows, experts, sizes, total})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(topk.scalar_type() == at::kFloat && topk.numel() == routes &&
              ids.numel() == routes && out.numel() == tokens * 2560 &&
              routed.numel() == routes * 2560 &&
              rows.numel() >= routes * kPack && experts.numel() >= routes &&
              sizes.numel() >= routes && total.numel() == 1 &&
              w.numel() == kEp4Experts * kEp4Intermediate * 320 &&
              s.numel() == kEp4Experts * 40 * 2560);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  ep4_w2_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const uint32_t*>(w.data_ptr()),
      reinterpret_cast<const half*>(s.data_ptr()), rows.data_ptr<int32_t>(),
      experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
      total.data_ptr<int32_t>(), reinterpret_cast<half*>(routed.data_ptr()));
  ep4_reduce_kernel<<<dim3(10, tokens), 256, 0, stream>>>(
      reinterpret_cast<const half*>(routed.data_ptr()), topk.data_ptr<float>(),
      ids.data_ptr<int32_t>(), reinterpret_cast<half*>(out.data_ptr()), tokens,
      static_cast<int>(expert_start));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
      "nvfp4_grouped_w2_sm70_out(Tensor(a!) out, Tensor(b!) routed, Tensor x, "
      "Tensor w, Tensor s, "
      "Tensor topk, Tensor rows, Tensor experts, Tensor sizes, Tensor total) "
      "-> ()");
  m.def(
      "nvfp4_grouped_ep4_w13_sm70_out(Tensor(a!) out, Tensor x, Tensor w, "
      "Tensor s, Tensor ids, Tensor(b!) rows, Tensor(c!) experts, Tensor(d!) "
      "sizes, Tensor(e!) total, int expert_start, int split, bool interleaved, "
      "int blocks) -> ()");
  m.def(
      "nvfp4_grouped_ep4_w2_sm70_out(Tensor(a!) out, Tensor(b!) routed, "
      "Tensor x, Tensor w, Tensor s, Tensor topk, Tensor ids, Tensor rows, "
      "Tensor experts, Tensor sizes, Tensor total, int expert_start, int "
      "blocks) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("nvfp4_grouped_w13_sm70_out", &run);
  m.impl("nvfp4_grouped_w2_sm70_out", &w2);
  m.impl("nvfp4_grouped_ep4_w13_sm70_out", &ep4_run);
  m.impl("nvfp4_grouped_ep4_w2_sm70_out", &ep4_w2);
}
