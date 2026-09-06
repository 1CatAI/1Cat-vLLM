// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
//
// Qwen3.8 TP4 native-g32 AWQ M=1. The quadpair-N m8n8k4 dataflow is
// derived from mxfp4_qpn_m1_sm70.cu / dnv2003/v100-skinny (MIT).
// See adjacent LICENSE.v100-skinny for the retained notice.
// Preserve FP16 scale/bias dequantization and per-route rounding; the
// CTA-local FP32 reduction is numerically, not bitwise, equivalent to
// the legacy TurboMind serial split-K route.

#include <torch/all.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kK = 2560;
constexpr int kN = 320;
constexpr int kExperts = 512;
constexpr int kRoutes = 10;
constexpr int kSplit = 16;

// Formats: existing 3B scalar (0) / 4B scale+bias metadata (2).
template <int Format>
__device__ __forceinline__ uint32_t read_awq_stats(const uint8_t* stats,
                                                   int group, int tile, int col,
                                                   int n) {
  uint32_t bits = 0;
  if constexpr (Format == 2) {
    return __ldg(reinterpret_cast<const uint32_t*>(stats) + group * n +
                 tile * 32 + col);
  } else {
    const auto* record =
        stats + (static_cast<size_t>(group) * n + tile * 32 + col) * 3;
    bits = static_cast<uint32_t>(__ldg(record)) |
           (static_cast<uint32_t>(__ldg(record + 1)) << 8) |
           (static_cast<uint32_t>(__ldg(record + 2)) << 16);
  }
  const half scale = __ushort_as_half(static_cast<uint16_t>(bits));
  const half zero = __int2half_rn(static_cast<int>((bits >> 16) & 0xff));
  const half bias = __hmul(__hneg(zero), scale);
  return (bits & 0xffffu) |
         (static_cast<uint32_t>(__half_as_ushort(bias)) << 16);
}

__device__ __forceinline__ void dequant_awq_u4x8(uint32_t packed,
                                                 uint32_t stats,
                                                 half2* decoded) {
  // Exact current TurboMind U4 -> half conversion, with the native FP16 bias
  // boundary. In particular, do not substitute (q - zero) * scale.
  uint32_t h[4];
  const uint32_t upper = __byte_perm(packed, 0, 0x4321);
  constexpr uint32_t lut = (0xf0 & 0xcc) | 0xaa;
  constexpr uint32_t bottom_mask = 0x000f000f;
  constexpr uint32_t top_mask = 0x00f000f0;
  constexpr uint32_t magic0 = 0x64006400;
  constexpr uint32_t magic1 = 0x54005400;
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[0])
      : "r"(packed), "n"(bottom_mask), "n"(magic0), "n"(lut));
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[1])
      : "r"(packed), "n"(top_mask), "n"(magic1), "n"(lut));
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[2])
      : "r"(upper), "n"(bottom_mask), "n"(magic0), "n"(lut));
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[3])
      : "r"(upper), "n"(top_mask), "n"(magic1), "n"(lut));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[0]) : "r"(h[0]), "r"(magic0));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[1]) : "r"(h[1]), "r"(magic1));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[2]) : "r"(h[2]), "r"(magic0));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[3]) : "r"(h[3]), "r"(magic1));
  const half scale = __ushort_as_half(static_cast<uint16_t>(stats));
  const half bias = __ushort_as_half(static_cast<uint16_t>(stats >> 16));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    decoded[i] =
        __hfma2(*reinterpret_cast<const half2*>(&h[i]),
                __halves2half2(scale, scale), __halves2half2(bias, bias));
  }
}

#define QPN_MMA(C, A0, A1, B0, B1)                                  \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};"                                  \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int Format>
__global__ void shared_qpn_w13_kernel(const half* input,
                                      const uint32_t* weights,
                                      const uint8_t* metadata,
                                      const int32_t* expert_ids, half* output) {
  __shared__ float partials[kSplit][32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int route = blockIdx.y;
  const int expert = __ldg(expert_ids + route);
  if (expert < 0 || expert >= kExperts) {
    if (threadIdx.x < 16)
      output[route * 160 + tile * 16 + threadIdx.x] = __float2half(0.f);
    return;
  }
  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = quadpair * 8 + a_row;
  const uint32_t* expert_weights =
      weights + static_cast<size_t>(expert) * kK * kN / 8;
  constexpr int stats_bytes = (kK / 32) * kN * (Format == 2 ? 4 : 3);
  const uint8_t* expert_stats =
      metadata + static_cast<size_t>(expert) * stats_bytes;
  float accum[8] = {};
  uint32_t stats = 0;
#pragma unroll 4
  for (int group = warp * 10; group < warp * 10 + 10; ++group) {
    const size_t base =
        (static_cast<size_t>(tile) * (kK / 8) + group * 2) * 32 + packed_col;
    const uint32_t packed0 = __ldcs(expert_weights + base);
    const uint32_t packed1 = __ldcs(expert_weights + base + 32);
    half2 decoded[8];

    // Each warp begins on an even K16 group. Reuse g32 metadata for both
    // halves without changing the common K16/MMA/FP32 accumulation order.
    if ((group & 1) == 0)
      stats =
          read_awq_stats<Format>(expert_stats, group / 2, tile, packed_col, kN);
    dequant_awq_u4x8(packed0, stats, decoded);
    dequant_awq_u4x8(packed1, stats, decoded + 4);

    const auto* b = reinterpret_cast<const uint32_t*>(decoded);
    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (a_row == 0) {
      input01 = *reinterpret_cast<const uint4*>(input + group * 16);
      input23 = *reinterpret_cast<const uint4*>(input + group * 16 + 8);
    }
    const auto* a0 = reinterpret_cast<const uint32_t*>(&input01);
    const auto* a1 = reinterpret_cast<const uint32_t*>(&input23);
    QPN_MMA(accum, a0[0], a0[1], b[0], b[1]);
    QPN_MMA(accum, a0[2], a0[3], b[2], b[3]);
    QPN_MMA(accum, a1[0], a1[1], b[4], b[5]);
    QPN_MMA(accum, a1[2], a1[3], b[6], b[7]);
  }
  if ((lane & 17) == 0) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int index = pair * 4 + offset;
        const int col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        partials[warp][quadpair * 8 + col] = accum[index];
      }
    }
  }
  __syncthreads();
  if (warp == 0) {
    float value = 0.f;
#pragma unroll
    for (int part = 0; part < kSplit; ++part) value += partials[part][lane];
    const half rounded = __float2half(value);
    const unsigned rounded_bits = __half_as_ushort(rounded);
    const int source_lane = (lane & 15) * 2;
    const half gate = __ushort_as_half(static_cast<unsigned short>(
        __shfl_sync(0xffffffffu, rounded_bits, source_lane)));
    const half up = __ushort_as_half(static_cast<unsigned short>(
        __shfl_sync(0xffffffffu, rounded_bits, source_lane + 1)));
    if (lane < 16) {
      const float gate_f = __half2float(gate);
      const half silu = __float2half(gate_f / (1.f + expf(-gate_f)));
      output[route * 160 + tile * 16 + lane] = __hmul(silu, up);
    }
  }
}

template <int Format>
__global__ void shared_qpn_w2_reduce_kernel(
    const half* __restrict__ input, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ metadata,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ topk_weights, half* __restrict__ output) {
  constexpr int k = 160;
  constexpr int n = 2560;
  __shared__ half route_outputs[kRoutes][32];
  const int lane = threadIdx.x & 31;
  const int route = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int expert = __ldg(expert_ids + route);
  float accum[8] = {};
  if (expert >= 0 && expert < kExperts) {
    const int quadpair = (lane >> 2) & 3;
    const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int packed_col = quadpair * 8 + a_row;
    const uint32_t* expert_weights =
        weights + static_cast<size_t>(expert) * k * n / 8;
    constexpr int bytes = (k / 32) * n * (Format == 2 ? 4 : 3);
    const uint8_t* expert_stats =
        metadata + static_cast<size_t>(expert) * bytes;
    const half* input_row = input + route * k;
    uint32_t stats = 0;
#pragma unroll
    for (int group = 0; group < k / 16; ++group) {
      const size_t base =
          (static_cast<size_t>(tile) * (k / 8) + group * 2) * 32 + packed_col;
      const uint32_t packed0 = __ldcs(expert_weights + base);
      const uint32_t packed1 = __ldcs(expert_weights + base + 32);
      half2 decoded[8];

      if ((group & 1) == 0)
        stats = read_awq_stats<Format>(expert_stats, group / 2, tile,
                                       packed_col, n);
      dequant_awq_u4x8(packed0, stats, decoded);
      dequant_awq_u4x8(packed1, stats, decoded + 4);

      const auto* b = reinterpret_cast<const uint32_t*>(decoded);
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      if (a_row == 0) {
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }
      const auto* a0 = reinterpret_cast<const uint32_t*>(&input01);
      const auto* a1 = reinterpret_cast<const uint32_t*>(&input23);
      QPN_MMA(accum, a0[0], a0[1], b[0], b[1]);
      QPN_MMA(accum, a0[2], a0[3], b[2], b[3]);
      QPN_MMA(accum, a1[0], a1[1], b[4], b[5]);
      QPN_MMA(accum, a1[2], a1[3], b[6], b[7]);
    }
    if ((lane & 17) == 0) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
        for (int offset = 0; offset < 2; ++offset) {
          const int index = pair * 4 + offset;
          const int col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
          route_outputs[route][quadpair * 8 + col] = __float2half(accum[index]);
        }
      }
    }
  } else if (lane < 4) {
#pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
      route_outputs[route][lane * 8 + offset] = __float2half(0.f);
    }
  }
  __syncthreads();
  if (route == 0) {
    float weighted = 0.f;
#pragma unroll
    for (int selected = 0; selected < kRoutes; ++selected) {
      // Preserve original router order and per-route FP16 materialization.
      weighted = fmaf(__ldg(topk_weights + selected),
                      __half2float(route_outputs[selected][lane]), weighted);
    }
    output[tile * 32 + lane] = __float2half(weighted);
  }
}

template <int W13Format, int W2Format>
void launch(const at::Tensor& input, const at::Tensor& w13,
            const at::Tensor& s13, const at::Tensor& w2, const at::Tensor& s2,
            const at::Tensor& ids, const at::Tensor& topk,
            at::Tensor& intermediate, at::Tensor& out, cudaStream_t stream) {
  shared_qpn_w13_kernel<W13Format><<<dim3(10, 10), 512, 0, stream>>>(
      reinterpret_cast<const half*>(input.const_data_ptr()),
      reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
      reinterpret_cast<const uint8_t*>(s13.const_data_ptr()),
      ids.const_data_ptr<int32_t>(),
      reinterpret_cast<half*>(intermediate.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  shared_qpn_w2_reduce_kernel<W2Format><<<80, 320, 0, stream>>>(
      reinterpret_cast<const half*>(intermediate.const_data_ptr()),
      reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
      reinterpret_cast<const uint8_t*>(s2.const_data_ptr()),
      ids.const_data_ptr<int32_t>(), topk.const_data_ptr<float>(),
      reinterpret_cast<half*>(out.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_tensor(const at::Tensor& tensor, at::ScalarType dtype,
                  at::IntArrayRef shape, const at::Device& device,
                  const char* name, uintptr_t alignment = 16) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device,
              "awq_qpn_m1: ", name, " must be CUDA on the input device");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.sizes() == shape &&
                  tensor.is_contiguous(),
              "awq_qpn_m1: ", name, " dtype/shape/contiguity mismatch");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(tensor.const_data_ptr()) % alignment == 0,
      "awq_qpn_m1: ", name, " alignment mismatch");
}

}  // namespace

void awq_moe_qpn_m1_sm70_out(at::Tensor out, at::Tensor intermediate,
                             const at::Tensor& input, const at::Tensor& w13,
                             const at::Tensor& s13, const at::Tensor& w2,
                             const at::Tensor& s2, const at::Tensor& ids,
                             const at::Tensor& topk) {
  const auto device = input.device();
  check_tensor(input, at::kHalf, {1, 2560}, device, "input");
  check_tensor(out, at::kHalf, {1, 2560}, device, "output");
  check_tensor(intermediate, at::kHalf, {10, 160}, device, "intermediate");
  check_tensor(w13, at::kInt, {512, 2560, 40}, device, "W13");
  check_tensor(w2, at::kInt, {512, 160, 320}, device, "W2");
  const bool compact = s13.scalar_type() == at::kByte;
  if (compact) {
    check_tensor(s13, at::kByte, {512, 80, 320, 3}, device, "W13 metadata");
    check_tensor(s2, at::kByte, {512, 5, 2560, 3}, device, "W2 metadata");
  } else {
    check_tensor(s13, at::kInt, {512, 80, 320}, device, "W13 metadata");
    check_tensor(s2, at::kInt, {512, 5, 2560}, device, "W2 metadata");
  }
  check_tensor(ids, at::kInt, {1, 10}, device, "expert IDs", 4);
  check_tensor(topk, at::kFloat, {1, 10}, device, "router weights", 4);
  for (const auto* tensor : {&input, &w13, &s13, &w2, &s2, &ids, &topk}) {
    at::assert_no_overlap(out, *tensor);
    at::assert_no_overlap(intermediate, *tensor);
  }
  at::assert_no_overlap(out, intermediate);
  const c10::cuda::CUDAGuard guard(device);
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "awq_qpn_m1 requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  // Reuse the prepared bank; no load-time repack or additional weight copy.
  if (compact) {
    launch<0, 0>(input, w13, s13, w2, s2, ids, topk, intermediate, out, stream);
  } else {
    launch<2, 2>(input, w13, s13, w2, s2, ids, topk, intermediate, out, stream);
  }
}
