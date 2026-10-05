// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The QPN2 execution layout is derived from dnv2003/v100-skinny (MIT).
// See LICENSE.v100-skinny in this directory for the retained MIT notice.

#include <mma.h>
#include <map>
#include <mutex>
#include <tuple>
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "nvfp4_qpn2_layout.cuh"
#include "activation_pack_sm70.cuh"

namespace {
__device__ __forceinline__ half2 fp8e4m3_to_half2(uint8_t value) {
  const unsigned short bits =
      ((static_cast<unsigned short>(value) & 0x80u) << 8) |
      ((static_cast<unsigned short>(value) & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

__device__ __forceinline__ half2 nvfp4_effective_scale(uint8_t value,
                                                       float global_scale) {
  // Match TurboMind's W4A16 weights: multiply the exact E4M3 group scale
  // by the FP32 global scale before rounding once to FP16. Rounding the
  // global factor first changes the model's weights on every decode step.
  const half raw = __low2half(fp8e4m3_to_half2(value));
  const half scaled =
      __float2half_rn(__fmul_rn(__half2float(raw), global_scale));
  return __halves2half2(scaled, scaled);
}

__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                               half2 output[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    // Undo the FP4 exponent bias on the code, not on its scale. Scaling
    // the scale by 2^14 first loses subnormals and can overflow even when
    // the final dequantized weights are finite.
    const half2 code = __hmul2(*reinterpret_cast<half2*>(&values[index]),
                               __float2half2_rn(16384.0f));
    output[index] = __hmul2(code, scale);
  }
}

#define VLLM_SM70_QPN2_MMA(C, A0, A1, B0, B1)                       \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// N128 tile: direct HMMA884 with shared A and register B.
// Each warp owns N32; all column/projection warps reuse each A stage.
// Read original K16 input fragments directly, without pack_k16_input.
template <int Rows, bool Gated>
__global__ void nvfp4_qpn2_m64_n128_kernel(const half* input,
                                           const uint8_t* codes,
                                           const uint8_t* scales, float* output,
                                           int width, int k, float gs) {
  constexpr int P = Gated ? 2 : 1, RowGroups = Rows == 64 ? 2 : 1,
                Warps = P * 4 * RowGroups, RowTiles = Rows / (8 * RowGroups),
                Parts = Gated ? 5 : 4;
  constexpr int PanelGroups = 8;
  __shared__ __align__(16) half as[2][PanelGroups][Rows][16];
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int projection = warp / (4 * RowGroups), rg = warp / 4 % RowGroups,
      nt = warp % 4;
  int tile = blockIdx.x * 4 + nt + projection * (width / 32);
  int local_row = (lane & 3) + ((lane & 16) ? 4 : 0), qp = (lane >> 2) & 3;
  int groups = k / 16, begin = groups * blockIdx.z / Parts,
      end = groups * (blockIdx.z + 1) / Parts;
  Nvfp4Qpn2CodeReader<true, true> reader(codes, tile, groups, lane);
  const uint8_t* sp = scales + static_cast<size_t>(tile) * groups * 32 + lane;
  float accum[RowTiles][8] = {};

  for (int v = tid; v < Rows * PanelGroups * 2; v += Warps * 32) {
    int row = v / (PanelGroups * 2), pg = (v / 2) % PanelGroups,
        kk = (v % 2) * 8;
    if (begin + pg < end)
      *reinterpret_cast<uint4*>(&as[0][pg][row][kk]) =
          *reinterpret_cast<const uint4*>(input + static_cast<size_t>(row) * k +
                                          (begin + pg) * 16 + kk);
  }
  __syncthreads();
  int stage = 0;
  for (int panel = begin; panel < end; panel += PanelGroups) {
    if (panel + PanelGroups < end) {
      for (int v = tid; v < Rows * PanelGroups * 2; v += Warps * 32) {
        int row = v / (PanelGroups * 2), pg = (v / 2) % PanelGroups,
            kk = v % 2 * 8;
        if (panel + PanelGroups + pg < end)
          *reinterpret_cast<uint4*>(&as[stage ^ 1][pg][row][kk]) =
              *reinterpret_cast<const uint4*>(
                  input + static_cast<size_t>(row) * k +
                  (panel + PanelGroups + pg) * 16 + kk);
      }
    }

    uint2 inflight[4];
    uint8_t inflight_scales[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      inflight[j] = panel + j < end ? reader.load(panel + j) : make_uint2(0, 0);
      inflight_scales[j] =
          panel + j < end ? __ldg(sp + static_cast<size_t>(panel + j) * 32) : 0;
    }
#pragma unroll
    for (int pg = 0; pg < PanelGroups; pg++) {
      if (panel + pg >= end) break;
      uint2 raw = inflight[pg % 4];
      uint8_t scale_raw = inflight_scales[pg % 4];
      if (pg + 4 < PanelGroups && panel + pg + 4 < end) {
        inflight[pg % 4] = reader.load(panel + pg + 4);
        inflight_scales[pg % 4] =
            __ldg(sp + static_cast<size_t>(panel + pg + 4) * 32);
      }
      half2 ws[8];
      half2 scale = nvfp4_effective_scale(scale_raw, gs);
      dequant_e2m1x8(raw.x, scale, ws);
      dequant_e2m1x8(raw.y, scale, ws + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(ws);
#pragma unroll
      for (int rt = 0; rt < RowTiles; rt++) {
        const half* a = &as[stage][pg][(rg * RowTiles + rt) * 8 + local_row][0];
        uint4 a01, a23;
        unsigned addr = __cvta_generic_to_shared(a);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3},[%4];"
                     : "=r"(a01.x), "=r"(a01.y), "=r"(a01.z), "=r"(a01.w)
                     : "r"(addr));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3},[%4];"
                     : "=r"(a23.x), "=r"(a23.y), "=r"(a23.z), "=r"(a23.w)
                     : "r"(addr + 16));
        VLLM_SM70_QPN2_MMA(accum[rt], a01.x, a01.y, b[0], b[1]);
        VLLM_SM70_QPN2_MMA(accum[rt], a01.z, a01.w, b[2], b[3]);
        VLLM_SM70_QPN2_MMA(accum[rt], a23.x, a23.y, b[4], b[5]);
        VLLM_SM70_QPN2_MMA(accum[rt], a23.z, a23.w, b[6], b[7]);
      }
    }
    __syncthreads();
    stage ^= 1;
  }
#pragma unroll
  for (int rt = 0; rt < RowTiles; rt++) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      output[(static_cast<size_t>(blockIdx.z) * P + projection) * Rows * width +
             (blockIdx.x * 4 + nt) * Rows * 32 + (rg * RowTiles + rt) * 256 +
             i * 32 + lane] = accum[rt][i];
    }
  }
}

template <bool Gated>
__global__ void nvfp4_qpn2_m64_n128_finish(half* output, const float* partial,
                                           int width, int rows) {
  constexpr int P = Gated ? 2 : 1, Parts = Gated ? 5 : 4;
  int tid = threadIdx.x, lane = tid & 31, i = tid >> 5, rt = blockIdx.y;
  int r = rt * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
  int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2) |
          (((lane >> 2) & 3) * 8);
  int count = rows * width;
  int index = blockIdx.x * rows * 32 + rt * 256 + tid;
  float gate = 0, up = 0;
#pragma unroll
  for (int z = 0; z < Parts; z++) {
    gate += partial[z * P * count + index];
    if constexpr (Gated) up += partial[(z * P + 1) * count + index];
  }
  half h = __float2half(gate);
  if constexpr (Gated) {
    float v = __half2float(h);
    h = __hmul(__float2half(v / (1.f + expf(-v))), __float2half(up));
  }
  output[r * width + blockIdx.x * 32 + c] = h;
}

}  // namespace

// The dispatcher admits only the two qualified TP4 MLP shapes. Each stream
// owns its partials; sharing across sequential layers avoids graph-pool copies
// of this scratch. Both shapes share a fixed-size buffer whose pointer stays
// live for earlier captured graphs. Every used partial is overwritten.
void nvfp4_qpn2_m64_n128_sm70_out(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor codes, torch::Tensor scales,
                                  double global_scale, bool gated_silu) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda() && codes.is_cuda() &&
                  scales.is_cuda() && input.device() == out.device() &&
                  input.device() == codes.device() &&
                  input.device() == scales.device(),
              "M64 QPN2 tensors must share a CUDA device");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat16 &&
          out.scalar_type() == torch::kFloat16 &&
          codes.scalar_type() == torch::kUInt8 &&
          scales.scalar_type() == torch::kUInt8 && input.is_contiguous() &&
          out.is_contiguous() && codes.is_contiguous() &&
          scales.is_contiguous(),
      "M64 QPN2 requires contiguous FP16 activations and uint8 layouts");
  const int rows = input.size(0), k = input.size(1), width = out.size(1);
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2 && rows == 64 &&
                  out.size(0) == rows &&
                  ((gated_silu && k == 5120 && width == 4352) ||
                   (!gated_silu && k == 4352 && width == 5120)),
              "M64 QPN2 requires a qualified TP4 MLP shape");
  const int projections = gated_silu ? 2 : 1;
  const int parts = gated_silu ? 5 : 4;
  TORCH_CHECK(codes.numel() == int64_t(k) * width * projections / 2 &&
                  scales.numel() == int64_t(k) * width * projections / 16,
              "M64 QPN2 layout size mismatch");
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  using Key = std::tuple<int, cudaStream_t>;
  static std::mutex mutex;
  static std::map<Key, torch::Tensor> scratch;
  constexpr int64_t kMaxElements = int64_t(5) * 2 * 64 * 4352;
  torch::Tensor storage;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto& cached = scratch[{input.get_device(), stream}];
    if (!cached.defined()) {
      cached =
          torch::empty({kMaxElements}, input.options().dtype(torch::kFloat32));
    }
    storage = cached;
  }
  const auto* a = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  const auto* b = codes.data_ptr<uint8_t>();
  const auto* s = scales.data_ptr<uint8_t>();
  auto* partial = storage.data_ptr<float>();
  auto* output = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  if (gated_silu) {
    nvfp4_qpn2_m64_n128_kernel<64, true>
        <<<dim3(width / 128, 1, parts), 512, 0, stream>>>(
            a, b, s, partial, width, k, global_scale);
    nvfp4_qpn2_m64_n128_finish<true>
        <<<dim3(width / 32, rows / 8), 256, 0, stream>>>(output, partial, width,
                                                         rows);
  } else {
    nvfp4_qpn2_m64_n128_kernel<64, false>
        <<<dim3(width / 128, 1, parts), 256, 0, stream>>>(
            a, b, s, partial, width, k, global_scale);
    nvfp4_qpn2_m64_n128_finish<false>
        <<<dim3(width / 32, rows / 8), 256, 0, stream>>>(output, partial, width,
                                                         rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
