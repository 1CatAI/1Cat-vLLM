// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The QPN2 execution layout is derived from dnv2003/v100-skinny (MIT).
// See LICENSE.v100-skinny in this directory for the retained MIT notice.

#include <mma.h>
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "../../../csrc/sm70_turbomind/ops/nvfp4_qpn2_layout.cuh"
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

// Research-only N128 tile, direct HMMA884 with shared A and register B.
// Each warp owns N32; all column/projection warps reuse each A stage.
template <int Rows, bool Gated>
__global__ void batch_n128_coalesced_kernel(const half* input,
                                            const uint8_t* codes,
                                            const uint8_t* scales,
                                            float* output, int width, int k,
                                            float gs) {
  constexpr int P = Gated ? 2 : 1, Warps = P * 4, RowTiles = Rows / 8,
                Parts = Gated ? 5 : 4;
  constexpr int PanelGroups = 8;
  __shared__ __align__(16) half as[2][PanelGroups][Rows][16];
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int projection = warp / 4, nt = warp % 4;
  int tile = blockIdx.x * 4 + nt + projection * (width / 32);
  int local_row = (lane & 3) + ((lane & 16) ? 4 : 0), qp = (lane >> 2) & 3;
  int groups = k / 16, begin = groups * blockIdx.z / Parts,
      end = groups * (blockIdx.z + 1) / Parts;
  Nvfp4Qpn2CodeReader<true> reader(codes, tile, groups, lane);
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
    uint2 next_codes = reader.load(panel);
    uint8_t next_scale = __ldg(sp + static_cast<size_t>(panel) * 32);
#pragma unroll 1
    for (int pg = 0; pg < PanelGroups && panel + pg < end; pg++) {
      uint2 raw = next_codes;
      uint8_t scale_raw = next_scale;
      if (pg + 1 < PanelGroups && panel + pg + 1 < end) {
        next_codes = reader.load(panel + pg + 1);
        next_scale = __ldg(sp + static_cast<size_t>(panel + pg + 1) * 32);
      }
      half2 ws[8];
      half2 scale = nvfp4_effective_scale(scale_raw, gs);
      dequant_e2m1x8(raw.x, scale, ws);
      dequant_e2m1x8(raw.y, scale, ws + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(ws);
#pragma unroll
      for (int rt = 0; rt < RowTiles; rt++) {
        const half* a = &as[stage][pg][rt * 8 + local_row][0];
        uint4 a01 = *reinterpret_cast<const uint4*>(a),
              a23 = *reinterpret_cast<const uint4*>(a + 8);
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
      int r = rt * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
      int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      output[(static_cast<size_t>(blockIdx.z) * P + projection) * Rows * width +
             (blockIdx.x * 4 + nt) * Rows * 32 + rt * 256 + i * 32 + lane] =
          accum[rt][i];
    }
  }
}

template <bool Gated>
__global__ void finish_n128(half* output, const float* partial, int width,
                            int rows) {
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

template <int Rows, bool Gated>
void launch(const half* a, const uint8_t* b, const uint8_t* s, float* tmp,
            int width, int k, float gs, cudaStream_t stream) {
  batch_n128_coalesced_kernel<Rows, Gated>
      <<<dim3(width / 128, 1, Gated ? 5 : 4), (Gated ? 8 : 4) * 32, 0,
         stream>>>(a, b, s, tmp, width, k, gs);
}
void run(torch::Tensor out, torch::Tensor input, torch::Tensor codes,
         torch::Tensor scales, torch::Tensor packed, torch::Tensor scratch,
         double gs, bool gated) {
  int m = input.size(0), k = input.size(1), width = out.size(1);
  TORCH_CHECK((m == 16 || m == 32 || m == 64) && k % 32 == 0 &&
              width % 128 == 0 && input.is_contiguous());
  TORCH_CHECK(scratch.numel() >= 8 * (gated ? 2 : 1) * m * width);
  at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* a = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* b = reinterpret_cast<const uint8_t*>(codes.data_ptr());
  auto* sc = scales.data_ptr<uint8_t>();
  auto* o = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  auto* tmp = scratch.data_ptr<float>();
#define LAUNCH(R)                                         \
  if (gated)                                              \
    launch<R, true>(a, b, sc, tmp, width, k, gs, stream); \
  else                                                    \
    launch<R, false>(a, b, sc, tmp, width, k, gs, stream)
  if (m == 16) {
    LAUNCH(16);
  } else if (m == 32) {
    LAUNCH(32);
  } else {
    LAUNCH(64);
  }
#undef LAUNCH
  if (gated)
    finish_n128<true>
        <<<dim3(width / 32, m / 8), 256, 0, stream>>>(o, tmp, width, m);
  else
    finish_n128<false>
        <<<dim3(width / 32, m / 8), 256, 0, stream>>>(o, tmp, width, m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY_FRAGMENT(_qwen_batch_screen, ops) {
  ops.def(
      "run(Tensor(a!) out, Tensor input, Tensor codes, Tensor scales, "
      "Tensor(b!) packed, Tensor(c!) scratch, float scale, bool gated) -> ()");
  ops.impl("run", torch::kCUDA, &run);
}
