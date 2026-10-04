// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only software load pipeline. No production dispatch is changed.
#define VLLM_NVFP4_QPN2_STANDALONE
#define VLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE
#include "../../csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"

namespace {

struct alignas(16) StageBuffers {
  unsigned codes[2][2048];
  unsigned char scales[2][1024];
};

__device__ __forceinline__ void load_stage(StageBuffers& staging,
                                           const unsigned char* codes,
                                           const unsigned char* scales,
                                           int tile, int stage, int buffer,
                                           int thread) {
  // Two producer warps load four independent 128-bit vectors before their
  // stores. The original TurboMind code layout and group scale bytes stay
  // intact.
#pragma unroll
  for (int batch = 0; batch < 2; ++batch) {
    uint4 values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int vector = thread + (batch * 4 + item) * 64;
      const int projection = vector >> 8;
      const int group_slot = (vector >> 4) & 15;
      const int group = (group_slot >> 1) * 40 + stage * 2 + (group_slot & 1);
      const int plane = (vector >> 3) & 1;
      const int column = (vector & 7) * 4;
      const size_t offset =
          (static_cast<size_t>(tile + projection * 136) * 320 * 64 +
           group * 64 + plane * 32 + column) *
          4;
      values[item] = __ldcs(reinterpret_cast<const uint4*>(codes + offset));
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int vector = thread + (batch * 4 + item) * 64;
      reinterpret_cast<uint4*>(staging.codes[buffer])[vector] = values[item];
    }
  }
  const int offset = thread * 16;
  const int projection = offset / 512;
  const int group_slot = (offset / 32) & 15;
  const int group = (group_slot >> 1) * 40 + stage * 2 + (group_slot & 1);
  const int lane = offset & 31;
  const size_t source =
      static_cast<size_t>(tile + projection * 136) * 320 * 32 + group * 32 +
      lane;
  reinterpret_cast<uint4*>(staging.scales[buffer])[thread] =
      __ldg(reinterpret_cast<const uint4*>(scales + source));
}

__global__ __launch_bounds__(576, 2) void nvfp4_gate_staged_sm70_kernel(
    const unsigned char* codes, const unsigned char* scales, const half* input,
    half* output, float global_scale) {
  __shared__ StageBuffers staging;
  __shared__ float partial[2][8][256];
  const int thread = threadIdx.x;
  const int tile = blockIdx.x;
  if (thread < 64) load_stage(staging, codes, scales, tile, 0, 0, thread);
  __syncthreads();

  const int consumer = thread - 64;
  const int lane = consumer & 31;
  const int warp = consumer >> 5;
  const int projection = warp / 8;
  const int split = warp & 7;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int column = quadpair * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  float accum[2][8] = {};
#pragma unroll 1
  for (int stage = 0; stage < 20; ++stage) {
    if (thread < 64) {
      if (stage + 1 < 20)
        load_stage(staging, codes, scales, tile, stage + 1, (stage + 1) & 1,
                   thread);
    } else {
#pragma unroll
      for (int within = 0; within < 2; ++within) {
        const int slot = split * 2 + within;
        const int base = projection * 1024 + slot * 64;
        const unsigned* stage_codes = staging.codes[stage & 1];
        const unsigned code0 = stage_codes[base + column];
        const unsigned code1 = stage_codes[base + 32 + column];
        const half2 scale = nvfp4_effective_scale(
            staging.scales[stage & 1][projection * 512 + slot * 32 + lane],
            global_scale);
        half2 weights[8];
        dequant_e2m1x8(code0, scale, weights);
        dequant_e2m1x8(code1, scale, weights + 4);
        const unsigned* b = reinterpret_cast<const unsigned*>(weights);
        const int group = split * 40 + stage * 2 + within;
        const half* x = input + row * 5120 + group * 16;
        const uint4 input0 = *reinterpret_cast<const uint4*>(x);
        const uint4 input1 = *reinterpret_cast<const uint4*>(x + 8);
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&input0);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&input1);
        VLLM_SM70_QPN2_MMA(accum[0], a0[0], a0[1], b[0], b[1]);
        VLLM_SM70_QPN2_MMA(accum[1], a0[2], a0[3], b[2], b[3]);
        VLLM_SM70_QPN2_MMA(accum[0], a1[0], a1[1], b[4], b[5]);
        VLLM_SM70_QPN2_MMA(accum[1], a1[2], a1[3], b[6], b[7]);
      }
    }
    __syncthreads();
  }
  if (thread >= 64) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int output_row = (index & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
      partial[projection][split][output_row * 32 + quadpair * 8 + output_col] =
          accum[0][index] + accum[1][index];
    }
  }
  __syncthreads();
  if (thread < 256) {
    float gate = 0.0f;
    float up = 0.0f;
#pragma unroll
    for (int split = 0; split < 8; ++split) {
      gate += partial[0][split][thread];
      up += partial[1][split][thread];
    }
    gate = __half2float(__float2half(gate));
    const half value =
        __hmul(__float2half(gate / (1.0f + expf(-gate))), __float2half(up));
    output[(thread / 32) * 4352 + tile * 32 + thread % 32] = value;
  }
}

void staged_gate(torch::Tensor out, torch::Tensor input, torch::Tensor codes,
                 torch::Tensor scales, double global_scale) {
  TORCH_CHECK(input.sizes() == torch::IntArrayRef({8, 5120}));
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({8, 4352}));
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kHalf &&
              out.scalar_type() == torch::kHalf);
  TORCH_CHECK(codes.is_cuda() && scales.is_cuda() && codes.is_contiguous() &&
              scales.is_contiguous());
  const c10::cuda::CUDAGuard guard(input.device());
  static std::once_flag configure;
  std::call_once(configure, [] {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(nvfp4_gate_staged_sm70_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared));
  });
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  nvfp4_gate_staged_sm70_kernel<<<136, 576, 0, stream>>>(
      static_cast<const unsigned char*>(codes.data_ptr()),
      static_cast<const unsigned char*>(scales.data_ptr()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<float>(global_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_nvfp4_stage_micro, ops) {
  ops.def(
      "gate(Tensor(a!) out, Tensor input, Tensor codes, Tensor scales, float "
      "global_scale) -> ()");
  ops.impl("gate", torch::kCUDA, &staged_gate);
}
