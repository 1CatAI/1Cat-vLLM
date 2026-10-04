// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Independent occupancy experiment; production dispatch is unchanged.
#define VLLM_NVFP4_QPN2_STANDALONE
#define VLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE
#include "../../csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"

namespace {

// Split the K reduction across small CTAs. NAcc=1 reduces the register budget;
// the larger grid can then occupy six 8-warp CTAs instead of two 16-warp CTAs.
// Weight decoding and HMMA precision are unchanged. The summation order is
// different, so model-level numerical admission is required before dispatch.
template <int Parts, int NAcc>
__global__
__launch_bounds__(256, NAcc == 1 ? 6 : 4) void nvfp4_external_split_sm70_kernel(
    const uint8_t* codes, const uint8_t* scales, const half* input,
    float* output, int width, int k, float global_scale) {
  constexpr int Warps = 8;
  __shared__ float partial[Warps][256];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int projection = blockIdx.z;
  const int tile = blockIdx.x + projection * (width / 32);
  const int part = blockIdx.y;
  const int groups = k / 16;
  const int per_warp = (groups + Parts * Warps - 1) / (Parts * Warps);
  const int begin = (part * Warps + warp) * per_warp;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const Nvfp4Qpn2CodeReader<true> reader(codes, tile, groups, lane);
  const uint8_t* scale_ptr =
      scales + static_cast<size_t>(tile) * groups * 32 + lane;
  float accum[NAcc][8] = {};
#pragma unroll 4
  for (int offset = 0; offset < per_warp; ++offset) {
    const int group = begin + offset;
    if (group < groups) {
      const uint2 packed = reader.load(group);
      const half2 scale = nvfp4_effective_scale(
          __ldg(scale_ptr + static_cast<size_t>(group) * 32), global_scale);
      half2 weights[8];
      dequant_e2m1x8(packed.x, scale, weights);
      dequant_e2m1x8(packed.y, scale, weights + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(weights);
      const half* x = input + static_cast<size_t>(row) * k + group * 16;
      const uint4 input01 = *reinterpret_cast<const uint4*>(x);
      const uint4 input23 = *reinterpret_cast<const uint4*>(x + 8);
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      VLLM_SM70_QPN2_MMA(accum[0], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_QPN2_MMA(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_QPN2_MMA(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_QPN2_MMA(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }
#pragma unroll
  for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[0][i] += accum[chain][i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partial[warp][r * 32 + ((lane >> 2) & 3) * 8 + c] = accum[0][i];
  }
  __syncthreads();
  const int e = threadIdx.x;
  float sum = 0.0f;
#pragma unroll
  for (int w = 0; w < Warps; ++w) sum += partial[w][e];
  const size_t position =
      ((static_cast<size_t>(projection) * Parts + part) * 8 + e / 32) * width +
      blockIdx.x * 32 + e % 32;
  output[position] = sum;
}

template <int Parts, bool Gated>
__global__ void nvfp4_external_epilogue_sm70_kernel(const float* partial,
                                                    half* out, int width) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 8 * width) return;
  float value = 0.0f;
  float up = 0.0f;
#pragma unroll
  for (int part = 0; part < Parts; ++part) {
    value += partial[static_cast<size_t>(part) * 8 * width + i];
    if constexpr (Gated) {
      up += partial[static_cast<size_t>(Parts + part) * 8 * width + i];
    }
  }
  half result = __float2half(value);
  if constexpr (Gated) {
    const float gate = __half2float(result);
    result =
        __hmul(__float2half(gate / (1.0f + expf(-gate))), __float2half(up));
  }
  out[i] = result;
}

template <int Parts, int NAcc, bool Gated>
void launch(torch::Tensor out, torch::Tensor partial, torch::Tensor input,
            torch::Tensor codes, torch::Tensor scales, double global_scale) {
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int width = out.size(1);
  const int k = input.size(1);
  nvfp4_external_split_sm70_kernel<Parts, NAcc>
      <<<dim3(width / 32, Parts, Gated ? 2 : 1), 256, 0, stream>>>(
          codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          partial.data_ptr<float>(), width, k,
          static_cast<float>(global_scale));
  nvfp4_external_epilogue_sm70_kernel<Parts, Gated>
      <<<(8 * width + 255) / 256, 256, 0, stream>>>(
          partial.data_ptr<float>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()), width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void split(torch::Tensor out, torch::Tensor partial, torch::Tensor input,
           torch::Tensor codes, torch::Tensor scales, double global_scale,
           bool gated, int64_t parts, int64_t chains) {
  check_qpn2_tensors(out, input, codes, scales, gated);
  TORCH_CHECK(input.size(0) == 8);
  TORCH_CHECK(partial.is_cuda() && partial.device() == input.device() &&
              partial.scalar_type() == torch::kFloat32 &&
              partial.is_contiguous());
  TORCH_CHECK(partial.numel() == (gated ? 2 : 1) * parts * out.numel());
  TORCH_CHECK((parts == 1 && chains == 2) || (parts == 4 && chains == 1));
  const c10::cuda::CUDAGuard guard(input.device());
  if (parts == 1) {
    if (gated)
      launch<1, 2, true>(out, partial, input, codes, scales, global_scale);
    else
      launch<1, 2, false>(out, partial, input, codes, scales, global_scale);
  } else {
    if (gated)
      launch<4, 1, true>(out, partial, input, codes, scales, global_scale);
    else
      launch<4, 1, false>(out, partial, input, codes, scales, global_scale);
  }
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_nvfp4_split_micro, ops) {
  ops.def(
      "run(Tensor(a!) out, Tensor(b!) partial, Tensor input, Tensor codes, "
      "Tensor scales, float global_scale, bool gated, int parts, "
      "int chains) -> ()");
  ops.impl("run", torch::kCUDA, &split);
}
