// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only N16 gate/up tile, using the original native weight layout.
// The QPN2 implementation derives from dnv2003/v100-skinny (MIT); see
// csrc/sm70_turbomind/ops/LICENSE.v100-skinny for the retained notice.
// Included after the production translation unit by the standalone builder.
// Unlike the earlier TurboMind NAcc2 screen, this uses native codes and NAcc1.
template <int SplitK, int NAcc>
__global__ void paired16_kernel(const uint8_t* __restrict__ codes,
                                const uint8_t* __restrict__ group_scales,
                                const half* __restrict__ input,
                                half* __restrict__ output, int hidden, int k,
                                int m, float global_scale) {
  __shared__ float partials[2][SplitK][128];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int projection = (lane >> 3) & 1;
  const int quadpair = (lane >> 2) & 1;
  const int output_tile = blockIdx.x;
  const int physical_lane = (lane & ~8) | ((output_tile & 1) << 3);
  const int tile = (output_tile >> 1) + projection * (hidden >> 5);
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups = k >> 4;
  const int per_warp = groups / SplitK;
  const int begin = warp * per_warp;
  const Nvfp4Qpn2CodeReader<false> reader(codes, tile, groups, physical_lane);
  const uint8_t* scale_ptr =
      group_scales + static_cast<size_t>(tile) * groups * 32 + physical_lane;
  float accum[NAcc][8];
#pragma unroll
  for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[chain][i] = 0.0f;
  }
#pragma unroll 4
  for (int group = begin; group < begin + per_warp; ++group) {
    const uint2 packed = reader.load(group);
    const half2 scale = nvfp4_effective_scale(
        __ldg(scale_ptr + static_cast<size_t>(group) * 32), global_scale);
    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (local_row < m) {
      const half* row = input + static_cast<size_t>(local_row) * k;
      input01 = *reinterpret_cast<const uint4*>(row + group * 16);
      input23 = *reinterpret_cast<const uint4*>(row + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    VLLM_SM70_QPN2_MMA(accum[0], a0[0], a0[1], b[0], b[1]);
    VLLM_SM70_QPN2_MMA(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
    VLLM_SM70_QPN2_MMA(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
    VLLM_SM70_QPN2_MMA(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
  }
#pragma unroll
  for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[0][i] += accum[chain][i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partials[projection][warp][row * 16 + quadpair * 8 + col] = accum[0][i];
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    float gate = 0.0f, up = 0.0f;
#pragma unroll
    for (int kw = 0; kw < SplitK; ++kw) {
      gate += partials[0][kw][threadIdx.x];
      up += partials[1][kw][threadIdx.x];
    }
    const int row = threadIdx.x >> 4;
    if (row < m) {
      const float g = __half2float(__float2half(gate));
      const half silu = __float2half(g / (1.0f + expf(-g)));
      output[static_cast<size_t>(row) * hidden + output_tile * 16 +
             (threadIdx.x & 15)] = __hmul(silu, __float2half(up));
    }
  }
}
void paired16(torch::Tensor out, torch::Tensor input, torch::Tensor codes,
              torch::Tensor scales, double global_scale) {
  check_qpn2_tensors(out, input, codes, scales, true);
  TORCH_CHECK(input.size(0) <= 8 && input.size(1) % 128 == 0);
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  paired16_kernel<8, 1>
      <<<out.size(1) / 16, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()), out.size(1),
          input.size(1), input.size(0), global_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY_FRAGMENT(_qpn2_native_n16, m) {
  m.def(
      "gated(Tensor(a!) out, Tensor input, Tensor codes, Tensor scales, float "
      "global_scale) -> ()");
  m.impl("gated", torch::kCUDA, &paired16);
}
