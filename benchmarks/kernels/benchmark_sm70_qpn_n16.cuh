// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research tile: more resident warps after reducing operand/register overhead.
// The K partition and accumulator order differ from the production kernel.

template <bool Gated>
__global__ void operand_n16(const uint8_t* codes, const uint8_t* scales,
                            const half* input, half* output,
                            float global_scale) {
  constexpr int K = Gated ? 5120 : 4352;
  constexpr int Groups = K / 16;
  constexpr int Splits = Gated ? 12 : 24;
  __shared__ float partial[24][128];
  const int lane = threadIdx.x & 31;
  const int physical_warp = threadIdx.x >> 5;
  const int half_warp = (lane >> 3) & 1;
  const int projection = Gated ? half_warp : 0;
  const int warp = Gated ? physical_warp : physical_warp + half_warp * 12;
  const int slot = Gated ? projection * 12 + warp : warp;
  const int quadpair = (lane >> 2) & 1;
  const int physical_lane = (lane & ~8) | ((blockIdx.x & 1) << 3);
  const int tile = blockIdx.x / 2 + projection * 136;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int begin = Groups * warp / Splits;
  const int end = Groups * (warp + 1) / Splits;
  const Nvfp4Qpn2CodeReader<false> reader(codes, tile, Groups, physical_lane);
  const uint8_t* scale_ptr = scales + tile * Groups * 32 + physical_lane;
  float accum[8] = {};
#pragma unroll 1
  for (int group = begin; group < end; ++group) {
    const uint2 packed = reader.load(group);
    const half2 scale = __hmul2(
        nvfp4_effective_scale(__ldg(scale_ptr + group * 32), global_scale),
        __float2half2_rn(16384.f));
    half2 decoded[8];
    range_decode(packed.x, scale, decoded);
    range_decode(packed.y, scale, decoded + 4);
    const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
    const half* activation = input + group * 128 + row * 16;
    const uint4 a0 = *reinterpret_cast<const uint4*>(activation);
    const uint4 a1 = *reinterpret_cast<const uint4*>(activation + 8);
    VLLM_SM70_QPN2_MMA(accum, a0.x, a0.y, b[0], b[1]);
    VLLM_SM70_QPN2_MMA(accum, a0.z, a0.w, b[2], b[3]);
    VLLM_SM70_QPN2_MMA(accum, a1.x, a1.y, b[4], b[5]);
    VLLM_SM70_QPN2_MMA(accum, a1.z, a1.w, b[6], b[7]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partial[slot][r * 16 + quadpair * 8 + c] = accum[i];
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    float value = 0, up = 0;
#pragma unroll 4
    for (int split = 0; split < Splits; ++split) {
      value += partial[split][threadIdx.x];
      if constexpr (Gated) up += partial[split + 12][threadIdx.x];
    }
    if constexpr (Gated) {
      const float g = __half2float(__float2half(value));
      const half silu = __float2half(g / (1.0f + expf(-g)));
      // Direct K16 publication for the down projection.
      output[blockIdx.x * 128 + threadIdx.x] = __hmul(silu, __float2half(up));
    } else {
      output[(threadIdx.x >> 4) * 5120 + blockIdx.x * 16 + (threadIdx.x & 15)] =
          __float2half(value);
    }
  }
}
