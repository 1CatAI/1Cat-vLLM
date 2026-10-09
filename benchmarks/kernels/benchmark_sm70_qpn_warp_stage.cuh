// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only loader/decoder and HMMA warp partition. One named barrier
// per consumer alternates full and empty phases; barrier 0 is reserved for
// the final whole-CTA reduction. No inter-CTA synchronization is used.

__device__ __forceinline__ void qpn_stage_sync(int consumer) {
  asm volatile("bar.sync %0, 64;" : : "r"(consumer + 1) : "memory");
}

__global__ __launch_bounds__(384) void operand_warp_stage_gate(
    const uint8_t* codes, const uint8_t* scales, const half* input,
    half* output, float global_scale) {
  constexpr int Groups = 320, Splits = 8, GroupsPerSplit = Groups / Splits;
  __shared__ uint4 staged[8][4][32];
  __shared__ float partial[2][8][128];
  const int lane = threadIdx.x & 31;
  const int physical_warp = threadIdx.x >> 5;
  const int projection = (lane >> 3) & 1;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  if (physical_warp < 4) {
    const int physical_lane = (lane & ~8) | ((blockIdx.x & 1) << 3);
    const int tile = blockIdx.x / 2 + projection * 136;
    const Nvfp4Qpn2CodeReader<false> reader(codes, tile, Groups, physical_lane);
    const uint8_t* scale_ptr = scales + tile * Groups * 32 + physical_lane;
#pragma unroll 1
    for (int step = 0; step < GroupsPerSplit; ++step) {
#pragma unroll
      for (int owned = 0; owned < 2; ++owned) {
        const int consumer = physical_warp + owned * 4;
        const int group = consumer * GroupsPerSplit + step;
        const uint2 packed = reader.load(group);
        const half2 scale =
            nvfp4_effective_scale(__ldg(scale_ptr + group * 32), global_scale);
        half2 decoded[8];
        dequant_e2m1x8(packed.x, scale, decoded);
        dequant_e2m1x8(packed.y, scale, decoded + 4);
        const half* a = input + group * 128 + row * 16;
        const uint4 a0 = *reinterpret_cast<const uint4*>(a);
        const uint4 a1 = *reinterpret_cast<const uint4*>(a + 8);
        const uint4 b0 = *reinterpret_cast<const uint4*>(decoded);
        const uint4 b1 = *reinterpret_cast<const uint4*>(decoded + 4);
        // The producer prepares the next operands while this consumer runs
        // HMMA on the previous group. Do not overwrite live shared operands.
        if (step) qpn_stage_sync(consumer);
        staged[consumer][0][lane] = b0;
        staged[consumer][1][lane] = b1;
        staged[consumer][2][lane] = a0;
        staged[consumer][3][lane] = a1;
        qpn_stage_sync(consumer);
      }
    }
    qpn_stage_sync(physical_warp);
    qpn_stage_sync(physical_warp + 4);
  } else {
    const int consumer = physical_warp - 4;
    float accum[8] = {};
#pragma unroll 1
    for (int step = 0; step < GroupsPerSplit; ++step) {
      qpn_stage_sync(consumer);
      const uint4 b0 = staged[consumer][0][lane];
      const uint4 b1 = staged[consumer][1][lane];
      const uint4 a0 = staged[consumer][2][lane];
      const uint4 a1 = staged[consumer][3][lane];
      VLLM_SM70_QPN2_MMA(accum, a0.x, a0.y, b0.x, b0.y);
      VLLM_SM70_QPN2_MMA(accum, a0.z, a0.w, b0.z, b0.w);
      VLLM_SM70_QPN2_MMA(accum, a1.x, a1.y, b1.x, b1.y);
      VLLM_SM70_QPN2_MMA(accum, a1.z, a1.w, b1.z, b1.w);
      qpn_stage_sync(consumer);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
      const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partial[projection][consumer][r * 16 + ((lane >> 2) & 1) * 8 + c] =
          accum[i];
    }
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    float gate = 0, up = 0;
#pragma unroll
    for (int split = 0; split < Splits; ++split) {
      gate += partial[0][split][threadIdx.x];
      up += partial[1][split][threadIdx.x];
    }
    const float g = __half2float(__float2half(gate));
    const half silu = __float2half(g / (1.f + expf(-g)));
    output[blockIdx.x * 128 + threadIdx.x] = __hmul(silu, __float2half(up));
  }
}
