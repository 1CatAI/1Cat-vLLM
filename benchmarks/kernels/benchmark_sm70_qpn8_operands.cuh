// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// N16 uses the two quad-pair halves for distinct K partitions. With 24
// logical partitions and 12 physical warps this targets four CTAs/SM, without
// a separate partial-reduction kernel. This changes the FP32 reduction order.
__global__ void qpn8_output_n16(const uint8_t* codes, const half* scales,
                                const half* input, half* output) {
  __shared__ float partial[24][128];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int logical_warp = warp + ((lane >> 3) & 1) * 12;
  const int quadpair = (lane >> 2) & 1;
  const int physical_lane = (lane & ~8) | ((blockIdx.x & 1) << 3);
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int begin = logical_warp * 4;
  const uint4* weight = reinterpret_cast<const uint4*>(codes) +
                        static_cast<size_t>(blockIdx.x / 2) * 96 * 32 +
                        begin * 32 + physical_lane;
  const half* activation = input + begin * 128 + row * 16;
  float accum[8] = {};
#pragma unroll 1
  for (int group = 0; group < 4; ++group) {
    const uint4 packed = __ldcs(weight + group * 32);
    half2 decoded[8];
    fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), decoded);
    fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), decoded + 4);
    const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
    const uint4 a0 = *reinterpret_cast<const uint4*>(activation + group * 128);
    const uint4 a1 =
        *reinterpret_cast<const uint4*>(activation + group * 128 + 8);
    VLLM_SM70_MMA_8N8K4(accum, a0.x, a0.y, b[0], b[1]);
    VLLM_SM70_MMA_8N8K4(accum, a0.z, a0.w, b[2], b[3]);
    VLLM_SM70_MMA_8N8K4(accum, a1.x, a1.y, b[4], b[5]);
    VLLM_SM70_MMA_8N8K4(accum, a1.z, a1.w, b[6], b[7]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partial[logical_warp][r * 16 + quadpair * 8 + c] = accum[i];
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    float value = 0;
#pragma unroll
    for (int split = 0; split < 24; ++split)
      value += partial[split][threadIdx.x];
    const int column = blockIdx.x * 16 + (threadIdx.x & 15);
    value *= __half2float(scales[column]);
    output[(threadIdx.x >> 4) * 5120 + column] = __float2half(value);
  }
}

void run(torch::Tensor output, torch::Tensor z_out, torch::Tensor b_out,
         torch::Tensor a_out, torch::Tensor input, torch::Tensor codes,
         torch::Tensor scales, torch::Tensor ba_weight, int64_t kind,
         int64_t variant) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
              input.is_contiguous() && input.dim() == 2 && input.size(0) == 8);
  TORCH_CHECK(kind >= 0 && kind <= 3);
  const int k = input.size(1), n = kind == 0 ? 4096 : output.size(1);
  TORCH_CHECK(k == (kind == 1 ? 1536 : kind == 3 ? 4352 : 5120));
  TORCH_CHECK(n == (kind == 0 ? 4096 : kind == 2 ? 3584 : 5120));
  TORCH_CHECK(output.dim() == 2 && output.size(0) == 8 &&
              output.size(1) == (kind == 0 ? 2560 : n));
  for (const auto& t : {output, z_out, b_out, a_out, ba_weight, scales}) {
    TORCH_CHECK(t.device() == input.device() && t.is_contiguous() &&
                t.scalar_type() == torch::kFloat16);
  }
  TORCH_CHECK(codes.device() == input.device() && codes.is_contiguous() &&
              codes.scalar_type() == torch::kUInt8 &&
              codes.numel() == static_cast<int64_t>(n) * k);
  TORCH_CHECK(scales.numel() == n);
  if (kind == 0) {
    TORCH_CHECK(z_out.numel() == 8 * 1536 && b_out.numel() == 8 * 12 &&
                a_out.numel() == 8 * 12 && ba_weight.numel() == 24 * 5120);
  }
  c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* w = codes.data_ptr<uint8_t>();
  const auto* s = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  const auto* bw =
      reinterpret_cast<const half*>(ba_weight.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  auto* z = reinterpret_cast<half*>(z_out.data_ptr<at::Half>());
  auto* b = reinterpret_cast<half*>(b_out.data_ptr<at::Half>());
  auto* a = reinterpret_cast<half*>(a_out.data_ptr<at::Half>());
  switch (variant) {
    // GENERATED_LAUNCHES
    case 4: {
      TORCH_CHECK(kind == 1, "N16 screen only covers the output projection");
      static bool configured = false;
      if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            qpn8_output_n16, cudaFuncAttributePreferredSharedMemoryCarveout,
            50));
        int resident = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, qpn8_output_n16, 384, 0));
        TORCH_CHECK(resident >= 4,
                    "N16 failed the four-CTA resource gate: ", resident);
        configured = true;
      }
      qpn8_output_n16<<<320, 384, 0, stream>>>(w, s, x, y);
      break;
    }
    default:
      TORCH_CHECK(false, "Unknown FP8 operand variant");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY(_qpn8_operands700, library) {
  library.def(
      "run(Tensor(a!) output, Tensor(b!) z_out, Tensor(c!) b_out, "
      "Tensor(d!) a_out, Tensor input, Tensor codes, Tensor scales, "
      "Tensor ba_weight, int kind, int variant) -> ()");
}
TORCH_LIBRARY_IMPL(_qpn8_operands700, CUDA, library) {
  library.impl("run", &run);
}
