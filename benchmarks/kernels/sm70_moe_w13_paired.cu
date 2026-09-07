// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Benchmark only. A warp reuses one A fragment for gate and up projections.
// Preserve both independent accumulation sequences and the ordered Split-K sum.
#define _C _C_moe_pair_reference
#include "../../csrc/sm70_turbomind/ops/nvfp4_grouped_decode_sm70.cu"
#undef _C
namespace {
template <int Split, bool Interleaved>
__global__ void paired_w13_kernel(const half* x, const uint32_t* weights,
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
  const int split = warp;
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + mma_row;
  const int route = mma_row < count ? rows[group_id * kPack + mma_row] : 0;
  float accum[2][8] = {};
  if (expert < kExperts) {
    const uint32_t* w = weights + static_cast<size_t>(expert) * 2560 * 40;
    const half* s = scales + static_cast<size_t>(expert) * 160 * 320;
    const half* input = x + static_cast<size_t>(route / 10) * 2560;
#pragma unroll 4
    for (int g = split * (160 / Split); g < (split + 1) * (160 / Split); ++g) {
      uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
      if (mma_row < count) {
        lo = *reinterpret_cast<const uint4*>(input + g * 16);
        hi = *reinterpret_cast<const uint4*>(input + g * 16 + 8);
      }

#pragma unroll
      for (int projection = 0; projection < 2; ++projection) {
        const int tile = Interleaved ? blockIdx.x * 2 + projection
                                     : blockIdx.x + projection * 5;

        const size_t offset =
            (static_cast<size_t>(tile) * 320 + g * 2) * 32 + col;
        const half scalar = __hmul(__ldg(s + (g * 10 + tile) * 32 + col),
                                   __float2half_rn(16384.0f));
        const half2 scale = __halves2half2(scalar, scalar);
        half2 decoded[8];
        decode(__ldcs(w + offset), scale, decoded);
        decode(__ldcs(w + offset + 32), scale, decoded + 4);
        const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
        PACKED_MMA(accum[projection], lo.x, lo.y, b[0], b[1]);
        PACKED_MMA(accum[projection], lo.z, lo.w, b[2], b[3]);
        PACKED_MMA(accum[projection], hi.x, hi.y, b[4], b[5]);
        PACKED_MMA(accum[projection], hi.z, hi.w, b[6], b[7]);
      }
    }
  }
#pragma unroll
  for (int projection = 0; projection < 2; ++projection) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partial[projection][split][r][quad * 8 + c] = accum[projection][i];
    }
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

void paired_w13(torch::Tensor out, torch::Tensor x, torch::Tensor w,
                torch::Tensor s, torch::Tensor ids, torch::Tensor rows,
                torch::Tensor experts, torch::Tensor sizes, torch::Tensor total,
                int64_t split, bool interleaved) {
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
  paired_w13_kernel<S, I><<<dim3(5, routes), 32 * S, 0, stream>>>(           \
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

template <int S>
std::vector<int64_t> resource_info() {
  cudaFuncAttributes attr;
  AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, paired_w13_kernel<S, true>));
  int blocks = 0;
  AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, paired_w13_kernel<S, true>, 32 * S, 0));
  return {attr.numRegs, static_cast<int64_t>(attr.sharedSizeBytes),
          static_cast<int64_t>(attr.localSizeBytes), blocks};
}
std::vector<int64_t> resources(int64_t split) {
  switch (split) {
    case 4:
      return resource_info<4>();
    case 5:
      return resource_info<5>();
    case 8:
      return resource_info<8>();
    default:
      TORCH_CHECK(false, "resource query supports production screen splits");
  }
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(_C_moe_pair, m) {
  m.def(
      "run(Tensor(a!) out, Tensor x, Tensor w, Tensor s, Tensor ids, "
      "Tensor(b!) rows, Tensor(c!) experts, Tensor(d!) sizes, Tensor(e!) "
      "total, "
      "int split, bool interleaved) -> ()");
  m.def("resources(int split) -> int[]", &resources);
}
TORCH_LIBRARY_IMPL(_C_moe_pair, CUDA, m) { m.impl("run", &paired_w13); }
