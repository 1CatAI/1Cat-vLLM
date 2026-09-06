// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Benchmark-only: colocate neighbouring N tiles of one expert within a CTA.
// Reuse the production plan/reduction and an isolated reference namespace.
#define _C _C_moe_locality_reference
#include "../../csrc/sm70_turbomind/ops/nvfp4_grouped_decode_sm70.cu"
#undef _C

namespace {
template <int WarpTiles>
__global__ void locality_w2_kernel(const half* x, const uint32_t* weights,
                                   const half* scales, const int32_t* rows,
                                   const int32_t* experts, const int32_t* sizes,
                                   const int32_t* total, half* out) {
  const int warp = threadIdx.x / 32;
  const int group = blockIdx.y * (4 / WarpTiles) + warp / WarpTiles;
  const int tile = blockIdx.x * WarpTiles + warp % WarpTiles;
  if (group >= *total) return;
  const int count = sizes[group], expert = experts[group];
  const int lane = threadIdx.x % 32, quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0), col = quad * 8 + r;
  const int route = r < count ? rows[group * kPack + r] : 0;
  float accum[8] = {};
  if (expert < kExperts) {
    const uint32_t* w = weights + static_cast<size_t>(expert) * 160 * 320;
    const half* s = scales + static_cast<size_t>(expert) * 10 * 2560;
    const half* input = x + static_cast<size_t>(route) * 160;
#pragma unroll
    for (int g = 0; g < 10; ++g) {
      const int offset = (tile * 20 + g * 2) * 32 + col;
      const half scalar = __hmul(__ldg(s + (g * 80 + tile) * 32 + col),
                                 __float2half_rn(16384.0f));
      const half2 scale = __halves2half2(scalar, scalar);
      half2 decoded[8];
      decode(__ldcs(w + offset), scale, decoded);
      decode(__ldcs(w + offset + 32), scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
      uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
      if (r < count) {
        lo = *reinterpret_cast<const uint4*>(input + g * 16);
        hi = *reinterpret_cast<const uint4*>(input + g * 16 + 8);
      }
      PACKED_MMA(accum, lo.x, lo.y, b[0], b[1]);
      PACKED_MMA(accum, lo.z, lo.w, b[2], b[3]);
      PACKED_MMA(accum, hi.x, hi.y, b[4], b[5]);
      PACKED_MMA(accum, hi.z, hi.w, b[6], b[7]);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    if (row < count) {
      const int dst = rows[group * kPack + row];
      out[static_cast<size_t>(dst) * 2560 + tile * 32 + quad * 8 + c] =
          __float2half_rn(accum[i]);
    }
  }
}

void locality_w2(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
                 torch::Tensor w, torch::Tensor s, torch::Tensor topk,
                 torch::Tensor rows, torch::Tensor experts, torch::Tensor sizes,
                 torch::Tensor total, int64_t warp_tiles) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 160 && x.size(0) % 10 == 0);
  const int routes = x.size(0), tokens = routes / 10;
  TORCH_CHECK(tokens >= 1 && tokens <= 16);
  for (const auto& t :
       {out, routed, x, w, s, topk, rows, experts, sizes, total})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  for (const auto& t : {out, routed, x, s})
    TORCH_CHECK(t.scalar_type() == at::kHalf);
  for (const auto& t : {w, rows, experts, sizes, total})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(topk.scalar_type() == at::kFloat && topk.numel() == routes &&
              out.numel() == tokens * 2560 && routed.numel() == routes * 2560 &&
              rows.numel() >= routes * 8 && experts.numel() >= routes &&
              sizes.numel() >= routes && total.numel() == 1 &&
              w.numel() == 512 * 160 * 320 && s.numel() == 512 * 10 * 2560);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
#define LAUNCH(W)                                                         \
  locality_w2_kernel<W>                                                   \
      <<<dim3(80 / W, (routes + 4 / W - 1) / (4 / W)), 128, 0, stream>>>( \
          reinterpret_cast<const half*>(x.data_ptr()),                    \
          reinterpret_cast<const uint32_t*>(w.data_ptr()),                \
          reinterpret_cast<const half*>(s.data_ptr()),                    \
          rows.data_ptr<int32_t>(), experts.data_ptr<int32_t>(),          \
          sizes.data_ptr<int32_t>(), total.data_ptr<int32_t>(),           \
          reinterpret_cast<half*>(routed.data_ptr()));
  switch (warp_tiles) {
    case 1:
      LAUNCH(1);
      break;
    case 2:
      LAUNCH(2);
      break;
    case 4:
      LAUNCH(4);
      break;
    default:
      TORCH_CHECK(false, "warp_tiles must be 1, 2 or 4");
  }
#undef LAUNCH
  reduce_kernel<<<dim3(10, tokens), 256, 0, stream>>>(
      reinterpret_cast<const half*>(routed.data_ptr()), topk.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int W>
std::vector<int64_t> resource_info() {
  cudaFuncAttributes attr;
  AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, locality_w2_kernel<W>));
  int blocks = 0;
  AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, locality_w2_kernel<W>, 128, 0));
  return {attr.numRegs, static_cast<int64_t>(attr.sharedSizeBytes),
          static_cast<int64_t>(attr.localSizeBytes), blocks};
}

std::vector<int64_t> resources(int64_t warp_tiles) {
  switch (warp_tiles) {
    case 1:
      return resource_info<1>();
    case 2:
      return resource_info<2>();
    case 4:
      return resource_info<4>();
    default:
      TORCH_CHECK(false, "warp_tiles must be 1, 2 or 4");
  }
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_moe_locality, m) {
  m.def(
      "w2(Tensor(a!) out, Tensor(b!) routed, Tensor x, Tensor w, Tensor s, "
      "Tensor topk, Tensor rows, Tensor experts, Tensor sizes, Tensor total, "
      "int warp_tiles) -> ()");
  m.def("resources(int warp_tiles) -> int[]", &resources);
}
TORCH_LIBRARY_IMPL(_C_moe_locality, CUDA, m) { m.impl("w2", &locality_w2); }
