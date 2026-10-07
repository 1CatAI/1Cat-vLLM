// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// TP4 greedy top-1 without gathering logits on SM70. Each rank holds its vocab
// shard's (max logit, global id) per row and exchanges the 8-byte pair in two
// recursive-doubling hops over its direct NVLink partners (logical r^1, then
// r^2) as LL words {value, id, tag, tag}. The reduction order is a total order
// (larger value wins, ties go to the smaller id, NaN loses), so every rank ends
// with the same id as an all-gather of the shard pairs followed by a first-max
// argmax. One warp, one lane per row; a device counter supplies the epoch tag,
// so the launch is CUDA-graph replayable.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
constexpr int kRanks = 4, kRows = 32;

__device__ __forceinline__ void st4(uint4* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint4 ld4(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ bool better(float v, unsigned id, float bv,
                                       unsigned bid) {
  if (v != v) return false;
  if (bv != bv) return true;
  return v > bv || (v == bv && id < bid);
}

struct Top1Args {
  const float* pairs;  // [M, 2] (value, id as float)
  int64_t* out;        // [M]
  uint4* buf[kRanks];  // receive buffers by logical rank: [2 parity][2 hop][32]
  unsigned* seq;
  int rank, M;
};

__global__ __launch_bounds__(32) void top1x_kernel(Top1Args a) {
  const int row = threadIdx.x;
  const unsigned tag = *reinterpret_cast<volatile unsigned*>(a.seq) + 1u;
  const unsigned par = tag & 1u;
  float v = __int_as_float(0x7fc00000);
  unsigned id = 0xffffffffu;
  if (row < a.M) {
    v = a.pairs[row * 2];
    id = static_cast<unsigned>(a.pairs[row * 2 + 1]);
  }
#pragma unroll
  for (int hop = 0; hop < 2; ++hop) {
    const int partner = a.rank ^ (1 << hop);
    const int slot = (par * 2 + hop) * kRows + row;
    if (row < a.M) {
      st4(a.buf[partner] + slot, make_uint4(__float_as_uint(v), id, tag, tag));
      uint4 w;
      const long long t0 = clock64();
      do {
        w = ld4(a.buf[a.rank] + slot);
        if (clock64() - t0 > 4000000000LL) __trap();
      } while (w.z != tag || w.w != tag);
      const float ov = __uint_as_float(w.x);
      if (better(ov, w.y, v, id)) v = ov, id = w.y;
    }
  }
  if (row < a.M) a.out[row] = static_cast<int64_t>(id);
  __syncwarp();
  if (row == 0) *reinterpret_cast<volatile unsigned*>(a.seq) = tag;
}
}  // namespace

void sm70_top1x_out(torch::Tensor out, torch::Tensor pairs,
                    std::vector<int64_t> buffers, torch::Tensor seq,
                    int64_t rank) {
  const c10::cuda::CUDAGuard guard(pairs.device());
  TORCH_CHECK(pairs.dtype() == torch::kFloat32 && pairs.is_contiguous() &&
                  pairs.dim() == 2 && pairs.size(1) == 2,
              "top1x expects contiguous FP32 [M, 2] pairs");
  TORCH_CHECK(out.dtype() == torch::kInt64 && out.numel() == pairs.size(0) &&
                  buffers.size() == kRanks && seq.numel() == 1,
              "invalid top1x outputs or buffers");
  Top1Args a{};
  a.pairs = pairs.data_ptr<float>();
  a.out = out.data_ptr<int64_t>();
  for (int q = 0; q < kRanks; ++q)
    a.buf[q] = reinterpret_cast<uint4*>(buffers[q]);
  a.seq = reinterpret_cast<unsigned*>(seq.data_ptr<int>());
  a.rank = static_cast<int>(rank);
  a.M = static_cast<int>(pairs.size(0));
  TORCH_CHECK(a.M >= 1 && a.M <= kRows, "top1x supports 1..32 rows");
  top1x_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
