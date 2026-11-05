// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// v41pidx: DS-V4.1 lightning-indexer prefill scores for one TP
// rank, fused on Volta's mma.m8n8k4 tensor cores. Behind
// VLLM_V41_INDEX_FUSE (vllm/models/deepseek_v41/index_fuse.py), default off.
//
// The path it replaces (attention_ops.select_compressed_positions, per
// all-reduce tile of R query rows, H = 8 index heads per rank on TP4):
//
//   s = torch.mm(q[R*H, 128], keys[N, 128].T)   fp16 out, fp32 HMMA accumulate
//   s.relu_().mul_(w[:, :, None])                fp16, product rounded once
//   p = s.sum(1)                                 fp32 accumulate, fp16 out
//
// writes the [R, H, N] fp16 tile to memory and reads it back three times.
// index_head_sum computes the same p[R, N] with the per-head scores held in
// registers:
//
// - each head's 128-long dot product is one chain of 32 m8n8k4 MMAs over the
//   K quads 0..31 in order, FP32 accumulate from +0, rounded once to fp16
//   (cvt.rn): the HMMA.884 instruction sequence of a cuBLAS s884 kernel with
//   K = 128. Whether cuBLAS feeds the quads in this order is a GPU question;
//   index_head_scores writes these per-head fp16 scores so the GPU check can
//   compare them with torch.mm bit for bit.
// - relu then the product with the head weight in FP32, rounded once to
//   fp16: what PyTorch's CUDA relu_ (clamp_min in FP32) and mul_ (opmath
//   FP32, one rounding) compute.
// - the head sum in FP32 from +0 in one of four orders (ORDER below; the
//   reduction config PyTorch picks for a [R, 8, N] sum over dim 1 decides
//   which one it uses), rounded once to fp16.
//
// Columns at or past `live` (keys that do not exist yet; today they score
// uninitialized rows and top-k never reads them) are written as +0.
//
// Fragment mapping of the full-warp m8n8k4 (8 rows x 32 columns x K 4 per
// warp instruction, quad pair p = lane bits 2-3 owns columns 8p..8p+7): the
// mapping proven by benchmarks/csrc/sm70_raw_hmma_qk_panel_micro.cu
// (raw_a_row, raw_b_col_n, raw_c_row, raw_c_col). A rows are 8 query tokens
// of one head, B columns are 32 keys; each head has its own accumulators, so
// the head sum needs no shuffles: a thread holds all 8 heads of its outputs.
//
// Upstream: the fused-scorer idea is 1Cat-vLLM #1280 (SM75 m16n8k8 scorer
// for DeepSeek-V4, Peuqui) and #1282 (sm7x tensor-core Triton scorer), and
// philbert440/deepseek-v4.1-v100 csrc/ds41_indexer_fused_sm70.cu (WMMA scorer
// + top-512, near-tie). #1347 notes that none of them reach V4.1's
// attention_ops path. This kernel is new code for that path and keeps the
// per-rank partial bitwise-targeted instead of near-tie.

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace v41pidx {

constexpr int kHeads = 8;    // index heads per TP rank (32 / TP4)
constexpr int kDim = 128;    // index head dim
constexpr int kQuads = kDim / 4;
constexpr int kTok = 8;      // query tokens per CTA = A rows of the MMA
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kWarpKeys = 32;  // B columns of one warp MMA
constexpr int kPasses = 4;
constexpr int kCtaKeys = kWarps * kWarpKeys * kPasses;  // 1024

enum Order : int { kSeq = 0, kIlp2 = 1, kIlp4 = 2, kTree = 3 };

__device__ __forceinline__ void mma884(float (&d)[8], uint32_t a0, uint32_t a1,
                                       uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
      "{%0, %1, %2, %3, %4, %5, %6, %7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
        "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }

// FP32 head sum of the eight fp16-valued products v[h], from PyTorch's +0
// identity, in PyTorch's reduce orders (Reduce.cuh thread_reduce_impl with
// vt0 interleaved accumulators, then block_y_reduce halving offsets):
//   kSeq  : one accumulator (vt0 1; vt0 8 adds the same values in the same order)
//   kIlp2 : vt0 2, block.y 1 (block.y 2 with vt0 4 adds in this same order)
//   kIlp4 : vt0 4, block.y 1
//   kTree : block.y 4 (or 8): ((v0+v4) + (v2+v6)) + ((v1+v5) + (v3+v7))
// Every leaf starts as 0 + v (sign of zero as PyTorch's identity leaves it).
template <int ORDER>
__device__ __forceinline__ float head_sum(const float (&v)[kHeads]) {
  float z[kHeads];
#pragma unroll
  for (int h = 0; h < kHeads; ++h) z[h] = add(0.0f, v[h]);
  if constexpr (ORDER == kSeq) {
    float t = z[0];
#pragma unroll
    for (int h = 1; h < kHeads; ++h) t = add(t, v[h]);
    return t;
  } else if constexpr (ORDER == kIlp2) {
    float a0 = z[0], a1 = z[1];
#pragma unroll
    for (int h = 2; h < kHeads; h += 2) {
      a0 = add(a0, v[h]);
      a1 = add(a1, v[h + 1]);
    }
    return add(a0, a1);
  } else if constexpr (ORDER == kIlp4) {
    const float a0 = add(z[0], v[4]), a1 = add(z[1], v[5]);
    const float a2 = add(z[2], v[6]), a3 = add(z[3], v[7]);
    return add(add(add(a0, a1), a2), a3);
  } else {
    static_assert(ORDER == kTree, "unknown head-sum order");
    const float u0 = add(z[0], z[4]), u1 = add(z[1], z[5]);
    const float u2 = add(z[2], z[6]), u3 = add(z[3], z[7]);
    return add(add(u0, u2), add(u1, u3));
  }
}

// grid (ceil(rows / 8), ceil(width / 1024)): x = token group (consecutive
// CTAs share a key range, so its keys come from L2), y = key range.
// HEADS_OUT: write the per-head fp16 scores [rows, 8, width] instead (GPU
// check of the MMA chain against torch.mm; no epilogue).
template <int ORDER, bool HEADS_OUT>
__global__ void __launch_bounds__(kThreads, 2)
    head_sum_kernel(half* __restrict__ out, const half* __restrict__ q,
                    const half* __restrict__ w, const half* __restrict__ keys,
                    int rows, int width, int live, int64_t out_stride) {
  // q of the CTA's 8 tokens as K-quad words: [head][quad][token], 16 KiB.
  __shared__ __align__(16) uint2 sq[kHeads * kQuads * kTok];
  const int tok0 = blockIdx.x * kTok;
  const int key_base = blockIdx.y * kCtaKeys;
  for (int i = threadIdx.x; i < kTok * kHeads * kQuads; i += kThreads) {
    const int quad = i % kQuads;
    const int h = (i / kQuads) % kHeads;
    const int t = i / (kQuads * kHeads);
    uint2 v = make_uint2(0u, 0u);
    if (tok0 + t < rows) {
      v = __ldg(reinterpret_cast<const uint2*>(
                    q + (static_cast<int64_t>(tok0 + t) * kHeads + h) * kDim) +
                quad);
    }
    sq[(h * kQuads + quad) * kTok + t] = v;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int a_row = (lane & 3) + ((lane >> 4) << 2);
  const int b_col = ((lane & 12) << 1) + (lane & 3) + ((lane >> 4) << 2);
  const int c_row0 = (lane & 1) + ((lane >> 4) << 2);  // + (r & 2)
  const int c_col0 = ((lane & 12) << 1) + (lane & 2);  // + (r & 4) + (r & 1)

  float wv[2][kHeads];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int t = tok0 + c_row0 + 2 * j;
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
      wv[j][h] = (!HEADS_OUT && t < rows)
                     ? __half2float(w[static_cast<int64_t>(t) * kHeads + h])
                     : 0.0f;
    }
  }

  for (int pass = 0; pass < kPasses; ++pass) {
    const int n0 = key_base + (pass * kWarps + warp) * kWarpKeys;
    if (n0 >= width) break;  // warp-uniform
    float acc[kHeads][8];
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
#pragma unroll
      for (int r = 0; r < 8; ++r) acc[h][r] = 0.0f;
    }
    const int key = n0 + b_col;
    const bool key_ok = key < live;
    const uint4* kp = reinterpret_cast<const uint4*>(
        keys + static_cast<int64_t>(key_ok ? key : 0) * kDim);
#pragma unroll 2
    for (int quad = 0; quad < kQuads; quad += 2) {
      // K quads `quad` and `quad + 1` of this lane's key column.
      const uint4 kb = key_ok ? __ldg(kp + quad / 2) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int h = 0; h < kHeads; ++h) {
        const uint2 a0 = sq[(h * kQuads + quad) * kTok + a_row];
        const uint2 a1 = sq[(h * kQuads + quad + 1) * kTok + a_row];
        mma884(acc[h], a0.x, a0.y, kb.x, kb.y);
        mma884(acc[h], a1.x, a1.y, kb.z, kb.w);
      }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int row = c_row0 + (r & 2);
      const int col = n0 + c_col0 + (r & 4) + (r & 1);
      const int t = tok0 + row;
      if (t >= rows || col >= width) continue;
      if constexpr (HEADS_OUT) {
#pragma unroll
        for (int h = 0; h < kHeads; ++h) {
          out[(static_cast<int64_t>(t) * kHeads + h) * out_stride + col] =
              __float2half_rn(acc[h][r]);
        }
      } else {
        half result = __float2half_rn(0.0f);
        if (col < live) {
          float v[kHeads];
#pragma unroll
          for (int h = 0; h < kHeads; ++h) {
            const float s = __half2float(__float2half_rn(acc[h][r]));
            const float relu = fmaxf(s, 0.0f);
            v[h] = __half2float(
                __float2half_rn(__fmul_rn(relu, wv[(r >> 1) & 1][h])));
          }
          result = __float2half_rn(head_sum<ORDER>(v));
        }
        out[static_cast<int64_t>(t) * out_stride + col] = result;
      }
    }
  }
}

template <int ORDER, bool HEADS_OUT>
void launch(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& w,
            const torch::Tensor& keys, int rows, int width, int live,
            int64_t out_stride) {
  const dim3 grid((rows + kTok - 1) / kTok, (width + kCtaKeys - 1) / kCtaKeys);
  const auto stream = at::cuda::getCurrentCUDAStream();
  head_sum_kernel<ORDER, HEADS_OUT><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const half*>(q.data_ptr()),
      HEADS_OUT ? nullptr : reinterpret_cast<const half*>(w.data_ptr()),
      reinterpret_cast<const half*>(keys.data_ptr()), rows, width, live,
      out_stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_common(const torch::Tensor& q, const torch::Tensor& keys,
                  int64_t live, int64_t width) {
  TORCH_CHECK(q.is_cuda() && keys.is_cuda(), "v41pidx: CUDA tensors only");
  TORCH_CHECK(q.scalar_type() == at::kHalf && keys.scalar_type() == at::kHalf,
              "v41pidx: fp16 q and keys");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeads && q.size(2) == kDim &&
                  q.is_contiguous(),
              "v41pidx: q must be contiguous [rows, 8, 128]");
  TORCH_CHECK(keys.dim() == 2 && keys.size(1) == kDim && keys.is_contiguous(),
              "v41pidx: keys must be contiguous [n, 128]");
  TORCH_CHECK(live >= 0 && live <= width && live <= keys.size(0),
              "v41pidx: live must be in [0, min(width, keys rows)]");
  TORCH_CHECK(width <= (int64_t{65535} * kCtaKeys) && q.size(0) <= (int64_t{1} << 30),
              "v41pidx: problem too large for the grid");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(keys.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(q.data_ptr()) % 8 == 0,
              "v41pidx: keys need 16-byte and q 8-byte alignment");
}

}  // namespace v41pidx

// out[r, c] = fp16 sum_h fp16(relu(fp16(q[r, h] . keys[c])) * w[r, h]) for
// c < live (head-sum order `order`, see head_sum), +0 for live <= c < width.
void v41pidx_index_head_sum(torch::Tensor out, torch::Tensor q, torch::Tensor w,
                           torch::Tensor keys, int64_t order, int64_t live) {
  using namespace v41pidx;
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf && out.dim() == 2 &&
                  out.stride(1) == 1 && out.size(0) == q.size(0),
              "v41pidx: out must be fp16 [rows, width] with unit column stride");
  check_common(q, keys, live, out.size(1));
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kHalf && w.dim() == 2 &&
                  w.size(0) == q.size(0) && w.size(1) == kHeads && w.is_contiguous(),
              "v41pidx: w must be contiguous fp16 [rows, 8]");
  TORCH_CHECK(out.stride(0) >= out.size(1), "v41pidx: overlapping out rows");
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const int rows = static_cast<int>(q.size(0));
  const int width = static_cast<int>(out.size(1));
  if (rows == 0 || width == 0) return;
  const int lv = static_cast<int>(live);
  const int64_t stride = out.stride(0);
  switch (order) {
    case kSeq: launch<kSeq, false>(out, q, w, keys, rows, width, lv, stride); break;
    case kIlp2: launch<kIlp2, false>(out, q, w, keys, rows, width, lv, stride); break;
    case kIlp4: launch<kIlp4, false>(out, q, w, keys, rows, width, lv, stride); break;
    case kTree: launch<kTree, false>(out, q, w, keys, rows, width, lv, stride); break;
    default: TORCH_CHECK(false, "v41pidx: order must be 0..3, got ", order);
  }
}

// out[r, h, c] = fp16(q[r, h] . keys[c]) for c < live (the MMA chain alone;
// GPU check against torch.mm). Columns live <= c < width are written +0.
void v41pidx_index_head_scores(torch::Tensor out, torch::Tensor q,
                              torch::Tensor keys, int64_t live) {
  using namespace v41pidx;
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf && out.dim() == 3 &&
                  out.size(0) == q.size(0) && out.size(1) == kHeads &&
                  out.is_contiguous(),
              "v41pidx: out must be contiguous fp16 [rows, 8, width]");
  check_common(q, keys, live, out.size(2));
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const int rows = static_cast<int>(q.size(0));
  const int width = static_cast<int>(out.size(2));
  if (rows == 0 || width == 0) return;
  launch<kSeq, true>(out, q, q, keys, rows, width, static_cast<int>(live), width);
}

TORCH_LIBRARY(v41pidx, m) {
  m.def(
      "index_head_sum(Tensor(a!) out, Tensor q, Tensor w, Tensor keys, "
      "int order, int live) -> ()",
      &v41pidx_index_head_sum);
  m.def(
      "index_head_scores(Tensor(a!) out, Tensor q, Tensor keys, int live) -> ()",
      &v41pidx_index_head_scores);
}
