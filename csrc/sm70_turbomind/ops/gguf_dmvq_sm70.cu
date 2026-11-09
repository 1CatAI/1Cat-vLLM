// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Based on the supplied dmvq.cu temporary original-record reader.
// dense_mv2 skeleton driving the native GGUF pair readers (all IQ/K formats).
// CTA = one N32 tile; its W warps take interleaved K128 groups (g = g0 + warp,
// step W) so neighbouring warps stream neighbouring records. Activations go
// through warp-private, double-buffered shared slots filled with coalesced
// loads; the next record is in flight while the current one is decoded.
// Partials reduce across warps in shared memory and across split-K CTAs by the
// last CTA in fixed order.
#include "gguf_pair_shared_a_sm70.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <torch/all.h>

namespace vllm::sm70_gguf {

template <class R>
__global__ void dq_book_dump(uint8_t* dst) {
  extern __shared__ uint4 smem_raw[];
  uint8_t* book = reinterpret_cast<uint8_t*>(smem_raw);
  R::initialize(book);
  __syncthreads();
  for (int i = threadIdx.x; i < R::kBookBytes; i += blockDim.x)
    dst[i] = book[i];
}

template <class R>
void prepare_book(torch::Tensor table) {
  TORCH_CHECK(table.numel() >= R::kBookBytes, "DMVQ codebook is too small");
  if constexpr (R::kBookBytes > 0) {
    dq_book_dump<R>
        <<<1, 256, R::kBookBytes, at::cuda::getCurrentCUDAStream()>>>(
            table.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <class R, int W>
__global__ void __launch_bounds__(32 * W)
    dmvq_kernel(half* __restrict__ out, const half* __restrict__ x, int ldx,
                int M, const uint8_t* __restrict__ weight,
                const uint4* __restrict__ gbook, float* __restrict__ ws,
                int* __restrict__ cnt, int n, int k) {
  extern __shared__ uint4 smem[];
  __shared__ int last;
  constexpr int BookVecs = (R::kBookBytes + 15) / 16;
  uint8_t* book = reinterpret_cast<uint8_t*>(smem);
  uint4* xs = smem + BookVecs;
  const int tile = blockIdx.x, sp = blockIdx.y, split = gridDim.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int quadpair = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quadpair * 8 + r;
  const int P = k / 128, pps = (P + split - 1) / split;
  const int g0 = min(P, sp * pps), g1 = min(P, g0 + pps);
  const int blocks_k = k / 256;
  uint4* slot = xs + warp * 256;
  uint4 X[4];
  auto xload = [&](int gg) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int idx = lane + 32 * j, c = idx >> 3, row = idx & 7;
      X[j] = make_uint4(0, 0, 0, 0);
      if (row < M)
        X[j] = __ldg(
            reinterpret_cast<const uint4*>(x + row * ldx + gg * 128 + c * 8));
    }
  };
  auto xstore = [&](int b) {
#pragma unroll
    for (int j = 0; j < 4; ++j) slot[b * 128 + lane + 32 * j] = X[j];
  };
  int g = g0 + warp;
  typename R::Record A;
  if (g < g1) {
    R rd(weight, tile, blocks_k, g, col);
    A = rd.load();
    xload(g);
  }
  for (int i = threadIdx.x; i < BookVecs; i += 32 * W)
    smem[i] = __ldg(gbook + i);
  if constexpr (R::kBookBytes % 16 != 0) R::initialize(book);
  __syncthreads();
  float acc[8] = {};
  if (g < g1) {
    xstore(0);
    __syncwarp();
    int cur = 0;
    while (true) {
      typename R::Record B;
      const int gn = g + W;
      const bool more = gn < g1;
      if (more) {
        R rd(weight, tile, blocks_k, gn, col);
        B = rd.load();
        xload(gn);
      }
      const uint4* xp = slot + cur * 128 + r;
#define DQ_SEG(S)                                               \
  {                                                             \
    const auto f = R::template fragment<S, 0>(A, book);         \
    const auto h = R::template fragment<S, 1>(A, book);         \
    const auto* b = reinterpret_cast<const uint32_t*>(&f);      \
    const auto* c = reinterpret_cast<const uint32_t*>(&h);      \
    const uint4 xa = xp[(2 * S) * 8], xb = xp[(2 * S + 1) * 8]; \
    native_pair_mma(acc, xa.x, xa.y, b[0], b[1]);               \
    native_pair_mma(acc, xa.z, xa.w, b[2], b[3]);               \
    native_pair_mma(acc, xb.x, xb.y, c[0], c[1]);               \
    native_pair_mma(acc, xb.z, xb.w, c[2], c[3]);               \
  }
      DQ_SEG(0);
      DQ_SEG(1);
      DQ_SEG(2);
      DQ_SEG(3);
      DQ_SEG(4);
      DQ_SEG(5);
      DQ_SEG(6);
      DQ_SEG(7);
#undef DQ_SEG
      if (!more) break;
      cur ^= 1;
      xstore(cur);
      __syncwarp();
      A = B;
      g = gn;
    }
  }
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int i = 0; i < 8; ++i) red[warp * 256 + lane * 8 + i] = acc[i];
  __syncthreads();
  auto write_out = [&](int v, float val) {
    const int lv = v >> 3, i = v & 7;
    const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
    const int c = (i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2);
    if (token < M)
      out[int64_t{token} * n + tile * 32 + ((lv >> 2) & 3) * 8 + c] =
          __float2half_rn(val);
  };
  for (int v = threadIdx.x; v < 256; v += 32 * W) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) s += red[w * 256 + v];
    if (split == 1)
      write_out(v, s);
    else
      __stcg(ws + (int64_t{tile} * split + sp) * 256 + v, s);
  }
  if (split == 1) return;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(cnt + tile, 1) == split - 1;
    if (last) __threadfence();
  }
  __syncthreads();
  if (!last) return;
  for (int v = threadIdx.x; v < 256; v += 32 * W) {
    float pv[8];
#pragma unroll
    for (int p = 0; p < 8; ++p)
      pv[p] =
          p < split ? __ldcg(ws + (int64_t{tile} * split + p) * 256 + v) : 0.f;
    float s = 0.f;
#pragma unroll
    for (int p = 0; p < 8; ++p) s += pv[p];
    write_out(v, s);
  }
  if (threadIdx.x == 0) cnt[tile] = 0;
}

template <class R, int W>
void dmvq_launch(torch::Tensor out, torch::Tensor x, torch::Tensor w,
                 torch::Tensor ws, torch::Tensor cnt, torch::Tensor table,
                 int split) {
  const int M = x.size(0), k = x.size(1), n = out.size(1);
  TORCH_CHECK(M <= 8 && n % 32 == 0 && k % 256 == 0 && split <= 8);
  TORCH_CHECK(w.numel() == int64_t{n} * (k / 256) * R::kBlockBytes);
  TORCH_CHECK(split == 1 || (ws.numel() >= int64_t{n / 32} * split * 256 &&
                             cnt.numel() >= n / 32));
  TORCH_CHECK(table.numel() >= R::kBookBytes, "invalid DMVQ codebook");
  const uint4* gbook =
      reinterpret_cast<const uint4*>(table.data_ptr<uint8_t>());
  const int book = (R::kBookBytes + 15) / 16 * 16;
  const int smem = std::max(book + W * 256 * 16, W * 256 * 4);
  auto kern = dmvq_kernel<R, W>;
  static bool init = false;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 90 * 1024));
    init = true;
  }
  kern<<<dim3(n / 32, split), 32 * W, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()), x.stride(0), M,
      w.data_ptr<uint8_t>(), gbook, split > 1 ? ws.data_ptr<float>() : nullptr,
      split > 1 ? cnt.data_ptr<int>() : nullptr, n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace vllm::sm70_gguf

void gguf_dmvq_sm70_out(torch::Tensor out, torch::Tensor x, torch::Tensor w,
                        torch::Tensor ws, torch::Tensor cnt,
                        torch::Tensor table, int64_t type, int64_t warps,
                        int64_t split) {
  using namespace vllm::sm70_gguf;
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.size(0) == 8 && x.stride(1) == 1,
              "DMVQ requires eight FP16 rows");
  const c10::cuda::CUDAGuard guard(x.device());
  auto storage = [&](torch::Tensor value, at::ScalarType dtype) {
    TORCH_CHECK(value.device() == x.device() && value.scalar_type() == dtype &&
                    value.is_contiguous(),
                "invalid DMVQ storage");
  };
  storage(out, at::kHalf);
  storage(w, at::kByte);
  storage(ws, at::kFloat);
  storage(cnt, at::kInt);
  storage(table, at::kByte);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == 8 && split >= 1 &&
                  split <= x.size(1) / 128 &&
                  (warps == 2 || warps == 4 || warps == 8),
              "invalid DMVQ geometry");
#define WS(R)                                                      \
  switch (warps) {                                                 \
    case 2:                                                        \
      return dmvq_launch<R, 2>(out, x, w, ws, cnt, table, split);  \
    case 4:                                                        \
      return dmvq_launch<R, 4>(out, x, w, ws, cnt, table, split);  \
    case 8:                                                        \
      return dmvq_launch<R, 8>(out, x, w, ws, cnt, table, split);  \
    default:                                                       \
      return dmvq_launch<R, 16>(out, x, w, ws, cnt, table, split); \
  }
  switch (type) {
#define T(X) \
  case X:    \
    WS(NativePairReader<X>)
    T(10);
    T(16);
    T(17);
    T(22);
    T(29);
#undef T
  }
#undef WS
  TORCH_CHECK(false, "unsupported type");
}

void gguf_dmvq_book_sm70_out(torch::Tensor table, int64_t type) {
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kByte &&
                  table.is_contiguous(),
              "invalid DMVQ table");
  const c10::cuda::CUDAGuard guard(table.device());
  using namespace vllm::sm70_gguf;
  switch (type) {
#define BOOK(T) \
  case T:       \
    return prepare_book<NativePairReader<T>>(table);
    BOOK(10);
    BOOK(16);
    BOOK(17);
    BOOK(22);
    BOOK(29);
#undef BOOK
  }
  TORCH_CHECK(false, "unsupported DMVQ codebook type");
}
