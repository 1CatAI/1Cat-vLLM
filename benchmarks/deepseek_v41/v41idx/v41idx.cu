// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// v41idx: VLLM_V41_INDEX_TILED decode index scorer for DeepSeek-V4.1 on SM70
// Bitwise replacement of v41k::index_scores_decode on allocated live keys.
//
// Contract (the same as v41k::index_scores_decode, which stays the reference):
//   q   fp16 [tokens * heads, 128]  (token-major rows), w fp16 [tokens * heads]
//   keys uint8 [capacity, 68]: 64 B E2M1 nibbles (even element low) + 4 biased
//        power-of-two scale exponents, one per 32 elements
//   positions int64 [>= tokens]; end = min((last_position + 1) / ratio, capacity)
//   out fp16, row stride out.size(1): out[t, n] for n < end only =
//     fp16( sum_h fp16( relu( fp16( q[t,h] . k[n] ) ) * w[t,h] ) )
//   with q . k accumulated as ONE fmaf chain over d = 0..127 in order, from
//   0.f, the key values rounded to fp16 first, and the head sum in h order.
//   Columns n >= end are never written.
//
// Why it is faster: the reference launches capacity / 32 CTAs (4096 at the
// stage-2 capacity of 131072 rows) and each thread computes one (key, token)
// pair with two shared-memory loads per FMA; at short context almost all CTAs
// only read positions and exit (25.6 / 48.9 us per call MEASURED at 0 keys),
// and at 300K keys it runs ~1.2 TFLOP/s (497 / 1061 us per call). Here:
//   - a persistent grid (<= 2 CTAs per SM) loops over 64-key blocks of the
//     live prefix only;
//   - q (FP32) and w are staged once per CTA; each 64-key block is decoded
//     once into shared memory as fp16 (exactly the reference's rounded key);
//   - each thread owns one token x H heads x 4 keys (4 * H accumulators) and
//     reads operands with 128-bit (q) / 64-bit (k) shared loads.
// Every (row, key) dot product is still the same fmaf chain in the same order
// on the same operand values, and the epilogue is the reference's, so the
// CPU host-compiled bodies below check bitwise equality. check_v41idx.py
// supplies GPU/reference and capture checks, which still require qualification
// after a rebase.
//
// The stage/tile bodies are __host__ __device__ so that v41idx_host_test.cu
// runs the exact kernel code on the CPU (no GPU) with the CTA schedule.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>

#ifndef V41IDX_HOST_TEST
  #include <mutex>
  #include <torch/all.h>
  #include <ATen/cuda/CUDAContext.h>
  #include <c10/cuda/CUDAGuard.h>
  #include <c10/cuda/CUDAException.h>
#endif

namespace v41idx {

constexpr int KB = 64;      // keys per block
constexpr int KL = 16;      // key lanes per token (threads per token)
constexpr int KJ = KB / KL; // keys per thread (4): kl, kl + 16, kl + 32, kl + 48
constexpr int QS = 132;     // q row stride in floats (528 B: 16-B aligned)
constexpr int KS = 132;     // key row stride in halves (264 B: 8-B aligned; 16
                            // consecutive rows start on banks 0, 2, .., 30)
constexpr int KEY_BYTES = 68;
constexpr int MAX_ROWS = 64;
constexpr int MAX_THREADS = 256;

__host__ __device__ constexpr int smem_bytes(int rows) {
  return rows * QS * 4 + ((rows + 3) & ~3) * 4 + KB * KS * 2;
}

__host__ __device__ inline float bits_float(uint32_t u) {
#ifdef __CUDA_ARCH__
  return __uint_as_float(u);
#else
  float f;
  memcpy(&f, &u, 4);
  return f;
#endif
}

// The reference's v41k_e2m1, operation for operation.
__host__ __device__ inline float e2m1(int code) {
  const int m = code & 7;
  const float mag = m < 2 ? 0.5f * m
                          : bits_float(uint32_t(((m >> 1) + 126) << 23) |
                                       uint32_t((m & 1) << 22));
  return (code & 8) ? -mag : mag;
}

// q rows -> FP32 shared rows (exactly __half2float of the input), w -> ws.
__host__ __device__ inline void stage_q(int tid, int nt, const __half* q,
                                        const __half* w, int rows, float* qs,
                                        float* ws) {
  for (int i = tid; i < rows * 128; i += nt) {
    const int r = i / 128, d = i % 128;
    qs[r * QS + d] = __half2float(q[r * 128 + d]);
  }
  for (int r = tid; r < rows; r += nt) ws[r] = __half2float(w[r]);
}

// Decode keys [n0, n0 + KB) into ks (fp16), zeros past end (never read for
// output). One item = one 4-byte word = 8 elements of one key row.
__host__ __device__ inline void stage_keys(int tid, int nt,
                                           const uint8_t* keys, int64_t n0,
                                           int64_t end, __half* ks) {
  for (int i = tid; i < KB * 16; i += nt) {
    const int k = i / 16, wd = i % 16;
    const int64_t n = n0 + k;
    __half h[8];
    if (n < end) {
      const uint8_t* row = keys + n * KEY_BYTES;
      uint32_t word;
#ifdef __CUDA_ARCH__
      word = *reinterpret_cast<const uint32_t*>(row + 4 * wd);
#else
      memcpy(&word, row + 4 * wd, 4);
#endif
      // byte b = 4 wd + c; its scale is row[64 + (2 b) / 32] = row[64 + wd / 4]
      const float scale = bits_float(uint32_t(row[64 + wd / 4]) << 23);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int byte = int((word >> (8 * c)) & 255u);
        const float lo = e2m1(byte & 15) * scale;
        const float hi = e2m1((byte >> 4) & 15) * scale;
        h[2 * c] = __float2half_rn(lo);
        h[2 * c + 1] = __float2half_rn(hi);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 8; ++c) h[c] = __float2half_rn(0.f);
    }
    __half* dst = ks + k * KS + 8 * wd;
#ifdef __CUDA_ARCH__
    uint2 a, b;
    memcpy(&a, h, 8);
    memcpy(&b, h + 4, 8);
    reinterpret_cast<uint2*>(dst)[0] = a;
    reinterpret_cast<uint2*>(dst)[1] = b;
#else
    memcpy(dst, h, 16);
#endif
  }
}

// One thread: token t, key lane kl (keys kl + 16 j), all H heads of t.
template <int H>
__host__ __device__ inline void tile(int t, int kl, const float* qs,
                                     const __half* ks, const float* ws,
                                     int64_t n0, int64_t end, __half* out,
                                     int64_t n_out) {
  float acc[H][KJ];
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int j = 0; j < KJ; ++j) acc[h][j] = 0.f;
#ifdef V41IDX_MUTANT_ORDER
  // CPU-test mutant only: reversed d order (must be caught as a mismatch).
  for (int d = 124; d >= 0; d -= 4) {
#else
  for (int d = 0; d < 128; d += 4) {
#endif
    float kv[KJ][4];
#pragma unroll
    for (int j = 0; j < KJ; ++j) {
      const __half* kp = ks + (kl + KL * j) * KS + d;
      __half2 a, b;
#ifdef __CUDA_ARCH__
      const uint2 raw = *reinterpret_cast<const uint2*>(kp);
      memcpy(&a, &raw.x, 4);
      memcpy(&b, &raw.y, 4);
#else
      memcpy(&a, kp, 4);
      memcpy(&b, kp + 2, 4);
#endif
      kv[j][0] = __low2float(a);
      kv[j][1] = __high2float(a);
      kv[j][2] = __low2float(b);
      kv[j][3] = __high2float(b);
    }
#pragma unroll
    for (int h = 0; h < H; ++h) {
      const float* qp = qs + (t * H + h) * QS + d;
      float qv[4];
#ifdef __CUDA_ARCH__
      const float4 q4 = *reinterpret_cast<const float4*>(qp);
      qv[0] = q4.x;
      qv[1] = q4.y;
      qv[2] = q4.z;
      qv[3] = q4.w;
#else
      memcpy(qv, qp, 16);
#endif
#ifdef V41IDX_MUTANT_ORDER
#pragma unroll
      for (int c = 3; c >= 0; --c)
#else
#pragma unroll
      for (int c = 0; c < 4; ++c)
#endif
#pragma unroll
        for (int j = 0; j < KJ; ++j)
          acc[h][j] = fmaf(qv[c], kv[j][c], acc[h][j]);
    }
  }
#pragma unroll
  for (int j = 0; j < KJ; ++j) {
    const int64_t n = n0 + kl + KL * j;
    if (n >= end) continue;
    float total = 0.f;
#pragma unroll
    for (int h = 0; h < H; ++h) {
      float s = __half2float(__float2half_rn(acc[h][j]));
      s = s > 0.f ? s : 0.f;
      total += __half2float(__float2half_rn(s * ws[t * H + h]));
    }
    out[int64_t(t) * n_out + n] = __float2half_rn(total);
  }
}

// Shared host/device bound: stale positions cannot extend an allocation.
__host__ __device__ inline int64_t live_end(int64_t position, int ratio,
                                           int64_t capacity) {
  const int64_t end = (position + 1) / ratio;
  return end < capacity ? end : capacity;
}

// Shared-memory carve-up for `rows` rows.
struct Smem {
  float* qs;
  float* ws;
  __half* ks;
};
__host__ __device__ inline Smem carve(unsigned char* base, int rows) {
  Smem s;
  s.qs = reinterpret_cast<float*>(base);
  s.ws = s.qs + rows * QS;
  s.ks = reinterpret_cast<__half*>(s.ws + ((rows + 3) & ~3));
  return s;
}

#ifndef V41IDX_HOST_TEST
template <int H>
__global__ void __launch_bounds__(MAX_THREADS)
    index_scores_tiled(const __half* __restrict__ q,
                       const __half* __restrict__ w,
                       const uint8_t* __restrict__ keys,
                       const int64_t* __restrict__ positions,
                       __half* __restrict__ out, int tokens, int64_t n_out,
                       int ratio, int64_t capacity) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int64_t end = live_end(positions[tokens - 1], ratio, capacity);
  int64_t n0 = int64_t(blockIdx.x) * KB;
  if (n0 >= end) return;
  const int rows = tokens * H;
  const Smem s = carve(smem, rows);
  const int tid = threadIdx.x, nt = blockDim.x;
  stage_q(tid, nt, q, w, rows, s.qs, s.ws);
  const int t = tid / KL, kl = tid % KL;
  for (; n0 < end; n0 += int64_t(gridDim.x) * KB) {
    stage_keys(tid, nt, keys, n0, end, s.ks);
    __syncthreads();
    if (t < tokens) tile<H>(t, kl, s.qs, s.ks, s.ws, n0, end, out, n_out);
    __syncthreads();
  }
}
#endif

}  // namespace v41idx

#ifndef V41IDX_HOST_TEST
  #include "v41idx_ref.cuh"

namespace {

int sm_count(int device) {
  static std::mutex mu;
  static int counts[64] = {0};
  std::lock_guard<std::mutex> lock(mu);
  TORCH_CHECK(device >= 0 && device < 64, "v41idx: device index out of range");
  if (counts[device] == 0) {
    int c = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount,
                                          device));
    counts[device] = c;
  }
  return counts[device];
}

template <int H>
void allow_smem(int device) {
  static std::mutex mu;
  static bool done[64] = {false};
  std::lock_guard<std::mutex> lock(mu);
  if (!done[device]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        v41idx::index_scores_tiled<H>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        v41idx::smem_bytes(v41idx::MAX_ROWS)));
    done[device] = true;
  }
}

void check_args(const torch::Tensor& out, const torch::Tensor& q,
                const torch::Tensor& w, const torch::Tensor& keys,
                const torch::Tensor& positions, int64_t heads, int64_t ratio,
                bool tiled) {
  const int64_t tokens = out.dim() == 2 ? out.size(0) : -1;
  TORCH_CHECK(out.is_cuda() && q.device() == out.device() &&
                  w.device() == out.device() &&
                  keys.device() == out.device() &&
                  positions.device() == out.device(),
              "v41idx: all tensors must be on one CUDA device");
  TORCH_CHECK(out.dim() == 2 && out.scalar_type() == torch::kHalf &&
                  out.stride(1) == 1 && out.stride(0) == out.size(1),
              "v41idx: out must be fp16 [tokens, n_out] with row stride n_out");
  if (tiled)
    TORCH_CHECK(heads == 4 || heads == 8,
                "v41idx: tiled scorer supports 4 or 8 index heads per rank, got ",
                heads);
  TORCH_CHECK(heads >= 1 && tokens >= 1 && tokens * heads <= 64,
              "v41idx: need 1 <= tokens * heads <= 64");
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.is_contiguous() &&
                  q.dim() == 2 && q.size(0) == tokens * heads &&
                  q.size(1) == 128,
              "v41idx: q must be contiguous fp16 [tokens * heads, 128]");
  TORCH_CHECK(w.scalar_type() == torch::kHalf && w.is_contiguous() &&
                  w.numel() == tokens * heads,
              "v41idx: w must be contiguous fp16 [tokens * heads]");
  TORCH_CHECK(keys.scalar_type() == torch::kUInt8 && keys.is_contiguous() &&
                  keys.dim() == 2 && keys.size(1) == 68 && keys.size(0) > 0,
              "v41idx: keys must be contiguous uint8 [capacity > 0, 68]");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
                  positions.is_contiguous() && positions.numel() >= tokens,
              "v41idx: positions must be contiguous int64 with >= tokens entries");
  TORCH_CHECK(out.size(1) >= keys.size(0),
              "v41idx: output width must cover key capacity");
  TORCH_CHECK(ratio >= 1, "v41idx: ratio must be >= 1");
}

void index_scores_tiled(torch::Tensor out, torch::Tensor q, torch::Tensor w,
                        torch::Tensor keys, torch::Tensor positions,
                        int64_t heads, int64_t ratio) {
  check_args(out, q, w, keys, positions, heads, ratio, true);
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int device = out.get_device();
  const int tokens = int(out.size(0));
  const int rows = tokens * int(heads);
  const int threads = ((tokens * v41idx::KL + 31) / 32) * 32;
  const int smem = v41idx::smem_bytes(rows);
  const int64_t blocks = (keys.size(0) + v41idx::KB - 1) / v41idx::KB;
  const int grid = int(std::min<int64_t>(blocks, 2 * int64_t(sm_count(device))));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* qp = reinterpret_cast<const __half*>(q.data_ptr());
  const auto* wp = reinterpret_cast<const __half*>(w.data_ptr());
  auto* op = reinterpret_cast<__half*>(out.data_ptr());
  if (heads == 8) {
    allow_smem<8>(device);
    v41idx::index_scores_tiled<8><<<grid, threads, smem, stream>>>(
        qp, wp, keys.data_ptr<uint8_t>(), positions.data_ptr<int64_t>(), op,
        tokens, out.size(1), int(ratio), keys.size(0));
  } else {
    allow_smem<4>(device);
    v41idx::index_scores_tiled<4><<<grid, threads, smem, stream>>>(
        qp, wp, keys.data_ptr<uint8_t>(), positions.data_ptr<int64_t>(), op,
        tokens, out.size(1), int(ratio), keys.size(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public v41k reference and launch configuration for check_v41idx.py;
// the independently loaded v41k operation is its primary reference.
void index_scores_ref(torch::Tensor out, torch::Tensor q, torch::Tensor w,
                      torch::Tensor keys, torch::Tensor positions,
                      int64_t heads, int64_t ratio) {
  check_args(out, q, w, keys, positions, heads, ratio, false);
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int tokens = out.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int KB = 32;
  const int blocks = (keys.size(0) + KB - 1) / KB;
  v41idx_ref::index_scores_decode<KB><<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr()),
      reinterpret_cast<const half*>(w.data_ptr()), keys.data_ptr<uint8_t>(),
      positions.data_ptr<int64_t>(), reinterpret_cast<half*>(out.data_ptr()),
      tokens, int(heads), out.size(1), int(ratio));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t version() { return 1; }

int64_t grid_for(int64_t capacity, int64_t device) {
  const int64_t blocks = (capacity + v41idx::KB - 1) / v41idx::KB;
  return std::min<int64_t>(blocks, 2 * int64_t(sm_count(int(device))));
}

}  // namespace

TORCH_LIBRARY(v41idx, m) {
  m.def(
      "index_scores_tiled(Tensor(a!) out, Tensor q, Tensor w, Tensor keys, "
      "Tensor positions, int heads, int ratio) -> ()",
      &index_scores_tiled);
  m.def(
      "index_scores_ref(Tensor(a!) out, Tensor q, Tensor w, Tensor keys, "
      "Tensor positions, int heads, int ratio) -> ()",
      &index_scores_ref);
  m.def("version() -> int", &version);
  m.def("grid_for(int capacity, int device) -> int", &grid_for);
}
// Shape-only capture/fake implementation. The out tensor is mutated in place.
namespace {
void index_scores_meta(torch::Tensor out, torch::Tensor q, torch::Tensor w,
                       torch::Tensor keys, torch::Tensor positions,
                       int64_t heads, int64_t ratio) {}
}
TORCH_LIBRARY_IMPL(v41idx, Meta, m) {
  m.impl("index_scores_tiled", &index_scores_meta);
}
#endif
