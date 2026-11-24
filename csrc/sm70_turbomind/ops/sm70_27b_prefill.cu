// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <type_traits>

namespace vllm_27b_sm70 {

// ============================================================================
// 1. Autotuned cuBLASLt FP32-Accumulation GEMM (CUBLAS_COMPUTE_32F)
//    - Caches optimal algo by (dev, K, N) -> ZERO autotune spike on tail M!
//    - Caches descriptors by (dev, M, K, N) -> minimal CPU launch latency!
//    - Works for ANY TP (TP=1, TP=2, TP=4, TP=8) and PCIe/NVLink alike.
// ============================================================================
struct KNKey {
  int dev;
  int64_t k, n;
  bool operator==(const KNKey& o) const {
    return dev == o.dev && k == o.k && n == o.n;
  }
};
struct KNKeyHash {
  size_t operator()(const KNKey& s) const noexcept {
    size_t h = static_cast<size_t>(s.dev);
    h = h * 1315423911ULL ^ static_cast<size_t>(s.k);
    h = h * 1315423911ULL ^ static_cast<size_t>(s.n);
    return h;
  }
};

struct MKNKey {
  int dev;
  int64_t m, k, n;
  bool operator==(const MKNKey& o) const {
    return dev == o.dev && m == o.m && k == o.k && n == o.n;
  }
};
struct MKNKeyHash {
  size_t operator()(const MKNKey& s) const noexcept {
    size_t h = static_cast<size_t>(s.dev);
    h = h * 1315423911ULL ^ static_cast<size_t>(s.m);
    h = h * 1315423911ULL ^ static_cast<size_t>(s.k);
    h = h * 1315423911ULL ^ static_cast<size_t>(s.n);
    return h;
  }
};

struct CachedAlgo {
  cublasLtMatmulAlgo_t algo;
  bool valid = false;
};

struct CachedDescs {
  cublasLtMatmulDesc_t opDesc = nullptr;
  cublasLtMatrixLayout_t layA = nullptr;
  cublasLtMatrixLayout_t layB = nullptr;
  cublasLtMatrixLayout_t layC = nullptr;
};

struct DevLtState {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 32ULL * 1024ULL * 1024ULL;
};

static std::mutex g_lt_mu;
static DevLtState g_dev_lt[16];
static std::unordered_map<KNKey, CachedAlgo, KNKeyHash> g_kn_algo_cache;
static std::unordered_map<MKNKey, CachedDescs, MKNKeyHash> g_desc_cache;

static DevLtState& get_dev_lt_locked(int dev) {
  auto& st = g_dev_lt[dev];
  if (!st.handle) {
    TORCH_CHECK(cublasLtCreate(&st.handle) == CUBLAS_STATUS_SUCCESS,
                "cublasLtCreate failed");
    TORCH_CHECK(cudaMalloc(&st.workspace, st.workspace_bytes) == cudaSuccess,
                "cudaMalloc cublasLt workspace failed");
  }
  return st;
}

static CachedDescs get_descs_locked(int dev, int64_t m, int64_t k, int64_t n) {
  const MKNKey key{dev, m, k, n};
  auto it = g_desc_cache.find(key);
  if (it != g_desc_cache.end()) return it->second;

  // Cap cache size to prevent unbounded growth on many random tail M values
  if (g_desc_cache.size() > 512) {
    for (auto& kv : g_desc_cache) {
      cublasLtMatrixLayoutDestroy(kv.second.layA);
      cublasLtMatrixLayoutDestroy(kv.second.layB);
      cublasLtMatrixLayoutDestroy(kv.second.layC);
      cublasLtMatmulDescDestroy(kv.second.opDesc);
    }
    g_desc_cache.clear();
  }

  CachedDescs d{};
  TORCH_CHECK(cublasLtMatmulDescCreate(&d.opDesc, CUBLAS_COMPUTE_32F,
                                       CUDA_R_32F) == CUBLAS_STATUS_SUCCESS);
  cublasOperation_t opN = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(d.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN,
                                 sizeof(opN));
  cublasLtMatmulDescSetAttribute(d.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN,
                                 sizeof(opN));
  cublasLtMatrixLayoutCreate(&d.layA, CUDA_R_16F, n, k, n);
  cublasLtMatrixLayoutCreate(&d.layB, CUDA_R_16F, k, m, k);
  cublasLtMatrixLayoutCreate(&d.layC, CUDA_R_16F, n, m, n);
  g_desc_cache[key] = d;
  return d;
}

void cublaslt_fp32_mm_out(torch::Tensor out, torch::Tensor x, torch::Tensor w) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 &&
                  w.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "must be FP16");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2, "must be 2D");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous(),
              "must be contiguous");
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = w.size(1);
  TORCH_CHECK(w.size(0) == k && out.size(0) == m && out.size(1) == n,
              "shape mismatch");
  if (m == 0 || n == 0 || k == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const int dev = x.device().index();
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  cublasLtHandle_t lt = nullptr;
  void* ws = nullptr;
  size_t ws_bytes = 0;
  CachedDescs d{};
  CachedAlgo cached{};
  const KNKey kn_key{dev, k, n};
  {
    std::lock_guard<std::mutex> lk(g_lt_mu);
    auto& st = get_dev_lt_locked(dev);
    lt = st.handle;
    ws = st.workspace;
    ws_bytes = st.workspace_bytes;
    d = get_descs_locked(dev, m, k, n);
    auto it = g_kn_algo_cache.find(kn_key);
    if (it != g_kn_algo_cache.end()) {
      cached = it->second;
    }
  }

  const float alpha = 1.0f, beta = 0.0f;

  // Autotune ONCE per (dev, K, N) on the first large M >= 1024!
  if (!cached.valid) {
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
        sizeof(ws_bytes));
    constexpr int kMax = 16;
    cublasLtMatmulHeuristicResult_t res[kMax];
    int nFound = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, d.opDesc, d.layA, d.layB, d.layC, d.layC,
                                   pref, kMax, res, &nFound);
    cublasLtMatmulPreferenceDestroy(pref);

    if (nFound > 0) {
      int best_idx = 0;
      if (nFound > 1 && m >= 1024) {
        int limit = nFound < 8 ? nFound : 8;
        float best_ms = 1e9f;
        float best_nosplit_ms = 1e9f;
        int best_nosplit_idx = -1;
        cudaEvent_t evS, evE;
        cudaEventCreate(&evS);
        cudaEventCreate(&evE);
        for (int i = 0; i < limit; ++i) {
          if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
          auto st0 = cublasLtMatmul(lt, d.opDesc, &alpha, w.data_ptr(), d.layA,
                                    x.data_ptr(), d.layB, &beta, out.data_ptr(),
                                    d.layC, out.data_ptr(), d.layC,
                                    &res[i].algo, ws, ws_bytes, stream);
          if (st0 != CUBLAS_STATUS_SUCCESS) continue;
          cudaEventRecord(evS, stream);
          for (int it = 0; it < 3; ++it) {
            cublasLtMatmul(lt, d.opDesc, &alpha, w.data_ptr(), d.layA,
                           x.data_ptr(), d.layB, &beta, out.data_ptr(), d.layC,
                           out.data_ptr(), d.layC, &res[i].algo, ws, ws_bytes,
                           stream);
          }
          cudaEventRecord(evE, stream);
          cudaEventSynchronize(evE);
          float ms = 0.0f;
          cudaEventElapsedTime(&ms, evS, evE);
          if (ms < best_ms) {
            best_ms = ms;
            best_idx = i;
          }
          if (res[i].workspaceSize == 0 && ms < best_nosplit_ms) {
            best_nosplit_ms = ms;
            best_nosplit_idx = i;
          }
        }
        cudaEventDestroy(evS);
        cudaEventDestroy(evE);
        // Prefer zero-workspace (no Memset launch) if within 1.5% of fastest!
        if (best_nosplit_idx >= 0 && best_nosplit_ms <= best_ms * 1.015f) {
          best_idx = best_nosplit_idx;
        }
        cached.algo = res[best_idx].algo;
        cached.valid = true;
        std::lock_guard<std::mutex> lk(g_lt_mu);
        g_kn_algo_cache[kn_key] = cached;
      } else {
        // Small M before any M>=1024 seen: use heuristic[0] without locking
        // into KN cache
        cached.algo = res[0].algo;
        cached.valid = true;
      }
    }
  }

  cublasStatus_t status = cublasLtMatmul(
      lt, d.opDesc, &alpha, w.data_ptr(), d.layA, x.data_ptr(), d.layB, &beta,
      out.data_ptr(), d.layC, out.data_ptr(), d.layC,
      cached.valid ? &cached.algo : nullptr, ws, ws_bytes, stream);

  if (status != CUBLAS_STATUS_SUCCESS && cached.valid) {
    // Fallback cleanly to cuBLASLt default heuristic if cached algo rejects an
    // unusual M
    status =
        cublasLtMatmul(lt, d.opDesc, &alpha, w.data_ptr(), d.layA, x.data_ptr(),
                       d.layB, &beta, out.data_ptr(), d.layC, out.data_ptr(),
                       d.layC, nullptr, ws, ws_bytes, stream);
  }
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "cublasLtMatmul failed with status ", static_cast<int>(status));
}

// ============================================================================
// 2. Vectorized 16B SiLU + Mul (FP32 math, exact parity with QPN4/QPN8 silu)
// ============================================================================
__global__ void silu_and_mul_vec_kernel(half* __restrict__ out,
                                        const half* __restrict__ gate_up,
                                        int rows, int hidden) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const half* row_in = gate_up + static_cast<size_t>(row) * hidden * 2;
  half* row_out = out + static_cast<size_t>(row) * hidden;
  if ((hidden & 7) == 0 && (reinterpret_cast<uintptr_t>(row_in) & 15u) == 0 &&
      (reinterpret_cast<uintptr_t>(row_out) & 15u) == 0) {
    const int vec_h = hidden >> 3;
    for (int v = threadIdx.x; v < vec_h; v += blockDim.x) {
      const uint4 g4 = reinterpret_cast<const uint4*>(row_in)[v];
      const uint4 u4 = reinterpret_cast<const uint4*>(row_in + hidden)[v];
      const half* gh = reinterpret_cast<const half*>(&g4);
      const half* uh = reinterpret_cast<const half*>(&u4);
      uint4 o4;
      half* oh = reinterpret_cast<half*>(&o4);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float gate = __half2float(gh[i]);
        const float silu = gate / (1.0f + __expf(-gate));
        oh[i] = __hmul(__float2half(silu), uh[i]);
      }
      reinterpret_cast<uint4*>(row_out)[v] = o4;
    }
    return;
  }
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float gate = __half2float(row_in[col]);
    const float silu = gate / (1.0f + __expf(-gate));
    row_out[col] = __hmul(__float2half(silu), row_in[hidden + col]);
  }
}

void silu_and_mul_out(torch::Tensor out, torch::Tensor gate_up) {
  const c10::cuda::CUDAGuard guard(gate_up.device());
  const int rows = static_cast<int>(out.size(0));
  const int hidden = static_cast<int>(out.size(1));
  if (rows == 0 || hidden == 0) return;
  silu_and_mul_vec_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate_up.data_ptr<at::Half>()), rows,
      hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// 3. Generic GDN RMSNormGated (D=128, ANY HV: TP1/TP2/TP4!)
//    Accepts strided z (zero z.contiguous() copy!). Exact FP32 RMSNorm + SiLU.
// ============================================================================
template <typename WeightT>
__global__ void __launch_bounds__(256, 4)
    gdn_rmsnorm_gated_d128_generic_kernel(half* __restrict__ out,
                                          const half* __restrict__ x,
                                          const half* __restrict__ z,
                                          const WeightT* __restrict__ weight,
                                          int num_tokens, int num_heads,
                                          int64_t z_row_stride, float eps) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head_idx_global = blockIdx.x * 8 + warp_in_block;
  const int total_heads = num_tokens * num_heads;
  if (head_idx_global >= total_heads) return;

  const int tok = head_idx_global / num_heads;
  const int h = head_idx_global - tok * num_heads;
  const int off = lane * 4;
  const size_t x_off = static_cast<size_t>(head_idx_global) * 128 + off;
  const size_t z_off = static_cast<size_t>(tok) * z_row_stride +
                       static_cast<size_t>(h) * 128 + off;

  uint2 xv = *reinterpret_cast<const uint2*>(x + x_off);
  uint2 zv = *reinterpret_cast<const uint2*>(z + z_off);

  const half2* xh = reinterpret_cast<const half2*>(&xv);
  const half2* zh = reinterpret_cast<const half2*>(&zv);
  float2 xf0 = __half22float2(xh[0]);
  float2 xf1 = __half22float2(xh[1]);
  float2 zf0 = __half22float2(zh[0]);
  float2 zf1 = __half22float2(zh[1]);

  float var = xf0.x * xf0.x + xf0.y * xf0.y + xf1.x * xf1.x + xf1.y * xf1.y;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) var += __shfl_xor_sync(0xffffffffu, var, d);
  const float inv = rsqrtf(var * (1.0f / 128.0f) + eps);

  float w0, w1, w2, w3;
  if constexpr (std::is_same_v<WeightT, half>) {
    uint2 wv = *reinterpret_cast<const uint2*>(weight + off);
    const half2* wh = reinterpret_cast<const half2*>(&wv);
    float2 wf0 = __half22float2(wh[0]);
    float2 wf1 = __half22float2(wh[1]);
    w0 = wf0.x;
    w1 = wf0.y;
    w2 = wf1.x;
    w3 = wf1.y;
  } else {
    float4 wf = *reinterpret_cast<const float4*>(weight + off);
    w0 = wf.x;
    w1 = wf.y;
    w2 = wf.z;
    w3 = wf.w;
  }

  auto silu = [](float v) { return v / (1.0f + __expf(-v)); };
  uint2 ov;
  half2* oh = reinterpret_cast<half2*>(&ov);
  oh[0] = __floats2half2_rn(xf0.x * inv * w0 * silu(zf0.x),
                            xf0.y * inv * w1 * silu(zf0.y));
  oh[1] = __floats2half2_rn(xf1.x * inv * w2 * silu(zf1.x),
                            xf1.y * inv * w3 * silu(zf1.y));
  *reinterpret_cast<uint2*>(out + x_off) = ov;
}

void gdn_rmsnorm_gated_d128_out(torch::Tensor out, torch::Tensor x,
                                torch::Tensor z, torch::Tensor weight,
                                double eps) {
  TORCH_CHECK(x.is_cuda() && z.is_cuda() && out.is_cuda() && weight.is_cuda());
  TORCH_CHECK(x.dim() == 3 && z.dim() == 3 && x.size(2) == 128 &&
              z.size(2) == 128);
  TORCH_CHECK(x.sizes() == z.sizes() && x.is_contiguous() &&
              out.is_contiguous());
  TORCH_CHECK(z.stride(1) == 128 && z.stride(2) == 1 && (z.stride(0) & 3) == 0);
  const int num_tokens = static_cast<int>(x.size(0));
  const int num_heads = static_cast<int>(x.size(1));
  if (num_tokens == 0 || num_heads == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int ctas = (num_tokens * num_heads + 7) / 8;
  if (weight.scalar_type() == torch::kFloat16) {
    gdn_rmsnorm_gated_d128_generic_kernel<half><<<ctas, 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(z.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()), num_tokens,
        num_heads, z.stride(0), static_cast<float>(eps));
  } else {
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32);
    gdn_rmsnorm_gated_d128_generic_kernel<float><<<ctas, 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(z.data_ptr<at::Half>()),
        weight.data_ptr<float>(), num_tokens, num_heads, z.stride(0),
        static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// 4. PUSH-Based Zero-Memcpy 16B Rotated NVLink P2P RS + GemmaRMSNorm + AG
//    Supports BOTH TP=2 (NGPU=2) AND TP=4 (NGPU=4), H=5120!
//    Bit-exact rank-ordered FP16 sum + FP32 residual + GemmaRMSNorm.
// ============================================================================
constexpr int kH = 5120;

template <int NGPU>
struct PeerPtrs {
  void* rs_ptrs[NGPU];
  void* out_ptrs[NGPU];
  int* flags[NGPU];
};

__device__ __forceinline__ uint4 ld_g128(const void* p) {
  return *reinterpret_cast<const uint4*>(p);
}
__device__ __forceinline__ void st_g128(void* p, uint4 v) {
  *reinterpret_cast<uint4*>(p) = v;
}
__device__ __forceinline__ void st_cs128(void* p, uint4 v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint4 hadd_u4(uint4 a, uint4 b) {
  uint4 r;
  half2* ra = reinterpret_cast<half2*>(&a);
  half2* rb = reinterpret_cast<half2*>(&b);
  half2* rr = reinterpret_cast<half2*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) rr[i] = __hadd2(ra[i], rb[i]);
  return r;
}

template <int NGPU>
__device__ __forceinline__ void cta_barrier(int* const flags[NGPU], int rank,
                                            int slot_phase, int epoch,
                                            int max_blocks) {
  __syncthreads();
  const int tid = threadIdx.x;
  if (tid == 0) __threadfence_system();
  __syncthreads();
  if (tid < NGPU) {
    const int b = blockIdx.x;
    const int off = (slot_phase * max_blocks + b) * NGPU;
    asm volatile(
        "st.relaxed.sys.global.s32 [%0], %1;" ::"l"(flags[tid] + off + rank),
        "r"(epoch)
        : "memory");
    int val = 0;
    int* src = flags[rank] + off + tid;
    do {
      asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
                   : "=r"(val)
                   : "l"(src)
                   : "memory");
    } while (val < epoch);
  }
  __syncthreads();
}

template <typename WeightT>
__device__ __forceinline__ void load_weights_16(
    const WeightT* __restrict__ weight, int tid, float wf[2][8]);

template <>
__device__ __forceinline__ void load_weights_16<half>(
    const half* __restrict__ weight, int tid, float wf[2][8]) {
  constexpr int T = 320;
  uint4 wp[2];
  wp[0] = ld_g128(reinterpret_cast<const uint4*>(weight) + tid);
  wp[1] = ld_g128(reinterpret_cast<const uint4*>(weight) + tid + T);
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    const half2* h2 = reinterpret_cast<const half2*>(&wp[it]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f2 = __half22float2(h2[j]);
      wf[it][2 * j] = f2.x + 1.0f;
      wf[it][2 * j + 1] = f2.y + 1.0f;
    }
  }
}

template <>
__device__ __forceinline__ void load_weights_16<float>(
    const float* __restrict__ weight, int tid, float wf[2][8]) {
  constexpr int T = 320;
  const float4* wp = reinterpret_cast<const float4*>(weight);
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    int idx = (tid + it * T) * 2;
    float4 a = wp[idx];
    float4 b = wp[idx + 1];
    wf[it][0] = a.x + 1.0f;
    wf[it][1] = a.y + 1.0f;
    wf[it][2] = a.z + 1.0f;
    wf[it][3] = a.w + 1.0f;
    wf[it][4] = b.x + 1.0f;
    wf[it][5] = b.y + 1.0f;
    wf[it][6] = b.z + 1.0f;
    wf[it][7] = b.w + 1.0f;
  }
}

template <int NGPU, typename WeightT>
__global__ void __launch_bounds__(320, 2)
    p2p_push_rs_gemma_norm_ag_kernel(PeerPtrs<NGPU> pp,
                                     const half* __restrict__ inp,
                                     const float* __restrict__ residual,
                                     const WeightT* __restrict__ weight,
                                     float* __restrict__ residual_out, int rank,
                                     int num_tokens, float eps, int slot,
                                     int epoch, int max_blocks) {
  static_assert(NGPU == 2 || NGPU == 4);
  constexpr int VECS = kH / 8;  // 640 uint4 per row
  constexpr int T = 320;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int tpr = (NGPU == 4) ? (num_tokens >> 2) : (num_tokens >> 1);
  const size_t shard_u4 = static_cast<size_t>(tpr) * VECS;

  __shared__ float warp_sum[10];
  __shared__ float inv_rms;

  const uint4* inp_u4 = reinterpret_cast<const uint4*>(inp);
  for (int lrow = blockIdx.x; lrow < tpr; lrow += gridDim.x) {
#pragma unroll
    for (int s = 1; s < NGPU; ++s) {
      const int peer = (rank + s) & (NGPU - 1);
      const size_t src_row = static_cast<size_t>(peer * tpr + lrow) * VECS;
      const size_t dst_row = static_cast<size_t>(rank) * shard_u4 +
                             static_cast<size_t>(lrow) * VECS;
      uint4* dst = reinterpret_cast<uint4*>(pp.rs_ptrs[peer]);
      uint4 v0 = ld_g128(inp_u4 + src_row + tid);
      uint4 v1 = ld_g128(inp_u4 + src_row + tid + T);
      st_cs128(dst + dst_row + tid, v0);
      st_cs128(dst + dst_row + tid + T, v1);
    }
  }

  cta_barrier<NGPU>(pp.flags, rank, slot * 2, epoch, max_blocks);

  float wf[2][8];
  load_weights_16<WeightT>(weight, tid, wf);

  const uint4* local_rs = reinterpret_cast<const uint4*>(pp.rs_ptrs[rank]);
  for (int lrow = blockIdx.x; lrow < tpr; lrow += gridDim.x) {
    const int row = rank * tpr + lrow;
    const size_t row_u4 = static_cast<size_t>(row) * VECS;

    float vals[2][8];
    float var = 0.0f;

#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int c = tid + it * T;
      uint4 pval[NGPU];
#pragma unroll
      for (int p = 0; p < NGPU; ++p) {
        if (p == rank) {
          pval[p] = ld_g128(inp_u4 + row_u4 + c);
        } else {
          pval[p] = ld_g128(local_rs + static_cast<size_t>(p) * shard_u4 +
                            static_cast<size_t>(lrow) * VECS + c);
        }
      }
      uint4 red = hadd_u4(pval[0], pval[1]);
      if constexpr (NGPU == 4) {
        red = hadd_u4(hadd_u4(red, pval[2]), pval[3]);
      }
      const half2* rh2 = reinterpret_cast<const half2*>(&red);

      const size_t g_idx = row_u4 + c;
      const float4* res_p =
          reinterpret_cast<const float4*>(residual) + g_idx * 2;
      float4* res_out_p = reinterpret_cast<float4*>(residual_out) + g_idx * 2;
      float4 r0 = res_p[0];
      float4 r1 = res_p[1];

      float2 f0 = __half22float2(rh2[0]);
      float2 f1 = __half22float2(rh2[1]);
      float2 f2 = __half22float2(rh2[2]);
      float2 f3 = __half22float2(rh2[3]);

      float4 o0 =
          make_float4(f0.x + r0.x, f0.y + r0.y, f1.x + r0.z, f1.y + r0.w);
      float4 o1 =
          make_float4(f2.x + r1.x, f2.y + r1.y, f3.x + r1.z, f3.y + r1.w);
      res_out_p[0] = o0;
      res_out_p[1] = o1;

      vals[it][0] = o0.x;
      vals[it][1] = o0.y;
      vals[it][2] = o0.z;
      vals[it][3] = o0.w;
      vals[it][4] = o1.x;
      vals[it][5] = o1.y;
      vals[it][6] = o1.z;
      vals[it][7] = o1.w;
      var += o0.x * o0.x + o0.y * o0.y + o0.z * o0.z + o0.w * o0.w +
             o1.x * o1.x + o1.y * o1.y + o1.z * o1.z + o1.w * o1.w;
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      var += __shfl_down_sync(0xffffffffu, var, off);
    if (lane == 0) warp_sum[warp] = var;
    __syncthreads();
    if (warp == 0) {
      float s = (lane < 10) ? warp_sum[lane] : 0.0f;
#pragma unroll
      for (int off = 8; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);
      if (lane == 0) inv_rms = rsqrtf(s * (1.0f / kH) + eps);
    }
    __syncthreads();
    const float sc = inv_rms;

#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const size_t g_idx = row_u4 + tid + it * T;
      uint4 out_pack;
      half2* oh2 = reinterpret_cast<half2*>(&out_pack);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        oh2[j] =
            __floats2half2_rn(vals[it][2 * j] * sc * wf[it][2 * j],
                              vals[it][2 * j + 1] * sc * wf[it][2 * j + 1]);
      }
#pragma unroll
      for (int s = 0; s < NGPU; ++s) {
        const int peer = (rank + s) & (NGPU - 1);
        if (peer == rank)
          st_g128(reinterpret_cast<uint4*>(pp.out_ptrs[peer]) + g_idx,
                  out_pack);
        else
          st_cs128(reinterpret_cast<uint4*>(pp.out_ptrs[peer]) + g_idx,
                   out_pack);
      }
    }
  }

  cta_barrier<NGPU>(pp.flags, rank, slot * 2 + 1, epoch, max_blocks);
}

template <int NGPU>
static torch::Tensor launch_p2p_push(
    torch::Tensor inp, torch::Tensor residual, torch::Tensor weight,
    torch::Tensor residual_out, const std::vector<int64_t>& rs_ptrs,
    const std::vector<int64_t>& out_ptrs, const std::vector<int64_t>& flag_ptrs,
    int64_t rank, double eps, int64_t slot, int64_t epoch, int64_t max_blocks) {
  const c10::cuda::CUDAGuard guard(inp.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  PeerPtrs<NGPU> pp{};
  for (int i = 0; i < NGPU; ++i) {
    pp.rs_ptrs[i] = reinterpret_cast<void*>(rs_ptrs[i]);
    pp.out_ptrs[i] = reinterpret_cast<void*>(out_ptrs[i]);
    pp.flags[i] = reinterpret_cast<int*>(flag_ptrs[i]);
  }
  const int num_tokens = static_cast<int>(inp.size(0));
  const int tpr = num_tokens / NGPU;
  const int blocks =
      tpr < static_cast<int>(max_blocks) ? tpr : static_cast<int>(max_blocks);

  if (weight.scalar_type() == torch::kFloat16) {
    p2p_push_rs_gemma_norm_ag_kernel<NGPU, half><<<blocks, 320, 0, stream>>>(
        pp, reinterpret_cast<const half*>(inp.data_ptr<at::Half>()),
        residual.data_ptr<float>(),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        residual_out.data_ptr<float>(), static_cast<int>(rank), num_tokens,
        static_cast<float>(eps), static_cast<int>(slot),
        static_cast<int>(epoch), static_cast<int>(max_blocks));
  } else {
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32);
    p2p_push_rs_gemma_norm_ag_kernel<NGPU, float><<<blocks, 320, 0, stream>>>(
        pp, reinterpret_cast<const half*>(inp.data_ptr<at::Half>()),
        residual.data_ptr<float>(), weight.data_ptr<float>(),
        residual_out.data_ptr<float>(), static_cast<int>(rank), num_tokens,
        static_cast<float>(eps), static_cast<int>(slot),
        static_cast<int>(epoch), static_cast<int>(max_blocks));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return torch::from_blob(
      pp.out_ptrs[rank], {num_tokens, kH}, [](void*) {}, inp.options());
}

torch::Tensor p2p_push_rs_gemma_norm_ag(
    torch::Tensor inp, torch::Tensor residual, torch::Tensor weight,
    torch::Tensor residual_out, std::vector<int64_t> rs_ptrs,
    std::vector<int64_t> out_ptrs, std::vector<int64_t> flag_ptrs, int64_t rank,
    double eps, int64_t slot, int64_t epoch, int64_t max_blocks) {
  const int64_t world_size = static_cast<int64_t>(rs_ptrs.size());
  TORCH_CHECK((world_size == 2 || world_size == 4) &&
              out_ptrs.size() == static_cast<size_t>(world_size) &&
              flag_ptrs.size() == static_cast<size_t>(world_size));
  TORCH_CHECK(inp.is_cuda() && inp.dim() == 2 && inp.size(1) == kH &&
              (inp.size(0) % world_size) == 0 &&
              inp.scalar_type() == torch::kFloat16 && inp.is_contiguous());
  TORCH_CHECK(residual.scalar_type() == torch::kFloat32 &&
              residual_out.scalar_type() == torch::kFloat32 &&
              residual.is_contiguous() && residual_out.is_contiguous());

  if (world_size == 4) {
    return launch_p2p_push<4>(inp, residual, weight, residual_out, rs_ptrs,
                              out_ptrs, flag_ptrs, rank, eps, slot, epoch,
                              max_blocks);
  }
  return launch_p2p_push<2>(inp, residual, weight, residual_out, rs_ptrs,
                            out_ptrs, flag_ptrs, rank, eps, slot, epoch,
                            max_blocks);
}

}  // namespace vllm_27b_sm70

TORCH_LIBRARY(vllm_sm70_27b_prefill, m) {
  m.def("cublaslt_fp32_mm_out(Tensor(a!) out, Tensor x, Tensor w) -> ()");
  m.impl("cublaslt_fp32_mm_out", torch::kCUDA,
         &vllm_27b_sm70::cublaslt_fp32_mm_out);

  m.def("silu_and_mul_out(Tensor(a!) out, Tensor gate_up) -> ()");
  m.impl("silu_and_mul_out", torch::kCUDA, &vllm_27b_sm70::silu_and_mul_out);

  m.def(
      "gdn_rmsnorm_gated_d128_out(Tensor(a!) out, Tensor x, Tensor z, "
      "Tensor weight, float eps) -> ()");
  m.impl("gdn_rmsnorm_gated_d128_out", torch::kCUDA,
         &vllm_27b_sm70::gdn_rmsnorm_gated_d128_out);

  m.def(
      "p2p_push_rs_gemma_norm_ag(Tensor inp, Tensor residual, Tensor weight, "
      "Tensor(a!) residual_out, int[] rs_ptrs, int[] out_ptrs, int[] "
      "flag_ptrs, "
      "int rank, float eps, int slot, int epoch, int max_blocks) -> Tensor");
  m.impl("p2p_push_rs_gemma_norm_ag", torch::kCUDA,
         &vllm_27b_sm70::p2p_push_rs_gemma_norm_ag);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
