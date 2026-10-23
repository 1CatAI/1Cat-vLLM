// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// QSA short-context decode attention for SM70 (contexts within the indexer
// budget, where QSA selects every visible token). Dense causal paged attention
// over [0, pos] per query token, split-K flash-decoding: grid (split, kv head,
// request); a CTA stacks the request's tokens x GROUP query heads as up to 32
// mma rows (wmma 16x16x16 fp16 -> fp32). Same arithmetic contract as
// _qsa_sparse_paged_gqa_splitk_kernel: scores * d^-0.5 * log2(e), exp2 online
// softmax, fp16 P.V with fp32 accumulation, fp16 output, then output *
// sigmoid(gate) in fp32.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

namespace qd {
constexpr int D = 256, RT = 32, KT = 32;
constexpr int QLD = D + 8, PLD = KT + 8, SLD = KT + 4, OLD = D + 4;

struct Args {
  const half* q;
  const half* kc;
  const half* vc;
  const int* block_table;
  const int* token_to_req;
  const int* pos;
  half* out;
  const half* gate;
  float* ws_o;   // [nreq][S][RT][D]
  float* ws_ml;  // [nreq][S][RT][2]
  long long sq_row, sq_head, sk_block, sk_tok, sk_head, sv_block, sv_tok,
      sv_head, so_row, so_head, sg_row, sg_head;
  int stab, T, G, HQ, BS, S;
  float scale_log2;
};

__device__ __forceinline__ void request_rows(const Args& a, int req, int& t0,
                                             int& nt) {
  t0 = -1;
  nt = 0;
  for (int t = 0; t < a.T; ++t)
    if (a.token_to_req[t] == req) {
      if (t0 < 0) t0 = t;
      ++nt;
    }
}

template <int NW>
__global__ void __launch_bounds__(32 * NW) partial(Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  half* qs = reinterpret_cast<half*>(smem);             // [RT][QLD]
  half* ks = qs + RT * QLD;                             // [KT][QLD]
  half* vs = ks + KT * QLD;                             // [KT][QLD]
  half* ps = vs + KT * QLD;                             // [RT][PLD]
  float* ss = reinterpret_cast<float*>(ps + RT * PLD);  // [RT][SLD]
  float* os = ss + RT * SLD;                            // [RT][OLD]
  __shared__ float m_r[RT], l_r[RT], al_r[RT];
  __shared__ int lim_r[RT];
  const int split = blockIdx.x, kvh = blockIdx.y, req = blockIdx.z,
            tid = threadIdx.x, warp = tid / 32;
  int t0, nt;
  request_rows(a, req, t0, nt);
  const int rows = nt * a.G;
  // query rows, key limits, state
  for (int i = tid; i < RT * (D / 8); i += blockDim.x) {
    const int r = i / (D / 8), c = (i % (D / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < rows) {
      const int t = t0 + r / a.G, h = kvh * a.G + r % a.G;
      v = *reinterpret_cast<const uint4*>(a.q + t * a.sq_row + h * a.sq_head +
                                          c);
    }
    *reinterpret_cast<uint4*>(qs + r * QLD + c) = v;
  }
  int maxpos = -1;
  for (int t = t0; t < t0 + nt; ++t) maxpos = max(maxpos, a.pos[t]);
  if (tid < RT) {
    lim_r[tid] = tid < rows ? a.pos[t0 + tid / a.G] : -1;
    m_r[tid] = -1.0e20f;
    l_r[tid] = 0.f;
  }
  for (int i = tid; i < RT * OLD; i += blockDim.x) os[i] = 0.f;
  // this split's key range: [k0, k1)
  const int nkeys = maxpos + 1;
  const int per = ((nkeys + a.S - 1) / a.S + KT - 1) / KT * KT;
  const int k0 = split * per, k1 = min(nkeys, k0 + per);
  __syncthreads();
  const int* bt = a.block_table + req * a.stab;
  for (int kb = k0; kb < k1; kb += KT) {
    // gather K/V tile (KT keys)
    for (int i = tid; i < KT * (D / 8); i += blockDim.x) {
      const int j = i / (D / 8), c = (i % (D / 8)) * 8, key = kb + j;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (key < k1) {
        const long long page = bt[key / a.BS], off = key % a.BS;
        kv = *reinterpret_cast<const uint4*>(
            a.kc + page * a.sk_block + off * a.sk_tok + kvh * a.sk_head + c);
        vv = *reinterpret_cast<const uint4*>(
            a.vc + page * a.sv_block + off * a.sv_tok + kvh * a.sv_head + c);
      }
      *reinterpret_cast<uint4*>(ks + j * QLD + c) = kv;
      *reinterpret_cast<uint4*>(vs + j * QLD + c) = vv;
    }
    __syncthreads();
    if (warp < 4) {  // S = Q K^T: 2 x 2 tiles of 16x16, one per warp
      const int tr = warp >> 1, tc = warp & 1;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
#pragma unroll
      for (int k = 0; k < D; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, qs + tr * 16 * QLD + k, QLD);
        wmma::load_matrix_sync(fb, ks + tc * 16 * QLD + k, QLD);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      wmma::store_matrix_sync(ss + tr * 16 * SLD + tc * 16, acc, SLD,
                              wmma::mem_row_major);
    }
    __syncthreads();
    if (tid < RT) {  // online softmax for row tid over this tile
      const int r = tid, lim = lim_r[r];
      float mx = m_r[r];
      float sc[KT];
#pragma unroll
      for (int j = 0; j < KT; ++j) {
        const bool ok = kb + j <= lim && kb + j < k1;
        sc[j] = ok ? ss[r * SLD + j] * a.scale_log2 : -1.0e20f;
        mx = fmaxf(mx, sc[j]);
      }
      const float alpha = exp2f(m_r[r] - mx);
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < KT; ++j) {
        const bool ok = kb + j <= lim && kb + j < k1;
        const float p = ok ? exp2f(sc[j] - mx) : 0.f;
        sum += p;
        ps[r * PLD + j] = __float2half_rn(p);
      }
      l_r[r] = l_r[r] * alpha + sum;
      m_r[r] = mx;
      al_r[r] = alpha;
    }
    __syncthreads();
    for (int i = tid; i < RT * D; i += blockDim.x)
      os[(i / D) * OLD + i % D] *= al_r[i / D];
    __syncthreads();
    // O += P V: 2 row tiles x 16 dim tiles, 8 per warp
    for (int tt = warp; tt < 2 * (D / 16); tt += NW) {
      const int tr = tt / (D / 16), tc = tt % (D / 16);
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::load_matrix_sync(acc, os + tr * 16 * OLD + tc * 16, OLD,
                             wmma::mem_row_major);
#pragma unroll
      for (int k = 0; k < KT; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
        wmma::load_matrix_sync(fa, ps + tr * 16 * PLD + k, PLD);
        wmma::load_matrix_sync(fb, vs + k * QLD + tc * 16, QLD);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      wmma::store_matrix_sync(os + tr * 16 * OLD + tc * 16, acc, OLD,
                              wmma::mem_row_major);
    }
    __syncthreads();
  }
  // partials
  float* wo =
      a.ws_o +
      ((static_cast<long long>(req) * gridDim.y + kvh) * a.S + split) * RT * D;
  float* wml =
      a.ws_ml +
      ((static_cast<long long>(req) * gridDim.y + kvh) * a.S + split) * RT * 2;
  for (int i = tid; i < RT * D; i += blockDim.x)
    wo[i] = os[(i / D) * OLD + i % D];
  if (tid < RT) {
    wml[tid * 2] = m_r[tid];
    wml[tid * 2 + 1] = l_r[tid];
  }
}

__global__ void __launch_bounds__(D) merge(Args a) {
  constexpr int MAXS = 128;
  __shared__ float w_s[MAXS], s_den;
  const int r = blockIdx.x, kvh = blockIdx.y, req = blockIdx.z, d = threadIdx.x;
  int t0, nt;
  request_rows(a, req, t0, nt);
  if (r >= nt * a.G) return;
  const long long base = (static_cast<long long>(req) * gridDim.y + kvh) * a.S;
  if (d < 32) {
    float M = -1.0e30f;
    for (int s = d; s < a.S; s += 32) {
      const float l = a.ws_ml[(base + s) * RT * 2 + r * 2 + 1];
      if (l > 0.f) M = fmaxf(M, a.ws_ml[(base + s) * RT * 2 + r * 2]);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1)
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    float den = 0.f;
    for (int s = d; s < a.S; s += 32) {
      const float l = a.ws_ml[(base + s) * RT * 2 + r * 2 + 1];
      const float w =
          l > 0.f ? exp2f(a.ws_ml[(base + s) * RT * 2 + r * 2] - M) : 0.f;
      w_s[s] = w;
      den += w * l;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
    if (d == 0) s_den = den;
  }
  __syncthreads();
  float num = 0.f;
  const float* op = a.ws_o + base * RT * D + r * D + d;
#pragma unroll 8
  for (int s = 0; s < a.S; ++s)
    num += w_s[s] * op[static_cast<long long>(s) * RT * D];
  const float den = s_den;
  const int t = t0 + r / a.G, h = kvh * a.G + r % a.G;
  float o = den > 0.f ? num / fmaxf(den, 1.0e-20f) : 0.f;
  if (a.gate) {
    o = __half2float(__float2half_rn(o));
    const float g = __half2float(a.gate[t * a.sg_row + h * a.sg_head + d]);
    o = o * (1.f / (1.f + __expf(-g)));
  }
  a.out[t * a.so_row + h * a.so_head + d] = __float2half_rn(o);
}
}  // namespace qd

void qsa_dense_decode_sm70_out(torch::Tensor out, torch::Tensor q,
                               torch::Tensor kc, torch::Tensor vc,
                               torch::Tensor block_table,
                               torch::Tensor token_to_req, torch::Tensor pos,
                               std::optional<torch::Tensor> gate,
                               torch::Tensor ws_o, torch::Tensor ws_ml,
                               int64_t nreq, int64_t splits, int64_t nw) {
  qd::Args a{};
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat16 &&
                  kc.scalar_type() == torch::kFloat16 &&
                  vc.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "qsa dense decode requires FP16 tensors");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  token_to_req.scalar_type() == torch::kInt32 &&
                  pos.scalar_type() == torch::kInt32,
              "qsa dense decode requires int32 metadata");
  TORCH_CHECK(kc.size(3) == qd::D && vc.size(3) == qd::D && q.size(2) == qd::D,
              "head size must be 256");
  const c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.size(2) == qd::D && kc.size(3) == qd::D);
  a.q = reinterpret_cast<const half*>(q.data_ptr());
  a.kc = reinterpret_cast<const half*>(kc.data_ptr());
  a.vc = reinterpret_cast<const half*>(vc.data_ptr());
  a.block_table = block_table.data_ptr<int>();
  a.token_to_req = token_to_req.data_ptr<int>();
  a.pos = pos.data_ptr<int>();
  a.out = reinterpret_cast<half*>(out.data_ptr());
  a.gate = gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr;
  a.ws_o = ws_o.data_ptr<float>();
  a.ws_ml = ws_ml.data_ptr<float>();
  a.sq_row = q.stride(0);
  a.sq_head = q.stride(1);
  a.sk_block = kc.stride(0);
  a.sk_tok = kc.stride(1);
  a.sk_head = kc.stride(2);
  a.sv_block = vc.stride(0);
  a.sv_tok = vc.stride(1);
  a.sv_head = vc.stride(2);
  a.so_row = out.stride(0);
  a.so_head = out.stride(1);
  if (gate) {
    a.sg_row = gate->stride(0);
    a.sg_head = gate->stride(1);
  }
  a.stab = block_table.stride(0);
  a.T = q.size(0);
  a.HQ = q.size(1);
  const int HK = kc.size(2);
  a.G = a.HQ / HK;
  a.BS = kc.size(1);
  a.S = splits;
  a.scale_log2 =
      (1.0f / sqrtf(static_cast<float>(qd::D))) * 1.4426950408889634f;
  TORCH_CHECK(a.G * 5 <= qd::RT || a.T <= qd::RT / a.G,
              "rows per request must fit in 32");
  TORCH_CHECK(splits <= 128, "splits");
  TORCH_CHECK(ws_o.numel() >= nreq * HK * splits * qd::RT * qd::D &&
              ws_ml.numel() >= nreq * HK * splits * qd::RT * 2);
  const size_t sm =
      (qd::RT * qd::QLD + 2 * qd::KT * qd::QLD + qd::RT * qd::PLD) * 2 +
      (qd::RT * qd::SLD + qd::RT * qd::OLD) * 4;
  auto st = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto w) {
    constexpr int NWV = decltype(w)::value;
    static bool init[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!init[dev & 15]) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qd::partial<NWV>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(sm)));
      init[dev & 15] = true;
    }
    qd::partial<NWV><<<dim3(splits, HK, nreq), 32 * NWV, sm, st>>>(a);
  };
  if (nw == 16)
    go(std::integral_constant<int, 16>{});
  else if (nw == 8)
    go(std::integral_constant<int, 8>{});
  else
    go(std::integral_constant<int, 4>{});
  qd::merge<<<dim3(qd::RT, HK, nreq), qd::D, 0, st>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
