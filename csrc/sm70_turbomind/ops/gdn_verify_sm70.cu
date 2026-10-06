// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// GDN target-verification recurrent update (sigmoid gating + gated delta rule) for SM70.
// Same contract as fused_sigmoid_gating_delta_rule_update_mixed_qkv on the spec-decode route:
// packed [T, q | k | v] fp16 rows, cu_seqlens, per-token state slots [N, S], num_accepted_tokens,
// fp32 state [slots, HV, V, K] updated in place for every token, o [T, HV, V] fp16.
//
// The Triton kernel runs one program per (V-tile of 32, head): 48 programs per rank at HV = 12,
// each walking the T tokens with global loads inside the loop. Here a CTA owns 16 state rows
// (4 warps x 4 rows, each lane 4 K columns of a row in registers): 96+ CTAs, every token's q/k/v
// and gate scalars are loaded up front, so the T-step chain is register math plus warp
// reductions, and per-token state stores are fire-and-forget float4s.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {
constexpr int KD = 128, TMAX = 8;

__device__ __forceinline__ float tof(float v) { return v; }
__device__ __forceinline__ float tof(half v) { return __half2float(v); }

struct P {
  const half* qkv;
  const half* a;
  const half* b;
  float* state;
  half* o;
  const int* cu;
  const int* idx;
  const int* nacc;
  long long s_state;  // state slot stride (elements)
  int s_idx;          // ssm_state_indices row stride
  int qkv_stride;
  int H, HV;
  int lda, ldb;  // row strides of a / b (elements), HV when contiguous
  float scale;
  // replay mode: rbuf holds two sides x nmax sequences of the previous/current round's
  // normalized k [TMAX][H][KD], raw v [TMAX][HV][KD] and (decay, beta) [TMAX][HV][2], fp32.
  float* rbuf;
  const int* rn;   // tokens of the previous round to replay (its accepted count), per sequence
  unsigned* ctr;   // [round, finished CTAs]
  int nmax;
};
__device__ __forceinline__ long long seq_floats(int H, int HV) { return (long long)TMAX * (H * KD + HV * KD + HV * 2); }

// Replay mode: the last CTA to finish advances the round counter (selects the buffer side).
__device__ __forceinline__ void round_done(const P& p) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.ctr + 1, 1u) == gridDim.x * gridDim.y - 1) {
      p.ctr[1] = 0;
      __threadfence();
      atomicAdd(p.ctr, 1u);
    }
  }
}

template <typename TA, typename TD, int WARPS, int ROWS, bool REPLAY, int TT>
__global__ __launch_bounds__(WARPS * 32) void gdn_verify(P p, const TA* A_log, const TD* dt_bias) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.y / p.HV, hv = blockIdx.y % p.HV, h = hv / (p.HV / p.H);
  const int bos = p.cu[n], T = p.cu[n + 1] - bos;
  if (T <= 0) {
    if (REPLAY) round_done(p);
    return;
  }
  const int slot0 = p.idx[n * p.s_idx + (REPLAY ? 0 : (p.nacc ? p.nacc[n] - 1 : 0))];
  if (slot0 < 0) {
    if (REPLAY) round_done(p);
    return;
  }
  const int row0 = blockIdx.x * (WARPS * ROWS) + warp * ROWS;
  const int qo = h * KD + lane * 4, ko = p.H * KD + h * KD + lane * 4, vo = 2 * p.H * KD + hv * KD + row0;

  // ---- everything the T-step chain needs, issued before the state load lands
  float4 st[ROWS];
#pragma unroll
  for (int j = 0; j < ROWS; ++j)
    st[j] = __ldcg(reinterpret_cast<const float4*>(p.state + slot0 * p.s_state + ((long long)hv * KD + row0 + j) * KD) + lane);
  // Replay: committed state + the previous round's first rn tokens = the round's initial state.
  const unsigned side = REPLAY ? (__ldcg(p.ctr) & 1u) : 0u;
  const long long SEQ = seq_floats(p.H, p.HV);
  float* cur = REPLAY ? p.rbuf + ((long long)side * p.nmax + n) * SEQ : nullptr;
  int R = 0;
  float4 rkv[TT];
  float rvv[TT][ROWS], rdc[TT], rbt[TT];
  if (REPLAY) {
    const float* prv = p.rbuf + ((long long)(side ^ 1u) * p.nmax + n) * SEQ;
    R = p.rn[n];
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      if (t < R) {
        rkv[t] = __ldcg(reinterpret_cast<const float4*>(prv + (t * p.H + h) * KD) + lane);
#pragma unroll
        for (int j = 0; j < ROWS; ++j) rvv[t][j] = __ldcg(prv + TMAX * p.H * KD + (t * p.HV + hv) * KD + row0 + j);
        const float2 gb = __ldcg(reinterpret_cast<const float2*>(prv + TMAX * (p.H + p.HV) * KD) + t * p.HV + hv);
        rdc[t] = gb.x, rbt[t] = gb.y;
      }
    }
  }
  uint2 qr[TT], kr[TT];
  float vr[TT][ROWS], ar[TT], br[TT];
  const float Al = tof(A_log[hv]), dtb = tof(dt_bias[hv]);
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    if (t < T) {
      const half* row = p.qkv + (long long)(bos + t) * p.qkv_stride;
      qr[t] = *reinterpret_cast<const uint2*>(row + qo);
      kr[t] = *reinterpret_cast<const uint2*>(row + ko);
#pragma unroll
      for (int j = 0; j < ROWS; ++j) vr[t][j] = __half2float(row[vo + j]);
      ar[t] = __half2float(p.a[(bos + t) * p.lda + hv]);
      br[t] = __half2float(p.b[(bos + t) * p.ldb + hv]);
    }
  }
  float hs[ROWS][4];
#pragma unroll
  for (int j = 0; j < ROWS; ++j) hs[j][0] = st[j].x, hs[j][1] = st[j].y, hs[j][2] = st[j].z, hs[j][3] = st[j].w;
  const float negA = -__expf(Al);
  if (REPLAY) {
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      if (t >= R) break;
      const float k[4] = {rkv[t].x, rkv[t].y, rkv[t].z, rkv[t].w};
      float d[ROWS];
#pragma unroll
      for (int j = 0; j < ROWS; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) hs[j][e] *= rdc[t];
        d[j] = hs[j][0] * k[0] + hs[j][1] * k[1] + hs[j][2] * k[2] + hs[j][3] * k[3];
      }
#pragma unroll
      for (int s = 16; s > 0; s >>= 1)
#pragma unroll
        for (int j = 0; j < ROWS; ++j) d[j] += __shfl_xor_sync(0xffffffff, d[j], s);
#pragma unroll
      for (int j = 0; j < ROWS; ++j) {
        const float u = (rvv[t][j] - d[j]) * rbt[t];
#pragma unroll
        for (int e = 0; e < 4; ++e) hs[j][e] += u * k[e];
      }
    }
    if (R > 0) {
#pragma unroll
      for (int j = 0; j < ROWS; ++j)
        __stcg(reinterpret_cast<float4*>(p.state + slot0 * p.s_state + ((long long)hv * KD + row0 + j) * KD) + lane,
               make_float4(hs[j][0], hs[j][1], hs[j][2], hs[j][3]));
    }
  }

#pragma unroll
  for (int t = 0; t < TT; ++t) {
    if (t >= T) break;
    float q[4], k[4];
    {
      const half2 q01 = *reinterpret_cast<const half2*>(&qr[t].x), q23 = *reinterpret_cast<const half2*>(&qr[t].y);
      const half2 k01 = *reinterpret_cast<const half2*>(&kr[t].x), k23 = *reinterpret_cast<const half2*>(&kr[t].y);
      q[0] = __low2float(q01), q[1] = __high2float(q01), q[2] = __low2float(q23), q[3] = __high2float(q23);
      k[0] = __low2float(k01), k[1] = __high2float(k01), k[2] = __low2float(k23), k[3] = __high2float(k23);
    }
    float sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    float sk = k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3];
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      sq += __shfl_xor_sync(0xffffffff, sq, s);
      sk += __shfl_xor_sync(0xffffffff, sk, s);
    }
    const float rq = rsqrtf(sq + 1e-6f) * p.scale, rk = rsqrtf(sk + 1e-6f);
#pragma unroll
    for (int e = 0; e < 4; ++e) q[e] *= rq, k[e] *= rk;
    const float x = ar[t] + dtb;
    const float sp = x <= 20.f ? logf(1.f + __expf(x)) : x;
    const float decay = __expf(negA * sp);
    const float beta = 1.f / (1.f + __expf(-br[t]));
    if (REPLAY) {
      if (blockIdx.x == 0 && hv % (p.HV / p.H) == 0 && warp == 0)
        __stcg(reinterpret_cast<float4*>(cur + (t * p.H + h) * KD) + lane, make_float4(k[0], k[1], k[2], k[3]));
      if (lane < ROWS) {
        float mine = vr[t][0];
#pragma unroll
        for (int j = 1; j < ROWS; ++j)
          if (lane == j) mine = vr[t][j];
        __stcg(cur + TMAX * p.H * KD + (t * p.HV + hv) * KD + row0 + lane, mine);
      }
      if (blockIdx.x == 0 && threadIdx.x == 0)
        __stcg(reinterpret_cast<float2*>(cur + TMAX * (p.H + p.HV) * KD) + t * p.HV + hv, make_float2(decay, beta));
    }
    float d[ROWS];
#pragma unroll
    for (int j = 0; j < ROWS; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) hs[j][e] *= decay;
      d[j] = hs[j][0] * k[0] + hs[j][1] * k[1] + hs[j][2] * k[2] + hs[j][3] * k[3];
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
#pragma unroll
      for (int j = 0; j < ROWS; ++j) d[j] += __shfl_xor_sync(0xffffffff, d[j], s);
    float ov[ROWS];
#pragma unroll
    for (int j = 0; j < ROWS; ++j) {
      const float u = (vr[t][j] - d[j]) * beta;
#pragma unroll
      for (int e = 0; e < 4; ++e) hs[j][e] += u * k[e];
      ov[j] = hs[j][0] * q[0] + hs[j][1] * q[1] + hs[j][2] * q[2] + hs[j][3] * q[3];
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
#pragma unroll
      for (int j = 0; j < ROWS; ++j) ov[j] += __shfl_xor_sync(0xffffffff, ov[j], s);
    if (lane < ROWS) {
      float mine = ov[0];
#pragma unroll
      for (int j = 1; j < ROWS; ++j)
        if (lane == j) mine = ov[j];
      p.o[((long long)(bos + t) * p.HV + hv) * KD + row0 + lane] = __float2half_rn(mine);
    }
    const int sl = REPLAY ? -1 : p.idx[n * p.s_idx + t];
    if (sl >= 0) {
#pragma unroll
      for (int j = 0; j < ROWS; ++j)
        __stcg(reinterpret_cast<float4*>(p.state + sl * p.s_state + ((long long)hv * KD + row0 + j) * KD) + lane,
               make_float4(hs[j][0], hs[j][1], hs[j][2], hs[j][3]));
    }
  }
  if (REPLAY) round_done(p);
}
}  // namespace

void sm70_gdn_verify_out(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias,
                    torch::Tensor state, torch::Tensor o, torch::Tensor cu, torch::Tensor idx,
                    std::optional<torch::Tensor> nacc, int64_t H, int64_t HV, double scale, int64_t cfg,
                    std::optional<torch::Tensor> rbuf, std::optional<torch::Tensor> rn, std::optional<torch::Tensor> ctr,
                    int64_t tmax) {
  const c10::cuda::CUDAGuard guard(qkv.device());
  TORCH_CHECK(tmax >= 1 && tmax <= TMAX);
  TORCH_CHECK(state.dtype() == torch::kFloat32 && state.size(-1) == KD && state.size(-2) == KD);
  TORCH_CHECK(qkv.stride(1) == 1 && qkv.dtype() == torch::kHalf);
  const int N = cu.size(0) - 1;
  TORCH_CHECK(qkv.size(0) <= N * TMAX);
  P p{};
  p.qkv = reinterpret_cast<const half*>(qkv.data_ptr());
  p.a = reinterpret_cast<const half*>(a.data_ptr());
  p.b = reinterpret_cast<const half*>(b.data_ptr());
  p.state = state.data_ptr<float>();
  p.o = reinterpret_cast<half*>(o.data_ptr());
  p.cu = cu.data_ptr<int>();
  p.idx = idx.data_ptr<int>();
  p.nacc = nacc ? nacc->data_ptr<int>() : nullptr;
  p.s_state = state.stride(0);
  p.s_idx = idx.dim() == 2 ? idx.stride(0) : 1;
  p.qkv_stride = qkv.stride(0);
  p.H = H;
  p.HV = HV;
  p.scale = static_cast<float>(scale);
  p.lda = a.dim() == 2 ? a.stride(0) : HV;
  p.ldb = b.dim() == 2 ? b.stride(0) : HV;
  TORCH_CHECK(a.stride(-1) == 1 && b.stride(-1) == 1, "a/b must be unit-stride in the head dim");
  const bool replay = rbuf.has_value();
  if (replay) {
    TORCH_CHECK(rn && ctr && rbuf->dtype() == torch::kFloat32);
    p.rbuf = rbuf->data_ptr<float>();
    p.rn = rn->data_ptr<int>();
    p.ctr = reinterpret_cast<unsigned*>(ctr->data_ptr<int>());
    const long long seq = (long long)TMAX * (H * KD + HV * KD + HV * 2);
    p.nmax = static_cast<int>(rbuf->numel() / (2 * seq));
    TORCH_CHECK(p.nmax >= N, "replay buffer too small");
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  auto go3 = [&](auto wr, auto rp, auto tt) {
    constexpr int W = decltype(wr)::first_type::value, R = decltype(wr)::second_type::value;
    constexpr bool RP = decltype(rp)::value;
    constexpr int TTV = decltype(tt)::value;
    const dim3 grid(KD / (W * R), N * HV);
    if (A_log.dtype() == torch::kFloat32 && dt_bias.dtype() == torch::kFloat32)
      gdn_verify<float, float, W, R, RP, TTV><<<grid, W * 32, 0, stream>>>(p, A_log.data_ptr<float>(), dt_bias.data_ptr<float>());
    else if (A_log.dtype() == torch::kFloat32 && dt_bias.dtype() == torch::kHalf)
      gdn_verify<float, half, W, R, RP, TTV><<<grid, W * 32, 0, stream>>>(p, A_log.data_ptr<float>(),
                                                                reinterpret_cast<const half*>(dt_bias.data_ptr()));
    else if (A_log.dtype() == torch::kHalf && dt_bias.dtype() == torch::kHalf)
      gdn_verify<half, half, W, R, RP, TTV><<<grid, W * 32, 0, stream>>>(p, reinterpret_cast<const half*>(A_log.data_ptr()),
                                                               reinterpret_cast<const half*>(dt_bias.data_ptr()));
    else
      TORCH_CHECK(false, "A_log/dt_bias dtypes");
  };
  auto go2 = [&](auto wr, auto rp) {
    if (tmax <= 1) go3(wr, rp, std::integral_constant<int, 1>{});
    else if (tmax <= 2) go3(wr, rp, std::integral_constant<int, 2>{});
    else if (tmax <= 4) go3(wr, rp, std::integral_constant<int, 4>{});
    else if (tmax <= 5) go3(wr, rp, std::integral_constant<int, 5>{});
    else go3(wr, rp, std::integral_constant<int, 8>{});
  };
  auto go = [&](auto wr) {
    if (replay)
      go2(wr, std::true_type{});
    else
      go2(wr, std::false_type{});
  };
  using I = std::integral_constant<int, 0>;
  switch (cfg) {
    case 1: go(std::pair<std::integral_constant<int, 2>, std::integral_constant<int, 4>>{}); break;
    default: TORCH_CHECK(false, "cfg");
  }
  (void)I{};
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

