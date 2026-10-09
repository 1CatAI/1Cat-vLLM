// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Weight-major task scheduling follows MonoMoE and Mirage MPK; integer
// decoding is shared with the existing GGUF dp4a reader. No TMA/FP8 MMA.
// Experimental M5 local routed chain. It has no model dispatch until the
// numerical, chain and same-wheel model gates pass.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include "gguf_dp4a.cuh"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"

namespace {
using turbomind::gemm::StridedPtr;
using vllm::sm70_gguf::Q8_1;
constexpr int M = 5, TopK = 10, Rows = 32, Threads = 512, Lanes = 16;
constexpr int MaxRoutes = M * TopK, MaxKGroups = 80, MaxHGroups = 5, Batch = 3;

__device__ uint64_t acquire(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ void release(uint64_t* p, uint64_t v) {
  asm volatile("st.release.gpu.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

struct Args {
  const half* x;
  const int* ids;
  const float* probabilities;
  const uint8_t* gate;
  const uint8_t* up;
  const StridedPtr* down;
  const StridedPtr* stats;
  half* output;
  Q8_1* hidden;
  half* routes;
  uint64_t* ready;
  uint64_t* epochs;
  int experts, n, k, stride, topk;
};

template <int Type>
struct State {
  using Dot = vllm::sm70_gguf::LatticeDot<Type>;
  uint32_t book[Dot::kBookWords], masks[16], presence[16], prefix[16];
  Q8_1 x[M][MaxKGroups];
  int unique[MaxRoutes], route[MaxRoutes][M], count;
  uint32_t token_mask[MaxRoutes], todo[Batch], work;
  uint64_t epoch;
  half activated[M][Rows];
  float partial[Batch][MaxHGroups][M][Rows];
  float final_sum[M][4][Rows];
};

template <int Type>
__device__ void prepare(const Args& a, State<Type>& s) {
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t < 16) s.presence[t] = 0;
  if (t < MaxRoutes) s.token_mask[t] = 0;
  for (int i = t; i < MaxRoutes * M; i += Threads) (&s.route[0][0])[i] = -1;
  if (!t) {
    s.epoch = a.epochs[blockIdx.x] + 1;
    a.epochs[blockIdx.x] = s.epoch;
  }
  __syncthreads();
  for (int r = t; r < M * a.topk; r += Threads) {
    const int e = a.ids[r];
    if (e >= 0 && e < a.experts) atomicOr(&s.presence[e / 32], 1u << (e % 32));
  }
  __syncthreads();
  if (!warp) {
    uint32_t mask = lane < 16 ? s.presence[lane] : 0;
    int pre = __popc(mask);
    const int count = pre;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      const int v = __shfl_up_sync(0xffffffffu, pre, d);
      if (lane >= d) pre += v;
    }
    if (lane < 16) s.prefix[lane] = pre - count;
    if (lane == 31) s.count = pre;
    for (int u = pre - count; mask; mask &= mask - 1, ++u)
      s.unique[u] = lane * 32 + __ffs(mask) - 1;
  }
  __syncthreads();
  for (int r = t; r < M * a.topk; r += Threads) {
    const int e = a.ids[r];
    if (e >= 0 && e < a.experts) {
      const int u = s.prefix[e / 32] +
                    __popc(s.presence[e / 32] & ((1u << (e % 32)) - 1));
      s.route[u][r / a.topk] = r;
      atomicOr(&s.token_mask[u], 1u << (r / a.topk));
    }
  }
  using Dot = typename State<Type>::Dot;
  Dot::initialize(s.book, s.masks);
  for (int i = warp; i < M * (a.k / 32); i += Threads / 32) {
    const int token = i / (a.k / 32), group = i % (a.k / 32);
    vllm::sm70_gguf::quantize_q8_1_warp(
        &s.x[token][group], __half2float(a.x[token * a.k + group * 32 + lane]));
  }
  __syncthreads();
}

template <int Type, bool Publish = true>
__device__ void gate_up(const Args& a, State<Type>& s, int task) {
  using Dot = typename State<Type>::Dot;
  const int groups = a.n / 32, u = task / groups, chunk = task % groups;
  const int t = threadIdx.x, lane = t % 32, row = t / Lanes;
  const int e = s.unique[u], col = chunk * Rows + row;
  const uint32_t mask = s.token_mask[u];
  const uint8_t* gate = a.gate + (int64_t(e) * a.n + col) * a.stride;
  const uint8_t* up = a.up + (int64_t(e) * a.n + col) * a.stride;
  float g[M] = {}, v[M] = {};
  for (int kg = lane % Lanes; kg < a.k / 32; kg += Lanes) {
    const auto gw = Dot::load_group(gate, kg, s.book, s.masks);
    const auto uw = Dot::load_group(up, kg, s.book, s.masks);
#pragma unroll
    for (int token = 0; token < M; ++token) {
      if (mask & (1u << token)) {
        g[token] += Dot::dot(gw, s.x[token][kg]);
        v[token] += Dot::dot(uw, s.x[token][kg]);
      }
    }
  }
#pragma unroll
  for (int token = 0; token < M; ++token) {
    if (mask & (1u << token)) {
#pragma unroll
      for (int d = Lanes / 2; d; d /= 2) {
        g[token] += __shfl_down_sync(0xffffffffu, g[token], d, Lanes);
        v[token] += __shfl_down_sync(0xffffffffu, v[token], d, Lanes);
      }
      if (lane % Lanes == 0) {
        const half gh = __float2half_rn(g[token]),
                   uh = __float2half_rn(v[token]);
        const float gf = __half2float(gh);
        s.activated[token][row] =
            __hmul(__float2half_rn(gf / (1.f + expf(-gf))), uh);
      }
    }
  }
  __syncthreads();
  const int warp = t / 32;
  if (warp < M && (mask & (1u << warp))) {
    const int r = s.route[u][warp];
    vllm::sm70_gguf::quantize_q8_1_warp(a.hidden + r * groups + chunk,
                                        __half2float(s.activated[warp][lane]));
  }
  if constexpr (Publish) {
    // Every writer completes its own global stores before the leader publishes.
    __threadfence();
    __syncthreads();
    if (!t) release(a.ready + task, s.epoch);
    __syncthreads();
  }
}

// Ablate weight reuse without resident consumers or per-CTA quantization.
// The input is produced once by the existing Q8_1 activation operator.
template <int Type>
__global__ __launch_bounds__(Threads,
                             1) void gate_reuse(Args a,
                                                const Q8_1* activation) {
  extern __shared__ __align__(16) uint8_t memory[];
  auto& s = *reinterpret_cast<State<Type>*>(memory);
  const int t = threadIdx.x, candidate = blockIdx.y;
  const int e = a.ids[candidate];
  if (t < M) s.route[0][t] = -1;
  if (!t) {
    s.unique[0] = e;
    s.count = M * a.topk;
    s.token_mask[0] = 0;
  }
  __syncthreads();
  if (t < M * a.topk && a.ids[t] == e) {
    atomicMin(&s.count, t);
    s.route[0][t / a.topk] = t;
    atomicOr(&s.token_mask[0], 1u << (t / a.topk));
  }
  __syncthreads();
  if (s.count != candidate || e < 0 || e >= a.experts) return;
  const int groups = a.k / 32;
  for (int i = t; i < M * groups * 9; i += Threads) {
    const int token = i / (groups * 9), offset = i % (groups * 9);
    if (s.token_mask[0] & (1u << token))
      reinterpret_cast<uint32_t*>(s.x[token])[offset] =
          reinterpret_cast<const uint32_t*>(activation)[i];
  }
  using Dot = typename State<Type>::Dot;
  Dot::initialize(s.book, s.masks);
  gate_up<Type, false>(a, s, blockIdx.x);
}

template <int Type, int Down>
__device__ void down_chunk(const Args& a, State<Type>& s, int base) {
  using Dot = vllm::sm70_gguf::CanonicalIntegerDot<Down>;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int b = warp / MaxHGroups, group = warp % MaxHGroups;
  const int groups = a.n / 32, col = blockIdx.x * Rows + lane;
  if (b < Batch && (s.work & (1u << warp))) {
    const int u = base + b, e = s.unique[u];
    const auto weight =
        Dot::load_group(a.down[e].ptr, a.stats[e].ptr, a.k, a.n, col, group);
#pragma unroll
    for (int token = 0; token < M; ++token) {
      const int r = s.route[u][token];
      if (r >= 0)
        s.partial[b][group][token][lane] =
            Dot::dot(weight, a.hidden[r * groups + group]);
    }
  }
  __syncthreads();
  if (warp < Batch * MaxHGroups && !lane && (s.work & (1u << warp)))
    atomicOr(&s.todo[b], 1u << group);
  __syncthreads();
}

template <int Type>
__device__ void finish_batch(const Args& a, State<Type>& s, int base) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int b = warp / M, token = warp % M, groups = a.n / 32;
  if (b < Batch && base + b < s.count) {
    const int r = s.route[base + b][token];
    if (r >= 0) {
      float sum = 0;
      for (int g = 0; g < groups; ++g) sum += s.partial[b][g][token][lane];
      a.routes[int64_t(r) * a.k + blockIdx.x * Rows + lane] =
          __float2half_rn(sum);
    }
  }
  __syncthreads();
}

template <int Type>
__device__ void unroute(const Args& a, State<Type>& s) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int col = blockIdx.x * Rows + lane;
  for (int token = warp / 4; token < M; token += Threads / 128) {
    float sum = 0;
    for (int r = warp % 4; r < a.topk; r += 4) {
      const int slot = token * a.topk + r, e = a.ids[slot];
      if (e >= 0 && e < a.experts)
        sum += __half2float(a.routes[int64_t(slot) * a.k + col]) *
               a.probabilities[slot];
    }
    s.final_sum[token][warp % 4][lane] = sum;
  }
  __syncthreads();
  if (warp < M) {
    float sum = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) sum += s.final_sum[warp][w][lane];
    a.output[warp * a.k + col] = __float2half_rn(sum);
  }
}

template <int Type, int Down, bool FullBatch = false>
__global__ __launch_bounds__(Threads, 1) void persistent(Args a) {
  extern __shared__ __align__(16) uint8_t memory[];
  auto& s = *reinterpret_cast<State<Type>*>(memory);
  prepare(a, s);
  const int groups = a.n / 32;
  int gate_task = blockIdx.x, base = 0;
  if (threadIdx.x < Batch) s.todo[threadIdx.x] = 0;
  __syncthreads();
  while (base < s.count) {
    if (!threadIdx.x) {
      uint32_t ready = 0;
      for (int b = 0; b < Batch && base + b < s.count; ++b)
        for (int g = 0; g < groups; ++g)
          if (!(s.todo[b] & (1u << g)) &&
              acquire(a.ready + (base + b) * groups + g) == s.epoch)
            ready |= 1u << (b * MaxHGroups + g);
      s.work = ready;
    }
    __syncthreads();
    uint32_t wanted = 0;
    if constexpr (FullBatch)
      for (int b = 0; b < Batch && base + b < s.count; ++b)
        wanted |= (((1u << groups) - 1) & ~s.todo[b]) << (b * MaxHGroups);
    const bool enough =
        !FullBatch || s.work == wanted || gate_task >= s.count * groups;
    if (s.work && enough) {
      down_chunk<Type, Down>(a, s, base);
      bool complete = true;
      for (int b = 0; b < Batch && base + b < s.count; ++b)
        complete &= s.todo[b] == (1u << groups) - 1;
      if (complete) {
        finish_batch(a, s, base);
        base += Batch;
        if (threadIdx.x < Batch) s.todo[threadIdx.x] = 0;
        __syncthreads();
      }
    } else if (gate_task < s.count * groups) {
      // Do not block on a consumer while its producer is still queued.
      gate_up<Type>(a, s, gate_task);
      gate_task += gridDim.x;
    }
  }
  unroute(a, s);
}

template <int Type, int Down, bool FullBatch = false>
void launch(const Args& a) {
  int sm_count, shared_per_sm, shared_limit;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, c10::cuda::current_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &shared_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
      c10::cuda::current_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit,
                                        cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                        c10::cuda::current_device()));
  const int bytes = std::max(int(sizeof(State<Type>)), shared_per_sm / 2 + 512);
  TORCH_CHECK(a.k / Rows <= sm_count && bytes <= shared_limit,
              "persistent residency unavailable");
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(persistent<Type, Down, FullBatch>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  int resident = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident, persistent<Type, Down, FullBatch>, Threads, bytes));
  TORCH_CHECK(resident >= 1, "persistent block cannot reside");
  persistent<Type, Down, FullBatch>
      <<<a.k / Rows, Threads, bytes, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(torch::Tensor out, torch::Tensor x, torch::Tensor ids,
         torch::Tensor probabilities, torch::Tensor gate, torch::Tensor up,
         torch::Tensor down, torch::Tensor stats, torch::Tensor hidden,
         torch::Tensor routes, torch::Tensor ready, torch::Tensor epochs,
         int64_t source_type, int64_t down_type, int64_t schedule) {
  TORCH_CHECK(schedule == 0 || schedule == 1, "invalid resident schedule");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) == M && x.size(1) > 0 && x.size(1) <= 2560 &&
          x.size(1) % 256 == 0,
      "persistent prototype requires M5 and K256..2560 divisible by 256");
  TORCH_CHECK(gate.dim() == 3 && up.sizes() == gate.sizes() &&
                  gate.size(0) <= 512 && gate.size(1) > 0 &&
                  gate.size(1) <= 160 && gate.size(1) % 32 == 0,
              "invalid persistent expert bank");
  TORCH_CHECK(ids.dim() == 2 && ids.size(0) == M && ids.size(1) > 0 &&
                  ids.size(1) <= TopK && probabilities.sizes() == ids.sizes(),
              "invalid routes");
  const int k = x.size(1), n = gate.size(1), experts = gate.size(0),
            topk = ids.size(1);
  TORCH_CHECK(experts > 0 && out.sizes() == x.sizes() &&
                  hidden.numel() == M * topk * (n / 32) * 36 &&
                  routes.numel() == M * topk * k &&
                  ready.numel() >= M * topk * (n / 32) &&
                  epochs.numel() == k / Rows,
              "invalid fixed workspace");
  TORCH_CHECK(source_type == 18 || source_type == 21 || source_type == 22,
              "requires IQ3_XXS/S or IQ2_S");
  TORCH_CHECK(down_type == 20 || down_type == 42,
              "requires canonical IQ4_NL or Q2_0 down");
  const int block_bytes = source_type == 18 ? 98 : source_type == 21 ? 110 : 82;
  TORCH_CHECK(gate.size(2) >= (k / 256) * block_bytes &&
                  down.nbytes() >= experts * sizeof(StridedPtr) &&
                  stats.nbytes() >= experts * sizeof(StridedPtr),
              "expert storage truncated");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16 &&
                  routes.scalar_type() == torch::kFloat16 &&
                  ids.scalar_type() == torch::kInt32 &&
                  probabilities.scalar_type() == torch::kFloat32 &&
                  ready.scalar_type() == torch::kInt64 &&
                  epochs.scalar_type() == torch::kInt64 &&
                  hidden.scalar_type() == torch::kUInt8 &&
                  gate.scalar_type() == torch::kUInt8 &&
                  up.scalar_type() == torch::kUInt8 &&
                  down.scalar_type() == torch::kUInt8 &&
                  stats.scalar_type() == torch::kUInt8,
              "invalid persistent dtype");
  for (const auto& t : {out, x, ids, probabilities, gate, up, down, stats,
                        hidden, routes, ready, epochs})
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == x.device(),
                "requires contiguous tensors on one GPU");
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 7 &&
                  at::cuda::getCurrentDeviceProperties()->minor == 0,
              "requires SM70");
  Args a{reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
         ids.data_ptr<int>(),
         probabilities.data_ptr<float>(),
         gate.data_ptr<uint8_t>(),
         up.data_ptr<uint8_t>(),
         reinterpret_cast<const StridedPtr*>(down.data_ptr()),
         reinterpret_cast<const StridedPtr*>(stats.data_ptr()),
         reinterpret_cast<half*>(out.data_ptr<at::Half>()),
         reinterpret_cast<Q8_1*>(hidden.data_ptr()),
         reinterpret_cast<half*>(routes.data_ptr<at::Half>()),
         reinterpret_cast<uint64_t*>(ready.data_ptr()),
         reinterpret_cast<uint64_t*>(epochs.data_ptr()),
         experts,
         n,
         k,
         int(gate.size(2)),
         topk};
#define CASE(TYPE)                 \
  case TYPE:                       \
    if (schedule == 1) {           \
      if (down_type == 20)         \
        launch<TYPE, 20, true>(a); \
      else                         \
        launch<TYPE, 42, true>(a); \
    } else {                       \
      if (down_type == 20)         \
        launch<TYPE, 20>(a);       \
      else                         \
        launch<TYPE, 42>(a);       \
    }                              \
    break
  switch (source_type) {
    CASE(18);
    CASE(21);
    CASE(22);
  }
#undef CASE
}

template <int Type>
void launch_gate_reuse(const Args& a, torch::Tensor activation) {
  gate_reuse<Type><<<dim3(a.n / Rows, M * a.topk), Threads, sizeof(State<Type>),
                     at::cuda::getCurrentCUDAStream()>>>(
      a, reinterpret_cast<const Q8_1*>(activation.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_gate_reuse(torch::Tensor hidden, torch::Tensor activation,
                    torch::Tensor ids, torch::Tensor gate, torch::Tensor up,
                    int64_t source_type) {
  TORCH_CHECK(activation.dim() == 3 && activation.size(0) == M &&
                  activation.size(1) > 0 && activation.size(1) <= MaxKGroups &&
                  activation.size(1) % 8 == 0 && activation.size(2) == 36,
              "requires M5 Q8_1 activation");
  TORCH_CHECK(ids.dim() == 2 && ids.size(0) == M && ids.size(1) > 0 &&
                  ids.size(1) <= TopK && gate.dim() == 3 &&
                  up.sizes() == gate.sizes() && gate.size(0) > 0 &&
                  gate.size(0) <= 512 && gate.size(1) > 0 &&
                  gate.size(1) <= 160 && gate.size(1) % Rows == 0,
              "invalid shared expert bank");
  TORCH_CHECK(source_type == 18 || source_type == 21 || source_type == 22,
              "requires IQ3_XXS/S or IQ2_S");
  const int k = activation.size(1) * 32, n = gate.size(1), topk = ids.size(1);
  const int block_bytes = source_type == 18 ? 98 : source_type == 21 ? 110 : 82;
  TORCH_CHECK(hidden.numel() == M * topk * (n / 32) * 36 &&
                  gate.size(2) >= (k / 256) * block_bytes &&
                  ids.scalar_type() == torch::kInt32,
              "invalid shared workspace");
  for (const auto& tensor : {hidden, activation, gate, up})
    TORCH_CHECK(tensor.scalar_type() == torch::kUInt8, "invalid shared dtype");
  for (const auto& tensor : {hidden, activation, ids, gate, up})
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                    tensor.device() == activation.device(),
                "requires one contiguous GPU");
  const c10::cuda::CUDAGuard guard(activation.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 7 &&
                  at::cuda::getCurrentDeviceProperties()->minor == 0,
              "requires SM70");
  Args a{};
  a.ids = ids.data_ptr<int>();
  a.gate = gate.data_ptr<uint8_t>();
  a.up = up.data_ptr<uint8_t>();
  a.hidden = reinterpret_cast<Q8_1*>(hidden.data_ptr());
  a.experts = gate.size(0);
  a.n = n;
  a.k = k;
  a.stride = gate.size(2);
  a.topk = topk;
  switch (source_type) {
    case 18:
      launch_gate_reuse<18>(a, activation);
      break;
    case 21:
      launch_gate_reuse<21>(a, activation);
      break;
    case 22:
      launch_gate_reuse<22>(a, activation);
      break;
  }
}
}  // namespace

TORCH_LIBRARY(sm70_gguf_persistent, m) {
  m.def(
      "run(Tensor(a!) out, Tensor x, Tensor ids, Tensor probabilities, Tensor "
      "gate, Tensor up, "
      "Tensor down, Tensor stats, Tensor(b!) hidden, Tensor(c!) routes, "
      "Tensor(d!) ready, "
      "Tensor(e!) epochs, int source_type, int down_type, int schedule=0) -> "
      "()");
  m.impl("run", torch::kCUDA, &run);
  m.def(
      "gate_reuse(Tensor(a!) hidden, Tensor activation, Tensor ids, "
      "Tensor gate, Tensor up, int source_type) -> ()");
  m.impl("gate_reuse", torch::kCUDA, &run_gate_reuse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
