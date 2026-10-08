// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Resident task ownership is informed by MonoMoE and Mirage MPK. Integer
// formulas and FP16/Q8_1 boundaries use the existing shared GGUF reader.
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
constexpr int M = 5, TopK = 10, MaxRoutes = 50, MaxGroups = 5;
constexpr int Threads = 128, GateRows = 8, OutRows = 32, WorkersPerStripe = 6;

struct Work {
  uint32_t presence[16], prefix[16];
  int unique[MaxRoutes], route[MaxRoutes][M], count;
  uint32_t mask[MaxRoutes], chunks[MaxRoutes * MaxGroups], experts[MaxRoutes];
};

struct Args {
  const half* x;
  const int* ids;
  const float* probabilities;
  const uint8_t* gate;
  const uint8_t* up;
  const StridedPtr* down;
  const StridedPtr* stats;
  Q8_1* activation;
  half* activated;
  Q8_1* hidden;
  half* routes;
  half* output;
  Work* work;
  int experts, n, k, stride, topk;
};

__device__ uint32_t acquire(const uint32_t* p) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.u32 %0, [%1];"
               : "=r"(value)
               : "l"(p)
               : "memory");
  return value;
}

__device__ uint32_t publish(uint32_t* p) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
               : "=r"(old)
               : "l"(p)
               : "memory");
  return old;
}

// Quantization and the small routing map share one existing kernel boundary.
// Resetting readiness here makes each captured invocation self-contained.
__global__ void prepare(Args a) {
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int group = blockIdx.x, groups = a.k / 32;
  for (int token = warp; token < M; token += Threads / 32)
    vllm::sm70_gguf::quantize_q8_1_warp(
        a.activation + token * groups + group,
        __half2float(a.x[token * a.k + group * 32 + lane]));
  if (blockIdx.x != 0) return;
  auto& w = *a.work;
  if (t < 16) w.presence[t] = 0;
  if (t < MaxRoutes) {
    w.mask[t] = 0;
    w.experts[t] = 0;
  }
  for (int i = t; i < MaxRoutes * M; i += Threads) (&w.route[0][0])[i] = -1;
  for (int i = t; i < MaxRoutes * MaxGroups; i += Threads) w.chunks[i] = 0;
  __syncthreads();
  if (t < M * a.topk) {
    const int e = a.ids[t];
    if (e >= 0 && e < a.experts) atomicOr(&w.presence[e / 32], 1u << (e % 32));
  }
  __syncthreads();
  if (!warp) {
    uint32_t bits = lane < 16 ? w.presence[lane] : 0;
    int prefix = __popc(bits), count = prefix;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      const int value = __shfl_up_sync(0xffffffffu, prefix, offset);
      if (lane >= offset) prefix += value;
    }
    if (lane < 16) w.prefix[lane] = prefix - count;
    if (lane == 31) w.count = prefix;
    for (int u = prefix - count; bits; bits &= bits - 1, ++u)
      w.unique[u] = lane * 32 + __ffs(bits) - 1;
  }
  __syncthreads();
  if (t < M * a.topk) {
    const int e = a.ids[t];
    if (e >= 0 && e < a.experts) {
      const int u = w.prefix[e / 32] +
                    __popc(w.presence[e / 32] & ((1u << (e % 32)) - 1));
      w.route[u][t / a.topk] = t;
      atomicOr(&w.mask[u], 1u << (t / a.topk));
    }
  }
}

template <int Type>
struct State {
  using Dot = vllm::sm70_gguf::LatticeDot<Type>;
  uint32_t book[Dot::kBookWords], masks[16];
  float partial[MaxGroups][M][OutRows];
  int route[M], expert;
  uint32_t mask, ready;
};

template <int Type>
__device__ void gate_task(const Args& a, State<Type>& s, int task) {
  using Dot = typename State<Type>::Dot;
  const int stripes = a.n / GateRows, u = task / stripes,
            stripe = task % stripes;
  const int t = threadIdx.x, lane = t % 32, row = t / 16;
  if (t < M) s.route[t] = a.work->route[u][t];
  if (!t) {
    s.expert = a.work->unique[u];
    s.mask = a.work->mask[u];
  }
  __syncthreads();
  const int col = stripe * GateRows + row;
  const uint8_t* gate = a.gate + (int64_t(s.expert) * a.n + col) * a.stride;
  const uint8_t* up = a.up + (int64_t(s.expert) * a.n + col) * a.stride;
  float g[M] = {}, v[M] = {};
  for (int group = lane % 16; group < a.k / 32; group += 16) {
    const auto gw = Dot::load_group(gate, group, s.book, s.masks);
    const auto uw = Dot::load_group(up, group, s.book, s.masks);
#pragma unroll
    for (int token = 0; token < M; ++token)
      if (s.mask & (1u << token)) {
        const auto& x = a.activation[token * (a.k / 32) + group];
        g[token] += Dot::dot(gw, x);
        v[token] += Dot::dot(uw, x);
      }
  }
#pragma unroll
  for (int token = 0; token < M; ++token)
    if (s.mask & (1u << token)) {
#pragma unroll
      for (int offset = 8; offset; offset /= 2) {
        g[token] += __shfl_down_sync(0xffffffffu, g[token], offset, 16);
        v[token] += __shfl_down_sync(0xffffffffu, v[token], offset, 16);
      }
      if (lane % 16 == 0) {
        const float g16 = __half2float(__float2half_rn(g[token]));
        const half value = __hmul(__float2half_rn(g16 / (1.f + expf(-g16))),
                                  __float2half_rn(v[token]));
        a.activated[s.route[token] * a.n + col] = value;
      }
    }
  __threadfence();
  __syncthreads();
  const int chunk = stripe / 4, groups = a.n / 32;
  if (!t) s.ready = publish(a.work->chunks + u * MaxGroups + chunk) == 3;
  __syncthreads();
  if (s.ready) {
    for (int token = t / 32; token < M; token += Threads / 32)
      if (s.mask & (1u << token))
        vllm::sm70_gguf::quantize_q8_1_warp(
            a.hidden + s.route[token] * groups + chunk,
            __half2float(
                a.activated[s.route[token] * a.n + chunk * 32 + lane]));
    __threadfence();
    __syncthreads();
    if (!t) publish(a.work->experts + u);
  }
  __syncthreads();
}

template <int Type, int Down>
__device__ void down_task(const Args& a, State<Type>& s, int u) {
  using Dot = vllm::sm70_gguf::CanonicalIntegerDot<Down>;
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int col = (blockIdx.x % (a.k / OutRows)) * OutRows + lane;
  if (t < M) s.route[t] = a.work->route[u][t];
  if (!t) s.expert = a.work->unique[u];
  __syncthreads();
  const int groups = a.n / 32;
  for (int group = warp; group < groups; group += Threads / 32) {
    const auto weight = Dot::load_group(
        a.down[s.expert].ptr, a.stats[s.expert].ptr, a.k, a.n, col, group);
#pragma unroll
    for (int token = 0; token < M; ++token)
      if (s.route[token] >= 0)
        s.partial[group][token][lane] =
            Dot::dot(weight, a.hidden[s.route[token] * groups + group]);
  }
  __syncthreads();
  for (int token = warp; token < M; token += Threads / 32)
    if (s.route[token] >= 0) {
      float sum = 0;
      for (int group = 0; group < groups; ++group)
        sum += s.partial[group][token][lane];
      a.routes[int64_t(s.route[token]) * a.k + col] = __float2half_rn(sum);
    }
  __syncthreads();
}

template <int Type, int Down>
__global__ __launch_bounds__(Threads, WorkersPerStripe) void worker(Args a) {
  extern __shared__ __align__(16) uint8_t memory[];
  auto& s = *reinterpret_cast<State<Type>*>(memory);
  using Dot = typename State<Type>::Dot;
  Dot::initialize(s.book, s.masks);
  const int count = a.work->count, columns = a.k / OutRows;
  int gate = blockIdx.x, down = blockIdx.x / columns;
  while (gate < count * (a.n / GateRows) || down < count) {
    if (!threadIdx.x)
      s.ready = down < count && acquire(a.work->experts + down) == a.n / 32;
    __syncthreads();
    if (s.ready) {
      down_task<Type, Down>(a, s, down);
      down += WorkersPerStripe;
    } else if (gate < count * (a.n / GateRows)) {
      gate_task(a, s, gate);
      gate += gridDim.x;
    }
  }
}

__global__ void unroute(Args a) {
  __shared__ float partial[4][32];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int token = blockIdx.y, col = blockIdx.x * 32 + lane;
  float sum = 0;
  for (int route = warp; route < a.topk; route += 4) {
    const int slot = token * a.topk + route, e = a.ids[slot];
    if (e >= 0 && e < a.experts)
      sum += __half2float(a.routes[int64_t(slot) * a.k + col]) *
             a.probabilities[slot];
  }
  partial[warp][lane] = sum;
  __syncthreads();
  if (!warp) {
    sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) sum += partial[i][lane];
    a.output[token * a.k + col] = __float2half_rn(sum);
  }
}

template <int Type, int Down>
void launch(const Args& a) {
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const int bytes = std::max(
      int(sizeof(State<Type>)),
      int(properties->sharedMemPerMultiprocessor / (WorkersPerStripe + 1)) +
          512);
  TORCH_CHECK(a.k / OutRows <= properties->multiProcessorCount &&
                  bytes <= properties->sharedMemPerBlockOptin,
              "resident task grid unavailable");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      worker<Type, Down>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  int resident = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident, worker<Type, Down>, Threads, bytes));
  TORCH_CHECK(resident >= WorkersPerStripe,
              "resident task occupancy unavailable");
  const auto stream = at::cuda::getCurrentCUDAStream();
  prepare<<<a.k / 32, Threads, 0, stream>>>(a);
  worker<Type, Down>
      <<<a.k / OutRows * WorkersPerStripe, Threads, bytes, stream>>>(a);
  unroute<<<dim3(a.k / 32, M), Threads, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(torch::Tensor out, torch::Tensor x, torch::Tensor ids,
         torch::Tensor probabilities, torch::Tensor gate, torch::Tensor up,
         torch::Tensor down, torch::Tensor stats, torch::Tensor activation,
         torch::Tensor activated, torch::Tensor hidden, torch::Tensor routes,
         torch::Tensor work, int64_t source_type, int64_t down_type) {
  TORCH_CHECK(x.dim() == 2 && x.size(0) == M && x.size(1) > 0 &&
                  x.size(1) <= 2560 && x.size(1) % 256 == 0,
              "task path requires M5");
  TORCH_CHECK(gate.dim() == 3 && up.sizes() == gate.sizes() &&
                  gate.size(0) > 0 && gate.size(0) <= 512 && gate.size(1) > 0 &&
                  gate.size(1) <= 160 && gate.size(1) % 32 == 0 &&
                  ids.dim() == 2 && ids.size(0) == M && ids.size(1) > 0 &&
                  ids.size(1) <= TopK && probabilities.sizes() == ids.sizes(),
              "invalid task expert routing");
  const int k = x.size(1), n = gate.size(1), experts = gate.size(0),
            topk = ids.size(1);
  TORCH_CHECK(source_type == 18 || source_type == 21 || source_type == 22,
              "requires IQ3_XXS/S or IQ2_S");
  TORCH_CHECK(down_type == 20 || down_type == 42,
              "requires canonical integer down");
  const int block_bytes = source_type == 18 ? 98 : source_type == 21 ? 110 : 82;
  TORCH_CHECK(
      gate.size(2) >= k / 256 * block_bytes && out.sizes() == x.sizes() &&
          activation.numel() == M * (k / 32) * 36 &&
          activated.numel() == M * topk * n &&
          hidden.numel() == M * topk * (n / 32) * 36 &&
          routes.numel() == M * topk * k && work.numel() >= sizeof(Work) &&
          down.nbytes() >= experts * sizeof(StridedPtr) &&
          stats.nbytes() >= experts * sizeof(StridedPtr),
      "invalid task workspace");
  for (const auto& tensor : {out, x, activated, routes})
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
                "requires FP16 boundaries");
  for (const auto& tensor : {activation, hidden, work, gate, up, down, stats})
    TORCH_CHECK(tensor.scalar_type() == torch::kUInt8, "requires byte storage");
  TORCH_CHECK(ids.scalar_type() == torch::kInt32 &&
                  probabilities.scalar_type() == torch::kFloat32,
              "requires int32 routing and FP32 probabilities");
  for (const auto& tensor : {out, x, ids, probabilities, gate, up, down, stats,
                             activation, activated, hidden, routes, work})
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                    tensor.device() == x.device(),
                "requires contiguous tensors on one GPU");
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 7 &&
                  at::cuda::getCurrentDeviceProperties()->minor == 0,
              "requires SM70");
  Args a{reinterpret_cast<const half*>(x.data_ptr()),
         ids.data_ptr<int>(),
         probabilities.data_ptr<float>(),
         gate.data_ptr<uint8_t>(),
         up.data_ptr<uint8_t>(),
         reinterpret_cast<const StridedPtr*>(down.data_ptr()),
         reinterpret_cast<const StridedPtr*>(stats.data_ptr()),
         reinterpret_cast<Q8_1*>(activation.data_ptr()),
         reinterpret_cast<half*>(activated.data_ptr()),
         reinterpret_cast<Q8_1*>(hidden.data_ptr()),
         reinterpret_cast<half*>(routes.data_ptr()),
         reinterpret_cast<half*>(out.data_ptr()),
         reinterpret_cast<Work*>(work.data_ptr()),
         experts,
         n,
         k,
         int(gate.size(2)),
         topk};
#define CASE(TYPE)         \
  case TYPE:               \
    if (down_type == 20)   \
      launch<TYPE, 20>(a); \
    else                   \
      launch<TYPE, 42>(a); \
    break
  switch (source_type) {
    CASE(18);
    CASE(21);
    CASE(22);
  }
#undef CASE
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(sm70_gguf_persistent, m) {
  m.def(
      "queued(Tensor(a!) out, Tensor x, Tensor ids, Tensor probabilities, "
      "Tensor gate, Tensor up, "
      "Tensor down, Tensor stats, Tensor(b!) activation, Tensor(c!) activated, "
      "Tensor(d!) hidden, Tensor(e!) routes, Tensor(f!) work, int source_type, "
      "int down_type) -> ()");
  m.impl("queued", torch::kCUDA, &run);
}
