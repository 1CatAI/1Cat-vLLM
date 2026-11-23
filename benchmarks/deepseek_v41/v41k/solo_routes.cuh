// SPDX-License-Identifier: Apache-2.0
// One stable CUDA IPC receive slot per DP peer. A sender publishes a stripe
// after all its stores are visible system-wide. A receiver returns the slot credit
// after consuming the stripe; the sender can start local experts immediately.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>

namespace v41_solo_native {
constexpr int rows = 6, hidden = 5120, topk = 6, packs = hidden / 8;
constexpr int packet_bytes = hidden * 2 + topk * 8;
constexpr int data_offset = 1024;
struct Peers {
  char* p[4];
};

__device__ __forceinline__ void publish(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}
__device__ __forceinline__ unsigned observe(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ uint4 packet_load(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
template <int stripes>
__global__ void broadcast_routes(const uint4* x, const float* prob,
                                 const int* ids, const int* real,
                                 const int* table, const int* counts, int width,
                                 int logical, int dead, uint4* ox, float* op,
                                 int* oi, Peers peers, int me, int owner,
                                 unsigned* sequence) {
  const int block = blockIdx.x, row = block / stripes, stripe = block % stripes;
  const int first = stripe * blockDim.x + threadIdx.x;
  const unsigned seq = sequence[block];
  // A sender owns two packet slots at every receiver. It reserves a free
  // slot before writing; the receiver returns its credit after the last read.
  // Per-sender slots also allow the active DP owner to change between graphs.
  for (int peer = 0; peer < 4; ++peer)
    peers.p[peer] += (owner * 2 + (seq & 1u)) * 65536;
  unsigned* local_flags = reinterpret_cast<unsigned*>(peers.p[me]);
  if (me == owner) {
    if (threadIdx.x < 4 && threadIdx.x != me) {
      unsigned* credit = local_flags + gridDim.x + block * 4 + threadIdx.x;
      while (observe(credit) != ~0u) {}
      publish(credit, 0u);
    }
    __syncthreads();

    for (int k = first; k < packs; k += blockDim.x * stripes) {
      uint4 value = x[row * packs + k];
      ox[row * packs + k] = value;
#pragma unroll
      for (int peer = 0; peer < 4; ++peer)
        if (peer != me)
          reinterpret_cast<uint4*>(peers.p[peer] + data_offset +
                                   row * packet_bytes)[k] = value;
    }
    if (first < topk) {
      int k = row * topk + first, e = ids[k], mapped = dead;
      if (row < *real && e >= 0 && e < logical)
        mapped = table[e * width + row % counts[e]];
      float value = prob[k];
      op[k] = value;
      oi[k] = mapped;
#pragma unroll
      for (int peer = 0; peer < 4; ++peer)
        if (peer != me) {
          char* meta =
              peers.p[peer] + data_offset + row * packet_bytes + hidden * 2;
          reinterpret_cast<float*>(meta)[first] = value;
          reinterpret_cast<int*>(meta)[topk + first] = mapped;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x < 4 && threadIdx.x != me) {
      publish(reinterpret_cast<unsigned*>(peers.p[threadIdx.x]) + block, seq);
    }
  } else {
    if (threadIdx.x == 0)
      while (observe(local_flags + block) != seq) {
      }
    __syncthreads();
    const char* packet = peers.p[me] + data_offset + row * packet_bytes;
    for (int k = first; k < packs; k += blockDim.x * stripes)
      ox[row * packs + k] =
          packet_load(reinterpret_cast<const uint4*>(packet) + k);
    if (first < topk) {
      const char* meta = packet + hidden * 2;
      op[row * topk + first] =
          reinterpret_cast<const volatile float*>(meta)[first];
      oi[row * topk + first] =
          reinterpret_cast<const volatile int*>(meta)[topk + first];
    }
    __syncthreads();
    if (threadIdx.x == 0)
      publish(reinterpret_cast<unsigned*>(peers.p[owner]) + gridDim.x +
                  block * 4 + me,
              ~0u);
  }
  __syncthreads();
  if (threadIdx.x == 0) sequence[block] = seq + 1;
}

void dispatch(at::Tensor x, at::Tensor p, at::Tensor ids, at::Tensor real,
              at::Tensor table, at::Tensor counts, int64_t dead, at::Tensor ox,
              at::Tensor op, at::Tensor oi, int64_t p0, int64_t p1, int64_t p2,
              int64_t p3, int64_t me, int64_t owner, at::Tensor seq,
              int64_t stripes) {
  TORCH_CHECK(me >= 0 && me < 4 && owner >= 0 && owner < 4 &&
              (stripes == 1 || stripes == 4));
  const int input_rows = me == owner ? rows : 0;
  TORCH_CHECK(x.sizes() == at::IntArrayRef({input_rows, hidden}) &&
              x.scalar_type() == at::kHalf);
  TORCH_CHECK(p.sizes() == at::IntArrayRef({input_rows, topk}) &&
              ids.sizes() == p.sizes());
  TORCH_CHECK(ox.sizes() == at::IntArrayRef({rows, hidden}) &&
              op.sizes() == at::IntArrayRef({rows, topk}) &&
              oi.sizes() == op.sizes());
  TORCH_CHECK(p.scalar_type() == at::kFloat && op.scalar_type() == at::kFloat &&
              ox.scalar_type() == at::kHalf);
  for (const auto& t : {ids, real, table, counts, oi, seq})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  for (const auto& t : {x, p, ids, real, table, counts, ox, op, oi, seq})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  TORCH_CHECK(table.dim() == 2 && table.size(0) == counts.numel() &&
              table.size(1) > 0 && real.numel() == 1 &&
              seq.numel() >= rows * stripes);
  c10::cuda::CUDAGuard guard(x.device());
  const Peers peers{{(char*)p0, (char*)p1, (char*)p2, (char*)p3}};
  const auto stream = at::cuda::getCurrentCUDAStream();
#define RUN(S)                                                              \
  broadcast_routes<S><<<rows * S, 128, 0, stream>>>(                        \
      (const uint4*)x.data_ptr(), p.data_ptr<float>(), ids.data_ptr<int>(), \
      real.data_ptr<int>(), table.data_ptr<int>(), counts.data_ptr<int>(),  \
      table.size(1), table.size(0), dead, (uint4*)ox.data_ptr(),            \
      op.data_ptr<float>(), oi.data_ptr<int>(), peers, me, owner,           \
      (unsigned*)seq.data_ptr<int>())
  if (stripes == 1) {
    RUN(1);
  } else {
    RUN(4);
  }
#undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace v41_solo_native
TORCH_LIBRARY(v41solo, m) {
  m.def(
      "dispatch(Tensor x, Tensor p, Tensor ids, Tensor real, Tensor table, "
      "Tensor counts, int dead, Tensor(a!) ox, Tensor(b!) op, Tensor(c!) oi, "
      "int p0, int p1, int p2, int p3, int me, int owner, Tensor(d!) seq, int "
      "stripes) -> ()");
}
TORCH_LIBRARY_IMPL(v41solo, CUDA, m) {
  m.impl("dispatch", v41_solo_native::dispatch);
}
