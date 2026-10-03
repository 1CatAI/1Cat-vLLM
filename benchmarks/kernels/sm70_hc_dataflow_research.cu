// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research only: not selected by production dispatch.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {
constexpr int Threads = 128, OutputTile = 8, DownTile = 2, MaxPeers = 8;
struct Peer {
  half* down;
  unsigned* down_flags;
  half* output;
  unsigned* up_flags;
};
struct Params {
  const half *residual, *block, *injection, *norm_weight, *down_weight,
      *up_weight;
  half *combined, *normalized;
  unsigned *norm_flags, *epochs;
  unsigned long long* trace;
  Peer peers[MaxPeers];
  int m, hidden, hc, low_rank, tp, rank, down_ctas, up_ctas;
  float eps;
};

__device__ unsigned acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ void release(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}
__device__ void wait(const unsigned* p, unsigned generation) {
  while (acquire(p) != generation) __nanosleep(128);
}
__device__ float warp_sum(float v) {
  for (int d = 16; d; d /= 2) v += __shfl_down_sync(0xffffffffu, v, d);
  return v;
}
__device__ float sigmoid(float v) { return 1.f / (1.f + expf(-v)); }

__device__ void mark(Params p, int row, int stage) {
  if (!threadIdx.x && p.trace) {
    unsigned long long time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    const int grid = p.hc + p.down_ctas + p.up_ctas;
    p.trace[(size_t(row) * grid + blockIdx.x) * 5 + stage] = time;
  }
}

// Each role has a fixed CTA owner. Cooperative admission guarantees that
// every producer can run while consumers wait, without any grid-wide barrier.
template <int M>
__global__ void __launch_bounds__(Threads) hc_dataflow(Params p) {
  extern __shared__ half cache[];
  const int t = threadIdx.x, lane = t & 31, warp = t / 32;
  const int total_hidden = p.hc * p.hidden;
  const int local_hidden = p.hidden / p.tp;
  const int local_lora = p.low_rank / p.tp;
  const int local_injection = p.hc / p.tp;
  const int local_down = local_lora + local_injection;
  const unsigned generation = p.epochs[blockIdx.x] + 1u;
  const int batch = M ? M : p.m;
  const Peer local = p.peers[p.rank];
  mark(p, 0, 0);

  // This load is independent of the current input, and occurs before polling.
  if (blockIdx.x >= p.hc && blockIdx.x < p.hc + p.down_ctas) {
    const int tile = blockIdx.x - p.hc;
    for (int i = t; i < DownTile * total_hidden; i += Threads) {
      const int row = tile * DownTile + i / total_hidden;
      const int global_row =
          row < local_lora
              ? p.rank * local_lora + row
              : p.low_rank + p.rank * local_injection + row - local_lora;
      cache[i] = row < local_down
                     ? p.down_weight[size_t(global_row) * total_hidden +
                                     i % total_hidden]
                     : __float2half_rn(0.f);
    }
  } else if (blockIdx.x >= p.hc + p.down_ctas) {
    const int tile = blockIdx.x - p.hc - p.down_ctas;
    for (int i = t; i < OutputTile * p.hc * p.low_rank; i += Threads) {
      const int r = i / p.low_rank;
      const int h = p.rank * local_hidden + tile * OutputTile + r % OutputTile;
      const int global_row = (r / OutputTile) * p.hidden + h;
      cache[i] = p.up_weight[size_t(global_row) * p.low_rank + i % p.low_rank];
    }
  }
  __syncthreads();
  mark(p, 0, 1);

#pragma unroll 1
  for (int row = 0; row < batch; ++row) {
    if (blockIdx.x < p.hc) {
      const int branch = blockIdx.x;
      const size_t base = size_t(row) * total_hidden + branch * p.hidden;
      const float injection =
          2.f * sigmoid(__half2float(p.injection[row * p.hc + branch]) / p.hc);
      float squares = 0.f;
      for (int h = t; h < p.hidden; h += Threads) {
        const half v = __float2half_rn(
            __half2float(p.residual[base + h]) +
            injection * __half2float(p.block[size_t(row) * p.hidden + h]));
        p.combined[base + h] = v;
        const float f = __half2float(v);
        squares = fmaf(f, f, squares);
      }
      squares = warp_sum(squares);
      auto* partial = reinterpret_cast<float*>(cache);
      if (!lane) partial[warp] = squares;
      __syncthreads();
      if (!t)
        partial[0] = rsqrtf(
            (partial[0] + partial[1] + partial[2] + partial[3]) / p.hidden +
            p.eps);
      __syncthreads();
      const float inverse = partial[0];
      for (int h = t; h < p.hidden; h += Threads) {
        const float y = __half2float(p.combined[base + h]) * inverse;
        p.normalized[base + h] = __float2half_rn(
            y + y * __half2float(p.norm_weight[branch * p.hidden + h]));
      }
      __threadfence();
      __syncthreads();
      if (!t) release(p.norm_flags + row * p.hc + branch, generation);
      mark(p, row, 2);
      mark(p, row, 3);
    } else if (blockIdx.x < p.hc + p.down_ctas) {
      const int tile = blockIdx.x - p.hc;
      for (int b = t; b < p.hc; b += Threads)
        wait(p.norm_flags + row * p.hc + b, generation);
      __syncthreads();
      mark(p, row, 2);
      float value[DownTile] = {0.f, 0.f};
      for (int k = t; k < total_hidden; k += Threads) {
        const float x =
            __half2float(p.normalized[size_t(row) * total_hidden + k]);
#pragma unroll
        for (int r = 0; r < DownTile; ++r)
          value[r] =
              fmaf(x, __half2float(cache[r * total_hidden + k]), value[r]);
      }
      auto* partial = reinterpret_cast<float*>(cache + DownTile * total_hidden);
#pragma unroll
      for (int r = 0; r < DownTile; ++r) {
        value[r] = warp_sum(value[r]);
        if (!lane) partial[r * 4 + warp] = value[r];
      }
      __syncthreads();
      if (t < DownTile && tile * DownTile + t < local_down) {
        const int local_row = tile * DownTile + t;
        const int global_row = local_row < local_lora
                                   ? p.rank * local_lora + local_row
                                   : p.low_rank + p.rank * local_injection +
                                         local_row - local_lora;
        const float projected = __half2float(
            __float2half_rn(partial[t * 4] + partial[t * 4 + 1] +
                            partial[t * 4 + 2] + partial[t * 4 + 3]));
        const float scaled = projected / p.hc;
        const half result = __float2half_rn(
            local_row < local_lora ? scaled * sigmoid(scaled) : projected);
        for (int dest = 0; dest < p.tp; ++dest)
          p.peers[dest].down[size_t(row) * (p.low_rank + p.hc) + global_row] =
              result;
        __threadfence_system();
      }
      __syncthreads();
      if (t < p.tp)
        release(
            p.peers[t].down_flags + (row * p.tp + p.rank) * p.down_ctas + tile,
            generation);
      mark(p, row, 3);
    } else {
      const int tile = blockIdx.x - p.hc - p.down_ctas;
      for (int i = t; i < p.tp * p.down_ctas; i += Threads)
        wait(local.down_flags + row * p.tp * p.down_ctas + i, generation);
      __syncthreads();
      mark(p, row, 2);
      auto* gates =
          reinterpret_cast<float*>(cache + OutputTile * p.hc * p.low_rank);
      for (int r = warp; r < OutputTile * p.hc; r += Threads / 32) {
        float sum = 0.f;
        for (int k = lane; k < p.low_rank; k += 32)
          sum = fmaf(
              __half2float(cache[r * p.low_rank + k]),
              __half2float(local.down[size_t(row) * (p.low_rank + p.hc) + k]),
              sum);
        sum = warp_sum(sum);
        if (!lane) gates[r] = sigmoid(__half2float(__float2half_rn(sum)));
      }
      __syncthreads();
      if (t < OutputTile) {
        const int h = p.rank * local_hidden + tile * OutputTile + t;
        float mixed = 0.f;
        for (int b = 0; b < p.hc; ++b)
          mixed = fmaf(
              gates[b * OutputTile + t],
              __half2float(
                  p.normalized[size_t(row) * total_hidden + b * p.hidden + h]),
              mixed);
        const half output = __float2half_rn(mixed / p.hc);
        for (int dest = 0; dest < p.tp; ++dest)
          p.peers[dest].output[size_t(row) * p.hidden + h] = output;
        __threadfence_system();
      }
      __syncthreads();
      mark(p, row, 3);
      if (t < p.tp) {
        release(p.peers[t].up_flags + (row * p.tp + p.rank) * p.up_ctas + tile,
                generation);
        wait(local.up_flags + (row * p.tp + t) * p.up_ctas + tile, generation);
      }
    }
    __syncthreads();
    mark(p, row, 4);
  }
  if (!t) p.epochs[blockIdx.x] = generation;
}

template <int M>
void launch(Params p, int shared, cudaStream_t stream) {
  int device, blocks;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      hc_dataflow<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, hc_dataflow<M>, Threads, shared));
  const int grid = p.hc + p.down_ctas + p.up_ctas;
  TORCH_CHECK(
      prop.cooperativeLaunch && grid <= blocks * prop.multiProcessorCount,
      "HC dataflow roles exceed cooperative resident capacity");
  void* args[] = {&p};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)hc_dataflow<M>, grid,
                                             Threads, args, shared, stream));
}

// Peer access is prepared outside capture. Tensor owners remain with Python.
void prepare_peers(const std::vector<torch::Tensor>& outputs) {
  for (const auto& source : outputs) {
    c10::cuda::CUDAGuard guard(source.device());
    for (const auto& dest : outputs) {
      if (source.device() == dest.device()) continue;
      int access;
      C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&access, source.get_device(),
                                             dest.get_device()));
      TORCH_CHECK(access, "HC dataflow requires direct peer access");
      const auto status = cudaDeviceEnablePeerAccess(dest.get_device(), 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled)
        cudaGetLastError();
      else
        C10_CUDA_CHECK(status);
    }
  }
}

void run(torch::Tensor residual, torch::Tensor block, torch::Tensor injection,
         torch::Tensor norm_weight, torch::Tensor down_weight,
         torch::Tensor up_weight, torch::Tensor combined,
         torch::Tensor normalized, torch::Tensor norm_flags,
         torch::Tensor epochs, const std::vector<torch::Tensor>& down,
         const std::vector<torch::Tensor>& down_flags,
         const std::vector<torch::Tensor>& output,
         const std::vector<torch::Tensor>& up_flags, int rank, double eps,
         std::optional<torch::Tensor> trace) {
  c10::cuda::CUDAGuard guard(residual.device());
  TORCH_CHECK(residual.dim() == 2 && block.dim() == 2 && injection.dim() == 2 &&
              down_weight.dim() == 2 && up_weight.dim() == 2);
  const int tp = output.size(), m = residual.size(0), hc = injection.size(1);
  const int hidden = block.size(1), low_rank = up_weight.size(1);
  TORCH_CHECK(m > 0 && hc > 0 && tp > 0 && tp <= MaxPeers && rank >= 0 &&
              rank < tp);
  TORCH_CHECK(block.size(0) == m && injection.size(0) == m && eps > 0);
  TORCH_CHECK(output[rank].device() == residual.device());
  TORCH_CHECK(hc % tp == 0 && low_rank % tp == 0 &&
              hidden % (tp * OutputTile) == 0);
  TORCH_CHECK(residual.size(1) == hc * hidden &&
              norm_weight.numel() == hc * hidden);
  TORCH_CHECK(down_weight.size(0) >= low_rank + hc &&
              down_weight.size(1) == hc * hidden);
  TORCH_CHECK(up_weight.size(0) == hc * hidden);
  const int dc = (low_rank / tp + hc / tp + DownTile - 1) / DownTile;
  const int uc = hidden / tp / OutputTile, grid = hc + dc + uc;
  for (const auto& tensor : {residual, block, injection, norm_weight,
                             down_weight, up_weight, combined, normalized})
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == residual.device() &&
                tensor.scalar_type() == torch::kFloat16 &&
                tensor.is_contiguous());
  TORCH_CHECK(combined.sizes() == residual.sizes() &&
              normalized.sizes() == residual.sizes());
  TORCH_CHECK(norm_flags.is_cuda() &&
              norm_flags.device() == residual.device() &&
              norm_flags.scalar_type() == torch::kInt32 &&
              norm_flags.is_contiguous() && norm_flags.numel() >= m * hc);
  TORCH_CHECK(epochs.is_cuda() && epochs.device() == residual.device() &&
              epochs.scalar_type() == torch::kInt32 && epochs.is_contiguous() &&
              epochs.numel() >= grid);
  TORCH_CHECK(down.size() == tp && down_flags.size() == tp &&
              up_flags.size() == tp);
  Params p{};
  if (trace) {
    TORCH_CHECK(trace->device() == residual.device() &&
                trace->scalar_type() == torch::kInt64 &&
                trace->is_contiguous() && trace->numel() >= m * grid * 5);
    p.trace = reinterpret_cast<unsigned long long*>(trace->data_ptr());
  }
  p.m = m;
  p.hidden = hidden;
  p.hc = hc;
  p.low_rank = low_rank;
  p.tp = tp;
  p.rank = rank;
  p.down_ctas = dc;
  p.up_ctas = uc;
  p.eps = eps;
  p.residual = (half*)residual.data_ptr();
  p.block = (half*)block.data_ptr();
  p.injection = (half*)injection.data_ptr();
  p.norm_weight = (half*)norm_weight.data_ptr();
  p.down_weight = (half*)down_weight.data_ptr();
  p.up_weight = (half*)up_weight.data_ptr();
  p.combined = (half*)combined.data_ptr();
  p.normalized = (half*)normalized.data_ptr();
  p.norm_flags = (unsigned*)norm_flags.data_ptr();
  p.epochs = (unsigned*)epochs.data_ptr();
  for (int i = 0; i < tp; ++i) {
    TORCH_CHECK(down[i].is_cuda() && output[i].device() == down[i].device() &&
                down_flags[i].device() == down[i].device() &&
                up_flags[i].device() == down[i].device());
    TORCH_CHECK(down[i].scalar_type() == torch::kFloat16 &&
                output[i].scalar_type() == torch::kFloat16 &&
                down_flags[i].scalar_type() == torch::kInt32 &&
                up_flags[i].scalar_type() == torch::kInt32);
    TORCH_CHECK(down[i].is_contiguous() && output[i].is_contiguous() &&
                down_flags[i].is_contiguous() && up_flags[i].is_contiguous());
    TORCH_CHECK(down[i].numel() >= m * (low_rank + hc) &&
                output[i].numel() >= m * hidden &&
                down_flags[i].numel() >= m * tp * dc &&
                up_flags[i].numel() >= m * tp * uc);
    p.peers[i] = {
        (half*)down[i].data_ptr(), (unsigned*)down_flags[i].data_ptr(),
        (half*)output[i].data_ptr(), (unsigned*)up_flags[i].data_ptr()};
  }
  const int shared = std::max({int(DownTile * hc * hidden * sizeof(half) + 32),
                               int(OutputTile * hc * low_rank * sizeof(half) +
                                   OutputTile * hc * sizeof(float)),
                               32});
  auto stream = at::cuda::getCurrentCUDAStream();
  if (m == 1)
    launch<1>(p, shared, stream);
  else if (m == 5)
    launch<5>(p, shared, stream);
  else
    launch<0>(p, shared,
              stream);  // Generic batch support, not a fallback path.
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare_peers", &prepare_peers);
  m.def("run", &run, pybind11::arg("residual"), pybind11::arg("block"),
        pybind11::arg("injection"), pybind11::arg("norm_weight"),
        pybind11::arg("down_weight"), pybind11::arg("up_weight"),
        pybind11::arg("combined"), pybind11::arg("normalized"),
        pybind11::arg("norm_flags"), pybind11::arg("epochs"),
        pybind11::arg("down"), pybind11::arg("down_flags"),
        pybind11::arg("output"), pybind11::arg("up_flags"),
        pybind11::arg("rank"), pybind11::arg("eps"),
        pybind11::arg("trace") = pybind11::none());
}
