// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Convolution, recurrence and norm with head-local delayed conv-state commit.
#include <torch/types.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda/atomic>
#include <cuda_fp16.h>

namespace {
constexpr int Threads = 128, Producers = 192, Blocks = 204;
__device__ float warp_sum(float x) {
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
__device__ float column_sum(float x) {
  for (int d = 8; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
__device__ void wait(uint32_t* p, uint32_t generation) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(*p);
  while (flag.load(cuda::memory_order_acquire) != generation) {
  }
}
__device__ float conv_value(const half* raw, const half* state,
                            const half* weight, int row, int channel, int slot,
                            int selected, int qstride, int cstride,
                            int cdstride, int ctstride) {
  float value = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int source_row = row - 3 + j;
    const half input = source_row < 0
                           ? state[slot * cstride + channel * cdstride +
                                   (selected + source_row + 3) * ctstride]
                           : raw[source_row * qstride + channel];
    value += __half2float(__hmul(input, weight[channel * 4 + j]));
  }
  return __half2float(__float2half_rn(value / (1.f + expf(-value))));
}
__device__ void commit_conv(const half* raw, half* state, int channel, int slot,
                            int selected, int m, int qstride, int cstride,
                            int cdstride, int ctstride) {
  half* destination = state + slot * cstride + channel * cdstride;
  const half old1 = destination[(selected + 1) * ctstride];
  const half old2 = destination[(selected + 2) * ctstride];
  destination[0] = old1;
  destination[ctstride] = old2;
  for (int row = 0; row < m; ++row)
    destination[(2 + row) * ctstride] = raw[row * qstride + channel];
}
__global__ __launch_bounds__(Threads, 3) void gdn_conv_tile_chain(
    const half* qkv, const half* a, const half* b, const float* al,
    const half* bias, const half* z, const half* weight, const float* initial,
    float* states, half* core, float* norms, uint32_t* flags, uint32_t* epochs,
    half* output, int m, float eps, half* conv, const half* cw, const int* ids,
    const int* accepted, int qstride, int cstride, int cdstride, int ctstride,
    int sstride, bool silu_gate) {
  const int t = threadIdx.x, block = blockIdx.x;
  const uint32_t generation = epochs[block] + 1;
  const int conv_slot = ids[0];
  const int selected = m == 1 ? 0 : accepted[0] - 1;
  const int initial_slot = ids[selected];
  __shared__ float q[128], k[128], values[8], sums[8], normalizer_q,
      normalizer_k, decay, beta;
  if (block < Producers) {
    const int hv = block / 16, tile = block % 16, hq = hv / 3, lane = t % 16,
              v = tile * 8 + t / 16;
    float state[8];
#pragma unroll
    for (int r = 0; r < 8; ++r)
      state[r] = initial[initial_slot * sstride + (hv * 128 + v) * 128 +
                         r * 16 + lane];
    for (int row = 0; row < m; ++row) {
      q[t] = conv_value(qkv, conv, cw, row, hq * 128 + t, conv_slot, selected,
                        qstride, cstride, cdstride, ctstride);
      k[t] = conv_value(qkv, conv, cw, row, 512 + hq * 128 + t, conv_slot,
                        selected, qstride, cstride, cdstride, ctstride);
      if (lane == 0)
        values[t / 16] =
            conv_value(qkv, conv, cw, row, 1024 + hv * 128 + v, conv_slot,
                       selected, qstride, cstride, cdstride, ctstride);
      float sq = warp_sum(q[t] * q[t]), sk = warp_sum(k[t] * k[t]);
      if ((t & 31) == 0) {
        sums[t / 32] = sq;
        sums[4 + t / 32] = sk;
      }
      __syncthreads();
      if (t == 0) {
        sq = 0;
        sk = 0;
        for (int w = 0; w < 4; ++w) {
          sq += sums[w];
          sk += sums[4 + w];
        }
        normalizer_q = rsqrtf(sq + 1e-6f);
        normalizer_k = rsqrtf(sk + 1e-6f);
        const float av =
            __half2float(a[row * 12 + hv]) + __half2float(bias[hv]);
        const float softplus = av > 20.f ? av : log1pf(expf(av));
        decay = expf(-expf(al[hv]) * softplus);
        beta = 1.f / (1.f + expf(-__half2float(b[row * 12 + hv])));
      }
      __syncthreads();
      q[t] *= normalizer_q;
      k[t] *= normalizer_k;
      __syncthreads();
      float dot = 0;
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        state[r] *= decay;
        dot = fmaf(state[r], k[r * 16 + lane], dot);
      }
      dot = column_sum(dot);
      const float delta = (values[t / 16] - dot) * beta;
      dot = 0;
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        state[r] = fmaf(delta, k[r * 16 + lane], state[r]);
        states[ids[row] * sstride + (hv * 128 + v) * 128 + r * 16 + lane] =
            state[r];
        dot = fmaf(state[r], q[r * 16 + lane], dot);
      }
      dot = column_sum(dot) * 0.08838834764831845f;
      if (lane == 0) {
        const half rounded = __float2half_rn(dot);
        core[(row * 12 + hv) * 128 + v] = rounded;
        const float fv = __half2float(rounded);
        sums[t / 16] = fv * fv;
      }
      __syncthreads();
      if (t == 0) {
        float sum = 0;
        for (int v = 0; v < 8; ++v) sum += sums[v];
        norms[row * Producers + block] = sum;
      }
      __threadfence();
      __syncthreads();
      if (t == 0) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(
            flags[row * Producers + block]);
        flag.store(generation, cuda::memory_order_release);
      }
      __syncthreads();
    }
  } else {
    const int hv = block - Producers;
    for (int row = 0; row < m; ++row) {
      float sum = 0;
      if (t < 16) {
        wait(flags + row * Producers + hv * 16 + t, generation);
        sum = norms[row * Producers + hv * 16 + t];
      }
      sum = warp_sum(sum);
      if (t == 0) normalizer_q = rsqrtf(sum / 128.f + eps);
      __syncthreads();
      const int index = (row * 12 + hv) * 128 + t;
      const float zv = __half2float(z[index]);
      const float value =
          __half2float(core[index]) * normalizer_q * __half2float(weight[t]);
      output[index] =
          __float2half_rn(value * (silu_gate ? zv : 1.f) / (1.f + expf(-zv)));
      __syncthreads();
    }
    // Each V channel is read only by this head's producers. Q/K channels
    // are shared by three heads: delay their commit until those 48 local
    // producers finish every token. This never waits on the whole grid.
    if (hv % 3 == 0 && t < 48)
      wait(flags + (m - 1) * Producers + (hv / 3) * 48 + t, generation);
    __syncthreads();
    commit_conv(qkv, conv, 1024 + hv * 128 + t, conv_slot, selected, m, qstride,
                cstride, cdstride, ctstride);
    if (hv % 3 == 0) {
      commit_conv(qkv, conv, (hv / 3) * 128 + t, conv_slot, selected, m,
                  qstride, cstride, cdstride, ctstride);
      commit_conv(qkv, conv, 512 + (hv / 3) * 128 + t, conv_slot, selected, m,
                  qstride, cstride, cdstride, ctstride);
    }
  }
  if (t == 0) epochs[block] = generation;
}
}  // namespace
void qwen38_gdn_conv_norm_sm70_out(
    torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor al,
    torch::Tensor bias, torch::Tensor z, torch::Tensor weight,
    torch::Tensor conv, torch::Tensor conv_weight, torch::Tensor state,
    torch::Tensor indices, torch::Tensor accepted, torch::Tensor core,
    torch::Tensor norms, torch::Tensor flags, torch::Tensor epochs,
    torch::Tensor output, double epsilon, bool silu_gate) {
  const c10::cuda::CUDAGuard guard(qkv.device());
  const int m = qkv.size(0);
  TORCH_CHECK((m == 1 || m == 5) && qkv.size(1) == 2560 && qkv.stride(1) == 1);
  TORCH_CHECK(conv_weight.sizes() == at::IntArrayRef({2560, 4}) &&
              conv_weight.is_contiguous());
  TORCH_CHECK(conv.size(1) == 2560 && conv.size(2) >= m + 2);
  TORCH_CHECK(state.stride(1) == 16384 && state.stride(2) == 128 &&
              state.stride(3) == 1);
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.stride(-1) == 1);
  TORCH_CHECK(accepted.scalar_type() == at::kInt && indices.numel() >= m);
  for (const auto& t :
       {qkv, a, b, bias, z, weight, conv, conv_weight, core, output})
    TORCH_CHECK(t.is_cuda() && t.device() == qkv.device() &&
                t.scalar_type() == at::kHalf);
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && z.is_contiguous());
  for (const auto& tensor : {al, state, norms})
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == qkv.device() &&
                tensor.scalar_type() == at::kFloat);
  for (const auto& tensor : {indices, accepted, flags, epochs})
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == qkv.device() &&
                tensor.scalar_type() == at::kInt && tensor.is_contiguous());
  TORCH_CHECK(al.numel() == 12 && al.is_contiguous() && bias.numel() == 12);
  TORCH_CHECK(a.sizes() == at::IntArrayRef({m, 12}) && b.sizes() == a.sizes());
  TORCH_CHECK(z.sizes() == at::IntArrayRef({m, 12, 128}) &&
              weight.numel() == 128);
  TORCH_CHECK(core.is_contiguous() && core.numel() >= m * 1536 &&
              norms.is_contiguous() && norms.numel() >= m * 192 &&
              flags.numel() >= m * 192 && epochs.numel() == Blocks &&
              accepted.numel() >= 1);
  TORCH_CHECK(output.sizes() == z.sizes() && output.is_contiguous());
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, gdn_conv_tile_chain, Threads, 0));
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0 &&
              properties->cooperativeLaunch &&
              active * properties->multiProcessorCount >= Blocks);
  const half* qp = (half*)qkv.data_ptr();
  const half* ap = (half*)a.data_ptr();
  const half* bp = (half*)b.data_ptr();
  const float* alp = al.data_ptr<float>();
  const half* dp = (half*)bias.data_ptr();
  const half* zp = (half*)z.data_ptr();
  const half* wp = (half*)weight.data_ptr();
  const float* ip = state.data_ptr<float>();
  float* sp = state.data_ptr<float>();
  half* cp = (half*)core.data_ptr();
  float* np = norms.data_ptr<float>();
  auto* fp = (uint32_t*)flags.data_ptr();
  auto* ep = (uint32_t*)epochs.data_ptr();
  half* op = (half*)output.data_ptr();
  half* cv = (half*)conv.data_ptr();
  const half* cw = (half*)conv_weight.data_ptr();
  const int* ids = indices.data_ptr<int>();
  const int* sel = accepted.data_ptr<int>();
  int qr = qkv.stride(0), cs = conv.stride(0), cd = conv.stride(1),
      ct = conv.stride(2), ss = state.stride(0);
  float eps = epsilon;
  void* args[] = {&qp,  &ap,  &bp, &alp, &dp, &zp,       &wp,  &ip,       &sp,
                  &cp,  &np,  &fp, &ep,  &op, (void*)&m, &eps, &cv,       &cw,
                  &ids, &sel, &qr, &cs,  &cd, &ct,       &ss,  &silu_gate};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      (const void*)gdn_conv_tile_chain, dim3(Blocks), dim3(Threads), args, 0,
      c10::cuda::getCurrentCUDAStream()));
}
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_gdn_conv_norm_sm70_out(Tensor qkv, Tensor a, Tensor b, Tensor "
      "al, "
      "Tensor bias, Tensor z, Tensor weight, Tensor(a!) conv, Tensor "
      "conv_weight, "
      "Tensor(b!) state, Tensor indices, Tensor accepted, Tensor(c!) core, "
      "Tensor(d!) norms, Tensor(e!) flags, Tensor(f!) epochs, Tensor(g!) "
      "output, "
      "float epsilon, bool silu_gate) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_gdn_conv_norm_sm70_out", &qwen38_gdn_conv_norm_sm70_out);
}
