// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Direct device E4M3/FP16 history: FP16 QK operands, FP32 QK/PV,
// probability and merge. No protected-hot ownership or staging is required.
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace device_qsa {
constexpr int D = 256, H = 6, C = 64, PAD = 264;
struct Args {
  const half* q;
  const unsigned char* history;
  const float* scales;
  const int* indices;
  const int* table;
  const int* requests;
  const int64_t* positions;
  const int* lengths;
  const half* gate;
  half* out;
  float* partial;
  float* state;
  int M, blocks, page, width, table_width, num_requests, splits;
  int64_t qs0, qs1, is0, ts0, os0, os1, gs0, gs1;
};

__device__ __forceinline__ float fast_exp2(float x) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

__device__ __forceinline__ float e4m3(unsigned c) {
  const unsigned sign = (c & 128u) << 24;
  const unsigned magnitude = c & 127u;
  if (magnitude == 127u) return __uint_as_float(sign | 0x7fc00000u);
  const unsigned normal = (magnitude << 20) + (120u << 23);
  const float subnormal = float(magnitude & 7u) * 0x1p-9f;
  const unsigned bits = magnitude < 8u ? __float_as_uint(subnormal) : normal;
  return __uint_as_float(bits | sign);
}

__device__ __forceinline__ void mma(float (&d)[8], uint32_t a0, uint32_t a1,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 decode8(const unsigned char* source,
                                         float scale) {
  const uint2 packed = __ldg(reinterpret_cast<const uint2*>(source));
  uint4 result;
  unsigned* words = reinterpret_cast<unsigned*>(&result);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned p = j < 2 ? packed.x : packed.y;
    const unsigned code0 = (p >> (16 * (j & 1))) & 255u;
    const unsigned code1 = (p >> (16 * (j & 1) + 8)) & 255u;
    const half lo = __float2half_rn(__fmul_rn(e4m3(code0), scale));
    const half hi = __float2half_rn(__fmul_rn(e4m3(code1), scale));
    words[j] =
        unsigned(__half_as_ushort(lo)) | (unsigned(__half_as_ushort(hi)) << 16);
  }
  return result;
}

template <bool FP8>
__global__ __launch_bounds__(256) void partial(Args a) {
  const int row = blockIdx.y, split = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  __shared__ __align__(16) half query[8][PAD];
  __shared__ __align__(16) half kv[C][PAD];
  __shared__ float scores[4][8][C];
  __shared__ int64_t slots[C];
  const int req = a.requests[row];
  const int64_t position = a.positions[row];
  const int length = req >= 0 && req < a.num_requests ? a.lengths[req] : 0;
  if (t < C) {
    const int col = split * C + t;
    const int logical = col < a.width ? a.indices[row * a.is0 + col] : -1;
    int64_t slot = -1;
    if (req >= 0 && req < a.num_requests && logical >= 0 &&
        logical <= position && logical < length &&
        logical / a.page < a.table_width) {
      const int physical = a.table[req * a.ts0 + logical / a.page];
      if (physical >= 0 && physical < a.blocks)
        slot = int64_t(physical) * a.page + logical % a.page;
    }
    slots[t] = slot;
  }
  const int head = t / 32, d8 = (t % 32) * 8;
  const uint4 qv = head < H ? __ldg(reinterpret_cast<const uint4*>(
                                  a.q + row * a.qs0 + head * a.qs1 + d8))
                            : make_uint4(0, 0, 0, 0);
  *reinterpret_cast<uint4*>(&query[head][d8]) = qv;
  __syncthreads();
  for (int v = t; v < C * 32; v += 256) {
    const int token = v / 32, dim = (v % 32) * 8;
    const int64_t slot = slots[token];
    uint4 value = make_uint4(0, 0, 0, 0);
    if (slot >= 0) {
      const int block = slot / a.page, off = slot % a.page;
      const unsigned char* src =
          a.history + (int64_t(block) * 2 * a.page + off) * D + dim;
      if constexpr (FP8)
        value = decode8(src, __ldg(a.scales + slot * 2));
      else
        value = __ldg(reinterpret_cast<const uint4*>(
            reinterpret_cast<const half*>(a.history) +
            (int64_t(block) * 2 * a.page + off) * D + dim));
    }
    *reinterpret_cast<uint4*>(&kv[token][dim]) = value;
  }
  __syncthreads();
  const int nt = warp & 1, kw = warp >> 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  float acc[8] = {};
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int k = kw * 64 + s * 16;
    const uint4 q0 = *reinterpret_cast<const uint4*>(&query[r8][k]);
    const uint4 q1 = *reinterpret_cast<const uint4*>(&query[r8][k + 8]);
    const uint4 k0 = *reinterpret_cast<const uint4*>(&kv[nr][k]);
    const uint4 k1 = *reinterpret_cast<const uint4*>(&kv[nr][k + 8]);
    mma(acc, q0.x, q0.y, k0.x, k0.y);
    mma(acc, q0.z, q0.w, k0.z, k0.w);
    mma(acc, q1.x, q1.y, k1.x, k1.y);
    mma(acc, q1.z, q1.w, k1.z, k1.w);
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = nt * 32 + ((lane >> 2) & 3) * 8 +
                    ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
    scores[kw][h][col] = acc[e];
  }
  __syncthreads();
  for (int v = t; v < 8 * C; v += 256) {
    const int h = v / C, k = v % C;
    const float sum = ((scores[0][h][k] + scores[1][h][k]) + scores[2][h][k]) +
                      scores[3][h][k];
    scores[0][h][k] =
        slots[k] >= 0 ? sum * (0.0625f * 1.4426950408889634f) : -1e20f;
  }
  __syncthreads();
  float maximum = fmaxf(scores[0][warp][lane], scores[0][warp][lane + 32]);
#pragma unroll
  for (int s = 16; s; s >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
  const float p0 =
      slots[lane] >= 0 ? fast_exp2(scores[0][warp][lane] - maximum) : 0.f;
  const float p1 = slots[lane + 32] >= 0
                       ? fast_exp2(scores[0][warp][lane + 32] - maximum)
                       : 0.f;
  scores[0][warp][lane] = p0;
  scores[0][warp][lane + 32] = p1;
  float sum = p0 + p1;
#pragma unroll
  for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
  if (lane == 0 && warp < H) {
    a.state[((row * H + warp) * a.splits + split) * 2] = maximum;
    a.state[((row * H + warp) * a.splits + split) * 2 + 1] = sum;
  }
  __syncthreads();
  // The shared key tile is retired. Reuse its storage for values, keeping
  // FP32 probabilities and PV FMAs instead of rounding probabilities to half.
  for (int v = t; v < C * 32; v += 256) {
    const int token = v / 32, dim = (v % 32) * 8;
    const int64_t slot = slots[token];
    uint4 value = make_uint4(0, 0, 0, 0);
    if (slot >= 0) {
      const int block = slot / a.page, off = slot % a.page;
      const unsigned char* src =
          a.history + ((int64_t(block) * 2 + 1) * a.page + off) * D + dim;
      if constexpr (FP8)
        value = decode8(src, __ldg(a.scales + slot * 2 + 1));
      else
        value = __ldg(reinterpret_cast<const uint4*>(
            reinterpret_cast<const half*>(a.history) +
            ((int64_t(block) * 2 + 1) * a.page + off) * D + dim));
    }
    *reinterpret_cast<uint4*>(&kv[token][dim]) = value;
  }
  __syncthreads();
  if (warp < H) {
    float result[8] = {};
#pragma unroll 1
    for (int k = 0; k < C; ++k) {
      const float probability = scores[0][warp][k];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        result[j] = __fmaf_rn(probability, __half2float(kv[k][lane + j * 32]),
                              result[j]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
      a.partial[((row * H + warp) * a.splits + split) * D + lane + j * 32] =
          result[j];
  }
}

__global__ void merge(Args a) {
  const int row = blockIdx.x / H, head = blockIdx.x % H, t = threadIdx.x;
  __shared__ float weights[64], denom_parts[64], global_maximum, denominator;
  const int64_t state_base = int64_t(row * H + head) * a.splits * 2;
  if (t < 32) {
    const int s0 = t, s1 = t + 32;
    const float m0 = s0 < a.splits && a.state[state_base + s0 * 2 + 1] > 0
                         ? a.state[state_base + s0 * 2]
                         : -1e20f;
    const float m1 = s1 < a.splits && a.state[state_base + s1 * 2 + 1] > 0
                         ? a.state[state_base + s1 * 2]
                         : -1e20f;
    float maximum = fmaxf(m0, m1);
#pragma unroll
    for (int s = 16; s; s >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
    if (!t) global_maximum = maximum;
  }
  __syncthreads();
  if (t < 64) {
    const bool valid = t < a.splits && a.state[state_base + t * 2 + 1] > 0;
    const float weight =
        valid ? fast_exp2(a.state[state_base + t * 2] - global_maximum) : 0.f;
    weights[t] = weight;
    denom_parts[t] = valid ? weight * a.state[state_base + t * 2 + 1] : 0.f;
  }
  __syncthreads();
  if (t < 32) {
    float sum = denom_parts[t] + denom_parts[t + 32];
#pragma unroll
    for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    if (!t) denominator = sum;
  }
  __syncthreads();
  float numerator = 0.f;
  for (int s = 0; s < a.splits; ++s)
    numerator = __fmaf_rn(weights[s],
                          a.partial[((row * H + head) * a.splits + s) * D + t],
                          numerator);
  float result = denominator > 0 ? __fdiv_rn(numerator, denominator) : 0.f;
  if (a.gate) {
    result = __half2float(__float2half_rn(result));
    const float gate = __half2float(a.gate[row * a.gs0 + head * a.gs1 + t]);
    result *= __fdiv_rn(1.f, 1.f + fast_exp2(-gate * 1.4426950408889634f));
  }
  a.out[row * a.os0 + head * a.os1 + t] = __float2half_rn(result);
}
}  // namespace device_qsa

void device_qsa_out(torch::Tensor q, torch::Tensor history,
                    torch::Tensor scales, torch::Tensor indices,
                    torch::Tensor table, torch::Tensor requests,
                    torch::Tensor positions, torch::Tensor lengths,
                    torch::Tensor out, std::optional<torch::Tensor> gate,
                    torch::Tensor partial, torch::Tensor state) {
  const c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 &&
              q.size(0) > 0 && q.size(0) <= 20 && q.size(1) == 6 &&
              q.size(2) == 256 && q.stride(2) == 1);
  TORCH_CHECK((history.scalar_type() == at::kByte ||
               history.scalar_type() == at::kHalf) &&
              history.dim() == 5 && history.size(1) == 2 &&
              history.size(3) == 1 && history.size(4) == 256 &&
              history.is_contiguous());
  TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.is_contiguous() &&
              scales.numel() == (history.scalar_type() == at::kByte
                                     ? history.size(0) * history.size(2) * 2
                                     : 2));
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.dim() == 2 &&
              indices.size(0) == q.size(0) && indices.stride(1) == 1 &&
              indices.size(1) > 0 && indices.size(1) <= 4096);
  TORCH_CHECK(table.scalar_type() == at::kInt && table.dim() == 2 &&
              table.stride(1) == 1 && requests.scalar_type() == at::kInt &&
              positions.scalar_type() == at::kLong &&
              lengths.scalar_type() == at::kInt);
  TORCH_CHECK(requests.numel() == q.size(0) && positions.numel() == q.size(0) &&
              lengths.numel() == table.size(0) && requests.is_contiguous() &&
              positions.is_contiguous() && lengths.is_contiguous());
  TORCH_CHECK(
      out.sizes() == q.sizes() && out.scalar_type() == at::kHalf &&
      out.stride(2) == 1 &&
      (!gate || (gate->sizes() == q.sizes() &&
                 gate->scalar_type() == at::kHalf && gate->stride(2) == 1)));
  for (const auto& tensor : {history, scales, indices, table, requests,
                             positions, lengths, out, partial, state})
    TORCH_CHECK(tensor.device() == q.device());
  if (gate) TORCH_CHECK(gate->device() == q.device());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
              q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0);
  auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 0);
  device_qsa::Args a{};
  a.q = reinterpret_cast<const half*>(q.data_ptr());
  a.history = reinterpret_cast<const unsigned char*>(history.data_ptr());
  a.scales = scales.data_ptr<float>();
  a.indices = indices.data_ptr<int>();
  a.table = table.data_ptr<int>();
  a.requests = requests.data_ptr<int>();
  a.positions = positions.data_ptr<int64_t>();
  a.lengths = lengths.data_ptr<int>();
  a.gate = gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr;
  a.out = reinterpret_cast<half*>(out.data_ptr());
  a.partial = partial.data_ptr<float>();
  a.state = state.data_ptr<float>();
  a.M = q.size(0);
  a.blocks = history.size(0);
  a.page = history.size(2);
  a.width = indices.size(1);
  a.table_width = table.size(1);
  a.num_requests = table.size(0);
  a.splits = (a.width + 63) / 64;
  TORCH_CHECK(partial.scalar_type() == at::kFloat &&
              state.scalar_type() == at::kFloat && partial.is_contiguous() &&
              state.is_contiguous() &&
              partial.numel() >= a.M * 6 * a.splits * 256 &&
              state.numel() >= a.M * 6 * a.splits * 2);
  a.qs0 = q.stride(0);
  a.qs1 = q.stride(1);
  a.is0 = indices.stride(0);
  a.ts0 = table.stride(0);
  a.os0 = out.stride(0);
  a.os1 = out.stride(1);
  a.gs0 = gate ? gate->stride(0) : 0;
  a.gs1 = gate ? gate->stride(1) : 0;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (history.scalar_type() == at::kByte)
    device_qsa::partial<true><<<dim3(a.splits, a.M), 256, 0, stream>>>(a);
  else
    device_qsa::partial<false><<<dim3(a.splits, a.M), 256, 0, stream>>>(a);
  device_qsa::merge<<<a.M * 6, 256, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY(vllm_sm70_qsa_device, m) {
  m.def(
      "run(Tensor q, Tensor history, Tensor scales, Tensor indices, Tensor "
      "table, Tensor requests, Tensor positions, Tensor lengths, Tensor(a!) "
      "out, Tensor? gate, Tensor(b!) partial, Tensor(c!) state) -> ()");
  m.impl("run", torch::kCUDA, &device_qsa_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
