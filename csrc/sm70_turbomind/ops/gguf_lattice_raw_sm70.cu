// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include "gguf_lattice_raw.cuh"

namespace {
template <int Type>
__device__ const uint8_t* stage_block(uint8_t* shared, const uint8_t* row,
                                      int block, int row_stride) {
  constexpr int bytes = vllm::sm70_gguf::LatticeRawDecoder<Type>::kBlockBytes;
  const int lane = threadIdx.x % 32;
  const int start = block * bytes, aligned = start & ~7, bias = start & 7;
  const int words = (bias + bytes + 7) / 8;
  if (lane < words && aligned + lane * 8 < row_stride)
    reinterpret_cast<uint64_t*>(shared)[lane] =
        *reinterpret_cast<const uint64_t*>(row + aligned + lane * 8);
  __syncwarp();
  return shared + bias;
}

template <int Type, class Output>
__global__ void raw_dequant_kernel(Output* out, const uint8_t* weight, int n,
                                   int k, int stride) {
  using Decode = vllm::sm70_gguf::LatticeRawDecoder<Type>;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  __shared__ __align__(16) uint8_t raw[4][120];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.x * 4 + warp, block = blockIdx.y;
  if (row >= n) return;
  const uint8_t* data = stage_block<Type>(
      raw[warp], weight + (int64_t)row * stride, block, stride);
  const auto values = Decode::template fragment<Output>(data, lane * 8, grid);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    out[(int64_t)row * k + block * 256 + lane * 8 + i] =
        static_cast<Output>(values[i]);
}

void validate_raw(torch::Tensor w, int type, int64_t n, int64_t k,
                  bool allow_xxs = false) {
  TORCH_CHECK(type == 21 || type == 22 || (allow_xxs && type == 18),
              "Unsupported raw GGUF lattice type");
  const int block_bytes = type == 18 ? 98 : type == 21 ? 110 : 82;
  const int64_t row_bytes = k / 256 * block_bytes;
  TORCH_CHECK(
      w.is_cuda() && w.scalar_type() == torch::kUInt8 && w.dim() == 2 &&
          w.is_contiguous() && w.size(0) == n && n > 0 && n <= INT_MAX &&
          k > 0 && k <= INT_MAX && k % 256 == 0 &&
          w.size(1) == (row_bytes + 7) / 8 * 8,
      "Raw GGUF storage must contain original rows with only 8-byte alignment");
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 0, "Raw GGUF requires SM70");
}

template <int Type>
void launch_dequant(torch::Tensor out, torch::Tensor weight,
                    cudaStream_t stream) {
  const dim3 grid((out.size(0) + 3) / 4, out.size(1) / 256);
  if (out.scalar_type() == torch::kFloat32)
    raw_dequant_kernel<Type><<<grid, 128, 0, stream>>>(
        out.data_ptr<float>(), weight.data_ptr<uint8_t>(), out.size(0),
        out.size(1), weight.size(1));
  else
    raw_dequant_kernel<Type><<<grid, 128, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()), weight.data_ptr<uint8_t>(),
        out.size(0), out.size(1), weight.size(1));
}
}  // namespace

void gguf_lattice_raw_dequantize_sm70_out(torch::Tensor out,
                                          torch::Tensor weight,
                                          int64_t source_type) {
  TORCH_CHECK(out.device() == weight.device() && out.dim() == 2 &&
                  out.is_contiguous() &&
                  (out.scalar_type() == torch::kFloat16 ||
                   out.scalar_type() == torch::kFloat32),
              "Raw GGUF dequant requires a contiguous FP16/FP32 [N,K] output");
  const c10::cuda::CUDAGuard guard(weight.device());
  validate_raw(weight, source_type, out.size(0), out.size(1), true);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (source_type == 18)
    launch_dequant<18>(out, weight, stream);
  else if (source_type == 21)
    launch_dequant<21>(out, weight, stream);
  else
    launch_dequant<22>(out, weight, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
// One CTA owns two output rows from each projection of an active expert.
// Repeated route slots return before initializing the shared codebook. Every
// original block is decoded once per token pair and output row, with no
// tensor-core M tile padding. Larger expert intervals use additional pairs.
template <int Type, int MaxTokens>
__global__ void raw_grouped_gate_up_kernel(
    half* gate, half* up, const half* input, const uint8_t* gate_weights,
    const uint8_t* up_weights, const int* offsets, const int64_t* ids, int n,
    int k, int stride) {
  const int slot = blockIdx.y;
  const int expert = ids[slot];
  if (slot && ids[slot - 1] == expert) return;
  const int begin = offsets[expert], end = offsets[expert + 1];
  using Decode = vllm::sm70_gguf::LatticeRawDecoder<Type>;
  __shared__ __align__(16) uint8_t grid[Decode::kCodebookBytes];
  __shared__ __align__(16) uint8_t raw[4][120];
  Decode::initialize(grid);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.x * 2 + warp % 2;
  if (row >= n) return;
  const uint8_t* weights = warp < 2 ? gate_weights : up_weights;
  half* output = warp < 2 ? gate : up;
  const uint8_t* row_data = weights + (int64_t{expert} * n + row) * stride;
  for (int chunk = begin; chunk < end; chunk += MaxTokens) {
    float sums[MaxTokens] = {};
    stage_block<Type>(raw[warp], row_data, 0, stride);
    for (int block = 0; block < k / 256; ++block) {
      // Use the existing raw vector's software prefetch: keep the next
      // original block in registers while decoding and multiplying this one.
      // SM70 has no cp.async. Per-token FMA and reduction order is unchanged.
      constexpr int bytes = Decode::kBlockBytes;
      const int next_start = (block + 1) * bytes;
      const int aligned = next_start & ~7, bias = next_start & 7;
      const int words = (bias + bytes + 7) / 8;
      const bool load_next =
          block + 1 < k / 256 && lane < words && aligned + lane * 8 < stride;
      uint64_t next_word = 0;
      if (load_next)
        next_word =
            *reinterpret_cast<const uint64_t*>(row_data + aligned + lane * 8);
      const auto* data = raw[warp] + (block * bytes & 7);
      const auto values = Decode::fragment(data, lane * 8, grid);
#pragma unroll
      for (int token = 0; token < MaxTokens; ++token) {
        if (chunk + token < end) {
          const auto* x =
              input + int64_t{chunk + token} * k + block * 256 + lane * 8;
          const uint4 loaded = *reinterpret_cast<const uint4*>(x);
          const auto& activation =
              reinterpret_cast<const turbomind::Array<half, 8>&>(loaded);
#pragma unroll
          for (int i = 0; i < 8; ++i)
            sums[token] =
                fmaf(__half2float(activation[i]), values[i], sums[token]);
        }
      }
      __syncwarp();
      if (load_next) reinterpret_cast<uint64_t*>(raw[warp])[lane] = next_word;
      __syncwarp();
    }
#pragma unroll
    for (int token = 0; token < MaxTokens; ++token) {
      if (chunk + token < end) {
#pragma unroll
        for (int distance = 16; distance > 0; distance /= 2)
          sums[token] += __shfl_down_sync(0xffffffffU, sums[token], distance);
        if (lane == 0)
          output[int64_t{chunk + token} * n + row] =
              __float2half_rn(sums[token]);
      }
    }
  }
}

template <int Type, int MaxTokens>
void launch_grouped_gate_up(torch::Tensor gate, torch::Tensor up,
                            torch::Tensor input, torch::Tensor gate_weights,
                            torch::Tensor up_weights, torch::Tensor offsets,
                            torch::Tensor ids, cudaStream_t stream) {
  raw_grouped_gate_up_kernel<Type, MaxTokens>
      <<<dim3((gate.size(1) + 1) / 2, input.size(0)), 128, 0, stream>>>(
          reinterpret_cast<half*>(gate.data_ptr()),
          reinterpret_cast<half*>(up.data_ptr()),
          reinterpret_cast<const half*>(input.data_ptr()),
          gate_weights.data_ptr<uint8_t>(), up_weights.data_ptr<uint8_t>(),
          offsets.data_ptr<int>(), ids.data_ptr<int64_t>(), gate.size(1),
          input.size(1), gate_weights.size(2));
}

template <int Type>
void dispatch_grouped_gate_up(torch::Tensor gate, torch::Tensor up,
                              torch::Tensor input, torch::Tensor gate_weights,
                              torch::Tensor up_weights, torch::Tensor offsets,
                              torch::Tensor ids, int tokens,
                              cudaStream_t stream) {
#define GROUPED_CASE(MAX)                                                      \
  launch_grouped_gate_up<Type, MAX>(gate, up, input, gate_weights, up_weights, \
                                    offsets, ids, stream)
  if (tokens <= 1) {
    GROUPED_CASE(1);
  } else {
    GROUPED_CASE(2);
  }
#undef GROUPED_CASE
}
}  // namespace

// IDs and offsets describe expert-sorted top-k routing with distinct experts
// per original token. Consequently an expert receives at most R / top_k
// tokens. This is the same prepared-routing contract as grouped GEMM.
void gguf_lattice_raw_grouped_gate_up_sm70_out(
    torch::Tensor gate, torch::Tensor up, torch::Tensor input,
    torch::Tensor gate_weights, torch::Tensor up_weights, torch::Tensor offsets,
    torch::Tensor ids, int64_t source_type, int64_t top_k) {
  TORCH_CHECK(source_type == 18 || source_type == 21 || source_type == 22,
              "Raw grouped gate/up supports IQ3_XXS, IQ3_S and IQ2_S");
  TORCH_CHECK(
      input.is_cuda() && input.dim() == 2 &&
          input.scalar_type() == torch::kFloat16 && input.is_contiguous(),
      "Raw grouped gate/up requires a contiguous FP16 activation matrix");
  for (const auto& tensor : {gate, up}) {
    TORCH_CHECK(tensor.device() == input.device() && tensor.dim() == 2 &&
                    tensor.scalar_type() == torch::kFloat16 &&
                    tensor.is_contiguous() && tensor.size(0) == input.size(0),
                "Raw grouped gate/up output descriptor mismatch");
  }
  TORCH_CHECK(gate.sizes() == up.sizes(), "Gate/up output shapes differ");
  for (const auto& tensor : {gate_weights, up_weights}) {
    TORCH_CHECK(tensor.device() == input.device() && tensor.dim() == 3 &&
                    tensor.scalar_type() == torch::kUInt8 &&
                    tensor.is_contiguous() && tensor.size(1) == gate.size(1),
                "Raw grouped gate/up requires prepared [E,N,bytes] weights");
  }
  TORCH_CHECK(gate_weights.sizes() == up_weights.sizes() &&
                  offsets.device() == input.device() && offsets.dim() == 1 &&
                  offsets.scalar_type() == torch::kInt32 &&
                  offsets.is_contiguous() &&
                  offsets.numel() == gate_weights.size(0) + 1 &&
                  ids.device() == input.device() && ids.dim() == 1 &&
                  ids.scalar_type() == torch::kInt64 && ids.is_contiguous() &&
                  ids.numel() == input.size(0),
              "Raw grouped gate/up route descriptor mismatch");
  const int64_t routes = input.size(0), k = input.size(1), n = gate.size(1);
  const int bytes = source_type == 18 ? 98 : source_type == 21 ? 110 : 82;
  TORCH_CHECK(top_k > 0 && top_k <= 16 && routes % top_k == 0 &&
                  routes / top_k <= 32 && n > 0 && n <= INT_MAX && k > 0 &&
                  k <= INT_MAX && k % 256 == 0 && gate_weights.size(0) > 0 &&
                  gate_weights.size(2) == (k / 256 * bytes + 7) / 8 * 8,
              "Raw grouped gate/up dimensions exceed the small-batch contract");
  if (!routes) return;
  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (source_type == 18)
    dispatch_grouped_gate_up<18>(gate, up, input, gate_weights, up_weights,
                                 offsets, ids, routes / top_k, stream);
  else if (source_type == 21)
    dispatch_grouped_gate_up<21>(gate, up, input, gate_weights, up_weights,
                                 offsets, ids, routes / top_k, stream);
  else
    dispatch_grouped_gate_up<22>(gate, up, input, gate_weights, up_weights,
                                 offsets, ids, routes / top_k, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
