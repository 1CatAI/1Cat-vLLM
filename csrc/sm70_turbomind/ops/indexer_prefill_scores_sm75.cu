// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Prefill index scores of the sparse-attention lightning indexer, fused on
// Turing's m16n8k8 tensor cores:
//
//   out[t, key] = scale[key] * sum_h w[t, h]
//                 * relu(fp16(q[t, h, :] . k[key, :]))
//
// The cuBLAS route stacks the heads into M and writes the [tokens * heads,
// keys] fp16 scores (64 MiB per 512-token chunk and 1024 keys) before a second
// kernel reads them back for the head sum. Here each warp keeps its token's
// queries for 32 heads in registers, multiplies them with a shared key tile,
// and reduces the heads in registers and through warp shuffles; only the
// [tokens, keys] result reaches memory. The keys stay e4m3 in global memory and
// are widened once per CTA while staged into shared memory. Scores are rounded
// to fp16 before the relu, as the cuBLAS route stores them.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <algorithm>

#include "mma_fragments_sm70.cuh"

namespace {

using vllm::sm70::ldmatrix_x4;
using vllm::sm70::mma_m16n8k8;
using vllm::sm70::shared_address;

constexpr int kHeadDim = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kHeadsPerWarp = 32;
constexpr int kFrags = kHeadsPerWarp / 16;
constexpr int kKeySteps = kHeadDim / 8;
constexpr int kKeyTile = 64;
constexpr int kChunks = kHeadDim / 8;  // 16-byte fp16 chunks per key row
constexpr int kStagedWords = kKeyTile * kHeadDim / 16 / kThreads;
// Keys per CTA: the queries are loaded once per CTA, so a longer key range
// amortises them; consecutive CTAs take the same range for different tokens
// and read its keys from L2.
constexpr int kKeysPerCta = 1024;
// The e4m3 bits are placed into fp16 positions without the exponent re-bias,
// which leaves every key value scaled by 2^-8; the scores take it back exactly.
constexpr float kKeyRebias = 256.0f;

// 256-byte rows: XOR the chunk with the low row bits so the eight rows of one
// ldmatrix phase land in distinct bank quads.
__device__ __forceinline__ int key_slot(int key, int chunk) {
  return key * kChunks + (chunk ^ (key & 7));
}

// Eight e4m3 values in memory order to four half2 pairs (2i, 2i + 1), each
// value times 2^-8.
__device__ __forceinline__ void e4m3x8_to_half2x4(uint2 bytes, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x3f803f80u;
  constexpr unsigned kPairs[4] = {0x0100, 0x0302, 0x0504, 0x0706};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned pair = __byte_perm(bytes.x, bytes.y, kPairs[i]);
    const unsigned value =
        ((pair << 8) & kSign) | ((pair << 7) & kExponentMantissa);
    out[i] = *reinterpret_cast<const half2*>(&value);
  }
}

__device__ __forceinline__ float weight_value(float value) { return value; }
__device__ __forceinline__ float weight_value(half value) {
  return __half2float(value);
}

template <int NumHeads, typename Weight>
__global__
__launch_bounds__(kThreads, 1) void indexer_prefill_scores_sm75_kernel(
    const half* __restrict__ q, const Weight* __restrict__ weights,
    const uint8_t* __restrict__ keys, const float* __restrict__ key_scales,
    float* __restrict__ out, int num_q, int num_k, int64_t q_stride,
    int64_t w_stride, int64_t key_stride, int64_t out_stride) {
  constexpr int kHeadWarps = NumHeads / kHeadsPerWarp;
  constexpr int kTokens = kWarps / kHeadWarps;
  static_assert(NumHeads % kHeadsPerWarp == 0 && kWarps % kHeadWarps == 0,
                "heads must fill whole warps");

  __shared__ __align__(16) uint4 key_tiles[2][kKeyTile * kChunks];
  // Head sums of the second head warp, by key tile parity: a buffer is
  // rewritten two tiles later, after the reader has passed the barrier
  // between.
  __shared__ float partial[2][kTokens][kKeyTile];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int group = lane >> 2;
  const int pair = lane & 3;
  const int token_local = warp / kHeadWarps;
  const int head_part = warp % kHeadWarps;
  const int token = blockIdx.x * kTokens + token_local;
  const bool active = token < num_q;
  const int key_begin = blockIdx.y * kKeysPerCta;
  const int key_end = min(num_k, key_begin + kKeysPerCta);
  const int tiles = (key_end - key_begin + kKeyTile - 1) / kKeyTile;

  // This warp's 32 heads of its token, the whole head dimension, as m16n8k8
  // A fragments: rows are heads, K is the head dimension.
  uint32_t a[kFrags][kKeySteps][2];
  float w[kFrags][2];
  const int head0 = head_part * kHeadsPerWarp;
#pragma unroll
  for (int f = 0; f < kFrags; ++f) {
    const int head = head0 + f * 16 + group;
#pragma unroll
    for (int s = 0; s < kKeySteps; ++s) {
      a[f][s][0] = 0;
      a[f][s][1] = 0;
      if (active) {
        const half* row =
            q + token * q_stride + head * kHeadDim + s * 8 + pair * 2;
        a[f][s][0] = *reinterpret_cast<const uint32_t*>(row);
        a[f][s][1] = *reinterpret_cast<const uint32_t*>(row + 8 * kHeadDim);
      }
    }
    w[f][0] = active ? weight_value(weights[token * w_stride + head]) : 0.0f;
    w[f][1] =
        active ? weight_value(weights[token * w_stride + head + 8]) : 0.0f;
  }

  uint4 staged[kStagedWords];
  auto load_keys = [&](int tile) {
#pragma unroll
    for (int i = 0; i < kStagedWords; ++i) {
      const int index = tid + i * kThreads;
      const int key = key_begin + tile * kKeyTile + index / (kHeadDim / 16);
      staged[i] = make_uint4(0, 0, 0, 0);
      if (key < key_end) {
        staged[i] = __ldg(reinterpret_cast<const uint4*>(
            keys + key * key_stride + (index % (kHeadDim / 16)) * 16));
      }
    }
  };
  auto store_keys = [&](int stage) {
#pragma unroll
    for (int i = 0; i < kStagedWords; ++i) {
      const int index = tid + i * kThreads;
      const int key = index / (kHeadDim / 16);
      const int chunk = (index % (kHeadDim / 16)) * 2;
      half2 values[8];
      e4m3x8_to_half2x4(make_uint2(staged[i].x, staged[i].y), values);
      e4m3x8_to_half2x4(make_uint2(staged[i].z, staged[i].w), values + 4);
      const uint4* packed = reinterpret_cast<const uint4*>(values);
      key_tiles[stage][key_slot(key, chunk)] = packed[0];
      key_tiles[stage][key_slot(key, chunk + 1)] = packed[1];
    }
  };

  load_keys(0);
  store_keys(0);
  __syncthreads();

  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile & 1;
    if (tile + 1 < tiles) {
      load_keys(tile + 1);
    }

    float acc[kFrags][kKeyTile / 8][4];
#pragma unroll
    for (int f = 0; f < kFrags; ++f) {
#pragma unroll
      for (int j = 0; j < kKeyTile / 8; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          acc[f][j][e] = 0.0f;
        }
      }
    }
#pragma unroll
    for (int kk = 0; kk < kKeySteps / 2; ++kk) {
      uint32_t b[kKeyTile / 8][2];
#pragma unroll
      for (int j = 0; j < kKeyTile / 8; j += 2) {
        const int key = j * 8 + (lane & 7) + ((lane >> 4) << 3);
        const int chunk = kk * 2 + ((lane >> 3) & 1);
        uint32_t r[4];
        ldmatrix_x4(r, shared_address(&key_tiles[stage][key_slot(key, chunk)]));
        b[j][0] = r[0];
        b[j][1] = r[1];
        b[j + 1][0] = r[2];
        b[j + 1][1] = r[3];
      }
#pragma unroll
      for (int f = 0; f < kFrags; ++f) {
#pragma unroll
        for (int j = 0; j < kKeyTile / 8; ++j) {
          mma_m16n8k8(acc[f][j], a[f][2 * kk][0], a[f][2 * kk][1], b[j][0]);
          mma_m16n8k8(acc[f][j], a[f][2 * kk + 1][0], a[f][2 * kk + 1][1],
                      b[j][1]);
        }
      }
    }

    // Lane (group, pair) holds heads 16f + group (+ 8) at keys 8j + 2 pair
    // (+ 1). Sum its four heads, then reduce over the eight groups while
    // scattering the keys, so that lane l ends with keys 2l and 2l + 1.
    float sums[kKeyTile / 8][2];
#pragma unroll
    for (int j = 0; j < kKeyTile / 8; ++j) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        float sum = 0.0f;
#pragma unroll
        for (int f = 0; f < kFrags; ++f) {
#pragma unroll
          for (int half_row = 0; half_row < 2; ++half_row) {
            const float score = __half2float(
                __float2half_rn(acc[f][j][half_row * 2 + c] * kKeyRebias));
            sum += fmaxf(score, 0.0f) * w[f][half_row];
          }
        }
        sums[j][c] = sum;
      }
    }
    const bool upper4 = (lane >> 4) & 1;
    float sums4[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const float send = upper4 ? sums[i][c] : sums[i + 4][c];
        const float keep = upper4 ? sums[i + 4][c] : sums[i][c];
        sums4[i][c] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
      }
    }
    const bool upper2 = (lane >> 3) & 1;
    float sums2[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const float send = upper2 ? sums4[i][c] : sums4[i + 2][c];
        const float keep = upper2 ? sums4[i + 2][c] : sums4[i][c];
        sums2[i][c] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
      }
    }
    const bool upper1 = (lane >> 2) & 1;
    float key_sum[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const float send = upper1 ? sums2[0][c] : sums2[1][c];
      const float keep = upper1 ? sums2[1][c] : sums2[0][c];
      key_sum[c] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
    }

    if constexpr (kHeadWarps > 1) {
      if (head_part == 1) {
        partial[stage][token_local][2 * lane] = key_sum[0];
        partial[stage][token_local][2 * lane + 1] = key_sum[1];
      }
    }
    if (tile + 1 < tiles) {
      store_keys(stage ^ 1);
    }
    __syncthreads();

    if (head_part == 0 && active) {
      if constexpr (kHeadWarps > 1) {
        key_sum[0] += partial[stage][token_local][2 * lane];
        key_sum[1] += partial[stage][token_local][2 * lane + 1];
      }
      const int key = key_begin + tile * kKeyTile + 2 * lane;
      float* row = out + token * out_stride;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        if (key + c < key_end) {
          row[key + c] = key_sum[c] * key_scales[key + c];
        }
      }
    }
  }
}

template <int NumHeads, typename Weight>
void launch(torch::Tensor& out, const torch::Tensor& q,
            const torch::Tensor& weights, const torch::Tensor& keys,
            const torch::Tensor& key_scales) {
  constexpr int kTokens = kWarps / (NumHeads / kHeadsPerWarp);
  const int num_q = static_cast<int>(q.size(0));
  const int num_k = static_cast<int>(keys.size(0));
  const dim3 grid((num_q + kTokens - 1) / kTokens,
                  (num_k + kKeysPerCta - 1) / kKeysPerCta);
  indexer_prefill_scores_sm75_kernel<NumHeads, Weight>
      <<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
          reinterpret_cast<const Weight*>(weights.data_ptr()),
          keys.data_ptr<uint8_t>(), key_scales.data_ptr<float>(),
          out.data_ptr<float>(), num_q, num_k, q.stride(0), weights.stride(0),
          keys.stride(0), out.stride(0));
}

template <int NumHeads>
void dispatch_weights(torch::Tensor& out, const torch::Tensor& q,
                      const torch::Tensor& weights, const torch::Tensor& keys,
                      const torch::Tensor& key_scales) {
  if (weights.scalar_type() == torch::kFloat32) {
    launch<NumHeads, float>(out, q, weights, keys, key_scales);
  } else {
    launch<NumHeads, half>(out, q, weights, keys, key_scales);
  }
}

}  // namespace

void indexer_prefill_scores_sm75(torch::Tensor out, torch::Tensor q,
                                 torch::Tensor weights, torch::Tensor keys,
                                 torch::Tensor key_scales) {
  TORCH_CHECK(out.is_cuda() && q.is_cuda() && weights.is_cuda() &&
                  keys.is_cuda() && key_scales.is_cuda(),
              "indexer_prefill_scores_sm75: tensors must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 && q.dim() == 3 &&
                  q.size(2) == kHeadDim && q.stride(2) == 1 &&
                  q.stride(1) == kHeadDim && q.stride(0) % 2 == 0,
              "indexer_prefill_scores_sm75: q must be float16 [tokens, heads, "
              "128] with contiguous heads and an even token stride");
  const int64_t num_q = q.size(0);
  const int64_t num_heads = q.size(1);
  TORCH_CHECK(num_heads == 32 || num_heads == 64,
              "indexer_prefill_scores_sm75: 32 or 64 heads");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == num_q &&
                  weights.size(1) == num_heads && weights.stride(1) == 1 &&
                  (weights.scalar_type() == torch::kFloat32 ||
                   weights.scalar_type() == torch::kFloat16),
              "indexer_prefill_scores_sm75: weights must be float32 or "
              "float16 [tokens, heads]");
  TORCH_CHECK(keys.scalar_type() == torch::kUInt8 && keys.dim() == 2 &&
                  keys.size(1) == kHeadDim && keys.stride(1) == 1 &&
                  keys.stride(0) % 16 == 0,
              "indexer_prefill_scores_sm75: keys must be uint8 [keys, 128] "
              "with 16-byte aligned rows");
  const int64_t num_k = keys.size(0);
  TORCH_CHECK(key_scales.scalar_type() == torch::kFloat32 &&
                  key_scales.dim() == 1 && key_scales.size(0) == num_k &&
                  key_scales.stride(0) == 1,
              "indexer_prefill_scores_sm75: key_scales must be float32 [keys]");
  TORCH_CHECK(
      out.scalar_type() == torch::kFloat32 && out.dim() == 2 &&
          out.size(0) == num_q && out.size(1) == num_k && out.stride(1) == 1,
      "indexer_prefill_scores_sm75: out must be float32 [tokens, keys]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(keys.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(q.data_ptr()) % 4 == 0,
              "indexer_prefill_scores_sm75: misaligned keys or queries");
  if (num_q == 0 || num_k == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 5,
              "indexer_prefill_scores_sm75: requires an SM75 device");
  if (num_heads == 64) {
    dispatch_weights<64>(out, q, weights, keys, key_scales);
  } else {
    dispatch_weights<32>(out, q, weights, keys, key_scales);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
