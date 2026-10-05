// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_iq3_nibble_book.cuh"
#include "src/turbomind/kernels/gemm/arch/mma_sm70.h"

#define VLLM_SM70_MMA_8N8K4(C, A0, A1, B0, B1)                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))
namespace {
// Reader only: original 26-bit packets stay tightly packed, without warp
// shuffles.
template <int CacheHint>
__device__ uint2 direct_packet(const uint8_t* tile, int width, int octet,
                               int col) {
  const uint32_t* words = reinterpret_cast<const uint32_t*>(tile) + octet * 26;
  const int bit = col * 26;
  const uint32_t low =
      CacheHint == 1 ? __ldcs(words + (bit >> 5)) : __ldcg(words + (bit >> 5));
  const uint32_t high = CacheHint == 1 ? __ldcs(words + (bit >> 5) + 1)
                                       : __ldcg(words + (bit >> 5) + 1);
  return make_uint2(low, high);
}

// Fixed fields in aligned uint4 records. Crossing fields use one PRMT
// byte window, with no dynamic funnelshift or address recomputation.

template <int SplitK, bool TwoChains, bool FullActivation = false,
          int CacheHint = 1>
__global__ __launch_bounds__(512, 2) void iq3_pair_kernel(
    half* __restrict__ output, const half* __restrict__ input,
    const uint8_t* __restrict__ gate, const uint8_t* __restrict__ up,
    int hidden, int k, int m, float* scratch, int* counters, int partitions) {
  constexpr int NAcc = TwoChains ? 2 : 1, RowTiles = 1;
  constexpr bool M1Only = false;
  using D = vllm::sm70_gguf::Iq3NibbleBookDecoder;
  __shared__ __align__(16) uint8_t grid[D::kSignedCodebookBytes];
  union alignas(16) Reduction {
    float partials[2][8][256];
    int last[2];
  };
  __shared__ Reduction storage;
  __shared__ half staged_a[4][8][136];
  D::initialize<true>(grid);
  auto& partials = storage.partials;
  for (int task = blockIdx.x; task < hidden / 32; task += gridDim.x) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int projection = warp_in_block / 8;
    const int tile_in_pair = (warp_in_block % 8) / 4;
    const int tile_n32 = (task / 2) * 2 + tile_in_pair;
    const int k_partition = task & 1;
    const int warp = warp_in_block % 4;
    const int hidden_tiles = hidden >> 5;
    const int tiles_n32 = hidden_tiles * 2;
    const int tile = blockIdx.x + projection * hidden_tiles;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int groups_per_warp = groups_k16 / SplitK;
    const int group_begin =
        k_partition * (groups_k16 / 2) + warp * groups_per_warp;
    const int col = quadpair * 8 + row;
    const uint8_t* source = projection ? up : gate;
    const uint8_t* macro = source + int64_t{tile_n32} * 32 * (k / 256) * 110;
    typename D::Parameters parameters{};
    float accum[RowTiles][NAcc][8] = {};
    const int blocks_k = k / 256;
    const int parts_per_warp = groups_per_warp / 8;
    const int part_begin = group_begin / 8;
    const uint8_t* payload = macro + part_begin * 1536 + col * 16;
    const uint32_t* high_ptr = reinterpret_cast<const uint32_t*>(
        macro + blocks_k * 3072 + part_begin * 128 + col * 4);
    const uint8_t* metadata = macro + blocks_k * 3328 + (part_begin / 2) * 192;
    const half* activation =
        input + static_cast<size_t>(row) * k + part_begin * 128;
    int half_block = part_begin & 1;
    for (int part = 0; part < parts_per_warp; ++part) {
      // One coalesced activation load per CTA thread, shared by two N32 tiles
      // and by gate/up. Padding by eight half elements retains 16B alignment.
      const int input_k_warp = threadIdx.x / 128;
      const int input_row = (threadIdx.x / 16) & 7;
      const int input_vector = threadIdx.x & 15;
      const int input_k = k_partition * (k / 2) +
                          input_k_warp * groups_per_warp * 16 + part * 128 +
                          input_vector * 8;
      const uint4 value = *reinterpret_cast<const uint4*>(
          input + static_cast<size_t>(input_row) * k + input_k);
      *reinterpret_cast<uint4*>(
          &staged_a[input_k_warp][input_row][input_vector * 8]) = value;
      __syncthreads();
      const half* current_a = &staged_a[warp][row][0];
      if (part == 0 || half_block == 0) {
        const half d = reinterpret_cast<const half*>(metadata)[col];
        parameters.base = __halves2half2(d, d);
        parameters.scales =
            *reinterpret_cast<const uint32_t*>(metadata + 64 + col * 4);
      }
      const uint4 indices0 = *reinterpret_cast<const uint4*>(payload);
      const uint4 indices1 = *reinterpret_cast<const uint4*>(payload + 512);
      const uint4 sign_words = *reinterpret_cast<const uint4*>(payload + 1024);
      const uint32_t tail = *high_ptr;
      const uint32_t words[13] = {
          indices0.x,   indices0.y,   indices0.z, indices0.w,   indices1.x,
          indices1.y,   indices1.z,   indices1.w, sign_words.x, sign_words.y,
          sign_words.z, sign_words.w, tail};
      {
        const int nibble = (parameters.scales >> (16 * half_block + 0)) & 15;
        const auto b0 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<0>(words),
            D::signed_index<13>(words), grid);
        const auto b1 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<26>(words),
            D::signed_index<39>(words), grid);
        const uint4 a0 = *reinterpret_cast<const uint4*>(current_a + 0);
        const uint4 a1 = *reinterpret_cast<const uint4*>(current_a + 8);
        const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
        const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
        VLLM_SM70_MMA_8N8K4(accum[0][0], a0.x, a0.y, b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][1 % NAcc], a0.z, a0.w, b[2], b[3]);
        VLLM_SM70_MMA_8N8K4(accum[0][2 % NAcc], a1.x, a1.y, c[0], c[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][3 % NAcc], a1.z, a1.w, c[2], c[3]);
      }
      {
        const int nibble = (parameters.scales >> (16 * half_block + 0)) & 15;
        const auto b0 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<52>(words),
            D::signed_index<65>(words), grid);
        const auto b1 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<78>(words),
            D::signed_index<91>(words), grid);
        const uint4 a0 = *reinterpret_cast<const uint4*>(current_a + 16);
        const uint4 a1 = *reinterpret_cast<const uint4*>(current_a + 24);
        const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
        const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
        VLLM_SM70_MMA_8N8K4(accum[0][0], a0.x, a0.y, b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][1 % NAcc], a0.z, a0.w, b[2], b[3]);
        VLLM_SM70_MMA_8N8K4(accum[0][2 % NAcc], a1.x, a1.y, c[0], c[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][3 % NAcc], a1.z, a1.w, c[2], c[3]);
      }
      {
        const int nibble = (parameters.scales >> (16 * half_block + 4)) & 15;
        const auto b0 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<104>(words),
            D::signed_index<117>(words), grid);
        const auto b1 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<130>(words),
            D::signed_index<143>(words), grid);
        const uint4 a0 = *reinterpret_cast<const uint4*>(current_a + 32);
        const uint4 a1 = *reinterpret_cast<const uint4*>(current_a + 40);
        const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
        const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
        VLLM_SM70_MMA_8N8K4(accum[0][0], a0.x, a0.y, b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][1 % NAcc], a0.z, a0.w, b[2], b[3]);
        VLLM_SM70_MMA_8N8K4(accum[0][2 % NAcc], a1.x, a1.y, c[0], c[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][3 % NAcc], a1.z, a1.w, c[2], c[3]);
      }
      {
        const int nibble = (parameters.scales >> (16 * half_block + 4)) & 15;
        const auto b0 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<156>(words),
            D::signed_index<169>(words), grid);
        const auto b1 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<182>(words),
            D::signed_index<195>(words), grid);
        const uint4 a0 = *reinterpret_cast<const uint4*>(current_a + 48);
        const uint4 a1 = *reinterpret_cast<const uint4*>(current_a + 56);
        const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
        const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
        VLLM_SM70_MMA_8N8K4(accum[0][0], a0.x, a0.y, b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][1 % NAcc], a0.z, a0.w, b[2], b[3]);
        VLLM_SM70_MMA_8N8K4(accum[0][2 % NAcc], a1.x, a1.y, c[0], c[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][3 % NAcc], a1.z, a1.w, c[2], c[3]);
      }
      {
        const int nibble = (parameters.scales >> (16 * half_block + 8)) & 15;
        const auto b0 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<208>(words),
            D::signed_index<221>(words), grid);
        const auto b1 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<234>(words),
            D::signed_index<247>(words), grid);
        const uint4 a0 = *reinterpret_cast<const uint4*>(current_a + 64);
        const uint4 a1 = *reinterpret_cast<const uint4*>(current_a + 72);
        const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
        const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
        VLLM_SM70_MMA_8N8K4(accum[0][0], a0.x, a0.y, b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][1 % NAcc], a0.z, a0.w, b[2], b[3]);
        VLLM_SM70_MMA_8N8K4(accum[0][2 % NAcc], a1.x, a1.y, c[0], c[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][3 % NAcc], a1.z, a1.w, c[2], c[3]);
      }
      {
        const int nibble = (parameters.scales >> (16 * half_block + 8)) & 15;
        const auto b0 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<260>(words),
            D::signed_index<273>(words), grid);
        const auto b1 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<286>(words),
            D::signed_index<299>(words), grid);
        const uint4 a0 = *reinterpret_cast<const uint4*>(current_a + 80);
        const uint4 a1 = *reinterpret_cast<const uint4*>(current_a + 88);
        const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
        const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
        VLLM_SM70_MMA_8N8K4(accum[0][0], a0.x, a0.y, b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][1 % NAcc], a0.z, a0.w, b[2], b[3]);
        VLLM_SM70_MMA_8N8K4(accum[0][2 % NAcc], a1.x, a1.y, c[0], c[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][3 % NAcc], a1.z, a1.w, c[2], c[3]);
      }
      {
        const int nibble = (parameters.scales >> (16 * half_block + 12)) & 15;
        const auto b0 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<312>(words),
            D::signed_index<325>(words), grid);
        const auto b1 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<338>(words),
            D::signed_index<351>(words), grid);
        const uint4 a0 = *reinterpret_cast<const uint4*>(current_a + 96);
        const uint4 a1 = *reinterpret_cast<const uint4*>(current_a + 104);
        const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
        const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
        VLLM_SM70_MMA_8N8K4(accum[0][0], a0.x, a0.y, b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][1 % NAcc], a0.z, a0.w, b[2], b[3]);
        VLLM_SM70_MMA_8N8K4(accum[0][2 % NAcc], a1.x, a1.y, c[0], c[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][3 % NAcc], a1.z, a1.w, c[2], c[3]);
      }
      {
        const int nibble = (parameters.scales >> (16 * half_block + 12)) & 15;
        const auto b0 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<364>(words),
            D::signed_index<377>(words), grid);
        const auto b1 = D::fragment_signed<half, true>(
            parameters, nibble, D::signed_index<390>(words),
            D::signed_index<403>(words), grid);
        const uint4 a0 = *reinterpret_cast<const uint4*>(current_a + 112);
        const uint4 a1 = *reinterpret_cast<const uint4*>(current_a + 120);
        const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
        const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
        VLLM_SM70_MMA_8N8K4(accum[0][0], a0.x, a0.y, b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][1 % NAcc], a0.z, a0.w, b[2], b[3]);
        VLLM_SM70_MMA_8N8K4(accum[0][2 % NAcc], a1.x, a1.y, c[0], c[1]);
        VLLM_SM70_MMA_8N8K4(accum[0][3 % NAcc], a1.z, a1.w, c[2], c[3]);
      }
      __syncthreads();
      payload += 1536;
      high_ptr += 32;
      activation += 128;
      metadata += half_block * 192;
      half_block ^= 1;
    }

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
      for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          accum[row_tile][0][i] += accum[row_tile][chain][i];
        }
      }
    }
    // The book remains live across tasks; partials become completion flags
    // only after all values are committed to the global workspace.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int output_row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partials[projection][tile_in_pair * 4 + warp]
              [output_row * 32 + quadpair * 8 + output_col] = accum[0][0][i];
    }
    __syncthreads();
    const int tile_local = threadIdx.x / 256;
    const int element = threadIdx.x % 256;
    const int tile_global = (task / 2) * 2 + tile_local;
    float sums[2] = {};
#pragma unroll
    for (int p = 0; p < 2; ++p) {
#pragma unroll
      for (int w = 0; w < 4; ++w)
        sums[p] += partials[p][tile_local * 4 + w][element];
      reinterpret_cast<volatile float*>(
          scratch)[(tile_global * 2 + k_partition) * 512 + p * 256 + element] =
          sums[p];
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x < 2)
      storage.last[threadIdx.x] =
          (atomicInc(reinterpret_cast<unsigned*>(counters) + (task / 2) * 2 +
                         threadIdx.x,
                     1) == 1);
    __syncthreads();
    if (storage.last[tile_local]) {
      const volatile float* data = scratch + tile_global * 1024;
      const float gate = data[element] + data[512 + element];
      const float up = data[256 + element] + data[768 + element];
      const float g = __half2float(__float2half(gate));
      const half silu = __float2half(g / (1.f + expf(-g)));
      output[static_cast<size_t>(element / 32) * hidden + tile_global * 32 +
             element % 32] = __hmul(silu, __float2half(up));
    }
    __syncthreads();
  }
}

}  // namespace
void shared_a_n64_gguf_iq3_pair_sm70_out(torch::Tensor out, torch::Tensor input,
                                         torch::Tensor gate, torch::Tensor up,
                                         int64_t split, bool prefetch,
                                         torch::Tensor scratch,
                                         torch::Tensor counters,
                                         int64_t partitions, bool staged) {
  TORCH_CHECK(
      out.is_cuda() && input.is_cuda() && gate.is_cuda() && up.is_cuda(),
      "IQ3 pair requires CUDA tensors");
  TORCH_CHECK(out.device() == input.device() &&
                  gate.device() == input.device() &&
                  up.device() == input.device(),
              "IQ3 pair device mismatch");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16 &&
                  gate.scalar_type() == torch::kUInt8 &&
                  up.scalar_type() == torch::kUInt8,
              "IQ3 pair dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  gate.is_contiguous() && up.is_contiguous(),
              "IQ3 pair requires contiguous tensors");
  TORCH_CHECK(
      input.dim() == 2 && out.dim() == 2 && input.size(0) == out.size(0),
      "IQ3 pair shape mismatch");
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  TORCH_CHECK(m == 8 && n > 0 && n % 32 == 0 && k > 0 && k % 256 == 0,
              "IQ3 pair requires M8, N32, K256");
  TORCH_CHECK(gate.numel() >= int64_t{n} * (k / 256) * 110 &&
                  up.numel() == gate.numel(),
              "IQ3 pair storage mismatch");
  const c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(scratch.device() == input.device() &&
                  counters.device() == input.device() &&
                  scratch.scalar_type() == torch::kFloat32 &&
                  counters.scalar_type() == torch::kInt32 &&
                  scratch.is_contiguous() && counters.is_contiguous() &&
                  scratch.numel() >= int64_t{n / 32} * 2 * 512 &&
                  counters.numel() >= n / 32,
              "IQ3 pair workspace mismatch");
  TORCH_CHECK(
      (partitions == 80 || partitions == 160) && n % 64 == 0 && k % 1024 == 0,
      "persistent pair requires grid80/160,N64,K1024");
  const auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      iq3_pair_kernel<8, false, true, 1>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  iq3_pair_kernel<8, false, true, 1><<<dim3(partitions), 512, 0, stream>>>(
      (half*)out.data_ptr(), (const half*)input.data_ptr(),
      gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), n, k, m,
      scratch.data_ptr<float>(), counters.data_ptr<int>(), 2);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef GGUF_IQ3_PAIR_RESEARCH
TORCH_LIBRARY(iq3_pair_shared_a_n64_research, ops) {
  ops.def(
      "pair(Tensor(a!) out, Tensor input, Tensor gate, Tensor up, int split, "
      "bool prefetch, Tensor(b!) scratch, Tensor(c!) counters, int partitions, "
      "bool staged) "
      "-> ()");
  ops.impl("pair", torch::kCUDA, &shared_a_n64_gguf_iq3_pair_sm70_out);
}
#endif
