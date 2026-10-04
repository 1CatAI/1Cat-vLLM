// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Mixed-CTA projection follows fp8_qpn8_sm70.cu (v100-skinny, MIT).
// See LICENSE.v100-skinny. Decode formulas come from gguf_lattice_compact.cuh.
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_lattice_compact.cuh"

#define MMA(C, A0, A1, B0, B1)                                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))
namespace {
// Fixed fields in aligned uint4 records. Crossing fields use one PRMT
// byte window, with no dynamic funnelshift or address recomputation.
template <int Bit, int Words>
__device__ __forceinline__ uint32_t
signed_index(const uint32_t (&words)[Words]) {
  constexpr int word = Bit / 32, offset = Bit % 32;
  if constexpr (offset + 13 <= 32)
    return (words[word] >> offset) & 8191;
  else {
    constexpr int byte = offset / 8;
    constexpr int selector =
        (byte + 3) * 4096 + (byte + 2) * 256 + (byte + 1) * 16 + byte;
    uint32_t window;
    asm("prmt.b32 %0,%1,%2,%3;"
        : "=r"(window)
        : "r"(words[word]), "r"(words[word + 1]), "n"(selector));
    return (window >> (offset % 8)) & 8191;
  }
}

template <int Split, bool Fused, bool Aligned>
__global__ void gguf_gdn_input_kernel(half* qkv, half* z, half* b, half* a,
                                      const half* input, const uint8_t* codes,
                                      const half* ba) {
  constexpr int K = 5120, N = 4096, Q = 2560, Z = 1536;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tile = blockIdx.x;
  using D = vllm::sm70_gguf::LatticeCompactDecoder<21>;
  union alignas(16) Storage {
    uint8_t grid[D::kSignedCodebookBytes];
    float partial[Split][256];
  };
  __shared__ Storage storage;
  auto& partial = storage.partial;
  if constexpr (Fused) {
    if (tile >= (Aligned ? N / 64 : N / 32)) {
      // The same two-row FP32 scalar reduction used by the qualified FP8 route.
      const int token = (tile - (Aligned ? N / 64 : N / 32)) / 12,
                group = threadIdx.x / 256;
      const int row = ((tile - (Aligned ? N / 64 : N / 32)) % 12) * 2 + group,
                t = threadIdx.x % 256;
      float value = 0;
      const half2* x = reinterpret_cast<const half2*>(input + token * K);
      const half2* w = reinterpret_cast<const half2*>(ba + row * K);
      for (int pair = t; pair < K / 2; pair += 256) {
        const float2 xv = __half22float2(__ldg(x + pair));
        const float2 wv = __half22float2(__ldg(w + pair));
        value = fmaf(xv.x, wv.x, value);
        value = fmaf(xv.y, wv.y, value);
      }
      for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
      if (lane == 0) partial[group][(t >> 5)] = value;
      __syncthreads();
      if ((t >> 5) == 0) {
        value = lane < 8 ? partial[group][lane] : 0;
        for (int offset = 16; offset; offset >>= 1)
          value += __shfl_down_sync(0xffffffff, value, offset);
        if (lane == 0) {
          if (row < 12)
            b[token * 12 + row] = __float2half_rn(value);
          else
            a[token * 12 + row - 12] = __float2half_rn(value);
        }
      }
      return;
    }
  }
  uint8_t* grid = storage.grid;
  D::initialize<true>(grid);
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0), quad = (lane >> 2) & 3,
            col = quad * 8 + r;
  const unsigned part_begin = warp * 5, part_end = part_begin + 5;
  const uint8_t* macro = codes + tile * (K / 256) * 32 * 110;
  typename D::Parameters parameters{};
  float accum[8] = {};
  const unsigned blocks_k = static_cast<unsigned>(K) >> 8;
  const uint8_t* payload = macro + part_begin * 512 + col * 16;
  const uint2* tail_ptr = reinterpret_cast<const uint2*>(
      macro + blocks_k * 2048 + part_begin * 256 + col * 8);
  const uint16_t* short_ptr = reinterpret_cast<const uint16_t*>(
      macro + blocks_k * 3072 + part_begin * 64 + col * 2);
  const uint8_t* metadata = macro + blocks_k * 3328 + (part_begin / 4) * 192;
  const half* activation = input + static_cast<size_t>(r) * K + part_begin * 64;
  unsigned in_block = part_begin & 3;
  for (unsigned part = part_begin; part < part_end; ++part) {
    if (part == part_begin || in_block == 0) {
      const half d = reinterpret_cast<const half*>(metadata)[col];
      parameters.base = __halves2half2(d, d);
      parameters.scales =
          *reinterpret_cast<const uint32_t*>(metadata + 64 + col * 4);
    }
    const uint4 indices = *reinterpret_cast<const uint4*>(payload);
    const uint2 tail = *tail_ptr;
    const uint32_t words[7] = {indices.x, indices.y, indices.z, indices.w,
                               tail.x,    tail.y,    *short_ptr};
    {
      const int nibble = (parameters.scales >> (8 * in_block + 0)) & 15;
      const auto b0 = D::fragment_signed<half, true>(
          parameters, nibble, signed_index<0>(words), signed_index<13>(words),
          grid);
      const auto b1 = D::fragment_signed<half, true>(
          parameters, nibble, signed_index<26>(words), signed_index<39>(words),
          grid);
      const uint4 a0 = *reinterpret_cast<const uint4*>(activation + 0);
      const uint4 a1 = *reinterpret_cast<const uint4*>(activation + 8);
      const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
      const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
      MMA(accum, a0.x, a0.y, b[0], b[1]);
      MMA(accum, a0.z, a0.w, b[2], b[3]);
      MMA(accum, a1.x, a1.y, c[0], c[1]);
      MMA(accum, a1.z, a1.w, c[2], c[3]);
    }
    {
      const int nibble = (parameters.scales >> (8 * in_block + 0)) & 15;
      const auto b0 = D::fragment_signed<half, true>(
          parameters, nibble, signed_index<52>(words), signed_index<65>(words),
          grid);
      const auto b1 = D::fragment_signed<half, true>(
          parameters, nibble, signed_index<78>(words), signed_index<91>(words),
          grid);
      const uint4 a0 = *reinterpret_cast<const uint4*>(activation + 16);
      const uint4 a1 = *reinterpret_cast<const uint4*>(activation + 24);
      const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
      const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
      MMA(accum, a0.x, a0.y, b[0], b[1]);
      MMA(accum, a0.z, a0.w, b[2], b[3]);
      MMA(accum, a1.x, a1.y, c[0], c[1]);
      MMA(accum, a1.z, a1.w, c[2], c[3]);
    }
    {
      const int nibble = (parameters.scales >> (8 * in_block + 4)) & 15;
      const auto b0 = D::fragment_signed<half, true>(
          parameters, nibble, signed_index<104>(words),
          signed_index<117>(words), grid);
      const auto b1 = D::fragment_signed<half, true>(
          parameters, nibble, signed_index<130>(words),
          signed_index<143>(words), grid);
      const uint4 a0 = *reinterpret_cast<const uint4*>(activation + 32);
      const uint4 a1 = *reinterpret_cast<const uint4*>(activation + 40);
      const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
      const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
      MMA(accum, a0.x, a0.y, b[0], b[1]);
      MMA(accum, a0.z, a0.w, b[2], b[3]);
      MMA(accum, a1.x, a1.y, c[0], c[1]);
      MMA(accum, a1.z, a1.w, c[2], c[3]);
    }
    {
      const int nibble = (parameters.scales >> (8 * in_block + 4)) & 15;
      const auto b0 = D::fragment_signed<half, true>(
          parameters, nibble, signed_index<156>(words),
          signed_index<169>(words), grid);
      const auto b1 = D::fragment_signed<half, true>(
          parameters, nibble, signed_index<182>(words),
          signed_index<195>(words), grid);
      const uint4 a0 = *reinterpret_cast<const uint4*>(activation + 48);
      const uint4 a1 = *reinterpret_cast<const uint4*>(activation + 56);
      const unsigned* b = reinterpret_cast<const unsigned*>(&b0);
      const unsigned* c = reinterpret_cast<const unsigned*>(&b1);
      MMA(accum, a0.x, a0.y, b[0], b[1]);
      MMA(accum, a0.z, a0.w, b[2], b[3]);
      MMA(accum, a1.x, a1.y, c[0], c[1]);
      MMA(accum, a1.z, a1.w, c[2], c[3]);
    }
    payload += 512;
    tail_ptr += 32;
    short_ptr += 32;
    activation += 64;
    metadata += (in_block == 3) * 192;
    in_block = (in_block + 1) & 3;
  }
  __syncthreads();  // All table reads finish before union storage becomes
                    // partials.
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int cc =
        quad * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
    partial[warp][rr * 32 + cc] = accum[i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < (Aligned ? 512 : 256); e += blockDim.x) {
    float value = 0;
#pragma unroll
    for (int s = 0; s < (Aligned ? 8 : Split); ++s)
      value += partial[(Aligned ? (e / 256) * 8 : 0) + s][e % 256];
    const int row = (e % 256) / 32,
              col = (Aligned ? tile * 64 + (e / 256) * 32 : tile * 32) + e % 32;
    if (col < Q)
      qkv[row * Q + col] = __float2half_rn(value);
    else
      z[row * Z + col - Q] = __float2half_rn(value);
  }
}
void run(torch::Tensor qkv, torch::Tensor z, torch::Tensor b, torch::Tensor a,
         torch::Tensor x, torch::Tensor codes, torch::Tensor ba) {
  TORCH_CHECK(x.is_cuda() && x.sizes() == at::IntArrayRef({8, 5120}) &&
              x.scalar_type() == at::kHalf);
  const c10::cuda::CUDAGuard guard(x.device());
  for (const auto& t : {qkv, z, b, a, x, ba})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() &&
                t.scalar_type() == at::kHalf);
  TORCH_CHECK(qkv.sizes() == at::IntArrayRef({8, 2560}) &&
              z.sizes() == at::IntArrayRef({8, 1536}) &&
              b.sizes() == at::IntArrayRef({8, 12}) && a.sizes() == b.sizes());
  TORCH_CHECK(codes.device() == x.device() &&
              codes.scalar_type() == at::kByte && codes.is_contiguous() &&
              codes.numel() == 4096 * 20 * 110 &&
              ba.sizes() == at::IntArrayRef({24, 5120}));
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(gguf_gdn_input_kernel<16, true, false>,
                           cudaFuncAttributePreferredSharedMemoryCarveout, 66));
  gguf_gdn_input_kernel<16, true, false><<<224, 512, 0, stream>>>(
      (half*)qkv.data_ptr(), (half*)z.data_ptr(), (half*)b.data_ptr(),
      (half*)a.data_ptr(), (const half*)x.data_ptr(), codes.data_ptr<uint8_t>(),
      (const half*)ba.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
#undef MMA
#ifdef GGUF_GDN_INPUT_SIGNED_INDEX64_RESEARCH
TORCH_LIBRARY(gguf_gdn_input_signed_index64_research, m) {
  m.def(
      "run(Tensor(a!) qkv, Tensor(b!) z, Tensor(c!) b, Tensor(d!) a, Tensor x, "
      "Tensor codes, Tensor ba) -> ()");
}
TORCH_LIBRARY_IMPL(gguf_gdn_input_signed_index64_research, CUDA, m) {
  m.impl("run", &run);
}
#endif
