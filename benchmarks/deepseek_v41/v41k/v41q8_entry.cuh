// v41q8 entry points, appended to the generated copy of the QPN8 kernels
// (csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu, MIT notice in LICENSE.v100-skinny)
// whose block-scale index is rewritten from 128 to 32 K values per scale:
// DeepSeek-V4.1 dense FP8 weights carry one UE8M0 scale per 32 x 32 block.

namespace v41q8 {

// codes: [N, K] E4M3 bytes -> [N/32][K/16][32 lanes][16] QPN8 fragment order.
void prepack(torch::Tensor packed, torch::Tensor codes) {
  TORCH_CHECK(packed.is_cuda() && codes.is_cuda() && codes.dim() == 2 &&
              codes.scalar_type() == torch::kUInt8 &&
              packed.scalar_type() == torch::kUInt8 &&
              packed.numel() == codes.numel() && codes.is_contiguous());
  const int n = codes.size(0), k = codes.size(1);
  TORCH_CHECK(n % 32 == 0 && k % 32 == 0, "v41q8 prepack needs N%32, K%32");
  const at::cuda::OptionalCUDAGuard guard(device_of(codes));
  const size_t numel = static_cast<size_t>(n) * k;
  fp8_qpn8_prepack_sm70_kernel<<<
      (numel + kQpn8PrepareThreads - 1) / kQpn8PrepareThreads,
      kQpn8PrepareThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      packed.data_ptr<uint8_t>(), codes.data_ptr<uint8_t>(), n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[M, N] = input[M, K] @ W^T, M in 1..8, scales [K/32, N/32] FP16
// holding 256 * 2^(e - 127) (the decoder's 2^-8 is folded in).
void gemm(torch::Tensor out, torch::Tensor input, torch::Tensor packed,
          torch::Tensor scales, int64_t split_k, int64_t chains) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat16 && input.is_contiguous() &&
              out.is_contiguous() && scales.scalar_type() == torch::kFloat16);
  const int m = input.size(0), k = input.size(1), n = out.size(1);
  TORCH_CHECK(m >= 1 && m <= 8 && out.size(0) == m, "v41q8: M must be 1..8");
  TORCH_CHECK(n % 32 == 0 && k % 32 == 0 && packed.numel() == (int64_t)n * k);
  TORCH_CHECK(scales.size(0) == k / 32 && scales.size(1) == n / 32,
              "v41q8: scales must be [K/32, N/32]");
  TORCH_CHECK((k / 16) % split_k == 0, "v41q8: K/16 must divide by split_k");
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* c = packed.data_ptr<uint8_t>();
  const auto* s = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(out.data_ptr<at::Half>());
#define V41Q8_CASE(S, A)                                                \
  if (split_k == S && chains == A) {                                    \
    launch_fp8_qpn8_sm70<S, A, true, false>(c, s, x, y, n, k, m, false, \
                                            stream);                    \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                     \
    return;                                                             \
  }
  // clang-format off
  V41Q8_CASE(4, 1)
  V41Q8_CASE(4, 2)
  V41Q8_CASE(8, 1)
  V41Q8_CASE(8, 2)
  V41Q8_CASE(16, 1)
  V41Q8_CASE(16, 2)
  V41Q8_CASE(32, 1)
  V41Q8_CASE(32, 2)
  // clang-format on
#undef V41Q8_CASE
  TORCH_CHECK(false, "v41q8: split_k in {4, 8, 16, 32}, chains in {1, 2}");
}

// out_a[M, Na] | out_b[M, N - Na] = input[M, K] @ W^T in one launch: the
// kernel's split-output epilogue writes each half contiguously (no copies).
void gemm2(torch::Tensor out_a, torch::Tensor out_b, torch::Tensor input,
           torch::Tensor packed, torch::Tensor scales, int64_t split_k,
           int64_t chains) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
              input.is_contiguous() && out_a.is_contiguous() &&
              out_b.is_contiguous() && out_a.scalar_type() == torch::kFloat16 &&
              out_b.scalar_type() == torch::kFloat16);
  const int m = input.size(0), k = input.size(1);
  const int na = out_a.size(1), n = na + out_b.size(1);
  TORCH_CHECK(m >= 1 && m <= 8 && out_a.size(0) == m && out_b.size(0) == m);
  TORCH_CHECK(n % 32 == 0 && k % 32 == 0 && packed.numel() == (int64_t)n * k);
  TORCH_CHECK(scales.size(0) == k / 32 && scales.size(1) == n / 32);
  TORCH_CHECK((k / 16) % split_k == 0, "v41q8: K/16 must divide by split_k");
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* c = packed.data_ptr<uint8_t>();
  const auto* s = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* ya = reinterpret_cast<half*>(out_a.data_ptr<at::Half>());
  auto* yb = reinterpret_cast<half*>(out_b.data_ptr<at::Half>());
#define V41Q8_CASE2(S, A)                                                      \
  if (split_k == S && chains == A) {                                           \
    fp8_qpn8_sm70_kernel<S, A, true, false, false, false, true, 1>             \
        <<<(n / 32), (32 * S), 0, stream>>>(c, s, x, ya, yb, nullptr, nullptr, \
                                            nullptr, nullptr, 0, na, n, k, m,  \
                                            false);                            \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                            \
    return;                                                                    \
  }
  // clang-format off
  V41Q8_CASE2(4, 2)
  V41Q8_CASE2(8, 2)
  V41Q8_CASE2(16, 2)
  V41Q8_CASE2(32, 2)
  V41Q8_CASE2(4, 1)
  V41Q8_CASE2(8, 1)
  V41Q8_CASE2(16, 1)
  V41Q8_CASE2(32, 1)
  // clang-format on
#undef V41Q8_CASE2
  TORCH_CHECK(false,
              "v41q8 gemm2: split_k in {4, 8, 16, 32}, chains in {1, 2}");
}

// ---- K split across CTAs (small N, large K) ----
// The QPN8 kernel runs one CTA per 32 output columns, so N = 512..1792 at
// K = 5120 occupies only 16..56 of 80 SMs (230..520 GB/s). Here grid.y CTAs
// share a column tile: each reduces its W warps in shared memory (warp
// order), writes one FP32 partial, and the last CTA of the tile (atomic
// ticket) sums the partials in CTA order, writes FP16 and re-arms the
// ticket for the next launch (CUDA graph safe). Deterministic.
template <int W, int NAcc>
__global__ void __launch_bounds__(W * 32)
    qpn8_ksplit_kernel(const uint8_t* __restrict__ codes,
                       const half* __restrict__ group_scales,
                       const half* __restrict__ input,
                       half* __restrict__ output, float* __restrict__ workspace,
                       int* __restrict__ tickets, int n, int k, int m,
                       int x_row_stride, int tiles_per_group) {
  __shared__ float partials[W][256];
  __shared__ bool last;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x, part = blockIdx.y, parts = gridDim.y;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / (W * parts);
  const int group_begin = (part * W + warp) * groups_per_warp;
  const int tiles_n32 = n >> 5;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half* scale_ptr = group_scales + tile;
  float accum[NAcc][8];
#pragma unroll
  for (int chain = 0; chain < NAcc; ++chain)
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[chain][i] = 0.0f;
  int loaded_scale_group = -1;
  half loaded_scale = __float2half(0.0f);
  // Grouped input (wo_a): column tile t reads K values of group t / tpg.
  const half* input_row = input + static_cast<size_t>(row) * x_row_stride +
                          static_cast<size_t>(tile / tiles_per_group) * k;
  const bool row_live = row < m;
#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const int scale_group = group >> 1;  // one UE8M0 scale per 32 K values
    if (scale_group != loaded_scale_group) {
      loaded_scale =
          __ldg(scale_ptr + static_cast<size_t>(scale_group) * tiles_n32);
      loaded_scale_group = scale_group;
    }
    const uint4 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    half2 weights[8];
    fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
    fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
    const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) weights[i] = __hmul2(weights[i], scale2);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
    uint4 input01 = make_uint4(0, 0, 0, 0), input23 = make_uint4(0, 0, 0, 0);
    if (row_live) {
      input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
      input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    VLLM_SM70_MMA_8N8K4(accum[0], a0[0], a0[1], b[0], b[1]);
    VLLM_SM70_MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
    VLLM_SM70_MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
    VLLM_SM70_MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
  }
#pragma unroll
  for (int chain = 1; chain < NAcc; ++chain)
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[0][i] += accum[chain][i];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int output_row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partials[warp][output_row * 32 + quadpair * 8 + output_col] = accum[0][i];
  }
  __syncthreads();
  const int live = m * 32;
  if (parts == 1) {
    for (int element = threadIdx.x; element < live; element += blockDim.x) {
      float value = 0.0f;
#pragma unroll
      for (int w = 0; w < W; ++w) value += partials[w][element];
      output[static_cast<size_t>(element >> 5) * n + tile * 32 +
             (element & 31)] = __float2half(value);
    }
    return;
  }
  float* mine = workspace + (static_cast<size_t>(tile) * parts + part) * 256;
  for (int element = threadIdx.x; element < live; element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int w = 0; w < W; ++w) value += partials[w][element];
    mine[element] = value;
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(tickets + tile, 1) == parts - 1;
  __syncthreads();
  if (!last) return;
  if (parts > 1) __threadfence();
  const float* all = workspace + static_cast<size_t>(tile) * parts * 256;
  for (int element = threadIdx.x; element < live; element += blockDim.x) {
    float value = 0.0f;
    for (int p = 0; p < parts; ++p) value += __ldcg(all + p * 256 + element);
    output[static_cast<size_t>(element >> 5) * n + tile * 32 + (element & 31)] =
        __float2half(value);
  }
  if (threadIdx.x == 0) tickets[tile] = 0;
}

static void launch_ksplit(torch::Tensor out, torch::Tensor input,
                          torch::Tensor packed, torch::Tensor scales,
                          torch::Tensor tickets, int64_t warps, int64_t parts,
                          int64_t chains, int m, int k, int n, int x_row_stride,
                          int tiles_per_group) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat16 && input.is_contiguous() &&
              out.is_contiguous() && scales.scalar_type() == torch::kFloat16 &&
              tickets.scalar_type() == torch::kInt32);
  TORCH_CHECK(m >= 1 && m <= 8 && out.size(0) == m, "v41q8: M must be 1..8");
  TORCH_CHECK(n % 32 == 0 && k % 32 == 0 && packed.numel() == (int64_t)n * k);
  TORCH_CHECK(scales.size(0) == k / 32 && scales.size(1) == n / 32);
  TORCH_CHECK(tickets.numel() >= n / 32, "v41q8: tickets must be [N/32]");
  TORCH_CHECK(parts >= 1 && (k / 16) % (warps * parts) == 0,
              "v41q8: K/16 must divide by warps * parts");
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto workspace = torch::empty({parts > 1 ? (n / 32) * parts * 256 : 1},
                                input.options().dtype(torch::kFloat32));
  const dim3 grid(n / 32, parts);
  const auto* c = packed.data_ptr<uint8_t>();
  const auto* s = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(out.data_ptr<at::Half>());
#define V41Q8_KS(W, A)                                                       \
  if (warps == W && chains == A) {                                           \
    qpn8_ksplit_kernel<W, A><<<grid, W * 32, 0, stream>>>(                   \
        c, s, x, y, workspace.data_ptr<float>(), tickets.data_ptr<int>(), n, \
        k, m, x_row_stride, tiles_per_group);                                \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    return;                                                                  \
  }
  // clang-format off
  V41Q8_KS(4, 1)
  V41Q8_KS(4, 2)
  V41Q8_KS(8, 1)
  V41Q8_KS(8, 2)
  V41Q8_KS(16, 1)
  V41Q8_KS(16, 2)
  V41Q8_KS(32, 1)
  V41Q8_KS(32, 2)
  // clang-format on
#undef V41Q8_KS
  TORCH_CHECK(false, "v41q8 ksplit: warps in {4, 8, 16, 32}, chains {1, 2}");
}

// gemm with grid.y = parts CTAs per column tile, W warps each; tickets is
// a zeroed int32 [N/32] owned by the weight (re-armed by every launch).
void gemm_ksplit(torch::Tensor out, torch::Tensor input, torch::Tensor packed,
                 torch::Tensor scales, torch::Tensor tickets, int64_t warps,
                 int64_t parts, int64_t chains) {
  const int k = input.size(1), n = out.size(1);
  launch_ksplit(out, input, packed, scales, tickets, warps, parts, chains,
                input.size(0), k, n, k, 1 << 30);
}

// Grouped projection in one launch: input [M, G, K], weight [G * Ng, K]
// prepacked as one matrix, out [M, G * Ng] (= the per-group results
// concatenated). Column tile t reads group t / (Ng / 32).
void gemm_grouped(torch::Tensor out, torch::Tensor input, torch::Tensor packed,
                  torch::Tensor scales, torch::Tensor tickets, int64_t warps,
                  int64_t parts, int64_t chains) {
  TORCH_CHECK(input.dim() == 3, "v41q8 gemm_grouped: input [M, G, K]");
  const int g = input.size(1), k = input.size(2), n = out.size(1);
  TORCH_CHECK(n % (32 * g) == 0,
              "v41q8 gemm_grouped: N must split into groups");
  launch_ksplit(out, input, packed, scales, tickets, warps, parts, chains,
                input.size(0), k, n, g * k, n / g / 32);
}

// Per-output-channel scales [N] FP16 holding 256 * scale (draft head).
void gemm_channel(torch::Tensor out, torch::Tensor input, torch::Tensor packed,
                  torch::Tensor scales, int64_t split_k) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat16 && input.is_contiguous() &&
              out.is_contiguous() && scales.scalar_type() == torch::kFloat16);
  const int m = input.size(0), k = input.size(1), n = out.size(1);
  TORCH_CHECK(m >= 1 && m <= 8 && out.size(0) == m, "v41q8: M must be 1..8");
  TORCH_CHECK(n % 32 == 0 && k % 16 == 0 && packed.numel() == (int64_t)n * k);
  TORCH_CHECK(scales.numel() == n, "v41q8: channel scales must be [N]");
  TORCH_CHECK((k / 16) % split_k == 0, "v41q8: K/16 must divide by split_k");
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* c = packed.data_ptr<uint8_t>();
  const auto* s = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  if (split_k == 16) {
    launch_fp8_qpn8_sm70<16, 2, true, false>(c, s, x, y, n, k, m, true, stream);
  } else if (split_k == 32) {
    launch_fp8_qpn8_sm70<32, 2, true, false>(c, s, x, y, n, k, m, true, stream);
  } else {
    TORCH_CHECK(false, "v41q8 gemm_channel: split_k in {16, 32}");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- multi-row QPN8 for batched decode (M up to 64) -------------------------
// qpn8_ksplit_kernel with RT 8-row tiles: each decoded weight fragment feeds
// the MMAs of every row tile, so the weights are read once for M <= 8 * RT
// rows (the Triton 16-row tiles re-decode them per tile). Same split-K CTAs,
// warp-order and CTA-order reductions, ticket re-arm (deterministic, graph
// safe). Shared memory W * RT * 1 KB <= 32 KB.
template <int W, int RT>
__global__ void __launch_bounds__(W * 32)
    qpn8_rows_kernel(const uint8_t* __restrict__ codes,
                     const half* __restrict__ group_scales,
                     const half* __restrict__ input, half* __restrict__ output,
                     float* __restrict__ workspace, int* __restrict__ tickets,
                     int n, int k, int m, int x_row_stride,
                     int tiles_per_group) {
  static_assert(W * RT <= 32, "qpn8 rows: W * RT * 1 KB of shared partials");
  __shared__ float partials[W][RT * 256];
  __shared__ bool last;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x, part = blockIdx.y, parts = gridDim.y;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / (W * parts);
  const int group_begin = (part * W + warp) * groups_per_warp;
  const int tiles_n32 = n >> 5;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half* scale_ptr = group_scales + tile;
  float accum[RT][8];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[r][i] = 0.0f;
  int loaded_scale_group = -1;
  half loaded_scale = __float2half(0.0f);
  const half* input_base =
      input + static_cast<size_t>(tile / tiles_per_group) * k;
#pragma unroll 2
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const int scale_group = group >> 1;  // one UE8M0 scale per 32 K values
    if (scale_group != loaded_scale_group) {
      loaded_scale =
          __ldg(scale_ptr + static_cast<size_t>(scale_group) * tiles_n32);
      loaded_scale_group = scale_group;
    }
    const uint4 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    half2 weights[8];
    fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
    fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
    const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) weights[i] = __hmul2(weights[i], scale2);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const int x_row = r * 8 + row;
      uint4 input01 = make_uint4(0, 0, 0, 0), input23 = make_uint4(0, 0, 0, 0);
      if (x_row < m) {
        const half* input_row =
            input_base + static_cast<size_t>(x_row) * x_row_stride;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      VLLM_SM70_MMA_8N8K4(accum[r], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_MMA_8N8K4(accum[r], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_MMA_8N8K4(accum[r], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_MMA_8N8K4(accum[r], a1[2], a1[3], b[6], b[7]);
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int output_row =
          r * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partials[warp][output_row * 32 + quadpair * 8 + output_col] = accum[r][i];
    }
  __syncthreads();
  const int live = m * 32;
  if (parts == 1) {
    for (int element = threadIdx.x; element < live; element += blockDim.x) {
      float value = 0.0f;
#pragma unroll
      for (int w = 0; w < W; ++w) value += partials[w][element];
      output[static_cast<size_t>(element >> 5) * n + tile * 32 +
             (element & 31)] = __float2half(value);
    }
    return;
  }
  float* mine =
      workspace + (static_cast<size_t>(tile) * parts + part) * RT * 256;
  for (int element = threadIdx.x; element < live; element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int w = 0; w < W; ++w) value += partials[w][element];
    mine[element] = value;
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(tickets + tile, 1) == parts - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const float* all = workspace + static_cast<size_t>(tile) * parts * RT * 256;
  for (int element = threadIdx.x; element < live; element += blockDim.x) {
    float value = 0.0f;
    for (int p = 0; p < parts; ++p)
      value += __ldcg(all + p * RT * 256 + element);
    output[static_cast<size_t>(element >> 5) * n + tile * 32 + (element & 31)] =
        __float2half(value);
  }
  if (threadIdx.x == 0) tickets[tile] = 0;
}

// out[M, N] = input @ W^T for M <= 64 (rows = 8 * RT); grouped like
// gemm_grouped when input is [M, G, K]. tickets: zeroed int32 [N/32].
void gemm_rows(torch::Tensor out, torch::Tensor input, torch::Tensor packed,
               torch::Tensor scales, torch::Tensor tickets, int64_t warps,
               int64_t parts, int64_t row_tiles) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat16 && input.is_contiguous() &&
              out.is_contiguous() && scales.scalar_type() == torch::kFloat16 &&
              tickets.scalar_type() == torch::kInt32);
  const bool grouped = input.dim() == 3;
  const int m = input.size(0), n = out.size(1);
  const int g = grouped ? input.size(1) : 1;
  const int k = input.size(grouped ? 2 : 1);
  TORCH_CHECK(m >= 1 && m <= 8 * row_tiles && out.size(0) == m,
              "v41q8 rows: M must be 1..8 * row_tiles");
  TORCH_CHECK(n % (32 * g) == 0 && k % 32 == 0 &&
              packed.numel() == (int64_t)n * k);
  TORCH_CHECK(scales.size(0) == k / 32 && scales.size(1) == n / 32);
  TORCH_CHECK(tickets.numel() >= n / 32, "v41q8: tickets must be [N/32]");
  TORCH_CHECK(parts >= 1 && (k / 16) % (warps * parts) == 0,
              "v41q8: K/16 must divide by warps * parts");
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto workspace =
      torch::empty({parts > 1 ? (n / 32) * parts * row_tiles * 256 : 1},
                   input.options().dtype(torch::kFloat32));
  const dim3 grid(n / 32, parts);
  const auto* c = packed.data_ptr<uint8_t>();
  const auto* s = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const int stride = grouped ? g * k : k;
  const int tpg = grouped ? n / g / 32 : 1 << 30;
#define V41Q8_ROWS(W, RT)                                                    \
  if (warps == W && row_tiles == RT) {                                       \
    qpn8_rows_kernel<W, RT><<<grid, W * 32, 0, stream>>>(                    \
        c, s, x, y, workspace.data_ptr<float>(), tickets.data_ptr<int>(), n, \
        k, m, stride, tpg);                                                  \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    return;                                                                  \
  }
  // clang-format off
  V41Q8_ROWS(4, 2)
  V41Q8_ROWS(4, 4)
  V41Q8_ROWS(4, 8)
  V41Q8_ROWS(8, 2)
  V41Q8_ROWS(8, 4)
  V41Q8_ROWS(16, 2)
  // clang-format on
#undef V41Q8_ROWS
  TORCH_CHECK(false,
              "v41q8 rows: (warps, row_tiles) in (4|8|16, 2|4|8), W*RT<=32");
}

}  // namespace v41q8

TORCH_LIBRARY(v41q8, m) {
  m.def("prepack(Tensor(a!) packed, Tensor codes) -> ()");
  m.def(
      "gemm(Tensor(a!) out, Tensor input, Tensor packed, Tensor scales, "
      "int split_k, int chains) -> ()");
  m.def(
      "gemm_channel(Tensor(a!) out, Tensor input, Tensor packed, Tensor "
      "scales, "
      "int split_k) -> ()");
  m.def(
      "gemm_ksplit(Tensor(a!) out, Tensor input, Tensor packed, Tensor scales, "
      "Tensor(b!) tickets, int warps, int parts, int chains) -> ()");
  m.def(
      "gemm2(Tensor(a!) out_a, Tensor(b!) out_b, Tensor input, Tensor packed, "
      "Tensor scales, int split_k, int chains) -> ()");
  m.def(
      "gemm_grouped(Tensor(a!) out, Tensor input, Tensor packed, Tensor "
      "scales, "
      "Tensor(b!) tickets, int warps, int parts, int chains) -> ()");
  m.def(
      "gemm_rows(Tensor(a!) out, Tensor input, Tensor packed, Tensor scales, "
      "Tensor(b!) tickets, int warps, int parts, int row_tiles) -> ()");
}

TORCH_LIBRARY_IMPL(v41q8, CUDA, m) {
  m.impl("prepack", &v41q8::prepack);
  m.impl("gemm", &v41q8::gemm);
  m.impl("gemm_channel", &v41q8::gemm_channel);
  m.impl("gemm_ksplit", &v41q8::gemm_ksplit);
  m.impl("gemm_grouped", &v41q8::gemm_grouped);
  m.impl("gemm2", &v41q8::gemm2);
  m.impl("gemm_rows", &v41q8::gemm_rows);
}
