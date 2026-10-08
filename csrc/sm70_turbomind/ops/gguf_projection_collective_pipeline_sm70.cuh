// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// One output tile is published while the other projection CTAs still run.
// Peer packets and FP32 rank/norm reduction order match custom_all_reduce.
#include "../../custom_all_reduce.cuh"

namespace {
constexpr int PipeParts = vllm::kSm70PushNormParts;
constexpr int PipePacks = vllm::kSm70PushNormRows * 640;

template <int Format, typename Weight = float>
__global__ __launch_bounds__(256) void projection_collective_pipeline(
    const __grid_constant__ Segs down, const half* x,
    const __grid_constant__ vllm::RankData buffers, int rank,
    const float* residual, const Weight* weight, half* normalized,
    float* residual_out, float epsilon) {
  using P = typename vllm::packed_t<half>::P;
  using A = typename vllm::packed_t<half>::A;
  char* local =
      const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[rank]));
  auto* meta = reinterpret_cast<volatile vllm::Sm70PushNormPacketMeta*>(
      local + vllm::kSm70PushNormPacketOffset);
  const int row = blockIdx.x / PipeParts, part = blockIdx.x % PipeParts;
  const int tid = threadIdx.x;
  const uint32_t generation = meta->generation[0] + 1;
  const int epoch = (generation & 1) * 4 * PipePacks;
  dense_tile<4, 2, Format, Format>(down, x, 4352, 8, 4352, 136, 34, 1, nullptr,
                                   nullptr, nullptr, blockIdx.x, 0);
  __syncthreads();
  // Publish this output tile immediately, while other tiles still compute.
  if (tid < 64) {
    const int pack = (tid / 8) * 640 + blockIdx.x * 8 + tid % 8;
    P value = reinterpret_cast<const P*>(down.s[0].out)[pack];
#pragma unroll
    for (int i = 0; i < P::size; ++i)
      vllm::sm70_push_escape_sentinel(value.data[i]);
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      char* remote =
          const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[peer]));
      vllm::sm70_push_store_volatile_16b(
          value,
          remote + vllm::kSm70PushNormOffset + vllm::kSm70PushNormMetaBytes +
              (epoch + rank * PipePacks) * sizeof(P),
          pack);
    }
  }
  // All eighty CTAs are guaranteed resident. A consumer can wait for another
  // tile without preventing its producer from being scheduled.
  if (blockIdx.x >= 8 * PipeParts) return;
  float values[8] = {}, variance = 0;
  const int pack = row * 640 + part * 128 + tid;
  if (tid < 128) {
    P peers[4];
    while (true) {
      bool missing = false;
#pragma unroll
      for (int peer = 0; peer < 4; ++peer) {
        vllm::sm70_push_load_volatile_16b(
            peers[peer],
            local + vllm::kSm70PushNormOffset + vllm::kSm70PushNormMetaBytes +
                (epoch + peer * PipePacks) * sizeof(P),
            pack);
#pragma unroll
        for (int i = 0; i < P::size; ++i)
          missing |= vllm::sm70_push_is_sentinel(peers[peer].data[i]);
      }
      if (!missing) break;
    }
    const P sum = vllm::sm70_push_reduce<P, 4, A>(peers);
#pragma unroll
    for (int i = 0; i < P::size; ++i) {
      const int index = pack * P::size + i;
      values[i] = __half2float(sum.data[i]) + residual[index];
      variance += values[i] * values[i];
      residual_out[index] = values[i];
    }
    P empty;
#pragma unroll
    for (int i = 0; i < P::size; ++i)
      *reinterpret_cast<uint16_t*>(&empty.data[i]) =
          vllm::kSm70Tp4PushAllreduceSentinel;
#pragma unroll
    for (int peer = 0; peer < 4; ++peer)
      vllm::sm70_push_store_volatile_16b(
          empty,
          local + vllm::kSm70PushNormOffset + vllm::kSm70PushNormMetaBytes +
              (epoch + peer * PipePacks) * sizeof(P),
          pack);
  }
  // Match the original 128-thread BlockReduce topology. The extra four
  // warps attend block barriers without contributing a second norm tree.
  using WarpReduce = cub::WarpReduce<float>;
  __shared__ typename WarpReduce::TempStorage warp_storage[4];
  __shared__ float warp_sum[4];
  __shared__ float inverse;
  if (tid < 128) {
    variance = WarpReduce(warp_storage[tid / 32]).Reduce(variance, CubAddOp{});
    if ((tid & 31) == 0) warp_sum[tid / 32] = variance;
  }
  __syncthreads();
  if (tid == 0) {
    variance = warp_sum[0];
#pragma unroll
    for (int warp = 1; warp < 4; ++warp) variance += warp_sum[warp];
  }
  __syncthreads();
  if (tid == 0) {
    const uint64_t packet =
        (uint64_t(generation) << 32) | __float_as_uint(variance);
    asm volatile(
        "st.volatile.global.u64 [%0], %1;" ::"l"(&meta->partial[row][part]),
        "l"(packet)
        : "memory");
  }
  __syncwarp();
  if (tid < 32) {
    float own = 0;
    if (tid < PipeParts) {
      uint64_t packet;
      do {
        asm volatile("ld.volatile.global.u64 %0, [%1];"
                     : "=l"(packet)
                     : "l"(&meta->partial[row][tid])
                     : "memory");
      } while (uint32_t(packet >> 32) != generation);
      own = __uint_as_float(uint32_t(packet));
    }
    __syncwarp();
    float total = 0;
#pragma unroll
    for (int p = 0; p < PipeParts; ++p)
      total += __shfl_sync(0xffffffff, own, p);
    if (tid == 0) {
      inverse = rsqrtf(total / 5120 + epsilon);
      if (part == 0) meta->generation[row] = generation;
    }
  }
  __syncthreads();
  if (tid < 128) {
    P value;
#pragma unroll
    for (int i = 0; i < P::size; ++i)
      value.data[i] =
          __float2half_rn(values[i] * inverse *
                          (vllm::sm70_gemma_rms_norm_to_float(
                               weight[(part * 128 + tid) * P::size + i]) +
                           1.f));
    reinterpret_cast<P*>(normalized)[pack] = value;
  }
}

template <int Format, typename Weight = float>
void run_pipeline(Segs down, const half* x, vllm::RankData buffers, int rank,
                  const float* residual, const Weight* weight, half* normalized,
                  float* residual_out, float epsilon, bool fused) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (!fused) {
    launch<4, 2, Format, Format>(down, x, 4352, 8, 4352, 136, 34, 1, nullptr,
                                 nullptr, 80, stream, nullptr);
    vllm::sm70_push_allreduce_gemma_rms_norm<Weight><<<40, 128, 0, stream>>>(
        buffers, down.s[0].out, residual, weight, normalized, residual_out,
        rank, buffers.ptrs[rank], epsilon);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const auto kernel = projection_collective_pipeline<Format, Weight>;
  constexpr int shared = 4 * 256 * 16 + TAB_VECS * 16;
  int active = 0, device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel,
                                                               256, shared));
  TORCH_CHECK(props.major == 7 && props.minor == 0 && props.cooperativeLaunch &&
                  props.multiProcessorCount >= 80 && active >= 1,
              "tile pipeline requires eighty resident CTAs");
  void* args[] = {&down,   &x,          &buffers,      &rank,   &residual,
                  &weight, &normalized, &residual_out, &epsilon};
  C10_CUDA_CHECK(
      cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel),
                                  dim3(80), dim3(256), args, shared, stream));
}

}  // namespace
