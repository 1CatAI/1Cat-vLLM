#include <atomic>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <torch/all.h>

#include "custom_all_reduce.cuh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

bool sm70_profile_trace_enabled() {
  const char* value = std::getenv("VLLM_SM70_PROFILE_TRACE");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

const char* scalar_type_name(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Float:
      return "float32";
    case at::ScalarType::Half:
      return "float16";
    case at::ScalarType::BFloat16:
      return "bfloat16";
    default:
      return "other";
  }
}

const char* capture_status_name(cudaStreamCaptureStatus status) {
  switch (status) {
    case cudaStreamCaptureStatusNone:
      return "none";
    case cudaStreamCaptureStatusActive:
      return "active";
    case cudaStreamCaptureStatusInvalidated:
      return "invalidated";
    default:
      return "unknown";
  }
}

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  vllm::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<vllm::Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new vllm::CustomAllreduce(ipc_ptrs, rank_data.data_ptr(),
                                            rank_data.numel(), rank, world_size,
                                            fully_connected);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() ==
          t.numel() * t.element_size());
}

#if !defined(USE_ROCM)
namespace vllm {

constexpr int kQwen38HcDownLocalElements = 88;
constexpr int kQwen38HcDownLiveElements = 84;
constexpr int kQwen38HcDownLocalLoraElements = 80;
constexpr int kQwen38HcDownLocalInjectionElements = 1;
constexpr int kQwen38HcDownLocalPaddingElements = 3;
constexpr int kQwen38HcDownGatheredElements =
    kQwen38HcDownLiveElements * kSm70Tp4PushAllreduceWorldSize;
constexpr int kQwen38HcGateLocalElements = 2560;
constexpr int kQwen38HcGateGatheredElements =
    kQwen38HcGateLocalElements * kSm70Tp4PushAllreduceWorldSize;
constexpr int kQwen38HcOutputLocalElements =
    kQwen38HcGateLocalElements / kSm70Tp4PushAllreduceWorldSize;

template <int ngpus, int Rank>
__global__ void __launch_bounds__(128, 1)
    sm70_qwen38_hc_down_push_allgather(RankData push_buffers,
                                       const half* __restrict__ input,
                                       half* __restrict__ output) {
  static_assert(ngpus == kSm70Tp4PushAllreduceWorldSize);
  using P = typename packed_t<half>::P;
  constexpr int kElementsPerPack = P::size;
  constexpr int kPackedElements = kQwen38HcDownLocalElements / kElementsPerPack;
  constexpr int kPackedStride = kSm70Qwen38HcDownPushBytes / sizeof(P);
  static_assert(kPackedElements <= kPackedStride);

  auto* local_storage =
      const_cast<char*>(reinterpret_cast<const char*>(push_buffers.ptrs[Rank]));
  auto* local_epochs = reinterpret_cast<uint32_t*>(
      local_storage + kSm70Qwen38HcPushSignalOffset);
  const uint32_t epoch = local_epochs[kSm70Qwen38HcDownEpochIndex];
  const int epoch_offset = epoch * ngpus * kPackedStride;
  const int offset = threadIdx.x;

  if (offset < kPackedElements) {
    P value = reinterpret_cast<const P*>(input)[offset];
  #pragma unroll
    for (int element = 0; element < P::size; ++element) {
      sm70_push_escape_sentinel(value.data[element]);
    }

  #pragma unroll
    for (int destination_rank = 0; destination_rank < ngpus;
         ++destination_rank) {
      if (destination_rank == Rank) continue;
      auto* destination_base = const_cast<char*>(
          reinterpret_cast<const char*>(push_buffers.ptrs[destination_rank]));
      void* destination = destination_base + kSm70Qwen38HcDownPushOffset +
                          (epoch_offset + Rank * kPackedStride) * sizeof(P);
      sm70_push_store_volatile_16b(value, destination, offset);
    }

    P peer_values[ngpus];
    peer_values[Rank] = value;
    while (true) {
      bool has_empty_slot = false;
  #pragma unroll
      for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
        if (source_rank == Rank) continue;
        const void* source =
            local_storage + kSm70Qwen38HcDownPushOffset +
            (epoch_offset + source_rank * kPackedStride) * sizeof(P);
        sm70_push_load_volatile_16b(peer_values[source_rank], source, offset);
  #pragma unroll
        for (int element = 0; element < P::size; ++element) {
          has_empty_slot |=
              sm70_push_is_sentinel(peer_values[source_rank].data[element]);
        }
      }
      if (!has_empty_slot) break;
    }

  #pragma unroll
    for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
  #pragma unroll
      for (int element = 0; element < P::size; ++element) {
        const int local_element = offset * kElementsPerPack + element;
        if (local_element < kQwen38HcDownLocalLoraElements) {
          output[source_rank * kQwen38HcDownLocalLoraElements + local_element] =
              peer_values[source_rank].data[element];
        } else if (local_element == kQwen38HcDownLocalLoraElements) {
          output[ngpus * kQwen38HcDownLocalLoraElements + source_rank] =
              peer_values[source_rank].data[element];
        } else if (local_element < kQwen38HcDownLiveElements) {
          const int local_padding = local_element -
                                    kQwen38HcDownLocalLoraElements -
                                    kQwen38HcDownLocalInjectionElements;
          output[ngpus * (kQwen38HcDownLocalLoraElements +
                          kQwen38HcDownLocalInjectionElements) +
                 source_rank * kQwen38HcDownLocalPaddingElements +
                 local_padding] = peer_values[source_rank].data[element];
        }
      }
    }

    P empty;
  #pragma unroll
    for (int element = 0; element < P::size; ++element) {
      *reinterpret_cast<uint16_t*>(&empty.data[element]) =
          kSm70Tp4PushAllreduceSentinel;
    }
  #pragma unroll
    for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
      if (source_rank == Rank) continue;
      void* source = local_storage + kSm70Qwen38HcDownPushOffset +
                     (epoch_offset + source_rank * kPackedStride) * sizeof(P);
      sm70_push_store_volatile_16b(empty, source, offset);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    local_epochs[kSm70Qwen38HcDownEpochIndex] =
        (epoch + 1) % kSm70Tp4PushAllreduceEpochs;
  }
}

DINLINE float qwen38_hc_sigmoid_fp32(float value) {
  constexpr uint32_t kLog2E = 0x3fb8aa3b;
  const float log2e = __uint_as_float(kLog2E);
  const float negated = __fsub_rn(0.0f, value);
  const float exponent = __fmul_rn(negated, log2e);
  float exp2;
  asm volatile("ex2.approx.f32 %0, %1;" : "=f"(exp2) : "f"(exponent));
  const float denominator = __fadd_rn(exp2, 1.0f);
  float result;
  asm volatile("div.full.f32 %0, %1, %2;"
               : "=f"(result)
               : "f"(1.0f), "f"(denominator));
  return result;
}

DINLINE float qwen38_hc_divide_by_count(float value) {
  float result;
  asm volatile("div.full.f32 %0, %1, %2;"
               : "=f"(result)
               : "f"(value), "f"(4.0f));
  return result;
}

// Stream-ordered TP4 HC calls share per-CTA counters. Cooperative launch
// keeps every CTA eligible to make progress while polling its remote peers.
// Pack an exact FP16 output and its 16-bit generation into one aligned word:
// readiness never escapes or changes a floating-point value (including NaNs).
// Two slots are sufficient: a rank cannot produce generation g+2 until every
// peer has produced g+1, hence completed its reads of g. Tag wrap is safe for
// the same reason; only adjacent generations can be in flight.
__device__ __forceinline__ uint4 qwen38_hc_load8(const half* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ float qwen38_hc_half_at(uint4 v, int i) {
  const uint32_t word = i < 2 ? v.x : i < 4 ? v.y : i < 6 ? v.z : v.w;
  return __half2float(__ushort_as_half((word >> ((i & 1) * 16)) & 0xffffu));
}

__global__ void __launch_bounds__(256)
    sm70_qwen38_hc_up_mix_push(const half* lora, const half* weight,
                               const half* branches, half* output,
                               RankData peers, int rank) {
  constexpr int Hidden = 4;
  // Constant parameter indices avoid materializing RankData in local memory.
  const void* local_peer = rank == 0   ? peers.ptrs[0]
                           : rank == 1 ? peers.ptrs[1]
                           : rank == 2 ? peers.ptrs[2]
                                       : peers.ptrs[3];
  auto* local = const_cast<char*>(reinterpret_cast<const char*>(local_peer));
  auto* counters =
      reinterpret_cast<uint32_t*>(local + kSm70Qwen38HcUpFusedEpochOffset);
  __shared__ float gates[Hidden * 4];
  __shared__ float partial[Hidden * 4][2];
  const int t = threadIdx.x;
  const int lane = t & 31, warp = t >> 5;
  const int pair = warp >> 1, kg = warp & 1, kp = kg * 32 + lane;
  uint4 lora_values;
  if (kp < 40) lora_values = qwen38_hc_load8(lora + kp * 8);
  // Same 8-term FMA chain, XOR tree, cross-warp add and FP16 gate boundary
  // as the accepted Triton up projection. Only the row assignment changes.
  #pragma unroll
  for (int group = 0; group < Hidden / 2; ++group) {
    const int a = group * 8 + pair, b = a + 4;
    const int ra = (a % 4) * 2560 + rank * 640 + blockIdx.x * Hidden + a / 4;
    const int rb = (b % 4) * 2560 + rank * 640 + blockIdx.x * Hidden + b / 4;
    float va = 0.f, vb = 0.f;
    if (kp < 40) {
      const int k = kp * 8;
      const uint4 wa = qwen38_hc_load8(weight + ra * 320 + k);
      const uint4 wb = qwen38_hc_load8(weight + rb * 320 + k);
      const float x1 = qwen38_hc_half_at(lora_values, 1);
      va = __fmul_rn(x1, qwen38_hc_half_at(wa, 1));
      vb = __fmul_rn(x1, qwen38_hc_half_at(wb, 1));
      va = __fmaf_rn(qwen38_hc_half_at(lora_values, 0),
                     qwen38_hc_half_at(wa, 0), va);
      vb = __fmaf_rn(qwen38_hc_half_at(lora_values, 0),
                     qwen38_hc_half_at(wb, 0), vb);
  #pragma unroll
      for (int e = 2; e < 8; ++e) {
        const float x = qwen38_hc_half_at(lora_values, e);
        va = __fmaf_rn(x, qwen38_hc_half_at(wa, e), va);
        vb = __fmaf_rn(x, qwen38_hc_half_at(wb, e), vb);
      }
    }
  #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      va = __fadd_rn(va, __shfl_xor_sync(0xffffffff, va, d));
      vb = __fadd_rn(vb, __shfl_xor_sync(0xffffffff, vb, d));
    }
    if (lane == 0) {
      partial[a][kg] = va;
      partial[b][kg] = vb;
    }
  }
  __syncthreads();
  if (t < Hidden * 4)
    gates[t] = qwen38_hc_sigmoid_fp32(
        __half2float(__float2half_rn(__fadd_rn(partial[t][0], partial[t][1]))));
  __syncthreads();
  if (t < Hidden) {
    const int h = blockIdx.x * Hidden + t;
    float mixed = 0.f;
  #pragma unroll
    for (int branch = 0; branch < 4; ++branch)
      mixed = __fmaf_rn(gates[t * 4 + branch],
                        __half2float(branches[branch * 2560 + rank * 640 + h]),
                        mixed);
    float scaled;
    asm("div.full.f32 %0, %1, %2;" : "=f"(scaled) : "f"(mixed), "f"(4.f));
    const half value = __float2half_rn(scaled);
    {
      const uint32_t generation = counters[blockIdx.x] + 1u;
      const uint32_t tag = generation & 0xffffu;
      const uint32_t packet = (tag << 16) | __half_as_ushort(value);
      const int slot = (generation & 1u) * 4 * 640;
  #pragma unroll
      for (int dest = 0; dest < 4; ++dest) {
        if (dest == rank) continue;
        auto* p = reinterpret_cast<uint32_t*>(
                      const_cast<char*>(
                          reinterpret_cast<const char*>(peers.ptrs[dest])) +
                      kSm70Qwen38HcUpFusedPacketOffset) +
                  slot + rank * 640 + h;
        asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(packet)
                     : "memory");
      }
      output[rank * 640 + h] = value;
  #pragma unroll
      for (int src = 0; src < 4; ++src) {
        if (src == rank) continue;
        auto* p = reinterpret_cast<uint32_t*>(
                      local + kSm70Qwen38HcUpFusedPacketOffset) +
                  slot + src * 640 + h;
        uint32_t received;
        do {
          asm volatile("ld.volatile.global.u32 %0, [%1];"
                       : "=r"(received)
                       : "l"(p)
                       : "memory");
        } while ((received >> 16) != tag);
        output[src * 640 + h] = __ushort_as_half(received & 0xffffu);
      }
    }
  }
  {
    __syncthreads();
    if (t == 0) counters[blockIdx.x] += 1;
  }
}

template <int ngpus, int Rank, bool GatherOutput = false>
__global__ void __launch_bounds__(512, 1)
    sm70_qwen38_hc_gate_push_mix(RankData push_buffers,
                                 const half* __restrict__ local_gate,
                                 const half* __restrict__ branches,
                                 half* __restrict__ output,
                                 int packed_elements) {
  static_assert(ngpus == kSm70Tp4PushAllreduceWorldSize);
  using P = typename packed_t<half>::P;
  constexpr int kPackedStride = kSm70Qwen38HcGatePushBytes / sizeof(P);

  auto* local_storage =
      const_cast<char*>(reinterpret_cast<const char*>(push_buffers.ptrs[Rank]));
  auto* local_epochs = reinterpret_cast<uint32_t*>(
      local_storage + kSm70Qwen38HcPushSignalOffset);
  const int epoch_index = kSm70Qwen38HcGateEpochIndexBase + blockIdx.x;
  const uint32_t epoch = local_epochs[epoch_index];
  const int epoch_offset = epoch * ngpus * kPackedStride;
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (offset < packed_elements) {
    P value = reinterpret_cast<const P*>(local_gate)[offset];
  #pragma unroll
    for (int element = 0; element < P::size; ++element) {
      sm70_push_escape_sentinel(value.data[element]);
    }

  #pragma unroll
    for (int destination_rank = 0; destination_rank < ngpus;
         ++destination_rank) {
      if (destination_rank == Rank) continue;
      auto* destination_base = const_cast<char*>(
          reinterpret_cast<const char*>(push_buffers.ptrs[destination_rank]));
      void* destination = destination_base + kSm70Qwen38HcGatePushOffset +
                          (epoch_offset + Rank * kPackedStride) * sizeof(P);
      sm70_push_store_volatile_16b(value, destination, offset);
    }

    P peer_values[ngpus];
    peer_values[Rank] = value;
    while (true) {
      bool has_empty_slot = false;
  #pragma unroll
      for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
        if (source_rank == Rank) continue;
        const void* source =
            local_storage + kSm70Qwen38HcGatePushOffset +
            (epoch_offset + source_rank * kPackedStride) * sizeof(P);
        sm70_push_load_volatile_16b(peer_values[source_rank], source, offset);
  #pragma unroll
        for (int element = 0; element < P::size; ++element) {
          has_empty_slot |=
              sm70_push_is_sentinel(peer_values[source_rank].data[element]);
        }
      }
      if (!has_empty_slot) break;
    }

    if constexpr (GatherOutput) {
      // Up already mixed all branches for 640 hidden coordinates. Gather
      // these final FP16 values, without a second arithmetic/rounding step.
      // Reuse the isolated HC gate channel and its existing epoch protocol;
      // the MoE/shared-expert stream uses a separate channel.
  #pragma unroll
      for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
        reinterpret_cast<P*>(
            output + source_rank * kQwen38HcOutputLocalElements)[offset] =
            peer_values[source_rank];
      }
    } else {
  #pragma unroll
      for (int element = 0; element < P::size; ++element) {
        const int hidden = offset * P::size + element;
        float result = 0.0f;
  #pragma unroll
        for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
          const float gate =
              __half2float(peer_values[source_rank].data[element]);
          const float branch = __half2float(
              branches[source_rank * kQwen38HcGateLocalElements + hidden]);
          result = __fmaf_rn(qwen38_hc_sigmoid_fp32(gate), branch, result);
        }
        output[hidden] = __float2half_rn(qwen38_hc_divide_by_count(result));
      }
    }

    P empty;
  #pragma unroll
    for (int element = 0; element < P::size; ++element) {
      *reinterpret_cast<uint16_t*>(&empty.data[element]) =
          kSm70Tp4PushAllreduceSentinel;
    }
  #pragma unroll
    for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
      if (source_rank == Rank) continue;
      void* source = local_storage + kSm70Qwen38HcGatePushOffset +
                     (epoch_offset + source_rank * kPackedStride) * sizeof(P);
      sm70_push_store_volatile_16b(empty, source, offset);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    local_epochs[epoch_index] = (epoch + 1) % kSm70Tp4PushAllreduceEpochs;
  }
}

}  // namespace vllm
#endif

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                                  cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->allreduce<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<float*>(out.data_ptr()),
                           out.numel());
      break;
    }
    case at::ScalarType::Half: {
      fa->allreduce<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<half*>(out.data_ptr()), out.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom allreduce only supports float32, float16 and bfloat16");
  }
}

namespace {

template <int kWorldSize>
void sm70_all_reduce_gemma_rms_norm_impl(
    fptr_t _fa, torch::Tensor& inp, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& normalized_out,
    torch::Tensor& residual_out, fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes, double epsilon) {
  constexpr int64_t kHiddenSize = vllm::kSm70GemmaRmsNormHiddenSize;
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(normalized_out.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(residual_out.scalar_type(), at::ScalarType::Float);
  if constexpr (kWorldSize == 4) {
    TORCH_CHECK(residual.scalar_type() == at::ScalarType::Float,
                "SM70 TP4 Gemma RMSNorm prototype residual must be float32.");
  } else {
    TORCH_CHECK(residual.scalar_type() == at::ScalarType::Half ||
                    residual.scalar_type() == at::ScalarType::Float,
                "SM70 Gemma RMSNorm prototype residual must be float16 or "
                "float32.");
  }
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half ||
                  weight.scalar_type() == at::ScalarType::Float,
              "SM70 Gemma RMSNorm prototype weight must be float16 or "
              "float32.");
  TORCH_CHECK_EQ(inp.dim(), 2);
  TORCH_CHECK_EQ(inp.size(1), kHiddenSize);
  TORCH_CHECK(residual.sizes() == inp.sizes(),
              "SM70 Gemma RMSNorm prototype residual shape must match inp.");
  TORCH_CHECK(normalized_out.sizes() == inp.sizes(),
              "SM70 Gemma RMSNorm prototype normalized_out shape must match "
              "inp.");
  TORCH_CHECK(
      residual_out.sizes() == inp.sizes(),
      "SM70 Gemma RMSNorm prototype residual_out shape must match inp.");
  TORCH_CHECK_EQ(weight.dim(), 1);
  TORCH_CHECK_EQ(weight.numel(), kHiddenSize);
  TORCH_CHECK_EQ(residual.get_device(), inp.get_device());
  TORCH_CHECK_EQ(weight.get_device(), inp.get_device());
  TORCH_CHECK_EQ(normalized_out.get_device(), inp.get_device());
  TORCH_CHECK_EQ(residual_out.get_device(), inp.get_device());
  TORCH_CHECK(inp.is_contiguous());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(normalized_out.is_contiguous());
  TORCH_CHECK(residual_out.is_contiguous());

  const auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                                  cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }

  const int num_tokens = static_cast<int>(inp.size(0));
  const int hidden_size = static_cast<int>(inp.size(1));
  const float epsilon_f = static_cast<float>(epsilon);

  if constexpr (kWorldSize == 4) {
    static const bool trace_enabled = sm70_profile_trace_enabled();
    static std::atomic<bool> logged_route{false};
    if (trace_enabled) {
      cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
      AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
      bool expected = false;
      if (capture_status == cudaStreamCaptureStatusActive &&
          logged_route.compare_exchange_strong(expected, true)) {
        std::cerr << "SM70 TP4 all_reduce_gemma_rms_norm op reached"
                  << " rank=" << fa->rank_ << " num_tokens=" << num_tokens
                  << " residual=" << scalar_type_name(residual.scalar_type())
                  << " capture=" << capture_status_name(capture_status)
                  << std::endl;
      }
    }
  }

  auto input_ptr = reinterpret_cast<half*>(reg_buffer);
  auto normalized_out_ptr = reinterpret_cast<half*>(normalized_out.data_ptr());
  auto residual_out_ptr = reinterpret_cast<float*>(residual_out.data_ptr());

  if (residual.scalar_type() == at::ScalarType::Float) {
    auto residual_ptr = reinterpret_cast<const float*>(residual.data_ptr());
    if (weight.scalar_type() == at::ScalarType::Float) {
      fa->sm70_allreduce_gemma_rms_norm<kWorldSize, float, float>(
          stream, input_ptr, residual_ptr,
          reinterpret_cast<const float*>(weight.data_ptr()), normalized_out_ptr,
          residual_out_ptr, num_tokens, hidden_size, epsilon_f);
    } else {
      fa->sm70_allreduce_gemma_rms_norm<kWorldSize, float, half>(
          stream, input_ptr, residual_ptr,
          reinterpret_cast<const half*>(weight.data_ptr()), normalized_out_ptr,
          residual_out_ptr, num_tokens, hidden_size, epsilon_f);
    }
  } else if constexpr (kWorldSize == 2) {
    auto residual_ptr = reinterpret_cast<const half*>(residual.data_ptr());
    if (weight.scalar_type() == at::ScalarType::Float) {
      fa->sm70_allreduce_gemma_rms_norm<kWorldSize, half, float>(
          stream, input_ptr, residual_ptr,
          reinterpret_cast<const float*>(weight.data_ptr()), normalized_out_ptr,
          residual_out_ptr, num_tokens, hidden_size, epsilon_f);
    } else {
      fa->sm70_allreduce_gemma_rms_norm<kWorldSize, half, half>(
          stream, input_ptr, residual_ptr,
          reinterpret_cast<const half*>(weight.data_ptr()), normalized_out_ptr,
          residual_out_ptr, num_tokens, hidden_size, epsilon_f);
    }
  }
}

}  // namespace

void sm70_tp2_all_reduce_gemma_rms_norm(
    fptr_t _fa, torch::Tensor& inp, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& normalized_out,
    torch::Tensor& residual_out, fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes, double epsilon) {
  sm70_all_reduce_gemma_rms_norm_impl<2>(
      _fa, inp, residual, weight, normalized_out, residual_out, _reg_buffer,
      reg_buffer_sz_bytes, epsilon);
}

void sm70_tp4_all_reduce_gemma_rms_norm(
    fptr_t _fa, torch::Tensor& inp, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& normalized_out,
    torch::Tensor& residual_out, fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes, double epsilon) {
  sm70_all_reduce_gemma_rms_norm_impl<4>(
      _fa, inp, residual, weight, normalized_out, residual_out, _reg_buffer,
      reg_buffer_sz_bytes, epsilon);
}

void sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather(
    fptr_t _fa, torch::Tensor& inp, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& normalized_out,
    torch::Tensor& residual_out, fptr_t _reg_input_buffer,
    fptr_t _reg_output_buffer, int64_t reg_buffer_sz_bytes, double epsilon) {
  constexpr int64_t kWorldSize = 4;
  constexpr int64_t kHiddenSize = vllm::kSm70GemmaRmsNormHiddenSize;
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(fa->world_size_, kWorldSize);
  TORCH_CHECK(fa->fully_connected_);
  TORCH_CHECK_EQ(inp.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(residual.scalar_type(), at::ScalarType::Float);
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half ||
              weight.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK_EQ(normalized_out.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(residual_out.scalar_type(), at::ScalarType::Float);
  TORCH_CHECK_EQ(inp.dim(), 2);
  TORCH_CHECK_EQ(inp.size(1), kHiddenSize);
  TORCH_CHECK_EQ(inp.size(0) % kWorldSize, 0);
  TORCH_CHECK_LE(inp.size(0) / kWorldSize,
                 vllm::kSm70LongPrefillMaxTokensPerRank);
  TORCH_CHECK(residual.sizes() == inp.sizes());
  TORCH_CHECK(normalized_out.sizes() == inp.sizes());
  TORCH_CHECK(residual_out.sizes() == inp.sizes());
  TORCH_CHECK_EQ(weight.dim(), 1);
  TORCH_CHECK_EQ(weight.numel(), kHiddenSize);
  TORCH_CHECK_EQ(residual.get_device(), inp.get_device());
  TORCH_CHECK_EQ(weight.get_device(), inp.get_device());
  TORCH_CHECK_EQ(normalized_out.get_device(), inp.get_device());
  TORCH_CHECK_EQ(residual_out.get_device(), inp.get_device());
  TORCH_CHECK(inp.is_contiguous());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(normalized_out.is_contiguous());
  TORCH_CHECK(residual_out.is_contiguous());

  auto* reg_input_buffer = reinterpret_cast<void*>(_reg_input_buffer);
  auto* reg_output_buffer = reinterpret_cast<void*>(_reg_output_buffer);
  const int64_t input_size = inp.numel() * inp.element_size();
  TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
  if (reg_input_buffer != nullptr) {
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_input_buffer, inp.data_ptr(), input_size,
                                  cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_input_buffer = inp.data_ptr();
  }
  if (reg_output_buffer == nullptr) {
    reg_output_buffer = normalized_out.data_ptr();
  }

  auto* input_ptr = reinterpret_cast<half*>(reg_input_buffer);
  auto* shared_output_ptr = reinterpret_cast<half*>(reg_output_buffer);
  auto* residual_ptr = reinterpret_cast<const float*>(residual.data_ptr());
  auto* residual_out_ptr = reinterpret_cast<float*>(residual_out.data_ptr());
  const int num_tokens = static_cast<int>(inp.size(0));
  const float epsilon_f = static_cast<float>(epsilon);
  if (weight.scalar_type() == at::ScalarType::Float) {
    fa->sm70_reduce_scatter_gemma_rms_norm_all_gather<4, float, float>(
        stream, input_ptr, shared_output_ptr, residual_ptr,
        reinterpret_cast<const float*>(weight.data_ptr()), residual_out_ptr,
        num_tokens, kHiddenSize, epsilon_f);
  } else {
    fa->sm70_reduce_scatter_gemma_rms_norm_all_gather<4, float, half>(
        stream, input_ptr, shared_output_ptr, residual_ptr,
        reinterpret_cast<const half*>(weight.data_ptr()), residual_out_ptr,
        num_tokens, kHiddenSize, epsilon_f);
  }
  if (_reg_output_buffer != 0) {
    AT_CUDA_CHECK(cudaMemcpyAsync(normalized_out.data_ptr(), reg_output_buffer,
                                  input_size, cudaMemcpyDeviceToDevice,
                                  stream));
  }
}

void all_reduce_sum2(fptr_t _fa, torch::Tensor& inp_a, torch::Tensor& inp_b,
                     torch::Tensor& out) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp_a));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));

  TORCH_CHECK_EQ(inp_a.scalar_type(), inp_b.scalar_type());
  TORCH_CHECK_EQ(inp_a.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp_a.numel(), inp_b.numel());
  TORCH_CHECK_EQ(inp_a.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(inp_a));
  TORCH_CHECK(_is_weak_contiguous(inp_b));
  TORCH_CHECK(_is_weak_contiguous(out));

  static std::atomic<bool> logged_sum2_route{false};
  bool expected = false;
  if (sm70_profile_trace_enabled() &&
      logged_sum2_route.compare_exchange_strong(expected, true)) {
    std::cerr << "SM70 custom all_reduce_sum2 op reached"
              << " rank=" << fa->rank_ << " world_size=" << fa->world_size_
              << " numel=" << out.numel()
              << " dtype=" << scalar_type_name(out.scalar_type())
              << " capture=" << capture_status_name(capture_status)
              << std::endl;
  }

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->allreduce_sum2<float>(
          stream, reinterpret_cast<float*>(inp_a.data_ptr()),
          reinterpret_cast<float*>(inp_b.data_ptr()),
          reinterpret_cast<float*>(out.data_ptr()), out.numel());
      break;
    }
    case at::ScalarType::Half: {
      fa->allreduce_sum2<half>(
          stream, reinterpret_cast<half*>(inp_a.data_ptr()),
          reinterpret_cast<half*>(inp_b.data_ptr()),
          reinterpret_cast<half*>(out.data_ptr()), out.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      fa->allreduce_sum2<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp_a.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(inp_b.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom allreduce sum2 only supports float32, float16 and bfloat16");
  }
}

void sm70_qwen38_hc_down_allgather(fptr_t _fa, torch::Tensor& input,
                                   torch::Tensor& output) {
#if defined(USE_ROCM)
  TORCH_CHECK(false, "SM70 Qwen3.8 HC all-gather is unavailable on ROCm");
#else
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK_EQ(fa->world_size_, vllm::kSm70Tp4PushAllreduceWorldSize);
  TORCH_CHECK(fa->fully_connected_ && fa->sm70_tp4_push_buffers_registered_);
  TORCH_CHECK_EQ(input.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(output.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(input.numel(), vllm::kQwen38HcDownLocalElements);
  TORCH_CHECK_EQ(output.numel(), vllm::kQwen38HcDownGatheredElements);
  TORCH_CHECK(_is_weak_contiguous(input) && _is_weak_contiguous(output));
  #define VLLM_LAUNCH_QWEN38_HC_DOWN(RANK)                                   \
    vllm::sm70_qwen38_hc_down_push_allgather<4, RANK><<<1, 32, 0, stream>>>( \
        fa->sm70_tp4_push_buffers_,                                          \
        reinterpret_cast<const half*>(input.data_ptr()),                     \
        reinterpret_cast<half*>(output.data_ptr()))
  switch (fa->rank_) {
    case 0:
      VLLM_LAUNCH_QWEN38_HC_DOWN(0);
      break;
    case 1:
      VLLM_LAUNCH_QWEN38_HC_DOWN(1);
      break;
    case 2:
      VLLM_LAUNCH_QWEN38_HC_DOWN(2);
      break;
    default:
      VLLM_LAUNCH_QWEN38_HC_DOWN(3);
      break;
  }
  #undef VLLM_LAUNCH_QWEN38_HC_DOWN
#endif
}

void sm70_qwen38_hc_gate_mix(fptr_t _fa, torch::Tensor& local_gate,
                             torch::Tensor& branches, torch::Tensor& output) {
#if defined(USE_ROCM)
  TORCH_CHECK(false, "SM70 Qwen3.8 HC gate-mix is unavailable on ROCm");
#else
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(local_gate));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK_EQ(fa->world_size_, vllm::kSm70Tp4PushAllreduceWorldSize);
  TORCH_CHECK(fa->fully_connected_ && fa->sm70_tp4_push_buffers_registered_);
  TORCH_CHECK_EQ(local_gate.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(branches.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(output.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(local_gate.numel(), vllm::kQwen38HcGateLocalElements);
  TORCH_CHECK_EQ(branches.numel(), vllm::kQwen38HcGateGatheredElements);
  TORCH_CHECK_EQ(output.numel(), vllm::kQwen38HcGateLocalElements);
  TORCH_CHECK(_is_weak_contiguous(local_gate) &&
              _is_weak_contiguous(branches) && _is_weak_contiguous(output));
  constexpr int kPackedElements =
      vllm::kQwen38HcGateLocalElements / vllm::packed_t<half>::P::size;
  constexpr int kThreads = 32;
  constexpr int kBlocks = (kPackedElements + kThreads - 1) / kThreads;
  #define VLLM_LAUNCH_QWEN38_HC_GATE(RANK)                        \
    vllm::sm70_qwen38_hc_gate_push_mix<4, RANK>                   \
        <<<kBlocks, kThreads, 0, stream>>>(                       \
            fa->sm70_tp4_push_buffers_,                           \
            reinterpret_cast<const half*>(local_gate.data_ptr()), \
            reinterpret_cast<const half*>(branches.data_ptr()),   \
            reinterpret_cast<half*>(output.data_ptr()), kPackedElements)
  switch (fa->rank_) {
    case 0:
      VLLM_LAUNCH_QWEN38_HC_GATE(0);
      break;
    case 1:
      VLLM_LAUNCH_QWEN38_HC_GATE(1);
      break;
    case 2:
      VLLM_LAUNCH_QWEN38_HC_GATE(2);
      break;
    default:
      VLLM_LAUNCH_QWEN38_HC_GATE(3);
      break;
  }
  #undef VLLM_LAUNCH_QWEN38_HC_GATE
#endif
}

void sm70_qwen38_hc_up_mix_allgather(fptr_t _fa, torch::Tensor& lora,
                                     torch::Tensor& weight,
                                     torch::Tensor& branches,
                                     torch::Tensor& output) {
#if defined(USE_ROCM)
  TORCH_CHECK(false, "SM70 Qwen3.8 HC up/mix is unavailable on ROCm");
#else
  TORCH_CHECK(lora.is_cuda());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(lora));
  for (const auto* tensor : {&lora, &weight, &branches, &output}) {
    TORCH_CHECK(tensor->device() == lora.device());
    TORCH_CHECK(tensor->scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(tensor->is_contiguous());
  }
  TORCH_CHECK(lora.numel() == 336 && weight.numel() == 10240 * 320 &&
              branches.numel() == 10240 && output.numel() == 2560);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(lora.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0);
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK(fa->world_size_ == 4 && fa->fully_connected_ &&
              fa->sm70_tp4_push_buffers_registered_);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const half* lp = reinterpret_cast<const half*>(lora.data_ptr());
  const half* wp = reinterpret_cast<const half*>(weight.data_ptr());
  const half* xp = reinterpret_cast<const half*>(branches.data_ptr());
  half* out = reinterpret_cast<half*>(output.data_ptr());
  auto peers = fa->sm70_tp4_push_buffers_;
  int rank = fa->rank_;
  void* args[] = {&lp, &wp, &xp, &out, &peers, &rank};
  CUDACHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(vllm::sm70_qwen38_hc_up_mix_push),
      dim3(vllm::kSm70Qwen38HcUpFusedBlocks), dim3(256), args, 0, stream));
#endif
}

void sm70_qwen38_hc_output_allgather(fptr_t _fa, torch::Tensor& local_block,
                                     torch::Tensor& output) {
#if defined(USE_ROCM)
  TORCH_CHECK(false,
              "SM70 Qwen3.8 HC output all-gather is unavailable on ROCm");
#else
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK(local_block.is_cuda() && output.is_cuda());
  TORCH_CHECK(local_block.device() == output.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(local_block));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK_EQ(fa->world_size_, vllm::kSm70Tp4PushAllreduceWorldSize);
  TORCH_CHECK(fa->fully_connected_ && fa->sm70_tp4_push_buffers_registered_);
  TORCH_CHECK_EQ(local_block.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(output.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(local_block.numel(), vllm::kQwen38HcOutputLocalElements);
  TORCH_CHECK_EQ(output.numel(), vllm::kQwen38HcGateLocalElements);
  TORCH_CHECK(local_block.is_contiguous() && output.is_contiguous());
  constexpr int kPackedElements =
      vllm::kQwen38HcOutputLocalElements / vllm::packed_t<half>::P::size;
  constexpr int kThreads = 32;
  constexpr int kBlocks = (kPackedElements + kThreads - 1) / kThreads;
  #define VLLM_LAUNCH_QWEN38_HC_OUTPUT(RANK)                                \
    vllm::sm70_qwen38_hc_gate_push_mix<4, RANK, true>                       \
        <<<kBlocks, kThreads, 0, stream>>>(                                 \
            fa->sm70_tp4_push_buffers_,                                     \
            reinterpret_cast<const half*>(local_block.data_ptr()), nullptr, \
            reinterpret_cast<half*>(output.data_ptr()), kPackedElements)
  switch (fa->rank_) {
    case 0:
      VLLM_LAUNCH_QWEN38_HC_OUTPUT(0);
      break;
    case 1:
      VLLM_LAUNCH_QWEN38_HC_OUTPUT(1);
      break;
    case 2:
      VLLM_LAUNCH_QWEN38_HC_OUTPUT(2);
      break;
    default:
      VLLM_LAUNCH_QWEN38_HC_OUTPUT(3);
      break;
  }
  #undef VLLM_LAUNCH_QWEN38_HC_OUTPUT
#endif
}

void top1_argmax(fptr_t _fa, torch::Tensor& input_pair, torch::Tensor& output,
                 fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input_pair));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(input_pair.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Long);
  TORCH_CHECK(input_pair.numel() == 2);
  TORCH_CHECK(output.numel() == 1);
  TORCH_CHECK(_is_weak_contiguous(input_pair));
  TORCH_CHECK(_is_weak_contiguous(output));

  auto input_size = input_pair.numel() * input_pair.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, input_pair.data_ptr(), input_size,
                                  cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = input_pair.data_ptr();
  }

  fa->top1_argmax(stream, reinterpret_cast<float*>(reg_buffer),
                  reinterpret_cast<int64_t*>(output.data_ptr()));
}

void tile_runtime_all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                             fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes,
                             int64_t tile_numel, int64_t engine_blocks,
                             int64_t compute_iters) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(tile_numel > 0);
  TORCH_CHECK(engine_blocks >= 0);
  TORCH_CHECK(compute_iters >= 0);

  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  TORCH_CHECK(reg_buffer != nullptr,
              "SM70 tile runtime prototype requires a registered staging "
              "buffer.");
  TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->tile_runtime_allreduce<float>(
          stream, reinterpret_cast<const float*>(inp.data_ptr()),
          reinterpret_cast<float*>(reg_buffer),
          reinterpret_cast<float*>(out.data_ptr()), out.numel(), tile_numel,
          engine_blocks, compute_iters);
      break;
    }
    case at::ScalarType::Half: {
      fa->tile_runtime_allreduce<half>(
          stream, reinterpret_cast<const half*>(inp.data_ptr()),
          reinterpret_cast<half*>(reg_buffer),
          reinterpret_cast<half*>(out.data_ptr()), out.numel(), tile_numel,
          engine_blocks, compute_iters);
      break;
    }
    default:
      throw std::runtime_error(
          "SM70 tile runtime prototype supports float32 and float16 only");
  }
}

void tile_runtime_all_reduce_engine(fptr_t _fa, torch::Tensor& inp,
                                    torch::Tensor& out, fptr_t _reg_buffer,
                                    int64_t reg_buffer_sz_bytes,
                                    int64_t tile_numel, int64_t producer_blocks,
                                    int64_t reducer_blocks,
                                    int64_t compute_iters) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(tile_numel > 0);
  TORCH_CHECK(producer_blocks >= 0);
  TORCH_CHECK(reducer_blocks >= 0);
  TORCH_CHECK(compute_iters >= 0);

  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  TORCH_CHECK(reg_buffer != nullptr,
              "SM70 tile runtime engine requires a registered staging buffer.");
  TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->tile_runtime_allreduce_engine<float>(
          stream, reinterpret_cast<const float*>(inp.data_ptr()),
          reinterpret_cast<float*>(reg_buffer),
          reinterpret_cast<float*>(out.data_ptr()), out.numel(), tile_numel,
          producer_blocks, reducer_blocks, compute_iters);
      break;
    }
    case at::ScalarType::Half: {
      fa->tile_runtime_allreduce_engine<half>(
          stream, reinterpret_cast<const half*>(inp.data_ptr()),
          reinterpret_cast<half*>(reg_buffer),
          reinterpret_cast<half*>(out.data_ptr()), out.numel(), tile_numel,
          producer_blocks, reducer_blocks, compute_iters);
      break;
    }
    default:
      throw std::runtime_error(
          "SM70 tile runtime engine supports float32 and float16 only");
  }
}

void tile_runtime_wait_reduce(fptr_t _fa, torch::Tensor& staging,
                              torch::Tensor& out, int64_t tile_numel,
                              int64_t reducer_blocks) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(staging));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(staging.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(staging.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(staging));
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(tile_numel > 0);
  TORCH_CHECK(reducer_blocks >= 0);

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->tile_runtime_wait_reduce<float>(
          stream, reinterpret_cast<float*>(staging.data_ptr()),
          reinterpret_cast<float*>(out.data_ptr()), out.numel(), tile_numel,
          reducer_blocks);
      break;
    }
    case at::ScalarType::Half: {
      fa->tile_runtime_wait_reduce<half>(
          stream, reinterpret_cast<half*>(staging.data_ptr()),
          reinterpret_cast<half*>(out.data_ptr()), out.numel(), tile_numel,
          reducer_blocks);
      break;
    }
    default:
      throw std::runtime_error(
          "SM70 tile runtime wait-reduce supports float32 and float16 only");
  }
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<vllm::CustomAllreduce*>(_fa);
}

int64_t meta_size() { return sizeof(vllm::Signal); }

int64_t sm70_tp4_push_allreduce_buffer_size() {
  return vllm::kSm70Tp4PushAllreduceBufferBytes;
}

int64_t sm70_tp8_hierarchical_push_allreduce_buffer_size() {
  return vllm::kSm70Tp8HierarchicalPushBufferBytes;
}

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

void register_sm70_tp4_push_allreduce_buffer(
    fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK_EQ(fake_ipc_ptrs.size(),
                 static_cast<size_t>(vllm::kSm70Tp4PushAllreduceWorldSize));
  TORCH_CHECK_EQ(fake_ipc_ptrs.size(), static_cast<size_t>(fa->world_size_));
  void* ipc_ptrs[vllm::kSm70Tp4PushAllreduceWorldSize];
  for (size_t peer = 0; peer < fake_ipc_ptrs.size(); ++peer) {
    ipc_ptrs[peer] = reinterpret_cast<void*>(fake_ipc_ptrs[peer]);
  }
  fa->register_sm70_tp4_push_buffer(ipc_ptrs);
}

void register_sm70_tp8_hierarchical_push_allreduce_buffer(
    fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK_EQ(fake_ipc_ptrs.size(),
                 static_cast<size_t>(vllm::kSm70Tp8HierarchicalPushWorldSize));
  TORCH_CHECK_EQ(fake_ipc_ptrs.size(), static_cast<size_t>(fa->world_size_));
  void* ipc_ptrs[vllm::kSm70Tp8HierarchicalPushWorldSize];
  for (size_t peer = 0; peer < fake_ipc_ptrs.size(); ++peer) {
    ipc_ptrs[peer] = reinterpret_cast<void*>(fake_ipc_ptrs[peer]);
  }
  fa->register_sm70_tp8_hierarchical_push_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  bytes.reserve(handles.size());
  fa->register_graph_buffers(bytes, offsets);
}

std::tuple<fptr_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size) {
  auto device_index = c10::cuda::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Allocate buffer
#if defined(USE_ROCM)
  // data buffers need to be "uncached" for signal on MI200
  AT_CUDA_CHECK(
      hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
#else
  AT_CUDA_CHECK(cudaMalloc((void**)&buffer, size));
#endif
  AT_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Create IPC memhandle for the allocated buffer.
  // Will use it in open_mem_handle.
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto handle =
      torch::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
  AT_CUDA_CHECK(
      cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle.data_ptr(), buffer));

  return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

fptr_t open_mem_handle(torch::Tensor& mem_handle) {
  void* ipc_ptr;
  AT_CUDA_CHECK(cudaIpcOpenMemHandle(
      (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)mem_handle.data_ptr()),
      cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(ipc_ptr);
}

void free_shared_buffer(fptr_t buffer) {
  AT_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
}
