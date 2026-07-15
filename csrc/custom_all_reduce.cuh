#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(USE_ROCM)
typedef __hip_bfloat16 nv_bfloat16;
#endif

#include <iostream>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <string>

#include "cub_helpers.h"
#include "sm70_tile_runtime_signal.cuh"

namespace vllm {
#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

// Maximal number of signal slots. The default production all-reduce still
// launches at most defaultBlockLimit CTAs unless explicitly overridden.
constexpr int kMaxBlocks = sm70_tile_runtime::kMaxBlocks;

// Default number of blocks in allreduce kernel.
#ifndef USE_ROCM
const int defaultBlockLimit = 36;
inline CUpointer_attribute rangeStartAddrAttr =
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR;
#else
const int defaultBlockLimit = 16;
inline hipPointer_attribute rangeStartAddrAttr =
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR;
#endif

constexpr size_t kSm70Tp2SmallAllreduceBytes = 40 * 1024;
constexpr int kSm70GemmaRmsNormHiddenSize = 5120;
constexpr int kSm70GemmaRmsNormThreads = 1024;

inline int sm70_gemma_rms_norm_threads() {
  const char* raw = std::getenv("VLLM_SM70_TP2_AR_GEMMA_RMS_THREADS");
  if (raw == nullptr) return kSm70GemmaRmsNormThreads;
  const int threads = std::atoi(raw);
  return threads == 256 || threads == 512 || threads == 1024
             ? threads
             : kSm70GemmaRmsNormThreads;
}

inline bool custom_allreduce_current_device_is_sm70() {
#ifndef USE_ROCM
  int device = 0;
  CUDACHECK(cudaGetDevice(&device));
  if (device >= 0 && device < 64) {
    static int cached_arch[64] = {};
    if (cached_arch[device] == 0) {
      cudaDeviceProp prop{};
      CUDACHECK(cudaGetDeviceProperties(&prop, device));
      cached_arch[device] = prop.major * 10 + prop.minor;
    }
    return cached_arch[device] == 70;
  }
  cudaDeviceProp prop{};
  CUDACHECK(cudaGetDeviceProperties(&prop, device));
  return prop.major == 7 && prop.minor == 0;
#else
  return false;
#endif
}

inline int custom_allreduce_block_limit(int default_limit,
                                        int world_size,
                                        size_t bytes) {
  const char* raw = std::getenv("VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT");
  if (raw == nullptr || raw[0] == '\0') {
    if (world_size == 2 && bytes <= kSm70Tp2SmallAllreduceBytes &&
        custom_allreduce_current_device_is_sm70()) {
      return 1;
    }
    return default_limit;
  }
  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed <= 0 || parsed > kMaxBlocks) {
    throw std::runtime_error(
        "Invalid VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT: " + std::string(raw) +
        ". Expected an integer in [1, " + std::to_string(kMaxBlocks) + "]");
  }
  return static_cast<int>(parsed);
}

inline int sm70_tp4_m5_allreduce_threads(int world_size,
                                         bool fully_connected,
                                         size_t bytes) {
  const char* raw = std::getenv("VLLM_SM70_TP4_M5_AR_THREADS");
  if (raw == nullptr || raw[0] == '\0') return 512;

  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' ||
      (parsed != 128 && parsed != 256 && parsed != 512)) {
    throw std::runtime_error(
        "Invalid VLLM_SM70_TP4_M5_AR_THREADS: " + std::string(raw) +
        ". Expected one of 128, 256, 512.");
  }
  if (world_size != 4 || !fully_connected ||
      bytes != 5 * kSm70GemmaRmsNormHiddenSize * sizeof(half) ||
      !custom_allreduce_current_device_is_sm70()) {
    return 512;
  }
  return static_cast<int>(parsed);
}

// Counter may overflow, but unsigned integer overflow is well-defined.
using FlagType = sm70_tile_runtime::FlagType;
using Signal = sm70_tile_runtime::Signal;
using RankData = sm70_tile_runtime::RankData;
using RankSignals = sm70_tile_runtime::RankSignals;

// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

// scalar cast functions
DINLINE float upcast_s(half val) { return __half2float(val); }

DINLINE float sm70_gemma_rms_norm_to_float(float val) { return val; }
DINLINE float sm70_gemma_rms_norm_to_float(half val) {
  return __half2float(val);
}

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float& assign_add(float& a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

#if !defined(USE_ROCM)

static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  #else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
               : "=r"(flag)
               : "l"(flag_addr));
  #endif
  return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

static DINLINE void st_flag_sys_visible(FlagType* flag_addr, FlagType flag) {
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;"
               ::"r"(flag),
               "l"(flag_addr)
               : "memory");
}

static DINLINE FlagType ld_flag_sys_visible(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.sys;"
               : "=r"(flag)
               : "l"(flag_addr)
               : "memory");
  return flag;
}

// This function is meant to be used as the first synchronization in the all
// reduce kernel. The all-reduce input is usually produced by kernels launched
// immediately before this kernel on each rank. Use a system memory fence around
// the peer-visible flag so faster upstream paths do not expose stale input to
// peer ranks.
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value
    // from peer.
    st_flag_sys_visible(peer_counter_ptr, flag);
    while (ld_flag_sys_visible(self_counter_ptr) != flag);
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// This function is meant to be used as the second or the final
// synchronization barrier in the all reduce kernel. If it's the final
// synchronization barrier, we don't need to make any visibility guarantees
// for prior memory accesses.
template <int ngpus, bool final_sync = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value from
    // peer.
    if constexpr (!final_sync) {
      st_flag_release(peer_counter_ptr, flag);
      while (ld_flag_acquire(self_counter_ptr) != flag);
    } else {
      st_flag_volatile(peer_counter_ptr, flag);
      while (ld_flag_volatile(self_counter_ptr) != flag);
    }
  }
  if constexpr (!final_sync) __syncthreads();

  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#else

template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    // simultaneously write to the corresponding flag of all ranks.
    // Latency = 1 p2p write
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],
                            flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE) < flag);
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus, bool final_sync = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    // simultaneously write to the corresponding flag of all ranks.
    // Latency = 1 p2p write
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->end[blockIdx.x][rank],
                            flag,
                            final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (
        __scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x],
                               final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                               __MEMORY_SCOPE_DEVICE) < flag);
  }
  if constexpr (!final_sync) __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#endif

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce_sum2(const P* ptrs_a[], const P* ptrs_b[], int idx) {
  P local = ptrs_a[0][idx];
  packed_assign_add(local, ptrs_b[0][idx]);
  A tmp = upcast(local);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    local = ptrs_a[i][idx];
    packed_assign_add(local, ptrs_b[i][idx]);
    packed_assign_add(tmp, upcast(local));
  }
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

// This prototype deliberately stays narrow: it mirrors the FP32 Gemma RMSNorm
// reduction order for a [tokens, 5120] FP16 projection. Each CTA handles one
// row, retains the normal all-reduce peer order, and applies only its local
// residual after the peer reduction.
template <int Threads, int ngpus, typename ResidualT, typename WeightT>
__global__ void __launch_bounds__(Threads, 1)
    sm70_peer_reduce_gemma_rms_norm(
        RankData* _dp, RankSignals sg, Signal* self_sg,
        const ResidualT* __restrict__ residual,
        const WeightT* __restrict__ weight, half* __restrict__ normalized_out,
        float* __restrict__ residual_out, int rank, float epsilon) {
  using P = typename packed_t<half>::P;
  using A = typename packed_t<half>::A;
  constexpr int kPackedWidth = P::size;
  constexpr int kVarianceVectorWidth = 4;

  __shared__ float residual_values[kSm70GemmaRmsNormHiddenSize];
  __shared__ float inverse_rms;
  using BlockReduce = cub::BlockReduce<float, Threads>;
  __shared__ typename BlockReduce::TempStorage reduce_store;

  const int row = blockIdx.x;
  const int row_offset = row * kSm70GemmaRmsNormHiddenSize;
  const int packed_per_row = kSm70GemmaRmsNormHiddenSize / kPackedWidth;
  const int tid = threadIdx.x;
  auto dp = *_dp;

  barrier_at_start<ngpus>(sg, self_sg, rank);

  for (int packed_idx = tid; packed_idx < packed_per_row;
       packed_idx += blockDim.x) {
    const P reduced =
        packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0],
                                    row * packed_per_row + packed_idx);
    const int element_offset = row_offset + packed_idx * kPackedWidth;
#pragma unroll
    for (int i = 0; i < kPackedWidth; ++i) {
      const float value = __half2float(reduced.data[i]) +
                          sm70_gemma_rms_norm_to_float(
                              residual[element_offset + i]);
      residual_values[packed_idx * kPackedWidth + i] = value;
      residual_out[element_offset + i] = value;
    }
  }
  __syncthreads();

  // Match rms_norm_kernel<float, 4, 2>: each thread consumes vector indices
  // tid, tid + blockDim.x, ... and CUB reduces the resulting partial sums.
  float variance = 0.0f;
  for (int vector_idx = tid;
       vector_idx < kSm70GemmaRmsNormHiddenSize / kVarianceVectorWidth;
       vector_idx += blockDim.x) {
    const int element_offset = vector_idx * kVarianceVectorWidth;
#pragma unroll
    for (int i = 0; i < kVarianceVectorWidth; ++i) {
      const float value = residual_values[element_offset + i];
      variance += value * value;
    }
  }
  variance = BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);

  if (tid == 0) {
    inverse_rms = rsqrtf(variance / kSm70GemmaRmsNormHiddenSize + epsilon);
  }
  __syncthreads();

  for (int vector_idx = tid;
       vector_idx < kSm70GemmaRmsNormHiddenSize / kVarianceVectorWidth;
       vector_idx += blockDim.x) {
    const int element_offset = vector_idx * kVarianceVectorWidth;
#pragma unroll
    for (int i = 0; i < kVarianceVectorWidth; ++i) {
      const int column = element_offset + i;
      const float gemma_weight =
          sm70_gemma_rms_norm_to_float(weight[column]) + 1.0f;
      normalized_out[row_offset + column] = __float2half_rn(
          residual_values[column] * inverse_rms * gemma_weight);
    }
  }

  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 1) sm70_tile_runtime_reduce_kernel(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    const T* __restrict__ input, T* __restrict__ staging,
    T* __restrict__ result, int rank, int packed_size, int tile_packed_size,
    int tile_count, int compute_iters) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  const int tid = threadIdx.x;
  auto dp = *_dp;

  for (int tile_id = blockIdx.x; tile_id < tile_count; tile_id += gridDim.x) {
    const int begin = tile_id * tile_packed_size;
    const int end = min(begin + tile_packed_size, packed_size);

    unsigned spin = static_cast<unsigned>(tid);
    for (int idx = begin + tid; idx < end; idx += blockDim.x) {
      P value = reinterpret_cast<const P*>(input)[idx];
      for (int iter = 0; iter < compute_iters; ++iter) {
#if !defined(USE_ROCM)
        asm volatile("mov.u32 %0, %0;" : "+r"(spin));
#endif
      }
      reinterpret_cast<P*>(staging)[idx] = value;
    }

    __syncthreads();

    const FlagType flag = self_sg->_flag[tile_id] + 1;
    if (tid < ngpus) {
      auto peer_flag = &sg.signals[tid]->start[tile_id][rank];
      st_flag_sys_visible(peer_flag, flag);
    }

    if (tid < ngpus) {
      auto self_flag = &self_sg->start[tile_id][tid];
      while (ld_flag_sys_visible(self_flag) != flag);
    }

    __syncthreads();

    for (int idx = begin + tid; idx < end; idx += blockDim.x) {
      reinterpret_cast<P*>(result)[idx] =
          packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
    }

    __syncthreads();
    if (tid == 0) {
      self_sg->_flag[tile_id] = flag;
    }
  }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 1) sm70_tile_runtime_engine_kernel(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    const T* __restrict__ input, T* __restrict__ staging,
    T* __restrict__ result, int rank, int packed_size, int tile_packed_size,
    int tile_count, int producer_blocks, int reducer_blocks,
    int compute_iters) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  const int tid = threadIdx.x;
  auto dp = *_dp;

  if (blockIdx.x < producer_blocks) {
    for (int tile_id = blockIdx.x; tile_id < tile_count;
         tile_id += producer_blocks) {
      const int begin = tile_id * tile_packed_size;
      const int end = min(begin + tile_packed_size, packed_size);

      unsigned spin = static_cast<unsigned>(tid);
      for (int idx = begin + tid; idx < end; idx += blockDim.x) {
        P value = reinterpret_cast<const P*>(input)[idx];
        for (int iter = 0; iter < compute_iters; ++iter) {
#if !defined(USE_ROCM)
          asm volatile("mov.u32 %0, %0;" : "+r"(spin));
#endif
        }
        reinterpret_cast<P*>(staging)[idx] = value;
      }

      __syncthreads();

      const FlagType flag = self_sg->_flag[tile_id] + 1;
      if (tid < ngpus) {
        auto peer_flag = &sg.signals[tid]->start[tile_id][rank];
        st_flag_sys_visible(peer_flag, flag);
      }
    }
    return;
  }

  const int reducer_block = blockIdx.x - producer_blocks;
  for (int tile_id = reducer_block; tile_id < tile_count;
       tile_id += reducer_blocks) {
    const int begin = tile_id * tile_packed_size;
    const int end = min(begin + tile_packed_size, packed_size);
    const FlagType flag = self_sg->_flag[tile_id] + 1;

    if (tid < ngpus) {
      auto self_flag = &self_sg->start[tile_id][tid];
      while (ld_flag_sys_visible(self_flag) != flag);
    }

    __syncthreads();

    for (int idx = begin + tid; idx < end; idx += blockDim.x) {
      reinterpret_cast<P*>(result)[idx] =
          packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
    }

    __syncthreads();
    if (tid == 0) {
      self_sg->_flag[tile_id] = flag;
    }
  }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 1)
    sm70_tile_runtime_wait_reduce_kernel(
        RankData* _dp, RankSignals sg, Signal* self_sg,
        T* __restrict__ result, int rank, int packed_size,
        int tile_packed_size, int tile_count) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;

  const int tid = threadIdx.x;
  auto dp = *_dp;

  for (int tile_id = blockIdx.x; tile_id < tile_count; tile_id += gridDim.x) {
    const int begin = tile_id * tile_packed_size;
    const int end = min(begin + tile_packed_size, packed_size);
    const FlagType flag = self_sg->_flag[tile_id] + 1;

    if (tid < ngpus) {
      auto self_flag = &self_sg->start[tile_id][tid];
      while (ld_flag_sys_visible(self_flag) != flag);
    }

    __syncthreads();

    for (int idx = begin + tid; idx < end; idx += blockDim.x) {
      reinterpret_cast<P*>(result)[idx] =
          packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
    }

    __syncthreads();
    if (tid == 0) {
      self_sg->_flag[tile_id] = flag;
    }
  }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_sum2_1stage(
    RankData* _dp_a, RankData* _dp_b, RankSignals sg, Signal* self_sg,
    T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp_a = *_dp_a;
  auto dp_b = *_dp_b;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] =
        packed_reduce_sum2<P, ngpus, A>((const P**)&dp_a.ptrs[0],
                                        (const P**)&dp_b.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_sum2_2stage(
    RankData* _dp_a, RankData* _dp_b, RankSignals sg, Signal* self_sg,
    T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs_a[ngpus];
  const P* ptrs_b[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs_a[i] = (const P*)_dp_a->ptrs[target];
    ptrs_b[i] = (const P*)_dp_b->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // Stage 1 mirrors cross_device_reduce_2stage, but each rank first forms
  // its local input_a + input_b value before the cross-rank reduction.
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] =
        packed_reduce_sum2<P, ngpus, A>(ptrs_a, ptrs_b, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // Stage 2 allgather is intentionally identical to cross_device_reduce_2stage
  // so the final reduction order matches custom_all_reduce(input_a + input_b).
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

template <int ngpus>
__global__ void cross_device_top1_argmax(RankData* _dp, RankSignals sg,
                                         Signal* self_sg, int64_t* output,
                                         int rank) {
  barrier_at_start<ngpus>(sg, self_sg, rank);

  if (threadIdx.x == 0) {
    float best_value = -std::numeric_limits<float>::infinity();
    int64_t best_index = std::numeric_limits<int64_t>::max();

#pragma unroll
    for (int i = 0; i < ngpus; ++i) {
      const float* pair = reinterpret_cast<const float*>(_dp->ptrs[i]);
      const float value = pair[0];
      const int64_t index = static_cast<int64_t>(llrintf(pair[1]));
      if (value > best_value || (value == best_value && index < best_index)) {
        best_value = value;
        best_index = index;
      }
    }
    output[0] = best_index;
  }

  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  // Full NVLink or xGMI connection between GPUs.
  bool fully_connected_;

  RankSignals sg_;
  // Stores a map from a pointer to its peer pointers from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph
  // capture time. Therefore, during capture, we increment the rank data
  // pointer and use that as the argument to the kernel. The kernel arguments
  // are stored in graph_unreg_buffers_. The actual peer pointers will be
  // filled in at the memory pointed to by the pointers in
  // graph_unreg_buffers_ when the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second
   * section is for storing the intermediate results required by some
   * allreduce algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool fully_connected = true)
      : rank_(rank),
        world_size_(world_size),
        fully_connected_(fully_connected),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr, rangeStartAddrAttr,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  /**
   * Register already-shared IPC pointers.
   */
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  RankData* rank_data_for_buffer(cudaStream_t stream, void* buffer,
                                 const char* op_name) {
    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(buffer);
    } else {
      auto it = buffers_.find(buffer);
      if (it == buffers_.end()) {
        throw std::runtime_error(std::string(op_name) +
                                 " buffer address " +
                                 std::to_string(
                                     reinterpret_cast<uint64_t>(buffer)) +
                                 " is not registered!");
      }
      ptrs = it->second;
    }
    return ptrs;
  }

  // Note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the
  // remote possibility of different allocation patterns between ranks. For
  // example, rank 1 may get the same input address for the second allreduce,
  // but rank 2 got a different address. IPC handles have internal reference
  // counting mechanism so overhead should be small.
  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search.
   * Using 36 blocks give the best or close to the best runtime on the devices
   * I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also
   * only take a small amount of SMs. Not quite sure the underlying reason,
   * but my guess is that too many SMs will cause contention on NVLink bus.
   */
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size,
                 int threads = 512, int block_limit = defaultBlockLimit) {
    block_limit = custom_allreduce_block_limit(
        block_limit, world_size_, static_cast<size_t>(size) * sizeof(T));
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple "
          "of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    threads = sm70_tp4_m5_allreduce_threads(world_size_, fully_connected_,
                                            bytes);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

    // Check environment variable once
    const char* env_algo = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    bool force_1stage = false;
    bool force_2stage = false;
    if (env_algo != nullptr) {
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        force_1stage = true;
      } else if (std::strcmp(env_algo, "2stage") == 0 ||
                 std::strcmp(env_algo, "twoshot") == 0) {
        force_2stage = true;
      } else {
        throw std::runtime_error(
            "Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: " + std::string(env_algo) +
            ". Valid values: 1stage, oneshot, 2stage, twoshot");
      }
    }

#define KL(ngpus, name)                                                       \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size);
#define REDUCE_CASE(ngpus)                              \
  case ngpus: {                                         \
    if (force_1stage) {                                 \
      KL(ngpus, cross_device_reduce_1stage);            \
    } else if (force_2stage) {                          \
      KL(ngpus, cross_device_reduce_2stage);            \
    } else {                                            \
      if (world_size_ == 2) {                           \
        KL(ngpus, cross_device_reduce_1stage);          \
      } else if (fully_connected_) {                    \
        if ((world_size_ <= 4 && bytes < 512 * 1024) || \
            (world_size_ <= 8 && bytes < 256 * 1024)) { \
          KL(ngpus, cross_device_reduce_1stage);        \
        } else {                                        \
          KL(ngpus, cross_device_reduce_2stage);        \
        }                                               \
      }                                                 \
    }                                                   \
    break;                                              \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual "
            "num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  template <int ngpus, typename ResidualT, typename WeightT>
  void sm70_allreduce_gemma_rms_norm(cudaStream_t stream, half* input,
                                      const ResidualT* residual,
                                      const WeightT* weight,
                                      half* normalized_out,
                                      float* residual_out, int num_tokens,
                                      int hidden_size, float epsilon) {
    if (world_size_ != ngpus || !custom_allreduce_current_device_is_sm70()) {
      throw std::runtime_error("SM70 Gemma RMSNorm prototype requires TP" +
                               std::to_string(ngpus) + " on an SM70 device.");
    }
    if (hidden_size != kSm70GemmaRmsNormHiddenSize) {
      throw std::runtime_error(
          "SM70 Gemma RMSNorm prototype requires hidden_size=" +
          std::to_string(kSm70GemmaRmsNormHiddenSize) + ".");
    }
    constexpr int kMaxTokens = ngpus == 4 ? 64 : kMaxBlocks;
    if (num_tokens <= 0 || num_tokens > kMaxTokens) {
      throw std::runtime_error(
          "SM70 Gemma RMSNorm prototype supports tokens in [1, " +
          std::to_string(kMaxTokens) + "]. Got " +
          std::to_string(num_tokens) + ".");
    }

    RankData* ptrs = rank_data_for_buffer(
        stream, input, "SM70 Gemma RMSNorm prototype input");
    if constexpr (ngpus == 4) {
      // Keep the same 1024-thread CUB reduction topology as vLLM rms_norm.
      // packed_reduce preserves the cross_device_reduce_1stage<half, 4>
      // rank order before the FP32 residual add.
      sm70_peer_reduce_gemma_rms_norm<kSm70GemmaRmsNormThreads, ngpus,
                                       ResidualT, WeightT>
          <<<num_tokens, kSm70GemmaRmsNormThreads, 0, stream>>>(
              ptrs, sg_, self_sg_, residual, weight, normalized_out,
              residual_out, rank_, epsilon);
      return;
    }

    const int threads = sm70_gemma_rms_norm_threads();
#define VLLM_LAUNCH_SM70_GEMMA_RMS_NORM(THREADS)                         \
  sm70_peer_reduce_gemma_rms_norm<THREADS, ngpus, ResidualT, WeightT>     \
      <<<num_tokens, THREADS, 0, stream>>>(                               \
          ptrs, sg_, self_sg_, residual, weight, normalized_out,          \
          residual_out, rank_, epsilon)
    switch (threads) {
      case 256:
        VLLM_LAUNCH_SM70_GEMMA_RMS_NORM(256);
        break;
      case 512:
        VLLM_LAUNCH_SM70_GEMMA_RMS_NORM(512);
        break;
      default:
        VLLM_LAUNCH_SM70_GEMMA_RMS_NORM(1024);
        break;
    }
#undef VLLM_LAUNCH_SM70_GEMMA_RMS_NORM
  }

  template <typename T>
  void allreduce_sum2(cudaStream_t stream, T* input_a, T* input_b, T* output,
                      int size, int threads = 512,
                      int block_limit = defaultBlockLimit) {
    block_limit = custom_allreduce_block_limit(
        block_limit, world_size_, static_cast<size_t>(size) * sizeof(T));
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce sum2 currently requires input length to be "
          "multiple of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs_a;
    RankData* ptrs_b;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs_a = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input_a);
      ptrs_b = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input_b);
    } else {
      auto it_a = buffers_.find(input_a);
      auto it_b = buffers_.find(input_b);
      if (it_a == buffers_.end() || it_b == buffers_.end())
        throw std::runtime_error(
            "custom allreduce sum2 input address is not registered!");
      ptrs_a = it_a->second;
      ptrs_b = it_b->second;
    }

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

    const char* env_algo = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    bool force_1stage = false;
    bool force_2stage = false;
    if (env_algo != nullptr) {
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        force_1stage = true;
      } else if (std::strcmp(env_algo, "2stage") == 0 ||
                 std::strcmp(env_algo, "twoshot") == 0) {
        force_2stage = true;
      } else {
        throw std::runtime_error(
            "Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: " + std::string(env_algo) +
            ". Valid values: 1stage, oneshot, 2stage, twoshot");
      }
    }

#define SUM2_KL(ngpus, name)                                                  \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs_a, ptrs_b, sg_,         \
                                                 self_sg_, output, rank_,     \
                                                 size);
#define SUM2_CASE(ngpus)                              \
  case ngpus: {                                      \
    if (force_1stage) {                              \
      SUM2_KL(ngpus, cross_device_reduce_sum2_1stage); \
    } else if (force_2stage) {                       \
      SUM2_KL(ngpus, cross_device_reduce_sum2_2stage); \
    } else {                                         \
      if (world_size_ == 2) {                        \
        SUM2_KL(ngpus, cross_device_reduce_sum2_1stage); \
      } else if (fully_connected_) {                 \
        if ((world_size_ <= 4 && bytes < 512 * 1024) || \
            (world_size_ <= 8 && bytes < 256 * 1024)) { \
          SUM2_KL(ngpus, cross_device_reduce_sum2_1stage); \
        } else {                                     \
          SUM2_KL(ngpus, cross_device_reduce_sum2_2stage); \
        }                                            \
      }                                              \
    }                                                \
    break;                                           \
  }

    switch (world_size_) {
      SUM2_CASE(2)
      SUM2_CASE(4)
      SUM2_CASE(6)
      SUM2_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce sum2 only supports num gpus in (2,4,6,8). "
            "Actual num gpus = " +
            std::to_string(world_size_));
    }
#undef SUM2_CASE
#undef SUM2_KL
  }

  template <typename T>
  void tile_runtime_allreduce(cudaStream_t stream, const T* input, T* staging,
                              T* output, int size, int tile_numel,
                              int engine_blocks, int compute_iters) {
    if (world_size_ != 2) {
      throw std::runtime_error(
          "SM70 tile runtime prototype currently supports only TP2.");
    }

    auto pack = packed_t<T>::P::size;
    if (size % pack != 0 || tile_numel % pack != 0) {
      throw std::runtime_error(
          "SM70 tile runtime prototype requires size and tile_numel to be "
          "multiples of " +
          std::to_string(pack));
    }
    if (tile_numel <= 0) {
      throw std::runtime_error("tile_numel must be positive.");
    }

    const int packed_size = size / pack;
    const int tile_packed_size = tile_numel / pack;
    const int tile_count = (packed_size + tile_packed_size - 1) / tile_packed_size;
    if (tile_count <= 0 || tile_count > kMaxBlocks) {
      throw std::runtime_error(
          "SM70 tile runtime prototype supports tile_count in [1, " +
          std::to_string(kMaxBlocks) + "]. Got " +
          std::to_string(tile_count));
    }

    auto it = buffers_.find(staging);
    if (it == buffers_.end()) {
      throw std::runtime_error(
          "tile runtime staging buffer address " +
          std::to_string(reinterpret_cast<uint64_t>(staging)) +
          " is not registered!");
    }
    RankData* ptrs = it->second;

    const int threads = 256;
    int blocks = engine_blocks > 0 ? engine_blocks : tile_count;
    blocks = std::max(1, std::min(blocks, tile_count));
    compute_iters = std::max(0, compute_iters);

#define TILE_RUNTIME_CASE(ngpus)                                             \
  case ngpus: {                                                              \
    sm70_tile_runtime_reduce_kernel<T, ngpus>                                \
        <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, input, staging, \
                                         output, rank_, packed_size,          \
                                         tile_packed_size, tile_count,        \
                                         compute_iters);                     \
    break;                                                                   \
  }

    switch (world_size_) {
      TILE_RUNTIME_CASE(2)
      default:
        throw std::runtime_error(
            "SM70 tile runtime prototype only supports world_size=2.");
    }
#undef TILE_RUNTIME_CASE
  }

  template <typename T>
  void tile_runtime_allreduce_engine(cudaStream_t stream, const T* input,
                                     T* staging, T* output, int size,
                                     int tile_numel, int producer_blocks,
                                     int reducer_blocks, int compute_iters) {
    if (world_size_ != 2) {
      throw std::runtime_error(
          "SM70 tile runtime engine currently supports only TP2.");
    }

    auto pack = packed_t<T>::P::size;
    if (size % pack != 0 || tile_numel % pack != 0) {
      throw std::runtime_error(
          "SM70 tile runtime engine requires size and tile_numel to be "
          "multiples of " +
          std::to_string(pack));
    }
    if (tile_numel <= 0) {
      throw std::runtime_error("tile_numel must be positive.");
    }

    const int packed_size = size / pack;
    const int tile_packed_size = tile_numel / pack;
    const int tile_count =
        (packed_size + tile_packed_size - 1) / tile_packed_size;
    if (tile_count <= 0 || tile_count > kMaxBlocks) {
      throw std::runtime_error(
          "SM70 tile runtime engine supports tile_count in [1, " +
          std::to_string(kMaxBlocks) + "]. Got " +
          std::to_string(tile_count));
    }

    auto it = buffers_.find(staging);
    if (it == buffers_.end()) {
      throw std::runtime_error(
          "tile runtime staging buffer address " +
          std::to_string(reinterpret_cast<uint64_t>(staging)) +
          " is not registered!");
    }
    RankData* ptrs = it->second;

    producer_blocks =
        producer_blocks > 0 ? producer_blocks : std::min(tile_count, 4);
    reducer_blocks = reducer_blocks > 0 ? reducer_blocks : tile_count;
    producer_blocks = std::max(1, std::min(producer_blocks, tile_count));
    reducer_blocks = std::max(1, std::min(reducer_blocks, tile_count));
    compute_iters = std::max(0, compute_iters);

    const int threads = 256;
    const int blocks = producer_blocks + reducer_blocks;

#define TILE_RUNTIME_ENGINE_CASE(ngpus)                                      \
  case ngpus: {                                                              \
    sm70_tile_runtime_engine_kernel<T, ngpus>                                \
        <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, input, staging, \
                                         output, rank_, packed_size,          \
                                         tile_packed_size, tile_count,        \
                                         producer_blocks, reducer_blocks,     \
                                         compute_iters);                     \
    break;                                                                   \
  }

    switch (world_size_) {
      TILE_RUNTIME_ENGINE_CASE(2)
      default:
        throw std::runtime_error(
            "SM70 tile runtime engine only supports world_size=2.");
    }
#undef TILE_RUNTIME_ENGINE_CASE
  }

  template <typename T>
  void tile_runtime_wait_reduce(cudaStream_t stream, T* staging, T* output,
                                int size, int tile_numel,
                                int reducer_blocks) {
    if (world_size_ != 2) {
      throw std::runtime_error(
          "SM70 tile runtime wait-reduce currently supports only TP2.");
    }

    auto pack = packed_t<T>::P::size;
    if (size % pack != 0 || tile_numel % pack != 0) {
      throw std::runtime_error(
          "SM70 tile runtime wait-reduce requires size and tile_numel to be "
          "multiples of " +
          std::to_string(pack));
    }
    if (tile_numel <= 0) {
      throw std::runtime_error("tile_numel must be positive.");
    }

    const int packed_size = size / pack;
    const int tile_packed_size = tile_numel / pack;
    const int tile_count =
        (packed_size + tile_packed_size - 1) / tile_packed_size;
    if (tile_count <= 0 || tile_count > kMaxBlocks) {
      throw std::runtime_error(
          "SM70 tile runtime wait-reduce supports tile_count in [1, " +
          std::to_string(kMaxBlocks) + "]. Got " +
          std::to_string(tile_count));
    }

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(staging);
    } else {
      auto it = buffers_.find(staging);
      if (it == buffers_.end()) {
        throw std::runtime_error(
            "tile runtime wait-reduce staging address " +
            std::to_string(reinterpret_cast<uint64_t>(staging)) +
            " is not registered!");
      }
      ptrs = it->second;
    }

    reducer_blocks =
        reducer_blocks > 0 ? reducer_blocks : std::min(tile_count, 4);
    reducer_blocks = std::max(1, std::min(reducer_blocks, tile_count));

    constexpr int threads = 256;

#define TILE_RUNTIME_WAIT_REDUCE_CASE(ngpus)                                 \
  case ngpus: {                                                              \
    sm70_tile_runtime_wait_reduce_kernel<T, ngpus>                           \
        <<<reducer_blocks, threads, 0, stream>>>(                            \
            ptrs, sg_, self_sg_, output, rank_, packed_size,                 \
            tile_packed_size, tile_count);                                   \
    break;                                                                   \
  }

    switch (world_size_) {
      TILE_RUNTIME_WAIT_REDUCE_CASE(2)
      default:
        throw std::runtime_error(
            "SM70 tile runtime wait-reduce only supports world_size=2.");
    }
#undef TILE_RUNTIME_WAIT_REDUCE_CASE
  }

  void top1_argmax(cudaStream_t stream, float* input_pair, int64_t* output) {
    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input_pair);
    } else {
      auto it = buffers_.find(input_pair);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input_pair)) +
            " is not registered!");
      ptrs = it->second;
    }

#define TOP1_CASE(ngpus)                                   \
  case ngpus: {                                            \
    cross_device_top1_argmax<ngpus><<<1, 32, 0, stream>>>( \
        ptrs, sg_, self_sg_, output, rank_);               \
    break;                                                 \
  }

    switch (world_size_) {
      TOP1_CASE(2)
      TOP1_CASE(4)
      TOP1_CASE(6)
      TOP1_CASE(8)
      default:
        throw std::runtime_error(
            "custom top1 argmax only supports num gpus in (2,4,6,8). Actual "
            "num gpus = " +
            std::to_string(world_size_));
    }
#undef TOP1_CASE
  }

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};

/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and
 add a template instantiation:
 * template void vllm::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 half *, int, int, int);
*/
}  // namespace vllm
