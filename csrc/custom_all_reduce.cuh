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
constexpr size_t kSm70Tp8HierarchicalAllreduce8KiBBytes = 4096 * sizeof(half);
constexpr size_t kSm70Tp8HierarchicalAllreduce64KiBBytes =
    8 * 4096 * sizeof(half);
constexpr int kSm70Tp8HierarchicalPushWorldSize = 8;
constexpr int kSm70Tp8HierarchicalPushCliqueSize = 4;
constexpr int kSm70Tp8HierarchicalPushThreads = 128;
constexpr int kSm70Tp8HierarchicalPushMaxBlocks = 64;
constexpr int kSm70Tp8HierarchicalPushEpochs = 2;
constexpr size_t kSm70Tp8HierarchicalPushSignalBytes =
    ((kSm70Tp8HierarchicalPushMaxBlocks * sizeof(uint32_t) + 127) / 128) * 128;
constexpr size_t kSm70Tp8HierarchicalPushCliqueBytes =
    kSm70Tp8HierarchicalPushEpochs * kSm70Tp8HierarchicalPushCliqueSize *
    kSm70Tp8HierarchicalAllreduce64KiBBytes;
// Each FP32 clique partial occupies twice the bytes of its FP16 input.
constexpr size_t kSm70Tp8HierarchicalPushCrossBytes =
    kSm70Tp8HierarchicalPushEpochs * 2 *
    kSm70Tp8HierarchicalAllreduce64KiBBytes;
constexpr size_t kSm70Tp8HierarchicalPushBufferBytes =
    kSm70Tp8HierarchicalPushSignalBytes + kSm70Tp8HierarchicalPushCliqueBytes +
    kSm70Tp8HierarchicalPushCrossBytes;
constexpr size_t kSm70Tp4MtpVerifierBytesPerRequest = 5 * 2048 * sizeof(half);
constexpr int kSm70Tp8CompletionSignalSlotBase = 2;
constexpr int kSm70GemmaRmsNormHiddenSize = 5120;
constexpr int kSm70GemmaRmsNormThreads = 1024;
constexpr int kSm70LongPrefillSignalBlocks =
    sm70_tile_runtime::kMaxSignalBlocks;
constexpr int kSm70LongPrefillMaxTokensPerRank = 2048;
// DFlash2 verifies eight positions at once. These constants mirror SGLang's
// two-epoch one-shot push collective while keeping its storage separate from
// vLLM's pull-collective metadata and intermediate buffers.
constexpr int kSm70Tp4PushAllreduceWorldSize = 4;
constexpr int kSm70Tp4PushAllreduceBlocks = 80;
constexpr int kSm70Tp4PushAllreduceThreads = 128;
constexpr int kSm70Tp4PushAllreduceEpochs = 2;
constexpr uint16_t kSm70Tp4PushAllreduceSentinel = 0x7f7f;
constexpr int kSm70Tp4PushAllreduceSentinelByte = 0x7f;
constexpr size_t kSm70Tp4PushAllreduceBytes =
    8 * kSm70GemmaRmsNormHiddenSize * sizeof(half);
constexpr size_t kSm70Tp4PushAllreduce8KiBBytes = 4096 * sizeof(half);
constexpr size_t kSm70Tp4PushAllreduceQwen4ExpBytes = 2560 * sizeof(half);
constexpr size_t kSm70Tp4PushAllreduceQwen4ExpMtp5Bytes =
    5 * 2560 * sizeof(half);
constexpr size_t kSm70Tp4PushAllreduceQwen38M4Bytes = 4 * 2560 * sizeof(half);
constexpr size_t kSm70Tp4PushAllreduceQwen38M8Bytes = 8 * 2560 * sizeof(half);
constexpr size_t kSm70Tp4PushAllreduceSignalBytes =
    ((kSm70Tp4PushAllreduceBlocks * sizeof(uint32_t) + 127) / 128) * 128;
constexpr size_t kSm70Tp4PushAllreduceGenericBufferBytes =
    kSm70Tp4PushAllreduceSignalBytes + kSm70Tp4PushAllreduceEpochs *
                                           kSm70Tp4PushAllreduceWorldSize *
                                           kSm70Tp4PushAllreduceBytes;
// HC decode can overlap the ordinary MoE push collective on vLLM's auxiliary
// stream. Keep both its epoch words and payloads disjoint so an HC poll cannot
// observe or clear a concurrently running all-reduce packet. The ordinary
// collective layout above remains unchanged.
constexpr int kSm70Qwen38HcGatePushBlocks = 10;
constexpr int kSm70Qwen38HcDownEpochIndex = 0;
constexpr int kSm70Qwen38HcGateEpochIndexBase = 1;
constexpr size_t kSm70Qwen38HcPushSignalOffset =
    kSm70Tp4PushAllreduceGenericBufferBytes;
constexpr size_t kSm70Qwen38HcPushSignalBytes = 128;
constexpr size_t kSm70Qwen38HcDownPushBytes = 256;
constexpr size_t kSm70Qwen38HcGatePushBytes = 2560 * sizeof(half);
constexpr size_t kSm70Qwen38HcDownPushOffset =
    kSm70Qwen38HcPushSignalOffset + kSm70Qwen38HcPushSignalBytes;
constexpr size_t kSm70Qwen38HcGatePushOffset =
    kSm70Qwen38HcDownPushOffset + kSm70Tp4PushAllreduceEpochs *
                                      kSm70Tp4PushAllreduceWorldSize *
                                      kSm70Qwen38HcDownPushBytes;
constexpr size_t kSm70Qwen38HcUpFusedEpochOffset =
    kSm70Qwen38HcGatePushOffset + kSm70Tp4PushAllreduceEpochs *
                                      kSm70Tp4PushAllreduceWorldSize *
                                      kSm70Qwen38HcGatePushBytes;
// The fused up/mix/gather uses 160 independent generation counters and exact
// half-plus-tag packets, separate from both legacy HC and auxiliary MoE data.
constexpr int kSm70Qwen38HcUpFusedBlocks = 160;
constexpr size_t kSm70Qwen38HcUpFusedPacketOffset =
    kSm70Qwen38HcUpFusedEpochOffset +
    kSm70Qwen38HcUpFusedBlocks * sizeof(uint32_t);
constexpr size_t kSm70Tp4PushAllreduceBufferBytes =
    kSm70Qwen38HcUpFusedPacketOffset +
    kSm70Tp4PushAllreduceEpochs * 4 * 640 * sizeof(uint32_t);
static_assert(kSm70Qwen38HcGateEpochIndexBase + kSm70Qwen38HcGatePushBlocks <=
              kSm70Qwen38HcPushSignalBytes / sizeof(uint32_t));

inline int sm70_tp4_push_allreduce_blocks(size_t bytes,
                                          bool allow_generic = false) {
  // Experimental message-size admission, independent of model or batch shape.
  // Each thread handles one 16-byte pack. Use the smallest covering grid,
  // bounded by the existing persistent buffer, including for known payloads.
  const char* small =
      std::getenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES");
  if (allow_generic && small != nullptr && std::strcmp(small, "1") == 0 &&
      bytes > 0 && bytes <= kSm70Tp4PushAllreduceBytes && bytes % 16 == 0) {
    return static_cast<int>((bytes + kSm70Tp4PushAllreduceThreads * 16 - 1) /
                            (kSm70Tp4PushAllreduceThreads * 16));
  }
  if (bytes == kSm70Tp4PushAllreduceBytes) {
    return kSm70Tp4PushAllreduceBlocks;
  }
  if (bytes == kSm70Tp4PushAllreduce8KiBBytes) {
    return 4;
  }
  if (bytes == kSm70Tp4PushAllreduceQwen4ExpBytes) {
    return 3;
  }
  const char* batch = std::getenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH");
  const bool batch_enabled = batch == nullptr || std::strcmp(batch, "1") == 0;
  if (batch_enabled && (bytes == kSm70Tp4PushAllreduceQwen38M4Bytes ||
                        bytes == kSm70Tp4PushAllreduceQwen38M8Bytes)) {
    const char* blocks =
        std::getenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH_BLOCKS");
    if (blocks != nullptr) {
      const int parsed = std::atoi(blocks);
      const int min_blocks = (bytes + kSm70Tp4PushAllreduceThreads * 16 - 1) /
                             (kSm70Tp4PushAllreduceThreads * 16);
      // This push kernel handles one pack per thread, without a grid-stride
      // loop. An undersized launch silently leaves the output tail unwritten.
      if (parsed >= min_blocks && parsed <= kSm70Tp4PushAllreduceBlocks) {
        return parsed;
      }
    }
    return bytes == kSm70Tp4PushAllreduceQwen38M4Bytes ? 10 : 20;
  }
  const char* mtp5 = std::getenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5");
  return bytes == kSm70Tp4PushAllreduceQwen4ExpMtp5Bytes && mtp5 != nullptr &&
                 std::strcmp(mtp5, "1") == 0
             ? 13
             : 0;
}

inline int sm70_gemma_rms_norm_threads() {
  const char* raw = std::getenv("VLLM_SM70_TP2_AR_GEMMA_RMS_THREADS");
  if (raw == nullptr) return kSm70GemmaRmsNormThreads;
  const int threads = std::atoi(raw);
  return threads == 256 || threads == 512 || threads == 1024
             ? threads
             : kSm70GemmaRmsNormThreads;
}

inline int sm70_tp4_long_fused_norm_threads() {
  const char* raw = std::getenv("VLLM_SM70_TP4_LONG_FUSED_NORM_THREADS");
  if (raw == nullptr) return 512;
  const int threads = std::atoi(raw);
  return threads == 256 || threads == 512 || threads == 1024 ? threads : 512;
}

inline int sm70_tp4_long_fused_norm_blocks() {
  const char* raw = std::getenv("VLLM_SM70_TP4_LONG_FUSED_NORM_BLOCKS");
  if (raw == nullptr) return 80;
  const int blocks = std::atoi(raw);
  return blocks >= 1 && blocks <= kSm70LongPrefillSignalBlocks ? blocks : 80;
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

inline bool sm70_tp8_hierarchical_custom_ar_enabled(int world_size,
                                                    bool fully_connected) {
  const char* raw = std::getenv("VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR");
  return raw != nullptr && std::atoi(raw) != 0 && world_size == 8 &&
         !fully_connected && custom_allreduce_current_device_is_sm70();
}

inline bool sm70_tp8_hierarchical_peer(int rank, int peer) {
  return rank / 4 == peer / 4 || rank + 4 == peer || peer + 4 == rank;
}

inline bool sm70_tp8_hierarchical_allreduce_size(size_t bytes) {
  return bytes == kSm70Tp8HierarchicalAllreduce8KiBBytes ||
         bytes == kSm70Tp8HierarchicalAllreduce64KiBBytes;
}

inline int sm70_tp8_hierarchical_push_blocks(size_t bytes) {
  const int default_blocks =
      bytes == kSm70Tp8HierarchicalAllreduce64KiBBytes ? 16 : 4;
  const char* raw = std::getenv("VLLM_SM70_TP8_HIERARCHICAL_PUSH_BLOCKS");
  if (raw == nullptr || raw[0] == '\0') return default_blocks;
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed < 1 ||
      parsed > kSm70Tp8HierarchicalPushMaxBlocks) {
    throw std::runtime_error(
        "Invalid VLLM_SM70_TP8_HIERARCHICAL_PUSH_BLOCKS: " + std::string(raw) +
        ". Expected an integer in [1, " +
        std::to_string(kSm70Tp8HierarchicalPushMaxBlocks) + "].");
  }
  return static_cast<int>(parsed);
}

inline int custom_allreduce_block_limit(int default_limit, int world_size,
                                        bool fully_connected, size_t bytes,
                                        bool tune_sm70_tp4_mtp) {
  const char* raw = std::getenv("VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT");
  if (raw == nullptr || raw[0] == '\0') {
    if (world_size == 2 && bytes <= kSm70Tp2SmallAllreduceBytes &&
        custom_allreduce_current_device_is_sm70()) {
      return 1;
    }
    const char* tune_raw = std::getenv("VLLM_SM70_TP4_MTP_AR_BLOCK_TUNING");
    const bool tuning_enabled =
        tune_raw != nullptr && tune_raw[0] != '\0' && std::atoi(tune_raw) != 0;
    if (tune_sm70_tp4_mtp && tuning_enabled &&
        default_limit == defaultBlockLimit && world_size == 4 &&
        fully_connected && custom_allreduce_current_device_is_sm70()) {
      if (bytes == kSm70Tp4MtpVerifierBytesPerRequest) {
        return 1;
      }
      if (bytes == 8 * kSm70Tp4MtpVerifierBytesPerRequest ||
          bytes == 12 * kSm70Tp4MtpVerifierBytesPerRequest ||
          bytes == 16 * kSm70Tp4MtpVerifierBytesPerRequest) {
        return 8;
      }
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

inline int sm70_tp4_m5_allreduce_threads(int world_size, bool fully_connected,
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

inline bool sm70_tp4_small_allreduce_pack32(int world_size,
                                            bool fully_connected,
                                            size_t bytes) {
  const char* raw = std::getenv("VLLM_SM70_TP4_SMALL_AR_PACK32");
  return raw != nullptr && std::atoi(raw) != 0 && world_size == 4 &&
         fully_connected && bytes == 5120 * sizeof(half) &&
         custom_allreduce_current_device_is_sm70();
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
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
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

static DINLINE void membar_sys() { asm volatile("membar.sys;" ::: "memory"); }

// The TP4 pack32 route waits with volatile peer loads, then executes one
// system fence after the matching flag is visible. This preserves the input
// visibility contract without putting a system fence in every poll iteration.
template <int ngpus>
DINLINE void sm70_pack32_barrier_at_start(const RankSignals& sg,
                                          Signal* self_sg, int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    const unsigned int peer = threadIdx.x;
    auto peer_counter_ptr = &sg.signals[peer]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][peer];
    st_flag_sys_visible(peer_counter_ptr, flag);
    while (ld_flag_volatile(self_counter_ptr) != flag);
    membar_sys();
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus>
DINLINE void sm70_pack32_barrier_at_end(const RankSignals& sg, Signal* self_sg,
                                        int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    const unsigned int peer = threadIdx.x;
    auto peer_counter_ptr = &sg.signals[peer]->end[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][peer];
    st_flag_volatile(peer_counter_ptr, flag);
    while (ld_flag_volatile(self_counter_ptr) != flag);
  }
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// This function is meant to be used as the first synchronization in the all
// reduce kernel. Publish the peer flag after a system fence, poll it with a
// volatile load, then execute one system fence after the matching value is
// visible. The post-poll fence preserves input visibility without putting a
// system fence in every spin-loop iteration.
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
    while (ld_flag_volatile(self_counter_ptr) != flag);
    membar_sys();
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
DINLINE void sm70_pack32_barrier_at_start(const RankSignals& sg,
                                          Signal* self_sg, int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    const unsigned int peer = threadIdx.x;
    __scoped_atomic_store_n(&sg.signals[peer]->start[blockIdx.x][rank], flag,
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][peer],
                                  __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE) < flag);
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus>
DINLINE void sm70_pack32_barrier_at_end(const RankSignals& sg, Signal* self_sg,
                                        int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    const unsigned int peer = threadIdx.x;
    __scoped_atomic_store_n(&sg.signals[peer]->end[blockIdx.x][rank], flag,
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_sg->end[blockIdx.x][peer],
                                  __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE) < flag);
  }
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

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

// Adapted from SGLang-V100's SM70-capable one-shot push collective at
// haohervchb/sglang-V100@845b9fdf7a7e. SGLang uses positive zero as its empty
// slot sentinel. This variant preinitializes slots with a reserved FP16 NaN
// payload instead, preserving every finite payload bit, including signed zero.
DINLINE void sm70_push_escape_sentinel(half& value) {
  auto* bits = reinterpret_cast<uint16_t*>(&value);
  if (*bits == kSm70Tp4PushAllreduceSentinel) *bits = 0x7e00u;
}

DINLINE bool sm70_push_is_sentinel(const half& value) {
  const auto* bits = reinterpret_cast<const uint16_t*>(&value);
  return *bits == kSm70Tp4PushAllreduceSentinel;
}

template <typename P>
DINLINE void sm70_push_load_volatile_16b(P& value, const void* address,
                                         int offset) {
  static_assert(alignof(P) == 16 && sizeof(P) == 16);
  const auto* source = reinterpret_cast<const P*>(address) + offset;
  uint4 bits;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
               : "l"(source));
  value = *reinterpret_cast<const P*>(&bits);
}

template <typename P>
DINLINE void sm70_push_store_volatile_16b(const P& value, void* address,
                                          int offset) {
  static_assert(alignof(P) == 16 && sizeof(P) == 16);
  const uint4 bits = *reinterpret_cast<const uint4*>(&value);
  auto* destination = reinterpret_cast<P*>(address) + offset;
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :
               : "r"(bits.x), "r"(bits.y), "r"(bits.z), "r"(bits.w),
                 "l"(destination));
}

template <typename P, int ngpus, typename A>
DINLINE P sm70_push_reduce(P (&values)[ngpus]) {
  A accumulator = upcast(values[0]);
#pragma unroll
  for (int rank = 1; rank < ngpus; ++rank) {
    packed_assign_add(accumulator, upcast(values[rank]));
  }
  return downcast<P>(accumulator);
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

template <int ngpus>
__global__ void __launch_bounds__(512, 1)
    sm70_cross_device_reduce_1stage_pack32(RankData* _dp, RankSignals sg,
                                           Signal* self_sg,
                                           half* __restrict__ result, int rank,
                                           int size) {
  using P = array_t<half, 16>;
  using A = array_t<float, 16>;
  auto dp = *_dp;
  sm70_pack32_barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    reinterpret_cast<P*>(result)[idx] =
        packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  sm70_pack32_barrier_at_end<ngpus>(sg, self_sg, rank);
}

template <int ngpus>
__global__ void __launch_bounds__(1024, 1)
    sm70_cross_device_reduce_1stage_push(RankData push_buffers,
                                         const half* __restrict__ input,
                                         half* __restrict__ output, int rank,
                                         int packed_size) {
  static_assert(ngpus == kSm70Tp4PushAllreduceWorldSize);
  using P = typename packed_t<half>::P;
  using A = typename packed_t<half>::A;

  auto* local_storage =
      const_cast<char*>(reinterpret_cast<const char*>(push_buffers.ptrs[rank]));
  auto* local_epochs = reinterpret_cast<uint32_t*>(local_storage);
  const uint32_t epoch = local_epochs[blockIdx.x];
  constexpr int packed_stride = kSm70Tp4PushAllreduceBytes / sizeof(P);
  const int epoch_offset = epoch * ngpus * packed_stride;
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (offset < packed_size) {
    P value = reinterpret_cast<const P*>(input)[offset];
#pragma unroll
    for (int element = 0; element < P::size; ++element) {
      sm70_push_escape_sentinel(value.data[element]);
    }

#pragma unroll
    for (int destination_rank = 0; destination_rank < ngpus;
         ++destination_rank) {
      auto* destination_base = const_cast<char*>(
          reinterpret_cast<const char*>(push_buffers.ptrs[destination_rank]));
      void* destination = destination_base + kSm70Tp4PushAllreduceSignalBytes +
                          (epoch_offset + rank * packed_stride) * sizeof(P);
      sm70_push_store_volatile_16b(value, destination, offset);
    }

    P peer_values[ngpus];
    while (true) {
      bool has_empty_slot = false;
#pragma unroll
      for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
        const void* source =
            local_storage + kSm70Tp4PushAllreduceSignalBytes +
            (epoch_offset + source_rank * packed_stride) * sizeof(P);
        sm70_push_load_volatile_16b(peer_values[source_rank], source, offset);
#pragma unroll
        for (int element = 0; element < P::size; ++element) {
          has_empty_slot |=
              sm70_push_is_sentinel(peer_values[source_rank].data[element]);
        }
      }
      if (!has_empty_slot) break;
    }

    reinterpret_cast<P*>(output)[offset] =
        sm70_push_reduce<P, ngpus, A>(peer_values);

    P empty;
#pragma unroll
    for (int element = 0; element < P::size; ++element) {
      *reinterpret_cast<uint16_t*>(&empty.data[element]) =
          kSm70Tp4PushAllreduceSentinel;
    }
#pragma unroll
    for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
      void* source = local_storage + kSm70Tp4PushAllreduceSignalBytes +
                     (epoch_offset + source_rank * packed_stride) * sizeof(P);
      sm70_push_store_volatile_16b(empty, source, offset);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    local_epochs[blockIdx.x] = (epoch + 1) % kSm70Tp4PushAllreduceEpochs;
  }
}

template <int ngpus>
__global__ void __launch_bounds__(1024, 1)
    sm70_cross_device_reduce_sum2_1stage_push(RankData push_buffers,
                                              const half* __restrict__ input_a,
                                              const half* __restrict__ input_b,
                                              half* __restrict__ output,
                                              int rank, int packed_size) {
  static_assert(ngpus == kSm70Tp4PushAllreduceWorldSize);
  using P = typename packed_t<half>::P;
  using A = typename packed_t<half>::A;

  auto* local_storage =
      const_cast<char*>(reinterpret_cast<const char*>(push_buffers.ptrs[rank]));
  auto* local_epochs = reinterpret_cast<uint32_t*>(local_storage);
  const uint32_t epoch = local_epochs[blockIdx.x];
  constexpr int packed_stride = kSm70Tp4PushAllreduceBytes / sizeof(P);
  const int epoch_offset = epoch * ngpus * packed_stride;
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (offset < packed_size) {
    P value_a = reinterpret_cast<const P*>(input_a)[offset];
    const P value_b = reinterpret_cast<const P*>(input_b)[offset];
    packed_assign_add(value_a, value_b);
#pragma unroll
    for (int element = 0; element < P::size; ++element) {
      sm70_push_escape_sentinel(value_a.data[element]);
    }

#pragma unroll
    for (int destination_rank = 0; destination_rank < ngpus;
         ++destination_rank) {
      auto* destination_base = const_cast<char*>(
          reinterpret_cast<const char*>(push_buffers.ptrs[destination_rank]));
      void* destination = destination_base + kSm70Tp4PushAllreduceSignalBytes +
                          (epoch_offset + rank * packed_stride) * sizeof(P);
      sm70_push_store_volatile_16b(value_a, destination, offset);
    }

    P peer_values[ngpus];
    while (true) {
      bool has_empty_slot = false;
#pragma unroll
      for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
        const void* source =
            local_storage + kSm70Tp4PushAllreduceSignalBytes +
            (epoch_offset + source_rank * packed_stride) * sizeof(P);
        sm70_push_load_volatile_16b(peer_values[source_rank], source, offset);
#pragma unroll
        for (int element = 0; element < P::size; ++element) {
          has_empty_slot |=
              sm70_push_is_sentinel(peer_values[source_rank].data[element]);
        }
      }
      if (!has_empty_slot) break;
    }

    reinterpret_cast<P*>(output)[offset] =
        sm70_push_reduce<P, ngpus, A>(peer_values);

    P empty;
#pragma unroll
    for (int element = 0; element < P::size; ++element) {
      *reinterpret_cast<uint16_t*>(&empty.data[element]) =
          kSm70Tp4PushAllreduceSentinel;
    }
#pragma unroll
    for (int source_rank = 0; source_rank < ngpus; ++source_rank) {
      void* source = local_storage + kSm70Tp4PushAllreduceSignalBytes +
                     (epoch_offset + source_rank * packed_stride) * sizeof(P);
      sm70_push_store_volatile_16b(empty, source, offset);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    local_epochs[blockIdx.x] = (epoch + 1) % kSm70Tp4PushAllreduceEpochs;
  }
}

// Fine-grained two-level push reduction for the 4+4 NVLink topology used by
// the TP8 GLM verifier. Every packed lane first pushes its FP16 input within
// its four-GPU clique, forms an ordered FP32 clique partial, and exchanges the
// partial only with the directly connected rank in the other clique. The
// resulting tree is exactly (r0+r1+r2+r3) + (r4+r5+r6+r7).
template <bool SumTwoInputs>
static __global__ void __launch_bounds__(kSm70Tp8HierarchicalPushThreads, 1)
    sm70_tp8_hierarchical_reduce_push(RankData push_buffers,
                                      const half* __restrict__ input_a,
                                      const half* __restrict__ input_b,
                                      half* __restrict__ output, int rank,
                                      int packed_size) {
  using P = typename packed_t<half>::P;
  using A = typename packed_t<half>::A;
  using F = array_t<float, 4>;

  constexpr int packed_stride =
      kSm70Tp8HierarchicalAllreduce64KiBBytes / sizeof(P);
  constexpr uint32_t fp32_sentinel = 0x7f7f7f7f;
  const int clique_base = rank < 4 ? 0 : 4;
  const int source_slot = rank - clique_base;
  const int pair_rank = rank < 4 ? rank + 4 : rank - 4;
  auto* local_storage =
      const_cast<char*>(reinterpret_cast<const char*>(push_buffers.ptrs[rank]));
  auto* local_epochs = reinterpret_cast<uint32_t*>(local_storage);
  const uint32_t epoch = local_epochs[blockIdx.x];
  const int clique_epoch_offset =
      epoch * kSm70Tp8HierarchicalPushCliqueSize * packed_stride;
  const int cross_epoch_offset = epoch * 2 * packed_stride;
  char* const local_clique =
      local_storage + kSm70Tp8HierarchicalPushSignalBytes;
  char* const local_cross = local_clique + kSm70Tp8HierarchicalPushCliqueBytes;

  for (int offset = blockIdx.x * blockDim.x + threadIdx.x; offset < packed_size;
       offset += gridDim.x * blockDim.x) {
    P value = reinterpret_cast<const P*>(input_a)[offset];
    if constexpr (SumTwoInputs) {
      packed_assign_add(value, reinterpret_cast<const P*>(input_b)[offset]);
    }
#pragma unroll
    for (int element = 0; element < P::size; ++element) {
      sm70_push_escape_sentinel(value.data[element]);
    }

#pragma unroll
    for (int destination_slot = 0;
         destination_slot < kSm70Tp8HierarchicalPushCliqueSize;
         ++destination_slot) {
      const int destination_rank = clique_base + destination_slot;
      auto* destination_storage = const_cast<char*>(
          reinterpret_cast<const char*>(push_buffers.ptrs[destination_rank]));
      void* destination =
          destination_storage + kSm70Tp8HierarchicalPushSignalBytes +
          (clique_epoch_offset + source_slot * packed_stride) * sizeof(P);
      sm70_push_store_volatile_16b(value, destination, offset);
    }

    P peer_values[kSm70Tp8HierarchicalPushCliqueSize];
    while (true) {
      bool has_empty_slot = false;
#pragma unroll
      for (int peer_slot = 0; peer_slot < kSm70Tp8HierarchicalPushCliqueSize;
           ++peer_slot) {
        const void* source =
            local_clique +
            (clique_epoch_offset + peer_slot * packed_stride) * sizeof(P);
        sm70_push_load_volatile_16b(peer_values[peer_slot], source, offset);
#pragma unroll
        for (int element = 0; element < P::size; ++element) {
          has_empty_slot |=
              sm70_push_is_sentinel(peer_values[peer_slot].data[element]);
        }
      }
      if (!has_empty_slot) break;
    }

    A partial = upcast(peer_values[0]);
#pragma unroll
    for (int peer_slot = 1; peer_slot < kSm70Tp8HierarchicalPushCliqueSize;
         ++peer_slot) {
      packed_assign_add(partial, upcast(peer_values[peer_slot]));
    }
#pragma unroll
    for (int element = 0; element < A::size; ++element) {
      auto* bits = reinterpret_cast<uint32_t*>(&partial.data[element]);
      if (*bits == fp32_sentinel) *bits = 0x7fc00000;
    }

    auto* pair_storage = const_cast<char*>(
        reinterpret_cast<const char*>(push_buffers.ptrs[pair_rank]));
    void* pair_cross = pair_storage + kSm70Tp8HierarchicalPushSignalBytes +
                       kSm70Tp8HierarchicalPushCliqueBytes;
    const auto* partial_words = reinterpret_cast<const F*>(&partial);
    sm70_push_store_volatile_16b(partial_words[0], pair_cross,
                                 cross_epoch_offset + 2 * offset);
    sm70_push_store_volatile_16b(partial_words[1], pair_cross,
                                 cross_epoch_offset + 2 * offset + 1);

    F pair_words[2];
    while (true) {
      bool has_empty_slot = false;
      sm70_push_load_volatile_16b(pair_words[0], local_cross,
                                  cross_epoch_offset + 2 * offset);
      sm70_push_load_volatile_16b(pair_words[1], local_cross,
                                  cross_epoch_offset + 2 * offset + 1);
#pragma unroll
      for (int word = 0; word < 2; ++word) {
#pragma unroll
        for (int element = 0; element < F::size; ++element) {
          has_empty_slot |=
              *reinterpret_cast<const uint32_t*>(
                  &pair_words[word].data[element]) == fp32_sentinel;
        }
      }
      if (!has_empty_slot) break;
    }

    A pair_partial;
#pragma unroll
    for (int element = 0; element < F::size; ++element) {
      pair_partial.data[element] = pair_words[0].data[element];
      pair_partial.data[element + F::size] = pair_words[1].data[element];
    }
    A total = rank < 4 ? partial : pair_partial;
    packed_assign_add(total, rank < 4 ? pair_partial : partial);
    reinterpret_cast<P*>(output)[offset] = downcast<P>(total);

    P empty_input;
#pragma unroll
    for (int element = 0; element < P::size; ++element) {
      *reinterpret_cast<uint16_t*>(&empty_input.data[element]) =
          kSm70Tp4PushAllreduceSentinel;
    }
#pragma unroll
    for (int peer_slot = 0; peer_slot < kSm70Tp8HierarchicalPushCliqueSize;
         ++peer_slot) {
      void* source =
          local_clique +
          (clique_epoch_offset + peer_slot * packed_stride) * sizeof(P);
      sm70_push_store_volatile_16b(empty_input, source, offset);
    }

    F empty_cross;
#pragma unroll
    for (int element = 0; element < F::size; ++element) {
      *reinterpret_cast<uint32_t*>(&empty_cross.data[element]) = fp32_sentinel;
    }
    sm70_push_store_volatile_16b(empty_cross, local_cross,
                                 cross_epoch_offset + 2 * offset);
    sm70_push_store_volatile_16b(empty_cross, local_cross,
                                 cross_epoch_offset + 2 * offset + 1);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    local_epochs[blockIdx.x] = (epoch + 1) % kSm70Tp8HierarchicalPushEpochs;
  }
}

// This prototype deliberately stays narrow: it mirrors the FP32 Gemma RMSNorm
// reduction order for a [tokens, 5120] FP16 projection. Each CTA handles one
// row, retains the normal all-reduce peer order, and applies only its local
// residual after the peer reduction.
template <int Threads, int ngpus, typename ResidualT, typename WeightT>
__global__ void __launch_bounds__(Threads, 1)
    sm70_peer_reduce_gemma_rms_norm(RankData* _dp, RankSignals sg,
                                    Signal* self_sg,
                                    const ResidualT* __restrict__ residual,
                                    const WeightT* __restrict__ weight,
                                    half* __restrict__ normalized_out,
                                    float* __restrict__ residual_out, int rank,
                                    float epsilon) {
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
    const P reduced = packed_reduce<P, ngpus, A>(
        (const P**)&dp.ptrs[0], row * packed_per_row + packed_idx);
    const int element_offset = row_offset + packed_idx * kPackedWidth;
#pragma unroll
    for (int i = 0; i < kPackedWidth; ++i) {
      const float value =
          __half2float(reduced.data[i]) +
          sm70_gemma_rms_norm_to_float(residual[element_offset + i]);
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
      normalized_out[row_offset + column] =
          __float2half_rn(residual_values[column] * inverse_rms * gemma_weight);
    }
  }

  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

// Benchmark-only SM70 counterpart of NCCL's fused LSA RMSNorm example:
// reduce-scatter token rows, apply the mixed-dtype Gemma RMSNorm locally, then
// all-gather by writing the owned normalized rows into every peer output.
// Each rank launches one CTA per owned token and therefore moves the same
// asymptotic peer traffic as ring reduce-scatter + all-gather, without
// materializing an intermediate all-reduce tensor.
template <int Threads, int ngpus, typename ResidualT, typename WeightT>
__global__ void __launch_bounds__(Threads, 1)
    sm70_peer_reduce_scatter_gemma_rms_norm_all_gather(
        RankData* _input_dp, RankData* _output_dp, RankSignals sg,
        Signal* self_sg, const ResidualT* __restrict__ residual,
        const WeightT* __restrict__ weight, float* __restrict__ residual_out,
        int rank, int num_tokens, float epsilon) {
  static_assert(ngpus == 4);
  using H4 = array_t<half, 4>;
  constexpr int kVarianceVectorWidth = 4;
  constexpr int kVectorsPerRow =
      kSm70GemmaRmsNormHiddenSize / kVarianceVectorWidth;
  constexpr int kVectorsPerThread = (kVectorsPerRow + Threads - 1) / Threads;

  __shared__ float inverse_rms;
  // Preserve the accepted local GemmaNorm reduction topology. The valid-item
  // count below limits participation to the threads in this launch.
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;

  const int tokens_per_rank = num_tokens / ngpus;
  const int tid = threadIdx.x;
  auto input_dp = *_input_dp;
  auto output_dp = *_output_dp;

  barrier_at_start<ngpus>(sg, self_sg, rank);

  for (int local_row = blockIdx.x; local_row < tokens_per_rank;
       local_row += gridDim.x) {
    const int row = rank * tokens_per_rank + local_row;
    const int vector_row_offset = row * kVectorsPerRow;
    float4 row_values[kVectorsPerThread];
    float variance = 0.0f;
#pragma unroll
    for (int iter = 0; iter < kVectorsPerThread; ++iter) {
      const int vector_idx = tid + iter * Threads;
      if (vector_idx >= kVectorsPerRow) continue;
      const int global_vector_idx = vector_row_offset + vector_idx;
      H4 reduced =
          reinterpret_cast<const H4*>(input_dp.ptrs[0])[global_vector_idx];
#pragma unroll
      for (int peer = 1; peer < ngpus; ++peer) {
        const H4 peer_value =
            reinterpret_cast<const H4*>(input_dp.ptrs[peer])[global_vector_idx];
#pragma unroll
        for (int i = 0; i < kVarianceVectorWidth; ++i) {
          // NCCL's f16 Sum specialization uses __hadd/__hadd2 at every ring
          // step. Preserve that FP16 rounding contract instead of accumulating
          // all four inputs in FP32 and rounding only once.
          reduced.data[i] = __hadd(reduced.data[i], peer_value.data[i]);
        }
      }
      const float4 residual_value =
          reinterpret_cast<const float4*>(residual)[global_vector_idx];
      float4 value;
      value.x = __half2float(reduced.data[0]) + residual_value.x;
      value.y = __half2float(reduced.data[1]) + residual_value.y;
      value.z = __half2float(reduced.data[2]) + residual_value.z;
      value.w = __half2float(reduced.data[3]) + residual_value.w;
      row_values[iter] = value;
      // Keep the public [M, H] residual shape so the compiled model graph does
      // not change shape.  Only this rank's persistent token shard is valid;
      // the next fused boundary reads the same shard and ignores other rows.
      reinterpret_cast<float4*>(residual_out)[global_vector_idx] = value;
      // Match the accepted local GemmaNorm exactly: each thread accumulates
      // vector indices tid, tid + Threads, ... in x/y/z/w order before CUB.
      variance += value.x * value.x;
      variance += value.y * value.y;
      variance += value.z * value.z;
      variance += value.w * value.w;
    }
    variance =
        BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
    if (tid == 0) {
      inverse_rms = rsqrtf(variance / kSm70GemmaRmsNormHiddenSize + epsilon);
    }
    __syncthreads();

    const float scale = inverse_rms;
#pragma unroll
    for (int iter = 0; iter < kVectorsPerThread; ++iter) {
      const int vector_idx = tid + iter * Threads;
      if (vector_idx >= kVectorsPerRow) continue;
      const int column = vector_idx * kVarianceVectorWidth;
      const float4 value = row_values[iter];
      H4 normalized_pack;
      normalized_pack.data[0] = __float2half_rn(
          value.x * scale *
          (sm70_gemma_rms_norm_to_float(weight[column]) + 1.0f));
      normalized_pack.data[1] = __float2half_rn(
          value.y * scale *
          (sm70_gemma_rms_norm_to_float(weight[column + 1]) + 1.0f));
      normalized_pack.data[2] = __float2half_rn(
          value.z * scale *
          (sm70_gemma_rms_norm_to_float(weight[column + 2]) + 1.0f));
      normalized_pack.data[3] = __float2half_rn(
          value.w * scale *
          (sm70_gemma_rms_norm_to_float(weight[column + 3]) + 1.0f));
#pragma unroll
      for (int peer = 0; peer < ngpus; ++peer) {
        auto* peer_output =
            reinterpret_cast<H4*>(const_cast<void*>(output_dp.ptrs[peer]));
        peer_output[vector_row_offset + vector_idx] = normalized_pack;
      }
    }
    // CUB's temporary storage is reused by the next persistent row.
    __syncthreads();
  }

  // The normalized rows are written directly into peer IPC buffers.  Unlike
  // an ordinary final all-reduce barrier, this barrier must publish those
  // remote stores before a peer starts its local output copy.
  barrier_at_end<ngpus>(sg, self_sg, rank);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 1) sm70_tile_runtime_reduce_kernel(
    RankData* _dp, RankSignals sg, Signal* self_sg, const T* __restrict__ input,
    T* __restrict__ staging, T* __restrict__ result, int rank, int packed_size,
    int tile_packed_size, int tile_count, int compute_iters) {
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
    RankData* _dp, RankSignals sg, Signal* self_sg, const T* __restrict__ input,
    T* __restrict__ staging, T* __restrict__ result, int rank, int packed_size,
    int tile_packed_size, int tile_count, int producer_blocks,
    int reducer_blocks, int compute_iters) {
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
    sm70_tile_runtime_wait_reduce_kernel(RankData* _dp, RankSignals sg,
                                         Signal* self_sg,
                                         T* __restrict__ result, int rank,
                                         int packed_size, int tile_packed_size,
                                         int tile_count) {
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
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_sum2_1stage(RankData* _dp_a, RankData* _dp_b,
                                    RankSignals sg, Signal* self_sg,
                                    T* __restrict__ result, int rank,
                                    int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp_a = *_dp_a;
  auto dp_b = *_dp_b;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce_sum2<P, ngpus, A>(
        (const P**)&dp_a.ptrs[0], (const P**)&dp_b.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

DINLINE FlagType sm70_tp8_clique_barrier(const RankSignals& sg, Signal* self_sg,
                                         int rank, int signal_slot) {
  const int tid = threadIdx.x;
  const int clique_base = rank < 4 ? 0 : 4;
  const FlagType flag = self_sg->_flag[0] + 1;
  if (tid < 4) {
    const int peer = clique_base + tid;
    st_flag_sys_visible(&sg.signals[peer]->start[signal_slot][rank], flag);
    while (ld_flag_sys_visible(&self_sg->start[signal_slot][peer]) != flag);
  }
  __syncthreads();
  if (tid == 0) self_sg->_flag[0] = flag;
  return flag;
}

static __global__ void __launch_bounds__(512, 1)
    sm70_tp8_hierarchical_reduce(RankData* _dp, RankSignals sg, Signal* self_sg,
                                 half* __restrict__ result, int rank,
                                 int packed_size) {
  using P = typename packed_t<half>::P;
  using A = typename packed_t<half>::A;

  const int tid = threadIdx.x;
  const int clique_base = rank < 4 ? 0 : 4;
  const int pair_rank = rank < 4 ? rank + 4 : rank - 4;
  auto dp = *_dp;

  // The counter advances twice per call. Alternate signal/data slots so one
  // clique may safely run one round ahead of its paired clique.
  const int partial_slot = (self_sg->_flag[0] >> 1) & 1;
  const FlagType clique_flag =
      sm70_tp8_clique_barrier(sg, self_sg, rank, partial_slot);
  A* const self_partial = get_tmp_buf<A>(self_sg) + partial_slot * packed_size;
  const A* const pair_partial =
      get_tmp_buf<A>(sg.signals[pair_rank]) + partial_slot * packed_size;

  for (int idx = tid; idx < packed_size; idx += blockDim.x) {
    A partial{};
    partial = upcast(reinterpret_cast<const P*>(dp.ptrs[clique_base])[idx]);
#pragma unroll
    for (int i = 1; i < 4; ++i) {
      packed_assign_add(
          partial,
          upcast(reinterpret_cast<const P*>(dp.ptrs[clique_base + i])[idx]));
    }
    self_partial[idx] = partial;
  }

  // Publish the FP32 clique partial only after every producer thread has made
  // its store visible to the paired GPU.
  __threadfence_system();
  __syncthreads();
  const FlagType pair_flag = clique_flag + 1;
  if (tid < 4) {
    const int peer = clique_base + tid;
    const int completion_slot = kSm70Tp8CompletionSignalSlotBase + partial_slot;
    st_flag_volatile(&sg.signals[peer]->end[completion_slot][rank], pair_flag);
    while (ld_flag_volatile(&self_sg->end[completion_slot][peer]) != pair_flag);
  } else if (tid == 4) {
    st_flag_release(&sg.signals[pair_rank]->end[partial_slot][rank], pair_flag);
    while (ld_flag_acquire(&self_sg->end[partial_slot][pair_rank]) !=
           pair_flag);
  }
  __syncthreads();

  for (int idx = tid; idx < packed_size; idx += blockDim.x) {
    // Both cliques form the same (ranks 0..3) + (ranks 4..7) FP32 tree,
    // independent of which side writes the output.
    A total = rank < 4 ? self_partial[idx] : pair_partial[idx];
    packed_assign_add(total, rank < 4 ? pair_partial[idx] : self_partial[idx]);
    reinterpret_cast<P*>(result)[idx] = downcast<P>(total);
  }
  if (tid == 0) self_sg->_flag[0] = pair_flag;
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
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_sum2_2stage(RankData* _dp_a, RankData* _dp_b,
                                    RankSignals sg, Signal* self_sg,
                                    T* __restrict__ result, int rank,
                                    int size) {
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
    tmp_out[idx - start] = packed_reduce_sum2<P, ngpus, A>(ptrs_a, ptrs_b, idx);
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
  RankData sm70_tp4_push_buffers_{};
  bool sm70_tp4_push_buffers_registered_ = false;
  RankData sm70_tp8_hierarchical_push_buffers_{};
  bool sm70_tp8_hierarchical_push_buffers_registered_ = false;

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

  void register_sm70_tp4_push_buffer(void** ptrs) {
    if (world_size_ != kSm70Tp4PushAllreduceWorldSize || !fully_connected_ ||
        !custom_allreduce_current_device_is_sm70()) {
      throw std::runtime_error(
          "SM70 push all-reduce requires fully-connected TP4 on SM70.");
    }
    for (int peer = 0; peer < world_size_; ++peer) {
      if (ptrs[peer] == nullptr) {
        throw std::runtime_error(
            "SM70 push all-reduce received a null peer buffer.");
      }
      sm70_tp4_push_buffers_.ptrs[peer] = ptrs[peer];
    }
    auto* generic_data =
        static_cast<char*>(ptrs[rank_]) + kSm70Tp4PushAllreduceSignalBytes;
    CUDACHECK(cudaMemset(generic_data, kSm70Tp4PushAllreduceSentinelByte,
                         kSm70Tp4PushAllreduceGenericBufferBytes -
                             kSm70Tp4PushAllreduceSignalBytes));
    auto* hc_signal =
        static_cast<char*>(ptrs[rank_]) + kSm70Qwen38HcPushSignalOffset;
    CUDACHECK(cudaMemset(hc_signal, 0, kSm70Qwen38HcPushSignalBytes));
    auto* hc_data =
        static_cast<char*>(ptrs[rank_]) + kSm70Qwen38HcDownPushOffset;
    CUDACHECK(cudaMemset(
        hc_data, kSm70Tp4PushAllreduceSentinelByte,
        kSm70Qwen38HcUpFusedEpochOffset - kSm70Qwen38HcDownPushOffset));
    // The first fused packet uses generation 1; zero is initially invalid.
    auto* hc_up =
        static_cast<char*>(ptrs[rank_]) + kSm70Qwen38HcUpFusedEpochOffset;
    CUDACHECK(cudaMemset(
        hc_up, 0,
        kSm70Tp4PushAllreduceBufferBytes - kSm70Qwen38HcUpFusedEpochOffset));
    sm70_tp4_push_buffers_registered_ = true;
  }

  void register_sm70_tp8_hierarchical_push_buffer(void** ptrs) {
    if (world_size_ != kSm70Tp8HierarchicalPushWorldSize || fully_connected_ ||
        !custom_allreduce_current_device_is_sm70()) {
      throw std::runtime_error(
          "SM70 hierarchical push all-reduce requires non-fully-connected "
          "TP8 on SM70.");
    }
    for (int peer = 0; peer < world_size_; ++peer) {
      if (sm70_tp8_hierarchical_peer(rank_, peer) && ptrs[peer] == nullptr) {
        throw std::runtime_error(
            "SM70 hierarchical push all-reduce received a null required "
            "peer buffer.");
      }
      sm70_tp8_hierarchical_push_buffers_.ptrs[peer] = ptrs[peer];
    }
    auto* local_data =
        static_cast<char*>(ptrs[rank_]) + kSm70Tp8HierarchicalPushSignalBytes;
    CUDACHECK(cudaMemset(local_data, kSm70Tp4PushAllreduceSentinelByte,
                         kSm70Tp8HierarchicalPushBufferBytes -
                             kSm70Tp8HierarchicalPushSignalBytes));
    sm70_tp8_hierarchical_push_buffers_registered_ = true;
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
        throw std::runtime_error(
            std::string(op_name) + " buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(buffer)) +
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
          if (sm70_tp8_hierarchical_custom_ar_enabled(world_size_,
                                                      fully_connected_) &&
              !sm70_tp8_hierarchical_peer(rank_, j)) {
            rd.ptrs[j] = nullptr;
            continue;
          }
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
        block_limit, world_size_, fully_connected_,
        static_cast<size_t>(size) * sizeof(T), true);
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
    if constexpr (std::is_same_v<T, half>) {
      // The push protocol amortizes peer polling across captured collective
      // chains. A lone eager call stays on the ordinary registered-buffer
      // pull path.
      if (sm70_tp4_push_buffers_registered_ &&
          status == cudaStreamCaptureStatusActive &&
          world_size_ == kSm70Tp4PushAllreduceWorldSize && fully_connected_ &&
          custom_allreduce_current_device_is_sm70()) {
        const int push_blocks = sm70_tp4_push_allreduce_blocks(bytes, true);
        if (push_blocks > 0) {
          sm70_cross_device_reduce_1stage_push<kSm70Tp4PushAllreduceWorldSize>
              <<<push_blocks, kSm70Tp4PushAllreduceThreads, 0, stream>>>(
                  sm70_tp4_push_buffers_, input, output, rank_, size);
          return;
        }
      }
      if (sm70_tp8_hierarchical_push_buffers_registered_ &&
          status == cudaStreamCaptureStatusActive &&
          sm70_tp8_hierarchical_custom_ar_enabled(world_size_,
                                                  fully_connected_) &&
          sm70_tp8_hierarchical_allreduce_size(bytes)) {
        const int push_blocks = sm70_tp8_hierarchical_push_blocks(bytes);
        sm70_tp8_hierarchical_reduce_push<false>
            <<<push_blocks, kSm70Tp8HierarchicalPushThreads, 0, stream>>>(
                sm70_tp8_hierarchical_push_buffers_, input, nullptr, output,
                rank_, size);
        return;
      }
      if (sm70_tp8_hierarchical_custom_ar_enabled(world_size_,
                                                  fully_connected_) &&
          sm70_tp8_hierarchical_allreduce_size(bytes)) {
        sm70_tp8_hierarchical_reduce<<<1, 512, 0, stream>>>(
            ptrs, sg_, self_sg_, output, rank_, size);
        return;
      }
    }
    threads =
        sm70_tp4_m5_allreduce_threads(world_size_, fully_connected_, bytes);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

    if constexpr (std::is_same_v<T, half>) {
      if (blocks == 1 && threads == 512 &&
          sm70_tp4_small_allreduce_pack32(world_size_, fully_connected_,
                                          bytes)) {
        sm70_cross_device_reduce_1stage_pack32<4><<<1, 512, 0, stream>>>(
            ptrs, sg_, self_sg_, output, rank_, bytes / 32);
        return;
      }
    }

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
                                     half* normalized_out, float* residual_out,
                                     int num_tokens, int hidden_size,
                                     float epsilon) {
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
          std::to_string(kMaxTokens) + "]. Got " + std::to_string(num_tokens) +
          ".");
    }

    RankData* ptrs = rank_data_for_buffer(stream, input,
                                          "SM70 Gemma RMSNorm prototype input");
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
#define VLLM_LAUNCH_SM70_GEMMA_RMS_NORM(THREADS)                          \
  sm70_peer_reduce_gemma_rms_norm<THREADS, ngpus, ResidualT, WeightT>     \
      <<<num_tokens, THREADS, 0, stream>>>(ptrs, sg_, self_sg_, residual, \
                                           weight, normalized_out,        \
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

  template <int ngpus, typename ResidualT, typename WeightT>
  void sm70_reduce_scatter_gemma_rms_norm_all_gather(
      cudaStream_t stream, half* input, half* shared_output,
      const ResidualT* residual, const WeightT* weight, float* residual_out,
      int num_tokens, int hidden_size, float epsilon) {
    if (world_size_ != ngpus || !fully_connected_ ||
        !custom_allreduce_current_device_is_sm70()) {
      throw std::runtime_error(
          "SM70 long-prefill fused norm requires fully connected TP" +
          std::to_string(ngpus) + " on SM70.");
    }
    if (hidden_size != kSm70GemmaRmsNormHiddenSize || num_tokens <= 0 ||
        num_tokens % ngpus != 0) {
      throw std::runtime_error(
          "SM70 long-prefill fused norm requires a TP-divisible [M, 5120] "
          "input.");
    }
    const int tokens_per_rank = num_tokens / ngpus;
    if (tokens_per_rank > kSm70LongPrefillMaxTokensPerRank) {
      throw std::runtime_error(
          "SM70 long-prefill fused norm exceeds the token-shard capacity.");
    }

    RankData* input_ptrs =
        rank_data_for_buffer(stream, input, "long-prefill fused norm input");
    RankData* output_ptrs = rank_data_for_buffer(
        stream, shared_output, "long-prefill fused norm output");
    const int threads = sm70_tp4_long_fused_norm_threads();
    const int blocks =
        std::min(tokens_per_rank, sm70_tp4_long_fused_norm_blocks());
#define VLLM_LAUNCH_SM70_TP4_LONG_FUSED_NORM(THREADS)                          \
  sm70_peer_reduce_scatter_gemma_rms_norm_all_gather<THREADS, ngpus,           \
                                                     ResidualT, WeightT>       \
      <<<blocks, THREADS, 0, stream>>>(input_ptrs, output_ptrs, sg_, self_sg_, \
                                       residual, weight, residual_out, rank_,  \
                                       num_tokens, epsilon)
    switch (threads) {
      case 256:
        VLLM_LAUNCH_SM70_TP4_LONG_FUSED_NORM(256);
        break;
      case 1024:
        VLLM_LAUNCH_SM70_TP4_LONG_FUSED_NORM(1024);
        break;
      default:
        VLLM_LAUNCH_SM70_TP4_LONG_FUSED_NORM(512);
        break;
    }
#undef VLLM_LAUNCH_SM70_TP4_LONG_FUSED_NORM
  }

  template <typename T>
  void allreduce_sum2(cudaStream_t stream, T* input_a, T* input_b, T* output,
                      int size, int threads = 512,
                      int block_limit = defaultBlockLimit) {
    block_limit = custom_allreduce_block_limit(
        block_limit, world_size_, fully_connected_,
        static_cast<size_t>(size) * sizeof(T), false);
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
    if constexpr (std::is_same_v<T, half>) {
      const char* batch =
          std::getenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH");
      const bool qwen38_batch =
          (batch == nullptr || std::strcmp(batch, "1") == 0) &&
          (bytes == kSm70Tp4PushAllreduceQwen38M4Bytes ||
           bytes == kSm70Tp4PushAllreduceQwen38M8Bytes ||
           bytes == kSm70Tp4PushAllreduceBytes);
      const char* mtp5 = std::getenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5");
      const bool qwen38_mtp5 = mtp5 != nullptr && std::strcmp(mtp5, "1") == 0 &&
                               bytes == kSm70Tp4PushAllreduceQwen4ExpMtp5Bytes;
      const char* qwen4_exp_m1 =
          std::getenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1");
      const bool qwen4_exp_m1_enabled =
          bytes == kSm70Tp4PushAllreduceQwen4ExpBytes &&
          (qwen4_exp_m1 == nullptr || std::strcmp(qwen4_exp_m1, "1") == 0);

      if (sm70_tp8_hierarchical_push_buffers_registered_ &&
          status == cudaStreamCaptureStatusActive &&
          sm70_tp8_hierarchical_custom_ar_enabled(world_size_,
                                                  fully_connected_) &&
          sm70_tp8_hierarchical_allreduce_size(bytes)) {
        const int push_blocks = sm70_tp8_hierarchical_push_blocks(bytes);
        sm70_tp8_hierarchical_reduce_push<true>
            <<<push_blocks, kSm70Tp8HierarchicalPushThreads, 0, stream>>>(
                sm70_tp8_hierarchical_push_buffers_, input_a, input_b, output,
                rank_, size);
        return;
      }
      if (sm70_tp4_push_buffers_registered_ &&
          status == cudaStreamCaptureStatusActive &&
          world_size_ == kSm70Tp4PushAllreduceWorldSize && fully_connected_ &&
          (qwen38_batch || qwen38_mtp5 || qwen4_exp_m1_enabled) &&
          custom_allreduce_current_device_is_sm70()) {
        const int push_blocks = sm70_tp4_push_allreduce_blocks(bytes);
        if (push_blocks > 0) {
          sm70_cross_device_reduce_sum2_1stage_push<
              kSm70Tp4PushAllreduceWorldSize>
              <<<push_blocks, kSm70Tp4PushAllreduceThreads, 0, stream>>>(
                  sm70_tp4_push_buffers_, input_a, input_b, output, rank_,
                  size);
          return;
        }
      }
    }
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

#define SUM2_KL(ngpus, name)                      \
  name<T, ngpus><<<blocks, threads, 0, stream>>>( \
      ptrs_a, ptrs_b, sg_, self_sg_, output, rank_, size);
#define SUM2_CASE(ngpus)                                   \
  case ngpus: {                                            \
    if (force_1stage) {                                    \
      SUM2_KL(ngpus, cross_device_reduce_sum2_1stage);     \
    } else if (force_2stage) {                             \
      SUM2_KL(ngpus, cross_device_reduce_sum2_2stage);     \
    } else {                                               \
      if (world_size_ == 2) {                              \
        SUM2_KL(ngpus, cross_device_reduce_sum2_1stage);   \
      } else if (fully_connected_) {                       \
        if ((world_size_ <= 4 && bytes < 512 * 1024) ||    \
            (world_size_ <= 8 && bytes < 256 * 1024)) {    \
          SUM2_KL(ngpus, cross_device_reduce_sum2_1stage); \
        } else {                                           \
          SUM2_KL(ngpus, cross_device_reduce_sum2_2stage); \
        }                                                  \
      }                                                    \
    }                                                      \
    break;                                                 \
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
    const int tile_count =
        (packed_size + tile_packed_size - 1) / tile_packed_size;
    if (tile_count <= 0 || tile_count > kMaxBlocks) {
      throw std::runtime_error(
          "SM70 tile runtime prototype supports tile_count in [1, " +
          std::to_string(kMaxBlocks) + "]. Got " + std::to_string(tile_count));
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

#define TILE_RUNTIME_CASE(ngpus)                                               \
  case ngpus: {                                                                \
    sm70_tile_runtime_reduce_kernel<T, ngpus><<<blocks, threads, 0, stream>>>( \
        ptrs, sg_, self_sg_, input, staging, output, rank_, packed_size,       \
        tile_packed_size, tile_count, compute_iters);                          \
    break;                                                                     \
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
          std::to_string(kMaxBlocks) + "]. Got " + std::to_string(tile_count));
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

#define TILE_RUNTIME_ENGINE_CASE(ngpus)                                        \
  case ngpus: {                                                                \
    sm70_tile_runtime_engine_kernel<T, ngpus><<<blocks, threads, 0, stream>>>( \
        ptrs, sg_, self_sg_, input, staging, output, rank_, packed_size,       \
        tile_packed_size, tile_count, producer_blocks, reducer_blocks,         \
        compute_iters);                                                        \
    break;                                                                     \
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
                                int size, int tile_numel, int reducer_blocks) {
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
          std::to_string(kMaxBlocks) + "]. Got " + std::to_string(tile_count));
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

#define TILE_RUNTIME_WAIT_REDUCE_CASE(ngpus)                                   \
  case ngpus: {                                                                \
    sm70_tile_runtime_wait_reduce_kernel<T, ngpus>                             \
        <<<reducer_blocks, threads, 0, stream>>>(                              \
            ptrs, sg_, self_sg_, output, rank_, packed_size, tile_packed_size, \
            tile_count);                                                       \
    break;                                                                     \
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

#define TOP1_CASE(ngpus)                                            \
  case ngpus: {                                                     \
    cross_device_top1_argmax<ngpus>                                 \
        <<<1, 32, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_); \
    break;                                                          \
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
