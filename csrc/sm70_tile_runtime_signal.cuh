#pragma once

#include <cstdint>

namespace vllm::sm70_tile_runtime {

// Keep the default all-reduce block limit unchanged, but allow the
// TileRT-style MLP down-proj path to publish one flag per TurboMind N tile.
constexpr int kMaxBlocks = 64;
constexpr int kMaxRanks = 8;

using FlagType = uint32_t;

struct Signal {
  alignas(128) FlagType start[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType end[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType _flag[kMaxBlocks];
};

struct __align__(16) RankData {
  const void* ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

#if !defined(USE_ROCM)
static __device__ __forceinline__ void store_flag_sys_visible(
    FlagType* flag_addr, FlagType flag) {
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;"
               ::"r"(flag),
               "l"(flag_addr)
               : "memory");
}

static __device__ __forceinline__ FlagType load_flag_sys_visible(
    FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.sys;"
               : "=r"(flag)
               : "l"(flag_addr)
               : "memory");
  return flag;
}
#endif

}  // namespace vllm::sm70_tile_runtime
