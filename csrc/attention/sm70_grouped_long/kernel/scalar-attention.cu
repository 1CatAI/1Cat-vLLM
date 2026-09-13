#include <cuda.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <string>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_radix_sort.cuh>

#include "fp8_kv_utils.cuh"
#include "fused_mma.h"

namespace {

int kv_cache_dtype_code_from_string(const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "auto" || kv_cache_dtype == "float16" ||
      kv_cache_dtype == "bfloat16") {
    return flash_v100::KV_CACHE_DTYPE_FP16;
  }
  if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    return flash_v100::KV_CACHE_DTYPE_FP8_E4M3;
  }
  if (kv_cache_dtype == "fp8_e5m2") {
    return flash_v100::KV_CACHE_DTYPE_FP8_E5M2;
  }
  return -1;
}

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kXQATCBlockN = 128;
constexpr int kXQATCStride = 128;
// QSA exposes selected four-token microblocks as virtual paged KV. Keep enough
// page slots for one full XQA tile at that minimum supported granularity.
constexpr int kXQATCPageIdsCapacity = kXQATCBlockN / 4;
constexpr int kXQATC256WideWarpCount = 8;
constexpr int kXQATC256WideThreads = kXQATC256WideWarpCount * kWarpSize;
constexpr int kXQATC256WideBlockM = 8;
constexpr int kXQATC256WideThreadsPerRow = kWarpSize;
constexpr int kXQATCG6DualCtaThreads = 6 * kWarpSize;
constexpr int kXQATCG6Pipeline8WarpThreads = 8 * kWarpSize;
constexpr int kXQARouteAllSeqLens = 0;
constexpr int kXQARouteShortSeqLens = -1;
constexpr int kXQARouteLongSeqLens = 1;
constexpr int kXQARouteP1024Sawtooth = 4;
constexpr int kXQARouteP256Sawtooth = 5;
constexpr int kXQARouteP1024SawtoothMid = 6;
constexpr int kXQARouteP1024SawtoothFinal = 7;
constexpr int kXQARouteRangeSeqLens = 8;
constexpr int kXQARouteWaveLongSeqLens = 9;
// The dense 256/128-half strides alias Volta shared-memory banks in the
// WMMA A/B fragment loads. Both padded strides remain 16-byte aligned.
constexpr int kXQATC256WidePaddedQStride = 264;
constexpr int kXQATC256WidePaddedKVStride = 136;
constexpr int kXQATC256WideAlignedPaddedQStride = 272;
constexpr int kXQATC256WideAlignedPaddedKVStride = 144;
constexpr int kXQATCQKPipelinePanelDim = 64;
constexpr int kXQATCQKPipelineKVStride = 72;
constexpr float kXQANegInf = -1.0e30f;

template <bool PADDED_SMEM, bool ALIGNED_PADDED_SMEM = false>
struct alignas(256) XQATCSmem256WideLayout {
  static_assert(!ALIGNED_PADDED_SMEM || PADDED_SMEM,
                "Aligned padding requires padded shared memory");
  static constexpr int kQStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedQStride
                                         : kXQATC256WidePaddedQStride)
                  : 256;
  static constexpr int kKVStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedKVStride
                                         : kXQATC256WidePaddedKVStride)
                  : kXQATCStride;
  static constexpr int kQKStride = kKVStride;
  alignas(16) __half q[kXQATC256WideBlockM * kQStride];
  union {
    alignas(16) __half k[kXQATCBlockN * kKVStride];
    alignas(16) __half v[kXQATCBlockN * kKVStride];
  } reuse_kv;
  struct {
    alignas(16) float s[kXQATC256WideBlockM * kXQATCBlockN];
    alignas(16) __half p[kXQATC256WideBlockM * kXQATCBlockN];
  } reuse_sp;
  alignas(16) float row_max[kXQATC256WideBlockM];
  alignas(16) float row_sum[kXQATC256WideBlockM];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];

  __device__ __forceinline__ __half* k_buffer(int) { return reuse_kv.k; }
  __device__ __forceinline__ __half* v_buffer() { return reuse_kv.v; }
};

template <bool PADDED_SMEM, bool ALIGNED_PADDED_SMEM = false>
struct alignas(256) XQATCQKPipelineSmem256WideLayout {
  static_assert(!ALIGNED_PADDED_SMEM || PADDED_SMEM,
                "Aligned padding requires padded shared memory");
  static constexpr int kQStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedQStride
                                         : kXQATC256WidePaddedQStride)
                  : 256;
  static constexpr int kKVStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedKVStride
                                         : kXQATC256WidePaddedKVStride)
                  : kXQATCStride;
  static constexpr int kQKStride = kXQATCQKPipelineKVStride;
  struct QKBuffers {
    alignas(16) __half panel[2][kXQATCBlockN * kQKStride];
  };

  alignas(16) __half q[kXQATC256WideBlockM * kQStride];
  union {
    QKBuffers qk;
    alignas(16) __half v[kXQATCBlockN * kKVStride];
  } reuse_kv;
  struct {
    alignas(16) float s[kXQATC256WideBlockM * kXQATCBlockN];
    alignas(16) __half p[kXQATC256WideBlockM * kXQATCBlockN];
  } reuse_sp;
  alignas(16) float row_max[kXQATC256WideBlockM];
  alignas(16) float row_sum[kXQATC256WideBlockM];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];

  __device__ __forceinline__ __half* k_buffer(int index) {
    return reuse_kv.qk.panel[index];
  }
  __device__ __forceinline__ __half* v_buffer() { return reuse_kv.v; }
};

constexpr int kXQATCStagedPVTileRows = 64;

template <bool PADDED_SMEM>
struct alignas(256) XQATCStagedPVSmem256Wide {
  static constexpr int kKVStride =
      PADDED_SMEM ? kXQATC256WidePaddedKVStride : kXQATCStride;
  alignas(16) __half v[kXQATCStagedPVTileRows * kKVStride];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];
};

bool xqa_padded_smem_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_PADDED_SMEM");
  return value == nullptr || value[0] != '0';
}

bool xqa_g6_dual_cta_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_DUAL_CTA");
  return value != nullptr && value[0] == '1';
}

bool xqa_e4m3_batch_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_E4M3_BATCH_XQA");
  return value == nullptr || value[0] != '0';
}

bool xqa_e4m3_batch_optimized_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_E4M3_BATCH_XQA_OPTIMIZED");
  return value == nullptr || value[0] != '0';
}

bool xqa_e4m3_page800_fastpath_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH");
  return value == nullptr || value[0] != '0';
}

bool xqa_e4m3_page800_fastpath_trace_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH_TRACE");
  return value != nullptr && value[0] == '1';
}

void trace_xqa_e4m3_page800_fastpath(const int batch_size,
                                     const int partition_size,
                                     const bool active) {
  if (!active || !xqa_e4m3_page800_fastpath_trace_enabled()) {
    return;
  }
  static std::atomic<bool> traced{false};
  if (!traced.exchange(true, std::memory_order_relaxed)) {
    TORCH_WARN("Flash-V100 E4M3 page800 fast path active: batch=", batch_size,
               ", partition_size=", partition_size,
               ", standard interleaved Hkv=1, PV=half2");
  }
}

bool xqa_e5m2_g6_dual_cta_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA");
  return value == nullptr || value[0] != '0';
}

bool xqa_e5m2_g6_split_reduce_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E5M2_G6_SPLIT_REDUCE");
  return value == nullptr || value[0] != '0';
}

bool xqa_e5m2_partition_page_ids_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_E5M2_PARTITION_PAGE_IDS");
  return value == nullptr || value[0] != '0';
}

bool xqa_e5m2_pair_load_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E5M2_PAIR_LOAD");
  return value == nullptr || value[0] != '0';
}

bool xqa_e5m2_batch_wide_load_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E5M2_BATCH_WIDE_LOAD");
  return value == nullptr || value[0] != '0';
}

bool dflash2_grouped_fixed_interleaved_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED");
  return value == nullptr || value[0] != '0';
}

bool dflash2_grouped_stage_page_ids_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS");
  return value == nullptr || value[0] != '0';
}

int xqa_e5m2_p1024_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E5M2_P1024_BEGIN");
  return value == nullptr ? 61633 : std::max(1, std::atoi(value));
}

int xqa_e5m2_scalar_xqa_seq_len() {
  const char* value = std::getenv("VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN");
  return value == nullptr ? 16384 : std::max(1, std::atoi(value));
}

bool xqa_e5m2_g6_dual_cta_trace_enabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

bool xqa_mtp5_dual_cta_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA");
  return value == nullptr || value[0] == '1';
}

bool xqa_g6_dual_cta_dense_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_DUAL_CTA_DENSE");
  return value != nullptr && value[0] == '1';
}

bool xqa_g6_p1024_auto_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_AUTO");
  return value == nullptr || value[0] != '0';
}

bool xqa_g6_p1024_auto_trace_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_AUTO_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

bool xqa_g6_p1024_sawtooth_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH");
  return value == nullptr || value[0] != '0';
}

bool xqa_e4m3_g6_p64_p256_auto_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO");
  return value != nullptr && value[0] == '1';
}

bool xqa_e4m3_g6_p64_p256_auto_trace_enabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

int xqa_e4m3_g6_p256_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P256_BEGIN");
  return value == nullptr ? 12288 : std::max(1, std::atoi(value));
}

int xqa_e4m3_g6_dual_cta_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_DUAL_CTA_BEGIN");
  return value == nullptr ? 32768 : std::max(1, std::atoi(value));
}

bool xqa_e4m3_g6_wave_partitions_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS");
  return value != nullptr && value[0] == '1';
}

bool xqa_e4m3_g6_merged_wave_launch_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH");
  return value != nullptr && value[0] == '1';
}

int xqa_e4m3_g6_p512_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P512_BEGIN");
  return value == nullptr ? 49152 : std::max(1, std::atoi(value));
}

int xqa_e4m3_g6_p896_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P896_BEGIN");
  return value == nullptr ? 98304 : std::max(1, std::atoi(value));
}

int xqa_e4m3_g6_p1664_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P1664_BEGIN");
  return value == nullptr ? 196608 : std::max(1, std::atoi(value));
}

bool decode_partition_size_overridden() {
  return std::getenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE") != nullptr;
}

bool xqa_g6_qk_pipeline_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE");
  return value == nullptr || value[0] != '0';
}

int xqa_g6_qk_pipeline_warps() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_WARPS");
  return value != nullptr && std::atoi(value) == 6 ? 6 : 8;
}

bool xqa_g6_qk_pipeline_trace_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

bool xqa_g6_p1024_sawtooth_trace_enabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

int xqa_g6_p1024_sawtooth_p1024_mid_seq_len() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_MID_SEQ_LEN");
  return value == nullptr ? 111104 : std::max(1, std::atoi(value));
}

int xqa_g6_p1024_sawtooth_p256_long_seq_len() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P256_LONG_SEQ_LEN");
  return value == nullptr ? 147841 : std::max(1, std::atoi(value));
}

int xqa_g6_p1024_sawtooth_p1024_final_seq_len() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_FINAL_SEQ_LEN");
  return value == nullptr ? 258176 : std::max(1, std::atoi(value));
}

bool xqa_split_reduce_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_SPLIT_REDUCE");
  return value != nullptr && value[0] == '1';
}

enum class XQABatchContextRoute : int {
  kDisabled = -1,
  kBaseline = 0,
  kDualCta = 1,
  kDualCtaSplit = 2,
};

bool xqa_batch_context_routing_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING");
  return value == nullptr || value[0] != '0';
}

bool xqa_batch_context_routing_trace_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING_TRACE");
  return value != nullptr && value[0] == '1';
}

XQABatchContextRoute select_xqa_batch_context_route(const int batch_size,
                                                    const int max_seq_len,
                                                    const int partition_size) {
  if (!xqa_batch_context_routing_enabled() || batch_size < 4 ||
      max_seq_len <= 0) {
    return XQABatchContextRoute::kDisabled;
  }
  if (batch_size <= 4) {
    if (max_seq_len <= 8191) {
      return XQABatchContextRoute::kBaseline;
    }
    if (max_seq_len <= 12287) {
      return XQABatchContextRoute::kDualCta;
    }
    return partition_size == 1024 ? XQABatchContextRoute::kBaseline
                                  : XQABatchContextRoute::kDualCtaSplit;
  }
  if (batch_size <= 8) {
    if (max_seq_len <= 4095) {
      return XQABatchContextRoute::kBaseline;
    }
    return max_seq_len <= 12287 ? XQABatchContextRoute::kDualCta
                                : XQABatchContextRoute::kDualCtaSplit;
  }
  if (batch_size <= 12) {
    if (max_seq_len <= 2047) {
      return XQABatchContextRoute::kBaseline;
    }
    return max_seq_len <= 16000 ? XQABatchContextRoute::kDualCta
                                : XQABatchContextRoute::kDualCtaSplit;
  }
  return max_seq_len <= 12287 ? XQABatchContextRoute::kBaseline
                              : XQABatchContextRoute::kDualCta;
}

const char* xqa_batch_context_route_name(const XQABatchContextRoute route) {
  switch (route) {
    case XQABatchContextRoute::kBaseline:
      return "baseline";
    case XQABatchContextRoute::kDualCta:
      return "dual_cta";
    case XQABatchContextRoute::kDualCtaSplit:
      return "dual_cta_split";
    default:
      return "disabled";
  }
}

void trace_xqa_batch_context_route(const int batch_size, const int max_seq_len,
                                   const int partition_size,
                                   const int block_size,
                                   const XQABatchContextRoute route) {
  if (!xqa_batch_context_routing_trace_enabled() ||
      route == XQABatchContextRoute::kDisabled) {
    return;
  }
  const int batch_class = batch_size <= 4    ? 0
                          : batch_size <= 8  ? 1
                          : batch_size <= 12 ? 2
                                             : 3;
  const int context_class = max_seq_len <= 4095    ? 0
                            : max_seq_len <= 8191  ? 1
                            : max_seq_len <= 12287 ? 2
                            : max_seq_len <= 16000 ? 3
                                                   : 4;
  const unsigned long long bit =
      1ULL << (batch_class * 15 + context_class * 3 + static_cast<int>(route));
  static std::atomic<unsigned long long> traced_routes{0};
  const unsigned long long previous =
      traced_routes.fetch_or(bit, std::memory_order_relaxed);
  if ((previous & bit) == 0) {
    TORCH_WARN("Flash-V100 XQA batch/context route active: batch=", batch_size,
               ", max_seq_len=", max_seq_len,
               ", partition_size=", partition_size, ", block_size=", block_size,
               ", route=", xqa_batch_context_route_name(route));
  }
}

int xqa_block16_layout_mode() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT");
  if (value == nullptr) {
    return 0;
  }
  const int mode = std::atoi(value);
  return mode == 1 || mode == 2 ? mode : 0;
}

bool xqa_block16_layout_required() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_REQUIRE");
  return value != nullptr && value[0] == '1';
}

bool xqa_block16_layout_trace_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

bool xqa_block784_index_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK784_INDEX");
  return value == nullptr || value[0] != '0';
}

bool xqa_block784_index_trace_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK784_INDEX_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

bool xqa_aligned_padded_smem_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM");
  return value != nullptr && value[0] == '1';
}

bool xqa_aligned_padded_smem_trace_enabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

int xqa_split_reduce_dim_tile() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_SPLIT_REDUCE_D_TILE");
  if (value == nullptr) {
    return 8;
  }
  const int dim_tile = std::atoi(value);
  return dim_tile == 8 || dim_tile == 16 || dim_tile == 32 ? dim_tile : 8;
}

template <int SEQ_LEN_ROUTE>
__device__ __forceinline__ bool xqa_seq_len_route_active(
    const int seq_len, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final) {
  if constexpr (SEQ_LEN_ROUTE == kXQARouteShortSeqLens) {
    return seq_len < route_seq_len_begin;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteLongSeqLens) {
    return seq_len >= route_seq_len_begin;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024Sawtooth) {
    return (seq_len >= route_seq_len_begin && seq_len < route_seq_len_end) ||
           seq_len >= route_seq_len_final;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP256Sawtooth) {
    return seq_len < route_seq_len_begin ||
           (seq_len >= route_seq_len_end && seq_len < route_seq_len_final);
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024SawtoothMid) {
    return seq_len >= route_seq_len_begin && seq_len < route_seq_len_end;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024SawtoothFinal) {
    return seq_len >= route_seq_len_final;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteRangeSeqLens) {
    return seq_len >= route_seq_len_begin && seq_len < route_seq_len_end;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens) {
    return seq_len >= route_seq_len_begin;
  }
  return true;
}

__device__ __forceinline__ int xqa_sawtooth_partition_size(
    const int seq_len, const int p1024_mid_seq_len, const int p256_long_seq_len,
    const int p1024_final_seq_len) {
  const bool use_p1024 =
      (seq_len >= p1024_mid_seq_len && seq_len < p256_long_seq_len) ||
      seq_len >= p1024_final_seq_len;
  return use_p1024 ? 1024 : 256;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float shared[NUM_WARPS];
  __shared__ float result;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = warp_reduce_sum(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  val = threadIdx.x < NUM_WARPS ? shared[lane] : 0.f;
  if (warp == 0) {
    val = warp_reduce_sum(val);
    if (lane == 0) {
      result = val;
    }
  }
  __syncthreads();
  return result;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float shared[NUM_WARPS];
  __shared__ float result;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = warp_reduce_max(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  val = threadIdx.x < NUM_WARPS ? shared[lane] : -1.0e20f;
  if (warp == 0) {
    val = warp_reduce_max(val);
    if (lane == 0) {
      result = val;
    }
  }
  __syncthreads();
  return result;
}

__device__ __forceinline__ uint32_t
fp8_e5m2_pair_to_half2_bits(const uint16_t raw_pair) {
  return (static_cast<uint32_t>(raw_pair & 0x00ffu) << 8) |
         (static_cast<uint32_t>(raw_pair & 0xff00u) << 16);
}

__device__ __forceinline__ uint4 fp8_e5m2_vector_to_half8(const uint64_t raw) {
  return make_uint4(
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 16)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 32)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint16_t fp8_e4m3fn_to_half_bits(const uint8_t raw) {
  const uint16_t sign = static_cast<uint16_t>(raw & 0x80u) << 8;
  const uint8_t magnitude = raw & 0x7fu;
  const uint8_t exponent = magnitude >> 3;
  const uint8_t mantissa = magnitude & 0x07u;
  if (magnitude == 0) {
    return sign;
  }
  if (exponent == 0) {
    // E4M3 subnormals are exact fp16 normals: mantissa * 2^-9.
    const uint16_t magnitude_bits =
        mantissa < 2
            ? 0x1800u
            : (mantissa < 4
                   ? static_cast<uint16_t>(0x1c00u | ((mantissa - 2) << 9))
                   : static_cast<uint16_t>(0x2000u | ((mantissa - 4) << 8)));
    return sign | magnitude_bits;
  }
  if (magnitude == 0x7fu) {
    return sign | 0x7e00u;
  }
  return sign | static_cast<uint16_t>((exponent + 8) << 10) |
         static_cast<uint16_t>(mantissa << 7);
}

__device__ __forceinline__ uint32_t
fp8_e4m3fn_pair_to_half2_bits(const uint16_t raw_pair) {
  return static_cast<uint32_t>(
             fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw_pair))) |
         (static_cast<uint32_t>(
              fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw_pair >> 8)))
          << 16);
}

__device__ __forceinline__ uint32_t
fp8_e4m3fn_pair_to_half2_bits_fast(const uint16_t raw_pair) {
  const uint8_t raw0 = static_cast<uint8_t>(raw_pair);
  const uint8_t raw1 = static_cast<uint8_t>(raw_pair >> 8);
  if ((raw0 & 0x7fu) == 0x7fu || (raw1 & 0x7fu) == 0x7fu) {
    return fp8_e4m3fn_pair_to_half2_bits(raw_pair);
  }

  // Moving a finite E4M3 encoding into the corresponding fp16 sign,
  // exponent, and mantissa fields represents exactly value / 256. A packed
  // half2 multiply restores both values without per-byte exponent branches.
  const uint32_t expanded = (static_cast<uint32_t>(raw_pair & 0x0080u) << 8) |
                            (static_cast<uint32_t>(raw_pair & 0x007fu) << 7) |
                            (static_cast<uint32_t>(raw_pair & 0x8000u) << 16) |
                            (static_cast<uint32_t>(raw_pair & 0x7f00u) << 15);
  union {
    uint32_t u;
    __half2 h2;
  } converter;
  converter.u = expanded;
  converter.h2 = __hmul2(converter.h2, __float2half2_rn(256.0f));
  return converter.u;
}

__device__ __forceinline__ uint4
fp8_e4m3fn_vector_to_half8(const uint64_t raw) {
  return make_uint4(
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 16)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 32)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint4
fp8_e4m3fn_vector_to_half8_fast(const uint64_t raw) {
  return make_uint4(
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 16)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 32)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint4 fp8_e4m3fn_vector_to_half8_lut(
    const uint64_t raw, const uint16_t* __restrict__ lut) {
  return make_uint4(
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 8)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 16)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 24)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 32)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 40)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 48)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 56)]) << 16));
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16,
          bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ uint4 load_xqa_tc_kv_vector(
    const void* __restrict__ kv_cache, const int* __restrict__ page_ids,
    const int copy_idx, const int panel_d_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const uint16_t* __restrict__ e4m3_lut = nullptr) {
  const int row = copy_idx / panel_d_stride_uint4;
  const int vec_col = copy_idx % panel_d_stride_uint4;
  const int token_offset = tile_page_offset + kv_tile_start + row;
  static_assert(BLOCK_SIZE == 0 || BLOCK_SIZE == 4 || BLOCK_SIZE == 16 ||
                    BLOCK_SIZE == 784 || BLOCK_SIZE == 800 ||
                    BLOCK_SIZE == 1568 || BLOCK_SIZE == 1648 ||
                    BLOCK_SIZE == 3296,
                "Unsupported paged-KV block-size specialization");
  static_assert(!CONTIGUOUS_HKV1_LAYOUT || BLOCK_SIZE == 16 ||
                    BLOCK_SIZE == 800 || BLOCK_SIZE == 1568 ||
                    BLOCK_SIZE == 1648 || BLOCK_SIZE == 3296,
                "The fixed-stride Hkv=1 layout requires a specialized page");
  int logical_block;
  int block_offset;
  if constexpr (BLOCK_SIZE == 4) {
    logical_block = token_offset >> 2;
    block_offset = token_offset & 3;
  } else if constexpr (BLOCK_SIZE == 16) {
    logical_block = token_offset >> 4;
    block_offset = token_offset & 15;
  } else if constexpr (BLOCK_SIZE == 784) {
    logical_block = token_offset / 784;
    block_offset = token_offset - logical_block * 784;
  } else if constexpr (BLOCK_SIZE == 800) {
    logical_block = token_offset >= 800;
    block_offset = token_offset - logical_block * 800;
  } else if constexpr (BLOCK_SIZE == 1568) {
    logical_block = token_offset / 1568;
    block_offset = token_offset - logical_block * 1568;
  } else if constexpr (BLOCK_SIZE == 1648) {
    logical_block = token_offset / 1648;
    block_offset = token_offset - logical_block * 1648;
  } else if constexpr (BLOCK_SIZE == 3296) {
    logical_block = token_offset / 3296;
    block_offset = token_offset - logical_block * 3296;
  } else {
    logical_block = token_offset / block_size;
    block_offset = token_offset % block_size;
  }
  const int physical_block = page_ids[logical_block];
  int64_t physical_offset;
  if constexpr (CONTIGUOUS_HKV1_LAYOUT) {
    constexpr int64_t kHeadDim = 256;
    constexpr int64_t kBlockStride =
        BLOCK_SIZE == 16 ? 16 * kHeadDim : 2 * BLOCK_SIZE * kHeadDim;
    physical_offset = static_cast<int64_t>(physical_block) * kBlockStride +
                      static_cast<int64_t>(block_offset) * kHeadDim +
                      panel_offset;
  } else {
    physical_offset = static_cast<int64_t>(physical_block) * block_stride +
                      static_cast<int64_t>(block_offset) * token_stride +
                      static_cast<int64_t>(kv_head_idx) * head_stride +
                      panel_offset;
  }
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    const uint4* cache_vec = reinterpret_cast<const uint4*>(kv_cache);
    return __ldg(&cache_vec[physical_offset / 8 + vec_col]);
  } else {
    static_assert(KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
                  "XQA only supports fp16, FP8 E4M3, and FP8 E5M2 KV");
    const uint64_t* cache_vec = reinterpret_cast<const uint64_t*>(kv_cache);
    const uint64_t raw = __ldg(&cache_vec[physical_offset / 8 + vec_col]);
    if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3) {
      if constexpr (E4M3_SHARED_LUT) {
        return fp8_e4m3fn_vector_to_half8_lut(raw, e4m3_lut);
      } else {
        return fp8_e4m3fn_vector_to_half8(raw);
      }
    } else {
      static_assert(!E4M3_SHARED_LUT,
                    "The E4M3 conversion LUT requires E4M3 KV");
      return fp8_e5m2_vector_to_half8(raw);
    }
  }
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16,
          bool FP8_PAIR_LOAD = false, bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ void load_xqa_tc_kv_panel(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx = threadIdx.x,
    const uint16_t* __restrict__ e4m3_lut = nullptr) {
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  if constexpr (FP8_PAIR_LOAD) {
    static_assert(KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
                  "Paired XQA loads require FP8 KV");
    static_assert(!E4M3_SHARED_LUT,
                  "Paired E4M3 conversion does not use the shared LUT");
    const int pair_stride = panel_d_stride_uint4 / 2;
    const int pair_count = valid_kv_tile_rows * pair_stride;
    for (int pair_idx = copy_thread_idx; pair_idx < pair_count;
         pair_idx += NUM_THREADS) {
      const int row = pair_idx / pair_stride;
      const int vec_pair = pair_idx % pair_stride;
      const int token_offset = tile_page_offset + kv_tile_start + row;
      int logical_block;
      int block_offset;
      if constexpr (BLOCK_SIZE == 4) {
        logical_block = token_offset >> 2;
        block_offset = token_offset & 3;
      } else if constexpr (BLOCK_SIZE == 16) {
        logical_block = token_offset >> 4;
        block_offset = token_offset & 15;
      } else if constexpr (BLOCK_SIZE == 784) {
        logical_block = token_offset / 784;
        block_offset = token_offset - logical_block * 784;
      } else if constexpr (BLOCK_SIZE == 800) {
        logical_block = token_offset >= 800;
        block_offset = token_offset - logical_block * 800;
      } else if constexpr (BLOCK_SIZE == 1568) {
        logical_block = token_offset / 1568;
        block_offset = token_offset - logical_block * 1568;
      } else if constexpr (BLOCK_SIZE == 1648) {
        logical_block = token_offset / 1648;
        block_offset = token_offset - logical_block * 1648;
      } else if constexpr (BLOCK_SIZE == 3296) {
        logical_block = token_offset / 3296;
        block_offset = token_offset - logical_block * 3296;
      } else {
        logical_block = token_offset / block_size;
        block_offset = token_offset % block_size;
      }
      const int physical_block = page_ids[logical_block];
      int64_t physical_offset;
      if constexpr (CONTIGUOUS_HKV1_LAYOUT) {
        constexpr int64_t kHeadDim = 256;
        constexpr int64_t kPhysicalBlockStride =
            BLOCK_SIZE == 16 ? 16 * kHeadDim : 2 * BLOCK_SIZE * kHeadDim;
        physical_offset =
            static_cast<int64_t>(physical_block) * kPhysicalBlockStride +
            static_cast<int64_t>(block_offset) * kHeadDim + panel_offset;
      } else {
        physical_offset = static_cast<int64_t>(physical_block) * block_stride +
                          static_cast<int64_t>(block_offset) * token_stride +
                          static_cast<int64_t>(kv_head_idx) * head_stride +
                          panel_offset;
      }
      const uint4 raw = __ldg(reinterpret_cast<const uint4*>(kv_cache) +
                              physical_offset / 16 + vec_pair);
      const int shared_offset = row * kv_smem_stride_uint4 + vec_pair * 2;
      const uint64_t raw_lo =
          static_cast<uint64_t>(raw.x) | (static_cast<uint64_t>(raw.y) << 32);
      shared_vec[shared_offset] =
          KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3
              ? fp8_e4m3fn_vector_to_half8_fast(raw_lo)
              : fp8_e5m2_vector_to_half8(raw_lo);
      const uint64_t raw_hi =
          static_cast<uint64_t>(raw.z) | (static_cast<uint64_t>(raw.w) << 32);
      shared_vec[shared_offset + 1] =
          KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3
              ? fp8_e4m3fn_vector_to_half8_fast(raw_hi)
              : fp8_e5m2_vector_to_half8(raw_hi);
    }
  } else {
    const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
    for (int copy_idx = copy_thread_idx; copy_idx < copy_count;
         copy_idx += NUM_THREADS) {
      const int row = copy_idx / panel_d_stride_uint4;
      const int vec_col = copy_idx % panel_d_stride_uint4;
      shared_vec[row * kv_smem_stride_uint4 + vec_col] =
          load_xqa_tc_kv_vector<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                                E4M3_SHARED_LUT>(
              kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
              tile_page_offset, kv_tile_start, block_size, kv_head_idx,
              block_stride, token_stride, head_stride, panel_offset, e4m3_lut);
    }
  }
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16,
          bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ void load_xqa_tc_kv_panel_and_zero(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx,
    const uint16_t* __restrict__ e4m3_lut = nullptr) {
  const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  constexpr int kLoadStages = 4;
  for (int copy_base = copy_thread_idx; copy_base < copy_count;
       copy_base += NUM_THREADS * kLoadStages) {
    uint4 staged[kLoadStages];
#pragma unroll
    for (int stage = 0; stage < kLoadStages; ++stage) {
      const int copy_idx = copy_base + stage * NUM_THREADS;
      if (copy_idx < copy_count) {
        staged[stage] =
            load_xqa_tc_kv_vector<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                                  E4M3_SHARED_LUT>(
                kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
                tile_page_offset, kv_tile_start, block_size, kv_head_idx,
                block_stride, token_stride, head_stride, panel_offset,
                e4m3_lut);
      }
    }
#pragma unroll
    for (int stage = 0; stage < kLoadStages; ++stage) {
      const int copy_idx = copy_base + stage * NUM_THREADS;
      if (copy_idx < copy_count) {
        const int row = copy_idx / panel_d_stride_uint4;
        const int vec_col = copy_idx % panel_d_stride_uint4;
        shared_vec[row * kv_smem_stride_uint4 + vec_col] = staged[stage];
      }
    }
  }
  for (int copy_idx = copy_thread_idx + copy_count;
       copy_idx < kXQATCBlockN * panel_d_stride_uint4;
       copy_idx += NUM_THREADS) {
    const int row = copy_idx / panel_d_stride_uint4;
    const int vec_col = copy_idx % panel_d_stride_uint4;
    shared_vec[row * kv_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
}

template <int D>
__device__ __forceinline__ float dot_qk_half2(const __half* __restrict__ q_ptr,
                                              const __half* __restrict__ k_ptr,
                                              const int lane) {
  static_assert(D % 2 == 0, "Head dim must be even for half2 dot");
  const __half2* q_ptr2 = reinterpret_cast<const __half2*>(q_ptr);
  const __half2* k_ptr2 = reinterpret_cast<const __half2*>(k_ptr);

  float acc = 0.f;
#pragma unroll
  for (int i = lane; i < D / 2; i += kWarpSize) {
    const float2 qv = __half22float2(q_ptr2[i]);
    const float2 kv = __half22float2(k_ptr2[i]);
    acc = fmaf(qv.x, kv.x, acc);
    acc = fmaf(qv.y, kv.y, acc);
  }
  return warp_reduce_sum(acc);
}

template <int D, int KV_DTYPE>
__device__ __forceinline__ float dot_qk_cache(const __half* __restrict__ q_ptr,
                                              const void* __restrict__ k_cache,
                                              const int64_t k_index_base,
                                              const int lane) {
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    const __half* k_ptr =
        reinterpret_cast<const __half*>(k_cache) + k_index_base;
    return dot_qk_half2<D>(q_ptr, k_ptr, lane);
  } else if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2) {
    static_assert(D % 2 == 0, "Head dim must be even for e5m2 half2 dot");
    const __half2* q_ptr2 = reinterpret_cast<const __half2*>(q_ptr);
    float acc = 0.f;
#pragma unroll
    for (int i = lane; i < D / 2; i += kWarpSize) {
      const float2 qv = __half22float2(q_ptr2[i]);
      const __half2 k_h2 = flash_v100::load_fp8_e5m2_half2_unscaled(
          k_cache, k_index_base + static_cast<int64_t>(i) * 2);
      const float2 kv = __half22float2(k_h2);
      acc = fmaf(qv.x, kv.x, acc);
      acc = fmaf(qv.y, kv.y, acc);
    }
    return warp_reduce_sum(acc);
  } else {
    float acc = 0.f;
#pragma unroll
    for (int d = lane; d < D; d += kWarpSize) {
      const float qv = __half2float(q_ptr[d]);
      const float kv = flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(
          k_cache, k_index_base + d);
      acc = fmaf(qv, kv, acc);
    }
    return warp_reduce_sum(acc);
  }
}

template <int D, int PARTITION_SIZE, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool ANCHORED_SWA = false,
          typename PARTIAL_T = __half>
__global__ void flash_attention_decode_partition_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, PARTIAL_T* __restrict__ tmp_out,
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions, const int batch_size,
    const int max_num_blocks, const int max_num_partitions,
    const int num_heads_q, const int num_heads_kv, const int block_size,
    const int64_t q_stride0, const int64_t q_stride1,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t k_block_stride,
    const int64_t k_token_stride, const int64_t k_head_stride,
    const int64_t v_block_stride, const int64_t v_token_stride,
    const int64_t v_head_stride, const float softmax_scale, const float k_scale,
    const float v_scale, const int window_size_left,
    const int window_size_right, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final,
    const int* __restrict__ anchor_lens, const int anchored_window) {
  static_assert(D == 256 && PARTITION_SIZE == 1024 &&
      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
      SEQ_LEN_ROUTE == 0 && !ANCHORED_SWA &&
      std::is_same_v<PARTIAL_T, float>, "Private scalar six-head contract");
  const int partition_idx = blockIdx.z;
  const int seq_len = seq_lens[0];
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len ||
      partition_idx >= max_num_partitions) return;
  const int effective_num_partitions = min(max_num_partitions,
      max(active_num_partitions[0], (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE));
  if (partition_idx >= effective_num_partitions) return;
  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;
  const float score_scale = softmax_scale * k_scale;
  __shared__ float kv_lut[256];
  kv_lut[threadIdx.x] = flash_v100::fp8_e4m3fn_to_float(
      static_cast<uint8_t>(threadIdx.x));
  __shared__ __half q_shared[6][D];
  __shared__ float scores_shared[6][PARTITION_SIZE];
  __shared__ int block_idx_shared[2];
  const int first_block = start_token_idx / block_size;
  const int first_offset = start_token_idx - first_block * block_size;
  const int first_page_tokens = min(part_tokens, block_size - first_offset);
  for (int i = threadIdx.x; i < 6 * D; i += blockDim.x)
    q_shared[i / D][i % D] = q[(i / D) * q_stride1 + i % D];
  if (threadIdx.x == 0) {
    block_idx_shared[0] = block_table[first_block];
    block_idx_shared[1] = first_page_tokens < part_tokens
        ? block_table[first_block + 1] : block_idx_shared[0];
  }
  __syncthreads();
  float local_max[6];
#pragma unroll
  for (int head = 0; head < 6; ++head) local_max[head] = -1.0e20f;
  for (int token_local = warp_idx; token_local < part_tokens;
       token_local += kWarpsPerBlock) {
    const bool second_page = token_local >= first_page_tokens;
    const int token_offset = second_page ? token_local - first_page_tokens
                                         : first_offset + token_local;
    const int64_t k_index =
        static_cast<int64_t>(block_idx_shared[second_page]) * k_block_stride +
        static_cast<int64_t>(token_offset) * k_token_stride;
    float score[6] = {};
    // Each head retains d=lane,lane+32,... and the original warp reduction.
    // Only independent heads share the decoded K value.
#pragma unroll
    for (int d = lane; d < D; d += kWarpSize) {
      const float kv = kv_lut[static_cast<const uint8_t*>(k_cache)[k_index + d]];
#pragma unroll
      for (int head = 0; head < 6; ++head)
        score[head] = fmaf(__half2float(q_shared[head][d]), kv, score[head]);
    }
#pragma unroll
    for (int head = 0; head < 6; ++head) {
      const float reduced = warp_reduce_sum(score[head]);
      if (lane == 0) {
        const float value = reduced * score_scale;
        scores_shared[head][token_local] = value;
        local_max[head] = fmaxf(local_max[head], value);
      }
    }
  }
  float inv_sum[6];
#pragma unroll
  for (int head = 0; head < 6; ++head) {
    const float part_max = block_reduce_max<kWarpsPerBlock>(local_max[head]);
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
      const float p = __expf(scores_shared[head][i] - part_max);
      scores_shared[head][i] = p;
      local_sum += p;
    }
    const float part_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
    inv_sum[head] = part_sum > 0.f ? 1.f / part_sum : 0.f;
    if (threadIdx.x == 0) {
      const int64_t stats_index = head * stats_stride1 + partition_idx;
      max_logits[stats_index] = part_max;
      exp_sums[stats_index] = part_sum;
    }
    // Complete reads of the reduction helper's shared result before the
    // next head reuses it; all 256 threads participate in every reduction.
    __syncthreads();
  }
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc[6] = {};
    for (int page = 0; page < 2; ++page) {
      const int begin = page == 0 ? 0 : first_page_tokens;
      const int end = page == 0 ? first_page_tokens : part_tokens;
      int64_t v_index = static_cast<int64_t>(block_idx_shared[page]) *
          v_block_stride + (page == 0 ? first_offset * v_token_stride : 0) + d;
      for (int i = begin; i < end; ++i, v_index += v_token_stride) {
        const float vv =
            kv_lut[static_cast<const uint8_t*>(v_cache)[v_index]];
#pragma unroll
        for (int head = 0; head < 6; ++head)
          acc[head] = fmaf(scores_shared[head][i], vv, acc[head]);
      }
    }
#pragma unroll
    for (int head = 0; head < 6; ++head) {
      const float out_scale = inv_sum[head] * v_scale;
      const int64_t tmp_out_base = head * tmp_out_stride1 +
          static_cast<int64_t>(partition_idx) * tmp_out_stride2;
      tmp_out[tmp_out_base + d] = acc[head] * out_scale;
    }
  }
}

template <int D, int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens,
          typename PARTIAL_T = __half>
__global__ void flash_attention_decode_reduce_kernel(
    const PARTIAL_T* __restrict__ tmp_out, const float* __restrict__ max_logits,
    const float* __restrict__ exp_sums, const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions, __half* __restrict__ out,
    const int batch_size, const int max_num_partitions, const int num_heads_q,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t out_stride0,
    const int64_t out_stride1, const int route_partition_size_begin,
    const int route_seq_len_begin, const int route_seq_len_end,
    const int route_seq_len_final) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;

  if (batch_idx >= batch_size || head_idx >= num_heads_q) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  int partition_size;
  if constexpr (PARTITION_SIZE == -1) {
    static_assert(SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens,
                  "Runtime wave partitions require the wave-long route");
    partition_size = seq_len < route_seq_len_end
                         ? 512
                         : (seq_len < route_seq_len_final ? 896 : 1664);
  } else if constexpr (PARTITION_SIZE == 0) {
    partition_size = seq_len < route_partition_size_begin ? 64 : 256;
  } else {
    partition_size = PARTITION_SIZE;
  }
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  (void)active_num_partitions;

  if (seq_len <= 0 || num_partitions <= 0) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      out[static_cast<int64_t>(batch_idx) * out_stride0 +
          static_cast<int64_t>(head_idx) * out_stride1 + d] = __float2half(0.f);
    }
    return;
  }

  extern __shared__ float shared_mem[];
  float* max_shared = shared_mem;
  float* weight_shared = shared_mem + max_num_partitions;

  float local_max = -1.0e20f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + i;
    const float m = max_logits[stats_index];
    max_shared[i] = m;
    local_max = fmaxf(local_max, m);
  }
  const float global_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + i;
    const float weight =
        exp_sums[stats_index] * __expf(max_shared[i] - global_max);
    weight_shared[i] = weight;
    local_sum += weight;
  }
  const float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
  __syncthreads();

  const int64_t out_base = static_cast<int64_t>(batch_idx) * out_stride0 +
                           static_cast<int64_t>(head_idx) * out_stride1;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < num_partitions; ++i) {
      acc = fmaf(weight_shared[i],
                 static_cast<float>(
                     tmp_out[tmp_out_base +
                             static_cast<int64_t>(i) * tmp_out_stride2 + d]),
                 acc);
    }
    out[out_base + d] = __float2half(acc * inv_global_sum);
  }
}

}  // namespace
at::Tensor scalar_attention_candidate(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, const at::Tensor& table, const at::Tensor& lengths,
    at::Tensor& partial, at::Tensor& maximum, at::Tensor& sums,
    const at::Tensor& active, float scale, float k_scale, float v_scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf &&
              q.sizes() == at::IntArrayRef({1, 6, 256}) && q.is_contiguous(),
              "Private scalar q1 probe requires contiguous FP16 [1,6,256]");
  TORCH_CHECK(k.dim() == 4 && k.size(2) == 1 && k.size(3) == 256 &&
              k.size(1) > 0 && k.scalar_type() == at::kByte &&
              v.sizes() == k.sizes() && v.scalar_type() == at::kByte,
              "Private scalar q1 probe requires E4M3 [pages,page,1,256]");
  for (const auto* t : {&k, &v})
    TORCH_CHECK(t->stride(3) == 1, "Head dimension must be contiguous");
  TORCH_CHECK(out.sizes() == q.sizes() && out.scalar_type() == at::kHalf &&
              out.is_contiguous(), "Output shape/dtype mismatch");
  TORCH_CHECK(table.dim() == 2 && table.size(0) == 1 && table.size(1) > 0 &&
              table.scalar_type() == at::kInt && table.is_contiguous() &&
              table.size(1) * k.size(1) <= 266240 &&
              lengths.sizes() == at::IntArrayRef({1}) &&
              lengths.scalar_type() == at::kInt && lengths.is_contiguous() &&
              active.sizes() == at::IntArrayRef({1}) &&
              active.scalar_type() == at::kInt && active.is_contiguous(),
              "Invalid scalar q1 page/length metadata");
  TORCH_CHECK(partial.sizes() == at::IntArrayRef({1, 6, 256, 256}) &&
              partial.scalar_type() == at::kFloat && partial.is_contiguous() &&
              maximum.sizes() == at::IntArrayRef({1, 6, 256}) &&
              sums.sizes() == maximum.sizes() &&
              maximum.scalar_type() == at::kFloat &&
              sums.scalar_type() == at::kFloat &&
              maximum.is_contiguous() && sums.is_contiguous(),
              "Scalar q1 requires unchanged FP32 partition workspaces");
  for (const auto* t : {&k, &v, static_cast<const at::Tensor*>(&out), &table,
      &lengths, static_cast<const at::Tensor*>(&partial),
      static_cast<const at::Tensor*>(&maximum),
      static_cast<const at::Tensor*>(&sums), &active})
    TORCH_CHECK(t->device() == q.device(), "Device mismatch");
  TORCH_CHECK(std::isfinite(scale) && std::isfinite(k_scale) &&
              std::isfinite(v_scale) && k_scale > 0 && v_scale > 0,
              "Finite positive KV scales required");
  c10::cuda::CUDAGuard guard(q.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0, "SM70 only");
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  flash_attention_decode_partition_kernel<256, 1024,
      flash_v100::KV_CACHE_DTYPE_FP8_E4M3, 0, false, float>
      <<<dim3(1, 1, 256), 256, 4096, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()), k.data_ptr(), v.data_ptr(),
      partial.data_ptr<float>(), maximum.data_ptr<float>(), sums.data_ptr<float>(),
      table.data_ptr<int>(), lengths.data_ptr<int>(), active.data_ptr<int>(),
      1, table.size(1), 256, 6, 1, k.size(1), q.stride(0), q.stride(1),
      partial.stride(0), partial.stride(1), partial.stride(2),
      maximum.stride(0), maximum.stride(1),
      k.stride(0), k.stride(1), k.stride(2),
      v.stride(0), v.stride(1), v.stride(2),
      scale, k_scale, v_scale, -1, -1, 0, 0, 0, nullptr, 0);
  flash_attention_decode_reduce_kernel<256, 1024, 0, float>
      <<<dim3(1, 6, 1), 256, 2 * 256 * sizeof(float), stream>>>(
      partial.data_ptr<float>(), maximum.data_ptr<float>(), sums.data_ptr<float>(),
      lengths.data_ptr<int>(), active.data_ptr<int>(),
      reinterpret_cast<__half*>(out.data_ptr()), 1, 256, 6,
      partial.stride(0), partial.stride(1), partial.stride(2),
      maximum.stride(0), maximum.stride(1), out.stride(0), out.stride(1),
      0, 0, 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
// Registered into the shipped FA2 extension rather than a private pybind
// module, so the compact scalar long-context tail is available by default
// instead of requiring an externally built DSO selected by a manifest.
namespace {
at::Tensor sm70_scalar_attention_entry(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, const at::Tensor& table, const at::Tensor& lengths,
    at::Tensor& partial, at::Tensor& maximum, at::Tensor& sums,
    const at::Tensor& active, double scale, double k_scale, double v_scale) {
  return scalar_attention_candidate(
      q, k, v, out, table, lengths, partial, maximum, sums, active,
      static_cast<float>(scale), static_cast<float>(k_scale),
      static_cast<float>(v_scale));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_vllm_fa2_C, ops) {
  ops.def(
      "sm70_scalar_attention_fwd(Tensor q, Tensor k, Tensor v, Tensor(a!) out, "
      "Tensor table, Tensor lengths, Tensor(a!) partial, Tensor(a!) maximum, "
      "Tensor(a!) sums, Tensor active, float scale, float k_scale, "
      "float v_scale) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(_vllm_fa2_C, CUDA, ops) {
  ops.impl("sm70_scalar_attention_fwd", &sm70_scalar_attention_entry);
}
