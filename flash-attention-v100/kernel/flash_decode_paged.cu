#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <string>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool ANCHORED_SWA = false>
__global__ void flash_attention_decode_partition_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, __half* __restrict__ tmp_out,
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
  // The anchored decode-window mask variant is only generated for the fp16-KV
  // configuration; every other instantiation keeps its original code path.
  static_assert(!ANCHORED_SWA || KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16,
                "ANCHORED_SWA requires an fp16 KV cache");
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || head_idx >= num_heads_q ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int query_pos = seq_len - 1;
  const int min_token_idx =
      window_size_left >= 0 ? max(0, query_pos - window_size_left) : 0;
  const int max_token_idx =
      window_size_right >= 0 ? min(seq_len - 1, query_pos + window_size_right)
                             : seq_len - 1;
  const int part_start = max(start_token_idx, min_token_idx);
  const int part_end = min(start_token_idx + PARTITION_SIZE, max_token_idx + 1);

  // Anchored decode window: keys below the per-request prompt length stay
  // globally visible; later (generated) keys must fall inside the sliding
  // window over generated positions. Keys inside [gap_lo, gap_hi) are masked.
  int gap_lo = seq_len;
  int gap_hi = seq_len;
  if constexpr (ANCHORED_SWA) {
    const int anchor_len = anchor_lens[batch_idx];
    gap_lo = anchor_len;
    gap_hi = max(anchor_len, query_pos - anchored_window + 1);
  }
  const int q_per_kv = num_heads_q / num_heads_kv;
  const int kv_head_idx = head_idx / q_per_kv;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;
  const float score_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                ? softmax_scale
                                : softmax_scale * k_scale;

  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1 +
      static_cast<int64_t>(partition_idx) * tmp_out_stride2;
  if (part_start >= part_end) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      tmp_out[tmp_out_base + d] = __float2half(0.f);
    }
    if (threadIdx.x == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = -1.0e20f;
      exp_sums[stats_index] = 0.f;
    }
    return;
  }

  const int part_tokens = part_end - part_start;

  __shared__ __half q_shared[D];
  __shared__ float scores_shared[PARTITION_SIZE];
  __shared__ int block_idx_shared[PARTITION_SIZE];
  __shared__ int block_offset_shared[PARTITION_SIZE];

  const __half* q_ptr = q + static_cast<int64_t>(batch_idx) * q_stride0 +
                        static_cast<int64_t>(head_idx) * q_stride1;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    q_shared[d] = q_ptr[d];
  }
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    const int token_idx = part_start + i;
    const int logical_block = token_idx / block_size;
    block_idx_shared[i] =
        block_table[batch_idx * max_num_blocks + logical_block];
    block_offset_shared[i] = token_idx - logical_block * block_size;
  }
  __syncthreads();

  float local_max = -1.0e20f;
  for (int token_local = warp_idx; token_local < part_tokens;
       token_local += kWarpsPerBlock) {
    const int physical_block = block_idx_shared[token_local];
    const int block_offset = block_offset_shared[token_local];
    const int64_t k_index =
        static_cast<int64_t>(physical_block) * k_block_stride +
        static_cast<int64_t>(block_offset) * k_token_stride +
        static_cast<int64_t>(kv_head_idx) * k_head_stride;

    float score = dot_qk_cache<D, KV_DTYPE>(q_shared, k_cache, k_index, lane);
    if (lane == 0) {
      if constexpr (ANCHORED_SWA) {
        const int token_idx = part_start + token_local;
        if (token_idx >= gap_lo && token_idx < gap_hi) {
          // Masked gap key: never contributes (nulled gap blocks included).
          scores_shared[token_local] = -1.0e30f;
        } else {
          score *= score_scale;
          scores_shared[token_local] = score;
          local_max = fmaxf(local_max, score);
        }
      } else {
        score *= score_scale;
        scores_shared[token_local] = score;
        local_max = fmaxf(local_max, score);
      }
    }
  }

  const float part_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    if constexpr (ANCHORED_SWA) {
      const int token_idx = part_start + i;
      if (token_idx >= gap_lo && token_idx < gap_hi) {
        // Exact zero weight for masked keys (no reliance on expf underflow).
        scores_shared[i] = 0.f;
        continue;
      }
    }
    const float p = __expf(scores_shared[i] - part_max);
    scores_shared[i] = p;
    local_sum += p;
  }
  const float part_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  const float inv_part_sum = part_sum > 0.f ? 1.f / part_sum : 0.f;
  __syncthreads();

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < part_tokens; ++i) {
      const int physical_block = block_idx_shared[i];
      const int block_offset = block_offset_shared[i];
      const int64_t v_index =
          static_cast<int64_t>(physical_block) * v_block_stride +
          static_cast<int64_t>(block_offset) * v_token_stride +
          static_cast<int64_t>(kv_head_idx) * v_head_stride + d;
      const float vv =
          flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(v_cache, v_index);
      acc = fmaf(scores_shared[i], vv, acc);
    }
    const float out_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                ? inv_part_sum
                                : inv_part_sum * v_scale;
    tmp_out[tmp_out_base + d] = __float2half(acc * out_scale);
  }

  if (threadIdx.x == 0) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
    max_logits[stats_index] = part_max;
    exp_sums[stats_index] = part_sum;
  }
}

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM, int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          bool ALIGNED_PADDED_SMEM, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool QK_SW_PIPELINE = false,
          bool PARTITION_PAGE_IDS = false, bool FP8_PAIR_LOAD = false,
          bool E4M3_SHARED_LUT = false>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_partition_kernel_256_wide(
        const __half* __restrict__ q, const void* __restrict__ k_cache,
        const void* __restrict__ v_cache, __half* __restrict__ tmp_out,
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
        const int64_t v_head_stride, const float softmax_scale,
        const float k_scale, const float v_scale, const int route_seq_len_begin,
        const int route_seq_len_end, const int route_seq_len_final) {
  constexpr int D = 256;
  constexpr int WMMA_M = 8;
  constexpr int WMMA_N = 32;
  constexpr int WMMA_K = 16;
  constexpr int kPVPanelDim = kXQATCStride;
  constexpr int kNumPVPanels = D / kPVPanelDim;
  constexpr int kQKPanelDim =
      QK_SW_PIPELINE ? kXQATCQKPipelinePanelDim : kXQATCStride;
  constexpr int kNumQKPanels = D / kQKPanelDim;
  using SmemLayout = std::conditional_t<
      QK_SW_PIPELINE,
      XQATCQKPipelineSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>,
      XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>>;
  // Keep softmax P in fp32 through PV for fp16 KV, avoiding a half round-trip.
  // Preserve the half-P path for fp8_e5m2 so quantized-cache behavior remains
  // bit-exact. This branch is resolved entirely at compile time.
  constexpr bool kKeepPfp32 = (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16);
  constexpr int q_global_stride_uint4 = D / 8;
  constexpr int q_smem_stride_uint4 = SmemLayout::kQStride / 8;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int pv_panel_d_stride_uint4 = kPVPanelDim / 8;
  constexpr int qk_smem_stride_uint4 =
      QK_SW_PIPELINE ? SmemLayout::kQKStride / 8 : kv_smem_stride_uint4;
  constexpr int qk_panel_d_stride_uint4 = kQKPanelDim / 8;
  constexpr int kAccumsPerThread = D / kWarpSize;
  constexpr bool kPVHalf2 = CONTIGUOUS_HKV1_LAYOUT &&
                            (BLOCK_SIZE == 800 || BLOCK_SIZE == 1568) &&
                            KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3;
  static_assert(GROUP_SIZE == 4 || GROUP_SIZE == 6 || GROUP_SIZE == 8,
                "Wide D=256 TC XQA kernel supports q_per_kv in {4, 6, 8}");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one full softmax/PV warp");
  static_assert(
      !QK_SW_PIPELINE || (GROUP_SIZE == 6 && (NUM_THREADS == 6 * kWarpSize ||
                                              NUM_THREADS == 8 * kWarpSize)),
      "The QK software pipeline requires a six- or eight-warp G6 "
      "kernel");
  static_assert(
      !E4M3_SHARED_LUT || KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3,
      "The shared conversion LUT is only valid for E4M3 KV");
  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
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
    const int partition_size_begin =
        route_seq_len_end > 0 ? route_seq_len_end : route_seq_len_begin;
    partition_size = seq_len < partition_size_begin ? 64 : 256;
  } else {
    partition_size = PARTITION_SIZE;
  }
  const int start_token_idx = partition_idx * partition_size;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + partition_size - 1) / partition_size;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(partition_size, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  uint16_t* e4m3_lut = nullptr;
  if constexpr (E4M3_SHARED_LUT) {
    e4m3_lut = reinterpret_cast<uint16_t*>(smem_raw + sizeof(SmemLayout));
    for (int raw = tid; raw < 256; raw += NUM_THREADS) {
      e4m3_lut[raw] = fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw));
    }
  }
  __half* sQ = smem.q;
  __half* sK = smem.k_buffer(0);
  __half* sV = smem.v_buffer();
  float* sS = smem.reuse_sp.s;
  __half* sP = smem.reuse_sp.p;
  float row_max_reg = kXQANegInf;
  float row_sum_reg = 0.f;
  float out_acc[kAccumsPerThread];
#pragma unroll
  for (int i = 0; i < kAccumsPerThread; ++i) {
    out_acc[i] = 0.f;
  }

  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
  for (int idx = tid; idx < GROUP_SIZE * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    const int64_t q_offset =
        static_cast<int64_t>(batch_idx) * q_stride0 +
        static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid;
       idx < (kXQATC256WideBlockM - GROUP_SIZE) * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = GROUP_SIZE + idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  int partition_start_page = 0;
  int partition_page_offset = 0;
  if constexpr (PARTITION_PAGE_IDS) {
    int partition_page_count;
    if constexpr (BLOCK_SIZE == 800) {
      partition_start_page = start_token_idx / 800;
      partition_page_offset = start_token_idx - partition_start_page * 800;
      partition_page_count = 1 + (partition_page_offset + part_tokens > 800);
    } else {
      partition_start_page = start_token_idx / block_size;
      partition_page_offset =
          start_token_idx - partition_start_page * block_size;
      partition_page_count =
          (partition_page_offset + part_tokens + block_size - 1) / block_size;
    }
    for (int idx = tid; idx < partition_page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[partition_start_page + idx]);
    }
    __syncthreads();
  }

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    int start_page = partition_start_page;
    int tile_page_offset;
    int page_count = 0;
    if constexpr (PARTITION_PAGE_IDS) {
      tile_page_offset = partition_page_offset + block_n * kXQATCBlockN;
    } else if constexpr (BLOCK_SIZE == 16) {
      start_page = tile_token_start >> 4;
      tile_page_offset = tile_token_start & 15;
      page_count = (tile_page_offset + valid_k_rows + 15) >> 4;
    } else if constexpr (BLOCK_SIZE == 784) {
      start_page = tile_token_start / 784;
      tile_page_offset = tile_token_start - start_page * 784;
      page_count = (tile_page_offset + valid_k_rows + 783) / 784;
    } else if constexpr (BLOCK_SIZE == 800) {
      start_page = tile_token_start / 800;
      tile_page_offset = tile_token_start - start_page * 800;
      page_count = 1 + (tile_page_offset + valid_k_rows > 800);
    } else if constexpr (BLOCK_SIZE == 1568) {
      start_page = tile_token_start / 1568;
      tile_page_offset = tile_token_start - start_page * 1568;
      page_count = (tile_page_offset + valid_k_rows + 1567) / 1568;
    } else {
      start_page = tile_token_start / block_size;
      tile_page_offset = tile_token_start - start_page * block_size;
      page_count =
          (tile_page_offset + valid_k_rows + block_size - 1) / block_size;
    }

    if constexpr (!PARTITION_PAGE_IDS) {
      for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
        smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
      }
      __syncthreads();
    }

    for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
         kv_tile_start += kXQATCBlockN) {
      const int valid_kv_tile_rows =
          min(kXQATCBlockN, valid_k_rows - kv_tile_start);
      // The QK fragments die before softmax/PV. This permits the compiler to
      // reuse their registers for the memory-latency-sensitive PV phase.
      volta::fragment<volta::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::row_major>
          qk_a_frag;
      volta::fragment<volta::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::col_major>
          qk_b_frag;
      volta::fragment<volta::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
          qk_acc_frag;
      if (warp_id < (kXQATCBlockN / WMMA_N)) {
        volta::fill_fragment(qk_acc_frag, 0.0f);
      }

      if constexpr (QK_SW_PIPELINE) {
        constexpr int kConsumerWarps = kXQATCBlockN / WMMA_N;
        constexpr int kProducerThreads =
            NUM_THREADS - kConsumerWarps * kWarpSize;
        const int producer_tid = tid - kConsumerWarps * kWarpSize;
        if (producer_tid >= 0) {
          load_xqa_tc_kv_panel_and_zero<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                                        kProducerThreads, KV_DTYPE,
                                        E4M3_SHARED_LUT>(
              smem.k_buffer(0), k_cache, smem.page_ids, valid_kv_tile_rows,
              qk_panel_d_stride_uint4, qk_smem_stride_uint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, 0, producer_tid, e4m3_lut);
        }
        __syncthreads();

#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          if (producer_tid >= 0 && panel_idx + 1 < kNumQKPanels) {
            load_xqa_tc_kv_panel_and_zero<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                                          kProducerThreads, KV_DTYPE,
                                          E4M3_SHARED_LUT>(
                smem.k_buffer((panel_idx + 1) & 1), k_cache, smem.page_ids,
                valid_kv_tile_rows, qk_panel_d_stride_uint4,
                qk_smem_stride_uint4, tile_page_offset, kv_tile_start,
                block_size, kv_head_idx, k_block_stride, k_token_stride,
                k_head_stride, panel_offset + kQKPanelDim, producer_tid,
                e4m3_lut);
          }
          if (warp_id < kConsumerWarps) {
            const int tile_n = warp_id * WMMA_N;
            const __half* sK_panel = smem.k_buffer(panel_idx & 1);
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / WMMA_K); ++k_tile) {
              const int k_offset = k_tile * WMMA_K;
              volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                      SmemLayout::kQStride);
              volta::load_matrix_sync(
                  qk_b_frag,
                  sK_panel + tile_n * SmemLayout::kQKStride + k_offset,
                  SmemLayout::kQKStride);
              volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      } else {
#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                               KV_DTYPE, FP8_PAIR_LOAD, E4M3_SHARED_LUT>(
              sK, k_cache, smem.page_ids, valid_kv_tile_rows,
              qk_panel_d_stride_uint4, qk_smem_stride_uint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, panel_offset, threadIdx.x,
              e4m3_lut);
          for (int idx = tid + valid_kv_tile_rows * qk_panel_d_stride_uint4;
               idx < kXQATCBlockN * qk_panel_d_stride_uint4;
               idx += NUM_THREADS) {
            const int row = idx / qk_panel_d_stride_uint4;
            const int vec_col = idx % qk_panel_d_stride_uint4;
            reinterpret_cast<uint4*>(sK)[row * qk_smem_stride_uint4 + vec_col] =
                make_uint4(0, 0, 0, 0);
          }
          __syncthreads();

          if (warp_id < (kXQATCBlockN / WMMA_N)) {
            const int tile_n = warp_id * WMMA_N;
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / WMMA_K); ++k_tile) {
              const int k_offset = k_tile * WMMA_K;
              volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                      SmemLayout::kQStride);
              volta::load_matrix_sync(
                  qk_b_frag, sK + tile_n * SmemLayout::kQKStride + k_offset,
                  SmemLayout::kQKStride);
              volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      }

      if (warp_id < (kXQATCBlockN / WMMA_N)) {
#pragma unroll
        for (int i = 0; i < qk_acc_frag.num_elements; ++i) {
          qk_acc_frag.x[i] *= KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                  ? softmax_scale
                                  : softmax_scale * k_scale;
        }
        volta::store_matrix_sync(sS + kv_tile_start + warp_id * WMMA_N,
                                 qk_acc_frag, kXQATCBlockN,
                                 volta::mem_row_major);
      }
      __syncthreads();
    }

    if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      const unsigned mask = 0xffffffffu;
      float* sS_row_f = sS + row * kXQATCBlockN;
      __half* sP_row_h = sP + row * kXQATCBlockN;
      float* sP_row_f = sS + row * kXQATCBlockN;
      const int vec_cols = valid_k_rows >> 2;
      const int tail_start = vec_cols << 2;
      const int vec_col = thread_in_row;

      float thread_max = kXQANegInf;
      __half2 packed_exp0 = __float22half2_rn(make_float2(0.f, 0.f));
      __half2 packed_exp1 = __float22half2_rn(make_float2(0.f, 0.f));
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        thread_max =
            fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < valid_k_rows;
           c += kXQATC256WideThreadsPerRow) {
        thread_max = fmaxf(thread_max, sS_row_f[c]);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_max =
            fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
      }

      const float row_max = __shfl_sync(mask, thread_max, 0, kWarpSize);
      const float old_max = __shfl_sync(mask, row_max_reg, 0, kWarpSize);
      const float new_max = fmaxf(old_max, row_max);
      const float exp_diff = __expf(old_max - new_max);

      float thread_sum = 0.f;
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        const float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
        const float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
        const float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
        const float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
        thread_sum += (e0 + e1) + (e2 + e3);
        if constexpr (kKeepPfp32) {
          reinterpret_cast<float4*>(sP_row_f)[vec_col] =
              make_float4(e0, e1, e2, e3);
        } else {
          packed_exp0 = __float22half2_rn(make_float2(e0, e1));
          packed_exp1 = __float22half2_rn(make_float2(e2, e3));
        }
      }

#pragma unroll
      for (int c = tail_start + thread_in_row; c < kXQATCBlockN;
           c += kXQATC256WideThreadsPerRow) {
        const float v = (c < valid_k_rows) ? sS_row_f[c] : kXQANegInf;
        const float e = __expf(fmaxf(v - new_max, -80.0f));
        thread_sum += (c < valid_k_rows) ? e : 0.0f;
        if constexpr (kKeepPfp32) {
          sP_row_f[c] = (c < valid_k_rows) ? e : 0.0f;
        } else {
          sP_row_h[c] =
              (c < valid_k_rows) ? __float2half_rn(e) : __float2half(0.f);
        }
      }

#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, o, kWarpSize);
      }

      const float row_sum = __shfl_sync(mask, thread_sum, 0, kWarpSize);
      const float old_sum = __shfl_sync(mask, row_sum_reg, 0, kWarpSize);

      if (thread_in_row == 0) {
        row_sum_reg = exp_diff * old_sum + row_sum;
        row_max_reg = new_max;
      }
      if constexpr (!kKeepPfp32) {
        __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);
        if (vec_col < vec_cols) {
          const int base_offset = vec_col * 2;
          sP_half2[base_offset] = packed_exp0;
          sP_half2[base_offset + 1] = packed_exp1;
        }
      }

      if (block_n > 0) {
#pragma unroll
        for (int i = 0; i < kAccumsPerThread; ++i) {
          out_acc[i] *= exp_diff;
        }
      }
    }
    __syncthreads();

    for (int panel_idx = 0; panel_idx < kNumPVPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPVPanelDim;
      for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
           kv_tile_start += kXQATCBlockN) {
        const int valid_kv_tile_rows =
            min(kXQATCBlockN, valid_k_rows - kv_tile_start);
        load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                             KV_DTYPE, FP8_PAIR_LOAD, E4M3_SHARED_LUT>(
            sV, v_cache, smem.page_ids, valid_kv_tile_rows,
            pv_panel_d_stride_uint4, kv_smem_stride_uint4, tile_page_offset,
            kv_tile_start, block_size, kv_head_idx, v_block_stride,
            v_token_stride, v_head_stride, panel_offset, threadIdx.x, e4m3_lut);
        for (int idx = tid + valid_kv_tile_rows * pv_panel_d_stride_uint4;
             idx < kXQATCBlockN * pv_panel_d_stride_uint4; idx += NUM_THREADS) {
          const int row = idx / pv_panel_d_stride_uint4;
          const int vec_col = idx % pv_panel_d_stride_uint4;
          reinterpret_cast<uint4*>(sV)[row * kv_smem_stride_uint4 + vec_col] =
              make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
          const int row = tid / kXQATC256WideThreadsPerRow;
          const __half* sP_row = sP + row * kXQATCBlockN + kv_tile_start;
          const float* sP_row_f = sS + row * kXQATCBlockN + kv_tile_start;
#pragma unroll
          for (int token = 0; token < kXQATCBlockN; ++token) {
            if (token >= valid_kv_tile_rows) {
              break;
            }
            float prob;
            if constexpr (kKeepPfp32) {
              prob = sP_row_f[token];
            } else {
              prob = __half2float(sP_row[token]);
            }
            const __half* sV_row = sV + token * SmemLayout::kKVStride;
            if constexpr (kPVHalf2) {
#pragma unroll
              for (int d_iter = 0; d_iter < (kPVPanelDim / (2 * kWarpSize));
                   ++d_iter) {
                const int local_d2 = lane_id + d_iter * kWarpSize;
                const float2 value = __half22float2(
                    reinterpret_cast<const __half2*>(sV_row)[local_d2]);
                const int acc_idx =
                    panel_idx * (kPVPanelDim / kWarpSize) + d_iter * 2;
                out_acc[acc_idx] = fmaf(prob, value.x, out_acc[acc_idx]);
                out_acc[acc_idx + 1] =
                    fmaf(prob, value.y, out_acc[acc_idx + 1]);
              }
            } else {
#pragma unroll
              for (int d_iter = 0; d_iter < (kPVPanelDim / kWarpSize);
                   ++d_iter) {
                const int local_d = lane_id + d_iter * kWarpSize;
                const int acc_idx =
                    panel_idx * (kPVPanelDim / kWarpSize) + d_iter;
                out_acc[acc_idx] =
                    fmaf(prob, __half2float(sV_row[local_d]), out_acc[acc_idx]);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    if (thread_in_row == 0) {
      smem.row_max[row] = row_max_reg;
      smem.row_sum[row] = row_sum_reg;
    }
  }
  __syncthreads();

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    const int head_idx = q_head_base + row;
    const float row_sum = smem.row_sum[row];
    const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
    __half* tmp_out_ptr = tmp_out +
                          static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
                          static_cast<int64_t>(head_idx) * tmp_out_stride1 +
                          static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    const float output_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                   ? inv_row_sum
                                   : inv_row_sum * v_scale;
    if constexpr (kPVHalf2) {
#pragma unroll
      for (int acc_idx = 0; acc_idx < kAccumsPerThread; ++acc_idx) {
        constexpr int kAccumsPerPanel = kPVPanelDim / kWarpSize;
        const int panel_idx = acc_idx / kAccumsPerPanel;
        const int panel_acc_idx = acc_idx % kAccumsPerPanel;
        const int d_iter = panel_acc_idx / 2;
        const int component = panel_acc_idx & 1;
        const int d = panel_idx * kPVPanelDim + d_iter * (2 * kWarpSize) +
                      thread_in_row * 2 + component;
        tmp_out_ptr[d] = __float2half(out_acc[acc_idx] * output_scale);
      }
    } else {
      for (int d = thread_in_row; d < D; d += kXQATC256WideThreadsPerRow) {
        tmp_out_ptr[d] = __float2half(out_acc[d / kWarpSize] * output_scale);
      }
    }
    if (thread_in_row == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = smem.row_max[row];
      exp_sums[stats_index] = row_sum;
    }
  }
}

// Exact grouped selector-verifier kernel for the admitted TP4 tensor contract.
//
// Independent-row decode scans the same paged KV prefix once for every
// verifier token. This kernel instead assigns one packed GQA group to each of
// eighty context splits. Every CTA handles all six query heads for all eight
// verifier tokens, so K/V is shared across the 48 rows before the exact
// split-softmax states are merged.
//
// The diagnostic two-pass variant intentionally uses:
//   1. QK computes the exact FP32 max/sum for each query row.
//   2. QK is recomputed, then half P x half V is accumulated with Volta WMMA.
// It remains a numerical oracle for the production one-pass register-rescale
// variant, which loads K once and preserves the same online-softmax/PV result.
//
// The verifier shape has one KV head and six query heads. Keep each CTA at 48
// rows for both supported widths: q8 packs six heads in one CTA, while q16
// packs three heads in each of two CTAs. The q16 route halves the number of
// context splits, preserving eighty CTAs and the q8 workspace byte count.
constexpr int kGroupedVerifyQ8MaxQ = 8;
constexpr int kGroupedVerifyQ16MaxQ = 16;
constexpr int kGroupedVerifyMaxSupportedQ = kGroupedVerifyQ16MaxQ;
constexpr int kGroupedVerifyHeads = 6;
constexpr int kGroupedVerifyHeadDim = 256;
constexpr int kGroupedVerifyRows = 48;
constexpr int kGroupedVerifyBlockN = 32;
constexpr int kGroupedVerifyQStride = 264;
constexpr int kGroupedVerifyKVStride = 264;
constexpr int kGroupedVerifyScoreStride = 32;
constexpr int kGroupedVerifyProbStride = 40;
constexpr int kGroupedVerifyPageIdsCapacity = 16;
// One packed head group times eighty context splits maps one 512-thread CTA to
// each of V100's eighty SMs. Short sequences still reduce active_splits using
// kGroupedVerifyMinTokensPerSplit inside the captured graph.
constexpr int kGroupedVerifyQ8Splits = 80;
constexpr int kGroupedVerifyWorkspaceRows =
    kGroupedVerifyQ8MaxQ * kGroupedVerifyQ8Splits;
// A packed CTA replaces two 256-thread head-group CTAs and can occupy only one
// SM. Use 64-token chunks to expose enough short-context parallelism without
// paying one split reduction per N32 tile; longer contexts saturate at the
// unchanged eighty splits. The single-query utility variant keeps the old
// 128-token split contract for its stricter FP32-reference error envelope.
constexpr int kGroupedVerifyMinTokensPerSplit = 64;
constexpr int kGroupedVerifySingleQueryMinTokensPerSplit = 128;
constexpr int kGroupedVerifyShortContextMaxTokens = 128;
constexpr int kGroupedVerifyThreads = 512;
constexpr int kGroupedVerifyWarps = kGroupedVerifyThreads / kWarpSize;
constexpr int kGroupedVerifyQKWarps =
    (kGroupedVerifyRows / 16) * (kGroupedVerifyBlockN / 16);
constexpr int kGroupedVerifyOutputTiles =
    (kGroupedVerifyRows / 16) * (kGroupedVerifyHeadDim / 16);
constexpr int kGroupedVerifyOutputTilesPerWarp =
    kGroupedVerifyOutputTiles / kGroupedVerifyWarps;

template <int MAX_QUERY_TOKENS>
struct GroupedVerifyTraits {
  static_assert(MAX_QUERY_TOKENS == kGroupedVerifyQ8MaxQ ||
                    MAX_QUERY_TOKENS == kGroupedVerifyQ16MaxQ,
                "grouped verifier supports q8 and q16 workspaces only");
  static constexpr int kHeadsPerCta = kGroupedVerifyRows / MAX_QUERY_TOKENS;
  static constexpr int kHeadGroups = kGroupedVerifyHeads / kHeadsPerCta;
  static constexpr int kSplits = kGroupedVerifyWorkspaceRows / MAX_QUERY_TOKENS;
  static_assert(MAX_QUERY_TOKENS * kHeadsPerCta == kGroupedVerifyRows,
                "grouped verifier CTA must retain 48 rows");
  static_assert(kGroupedVerifyHeads % kHeadsPerCta == 0,
                "query heads must divide evenly across grouped CTAs");
  static_assert(kSplits * MAX_QUERY_TOKENS == kGroupedVerifyWorkspaceRows,
                "q8 and q16 workspaces must have equal element counts");
};

struct alignas(256) GroupedVerifySmem {
  union {
    struct {
      alignas(16) __half q[kGroupedVerifyRows * kGroupedVerifyQStride];
      alignas(16) __half kv[kGroupedVerifyBlockN * kGroupedVerifyKVStride];
      alignas(16) float scores[kGroupedVerifyRows * kGroupedVerifyScoreStride];
      alignas(16) __half probs[kGroupedVerifyRows * kGroupedVerifyProbStride];
    } compute;
    alignas(16) float output[kGroupedVerifyRows * kGroupedVerifyHeadDim];
  } storage;
  alignas(16) float row_max[kGroupedVerifyRows];
  alignas(16) float row_sum[kGroupedVerifyRows];
  alignas(16) float row_scale[kGroupedVerifyRows];
  alignas(16) int page_ids[kGroupedVerifyPageIdsCapacity];
  alignas(16) uint32_t sparse_token_masks[kGroupedVerifyBlockN / 4];
};

static_assert(sizeof(GroupedVerifySmem) <= 64 * 1024,
              "packed grouped verifier must fit Volta's 64 KiB opt-in budget");
static_assert(kGroupedVerifyOutputTiles % kGroupedVerifyWarps == 0,
              "output tiles must divide evenly across warps");

template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY>
__device__ __forceinline__ int grouped_verify_active_splits(
    const int total_kv) {
  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;
  constexpr int kMinTokensPerSplit =
      SINGLE_QUERY ? kGroupedVerifySingleQueryMinTokensPerSplit
                   : kGroupedVerifyMinTokensPerSplit;
  const int active_splits =
      min(Traits::kSplits,
          max(1, (total_kv + kMinTokensPerSplit - 1) / kMinTokensPerSplit));
  if constexpr (SINGLE_QUERY) {
    return active_splits;
  }
  return total_kv <= kGroupedVerifyShortContextMaxTokens ? 1 : active_splits;
}

__device__ __forceinline__ void grouped_verify_qk(
    const __half* __restrict__ shared_q, const __half* __restrict__ shared_k,
    float* __restrict__ shared_scores, const float qk_scale,
    const int active_m_tiles) {
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id >= kGroupedVerifyQKWarps) {
    return;
  }

  const int m_tile = warp_id / (kGroupedVerifyBlockN / 16);
  if ((active_m_tiles & (1 << m_tile)) == 0) {
    return;
  }
  const int n_tile = warp_id % (kGroupedVerifyBlockN / 16);
  volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
      q_fragment;
  volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::col_major>
      k_fragment;
  volta::fragment<volta::accumulator, 16, 16, 16, float> score_fragment;
  volta::fill_fragment(score_fragment, 0.0f);

#pragma unroll
  for (int k_offset = 0; k_offset < kGroupedVerifyHeadDim; k_offset += 16) {
    volta::load_matrix_sync(
        q_fragment, shared_q + m_tile * 16 * kGroupedVerifyQStride + k_offset,
        kGroupedVerifyQStride);
    // K is row-major [N, D].  The same bytes represent K^T as a col-major
    // [D, N] matrix, which is the B operand needed by Q @ K^T.
    volta::load_matrix_sync(
        k_fragment, shared_k + n_tile * 16 * kGroupedVerifyKVStride + k_offset,
        kGroupedVerifyKVStride);
    volta::mma_sync(score_fragment, q_fragment, k_fragment, score_fragment);
  }
#pragma unroll
  for (int i = 0; i < score_fragment.num_elements; ++i) {
    score_fragment.x[i] *= qk_scale;
  }
  volta::store_matrix_sync(
      shared_scores + m_tile * 16 * kGroupedVerifyScoreStride + n_tile * 16,
      score_fragment, kGroupedVerifyScoreStride, volta::mem_row_major);
}

__device__ __forceinline__ void grouped_verify_scale_output_fragment(
    volta::fragment<volta::accumulator, 16, 16, 16, float>& fragment,
    const float* __restrict__ row_scale, const int tile_row_start) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const float first_scale = row_scale[tile_row_start + row];
  const float second_scale = row_scale[tile_row_start + row + 2];
  fragment.x[0] *= first_scale;
  fragment.x[1] *= first_scale;
  fragment.x[2] *= second_scale;
  fragment.x[3] *= second_scale;
  fragment.x[4] *= first_scale;
  fragment.x[5] *= first_scale;
  fragment.x[6] *= second_scale;
  fragment.x[7] *= second_scale;
}

template <bool SPARSE_PAGE4>
__device__ __forceinline__ bool grouped_verify_key_visible(
    const uint32_t* __restrict__ sparse_tile_masks, const int token_idx,
    const int query_len, const int head_idx, const int kv_idx,
    const int valid_k_rows, const int lane_or_col, const int prefix_kv_len) {
  const bool row_valid = token_idx < query_len &&
                         head_idx < kGroupedVerifyHeads &&
                         lane_or_col < valid_k_rows;
  if (!row_valid) {
    return false;
  }
  if constexpr (SPARSE_PAGE4) {
    const uint32_t token_mask =
        sparse_tile_masks[(kv_idx & (kGroupedVerifyBlockN - 1)) >> 2];
    const int mask_bit = token_idx * 4 + (kv_idx & 3);
    return (token_mask & (1u << mask_bit)) != 0;
  }
  return kv_idx <= prefix_kv_len + token_idx;
}

template <int MAX_QUERY_TOKENS, bool TWO_PASS, int PAGE_BLOCK_SIZE = 0,
          bool SINGLE_QUERY = false, bool CONTIGUOUS_HKV1_LAYOUT = false,
          bool STAGE_PARTITION_PAGE_IDS = false,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
          bool SPARSE_PAGE4 = false>
__global__
__launch_bounds__(kGroupedVerifyThreads, 1) void flash_attention_grouped_verify_e5m2_partial_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, __half* __restrict__ partial_out,
    float* __restrict__ partial_lse, const int query_len,
    const int max_num_blocks, const int page_block_size,
    const int64_t k_block_stride, const int64_t k_token_stride,
    const int64_t k_head_stride, const int64_t v_block_stride,
    const int64_t v_token_stride, const int64_t v_head_stride,
    const float qk_scale, const float v_scale,
    const uint32_t* __restrict__ sparse_token_masks = nullptr,
    const int num_groups = 1) {
  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;
  const int head_group = blockIdx.x;
  const int split_id = blockIdx.y;
  const int group_idx = SPARSE_PAGE4 ? blockIdx.z : 0;
  if (head_group >= Traits::kHeadGroups || split_id >= Traits::kSplits ||
      group_idx >= num_groups || query_len <= 0 ||
      query_len > MAX_QUERY_TOKENS) {
    return;
  }

  const int total_kv = seq_lens[SPARSE_PAGE4 ? group_idx : 0];
  if (total_kv <= 0) {
    if constexpr (SPARSE_PAGE4) {
      constexpr int kGroupOutputElements =
          kGroupedVerifyQ8MaxQ * kGroupedVerifyHeads * kGroupedVerifyHeadDim;
      const int64_t group_output_offset =
          static_cast<int64_t>(group_idx) * kGroupOutputElements;
      for (int idx = threadIdx.x; idx < kGroupOutputElements;
           idx += kGroupedVerifyThreads) {
        partial_out[group_output_offset + idx] = __float2half_rn(0.0f);
      }
      constexpr int kGroupLseElements =
          kGroupedVerifyQ8MaxQ * kGroupedVerifyHeads;
      if (threadIdx.x < kGroupLseElements) {
        partial_lse[static_cast<int64_t>(group_idx) * kGroupLseElements +
                    threadIdx.x] = kXQANegInf;
      }
    }
    return;
  }
  const int active_splits =
      SPARSE_PAGE4
          ? 1
          : grouped_verify_active_splits<MAX_QUERY_TOKENS, SINGLE_QUERY>(
                total_kv);
  if (split_id >= active_splits) {
    return;
  }
  const int total_tiles =
      (total_kv + kGroupedVerifyBlockN - 1) / kGroupedVerifyBlockN;
  const int base_tiles = total_tiles / active_splits;
  const int extra_tiles = total_tiles - base_tiles * active_splits;
  const int split_tile_start =
      split_id * base_tiles + min(split_id, extra_tiles);
  const int split_tiles = base_tiles + (split_id < extra_tiles ? 1 : 0);
  const int split_start = split_tile_start * kGroupedVerifyBlockN;
  const int split_end =
      min(total_kv, split_start + split_tiles * kGroupedVerifyBlockN);
  const int prefix_kv_len = max(0, total_kv - query_len);
  const int head_start = head_group * Traits::kHeadsPerCta;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  extern __shared__ char grouped_verify_smem_raw[];
  GroupedVerifySmem& smem =
      *reinterpret_cast<GroupedVerifySmem*>(grouped_verify_smem_raw);
  __half* shared_q = smem.storage.compute.q;
  __half* shared_kv = smem.storage.compute.kv;
  float* shared_scores = smem.storage.compute.scores;
  __half* shared_probs = smem.storage.compute.probs;
  const int* page_ids =
      block_table + static_cast<int64_t>(group_idx) * max_num_blocks;
  int split_page_offset = 0;
  bool use_staged_page_ids = false;
  if constexpr (STAGE_PARTITION_PAGE_IDS) {
    static_assert(PAGE_BLOCK_SIZE > 0,
                  "partition page staging requires a specialized page size");
    const int split_start_page = split_start / PAGE_BLOCK_SIZE;
    split_page_offset = split_start - split_start_page * PAGE_BLOCK_SIZE;
    const int split_page_count =
        (split_page_offset + split_end - split_start + PAGE_BLOCK_SIZE - 1) /
        PAGE_BLOCK_SIZE;
    use_staged_page_ids = split_page_count <= kGroupedVerifyPageIdsCapacity;
    if (use_staged_page_ids) {
      for (int idx = tid; idx < split_page_count;
           idx += kGroupedVerifyThreads) {
        smem.page_ids[idx] = __ldg(&block_table[split_start_page + idx]);
      }
    }
    if (use_staged_page_ids) {
      page_ids = smem.page_ids;
    }
  }

  constexpr int kVecsPerRow = kGroupedVerifyHeadDim / 8;
  constexpr int kSharedQVecsPerRow = kGroupedVerifyQStride / 8;
  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* shared_q_vec = reinterpret_cast<uint4*>(shared_q);
  for (int idx = tid; idx < kGroupedVerifyRows * kVecsPerRow;
       idx += kGroupedVerifyThreads) {
    const int row = idx / kVecsPerRow;
    const int vec_col = idx % kVecsPerRow;
    const int token_idx = row / Traits::kHeadsPerCta;
    const int local_head = row % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const int64_t query_row =
          static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
      shared_q_vec[row * kSharedQVecsPerRow + vec_col] = __ldg(
          q_vec + (query_row * kGroupedVerifyHeads + head_idx) * kVecsPerRow +
          vec_col);
    } else {
      shared_q_vec[row * kSharedQVecsPerRow + vec_col] = make_uint4(0, 0, 0, 0);
    }
  }
  if (tid < kGroupedVerifyRows) {
    smem.row_max[tid] = kXQANegInf;
    smem.row_sum[tid] = 0.0f;
    smem.row_scale[tid] = 1.0f;
  }
  // This barrier publishes both Q and the optional split-local page IDs.
  __syncthreads();

  constexpr int kPanelStrideVec = kGroupedVerifyHeadDim / 8;
  constexpr int kSharedStrideVec = kGroupedVerifyKVStride / 8;

  volta::fragment<volta::accumulator, 16, 16, 16, float>
      output_fragments[kGroupedVerifyOutputTilesPerWarp];
#pragma unroll
  for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
       ++fragment_idx) {
    volta::fill_fragment(output_fragments[fragment_idx], 0.0f);
  }

  // The conservative baseline computes exact per-split FP32 max/sum in a
  // separate pass. The candidate below fuses this state update with P x V.
  if constexpr (TWO_PASS) {
    for (int tile_start = split_start; tile_start < split_end;
         tile_start += kGroupedVerifyBlockN) {
      const int valid_k_rows =
          min(kGroupedVerifyBlockN, split_end - tile_start);
      if constexpr (SPARSE_PAGE4) {
        const int valid_sparse_pages = (valid_k_rows + 3) / 4;
        if (tid < kGroupedVerifyBlockN / 4) {
          smem.sparse_token_masks[tid] =
              tid < valid_sparse_pages
                  ? __ldg(sparse_token_masks +
                          static_cast<int64_t>(group_idx) * max_num_blocks +
                          (tile_start >> 2) + tid)
                  : 0;
        }
      }
      const int tile_page_offset =
          use_staged_page_ids ? split_page_offset + tile_start - split_start
                              : tile_start;
      load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                           kGroupedVerifyThreads, KV_DTYPE,
                           KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2>(
          shared_kv, k_cache, page_ids, valid_k_rows, kPanelStrideVec,
          kSharedStrideVec, tile_page_offset, 0, page_block_size, 0,
          k_block_stride, k_token_stride, k_head_stride, 0);
      for (int idx = tid + valid_k_rows * kSharedStrideVec;
           idx < kGroupedVerifyBlockN * kSharedStrideVec;
           idx += kGroupedVerifyThreads) {
        reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
      }
      __syncthreads();

      int active_m_tiles = 0x7;
      if constexpr (SPARSE_PAGE4) {
        uint32_t active_query_nibbles = 0;
#pragma unroll
        for (int page = 0; page < kGroupedVerifyBlockN / 4; ++page) {
          active_query_nibbles |= smem.sparse_token_masks[page];
        }
        active_m_tiles = 0;
#pragma unroll
        for (int token = 0; token < kGroupedVerifyQ8MaxQ; ++token) {
          if ((active_query_nibbles & (0xFu << (token * 4))) != 0) {
            const int first_row = token * kGroupedVerifyHeads;
            const int last_row = first_row + kGroupedVerifyHeads - 1;
            active_m_tiles |= 1 << (first_row / 16);
            active_m_tiles |= 1 << (last_row / 16);
          }
        }
      }
      grouped_verify_qk(shared_q, shared_kv, shared_scores, qk_scale,
                        active_m_tiles);
      __syncthreads();

#pragma unroll
      for (int row = warp_id; row < kGroupedVerifyRows;
           row += kGroupedVerifyWarps) {
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + lane_id;
        const bool visible = grouped_verify_key_visible<SPARSE_PAGE4>(
            smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
            valid_k_rows, lane_id, prefix_kv_len);
        const float score =
            visible ? shared_scores[row * kGroupedVerifyScoreStride + lane_id]
                    : kXQANegInf;
        const float tile_max_lane = warp_reduce_max(score);
        const float tile_max = __shfl_sync(0xffffffffu, tile_max_lane, 0);
        const float probability =
            visible ? __expf(fmaxf(score - tile_max, -80.0f)) : 0.0f;
        const float tile_sum_lane = warp_reduce_sum(probability);
        const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
        if (lane_id == 0 && tile_sum > 0.0f) {
          const float old_max = smem.row_max[row];
          const float old_sum = smem.row_sum[row];
          const float new_max = fmaxf(old_max, tile_max);
          smem.row_sum[row] =
              old_sum * __expf(fmaxf(old_max - new_max, -80.0f)) +
              tile_sum * __expf(fmaxf(tile_max - new_max, -80.0f));
          smem.row_max[row] = new_max;
        }
      }
      __syncthreads();
    }
  }

  // Recompute QK for the conservative path, or consume it once while updating
  // the online state for the fused path. Both form half P and use identical
  // Volta WMMA P @ V arithmetic.
  for (int tile_start = split_start; tile_start < split_end;
       tile_start += kGroupedVerifyBlockN) {
    const int valid_k_rows = min(kGroupedVerifyBlockN, split_end - tile_start);
    if constexpr (SPARSE_PAGE4) {
      const int valid_sparse_pages = (valid_k_rows + 3) / 4;
      if (tid < kGroupedVerifyBlockN / 4) {
        smem.sparse_token_masks[tid] =
            tid < valid_sparse_pages
                ? __ldg(sparse_token_masks +
                        static_cast<int64_t>(group_idx) * max_num_blocks +
                        (tile_start >> 2) + tid)
                : 0;
      }
    }
    const int tile_page_offset =
        use_staged_page_ids ? split_page_offset + tile_start - split_start
                            : tile_start;
    load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                         kGroupedVerifyThreads, KV_DTYPE,
                         KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2>(
        shared_kv, k_cache, page_ids, valid_k_rows, kPanelStrideVec,
        kSharedStrideVec, tile_page_offset, 0, page_block_size, 0,
        k_block_stride, k_token_stride, k_head_stride, 0);
    for (int idx = tid + valid_k_rows * kSharedStrideVec;
         idx < kGroupedVerifyBlockN * kSharedStrideVec;
         idx += kGroupedVerifyThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    int active_m_tiles = 0x7;
    if constexpr (SPARSE_PAGE4) {
      uint32_t active_query_nibbles = 0;
#pragma unroll
      for (int page = 0; page < kGroupedVerifyBlockN / 4; ++page) {
        active_query_nibbles |= smem.sparse_token_masks[page];
      }
      active_m_tiles = 0;
#pragma unroll
      for (int token = 0; token < kGroupedVerifyQ8MaxQ; ++token) {
        if ((active_query_nibbles & (0xFu << (token * 4))) != 0) {
          const int first_row = token * kGroupedVerifyHeads;
          const int last_row = first_row + kGroupedVerifyHeads - 1;
          active_m_tiles |= 1 << (first_row / 16);
          active_m_tiles |= 1 << (last_row / 16);
        }
      }
    }
    grouped_verify_qk(shared_q, shared_kv, shared_scores, qk_scale,
                      active_m_tiles);
    __syncthreads();

    if constexpr (TWO_PASS) {
      for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyBlockN;
           idx += kGroupedVerifyThreads) {
        const int row = idx / kGroupedVerifyBlockN;
        const int col = idx % kGroupedVerifyBlockN;
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + col;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, col, prefix_kv_len) &&
            smem.row_sum[row] > 0.0f;
        const float probability =
            visible ? __expf(fmaxf(
                          shared_scores[row * kGroupedVerifyScoreStride + col] -
                              smem.row_max[row],
                          -80.0f))
                    : 0.0f;
        shared_probs[row * kGroupedVerifyProbStride + col] =
            __float2half_rn(probability);
      }
      __syncthreads();
    } else {
#pragma unroll
      for (int row = warp_id; row < kGroupedVerifyRows;
           row += kGroupedVerifyWarps) {
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + lane_id;
        const bool visible = grouped_verify_key_visible<SPARSE_PAGE4>(
            smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
            valid_k_rows, lane_id, prefix_kv_len);
        const float score =
            visible ? shared_scores[row * kGroupedVerifyScoreStride + lane_id]
                    : kXQANegInf;
        const float tile_max_lane = warp_reduce_max(score);
        const float tile_max = __shfl_sync(0xffffffffu, tile_max_lane, 0);
        const float old_max = smem.row_max[row];
        const float new_max = fmaxf(old_max, tile_max);
        const float probability =
            visible ? __expf(fmaxf(score - new_max, -80.0f)) : 0.0f;
        const float tile_sum_lane = warp_reduce_sum(probability);
        const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
        const float exp_diff =
            tile_sum > 0.0f ? __expf(fmaxf(old_max - new_max, -80.0f)) : 1.0f;
        shared_probs[row * kGroupedVerifyProbStride + lane_id] =
            __float2half_rn(probability);
        if (lane_id == 0) {
          if (tile_sum > 0.0f) {
            smem.row_sum[row] = smem.row_sum[row] * exp_diff + tile_sum;
            smem.row_max[row] = new_max;
          }
          smem.row_scale[row] = exp_diff;
        }
      }
      __syncthreads();
#pragma unroll
      for (int fragment_idx = 0;
           fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
        const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
        const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
        grouped_verify_scale_output_fragment(output_fragments[fragment_idx],
                                             smem.row_scale, m_tile * 16);
      }
    }

    load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                         kGroupedVerifyThreads, KV_DTYPE,
                         KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2>(
        shared_kv, v_cache, page_ids, valid_k_rows, kPanelStrideVec,
        kSharedStrideVec, tile_page_offset, 0, page_block_size, 0,
        v_block_stride, v_token_stride, v_head_stride, 0);
    for (int idx = tid + valid_k_rows * kSharedStrideVec;
         idx < kGroupedVerifyBlockN * kSharedStrideVec;
         idx += kGroupedVerifyThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

#pragma unroll
    for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
         ++fragment_idx) {
      const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
      const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
      const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
      if constexpr (SPARSE_PAGE4) {
        if ((active_m_tiles & (1 << m_tile)) == 0) {
          continue;
        }
      }
      volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
          probability_fragment;
      volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::row_major>
          value_fragment;
#pragma unroll
      for (int k_offset = 0; k_offset < kGroupedVerifyBlockN; k_offset += 16) {
        volta::load_matrix_sync(
            probability_fragment,
            shared_probs + m_tile * 16 * kGroupedVerifyProbStride + k_offset,
            kGroupedVerifyProbStride);
        volta::load_matrix_sync(
            value_fragment,
            shared_kv + k_offset * kGroupedVerifyKVStride + d_tile * 16,
            kGroupedVerifyKVStride);
        volta::mma_sync(output_fragments[fragment_idx], probability_fragment,
                        value_fragment, output_fragments[fragment_idx]);
      }
    }
    __syncthreads();
  }

  // The compute buffers are dead. Reuse their storage for the dense FP32
  // partial output, then normalize and write only real query/head rows.
  __syncthreads();
  float* shared_output = smem.storage.output;
#pragma unroll
  for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
       ++fragment_idx) {
    const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
    const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
    const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
    volta::store_matrix_sync(
        shared_output + m_tile * 16 * kGroupedVerifyHeadDim + d_tile * 16,
        output_fragments[fragment_idx], kGroupedVerifyHeadDim,
        volta::mem_row_major);
  }
  __syncthreads();

  for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyHeadDim;
       idx += kGroupedVerifyThreads) {
    const int row = idx / kGroupedVerifyHeadDim;
    const int d = idx % kGroupedVerifyHeadDim;
    const int token_idx = row / Traits::kHeadsPerCta;
    const int local_head = row % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const float sum = smem.row_sum[row];
      const float scale = sum > 0.0f ? v_scale / sum : 0.0f;
      int64_t output_idx;
      if constexpr (SPARSE_PAGE4) {
        const int64_t global_token_idx =
            static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
        output_idx = (global_token_idx * kGroupedVerifyHeads + head_idx) *
                         kGroupedVerifyHeadDim +
                     d;
      } else {
        output_idx =
            (((static_cast<int64_t>(split_id) * MAX_QUERY_TOKENS + token_idx) *
                  kGroupedVerifyHeads +
              head_idx) *
                 kGroupedVerifyHeadDim +
             d);
      }
      partial_out[output_idx] = __float2half_rn(shared_output[idx] * scale);
    }
  }
  if (tid < kGroupedVerifyRows) {
    const int token_idx = tid / Traits::kHeadsPerCta;
    const int local_head = tid % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const float sum = smem.row_sum[tid];
      int64_t lse_idx;
      if constexpr (SPARSE_PAGE4) {
        const int64_t global_token_idx =
            static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
        lse_idx = global_token_idx * kGroupedVerifyHeads + head_idx;
      } else {
        lse_idx =
            (static_cast<int64_t>(split_id) * MAX_QUERY_TOKENS + token_idx) *
                kGroupedVerifyHeads +
            head_idx;
      }
      partial_lse[lse_idx] =
          sum > 0.0f ? smem.row_max[tid] + logf(sum) : kXQANegInf;
    }
  }
}

template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY>
__global__
__launch_bounds__(kGroupedVerifyThreads) void flash_attention_grouped_verify_e5m2_combine_kernel(
    const __half* __restrict__ partial_out,
    const float* __restrict__ partial_lse, const int* __restrict__ seq_lens,
    __half* __restrict__ out, const int query_len) {
  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;
  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  if (token_idx >= query_len || head_idx >= kGroupedVerifyHeads) {
    return;
  }
  const int active_splits =
      grouped_verify_active_splits<MAX_QUERY_TOKENS, SINGLE_QUERY>(seq_lens[0]);
  __shared__ float split_lse[Traits::kSplits];
  __shared__ float final_max;
  __shared__ float final_inv_sum;

  if (threadIdx.x < Traits::kSplits) {
    const int64_t lse_idx =
        (static_cast<int64_t>(threadIdx.x) * MAX_QUERY_TOKENS + token_idx) *
            kGroupedVerifyHeads +
        head_idx;
    split_lse[threadIdx.x] =
        threadIdx.x < active_splits ? partial_lse[lse_idx] : kXQANegInf;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float max_lse = kXQANegInf;
    for (int split = 0; split < active_splits; ++split) {
      max_lse = fmaxf(max_lse, split_lse[split]);
    }
    float sum = 0.0f;
    for (int split = 0; split < active_splits; ++split) {
      if (split_lse[split] > -1.0e20f) {
        sum += __expf(fmaxf(split_lse[split] - max_lse, -80.0f));
      }
    }
    final_max = max_lse;
    final_inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
  }
  __syncthreads();

  for (int d = threadIdx.x; d < kGroupedVerifyHeadDim;
       d += kGroupedVerifyThreads) {
    float accumulator = 0.0f;
    for (int split = 0; split < active_splits; ++split) {
      if (split_lse[split] > -1.0e20f) {
        const float weight =
            __expf(fmaxf(split_lse[split] - final_max, -80.0f)) * final_inv_sum;
        const int64_t partial_idx =
            (((static_cast<int64_t>(split) * MAX_QUERY_TOKENS + token_idx) *
                  kGroupedVerifyHeads +
              head_idx) *
                 kGroupedVerifyHeadDim +
             d);
        accumulator =
            fmaf(weight, __half2float(partial_out[partial_idx]), accumulator);
      }
    }
    out[(token_idx * kGroupedVerifyHeads + head_idx) * kGroupedVerifyHeadDim +
        d] = __float2half_rn(accumulator);
  }
}

template <int D, int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens>
__global__ void flash_attention_decode_reduce_kernel(
    const __half* __restrict__ tmp_out, const float* __restrict__ max_logits,
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
      acc = fmaf(
          weight_shared[i],
          __half2float(tmp_out[tmp_out_base +
                               static_cast<int64_t>(i) * tmp_out_stride2 + d]),
          acc);
    }
    out[out_base + d] = __float2half(acc * inv_global_sum);
  }
}

// This separates only the cross-partition reducer. The stats kernel preserves
// the original block-wide max/sum tree, and each output dimension retains its
// original ascending-partition FMA order.
template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_qk_softmax_staged_kernel_256_wide(
        const __half* __restrict__ q, const __half* __restrict__ k_cache,
        __half* __restrict__ probabilities, float* __restrict__ max_logits,
        float* __restrict__ exp_sums, float* __restrict__ online_rescales,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t q_stride0, const int64_t q_stride1,
        const int64_t probability_stride0, const int64_t probability_stride1,
        const int64_t probability_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t online_rescale_stride0,
        const int64_t online_rescale_stride1, const int64_t k_block_stride,
        const int64_t k_token_stride, const int64_t k_head_stride,
        const float softmax_scale) {
  constexpr int D = 256;
  constexpr int WMMA_M = 8;
  constexpr int WMMA_N = 32;
  constexpr int WMMA_K = 16;
  constexpr int kPanelDim = kXQATCStride;
  constexpr int kNumPanels = D / kPanelDim;
  using SmemLayout = XQATCSmem256WideLayout<PADDED_SMEM>;
  constexpr int q_global_stride_uint4 = D / 8;
  constexpr int q_smem_stride_uint4 = SmemLayout::kQStride / 8;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int panel_d_stride_uint4 = kPanelDim / 8;
  static_assert(GROUP_SIZE == 6,
                "Staged D=256 TC XQA is specialized for q_per_kv=6");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one softmax warp");

  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  __half* sQ = smem.q;
  __half* sK = smem.reuse_kv.k;
  float* sS = smem.reuse_sp.s;
  __half* sP = smem.reuse_sp.p;
  float row_max_reg = kXQANegInf;
  float row_sum_reg = 0.f;
  float row_rescale_reg = 1.f;

  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
  for (int idx = tid; idx < GROUP_SIZE * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    const int64_t q_offset =
        static_cast<int64_t>(batch_idx) * q_stride0 +
        static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid;
       idx < (kXQATC256WideBlockM - GROUP_SIZE) * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = GROUP_SIZE + idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    const int start_page = tile_token_start / block_size;
    const int tile_page_offset = tile_token_start - start_page * block_size;
    const int page_count =
        (tile_page_offset + valid_k_rows + block_size - 1) / block_size;

    for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
    }
    __syncthreads();

    volta::fragment<volta::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                    volta::row_major>
        qk_a_frag;
    volta::fragment<volta::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                    volta::col_major>
        qk_b_frag;
    volta::fragment<volta::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        qk_acc_frag;
    if (warp_id < (kXQATCBlockN / WMMA_N)) {
      volta::fill_fragment(qk_acc_frag, 0.0f);
    }

    for (int panel_idx = 0; panel_idx < kNumPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPanelDim;
      for (int idx = tid; idx < valid_k_rows * panel_d_stride_uint4;
           idx += NUM_THREADS) {
        const int row = idx / panel_d_stride_uint4;
        const int vec_col = idx % panel_d_stride_uint4;
        const int token_offset = tile_page_offset + row;
        const int physical_block = smem.page_ids[token_offset / block_size];
        const int block_offset = token_offset % block_size;
        const int64_t physical_offset_half_elements =
            static_cast<int64_t>(physical_block) * k_block_stride +
            static_cast<int64_t>(block_offset) * k_token_stride +
            static_cast<int64_t>(kv_head_idx) * k_head_stride + panel_offset;
        const uint4* k_vec = reinterpret_cast<const uint4*>(k_cache);
        reinterpret_cast<uint4*>(sK)[row * kv_smem_stride_uint4 + vec_col] =
            __ldg(&k_vec[physical_offset_half_elements / 8 + vec_col]);
      }
      for (int idx = tid + valid_k_rows * panel_d_stride_uint4;
           idx < kXQATCBlockN * panel_d_stride_uint4; idx += NUM_THREADS) {
        reinterpret_cast<uint4*>(sK)[idx] = make_uint4(0, 0, 0, 0);
      }
      __syncthreads();

      if (warp_id < (kXQATCBlockN / WMMA_N)) {
        const int tile_n = warp_id * WMMA_N;
#pragma unroll
        for (int k_tile = 0; k_tile < (kPanelDim / WMMA_K); ++k_tile) {
          const int k_offset = k_tile * WMMA_K;
          volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                  SmemLayout::kQStride);
          volta::load_matrix_sync(
              qk_b_frag, sK + tile_n * SmemLayout::kKVStride + k_offset,
              SmemLayout::kKVStride);
          volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
        }
      }
      __syncthreads();
    }

    if (warp_id < (kXQATCBlockN / WMMA_N)) {
#pragma unroll
      for (int i = 0; i < qk_acc_frag.num_elements; ++i) {
        qk_acc_frag.x[i] *= softmax_scale;
      }
      volta::store_matrix_sync(sS + warp_id * WMMA_N, qk_acc_frag, kXQATCBlockN,
                               volta::mem_row_major);
    }
    __syncthreads();

    if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      const unsigned mask = 0xffffffffu;
      float* sS_row_f = sS + row * kXQATCBlockN;
      __half* sP_row_h = sP + row * kXQATCBlockN;
      const int vec_cols = valid_k_rows >> 2;
      const int tail_start = vec_cols << 2;
      const int vec_col = thread_in_row;

      float thread_max = kXQANegInf;
      __half2 packed_exp0 = __float22half2_rn(make_float2(0.f, 0.f));
      __half2 packed_exp1 = __float22half2_rn(make_float2(0.f, 0.f));
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        thread_max =
            fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < valid_k_rows;
           c += kXQATC256WideThreadsPerRow) {
        thread_max = fmaxf(thread_max, sS_row_f[c]);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_max =
            fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
      }

      const float row_max = __shfl_sync(mask, thread_max, 0, kWarpSize);
      const float old_max = __shfl_sync(mask, row_max_reg, 0, kWarpSize);
      const float new_max = fmaxf(old_max, row_max);
      const float exp_diff = __expf(old_max - new_max);

      float thread_sum = 0.f;
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        const float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
        const float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
        const float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
        const float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
        thread_sum += (e0 + e1) + (e2 + e3);
        packed_exp0 = __float22half2_rn(make_float2(e0, e1));
        packed_exp1 = __float22half2_rn(make_float2(e2, e3));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < kXQATCBlockN;
           c += kXQATC256WideThreadsPerRow) {
        const float v = (c < valid_k_rows) ? sS_row_f[c] : kXQANegInf;
        const float e = __expf(fmaxf(v - new_max, -80.0f));
        thread_sum += (c < valid_k_rows) ? e : 0.0f;
        sP_row_h[c] =
            (c < valid_k_rows) ? __float2half_rn(e) : __float2half(0.f);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, o, kWarpSize);
      }

      const float row_sum = __shfl_sync(mask, thread_sum, 0, kWarpSize);
      const float old_sum = __shfl_sync(mask, row_sum_reg, 0, kWarpSize);
      if (thread_in_row == 0) {
        row_sum_reg = exp_diff * old_sum + row_sum;
        row_max_reg = new_max;
      }
      if (block_n > 0) {
        row_rescale_reg = exp_diff;
      }

      __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);
      if (vec_col < vec_cols) {
        const int base_offset = vec_col * 2;
        sP_half2[base_offset] = packed_exp0;
        sP_half2[base_offset + 1] = packed_exp1;
      }
    }
    __syncthreads();

    for (int idx = tid; idx < GROUP_SIZE * valid_k_rows; idx += NUM_THREADS) {
      const int row = idx / valid_k_rows;
      const int token = idx % valid_k_rows;
      const int head_idx = q_head_base + row;
      __half* probability_ptr =
          probabilities +
          static_cast<int64_t>(batch_idx) * probability_stride0 +
          static_cast<int64_t>(head_idx) * probability_stride1 +
          static_cast<int64_t>(partition_idx) * probability_stride2 +
          block_n * kXQATCBlockN + token;
      *probability_ptr = sP[row * kXQATCBlockN + token];
    }
    __syncthreads();
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    if (thread_in_row == 0) {
      const int head_idx = q_head_base + row;
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = row_max_reg;
      exp_sums[stats_index] = row_sum_reg;
      online_rescales[static_cast<int64_t>(batch_idx) * online_rescale_stride0 +
                      static_cast<int64_t>(head_idx) * online_rescale_stride1 +
                      partition_idx] = row_rescale_reg;
    }
  }
}

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_pv_staged_kernel_256_wide(
        const __half* __restrict__ v_cache, const __half* probabilities,
        __half* tmp_out, const float* __restrict__ exp_sums,
        const float* __restrict__ online_rescales,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t online_rescale_stride0,
        const int64_t online_rescale_stride1, const int64_t v_block_stride,
        const int64_t v_token_stride, const int64_t v_head_stride) {
  constexpr int D = 256;
  constexpr int kPanelDim = kXQATCStride;
  constexpr int kNumPanels = D / kPanelDim;
  constexpr int kAccumsPerThread = D / kWarpSize;
  using SmemLayout = XQATCStagedPVSmem256Wide<PADDED_SMEM>;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int panel_d_stride_uint4 = kPanelDim / 8;
  static_assert(GROUP_SIZE == 6,
                "Staged D=256 TC XQA is specialized for q_per_kv=6");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one PV warp");

  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) {
    return;
  }

  const int tid = threadIdx.x;
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;
  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  __half* sV = smem.v;
  float out_acc[kAccumsPerThread];
#pragma unroll
  for (int i = 0; i < kAccumsPerThread; ++i) {
    out_acc[i] = 0.f;
  }

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    const int start_page = tile_token_start / block_size;
    const int tile_page_offset = tile_token_start - start_page * block_size;
    const int page_count =
        (tile_page_offset + valid_k_rows + block_size - 1) / block_size;
    for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
    }
    __syncthreads();

    if (block_n > 0 && tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      float rescale = 1.f;
      if (thread_in_row == 0) {
        const int head_idx = q_head_base + row;
        rescale = online_rescales[static_cast<int64_t>(batch_idx) *
                                      online_rescale_stride0 +
                                  static_cast<int64_t>(head_idx) *
                                      online_rescale_stride1 +
                                  partition_idx];
      }
      rescale = __shfl_sync(0xffffffffu, rescale, 0, kWarpSize);
#pragma unroll
      for (int i = 0; i < kAccumsPerThread; ++i) {
        out_acc[i] *= rescale;
      }
    }

    for (int panel_idx = 0; panel_idx < kNumPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPanelDim;
      for (int v_tile_start = 0; v_tile_start < valid_k_rows;
           v_tile_start += kXQATCStagedPVTileRows) {
        const int valid_v_rows =
            min(kXQATCStagedPVTileRows, valid_k_rows - v_tile_start);
        for (int idx = tid; idx < valid_v_rows * panel_d_stride_uint4;
             idx += NUM_THREADS) {
          const int row = idx / panel_d_stride_uint4;
          const int vec_col = idx % panel_d_stride_uint4;
          const int token_offset = tile_page_offset + v_tile_start + row;
          const int physical_block = smem.page_ids[token_offset / block_size];
          const int block_offset = token_offset % block_size;
          const int64_t physical_offset_half_elements =
              static_cast<int64_t>(physical_block) * v_block_stride +
              static_cast<int64_t>(block_offset) * v_token_stride +
              static_cast<int64_t>(kv_head_idx) * v_head_stride + panel_offset;
          const uint4* v_vec = reinterpret_cast<const uint4*>(v_cache);
          reinterpret_cast<uint4*>(sV)[row * kv_smem_stride_uint4 + vec_col] =
              __ldg(&v_vec[physical_offset_half_elements / 8 + vec_col]);
        }
        for (int idx = tid + valid_v_rows * panel_d_stride_uint4;
             idx < kXQATCStagedPVTileRows * panel_d_stride_uint4;
             idx += NUM_THREADS) {
          reinterpret_cast<uint4*>(sV)[idx] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
          const int row = tid / kXQATC256WideThreadsPerRow;
          const int head_idx = q_head_base + row;
          const __half* probability_ptr =
              probabilities +
              static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
              static_cast<int64_t>(head_idx) * tmp_out_stride1 +
              static_cast<int64_t>(partition_idx) * tmp_out_stride2 +
              block_n * kXQATCBlockN + v_tile_start;
#pragma unroll
          for (int token = 0; token < kXQATCStagedPVTileRows; ++token) {
            if (token >= valid_v_rows) {
              break;
            }
            float prob = 0.f;
            if (lane_id == 0) {
              prob = __half2float(probability_ptr[token]);
            }
            prob = __shfl_sync(0xffffffffu, prob, 0, kWarpSize);
            const __half* sV_row = sV + token * SmemLayout::kKVStride;
#pragma unroll
            for (int d_iter = 0; d_iter < (kPanelDim / kWarpSize); ++d_iter) {
              const int local_d = lane_id + d_iter * kWarpSize;
              const int acc_idx = panel_idx * (kPanelDim / kWarpSize) + d_iter;
              out_acc[acc_idx] =
                  fmaf(prob, __half2float(sV_row[local_d]), out_acc[acc_idx]);
            }
          }
        }
        __syncthreads();
      }
    }
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    const int head_idx = q_head_base + row;
    const float row_sum =
        exp_sums[static_cast<int64_t>(batch_idx) * stats_stride0 +
                 static_cast<int64_t>(head_idx) * stats_stride1 +
                 partition_idx];
    const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
    __half* tmp_out_ptr = tmp_out +
                          static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
                          static_cast<int64_t>(head_idx) * tmp_out_stride1 +
                          static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    for (int d = thread_in_row; d < D; d += kXQATC256WideThreadsPerRow) {
      tmp_out_ptr[d] = __float2half(out_acc[d / kWarpSize] * inv_row_sum);
    }
  }
}

template <int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens>
__global__ void flash_attention_decode_xqa_reduce_stats_kernel(
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ seq_lens, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t stats_stride0, const int64_t stats_stride1,
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
  const int partition_size =
      PARTITION_SIZE == 0
          ? xqa_sawtooth_partition_size(seq_len, route_seq_len_begin,
                                        route_seq_len_end, route_seq_len_final)
          : PARTITION_SIZE;
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  if (seq_len <= 0 || num_partitions <= 0) {
    return;
  }

  extern __shared__ float max_shared[];
  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;

  float local_max = -1.0e20f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float m = max_logits[stats_base + i];
    max_shared[i] = m;
    local_max = fmaxf(local_max, m);
  }
  const float global_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float weight =
        exp_sums[stats_base + i] * __expf(max_shared[i] - global_max);
    max_logits[stats_base + i] = weight;
    local_sum += weight;
  }
  const float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  if (threadIdx.x == 0) {
    // The partition sum is dead after weights have been materialized.
    exp_sums[stats_base] = global_sum;
  }
}

template <int D, int PARTITION_SIZE, int D_TILE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens>
__global__ void flash_attention_decode_xqa_reduce_output_kernel(
    const __half* __restrict__ tmp_out, const float* __restrict__ weights,
    const float* __restrict__ global_sums, const int* __restrict__ seq_lens,
    __half* __restrict__ out, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t out_stride0,
    const int64_t out_stride1, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final) {
  static_assert(D_TILE > 0 && D_TILE <= kWarpSize,
                "Split-reduce dimension tile must fit in one warp");
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int d = blockIdx.z * D_TILE + threadIdx.x;
  if (batch_idx >= batch_size || head_idx >= num_heads_q || d >= D) {
    return;
  }

  const int64_t out_index = static_cast<int64_t>(batch_idx) * out_stride0 +
                            static_cast<int64_t>(head_idx) * out_stride1 + d;
  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  const int partition_size =
      PARTITION_SIZE == 0
          ? xqa_sawtooth_partition_size(seq_len, route_seq_len_begin,
                                        route_seq_len_end, route_seq_len_final)
          : PARTITION_SIZE;
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  if (seq_len <= 0 || num_partitions <= 0) {
    out[out_index] = __float2half(0.f);
    return;
  }

  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;
  const float global_sum = global_sums[stats_base];
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;

  float acc = 0.f;
  for (int i = 0; i < num_partitions; ++i) {
    acc = fmaf(
        weights[stats_base + i],
        __half2float(tmp_out[tmp_out_base +
                             static_cast<int64_t>(i) * tmp_out_stride2 + d]),
        acc);
  }
  out[out_index] = __float2half(acc * inv_global_sum);
}

template <int D, int PARTITION_SIZE, int KV_DTYPE>
__global__ void flash_attention_decode_qk_scores_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    float* __restrict__ scores, const int batch_size, const int max_num_blocks,
    const int max_num_partitions, const int num_heads_q, const int num_heads_kv,
    const int block_size, const int64_t q_stride0, const int64_t q_stride1,
    const int64_t scores_stride0, const int64_t scores_stride1,
    const int64_t scores_stride2, const int64_t k_block_stride,
    const int64_t k_token_stride, const int64_t k_head_stride,
    const float softmax_scale, const float k_scale) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || head_idx >= num_heads_q ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }

  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int q_per_kv = num_heads_q / num_heads_kv;
  const int kv_head_idx = head_idx / q_per_kv;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;
  const float score_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                ? softmax_scale
                                : softmax_scale * k_scale;

  __shared__ __half q_shared[D];
  __shared__ int block_idx_shared[PARTITION_SIZE];
  __shared__ int block_offset_shared[PARTITION_SIZE];

  const __half* q_ptr = q + static_cast<int64_t>(batch_idx) * q_stride0 +
                        static_cast<int64_t>(head_idx) * q_stride1;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    q_shared[d] = q_ptr[d];
  }
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    const int token_idx = start_token_idx + i;
    const int logical_block = token_idx / block_size;
    block_idx_shared[i] =
        block_table[batch_idx * max_num_blocks + logical_block];
    block_offset_shared[i] = token_idx - logical_block * block_size;
  }
  __syncthreads();

  const int64_t score_base =
      static_cast<int64_t>(batch_idx) * scores_stride0 +
      static_cast<int64_t>(head_idx) * scores_stride1 +
      static_cast<int64_t>(partition_idx) * scores_stride2;
  for (int token_local = warp_idx; token_local < part_tokens;
       token_local += kWarpsPerBlock) {
    const int physical_block = block_idx_shared[token_local];
    const int block_offset = block_offset_shared[token_local];
    const int64_t k_index =
        static_cast<int64_t>(physical_block) * k_block_stride +
        static_cast<int64_t>(block_offset) * k_token_stride +
        static_cast<int64_t>(kv_head_idx) * k_head_stride;

    float score = dot_qk_cache<D, KV_DTYPE>(q_shared, k_cache, k_index, lane);
    if (lane == 0) {
      scores[score_base + token_local] = score * score_scale;
    }
  }
}

template <int D, int PARTITION_SIZE, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens>
void launch_flash_attention_decode_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    at::Tensor& out, const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& tmp_out, at::Tensor& max_logits, at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions, const float softmax_scale,
    const int launch_num_partitions, const float k_scale, const float v_scale,
    const int window_size_left, const int window_size_right,
    cudaStream_t stream, const int route_seq_len_begin = 0,
    const int route_seq_len_end = 0, const int route_seq_len_final = 0,
    const bool launch_reduce = true, const int* anchor_lens = nullptr,
    const int anchored_window = 0) {
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int block_size = k_cache.size(1);
  const int max_num_blocks = block_table.size(1);
  const int max_num_partitions = launch_num_partitions;

  const dim3 partition_grid(batch_size, num_heads_q, max_num_partitions);
  const dim3 reduce_grid(batch_size, num_heads_q, 1);
  const dim3 block(kThreadsPerBlock);
  const size_t reduce_shared_mem =
      static_cast<size_t>(2 * max_num_partitions) * sizeof(float);

  const bool use_anchored = anchor_lens != nullptr && anchored_window > 0;
  // Second kernel version: the anchored decode-window mask is a separate
  // template instantiation, generated only for the fp16-KV configuration;
  // the non-anchored instantiations stay untouched.
  const auto launch_partition = [&](auto anchored_tag) {
    constexpr bool kAnchored = decltype(anchored_tag)::value;
    flash_attention_decode_partition_kernel<D, PARTITION_SIZE, KV_DTYPE,
                                            SEQ_LEN_ROUTE, kAnchored>
        <<<partition_grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            k_cache.data_ptr(), v_cache.data_ptr(),
            reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
            active_num_partitions.data_ptr<int>(), batch_size, max_num_blocks,
            max_num_partitions, num_heads_q, num_heads_kv, block_size,
            q.stride(0), q.stride(1), tmp_out.stride(0), tmp_out.stride(1),
            tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            softmax_scale, k_scale, v_scale, window_size_left,
            window_size_right, route_seq_len_begin, route_seq_len_end,
            route_seq_len_final, anchor_lens, anchored_window);
  };
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    if (use_anchored) {
      launch_partition(std::true_type{});
    } else {
      launch_partition(std::false_type{});
    }
  } else {
    TORCH_CHECK(!use_anchored,
                "anchored decode window requires an fp16 KV cache");
    launch_partition(std::false_type{});
  }

  if (!launch_reduce) {
    return;
  }

  flash_attention_decode_reduce_kernel<D, PARTITION_SIZE>
      <<<reduce_grid, block, reduce_shared_mem, stream>>>(
          reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
          max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
          seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
          reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size,
          max_num_partitions, num_heads_q, tmp_out.stride(0), tmp_out.stride(1),
          tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),
          out.stride(0), out.stride(1), 0, 0, 0, 0);
}

template <int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens>
void launch_flash_attention_decode_xqa_split_reduce(
    at::Tensor& out, const at::Tensor& seq_lens, const at::Tensor& tmp_out,
    at::Tensor& max_logits, at::Tensor& exp_sums,
    const int launch_num_partitions, const int dim_tile, cudaStream_t stream,
    const int route_seq_len_begin = 0, const int route_seq_len_end = 0,
    const int route_seq_len_final = 0) {
  const int batch_size = out.size(0);
  const int num_heads_q = out.size(1);
  const dim3 stats_grid(batch_size, num_heads_q, 1);
  const dim3 stats_block(kThreadsPerBlock);
  const size_t stats_shared_mem =
      static_cast<size_t>(launch_num_partitions) * sizeof(float);
  flash_attention_decode_xqa_reduce_stats_kernel<PARTITION_SIZE, SEQ_LEN_ROUTE>
      <<<stats_grid, stats_block, stats_shared_mem, stream>>>(
          max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
          seq_lens.data_ptr<int>(), batch_size, launch_num_partitions,
          num_heads_q, max_logits.stride(0), max_logits.stride(1),
          route_seq_len_begin, route_seq_len_end, route_seq_len_final);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_SPLIT_REDUCE_OUTPUT(D_TILE)                                   \
  do {                                                                       \
    const dim3 output_grid(batch_size, num_heads_q,                          \
                           (256 + D_TILE - 1) / D_TILE);                     \
    const dim3 output_block(D_TILE);                                         \
    flash_attention_decode_xqa_reduce_output_kernel<256, PARTITION_SIZE,     \
                                                    D_TILE, SEQ_LEN_ROUTE>   \
        <<<output_grid, output_block, 0, stream>>>(                          \
            reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),   \
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),        \
            seq_lens.data_ptr<int>(),                                        \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size, \
            launch_num_partitions, num_heads_q, tmp_out.stride(0),           \
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),      \
            max_logits.stride(1), out.stride(0), out.stride(1),              \
            route_seq_len_begin, route_seq_len_end, route_seq_len_final);    \
  } while (0)

  switch (dim_tile) {
    case 16:
      LAUNCH_SPLIT_REDUCE_OUTPUT(16);
      break;
    case 32:
      LAUNCH_SPLIT_REDUCE_OUTPUT(32);
      break;
    default:
      LAUNCH_SPLIT_REDUCE_OUTPUT(8);
      break;
  }
#undef LAUNCH_SPLIT_REDUCE_OUTPUT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM,
          int NUM_THREADS = kXQATC256WideThreads, int MIN_BLOCKS_PER_SM = 1,
          int BLOCK_SIZE = 0, bool CONTIGUOUS_HKV1_LAYOUT = false,
          bool ALIGNED_PADDED_SMEM = false,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool QK_SW_PIPELINE = false,
          bool PARTITION_PAGE_IDS = false, bool FP8_PAIR_LOAD = false,
          int KV_DTYPE_OVERRIDE = -1, bool E4M3_SHARED_LUT = false>
void launch_flash_attention_decode_paged_xqa_tc_256_wide(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    at::Tensor& out, const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& tmp_out, at::Tensor& max_logits, at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions, const float softmax_scale,
    const float k_scale, const float v_scale, const int launch_num_partitions,
    const bool use_split_reduce, const int split_reduce_dim_tile,
    cudaStream_t stream, const int route_seq_len_begin = 0,
    const int route_seq_len_end = 0, const int route_seq_len_final = 0,
    const bool launch_reduce = true) {
  static_assert(!E4M3_SHARED_LUT ||
                    KV_DTYPE_OVERRIDE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3,
                "The shared conversion LUT requires an E4M3 specialization");
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int max_num_blocks = block_table.size(1);
  const dim3 partition_grid(batch_size, num_heads_kv, launch_num_partitions);
  using SmemLayout = std::conditional_t<
      QK_SW_PIPELINE,
      XQATCQKPipelineSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>,
      XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>>;
  const size_t shared_mem =
      sizeof(SmemLayout) + (E4M3_SHARED_LUT ? 256 * sizeof(uint16_t) : 0);
#define LAUNCH_XQA_PARTITION(KV_DTYPE)                                         \
  do {                                                                         \
    auto partition_kernel =                                                    \
        (void*)flash_attention_decode_xqa_tc_partition_kernel_256_wide<        \
            PARTITION_SIZE, GROUP_SIZE, PADDED_SMEM, NUM_THREADS,              \
            MIN_BLOCKS_PER_SM, BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,             \
            ALIGNED_PADDED_SMEM, KV_DTYPE, SEQ_LEN_ROUTE, QK_SW_PIPELINE,      \
            PARTITION_PAGE_IDS, FP8_PAIR_LOAD, E4M3_SHARED_LUT>;               \
    cudaFuncSetAttribute(partition_kernel,                                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         shared_mem);                                          \
    if constexpr (MIN_BLOCKS_PER_SM > 1) {                                     \
      cudaFuncSetAttribute(partition_kernel,                                   \
                           cudaFuncAttributePreferredSharedMemoryCarveout,     \
                           100);                                               \
    }                                                                          \
    flash_attention_decode_xqa_tc_partition_kernel_256_wide<                   \
        PARTITION_SIZE, GROUP_SIZE, PADDED_SMEM, NUM_THREADS,                  \
        MIN_BLOCKS_PER_SM, BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,                 \
        ALIGNED_PADDED_SMEM, KV_DTYPE, SEQ_LEN_ROUTE, QK_SW_PIPELINE,          \
        PARTITION_PAGE_IDS, FP8_PAIR_LOAD, E4M3_SHARED_LUT>                    \
        <<<partition_grid, NUM_THREADS, shared_mem, stream>>>(                 \
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),           \
            k_cache.data_ptr(), v_cache.data_ptr(),                            \
            reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),           \
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),          \
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),             \
            active_num_partitions.data_ptr<int>(), batch_size, max_num_blocks, \
            launch_num_partitions, num_heads_q, num_heads_kv, k_cache.size(1), \
            q.stride(0), q.stride(1), tmp_out.stride(0), tmp_out.stride(1),    \
            tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),     \
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),           \
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),           \
            softmax_scale, k_scale, v_scale, route_seq_len_begin,              \
            route_seq_len_end, route_seq_len_final);                           \
  } while (0)

  if constexpr (KV_DTYPE_OVERRIDE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3) {
    static_assert(!(FP8_PAIR_LOAD && E4M3_SHARED_LUT),
                  "Paired E4M3 conversion and the shared LUT are exclusive");
    TORCH_CHECK(k_cache.scalar_type() == at::kByte,
                "E4M3 XQA requires uint8 KV cache");
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP8_E4M3);
  } else if constexpr (FP8_PAIR_LOAD) {
    static_assert(!E4M3_SHARED_LUT, "The E4M3 conversion LUT requires E4M3 KV");
    TORCH_CHECK(k_cache.scalar_type() == at::kByte,
                "Paired XQA loads require uint8 FP8 KV cache");
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP8_E5M2);
  } else if (k_cache.scalar_type() == at::kByte) {
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP8_E5M2);
  } else {
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP16);
  }
#undef LAUNCH_XQA_PARTITION
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (!launch_reduce) {
    return;
  }

  if constexpr (PARTITION_SIZE == -1) {
    TORCH_CHECK(!use_split_reduce,
                "Runtime wave partitions do not support split reduction");
  }
  if (use_split_reduce) {
    if constexpr (PARTITION_SIZE != -1) {
      launch_flash_attention_decode_xqa_split_reduce<PARTITION_SIZE>(
          out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
          split_reduce_dim_tile, stream);
    }
  } else {
    const dim3 reduce_grid(batch_size, num_heads_q, 1);
    const dim3 block(kThreadsPerBlock);
    const size_t reduce_shared_mem =
        static_cast<size_t>(2 * launch_num_partitions) * sizeof(float);
    flash_attention_decode_reduce_kernel<256, PARTITION_SIZE, SEQ_LEN_ROUTE>
        <<<reduce_grid, block, reduce_shared_mem, stream>>>(
            reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
            seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size,
            launch_num_partitions, num_heads_q, tmp_out.stride(0),
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),
            max_logits.stride(1), out.stride(0), out.stride(1),
            route_seq_len_end > 0 ? route_seq_len_end : route_seq_len_begin,
            route_seq_len_begin, route_seq_len_end, route_seq_len_final);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens,
          bool FIXED_INTERLEAVED_HKV1_LAYOUT = false>
void launch_flash_attention_decode_paged_xqa_e4m3_g6_page1568(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    at::Tensor& out, const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& tmp_out, at::Tensor& max_logits, at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions, const float softmax_scale,
    const float k_scale, const float v_scale, const int launch_num_partitions,
    cudaStream_t stream, const int route_seq_len_begin = 0,
    const int route_seq_len_end = 0, const int route_seq_len_final = 0,
    const bool launch_reduce = true) {
  launch_flash_attention_decode_paged_xqa_tc_256_wide<
      PARTITION_SIZE, 6, true, kXQATCG6DualCtaThreads, 2, 1568,
      FIXED_INTERLEAVED_HKV1_LAYOUT, false, SEQ_LEN_ROUTE, false, true, false,
      flash_v100::KV_CACHE_DTYPE_FP8_E4M3, true>(
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
      exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
      launch_num_partitions, false, 8, stream, route_seq_len_begin,
      route_seq_len_end, route_seq_len_final, launch_reduce);
}

void launch_flash_attention_decode_paged_xqa_tc_256_staged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    at::Tensor& out, const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& tmp_out, at::Tensor& max_logits, at::Tensor& exp_sums,
    at::Tensor& online_rescales, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int launch_num_partitions,
    const bool use_split_reduce, const int split_reduce_dim_tile,
    cudaStream_t stream) {
  constexpr int kGroupSize = 6;
  constexpr int kQKThreads = kXQATCG6DualCtaThreads;
  constexpr int kPVThreads = kXQATCG6DualCtaThreads;
  constexpr int kQKMinBlocksPerSM = 2;
  constexpr int kPVMinBlocksPerSM = 4;
  using QKSmem = XQATCSmem256WideLayout<true>;
  using PVSmem = XQATCStagedPVSmem256Wide<true>;

  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int max_num_blocks = block_table.size(1);
  const dim3 partition_grid(batch_size, num_heads_kv, launch_num_partitions);
  const size_t qk_shared_mem = sizeof(QKSmem);
  const size_t pv_shared_mem = sizeof(PVSmem);
  auto qk_kernel =
      (void*)flash_attention_decode_xqa_tc_qk_softmax_staged_kernel_256_wide<
          256, kGroupSize, true, kQKThreads, kQKMinBlocksPerSM>;
  auto pv_kernel =
      (void*)flash_attention_decode_xqa_tc_pv_staged_kernel_256_wide<
          256, kGroupSize, true, kPVThreads, kPVMinBlocksPerSM>;
  const cudaError_t qk_smem_status = cudaFuncSetAttribute(
      qk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, qk_shared_mem);
  TORCH_CHECK(qk_smem_status == cudaSuccess,
              "Failed to set staged XQA QK shared memory: ",
              cudaGetErrorString(qk_smem_status));
  const cudaError_t qk_carveout_status = cudaFuncSetAttribute(
      qk_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(qk_carveout_status == cudaSuccess,
              "Failed to set staged XQA QK shared-memory carveout: ",
              cudaGetErrorString(qk_carveout_status));
  const cudaError_t pv_smem_status = cudaFuncSetAttribute(
      pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pv_shared_mem);
  TORCH_CHECK(pv_smem_status == cudaSuccess,
              "Failed to set staged XQA PV shared memory: ",
              cudaGetErrorString(pv_smem_status));
  const cudaError_t pv_carveout_status = cudaFuncSetAttribute(
      pv_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(pv_carveout_status == cudaSuccess,
              "Failed to set staged XQA PV shared-memory carveout: ",
              cudaGetErrorString(pv_carveout_status));

  flash_attention_decode_xqa_tc_qk_softmax_staged_kernel_256_wide<
      256, kGroupSize, true, kQKThreads, kQKMinBlocksPerSM>
      <<<partition_grid, kQKThreads, qk_shared_mem, stream>>>(
          reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),
          max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
          online_rescales.data_ptr<float>(), block_table.data_ptr<int>(),
          seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
          batch_size, max_num_blocks, launch_num_partitions, num_heads_q,
          num_heads_kv, k_cache.size(1), q.stride(0), q.stride(1),
          tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2),
          max_logits.stride(0), max_logits.stride(1), online_rescales.stride(0),
          online_rescales.stride(1), k_cache.stride(0), k_cache.stride(1),
          k_cache.stride(2), softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  flash_attention_decode_xqa_tc_pv_staged_kernel_256_wide<
      256, kGroupSize, true, kPVThreads, kPVMinBlocksPerSM>
      <<<partition_grid, kPVThreads, pv_shared_mem, stream>>>(
          reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),
          exp_sums.data_ptr<float>(), online_rescales.data_ptr<float>(),
          block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
          active_num_partitions.data_ptr<int>(), batch_size, max_num_blocks,
          launch_num_partitions, num_heads_q, num_heads_kv, v_cache.size(1),
          tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2),
          max_logits.stride(0), max_logits.stride(1), online_rescales.stride(0),
          online_rescales.stride(1), v_cache.stride(0), v_cache.stride(1),
          v_cache.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (use_split_reduce) {
    launch_flash_attention_decode_xqa_split_reduce<256>(
        out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
        split_reduce_dim_tile, stream);
  } else {
    const dim3 reduce_grid(batch_size, num_heads_q, 1);
    const dim3 reduce_block(kThreadsPerBlock);
    const size_t reduce_shared_mem =
        static_cast<size_t>(2 * launch_num_partitions) * sizeof(float);
    flash_attention_decode_reduce_kernel<256, 256>
        <<<reduce_grid, reduce_block, reduce_shared_mem, stream>>>(
            reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
            seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size,
            launch_num_partitions, num_heads_q, tmp_out.stride(0),
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),
            max_logits.stride(1), out.stride(0), out.stride(1), 0, 0, 0, 0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int D, int PARTITION_SIZE, int KV_DTYPE>
void launch_flash_attention_decode_qk_scores(
    const at::Tensor& q, const at::Tensor& k_cache,
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& scores, const float softmax_scale, const float k_scale,
    cudaStream_t stream) {
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int block_size = k_cache.size(1);
  const int max_num_blocks = block_table.size(1);
  const int max_num_partitions = scores.size(2);

  const dim3 grid(batch_size, num_heads_q, max_num_partitions);
  const dim3 block(kThreadsPerBlock);

  flash_attention_decode_qk_scores_kernel<D, PARTITION_SIZE, KV_DTYPE>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
          k_cache.data_ptr(), block_table.data_ptr<int>(),
          seq_lens.data_ptr<int>(), scores.data_ptr<float>(), batch_size,
          max_num_blocks, max_num_partitions, num_heads_q, num_heads_kv,
          block_size, q.stride(0), q.stride(1), scores.stride(0),
          scores.stride(1), scores.stride(2), k_cache.stride(0),
          k_cache.stride(1), k_cache.stride(2), softmax_scale, k_scale);
}

constexpr int kGroupedSparseQueries = 8;
constexpr int kGroupedSparsePlannerThreads = 512;
constexpr int kGroupedSparseHashCapacity = 8192;
constexpr unsigned long long kGroupedSparseEmptyEntry = 0x00000000ffffffffULL;

__device__ __forceinline__ void grouped_sparse_hash_insert(
    unsigned long long* __restrict__ hash_table, const int physical_microblock,
    const uint32_t token_mask) {
  if (physical_microblock < 0 || token_mask == 0) {
    return;
  }
  int slot = (static_cast<uint32_t>(physical_microblock) * 2654435761u) &
             (kGroupedSparseHashCapacity - 1);
  const unsigned long long desired =
      (static_cast<unsigned long long>(token_mask) << 32) |
      static_cast<uint32_t>(physical_microblock);
#pragma unroll 1
  for (int probe = 0; probe < kGroupedSparseHashCapacity; ++probe) {
    const unsigned long long old =
        atomicCAS(hash_table + slot, kGroupedSparseEmptyEntry, desired);
    if (old == kGroupedSparseEmptyEntry) {
      return;
    }
    if (static_cast<uint32_t>(old) ==
        static_cast<uint32_t>(physical_microblock)) {
      atomicOr(hash_table + slot, static_cast<unsigned long long>(token_mask)
                                      << 32);
      return;
    }
    slot = (slot + 1) & (kGroupedSparseHashCapacity - 1);
  }
}

__device__ __forceinline__ int grouped_sparse_physical_microblock(
    const int token, const int request_idx,
    const int* __restrict__ request_block_table,
    const int64_t request_block_table_stride, const int block_table_width,
    const int page_size, const int physical_page_stride,
    const int num_cache_blocks) {
  if (token < 0) {
    return -1;
  }
  const int logical_page = token / page_size;
  if (logical_page < 0 || logical_page >= block_table_width) {
    return -1;
  }
  const int page_offset = token - logical_page * page_size;
  const int physical_page =
      __ldg(request_block_table +
            static_cast<int64_t>(request_idx) * request_block_table_stride +
            logical_page);
  if (physical_page < 0 || physical_page >= num_cache_blocks) {
    return -1;
  }
  return physical_page * physical_page_stride + page_offset / 4;
}

__device__ __forceinline__ int grouped_sparse_active_m_tiles(
    const uint32_t token_mask) {
  int active_m_tiles = 0;
#pragma unroll
  for (int query = 0; query < kGroupedSparseQueries; ++query) {
    if ((token_mask & (0xFu << (query * 4))) != 0) {
      const int first_row = query * kGroupedVerifyHeads;
      const int last_row = first_row + kGroupedVerifyHeads - 1;
      active_m_tiles |= 1 << (first_row / 16);
      active_m_tiles |= 1 << (last_row / 16);
    }
  }
  return active_m_tiles;
}

__global__
__launch_bounds__(kGroupedSparsePlannerThreads, 1) void grouped_sparse_page4_plan_kernel(
    const int* __restrict__ logical_indices,
    const int* __restrict__ request_block_table,
    const int* __restrict__ token_to_req,
    const int64_t* __restrict__ query_positions,
    const int* __restrict__ sequence_lengths, int* __restrict__ output_blocks,
    uint32_t* __restrict__ output_masks, int* __restrict__ output_seq_lens,
    const int selection_width, const int64_t logical_indices_stride,
    const int64_t request_block_table_stride, const int num_requests,
    const int block_table_width, const int output_width, const int page_size,
    const int physical_page_stride, const int num_cache_blocks) {
  const int group_idx = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ int category_counts[8];
  __shared__ int category_offsets[8];
  __shared__ int category_cursors[8];
  __shared__ int
      warp_category_prefix[(kGroupedSparsePlannerThreads / kWarpSize) * 8];
  extern __shared__ unsigned long long hash_table[];
  for (int slot = tid; slot < kGroupedSparseHashCapacity;
       slot += kGroupedSparsePlannerThreads) {
    hash_table[slot] = kGroupedSparseEmptyEntry;
  }
  __syncthreads();

  const int full_page4_count = selection_width / 4;
  for (int selected_page = tid; selected_page < full_page4_count;
       selected_page += kGroupedSparsePlannerThreads) {
#pragma unroll
    for (int query = 0; query < kGroupedSparseQueries; ++query) {
      const int row = group_idx * kGroupedSparseQueries + query;
      const int request_idx = __ldg(token_to_req + row);
      if (request_idx < 0 || request_idx >= num_requests) {
        continue;
      }
      const int sequence_length = __ldg(sequence_lengths + request_idx);
      const int64_t query_visible_tokens = __ldg(query_positions + row) + 1;
      const int visible_tokens =
          query_visible_tokens <= 0
              ? 0
              : (query_visible_tokens < sequence_length
                     ? static_cast<int>(query_visible_tokens)
                     : max(sequence_length, 0));
      const int row_complete_page4_count =
          min(min(visible_tokens / 4, sequence_length / 4), full_page4_count);
      if (selected_page >= row_complete_page4_count) {
        continue;
      }
      const int* selected = logical_indices +
                            static_cast<int64_t>(row) * logical_indices_stride +
                            selected_page * 4;
      const int first_token = __ldg(selected);
      if (first_token < 0) {
        continue;
      }
      const bool full_page4 = __ldg(selected + 1) == first_token + 1 &&
                              __ldg(selected + 2) == first_token + 2 &&
                              __ldg(selected + 3) == first_token + 3 &&
                              (first_token & 3) == 0 &&
                              first_token + 3 < sequence_length;
      if (full_page4) {
        const int physical_microblock = grouped_sparse_physical_microblock(
            first_token, request_idx, request_block_table,
            request_block_table_stride, block_table_width, page_size,
            physical_page_stride, num_cache_blocks);
        grouped_sparse_hash_insert(hash_table, physical_microblock,
                                   0xFu << (query * 4));
      } else {
#pragma unroll
        for (int token_offset = 0; token_offset < 4; ++token_offset) {
          const int token = __ldg(selected + token_offset);
          if (token >= 0 && token < sequence_length) {
            const int physical_microblock = grouped_sparse_physical_microblock(
                token, request_idx, request_block_table,
                request_block_table_stride, block_table_width, page_size,
                physical_page_stride, num_cache_blocks);
            grouped_sparse_hash_insert(hash_table, physical_microblock,
                                       1u << (query * 4 + (token & 3)));
          }
        }
      }
    }
  }
  if (tid < kGroupedSparseQueries) {
    const int query = tid;
    const int row = group_idx * kGroupedSparseQueries + query;
    const int request_idx = __ldg(token_to_req + row);
    if (request_idx >= 0 && request_idx < num_requests) {
      const int sequence_length = __ldg(sequence_lengths + request_idx);
      const int64_t query_visible_tokens = __ldg(query_positions + row) + 1;
      const int visible_tokens =
          query_visible_tokens <= 0
              ? 0
              : (query_visible_tokens < sequence_length
                     ? static_cast<int>(query_visible_tokens)
                     : max(sequence_length, 0));
      const int complete_page4_count =
          min(min(visible_tokens / 4, sequence_length / 4), full_page4_count);
      const int tail_count = visible_tokens & 3;
      const int tail_index = complete_page4_count * 4;
      const int selected_tail_token =
          tail_index < selection_width
              ? __ldg(logical_indices +
                      static_cast<int64_t>(row) * logical_indices_stride +
                      tail_index)
              : -1;
      const int expected_tail_token = (visible_tokens / 4) * 4;
      if (tail_count > 0 && selected_tail_token == expected_tail_token &&
          selected_tail_token < sequence_length) {
        const int physical_microblock = grouped_sparse_physical_microblock(
            selected_tail_token, request_idx, request_block_table,
            request_block_table_stride, block_table_width, page_size,
            physical_page_stride, num_cache_blocks);
        const uint32_t tail_mask = ((1u << tail_count) - 1) << (query * 4);
        grouped_sparse_hash_insert(hash_table, physical_microblock, tail_mask);
      }
    }
  }
  __syncthreads();

  if (tid < 8) {
    category_counts[tid] = 0;
    category_offsets[tid] = 0;
    category_cursors[tid] = 0;
  }
  __syncthreads();
  for (int slot = tid; slot < kGroupedSparseHashCapacity;
       slot += kGroupedSparsePlannerThreads) {
    const unsigned long long entry = hash_table[slot];
    if (static_cast<uint32_t>(entry) != 0xffffffffu) {
      const int category =
          grouped_sparse_active_m_tiles(static_cast<uint32_t>(entry >> 32));
      atomicAdd(category_counts + category, 1);
    }
  }
  __syncthreads();
  if (tid == 0) {
    int padded_offset = 0;
#pragma unroll
    for (int category = 1; category < 8; ++category) {
      category_offsets[category] = padded_offset;
      padded_offset += (category_counts[category] + 7) & ~7;
    }
    category_offsets[0] = padded_offset;
  }
  __syncthreads();
  constexpr int kPlannerWarps = kGroupedSparsePlannerThreads / kWarpSize;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  for (int chunk_start = 0; chunk_start < kGroupedSparseHashCapacity;
       chunk_start += kGroupedSparsePlannerThreads) {
    const unsigned long long entry = hash_table[chunk_start + tid];
    const uint32_t physical_microblock = static_cast<uint32_t>(entry);
    int category = 0;
    if (physical_microblock != 0xffffffffu) {
      const uint32_t token_mask = static_cast<uint32_t>(entry >> 32);
      category = grouped_sparse_active_m_tiles(token_mask);
    }
    unsigned category_lanes = 0;
#pragma unroll
    for (int scan_category = 1; scan_category < 8; ++scan_category) {
      const unsigned lanes =
          __ballot_sync(0xffffffffu, category == scan_category);
      if (category == scan_category) {
        category_lanes = lanes;
      }
      if (lane == 0) {
        warp_category_prefix[warp * 8 + scan_category] = __popc(lanes);
      }
    }
    __syncthreads();
    if (tid > 0 && tid < 8) {
      int prefix = category_cursors[tid];
#pragma unroll
      for (int scan_warp = 0; scan_warp < kPlannerWarps; ++scan_warp) {
        const int count = warp_category_prefix[scan_warp * 8 + tid];
        warp_category_prefix[scan_warp * 8 + tid] = prefix;
        prefix += count;
      }
      category_cursors[tid] = prefix;
    }
    __syncthreads();
    if (category != 0) {
      const int category_rank = warp_category_prefix[warp * 8 + category] +
                                __popc(category_lanes & ((1u << lane) - 1));
      const int output_idx = category_offsets[category] + category_rank;
      if (output_idx < output_width) {
        output_blocks[static_cast<int64_t>(group_idx) * output_width +
                      output_idx] = static_cast<int>(physical_microblock);
        output_masks[static_cast<int64_t>(group_idx) * output_width +
                     output_idx] = static_cast<uint32_t>(entry >> 32);
      }
    }
    __syncthreads();
  }
  __syncthreads();
  if (tid > 0 && tid < 8) {
    const int category = tid;
    const int padded_count = (category_counts[category] + 7) & ~7;
    for (int local_idx = category_counts[category]; local_idx < padded_count;
         ++local_idx) {
      const int output_idx = category_offsets[category] + local_idx;
      if (output_idx < output_width) {
        output_blocks[static_cast<int64_t>(group_idx) * output_width +
                      output_idx] = 0;
        output_masks[static_cast<int64_t>(group_idx) * output_width +
                     output_idx] = 0;
      }
    }
  }
  if (tid == 0) {
    output_seq_lens[group_idx] = min(category_offsets[0], output_width) * 4;
  }
}

}  // namespace

at::Tensor flash_attention_grouped_sparse_page4_plan(
    const at::Tensor& logical_indices, const at::Tensor& block_table,
    const at::Tensor& token_to_req, const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths, at::Tensor& output_blocks,
    at::Tensor& output_masks, at::Tensor& output_seq_lens, const int page_size,
    const int physical_page_stride, const int num_cache_blocks) {
  TORCH_CHECK(logical_indices.is_cuda() && block_table.is_cuda() &&
                  token_to_req.is_cuda() && query_positions.is_cuda() &&
                  sequence_lengths.is_cuda() && output_blocks.is_cuda() &&
                  output_masks.is_cuda() && output_seq_lens.is_cuda(),
              "grouped sparse page4 planner tensors must be CUDA tensors");
  TORCH_CHECK(logical_indices.dtype() == torch::kInt32 &&
                  block_table.dtype() == torch::kInt32 &&
                  token_to_req.dtype() == torch::kInt32 &&
                  query_positions.dtype() == torch::kInt64 &&
                  sequence_lengths.dtype() == torch::kInt32 &&
                  output_blocks.dtype() == torch::kInt32 &&
                  output_masks.scalar_type() == at::ScalarType::UInt32 &&
                  output_seq_lens.dtype() == torch::kInt32,
              "grouped sparse page4 planner requires int32/uint32 metadata");
  TORCH_CHECK(logical_indices.dim() == 2 && logical_indices.size(0) > 0 &&
                  logical_indices.size(0) % kGroupedSparseQueries == 0 &&
                  logical_indices.size(1) == 2051,
              "grouped sparse page4 planner requires [8*N, 2051] indices");
  const int64_t num_groups = logical_indices.size(0) / kGroupedSparseQueries;
  TORCH_CHECK(
      block_table.dim() == 2 &&
          token_to_req.sizes() == at::IntArrayRef({logical_indices.size(0)}),
      "grouped sparse page4 planner request metadata is invalid");
  TORCH_CHECK(
      query_positions.sizes() == at::IntArrayRef({logical_indices.size(0)}) &&
          sequence_lengths.sizes() == at::IntArrayRef({block_table.size(0)}),
      "grouped sparse page4 planner visibility metadata is invalid");
  TORCH_CHECK(output_blocks.dim() == 2 && output_blocks.size(0) == num_groups &&
                  output_blocks.size(1) >= 4160 &&
                  output_masks.sizes() == output_blocks.sizes() &&
                  output_seq_lens.sizes() == at::IntArrayRef({num_groups}),
              "grouped sparse page4 planner outputs must be [groups, >=4160]");
  TORCH_CHECK(
      logical_indices.is_contiguous() && block_table.is_contiguous() &&
          token_to_req.is_contiguous() && query_positions.is_contiguous() &&
          sequence_lengths.is_contiguous() && output_blocks.is_contiguous() &&
          output_masks.is_contiguous() && output_seq_lens.is_contiguous(),
      "grouped sparse page4 planner metadata must be contiguous");
  TORCH_CHECK(page_size > 0 && page_size % 4 == 0 && physical_page_stride > 0 &&
                  num_cache_blocks > 0,
              "grouped sparse page4 planner requires page_size divisible by 4");
  TORCH_CHECK(logical_indices.device() == block_table.device() &&
                  logical_indices.device() == token_to_req.device() &&
                  logical_indices.device() == query_positions.device() &&
                  logical_indices.device() == sequence_lengths.device() &&
                  logical_indices.device() == output_blocks.device() &&
                  logical_indices.device() == output_masks.device() &&
                  logical_indices.device() == output_seq_lens.device(),
              "grouped sparse page4 planner tensors must share one device");

  c10::cuda::CUDAGuard device_guard(logical_indices.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "grouped sparse page4 planner supports SM70 only");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr size_t kPlannerSharedMemory =
      kGroupedSparseHashCapacity * sizeof(unsigned long long);
  const cudaError_t smem_status = cudaFuncSetAttribute(
      grouped_sparse_page4_plan_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kPlannerSharedMemory);
  TORCH_CHECK(smem_status == cudaSuccess,
              "Failed to set grouped sparse page4 planner shared memory: ",
              cudaGetErrorString(smem_status));
  grouped_sparse_page4_plan_kernel<<<static_cast<unsigned>(num_groups),
                                     kGroupedSparsePlannerThreads,
                                     kPlannerSharedMemory, stream>>>(
      logical_indices.data_ptr<int>(), block_table.data_ptr<int>(),
      token_to_req.data_ptr<int>(), query_positions.data_ptr<int64_t>(),
      sequence_lengths.data_ptr<int>(), output_blocks.data_ptr<int>(),
      output_masks.data_ptr<uint32_t>(), output_seq_lens.data_ptr<int>(),
      static_cast<int>(logical_indices.size(1)), logical_indices.stride(0),
      block_table.stride(0), static_cast<int>(block_table.size(0)),
      static_cast<int>(block_table.size(1)),
      static_cast<int>(output_blocks.size(1)), page_size, physical_page_stride,
      num_cache_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output_blocks;
}

int64_t flash_attention_grouped_verify_max_query_tokens() {
  return kGroupedVerifyMaxSupportedQ;
}

at::Tensor flash_attention_grouped_verify_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& partial_out,
    at::Tensor& partial_lse, const float softmax_scale,
    const std::string& kv_cache_dtype, const float k_scale, const float v_scale,
    const bool one_pass) {
  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
              "grouped verify q/K/V must be CUDA tensors");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "grouped verify metadata must be CUDA tensors");
  TORCH_CHECK(partial_out.is_cuda() && partial_lse.is_cuda(),
              "grouped verify workspaces must be CUDA tensors");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "grouped verify q must be fp16");
  TORCH_CHECK(kv_cache_dtype == "fp8_e5m2",
              "grouped verify prototype supports fp8_e5m2 KV only");
  TORCH_CHECK(
      k_cache.dtype() == torch::kUInt8 && v_cache.dtype() == torch::kUInt8,
      "grouped verify E5M2 cache must use uint8 storage");
  TORCH_CHECK(
      block_table.dtype() == torch::kInt32 && seq_lens.dtype() == torch::kInt32,
      "grouped verify block_table/seq_lens must be int32");
  TORCH_CHECK(partial_out.dtype() == torch::kFloat16 &&
                  partial_lse.dtype() == torch::kFloat32,
              "grouped verify workspaces must be fp16/fp32");
  TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 &&
                  q.size(0) <= kGroupedVerifyMaxSupportedQ &&
                  q.size(1) == kGroupedVerifyHeads &&
                  q.size(2) == kGroupedVerifyHeadDim,
              "grouped verify q must have shape [1..16, 6, 256]");
  const bool wide_query = q.size(0) > kGroupedVerifyQ8MaxQ;
  const int max_query_tokens =
      wide_query ? kGroupedVerifyQ16MaxQ : kGroupedVerifyQ8MaxQ;
  const int grouped_splits = kGroupedVerifyWorkspaceRows / max_query_tokens;
  const int heads_per_cta = kGroupedVerifyRows / max_query_tokens;
  const int head_groups = kGroupedVerifyHeads / heads_per_cta;
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 &&
                  k_cache.sizes() == v_cache.sizes() && k_cache.size(1) > 0 &&
                  k_cache.size(2) == 1 &&
                  k_cache.size(3) == kGroupedVerifyHeadDim,
              "grouped verify KV must have shape [blocks, page, 1, 256]");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == 1,
              "grouped verify block_table must have shape [1, blocks]");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) >= 1,
              "grouped verify seq_lens must cover one sequence");
  TORCH_CHECK(q.is_contiguous(),
              "grouped verify q must be contiguous [M, H, D]");
  TORCH_CHECK(block_table.is_contiguous() && seq_lens.is_contiguous(),
              "grouped verify metadata must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
              "grouped verify KV head dimension must be contiguous");
  TORCH_CHECK(partial_out.is_contiguous() && partial_lse.is_contiguous(),
              "grouped verify workspaces must be contiguous");
  TORCH_CHECK(partial_out.sizes() ==
                  at::IntArrayRef({grouped_splits, max_query_tokens,
                                   kGroupedVerifyHeads, kGroupedVerifyHeadDim}),
              "partial_out must have shape [80, 8, 6, 256] or "
              "[40, 16, 6, 256]");
  TORCH_CHECK(
      partial_lse.sizes() == at::IntArrayRef({grouped_splits, max_query_tokens,
                                              kGroupedVerifyHeads}),
      "partial_lse must have shape [80, 8, 6] or [40, 16, 6]");
  TORCH_CHECK(k_scale > 0.0f && v_scale > 0.0f,
              "grouped verify E5M2 K/V scales must be positive");

  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda() && out.device() == q.device() &&
                  out.dtype() == torch::kFloat16 && out.sizes() == q.sizes() &&
                  out.is_contiguous(),
              "grouped verify out must be contiguous fp16 and q-shaped");
  TORCH_CHECK(
      q.device() == k_cache.device() && q.device() == v_cache.device() &&
          q.device() == block_table.device() &&
          q.device() == seq_lens.device() &&
          q.device() == partial_out.device() &&
          q.device() == partial_lse.device() && q.device() == out.device(),
      "all grouped verify tensors must be on the same device");

  c10::cuda::CUDAGuard device_guard(q.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "grouped verify prototype supports SM70 only");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const dim3 partial_grid(head_groups, grouped_splits, 1);
  const size_t partial_shared_mem = sizeof(GroupedVerifySmem);
#define LAUNCH_GROUPED_VERIFY_PARTIAL(MAX_QUERY_TOKENS, TWO_PASS, PAGE_SIZE,   \
                                      SINGLE_QUERY, CONTIGUOUS_LAYOUT,         \
                                      STAGE_PAGE_IDS)                          \
  do {                                                                         \
    auto partial_kernel =                                                      \
        (void*)flash_attention_grouped_verify_e5m2_partial_kernel<             \
            MAX_QUERY_TOKENS, TWO_PASS, PAGE_SIZE, SINGLE_QUERY,               \
            CONTIGUOUS_LAYOUT, STAGE_PAGE_IDS>;                                \
    const cudaError_t smem_status = cudaFuncSetAttribute(                      \
        partial_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,           \
        partial_shared_mem);                                                   \
    TORCH_CHECK(smem_status == cudaSuccess,                                    \
                "Failed to set packed grouped-verifier shared memory: ",       \
                cudaGetErrorString(smem_status));                              \
    const cudaError_t carveout_status = cudaFuncSetAttribute(                  \
        partial_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);  \
    TORCH_CHECK(carveout_status == cudaSuccess,                                \
                "Failed to set packed grouped-verifier shared carveout: ",     \
                cudaGetErrorString(carveout_status));                          \
    flash_attention_grouped_verify_e5m2_partial_kernel<                        \
        MAX_QUERY_TOKENS, TWO_PASS, PAGE_SIZE, SINGLE_QUERY,                   \
        CONTIGUOUS_LAYOUT, STAGE_PAGE_IDS>                                     \
        <<<partial_grid, kGroupedVerifyThreads, partial_shared_mem, stream>>>( \
            reinterpret_cast<const __half*>(q.data_ptr()), k_cache.data_ptr(), \
            v_cache.data_ptr(), block_table.data_ptr<int>(),                   \
            seq_lens.data_ptr<int>(),                                          \
            reinterpret_cast<__half*>(partial_out.data_ptr()),                 \
            partial_lse.data_ptr<float>(), static_cast<int>(q.size(0)),        \
            static_cast<int>(block_table.size(1)),                             \
            static_cast<int>(k_cache.size(1)), k_cache.stride(0),              \
            k_cache.stride(1), k_cache.stride(2), v_cache.stride(0),           \
            v_cache.stride(1), v_cache.stride(2), softmax_scale * k_scale,     \
            v_scale, nullptr, 1);                                              \
  } while (0)

#define DISPATCH_GROUPED_VERIFY_PARTIAL(MAX_QUERY_TOKENS, TWO_PASS,           \
                                        SINGLE_QUERY)                         \
  do {                                                                        \
    const int64_t page_size = k_cache.size(1);                                \
    const int64_t fixed_block_stride = 2 * page_size * kGroupedVerifyHeadDim; \
    const bool fixed_interleaved_layout =                                     \
        dflash2_grouped_fixed_interleaved_enabled() &&                        \
        (page_size == 1648 || page_size == 3296) &&                           \
        k_cache.stride(0) == fixed_block_stride &&                            \
        v_cache.stride(0) == fixed_block_stride &&                            \
        k_cache.stride(1) == kGroupedVerifyHeadDim &&                         \
        v_cache.stride(1) == kGroupedVerifyHeadDim &&                         \
        k_cache.stride(2) == kGroupedVerifyHeadDim &&                         \
        v_cache.stride(2) == kGroupedVerifyHeadDim;                           \
    const bool stage_page_ids =                                               \
        fixed_interleaved_layout && dflash2_grouped_stage_page_ids_enabled(); \
    if (stage_page_ids && page_size == 1648) {                                \
      LAUNCH_GROUPED_VERIFY_PARTIAL(MAX_QUERY_TOKENS, TWO_PASS, 1648,         \
                                    SINGLE_QUERY, true, true);                \
    } else if (stage_page_ids) {                                              \
      LAUNCH_GROUPED_VERIFY_PARTIAL(MAX_QUERY_TOKENS, TWO_PASS, 3296,         \
                                    SINGLE_QUERY, true, true);                \
    } else if (fixed_interleaved_layout && page_size == 1648) {               \
      LAUNCH_GROUPED_VERIFY_PARTIAL(MAX_QUERY_TOKENS, TWO_PASS, 1648,         \
                                    SINGLE_QUERY, true, false);               \
    } else if (fixed_interleaved_layout) {                                    \
      LAUNCH_GROUPED_VERIFY_PARTIAL(MAX_QUERY_TOKENS, TWO_PASS, 3296,         \
                                    SINGLE_QUERY, true, false);               \
    } else if (page_size == 1648) {                                           \
      LAUNCH_GROUPED_VERIFY_PARTIAL(MAX_QUERY_TOKENS, TWO_PASS, 1648,         \
                                    SINGLE_QUERY, false, false);              \
    } else if (page_size == 3296) {                                           \
      LAUNCH_GROUPED_VERIFY_PARTIAL(MAX_QUERY_TOKENS, TWO_PASS, 3296,         \
                                    SINGLE_QUERY, false, false);              \
    } else {                                                                  \
      LAUNCH_GROUPED_VERIFY_PARTIAL(MAX_QUERY_TOKENS, TWO_PASS, 0,            \
                                    SINGLE_QUERY, false, false);              \
    }                                                                         \
  } while (0)

  const bool single_query = q.size(0) == 1;
  if (wide_query && one_pass) {
    DISPATCH_GROUPED_VERIFY_PARTIAL(kGroupedVerifyQ16MaxQ, false, false);
  } else if (wide_query) {
    DISPATCH_GROUPED_VERIFY_PARTIAL(kGroupedVerifyQ16MaxQ, true, false);
  } else if (one_pass && single_query) {
    DISPATCH_GROUPED_VERIFY_PARTIAL(kGroupedVerifyQ8MaxQ, false, true);
  } else if (one_pass) {
    DISPATCH_GROUPED_VERIFY_PARTIAL(kGroupedVerifyQ8MaxQ, false, false);
  } else if (single_query) {
    DISPATCH_GROUPED_VERIFY_PARTIAL(kGroupedVerifyQ8MaxQ, true, true);
  } else {
    DISPATCH_GROUPED_VERIFY_PARTIAL(kGroupedVerifyQ8MaxQ, true, false);
  }
#undef DISPATCH_GROUPED_VERIFY_PARTIAL
#undef LAUNCH_GROUPED_VERIFY_PARTIAL
  const dim3 combine_grid(static_cast<unsigned>(q.size(0)), kGroupedVerifyHeads,
                          1);
#define LAUNCH_GROUPED_VERIFY_COMBINE(MAX_QUERY_TOKENS, SINGLE_QUERY)  \
  flash_attention_grouped_verify_e5m2_combine_kernel<MAX_QUERY_TOKENS, \
                                                     SINGLE_QUERY>     \
      <<<combine_grid, kGroupedVerifyThreads, 0, stream>>>(            \
          reinterpret_cast<const __half*>(partial_out.data_ptr()),     \
          partial_lse.data_ptr<float>(), seq_lens.data_ptr<int>(),     \
          reinterpret_cast<__half*>(out.data_ptr()),                   \
          static_cast<int>(q.size(0)))
  if (wide_query) {
    LAUNCH_GROUPED_VERIFY_COMBINE(kGroupedVerifyQ16MaxQ, false);
  } else if (single_query) {
    LAUNCH_GROUPED_VERIFY_COMBINE(kGroupedVerifyQ8MaxQ, true);
  } else {
    LAUNCH_GROUPED_VERIFY_COMBINE(kGroupedVerifyQ8MaxQ, false);
  }
#undef LAUNCH_GROUPED_VERIFY_COMBINE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor flash_attention_grouped_sparse_page4(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& token_masks, const at::Tensor& seq_lens, at::Tensor& lse,
    const float softmax_scale, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale) {
  constexpr int kQueriesPerGroup = kGroupedVerifyQ8MaxQ;
  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
              "grouped sparse page4 q/K/V must be CUDA tensors");
  TORCH_CHECK(block_table.is_cuda() && token_masks.is_cuda() &&
                  seq_lens.is_cuda() && lse.is_cuda(),
              "grouped sparse page4 metadata must be CUDA tensors");
  const bool e4m3_kv = kv_cache_dtype == "fp8" ||
                       kv_cache_dtype == "fp8_e4m3";
  TORCH_CHECK(q.dtype() == torch::kFloat16,
              "grouped sparse page4 requires fp16 queries");
  TORCH_CHECK(
      e4m3_kv ? (k_cache.dtype() == torch::kUInt8 &&
                  v_cache.dtype() == torch::kUInt8)
               : (k_cache.dtype() == torch::kFloat16 &&
                  v_cache.dtype() == torch::kFloat16),
      "grouped sparse page4 KV storage does not match kv_cache_dtype");
  TORCH_CHECK(e4m3_kv || kv_cache_dtype == "auto" ||
                  kv_cache_dtype == "float16",
              "grouped sparse page4 supports fp16 and fp8_e4m3 KV only");
  TORCH_CHECK(!e4m3_kv ||
                  (std::isfinite(k_scale) && std::isfinite(v_scale) &&
                   k_scale > 0.0f && v_scale > 0.0f),
              "grouped sparse page4 E4M3 scales must be finite and positive");
  TORCH_CHECK(block_table.dtype() == torch::kInt32 &&
                  seq_lens.dtype() == torch::kInt32 &&
                  token_masks.scalar_type() == at::ScalarType::UInt32,
              "grouped sparse page4 tables must be int32/uint32");
  TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 &&
                  q.size(0) % kQueriesPerGroup == 0 &&
                  q.size(1) == kGroupedVerifyHeads &&
                  q.size(2) == kGroupedVerifyHeadDim,
              "grouped sparse page4 q must have shape [8*N, 6, 256]");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 &&
                  k_cache.sizes() == v_cache.sizes() && k_cache.size(1) == 4 &&
                  k_cache.size(2) == 1 &&
                  k_cache.size(3) == kGroupedVerifyHeadDim,
              "grouped sparse page4 KV must have shape [blocks, 4, 1, 256]");
  const int64_t num_groups = q.size(0) / kQueriesPerGroup;
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == num_groups &&
                  token_masks.sizes() == block_table.sizes(),
              "grouped sparse page4 block IDs/masks must be [groups, pages]");
  TORCH_CHECK(seq_lens.sizes() == at::IntArrayRef({num_groups}),
              "grouped sparse page4 seq_lens must have shape [groups]");
  TORCH_CHECK(q.is_contiguous() && block_table.is_contiguous() &&
                  token_masks.is_contiguous() && seq_lens.is_contiguous(),
              "grouped sparse page4 q/metadata must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
              "grouped sparse page4 KV head dimension must be contiguous");
  TORCH_CHECK(
      lse.sizes() == at::IntArrayRef({q.size(0), kGroupedVerifyHeads}) &&
          lse.dtype() == torch::kFloat32 && lse.is_contiguous(),
      "grouped sparse page4 lse must be contiguous [rows, 6] fp32");

  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda() && out.device() == q.device() &&
                  out.dtype() == torch::kFloat16 && out.sizes() == q.sizes() &&
                  out.is_contiguous(),
              "grouped sparse page4 out must be contiguous fp16 and q-shaped");
  TORCH_CHECK(q.device() == k_cache.device() &&
                  q.device() == v_cache.device() &&
                  q.device() == block_table.device() &&
                  q.device() == token_masks.device() &&
                  q.device() == seq_lens.device() &&
                  q.device() == lse.device() && q.device() == out.device(),
              "all grouped sparse page4 tensors must be on the same device");

  c10::cuda::CUDAGuard device_guard(q.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "grouped sparse page4 supports SM70 only");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(1, 1, static_cast<unsigned>(num_groups));
  const size_t shared_mem = sizeof(GroupedVerifySmem);
#define LAUNCH_GROUPED_SPARSE_PAGE4(KV_DTYPE_CODE)                           \
  do {                                                                        \
    auto kernel =                                                             \
        (void*)flash_attention_grouped_verify_e5m2_partial_kernel<            \
            kQueriesPerGroup, false, 4, false, false, false, KV_DTYPE_CODE,   \
            true>;                                                            \
    const cudaError_t smem_status = cudaFuncSetAttribute(                     \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);     \
    TORCH_CHECK(smem_status == cudaSuccess,                                   \
                "Failed to set grouped sparse page4 shared memory: ",        \
                cudaGetErrorString(smem_status));                             \
    flash_attention_grouped_verify_e5m2_partial_kernel<                       \
        kQueriesPerGroup, false, 4, false, false, false, KV_DTYPE_CODE, true> \
        <<<grid, kGroupedVerifyThreads, shared_mem, stream>>>(                 \
            reinterpret_cast<const __half*>(q.data_ptr()),                    \
            k_cache.data_ptr(), v_cache.data_ptr(),                           \
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),            \
            reinterpret_cast<__half*>(out.data_ptr()), lse.data_ptr<float>(), \
            kQueriesPerGroup, static_cast<int>(block_table.size(1)), 4,       \
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),          \
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),          \
            softmax_scale * (e4m3_kv ? k_scale : 1.0f),                       \
            e4m3_kv ? v_scale : 1.0f, token_masks.data_ptr<uint32_t>(),       \
            static_cast<int>(num_groups));                                    \
  } while (0)
  if (e4m3_kv) {
    LAUNCH_GROUPED_SPARSE_PAGE4(flash_v100::KV_CACHE_DTYPE_FP8_E4M3);
  } else {
    LAUNCH_GROUPED_SPARSE_PAGE4(flash_v100::KV_CACHE_DTYPE_FP16);
  }
#undef LAUNCH_GROUPED_SPARSE_PAGE4
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor flash_attention_decode_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
              "k/v cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
              "workspace tensors must be on CUDA");
  TORCH_CHECK(active_num_partitions.is_cuda(),
              "active_num_partitions must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  TORCH_CHECK(kv_dtype_code >= 0,
              "Unsupported kv_cache_dtype: ", kv_cache_dtype);
  if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
    TORCH_CHECK(v_cache.dtype() == torch::kFloat16, "v_cache must be fp16");
  } else {
    TORCH_CHECK(k_cache.dtype() == torch::kUInt8,
                "fp8 k_cache must be stored as uint8");
    TORCH_CHECK(v_cache.dtype() == torch::kUInt8,
                "fp8 v_cache must be stored as uint8");
    TORCH_CHECK(k_scale > 0.f && v_scale > 0.f,
                "fp8 k/v scales must be positive");
  }
  TORCH_CHECK(tmp_out.dtype() == torch::kFloat16, "tmp_out must be fp16");
  TORCH_CHECK(max_logits.dtype() == torch::kFloat32, "max_logits must be fp32");
  TORCH_CHECK(exp_sums.dtype() == torch::kFloat32, "exp_sums must be fp32");
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(active_num_partitions.dtype() == torch::kInt32,
              "active_num_partitions must be int32");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4,
              "k_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(v_cache.dim() == 4,
              "v_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(tmp_out.dim() == 4, "tmp_out must have shape [B_cap, H, P, D]");
  TORCH_CHECK(max_logits.dim() == 3,
              "max_logits must have shape [B_cap, H, P]");
  TORCH_CHECK(exp_sums.dim() == 3, "exp_sums must have shape [B_cap, H, P]");
  TORCH_CHECK(
      active_num_partitions.dim() == 1 && active_num_partitions.numel() == 1,
      "active_num_partitions must have shape [1]");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1, "k_cache last dim must be contiguous");
  TORCH_CHECK(v_cache.stride(-1) == 1, "v_cache last dim must be contiguous");
  TORCH_CHECK(tmp_out.stride(-1) == 1, "tmp_out last dim must be contiguous");

  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int head_dim = q.size(2);
  const int num_heads_kv = k_cache.size(2);

  TORCH_CHECK(q.size(0) <= block_table.size(0),
              "block_table batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= seq_lens.size(0),
              "seq_lens batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= tmp_out.size(0),
              "tmp_out batch size must cover q batch size");
  TORCH_CHECK(num_heads_q == tmp_out.size(1),
              "tmp_out head dimension mismatch");
  TORCH_CHECK(head_dim == tmp_out.size(3), "tmp_out head_dim mismatch");
  TORCH_CHECK(max_logits.size(0) == tmp_out.size(0) &&
                  max_logits.size(1) == tmp_out.size(1) &&
                  max_logits.size(2) == tmp_out.size(2),
              "max_logits shape mismatch");
  TORCH_CHECK(exp_sums.sizes() == max_logits.sizes(),
              "exp_sums shape mismatch");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  TORCH_CHECK(k_cache.size(3) == head_dim, "k_cache head_dim mismatch");
  TORCH_CHECK(v_cache.size(3) == head_dim, "v_cache head_dim mismatch");
  TORCH_CHECK(
      partition_size == 256 || partition_size == 512 || partition_size == 1024,
      "Unsupported decode partition_size: ", partition_size);
  TORCH_CHECK(
      launch_num_partitions > 0 && launch_num_partitions <= tmp_out.size(2),
      "launch_num_partitions must be in (0, tmp_out.size(2)]");
  TORCH_CHECK(window_size_left >= -1 && window_size_right >= -1,
              "window sizes must be >= -1");

  TORCH_CHECK(anchored_window >= 0 && anchored_window <= INT_MAX,
              "anchored_window must fit in a non-negative int32");
  TORCH_CHECK((anchored_window > 0) == anchor_lens.has_value(),
              "anchor_lens and a positive anchored_window must be provided "
              "together");
  const bool use_anchored = anchor_lens.has_value();
  const int* anchor_lens_ptr = nullptr;
  if (use_anchored) {
    const at::Tensor& anchors = anchor_lens.value();
    TORCH_CHECK(anchors.is_cuda(), "anchor_lens must be on CUDA");
    TORCH_CHECK(anchors.device() == q.device(),
                "anchor_lens must be on the q device");
    TORCH_CHECK(anchors.dtype() == torch::kInt32, "anchor_lens must be int32");
    TORCH_CHECK(anchors.dim() == 1 && anchors.size(0) == batch_size,
                "anchor_lens must have shape [B]");
    TORCH_CHECK(anchors.is_contiguous(), "anchor_lens must be contiguous");
    TORCH_CHECK(kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16,
                "anchored decode window requires an fp16 KV cache");
    TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
                "anchored decode window cannot be combined with "
                "sliding-window attention");
    anchor_lens_ptr = anchors.data_ptr<int>();
  }

  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");
  TORCH_CHECK(out.stride(-1) == 1, "out last dim must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  c10::cuda::CUDAGuard device_guard(q.device());

#define LAUNCH_TYPED(HDIM, PARTITION, KV_DTYPE_CODE)                          \
  launch_flash_attention_decode_paged<HDIM, PARTITION, KV_DTYPE_CODE>(        \
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,   \
      exp_sums, active_num_partitions, softmax_scale, launch_num_partitions,  \
      k_scale, v_scale, window_size_left, window_size_right, stream, 0, 0, 0, \
      true, anchor_lens_ptr, static_cast<int>(anchored_window))

#define LAUNCH_BY_KV_DTYPE(HDIM, PARTITION)                                 \
  do {                                                                      \
    switch (kv_dtype_code) {                                                \
      case flash_v100::KV_CACHE_DTYPE_FP16:                                 \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP16);     \
        break;                                                              \
      case flash_v100::KV_CACHE_DTYPE_FP8_E4M3:                             \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E4M3); \
        break;                                                              \
      case flash_v100::KV_CACHE_DTYPE_FP8_E5M2:                             \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E5M2); \
        break;                                                              \
      default:                                                              \
        TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype); \
    }                                                                       \
  } while (0)

#define LAUNCH_BY_PARTITION(HDIM)                                           \
  do {                                                                      \
    switch (partition_size) {                                               \
      case 256:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 256);                                      \
        break;                                                              \
      case 512:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 512);                                      \
        break;                                                              \
      case 1024:                                                            \
        LAUNCH_BY_KV_DTYPE(HDIM, 1024);                                     \
        break;                                                              \
      default:                                                              \
        TORCH_CHECK(false,                                                  \
                    "Unsupported decode partition_size: ", partition_size); \
    }                                                                       \
  } while (0)

  switch (head_dim) {
    case 64:
      LAUNCH_BY_PARTITION(64);
      break;
    case 80:
      LAUNCH_BY_PARTITION(80);
      break;
    case 96:
      LAUNCH_BY_PARTITION(96);
      break;
    case 112:
      LAUNCH_BY_PARTITION(112);
      break;
    case 128:
      LAUNCH_BY_PARTITION(128);
      break;
    case 256:
      LAUNCH_BY_PARTITION(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head_dim for paged decode: ", head_dim);
  }

#undef LAUNCH_BY_PARTITION
#undef LAUNCH_BY_KV_DTYPE
#undef LAUNCH_TYPED

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor flash_attention_decode_paged_xqa(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const int batch_context_max_seq_len) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
              "k_cache and v_cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
              "decode workspaces must be on CUDA");
  TORCH_CHECK(active_num_partitions.is_cuda(),
              "active_num_partitions must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  TORCH_CHECK(kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16 ||
                  kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                  kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
              "XQA decode supports fp16, fp8_e4m3, and fp8_e5m2 KV cache "
              "only");
  if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16 &&
                    v_cache.dtype() == torch::kFloat16,
                "fp16 XQA requires fp16 K/V tensors");
  } else {
    TORCH_CHECK(
        k_cache.dtype() == torch::kUInt8 && v_cache.dtype() == torch::kUInt8,
        "FP8 XQA requires uint8 K/V tensors");
  }
  TORCH_CHECK(k_scale > 0.f && v_scale > 0.f,
              "XQA K/V scales must be positive");
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(active_num_partitions.dtype() == torch::kInt32,
              "active_num_partitions must be int32");
  TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
              "XQA decode does not support sliding-window attention");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
              "KV cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(
      active_num_partitions.dim() == 1 && active_num_partitions.numel() == 1,
      "active_num_partitions must have shape [1]");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
              "KV cache last dim must be contiguous");
  TORCH_CHECK(q.size(0) <= block_table.size(0),
              "block_table batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= seq_lens.size(0),
              "seq_lens batch size must cover q batch size");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "K/V cache shape mismatch");
  TORCH_CHECK(k_cache.size(3) == q.size(2), "KV head_dim mismatch");
  TORCH_CHECK(q.size(2) == 256, "XQA decode supports head_dim=256 only");
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  TORCH_CHECK(num_heads_kv > 0 && num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  const int q_per_kv = num_heads_q / num_heads_kv;
  TORCH_CHECK(q_per_kv == 4 || q_per_kv == 6 || q_per_kv == 8,
              "XQA decode supports q_per_kv in {4, 6, 8}, got ", q_per_kv);
  TORCH_CHECK(partition_size == 64 || partition_size == 128 ||
                  partition_size == 256 || partition_size == 512 ||
                  partition_size == 1024 ||
                  ((partition_size == 896 || partition_size == 1664) &&
                   kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E4M3),
              "Unsupported XQA decode partition_size: ", partition_size);
  TORCH_CHECK(launch_num_partitions > 0,
              "launch_num_partitions must be positive");
  TORCH_CHECK(tmp_out.dtype() == torch::kFloat16,
              "XQA decode tmp_out must be fp16");
  TORCH_CHECK(tmp_out.size(0) >= q.size(0) && tmp_out.size(1) >= q.size(1) &&
                  tmp_out.size(2) >= launch_num_partitions &&
                  tmp_out.size(3) == q.size(2),
              "tmp_out shape does not cover XQA launch");
  TORCH_CHECK(max_logits.size(0) >= q.size(0) &&
                  max_logits.size(1) >= q.size(1) &&
                  max_logits.size(2) >= launch_num_partitions,
              "max_logits shape does not cover XQA launch");
  TORCH_CHECK(exp_sums.size(0) >= q.size(0) && exp_sums.size(1) >= q.size(1) &&
                  exp_sums.size(2) >= launch_num_partitions,
              "exp_sums shape does not cover XQA launch");

  c10::cuda::CUDAGuard device_guard(q.device());
  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");
  TORCH_CHECK(out.stride(-1) == 1, "out last dim must be contiguous");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E4M3) {
    const bool e4m3_batch_allowed =
        q.size(0) > 1 && q.size(0) <= 16 && xqa_e4m3_batch_enabled();
    const bool use_large_partition =
        q.size(0) == 1 &&
        (partition_size == 512 || partition_size == 896 ||
         partition_size == 1024 || partition_size == 1664) &&
        k_cache.size(1) == 1568 && k_cache.size(2) == 1;
    TORCH_CHECK((q.size(0) == 1 || e4m3_batch_allowed) && q_per_kv == 6 &&
                    (partition_size == 64 || partition_size == 128 ||
                     partition_size == 256 || use_large_partition),
                "E4M3 XQA supports B=1, or B=2..16 when "
                "VLLM_FLASH_V100_E4M3_BATCH_XQA=1; q_per_kv=6 and D=256 are "
                "required. Page-1568/Hkv=1 B1 additionally supports partition "
                "sizes 512, 896, 1024, and 1664");
    const bool e4m3_batch_optimized =
        q.size(0) > 1 && xqa_e4m3_batch_optimized_enabled() &&
        k_cache.size(1) >= 256 && k_cache.size(1) % 16 == 0 &&
        k_cache.size(2) == 1 && k_cache.scalar_type() == at::kByte;
    const XQABatchContextRoute e4m3_batch_route =
        e4m3_batch_optimized && batch_context_max_seq_len > 0 && q.size(0) >= 4
            ? select_xqa_batch_context_route(
                  q.size(0), batch_context_max_seq_len, partition_size)
            : XQABatchContextRoute::kDisabled;
    const bool e4m3_dual_cta =
        e4m3_batch_route == XQABatchContextRoute::kDualCta ||
        e4m3_batch_route == XQABatchContextRoute::kDualCtaSplit;
    const bool e4m3_split_reduce =
        e4m3_batch_route == XQABatchContextRoute::kDualCtaSplit;
    constexpr int64_t kPage800HeadDim256Elements = 800 * 256;
    const bool e4m3_page800_contiguous =
        xqa_e4m3_page800_fastpath_enabled() && k_cache.size(1) == 800 &&
        k_cache.size(2) == 1 && k_cache.size(3) == 256 &&
        k_cache.stride(0) == 2 * kPage800HeadDim256Elements &&
        k_cache.stride(1) == 256 && k_cache.stride(2) == 256 &&
        k_cache.stride(3) == 1 &&
        v_cache.stride(0) == 2 * kPage800HeadDim256Elements &&
        v_cache.stride(1) == 256 && v_cache.stride(2) == 256 &&
        v_cache.stride(3) == 1;
    trace_xqa_batch_context_route(q.size(0), batch_context_max_seq_len,
                                  partition_size, k_cache.size(1),
                                  e4m3_batch_route);
    trace_xqa_e4m3_page800_fastpath(q.size(0), partition_size,
                                    e4m3_dual_cta && e4m3_page800_contiguous);
    const int split_reduce_dim_tile = xqa_split_reduce_dim_tile();

#define LAUNCH_E4M3_DUAL_CTA(PARTITION, PAGE_SIZE, CONTIGUOUS)              \
  launch_flash_attention_decode_paged_xqa_tc_256_wide<                      \
      PARTITION, 6, true, kXQATCG6DualCtaThreads, 2, PAGE_SIZE, CONTIGUOUS, \
      false, kXQARouteAllSeqLens, false, true, true,                        \
      flash_v100::KV_CACHE_DTYPE_FP8_E4M3>(                                 \
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits, \
      exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,     \
      launch_num_partitions, e4m3_split_reduce, split_reduce_dim_tile, stream)

#define LAUNCH_E4M3_BATCH_XQA(PARTITION)                                       \
  do {                                                                         \
    if (e4m3_dual_cta) {                                                       \
      if (e4m3_page800_contiguous) {                                           \
        LAUNCH_E4M3_DUAL_CTA(PARTITION, 800, true);                            \
      } else {                                                                 \
        LAUNCH_E4M3_DUAL_CTA(PARTITION, 0, false);                             \
      }                                                                        \
    } else if (e4m3_batch_optimized) {                                         \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<                     \
          PARTITION, 6, true, kXQATC256WideThreads, 1, 0, false, false,        \
          kXQARouteAllSeqLens, false, true, true,                              \
          flash_v100::KV_CACHE_DTYPE_FP8_E4M3>(                                \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale, \
          v_scale, launch_num_partitions, false, split_reduce_dim_tile,        \
          stream);                                                             \
    } else {                                                                   \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<                     \
          PARTITION, 6, true, kXQATC256WideThreads, 1, 0, false, false,        \
          kXQARouteAllSeqLens, false, false, false,                            \
          flash_v100::KV_CACHE_DTYPE_FP8_E4M3>(                                \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale, \
          v_scale, launch_num_partitions, false, 8, stream);                   \
    }                                                                          \
  } while (0)

    const bool use_p64_p256_auto =
        q.size(0) == 1 && partition_size == 64 && k_cache.size(1) == 1568 &&
        k_cache.size(2) == 1 && !decode_partition_size_overridden() &&
        xqa_e4m3_g6_p64_p256_auto_enabled();
    constexpr int64_t kPage1568HeadDim256Elements = 1568 * 256;
    const bool fixed_interleaved_hkv1_layout =
        k_cache.size(3) == 256 &&
        k_cache.stride(0) == 2 * kPage1568HeadDim256Elements &&
        k_cache.stride(1) == 256 && k_cache.stride(2) == 256 &&
        k_cache.stride(3) == 1 &&
        v_cache.stride(0) == 2 * kPage1568HeadDim256Elements &&
        v_cache.stride(1) == 256 && v_cache.stride(2) == 256 &&
        v_cache.stride(3) == 1;
    if (use_p64_p256_auto) {
      const int p256_begin = xqa_e4m3_g6_p256_begin();
      const int p512_begin = xqa_e4m3_g6_p512_begin();
      const bool use_wave_partitions = xqa_e4m3_g6_wave_partitions_enabled() &&
                                       batch_context_max_seq_len >= p512_begin;
      if (use_wave_partitions) {
        const int p896_begin = xqa_e4m3_g6_p896_begin();
        const int p1664_begin = xqa_e4m3_g6_p1664_begin();
        TORCH_CHECK(p256_begin < p512_begin && p512_begin < p896_begin &&
                        p896_begin < p1664_begin,
                    "Invalid E4M3 wave-partition thresholds: ", p256_begin,
                    ", ", p512_begin, ", ", p896_begin, ", ", p1664_begin);

        const int p64_launch_num_partitions = std::max(
            1, std::min(launch_num_partitions, (p256_begin + 62) / 64));
        const int p256_launch_num_partitions = (launch_num_partitions + 3) / 4;
        const int p512_launch_num_partitions = (launch_num_partitions + 7) / 8;
        const int p896_launch_num_partitions =
            (launch_num_partitions + 13) / 14;
        const int p1664_launch_num_partitions =
            (launch_num_partitions + 25) / 26;
        const int p256_mid_launch_num_partitions = std::max(
            1, std::min(p256_launch_num_partitions, (p512_begin + 254) / 256));
        const int short_launch_num_partitions =
            std::max(p64_launch_num_partitions, p256_mid_launch_num_partitions);
        const int p512_route_launch_num_partitions = std::max(
            1, std::min(p512_launch_num_partitions, (p896_begin + 510) / 512));
        const int p896_route_launch_num_partitions = std::max(
            1, std::min(p896_launch_num_partitions, (p1664_begin + 894) / 896));
        const int wave_long_launch_num_partitions = std::max(
            {p512_route_launch_num_partitions, p896_route_launch_num_partitions,
             p1664_launch_num_partitions});
        const bool use_merged_wave_launch =
            xqa_e4m3_g6_merged_wave_launch_enabled();

        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            0, 6, true, kXQATC256WideThreads, 1, 0, false, false,
            kXQARouteShortSeqLens, false, false, false,
            flash_v100::KV_CACHE_DTYPE_FP8_E4M3>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, short_launch_num_partitions, false, 8, stream, p512_begin,
            p256_begin, 0, false);
        if (use_merged_wave_launch) {
          if (fixed_interleaved_hkv1_layout) {
            launch_flash_attention_decode_paged_xqa_e4m3_g6_page1568<
                -1, kXQARouteWaveLongSeqLens, true>(
                q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
                max_logits, exp_sums, active_num_partitions, softmax_scale,
                k_scale, v_scale, wave_long_launch_num_partitions, stream,
                p512_begin, p896_begin, p1664_begin, false);
          } else {
            launch_flash_attention_decode_paged_xqa_e4m3_g6_page1568<
                -1, kXQARouteWaveLongSeqLens>(
                q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
                max_logits, exp_sums, active_num_partitions, softmax_scale,
                k_scale, v_scale, wave_long_launch_num_partitions, stream,
                p512_begin, p896_begin, p1664_begin, false);
          }
        } else {
          launch_flash_attention_decode_paged_xqa_e4m3_g6_page1568<
              512, kXQARouteRangeSeqLens>(
              q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
              max_logits, exp_sums, active_num_partitions, softmax_scale,
              k_scale, v_scale, p512_route_launch_num_partitions, stream,
              p512_begin, p896_begin, 0, false);
          launch_flash_attention_decode_paged_xqa_e4m3_g6_page1568<
              896, kXQARouteRangeSeqLens>(
              q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
              max_logits, exp_sums, active_num_partitions, softmax_scale,
              k_scale, v_scale, p896_route_launch_num_partitions, stream,
              p896_begin, p1664_begin, 0, false);
          launch_flash_attention_decode_paged_xqa_e4m3_g6_page1568<
              1664, kXQARouteLongSeqLens>(
              q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
              max_logits, exp_sums, active_num_partitions, softmax_scale,
              k_scale, v_scale, p1664_launch_num_partitions, stream,
              p1664_begin, 0, 0, false);
        }

        const dim3 reduce_grid(q.size(0), q.size(1), 1);
        const dim3 reduce_block(kThreadsPerBlock);
#define LAUNCH_E4M3_WAVE_REDUCER(PARTITION_SIZE, SEQ_ROUTE, MAX_PARTITIONS,  \
                                 PARTITION_BEGIN, ROUTE_BEGIN, ROUTE_END,    \
                                 ROUTE_FINAL)                                \
  do {                                                                       \
    const size_t reduce_shared_mem =                                         \
        static_cast<size_t>(2 * (MAX_PARTITIONS)) * sizeof(float);           \
    flash_attention_decode_reduce_kernel<256, PARTITION_SIZE, SEQ_ROUTE>     \
        <<<reduce_grid, reduce_block, reduce_shared_mem, stream>>>(          \
            reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),   \
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),        \
            seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(), \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), q.size(0),  \
            MAX_PARTITIONS, q.size(1), tmp_out.stride(0), tmp_out.stride(1), \
            tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),   \
            out.stride(0), out.stride(1), PARTITION_BEGIN, ROUTE_BEGIN,      \
            ROUTE_END, ROUTE_FINAL);                                         \
  } while (0)
        LAUNCH_E4M3_WAVE_REDUCER(0, kXQARouteShortSeqLens,
                                 short_launch_num_partitions, p256_begin,
                                 p512_begin, 0, 0);
        if (use_merged_wave_launch) {
          LAUNCH_E4M3_WAVE_REDUCER(-1, kXQARouteWaveLongSeqLens,
                                   wave_long_launch_num_partitions, 0,
                                   p512_begin, p896_begin, p1664_begin);
        } else {
          LAUNCH_E4M3_WAVE_REDUCER(512, kXQARouteRangeSeqLens,
                                   p512_route_launch_num_partitions, 0,
                                   p512_begin, p896_begin, 0);
          LAUNCH_E4M3_WAVE_REDUCER(896, kXQARouteRangeSeqLens,
                                   p896_route_launch_num_partitions, 0,
                                   p896_begin, p1664_begin, 0);
          LAUNCH_E4M3_WAVE_REDUCER(1664, kXQARouteLongSeqLens,
                                   p1664_launch_num_partitions, 0, p1664_begin,
                                   0, 0);
        }
#undef LAUNCH_E4M3_WAVE_REDUCER
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        static bool traced_wave_partitions = false;
        if (xqa_e4m3_g6_p64_p256_auto_trace_enabled() &&
            !traced_wave_partitions) {
          TORCH_WARN(
              "Flash-V100 XQA E4M3 G6 wave-partition route active; "
              "thresholds p256/p512/p896/p1664=",
              p256_begin, "/", p512_begin, "/", p896_begin, "/", p1664_begin,
              ", launch partitions short/p512/p896/p1664=",
              short_launch_num_partitions, "/",
              p512_route_launch_num_partitions, "/",
              p896_route_launch_num_partitions, "/",
              p1664_launch_num_partitions,
              ", merged_long=", use_merged_wave_launch,
              ", converter=shared-lut");
          traced_wave_partitions = true;
        }
        return out;
      }
      const int dual_cta_begin =
          std::max(p256_begin, xqa_e4m3_g6_dual_cta_begin());
      const int p64_launch_num_partitions =
          std::max(1, std::min(launch_num_partitions, (p256_begin + 62) / 64));
      const int p256_launch_num_partitions = (launch_num_partitions + 3) / 4;
      const int p256_mid_launch_num_partitions = std::max(
          1,
          std::min(p256_launch_num_partitions, (dual_cta_begin + 254) / 256));
      const int short_launch_num_partitions =
          std::max(p64_launch_num_partitions, p256_mid_launch_num_partitions);
      const int reduce_num_partitions =
          std::max(short_launch_num_partitions, p256_launch_num_partitions);

      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          0, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteShortSeqLens, false, false, false,
          flash_v100::KV_CACHE_DTYPE_FP8_E4M3>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          short_launch_num_partitions, false, 8, stream, dual_cta_begin,
          p256_begin, 0, false);
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATCG6DualCtaThreads, 2, 1568, false, false,
          kXQARouteLongSeqLens, false, false, false,
          flash_v100::KV_CACHE_DTYPE_FP8_E4M3, true>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          p256_launch_num_partitions, false, 8, stream, dual_cta_begin, 0, 0,
          false);

      const dim3 reduce_grid(q.size(0), q.size(1), 1);
      const dim3 reduce_block(kThreadsPerBlock);
      const size_t reduce_shared_mem =
          static_cast<size_t>(2 * reduce_num_partitions) * sizeof(float);
      flash_attention_decode_reduce_kernel<256, 0>
          <<<reduce_grid, reduce_block, reduce_shared_mem, stream>>>(
              reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
              max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
              seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
              reinterpret_cast<__half*>(out.data_ptr<at::Half>()), q.size(0),
              reduce_num_partitions, q.size(1), tmp_out.stride(0),
              tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),
              max_logits.stride(1), out.stride(0), out.stride(1), p256_begin, 0,
              0, 0);

      static bool traced_p64_p256_auto = false;
      if (xqa_e4m3_g6_p64_p256_auto_trace_enabled() && !traced_p64_p256_auto) {
        TORCH_WARN(
            "Flash-V100 XQA E4M3 G6 device-side p64/p256 route active; "
            "p256 threshold=",
            p256_begin, ", p64 launch partitions=", p64_launch_num_partitions,
            ", dual-CTA threshold=", dual_cta_begin,
            ", short/mid launch partitions=", short_launch_num_partitions,
            ", long p256 launch partitions=", p256_launch_num_partitions,
            ", routed launch partitions=", reduce_num_partitions);
        traced_p64_p256_auto = true;
      }
    } else if (q.size(0) == 1 && partition_size == 64) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          64, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteAllSeqLens, false, false, false,
          flash_v100::KV_CACHE_DTYPE_FP8_E4M3>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, false, 8, stream);
    } else if (q.size(0) == 1 && partition_size == 128) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          128, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteAllSeqLens, false, false, false,
          flash_v100::KV_CACHE_DTYPE_FP8_E4M3>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, false, 8, stream);
    } else if (q.size(0) == 1 && partition_size == 512) {
      launch_flash_attention_decode_paged_xqa_e4m3_g6_page1568<512>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, stream);
    } else if (q.size(0) == 1 && partition_size == 896) {
      launch_flash_attention_decode_paged_xqa_e4m3_g6_page1568<896>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, stream);
    } else if (q.size(0) == 1 && partition_size == 1024) {
      launch_flash_attention_decode_paged_xqa_e4m3_g6_page1568<1024>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, stream);
    } else if (q.size(0) == 1 && partition_size == 1664) {
      launch_flash_attention_decode_paged_xqa_e4m3_g6_page1568<1664>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, stream);
    } else if (partition_size == 64) {
      LAUNCH_E4M3_BATCH_XQA(64);
    } else if (partition_size == 128) {
      LAUNCH_E4M3_BATCH_XQA(128);
    } else {
      LAUNCH_E4M3_BATCH_XQA(256);
    }
#undef LAUNCH_E4M3_BATCH_XQA
#undef LAUNCH_E4M3_DUAL_CTA
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
  // Default only the shape with exact direct and model-level evidence.
  const bool padded_smem_enabled = xqa_padded_smem_enabled();
  const bool use_padded_smem =
      padded_smem_enabled && q_per_kv == 6 && partition_size == 256;
  const bool use_g6_dual_cta_dense = !use_padded_smem &&
                                     k_cache.size(1) == 784 &&
                                     xqa_g6_dual_cta_dense_enabled();
  const bool use_g6_p1024_auto =
      q.size(0) == 1 && q_per_kv == 6 && partition_size == 1024 &&
      k_cache.size(1) == 784 && k_cache.size(2) == 1 &&
      xqa_g6_p1024_auto_enabled();
  const bool use_g6_p1024_sawtooth =
      q.size(0) == 1 && q_per_kv == 6 && partition_size == 256 &&
      k_cache.size(1) == 784 && k_cache.size(2) == 1 &&
      !decode_partition_size_overridden() && xqa_block784_index_enabled() &&
      xqa_g6_p1024_sawtooth_enabled();
  const bool use_g6_qk_pipeline = use_g6_p1024_sawtooth &&
                                  k_cache.scalar_type() == at::kHalf &&
                                  xqa_g6_qk_pipeline_enabled();
  const int g6_qk_pipeline_warps =
      use_g6_qk_pipeline ? xqa_g6_qk_pipeline_warps() : 0;
  const int g6_p1024_route_seq_len =
      use_g6_p1024_auto ? (at::cuda::getDeviceProperties(q.get_device())
                               ->multiProcessorCount +
                           1) *
                              1024
                        : 0;
  const int g6_p1024_sawtooth_p1024_mid_seq_len =
      use_g6_p1024_sawtooth ? xqa_g6_p1024_sawtooth_p1024_mid_seq_len() : 0;
  const int g6_p1024_sawtooth_p256_long_seq_len =
      use_g6_p1024_sawtooth ? xqa_g6_p1024_sawtooth_p256_long_seq_len() : 0;
  const int g6_p1024_sawtooth_p1024_final_seq_len =
      use_g6_p1024_sawtooth ? xqa_g6_p1024_sawtooth_p1024_final_seq_len() : 0;
  TORCH_CHECK(
      !use_g6_p1024_sawtooth || (g6_p1024_sawtooth_p1024_mid_seq_len <
                                     g6_p1024_sawtooth_p256_long_seq_len &&
                                 g6_p1024_sawtooth_p256_long_seq_len <
                                     g6_p1024_sawtooth_p1024_final_seq_len),
      "Invalid p1024/p256 sawtooth thresholds: ",
      g6_p1024_sawtooth_p1024_mid_seq_len, ", ",
      g6_p1024_sawtooth_p256_long_seq_len, ", ",
      g6_p1024_sawtooth_p1024_final_seq_len);
  const bool use_mtp5_dual_cta =
      q.size(0) == 5 && q_per_kv == 6 && k_cache.size(1) == 1616 &&
      k_cache.size(2) > 0 &&
      kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 &&
      xqa_mtp5_dual_cta_enabled();
  const bool use_e5m2_g6_dual_cta =
      q.size(0) == 1 && q_per_kv == 6 && partition_size == 256 &&
      k_cache.size(1) >= 256 && k_cache.size(1) % 16 == 0 &&
      k_cache.size(2) == 1 && k_cache.scalar_type() == at::kByte &&
      kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 &&
      !decode_partition_size_overridden() && xqa_e5m2_g6_dual_cta_enabled() &&
      xqa_e5m2_g6_split_reduce_enabled();
  const bool batch_context_page_supported =
      k_cache.size(1) == 16 ||
      (k_cache.size(1) >= 256 && k_cache.size(1) % 16 == 0);
  const bool use_e5m2_g6_batch_context_route =
      batch_context_max_seq_len > 0 && q.size(0) >= 4 && q_per_kv == 6 &&
      batch_context_page_supported && k_cache.size(2) == 1 &&
      k_cache.scalar_type() == at::kByte &&
      kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 &&
      !decode_partition_size_overridden();
  const XQABatchContextRoute batch_context_route =
      use_e5m2_g6_batch_context_route
          ? select_xqa_batch_context_route(q.size(0), batch_context_max_seq_len,
                                           partition_size)
          : XQABatchContextRoute::kDisabled;
  const int e5m2_g6_dual_cta_seq_len =
      use_e5m2_g6_dual_cta
          ? at::cuda::getDeviceProperties(q.get_device())->multiProcessorCount *
                    1024 +
                1
          : 0;
  const int e5m2_p1024_begin =
      use_e5m2_g6_dual_cta ? xqa_e5m2_p1024_begin() : 0;
  const int e5m2_scalar_xqa_seq_len =
      use_e5m2_g6_dual_cta
          ? std::min(xqa_e5m2_scalar_xqa_seq_len(), e5m2_p1024_begin)
          : 0;
  TORCH_CHECK(
      !use_e5m2_g6_dual_cta || e5m2_p1024_begin < e5m2_g6_dual_cta_seq_len,
      "Invalid E5M2 G6 partition thresholds: ", e5m2_p1024_begin, ", ",
      e5m2_g6_dual_cta_seq_len);
  const int e5m2_p1024_launch_num_partitions = (launch_num_partitions + 3) / 4;
  const int e5m2_p1024_one_cta_launch_num_partitions =
      use_e5m2_g6_dual_cta
          ? std::min(e5m2_p1024_launch_num_partitions,
                     at::cuda::getDeviceProperties(q.get_device())
                         ->multiProcessorCount)
          : 0;
  const bool use_e5m2_partition_page_ids =
      use_e5m2_g6_dual_cta && xqa_e5m2_partition_page_ids_enabled();
  const bool use_e5m2_pair_load =
      use_e5m2_partition_page_ids && xqa_e5m2_pair_load_enabled();
  // The B1 E5M2 route already amortizes page-table lookups across a p256
  // partition and converts two half8 vectors from one aligned 128-bit load.
  // Reuse that exact load path for the batch/context route without changing
  // its partition, softmax, PV, or reduction order. Restrict the page size so
  // one partition always fits the existing shared page-id capacity.
  const bool use_e5m2_batch_wide_load =
      use_e5m2_g6_batch_context_route && partition_size == 256 &&
      k_cache.size(1) >= 256 && k_cache.size(1) % 16 == 0 &&
      (batch_context_route == XQABatchContextRoute::kDualCta ||
       batch_context_route == XQABatchContextRoute::kDualCtaSplit) &&
      xqa_e5m2_batch_wide_load_enabled();
  const bool use_qsa_page4 = q.size(0) >= 4096 && q_per_kv == 6 &&
                             (partition_size == 256 || partition_size == 512 ||
                              partition_size == 1024) &&
                             k_cache.size(1) == 4 && k_cache.size(2) == 1 &&
                             k_cache.scalar_type() == at::kHalf &&
                             block_table.size(1) == 513;
  const bool use_g6_dual_cta =
      use_qsa_page4 || use_g6_p1024_auto || use_g6_p1024_sawtooth ||
      use_mtp5_dual_cta || use_e5m2_g6_dual_cta ||
      batch_context_route == XQABatchContextRoute::kDualCta ||
      batch_context_route == XQABatchContextRoute::kDualCtaSplit ||
      (xqa_g6_dual_cta_enabled() && (use_padded_smem || use_g6_dual_cta_dense));
  const bool use_split_reduce =
      use_g6_p1024_auto || use_g6_p1024_sawtooth || use_e5m2_g6_dual_cta ||
      batch_context_route == XQABatchContextRoute::kDualCtaSplit ||
      (use_g6_dual_cta && xqa_split_reduce_enabled());
  const bool supports_block16_index = use_g6_dual_cta && k_cache.size(1) == 16;
  const bool use_block4_index = use_g6_dual_cta && k_cache.size(1) == 4;
  const bool supports_block16_contiguous_layout =
      supports_block16_index && k_cache.size(2) == 1 &&
      k_cache.stride(0) == 4096 && k_cache.stride(1) == 256 &&
      k_cache.stride(2) == 256 && k_cache.stride(3) == 1 &&
      v_cache.stride(0) == 4096 && v_cache.stride(1) == 256 &&
      v_cache.stride(2) == 256 && v_cache.stride(3) == 1;
  const bool use_block784_index =
      use_g6_dual_cta && k_cache.size(1) == 784 && xqa_block784_index_enabled();
  const bool use_aligned_padded_smem = use_padded_smem && use_block784_index &&
                                       xqa_aligned_padded_smem_enabled();
  const int requested_block16_layout_mode = xqa_block16_layout_mode();
  const int block16_layout_mode =
      requested_block16_layout_mode == 1 && supports_block16_index ? 1
      : requested_block16_layout_mode == 2 && supports_block16_contiguous_layout
          ? 2
          : 0;
  if (xqa_block16_layout_required() && requested_block16_layout_mode != 0) {
    TORCH_CHECK(
        block16_layout_mode == requested_block16_layout_mode,
        "Requested block16 XQA mode ", requested_block16_layout_mode,
        " but the live KV cache does not satisfy its layout gate: block_size=",
        k_cache.size(1), ", num_kv_heads=", k_cache.size(2), ", k_strides=[",
        k_cache.stride(0), ",", k_cache.stride(1), ",", k_cache.stride(2), ",",
        k_cache.stride(3), "]");
  }
  trace_xqa_batch_context_route(q.size(0), batch_context_max_seq_len,
                                partition_size, k_cache.size(1),
                                batch_context_route);
  static bool traced_e5m2_batch_wide_load = false;
  if (xqa_batch_context_routing_trace_enabled() && use_e5m2_batch_wide_load &&
      !traced_e5m2_batch_wide_load) {
    TORCH_WARN(
        "Flash-V100 XQA batched E5M2 partition page reuse and paired "
        "128-bit loads active");
    traced_e5m2_batch_wide_load = true;
  }
  static bool traced_block16_layout = false;
  if (xqa_block16_layout_trace_enabled() && block16_layout_mode != 0 &&
      !traced_block16_layout) {
    TORCH_WARN("Flash-V100 XQA block16 mode ", block16_layout_mode,
               " active for KV shape [blocks,", k_cache.size(1), ",",
               k_cache.size(2), ",", k_cache.size(3), "]");
    traced_block16_layout = true;
  }
  static bool traced_block784_index = false;
  if (xqa_block784_index_trace_enabled() && use_block784_index &&
      !traced_block784_index) {
    TORCH_WARN(
        "Flash-V100 XQA block784 index specialization active for KV "
        "shape [blocks,",
        k_cache.size(1), ",", k_cache.size(2), ",", k_cache.size(3), "]");
    traced_block784_index = true;
  }
  static bool traced_g6_p1024_auto = false;
  if (xqa_g6_p1024_auto_trace_enabled() && use_g6_p1024_auto &&
      !traced_g6_p1024_auto) {
    TORCH_WARN(
        "Flash-V100 XQA p1024 dynamic one/two-CTA route active; "
        "long-sequence threshold=",
        g6_p1024_route_seq_len);
    traced_g6_p1024_auto = true;
  }
  static bool traced_g6_p1024_sawtooth = false;
  if (xqa_g6_p1024_sawtooth_trace_enabled() && use_g6_p1024_sawtooth &&
      !traced_g6_p1024_sawtooth) {
    TORCH_WARN("Flash-V100 XQA p1024/p256 sawtooth route active; thresholds=",
               g6_p1024_sawtooth_p1024_mid_seq_len, ", ",
               g6_p1024_sawtooth_p256_long_seq_len, ", ",
               g6_p1024_sawtooth_p1024_final_seq_len);
    traced_g6_p1024_sawtooth = true;
  }
  static bool traced_g6_qk_pipeline = false;
  if (xqa_g6_qk_pipeline_trace_enabled() && use_g6_qk_pipeline &&
      !traced_g6_qk_pipeline) {
    TORCH_WARN(
        "Flash-V100 XQA G6 fp16-KV QK K64 producer/consumer "
        "pipeline active for the mid p1024 range with baseline final-range "
        "fallback; warps=",
        g6_qk_pipeline_warps);
    traced_g6_qk_pipeline = true;
  }
  static bool traced_e5m2_g6_dual_cta = false;
  if (xqa_e5m2_g6_dual_cta_trace_enabled() && use_e5m2_g6_dual_cta &&
      !traced_e5m2_g6_dual_cta) {
    TORCH_WARN(
        "Flash-V100 XQA E5M2 G6 device-side p256/p1024 route active; "
        "thresholds=",
        e5m2_p1024_begin, ", ", e5m2_g6_dual_cta_seq_len, ", KV shape [blocks,",
        k_cache.size(1), ",", k_cache.size(2), ",", k_cache.size(3),
        "], split_reduce=", use_split_reduce,
        ", partition_page_ids=", use_e5m2_partition_page_ids,
        ", pair_load=", use_e5m2_pair_load,
        ", scalar_xqa_seq_len=", e5m2_scalar_xqa_seq_len);
    traced_e5m2_g6_dual_cta = true;
  }
  static bool traced_aligned_padded_smem = false;
  if (xqa_aligned_padded_smem_trace_enabled() && use_aligned_padded_smem &&
      !traced_aligned_padded_smem) {
    TORCH_WARN("Flash-V100 XQA aligned padded shared layout active");
    traced_aligned_padded_smem = true;
  }
  const int split_reduce_dim_tile = xqa_split_reduce_dim_tile();

#define LAUNCH_XQA_WIDE(GROUP_SIZE, PARTITION)                                 \
  do {                                                                         \
    if (use_padded_smem) {                                                     \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<PARTITION,           \
                                                          GROUP_SIZE, true>(   \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale, \
          v_scale, launch_num_partitions, false, 8, stream);                   \
    } else {                                                                   \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<PARTITION,           \
                                                          GROUP_SIZE, false>(  \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale, \
          v_scale, launch_num_partitions, false, 8, stream);                   \
    }                                                                          \
  } while (0)

#define DISPATCH_PARTITION(GROUP_SIZE)                                   \
  do {                                                                   \
    switch (partition_size) {                                            \
      case 256:                                                          \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 256);                                \
        break;                                                           \
      case 512:                                                          \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 512);                                \
        break;                                                           \
      case 1024:                                                         \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 1024);                               \
        break;                                                           \
      default:                                                           \
        TORCH_CHECK(false,                                               \
                    "Unsupported XQA partition_size: ", partition_size); \
    }                                                                    \
  } while (0)

  if (use_e5m2_g6_dual_cta) {
    launch_flash_attention_decode_paged<
        256, 256, flash_v100::KV_CACHE_DTYPE_FP8_E5M2, kXQARouteShortSeqLens>(
        q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
        exp_sums, active_num_partitions, softmax_scale, launch_num_partitions,
        k_scale, v_scale, window_size_left, window_size_right, stream,
        e5m2_scalar_xqa_seq_len, 0, 0, false);
    if (use_e5m2_pair_load) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteP1024SawtoothMid, false, true, true>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, false, split_reduce_dim_tile, stream,
          e5m2_scalar_xqa_seq_len, e5m2_p1024_begin, 0, false);
    } else if (use_e5m2_partition_page_ids) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteP1024SawtoothMid, false, true>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, false, split_reduce_dim_tile, stream,
          e5m2_scalar_xqa_seq_len, e5m2_p1024_begin, 0, false);
    } else {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteP1024SawtoothMid>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, false, split_reduce_dim_tile, stream,
          e5m2_scalar_xqa_seq_len, e5m2_p1024_begin, 0, false);
    }
    if (use_e5m2_partition_page_ids) {
      if (use_e5m2_pair_load) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATC256WideThreads, 1, 0, false, false,
            kXQARouteP1024SawtoothMid, false, true, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, e5m2_p1024_one_cta_launch_num_partitions, false,
            split_reduce_dim_tile, stream, e5m2_p1024_begin,
            e5m2_g6_dual_cta_seq_len, 0, false);
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATCG6DualCtaThreads, 2, 0, false, false,
            kXQARouteLongSeqLens, false, true, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, e5m2_p1024_launch_num_partitions, false,
            split_reduce_dim_tile, stream, e5m2_g6_dual_cta_seq_len, 0, 0,
            false);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATC256WideThreads, 1, 0, false, false,
            kXQARouteP1024SawtoothMid, false, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, e5m2_p1024_one_cta_launch_num_partitions, false,
            split_reduce_dim_tile, stream, e5m2_p1024_begin,
            e5m2_g6_dual_cta_seq_len, 0, false);
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATCG6DualCtaThreads, 2, 0, false, false,
            kXQARouteLongSeqLens, false, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, e5m2_p1024_launch_num_partitions, false,
            split_reduce_dim_tile, stream, e5m2_g6_dual_cta_seq_len, 0, 0,
            false);
      }
    } else {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteP1024SawtoothMid>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          e5m2_p1024_one_cta_launch_num_partitions, false,
          split_reduce_dim_tile, stream, e5m2_p1024_begin,
          e5m2_g6_dual_cta_seq_len, 0, false);
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATCG6DualCtaThreads, 2, 0, false, false,
          kXQARouteLongSeqLens>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          e5m2_p1024_launch_num_partitions, false, split_reduce_dim_tile,
          stream, e5m2_g6_dual_cta_seq_len, 0, 0, false);
    }
    launch_flash_attention_decode_xqa_split_reduce<0>(
        out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
        split_reduce_dim_tile, stream, e5m2_p1024_begin,
        e5m2_g6_dual_cta_seq_len, e5m2_g6_dual_cta_seq_len);
  } else if (use_g6_p1024_sawtooth) {
    const int p1024_launch_num_partitions = (launch_num_partitions + 3) / 4;
    if (use_g6_qk_pipeline) {
      if (g6_qk_pipeline_warps == 8) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATCG6Pipeline8WarpThreads, 2, 784, false, false,
            kXQARouteP1024SawtoothMid, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, p1024_launch_num_partitions, false, split_reduce_dim_tile,
            stream, g6_p1024_sawtooth_p1024_mid_seq_len,
            g6_p1024_sawtooth_p256_long_seq_len,
            g6_p1024_sawtooth_p1024_final_seq_len, false);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATCG6DualCtaThreads, 2, 784, false, false,
            kXQARouteP1024SawtoothMid, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, p1024_launch_num_partitions, false, split_reduce_dim_tile,
            stream, g6_p1024_sawtooth_p1024_mid_seq_len,
            g6_p1024_sawtooth_p256_long_seq_len,
            g6_p1024_sawtooth_p1024_final_seq_len, false);
      }
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATCG6DualCtaThreads, 2, 784, false, false,
          kXQARouteP1024SawtoothFinal>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          p1024_launch_num_partitions, false, split_reduce_dim_tile, stream,
          g6_p1024_sawtooth_p1024_mid_seq_len,
          g6_p1024_sawtooth_p256_long_seq_len,
          g6_p1024_sawtooth_p1024_final_seq_len, false);
    } else {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATCG6DualCtaThreads, 2, 784, false, false,
          kXQARouteP1024Sawtooth>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          p1024_launch_num_partitions, false, split_reduce_dim_tile, stream,
          g6_p1024_sawtooth_p1024_mid_seq_len,
          g6_p1024_sawtooth_p256_long_seq_len,
          g6_p1024_sawtooth_p1024_final_seq_len, false);
    }
    launch_flash_attention_decode_paged_xqa_tc_256_wide<
        256, 6, true, kXQATCG6DualCtaThreads, 2, 784, false, false,
        kXQARouteP256Sawtooth>(
        q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
        exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
        launch_num_partitions, false, split_reduce_dim_tile, stream,
        g6_p1024_sawtooth_p1024_mid_seq_len,
        g6_p1024_sawtooth_p256_long_seq_len,
        g6_p1024_sawtooth_p1024_final_seq_len, false);
    // Both partition routes write the selected result at the front of the
    // shared p256 workspace. Select the matching reduction width on device so
    // one stats/output pair serves every replay length.
    launch_flash_attention_decode_xqa_split_reduce<0>(
        out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
        split_reduce_dim_tile, stream, g6_p1024_sawtooth_p1024_mid_seq_len,
        g6_p1024_sawtooth_p256_long_seq_len,
        g6_p1024_sawtooth_p1024_final_seq_len);
  } else if (use_g6_p1024_auto) {
    launch_flash_attention_decode_paged_xqa_tc_256_wide<
        1024, 6, false, kXQATC256WideThreads, 1, 784, false, false,
        kXQARouteShortSeqLens>(
        q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
        exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
        launch_num_partitions, false, split_reduce_dim_tile, stream,
        g6_p1024_route_seq_len, 0, 0, false);
    launch_flash_attention_decode_paged_xqa_tc_256_wide<
        1024, 6, false, kXQATCG6DualCtaThreads, 2, 784, false, false,
        kXQARouteLongSeqLens>(
        q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
        exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
        launch_num_partitions, true, split_reduce_dim_tile, stream,
        g6_p1024_route_seq_len, 0, 0, true);
  } else if (use_g6_dual_cta) {
    if (use_block784_index) {
      if (partition_size == 1024) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATCG6DualCtaThreads, 2, 784, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else if (partition_size == 512) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            512, 6, false, kXQATCG6DualCtaThreads, 2, 784, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else if (use_aligned_padded_smem) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 784, false, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else if (use_padded_smem) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 784, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, false, kXQATCG6DualCtaThreads, 2, 784, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      }
    } else if (use_block4_index) {
      if (partition_size == 256) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 4, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else if (partition_size == 512) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            512, 6, true, kXQATCG6DualCtaThreads, 2, 4, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, true, kXQATCG6DualCtaThreads, 2, 4, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      }
    } else if (block16_layout_mode == 2) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATCG6DualCtaThreads, 2, 16, true>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    } else if (block16_layout_mode == 1) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATCG6DualCtaThreads, 2, 16, false>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    } else if (partition_size == 256) {
      if (use_e5m2_batch_wide_load) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 0, false, false,
            kXQARouteAllSeqLens, false, true, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 0, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      }
    } else if (partition_size == 512) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          512, 6, false, kXQATCG6DualCtaThreads, 2, 0, false>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    } else {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATCG6DualCtaThreads, 2, 0, false>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    }
  } else if (q_per_kv == 4) {
    DISPATCH_PARTITION(4);
  } else if (q_per_kv == 6) {
    DISPATCH_PARTITION(6);
  } else {
    DISPATCH_PARTITION(8);
  }

#undef DISPATCH_PARTITION
#undef LAUNCH_XQA_WIDE

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor flash_attention_decode_paged_xqa_staged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, at::Tensor& online_rescales,
    const at::Tensor& active_num_partitions, const float softmax_scale,
    const int partition_size, const int launch_num_partitions,
    const std::string& kv_cache_dtype, const float k_scale, const float v_scale,
    const int window_size_left, const int window_size_right) {
  (void)k_scale;
  (void)v_scale;
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
              "k_cache and v_cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda() &&
                  online_rescales.is_cuda(),
              "staged XQA workspaces must be on CUDA");
  TORCH_CHECK(active_num_partitions.is_cuda(),
              "active_num_partitions must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  TORCH_CHECK(
      k_cache.dtype() == torch::kFloat16 && v_cache.dtype() == torch::kFloat16,
      "staged XQA supports fp16 KV cache only");
  TORCH_CHECK(kv_cache_dtype == "auto" || kv_cache_dtype == "bfloat16",
              "staged XQA supports fp16 KV cache only");
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(active_num_partitions.dtype() == torch::kInt32,
              "active_num_partitions must be int32");
  TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
              "staged XQA does not support sliding-window attention");
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4,
              "staged XQA expects q [B,H,D] and paged KV [blocks,T,H,D]");
  TORCH_CHECK(block_table.dim() == 2 && seq_lens.dim() == 1,
              "staged XQA block_table/seq_lens rank mismatch");
  TORCH_CHECK(
      active_num_partitions.dim() == 1 && active_num_partitions.numel() == 1,
      "active_num_partitions must have shape [1]");
  TORCH_CHECK(
      q.stride(-1) == 1 && k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
      "staged XQA requires contiguous head dimensions");
  TORCH_CHECK(q.size(0) <= block_table.size(0) && q.size(0) <= seq_lens.size(0),
              "staged XQA metadata batch capacity is too small");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "K/V cache shape mismatch");
  TORCH_CHECK(q.size(2) == 256 && k_cache.size(3) == 256,
              "staged XQA supports D=256 only");
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  TORCH_CHECK(num_heads_kv > 0 && num_heads_q == 6 * num_heads_kv,
              "staged XQA supports q_per_kv=6 only");
  TORCH_CHECK(partition_size == 256, "staged XQA supports p256 only");
  TORCH_CHECK(launch_num_partitions > 0,
              "launch_num_partitions must be positive");
  TORCH_CHECK(tmp_out.dtype() == torch::kFloat16 &&
                  tmp_out.size(0) >= q.size(0) &&
                  tmp_out.size(1) >= q.size(1) &&
                  tmp_out.size(2) >= launch_num_partitions &&
                  tmp_out.size(3) == q.size(2),
              "tmp_out shape does not cover staged XQA launch");
  TORCH_CHECK(max_logits.dtype() == torch::kFloat32 &&
                  max_logits.size(0) >= q.size(0) &&
                  max_logits.size(1) >= q.size(1) &&
                  max_logits.size(2) >= launch_num_partitions,
              "max_logits shape does not cover staged XQA launch");
  TORCH_CHECK(exp_sums.sizes() == max_logits.sizes(),
              "exp_sums shape mismatch");
  TORCH_CHECK(online_rescales.dtype() == torch::kFloat32 &&
                  online_rescales.size(0) >= q.size(0) &&
                  online_rescales.size(1) >= q.size(1) &&
                  online_rescales.size(2) >= launch_num_partitions,
              "online_rescales shape does not cover staged XQA launch");

  c10::cuda::CUDAGuard device_guard(q.device());
  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kFloat16 &&
                  out.sizes() == q.sizes() && out.stride(-1) == 1,
              "staged XQA output must be contiguous fp16 q-shaped CUDA tensor");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  launch_flash_attention_decode_paged_xqa_tc_256_staged(
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
      exp_sums, online_rescales, active_num_partitions, softmax_scale,
      launch_num_partitions, xqa_split_reduce_enabled(),
      xqa_split_reduce_dim_tile(), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor flash_attention_decode_qk_scores(
    const at::Tensor& q, const at::Tensor& k_cache,
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    const float softmax_scale, const int partition_size,
    const std::string& kv_cache_dtype, const float k_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  TORCH_CHECK(kv_dtype_code >= 0,
              "Unsupported kv_cache_dtype: ", kv_cache_dtype);
  if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
  } else {
    TORCH_CHECK(k_cache.dtype() == torch::kUInt8,
                "fp8 k_cache must be stored as uint8");
    TORCH_CHECK(k_scale > 0.f, "fp8 k scale must be positive");
  }
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4,
              "k_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1, "k_cache last dim must be contiguous");

  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int head_dim = q.size(2);
  const int num_heads_kv = k_cache.size(2);
  const int block_size = k_cache.size(1);
  const int max_num_blocks = block_table.size(1);
  const int max_num_partitions =
      (max_num_blocks * block_size + partition_size - 1) / partition_size;

  TORCH_CHECK(q.size(0) <= block_table.size(0),
              "block_table batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= seq_lens.size(0),
              "seq_lens batch size must cover q batch size");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  TORCH_CHECK(k_cache.size(3) == head_dim, "k_cache head_dim mismatch");
  TORCH_CHECK(
      partition_size == 256 || partition_size == 512 || partition_size == 1024,
      "Unsupported decode partition_size: ", partition_size);

  c10::cuda::CUDAGuard device_guard(q.device());
  auto scores =
      torch::full({batch_size, num_heads_q, max_num_partitions, partition_size},
                  -1.0e30f, q.options().dtype(torch::kFloat32));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

#define LAUNCH_TYPED(HDIM, PARTITION, KV_DTYPE_CODE)                       \
  launch_flash_attention_decode_qk_scores<HDIM, PARTITION, KV_DTYPE_CODE>( \
      q, k_cache, block_table, seq_lens, scores, softmax_scale, k_scale,   \
      stream)

#define LAUNCH_BY_KV_DTYPE(HDIM, PARTITION)                                 \
  do {                                                                      \
    switch (kv_dtype_code) {                                                \
      case flash_v100::KV_CACHE_DTYPE_FP16:                                 \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP16);     \
        break;                                                              \
      case flash_v100::KV_CACHE_DTYPE_FP8_E4M3:                             \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E4M3); \
        break;                                                              \
      case flash_v100::KV_CACHE_DTYPE_FP8_E5M2:                             \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E5M2); \
        break;                                                              \
      default:                                                              \
        TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype); \
    }                                                                       \
  } while (0)

#define LAUNCH_BY_PARTITION(HDIM)                                           \
  do {                                                                      \
    switch (partition_size) {                                               \
      case 256:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 256);                                      \
        break;                                                              \
      case 512:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 512);                                      \
        break;                                                              \
      case 1024:                                                            \
        LAUNCH_BY_KV_DTYPE(HDIM, 1024);                                     \
        break;                                                              \
      default:                                                              \
        TORCH_CHECK(false,                                                  \
                    "Unsupported decode partition_size: ", partition_size); \
    }                                                                       \
  } while (0)

  switch (head_dim) {
    case 64:
      LAUNCH_BY_PARTITION(64);
      break;
    case 80:
      LAUNCH_BY_PARTITION(80);
      break;
    case 96:
      LAUNCH_BY_PARTITION(96);
      break;
    case 112:
      LAUNCH_BY_PARTITION(112);
      break;
    case 128:
      LAUNCH_BY_PARTITION(128);
      break;
    case 256:
      LAUNCH_BY_PARTITION(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head_dim for paged decode: ", head_dim);
  }

#undef LAUNCH_BY_PARTITION
#undef LAUNCH_BY_KV_DTYPE
#undef LAUNCH_TYPED

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}
