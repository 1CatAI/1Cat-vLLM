#pragma once

#include <c10/cuda/CUDAException.h>

#include <cuda_runtime_api.h>
#include <torch/library.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "cutlass/gemm/gemm.h"

namespace marlin::sm70 {

constexpr int kCtaK = 32;
constexpr int kQuantTileK = 16;
constexpr int kQuantTileN = 64;

template <typename SharedStorage, typename Kernel>
inline size_t configure_sm70_dynamic_smem(Kernel kernel) {
  size_t smem_bytes = sizeof(SharedStorage);
  if (smem_bytes >= (48u << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }
  return smem_bytes;
}

constexpr int sm70_const_min(int lhs, int rhs) {
  return lhs < rhs ? lhs : rhs;
}

constexpr bool sm70_marlin_basic_geometry_values_supported(
    int cta_m, int cta_n, int cta_k, int warps, int warp_m, int warp_n,
    int warp_k) {
  return (cta_m == 32 || cta_m == 64 || cta_m == 128 || cta_m == 256) &&
         (cta_n == 64 || cta_n == 128 || cta_n == 256) &&
         (cta_k == 16 || cta_k == 32 || cta_k == 64 || cta_k == 128) &&
         (warps == 4 || warps == 8) &&
         (warp_m == 32 || warp_m == 64) &&
         (warp_n == 32 || warp_n == 64) &&
         (warp_k == 16 || warp_k == 32);
}

constexpr bool sm70_marlin_warp_decomposition_supported(
    int cta_m, int cta_n, int cta_k, int warps, int warp_m, int warp_n,
    int warp_k) {
  if (!sm70_marlin_basic_geometry_values_supported(
          cta_m, cta_n, cta_k, warps, warp_m, warp_n, warp_k)) {
    return false;
  }
  if (cta_m % warp_m != 0 || cta_n % warp_n != 0 ||
      cta_k % warp_k != 0) {
    return false;
  }
  return (cta_m / warp_m) * (cta_n / warp_n) * (cta_k / warp_k) == warps;
}

constexpr bool sm70_marlin_pitchlinear_threadmap_supported(
    int shape_contiguous, int shape_strided, int threads,
    int warp_thread_contiguous, int warp_thread_strided,
    int elements_per_access) {
  if (shape_contiguous % elements_per_access != 0) {
    return false;
  }
  int const shape_access_contiguous =
      shape_contiguous / elements_per_access;
  int const shape_access_strided = shape_strided;
  if (shape_access_contiguous % warp_thread_contiguous != 0 ||
      shape_access_strided % warp_thread_strided != 0) {
    return false;
  }
  int const warp_access_contiguous =
      shape_access_contiguous / warp_thread_contiguous;
  int const warp_access_strided =
      shape_access_strided / warp_thread_strided;
  int const warp_count = threads / 32;
  if (warp_access_strided <= 0 || warp_count <= 0) {
    return false;
  }
  int const warps_strided =
      warp_access_strided >= warp_count ? warp_count : warp_access_strided;
  if (warps_strided <= 0) {
    return false;
  }
  int const warps_contiguous =
      warp_count > warp_access_strided ? warp_count / warps_strided : 1;
  if (warps_contiguous <= 0) {
    return false;
  }
  int const iterations_contiguous =
      warp_access_contiguous / warps_contiguous;
  int const iterations_strided = warp_access_strided / warps_strided;
  return iterations_contiguous > 0 && iterations_strided > 0;
}

constexpr bool sm70_marlin_b_threadmap_matches_quant_iterator(
    int cta_n, int cta_k, int threads) {
  constexpr int kElementsPerAccess = 8;
  constexpr int kWarpThreadContiguous = 8;
  constexpr int kWarpThreadStrided = 4;
  if (!sm70_marlin_pitchlinear_threadmap_supported(
          cta_n, cta_k, threads, kWarpThreadContiguous,
          kWarpThreadStrided, kElementsPerAccess)) {
    return false;
  }
  int const shape_access_contiguous = cta_n / kElementsPerAccess;
  int const shape_access_strided = cta_k;
  int const warp_access_contiguous =
      shape_access_contiguous / kWarpThreadContiguous;
  int const warp_access_strided =
      shape_access_strided / kWarpThreadStrided;
  int const warp_count = threads / 32;
  int const warps_strided =
      warp_access_strided >= warp_count ? warp_count : warp_access_strided;
  int const warps_contiguous =
      warp_count > warp_access_strided ? warp_count / warps_strided : 1;
  int const iterations_contiguous =
      warp_access_contiguous / warps_contiguous;
  int const delta_contiguous =
      kWarpThreadContiguous * kElementsPerAccess;
  return iterations_contiguous == cta_n / kQuantTileN &&
         delta_contiguous == kQuantTileN;
}

constexpr bool sm70_marlin_volta_epilogue_supported(
    int cta_m, int cta_n, int warps, int warp_m) {
  constexpr int kShapeRow = 4;
  constexpr int kShapeGroup = 4;
  constexpr int kElementsPerAccess = 8;
  constexpr int kElementBits = 16;
  constexpr int kWarpSize = 32;
  int const shape_cluster = cta_m / warp_m;
  if (shape_cluster <= 0) {
    return false;
  }
  int const warps_remaining_for_groups =
      shape_cluster > warps ? 1 : warps / shape_cluster;
  if (warps_remaining_for_groups <= 0) {
    return false;
  }
  int const warps_remaining_for_rows =
      kShapeGroup > warps_remaining_for_groups
          ? 1
          : warps_remaining_for_groups / kShapeGroup;
  if (warps_remaining_for_rows <= 0) {
    return false;
  }

  if (kShapeRow <= warps_remaining_for_rows) {
    return cta_n / kElementsPerAccess / kWarpSize > 0;
  }

  int const row_shape = kShapeRow / warps_remaining_for_rows;
  if (row_shape <= 0) {
    return false;
  }
  int const shape_width = cta_n / kElementsPerAccess;
  int const target_memory_access_width =
      256 / (kElementsPerAccess * kElementBits / 8);
  int const target_access_rows = kWarpSize / target_memory_access_width;
  int const access_width =
      target_access_rows > row_shape
          ? kWarpSize / row_shape
          : sm70_const_min(
                shape_width,
                sm70_const_min(kWarpSize, target_memory_access_width));
  if (access_width <= 0) {
    return false;
  }
  int const access_rows =
      target_access_rows > row_shape
          ? row_shape
          : sm70_const_min(kShapeRow, kWarpSize / access_width);
  if (access_rows <= 0) {
    return false;
  }
  return row_shape / access_rows > 0 && shape_width / access_width > 0;
}

constexpr bool sm70_marlin_cta_geometry_is_supported_constexpr(
    int cta_m, int cta_n, int cta_k, int warps, int warp_m, int warp_n,
    int warp_k) {
  if (!sm70_marlin_warp_decomposition_supported(
          cta_m, cta_n, cta_k, warps, warp_m, warp_n, warp_k)) {
    return false;
  }
  int const threads = warps * 32;
  // Current SM70 Marlin uses CUTLASS' Volta row-major/row-major
  // DefaultMmaCore. Its A/B thread maps are fixed to 128-bit half loads.
  // CTA_K=16 is a mathematical GEMM shape but does not provide enough
  // contiguous A accesses for that thread map.
  if (!sm70_marlin_pitchlinear_threadmap_supported(
          cta_k, cta_m, threads, 4, 8, 8)) {
    return false;
  }
  if (!sm70_marlin_b_threadmap_matches_quant_iterator(cta_n, cta_k,
                                                      threads)) {
    return false;
  }
  return sm70_marlin_volta_epilogue_supported(cta_m, cta_n, warps, warp_m);
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK>
struct Sm70WarpShape {
  static_assert(CtaM == 32 || CtaM == 64 || CtaM == 128 || CtaM == 256,
                "SM70 Marlin supports CTA_M in {32, 64, 128, 256}.");
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 Marlin supports CTA_N in {64, 128, 256}.");
  static_assert(CtaK == 16 || CtaK == 32 || CtaK == 64 || CtaK == 128,
                "SM70 Marlin supports CTA_K in {16, 32, 64, 128}.");
  static_assert(Warps == 4 || Warps == 8,
                "SM70 Marlin supports 4 or 8 warps.");
  static_assert(WarpM == 32 || WarpM == 64,
                "SM70 Marlin supports Warp_M in {32, 64}.");
  static_assert(WarpN == 32 || WarpN == 64,
                "SM70 Marlin supports Warp_N in {32, 64}.");
  static_assert(WarpK == 16 || WarpK == 32,
                "SM70 Marlin supports Warp_K in {16, 32}.");
  static_assert(CtaM % WarpM == 0,
                "SM70 Marlin requires CTA_M divisible by Warp_M.");
  static_assert(CtaN % WarpN == 0,
                "SM70 Marlin requires CTA_N divisible by Warp_N.");
  static_assert(CtaK % WarpK == 0,
                "SM70 Marlin requires CTA_K divisible by Warp_K.");
  static_assert((CtaM / WarpM) * (CtaN / WarpN) * (CtaK / WarpK) == Warps,
                "SM70 Marlin explicit warp shape must decompose the CTA into "
                "the requested warp count.");
  static_assert(sm70_marlin_cta_geometry_is_supported_constexpr(
                    CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK),
                "SM70 Marlin CTA geometry is not supported by the current "
                "CUTLASS Volta thread maps, epilogue, and quantized B "
                "iterator contract.");

  using Type = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
};

struct Sm70CtaGeometry {
  int cta_m;
  int cta_n;
  int cta_k;
  int warps;
  int warp_m;
  int warp_n;
  int warp_k;
};

struct Sm70MarlinAutoParams {
  Sm70CtaGeometry geometry;
  int requested_split_k;
  bool use_metadata_vector_words;
  int packed_macro_n;
};

struct Sm70MarlinDenseAutoParamsContext {
  char const* quant_format;
  int64_t group_size;
  int64_t size_m;
  int64_t size_n;
  int64_t size_k;
  int packed_macro_n;
};

struct Sm70MarlinMoeAutoParamsContext {
  char const* quant_format;
  int64_t group_size;
  int64_t moe_block_size;
  int64_t top_k;
  int64_t size_m;
  int64_t size_n;
  int64_t size_k;
  int packed_macro_n;
};

inline constexpr char const* kSupportedSm70MarlinCtaGeometries =
    "{CTA_M}x{CTA_N}x{CTA_K}x{Warps}x{WarpM}x{WarpN}x{WarpK} "
    "with CTA_M in {32,64,128,256}, CTA_N in {64,128,256}, "
    "CTA_K in {16,32,64,128}, Warps in {4,8}, WarpM in {32,64}, "
    "WarpN in {32,64}, WarpK in {16,32}, a valid CTA/warp "
    "decomposition, current CUTLASS Volta thread-map support, and "
    "phase-aware SM70 Marlin warp-K offset support.";

inline bool sm70_marlin_cta_geometry_is_supported(
    Sm70CtaGeometry geometry) {
  return sm70_marlin_cta_geometry_is_supported_constexpr(
      geometry.cta_m, geometry.cta_n, geometry.cta_k, geometry.warps,
      geometry.warp_m, geometry.warp_n, geometry.warp_k);
}

inline int sm70_marlin_auto_packed_macro_n(int64_t size_n) {
  if (size_n % 256 == 0) {
    return 256;
  }
  if (size_n % 128 == 0) {
    return 128;
  }
  if (size_n % 64 == 0) {
    return 64;
  }
  TORCH_CHECK(false, "SM70 Marlin requires size_n divisible "
                     "by 64. Got size_n = ", size_n, ".");
  return 0;
}

inline bool sm70_marlin_packed_macro_n_is_supported(int packed_macro_n) {
  return packed_macro_n == 64 || packed_macro_n == 128 || packed_macro_n == 256;
}

inline bool sm70_marlin_requested_split_k_is_supported(
    int requested_split_k) {
  return requested_split_k == 1 || requested_split_k == 2 ||
         requested_split_k == 4 || requested_split_k == 8;
}

inline Sm70MarlinAutoParams sm70_marlin_default_auto_params(
    int packed_macro_n);

inline Sm70CtaGeometry sm70_marlin_default_geometry_from_packed_macro_n(
    int packed_macro_n) {
  switch (packed_macro_n) {
    case 64:
      return {64, 64, 32, 4, 32, 32, 32};
    case 128:
      return {32, 128, 32, 4, 32, 32, 32};
    case 256:
      return {32, 256, 32, 4, 32, 64, 32};
    default:
      TORCH_CHECK(false, "Unsupported SM70 Marlin packed macro-N=",
                  packed_macro_n,
                  ".");
  }
  return {0, 0, 0, 0, 0, 0, 0};
}

inline Sm70MarlinAutoParams sm70_marlin_default_auto_params(
    int packed_macro_n) {
  return {sm70_marlin_default_geometry_from_packed_macro_n(packed_macro_n),
          1,
          true,
          packed_macro_n};
}

inline bool sm70_marlin_parse_int_component(char const*& cursor, int& value) {
  char* end = nullptr;
  long parsed = std::strtol(cursor, &end, 10);
  if (end == cursor) {
    return false;
  }
  value = static_cast<int>(parsed);
  cursor = end;
  return true;
}

inline bool sm70_marlin_parse_geometry_env_value(char const* value,
                                                 Sm70CtaGeometry& geometry) {
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  char const* cursor = value;
  int fields[7] = {0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 7; ++i) {
    if (!sm70_marlin_parse_int_component(cursor, fields[i])) {
      return false;
    }
    if (i < 6) {
      if (*cursor != 'x') {
        return false;
      }
      ++cursor;
    }
  }
  if (*cursor != '\0') {
    return false;
  }
  geometry = {fields[0], fields[1], fields[2], fields[3],
              fields[4], fields[5], fields[6]};
  return true;
}

inline bool sm70_marlin_try_get_geometry_env(char const* env_name,
                                             Sm70CtaGeometry& geometry) {
  char const* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  TORCH_CHECK(sm70_marlin_parse_geometry_env_value(value, geometry),
              "Invalid ", env_name, " value '", value,
              "'. Expected {CTA_M}x{CTA_N}x{CTA_K}x{Warps}x{WarpM}x{WarpN}x{WarpK}.");
  return true;
}

inline bool sm70_marlin_try_get_split_k_env(char const* env_name,
                                            int& requested_split_k) {
  char const* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  char const* cursor = value;
  int parsed = 0;
  TORCH_CHECK(sm70_marlin_parse_int_component(cursor, parsed) &&
                  *cursor == '\0' &&
                  sm70_marlin_requested_split_k_is_supported(parsed),
              "Invalid ", env_name, " value '", value,
              "'. Expected one of 1, 2, 4, or 8.");
  requested_split_k = parsed;
  return true;
}

inline bool sm70_marlin_try_get_metadata_env(
    char const* env_name, bool& use_metadata_vector_words) {
  char const* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  if (std::strcmp(value, "vector_words") == 0) {
    use_metadata_vector_words = true;
    return true;
  }
  if (std::strcmp(value, "lane_vectors") == 0) {
    use_metadata_vector_words = false;
    return true;
  }
  TORCH_CHECK(false, "Invalid ", env_name, " value '", value,
              "'. Expected vector_words or lane_vectors.");
  return false;
}

inline bool sm70_marlin_env_is_set(char const* env_name) {
  char const* value = std::getenv(env_name);
  return value != nullptr && value[0] != '\0';
}

inline bool sm70_marlin_any_auto_env_is_set(
    char const* geometry_env_name, char const* split_k_env_name,
    char const* metadata_env_name) {
  return sm70_marlin_env_is_set(geometry_env_name) ||
         sm70_marlin_env_is_set(split_k_env_name) ||
         sm70_marlin_env_is_set(metadata_env_name);
}

inline void validate_sm70_marlin_packed_macro_n(char const* op_name,
                                                Sm70CtaGeometry geometry,
                                                int packed_macro_n) {
  TORCH_CHECK(sm70_marlin_packed_macro_n_is_supported(packed_macro_n),
              "Unsupported SM70 Marlin packed macro-N for ", op_name, ": ",
              packed_macro_n, ".");
  TORCH_CHECK(packed_macro_n % geometry.cta_n == 0,
              "SM70 Marlin ", op_name,
              " requires PackedMacroN divisible by CTA_N. Got PackedMacroN=",
              packed_macro_n, ", CTA_N=", geometry.cta_n, ".");
}

inline void validate_sm70_marlin_auto_params(
    char const* op_name, Sm70MarlinAutoParams params) {
  TORCH_CHECK(sm70_marlin_cta_geometry_is_supported(params.geometry),
              "Unsupported SM70 Marlin CTA geometry for ", op_name, ": ",
              params.geometry.cta_m, "x", params.geometry.cta_n, "x",
              params.geometry.cta_k, "x", params.geometry.warps, "x",
              params.geometry.warp_m, "x", params.geometry.warp_n, "x",
              params.geometry.warp_k, ". Supported geometries are ",
              kSupportedSm70MarlinCtaGeometries, ".");
  TORCH_CHECK(
      sm70_marlin_requested_split_k_is_supported(params.requested_split_k),
      "Unsupported SM70 Marlin requested split-K for ", op_name, ": ",
      params.requested_split_k, ". Expected one of 1, 2, 4, or 8.");
  validate_sm70_marlin_packed_macro_n(op_name, params.geometry,
                                      params.packed_macro_n);
}

inline Sm70MarlinAutoParams sm70_marlin_auto_params_from_env(
    char const* op_name, char const* geometry_env_name,
    char const* split_k_env_name, char const* metadata_env_name,
    int packed_macro_n) {
  Sm70MarlinAutoParams params =
      sm70_marlin_default_auto_params(packed_macro_n);
  sm70_marlin_try_get_geometry_env(geometry_env_name, params.geometry);
  sm70_marlin_try_get_split_k_env(split_k_env_name, params.requested_split_k);
  sm70_marlin_try_get_metadata_env(metadata_env_name,
                                   params.use_metadata_vector_words);
  validate_sm70_marlin_auto_params(op_name, params);
  return params;
}

inline bool sm70_marlin_dense_auto_env_is_set() {
  return sm70_marlin_any_auto_env_is_set(
      "SM70_MARLIN_DENSE_CTA_GEOMETRY", "SM70_MARLIN_DENSE_SPLIT_K",
      "SM70_MARLIN_DENSE_METADATA_CACHE");
}

inline Sm70MarlinAutoParams sm70_marlin_dense_auto_params_from_env(
    Sm70MarlinDenseAutoParamsContext const& ctx) {
  return sm70_marlin_auto_params_from_env(
      "Dense", "SM70_MARLIN_DENSE_CTA_GEOMETRY",
      "SM70_MARLIN_DENSE_SPLIT_K", "SM70_MARLIN_DENSE_METADATA_CACHE",
      ctx.packed_macro_n);
}

inline bool
sm70_marlin_dense_try_select_quanttrio_qwen3_6_27b_awq_params(
    Sm70MarlinDenseAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint4") != 0 ||
      ctx.group_size != 128 || ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry,
                              int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (ctx.size_n == 4352 && ctx.size_k == 5120) {
    if (ctx.size_m <= 16) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 48) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m < 128) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 768) {
    if (ctx.size_m <= 16) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
    }
    if (ctx.size_m <= 48) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, true);
    }
    if (ctx.size_m < 128) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 1536) {
    if (ctx.size_m <= 16) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, false);
    }
    if (ctx.size_m <= 48) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m < 128) {
      return set_params({64, 64, 64, 4, 64, 64, 16}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 2176) {
    if (ctx.size_m <= 16) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 48) {
      return set_params({32, 64, 128, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.size_m < 128) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 4352) {
    if (ctx.size_m <= 16) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 48) {
      return set_params({32, 256, 64, 8, 32, 64, 32}, 4, true);
    }
    if (ctx.size_m < 128) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 8704 && ctx.size_k == 5120) {
    if (ctx.size_m <= 48) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m < 128) {
      return set_params({64, 128, 128, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  return false;
}

inline bool
sm70_marlin_dense_try_select_cyankiwi_qwen3_6_27b_awq_bf16_nvfp4_params(
    Sm70MarlinDenseAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "nvfp4") != 0 ||
      ctx.group_size != 16 || ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry,
                              int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (ctx.size_n == 2048 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32,128,32,4,32,32,32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,64,16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64,64,64,4,32,64,32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, false);
  }

  if (ctx.size_n == 3584 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32,128,32,4,32,32,32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,64,16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64,64,32,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 4352 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32,128,32,4,32,32,32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64,64,32,4,32,32,32}, 4, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, false);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 768) {
    if (ctx.size_m <= 1) {
      return set_params({64,64,64,4,32,64,32}, 1, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,64,16}, 1, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,32,4,32,32,16}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, false);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 1536) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,64,4,32,64,16}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({64,64,32,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64,64,32,4,32,32,32}, 1, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 2176) {
    if (ctx.size_m <= 1) {
      return set_params({32,128,32,4,32,32,32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,128,4,32,64,32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,128,128,8,32,64,32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 4352) {
    if (ctx.size_m <= 1) {
      return set_params({32,128,32,4,32,32,32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,128,64,8,32,32,32}, 2, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,128,128,8,32,64,32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 8704 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32,128,32,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,128,32,4,32,32,32}, 4, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({64,128,128,8,64,64,32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  return false;
}

inline bool
sm70_marlin_dense_try_select_nvidia_glm_4_7_nvfp4_params(
    Sm70MarlinDenseAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "nvfp4") != 0 ||
      ctx.group_size != 16 || ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry,
                              int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (ctx.size_n == 384 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,64,4,32,64,16}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 8, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,64,16}, 8, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64,64,64,4,64,32,32}, 1, true);
    }
    return set_params({64,128,32,4,32,64,32}, 1, true);
  }

  if (ctx.size_n == 768 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,64,4,32,32,32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,64,16}, 8, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,64,32,4,32,64,32}, 1, true);
    }
    return set_params({64,128,32,4,32,64,32}, 1, true);
  }

  if (ctx.size_n == 3072 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32,128,32,4,32,32,32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,64,16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64,64,32,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,128,64,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 192) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,32,4,32,32,16}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,32,4,32,32,16}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,128,32,4,64,64,32}, 1, true);
    }
    return set_params({128,128,32,4,64,64,32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 384) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,32,4,32,32,16}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,32,4,32,32,16}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,32,4,32,32,16}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 1536) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,128,32,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 3072) {
    if (ctx.size_m <= 1) {
      return set_params({32,128,32,4,32,32,32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,64,16}, 2, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,128,128,8,32,64,32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 6144 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32,128,32,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,128,32,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64,128,32,4,32,64,32}, 4, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  return false;
}

inline bool
sm70_marlin_dense_try_select_nvidia_qwen3_6_35b_a3b_nvfp4_params(
    Sm70MarlinDenseAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr || ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry,
                              int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (std::strcmp(ctx.quant_format, "fp8_e4m3") == 0 &&
      ctx.group_size == 512 &&
      ctx.size_n == 2048 && ctx.size_k == 512) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,32,4,32,32,16}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,32,4,32,32,16}, 1, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,32,4,32,32,16}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (std::strcmp(ctx.quant_format, "fp8_e4m3") == 0 &&
      ctx.group_size == 1024 &&
      ctx.size_n == 2048 && ctx.size_k == 1024) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (std::strcmp(ctx.quant_format, "fp8_e4m3") == 0 &&
      ctx.group_size == 2048 &&
      ctx.size_n == 1536 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,32,4,32,32,16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64,128,32,4,32,64,32}, 1, false);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (std::strcmp(ctx.quant_format, "fp8_e4m3") == 0 &&
      ctx.group_size == 2048 &&
      ctx.size_n == 2560 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,32,4,32,32,16}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 2, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,64,16}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (std::strcmp(ctx.quant_format, "nvfp4") == 0 &&
      ctx.group_size == 16 &&
      ctx.size_n == 128 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,32,4,32,32,16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64,64,64,4,32,64,32}, 1, true);
    }
    return set_params({128,64,128,8,64,64,32}, 1, true);
  }

  if (std::strcmp(ctx.quant_format, "nvfp4") == 0 &&
      ctx.group_size == 16 &&
      ctx.size_n == 256 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,32,4,32,32,16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,64,128,8,64,64,32}, 1, false);
    }
    return set_params({128,64,32,4,32,64,32}, 1, false);
  }

  if (std::strcmp(ctx.quant_format, "nvfp4") == 0 &&
      ctx.group_size == 16 &&
      ctx.size_n == 2048 && ctx.size_k == 64) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,64,4,32,64,16}, 1, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,64,16}, 1, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({32,128,32,4,32,32,32}, 1, true);
    }
    return set_params({128,64,32,4,32,64,32}, 1, false);
  }

  if (std::strcmp(ctx.quant_format, "nvfp4") == 0 &&
      ctx.group_size == 16 &&
      ctx.size_n == 2048 && ctx.size_k == 128) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,64,4,32,32,32}, 1, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,64,16}, 1, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,64,32,4,32,64,32}, 1, true);
    }
    return set_params({128,64,32,4,32,64,32}, 1, true);
  }

  return false;
}

inline bool
sm70_marlin_dense_try_select_nvidia_qwen3_next_80b_a3b_thinking_nvfp4_params(
    Sm70MarlinDenseAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "nvfp4") != 0 ||
      ctx.group_size != 16 || ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry,
                              int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (ctx.size_n == 128 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,32,4,32,32,16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,32,4,32,32,16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64,64,64,4,32,64,32}, 1, true);
    }
    return set_params({128,64,128,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 256 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,32,4,32,32,16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,32,32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,64,128,8,64,64,32}, 1, true);
    }
    return set_params({128,64,32,4,32,64,32}, 1, true);
  }

  if (ctx.size_n == 2048 && ctx.size_k == 64) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,64,4,32,64,16}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,32,32}, 1, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64,64,32,4,32,32,32}, 1, true);
    }
    return set_params({128,64,32,4,32,64,32}, 1, true);
  }

  if (ctx.size_n == 2048 && ctx.size_k == 128) {
    if (ctx.size_m <= 1) {
      return set_params({32,128,32,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,64,16}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,64,32,4,32,64,32}, 1, true);
    }
    return set_params({128,64,32,4,32,64,32}, 1, false);
  }

  if (ctx.size_n == 2048 && ctx.size_k == 512) {
    if (ctx.size_m <= 1) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 1, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,32,4,32,32,16}, 1, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,128,32,4,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  if (ctx.size_n == 2048 && ctx.size_k == 1024) {
    if (ctx.size_m <= 1) {
      return set_params({64,64,32,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32,64,64,4,32,32,32}, 1, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32,64,64,4,32,32,32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128,256,32,8,64,64,32}, 1, true);
    }
    return set_params({128,256,32,8,64,64,32}, 1, true);
  }

  return false;
}

inline bool
sm70_marlin_dense_try_select_cyankiwi_minimax_m2_7_awq_4bit_params(
    Sm70MarlinDenseAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint4b8") != 0 ||
      ctx.group_size != 32 || ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry,
                              int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  // size_n=1024, size_k=3072  (tp8 qkv_proj)
  if (ctx.size_n == 1024 && ctx.size_k == 3072) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=2048, size_k=3072  (tp4 qkv_proj)
  if (ctx.size_n == 2048 && ctx.size_k == 3072) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({64, 64, 32, 4, 64, 32, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=3072, size_k=768  (tp8 o_proj)
  if (ctx.size_n == 3072 && ctx.size_k == 768) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 64, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 64, 4, 32, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=3072, size_k=1536  (tp4 o_proj)
  if (ctx.size_n == 3072 && ctx.size_k == 1536) {
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 64, 4, 64, 64, 16}, 1, false);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  return false;
}

inline bool
sm70_marlin_dense_try_select_cyankiwi_glm_4_7_awq_4bit_params(
    Sm70MarlinDenseAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint4b8") != 0 ||
      ctx.group_size != 32 || ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry,
                              int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  // size_n=1792, size_k=5120  (tp8 qkv_proj)
  if (ctx.size_n == 1792 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 64, 8, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64, 128, 32, 4, 64, 32, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=3584, size_k=5120  (tp4 qkv_proj)
  if (ctx.size_n == 3584 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 64, 4, 32, 64, 32}, 4, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=5120, size_k=1536  (tp8 o_proj)
  if (ctx.size_n == 5120 && ctx.size_k == 1536) {
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=5120, size_k=3072  (tp4 o_proj)
  if (ctx.size_n == 5120 && ctx.size_k == 3072) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 64, 8, 32, 32, 32}, 2, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  return false;
}

inline bool
sm70_marlin_dense_try_select_quanttrio_glm_4_7_awq_params(
    Sm70MarlinDenseAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint4") != 0 ||
      ctx.group_size != 128 || ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry,
                              int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (ctx.size_n == 1792 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, false);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 3072 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, false);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 3584 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, false);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 1536) {
    if (ctx.size_m <= 1) {
      return set_params({64, 128, 32, 4, 32, 64, 32}, 4, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 64, 4, 64, 64, 16}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 3072) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 2, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 6144 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 2, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  return false;
}

inline Sm70MarlinAutoParams sm70_marlin_dense_auto_params(
    char const* quant_format, int64_t group_size, int64_t size_m,
    int64_t size_n, int64_t size_k) {
  Sm70MarlinDenseAutoParamsContext const ctx{
      quant_format, group_size, size_m, size_n, size_k,
      sm70_marlin_auto_packed_macro_n(size_n)};

  if (sm70_marlin_dense_auto_env_is_set()) {
    return sm70_marlin_dense_auto_params_from_env(ctx);
  }

  Sm70MarlinAutoParams params{};
  if (sm70_marlin_dense_try_select_quanttrio_qwen3_6_27b_awq_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_cyankiwi_qwen3_6_27b_awq_bf16_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_nvidia_glm_4_7_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_nvidia_qwen3_6_35b_a3b_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_nvidia_qwen3_next_80b_a3b_thinking_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_cyankiwi_minimax_m2_7_awq_4bit_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_cyankiwi_glm_4_7_awq_4bit_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_quanttrio_glm_4_7_awq_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }

  return sm70_marlin_default_auto_params(ctx.packed_macro_n);
}

inline void validate_sm70_marlin_dense_cta_geometry_supported(char const* op_name,
                                                              Sm70CtaGeometry geometry) {
  TORCH_CHECK(sm70_marlin_cta_geometry_is_supported(geometry),
              "Unsupported SM70 Marlin CTA geometry for ", op_name, ": ",
              geometry.cta_m, "x", geometry.cta_n, "x", geometry.cta_k,
              "x", geometry.warps, "x", geometry.warp_m, "x",
              geometry.warp_n, "x", geometry.warp_k,
              ". Supported geometries are ",
              kSupportedSm70MarlinCtaGeometries, ".");
}

inline void validate_sm70_marlin_dense_cta_n_alignment(char const* op_name,
                                                       Sm70CtaGeometry geometry,
                                                       int64_t size_n) {
  TORCH_CHECK(
      size_n % geometry.cta_n == 0 && size_n % kQuantTileN == 0,
      "SM70 Marlin requires size_n divisible by both CTA_N and 64 for ",
      op_name, " with CTA geometry ", geometry.cta_m, "x", geometry.cta_n,
      "x", geometry.cta_k, "x", geometry.warps, "x", geometry.warp_m,
      "x", geometry.warp_n, "x", geometry.warp_k, ". Got size_n = ",
      size_n, ".");
}

}  // namespace marlin::sm70
