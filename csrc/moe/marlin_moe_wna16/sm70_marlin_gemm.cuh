#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "quantization/marlin/sm70_marlin_common.cuh"
#include "quantization/marlin/sm70_marlin_gemm.cuh"
#include "quantization/marlin/sm70_marlin_mma.cuh"
#include "quantization/marlin/sm70_marlin_splitk.cuh"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

namespace marlin_moe_wna16 {

using marlin::sm70::Sm70CtaGeometry;
using marlin::sm70::Sm70MarlinAutoParams;
using marlin::sm70::Sm70MarlinMoeAutoParamsContext;
using marlin::sm70::Sm70SplitKPartition;
using marlin::sm70::Sm70MarlinMmaPipelined;
using marlin::sm70::Sm70WarpShape;
using marlin::sm70::configure_sm70_dynamic_smem;
using marlin::sm70::kQuantTileK;
using marlin::sm70::kQuantTileN;
using marlin::sm70::sm70_marlin_any_auto_env_is_set;
using marlin::sm70::sm70_marlin_auto_packed_macro_n;
using marlin::sm70::sm70_marlin_auto_params_from_env;
using marlin::sm70::sm70_marlin_cta_geometry_is_supported;
using marlin::sm70::sm70_marlin_default_auto_params;
using marlin::sm70::sm70_active_split_k;
using marlin::sm70::sm70_splitk_partition;
using marlin::sm70::validate_sm70_marlin_auto_params;

inline constexpr char const* kSupportedSm70MarlinMoeCtaGeometries =
    "{CTA_M}x{CTA_N}x{CTA_K}x{Warps}x{WarpM}x{WarpN}x{WarpK} "
    "with CTA_M in {32,64}, CTA_N in {64,128,256}, "
    "CTA_K in {16,32,64,128}, Warps in {4,8}, WarpM in {32,64}, "
    "WarpN in {32,64}, WarpK in {16,32}, a valid CTA/warp "
    "decomposition, current CUTLASS Volta thread-map support, and "
    "phase-aware SM70 Marlin warp-K offset support. CTA_M=128/256 "
    "geometries are dense-only.";

inline bool sm70_marlin_moe_auto_env_is_set() {
  return sm70_marlin_any_auto_env_is_set(
      "SM70_MARLIN_MOE_CTA_GEOMETRY", "SM70_MARLIN_MOE_SPLIT_K",
      "SM70_MARLIN_MOE_METADATA_CACHE");
}

inline Sm70MarlinAutoParams sm70_marlin_moe_auto_params_from_env(
    Sm70MarlinMoeAutoParamsContext const& ctx) {
  return sm70_marlin_auto_params_from_env(
      "MoE", "SM70_MARLIN_MOE_CTA_GEOMETRY", "SM70_MARLIN_MOE_SPLIT_K",
      "SM70_MARLIN_MOE_METADATA_CACHE", ctx.packed_macro_n);
}

inline bool sm70_marlin_moe_try_select_quanttrio_qwen3_6_35b_a3b_awq_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  if (ctx.top_k == 1 && ctx.size_n == 2048) {
    if (ctx.moe_block_size == 8 && ctx.size_k == 128) {
      if (ctx.size_m <= 128) {
        return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
      }
      return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
    }
    if (ctx.moe_block_size == 8 && ctx.size_k == 512) {
      if (ctx.size_m <= 128) {
        return set_params({32, 128, 32, 4, 32, 64, 16}, 1, true);
      }
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if ((ctx.moe_block_size == 16 || ctx.moe_block_size == 32) &&
        ctx.size_k == 512) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.moe_block_size == 64 && ctx.size_k == 128) {
      if (ctx.size_m < 24576) {
        return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
      }
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.moe_block_size == 64 && ctx.size_k == 512) {
      return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
    }
  }

  if (ctx.top_k == 8 && ctx.size_k == 2048) {
    if (ctx.moe_block_size == 8 && ctx.size_n == 128) {
      if (ctx.size_m <= 16) {
        return set_params({32, 64, 32, 4, 32, 32, 16}, 8, true);
      }
      if (ctx.size_m <= 48) {
        return set_params({32, 128, 64, 8, 32, 32, 32}, 2, true);
      }
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    if (ctx.moe_block_size == 8 && ctx.size_n == 256) {
      if (ctx.size_m <= 16) {
        return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
      }
      if (ctx.size_m <= 48) {
        return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
      }
      return set_params({32, 256, 64, 8, 32, 64, 32}, 1, true);
    }
    if (ctx.moe_block_size == 8 && ctx.size_n == 1024) {
      if (ctx.size_m <= 16) {
        return set_params({32, 256, 32, 4, 32, 64, 32}, 4, true);
      }
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.moe_block_size == 16 && ctx.size_n == 1024) {
      if (ctx.size_m <= 48) {
        return set_params({32, 256, 64, 8, 32, 64, 32}, 1, false);
      }
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.moe_block_size == 32 && ctx.size_n == 1024) {
      return set_params({32, 256, 64, 8, 32, 64, 32}, 1, true);
    }
    if (ctx.moe_block_size == 64 && ctx.size_n == 128) {
      if (ctx.size_m < 3072) {
        return set_params({64, 128, 32, 4, 32, 64, 32}, 1, true);
      }
      return set_params({64, 128, 32, 8, 32, 32, 32}, 1, false);
    }
    if (ctx.moe_block_size == 64 &&
        (ctx.size_n == 256 || ctx.size_n == 1024)) {
      return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_quanttrio_qwen3_5_122b_a10b_awq_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1: down projection, size_n=3072
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 128) {
        if (ctx.size_m <= 8) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
        }
        return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
      }
      if (ctx.size_k == 256) {
        if (ctx.size_m <= 8) {
          return set_params({32, 256, 32, 4, 32, 64, 32}, 1, false);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
      if (ctx.size_k == 1024) {
        if (ctx.size_m <= 8) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
    }

    // top_k=8: gate-up projection, size_k=3072
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 256) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
        }
        if (ctx.size_m <= 32) {
          return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
        }
        return set_params({32, 256, 64, 8, 32, 64, 32}, 1, true);
      }
      if (ctx.size_n == 512) {
        if (ctx.size_m <= 1) {
          return set_params({32, 256, 32, 4, 32, 64, 32}, 8, true);
        }
        if (ctx.size_m <= 32) {
          return set_params({32, 256, 64, 8, 32, 64, 32}, 1, false);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
      if (ctx.size_n == 2048) {
        if (ctx.size_m <= 1) {
          return set_params({32, 256, 32, 4, 32, 64, 32}, 2, true);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    if (ctx.top_k == 1 && ctx.size_n == 3072 && ctx.size_k == 1024) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.top_k == 8 && ctx.size_n == 2048 && ctx.size_k == 3072) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    if (ctx.top_k == 1 && ctx.size_n == 3072 && ctx.size_k == 1024) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.top_k == 8 && ctx.size_n == 2048 && ctx.size_k == 3072) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 128) {
        if (ctx.size_m <= 16384) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
      if (ctx.size_k == 256) {
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
      if (ctx.size_k == 1024) {
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 256 || ctx.size_n == 512 || ctx.size_n == 2048) {
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_quanttrio_minimax_m2_7_awq_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1: down projection, size_n=3072
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 384) {
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
      if (ctx.size_k == 1536) {
        if (ctx.size_m <= 8) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
    }

    // top_k=8: gate-up projection, size_k=3072
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 384) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
        }
        if (ctx.size_m <= 32) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 2, true);
        }
        return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
      }
      if (ctx.size_n == 768) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
        }
        return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
      }
      if (ctx.size_n == 3072) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    if (ctx.top_k == 1 && ctx.size_n == 3072 && ctx.size_k == 1536) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.top_k == 8 && ctx.size_n == 3072 && ctx.size_k == 3072) {
      if (ctx.size_m <= 32) {
        return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
      }
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    if (ctx.top_k == 1 && ctx.size_n == 3072 && ctx.size_k == 1536) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.top_k == 8 && ctx.size_n == 3072 && ctx.size_k == 3072) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 384 || ctx.size_k == 1536) {
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 384) {
        return set_params({64, 128, 32, 4, 64, 64, 16}, 1, false);
      }
      if (ctx.size_n == 768 || ctx.size_n == 3072) {
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_cyankiwi_qwen3_5_122b_a10b_awq_8bit_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint8b128") != 0 ||
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 128) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 256) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 1024) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 256) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,128,8,32,64,32}, 1, true);
          }
          return set_params({32,128,64,8,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 512) {
          if (ctx.size_m <= 1) {
            return set_params({32,256,32,4,32,64,32}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,256,64,8,32,64,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 2048) {
          if (ctx.size_m <= 1) {
            return set_params({32,256,32,4,32,64,32}, 2, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 1024) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 2048) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 1024) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 2048) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 128) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 256) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 1024) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 256) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 512) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 2048) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_cyankiwi_qwen3_6_35b_a3b_awq_4bit_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint4") != 0 ||
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
          if (ctx.size_m <= 8) {
            return set_params({32,64,32,4,32,32,16}, 1, false);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
          if (ctx.size_m <= 8) {
            return set_params({32,64,32,4,32,32,16}, 1, true);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,32,4,32,64,16}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 8, false);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,64,8,32,64,16}, 2, false);
          }
          return set_params({32,128,128,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,128,8,32,64,32}, 1, true);
          }
          return set_params({32,256,64,8,32,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
          if (ctx.size_m <= 1) {
            return set_params({32,256,32,4,32,64,32}, 4, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
          if (ctx.size_m <= 32) {
            return set_params({32,256,64,8,32,64,32}, 1, false);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
          if (ctx.size_m <= 2048) {
            return set_params({64,128,32,4,64,64,16}, 1, false);
          }
          return set_params({64,128,32,8,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_cyankiwi_qwen3_6_35b_a3b_awq_nvfp4_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,64,4,32,64,32}, 1, true);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,32,4,32,32,32}, 1, false);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,64,4,32,64,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
          if (ctx.size_m <= 1) {
            return set_params({32,64,32,4,32,32,16}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,64,8,32,32,32}, 4, true);
          }
          return set_params({32,128,128,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,128,8,32,64,32}, 1, true);
          }
          return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 4, false);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({32,128,32,4,32,32,32}, 1, false);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
          if (ctx.size_m <= 32) {
            return set_params({32,256,64,8,32,64,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({32,256,64,8,32,64,32}, 1, false);
      }
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
          if (ctx.size_m <= 16384) {
            return set_params({64,64,32,4,32,32,32}, 1, true);
          }
          return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
          if (ctx.size_m <= 16384) {
            return set_params({64,256,32,4,64,64,32}, 1, false);
          }
          return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
          if (ctx.size_m <= 2048) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({64,128,32,8,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
          if (ctx.size_m <= 2048) {
            return set_params({64,256,32,4,64,64,32}, 1, true);
          }
          return set_params({64,256,64,8,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_cyankiwi_qwen3_coder_next_awq_4bit_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
          if (ctx.size_m <= 10) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          if (ctx.size_m <= 320) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
          if (ctx.size_m <= 10) {
            return set_params({32,128,64,4,32,64,32}, 1, true);
          }
          if (ctx.size_m <= 320) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
          if (ctx.size_m <= 1) {
            return set_params({32,64,32,4,32,32,16}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,64,8,32,64,16}, 2, true);
          }
          return set_params({32,128,128,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,128,8,32,64,32}, 1, true);
          }
          return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
          if (ctx.size_m <= 1) {
            return set_params({32,256,32,4,32,64,32}, 4, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 48 ===
  if (ctx.moe_block_size == 48) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
        return set_params({64,128,32,8,32,32,32}, 1, false);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
        return set_params({64,128,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_cyankiwi_qwen3_coder_next_awq_8bit_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint8b128") != 0 ||
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
          if (ctx.size_m <= 10) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          if (ctx.size_m <= 320) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
          if (ctx.size_m <= 10) {
            return set_params({32,256,64,8,32,64,32}, 1, true);
          }
          if (ctx.size_m <= 320) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
          if (ctx.size_m <= 1) {
            return set_params({32,64,32,4,32,32,16}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,64,8,32,64,16}, 2, true);
          }
          return set_params({32,128,128,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
          if (ctx.size_m <= 1) {
            return set_params({32,256,32,4,32,64,32}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,128,8,32,64,32}, 1, true);
          }
          return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,128,8,32,64,32}, 1, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 48 ===
  if (ctx.moe_block_size == 48) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
        return set_params({64,128,32,8,32,32,32}, 1, false);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
        return set_params({64,128,32,8,32,32,32}, 1, false);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_nvidia_glm_4_7_nvfp4_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 192) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,32,4,32,32,32}, 1, false);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,128,32,4,32,32,32}, 1, false);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 384) {
          if (ctx.size_m <= 8) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 1536) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, false);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 384) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,32,4,32,64,16}, 2, true);
          }
          return set_params({32,128,32,4,32,64,16}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 768) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,64,16}, 4, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,32,4,32,64,16}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 4, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 3072) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 8, true);
          }
          return set_params({32,128,32,4,32,32,32}, 2, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 1536) {
          if (ctx.size_m <= 256) {
            return set_params({32,256,64,8,32,64,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 3072) {
          if (ctx.size_m <= 32) {
            return set_params({32,128,32,4,32,32,32}, 4, true);
          }
          return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 1536) {
        return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 3072) {
        return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 192) {
          if (ctx.size_m <= 16384) {
            return set_params({64,256,32,4,64,64,32}, 1, true);
          }
          return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 384) {
          if (ctx.size_m <= 16384) {
            return set_params({64,256,32,4,64,64,32}, 1, false);
          }
          return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 1536) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 384) {
          if (ctx.size_m <= 2048) {
            return set_params({64,128,32,8,32,32,32}, 1, false);
          }
          return set_params({64,128,32,8,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 768) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 3072) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_nvidia_minimax_m2_7_nvfp4_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 192) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 384) {
          if (ctx.size_m <= 8) {
            return set_params({32,256,32,4,32,64,32}, 1, true);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,256,32,4,32,64,32}, 1, false);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 1536) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,32,4,32,64,16}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 384) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 8, false);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,32,4,32,64,16}, 2, true);
          }
          return set_params({32,128,32,4,32,64,16}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 768) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 4, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,32,4,32,64,16}, 1, true);
          }
          return set_params({32,128,32,4,32,64,16}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 3072) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 4, true);
          }
          return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 1536) {
          if (ctx.size_m <= 256) {
            return set_params({32,256,64,8,32,64,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 3072) {
          if (ctx.size_m <= 32) {
            return set_params({32,256,64,8,32,64,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 1536) {
        return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 3072) {
        return set_params({32,256,64,8,32,64,32}, 1, false);
      }
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 192) {
          if (ctx.size_m <= 16384) {
            return set_params({64,256,32,4,64,64,32}, 1, true);
          }
          return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 384) {
          if (ctx.size_m <= 16384) {
            return set_params({64,256,32,4,64,64,32}, 1, false);
          }
          return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 1536) {
          if (ctx.size_m <= 16384) {
            return set_params({64,256,32,4,64,64,32}, 1, true);
          }
          return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 384) {
          if (ctx.size_m <= 2048) {
            return set_params({64,128,32,8,32,32,32}, 1, true);
          }
          return set_params({64,128,32,8,32,32,32}, 1, false);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 768) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 3072) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_nvidia_qwen3_6_35b_a3b_nvfp4_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
          if (ctx.size_m <= 8) {
            return set_params({32,64,64,4,32,32,32}, 1, true);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
          if (ctx.size_m <= 8) {
            return set_params({32,64,64,4,32,32,32}, 1, true);
          }
          if (ctx.size_m <= 256) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
          if (ctx.size_m <= 8) {
            return set_params({32,128,64,4,32,64,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
          if (ctx.size_m <= 1) {
            return set_params({32,64,32,4,32,32,16}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,64,8,32,32,32}, 4, true);
          }
          return set_params({32,128,128,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,128,8,32,64,32}, 1, true);
          }
          return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 4, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
          if (ctx.size_m <= 32) {
            return set_params({32,256,64,8,32,64,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
          if (ctx.size_m <= 16384) {
            return set_params({64,64,32,4,32,32,32}, 1, false);
          }
          return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
          if (ctx.size_m <= 16384) {
            return set_params({64,256,32,4,64,64,32}, 1, false);
          }
          return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    // top_k=8
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
          if (ctx.size_m <= 2048) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({64,128,32,8,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
          if (ctx.size_m <= 2048) {
            return set_params({64,256,32,4,64,64,32}, 1, true);
          }
          return set_params({64,256,64,8,64,64,32}, 1, false);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
          if (ctx.size_m <= 2048) {
            return set_params({64,256,32,4,64,64,32}, 1, false);
          }
          return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_nvidia_qwen3_next_80b_a3b_thinking_nvfp4_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
          if (ctx.size_m <= 10) {
            return set_params({32,128,32,4,32,32,32}, 1, false);
          }
          if (ctx.size_m <= 320) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
          if (ctx.size_m <= 10) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          if (ctx.size_m <= 320) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,256,32,4,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
          if (ctx.size_m <= 10) {
            return set_params({32,256,64,8,32,64,32}, 1, true);
          }
          if (ctx.size_m <= 320) {
            return set_params({32,128,32,4,32,32,32}, 1, true);
          }
          return set_params({32,128,32,4,32,32,32}, 1, false);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
          if (ctx.size_m <= 1) {
            return set_params({32,64,32,4,32,32,16}, 8, false);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,64,8,32,64,16}, 2, true);
          }
          return set_params({32,128,128,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,32,4,32,32,32}, 8, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,128,128,8,32,64,32}, 1, true);
          }
          return set_params({32,256,64,8,32,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
          if (ctx.size_m <= 1) {
            return set_params({32,128,128,8,32,64,32}, 1, true);
          }
          if (ctx.size_m <= 32) {
            return set_params({32,256,32,4,32,64,32}, 1, false);
          }
          return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
        return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({32,128,32,4,32,32,32}, 1, true);
      }
    }
  }

  // === moe_block_size == 48 ===
  if (ctx.moe_block_size == 48) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
        return set_params({64,128,32,8,32,32,32}, 1, false);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
        return set_params({64,256,32,4,64,64,32}, 1, false);
      }
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 64) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 128) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 1 && ctx.size_n == 2048) {
      if (ctx.size_k == 512) {
          if (ctx.size_m <= 20480) {
            return set_params({64,256,32,4,64,64,32}, 1, false);
          }
          return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    // top_k=10
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 128) {
        return set_params({64,128,32,8,32,32,32}, 1, true);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 256) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
    if (ctx.top_k == 10 && ctx.size_k == 2048) {
      if (ctx.size_n == 1024) {
        return set_params({64,256,32,4,64,64,32}, 1, true);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_cyankiwi_minimax_m2_7_awq_4bit_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1: down projection, size_n=3072
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 192) {
        if (ctx.size_m <= 8) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
      if (ctx.size_k == 384) {
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
      if (ctx.size_k == 1536) {
        if (ctx.size_m <= 8) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
    }

    // top_k=8: gate-up projection, size_k=3072
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 384) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
        }
        if (ctx.size_m <= 32) {
          return set_params({32, 128, 32, 4, 32, 64, 16}, 2, true);
        }
        return set_params({32, 128, 32, 4, 32, 64, 16}, 1, true);
      }
      if (ctx.size_n == 768) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
        }
        return set_params({32, 128, 32, 4, 32, 64, 16}, 1, true);
      }
      if (ctx.size_n == 3072) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1: down projection, size_n=3072
    if (ctx.top_k == 1 && ctx.size_n == 3072 && ctx.size_k == 1536) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    // top_k=8: gate-up projection, size_k=3072
    if (ctx.top_k == 8 && ctx.size_n == 3072 && ctx.size_k == 3072) {
      if (ctx.size_m <= 32) {
        return set_params({32, 128, 32, 4, 32, 64, 16}, 1, true);
      }
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    // top_k=1: down projection, size_n=3072
    if (ctx.top_k == 1 && ctx.size_n == 3072 && ctx.size_k == 1536) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    // top_k=8: gate-up projection, size_k=3072
    if (ctx.top_k == 8 && ctx.size_n == 3072 && ctx.size_k == 3072) {
      return set_params({32, 128, 32, 4, 32, 64, 16}, 1, true);
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1: down projection, size_n=3072
    if (ctx.top_k == 1 && ctx.size_n == 3072) {
      if (ctx.size_k == 192 || ctx.size_k == 384 || ctx.size_k == 1536) {
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
    }
    // top_k=8: gate-up projection, size_k=3072
    if (ctx.top_k == 8 && ctx.size_k == 3072) {
      if (ctx.size_n == 384) {
        if (ctx.size_m <= 2048) {
          return set_params({64, 128, 32, 4, 32, 64, 32}, 1, true);
        }
        return set_params({64, 128, 32, 8, 32, 32, 32}, 1, false);
      }
      if (ctx.size_n == 768) {
        if (ctx.size_m <= 2048) {
          return set_params({64, 256, 32, 4, 64, 64, 32}, 1, true);
        }
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
      if (ctx.size_n == 3072) {
        if (ctx.size_m <= 2048) {
          return set_params({64, 256, 32, 4, 64, 64, 32}, 1, true);
        }
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_cyankiwi_glm_4_7_awq_4bit_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1: down projection, size_n=5120
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 192 || ctx.size_k == 384 || ctx.size_k == 1536) {
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
    }

    // top_k=8: gate-up projection, size_k=5120
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 384) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
        }
        if (ctx.size_m <= 32) {
          return set_params({32, 128, 32, 4, 32, 64, 16}, 2, true);
        }
        return set_params({32, 128, 32, 4, 32, 64, 16}, 1, true);
      }
      if (ctx.size_n == 768) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 64, 16}, 4, false);
        }
        if (ctx.size_m <= 32) {
          return set_params({32, 128, 32, 4, 32, 64, 16}, 1, true);
        }
        return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
      }
      if (ctx.size_n == 3072) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
        }
        return set_params({32, 128, 32, 4, 32, 32, 32}, 2, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    // top_k=1: down projection, size_n=5120
    if (ctx.top_k == 1 && ctx.size_n == 5120 && ctx.size_k == 1536) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    // top_k=8: gate-up projection, size_k=5120
    if (ctx.top_k == 8 && ctx.size_n == 3072 && ctx.size_k == 5120) {
      if (ctx.size_m <= 32) {
        return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
      }
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    // top_k=1: down projection, size_n=5120
    if (ctx.top_k == 1 && ctx.size_n == 5120 && ctx.size_k == 1536) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    // top_k=8: gate-up projection, size_k=5120
    if (ctx.top_k == 8 && ctx.size_n == 3072 && ctx.size_k == 5120) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    // top_k=1: down projection, size_n=5120
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 192 || ctx.size_k == 384 || ctx.size_k == 1536) {
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
    }
    // top_k=8: gate-up projection, size_k=5120
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 384) {
        return set_params({64, 128, 64, 4, 64, 64, 32}, 1, false);
      }
      if (ctx.size_n == 768) {
        if (ctx.size_m <= 2048) {
          return set_params({64, 256, 32, 4, 64, 64, 32}, 1, true);
        }
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
      if (ctx.size_n == 3072) {
        if (ctx.size_m <= 2048) {
          return set_params({64, 256, 32, 4, 64, 64, 32}, 1, true);
        }
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
    }
  }

  return false;
}

inline bool sm70_marlin_moe_try_select_quanttrio_glm_4_7_awq_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
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

  // === moe_block_size == 8 ===
  if (ctx.moe_block_size == 8) {
    // top_k=1: down projection, size_n=5120
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 384 || ctx.size_k == 1536) {
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
    }

    // top_k=8: gate-up projection, size_k=5120
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 384) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 8, false);
        }
        if (ctx.size_m <= 32) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 2, true);
        }
        return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
      }
      if (ctx.size_n == 768) {
        if (ctx.size_m <= 1) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 4, false);
        }
        if (ctx.size_m <= 32) {
          return set_params({32, 128, 32, 4, 32, 32, 32}, 1, false);
        }
        return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
      }
      if (ctx.size_n == 3072) {
        if (ctx.size_m <= 1) {
          return set_params({32, 256, 32, 4, 32, 64, 32}, 8, true);
        }
        return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
      }
    }
  }

  // === moe_block_size == 16 ===
  if (ctx.moe_block_size == 16) {
    if (ctx.top_k == 1 && ctx.size_n == 5120 && ctx.size_k == 1536) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.top_k == 8 && ctx.size_n == 3072 && ctx.size_k == 5120) {
      if (ctx.size_m <= 32) {
        return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
      }
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
  }

  // === moe_block_size == 32 ===
  if (ctx.moe_block_size == 32) {
    if (ctx.top_k == 1 && ctx.size_n == 5120 && ctx.size_k == 1536) {
      return set_params({32, 256, 32, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.top_k == 8 && ctx.size_n == 3072 && ctx.size_k == 5120) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
    }
  }

  // === moe_block_size == 64 ===
  if (ctx.moe_block_size == 64) {
    if (ctx.top_k == 1 && ctx.size_n == 5120) {
      if (ctx.size_k == 384 || ctx.size_k == 1536) {
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
    }
    if (ctx.top_k == 8 && ctx.size_k == 5120) {
      if (ctx.size_n == 384) {
        return set_params({64, 128, 32, 4, 64, 64, 16}, 1, false);
      }
      if (ctx.size_n == 768 || ctx.size_n == 3072) {
        return set_params({64, 256, 32, 4, 64, 64, 32}, 1, false);
      }
    }
  }

  return false;
}

inline Sm70MarlinAutoParams sm70_marlin_moe_auto_stage_params(
    char const* quant_format, int64_t group_size,
    int64_t moe_block_size, int64_t top_k, int64_t size_m,
    int64_t size_n, int64_t size_k) {
  Sm70MarlinMoeAutoParamsContext const ctx{
      quant_format, group_size, moe_block_size, top_k, size_m, size_n, size_k,
      sm70_marlin_auto_packed_macro_n(size_n)};

  if (sm70_marlin_moe_auto_env_is_set()) {
    return sm70_marlin_moe_auto_params_from_env(ctx);
  }

  Sm70MarlinAutoParams params{};
  if (sm70_marlin_moe_try_select_quanttrio_qwen3_6_35b_a3b_awq_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_quanttrio_qwen3_5_122b_a10b_awq_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_quanttrio_minimax_m2_7_awq_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_cyankiwi_qwen3_5_122b_a10b_awq_8bit_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_cyankiwi_qwen3_6_35b_a3b_awq_4bit_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_cyankiwi_qwen3_6_35b_a3b_awq_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_cyankiwi_qwen3_coder_next_awq_4bit_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_cyankiwi_qwen3_coder_next_awq_8bit_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_nvidia_glm_4_7_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_nvidia_minimax_m2_7_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_nvidia_qwen3_6_35b_a3b_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_nvidia_qwen3_next_80b_a3b_thinking_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_cyankiwi_minimax_m2_7_awq_4bit_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_cyankiwi_glm_4_7_awq_4bit_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }
  if (sm70_marlin_moe_try_select_quanttrio_glm_4_7_awq_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("MoE", params);
    return params;
  }

  return sm70_marlin_default_auto_params(ctx.packed_macro_n);
}

inline bool sm70_marlin_moe_cta_geometry_is_supported(
    Sm70CtaGeometry geometry) {
  return (geometry.cta_m == 32 || geometry.cta_m == 64) &&
         sm70_marlin_cta_geometry_is_supported(geometry);
}

inline void validate_sm70_marlin_moe_stage_cta_geometry_supported(char const* op_name,
                                                                  Sm70CtaGeometry geometry) {
  TORCH_CHECK(sm70_marlin_moe_cta_geometry_is_supported(geometry),
              "Unsupported SM70 Marlin MoE CTA geometry for ", op_name, ": ",
              geometry.cta_m, "x", geometry.cta_n, "x", geometry.cta_k,
              "x", geometry.warps, "x", geometry.warp_m, "x",
              geometry.warp_n, "x", geometry.warp_k,
              ". Supported geometries are ",
              kSupportedSm70MarlinMoeCtaGeometries, ".");
}

inline void validate_sm70_marlin_moe_stage_cta_n_alignment(char const* op_name,
                                                           Sm70CtaGeometry geometry,
                                                           int64_t size_n) {
  TORCH_CHECK(
      size_n % geometry.cta_n == 0 && size_n % kQuantTileN == 0,
      "SM70 Marlin MoE requires size_n divisible by both CTA_N and 64 for ",
      op_name, " with CTA geometry ", geometry.cta_m, "x", geometry.cta_n,
      "x", geometry.cta_k, "x", geometry.warps, "x", geometry.warp_m,
      "x", geometry.warp_n, "x", geometry.warp_k, ". Got size_n = ",
      size_n, ".");
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int PackedMacroN, typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_group_size(
    Launcher const& launcher, int64_t group_size, char const* quant_name) {
  switch (group_size) {
    case -1:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, -1, PackedMacroN>();
    case 32:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, 32, PackedMacroN>();
    case 64:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, 64, PackedMacroN>();
    case 128:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, 128, PackedMacroN>();
    default:
      TORCH_CHECK(false, "SM70 Marlin MoE ", quant_name,
                  " supports only group_size -1, 32, 64, or 128. Got ",
                  group_size, ".");
  }
  return torch::Tensor();
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int PackedMacroN, typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_group_size_with_group32(
    Launcher const& launcher, int64_t group_size, char const* quant_name) {
  switch (group_size) {
    case -1:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, -1, PackedMacroN>();
    case 32:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, 32, PackedMacroN>();
    case 64:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, 64, PackedMacroN>();
    case 128:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, 128, PackedMacroN>();
    default:
      TORCH_CHECK(false, "SM70 Marlin MoE ", quant_name,
                  " supports only group_size -1, 32, 64, or 128. Got ",
                  group_size, ".");
  }
  return torch::Tensor();
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int PackedMacroN, typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_fp8_group_size(
    Launcher const& launcher, int64_t group_size) {
  switch (group_size) {
    case -1:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, -1, PackedMacroN>();
    case 128:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, 128, PackedMacroN>();
    default:
      TORCH_CHECK(false,
                  "SM70 Marlin MoE fp8_e4m3 supports only group_size -1 "
                  "or 128. Got ",
                  group_size, ".");
  }
  return torch::Tensor();
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int GroupSize, int PackedMacroN, typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_fixed_group_size(
    Launcher const& launcher, int64_t group_size, char const* quant_name) {
  if (group_size == GroupSize) {
    return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM, WarpN,
                                        WarpK, GroupSize, PackedMacroN>();
  }
  TORCH_CHECK(false, "SM70 Marlin MoE ", quant_name,
              " supports only group_size ", GroupSize, ". Got ", group_size,
              ".");
  return torch::Tensor();
}

template <typename Launcher>
struct Sm70MarlinMoeGroupSizeDispatchLauncher {
  Launcher const& inner;
  int64_t group_size;
  char const* quant_name;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
            int WarpK, int PackedMacroN>
  torch::Tensor operator()() const {
    return dispatch_sm70_marlin_moe_group_size<CtaM, CtaN, CtaK, Warps, WarpM,
                                               WarpN, WarpK, PackedMacroN>(
        inner, group_size, quant_name);
  }
};

template <typename Launcher>
struct Sm70MarlinMoeGroupSize32DispatchLauncher {
  Launcher const& inner;
  int64_t group_size;
  char const* quant_name;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
            int WarpK, int PackedMacroN>
  torch::Tensor operator()() const {
    return dispatch_sm70_marlin_moe_group_size_with_group32<
        CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK, PackedMacroN>(
        inner, group_size, quant_name);
  }
};

template <typename Launcher>
struct Sm70MarlinMoeFp8GroupSizeDispatchLauncher {
  Launcher const& inner;
  int64_t group_size;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
            int WarpK, int PackedMacroN>
  torch::Tensor operator()() const {
    return dispatch_sm70_marlin_moe_fp8_group_size<CtaM, CtaN, CtaK, Warps,
                                                   WarpM, WarpN, WarpK,
                                                   PackedMacroN>(
        inner, group_size);
  }
};

template <int GroupSize, typename Launcher>
struct Sm70MarlinMoeFixedGroupSizeDispatchLauncher {
  Launcher const& inner;
  int64_t group_size;
  char const* quant_name;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
            int WarpK, int PackedMacroN>
  torch::Tensor operator()() const {
    return dispatch_sm70_marlin_moe_fixed_group_size<
        CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK, GroupSize,
        PackedMacroN>(inner, group_size, quant_name);
  }
};

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_cta_geometry(
    Launcher const& launcher, Sm70CtaGeometry geometry, int packed_macro_n,
    char const* quant_name) {
#define DISPATCH_SM70_MOE_PACKED_MACRO_N(CM, CN, CK, W, WM, WN, WK, PMN)  \
  if (packed_macro_n == PMN) {                                             \
    return launcher.template operator()<CM, CN, CK, W, WM, WN, WK, PMN>(); \
  }

#define DISPATCH_SM70_MOE_GEOMETRY(CM, CN, CK, W, WM, WN, WK, PMN)         \
  if (geometry.cta_m == CM && geometry.cta_n == CN &&                      \
      geometry.cta_k == CK && geometry.warps == W &&                       \
      geometry.warp_m == WM && geometry.warp_n == WN &&                    \
      geometry.warp_k == WK) {                                             \
    DISPATCH_SM70_MOE_PACKED_MACRO_N(CM, CN, CK, W, WM, WN, WK, PMN)       \
  }

#define FOR_EACH_SM70_MOE_GEOMETRY(M)                                     \
  M(32, 64, 32, 4, 32, 32, 16, 128)                                       \
  M(32, 64, 32, 4, 32, 32, 16, 256)                                       \
  M(32, 64, 64, 4, 32, 32, 32, 128)                                       \
  M(32, 64, 64, 4, 32, 32, 32, 256)                                       \
  M(32, 128, 32, 4, 32, 32, 32, 128)                                      \
  M(32, 128, 32, 4, 32, 32, 32, 256)                                      \
  M(32, 128, 32, 4, 32, 64, 16, 128)                                      \
  M(32, 128, 32, 4, 32, 64, 16, 256)                                      \
  M(32, 128, 64, 4, 32, 64, 32, 128)                                      \
  M(32, 128, 64, 4, 32, 64, 32, 256)                                      \
  M(32, 128, 64, 8, 32, 32, 32, 128)                                      \
  M(32, 128, 64, 8, 32, 32, 32, 256)                                      \
  M(32, 128, 64, 8, 32, 64, 16, 128)                                      \
  M(32, 128, 64, 8, 32, 64, 16, 256)                                      \
  M(32, 128, 128, 8, 32, 64, 32, 128)                                     \
  M(32, 128, 128, 8, 32, 64, 32, 256)                                     \
  M(32, 256, 32, 4, 32, 64, 32, 256)                                      \
  M(32, 256, 64, 8, 32, 64, 32, 256)                                      \
  M(64, 64, 32, 4, 32, 32, 32, 64)                                        \
  M(64, 64, 32, 4, 32, 32, 32, 128)                                       \
  M(64, 64, 32, 4, 32, 32, 32, 256)                                       \
  M(64, 128, 32, 4, 32, 64, 32, 128)                                      \
  M(64, 128, 32, 4, 32, 64, 32, 256)                                      \
  M(64, 128, 32, 4, 64, 64, 16, 128)                                      \
  M(64, 128, 32, 4, 64, 64, 16, 256)                                      \
  M(64, 128, 32, 8, 32, 32, 32, 128)                                      \
  M(64, 128, 32, 8, 32, 32, 32, 256)                                      \
  M(64, 128, 64, 4, 64, 64, 32, 128)                                      \
  M(64, 128, 64, 4, 64, 64, 32, 256)                                      \
  M(64, 256, 32, 4, 64, 64, 32, 256)                                      \
  M(64, 256, 64, 8, 64, 64, 32, 256)

  FOR_EACH_SM70_MOE_GEOMETRY(DISPATCH_SM70_MOE_GEOMETRY)

#undef FOR_EACH_SM70_MOE_GEOMETRY
#undef DISPATCH_SM70_MOE_GEOMETRY
#undef DISPATCH_SM70_MOE_PACKED_MACRO_N

  TORCH_CHECK(false, "Unreachable SM70 Marlin MoE ", quant_name,
              " CTA geometry dispatch.");
}

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_geometry(Launcher const& launcher,
                                                Sm70CtaGeometry geometry,
                                                int packed_macro_n,
                                                int64_t group_size,
                                                char const* quant_name) {
  return dispatch_sm70_marlin_moe_cta_geometry(
      Sm70MarlinMoeGroupSizeDispatchLauncher<Launcher>{
          launcher, group_size, quant_name},
      geometry, packed_macro_n, quant_name);
}

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_geometry_with_group32(
    Launcher const& launcher, Sm70CtaGeometry geometry, int packed_macro_n,
    int64_t group_size, char const* quant_name) {
  return dispatch_sm70_marlin_moe_cta_geometry(
      Sm70MarlinMoeGroupSize32DispatchLauncher<Launcher>{
          launcher, group_size, quant_name},
      geometry, packed_macro_n, quant_name);
}

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_fp8_geometry(
    Launcher const& launcher, Sm70CtaGeometry geometry, int packed_macro_n,
    int64_t group_size) {
  return dispatch_sm70_marlin_moe_cta_geometry(
      Sm70MarlinMoeFp8GroupSizeDispatchLauncher<Launcher>{launcher,
                                                          group_size},
      geometry, packed_macro_n, "fp8_e4m3");
}

template <int GroupSize, typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_fixed_group_geometry(
    Launcher const& launcher, Sm70CtaGeometry geometry, int packed_macro_n,
    int64_t group_size, char const* quant_name) {
  return dispatch_sm70_marlin_moe_cta_geometry(
      Sm70MarlinMoeFixedGroupSizeDispatchLauncher<GroupSize, Launcher>{
          launcher, group_size, quant_name},
      geometry, packed_macro_n, quant_name);
}

inline int moe_route_tile_count(int64_t padded_tokens, int64_t moe_block_size,
                                int cta_m) {
  int64_t const moe_blocks =
      (padded_tokens + moe_block_size - 1) / moe_block_size;
  int64_t const m_tiles_per_block = (moe_block_size + cta_m - 1) / cta_m;
  return static_cast<int>(moe_blocks * m_tiles_per_block);
}

inline int moe_n_tile_count(int64_t size_n, int cta_n) {
  return static_cast<int>(size_n / cta_n);
}

template <int CtaM>
CUTLASS_HOST_DEVICE void decode_moe_route_tile(int route_tile,
                                               int moe_block_size,
                                               int& moe_block,
                                               int& local_m_offset) {
  int const m_tiles_per_block = (moe_block_size + CtaM - 1) / CtaM;
  moe_block = route_tile / m_tiles_per_block;
  int const local_m_tile = route_tile - moe_block * m_tiles_per_block;
  local_m_offset = local_m_tile * CtaM;
}

template <typename Shape_, typename ThreadMap_>
class Sm70MoeGatherIteratorA {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  using Element = cutlass::half_t;
  using Layout = cutlass::layout::RowMajor;
  using TensorCoord = cutlass::MatrixCoord;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  struct Params {
    int lda;
    int moe_block_size;
    int top_k;
    int size_m;
    int expanded_token_count;
    int padded_tokens;

    CUTLASS_HOST_DEVICE
    Params()
        : lda(0),
          moe_block_size(0),
          top_k(0),
          size_m(0),
          expanded_token_count(0),
          padded_tokens(0) {}

    CUTLASS_HOST_DEVICE
    Params(int lda_, int moe_block_size_, int top_k_, int size_m_,
           int expanded_token_count_, int padded_tokens_)
        : lda(lda_),
          moe_block_size(moe_block_size_),
          top_k(top_k_),
          size_m(size_m_),
          expanded_token_count(expanded_token_count_),
          padded_tokens(padded_tokens_) {}
  };

 private:
  cutlass::half_t const* a_;
  int32_t const* sorted_token_ids_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int moe_block_;
  int local_m_offset_;
  int k_offset_;
  bool mask_enabled_;

 public:
  CUTLASS_DEVICE
  Sm70MoeGatherIteratorA(Params const& params,
                         cutlass::half_t const* __restrict__ a,
                         int32_t const* __restrict__ sorted_token_ids,
                         int thread_id, int moe_block, int local_m_offset,
                         int k_offset)
      : a_(a),
        sorted_token_ids_(sorted_token_ids),
        params_(params),
        thread_offset_(ThreadMap::initial_offset(thread_id)),
        moe_block_(moe_block),
        local_m_offset_(local_m_offset),
        k_offset_(k_offset),
        mask_enabled_(true) {}

  CUTLASS_DEVICE
  Sm70MoeGatherIteratorA& operator++() {
    k_offset_ += Shape::kK;
    return *this;
  }

  CUTLASS_DEVICE
  void clear_mask(bool enable = true) {
    if (enable) {
      mask_enabled_ = false;
    }
  }

  CUTLASS_DEVICE
  void enable_mask() { mask_enabled_ = true; }

  CUTLASS_DEVICE
  void load(Fragment& frag) const {
    cutlass::half_t* frag_ptr = frag.data();
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < Fragment::kElements; ++idx) {
      frag_ptr[idx] = cutlass::half_t(0);
    }

    if (!mask_enabled_) {
      return;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      int const local_row =
          local_m_offset_ + thread_offset_.strided() +
          s * ThreadMap::Delta::kStrided;
      int const route_row = moe_block_ * params_.moe_block_size + local_row;

      int sorted_id = -1;
      bool valid_row = local_row < params_.moe_block_size &&
                       route_row < params_.padded_tokens;
      if (valid_row) {
        sorted_id = sorted_token_ids_[route_row];
        valid_row =
            sorted_id >= 0 && sorted_id < params_.expanded_token_count;
      }

      int const token_row = valid_row ? (sorted_id / params_.top_k) : 0;

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int const logical_k =
            k_offset_ + thread_offset_.contiguous() +
            c * ThreadMap::Delta::kContiguous;
        int const frag_base =
            (c + s * ThreadMap::Iterations::kContiguous) *
            ThreadMap::kElementsPerAccess;
        bool const valid = valid_row && logical_k < params_.lda;

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
          int const k_element = logical_k + e;
          frag_ptr[frag_base + e] =
              (valid && k_element < params_.lda)
                  ? a_[int64_t(token_row) * params_.lda + k_element]
                  : cutlass::half_t(0);
        }
      }
    }
  }
};

template <typename Spec, int CtaM, int CtaN, int CtaK, int Warps, int WarpM,
          int WarpN, int WarpK, int GroupSize, int PackedMacroN>
struct Sm70MarlinMoeGemmTraits {
  static_assert(CtaM == 32 || CtaM == 64,
                "SM70 Marlin MoE supports CTA_M in {32, 64}.");
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 Marlin MoE supports CTA_N in {64, 128, 256}.");
  static_assert(CtaK == 16 || CtaK == 32 || CtaK == 64 || CtaK == 128,
                "SM70 Marlin MoE supports CTA_K in {16, 32, 64, 128}.");
  static_assert(Warps == 4 || Warps == 8,
                "SM70 Marlin MoE supports 4 or 8 warps.");
  static_assert(PackedMacroN == 64 || PackedMacroN == 128 ||
                    PackedMacroN == 256,
                "SM70 Marlin MoE packed macro-N must be 64, 128, or 256.");
  static_assert(PackedMacroN % CtaN == 0,
                "SM70 Marlin MoE packed macro-N must be divisible by CTA_N.");
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using GemmSpec = Spec;
  static int const kGroupSize = GroupSize;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<CtaM, CtaN, CtaK>;
  using WarpShape =
      typename Sm70WarpShape<CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK>::Type;
  static_assert(WarpShape::kM <= 64 && WarpShape::kN <= 64,
                "SM70 Marlin MoE keeps per-warp M/N no larger than 64.");
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAdd>;
  static_assert(MmaCore::kThreads == Warps * 32,
                "SM70 Marlin MoE launch threads must match CUTLASS warp count.");
  using IteratorA = typename Spec::template IteratorA<
      ThreadblockShape, typename MmaCore::IteratorThreadMapA>;
  using IteratorB = typename Spec::template IteratorB<
      ThreadblockShape, typename MmaCore::IteratorThreadMapB, GroupSize,
      PackedMacroN>;
  using Mma = Sm70MarlinMmaPipelined<
      ThreadblockShape, IteratorA, typename MmaCore::SmemIteratorA, IteratorB,
      typename MmaCore::SmemIteratorB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy>;
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementAccumulator>;
  using ExpectedSmemLayoutB =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
          cutlass::sizeof_bits<ElementB>::value>;
  using ActualSmemLayoutB = typename Mma::SmemIteratorB::Layout;
  static_assert(std::is_same<ActualSmemLayoutB, ExpectedSmemLayoutB>::value,
                "SM70 Marlin MoE B operand must use CUTLASS' predefined Volta "
                "B-congruous shared-memory layout.");
  static int const kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, OutputOp,
          OutputOp::kCount>::Epilogue;

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };
};

template <typename Traits>
class Sm70MoeScatterEpilogue {
 public:
  using CutlassEpilogue = typename Traits::Epilogue;
  using SharedStorage = typename CutlassEpilogue::Base::SharedStorage;
  using AccumulatorTile = typename CutlassEpilogue::AccumulatorTile;
  using AccumulatorFragmentIterator =
      typename CutlassEpilogue::AccumulatorFragmentIterator;
  using WarpTileIterator = typename CutlassEpilogue::WarpTileIterator;
  using SharedLoadIterator = typename CutlassEpilogue::SharedLoadIterator;
  using OutputTileIterator = typename CutlassEpilogue::OutputTileIterator;
  using ThreadMap = typename OutputTileIterator::ThreadMap;

 private:
  WarpTileIterator warp_tile_iterator_;
  SharedLoadIterator shared_load_iterator_;

  CUTLASS_DEVICE
  void store_fragment(OutputTileIterator const& destination_iterator,
                      typename SharedLoadIterator::Fragment const& frag,
                      int32_t const* __restrict__ sorted_token_ids,
                      float const* __restrict__ topk_weights,
                      cutlass::half_t* __restrict__ c,
                      int n, int moe_block, int local_m_offset,
                      int moe_block_size, int expanded_token_count,
                      int padded_tokens, bool mul_topk_weights,
                      bool atomic_store, float output_scale) const {
    float const* frag_ptr = reinterpret_cast<float const*>(&frag);
    half* c_half = reinterpret_cast<half*>(c);
    int const thread_start_row = destination_iterator.thread_start_row();
    int const thread_start_column = destination_iterator.thread_start_column();

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
         ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int const frag_row_idx =
              row + ThreadMap::Iterations::kRow *
                        (group + ThreadMap::Iterations::kGroup * cluster);
          int const row_offset =
              row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster;
          int const local_row =
              local_m_offset + thread_start_row + row_offset;
          int const route_row = moe_block * moe_block_size + local_row;
          bool valid_row =
              local_row < moe_block_size && route_row < padded_tokens;
          int sorted_id = -1;
          if (valid_row) {
            sorted_id = sorted_token_ids[route_row];
            valid_row =
                sorted_id >= 0 && sorted_id < expanded_token_count;
          }
          float const route_scale =
              (valid_row && mul_topk_weights) ? topk_weights[sorted_id] : 1.0f;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            int const logical_column_base =
                thread_start_column + column * ThreadMap::Delta::kColumn;
            int const frag_base =
                (frag_row_idx * ThreadMap::Iterations::kColumn + column) *
                ThreadMap::kElementsPerAccess;

            if (valid_row) {
              CUTLASS_PRAGMA_UNROLL
              for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
                int64_t const offset =
                    int64_t(sorted_id) * n + logical_column_base + e;
                float const value =
                    frag_ptr[frag_base + e] * route_scale * output_scale;
                if (atomic_store) {
                  atomicAdd(c_half + offset, __float2half_rn(value));
                } else {
                  c[offset] = cutlass::half_t(value);
                }
              }
            }
          }
        }
      }
    }
  }

 public:
  CUTLASS_DEVICE
  Sm70MoeScatterEpilogue(SharedStorage& shared_storage, int thread_idx,
                         int warp_idx, int lane_idx)
      : warp_tile_iterator_(shared_storage.reference(), lane_idx),
        shared_load_iterator_(shared_storage.reference(), thread_idx) {
    using WarpCount = typename CutlassEpilogue::WarpCount;
    int const warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);
    int const warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int const warp_m = warp_mn % WarpCount::kM;
    int const warp_n = warp_mn / WarpCount::kM;
    cutlass::MatrixCoord warp_offset{warp_k * WarpCount::kM + warp_m,
                                     warp_n};
    warp_tile_iterator_.add_tile_offset(warp_offset);
  }

  CUTLASS_DEVICE
  void operator()(OutputTileIterator destination_iterator,
                  AccumulatorTile const& accumulators,
                  int32_t const* __restrict__ sorted_token_ids,
                  float const* __restrict__ topk_weights,
                  cutlass::half_t* __restrict__ c,
                  int n, int moe_block, int local_m_offset,
                  int moe_block_size, int expanded_token_count,
                  int padded_tokens, bool mul_topk_weights,
                  bool atomic_store, float output_scale = 1.0f) {
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      __syncthreads();

      typename AccumulatorFragmentIterator::Fragment accum_fragment;
      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;
      warp_tile_iterator_.store(accum_fragment);

      __syncthreads();

      typename SharedLoadIterator::Fragment aligned_accum_fragment;
      shared_load_iterator_.load(aligned_accum_fragment);

      if (CutlassEpilogue::kPartitionsK > 1) {
        cutlass::plus<typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < CutlassEpilogue::kPartitionsK; ++i) {
          typename SharedLoadIterator::Fragment aligned_addend_fragment;
          shared_load_iterator_.add_pointer_offset(
              CutlassEpilogue::kSmemPointerOffset);
          shared_load_iterator_.load(aligned_addend_fragment);
          aligned_accum_fragment =
              add_fragments(aligned_accum_fragment, aligned_addend_fragment);
        }

        shared_load_iterator_.add_pointer_offset(
            (1 - CutlassEpilogue::kPartitionsK) *
            CutlassEpilogue::kSmemPointerOffset);
      }

      store_fragment(destination_iterator, aligned_accum_fragment,
                     sorted_token_ids, topk_weights, c, n, moe_block,
                     local_m_offset, moe_block_size, expanded_token_count,
                     padded_tokens, mul_topk_weights, atomic_store,
                     output_scale);
      ++destination_iterator;
    }
  }
};

template <typename Traits, bool SplitK>
__global__ __launch_bounds__(Traits::MmaCore::kThreads, 1)
void sm70_marlin_moe_gemm_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    typename Traits::GemmSpec::ScaleElement const* __restrict__ b_scales,
    typename Traits::GemmSpec::ZeroElement const* __restrict__ b_zeros,
    float const* __restrict__ global_scale,
    cutlass::half_t* __restrict__ c,
    int32_t const* __restrict__ sorted_token_ids,
    int32_t const* __restrict__ expert_ids,
    int32_t const* __restrict__ num_tokens_past_padded,
    float const* __restrict__ topk_weights, int moe_block_size, int top_k,
    bool mul_topk_weights, int m, int n, int k, int lda, int requested_split_k) {
  using Mma = typename Traits::Mma;
  using Epilogue = Sm70MoeScatterEpilogue<Traits>;
  constexpr int CtaM = Traits::ThreadblockShape::kM;
  constexpr int CtaN = Traits::ThreadblockShape::kN;
  constexpr int CtaK = Traits::ThreadblockShape::kK;
  using Spec = typename Traits::GemmSpec;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int const thread_idx = threadIdx.x;
  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const lane_idx = threadIdx.x % 32;

  int const padded_tokens = num_tokens_past_padded[0];
  int moe_block = 0;
  int local_m_offset = 0;
  decode_moe_route_tile<CtaM>(int(blockIdx.x), moe_block_size, moe_block,
                              local_m_offset);
  if (moe_block * moe_block_size >= padded_tokens) {
    return;
  }

  int const expert = expert_ids[moe_block];
  if (expert < 0) {
    return;
  }

  int k_begin = 0;
  int partition_k = k;
  if constexpr (SplitK) {
    Sm70SplitKPartition const partition =
        sm70_splitk_partition<Traits::kGroupSize, CtaK>(
            k, requested_split_k, int(blockIdx.z));
    if (partition.partition_k == 0) {
      return;
    }
    k_begin = partition.k_begin;
    partition_k = partition.partition_k;
  }

  int const n_offset = int(blockIdx.y) * CtaN;

  typename Mma::IteratorA iterator_A(
      typename Mma::IteratorA::Params(lda, moe_block_size, top_k, m,
                                      m * top_k, padded_tokens),
      a, sorted_token_ids, thread_idx, moe_block, local_m_offset, k_begin);
  typename Mma::IteratorB iterator_B(
      typename Mma::IteratorB::Params(k, n),
      reinterpret_cast<uint32_t const*>(b_q_weight), b_scales, b_zeros,
      thread_idx, expert, k_begin, n_offset);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  int const gemm_k_iterations =
      SplitK ? (partition_k / CtaK) : ((k + CtaK - 1) / CtaK);
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename Epilogue::OutputTileIterator iterator_D(
      typename Epilogue::OutputTileIterator::Params(
          typename Traits::LayoutC(n)),
      c, cutlass::MatrixCoord(CtaM, n), thread_idx,
      cutlass::MatrixCoord(0, n_offset));

  float output_scale = 1.0f;
  if constexpr (Spec::kUsesGlobalScale) {
    output_scale = global_scale[expert];
  }
  Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(iterator_D, accumulators, sorted_token_ids, topk_weights, c, n,
           moe_block, local_m_offset, moe_block_size, m * top_k,
           padded_tokens, mul_topk_weights, SplitK, output_scale);
}

template <typename Traits>
torch::Tensor launch_sm70_marlin_moe_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zeros,
    torch::Tensor& global_scale, torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids, torch::Tensor& num_tokens_past_padded,
    torch::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, int64_t size_m, int64_t size_n, int64_t size_k,
    int requested_split_k) {
  using Spec = typename Traits::GemmSpec;
  using SharedStorage = typename Traits::SharedStorage;
  constexpr int Warps = Traits::MmaCore::kThreads / 32;
  constexpr int CtaM = Traits::ThreadblockShape::kM;
  constexpr int CtaN = Traits::ThreadblockShape::kN;
  constexpr int CtaK = Traits::ThreadblockShape::kK;

  auto kernel = sm70_marlin_moe_gemm_kernel<Traits, false>;
  size_t smem_bytes = configure_sm70_dynamic_smem<SharedStorage>(kernel);
  dim3 block(Warps * 32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  int const route_tiles =
      moe_route_tile_count(sorted_token_ids.numel(), moe_block_size, CtaM);
  dim3 grid(static_cast<unsigned>(route_tiles),
            static_cast<unsigned>(moe_n_tile_count(size_n, CtaN)));

  auto const* b_scales_ptr =
      reinterpret_cast<typename Spec::ScaleElement const*>(b_scales.data_ptr());
  auto const* b_zeros_ptr =
      reinterpret_cast<typename Spec::ZeroElement const*>(b_zeros.data_ptr());
  float const* global_scale_ptr =
      global_scale.numel() == 0 ? nullptr : global_scale.data_ptr<float>();

  if (requested_split_k == 1) {
    kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
        reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
        b_scales_ptr, b_zeros_ptr, global_scale_ptr,
        reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
        sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_past_padded.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(), int(moe_block_size), int(top_k),
        mul_topk_weights, int(size_m), int(size_n), int(size_k),
        int(a.stride(0)), requested_split_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
  }

  TORCH_CHECK(size_k % int64_t(CtaK) == 0,
              "SM70 Marlin MoE requires K divisible by CTA_K=", CtaK,
              " for requested_split_k > 1. Got K=", size_k,
              ", requested_split_k=", requested_split_k, ".");

  auto split_kernel = sm70_marlin_moe_gemm_kernel<Traits, true>;
  smem_bytes = configure_sm70_dynamic_smem<SharedStorage>(split_kernel);

  int64_t const numel = size_m * top_k * size_n;
  C10_CUDA_CHECK(cudaMemsetAsync(
      c.data_ptr<at::Half>(), 0,
      static_cast<size_t>(numel) * sizeof(at::Half), stream));

  int const active_split_k =
      sm70_active_split_k(static_cast<int>(size_k), requested_split_k, CtaK);
  grid.z = static_cast<unsigned>(active_split_k);
  split_kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
      b_scales_ptr, b_zeros_ptr, global_scale_ptr,
      reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
      sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(), num_tokens_past_padded.data_ptr<int32_t>(),
      topk_weights.data_ptr<float>(), int(moe_block_size), int(top_k),
      mul_topk_weights, int(size_m), int(size_n), int(size_k),
      int(a.stride(0)), requested_split_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

}  // namespace marlin_moe_wna16
