#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>
#include <type_traits>

#include "quantization/marlin/sm70_marlin_common.cuh"
#include "quantization/marlin/sm70_marlin_splitk.cuh"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

namespace marlin_moe_wna16 {

using marlin::sm70::Sm70CtaGeometry;
using marlin::sm70::Sm70SplitKPartition;
using marlin::sm70::Sm70WarpShape;
using marlin::sm70::configure_sm70_dynamic_smem;
using marlin::sm70::kCtaK;
using marlin::sm70::kQuantTileK;
using marlin::sm70::kQuantTileN;
using marlin::sm70::sm70_marlin_moe_auto_stage_requested_split_k;
using marlin::sm70::sm70_active_split_k;
using marlin::sm70::sm70_splitk_partition;

inline constexpr char const* kSupportedSm70MarlinMoeCtaGeometries =
    "32x128x4, 32x256x4, 64x64x4, 64x128x4, 64x128x8, "
    "64x256x4, and 64x256x8";

inline int sm70_marlin_moe_auto_cta_n(int64_t size_n) {
  if (size_n % 256 == 0) {
    return 256;
  }
  if (size_n % 128 == 0) {
    return 128;
  }
  if (size_n % 64 == 0) {
    return 64;
  }
  TORCH_CHECK(false, "SM70 Marlin MoE requires size_n divisible by 64. "
                     "Got size_n = ", size_n, ".");
  return 0;
}

inline Sm70CtaGeometry sm70_marlin_moe_auto_stage_cta_geometry_from_cta_n(int64_t tokens,
                                                            int auto_cta_n) {
  switch (auto_cta_n) {
    case 64:
      return {64, 64, 4};
    case 128:
      return tokens >= 4096 ? Sm70CtaGeometry{64, 128, 8}
                            : Sm70CtaGeometry{32, 128, 4};
    case 256:
      return tokens >= 1024 ? Sm70CtaGeometry{64, 256, 4}
                            : Sm70CtaGeometry{32, 256, 4};
    default:
      TORCH_CHECK(false, "Unsupported SM70 Marlin MoE auto CTA_N=",
                  auto_cta_n, ".");
  }
  return {0, 0, 0};
}

inline Sm70CtaGeometry sm70_marlin_moe_u4_zp_auto_stage_cta_geometry_from_cta_n(
    int64_t tokens, int auto_cta_n, int64_t group_size) {
  if (auto_cta_n == 256 && group_size == -1) {
    return {32, 256, 4};
  }
  return sm70_marlin_moe_auto_stage_cta_geometry_from_cta_n(tokens, auto_cta_n);
}

inline Sm70CtaGeometry sm70_marlin_moe_u8_zp_auto_stage_cta_geometry_from_cta_n(
    int64_t tokens, int auto_cta_n, int64_t group_size) {
  if (auto_cta_n == 256 && group_size == -1 && tokens >= 1024) {
    return {64, 256, 8};
  }
  return sm70_marlin_moe_auto_stage_cta_geometry_from_cta_n(tokens, auto_cta_n);
}

inline Sm70CtaGeometry sm70_marlin_moe_auto_stage_cta_geometry(
    int64_t tokens, int64_t size_n) {
  int const auto_cta_n = sm70_marlin_moe_auto_cta_n(size_n);
  return sm70_marlin_moe_auto_stage_cta_geometry_from_cta_n(tokens, auto_cta_n);
}

inline Sm70CtaGeometry sm70_marlin_moe_u4_zp_auto_stage_cta_geometry(
    int64_t tokens, int64_t size_n, int64_t group_size) {
  int const auto_cta_n = sm70_marlin_moe_auto_cta_n(size_n);
  return sm70_marlin_moe_u4_zp_auto_stage_cta_geometry_from_cta_n(
      tokens, auto_cta_n, group_size);
}

inline Sm70CtaGeometry sm70_marlin_moe_u8_zp_auto_stage_cta_geometry(
    int64_t tokens, int64_t size_n, int64_t group_size) {
  int const auto_cta_n = sm70_marlin_moe_auto_cta_n(size_n);
  return sm70_marlin_moe_u8_zp_auto_stage_cta_geometry_from_cta_n(
      tokens, auto_cta_n, group_size);
}

inline bool sm70_marlin_moe_cta_geometry_is_supported(
    Sm70CtaGeometry geometry) {
  int const cta_m = geometry.cta_m;
  int const cta_n = geometry.cta_n;
  int const warps = geometry.warps;
  return (cta_m == 32 && cta_n == 128 && warps == 4) ||
         (cta_m == 32 && cta_n == 256 && warps == 4) ||
         (cta_m == 64 && cta_n == 64 && warps == 4) ||
         (cta_m == 64 && cta_n == 128 && warps == 4) ||
         (cta_m == 64 && cta_n == 128 && warps == 8) ||
         (cta_m == 64 && cta_n == 256 && warps == 4) ||
         (cta_m == 64 && cta_n == 256 && warps == 8);
}

inline void validate_sm70_marlin_moe_stage_cta_geometry_supported(char const* op_name,
                                                                  Sm70CtaGeometry geometry) {
  TORCH_CHECK(sm70_marlin_moe_cta_geometry_is_supported(geometry),
              "Unsupported SM70 Marlin MoE CTA geometry for ", op_name, ": ",
              geometry.cta_m, "x", geometry.cta_n, "x", geometry.warps,
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
      "x", geometry.warps, ". Got size_n = ", size_n, ".");
}

template <int CtaM, int CtaN, int Warps, typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_group_size(
    Launcher const& launcher, int64_t group_size, char const* quant_name) {
  switch (group_size) {
    case -1:
      return launcher.template operator()<CtaM, CtaN, Warps, -1>();
    case 32:
      return launcher.template operator()<CtaM, CtaN, Warps, 32>();
    case 64:
      return launcher.template operator()<CtaM, CtaN, Warps, 64>();
    case 128:
      return launcher.template operator()<CtaM, CtaN, Warps, 128>();
    default:
      TORCH_CHECK(false, "SM70 Marlin MoE ", quant_name,
                  " supports only group_size -1, 32, 64, or 128. Got ",
                  group_size, ".");
  }
  return torch::Tensor();
}

template <int CtaM, int CtaN, int Warps, typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_fp8_group_size(
    Launcher const& launcher, int64_t group_size) {
  switch (group_size) {
    case -1:
      return launcher.template operator()<CtaM, CtaN, Warps, -1>();
    case 128:
      return launcher.template operator()<CtaM, CtaN, Warps, 128>();
    default:
      TORCH_CHECK(false,
                  "SM70 Marlin MoE FP8 supports only group_size -1 "
                  "or 128. Got ",
                  group_size, ".");
  }
  return torch::Tensor();
}

template <int CtaM, int CtaN, int Warps, int GroupSize, typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_fixed_group_size(
    Launcher const& launcher, int64_t group_size, char const* quant_name) {
  if (group_size == GroupSize) {
    return launcher.template operator()<CtaM, CtaN, Warps, GroupSize>();
  }
  TORCH_CHECK(false, "SM70 Marlin MoE ", quant_name,
              " supports only group_size ", GroupSize, ". Got ", group_size,
              ".");
  return torch::Tensor();
}

struct Sm70MarlinMoeGroupSizeDispatch {
  char const* quant_name;

  template <int CtaM, int CtaN, int Warps, typename Launcher>
  torch::Tensor operator()(Launcher const& launcher, int64_t group_size) const {
    return dispatch_sm70_marlin_moe_group_size<CtaM, CtaN, Warps>(
        launcher, group_size, quant_name);
  }
};

struct Sm70MarlinMoeFp8GroupSizeDispatch {
  template <int CtaM, int CtaN, int Warps, typename Launcher>
  torch::Tensor operator()(Launcher const& launcher, int64_t group_size) const {
    return dispatch_sm70_marlin_moe_fp8_group_size<CtaM, CtaN, Warps>(
        launcher, group_size);
  }
};

template <int GroupSize>
struct Sm70MarlinMoeFixedGroupSizeDispatch {
  char const* quant_name;

  template <int CtaM, int CtaN, int Warps, typename Launcher>
  torch::Tensor operator()(Launcher const& launcher, int64_t group_size) const {
    return dispatch_sm70_marlin_moe_fixed_group_size<CtaM, CtaN, Warps,
                                                     GroupSize>(
        launcher, group_size, quant_name);
  }
};

template <typename Launcher, typename Dispatch>
torch::Tensor dispatch_sm70_marlin_moe_cta_geometry(
    Launcher const& launcher, Sm70CtaGeometry geometry, int64_t group_size,
    Dispatch dispatch) {
#define DISPATCH_SM70_MOE_SHARED_GEOMETRY(CM, CN, W)                      \
  if (geometry.cta_m == CM && geometry.cta_n == CN &&                       \
      geometry.warps == W) {                                                \
    return dispatch.template operator()<CM, CN, W>(launcher, group_size);    \
  }

  DISPATCH_SM70_MOE_SHARED_GEOMETRY(32, 128, 4)
  DISPATCH_SM70_MOE_SHARED_GEOMETRY(32, 256, 4)
  DISPATCH_SM70_MOE_SHARED_GEOMETRY(64, 64, 4)
  DISPATCH_SM70_MOE_SHARED_GEOMETRY(64, 128, 4)
  DISPATCH_SM70_MOE_SHARED_GEOMETRY(64, 128, 8)
  DISPATCH_SM70_MOE_SHARED_GEOMETRY(64, 256, 4)
  DISPATCH_SM70_MOE_SHARED_GEOMETRY(64, 256, 8)

#undef DISPATCH_SM70_MOE_SHARED_GEOMETRY

  TORCH_CHECK(false, "Unreachable SM70 Marlin MoE CTA dispatch.");
}

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_geometry(Launcher const& launcher,
                                                Sm70CtaGeometry geometry,
                                                int64_t group_size,
                                                char const* quant_name) {
  return dispatch_sm70_marlin_moe_cta_geometry(
      launcher, geometry, group_size,
      Sm70MarlinMoeGroupSizeDispatch{quant_name});
}

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_fp8_geometry(
    Launcher const& launcher, Sm70CtaGeometry geometry, int64_t group_size) {
  return dispatch_sm70_marlin_moe_cta_geometry(launcher, geometry, group_size,
                                               Sm70MarlinMoeFp8GroupSizeDispatch{});
}

template <int GroupSize, typename Launcher>
torch::Tensor dispatch_sm70_marlin_moe_fixed_group_geometry(
    Launcher const& launcher, Sm70CtaGeometry geometry, int64_t group_size,
    char const* quant_name) {
  return dispatch_sm70_marlin_moe_cta_geometry(
      launcher, geometry, group_size,
      Sm70MarlinMoeFixedGroupSizeDispatch<GroupSize>{quant_name});
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
  static_assert(Shape::kK == kCtaK,
                "SM70 Marlin MoE IteratorA expects CTA_K=32.");

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

template <typename Spec, int CtaM, int CtaN, int Warps, int GroupSize>
struct Sm70MarlinMoeGemmTraits {
  static_assert(CtaM == 32 || CtaM == 64,
                "SM70 Marlin MoE supports CTA_M in {32, 64}.");
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 Marlin MoE supports CTA_N in {64, 128, 256}.");
  static_assert(Warps == 4 || Warps == 8,
                "SM70 Marlin MoE supports 4 or 8 warps.");
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using GemmSpec = Spec;
  static int const kGroupSize = GroupSize;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<CtaM, CtaN, kCtaK>;
  using WarpShape = typename Sm70WarpShape<CtaM, CtaN, Warps>::Type;
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
      ThreadblockShape, typename MmaCore::IteratorThreadMapB, GroupSize>;
  using Mma = cutlass::gemm::threadblock::MmaPipelined<
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
        sm70_splitk_partition<Traits::kGroupSize>(k, requested_split_k, int(blockIdx.z));
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
      SplitK ? (partition_k / kCtaK) : ((k + kCtaK - 1) / kCtaK);
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

  TORCH_CHECK(size_k % int64_t(kCtaK) == 0,
              "SM70 Marlin MoE requires K divisible by 32 for requested_split_k > 1. "
              "Got K=",
              size_k, ", requested_split_k=", requested_split_k, ".");

  auto split_kernel = sm70_marlin_moe_gemm_kernel<Traits, true>;
  smem_bytes = configure_sm70_dynamic_smem<SharedStorage>(split_kernel);

  int64_t const numel = size_m * top_k * size_n;
  C10_CUDA_CHECK(cudaMemsetAsync(
      c.data_ptr<at::Half>(), 0,
      static_cast<size_t>(numel) * sizeof(at::Half), stream));

  int const active_split_k =
      sm70_active_split_k(static_cast<int>(size_k), requested_split_k);
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
