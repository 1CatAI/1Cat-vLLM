#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime_api.h>
#include <torch/library.h>
#include <torch/types.h>

#include <type_traits>

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "quantization/marlin/sm70_marlin_common.cuh"

namespace marlin::sm70 {

template <typename Spec, int CtaM, int CtaN, int Warps, int GroupSize>
struct Sm70MarlinGemmTraits {
  static_assert(CtaM == 32 || CtaM == 64 || CtaM == 128 || CtaM == 256,
                "SM70 Marlin supports CTA_M in {32, 64, 128, 256}.");
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 Marlin supports CTA_N in {64, 128, 256}.");
  static_assert(Warps == 4 || Warps == 8,
                "SM70 Marlin supports 4 or 8 warps.");
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<CtaM, CtaN, kCtaK>;
  using WarpShape = typename Sm70WarpShape<CtaM, CtaN, Warps>::Type;
  static_assert(WarpShape::kM <= 64 && WarpShape::kN <= 64,
                "SM70 Marlin keeps per-warp M/N no larger than 64.");
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAdd>;
  static_assert(MmaCore::kThreads == Warps * 32,
                "SM70 Marlin launch threads must match CUTLASS warp count.");
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA,
      128 / cutlass::sizeof_bits<ElementA>::value>;
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
                "SM70 Marlin B operand must use CUTLASS' predefined Volta "
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

inline dim3 sm70_marlin_cta_grid(int64_t size_m, int64_t size_n, int cta_m,
                                 int cta_n) {
  return dim3(static_cast<unsigned>((size_m + cta_m - 1) / cta_m),
              static_cast<unsigned>(size_n / cta_n));
}

template <int CtaM, int CtaN, int Warps, typename Launcher>
torch::Tensor dispatch_sm70_marlin_group_size(Launcher const& launcher,
                                              int64_t group_size,
                                              char const* quant_name) {
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
      TORCH_CHECK(false, "SM70 Marlin ", quant_name,
                  " supports only group_size -1, 32, 64, or 128. Got ",
                  group_size, ".");
  }
  return torch::Tensor();
}

template <int CtaM, int CtaN, int Warps, typename Launcher>
torch::Tensor dispatch_sm70_marlin_fp8_group_size(Launcher const& launcher,
                                                  int64_t group_size) {
  switch (group_size) {
    case -1:
      return launcher.template operator()<CtaM, CtaN, Warps, -1>();
    case 128:
      return launcher.template operator()<CtaM, CtaN, Warps, 128>();
    default:
      TORCH_CHECK(false,
                  "SM70 Marlin FP8 supports only group_size -1 or 128. Got ",
                  group_size, ".");
  }
  return torch::Tensor();
}

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_geometry(Launcher const& launcher,
                                            Sm70CtaGeometry geometry,
                                            int64_t group_size,
                                            char const* quant_name) {
#define DISPATCH_SM70_GEOMETRY(CM, CN, W)                              \
  if (geometry.cta_m == CM && geometry.cta_n == CN &&                     \
      geometry.warps == W) {                                              \
    return dispatch_sm70_marlin_group_size<CM, CN, W>(launcher,            \
                                                       group_size,         \
                                                       quant_name);        \
  }

  DISPATCH_SM70_GEOMETRY(32, 128, 4)
  DISPATCH_SM70_GEOMETRY(32, 256, 4)
  DISPATCH_SM70_GEOMETRY(64, 64, 4)
  DISPATCH_SM70_GEOMETRY(64, 128, 4)
  DISPATCH_SM70_GEOMETRY(64, 128, 8)
  DISPATCH_SM70_GEOMETRY(64, 256, 4)
  DISPATCH_SM70_GEOMETRY(64, 256, 8)
  DISPATCH_SM70_GEOMETRY(128, 64, 4)
  DISPATCH_SM70_GEOMETRY(128, 64, 8)
  DISPATCH_SM70_GEOMETRY(128, 128, 4)
  DISPATCH_SM70_GEOMETRY(128, 128, 8)
  DISPATCH_SM70_GEOMETRY(128, 256, 8)
  DISPATCH_SM70_GEOMETRY(256, 64, 4)
  DISPATCH_SM70_GEOMETRY(256, 64, 8)
  DISPATCH_SM70_GEOMETRY(256, 128, 8)

#undef DISPATCH_SM70_GEOMETRY

  TORCH_CHECK(false, "Unreachable SM70 Marlin ", quant_name,
              " CTA geometry dispatch.");
}

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_fp8_geometry(Launcher const& launcher,
                                                Sm70CtaGeometry geometry,
                                                int64_t group_size) {
#define DISPATCH_SM70_FP8_GEOMETRY(CM, CN, W)                         \
  if (geometry.cta_m == CM && geometry.cta_n == CN &&                    \
      geometry.warps == W) {                                             \
    return dispatch_sm70_marlin_fp8_group_size<CM, CN, W>(launcher,       \
                                                          group_size);   \
  }

  DISPATCH_SM70_FP8_GEOMETRY(32, 128, 4)
  DISPATCH_SM70_FP8_GEOMETRY(32, 256, 4)
  DISPATCH_SM70_FP8_GEOMETRY(64, 64, 4)
  DISPATCH_SM70_FP8_GEOMETRY(64, 128, 4)
  DISPATCH_SM70_FP8_GEOMETRY(64, 128, 8)
  DISPATCH_SM70_FP8_GEOMETRY(64, 256, 4)
  DISPATCH_SM70_FP8_GEOMETRY(64, 256, 8)
  DISPATCH_SM70_FP8_GEOMETRY(128, 64, 4)
  DISPATCH_SM70_FP8_GEOMETRY(128, 64, 8)
  DISPATCH_SM70_FP8_GEOMETRY(128, 128, 4)
  DISPATCH_SM70_FP8_GEOMETRY(128, 128, 8)
  DISPATCH_SM70_FP8_GEOMETRY(128, 256, 8)
  DISPATCH_SM70_FP8_GEOMETRY(256, 64, 4)
  DISPATCH_SM70_FP8_GEOMETRY(256, 64, 8)
  DISPATCH_SM70_FP8_GEOMETRY(256, 128, 8)

#undef DISPATCH_SM70_FP8_GEOMETRY

  TORCH_CHECK(false, "Unreachable SM70 Marlin FP8 CTA geometry dispatch.");
}

template <int CtaM, int CtaN, int Warps, int GroupSize, typename Launcher>
torch::Tensor dispatch_sm70_marlin_fixed_group_size(Launcher const& launcher,
                                                    int64_t group_size,
                                                    char const* quant_name) {
  if (group_size == GroupSize) {
    return launcher.template operator()<CtaM, CtaN, Warps, GroupSize>();
  }
  TORCH_CHECK(false, "SM70 Marlin ", quant_name, " supports only group_size ",
              GroupSize, ". Got ", group_size, ".");
  return torch::Tensor();
}

template <int GroupSize, typename Launcher>
torch::Tensor dispatch_sm70_marlin_fixed_group_geometry(
    Launcher const& launcher, Sm70CtaGeometry geometry, int64_t group_size,
    char const* quant_name) {
#define DISPATCH_SM70_FIXED_GROUP_GEOMETRY(CM, CN, W)                    \
  if (geometry.cta_m == CM && geometry.cta_n == CN &&                      \
      geometry.warps == W) {                                               \
    return dispatch_sm70_marlin_fixed_group_size<CM, CN, W, GroupSize>(     \
        launcher, group_size, quant_name);                                  \
  }

  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(32, 128, 4)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(32, 256, 4)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(64, 64, 4)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(64, 128, 4)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(64, 128, 8)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(64, 256, 4)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(64, 256, 8)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(128, 64, 4)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(128, 64, 8)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(128, 128, 4)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(128, 128, 8)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(128, 256, 8)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(256, 64, 4)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(256, 64, 8)
  DISPATCH_SM70_FIXED_GROUP_GEOMETRY(256, 128, 8)

#undef DISPATCH_SM70_FIXED_GROUP_GEOMETRY

  TORCH_CHECK(false, "Unreachable SM70 Marlin ", quant_name,
              " CTA geometry dispatch.");
}

}  // namespace marlin::sm70
