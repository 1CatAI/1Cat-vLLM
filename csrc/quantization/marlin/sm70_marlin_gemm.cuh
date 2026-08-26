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
#include "quantization/marlin/sm70_marlin_mma.cuh"

namespace marlin::sm70 {

template <typename Spec, int CtaM, int CtaN, int CtaK, int Warps, int WarpM,
          int WarpN, int WarpK, int GroupSize, int PackedMacroN>
struct Sm70MarlinGemmTraits {
  static_assert(CtaM == 32 || CtaM == 64 || CtaM == 128 || CtaM == 256,
                "SM70 Marlin supports CTA_M in {32, 64, 128, 256}.");
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 Marlin supports CTA_N in {64, 128, 256}.");
  static_assert(CtaK == 16 || CtaK == 32 || CtaK == 64 || CtaK == 128,
                "SM70 Marlin supports CTA_K in {16, 32, 64, 128}.");
  static_assert(Warps == 4 || Warps == 8,
                "SM70 Marlin supports 4 or 8 warps.");
  static_assert(PackedMacroN == 64 || PackedMacroN == 128 ||
                    PackedMacroN == 256,
                "SM70 Marlin packed macro-N must be 64, 128, or 256.");
  static_assert(PackedMacroN % CtaN == 0,
                "SM70 Marlin packed macro-N must be divisible by CTA_N.");
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<CtaM, CtaN, CtaK>;
  using WarpShape =
      typename Sm70WarpShape<CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK>::Type;
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

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int PackedMacroN, typename Launcher>
torch::Tensor dispatch_sm70_marlin_group_size(Launcher const& launcher,
                                              int64_t group_size,
                                              char const* quant_name) {
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
      TORCH_CHECK(false, "SM70 Marlin ", quant_name,
                  " supports only group_size -1, 32, 64, or 128. Got ",
                  group_size, ".");
  }
  return torch::Tensor();
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int PackedMacroN, typename Launcher>
torch::Tensor dispatch_sm70_marlin_group_size_with_group32(
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
      TORCH_CHECK(false, "SM70 Marlin ", quant_name,
                  " supports only group_size -1, 32, 64, or 128. Got ",
                  group_size, ".");
  }
  return torch::Tensor();
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int PackedMacroN, typename Launcher>
torch::Tensor dispatch_sm70_marlin_fp8_group_size(Launcher const& launcher,
                                                  int64_t group_size) {
  switch (group_size) {
    case -1:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, -1, PackedMacroN>();
    case 128:
      return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM,
                                          WarpN, WarpK, 128, PackedMacroN>();
    default:
      TORCH_CHECK(false,
                  "SM70 Marlin fp8_e4m3 supports only group_size -1 or 128. Got ",
                  group_size, ".");
  }
  return torch::Tensor();
}

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_cta_geometry(Launcher const& launcher,
                                                Sm70CtaGeometry geometry,
                                                int packed_macro_n,
                                                char const* quant_name) {
#define DISPATCH_SM70_PACKED_MACRO_N(CM, CN, CK, W, WM, WN, WK, PMN)       \
  if (packed_macro_n == PMN) {                                             \
    return launcher.template operator()<CM, CN, CK, W, WM, WN, WK, PMN>(); \
  }

#define DISPATCH_SM70_GEOMETRY(CM, CN, CK, W, WM, WN, WK, PMN)             \
  if (geometry.cta_m == CM && geometry.cta_n == CN &&                      \
      geometry.cta_k == CK && geometry.warps == W &&                       \
      geometry.warp_m == WM && geometry.warp_n == WN &&                    \
      geometry.warp_k == WK) {                                             \
    DISPATCH_SM70_PACKED_MACRO_N(CM, CN, CK, W, WM, WN, WK, PMN)           \
  }

#define FOR_EACH_SM70_GEOMETRY(M)                                         \
  M(32, 64, 32, 4, 32, 32, 16, 128)                                       \
  M(32, 64, 32, 4, 32, 32, 16, 256)                                       \
  M(32, 64, 64, 4, 32, 32, 32, 128)                                       \
  M(32, 64, 64, 4, 32, 32, 32, 256)                                       \
  M(32, 64, 64, 4, 32, 64, 16, 128)                                       \
  M(32, 64, 64, 4, 32, 64, 16, 256)                                       \
  M(32, 64, 128, 4, 32, 64, 32, 128)                                      \
  M(32, 64, 128, 4, 32, 64, 32, 256)                                      \
  M(32, 128, 32, 4, 32, 32, 32, 128)                                      \
  M(32, 128, 32, 4, 32, 32, 32, 256)                                      \
  M(32, 128, 64, 4, 32, 64, 32, 128)                                      \
  M(32, 128, 64, 4, 32, 64, 32, 256)                                      \
  M(32, 128, 64, 8, 32, 32, 32, 128)                                      \
  M(32, 128, 64, 8, 32, 32, 32, 256)                                      \
  M(32, 128, 128, 8, 32, 64, 32, 128)                                     \
  M(32, 128, 128, 8, 32, 64, 32, 256)                                     \
  M(32, 256, 32, 4, 32, 64, 32, 256)                                      \
  M(32, 256, 64, 8, 32, 64, 32, 256)                                      \
  M(64, 64, 32, 4, 32, 32, 32, 64)                                        \
  M(64, 64, 32, 4, 32, 32, 32, 128)                                       \
  M(64, 64, 32, 4, 32, 32, 32, 256)                                       \
  M(64, 64, 32, 4, 64, 32, 16, 128)                                       \
  M(64, 64, 32, 4, 64, 32, 16, 256)                                       \
  M(64, 64, 64, 4, 32, 64, 32, 128)                                       \
  M(64, 64, 64, 4, 32, 64, 32, 256)                                       \
  M(64, 64, 64, 4, 64, 32, 32, 128)                                       \
  M(64, 64, 64, 4, 64, 32, 32, 256)                                       \
  M(64, 64, 64, 4, 64, 64, 16, 128)                                       \
  M(64, 64, 64, 4, 64, 64, 16, 256)                                       \
  M(64, 128, 32, 4, 32, 64, 32, 128)                                      \
  M(64, 128, 32, 4, 32, 64, 32, 256)                                      \
  M(64, 128, 32, 4, 64, 32, 32, 128)                                      \
  M(64, 128, 32, 4, 64, 32, 32, 256)                                      \
  M(64, 128, 128, 8, 64, 64, 32, 128)                                     \
  M(64, 128, 128, 8, 64, 64, 32, 256)                                     \
  M(128, 64, 32, 4, 32, 64, 32, 128)                                      \
  M(128, 64, 32, 4, 32, 64, 32, 256)                                      \
  M(128, 64, 128, 8, 64, 64, 32, 128)                                     \
  M(128, 64, 128, 8, 64, 64, 32, 256)                                     \
  M(128, 128, 32, 4, 64, 64, 32, 128)                                     \
  M(128, 128, 32, 4, 64, 64, 32, 256)                                     \
  M(128, 128, 64, 8, 64, 64, 32, 128)                                     \
  M(128, 128, 64, 8, 64, 64, 32, 256)                                     \
  M(128, 256, 32, 8, 64, 64, 32, 256)

  FOR_EACH_SM70_GEOMETRY(DISPATCH_SM70_GEOMETRY)

#undef FOR_EACH_SM70_GEOMETRY
#undef DISPATCH_SM70_GEOMETRY
#undef DISPATCH_SM70_PACKED_MACRO_N

  TORCH_CHECK(false, "Unreachable SM70 Marlin ", quant_name,
              " CTA geometry dispatch.");
}

template <typename Launcher>
struct Sm70MarlinGroupSizeDispatchLauncher {
  Launcher const& inner;
  int64_t group_size;
  char const* quant_name;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
            int WarpK, int PackedMacroN>
  torch::Tensor operator()() const {
    return dispatch_sm70_marlin_group_size<CtaM, CtaN, CtaK, Warps, WarpM,
                                           WarpN, WarpK, PackedMacroN>(
        inner, group_size, quant_name);
  }
};

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_geometry(Launcher const& launcher,
                                            Sm70CtaGeometry geometry,
                                            int packed_macro_n,
                                            int64_t group_size,
                                            char const* quant_name) {
  return dispatch_sm70_marlin_cta_geometry(
      Sm70MarlinGroupSizeDispatchLauncher<Launcher>{
          launcher, group_size, quant_name},
      geometry, packed_macro_n, quant_name);
}

template <typename Launcher>
struct Sm70MarlinFp8GroupSizeDispatchLauncher {
  Launcher const& inner;
  int64_t group_size;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
            int WarpK, int PackedMacroN>
  torch::Tensor operator()() const {
    return dispatch_sm70_marlin_fp8_group_size<CtaM, CtaN, CtaK, Warps, WarpM,
                                               WarpN, WarpK, PackedMacroN>(
        inner, group_size);
  }
};

template <typename Launcher>
struct Sm70MarlinGroupSize32DispatchLauncher {
  Launcher const& inner;
  int64_t group_size;
  char const* quant_name;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
            int WarpK, int PackedMacroN>
  torch::Tensor operator()() const {
    return dispatch_sm70_marlin_group_size_with_group32<
        CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK, PackedMacroN>(
        inner, group_size, quant_name);
  }
};

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_geometry_with_group32(
    Launcher const& launcher, Sm70CtaGeometry geometry, int packed_macro_n,
    int64_t group_size, char const* quant_name) {
  return dispatch_sm70_marlin_cta_geometry(
      Sm70MarlinGroupSize32DispatchLauncher<Launcher>{
          launcher, group_size, quant_name},
      geometry, packed_macro_n, quant_name);
}

template <typename Launcher>
torch::Tensor dispatch_sm70_marlin_fp8_geometry(Launcher const& launcher,
                                                Sm70CtaGeometry geometry,
                                                int packed_macro_n,
                                                int64_t group_size) {
  return dispatch_sm70_marlin_cta_geometry(
      Sm70MarlinFp8GroupSizeDispatchLauncher<Launcher>{launcher, group_size},
      geometry, packed_macro_n, "fp8_e4m3");
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int GroupSize, int PackedMacroN, typename Launcher>
torch::Tensor dispatch_sm70_marlin_fixed_group_size(Launcher const& launcher,
                                                    int64_t group_size,
                                                    char const* quant_name) {
  if (group_size == GroupSize) {
    return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM, WarpN,
                                        WarpK, GroupSize, PackedMacroN>();
  }
  TORCH_CHECK(false, "SM70 Marlin ", quant_name, " supports only group_size ",
              GroupSize, ". Got ", group_size, ".");
  return torch::Tensor();
}

template <int GroupSize, typename Launcher>
struct Sm70MarlinFixedGroupDispatchLauncher {
  Launcher const& inner;
  int64_t group_size;
  char const* quant_name;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
            int WarpK, int PackedMacroN>
  torch::Tensor operator()() const {
    return dispatch_sm70_marlin_fixed_group_size<
        CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK, GroupSize,
        PackedMacroN>(inner, group_size, quant_name);
  }
};

template <int GroupSize, typename Launcher>
torch::Tensor dispatch_sm70_marlin_fixed_group_geometry(
    Launcher const& launcher, Sm70CtaGeometry geometry, int packed_macro_n,
    int64_t group_size, char const* quant_name) {
  return dispatch_sm70_marlin_cta_geometry(
      Sm70MarlinFixedGroupDispatchLauncher<GroupSize, Launcher>{
          launcher, group_size, quant_name},
      geometry, packed_macro_n, quant_name);
}

}  // namespace marlin::sm70
