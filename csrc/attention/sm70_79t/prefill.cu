// SPDX-License-Identifier: BSD-3-Clause
#include "shared_workspace.h"
#include <cstring>

/***************************************************************************************************
 * End-to-end prefix architecture screen for V100/SM70.
 *
 * QK GEMM writes FP16 logits and row statistics. PV normalizes logits in its
 * A-operand transform and writes one reusable FP16 block partial. A float
 * online accumulator folds that partial into prefix softmax state before the
 * next block, avoiding resident storage for every block partial.
 **************************************************************************************************/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if defined(PREFIX_TAIL_WAVE_GATED)
  #include <cuda.h>
#endif
#include <math_constants.h>
#include <mma.h>
#if defined(PREFIX_QK_CUBLAS_RAW)
  #include <cublas_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#if defined(PREFIX_TORCH_EXTENSION)
  #include <ATen/ATen.h>
  #include <ATen/cuda/CUDAContext.h>
  #include <c10/cuda/CUDAGuard.h>
  #include <c10/cuda/CUDAException.h>
  #include "namespace_config.h"

  // Keep the torch-port diagnostic actionable: the generic c10 CUDA exception
  // only reports "invalid argument", which hides which runtime operation is
  // incompatible with the worker process.
  #undef C10_CUDA_CHECK
  #define C10_CUDA_CHECK(EXPR)                                              \
    do {                                                                    \
      const cudaError_t _onecat_err = (EXPR);                               \
      if (_onecat_err != cudaSuccess) {                                     \
        TORCH_CHECK(false, "79T torch-port CUDA call failed at ", __FILE__, \
                    ":", __LINE__, " (", #EXPR,                             \
                    "): ", cudaGetErrorString(_onecat_err));                \
      }                                                                     \
    } while (0)
#endif

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/vector.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/epilogue/threadblock/default_epilogue_direct_store.h"
#include "include/cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/layout/matrix.h"

#ifndef PREFIX_TORCH_QUERY_TOKENS
  #define PREFIX_TORCH_QUERY_TOKENS 8000
#endif
#ifndef PREFIX_TORCH_ARCHITECTURE_FUNCTION
  #define PREFIX_TORCH_ARCHITECTURE_FUNCTION sm70_d256_gqa_architecture_fwd
#endif
#include "cutlass/numeric_conversion.h"
#include "gemm_with_softmax.h"

namespace {

#ifndef PREFIX_PV_HALF2_TAYLOR_DEGREE
  #define PREFIX_PV_HALF2_TAYLOR_DEGREE 3
#endif
#ifndef PREFIX_PV_HALF2_TAYLOR_C1
  #define PREFIX_PV_HALF2_TAYLOR_C1 1.0f
#endif
#ifndef PREFIX_PV_HALF2_TAYLOR_C2
  #define PREFIX_PV_HALF2_TAYLOR_C2 0.5f
#endif
#ifndef PREFIX_PV_HALF2_TAYLOR_C3
  #define PREFIX_PV_HALF2_TAYLOR_C3 (1.0f / 6.0f)
#endif
#ifndef PREFIX_PV_HALF2_TAYLOR_C4
  #define PREFIX_PV_HALF2_TAYLOR_C4 (1.0f / 24.0f)
#endif
#ifndef PREFIX_PV_HALF2_TAYLOR_C5
  #define PREFIX_PV_HALF2_TAYLOR_C5 (1.0f / 120.0f)
#endif
#ifndef PREFIX_PV_HALF2_RANGE_REDUCTION
  #define PREFIX_PV_HALF2_RANGE_REDUCTION 1
#endif
static_assert(PREFIX_PV_HALF2_RANGE_REDUCTION == 1 ||
                  PREFIX_PV_HALF2_RANGE_REDUCTION == 2 ||
                  PREFIX_PV_HALF2_RANGE_REDUCTION == 4 ||
                  PREFIX_PV_HALF2_RANGE_REDUCTION == 8 ||
                  PREFIX_PV_HALF2_RANGE_REDUCTION == 16,
              "half2 exp range reduction must be a power of two in [1, 16]");

#if defined(PREFIX_QK_BLOCKED_TRANSPOSE) && !defined(PREFIX_QK_CUBLAS_RAW)
  #error "blocked K transpose currently requires the cuBLAS raw-QK route"
#endif
#if defined(PREFIX_QK_BLOCKED_TRANSPOSE) &&      \
    (defined(PREFIX_MATERIALIZED_SLICED_TAIL) || \
     defined(PREFIX_CUBLAS_SLICED_TAIL) || defined(PREFIX_BATCHED_TRI_TAIL))
  #error "blocked K transpose is not implemented for the materialized tail"
#endif
#if defined(PREFIX_QK_LOG2_SCORES) && \
    (!defined(PREFIX_QK_CUBLAS_RAW) || defined(PREFIX_PV_POLY_EXP2))
  #error "log2-domain scores require exact-exp cuBLAS raw QK"
#endif
#if defined(PREFIX_QK_ASYNC_EXP_PIPELINE) &&                              \
    (!defined(PREFIX_QK_CUBLAS_RAW) || !defined(PREFIX_QK_LOG2_SCORES) || \
     !defined(PREFIX_PV_PREEXP_UNNORMALIZED))
  #error "async exp pipeline requires log2 cuBLAS QK and pre-exp PV"
#endif
#if defined(PREFIX_PHASE_BATCHED_MATERIALIZATION) && \
    !defined(PREFIX_QK_CUBLAS_RAW)
  #error "phase-batched materialization currently requires cuBLAS raw QK"
#endif
#if defined(PREFIX_PHASE_SINGLE_PV) && \
    !defined(PREFIX_PHASE_BATCHED_MATERIALIZATION)
  #error "single-PV phase requires full phase-batched score materialization"
#endif
#if defined(PREFIX_GROUPED_QK_PV) &&               \
    (!defined(PREFIX_QK_CUBLAS_RAW) ||             \
     !defined(PREFIX_PV_DIRECT_FP16_ACCUMULATE) || \
     !defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE))
  #error \
      "grouped QK/PV requires raw cuBLAS QK, FP16 direct accumulation, and transposed scores"
#endif
#if defined(PREFIX_PV_FUSED_PREFIX_SUM) &&          \
    (!defined(PREFIX_QK_CUBLAS_RAW) ||              \
     !defined(PREFIX_PV_DIRECT_FP16_ACCUMULATE) ||  \
     !defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE) || \
     defined(PREFIX_GROUPED_QK_PV))
  #error \
      "fused prefix sum requires serial raw cuBLAS QK, FP16 direct accumulation, and transposed scores"
#endif
#if defined(PREFIX_PV_WARP_SPECIALIZED) &&          \
    (!defined(PREFIX_QK_CUBLAS_RAW) ||              \
     !defined(PREFIX_PV_DIRECT_FP16_ACCUMULATE) ||  \
     !defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE) || \
     defined(PREFIX_BATCHED_TRI_TAIL))
  #error \
      "warp-specialized PV requires raw cuBLAS QK, FP16 direct accumulation, transposed scores, and serial prefix blocks"
#endif
#if defined(PREFIX_QK_DIRECT_PROB) &&           \
    (!defined(PREFIX_QK_PRETRANSPOSE_INPUTS) || \
     !defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE))
  #error "direct-probability QK requires pretransposed inputs and scores"
#endif
#if defined(PREFIX_TAIL_WAVE_GATED) &&        \
    (!defined(PREFIX_TAIL_IDLE_SM_OVERLAP) || \
     !defined(PREFIX_TAIL_IDLE_SM_FINE_PV) || \
     !defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE))
  #error "wave-gated tail requires direct fine-PV idle-SM overlap"
#endif
#if defined(PREFIX_QK_SUPERBLOCK_PV_MICROTILES) &&                             \
    (!defined(PREFIX_QK_CUBLAS_RAW) ||                                         \
     !defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE) ||                            \
     !defined(PREFIX_PV_DIRECT_FP16_ACCUMULATE) ||                             \
     !defined(PREFIX_PV_FUSED_PREFIX_SUM) || defined(PREFIX_QK_PV_PIPELINE) || \
     defined(PREFIX_QK_ASYNC_EXP_PIPELINE) ||                                  \
     defined(PREFIX_PHASE_BATCHED_MATERIALIZATION) ||                          \
     defined(PREFIX_GROUPED_QK_PV) || defined(PREFIX_TAIL_PV_OVERLAP))
  #error "QK superblocks require the serial raw-QK, fused-sum FP16-PV route"
#endif

using Element = cutlass::half_t;

#if defined(PREFIX_SCORE_INT8)
  #if defined(PREFIX_PROB_UINT8)
using ScoreElement = uint8_t;
  #else
using ScoreElement = int8_t;
  #endif
  #ifndef PREFIX_STORED_SCORE_SCALE
    #define PREFIX_STORED_SCORE_SCALE 0.0625f
  #endif
constexpr float kStoredScoreScale = PREFIX_STORED_SCORE_SCALE;
constexpr float kScoreOutputAlpha = 0.0625f / kStoredScoreScale;
#else
using ScoreElement = Element;
constexpr float kStoredScoreScale = 1.0f;
#endif

#ifndef QK_TB_M
  #define QK_TB_M 128
#endif
#ifndef QK_TB_N
  #define QK_TB_N 128
#endif
#ifndef QK_WARP_M
  #define QK_WARP_M 32
#endif
#ifndef QK_WARP_N
  #define QK_WARP_N 64
#endif
#ifndef QK_STAGES
  #define QK_STAGES 2
#endif
#ifndef QK_OUTPUT_ELEMENTS
  #define QK_OUTPUT_ELEMENTS 8
#endif

#if defined(PREFIX_QK_PRETRANSPOSE_INPUTS)
  #if !defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
    #error "pretransposed QK inputs require the transposed score workspace"
  #endif
using QKLayoutA = cutlass::layout::ColumnMajor;
using QKLayoutB = cutlass::layout::RowMajor;
#else
using QKLayoutA = cutlass::layout::RowMajor;
using QKLayoutB = cutlass::layout::ColumnMajor;
#endif

#if defined(PREFIX_MATERIALIZED_SLICED_TAIL)
  #if !defined(PREFIX_FULL_ENDPOINT) ||          \
      !defined(PREFIX_QK_PRETRANSPOSE_INPUTS) || \
      !defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
    #error \
        "materialized sliced tail requires full endpoint and pretransposed QK"
  #endif
static_assert(!std::is_integral<ScoreElement>::value,
              "materialized sliced tail requires FP16 probabilities");
#endif

#if defined(PREFIX_CUBLAS_SLICED_TAIL)
  #if !defined(PREFIX_FULL_ENDPOINT) ||              \
      !defined(PREFIX_QK_PRETRANSPOSE_INPUTS) ||     \
      !defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE) || \
      !defined(PREFIX_QK_CUBLAS_RAW) ||              \
      !defined(PREFIX_PV_DIRECT_FP16_ACCUMULATE)
    #error \
        "cuBLAS sliced tail requires full endpoint, raw cuBLAS QK, pretransposed inputs, and FP16 PV"
  #endif
static_assert(!std::is_integral<ScoreElement>::value,
              "cuBLAS sliced tail requires FP16 scores");
  #ifndef PREFIX_TAIL_SLICE_TOKENS
    #define PREFIX_TAIL_SLICE_TOKENS 1000
  #endif
#endif

#if defined(PREFIX_BATCHED_TRI_TAIL)
  #if !defined(PREFIX_FULL_ENDPOINT) ||              \
      !defined(PREFIX_QK_PRETRANSPOSE_INPUTS) ||     \
      !defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE) || \
      !defined(PREFIX_QK_CUBLAS_RAW) ||              \
      !defined(PREFIX_PV_DIRECT_FP16_ACCUMULATE)
    #error \
        "batched triangular tail requires full endpoint, raw cuBLAS QK, pretransposed inputs, and FP16 PV"
  #endif
static_assert(!std::is_integral<ScoreElement>::value,
              "batched triangular tail requires FP16 scores");
  #ifndef PREFIX_BATCHED_TAIL_TILE_TOKENS
    #define PREFIX_BATCHED_TAIL_TILE_TOKENS 1000
  #endif
  #ifndef PREFIX_BATCHED_TAIL_RESIDUAL_TOKENS
    #define PREFIX_BATCHED_TAIL_RESIDUAL_TOKENS 0
  #endif
  #ifndef PREFIX_BATCHED_TRI_PV_PAD_TOKENS
    #define PREFIX_BATCHED_TRI_PV_PAD_TOKENS 0
  #endif
  #ifndef PREFIX_BATCHED_TAIL_QK_ALGO
    #define PREFIX_BATCHED_TAIL_QK_ALGO CUBLAS_GEMM_ALGO9_TENSOR_OP
  #endif
  #ifndef PREFIX_TAIL_QK_TB_M
    #define PREFIX_TAIL_QK_TB_M 128
  #endif
  #ifndef PREFIX_TAIL_QK_TB_N
    #define PREFIX_TAIL_QK_TB_N 128
  #endif
  #ifndef PREFIX_TAIL_QK_WARP_M
    #define PREFIX_TAIL_QK_WARP_M 32
  #endif
  #ifndef PREFIX_TAIL_QK_WARP_N
    #define PREFIX_TAIL_QK_WARP_N 64
  #endif
  #if defined(PREFIX_BATCHED_TRI_FUSE_CAUSAL_MASK) && \
      !defined(PREFIX_BATCHED_TRI_FUSED_PV)
    #error "fused causal mask requires fused triangular-tail PV"
  #endif
  #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
    #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV) && \
        !defined(PREFIX_TAIL_FINE_PV_GROUP_TILES)
      #define PREFIX_TAIL_FINE_PV_GROUP_TILES 4
    #endif
    #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
      #if defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE) && \
          !defined(PREFIX_BATCHED_TRI_REPAIR_FIRST_TILE)
        #error "direct fine-PV accumulation requires exact first-tile repair"
      #endif
      #if defined(PREFIX_BATCHED_TRI_FUSED_PV) ||         \
          defined(PREFIX_BATCHED_TRI_FUSE_CAUSAL_MASK) || \
          defined(PREFIX_BATCHED_TRI_FUSE_NORMALIZE) ||   \
          !defined(PREFIX_PV_FUSED_PREFIX_SUM)
        #error \
            "fine-grained idle-SM tail PV requires non-fused tail tasks and fused prefix sums"
      #endif
    #elif !defined(PREFIX_BATCHED_TRI_FUSED_PV) ||       \
        !defined(PREFIX_BATCHED_TRI_FUSE_CAUSAL_MASK) || \
        !defined(PREFIX_BATCHED_TRI_FUSE_NORMALIZE) ||   \
        !defined(PREFIX_PV_FUSED_PREFIX_SUM)
      #error \
          "idle-SM tail overlap requires the fully fused tail and fused prefix sums"
    #endif
  #endif
  #if defined(PREFIX_BATCHED_TRI_REPAIR_FIRST_TILE)
    #ifndef PREFIX_BATCHED_TRI_REPAIR_TOKENS
      #define PREFIX_BATCHED_TRI_REPAIR_TOKENS 64
    #endif
  #endif
#endif
#if defined(PREFIX_GROUPED_QK_PV)
  #ifndef PREFIX_GROUPED_QK_PV_BLOCKS
    #define PREFIX_GROUPED_QK_PV_BLOCKS 2
  #endif
static_assert(PREFIX_GROUPED_QK_PV_BLOCKS >= 2,
              "grouped QK/PV needs at least two score buffers");
#endif
using QKThreadblockShape = cutlass::gemm::GemmShape<QK_TB_M, QK_TB_N, 32>;
using QKWarpShape = cutlass::gemm::GemmShape<QK_WARP_M, QK_WARP_N, 32>;
using QKInstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
#if defined(PREFIX_SCORE_INT8)
using QKOutputOpBase = cutlass::epilogue::thread::LinearCombinationClamp<
    ScoreElement, QK_OUTPUT_ELEMENTS, Element, float>;
struct QKOutputOp : QKOutputOpBase {
  using Params = typename QKOutputOpBase::Params;
  static constexpr cutlass::epilogue::thread::ScaleType::Kind kScale =
      cutlass::epilogue::thread::ScaleType::Default;

  CUTLASS_HOST_DEVICE
  explicit QKOutputOp(Params const& params) : QKOutputOpBase(params) {}
};
#else
using QKOutputOp = cutlass::epilogue::thread::LinearCombination<
    ScoreElement, QK_OUTPUT_ELEMENTS, Element, Element>;
#endif
#if defined(PREFIX_QK_RAW_FIXED_SHIFT)
static_assert(!std::is_integral<ScoreElement>::value,
              "raw fixed-shift QK currently requires FP16 scores");
using QKRawSwizzle =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using QKRawKernel = typename cutlass::gemm::kernel::DefaultGemm<
    Element, QKLayoutA, 8, Element, QKLayoutB, 8, ScoreElement,
    cutlass::layout::RowMajor, Element, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, QKThreadblockShape, QKWarpShape, QKInstructionShape,
    QKOutputOp, QKRawSwizzle, QK_STAGES, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
#else
using QKGemm =
    cutlass::GemmSoftmax<Element, QKLayoutA, Element, QKLayoutB, ScoreElement,
                         Element, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm70, QKThreadblockShape, QKWarpShape,
                         QKInstructionShape, QKOutputOp, QK_STAGES,
                         cutlass::MatrixShape<1, 1024>>;
#endif

#if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
struct TailQKBatchedZSwizzle
    : cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> {
  using Base = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  CUTLASS_DEVICE
  static cutlass::gemm::GemmCoord get_tile_offset(int log_tile) {
    auto offset = Base::get_tile_offset(log_tile);
    return {offset.m(), offset.n(), 0};
  }
};
using TailQKThreadblockShape =
    cutlass::gemm::GemmShape<PREFIX_TAIL_QK_TB_M, PREFIX_TAIL_QK_TB_N, 32>;
using TailQKWarpShape =
    cutlass::gemm::GemmShape<PREFIX_TAIL_QK_WARP_M, PREFIX_TAIL_QK_WARP_N, 32>;
using TailQKKernel = typename cutlass::gemm::kernel::DefaultGemm<
    Element, cutlass::layout::ColumnMajor, 8, Element,
    cutlass::layout::RowMajor, 8, ScoreElement, cutlass::layout::RowMajor,
    Element, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    TailQKThreadblockShape, TailQKWarpShape, QKInstructionShape, QKOutputOp,
    TailQKBatchedZSwizzle, 2, false, cutlass::arch::OpMultiplyAdd>::GemmKernel;
#endif

#if defined(PREFIX_QK_DIRECT_PROB)
template <typename T>
struct QKFixedShiftExp {
  CUTLASS_HOST_DEVICE
  T operator()(T const& input) const {
    cutlass::fast_exp_op<T> exponential;
    return exponential(input);
  }
};

using QKDirectOutputOp = cutlass::epilogue::thread::LinearCombinationGeneric<
    QKFixedShiftExp, ScoreElement, 2, Element, float,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
using QKDirectSwizzle =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using QKDirectBaseKernel = typename cutlass::gemm::kernel::DefaultGemm<
    Element, QKLayoutA, 8, Element, QKLayoutB, 8, ScoreElement,
    cutlass::layout::RowMajor, Element, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, QKThreadblockShape, QKWarpShape, QKInstructionShape,
    QKDirectOutputOp, QKDirectSwizzle, QK_STAGES, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
using QKDirectEpilogue =
    typename cutlass::epilogue::threadblock::DefaultEpilogueDirectStore<
        typename QKDirectBaseKernel::Epilogue>::Epilogue;
using QKDirectKernel =
    cutlass::gemm::kernel::Gemm<typename QKDirectBaseKernel::Mma,
                                QKDirectEpilogue, QKDirectSwizzle, false>;
#endif

#ifndef PV_TB_M
  #define PV_TB_M 64
#endif
#ifndef PV_TB_N
  #define PV_TB_N 128
#endif
#ifndef PV_WARP_M
  #define PV_WARP_M 32
#endif
#ifndef PV_WARP_N
  #define PV_WARP_N 64
#endif
#ifndef PV_TB_K
  #define PV_TB_K 32
#endif
#ifndef PV_WARP_K
  #define PV_WARP_K 32
#endif
#ifndef PV_STAGES
  #define PV_STAGES 2
#endif
#ifndef PREFIX_TORCH_BLOCK_N
  #define PREFIX_TORCH_BLOCK_N 8192
#endif
#if defined(PREFIX_TORCH_EXTENSION) && PV_TB_M == 128 && PV_TB_N == 256 && \
    PV_WARP_M == 64 && PV_WARP_N == 64
  // M128/W64 uses two contiguous A accesses per thread.  Keep both 64-row
  // groups distinct and reduce their K partitions across the eight warps.
  #define PREFIX_PV_M128_W64_ROW_SUM
#endif

#if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
using PVLayoutA = cutlass::layout::ColumnMajor;
#else
using PVLayoutA = cutlass::layout::RowMajor;
#endif
using PVLayoutB = cutlass::layout::RowMajor;
using PVLayoutC = cutlass::layout::RowMajor;
#if defined(PREFIX_PV_FP32_MMA_ACCUMULATE)
// Volta Tensor Cores accept FP16 operands with FP32 accumulation.  Keeping
// the long-K dot product in FP32 makes larger prefix blocks numerically viable
// without materializing an FP32 probability or output workspace.
using PVAccumulator = float;
#else
using PVAccumulator = Element;
#endif
using PVThreadblockShape = cutlass::gemm::GemmShape<PV_TB_M, PV_TB_N, PV_TB_K>;
using PVWarpShape = cutlass::gemm::GemmShape<PV_WARP_M, PV_WARP_N, PV_WARP_K>;
using PVInstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
#if defined(PREFIX_BATCHED_TRI_TAIL)
struct PVBatchedZSwizzle
    : cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> {
  using Base = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  CUTLASS_DEVICE
  static cutlass::gemm::GemmCoord get_tile_offset(int log_tile) {
    auto offset = Base::get_tile_offset(log_tile);
    // blockIdx.z selects a triangular-tail task in the batched wrapper; it is
    // not a split-K coordinate for the underlying GEMM.
    return {offset.m(), offset.n(), 0};
  }
};
using PVSwizzle = PVBatchedZSwizzle;
#else
using PVSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
#endif
#if defined(PREFIX_PV_DIRECT_FP32_ACCUMULATE)
using PVOutputElement = float;
using PVOutputOp =
    cutlass::epilogue::thread::LinearCombination<PVOutputElement, 4,
                                                 PVAccumulator, float>;
#else
using PVOutputElement = Element;
using PVOutputOp =
    cutlass::epilogue::thread::LinearCombination<Element, 8, PVAccumulator,
                                                 float>;
#endif

#ifndef PREFIX_PV_ALIGNMENT
  #define PREFIX_PV_ALIGNMENT 8
#endif
constexpr int kPVAlignment = PREFIX_PV_ALIGNMENT;
static_assert(kPVAlignment == 4 || kPVAlignment == 8);

using PVDefaultKernel = typename cutlass::gemm::kernel::DefaultGemm<
    Element, PVLayoutA, kPVAlignment, Element, PVLayoutB, kPVAlignment,
    PVOutputElement, PVLayoutC, PVAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, PVThreadblockShape, PVWarpShape, PVInstructionShape,
    PVOutputOp, PVSwizzle, PV_STAGES, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;

using PVDefaultMma = typename PVDefaultKernel::Mma;
using PVHalfIteratorA = typename PVDefaultMma::IteratorA;
using PVIteratorB = typename PVDefaultMma::IteratorB;
using PVSmemIteratorA = typename PVDefaultMma::SmemIteratorA;
using PVSmemIteratorB = typename PVDefaultMma::SmemIteratorB;

#if defined(PREFIX_SCORE_INT8)
using PVIteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
    cutlass::MatrixShape<PVThreadblockShape::kM, PVThreadblockShape::kK>,
    ScoreElement, PVLayoutA, 1, typename PVHalfIteratorA::ThreadMap,
    PVHalfIteratorA::ThreadMap::kElementsPerAccess>;
#else
using PVIteratorA = PVHalfIteratorA;
#endif

__device__ float const* g_row_max = nullptr;
__device__ float const* g_row_inv_sum = nullptr;
__device__ float* g_row_sum_out = nullptr;
__device__ int g_rows = 0;
__device__ float* g_tail_row_sum_out = nullptr;
__device__ int g_tail_rows = 0;
__device__ int g_pv_task_base = 0;
#if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
__global__ void set_pv_task_base_kernel(int task_base) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_pv_task_base = task_base;
  }
}
#endif
__device__ __forceinline__ int pv_task_index() {
#if defined(PREFIX_BATCHED_TRI_REVERSE_PV_TASKS)
  int local_task = int(gridDim.z) - 1 - int(blockIdx.z);
#else
  int local_task = int(blockIdx.z);
#endif
  return g_pv_task_base + local_task;
}
#if defined(PREFIX_TORCH_STABLE_ROWS)
  #include "stable_rows.cuh"
#endif
__device__ __forceinline__ int64_t row_sum_output_index(int row) {
#if defined(PREFIX_BATCHED_TRI_TAIL)
  return int64_t(pv_task_index()) * g_rows + row;
#else
  return row;
#endif
}

#if defined(PREFIX_PV_WARP_SPECIALIZED)
template <int BarrierId>
__device__ __forceinline__ void prefix_named_barrier_arrive() {
  constexpr int kThreads = 256;
  asm volatile("bar.arrive %0, %1;" ::"r"(BarrierId), "r"(kThreads) : "memory");
}

template <int BarrierId>
__device__ __forceinline__ void prefix_named_barrier_sync() {
  constexpr int kThreads = 256;
  asm volatile("bar.sync %0, %1;" ::"r"(BarrierId), "r"(kThreads) : "memory");
}

struct alignas(16) WarpSpecializedPVSharedStorage {
  __half probabilities[2][32 * 32];
  __half values[2][32 * 256];
  float output_tile[4][16 * 16];
};

__global__ __launch_bounds__(256, 2) void warp_specialized_pv_kernel(
    __half const* __restrict__ scores, __half const* __restrict__ value,
    __half* __restrict__ output, int rows, int k, bool accumulate) {
  using namespace nvcuda;
  __shared__ WarpSpecializedPVSharedStorage shared;
  int warp = int(threadIdx.x) >> 5;
  int lane = int(threadIdx.x) & 31;
  int row_base = int(blockIdx.x) * 32;
  int k_tiles = (k + 31) / 32;

  if (warp < 4) {
    float row_sum[8] = {};
    int producer_thread = int(threadIdx.x);
    for (int tile = 0; tile < k_tiles; ++tile) {
      int stage = tile & 1;
      if (tile >= 2) {
        if (stage == 0) {
          prefix_named_barrier_sync<2>();
        } else {
          prefix_named_barrier_sync<3>();
        }
      }
      int key_base = tile * 32;
  #pragma unroll
      for (int local = 0; local < 8; ++local) {
        int local_row = warp * 8 + local;
        int row = row_base + local_row;
        int key_index = key_base + lane;
        float weight = 0.0f;
        if (row < rows && key_index < k) {
          float score = __half2float(scores[int64_t(key_index) * rows + row]);
  #if defined(PREFIX_QK_LOG2_SCORES)
          weight = exp2f(score);
  #else
          weight = exp2f(score * 1.4426950408889634f);
  #endif
        }
        shared.probabilities[stage][local_row * 32 + lane] =
            __float2half_rn(weight);
        row_sum[local] += weight;
      }

      auto* shared_vectors = reinterpret_cast<int4*>(shared.values[stage]);
      auto const* value_vectors =
          reinterpret_cast<int4 const*>(value + int64_t(key_base) * 256);
      constexpr int kVectors = 32 * 256 / 8;
      if (key_base + 32 <= k) {
        for (int vector = producer_thread; vector < kVectors; vector += 128) {
          shared_vectors[vector] = value_vectors[vector];
        }
      } else {
        for (int element = producer_thread; element < 32 * 256;
             element += 128) {
          int key_in_tile = element / 256;
          int d = element - key_in_tile * 256;
          shared.values[stage][element] =
              key_base + key_in_tile < k
                  ? value[int64_t(key_base + key_in_tile) * 256 + d]
                  : __float2half(0.0f);
        }
      }
      if (stage == 0) {
        prefix_named_barrier_arrive<0>();
      } else {
        prefix_named_barrier_arrive<1>();
      }
    }

  #pragma unroll
    for (int local = 0; local < 8; ++local) {
      float sum = row_sum[local];
  #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      }
      int row = row_base + warp * 8 + local;
      if (lane == 0 && row < rows) {
        g_row_sum_out[row_sum_output_index(row)] = sum;
      }
    }
  } else {
    int consumer = warp - 4;
    using FragmentA =
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
    using FragmentB =
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    FragmentC accumulators[2][4];
  #pragma unroll
    for (int m = 0; m < 2; ++m) {
  #pragma unroll
      for (int n = 0; n < 4; ++n) {
        wmma::fill_fragment(accumulators[m][n], 0.0f);
      }
    }

    for (int tile = 0; tile < k_tiles; ++tile) {
      int stage = tile & 1;
      if (stage == 0) {
        prefix_named_barrier_sync<0>();
      } else {
        prefix_named_barrier_sync<1>();
      }
  #pragma unroll
      for (int kk = 0; kk < 32; kk += 16) {
  #pragma unroll
        for (int m = 0; m < 2; ++m) {
          FragmentA fragment_a;
          wmma::load_matrix_sync(
              fragment_a, shared.probabilities[stage] + m * 16 * 32 + kk, 32);
  #pragma unroll
          for (int n = 0; n < 4; ++n) {
            FragmentB fragment_b;
            wmma::load_matrix_sync(
                fragment_b,
                shared.values[stage] + kk * 256 + consumer * 64 + n * 16, 256);
            wmma::mma_sync(accumulators[m][n], fragment_a, fragment_b,
                           accumulators[m][n]);
          }
        }
      }
      if (stage == 0) {
        prefix_named_barrier_arrive<2>();
      } else {
        prefix_named_barrier_arrive<3>();
      }
    }

  #pragma unroll
    for (int m = 0; m < 2; ++m) {
  #pragma unroll
      for (int n = 0; n < 4; ++n) {
        float* scratch = shared.output_tile[consumer];
        wmma::store_matrix_sync(scratch, accumulators[m][n], 16,
                                wmma::mem_row_major);
        __syncwarp();
        for (int index = lane; index < 16 * 16; index += 32) {
          int row = row_base + m * 16 + index / 16;
          int column = consumer * 64 + n * 16 + index % 16;
          if (row < rows) {
            int64_t output_index = int64_t(row) * 256 + column;
            float result = scratch[index];
            if (accumulate) {
              result += __half2float(output[output_index]);
            }
            output[output_index] = __float2half_rn(result);
          }
        }
        __syncwarp();
      }
    }
  }
}
#endif
#if defined(PREFIX_PV_FP16_EXP_LUT)
static_assert(std::is_same<ScoreElement, Element>::value,
              "the full-domain exp LUT requires FP16 scores");
  #ifndef PREFIX_PV_FP16_EXP_LUT_BITS
    #define PREFIX_PV_FP16_EXP_LUT_BITS 16
  #endif
static_assert(PREFIX_PV_FP16_EXP_LUT_BITS >= 8 &&
                  PREFIX_PV_FP16_EXP_LUT_BITS <= 16,
              "FP16 exp LUT index bits must be in [8, 16]");
constexpr int kPrefixFP16ExpLUTBits = PREFIX_PV_FP16_EXP_LUT_BITS;
constexpr int kPrefixFP16ExpLUTDropBits = 16 - kPrefixFP16ExpLUTBits;
constexpr int kPrefixFP16ExpLUTSize = 1 << kPrefixFP16ExpLUTBits;
__device__ float g_prefix_fp16_exp_lut[kPrefixFP16ExpLUTSize];

__global__ void initialize_prefix_fp16_exp_lut() {
  int index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (index < kPrefixFP16ExpLUTSize) {
    union HalfBits {
      uint16_t bits;
      __half value;
    } score;
    constexpr int kBucketMask = (1 << kPrefixFP16ExpLUTDropBits) - 1;
    score.bits =
        uint16_t((index << kPrefixFP16ExpLUTDropBits) + kBucketMask / 2);
    float value = __half2float(score.value);
    constexpr float kLog2E = 1.4426950408889634f;
    g_prefix_fp16_exp_lut[index] =
        isfinite(value) ? exp2f(value * kLog2E) : 0.0f;
  }
}
#endif

#if defined(PREFIX_PV_POLY_EXP2)
// Cubic least-squares approximation of 2^fraction on [0, 1]. Splitting the
// integer exponent into the IEEE-754 exponent field replaces one MUFU.EX2
// with independent FP32 FMA work that can overlap the Volta tensor pipe.
// The sampled maximum relative approximation error is 1.881e-4.
__device__ __forceinline__ float prefix_poly_exp2(float input) {
  float clamped = fminf(fmaxf(input, -126.0f), 0.0f);
  int exponent = __float2int_rd(clamped);
  float fraction = clamped - float(exponent);
  float polynomial =
      fmaf(fraction,
           fmaf(fraction, fmaf(fraction, 0.0790198880f, 0.2241262340f),
                0.6968388310f),
           0.9998119090f);
  float scale = __int_as_float((exponent + 127) << 23);
  return scale * polynomial;
}
#endif

#if defined(PREFIX_QK_FULL_STATS)
struct ExpRowSumTransformA {
  using InputFragment = typename PVIteratorA::Fragment;
  using OutputFragment = cutlass::Array<typename PVSmemIteratorA::Element,
                                        InputFragment::kElements>;
  using ThreadMap = typename PVIteratorA::ThreadMap;
  static constexpr int kAccessesPerVector =
      PVIteratorA::UnderlyingIterator::kAccessesPerVector;
  static constexpr int kContiguousIterations =
      ThreadMap::Iterations::kContiguous;
  static constexpr int kStridedIterations = ThreadMap::Iterations::kStrided;

  float row_max[kStridedIterations];
  float row_inv_sum[kStridedIterations];
  #if defined(PREFIX_SCORE_EXP_LUT)
  int row_max_q[kStridedIterations];
  #endif

  CUTLASS_DEVICE
  ExpRowSumTransformA() {
    auto thread_offset = ThreadMap::initial_offset(threadIdx.x);
  #pragma unroll
    for (int s = 0; s < kStridedIterations; ++s) {
      int row = blockIdx.x * PVThreadblockShape::kM + thread_offset.strided() +
                s * ThreadMap::Delta::kStrided;
      row_max[s] = row < g_rows ? g_row_max[row] : 0.0f;
      row_inv_sum[s] = row < g_rows ? g_row_inv_sum[row] : 0.0f;
  #if defined(PREFIX_SCORE_EXP_LUT)
      row_max_q[s] = __float2int_rn(row_max[s] / kStoredScoreScale);
  #endif
    }
  }

  CUTLASS_DEVICE
  OutputFragment operator()(InputFragment const& input) {
    OutputFragment output;
    constexpr float kLog2E = 1.4426950408889634f;
    constexpr int kElementsPerAccess = PVIteratorA::AccessType::kElements;
    using OutputAccess =
        cutlass::Array<typename PVSmemIteratorA::Element, kElementsPerAccess>;
    auto const* input_access =
        reinterpret_cast<typename PVIteratorA::AccessType const*>(&input);
    auto* output_access = reinterpret_cast<OutputAccess*>(&output);
  #pragma unroll
    for (int s = 0; s < kStridedIterations; ++s) {
  #pragma unroll
      for (int c = 0; c < kContiguousIterations; ++c) {
  #pragma unroll
        for (int v = 0; v < kAccessesPerVector; ++v) {
          int index = v + kAccessesPerVector * (c + s * kContiguousIterations);
          OutputAccess transformed;
  #pragma unroll
          for (int e = 0; e < kElementsPerAccess; ++e) {
  #if defined(PREFIX_PV_PREEXP)
            float probability =
                static_cast<float>(input_access[index][e]) * kStoredScoreScale;
            transformed[e] = Element(probability * row_inv_sum[s]);
  #elif defined(PREFIX_SCORE_EXP_LUT)
            int difference = int(input_access[index][e]) - row_max_q[s];
            int lut_index =
                difference >= 0 ? 0 : (difference <= -255 ? 255 : -difference);
            float probability =
                __ldg(cutlass::epilogue::threadblock::g_prefix_score_exp_lut +
                      lut_index);
            transformed[e] = Element(probability * row_inv_sum[s]);
  #else
            float value = static_cast<float>(input_access[index][e]) *
    #if defined(PREFIX_QK_DEFER_SCALE_TO_PV)
                          0.0625f;
    #else
                          kStoredScoreScale;
    #endif
            transformed[e] =
                Element(exp2f((value - row_max[s]) * kLog2E) * row_inv_sum[s]);
  #endif
          }
          output_access[index] = transformed;
        }
      }
    }
    return output;
  }
};
#else
template <bool FuseCausalMask>
struct ExpRowSumTransformAImpl {
  using InputFragment = typename PVIteratorA::Fragment;
  using OutputFragment = cutlass::Array<typename PVSmemIteratorA::Element,
                                        InputFragment::kElements>;
  using ThreadMap = typename PVIteratorA::ThreadMap;

  static constexpr int kAccessesPerVector =
      PVIteratorA::UnderlyingIterator::kAccessesPerVector;
  static constexpr int kContiguousIterations =
      ThreadMap::Iterations::kContiguous;
  static constexpr int kStridedIterations = ThreadMap::Iterations::kStrided;
  static constexpr int kElementsPerAccess = PVIteratorA::AccessType::kElements;
  static constexpr int kVectorAccessCount =
      kAccessesPerVector * kContiguousIterations * kStridedIterations;

  CUTLASS_DEVICE
  static int active_rows() {
    if constexpr (FuseCausalMask) {
  #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
      return g_tail_rows;
  #else
      return g_rows;
  #endif
    } else {
      return g_rows;
    }
  }

  CUTLASS_DEVICE
  static float* active_row_sum_out() {
    if constexpr (FuseCausalMask) {
  #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
      return g_tail_row_sum_out;
  #else
      return g_row_sum_out;
  #endif
    } else {
      return g_row_sum_out;
    }
  }

  CUTLASS_DEVICE
  static int64_t active_row_sum_index(int row) {
    if constexpr (FuseCausalMask) {
  #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
      return int64_t(pv_task_index()) * active_rows() + row;
  #else
      return row_sum_output_index(row);
  #endif
    } else {
      return row;
    }
  }

  #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
    #if defined(PREFIX_PV_M128_W64_ROW_SUM)
  float row_max[kContiguousIterations * kElementsPerAccess];
    #else
  float row_max[kElementsPerAccess];
    #endif
    #if defined(PREFIX_PV_SKIP_ROW_SUM)
      // Performance diagnostic: retain the exact exp transform but omit the
      // denominator state and final reduction.
    #elif defined(PREFIX_PV_COMPACT_DISTRIBUTED_ROW_SUM)
  // The M192 producer map has three 64-row vectors per thread.  Reduce the
  // four K lanes for each row immediately, then distribute each vector's 64
  // rows over all 32 lanes (two rows per lane).  Six persistent FP32 values
  // replace the generic 24-value per-thread state.
  float row_sum[kContiguousIterations * 2];
    #elif defined(PREFIX_PV_M128_DISTRIBUTED_ROW_SUM)
  float row_sum[kElementsPerAccess];
    #elif defined(PREFIX_PV_M128_W64_ROW_SUM)
  float row_sum[kContiguousIterations * kElementsPerAccess];
    #elif defined(PREFIX_PV_GENERIC_SHARED_ROW_SUM)
  float row_sum[kVectorAccessCount * kElementsPerAccess];
    #else
  float row_sum[kElementsPerAccess];
    #endif
  #else
  float row_max[kStridedIterations];
  float row_sum[kStridedIterations];
  #endif
  bool valid = true;
  int k_tile = 0;
  #if defined(PREFIX_TORCH_STABLE_ROWS)
  bool score_outlier = false;
  #endif

  CUTLASS_DEVICE
  void set_valid(bool is_valid) { valid = is_valid; }

  CUTLASS_DEVICE
  ExpRowSumTransformAImpl() {
  #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
    #if defined(PREFIX_PV_M128_W64_ROW_SUM)
    auto offset = ThreadMap::initial_offset(threadIdx.x);
      #pragma unroll
    for (int contiguous = 0; contiguous < kContiguousIterations; ++contiguous) {
      #pragma unroll
      for (int element = 0; element < kElementsPerAccess; ++element) {
      #if defined(PREFIX_TORCH_STABLE_ROWS)
        int row = int(blockIdx.x) * PVThreadblockShape::kM +
                  offset.contiguous() +
                  contiguous * ThreadMap::Delta::kContiguous + element;
        if constexpr (FuseCausalMask) {
          row += stable_tail_query_tile() * PREFIX_BATCHED_TAIL_TILE_TOKENS * 6;
          row_max[contiguous * kElementsPerAccess + element] =
              __ldg(g_79t_tail_row_max + row);
        } else {
          row_max[contiguous * kElementsPerAccess + element] =
              __ldg(g_row_max + row);
        }
      #else
        row_max[contiguous * kElementsPerAccess + element] = 0.0f;
      #endif
      }
    }
    #else
      #pragma unroll
    for (int element = 0; element < kElementsPerAccess; ++element) {
      #if defined(PREFIX_TORCH_STABLE_ROWS)
      auto offset = ThreadMap::initial_offset(threadIdx.x);
      int row = int(blockIdx.x) * PVThreadblockShape::kM + offset.contiguous() +
                element;
      if constexpr (FuseCausalMask) {
        row += stable_tail_query_tile() * PREFIX_BATCHED_TAIL_TILE_TOKENS * 6;
        row_max[element] = __ldg(g_79t_tail_row_max + row);
      } else {
        row_max[element] = __ldg(g_row_max + row);
      }
      #else
      row_max[element] = 0.0f;
      #endif
    }
    #endif
    #if defined(PREFIX_PV_SKIP_ROW_SUM)
    #elif defined(PREFIX_PV_COMPACT_DISTRIBUTED_ROW_SUM)
    static_assert(
        kAccessesPerVector == 1 && kContiguousIterations == 3 &&
            kStridedIterations == 1 && kElementsPerAccess == 8 &&
            ThreadMap::Detail::WarpThreadArrangement::kContiguous == 8 &&
            ThreadMap::Detail::WarpThreadArrangement::kStrided == 4,
        "distributed row sum targets the M192 SM70 producer map");
      #pragma unroll
    for (int element = 0; element < kContiguousIterations * 2; ++element) {
      row_sum[element] = 0.0f;
    }
    #elif defined(PREFIX_PV_M128_DISTRIBUTED_ROW_SUM)
    static_assert(
        kAccessesPerVector == 1 && kContiguousIterations == 1 &&
            kStridedIterations == 1 && kElementsPerAccess == 8 &&
            ThreadMap::Detail::WarpThreadArrangement::kContiguous == 8 &&
            ThreadMap::Detail::WarpThreadArrangement::kStrided == 4,
        "distributed row sum targets the M128 SM70 producer map");
      #pragma unroll
    for (int element = 0; element < kElementsPerAccess; ++element) {
      row_sum[element] = 0.0f;
    }
    #elif defined(PREFIX_PV_M128_W64_ROW_SUM)
    static_assert(kAccessesPerVector == 1 && kContiguousIterations == 2 &&
                      kStridedIterations == 1 && kElementsPerAccess == 8,
                  "M128/W64 row sum targets the two-access SM70 map");
      #pragma unroll
    for (int element = 0; element < kContiguousIterations * kElementsPerAccess;
         ++element) {
      row_sum[element] = 0.0f;
    }
    #elif defined(PREFIX_PV_GENERIC_SHARED_ROW_SUM)
      #pragma unroll
    for (int element = 0; element < kVectorAccessCount * kElementsPerAccess;
         ++element) {
      row_sum[element] = 0.0f;
    }
    #else
      #pragma unroll
    for (int element = 0; element < kElementsPerAccess; ++element) {
      row_sum[element] = 0.0f;
    }
    #endif
  #else
    auto thread_offset = ThreadMap::initial_offset(threadIdx.x);
    #pragma unroll
    for (int s = 0; s < kStridedIterations; ++s) {
      int row = blockIdx.x * PVThreadblockShape::kM + thread_offset.strided() +
                s * ThreadMap::Delta::kStrided;
      row_max[s] = row < active_rows() ? g_row_max[row] : 0.0f;
      row_sum[s] = 0.0f;
    }
  #endif
  }

  CUTLASS_DEVICE
  OutputFragment operator()(InputFragment const& input) {
    if (!valid
  #if defined(PREFIX_PV_COMPACT_256_LOADERS)
        || threadIdx.x >= 256
  #endif
    ) {
      OutputFragment output;
      output.clear();
      return output;
    }
    OutputFragment output;
    constexpr float kLog2E = 1.4426950408889634f;
    using OutputAccess =
        cutlass::Array<typename PVSmemIteratorA::Element, kElementsPerAccess>;
    auto const* input_access =
        reinterpret_cast<typename PVIteratorA::AccessType const*>(&input);
    auto* output_access = reinterpret_cast<OutputAccess*>(&output);
  #if defined(PREFIX_BATCHED_TRI_FUSE_CAUSAL_MASK)
    static_assert(kVectorAccessCount == 1 && kContiguousIterations == 1 &&
                      kStridedIterations == 1,
                  "fused causal mask currently targets the M128/K32 PV map");
    auto thread_offset = ThreadMap::initial_offset(threadIdx.x);
    int current_k_tile = 0;
    if constexpr (FuseCausalMask) {
      current_k_tile = k_tile++;
    }
  #endif
  #pragma unroll
    for (int s = 0; s < kStridedIterations; ++s) {
  #pragma unroll
      for (int c = 0; c < kContiguousIterations; ++c) {
  #pragma unroll
        for (int v = 0; v < kAccessesPerVector; ++v) {
          int index = v + kAccessesPerVector * (c + s * kContiguousIterations);
          OutputAccess transformed;
  #if defined(PREFIX_PV_HALF2_TAYLOR_EXP)
          // The accepted raw-cuBLAS route stores an already-scaled FP16 QK
          // score, then applies kStoredScoreScale in this transform.  The
          // frozen shape keeps that final exp input within roughly +/-0.125.
          // Evaluate a degree-3/4/5 Taylor polynomial for exp(x) two rows at
          // a time on the SM70 FP16x2 ALUs.  Causal-tail tiles retain the
          // exact scalar path below because their diagonal mask contains
          // -inf.
          if constexpr (!FuseCausalMask) {
            static_assert(
                kElementsPerAccess % 2 == 0,
                "packed Taylor exp requires even-width vector accesses");
            __half2 scale2 = __float2half2_rn(
                kStoredScoreScale / float(PREFIX_PV_HALF2_RANGE_REDUCTION));
            __half2 one2 = __float2half2_rn(1.0f);
            __half2 linear2 = __float2half2_rn(PREFIX_PV_HALF2_TAYLOR_C1);
            __half2 quadratic2 = __float2half2_rn(PREFIX_PV_HALF2_TAYLOR_C2);
            __half2 cubic2 = __float2half2_rn(PREFIX_PV_HALF2_TAYLOR_C3);
    #if PREFIX_PV_HALF2_TAYLOR_DEGREE >= 4
            __half2 quartic2 = __float2half2_rn(PREFIX_PV_HALF2_TAYLOR_C4);
    #endif
    #if PREFIX_PV_HALF2_TAYLOR_DEGREE >= 5
            __half2 quintic2 = __float2half2_rn(PREFIX_PV_HALF2_TAYLOR_C5);
    #endif
    #pragma unroll
            for (int pair = 0; pair < kElementsPerAccess / 2; ++pair) {
              Element score_low = input_access[index][2 * pair];
              Element score_high = input_access[index][2 * pair + 1];
              __half2 score2 =
                  __halves2half2(__ushort_as_half(score_low.raw()),
                                 __ushort_as_half(score_high.raw()));
              __half2 x2 = __hmul2(score2, scale2);
    #if PREFIX_PV_HALF2_TAYLOR_DEGREE == 3
              __half2 polynomial2 = __hfma2(x2, cubic2, quadratic2);
    #elif PREFIX_PV_HALF2_TAYLOR_DEGREE == 4
              __half2 polynomial2 = __hfma2(x2, quartic2, cubic2);
              polynomial2 = __hfma2(x2, polynomial2, quadratic2);
    #elif PREFIX_PV_HALF2_TAYLOR_DEGREE == 5
              __half2 polynomial2 = __hfma2(x2, quintic2, quartic2);
              polynomial2 = __hfma2(x2, polynomial2, cubic2);
              polynomial2 = __hfma2(x2, polynomial2, quadratic2);
    #else
      #error "PREFIX_PV_HALF2_TAYLOR_DEGREE must be 3, 4, or 5"
    #endif
              polynomial2 = __hfma2(x2, polynomial2, linear2);
              polynomial2 = __hfma2(x2, polynomial2, one2);
    #if PREFIX_PV_HALF2_RANGE_REDUCTION >= 2
              polynomial2 = __hmul2(polynomial2, polynomial2);
    #endif
    #if PREFIX_PV_HALF2_RANGE_REDUCTION >= 4
              polynomial2 = __hmul2(polynomial2, polynomial2);
    #endif
    #if PREFIX_PV_HALF2_RANGE_REDUCTION >= 8
              polynomial2 = __hmul2(polynomial2, polynomial2);
    #endif
    #if PREFIX_PV_HALF2_RANGE_REDUCTION >= 16
              polynomial2 = __hmul2(polynomial2, polynomial2);
    #endif
              transformed[2 * pair] =
                  Element::bitcast(__half_as_ushort(__low2half(polynomial2)));
              transformed[2 * pair + 1] =
                  Element::bitcast(__half_as_ushort(__high2half(polynomial2)));
              if (blockIdx.y == 0) {
                float2 weights = __half22float2(polynomial2);
    #if defined(PREFIX_PV_GENERIC_SHARED_ROW_SUM) || \
        defined(PREFIX_PV_M128_W64_ROW_SUM)
                row_sum[index * kElementsPerAccess + 2 * pair] += weights.x;
                row_sum[index * kElementsPerAccess + 2 * pair + 1] += weights.y;
    #else
                row_sum[2 * pair] += weights.x;
                row_sum[2 * pair + 1] += weights.y;
    #endif
              }
            }
            output_access[index] = transformed;
            continue;
          }
  #endif
  #pragma unroll
          for (int e = 0; e < kElementsPerAccess; ++e) {
            float value = static_cast<float>(input_access[index][e]) *
  #if defined(PREFIX_QK_DEFER_SCALE_TO_PV)
                          (FuseCausalMask ? kStoredScoreScale : 0.0625f);
  #else
                          kStoredScoreScale;
  #endif
  #if defined(PREFIX_BATCHED_TRI_FUSE_CAUSAL_MASK)
            float weight = 0.0f;
            if constexpr (FuseCausalMask) {
              int local_row = int(blockIdx.x) * PVThreadblockShape::kM +
                              thread_offset.contiguous() + e;
              int local_key = current_k_tile * PVThreadblockShape::kK +
                              thread_offset.strided();
              bool causal_masked =
                  local_key >
                  pv_task_index() * PREFIX_BATCHED_TAIL_TILE_TOKENS +
                      local_row / 6;
              weight =
                  causal_masked ? 0.0f : exp2f((value - row_max[e]) * kLog2E);
            } else {
    #if defined(PREFIX_QK_LOG2_SCORES)
              weight = exp2f(value);
    #else
              weight = exp2f((value - row_max[e]) * kLog2E);
    #endif
            }
  #elif defined(PREFIX_PV_FP16_EXP_LUT)
            Element score_value = input_access[index][e];
            uint16_t score_bits = score_value.raw();
            float weight =
                __ldg(&g_prefix_fp16_exp_lut[score_bits >>
                                             kPrefixFP16ExpLUTDropBits]);
  #elif defined(PREFIX_PV_POLY_EXP2)
                float weight = prefix_poly_exp2((value - row_max[
    #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
                                                             e
    #else
                                                             s
    #endif
            ]) * kLog2E);
  #else
    #if defined(PREFIX_QK_LOG2_SCORES)
                float weight = exp2f(value);
    #else
      #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
                float const maximum = row_max[
        #if defined(PREFIX_PV_M128_W64_ROW_SUM)
                    index * kElementsPerAccess + e
        #else
                    e
        #endif
            ];
      #else
                float const maximum = row_max[s];
      #endif
      #if defined(PREFIX_TORCH_STABLE_ROWS)
            if constexpr (!FuseCausalMask) {
              score_outlier |= value - maximum > kStableMaxExpInput;
            }
            float weight = stable_exp(value, maximum);
      #else
            float weight = exp2f((value - maximum) * kLog2E);
      #endif
    #endif
  #endif
            if (blockIdx.y == 0) {
  #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
    #if defined(PREFIX_PV_SKIP_ROW_SUM)
    #elif defined(PREFIX_PV_COMPACT_DISTRIBUTED_ROW_SUM)
              float warp_sum = weight;
              warp_sum += __shfl_xor_sync(0xffffffffu, warp_sum, 8);
              warp_sum += __shfl_xor_sync(0xffffffffu, warp_sum, 16);
              int lane_k_group = (int(threadIdx.x) & 31) >> 3;
              if (lane_k_group == (e >> 1)) {
                row_sum[c * 2 + (e & 1)] += warp_sum;
              }
    #elif defined(PREFIX_PV_M128_DISTRIBUTED_ROW_SUM)
              row_sum[e] += weight;
    #elif defined(PREFIX_PV_M128_W64_ROW_SUM)
              row_sum[index * kElementsPerAccess + e] += weight;
    #elif defined(PREFIX_PV_GENERIC_SHARED_ROW_SUM)
              row_sum[index * kElementsPerAccess + e] += weight;
    #else
              row_sum[e] += weight;
    #endif
  #else
              row_sum[s] += weight;
  #endif
            }
            transformed[e] = Element(weight);
          }
          output_access[index] = transformed;
        }
      }
    }
    return output;
  }

  #if defined(PREFIX_PV_VECTOR_PIPELINED_TRANSFORM)
  CUTLASS_DEVICE
  void transform_access(InputFragment const& input, OutputFragment& output,
                        int index) {
    using OutputAccess =
        cutlass::Array<typename PVSmemIteratorA::Element, kElementsPerAccess>;
    auto const* input_access =
        reinterpret_cast<typename PVIteratorA::AccessType const*>(&input);
    auto* output_access = reinterpret_cast<OutputAccess*>(&output);
    OutputAccess transformed;
    constexpr float kLog2E = 1.4426950408889634f;
    #pragma unroll
    for (int e = 0; e < kElementsPerAccess; ++e) {
      float weight = 0.0f;
      if (valid) {
        float value = static_cast<float>(input_access[index][e]) *
    #if defined(PREFIX_QK_DEFER_SCALE_TO_PV)
                      (FuseCausalMask ? kStoredScoreScale : 0.0625f);
    #else
                      kStoredScoreScale;
    #endif
    #if defined(PREFIX_PV_FP16_EXP_LUT)
        Element score_value = input_access[index][e];
        uint16_t score_bits = score_value.raw();
        weight = __ldg(
            &g_prefix_fp16_exp_lut[score_bits >> kPrefixFP16ExpLUTDropBits]);
    #elif defined(PREFIX_QK_LOG2_SCORES)
        weight = exp2f(value);
    #else
            weight = exp2f((value - row_max[e]) * kLog2E);
    #endif
        if (blockIdx.y == 0) {
          row_sum[e] += weight;
        }
      }
      transformed[e] = Element(weight);
    }
    output_access[index] = transformed;
  }
  #endif

  CUTLASS_DEVICE
  void finalize() {
  #if defined(PREFIX_TORCH_STABLE_ROWS)
    if constexpr (!FuseCausalMask) {
      int outlier = __syncthreads_or(score_outlier);
      if (threadIdx.x == 0 && outlier) {
        atomicExch(g_79t_prefix_outliers + blockIdx.x, 1);
      }
    }
  #endif
  #if defined(PREFIX_PV_SKIP_ROW_SUM)
    return;
  #else
    if (blockIdx.y != 0) {
      return;
    }
    #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
      #if defined(PREFIX_PV_COMPACT_DISTRIBUTED_ROW_SUM)
    constexpr int kProducerWarps = 8;
    __shared__ float warp_row_sum[kProducerWarps][PVThreadblockShape::kM];
    if (threadIdx.x < 256) {
      int warp = int(threadIdx.x) >> 5;
      int lane = int(threadIdx.x) & 31;
      int row_base = (lane & 7) * kElementsPerAccess;
      int owned_element = (lane >> 3) * 2;
        #pragma unroll
      for (int contiguous = 0; contiguous < kContiguousIterations;
           ++contiguous) {
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
          int local_row = row_base +
                          contiguous * ThreadMap::Delta::kContiguous +
                          owned_element + slot;
          warp_row_sum[warp][local_row] = row_sum[contiguous * 2 + slot];
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < PVThreadblockShape::kM) {
      int local_row = int(threadIdx.x);
      float sum = 0.0f;
        #pragma unroll
      for (int warp = 0; warp < kProducerWarps; ++warp) {
        sum += warp_row_sum[warp][local_row];
      }
      int row = int(blockIdx.x) * PVThreadblockShape::kM + local_row;
      if (row < active_rows()) {
        #if defined(PREFIX_PV_FUSED_PREFIX_SUM)
        active_row_sum_out()[active_row_sum_index(row)] += sum;
        #else
        active_row_sum_out()[active_row_sum_index(row)] = sum;
        #endif
      }
    }
      #elif defined(PREFIX_PV_M128_DISTRIBUTED_ROW_SUM)
    static_assert(PVThreadblockShape::kM == 128 &&
                      PVThreadblockShape::kN == 256 &&
                      PVThreadblockShape::kK == 32 &&
                      PVDefaultKernel::kThreadCount == 512,
                  "M128 distributed row sum requires the 16-warp PV tile");
    // The M128 producer map interleaves its two 64-row groups across even
    // and odd warps.  This topology is identical for the prefix and causal
    // tail instantiations; causal masking only changes each lane's weight.
    // Use warp shuffles for both paths instead of issuing 4096 shared-memory
    // atomics per tail CTA.
    constexpr int kProducerWarps = 16;
    constexpr int kRowsPerWarp = 64;
    __shared__ float warp_row_sum[kProducerWarps][kRowsPerWarp];
    int warp = int(threadIdx.x) >> 5;
    int lane = int(threadIdx.x) & 31;
        #pragma unroll
    for (int element = 0; element < kElementsPerAccess; ++element) {
      float sum = row_sum[element];
      sum += __shfl_xor_sync(0xffffffffu, sum, 8);
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      if (lane < 8) {
        warp_row_sum[warp][lane * kElementsPerAccess + element] = sum;
      }
    }
    __syncthreads();
    if (threadIdx.x < PVThreadblockShape::kM) {
      int local_row = int(threadIdx.x);
      int row_group = local_row / kRowsPerWarp;
      int row_in_group = local_row - row_group * kRowsPerWarp;
      float sum = 0.0f;
        #pragma unroll
      for (int warp_group = 0; warp_group < kProducerWarps / 2; ++warp_group) {
        sum += warp_row_sum[row_group + 2 * warp_group][row_in_group];
      }
      int row = int(blockIdx.x) * PVThreadblockShape::kM + local_row;
      if (row < active_rows()) {
        #if defined(PREFIX_PV_FUSED_PREFIX_SUM)
        active_row_sum_out()[active_row_sum_index(row)] += sum;
        #else
        active_row_sum_out()[active_row_sum_index(row)] = sum;
        #endif
      }
    }
      #elif defined(PREFIX_PV_M128_W64_ROW_SUM)
    static_assert(
        PVThreadblockShape::kM == 128 && PVThreadblockShape::kN == 256 &&
            PVThreadblockShape::kK == 32 && PVWarpShape::kM == 64 &&
            PVWarpShape::kN == 64 && PVDefaultKernel::kThreadCount == 256 &&
            kContiguousIterations == 2 && kStridedIterations == 1 &&
            kAccessesPerVector == 1 && kElementsPerAccess == 8,
        "M128/W64 row sum requires the 8-warp two-access PV tile");
    constexpr int kProducerWarps = 8;
    constexpr int kRowsPerAccess = 64;
    __shared__ float warp_row_sum[kProducerWarps][128];
    int warp = int(threadIdx.x) >> 5;
    int lane = int(threadIdx.x) & 31;
        #pragma unroll
    for (int contiguous = 0; contiguous < kContiguousIterations; ++contiguous) {
        #pragma unroll
      for (int element = 0; element < kElementsPerAccess; ++element) {
        float sum = row_sum[contiguous * kElementsPerAccess + element];
        sum += __shfl_xor_sync(0xffffffffu, sum, 8);
        sum += __shfl_xor_sync(0xffffffffu, sum, 16);
        if (lane < 8) {
          int local_row =
              contiguous * kRowsPerAccess + lane * kElementsPerAccess + element;
          warp_row_sum[warp][local_row] = sum;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < PVThreadblockShape::kM) {
      int local_row = int(threadIdx.x);
      float sum = 0.0f;
        #pragma unroll
      for (int producer_warp = 0; producer_warp < kProducerWarps;
           ++producer_warp) {
        sum += warp_row_sum[producer_warp][local_row];
      }
      int row = int(blockIdx.x) * PVThreadblockShape::kM + local_row;
      if (row < active_rows()) {
        #if defined(PREFIX_PV_FUSED_PREFIX_SUM)
        active_row_sum_out()[active_row_sum_index(row)] += sum;
        #else
        active_row_sum_out()[active_row_sum_index(row)] = sum;
        #endif
      }
    }
      #elif defined(PREFIX_PV_GENERIC_SHARED_ROW_SUM)
    static_assert(kAccessesPerVector == 1,
                  "generic row sum currently expects one access per vector");
        #if defined(PREFIX_PV_GENERIC_GLOBAL_ROW_SUM)
    auto thread_offset = ThreadMap::initial_offset(threadIdx.x);
          #pragma unroll
    for (int strided = 0; strided < kStridedIterations; ++strided) {
          #pragma unroll
      for (int contiguous = 0; contiguous < kContiguousIterations;
           ++contiguous) {
          #pragma unroll
        for (int vector = 0; vector < kAccessesPerVector; ++vector) {
          int index =
              vector + kAccessesPerVector *
                           (contiguous + strided * kContiguousIterations);
          #pragma unroll
          for (int element = 0; element < kElementsPerAccess; ++element) {
            int local_row = thread_offset.contiguous() +
                            contiguous * ThreadMap::Delta::kContiguous +
                            element;
            int row = int(blockIdx.x) * PVThreadblockShape::kM + local_row;
            if (local_row < PVThreadblockShape::kM && row < active_rows()) {
              atomicAdd(&active_row_sum_out()[active_row_sum_index(row)],
                        row_sum[index * kElementsPerAccess + element]);
            }
          }
        }
      }
    }
        #else
    __shared__ float shared_row_sum[PVThreadblockShape::kM];
    for (int local_row = int(threadIdx.x); local_row < PVThreadblockShape::kM;
         local_row += int(blockDim.x)) {
      shared_row_sum[local_row] = 0.0f;
    }
    __syncthreads();
    auto thread_offset = ThreadMap::initial_offset(threadIdx.x);
          #pragma unroll
    for (int strided = 0; strided < kStridedIterations; ++strided) {
          #pragma unroll
      for (int contiguous = 0; contiguous < kContiguousIterations;
           ++contiguous) {
          #pragma unroll
        for (int vector = 0; vector < kAccessesPerVector; ++vector) {
          int index =
              vector + kAccessesPerVector *
                           (contiguous + strided * kContiguousIterations);
          #pragma unroll
          for (int element = 0; element < kElementsPerAccess; ++element) {
            int local_row = thread_offset.contiguous() +
                            contiguous * ThreadMap::Delta::kContiguous +
                            element;
            if (local_row < PVThreadblockShape::kM) {
              atomicAdd(&shared_row_sum[local_row],
                        row_sum[index * kElementsPerAccess + element]);
            }
          }
        }
      }
    }
    __syncthreads();
    for (int local_row = int(threadIdx.x); local_row < PVThreadblockShape::kM;
         local_row += int(blockDim.x)) {
      int row = int(blockIdx.x) * PVThreadblockShape::kM + local_row;
      if (row < active_rows()) {
          #if defined(PREFIX_PV_FUSED_PREFIX_SUM)
        active_row_sum_out()[active_row_sum_index(row)] +=
            shared_row_sum[local_row];
          #else
        active_row_sum_out()[active_row_sum_index(row)] =
            shared_row_sum[local_row];
          #endif
      }
    }
        #endif
      #else
    constexpr int kRowsPerWarpGroup = 8 * kElementsPerAccess;
    constexpr int kPVWarpCount = PVDefaultKernel::kThreadCount / 32;
    constexpr int kPVRowGroups = PVThreadblockShape::kM / kRowsPerWarpGroup;
    constexpr int kWarpsPerRowGroup = kPVWarpCount / kPVRowGroups;
    static_assert(PVThreadblockShape::kM % kRowsPerWarpGroup == 0);
    static_assert(kPVWarpCount % kPVRowGroups == 0,
                  "PV row-sum topology must evenly cover 64-row groups");
    __shared__ float warp_row_sum[kPVWarpCount][kRowsPerWarpGroup];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
        #pragma unroll
    for (int element = 0; element < kElementsPerAccess; ++element) {
      float sum = row_sum[element];
      sum += __shfl_xor_sync(0xffffffffu, sum, 8);
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      if (lane < 8) {
        warp_row_sum[warp][lane * kElementsPerAccess + element] = sum;
      }
    }
    __syncthreads();
    if (threadIdx.x < PVThreadblockShape::kM) {
      int local_row = threadIdx.x;
      int row_group = local_row / kRowsPerWarpGroup;
      int row_in_group = local_row - row_group * kRowsPerWarpGroup;
      float sum = 0.0f;
        #pragma unroll
      for (int warp_offset = 0; warp_offset < kWarpsPerRowGroup;
           ++warp_offset) {
        // The SM70 pitch-linear producer map interleaves M row groups in
        // warp-id order: M128 uses even warps for rows 0..63 and odd warps
        // for rows 64..127.  Contiguous warp ranges silently mix the two
        // denominators on the 8-warp FP32-accumulator topology.
        sum +=
            warp_row_sum[row_group + kPVRowGroups * warp_offset][row_in_group];
      }
      int row = blockIdx.x * PVThreadblockShape::kM + local_row;
      if (row < active_rows()) {
        #if defined(PREFIX_PV_FUSED_PREFIX_SUM)
        active_row_sum_out()[active_row_sum_index(row)] += sum;
        #else
        active_row_sum_out()[active_row_sum_index(row)] = sum;
        #endif
      }
    }
      #endif
    #else
    constexpr int kThreadsPerRow =
        ThreadMap::Detail::WarpThreadArrangement::kContiguous;
    auto thread_offset = ThreadMap::initial_offset(threadIdx.x);
    int lane = threadIdx.x & 31;
      #pragma unroll
    for (int s = 0; s < kStridedIterations; ++s) {
      float sum = row_sum[s];
      #pragma unroll
      for (int offset = 1; offset < kThreadsPerRow; offset <<= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
      }
      int row = blockIdx.x * PVThreadblockShape::kM + thread_offset.strided() +
                s * ThreadMap::Delta::kStrided;
      if ((lane % kThreadsPerRow) == 0 && row < g_rows) {
      #if defined(PREFIX_QK_RAW_FIXED_SHIFT)
        // All prefix blocks use the same logit shift, so their probability
        // masses are directly additive. Blocks execute in stream order.
        g_row_sum_out[row_sum_output_index(row)] += sum;
      #else
        g_row_sum_out[row_sum_output_index(row)] = sum;
      #endif
      }
    }
    #endif
  #endif
  }
};
using ExpRowSumTransformA = ExpRowSumTransformAImpl<false>;
  #if defined(PREFIX_BATCHED_TRI_FUSE_CAUSAL_MASK) || \
      defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
using TailExpRowSumTransformA = ExpRowSumTransformAImpl<true>;
  #endif
#endif

#if defined(PREFIX_PV_COMPACT_256_LOADERS)
static_assert(PVThreadblockShape::kM == 192 && PVThreadblockShape::kN == 256 &&
                  PVThreadblockShape::kK == 32 && PVWarpShape::kM == 64 &&
                  PVWarpShape::kN == 64 && PVDefaultKernel::kThreadCount == 384,
              "compact PV loaders target the SM70 M192/N256 12-warp tile");
#endif

#if defined(PREFIX_PV_PREEXP_UNNORMALIZED)
  #if defined(PREFIX_PV_ACCUMULATE_PREEXP_SUM)
struct PVTransformA {
  using InputFragment = typename PVIteratorA::Fragment;
  using OutputFragment = cutlass::Array<typename PVSmemIteratorA::Element,
                                        InputFragment::kElements>;
  using ThreadMap = typename PVIteratorA::ThreadMap;

  static constexpr int kAccessesPerVector =
      PVIteratorA::UnderlyingIterator::kAccessesPerVector;
  static constexpr int kContiguousIterations =
      ThreadMap::Iterations::kContiguous;
  static constexpr int kStridedIterations = ThreadMap::Iterations::kStrided;

  static constexpr int kElementsPerAccess = PVIteratorA::AccessType::kElements;
    #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
  float row_sum[kElementsPerAccess];
    #else
  float row_sum[kStridedIterations];
    #endif
  bool valid = true;

  CUTLASS_DEVICE
  PVTransformA() {
    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row <
    #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
                      kElementsPerAccess
    #else
                      kStridedIterations
    #endif
         ;
         ++row) {
      row_sum[row] = 0.0f;
    }
  }

  CUTLASS_DEVICE
  void set_valid(bool is_valid) { valid = is_valid; }

  CUTLASS_DEVICE
  OutputFragment operator()(InputFragment const& input) {
    OutputFragment output;
    if (!valid) {
      output.clear();
      return output;
    }
    using OutputAccess =
        cutlass::Array<typename PVSmemIteratorA::Element, kElementsPerAccess>;
    auto const* input_access =
        reinterpret_cast<typename PVIteratorA::AccessType const*>(&input);
    auto* output_access = reinterpret_cast<OutputAccess*>(&output);
    CUTLASS_PRAGMA_UNROLL
    for (int strided = 0; strided < kStridedIterations; ++strided) {
      CUTLASS_PRAGMA_UNROLL
      for (int contiguous = 0; contiguous < kContiguousIterations;
           ++contiguous) {
        CUTLASS_PRAGMA_UNROLL
        for (int vector = 0; vector < kAccessesPerVector; ++vector) {
          int index =
              vector + kAccessesPerVector *
                           (contiguous + strided * kContiguousIterations);
          OutputAccess converted;
          CUTLASS_PRAGMA_UNROLL
          for (int element = 0; element < kElementsPerAccess; ++element) {
            float probability =
                float(input_access[index][element]) * kStoredScoreScale;
    #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
            row_sum[element] += probability;
    #else
            row_sum[strided] += probability;
    #endif
            converted[element] = typename PVSmemIteratorA::Element(probability);
          }
          output_access[index] = converted;
        }
      }
    }
    return output;
  }

  CUTLASS_DEVICE
  void finalize() {
    if (blockIdx.y != 0) {
      return;
    }
    #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
    // Column-major A makes the eight elements of each vector eight distinct
    // query rows. Four lanes separated by eight cover K within a warp, and
    // the four warps cover the remaining K stripes. The input-transform
    // object persists for the full GEMM K loop, so this is the complete row
    // mass for the prefix block without a separate global reduction kernel.
    __shared__ float warp_row_sum[4][PVThreadblockShape::kM];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row_group = lane & 7;
    CUTLASS_PRAGMA_UNROLL
    for (int element = 0; element < kElementsPerAccess; ++element) {
      float sum = row_sum[element];
      sum += __shfl_xor_sync(0xffffffffu, sum, 8);
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      if (lane < 8) {
        warp_row_sum[warp][row_group * kElementsPerAccess + element] = sum;
      }
    }
    __syncthreads();
    if (threadIdx.x < PVThreadblockShape::kM) {
      int local_row = threadIdx.x;
      float sum = warp_row_sum[0][local_row] + warp_row_sum[1][local_row] +
                  warp_row_sum[2][local_row] + warp_row_sum[3][local_row];
      int row = blockIdx.x * PVThreadblockShape::kM + local_row;
      if (row < g_rows) {
        g_row_sum_out[row_sum_output_index(row)] = 1.0f / sum;
      }
    }
    #else
    constexpr int kThreadsPerRow =
        ThreadMap::Detail::WarpThreadArrangement::kContiguous;
    auto thread_offset = ThreadMap::initial_offset(threadIdx.x);
    int lane = threadIdx.x & 31;
    CUTLASS_PRAGMA_UNROLL
    for (int strided = 0; strided < kStridedIterations; ++strided) {
      float sum = row_sum[strided];
      CUTLASS_PRAGMA_UNROLL
      for (int offset = 1; offset < kThreadsPerRow; offset <<= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
      }
      int row = blockIdx.x * PVThreadblockShape::kM + thread_offset.strided() +
                strided * ThreadMap::Delta::kStrided;
      if ((lane % kThreadsPerRow) == 0 && row < g_rows) {
        g_row_sum_out[row_sum_output_index(row)] = 1.0f / sum;
      }
    }
    #endif
  }
};
  #elif defined(PREFIX_SCORE_INT8)
struct PVTransformA {
  using InputFragment = typename PVIteratorA::Fragment;
  using OutputFragment = cutlass::Array<typename PVSmemIteratorA::Element,
                                        InputFragment::kElements>;

  CUTLASS_DEVICE
  OutputFragment operator()(InputFragment const& input) const {
    OutputFragment output;
    CUTLASS_PRAGMA_UNROLL
    for (int element = 0; element < InputFragment::kElements; ++element) {
      output[element] = typename PVSmemIteratorA::Element(
          float(input[element]) * kStoredScoreScale);
    }
    return output;
  }
};
  #else
using PVTransformA =
    cutlass::NumericArrayConverter<typename PVSmemIteratorA::Element,
                                   typename PVIteratorA::Element,
                                   PVIteratorA::Fragment::kElements>;
  #endif
#else
using PVTransformA = ExpRowSumTransformA;
#endif
using PVTransformB =
    cutlass::NumericArrayConverter<typename PVSmemIteratorB::Element,
                                   typename PVIteratorB::Element,
                                   PVIteratorB::Fragment::kElements>;
using PVMma = cutlass::gemm::threadblock::MmaPipelined79T<
    typename PVDefaultMma::Shape, PVIteratorA, PVSmemIteratorA, PVIteratorB,
    PVSmemIteratorB, PVAccumulator, PVLayoutC, typename PVDefaultMma::Policy,
    PVTransformA, PVTransformB>;
using PVKernel =
    cutlass::gemm::kernel::Gemm<PVMma, typename PVDefaultKernel::Epilogue,
                                PVSwizzle, false>;
#if defined(PREFIX_TORCH_PREFIX_FP32_OUTPUT)
using PrefixFloatPVOutputOp =
    cutlass::epilogue::thread::LinearCombination<float, 4, PVAccumulator,
                                                 float>;
using PrefixFloatPVDefaultKernel = typename cutlass::gemm::kernel::DefaultGemm<
    Element, PVLayoutA, kPVAlignment, Element, PVLayoutB, kPVAlignment, float,
    PVLayoutC, PVAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, PVThreadblockShape, PVWarpShape, PVInstructionShape,
    PrefixFloatPVOutputOp, PVSwizzle, PV_STAGES, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
using PrefixFloatPVKernel = cutlass::gemm::kernel::Gemm<
    PVMma, typename PrefixFloatPVDefaultKernel::Epilogue, PVSwizzle, false>;
  #if defined(PREFIX_TORCH_STABLE_ROWS)
__global__ void repair_prefix_pv_kernel(
    typename PrefixFloatPVKernel::Params params) {
  if (!g_79t_prefix_outliers[blockIdx.x]) return;
  extern __shared__ int shared_storage_base[];
  auto* shared_storage =
      reinterpret_cast<typename PrefixFloatPVKernel::SharedStorage*>(
          shared_storage_base);
  PrefixFloatPVKernel operation;
  operation(params, *shared_storage);
}
  #endif
#endif
#if defined(PREFIX_BATCHED_TRI_FUSE_CAUSAL_MASK) || \
    defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
using TailPVMma = cutlass::gemm::threadblock::MmaPipelined79T<
    typename PVDefaultMma::Shape, PVIteratorA, PVSmemIteratorA, PVIteratorB,
    PVSmemIteratorB, PVAccumulator, PVLayoutC, typename PVDefaultMma::Policy,
    TailExpRowSumTransformA, PVTransformB>;
using TailPVKernel =
    cutlass::gemm::kernel::Gemm<TailPVMma, typename PVDefaultKernel::Epilogue,
                                PVSwizzle, false>;
#else
using TailPVKernel = PVKernel;
#endif

#if defined(PREFIX_TAIL_WAVE_GATED)
__global__ void prefix_pv_wave_signal_kernel(typename PVKernel::Params params,
                                             unsigned int* completed,
                                             unsigned int* ready,
                                             unsigned int ready_after) {
  extern __shared__ int shared_storage_base[];
  auto* shared_storage =
      reinterpret_cast<typename PVKernel::SharedStorage*>(shared_storage_base);
  PVKernel operation;
  operation(params, *shared_storage);
  if (threadIdx.x == 0) {
    unsigned int completion = atomicAdd(completed, 1u) + 1u;
    if (completion == ready_after) {
      __threadfence_system();
      atomicExch(ready, 1u);
    }
  }
}
#endif

#if defined(PREFIX_BATCHED_TRI_TAIL)
  #if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
__global__ void batched_tri_tail_qk_kernel(
    typename TailQKKernel::Params const* params) {
  extern __shared__ int shared_storage_base[];
  auto* shared_storage =
      reinterpret_cast<typename TailQKKernel::SharedStorage*>(
          shared_storage_base);
  TailQKKernel operation;
  operation(params[blockIdx.z], *shared_storage);
}
  #endif

__global__ void batched_tri_tail_pv_kernel(
    typename TailPVKernel::Params const* params
  #if defined(PREFIX_BATCHED_TRI_FUSE_NORMALIZE)
    ,
    __half* numerator_and_output, float const* tile_sums, float* state_max,
    float* state_sum, int tile_rows
  #endif
) {
  extern __shared__ int shared_storage_base[];
  auto* shared_storage =
      reinterpret_cast<typename TailPVKernel::SharedStorage*>(
          shared_storage_base);
  TailPVKernel operation;
  int task = pv_task_index();
  operation(params[task], *shared_storage);
  #if defined(PREFIX_BATCHED_TRI_FUSE_NORMALIZE)
  __syncthreads();
  float* row_inverse = reinterpret_cast<float*>(shared_storage_base);
  int row_base = int(blockIdx.x) * PVThreadblockShape::kM;
  if (int(threadIdx.x) < PVThreadblockShape::kM) {
    int local_row = row_base + int(threadIdx.x);
    if (local_row < tile_rows) {
      float sum = tile_sums[int64_t(task) * tile_rows + local_row];
      row_inverse[threadIdx.x] = 1.0f / sum;
      int global_row = task * tile_rows + local_row;
      state_max[global_row] = 0.0f;
      state_sum[global_row] = sum;
    }
  }
  __syncthreads();
  constexpr int kPairsPerRow = 256 / 2;
  constexpr int kPairsPerTile = PVThreadblockShape::kM * kPairsPerRow;
  auto* output_half2 = reinterpret_cast<__half2*>(numerator_and_output);
  for (int pair = int(threadIdx.x); pair < kPairsPerTile;
       pair += int(blockDim.x)) {
    int local_row_in_cta = pair / kPairsPerRow;
    int local_row = row_base + local_row_in_cta;
    if (local_row < tile_rows) {
      int global_row = task * tile_rows + local_row;
      int64_t output_pair =
          int64_t(global_row) * kPairsPerRow + pair % kPairsPerRow;
      __half2 scale = __float2half2_rn(row_inverse[local_row_in_cta]);
      output_half2[output_pair] = __hmul2(output_half2[output_pair], scale);
    }
  }
  #endif
}
#endif

void check(cudaError_t result, char const* operation) {
  if (result != cudaSuccess) {
    std::cerr << operation << ": " << cudaGetErrorString(result) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#if defined(PREFIX_TAIL_WAVE_GATED)
void check(CUresult result, char const* operation) {
  if (result != CUDA_SUCCESS) {
    char const* error_name = nullptr;
    char const* error_string = nullptr;
    cuGetErrorName(result, &error_name);
    cuGetErrorString(result, &error_string);
    std::cerr << operation << ": "
              << (error_name == nullptr ? "CUDA_ERROR" : error_name) << " ("
              << (error_string == nullptr ? "unknown" : error_string) << ")\n";
    std::exit(EXIT_FAILURE);
  }
}
#endif

void check(cutlass::Status status, char const* operation) {
  if (status != cutlass::Status::kSuccess) {
    std::cerr << operation << ": CUTLASS status " << int(status) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#if defined(PREFIX_QK_CUBLAS_RAW)
void check(cublasStatus_t status, char const* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << operation << ": cuBLAS status " << int(status) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

struct CublasQKLauncher {
  cublasHandle_t handle;
  Element* query_transposed;
  Element* key_transposed;
  ScoreElement* scores;
  int rows;
  int width;
  int query_stride;
  int key_stride;

  void launch(cudaStream_t stream) const {
    check(cublasSetStream(handle, stream), "set cuBLAS QK stream");
  #if defined(PREFIX_QK_CUBLAS_FP32_ACCUM)
    cublasGemmAlgo_t qk_algorithm = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  #else
    cublasGemmAlgo_t qk_algorithm = CUBLAS_GEMM_ALGO9_TENSOR_OP;
  #endif
    if (char const* runtime_algorithm =
            std::getenv("PREFIX_QK_CUBLAS_ALGO_RUNTIME")) {
      qk_algorithm =
          static_cast<cublasGemmAlgo_t>(std::atoi(runtime_algorithm));
    }
  #if defined(PREFIX_QK_CUBLAS_FP32_ACCUM)
    #if defined(PREFIX_QK_LOG2_SCORES)
    float alpha = 0.0625f * 1.4426950408889634f;
    #elif defined(PREFIX_QK_DEFER_SCALE_TO_PV)
    float alpha = 1.0f;
    #else
    float alpha = 0.0625f;
    #endif
    float beta = 0.0f;
    constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F;
  #else
    #if defined(PREFIX_QK_LOG2_SCORES)
    __half alpha = __float2half(0.0625f * 1.4426950408889634f);
    #elif defined(PREFIX_QK_DEFER_SCALE_TO_PV)
    __half alpha = __float2half(1.0f);
    #else
    __half alpha = __float2half(0.0625f);
    #endif
    __half beta = __float2half(0.0f);
    constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_16F;
  #endif
    check(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, rows, width, 256,
                       &alpha, query_transposed, CUDA_R_16F, query_stride,
                       key_transposed, CUDA_R_16F, key_stride, &beta, scores,
                       CUDA_R_16F, rows, kComputeType, qk_algorithm),
          "launch cuBLAS raw QK");
  }
};
#endif

#if defined(PREFIX_PV_WARP_SPECIALIZED)
struct PVLauncher {
  ScoreElement* scores;
  Element* value;
  PVOutputElement* output;
  int rows;
  int k;
  bool accumulate;

  PVLauncher(ScoreElement* scores_, Element* value_, PVOutputElement* output_,
             int rows_, int k_, bool accumulate_ = false)
      : scores(scores_),
        value(value_),
        output(output_),
        rows(rows_),
        k(k_),
        accumulate(accumulate_) {
    static_assert(std::is_same<ScoreElement, Element>::value);
    static_assert(std::is_same<PVOutputElement, Element>::value);
  }

  void launch(cudaStream_t stream) const {
    warp_specialized_pv_kernel<<<(rows + 31) / 32, 256, 0, stream>>>(
        reinterpret_cast<__half const*>(scores),
        reinterpret_cast<__half const*>(value),
        reinterpret_cast<__half*>(output), rows, k, accumulate);
  }
};
#else
struct PVLauncher {
  typename PVKernel::Params params;
  dim3 grid;
  dim3 block;
  int smem_bytes;

  PVLauncher(ScoreElement* scores, Element* value, PVOutputElement* output,
             int rows, int k, bool accumulate = false) {
    cutlass::gemm::GemmCoord problem(rows, 256, k);
    PVSwizzle swizzle;
    auto tiled_shape =
        swizzle.get_tiled_shape(problem,
                                {PVThreadblockShape::kM, PVThreadblockShape::kN,
                                 PVThreadblockShape::kK},
                                1);
    params = typename PVKernel::Params(
        problem, tiled_shape,
  #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
        {scores, PVLayoutA(rows)},
  #else
        {scores, PVLayoutA(k)},
  #endif
        {value, PVLayoutB(256)}, {output, PVLayoutC(256)},
        {output, PVLayoutC(256)},
        typename PVOutputOp::Params(1.0f, accumulate ? 1.0f : 0.0f), nullptr);
    grid = swizzle.get_grid_shape(tiled_shape);
    block = dim3(PVKernel::kThreadCount, 1, 1);
    smem_bytes = int(sizeof(typename PVKernel::SharedStorage));
    if (smem_bytes >= 48 * 1024) {
      check(cudaFuncSetAttribute(cutlass::Kernel<PVKernel>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_bytes),
            "set PV dynamic shared memory");
  #if defined(PREFIX_TAIL_WAVE_GATED)
      check(cudaFuncSetAttribute(prefix_pv_wave_signal_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_bytes),
            "set signaling PV dynamic shared memory");
  #endif
    }
  }

  void launch(cudaStream_t stream) const {
    cutlass::Kernel<PVKernel><<<grid, block, smem_bytes, stream>>>(params);
    check(cudaGetLastError(), "launch prefix PV");
  }

  #if defined(PREFIX_TAIL_WAVE_GATED)
  void launch_wave_signaled(cudaStream_t stream, unsigned int* completed,
                            unsigned int* ready,
                            unsigned int ready_after) const {
    prefix_pv_wave_signal_kernel<<<grid, block, smem_bytes, stream>>>(
        params, completed, ready, ready_after);
  }
  #endif
};

  #if defined(PREFIX_TORCH_PREFIX_FP32_OUTPUT)
struct PrefixFloatPVLauncher {
  typename PrefixFloatPVKernel::Params params;
  dim3 grid;
  dim3 block;
  int smem_bytes;

  PrefixFloatPVLauncher(ScoreElement* scores, Element* value, float* output,
                        int rows, int k, bool accumulate = false) {
    cutlass::gemm::GemmCoord problem(rows, 256, k);
    PVSwizzle swizzle;
    auto tiled_shape =
        swizzle.get_tiled_shape(problem,
                                {PVThreadblockShape::kM, PVThreadblockShape::kN,
                                 PVThreadblockShape::kK},
                                1);
    params = typename PrefixFloatPVKernel::Params(
        problem, tiled_shape, {scores, PVLayoutA(rows)},
        {value, PVLayoutB(256)}, {output, PVLayoutC(256)},
        {output, PVLayoutC(256)},
        typename PrefixFloatPVOutputOp::Params(1.0f, accumulate ? 1.0f : 0.0f),
        nullptr);
    grid = swizzle.get_grid_shape(tiled_shape);
    block = dim3(PrefixFloatPVKernel::kThreadCount, 1, 1);
    smem_bytes = int(sizeof(typename PrefixFloatPVKernel::SharedStorage));
    if (smem_bytes >= 48 * 1024) {
      check(cudaFuncSetAttribute(cutlass::Kernel<PrefixFloatPVKernel>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_bytes),
            "set FP32-output prefix PV dynamic shared memory");
  #if defined(PREFIX_TORCH_STABLE_ROWS)
      check(cudaFuncSetAttribute(repair_prefix_pv_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_bytes),
            "set repaired prefix PV dynamic shared memory");
  #endif
    }
  }

  void launch(cudaStream_t stream) const {
    cutlass::Kernel<PrefixFloatPVKernel>
        <<<grid, block, smem_bytes, stream>>>(params);
  }
  #if defined(PREFIX_TORCH_STABLE_ROWS)
  void repair(cudaStream_t stream) const {
    repair_prefix_pv_kernel<<<grid, block, smem_bytes, stream>>>(params);
  }
  #endif
};
  #endif

  #if defined(PREFIX_BATCHED_TRI_FUSE_CAUSAL_MASK) || \
      defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
struct TailPVLauncher {
  typename TailPVKernel::Params params;
  dim3 grid;
  dim3 block;
  int smem_bytes;

  TailPVLauncher(ScoreElement* scores, Element* value, PVOutputElement* output,
                 int rows, int k, bool accumulate = false) {
    cutlass::gemm::GemmCoord problem(rows, 256, k);
    PVSwizzle swizzle;
    auto tiled_shape =
        swizzle.get_tiled_shape(problem,
                                {PVThreadblockShape::kM, PVThreadblockShape::kN,
                                 PVThreadblockShape::kK},
                                1);
    params = typename TailPVKernel::Params(
        problem, tiled_shape, {scores, PVLayoutA(rows)},
        {value, PVLayoutB(256)}, {output, PVLayoutC(256)},
        {output, PVLayoutC(256)},
        typename PVOutputOp::Params(1.0f, accumulate ? 1.0f : 0.0f), nullptr);
    grid = swizzle.get_grid_shape(tiled_shape);
    block = dim3(TailPVKernel::kThreadCount, 1, 1);
    smem_bytes = int(sizeof(typename TailPVKernel::SharedStorage));
  }
};
  #else
using TailPVLauncher = PVLauncher;
  #endif
#endif

#if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
struct TailQKLauncher {
  typename TailQKKernel::Params params;
  dim3 grid;
  dim3 block;
  int smem_bytes;

  TailQKLauncher(Element* query_transposed, Element* key_transposed,
                 ScoreElement* scores, int rows, int columns, int query_stride,
                 int key_stride) {
    cutlass::gemm::GemmCoord problem(columns, rows, 256);
    TailQKBatchedZSwizzle swizzle;
    auto tiled_shape = swizzle.get_tiled_shape(
        problem,
        {TailQKThreadblockShape::kM, TailQKThreadblockShape::kN,
         TailQKThreadblockShape::kK},
        1);
    params = typename TailQKKernel::Params(
        problem, tiled_shape,
        {key_transposed, cutlass::layout::ColumnMajor(key_stride)},
        {query_transposed, cutlass::layout::RowMajor(query_stride)},
        {scores, cutlass::layout::RowMajor(rows)},
        {scores, cutlass::layout::RowMajor(rows)},
        typename QKOutputOp::Params(Element(0.0625f), Element(0.0f)), nullptr);
    grid = swizzle.get_grid_shape(tiled_shape);
    block = dim3(TailQKKernel::kThreadCount, 1, 1);
    smem_bytes = int(sizeof(typename TailQKKernel::SharedStorage));
  }
};
#endif

#if defined(PREFIX_QK_DIRECT_PROB)
struct QKDirectLauncher {
  typename QKDirectKernel::Params params;
  dim3 grid;
  dim3 block;
  int smem_bytes;

  QKDirectLauncher(Element* key_transposed, Element* query_transposed,
                   ScoreElement* scores, int rows, int columns, int key_stride,
                   int query_stride) {
    cutlass::gemm::GemmCoord problem(rows, columns, 256);
    QKDirectSwizzle swizzle;
    auto tiled_shape =
        swizzle.get_tiled_shape(problem,
                                {QKThreadblockShape::kM, QKThreadblockShape::kN,
                                 QKThreadblockShape::kK},
                                1);
    params = typename QKDirectKernel::Params(
        problem, tiled_shape, {key_transposed, QKLayoutA(key_stride)},
        {query_transposed, QKLayoutB(query_stride)},
        {scores, cutlass::layout::RowMajor(columns)},
        {scores, cutlass::layout::RowMajor(columns)},
        typename QKDirectOutputOp::Params(0.0625f, 0.0f), nullptr);
    grid = swizzle.get_grid_shape(tiled_shape);
    block = dim3(QKDirectKernel::kThreadCount, 1, 1);
    smem_bytes = int(sizeof(typename QKDirectKernel::SharedStorage));
    if (smem_bytes >= 48 * 1024) {
      check(cudaFuncSetAttribute(cutlass::Kernel<QKDirectKernel>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_bytes),
            "set direct-probability QK dynamic shared memory");
    }
  }

  void launch(cudaStream_t stream) const {
    cutlass::Kernel<QKDirectKernel>
        <<<grid, block, smem_bytes, stream>>>(params);
  }
};
#endif

#if defined(PREFIX_QK_RAW_FIXED_SHIFT)
struct QKRawLauncher {
  typename QKRawKernel::Params params;
  dim3 grid;
  dim3 block;
  int smem_bytes;

  QKRawLauncher(Element* query, Element* key, ScoreElement* scores, int rows,
                int columns) {
    cutlass::gemm::GemmCoord problem(rows, columns, 256);
    QKRawSwizzle swizzle;
    auto tiled_shape =
        swizzle.get_tiled_shape(problem,
                                {QKThreadblockShape::kM, QKThreadblockShape::kN,
                                 QKThreadblockShape::kK},
                                1);
    params = typename QKRawKernel::Params(
        problem, tiled_shape, {query, QKLayoutA(256)}, {key, QKLayoutB(256)},
        {scores, cutlass::layout::RowMajor(columns)},
        {scores, cutlass::layout::RowMajor(columns)},
        typename QKOutputOp::Params(Element(0.0625f), Element(0.0f)), nullptr);
    grid = swizzle.get_grid_shape(tiled_shape);
    block = dim3(QKRawKernel::kThreadCount, 1, 1);
    smem_bytes = int(sizeof(typename QKRawKernel::SharedStorage));
    if (smem_bytes >= 48 * 1024) {
      check(cudaFuncSetAttribute(
  #if defined(PREFIX_QK_MIN_BLOCKS_PER_SM)
                cutlass::PrefixQKBoundedKernel<QKRawKernel>,
  #else
                cutlass::Kernel<QKRawKernel>,
  #endif
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes),
            "set raw QK dynamic shared memory");
    }
  }

  void launch(cudaStream_t stream) const {
  #if defined(PREFIX_QK_MIN_BLOCKS_PER_SM)
    cutlass::PrefixQKBoundedKernel<QKRawKernel>
        <<<grid, block, smem_bytes, stream>>>(params);
  #else
    cutlass::Kernel<QKRawKernel><<<grid, block, smem_bytes, stream>>>(params);
  #endif
  }
};
#endif

__global__ void prepare_weights(float const* block_max, float const* block_sum,
                                float* weights, float* state_max,
                                float* state_sum, int blocks, int rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  float global_max = -CUDART_INF_F;
  for (int block = 0; block < blocks; ++block) {
    global_max = fmaxf(global_max, block_max[block * rows + row]);
  }
  float denominator = 0.0f;
  for (int block = 0; block < blocks; ++block) {
    float scale = exp2f((block_max[block * rows + row] - global_max) *
                        1.4426950408889634f);
#if defined(PREFIX_QK_FULL_STATS)
    float scaled_sum = scale / block_sum[block * rows + row];
    weights[block * rows + row] = scaled_sum;
    denominator += scaled_sum;
#else
    weights[block * rows + row] = scale;
    denominator += scale * block_sum[block * rows + row];
#endif
  }
  float inverse = 1.0f / denominator;
  for (int block = 0; block < blocks; ++block) {
    weights[block * rows + row] *= inverse;
  }
  state_max[row] = global_max;
  state_sum[row] = denominator;
}

__global__ void merge_partials(__half const* partials, float const* weights,
                               __half* output, int blocks, int rows) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kPairsPerRow = 128;
  int total_pairs = rows * kPairsPerRow;
  if (pair >= total_pairs) {
    return;
  }
  int row = pair / kPairsPerRow;
  int column_pair = pair - row * kPairsPerRow;
  float2 accumulator = make_float2(0.0f, 0.0f);
  auto const* partials2 = reinterpret_cast<__half2 const*>(partials);
  auto* output2 = reinterpret_cast<__half2*>(output);
  for (int block = 0; block < blocks; ++block) {
    int index = (block * rows + row) * kPairsPerRow + column_pair;
    float2 value = __half22float2(partials2[index]);
    float weight = weights[block * rows + row];
    accumulator.x = fmaf(weight, value.x, accumulator.x);
    accumulator.y = fmaf(weight, value.y, accumulator.y);
  }
  output2[pair] = __floats2half2_rn(accumulator.x, accumulator.y);
}

__global__ void update_prefix_accumulator(__half const* block_output,
                                          float const* block_max,
                                          float const* block_inv_sum,
                                          float* prefix_accumulator,
                                          float* prefix_max, float* prefix_sum,
                                          int rows, bool initialize) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  if (row >= rows || d >= 256) {
    return;
  }
  __shared__ float scales[3];
  if (d == 0) {
    constexpr float kLog2E = 1.4426950408889634f;
    float next_max = block_max[row];
#if defined(PREFIX_PV_UNNORMALIZED)
  #if defined(PREFIX_PV_PREEXP_UNNORMALIZED)
    float next_mass = 1.0f / block_inv_sum[row];
  #else
    float next_mass = block_inv_sum[row];
  #endif
#else
    float next_mass = 1.0f / block_inv_sum[row];
#endif
    if (initialize) {
      scales[0] = 0.0f;
#if defined(PREFIX_PV_UNNORMALIZED)
      scales[1] = 1.0f;
#else
      scales[1] = next_mass;
#endif
      scales[2] = next_mass;
      prefix_max[row] = next_max;
    } else {
      float old_max = prefix_max[row];
      float global_max = fmaxf(old_max, next_max);
      float old_scale = exp2f((old_max - global_max) * kLog2E);
      float next_scale = exp2f((next_max - global_max) * kLog2E);
      scales[0] = old_scale;
#if defined(PREFIX_PV_UNNORMALIZED)
      scales[1] = next_scale;
#else
      scales[1] = next_mass * next_scale;
#endif
      scales[2] = prefix_sum[row] * old_scale + next_mass * next_scale;
      prefix_max[row] = global_max;
    }
    prefix_sum[row] = scales[2];
  }
  __syncthreads();
  int64_t element = int64_t(row) * 256 + d;
  float block_value = float(block_output[element]);
  float old_value = initialize ? 0.0f : prefix_accumulator[element];
  prefix_accumulator[element] =
      fmaf(old_value, scales[0], block_value * scales[1]);
}

__global__ void prepare_prefix_update(float const* block_max,
                                      float const* block_inv_sum,
                                      float* prefix_max, float* prefix_sum,
                                      float* old_scales, float* block_scales,
                                      int rows, bool initialize) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  constexpr float kLog2E = 1.4426950408889634f;
#if defined(PREFIX_FIXED_STATE_NO_RESET) && \
    (defined(PREFIX_QK_CUBLAS_RAW) || defined(PREFIX_QK_RAW_FIXED_SHIFT))
  float next_max = 0.0f;
#else
  float next_max = block_max[row];
#endif
#if defined(PREFIX_PV_UNNORMALIZED)
  #if defined(PREFIX_PV_PREEXP_UNNORMALIZED)
  float next_mass = 1.0f / block_inv_sum[row];
  #else
  float next_mass = block_inv_sum[row];
  #endif
#else
  float next_mass = 1.0f / block_inv_sum[row];
#endif
  if (initialize) {
    old_scales[row] = 0.0f;
#if defined(PREFIX_PV_UNNORMALIZED)
    block_scales[row] = 1.0f;
#else
    block_scales[row] = next_mass;
#endif
    prefix_max[row] = next_max;
    prefix_sum[row] = next_mass;
    return;
  }
  float old_max = prefix_max[row];
  float global_max = fmaxf(old_max, next_max);
  float old_scale = exp2f((old_max - global_max) * kLog2E);
#if defined(PREFIX_PV_UNNORMALIZED)
  float block_scale = exp2f((next_max - global_max) * kLog2E);
#else
  float block_scale = next_mass * exp2f((next_max - global_max) * kLog2E);
#endif
  old_scales[row] = old_scale;
  block_scales[row] = block_scale;
  prefix_max[row] = global_max;
#if defined(PREFIX_PV_UNNORMALIZED)
  prefix_sum[row] = prefix_sum[row] * old_scale + next_mass * block_scale;
#else
  prefix_sum[row] = prefix_sum[row] * old_scale + block_scale;
#endif
}

__global__ void apply_prefix_update_half2(__half const* block_output,
                                          float* prefix_accumulator,
                                          float const* old_scales,
                                          float const* block_scales, int rows,
                                          bool initialize) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kPairsPerRow = 128;
  if (pair >= rows * kPairsPerRow) {
    return;
  }
  int row = pair / kPairsPerRow;
  auto const* block2 = reinterpret_cast<__half2 const*>(block_output);
  auto* accumulator2 = reinterpret_cast<float2*>(prefix_accumulator);
  float2 block_value = __half22float2(block2[pair]);
  float2 old_value = initialize ? make_float2(0.0f, 0.0f) : accumulator2[pair];
  float old_scale = old_scales[row];
  float block_scale = block_scales[row];
  accumulator2[pair] =
      make_float2(fmaf(old_value.x, old_scale, block_value.x * block_scale),
                  fmaf(old_value.y, old_scale, block_value.y * block_scale));
}

__global__ void finalize_prefix_accumulator(float const* prefix_accumulator,
                                            float const* prefix_sum,
                                            __half* output, int rows) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kPairsPerRow = 128;
  if (pair >= rows * kPairsPerRow) {
    return;
  }
  int row = pair / kPairsPerRow;
  int64_t first = int64_t(pair) * 2;
  float inverse = 1.0f / prefix_sum[row];
  output[first] = __float2half_rn(prefix_accumulator[first] * inverse);
  output[first + 1] = __float2half_rn(prefix_accumulator[first + 1] * inverse);
}

__global__ void merge_prefix_accumulator_tail(
    float const* prefix_accumulator, float const* prefix_max,
    float const* prefix_sum, __half const* tail_output, float const* tail_max,
    float const* tail_sum, __half* output, int rows) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  __shared__ float masses[3];
  if (d == 0) {
    constexpr float kLog2E = 1.4426950408889634f;
#if defined(PREFIX_PV_FUSED_PREFIX_SUM)
    float prefix_peak = 0.0f;
#else
    float prefix_peak = prefix_max[row];
#endif
    float global_max = fmaxf(prefix_peak, tail_max[row]);
    masses[0] = exp2f((prefix_peak - global_max) * kLog2E);
    masses[1] = tail_sum[row] * exp2f((tail_max[row] - global_max) * kLog2E);
    masses[2] = 1.0f / (prefix_sum[row] * masses[0] + masses[1]);
  }
  __syncthreads();
  int64_t element = int64_t(row) * 256 + d;
  float numerator = prefix_accumulator[element] * masses[0] +
                    float(tail_output[element]) * masses[1];
  output[element] = __float2half_rn(numerator * masses[2]);
}

#if defined(PREFIX_PV_DIRECT_FP16_ACCUMULATE)
__global__ void merge_prefix_half_accumulator_tail(
    __half const* prefix_accumulator, float const* prefix_max,
    float const* prefix_sum, __half const* tail_output, float const* tail_max,
    float const* tail_sum, __half* output, int rows) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  __shared__ float masses[3];
  if (d == 0) {
    constexpr float kLog2E = 1.4426950408889634f;
  #if defined(PREFIX_PV_FUSED_PREFIX_SUM)
    float prefix_peak = 0.0f;
  #else
    float prefix_peak = prefix_max[row];
  #endif
    float global_max = fmaxf(prefix_peak, tail_max[row]);
    masses[0] = exp2f((prefix_peak - global_max) * kLog2E);
    masses[1] = tail_sum[row] * exp2f((tail_max[row] - global_max) * kLog2E);
    masses[2] = 1.0f / (prefix_sum[row] * masses[0] + masses[1]);
  }
  __syncthreads();
  int64_t element = int64_t(row) * 256 + d;
  float numerator = float(prefix_accumulator[element]) * masses[0] +
                    float(tail_output[element]) * masses[1];
  output[element] = __float2half_rn(numerator * masses[2]);
}
#endif

__global__ void merge_prefix_tail(__half const* prefix_output,
                                  float const* prefix_max,
                                  float const* prefix_sum,
                                  __half const* tail_output,
                                  float const* tail_max, float const* tail_sum,
                                  __half* output, int rows) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  __shared__ float masses[3];
  if (d == 0) {
    float global_max = fmaxf(prefix_max[row], tail_max[row]);
    float prefix_scale =
        exp2f((prefix_max[row] - global_max) * 1.4426950408889634f);
    float tail_scale =
        exp2f((tail_max[row] - global_max) * 1.4426950408889634f);
    masses[0] = prefix_sum[row] * prefix_scale;
    masses[1] = tail_sum[row] * tail_scale;
    masses[2] = 1.0f / (masses[0] + masses[1]);
  }
  __syncthreads();
  int64_t element = int64_t(row) * 256 + d;
  float numerator = float(prefix_output[element]) * masses[0] +
                    float(tail_output[element]) * masses[1];
  output[element] = __float2half_rn(numerator * masses[2]);
}

using DenseTailRaw = cudaError_t (*)(const void*, const void*, const void*,
                                     float*, float*, void*, int, int, int, int,
                                     float, cudaStream_t);

__global__ void fill_pattern(__half* data, size_t elements, uint32_t seed) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  uint32_t value = uint32_t(index) ^ seed;
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  float uniform = float(value & 0xffffu) * (2.0f / 65535.0f) - 1.0f;
  data[index] = __float2half_rn(uniform);
}

#if defined(PREFIX_QK_ASYNC_EXP_PIPELINE)
__global__ void exp2_scores_inplace(__half2* scores, size_t pairs) {
  size_t pair = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) {
    return;
  }
  float2 logits = __half22float2(scores[pair]);
  scores[pair] = __floats2half2_rn(exp2f(logits.x), exp2f(logits.y));
}
#endif

#if defined(PREFIX_MATERIALIZED_SLICED_TAIL) || \
    defined(PREFIX_CUBLAS_SLICED_TAIL) || defined(PREFIX_BATCHED_TRI_TAIL)
__global__ void mask_sliced_tail_probability(__half* scores, int slice_start,
                                             int slice_tokens, int rows,
                                             int heads) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elements = int64_t(slice_tokens) * rows;
  if (index >= elements) {
    return;
  }
  int key_in_slice = int(index / rows);
  int query_row = int(index - int64_t(key_in_slice) * rows);
  int key_position = slice_start + key_in_slice;
  int query_position = slice_start + query_row / heads;
  if (key_position > query_position) {
  #if defined(PREFIX_CUBLAS_SLICED_TAIL)
    scores[int64_t(key_position) * rows + query_row] =
        __float2half(-CUDART_INF_F);
  #else
    scores[int64_t(key_position) * rows + query_row] = __float2half(0.0f);
  #endif
  }
}

__global__ void finalize_sliced_tail(float const* numerator,
                                     float const* inverse_sum, __half* output,
                                     float* state_max, float* state_sum,
                                     int output_row_start, int rows) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  if (row >= rows || d >= 256) {
    return;
  }
  float inverse = inverse_sum[row];
  int64_t local_element = int64_t(row) * 256 + d;
  int64_t output_element = int64_t(output_row_start + row) * 256 + d;
  output[output_element] = __float2half_rn(numerator[local_element] * inverse);
  if (d == 0) {
    state_max[output_row_start + row] = 0.0f;
    state_sum[output_row_start + row] = 1.0f / inverse;
  }
}
#endif

#if defined(PREFIX_CUBLAS_SLICED_TAIL)
__global__ void finalize_cublas_sliced_tail(__half* numerator_and_output,
                                            float const* row_sum,
                                            float* state_max, float* state_sum,
                                            int output_row_start, int rows) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  if (row >= rows || d >= 256) {
    return;
  }
  float sum = row_sum[row];
  int64_t element = int64_t(output_row_start + row) * 256 + d;
  float numerator = __half2float(numerator_and_output[element]);
  numerator_and_output[element] = __float2half_rn(numerator / sum);
  if (d == 0) {
    // The raw QK/PV path uses one common zero logit shift, so all masses are
    // directly mergeable with the prefix fixed-shift state.
    state_max[output_row_start + row] = 0.0f;
    state_sum[output_row_start + row] = sum;
  }
}
#endif

#if defined(PREFIX_BATCHED_TRI_TAIL)
__global__ void mask_batched_tri_tail_diagonal(__half* scores, int tile_rows,
                                               int tile_tokens,
                                               int pv_pad_tokens) {
  int query_tile = int(blockIdx.y);
  int64_t query_score_offset =
      int64_t(tile_rows) *
      (int64_t(tile_tokens) * query_tile * (query_tile + 1) / 2 +
       int64_t(pv_pad_tokens) * query_tile);
  int64_t diagonal_offset =
      query_score_offset + int64_t(query_tile) * tile_rows * tile_tokens;
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elements = int64_t(tile_rows) * tile_tokens;
  if (index < elements) {
    int key_in_tile = int(index / tile_rows);
    int query_row = int(index - int64_t(key_in_tile) * tile_rows);
    if (key_in_tile > query_row / 6) {
      scores[diagonal_offset + index] = __float2half(-CUDART_INF_F);
    }
  }
  int64_t pad_elements = int64_t(tile_rows) * pv_pad_tokens;
  if (index < pad_elements) {
    int64_t pad_offset =
        query_score_offset + int64_t(query_tile + 1) * tile_rows * tile_tokens;
    scores[pad_offset + index] = __float2half(-CUDART_INF_F);
  }
}

__global__ void finalize_batched_tri_tail(__half const* partial_numerators,
                                          float const* partial_sums,
                                          __half* output, float* state_max,
                                          float* state_sum, int tile_rows,
                                          int rows) {
  int global_row = int(blockIdx.x);
  int d = int(threadIdx.x);
  if (global_row >= rows || d >= 256) {
    return;
  }
  int query_tile = global_row / tile_rows;
  int local_row = global_row - query_tile * tile_rows;
  int first_task = query_tile * (query_tile + 1) / 2;
  float numerator = 0.0f;
  float sum = 0.0f;
  for (int key_tile = 0; key_tile <= query_tile; ++key_tile) {
    int task = first_task + key_tile;
    int64_t partial_row = int64_t(task) * tile_rows + local_row;
    numerator += __half2float(partial_numerators[partial_row * 256 + d]);
    if (d == 0) {
      sum += partial_sums[partial_row];
    }
  }
  __shared__ float shared_sum;
  if (d == 0) {
    shared_sum = sum;
    state_max[global_row] = 0.0f;
    state_sum[global_row] = sum;
  }
  __syncthreads();
  output[int64_t(global_row) * 256 + d] =
      __float2half_rn(numerator / shared_sum);
}

  #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
__global__ void finalize_grouped_batched_tri_tail(
    __half const* partial_numerators, float const* partial_sums, __half* output,
    float* state_max, float* state_sum, int tile_rows, int rows) {
  int global_row = int(blockIdx.x);
  int d = int(threadIdx.x);
  if (global_row >= rows || d >= 256) {
    return;
  }
  constexpr int kGroupTiles = PREFIX_TAIL_FINE_PV_GROUP_TILES;
  int query_tile = global_row / tile_rows;
  int local_row = global_row - query_tile * tile_rows;
  int full_groups = query_tile / kGroupTiles;
  int residual_queries = query_tile % kGroupTiles;
  int first_task = kGroupTiles * full_groups * (full_groups + 1) / 2 +
                   residual_queries * (full_groups + 1);
  int tasks = (query_tile + 1 + kGroupTiles - 1) / kGroupTiles;
  float numerator = 0.0f;
  float sum = 0.0f;
  for (int task_offset = 0; task_offset < tasks; ++task_offset) {
    int task = first_task + task_offset;
    int64_t partial_row = int64_t(task) * tile_rows + local_row;
    numerator += __half2float(partial_numerators[partial_row * 256 + d]);
    if (d == 0) {
      sum += partial_sums[partial_row];
    }
  }
  __shared__ float shared_sum;
  if (d == 0) {
    shared_sum = sum;
    state_max[global_row] = 0.0f;
    state_sum[global_row] = sum;
  }
  __syncthreads();
  output[int64_t(global_row) * 256 + d] =
      __float2half_rn(numerator / shared_sum);
}

    #if defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE)
__global__ void finalize_round_major_tail_state(float const* partial_sums,
                                                float* state_max,
                                                float* state_sum, int tile_rows,
                                                int rows) {
  int global_row = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (global_row >= rows) {
    return;
  }
  constexpr int kGroupTiles = PREFIX_TAIL_FINE_PV_GROUP_TILES;
  constexpr int kQueryTiles =
      PREFIX_TORCH_QUERY_TOKENS / PREFIX_BATCHED_TAIL_TILE_TOKENS;
  int query_tile = global_row / tile_rows;
  int local_row = global_row - query_tile * tile_rows;
  int rounds = (query_tile + 1 + kGroupTiles - 1) / kGroupTiles;
  float sum = 0.0f;
  for (int round = 0; round < rounds; ++round) {
    int first_task =
        round * kQueryTiles - kGroupTiles * round * (round - 1) / 2;
    int task = first_task + query_tile - round * kGroupTiles;
    sum += partial_sums[int64_t(task) * tile_rows + local_row];
  }
      #if !defined(PREFIX_TORCH_STABLE_ROWS)
  state_max[global_row] = 0.0f;
      #endif
  state_sum[global_row] = sum;
}

__global__ void merge_prefix_direct_round_major_tail(
    __half const* prefix_numerator, float const* prefix_sum,
    __half const* tail_numerator_or_output, float const* tail_max,
    float const* tail_sum, __half* output, int rows, int repaired_rows) {
  int row = int(blockIdx.x);
  int pair = int(threadIdx.x);
  if (row >= rows || pair >= 128) {
    return;
  }
  __shared__ float coefficients[3];
  if (pair == 0) {
    float prefix_scale = 1.0f;
    float tail_mass = tail_sum[row];
    float tail_scale = 1.0f;
    if (row < repaired_rows) {
      constexpr float kLog2E = 1.4426950408889634f;
      float tail_peak = tail_max[row];
      float global_peak = fmaxf(0.0f, tail_peak);
      prefix_scale = exp2f(-global_peak * kLog2E);
      tail_mass *= exp2f((tail_peak - global_peak) * kLog2E);
      tail_scale = tail_mass;
    }
    coefficients[0] = prefix_scale;
    coefficients[1] = tail_scale;
    coefficients[2] = 1.0f / (prefix_sum[row] * prefix_scale + tail_mass);
  }
  __syncthreads();
  int64_t element_pair = int64_t(row) * 128 + pair;
  auto const* prefix2 = reinterpret_cast<__half2 const*>(prefix_numerator);
  auto const* tail2 =
      reinterpret_cast<__half2 const*>(tail_numerator_or_output);
  auto* output2 = reinterpret_cast<__half2*>(output);
  float2 prefix_value = __half22float2(prefix2[element_pair]);
  float2 tail_value = __half22float2(tail2[element_pair]);
  float numerator0 =
      fmaf(prefix_value.x, coefficients[0], tail_value.x * coefficients[1]);
  float numerator1 =
      fmaf(prefix_value.y, coefficients[0], tail_value.y * coefficients[1]);
  output2[element_pair] = __floats2half2_rn(numerator0 * coefficients[2],
                                            numerator1 * coefficients[2]);
}

      #if defined(PREFIX_TORCH_PREFIX_FP32_OUTPUT)
__global__ void merge_float_prefix_direct_round_major_tail(
    float const* prefix_numerator, float const* prefix_sum,
    __half const* tail_numerator_or_output, float const* tail_max,
    float const* tail_sum, __half* output, int rows, int repaired_rows) {
  int row = int(blockIdx.x);
  int pair = int(threadIdx.x);
  if (row >= rows || pair >= 128) {
    return;
  }
  __shared__ float coefficients[3];
  if (pair == 0) {
    float prefix_scale = 1.0f;
    float tail_mass = tail_sum[row];
    float tail_scale = 1.0f;
    if (row < repaired_rows) {
      constexpr float kLog2E = 1.4426950408889634f;
      float tail_peak = tail_max[row];
      float global_peak = fmaxf(0.0f, tail_peak);
      prefix_scale = exp2f(-global_peak * kLog2E);
      tail_mass *= exp2f((tail_peak - global_peak) * kLog2E);
      tail_scale = tail_mass;
    }
    coefficients[0] = prefix_scale;
    coefficients[1] = tail_scale;
    coefficients[2] = 1.0f / (prefix_sum[row] * prefix_scale + tail_mass);
  }
  __syncthreads();
  int64_t element_pair = int64_t(row) * 128 + pair;
  auto const* prefix2 = reinterpret_cast<float2 const*>(prefix_numerator);
  auto const* tail2 =
      reinterpret_cast<__half2 const*>(tail_numerator_or_output);
  auto* output2 = reinterpret_cast<__half2*>(output);
  float2 prefix_value = prefix2[element_pair];
  float2 tail_value = __half22float2(tail2[element_pair]);
  float numerator0 =
      fmaf(prefix_value.x, coefficients[0], tail_value.x * coefficients[1]);
  float numerator1 =
      fmaf(prefix_value.y, coefficients[0], tail_value.y * coefficients[1]);
  output2[element_pair] = __floats2half2_rn(numerator0 * coefficients[2],
                                            numerator1 * coefficients[2]);
}
      #endif

      #if defined(PREFIX_TAIL_FUSE_STATE_IN_MERGE)
__global__ void merge_prefix_direct_round_major_tail_fused_state(
    __half const* prefix_numerator, float const* prefix_sum,
    __half const* tail_numerator_or_output, float const* repaired_tail_max,
    float const* repaired_tail_sum, float const* round_partial_sums,
    __half* output, int rows, int repaired_rows, int tile_rows) {
  int row = int(blockIdx.x);
  int pair = int(threadIdx.x);
  if (row >= rows || pair >= 128) {
    return;
  }
  __shared__ float coefficients[3];
  if (pair == 0) {
    float prefix_scale = 1.0f;
    float tail_mass = 0.0f;
    float tail_scale = 1.0f;
    if (row < repaired_rows) {
      constexpr float kLog2E = 1.4426950408889634f;
      float tail_peak = repaired_tail_max[row];
      tail_mass = repaired_tail_sum[row];
      float global_peak = fmaxf(0.0f, tail_peak);
      prefix_scale = exp2f(-global_peak * kLog2E);
      tail_mass *= exp2f((tail_peak - global_peak) * kLog2E);
      tail_scale = tail_mass;
    } else {
      constexpr int kGroupTiles = PREFIX_TAIL_FINE_PV_GROUP_TILES;
      constexpr int kQueryTiles =
          PREFIX_TORCH_QUERY_TOKENS / PREFIX_BATCHED_TAIL_TILE_TOKENS;
      constexpr int kMaxRounds = (kQueryTiles + kGroupTiles - 1) / kGroupTiles;
      int query_tile = row / tile_rows;
      int local_row = row - query_tile * tile_rows;
      int rounds = (query_tile + 1 + kGroupTiles - 1) / kGroupTiles;
        #pragma unroll
      for (int round = 0; round < kMaxRounds; ++round) {
        if (round < rounds) {
          int first_task =
              round * kQueryTiles - kGroupTiles * round * (round - 1) / 2;
          int task = first_task + query_tile - round * kGroupTiles;
          tail_mass +=
              round_partial_sums[int64_t(task) * tile_rows + local_row];
        }
      }
    }
    coefficients[0] = prefix_scale;
    coefficients[1] = tail_scale;
    coefficients[2] = 1.0f / (prefix_sum[row] * prefix_scale + tail_mass);
  }
  __syncthreads();
  int64_t element_pair = int64_t(row) * 128 + pair;
  auto const* prefix2 = reinterpret_cast<__half2 const*>(prefix_numerator);
  auto const* tail2 =
      reinterpret_cast<__half2 const*>(tail_numerator_or_output);
  auto* output2 = reinterpret_cast<__half2*>(output);
  float2 prefix_value = __half22float2(prefix2[element_pair]);
  float2 tail_value = __half22float2(tail2[element_pair]);
  float numerator0 =
      fmaf(prefix_value.x, coefficients[0], tail_value.x * coefficients[1]);
  float numerator1 =
      fmaf(prefix_value.y, coefficients[0], tail_value.y * coefficients[1]);
  output2[element_pair] = __floats2half2_rn(numerator0 * coefficients[2],
                                            numerator1 * coefficients[2]);
}
      #endif
    #endif
  #endif

  #if defined(PREFIX_BATCHED_TRI_FUSED_PV)
__global__ void finalize_batched_tri_fused_pv(__half* numerator_and_output,
                                              float const* tile_sums,
                                              float* state_max,
                                              float* state_sum, int tile_rows,
                                              int rows) {
  int global_row = int(blockIdx.x);
  int d = int(threadIdx.x);
  if (global_row >= rows || d >= 256) {
    return;
  }
  int query_tile = global_row / tile_rows;
  int local_row = global_row - query_tile * tile_rows;
  float sum = tile_sums[int64_t(query_tile) * tile_rows + local_row];
  int64_t element = int64_t(global_row) * 256 + d;
  numerator_and_output[element] =
      __float2half_rn(__half2float(numerator_and_output[element]) / sum);
  if (d == 0) {
    state_max[global_row] = 0.0f;
    state_sum[global_row] = sum;
  }
}
  #endif
#endif

#if defined(PREFIX_QK_PRETRANSPOSE_INPUTS)
// Convert an input [rows, columns] row-major matrix into a physically
// contiguous [columns, rows] matrix. Q and the fully visible K prefix are
// each transposed once per attention call, then reused by every QK block.
__global__ void transpose_half_32x32(__half const* input, __half* output,
                                     int rows, int columns) {
  __shared__ __half tile[32][33];
  int input_column = int(blockIdx.x) * 32 + int(threadIdx.x);
  int input_row = int(blockIdx.y) * 32 + int(threadIdx.y);
  #pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    int row = input_row + offset;
    if (row < rows && input_column < columns) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[int64_t(row) * columns + input_column];
    }
  }
  __syncthreads();
  int output_column = int(blockIdx.y) * 32 + int(threadIdx.x);
  int output_row = int(blockIdx.x) * 32 + int(threadIdx.y);
  #pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    int row = output_row + offset;
    if (row < columns && output_column < rows) {
      output[int64_t(row) * rows + output_column] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

  #if defined(PREFIX_QK_BLOCKED_TRANSPOSE)
// Transpose K while packing each runtime KV block into its own tight
// [head_dim, block_n] column-major panel. This preserves the coalesced tiled
// transpose but gives cuBLAS lda=block_n instead of lda=the full 120K prefix.
__global__ void transpose_half_blocked_32x32(__half const* input,
                                             __half* output, int rows,
                                             int columns, int block_n) {
  __shared__ __half tile[32][33];
  int input_column = int(blockIdx.x) * 32 + int(threadIdx.x);
  int input_row = int(blockIdx.y) * 32 + int(threadIdx.y);
    #pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    int row = input_row + offset;
    if (row < rows && input_column < columns) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[int64_t(row) * columns + input_column];
    }
  }
  __syncthreads();
  int output_dimension = int(blockIdx.x) * 32 + int(threadIdx.y);
  int input_global_row = int(blockIdx.y) * 32 + int(threadIdx.x);
    #pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    int dimension = output_dimension + offset;
    if (input_global_row < rows && dimension < columns) {
      int panel = input_global_row / block_n;
      int within_panel = input_global_row - panel * block_n;
      output[(int64_t(panel) * columns + dimension) * block_n + within_panel] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}
  #endif
#endif

#if defined(PREFIX_DEBUG_QK_STATE)
__global__ void count_invalid_scores(ScoreElement const* scores,
                                     size_t elements,
                                     unsigned long long* invalid,
                                     unsigned int* max_bits) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    float value = float(scores[index]);
    if (!isfinite(value)) {
      atomicAdd(invalid, 1ull);
    } else {
      atomicMax(max_bits, __float_as_uint(fabsf(value)));
    }
  }
}
#endif

#if defined(PREFIX_DEBUG_PREFIX_STATE)
__global__ void count_invalid_prefix_state(float const* values, size_t elements,
                                           unsigned long long* invalid) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements && !isfinite(values[index])) {
    atomicAdd(invalid, 1ull);
  }
}
#endif

struct BlockOperators {
  int width;
#if defined(PREFIX_QK_CUBLAS_RAW)
  std::unique_ptr<CublasQKLauncher> qk;
#elif defined(PREFIX_QK_DIRECT_PROB)
  std::unique_ptr<QKDirectLauncher> qk;
#elif defined(PREFIX_QK_RAW_FIXED_SHIFT)
  std::unique_ptr<QKRawLauncher> qk;
#else
  QKGemm qk;
#endif
  std::unique_ptr<PVLauncher> pv;
#if defined(PREFIX_QK_SUPERBLOCK_PV_MICROTILES)
  std::vector<std::unique_ptr<PVLauncher>> pv_microtiles;
#endif
#if defined(PREFIX_QK_PV_PIPELINE) || defined(PREFIX_QK_ASYNC_EXP_PIPELINE) || \
    defined(PREFIX_PHASE_BATCHED_MATERIALIZATION) ||                           \
    defined(PREFIX_GROUPED_QK_PV) || defined(PREFIX_TAIL_PV_OVERLAP)
  float* qk_norm;
  float* qk_sum;
  int buffer;
#endif
};

#if defined(PREFIX_MATERIALIZED_SLICED_TAIL)
struct TailSliceOperators {
  int query_token_start;
  int query_row_start;
  int rows;
  int width;
  QKGemm qk;
  std::unique_ptr<PVLauncher> pv;
};
#endif

#if defined(PREFIX_CUBLAS_SLICED_TAIL)
struct CublasTailSliceOperators {
  int query_token_start;
  int query_row_start;
  int rows;
  int width;
  std::unique_ptr<CublasQKLauncher> qk;
  std::unique_ptr<PVLauncher> pv;
};
#endif

}  // namespace

#if !defined(PREFIX_TORCH_EXTENSION)
extern "C" cudaError_t onecat_sm70_d256_dense_state_raw(
    const void*, const void*, const void*, float*, float*, void*, int, int, int,
    int, float, cudaStream_t);
int main(int argc, char** argv) {
  #if defined(PREFIX_DEBUG_THREADMAP)
  using QKOutputIterator =
      typename QKGemm::DefaultGemmKernel::Epilogue::OutputTileIterator;
  using QKThreadMap = typename QKOutputIterator::ThreadMap;
  using QKEpilogue = typename QKGemm::Epilogue;
  using QKAccumulatorIterator =
      typename QKEpilogue::AccumulatorFragmentIterator;
  using QKSharedLoadIterator = typename QKEpilogue::SharedLoadIterator;
  std::vector<int> coverage(size_t(QK_TB_M) * QK_TB_N, 0);
  for (int thread = 0; thread < QKGemm::DefaultGemmKernel::kThreadCount;
       ++thread) {
    cutlass::MatrixCoord initial = QKThreadMap::initial_offset(thread);
    int iterator_row = initial.row();
    int row_state = 0;
    int group_state = 0;
    int cluster_state = 0;
    for (int step = 0; step < QKOutputIterator::kIterations; ++step) {
      for (int iteration = 0; iteration < QKThreadMap::Iterations::kCount;
           ++iteration) {
        cutlass::MatrixCoord delta = QKThreadMap::iteration_offset(iteration);
        int row = iterator_row + delta.row();
        int column = initial.column() + delta.column();
        for (int element = 0; element < QKThreadMap::kElementsPerAccess;
             ++element) {
          int current_column = column + element;
          if (row >= 0 && row < QK_TB_M && current_column >= 0 &&
              current_column < QK_TB_N) {
            ++coverage[size_t(row) * QK_TB_N + current_column];
          }
        }
      }

      ++row_state;
      iterator_row += QKThreadMap::Shape::kRow;
      if (row_state == QKThreadMap::Count::kRow) {
        row_state = 0;
        ++group_state;
        iterator_row += (QKThreadMap::Shape::kGroup - 1) *
                        QKThreadMap::Shape::kRow * QKThreadMap::Count::kRow;
        if (group_state == QKThreadMap::Count::kGroup) {
          group_state = 0;
          ++cluster_state;
          iterator_row += QKThreadMap::Count::kGroup *
                          QKThreadMap::Shape::kGroup *
                          QKThreadMap::Count::kRow * QKThreadMap::Shape::kRow;
          if (cluster_state == QKThreadMap::Count::kCluster) {
            cluster_state = 0;
            iterator_row +=
                QKThreadMap::Shape::kGroup * QKThreadMap::Shape::kRow *
                QKThreadMap::Shape::kCluster * QKThreadMap::Shape::kTile;
          }
        }
      }
    }
  }
  size_t missing = 0;
  size_t duplicate = 0;
  int minimum = coverage.empty() ? 0 : coverage[0];
  int maximum = minimum;
  for (int count : coverage) {
    missing += count == 0;
    duplicate += count > 1;
    minimum = std::min(minimum, count);
    maximum = std::max(maximum, count);
  }
  std::cout << "threadmap threads=" << QKGemm::DefaultGemmKernel::kThreadCount
            << " visitor_iterations=" << QKGemm::EpilogueVisitor::kIterations
            << " iterator_iterations=" << QKOutputIterator::kIterations
            << " iterations=" << QKThreadMap::Iterations::kCount
            << " access=" << QKThreadMap::kElementsPerAccess
            << " shape=" << QKThreadMap::Shape::kRow << "x"
            << QKThreadMap::Shape::kColumn
            << " count=" << QKThreadMap::Count::kRow << "x"
            << QKThreadMap::Count::kGroup << "x" << QKThreadMap::Count::kCluster
            << "x" << QKThreadMap::Count::kTile << " accumulator_elements="
            << QKEpilogue::AccumulatorTile::kElements
            << " accumulator_fragment_elements="
            << QKAccumulatorIterator::Fragment::kElements
            << " accumulator_iterator_iterations="
            << QKAccumulatorIterator::kIterations
            << " shared_load_fragment_elements="
            << QKSharedLoadIterator::Fragment::kElements
            << " shared_load_columns="
            << QKSharedLoadIterator::ThreadMap::Iterations::kColumn
            << " visitor_fragment_count="
            << (QKEpilogue::AccumulatorTile::kElements /
                (QKGemm::EpilogueVisitor::kIterations *
                 QKEpilogue::AccumulatorAccessType::kElements))
            << " missing=" << missing << " duplicate=" << duplicate
            << " min=" << minimum << " max=" << maximum << "\n";

  using QKMma = typename QKGemm::DefaultGemmKernel::Mma;
  using QKBThreadMap = typename QKMma::IteratorB::ThreadMap;
  std::vector<int> b_coverage(
      size_t(QKThreadblockShape::kK) * QKThreadblockShape::kN, 0);
  for (int thread = 0; thread < QKBThreadMap::kThreads; ++thread) {
    auto initial = QKBThreadMap::initial_offset(thread);
    for (int strided = 0; strided < QKBThreadMap::Iterations::kStrided;
         ++strided) {
      for (int contiguous = 0;
           contiguous < QKBThreadMap::Iterations::kContiguous; ++contiguous) {
        int row = initial.strided() + strided * QKBThreadMap::Delta::kStrided;
        int column = initial.contiguous() +
                     contiguous * QKBThreadMap::Delta::kContiguous;
        for (int element = 0; element < QKBThreadMap::kElementsPerAccess;
             ++element) {
          int current_column = column + element;
          if (row >= 0 && row < QKThreadblockShape::kN && current_column >= 0 &&
              current_column < QKThreadblockShape::kK) {
            ++b_coverage[size_t(row) * QKThreadblockShape::kK + current_column];
          }
        }
      }
    }
  }
  size_t b_missing = 0;
  size_t b_duplicate = 0;
  int b_minimum = b_coverage.empty() ? 0 : b_coverage[0];
  int b_maximum = b_minimum;
  for (int count : b_coverage) {
    b_missing += count == 0;
    b_duplicate += count > 1;
    b_minimum = std::min(b_minimum, count);
    b_maximum = std::max(b_maximum, count);
  }
  std::cout << "b_threadmap threads=" << QKBThreadMap::kThreads
            << " iterations=" << QKBThreadMap::Iterations::kContiguous << "x"
            << QKBThreadMap::Iterations::kStrided
            << " delta=" << QKBThreadMap::Delta::kContiguous << "x"
            << QKBThreadMap::Delta::kStrided
            << " access=" << QKBThreadMap::kElementsPerAccess
            << " missing=" << b_missing << " duplicate=" << b_duplicate
            << " min=" << b_minimum << " max=" << b_maximum << "\n";
  using PVThreadMap = typename PVIteratorA::ThreadMap;
  std::cout << "pv_a_threadmap threads=" << PVThreadMap::kThreads
            << " iterations=" << PVThreadMap::Iterations::kContiguous << "x"
            << PVThreadMap::Iterations::kStrided
            << " delta=" << PVThreadMap::Delta::kContiguous << "x"
            << PVThreadMap::Delta::kStrided << " arrangement="
            << PVThreadMap::Detail::WarpThreadArrangement::kContiguous << "x"
            << PVThreadMap::Detail::WarpThreadArrangement::kStrided
            << " access=" << PVThreadMap::kElementsPerAccess
            << " accesses_per_vector="
            << PVIteratorA::UnderlyingIterator::kAccessesPerVector << "\n";
  return EXIT_SUCCESS;
  #endif
  int block_n = argc > 1 ? std::atoi(argv[1]) : 2048;
  int iterations = argc > 2 ? std::atoi(argv[2]) : 5;
  // Exact-tail entry is linked into this binary; no DSO argument.
  constexpr int kRows = 48000;
  constexpr int kHeadDim = 256;
  constexpr int kPrefix = 120000;
  constexpr int kTail = 8000;
  constexpr int kTotalKV = kPrefix + kTail;
  #if defined(PREFIX_BATCHED_TRI_TAIL)
  constexpr int kBatchedTailTileTokens = PREFIX_BATCHED_TAIL_TILE_TOKENS;
  cublasGemmAlgo_t batched_tail_qk_algo = PREFIX_BATCHED_TAIL_QK_ALGO;
  if (char const* runtime_algo =
          std::getenv("PREFIX_BATCHED_TAIL_QK_ALGO_RUNTIME")) {
    batched_tail_qk_algo =
        static_cast<cublasGemmAlgo_t>(std::atoi(runtime_algo));
  }
    #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  int tail_overlap_mode = 1;
  if (char const* runtime_mode = std::getenv("PREFIX_TAIL_OVERLAP_MODE")) {
    tail_overlap_mode = std::atoi(runtime_mode);
  }
  if (tail_overlap_mode < 0 || tail_overlap_mode > 3) {
    std::cerr << "PREFIX_TAIL_OVERLAP_MODE must be 0, 1, 2, or 3\n";
    return EXIT_FAILURE;
  }
  int tail_overlap_start_block = 0;
  if (char const* runtime_start =
          std::getenv("PREFIX_TAIL_OVERLAP_START_BLOCK")) {
    tail_overlap_start_block = std::atoi(runtime_start);
  }
      #if defined(PREFIX_TAIL_WAVE_GATED)
  int tail_wave_ready_after = 240;
  if (char const* runtime_ready_after =
          std::getenv("PREFIX_TAIL_WAVE_READY_AFTER")) {
    tail_wave_ready_after = std::atoi(runtime_ready_after);
  }
  bool tail_wave_continuous_qk = false;
  if (char const* runtime_continuous_qk =
          std::getenv("PREFIX_TAIL_WAVE_CONTINUOUS_QK")) {
    tail_wave_continuous_qk = std::atoi(runtime_continuous_qk) != 0;
  }
      #endif
    #endif
  constexpr int kBatchedTailPVPadTokens = PREFIX_BATCHED_TRI_PV_PAD_TOKENS;
  constexpr int kBatchedTailResidualTokens =
      PREFIX_BATCHED_TAIL_RESIDUAL_TOKENS;
  constexpr int kBatchedTailCoveredTokens = kTail - kBatchedTailResidualTokens;
  constexpr int kBatchedTailTiles =
      kBatchedTailCoveredTokens / kBatchedTailTileTokens;
  constexpr int kBatchedTailTileRows = kBatchedTailTileTokens * 6;
  constexpr int kBatchedTailCoveredRows = kBatchedTailCoveredTokens * 6;
  constexpr int kBatchedTailTasks =
      kBatchedTailTiles * (kBatchedTailTiles + 1) / 2;
    #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
  constexpr int kFinePVGroupTiles = PREFIX_TAIL_FINE_PV_GROUP_TILES;
  constexpr int kFinePVFullGroups = kBatchedTailTiles / kFinePVGroupTiles;
  constexpr int kFinePVResidualQueries = kBatchedTailTiles % kFinePVGroupTiles;
  constexpr int kFinePVTasks =
      kFinePVGroupTiles * kFinePVFullGroups * (kFinePVFullGroups + 1) / 2 +
      kFinePVResidualQueries * (kFinePVFullGroups + 1);
  static_assert(kFinePVGroupTiles > 0);
    #endif
  static_assert(kBatchedTailResidualTokens >= 0 &&
                kBatchedTailResidualTokens < kTail);
  static_assert(kBatchedTailCoveredTokens % kBatchedTailTileTokens == 0);
  static_assert(
      kBatchedTailResidualTokens == 0 || kBatchedTailResidualTokens % 64 == 0,
      "exact residual tail requires a 64-token query multiple");
  static_assert(kBatchedTailTileTokens % 8 == 0,
                "batched-tail K tile pointers must be 128-bit aligned");
  static_assert(kBatchedTailTileRows % 8 == 0,
                "batched-tail score leading dimension must be 128-bit aligned");
  static_assert(
      kBatchedTailPVPadTokens >= 0 && kBatchedTailPVPadTokens % 8 == 0,
      "batched-tail PV padding must preserve 128-bit alignment");
    #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  static_assert(
      kBatchedTailTiles == 25 && kBatchedTailResidualTokens == 0 &&
          (PVThreadblockShape::kM == 64 || PVThreadblockShape::kM == 96 ||
           PVThreadblockShape::kM == 128) &&
          (PVThreadblockShape::kN == 128 || PVThreadblockShape::kN == 256) &&
          (PVDefaultKernel::kThreadCount == 128 ||
           PVDefaultKernel::kThreadCount == 256 ||
           PVDefaultKernel::kThreadCount == 384 ||
           PVDefaultKernel::kThreadCount == 512),
      "idle-SM overlap requires a validated Q8000 N128/N256 route");
    #endif
    #if defined(PREFIX_BATCHED_TRI_REPAIR_FIRST_TILE)
  constexpr int kBatchedTailRepairTokens = PREFIX_BATCHED_TRI_REPAIR_TOKENS;
  static_assert(kBatchedTailRepairTokens > 0 &&
                    kBatchedTailRepairTokens <= kBatchedTailTileTokens &&
                    kBatchedTailRepairTokens % 64 == 0,
                "exact first-tile repair must be a positive 64-token multiple "
                "within the first tile");
    #endif
  #endif
  #if defined(PREFIX_FULL_ENDPOINT)
  constexpr int kStoredKV = kTotalKV
    #if defined(PREFIX_BATCHED_TRI_TAIL)
                            + kBatchedTailPVPadTokens
    #endif
      ;
  #else
  constexpr int kStoredKV = kPrefix;
  #endif
  int blocks = (kPrefix + block_n - 1) / block_n;
  #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  if (tail_overlap_start_block < 0 || tail_overlap_start_block >= blocks) {
    std::cerr << "PREFIX_TAIL_OVERLAP_START_BLOCK is outside prefix blocks\n";
    return EXIT_FAILURE;
  }
    #if defined(PREFIX_TAIL_WAVE_GATED)
  if (tail_overlap_start_block != 0 || blocks != 15) {
    std::cerr
        << "wave-gated tail requires start block 0 and 15 prefix blocks\n";
    return EXIT_FAILURE;
  }
  if (tail_wave_ready_after <= 0 || tail_wave_ready_after >= 375) {
    std::cerr << "PREFIX_TAIL_WAVE_READY_AFTER must be in [1, 374]\n";
    return EXIT_FAILURE;
  }
    #endif
  #endif

  Element* query = nullptr;
  Element* key = nullptr;
  Element* value = nullptr;
  #if defined(PREFIX_QK_PRETRANSPOSE_INPUTS)
  Element* query_transposed = nullptr;
  Element* key_transposed = nullptr;
  #endif
  ScoreElement* scores = nullptr;
  Element* partials = nullptr;
  Element* output = nullptr;
  float* qk_norm = nullptr;
  float* qk_sum = nullptr;
  float* prefix_accumulator = nullptr;
  float* prefix_max = nullptr;
  float* prefix_sum = nullptr;
  float* old_scales = nullptr;
  float* block_scales = nullptr;
  #if defined(PREFIX_FULL_ENDPOINT)
  Element* tail_output = nullptr;
  Element* final_output = nullptr;
  Element* exact_output = nullptr;
  float* tail_max = nullptr;
  float* tail_sum = nullptr;
    #if defined(PREFIX_BATCHED_TRI_TAIL)
  Element* batched_tail_partials = nullptr;
  TailPVKernel::Params* batched_tail_pv_params = nullptr;
      #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  ScoreElement* tail_overlap_scores = nullptr;
      #endif
      #if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
  TailQKKernel::Params* batched_tail_qk_params = nullptr;
  dim3 batched_tail_qk_grid;
  dim3 batched_tail_qk_block;
  int batched_tail_qk_smem_bytes = 0;
      #endif
  Element** batched_tail_q_ptrs = nullptr;
  Element** batched_tail_k_ptrs = nullptr;
  ScoreElement** batched_tail_score_ptrs = nullptr;
  dim3 batched_tail_pv_grid;
  dim3 batched_tail_pv_block;
  int batched_tail_pv_smem_bytes = 0;
    #endif
    #if defined(PREFIX_MATERIALIZED_SLICED_TAIL)
  float* tail_numerator = nullptr;
    #endif
  #endif

  #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
  int qk_tiles_n =
      (kRows + QKThreadblockShape::kN - 1) / QKThreadblockShape::kN;
  #else
  int qk_tiles_n =
      (block_n + QKThreadblockShape::kN - 1) / QKThreadblockShape::kN;
  #endif
  size_t partial_elements = size_t(kRows) * kHeadDim;
  #if defined(PREFIX_GROUPED_QK_PV)
  constexpr int kPipelineBuffers = PREFIX_GROUPED_QK_PV_BLOCKS;
  #elif defined(PREFIX_QK_PV_PIPELINE) || \
      defined(PREFIX_QK_ASYNC_EXP_PIPELINE) || defined(PREFIX_TAIL_PV_OVERLAP)
  constexpr int kPipelineBuffers = 2;
  #else
  constexpr int kPipelineBuffers = 1;
  #endif
  int score_buffer_count =
  #if defined(PREFIX_PHASE_BATCHED_MATERIALIZATION)
      blocks;
  #else
      kPipelineBuffers;
  #endif
  size_t score_buffer_elements = size_t(kRows) * block_n;
  #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
  size_t qk_stat_buffer_elements = size_t(qk_tiles_n) * block_n;
  #else
  size_t qk_stat_buffer_elements = size_t(qk_tiles_n) * kRows;
  #endif
  check(cudaMalloc(&query, size_t(kRows) * kHeadDim * sizeof(Element)),
        "allocate Q");
  check(cudaMalloc(&key, size_t(kStoredKV) * kHeadDim * sizeof(Element)),
        "allocate K");
  check(cudaMalloc(&value, size_t(kStoredKV) * kHeadDim * sizeof(Element)),
        "allocate V");
  #if defined(PREFIX_QK_PRETRANSPOSE_INPUTS)
  constexpr int kTransposedKRows =
    #if defined(PREFIX_MATERIALIZED_SLICED_TAIL) || \
        defined(PREFIX_CUBLAS_SLICED_TAIL) || defined(PREFIX_BATCHED_TRI_TAIL)
      kStoredKV;
    #else
      kPrefix;
    #endif
  check(
      cudaMalloc(&query_transposed, size_t(kRows) * kHeadDim * sizeof(Element)),
      "allocate transposed Q");
  size_t transposed_k_elements = size_t(kTransposedKRows) * kHeadDim;
    #if defined(PREFIX_QK_BLOCKED_TRANSPOSE)
  transposed_k_elements = size_t(blocks) * block_n * kHeadDim;
    #endif
  check(cudaMalloc(&key_transposed, transposed_k_elements * sizeof(Element)),
        "allocate transposed prefix K");
  #endif
  check(cudaMalloc(&scores, size_t(score_buffer_count) * score_buffer_elements *
                                sizeof(ScoreElement)),
        "allocate scores");
  check(cudaMalloc(&partials, partial_elements * sizeof(Element)),
        "allocate partials");
  check(cudaMalloc(&output, size_t(kRows) * kHeadDim * sizeof(Element)),
        "allocate output");
  check(cudaMalloc(&qk_norm, size_t(score_buffer_count) *
                                 qk_stat_buffer_elements * sizeof(float)),
        "allocate QK max");
  check(cudaMalloc(&qk_sum, size_t(score_buffer_count) *
                                qk_stat_buffer_elements * sizeof(float)),
        "allocate QK sum");
  check(
      cudaMalloc(&prefix_accumulator, size_t(kRows) * kHeadDim * sizeof(float)),
      "allocate prefix accumulator");
  check(cudaMalloc(&prefix_max, size_t(kRows) * sizeof(float)),
        "allocate prefix max");
  check(cudaMalloc(&prefix_sum, size_t(kRows) * sizeof(float)),
        "allocate prefix sum");
  check(cudaMalloc(&old_scales, size_t(kRows) * sizeof(float)),
        "allocate old scales");
  check(cudaMalloc(&block_scales, size_t(kRows) * sizeof(float)),
        "allocate block scales");
  #if defined(PREFIX_FULL_ENDPOINT)
  check(cudaMalloc(&tail_output, size_t(kRows) * kHeadDim * sizeof(Element)),
        "allocate tail output");
  check(cudaMalloc(&final_output, size_t(kRows) * kHeadDim * sizeof(Element)),
        "allocate final output");
  check(cudaMalloc(&exact_output, size_t(kRows) * kHeadDim * sizeof(Element)),
        "allocate exact output");
  check(cudaMalloc(&tail_max, size_t(kRows) * sizeof(float)),
        "allocate tail max");
  check(cudaMalloc(&tail_sum, size_t(kRows) * sizeof(float)),
        "allocate tail sum");
    #if defined(PREFIX_BATCHED_TRI_TAIL)
      #if !defined(PREFIX_BATCHED_TRI_FUSED_PV) && \
          !defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE)
  size_t batched_tail_partial_elements = size_t(
        #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
                                             kFinePVTasks
        #else
                                             kBatchedTailTasks
        #endif
                                             ) *
                                         kBatchedTailTileRows * kHeadDim;
  check(cudaMalloc(&batched_tail_partials,
                   batched_tail_partial_elements * sizeof(Element)),
        "allocate batched-tail partial numerators");
      #endif
  check(cudaMalloc(&batched_tail_pv_params, size_t(
      #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
                                                kFinePVTasks
      #else
                                                kBatchedTailTasks
      #endif
                                                ) *
                                                sizeof(TailPVKernel::Params)),
        "allocate batched-tail PV params");
      #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  check(cudaMalloc(&tail_overlap_scores,
                   (size_t(kBatchedTailTasks) * kBatchedTailTileRows *
                        kBatchedTailTileTokens +
                    size_t(kBatchedTailTiles) * kBatchedTailTileRows *
                        kBatchedTailPVPadTokens) *
                       sizeof(ScoreElement)),
        "allocate idle-SM full triangular-tail scores");
      #endif
      #if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
  check(cudaMalloc(&batched_tail_qk_params,
                   size_t(kBatchedTailTasks) * sizeof(TailQKKernel::Params)),
        "allocate batched-tail QK params");
      #endif
  check(cudaMalloc(&batched_tail_q_ptrs,
                   size_t(kBatchedTailTasks) * sizeof(Element*)),
        "allocate batched-tail Q pointer array");
  check(cudaMalloc(&batched_tail_k_ptrs,
                   size_t(kBatchedTailTasks) * sizeof(Element*)),
        "allocate batched-tail K pointer array");
  check(cudaMalloc(&batched_tail_score_ptrs,
                   size_t(kBatchedTailTasks) * sizeof(ScoreElement*)),
        "allocate batched-tail score pointer array");
    #endif
    #if defined(PREFIX_MATERIALIZED_SLICED_TAIL)
  constexpr int kTailSliceTokens = 500;
  constexpr int kTailSliceRows = kTailSliceTokens * 6;
  static_assert(kTail % kTailSliceTokens == 0);
  check(cudaMalloc(&tail_numerator,
                   size_t(kTailSliceRows) * kHeadDim * sizeof(float)),
        "allocate sliced-tail numerator");
    #endif
  #endif
  size_t query_elements = size_t(kRows) * kHeadDim;
  size_t kv_elements = size_t(kStoredKV) * kHeadDim;
  fill_pattern<<<(query_elements + 255) / 256, 256>>>(
      reinterpret_cast<__half*>(query), query_elements, 0x12345678u);
  fill_pattern<<<(kv_elements + 255) / 256, 256>>>(
      reinterpret_cast<__half*>(key), kv_elements, 0x9abcdef0u);
  fill_pattern<<<(kv_elements + 255) / 256, 256>>>(
      reinterpret_cast<__half*>(value), kv_elements, 0x31415926u);
  check(cudaGetLastError(), "initialize deterministic inputs");
  #if defined(PREFIX_PV_FP16_EXP_LUT)
  initialize_prefix_fp16_exp_lut<<<(kPrefixFP16ExpLUTSize + 255) / 256,
                                   256>>>();
  check(cudaGetLastError(), "initialize full-domain FP16 exp LUT");
  #endif

  #if defined(PREFIX_FULL_ENDPOINT)
  // Source-complete build: the exact tail entry is linked into this binary
  // instead of being dlopen'd from a prebuilt extension.
  DenseTailRaw tail_raw = &onecat_sm70_d256_dense_state_raw;
  #endif

  #if defined(PREFIX_QK_CUBLAS_RAW)
  cublasHandle_t qk_cublas = nullptr;
  check(cublasCreate(&qk_cublas), "create cuBLAS QK handle");
  check(cublasSetMathMode(qk_cublas, CUBLAS_TENSOR_OP_MATH),
        "enable cuBLAS tensor-op math");
    #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  cublasHandle_t tail_overlap_cublas = nullptr;
  check(cublasCreate(&tail_overlap_cublas),
        "create idle-SM tail cuBLAS handle");
  check(cublasSetMathMode(tail_overlap_cublas, CUBLAS_TENSOR_OP_MATH),
        "enable idle-SM tail tensor-op math");
    #endif
  #endif

  std::vector<BlockOperators> operators;
  operators.reserve(blocks);
  for (int block = 0; block < blocks; ++block) {
    int begin = block * block_n;
    int width = std::min(block_n, kPrefix - begin);
    BlockOperators operation;
    operation.width = width;
  #if defined(PREFIX_PHASE_BATCHED_MATERIALIZATION)
    operation.buffer = block;
    ScoreElement* block_scores =
        scores + size_t(operation.buffer) * score_buffer_elements;
    operation.qk_norm =
        qk_norm + size_t(operation.buffer) * qk_stat_buffer_elements;
    operation.qk_sum =
        qk_sum + size_t(operation.buffer) * qk_stat_buffer_elements;
  #elif defined(PREFIX_QK_PV_PIPELINE) ||      \
      defined(PREFIX_QK_ASYNC_EXP_PIPELINE) || \
      defined(PREFIX_GROUPED_QK_PV) || defined(PREFIX_TAIL_PV_OVERLAP)
    operation.buffer = block % kPipelineBuffers;
    ScoreElement* block_scores =
        scores + size_t(operation.buffer) * score_buffer_elements;
    operation.qk_norm =
        qk_norm + size_t(operation.buffer) * qk_stat_buffer_elements;
    operation.qk_sum =
        qk_sum + size_t(operation.buffer) * qk_stat_buffer_elements;
  #else
    ScoreElement* block_scores = scores;
    float* block_qk_norm = qk_norm;
    float* block_qk_sum = qk_sum;
  #endif
  #if defined(PREFIX_QK_CUBLAS_RAW)
    operation.qk = std::make_unique<CublasQKLauncher>(
        CublasQKLauncher{qk_cublas, query_transposed,
    #if defined(PREFIX_QK_BLOCKED_TRANSPOSE)
                         key_transposed + size_t(block) * block_n * kHeadDim,
    #else
                         key_transposed + begin,
    #endif
                         block_scores, kRows, width, kRows,
    #if defined(PREFIX_QK_BLOCKED_TRANSPOSE)
                         block_n});
    #else
                         kTransposedKRows});
    #endif
  #elif defined(PREFIX_QK_DIRECT_PROB)
    operation.qk = std::make_unique<QKDirectLauncher>(
        key_transposed + begin, query_transposed, block_scores, width, kRows,
        kTransposedKRows, kRows);
  #elif defined(PREFIX_QK_RAW_FIXED_SHIFT)
    operation.qk = std::make_unique<QKRawLauncher>(
        query, key + size_t(begin) * kHeadDim, block_scores, kRows, width);
  #else
    typename QKGemm::Arguments arguments(
    #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
        {width, kRows, kHeadDim}, 1,
      #if defined(PREFIX_QK_PRETRANSPOSE_INPUTS)
        {key_transposed + begin, QKLayoutA(kTransposedKRows)},
        {query_transposed, QKLayoutB(kRows)},
      #else
        {key + size_t(begin) * kHeadDim, QKLayoutA(kHeadDim)},
        {query, QKLayoutB(kHeadDim)},
      #endif
        {block_scores, typename QKGemm::LayoutC(kRows)},
        {block_scores, typename QKGemm::LayoutC(kRows)},
    #else
        {kRows, width, kHeadDim}, 1, {query, QKLayoutA(kHeadDim)},
        {key + size_t(begin) * kHeadDim, QKLayoutB(kHeadDim)},
        {block_scores, typename QKGemm::LayoutC(width)},
        {block_scores, typename QKGemm::LayoutC(width)},
    #endif
    #if defined(PREFIX_SCORE_INT8)
        {kScoreOutputAlpha, 0.0f},
    #else
        {Element(0.0625f), Element(0.0f)},
    #endif
    #if defined(PREFIX_QK_PV_PIPELINE)
      #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
        {operation.qk_norm, typename QKGemm::LayoutN(width)},
        {operation.qk_sum, typename QKGemm::LayoutS(width)},
      #else
        {operation.qk_norm, typename QKGemm::LayoutN(kRows)},
        {operation.qk_sum, typename QKGemm::LayoutS(kRows)},
      #endif
    #else
      #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
        {block_qk_norm, typename QKGemm::LayoutN(width)},
        {block_qk_sum, typename QKGemm::LayoutS(width)},
      #else
        {block_qk_norm, typename QKGemm::LayoutN(kRows)},
        {block_qk_sum, typename QKGemm::LayoutS(kRows)},
      #endif
    #endif
    #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
        {block_scores, typename QKGemm::LayoutSoft(kRows)});
    #else
        {block_scores, typename QKGemm::LayoutSoft(width)});
    #endif
    check(operation.qk.initialize(arguments), "initialize QK");
  #endif
  #if defined(PREFIX_QK_SUPERBLOCK_PV_MICROTILES)
    #ifndef PREFIX_QK_SUPERBLOCK_PV_TILE_TOKENS
      #define PREFIX_QK_SUPERBLOCK_PV_TILE_TOKENS 8192
    #endif
    static_assert(
        PREFIX_QK_SUPERBLOCK_PV_TILE_TOKENS > 0 &&
            PREFIX_QK_SUPERBLOCK_PV_TILE_TOKENS % PVThreadblockShape::kK == 0,
        "PV microtile K must be a positive whole MMA tile");
    for (int micro_begin = 0; micro_begin < width;
         micro_begin += PREFIX_QK_SUPERBLOCK_PV_TILE_TOKENS) {
      int micro_width =
          std::min(PREFIX_QK_SUPERBLOCK_PV_TILE_TOKENS, width - micro_begin);
      operation.pv_microtiles.push_back(std::make_unique<PVLauncher>(
          block_scores + size_t(micro_begin) * kRows,
          value + size_t(begin + micro_begin) * kHeadDim,
    #if defined(PREFIX_PV_DIRECT_FP32_ACCUMULATE)
          prefix_accumulator,
    #else
          partials,
    #endif
          kRows, micro_width, begin + micro_begin != 0));
    }
  #else
    operation.pv = std::make_unique<PVLauncher>(
        block_scores, value + size_t(begin) * kHeadDim,
    #if defined(PREFIX_PV_DIRECT_FP32_ACCUMULATE)
        prefix_accumulator,
    #else
        partials,
    #endif
        kRows, width,
    #if defined(PREFIX_PV_DIRECT_FP32_ACCUMULATE) || \
        defined(PREFIX_PV_DIRECT_FP16_ACCUMULATE)
        block != 0);
    #else
        false);
    #endif
  #endif
    operators.push_back(std::move(operation));
  }

  #if defined(PREFIX_GROUPED_QK_PV)
  std::vector<std::unique_ptr<PVLauncher>> grouped_prefix_pv;
  grouped_prefix_pv.reserve((blocks + kPipelineBuffers - 1) / kPipelineBuffers);
  for (int group_begin = 0; group_begin < blocks;
       group_begin += kPipelineBuffers) {
    int group_end = std::min(group_begin + kPipelineBuffers, blocks);
    int group_width = 0;
    for (int block = group_begin; block < group_end; ++block) {
      group_width += operators[block].width;
    }
    grouped_prefix_pv.push_back(std::make_unique<PVLauncher>(
        scores, value + size_t(group_begin) * block_n * kHeadDim, partials,
        kRows, group_width, group_begin != 0));
  }
  #endif

  #if defined(PREFIX_BATCHED_TRI_TAIL)
  size_t batched_tail_score_elements =
      size_t(kBatchedTailTasks) * kBatchedTailTileRows *
          kBatchedTailTileTokens +
      size_t(kBatchedTailTiles) * kBatchedTailTileRows *
          kBatchedTailPVPadTokens;
    #if !defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  if (score_buffer_elements < batched_tail_score_elements) {
    std::cerr << "score workspace is too small for batched triangular tail\n";
    return EXIT_FAILURE;
  }
    #endif
  std::vector<Element*> host_tail_q_ptrs;
  std::vector<Element*> host_tail_k_ptrs;
  std::vector<ScoreElement*> host_tail_score_ptrs;
  std::vector<TailPVKernel::Params> host_tail_pv_params;
    #if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
  std::vector<TailQKKernel::Params> host_tail_qk_params;
  host_tail_qk_params.reserve(kBatchedTailTasks);
    #endif
    #if defined(PREFIX_BATCHED_TRI_QK_SERIAL)
  std::vector<std::unique_ptr<CublasQKLauncher>> batched_tail_serial_qk;
  batched_tail_serial_qk.reserve(kBatchedTailTasks);
    #endif
  host_tail_q_ptrs.reserve(kBatchedTailTasks);
  host_tail_k_ptrs.reserve(kBatchedTailTasks);
  host_tail_score_ptrs.reserve(kBatchedTailTasks);
  host_tail_pv_params.reserve(
    #if defined(PREFIX_BATCHED_TRI_FUSED_PV)
      kBatchedTailTiles);
    #elif defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
      kFinePVTasks);
    #else
      kBatchedTailTasks);
    #endif
  int batched_tail_task = 0;
    #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
  int fine_pv_task = 0;
    #endif
  size_t batched_tail_query_score_offset = 0;
  for (int query_tile = 0; query_tile < kBatchedTailTiles; ++query_tile) {
    #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
    auto* query_scores = tail_overlap_scores + batched_tail_query_score_offset;
    #else
    auto* query_scores = scores + batched_tail_query_score_offset;
    #endif
    for (int key_tile = 0; key_tile <= query_tile; ++key_tile) {
      auto* task_scores = query_scores + size_t(key_tile) *
                                             kBatchedTailTileRows *
                                             kBatchedTailTileTokens;
      host_tail_q_ptrs.push_back(query_transposed +
                                 size_t(query_tile) * kBatchedTailTileRows);
      host_tail_k_ptrs.push_back(key_transposed + kPrefix +
                                 size_t(key_tile) * kBatchedTailTileTokens);
      host_tail_score_ptrs.push_back(task_scores);
    #if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
      TailQKLauncher task_qk(
          query_transposed + size_t(query_tile) * kBatchedTailTileRows,
          key_transposed + kPrefix + size_t(key_tile) * kBatchedTailTileTokens,
          task_scores, kBatchedTailTileRows, kBatchedTailTileTokens, kRows,
          kStoredKV);
      host_tail_qk_params.push_back(task_qk.params);
      if (batched_tail_task == 0) {
        batched_tail_qk_grid = task_qk.grid;
        batched_tail_qk_block = task_qk.block;
        batched_tail_qk_smem_bytes = task_qk.smem_bytes;
      }
    #endif
    #if defined(PREFIX_BATCHED_TRI_QK_SERIAL)
      batched_tail_serial_qk.push_back(
          std::make_unique<CublasQKLauncher>(CublasQKLauncher{
              qk_cublas,
              query_transposed + size_t(query_tile) * kBatchedTailTileRows,
              key_transposed + kPrefix +
                  size_t(key_tile) * kBatchedTailTileTokens,
              task_scores, kBatchedTailTileRows, kBatchedTailTileTokens, kRows,
              kStoredKV}));
    #endif
    #if !defined(PREFIX_BATCHED_TRI_FUSED_PV) && \
        !defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
      TailPVLauncher task_pv(
          task_scores,
          value +
              size_t(kPrefix + key_tile * kBatchedTailTileTokens) * kHeadDim,
          batched_tail_partials +
              size_t(batched_tail_task) * kBatchedTailTileRows * kHeadDim,
          kBatchedTailTileRows, kBatchedTailTileTokens, false);
      host_tail_pv_params.push_back(task_pv.params);
      if (batched_tail_task == 0) {
        batched_tail_pv_grid = task_pv.grid;
        batched_tail_pv_block = task_pv.block;
        batched_tail_pv_smem_bytes = task_pv.smem_bytes;
      }
    #endif
      ++batched_tail_task;
    }
    #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV) && \
        !defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE)
    for (int key_tile_begin = 0; key_tile_begin <= query_tile;
         key_tile_begin += kFinePVGroupTiles) {
      int key_tiles =
          std::min(kFinePVGroupTiles, query_tile + 1 - key_tile_begin);
      TailPVLauncher task_pv(
          query_scores + size_t(key_tile_begin) * kBatchedTailTileRows *
                             kBatchedTailTileTokens,
          value + size_t(kPrefix + key_tile_begin * kBatchedTailTileTokens) *
                      kHeadDim,
          batched_tail_partials +
              size_t(fine_pv_task) * kBatchedTailTileRows * kHeadDim,
          kBatchedTailTileRows, key_tiles * kBatchedTailTileTokens, false);
      host_tail_pv_params.push_back(task_pv.params);
      if (fine_pv_task == 0) {
        batched_tail_pv_grid = task_pv.grid;
        batched_tail_pv_block = task_pv.block;
        batched_tail_pv_smem_bytes = task_pv.smem_bytes;
      }
      ++fine_pv_task;
    }
    #endif
    #if defined(PREFIX_BATCHED_TRI_FUSED_PV)
    TailPVLauncher query_pv(
        query_scores, value + size_t(kPrefix) * kHeadDim,
        tail_output + size_t(query_tile) * kBatchedTailTileRows * kHeadDim,
        kBatchedTailTileRows,
        (query_tile + 1) * kBatchedTailTileTokens + kBatchedTailPVPadTokens,
        false);
    host_tail_pv_params.push_back(query_pv.params);
    if (query_tile == 0) {
      batched_tail_pv_grid = query_pv.grid;
      batched_tail_pv_block = query_pv.block;
      batched_tail_pv_smem_bytes = query_pv.smem_bytes;
    }
    #endif
    batched_tail_query_score_offset +=
        size_t(kBatchedTailTileRows) *
        ((query_tile + 1) * kBatchedTailTileTokens + kBatchedTailPVPadTokens);
  }
    #if defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE)
  constexpr int kFinePVRounds =
      (kBatchedTailTiles + kFinePVGroupTiles - 1) / kFinePVGroupTiles;
  for (int round = 0; round < kFinePVRounds; ++round) {
    int key_tile_begin = round * kFinePVGroupTiles;
    for (int query_tile = key_tile_begin; query_tile < kBatchedTailTiles;
         ++query_tile) {
      int key_tiles =
          std::min(kFinePVGroupTiles, query_tile + 1 - key_tile_begin);
      size_t query_score_offset =
          size_t(kBatchedTailTileRows) *
          (size_t(kBatchedTailTileTokens) * query_tile * (query_tile + 1) / 2 +
           size_t(kBatchedTailPVPadTokens) * query_tile);
      TailPVLauncher task_pv(
          tail_overlap_scores + query_score_offset +
              size_t(key_tile_begin) * kBatchedTailTileRows *
                  kBatchedTailTileTokens,
          value + size_t(kPrefix + key_tile_begin * kBatchedTailTileTokens) *
                      kHeadDim,
          tail_output + size_t(query_tile) * kBatchedTailTileRows * kHeadDim,
          kBatchedTailTileRows, key_tiles * kBatchedTailTileTokens, round != 0);
      host_tail_pv_params.push_back(task_pv.params);
      if (fine_pv_task == 0) {
        batched_tail_pv_grid = task_pv.grid;
        batched_tail_pv_block = task_pv.block;
        batched_tail_pv_smem_bytes = task_pv.smem_bytes;
      }
      ++fine_pv_task;
    }
  }
    #endif
  batched_tail_pv_grid.z = unsigned(host_tail_pv_params.size());
    #if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
  batched_tail_qk_grid.z = unsigned(host_tail_qk_params.size());
    #endif
  check(cudaMemcpy(batched_tail_q_ptrs, host_tail_q_ptrs.data(),
                   size_t(kBatchedTailTasks) * sizeof(Element*),
                   cudaMemcpyHostToDevice),
        "copy batched-tail Q pointer array");
  check(cudaMemcpy(batched_tail_k_ptrs, host_tail_k_ptrs.data(),
                   size_t(kBatchedTailTasks) * sizeof(Element*),
                   cudaMemcpyHostToDevice),
        "copy batched-tail K pointer array");
  check(cudaMemcpy(batched_tail_score_ptrs, host_tail_score_ptrs.data(),
                   size_t(kBatchedTailTasks) * sizeof(ScoreElement*),
                   cudaMemcpyHostToDevice),
        "copy batched-tail score pointer array");
  check(cudaMemcpy(batched_tail_pv_params, host_tail_pv_params.data(),
                   host_tail_pv_params.size() * sizeof(TailPVKernel::Params),
                   cudaMemcpyHostToDevice),
        "copy batched-tail PV params");
    #if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
  check(cudaMemcpy(batched_tail_qk_params, host_tail_qk_params.data(),
                   host_tail_qk_params.size() * sizeof(TailQKKernel::Params),
                   cudaMemcpyHostToDevice),
        "copy batched-tail QK params");
  if (batched_tail_qk_smem_bytes >= 48 * 1024) {
    check(cudaFuncSetAttribute(batched_tri_tail_qk_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               batched_tail_qk_smem_bytes),
          "set batched-tail QK dynamic shared memory");
  }
    #endif
  if (batched_tail_pv_smem_bytes >= 48 * 1024) {
    check(cudaFuncSetAttribute(batched_tri_tail_pv_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               batched_tail_pv_smem_bytes),
          "set batched-tail PV dynamic shared memory");
  }
  #endif

  #if defined(PREFIX_CUBLAS_SLICED_TAIL)
  constexpr int kCublasTailSliceTokens = PREFIX_TAIL_SLICE_TOKENS;
  constexpr int kCublasTailSliceRows = kCublasTailSliceTokens * 6;
  static_assert(kTail % kCublasTailSliceTokens == 0,
                "tail slice size must divide 8000 tokens");
  std::vector<CublasTailSliceOperators> cublas_tail_operators;
  cublas_tail_operators.reserve(kTail / kCublasTailSliceTokens);
  for (int slice = 0; slice < kTail / kCublasTailSliceTokens; ++slice) {
    CublasTailSliceOperators operation;
    operation.query_token_start = slice * kCublasTailSliceTokens;
    operation.query_row_start = operation.query_token_start * 6;
    operation.rows = kCublasTailSliceRows;
    operation.width = (slice + 1) * kCublasTailSliceTokens;
    operation.qk = std::make_unique<CublasQKLauncher>(CublasQKLauncher{
        qk_cublas, query_transposed + operation.query_row_start,
        key_transposed + kPrefix, scores, operation.rows, operation.width,
        kRows, kStoredKV});
    operation.pv = std::make_unique<PVLauncher>(
        scores, value + size_t(kPrefix) * kHeadDim,
        tail_output + size_t(operation.query_row_start) * kHeadDim,
        operation.rows, operation.width, false);
    cublas_tail_operators.push_back(std::move(operation));
  }
  #endif

  #if defined(PREFIX_PHASE_SINGLE_PV)
  auto full_prefix_pv = std::make_unique<PVLauncher>(scores, value,
    #if defined(PREFIX_PV_DIRECT_FP32_ACCUMULATE)
                                                     prefix_accumulator,
    #else
                                                     partials,
    #endif
                                                     kRows, kPrefix, false);
  #endif

  #if defined(PREFIX_MATERIALIZED_SLICED_TAIL)
  std::vector<TailSliceOperators> tail_operators;
  tail_operators.reserve(kTail / kTailSliceTokens);
  for (int slice = 0; slice < kTail / kTailSliceTokens; ++slice) {
    TailSliceOperators operation;
    operation.query_token_start = slice * kTailSliceTokens;
    operation.query_row_start = operation.query_token_start * 6;
    operation.rows = kTailSliceRows;
    operation.width = (slice + 1) * kTailSliceTokens;
    typename QKGemm::Arguments arguments(
        {operation.width, operation.rows, kHeadDim}, 1,
        {key_transposed + kPrefix, QKLayoutA(kTransposedKRows)},
        {query_transposed + operation.query_row_start, QKLayoutB(kRows)},
        {scores, typename QKGemm::LayoutC(operation.rows)},
        {scores, typename QKGemm::LayoutC(operation.rows)},
        {Element(0.0625f), Element(0.0f)},
        {qk_norm, typename QKGemm::LayoutN(operation.width)},
        {qk_sum, typename QKGemm::LayoutS(operation.width)},
        {scores, typename QKGemm::LayoutSoft(operation.rows)});
    check(operation.qk.initialize(arguments), "initialize sliced-tail QK");
    operation.pv = std::make_unique<PVLauncher>(
        scores, value + size_t(kPrefix) * kHeadDim, tail_numerator,
        operation.rows, operation.width, false);
    tail_operators.push_back(std::move(operation));
  }
  #endif

  check(cudaMemcpyToSymbol(g_rows, &kRows, sizeof(kRows)), "set row count");
  #if defined(PREFIX_SCORE_EXP_LUT)
  float score_exp_lut[256];
  for (int index = 0; index < 256; ++index) {
    score_exp_lut[index] = std::exp(-float(index) * kStoredScoreScale);
  }
  check(
      cudaMemcpyToSymbol(cutlass::epilogue::threadblock::g_prefix_score_exp_lut,
                         score_exp_lut, sizeof(score_exp_lut)),
      "initialize quantized score exponential LUT");
  #endif
  #if defined(PREFIX_QK_CUBLAS_RAW) || defined(PREFIX_QK_RAW_FIXED_SHIFT)
  float const* persistent_max_ptr = prefix_max;
  #else
  float const* persistent_max_ptr = qk_norm;
  #endif
  check(cudaMemcpyToSymbol(g_row_max, &persistent_max_ptr,
                           sizeof(persistent_max_ptr)),
        "set persistent PV max pointer");
  #if defined(PREFIX_QK_FULL_STATS)
  float const* persistent_inv_sum_ptr = qk_sum;
  check(cudaMemcpyToSymbol(g_row_inv_sum, &persistent_inv_sum_ptr,
                           sizeof(persistent_inv_sum_ptr)),
        "set persistent PV inverse-sum pointer");
  #else
    #if defined(PREFIX_QK_RAW_FIXED_SHIFT)
  float* persistent_sum_ptr = prefix_sum;
    #elif defined(PREFIX_PV_FUSED_PREFIX_SUM)
  float* persistent_sum_ptr = prefix_sum;
    #else
  float* persistent_sum_ptr = qk_sum;
    #endif
  check(cudaMemcpyToSymbol(g_row_sum_out, &persistent_sum_ptr,
                           sizeof(persistent_sum_ptr)),
        "set persistent PV sum pointer");
  #endif
  #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  int persistent_tail_rows = kBatchedTailTileRows;
  float* persistent_tail_sum_ptr = qk_sum;
  int persistent_tail_task_base = 0;
  check(cudaMemcpyToSymbol(g_tail_rows, &persistent_tail_rows,
                           sizeof(persistent_tail_rows)),
        "set idle-SM tail row count");
  check(cudaMemcpyToSymbol(g_tail_row_sum_out, &persistent_tail_sum_ptr,
                           sizeof(persistent_tail_sum_ptr)),
        "set idle-SM tail sum pointer");
  check(cudaMemcpyToSymbol(g_pv_task_base, &persistent_tail_task_base,
                           sizeof(persistent_tail_task_base)),
        "initialize idle-SM tail task base");
  #endif
  cudaStream_t stream = nullptr;
  #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  cudaStream_t tail_overlap_stream = nullptr;
    #if defined(PREFIX_TAIL_WAVE_GATED)
  unsigned int* tail_wave_completed = nullptr;
  unsigned int* tail_wave_ready = nullptr;
  check(cudaMalloc(&tail_wave_completed, size_t(blocks) * sizeof(unsigned int)),
        "allocate prefix-PV wave completion counters");
  check(cudaMalloc(&tail_wave_ready, size_t(blocks) * sizeof(unsigned int)),
        "allocate prefix-PV wave ready flags");
    #endif
  int least_priority = 0;
  int greatest_priority = 0;
  check(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
        "query CUDA stream priorities");
  check(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                     greatest_priority),
        "create high-priority prefix stream");
  check(cudaStreamCreateWithPriority(&tail_overlap_stream,
                                     cudaStreamNonBlocking, least_priority),
        "create low-priority idle-SM tail stream");
  check(cublasSetStream(tail_overlap_cublas, tail_overlap_stream),
        "set idle-SM tail cuBLAS stream");
  cudaEvent_t tail_overlap_ready = nullptr;
  check(cudaEventCreateWithFlags(&tail_overlap_ready, cudaEventDisableTiming),
        "create idle-SM tail-ready event");
  cudaEvent_t tail_overlap_prefix_pv_ready = nullptr;
  cudaEvent_t tail_overlap_start = nullptr;
  cudaEvent_t tail_overlap_qk_done = nullptr;
  cudaEvent_t tail_overlap_done = nullptr;
  cudaEvent_t tail_overlap_prefix_complete = nullptr;
  std::vector<cudaEvent_t> tail_overlap_pv_ready(8, nullptr);
  std::vector<cudaEvent_t> tail_overlap_pv_done(8, nullptr);
  std::vector<int> tail_overlap_query_indices(kBatchedTailTiles);
  for (int query_tile = 0; query_tile < kBatchedTailTiles; ++query_tile) {
    tail_overlap_query_indices[query_tile] = query_tile;
  }
  check(cudaEventCreateWithFlags(&tail_overlap_prefix_pv_ready,
                                 cudaEventDisableTiming),
        "create idle-SM first-prefix-PV event");
  check(cudaEventCreate(&tail_overlap_start),
        "create idle-SM tail-start event");
  check(cudaEventCreate(&tail_overlap_qk_done),
        "create idle-SM tail-QK-done event");
  check(cudaEventCreate(&tail_overlap_done), "create idle-SM tail-done event");
  check(cudaEventCreate(&tail_overlap_prefix_complete),
        "create idle-SM prefix-complete event");
  for (int group = 0; group < 8; ++group) {
    check(cudaEventCreateWithFlags(&tail_overlap_pv_ready[group],
                                   cudaEventDisableTiming),
          "create idle-SM grouped-PV-ready event");
    check(cudaEventCreate(&tail_overlap_pv_done[group]),
          "create idle-SM grouped-PV-done event");
  }
  #endif
  #if defined(PREFIX_DEBUG_PREFIX_STATE)
  unsigned long long* debug_invalid_prefix = nullptr;
  check(cudaMalloc(&debug_invalid_prefix, sizeof(unsigned long long)),
        "allocate debug prefix-state counter");
  #endif
  #if defined(PREFIX_DEBUG_QK_STATE)
    #ifndef PREFIX_DEBUG_QK_BLOCK_INDEX
      #define PREFIX_DEBUG_QK_BLOCK_INDEX 0
    #endif
  constexpr int kDebugQKBlock = PREFIX_DEBUG_QK_BLOCK_INDEX;
  static_assert(kDebugQKBlock >= 0, "debug block must be nonnegative");
  if (kDebugQKBlock >= blocks) {
    std::cerr << "debug QK block is outside the prefix\n";
    return EXIT_FAILURE;
  }
  size_t debug_stat_elements = qk_stat_buffer_elements;
  // Raw cuBLAS QK deliberately omits the CUTLASS max/sum workspaces.  The
  // debug report still expects those buffers, so provision poison-only
  // scratch storage instead of rejecting the score-range diagnostic.
  if (qk_norm == nullptr) {
    check(cudaMalloc(&qk_norm, debug_stat_elements * sizeof(float)),
          "allocate debug-only QK max workspace");
  }
  if (qk_sum == nullptr) {
    check(cudaMalloc(&qk_sum, debug_stat_elements * sizeof(float)),
          "allocate debug-only QK sum workspace");
  }
  check(cudaMemset(qk_norm, 0xff, debug_stat_elements * sizeof(float)),
        "poison debug QK max workspace");
  check(cudaMemset(qk_sum, 0xff, debug_stat_elements * sizeof(float)),
        "poison debug QK sum workspace");
  check(
      cudaMemset(scores, 0xff, size_t(kRows) * block_n * sizeof(ScoreElement)),
      "poison debug QK score workspace");
    #if defined(PREFIX_QK_PRETRANSPOSE_INPUTS)
  {
    dim3 transpose_threads(32, 8);
    dim3 query_transpose_grid((kHeadDim + 31) / 32, (kRows + 31) / 32);
    dim3 key_transpose_grid((kHeadDim + 31) / 32, (kTransposedKRows + 31) / 32);
    transpose_half_32x32<<<query_transpose_grid, transpose_threads, 0,
                           stream>>>(
        reinterpret_cast<__half const*>(query),
        reinterpret_cast<__half*>(query_transposed), kRows, kHeadDim);
    transpose_half_32x32<<<key_transpose_grid, transpose_threads, 0, stream>>>(
        reinterpret_cast<__half const*>(key),
        reinterpret_cast<__half*>(key_transposed), kTransposedKRows, kHeadDim);
    check(cudaGetLastError(), "prepare debug QK transposed inputs");
  }
    #endif
    #if defined(PREFIX_QK_CUBLAS_RAW) || defined(PREFIX_QK_DIRECT_PROB) || \
        defined(PREFIX_QK_RAW_FIXED_SHIFT)
  operators[kDebugQKBlock].qk->launch(stream);
  check(cudaGetLastError(), "launch debug raw QK");
    #else
  check(operators[kDebugQKBlock].qk(stream), "launch debug QK");
    #endif
  check(cudaDeviceSynchronize(), "synchronize debug QK");
  unsigned long long* device_invalid_scores = nullptr;
  unsigned int* device_max_score_bits = nullptr;
  unsigned long long invalid_scores_all = 0;
  unsigned int max_score_bits = 0;
  check(cudaMalloc(&device_invalid_scores, sizeof(unsigned long long)),
        "allocate debug invalid-score counter");
  check(cudaMemset(device_invalid_scores, 0, sizeof(unsigned long long)),
        "clear debug invalid-score counter");
  check(cudaMalloc(&device_max_score_bits, sizeof(unsigned int)),
        "allocate debug max-score bits");
  check(cudaMemset(device_max_score_bits, 0, sizeof(unsigned int)),
        "clear debug max-score bits");
  size_t debug_score_elements = size_t(kRows) * operators[kDebugQKBlock].width;
  count_invalid_scores<<<(debug_score_elements + 255) / 256, 256>>>(
      scores, debug_score_elements, device_invalid_scores,
      device_max_score_bits);
  check(cudaGetLastError(), "count debug invalid scores");
  check(cudaMemcpy(&invalid_scores_all, device_invalid_scores,
                   sizeof(unsigned long long), cudaMemcpyDeviceToHost),
        "copy debug invalid-score counter");
  check(cudaMemcpy(&max_score_bits, device_max_score_bits, sizeof(unsigned int),
                   cudaMemcpyDeviceToHost),
        "copy debug max-score bits");
  float max_abs_score = 0.0f;
  std::memcpy(&max_abs_score, &max_score_bits, sizeof(float));
  std::vector<float> host_max(debug_stat_elements);
  std::vector<float> host_inv_sum(debug_stat_elements);
  std::vector<ScoreElement> host_scores(block_n);
  check(cudaMemcpy(host_max.data(), qk_norm, host_max.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "copy debug QK max");
  check(cudaMemcpy(host_inv_sum.data(), qk_sum,
                   host_inv_sum.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "copy debug QK inverse sum");
  check(cudaMemcpy(host_scores.data(), scores,
                   host_scores.size() * sizeof(ScoreElement),
                   cudaMemcpyDeviceToHost),
        "copy debug QK scores");
  size_t invalid_max = 0;
  size_t invalid_inv_sum = 0;
  size_t invalid_max_all = 0;
  size_t invalid_inv_sum_all = 0;
  size_t invalid_scores = 0;
  float min_inv_sum = std::numeric_limits<float>::infinity();
  float max_inv_sum = 0.0f;
  for (int row = 0; row < kRows; ++row) {
    invalid_max += !std::isfinite(host_max[row]);
    invalid_inv_sum += !std::isfinite(host_inv_sum[row]);
    if (std::isfinite(host_inv_sum[row])) {
      min_inv_sum = std::min(min_inv_sum, host_inv_sum[row]);
      max_inv_sum = std::max(max_inv_sum, host_inv_sum[row]);
    }
  }
  for (size_t index = 0; index < debug_stat_elements; ++index) {
    invalid_max_all += !std::isfinite(host_max[index]);
    invalid_inv_sum_all += !std::isfinite(host_inv_sum[index]);
  }
  for (int column = 0; column < block_n; ++column) {
    invalid_scores += !std::isfinite(float(host_scores[column]));
  }
  std::vector<std::pair<int, int>> missing_score_runs;
  for (int column = 0; column < block_n;) {
    if (std::isfinite(float(host_scores[column]))) {
      ++column;
      continue;
    }
    int begin = column;
    while (column < block_n && !std::isfinite(float(host_scores[column]))) {
      ++column;
    }
    missing_score_runs.emplace_back(begin, column);
  }
  std::cout << "debug_qk invalid_max=" << invalid_max
            << " invalid_inv_sum=" << invalid_inv_sum
            << " invalid_max_all=" << invalid_max_all
            << " invalid_inv_sum_all=" << invalid_inv_sum_all
            << " invalid_scores_row0=" << invalid_scores
            << " invalid_scores_all=" << invalid_scores_all
            << " max_abs_score=" << max_abs_score
            << " min_inv_sum=" << min_inv_sum << " max_inv_sum=" << max_inv_sum
            << " row0_max=" << host_max[0]
            << " row0_inv_sum=" << host_inv_sum[0]
            << " row0_score0=" << float(host_scores[0]) << "\n";
  std::cout << "debug_qk row0_missing_runs=";
  size_t reported_runs = std::min<size_t>(missing_score_runs.size(), 64);
  for (size_t index = 0; index < reported_runs; ++index) {
    if (index) {
      std::cout << ",";
    }
    std::cout << "[" << missing_score_runs[index].first << ","
              << missing_score_runs[index].second << ")";
  }
  if (reported_runs < missing_score_runs.size()) {
    std::cout << ",...";
  }
  std::cout << " total_runs=" << missing_score_runs.size() << "\n";
  return EXIT_SUCCESS;
  #endif
  #if defined(PREFIX_FULL_ENDPOINT) && defined(PREFIX_CONCURRENT_TAIL)
  cudaStream_t concurrent_tail_stream = nullptr;
  cudaEvent_t concurrent_tail_ready = nullptr;
  cudaEvent_t concurrent_tail_complete = nullptr;
  check(
      cudaStreamCreateWithFlags(&concurrent_tail_stream, cudaStreamNonBlocking),
      "create concurrent tail stream");
  check(
      cudaEventCreateWithFlags(&concurrent_tail_ready, cudaEventDisableTiming),
      "create concurrent tail ready event");
  check(cudaEventCreateWithFlags(&concurrent_tail_complete,
                                 cudaEventDisableTiming),
        "create concurrent tail completion event");
  #endif
  #if defined(PREFIX_QK_ASYNC_EXP_PIPELINE)
  cudaStream_t exp_stream = nullptr;
  cudaEvent_t async_qk_ready[kPipelineBuffers] = {};
  cudaEvent_t async_exp_ready[kPipelineBuffers] = {};
  cudaEvent_t async_pv_done[kPipelineBuffers] = {};
  check(cudaStreamCreateWithFlags(&exp_stream, cudaStreamNonBlocking),
        "create async exp stream");
  for (int buffer = 0; buffer < kPipelineBuffers; ++buffer) {
    check(cudaEventCreateWithFlags(&async_qk_ready[buffer],
                                   cudaEventDisableTiming),
          "create async QK-ready event");
    check(cudaEventCreateWithFlags(&async_exp_ready[buffer],
                                   cudaEventDisableTiming),
          "create async exp-ready event");
    check(cudaEventCreateWithFlags(&async_pv_done[buffer],
                                   cudaEventDisableTiming),
          "create async PV-done event");
  }
  check(cudaDeviceSynchronize(), "synchronize async-exp inputs");
  #endif
  #if defined(PREFIX_QK_PV_PIPELINE)
  cudaStream_t qk_stream = nullptr;
  cudaStream_t pv_stream = nullptr;
  cudaEvent_t qk_ready[kPipelineBuffers] = {};
  cudaEvent_t pv_done[kPipelineBuffers] = {};
  check(cudaStreamCreateWithFlags(&qk_stream, cudaStreamNonBlocking),
        "create QK pipeline stream");
  check(cudaStreamCreateWithFlags(&pv_stream, cudaStreamNonBlocking),
        "create PV pipeline stream");
  for (int buffer = 0; buffer < kPipelineBuffers; ++buffer) {
    check(cudaEventCreateWithFlags(&qk_ready[buffer], cudaEventDisableTiming),
          "create QK-ready event");
    check(cudaEventCreateWithFlags(&pv_done[buffer], cudaEventDisableTiming),
          "create PV-done event");
  }
  check(cudaDeviceSynchronize(), "synchronize pipeline inputs");
  #endif
  #if defined(PREFIX_PHASE_DETAIL) && !defined(PREFIX_QK_PV_PIPELINE) && \
      !defined(PREFIX_QK_ASYNC_EXP_PIPELINE) &&                          \
      !defined(PREFIX_PHASE_BATCHED_MATERIALIZATION)
  std::vector<cudaEvent_t> detail_qk_start(blocks);
  std::vector<cudaEvent_t> detail_qk_done(blocks);
  std::vector<cudaEvent_t> detail_pv_done(blocks);
  std::vector<cudaEvent_t> detail_state_done(blocks);
  for (int block = 0; block < blocks; ++block) {
    check(cudaEventCreate(&detail_qk_start[block]), "create detail QK start");
    check(cudaEventCreate(&detail_qk_done[block]), "create detail QK done");
    check(cudaEventCreate(&detail_pv_done[block]), "create detail PV done");
    check(cudaEventCreate(&detail_state_done[block]),
          "create detail state done");
  }
  #endif
  #if defined(PREFIX_BATCHED_TRI_TAIL)
  cudaEvent_t batched_tail_qk_start;
  cudaEvent_t batched_tail_qk_done;
  cudaEvent_t batched_tail_mask_done;
  cudaEvent_t batched_tail_pv_done;
  cudaEvent_t batched_tail_finalize_done;
  check(cudaEventCreate(&batched_tail_qk_start),
        "create batched-tail QK start");
  check(cudaEventCreate(&batched_tail_qk_done), "create batched-tail QK done");
  check(cudaEventCreate(&batched_tail_mask_done),
        "create batched-tail mask done");
  check(cudaEventCreate(&batched_tail_pv_done), "create batched-tail PV done");
  check(cudaEventCreate(&batched_tail_finalize_done),
        "create batched-tail finalize done");
  #endif
  #if defined(PREFIX_QK_PRETRANSPOSE_INPUTS) && \
      defined(PREFIX_UPSTREAM_TRANSPOSED_QK)
  {
    dim3 transpose_threads(32, 8);
    dim3 query_transpose_grid((kHeadDim + 31) / 32, (kRows + 31) / 32);
    dim3 key_transpose_grid((kHeadDim + 31) / 32, (kTransposedKRows + 31) / 32);
    transpose_half_32x32<<<query_transpose_grid, transpose_threads, 0,
                           stream>>>(
        reinterpret_cast<__half const*>(query),
        reinterpret_cast<__half*>(query_transposed), kRows, kHeadDim);
    #if defined(PREFIX_QK_BLOCKED_TRANSPOSE)
    transpose_half_blocked_32x32<<<key_transpose_grid, transpose_threads, 0,
                                   stream>>>(
        reinterpret_cast<__half const*>(key),
        reinterpret_cast<__half*>(key_transposed), kTransposedKRows, kHeadDim,
        block_n);
    #else
    transpose_half_32x32<<<key_transpose_grid, transpose_threads, 0, stream>>>(
        reinterpret_cast<__half const*>(key),
        reinterpret_cast<__half*>(key_transposed), kTransposedKRows, kHeadDim);
    #endif
    check(cudaGetLastError(), "prepare upstream-transposed QK inputs");
    check(cudaDeviceSynchronize(), "synchronize upstream-transposed QK inputs");
  }
  #endif
  auto launch = [&](cudaEvent_t blocks_done, cudaEvent_t prefix_done,
                    cudaEvent_t tail_done) {
  #if defined(PREFIX_MATERIALIZED_SLICED_TAIL) || \
      defined(PREFIX_CUBLAS_SLICED_TAIL) || defined(PREFIX_BATCHED_TRI_TAIL)
    check(cudaMemcpyToSymbolAsync(g_rows, &kRows, sizeof(kRows), 0,
                                  cudaMemcpyHostToDevice, stream),
          "restore prefix row count");
    float* prefix_sum_output = qk_sum;
    #if defined(PREFIX_PV_FUSED_PREFIX_SUM)
    prefix_sum_output = prefix_sum;
    #endif
    check(cudaMemcpyToSymbolAsync(g_row_sum_out, &prefix_sum_output,
                                  sizeof(prefix_sum_output), 0,
                                  cudaMemcpyHostToDevice, stream),
          "restore prefix PV sum pointer");
  #endif
  #if defined(PREFIX_QK_PRETRANSPOSE_INPUTS) && \
      !defined(PREFIX_UPSTREAM_TRANSPOSED_QK)
    dim3 transpose_threads(32, 8);
    dim3 query_transpose_grid((kHeadDim + 31) / 32, (kRows + 31) / 32);
    dim3 key_transpose_grid((kHeadDim + 31) / 32, (kTransposedKRows + 31) / 32);
    transpose_half_32x32<<<query_transpose_grid, transpose_threads, 0,
                           stream>>>(
        reinterpret_cast<__half const*>(query),
        reinterpret_cast<__half*>(query_transposed), kRows, kHeadDim);
    #if defined(PREFIX_QK_BLOCKED_TRANSPOSE)
    transpose_half_blocked_32x32<<<key_transpose_grid, transpose_threads, 0,
                                   stream>>>(
        reinterpret_cast<__half const*>(key),
        reinterpret_cast<__half*>(key_transposed), kTransposedKRows, kHeadDim,
        block_n);
    #else
    transpose_half_32x32<<<key_transpose_grid, transpose_threads, 0, stream>>>(
        reinterpret_cast<__half const*>(key),
        reinterpret_cast<__half*>(key_transposed), kTransposedKRows, kHeadDim);
    #endif
    check(cudaGetLastError(), "transpose QK inputs");
  #endif
  #if defined(PREFIX_TRANSPOSED_SCORE_WORKSPACE)
    #if !defined(PREFIX_FIXED_STATE_NO_RESET)
    check(cudaMemsetAsync(qk_norm, 0,
                          size_t(score_buffer_count) * qk_stat_buffer_elements *
                              sizeof(float),
                          stream),
          "reset transposed-score block maxima");
    #endif
  #endif
  #if defined(PREFIX_QK_CUBLAS_RAW) || defined(PREFIX_QK_RAW_FIXED_SHIFT)
      // The accepted fixed-probability route uses shift zero. Reset only the
      // scalar mass state; the first PV GEMM overwrites the FP32 numerator.
    #if defined(PREFIX_PV_FUSED_PREFIX_SUM)
    check(cudaMemsetAsync(prefix_sum, 0, size_t(kRows) * sizeof(float), stream),
          "reset fused fixed-shift prefix sum");
    #elif !defined(PREFIX_FIXED_STATE_NO_RESET)
    check(cudaMemsetAsync(prefix_max, 0, size_t(kRows) * sizeof(float), stream),
          "reset fixed-shift prefix max");
    check(cudaMemsetAsync(prefix_sum, 0, size_t(kRows) * sizeof(float), stream),
          "reset fixed-shift prefix sum");
    #endif
  #endif
  #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
    // Each accepted prefix PV launch has only 63 resident CTAs on an 80-SM
    // V100.  Keep one low-priority full-tail pipeline resident across prefix
    // windows so it consumes idle slots without paying 25 split-GEMM calls.
    check(cudaMemsetAsync(qk_sum, 0,
                          size_t(
    #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
                              kFinePVTasks
    #else
                              kBatchedTailTiles
    #endif
                              ) *
                              kBatchedTailTileRows * sizeof(float),
                          stream),
          "reset idle-SM tail row sums");
    #if defined(PREFIX_TAIL_WAVE_GATED)
    check(cudaMemsetAsync(tail_wave_completed, 0,
                          size_t(blocks) * sizeof(unsigned int), stream),
          "reset prefix-PV wave completion counters");
    check(cudaMemsetAsync(tail_wave_ready, 0,
                          size_t(blocks) * sizeof(unsigned int), stream),
          "reset prefix-PV wave ready flags");
    #endif
    check(cudaEventRecord(tail_overlap_ready, stream),
          "record idle-SM tail state readiness");
    check(cudaStreamWaitEvent(tail_overlap_stream, tail_overlap_ready, 0),
          "wait for idle-SM tail state readiness");

    #if defined(PREFIX_TAIL_WAVE_GATED)
    auto wait_for_prefix_pv_tail_wave = [&](int block) {
      check(cuStreamWaitValue32(
                reinterpret_cast<CUstream>(tail_overlap_stream),
                reinterpret_cast<CUdeviceptr>(tail_wave_ready + block), 1u,
                CU_STREAM_WAIT_VALUE_GEQ),
            "wait for underfilled prefix-PV wave");
    };

    auto launch_wave_continuous_qk = [&]() {
      check(cudaStreamWaitEvent(tail_overlap_stream,
                                tail_overlap_prefix_pv_ready, 0),
            "wait for hybrid tail-QK start window");
      check(cudaEventRecord(tail_overlap_start, tail_overlap_stream),
            "record hybrid tail start");
      __half alpha = __float2half(0.0625f);
      __half beta = __float2half(0.0f);
      check(cublasGemmBatchedEx(
                tail_overlap_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                kBatchedTailTileRows, kBatchedTailTileTokens, kHeadDim, &alpha,
                reinterpret_cast<void const* const*>(batched_tail_q_ptrs),
                CUDA_R_16F, kRows,
                reinterpret_cast<void const* const*>(batched_tail_k_ptrs),
                CUDA_R_16F, kStoredKV, &beta,
                reinterpret_cast<void* const*>(batched_tail_score_ptrs),
                CUDA_R_16F, kBatchedTailTileRows, kBatchedTailTasks,
                CUBLAS_COMPUTE_16F, batched_tail_qk_algo),
            "launch hybrid continuous triangular-tail QK");
      check(cudaEventRecord(tail_overlap_qk_done, tail_overlap_stream),
            "record hybrid tail QK completion");
      int64_t mask_elements =
          int64_t(kBatchedTailTileRows) *
          std::max(kBatchedTailTileTokens, kBatchedTailPVPadTokens);
      dim3 mask_grid((mask_elements + 255) / 256, kBatchedTailTiles);
      mask_batched_tri_tail_diagonal<<<mask_grid, 256, 0,
                                       tail_overlap_stream>>>(
          reinterpret_cast<__half*>(tail_overlap_scores), kBatchedTailTileRows,
          kBatchedTailTileTokens, kBatchedTailPVPadTokens);
      check(cudaGetLastError(), "mask hybrid triangular-tail scores");
    };

    auto schedule_wave_gated_tail = [&](int block) {
      constexpr int kQKWaveGroups = 7;
      constexpr int kPVWaveGroups = 8;
      if (tail_wave_continuous_qk) {
        if (block == 0) {
          launch_wave_continuous_qk();
        }
        if (block < kQKWaveGroups) {
          return;
        }
        wait_for_prefix_pv_tail_wave(block);
      } else {
        wait_for_prefix_pv_tail_wave(block);
        if (block == 0) {
          check(cudaEventRecord(tail_overlap_start, tail_overlap_stream),
                "record wave-gated tail start");
        }
      }
      if (!tail_wave_continuous_qk && block < kQKWaveGroups) {
        int first_task = kBatchedTailTasks * block / kQKWaveGroups;
        int task_end = kBatchedTailTasks * (block + 1) / kQKWaveGroups;
        int task_count = task_end - first_task;
        __half alpha = __float2half(0.0625f);
        __half beta = __float2half(0.0f);
        check(cublasGemmBatchedEx(tail_overlap_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                  kBatchedTailTileRows, kBatchedTailTileTokens,
                                  kHeadDim, &alpha,
                                  reinterpret_cast<void const* const*>(
                                      batched_tail_q_ptrs + first_task),
                                  CUDA_R_16F, kRows,
                                  reinterpret_cast<void const* const*>(
                                      batched_tail_k_ptrs + first_task),
                                  CUDA_R_16F, kStoredKV, &beta,
                                  reinterpret_cast<void* const*>(
                                      batched_tail_score_ptrs + first_task),
                                  CUDA_R_16F, kBatchedTailTileRows, task_count,
                                  CUBLAS_COMPUTE_16F, batched_tail_qk_algo),
              "launch wave-gated triangular-tail QK group");
        if (block + 1 == kQKWaveGroups) {
          check(cudaEventRecord(tail_overlap_qk_done, tail_overlap_stream),
                "record wave-gated tail QK completion");
          int64_t mask_elements =
              int64_t(kBatchedTailTileRows) *
              std::max(kBatchedTailTileTokens, kBatchedTailPVPadTokens);
          dim3 mask_grid((mask_elements + 255) / 256, kBatchedTailTiles);
          mask_batched_tri_tail_diagonal<<<mask_grid, 256, 0,
                                           tail_overlap_stream>>>(
              reinterpret_cast<__half*>(tail_overlap_scores),
              kBatchedTailTileRows, kBatchedTailTileTokens,
              kBatchedTailPVPadTokens);
          check(cudaGetLastError(), "mask wave-gated triangular-tail scores");
        }
        return;
      }

      int pv_group = block - kQKWaveGroups;
      int first_task = kFinePVTasks * pv_group / kPVWaveGroups;
      int task_end = kFinePVTasks * (pv_group + 1) / kPVWaveGroups;
      int task_count = task_end - first_task;
      set_pv_task_base_kernel<<<1, 1, 0, tail_overlap_stream>>>(first_task);
      dim3 pv_grid = batched_tail_pv_grid;
      pv_grid.z = unsigned(task_count);
      batched_tri_tail_pv_kernel<<<pv_grid, batched_tail_pv_block,
                                   batched_tail_pv_smem_bytes,
                                   tail_overlap_stream>>>(
          batched_tail_pv_params);
      check(cudaGetLastError(), "launch wave-gated triangular-tail PV group");
      if (pv_group + 1 == kPVWaveGroups) {
        check(cudaEventRecord(tail_overlap_done, tail_overlap_stream),
              "record wave-gated full-tail completion");
      }
    };
    #else
    auto schedule_idle_tail = [&]() {
      check(cudaStreamWaitEvent(tail_overlap_stream,
                                tail_overlap_prefix_pv_ready, 0),
            "wait for first idle-SM prefix-PV window");
      check(cudaEventRecord(tail_overlap_start, tail_overlap_stream),
            "record idle-SM full-tail start");
      #if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
      batched_tri_tail_qk_kernel<<<batched_tail_qk_grid, batched_tail_qk_block,
                                   batched_tail_qk_smem_bytes,
                                   tail_overlap_stream>>>(
          batched_tail_qk_params);
      check(cudaGetLastError(), "launch idle-SM CUTLASS triangular-tail QK");
      #else
      __half alpha = __float2half(0.0625f);
      __half beta = __float2half(0.0f);
      check(cublasGemmBatchedEx(
                tail_overlap_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                kBatchedTailTileRows, kBatchedTailTileTokens, kHeadDim, &alpha,
                reinterpret_cast<void const* const*>(batched_tail_q_ptrs),
                CUDA_R_16F, kRows,
                reinterpret_cast<void const* const*>(batched_tail_k_ptrs),
                CUDA_R_16F, kStoredKV, &beta,
                reinterpret_cast<void* const*>(batched_tail_score_ptrs),
                CUDA_R_16F, kBatchedTailTileRows, kBatchedTailTasks,
                CUBLAS_COMPUTE_16F, batched_tail_qk_algo),
            "launch idle-SM full triangular-tail QK");
      #endif
      check(cudaEventRecord(tail_overlap_qk_done, tail_overlap_stream),
            "record idle-SM full-tail QK done");
      #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
      int64_t fine_mask_elements =
          int64_t(kBatchedTailTileRows) *
          std::max(kBatchedTailTileTokens, kBatchedTailPVPadTokens);
      dim3 fine_mask_grid((fine_mask_elements + 255) / 256, kBatchedTailTiles);
      mask_batched_tri_tail_diagonal<<<fine_mask_grid, 256, 0,
                                       tail_overlap_stream>>>(
          reinterpret_cast<__half*>(tail_overlap_scores), kBatchedTailTileRows,
          kBatchedTailTileTokens, kBatchedTailPVPadTokens);
      check(cudaGetLastError(), "mask fine-grained idle-SM tail scores");
        #if defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE)
      int direct_task_base = 0;
      constexpr int kDirectRounds =
          (kBatchedTailTiles + kFinePVGroupTiles - 1) / kFinePVGroupTiles;
      for (int round = 0; round < kDirectRounds; ++round) {
        int round_tasks = kBatchedTailTiles - round * kFinePVGroupTiles;
        set_pv_task_base_kernel<<<1, 1, 0, tail_overlap_stream>>>(
            direct_task_base);
        dim3 round_grid = batched_tail_pv_grid;
        round_grid.z = unsigned(round_tasks);
        batched_tri_tail_pv_kernel<<<round_grid, batched_tail_pv_block,
                                     batched_tail_pv_smem_bytes,
                                     tail_overlap_stream>>>(
            batched_tail_pv_params);
        direct_task_base += round_tasks;
      }
        #else
      batched_tri_tail_pv_kernel<<<batched_tail_pv_grid, batched_tail_pv_block,
                                   batched_tail_pv_smem_bytes,
                                   tail_overlap_stream>>>(
          batched_tail_pv_params);
        #endif
      check(cudaGetLastError(),
            "launch fine-grained idle-SM triangular-tail PV");
      check(cudaEventRecord(tail_overlap_done, tail_overlap_stream),
            "record fine-grained idle-SM full-tail completion");
      #else
      if (tail_overlap_mode == 1) {
        batched_tri_tail_pv_kernel<<<
            batched_tail_pv_grid, batched_tail_pv_block,
            batched_tail_pv_smem_bytes, tail_overlap_stream>>>(
            batched_tail_pv_params, reinterpret_cast<__half*>(tail_output),
            qk_sum, tail_max, tail_sum, kBatchedTailTileRows);
        check(cudaGetLastError(),
              "launch continuous idle-SM full triangular-tail PV");
        check(cudaEventRecord(tail_overlap_done, tail_overlap_stream),
              "record continuous idle-SM full-tail completion");
      } else if (tail_overlap_mode == 0) {
        check(cudaEventRecord(tail_overlap_done, tail_overlap_stream),
              "record idle-SM tail-QK-only completion");
      }
      #endif
    };
    #endif

    #if !defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
    auto launch_idle_tail_pv_range = [&](int first_query, int query_count) {
      set_pv_task_base_kernel<<<1, 1, 0, tail_overlap_stream>>>(first_query);
      check(cudaGetLastError(), "set idle-SM grouped-PV query range");
      dim3 query_pv_grid = batched_tail_pv_grid;
      query_pv_grid.z = unsigned(query_count);
      batched_tri_tail_pv_kernel<<<query_pv_grid, batched_tail_pv_block,
                                   batched_tail_pv_smem_bytes,
                                   tail_overlap_stream>>>(
          batched_tail_pv_params, reinterpret_cast<__half*>(tail_output),
          qk_sum, tail_max, tail_sum, kBatchedTailTileRows);
      check(cudaGetLastError(), "launch idle-SM grouped triangular-tail PV");
    };

    auto schedule_idle_tail_pv_group = [&](int group) {
      check(cudaStreamWaitEvent(tail_overlap_stream,
                                tail_overlap_pv_ready[group], 0),
            "wait for idle-SM grouped-PV window");
      switch (group) {
        case 0:
          launch_idle_tail_pv_range(0, 9);
          break;
        case 1:
          launch_idle_tail_pv_range(9, 4);
          break;
        case 2:
          launch_idle_tail_pv_range(13, 3);
          break;
        case 3:
          launch_idle_tail_pv_range(16, 2);
          break;
        case 4:
          launch_idle_tail_pv_range(18, 2);
          break;
        case 5:
          launch_idle_tail_pv_range(20, 2);
          break;
        case 6:
          launch_idle_tail_pv_range(22, 2);
          break;
        default:
          launch_idle_tail_pv_range(24, 1);
          break;
      }
      check(cudaEventRecord(tail_overlap_pv_done[group], tail_overlap_stream),
            "record idle-SM grouped-PV completion");
      if (group == 7) {
        check(cudaEventRecord(tail_overlap_done, tail_overlap_stream),
              "record idle-SM full-tail completion");
      }
      check(cudaStreamWaitEvent(stream, tail_overlap_pv_done[group], 0),
            "join idle-SM grouped PV before next prefix QK");
    };

    auto schedule_idle_tail_full_pv = [&]() {
      check(
          cudaStreamWaitEvent(tail_overlap_stream, tail_overlap_pv_ready[0], 0),
          "wait for gated idle-SM full-PV window");
      batched_tri_tail_pv_kernel<<<batched_tail_pv_grid, batched_tail_pv_block,
                                   batched_tail_pv_smem_bytes,
                                   tail_overlap_stream>>>(
          batched_tail_pv_params, reinterpret_cast<__half*>(tail_output),
          qk_sum, tail_max, tail_sum, kBatchedTailTileRows);
      check(cudaGetLastError(), "launch gated idle-SM full triangular-tail PV");
      check(cudaEventRecord(tail_overlap_pv_done[0], tail_overlap_stream),
            "record gated idle-SM full-PV completion");
      check(cudaEventRecord(tail_overlap_done, tail_overlap_stream),
            "record gated idle-SM full-tail completion");
    };
    #endif
  #endif
  #if defined(PREFIX_FULL_ENDPOINT) && defined(PREFIX_CONCURRENT_TAIL) && \
      !defined(PREFIX_TAIL_PV_OVERLAP)
    check(cudaEventRecord(concurrent_tail_ready, stream),
          "record concurrent tail input readiness");
    check(cudaStreamWaitEvent(concurrent_tail_stream, concurrent_tail_ready, 0),
          "wait for concurrent tail inputs");
    check(tail_raw(query, key + size_t(kPrefix) * kHeadDim,
                   value + size_t(kPrefix) * kHeadDim, tail_max, tail_sum,
                   tail_output, kTail, kTail, 6, 1, 0.0625f,
                   concurrent_tail_stream),
          "launch concurrent exact causal tail");
    check(cudaEventRecord(concurrent_tail_complete, concurrent_tail_stream),
          "record concurrent tail completion");
  #endif
  #if defined(PREFIX_PHASE_BATCHED_MATERIALIZATION)
    for (int block = 0; block < blocks; ++block) {
      operators[block].qk->launch(stream);
      check(cudaGetLastError(), "launch phase-batched raw QK");
    }
    #if defined(PREFIX_PHASE_SINGLE_PV)
    float* full_sum_output = qk_sum;
    check(cudaMemcpyToSymbolAsync(g_row_sum_out, &full_sum_output,
                                  sizeof(full_sum_output), 0,
                                  cudaMemcpyHostToDevice, stream),
          "set full-prefix PV sum pointer");
    full_prefix_pv->launch(stream);
    prepare_prefix_update<<<(kRows + 255) / 256, 256, 0, stream>>>(
        qk_norm, qk_sum, prefix_max, prefix_sum, old_scales, block_scales,
        kRows, true);
    #else
    for (int block = 0; block < blocks; ++block) {
      float* batched_sum_output = operators[block].qk_sum;
      check(cudaMemcpyToSymbolAsync(g_row_sum_out, &batched_sum_output,
                                    sizeof(batched_sum_output), 0,
                                    cudaMemcpyHostToDevice, stream),
            "set phase-batched PV sum pointer");
      operators[block].pv->launch(stream);
      prepare_prefix_update<<<(kRows + 255) / 256, 256, 0, stream>>>(
          operators[block].qk_norm, operators[block].qk_sum, prefix_max,
          prefix_sum, old_scales, block_scales, kRows, block == 0);
    }
    #endif
  #elif defined(PREFIX_GROUPED_QK_PV)
    int group_index = 0;
    for (int group_begin = 0; group_begin < blocks;
         group_begin += kPipelineBuffers, ++group_index) {
      int group_end = std::min(group_begin + kPipelineBuffers, blocks);
      for (int block = group_begin; block < group_end; ++block) {
    #if defined(PREFIX_PHASE_DETAIL)
        if (blocks_done != nullptr) {
          check(cudaEventRecord(detail_qk_start[block], stream),
                "record grouped-prefix QK start");
        }
    #endif
        operators[block].qk->launch(stream);
        check(cudaGetLastError(), "launch grouped-prefix raw QK");
    #if defined(PREFIX_PHASE_DETAIL)
        if (blocks_done != nullptr) {
          check(cudaEventRecord(detail_qk_done[block], stream),
                "record grouped-prefix QK done");
        }
    #endif
      }
      float* grouped_sum_output = qk_sum;
      check(cudaMemcpyToSymbolAsync(g_row_sum_out, &grouped_sum_output,
                                    sizeof(grouped_sum_output), 0,
                                    cudaMemcpyHostToDevice, stream),
            "set grouped-prefix PV sum pointer");
      grouped_prefix_pv[group_index]->launch(stream);
    #if defined(PREFIX_PHASE_DETAIL)
      if (blocks_done != nullptr) {
        for (int block = group_begin; block < group_end; ++block) {
          check(cudaEventRecord(detail_pv_done[block], stream),
                "record grouped-prefix PV done");
        }
      }
    #endif
      prepare_prefix_update<<<(kRows + 255) / 256, 256, 0, stream>>>(
          qk_norm, qk_sum, prefix_max, prefix_sum, old_scales, block_scales,
          kRows, group_begin == 0);
    #if defined(PREFIX_PHASE_DETAIL)
      if (blocks_done != nullptr) {
        for (int block = group_begin; block < group_end; ++block) {
          check(cudaEventRecord(detail_state_done[block], stream),
                "record grouped-prefix state done");
        }
      }
    #endif
    }
  #elif defined(PREFIX_TAIL_PV_OVERLAP)
    for (int group_begin = 0; group_begin < blocks;
         group_begin += kPipelineBuffers) {
      int group_end = std::min(group_begin + kPipelineBuffers, blocks);
      for (int block = group_begin; block < group_end; ++block) {
    #if defined(PREFIX_PHASE_DETAIL)
        if (blocks_done != nullptr) {
          check(cudaEventRecord(detail_qk_start[block], stream),
                "record grouped QK start");
        }
    #endif
        operators[block].qk->launch(stream);
        check(cudaGetLastError(), "launch tail/PV-overlap QK");
    #if defined(PREFIX_PHASE_DETAIL)
        if (blocks_done != nullptr) {
          check(cudaEventRecord(detail_qk_done[block], stream),
                "record grouped QK done");
        }
    #endif
      }
      if (group_begin == 0) {
        check(cudaEventRecord(concurrent_tail_ready, stream),
              "record delayed-tail readiness");
        check(cudaStreamWaitEvent(concurrent_tail_stream, concurrent_tail_ready,
                                  0),
              "wait for delayed-tail readiness");
        check(tail_raw(query, key + size_t(kPrefix) * kHeadDim,
                       value + size_t(kPrefix) * kHeadDim, tail_max, tail_sum,
                       tail_output, kTail, kTail, 6, 1, 0.0625f,
                       concurrent_tail_stream),
              "launch tail against PV-only window");
        check(cudaEventRecord(concurrent_tail_complete, concurrent_tail_stream),
              "record delayed-tail completion");
      }
      for (int block = group_begin; block < group_end; ++block) {
        float* grouped_sum_output = operators[block].qk_sum;
        check(cudaMemcpyToSymbolAsync(g_row_sum_out, &grouped_sum_output,
                                      sizeof(grouped_sum_output), 0,
                                      cudaMemcpyHostToDevice, stream),
              "set grouped PV sum pointer");
        operators[block].pv->launch(stream);
    #if defined(PREFIX_PHASE_DETAIL)
        if (blocks_done != nullptr) {
          check(cudaEventRecord(detail_pv_done[block], stream),
                "record grouped PV done");
        }
    #endif
        prepare_prefix_update<<<(kRows + 255) / 256, 256, 0, stream>>>(
            operators[block].qk_norm, operators[block].qk_sum, prefix_max,
            prefix_sum, old_scales, block_scales, kRows, block == 0);
    #if defined(PREFIX_PHASE_DETAIL)
        if (blocks_done != nullptr) {
          check(cudaEventRecord(detail_state_done[block], stream),
                "record grouped state done");
        }
    #endif
      }
    }
  #elif defined(PREFIX_QK_ASYNC_EXP_PIPELINE)
    for (int group_begin = 0; group_begin < blocks;
         group_begin += kPipelineBuffers) {
      int group_end = std::min(group_begin + kPipelineBuffers, blocks);
      for (int block = group_begin; block < group_end; ++block) {
        int buffer = operators[block].buffer;
        operators[block].qk->launch(stream);
        check(cudaGetLastError(), "launch async-exp raw QK");
        check(cudaEventRecord(async_qk_ready[buffer], stream),
              "record async-exp QK ready");
        check(cudaStreamWaitEvent(exp_stream, async_qk_ready[buffer], 0),
              "wait for async-exp QK");
        size_t score_pairs = size_t(kRows) * operators[block].width / 2;
        exp2_scores_inplace<<<(score_pairs + 255) / 256, 256, 0, exp_stream>>>(
            reinterpret_cast<__half2*>(scores +
                                       size_t(buffer) * score_buffer_elements),
            score_pairs);
        check(cudaGetLastError(), "launch async score exp2");
        check(cudaEventRecord(async_exp_ready[buffer], exp_stream),
              "record async exp ready");
      }
      for (int block = group_begin; block < group_end; ++block) {
        int buffer = operators[block].buffer;
        check(cudaStreamWaitEvent(stream, async_exp_ready[buffer], 0),
              "wait for async score exp2");
        float* async_sum_output = operators[block].qk_sum;
        check(cudaMemcpyToSymbolAsync(g_row_sum_out, &async_sum_output,
                                      sizeof(async_sum_output), 0,
                                      cudaMemcpyHostToDevice, stream),
              "set async-exp PV sum pointer");
        operators[block].pv->launch(stream);
        prepare_prefix_update<<<(kRows + 255) / 256, 256, 0, stream>>>(
            operators[block].qk_norm, operators[block].qk_sum, prefix_max,
            prefix_sum, old_scales, block_scales, kRows, block == 0);
        check(cudaEventRecord(async_pv_done[buffer], stream),
              "record async-exp PV done");
      }
    }
  #elif defined(PREFIX_QK_PV_PIPELINE)
    for (int block = 0; block < blocks; ++block) {
      int buffer = operators[block].buffer;
      if (block >= kPipelineBuffers) {
        check(cudaStreamWaitEvent(qk_stream, pv_done[buffer], 0),
              "wait to reuse score pipeline buffer");
      }
    #if defined(PREFIX_QK_CUBLAS_RAW) || defined(PREFIX_QK_DIRECT_PROB) || \
        defined(PREFIX_QK_RAW_FIXED_SHIFT)
      operators[block].qk->launch(qk_stream);
      check(cudaGetLastError(), "launch pipelined raw QK");
    #else
      check(operators[block].qk(qk_stream), "launch pipelined QK");
    #endif
      check(cudaEventRecord(qk_ready[buffer], qk_stream),
            "record pipelined QK ready");
      check(cudaStreamWaitEvent(pv_stream, qk_ready[buffer], 0),
            "wait for pipelined QK");
    #if defined(PREFIX_QK_CUBLAS_RAW)
      float* pipelined_sum_output = operators[block].qk_sum;
      check(cudaMemcpyToSymbolAsync(g_row_sum_out, &pipelined_sum_output,
                                    sizeof(pipelined_sum_output), 0,
                                    cudaMemcpyHostToDevice, pv_stream),
            "set pipelined PV sum pointer");
    #endif
      operators[block].pv->launch(pv_stream);
      prepare_prefix_update<<<(kRows + 255) / 256, 256, 0, pv_stream>>>(
          operators[block].qk_norm, operators[block].qk_sum, prefix_max,
          prefix_sum, old_scales, block_scales, kRows, block == 0);
      check(cudaEventRecord(pv_done[buffer], pv_stream),
            "record pipelined PV done");
    }
    check(cudaStreamWaitEvent(stream, pv_done[operators.back().buffer], 0),
          "join QK/PV pipeline");
  #else
    for (int block = 0; block < blocks; ++block) {
    #if defined(PREFIX_PHASE_DETAIL)
      if (blocks_done != nullptr) {
        check(cudaEventRecord(detail_qk_start[block], stream),
              "record detail QK start");
      }
    #endif
    #if defined(PREFIX_QK_CUBLAS_RAW) || defined(PREFIX_QK_DIRECT_PROB) || \
        defined(PREFIX_QK_RAW_FIXED_SHIFT)
      operators[block].qk->launch(stream);
      check(cudaGetLastError(), "launch raw fixed-shift QK");
    #else
      check(operators[block].qk(stream), "launch QK max");
    #endif
    #if defined(PREFIX_PHASE_DETAIL)
      if (blocks_done != nullptr) {
        check(cudaEventRecord(detail_qk_done[block], stream),
              "record detail QK done");
      }
    #endif
    #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
      #if defined(PREFIX_TAIL_WAVE_GATED)
      if (block == 0 && tail_wave_continuous_qk) {
        check(cudaEventRecord(tail_overlap_prefix_pv_ready, stream),
              "record hybrid tail-QK prefix-PV readiness");
      }
      #else
      if (block == tail_overlap_start_block) {
        // This event becomes ready immediately before prefix PV. Stream
        // priority admits prefix CTA waves before the persistent low-priority
        // tail pipeline consumes remaining scheduling slots.
        check(cudaEventRecord(tail_overlap_prefix_pv_ready, stream),
              "record first idle-SM prefix-PV readiness");
      }
      #endif
      #if !defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
      if (block == 7 && tail_overlap_mode == 2) {
        check(cudaEventRecord(tail_overlap_pv_ready[0], stream),
              "record gated idle-SM full-PV readiness");
      }
      if (block >= 7 && tail_overlap_mode == 3) {
        check(cudaEventRecord(tail_overlap_pv_ready[block - 7], stream),
              "record idle-SM range-PV readiness");
      }
      #endif
    #endif
    #if defined(PREFIX_TAIL_WAVE_GATED)
      operators[block].pv->launch_wave_signaled(
          stream, tail_wave_completed + block, tail_wave_ready + block,
          unsigned(tail_wave_ready_after));
    #else
      #if defined(PREFIX_QK_SUPERBLOCK_PV_MICROTILES)
      for (auto const& pv_microtile : operators[block].pv_microtiles) {
        pv_microtile->launch(stream);
      }
      #else
      operators[block].pv->launch(stream);
      #endif
    #endif
    #if defined(PREFIX_PHASE_DETAIL)
      if (blocks_done != nullptr) {
        check(cudaEventRecord(detail_pv_done[block], stream),
              "record detail PV done");
      }
    #endif
    #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
      #if defined(PREFIX_TAIL_WAVE_GATED)
      schedule_wave_gated_tail(block);
      #else
      if (block == tail_overlap_start_block) {
        schedule_idle_tail();
      }
      #endif
      #if !defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
      if (block == 7 && tail_overlap_mode == 2) {
        schedule_idle_tail_full_pv();
      }
      if (block >= 7 && tail_overlap_mode == 3) {
        schedule_idle_tail_pv_group(block - 7);
      }
      #endif
    #endif
    #if !defined(PREFIX_QK_RAW_FIXED_SHIFT) && \
        !defined(PREFIX_PV_FUSED_PREFIX_SUM)
      #if defined(PREFIX_DEBUG_PREFIX_STATE)
      check(cudaMemsetAsync(debug_invalid_prefix, 0, sizeof(unsigned long long),
                            stream),
            "clear debug prefix-state counter");
      size_t prefix_elements = size_t(kRows) * kHeadDim;
      count_invalid_prefix_state<<<(prefix_elements + 255) / 256, 256, 0,
                                   stream>>>(
          prefix_accumulator, prefix_elements, debug_invalid_prefix);
      check(cudaGetLastError(), "count debug invalid prefix state");
      unsigned long long host_invalid_prefix = 0;
      check(cudaMemcpyAsync(&host_invalid_prefix, debug_invalid_prefix,
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "copy debug invalid prefix-state counter");
      check(cudaStreamSynchronize(stream),
            "synchronize debug prefix-state counter");
      std::cout << "debug_prefix block=" << block
                << " invalid=" << host_invalid_prefix << "\n";
      #endif
      prepare_prefix_update<<<(kRows + 255) / 256, 256, 0, stream>>>(
          qk_norm, qk_sum, prefix_max, prefix_sum, old_scales, block_scales,
          kRows, block == 0);
      #if !defined(PREFIX_PV_DIRECT_FP32_ACCUMULATE) && \
          !defined(PREFIX_PV_DIRECT_FP16_ACCUMULATE)
      int pairs = kRows * kHeadDim / 2;
      apply_prefix_update_half2<<<(pairs + 255) / 256, 256, 0, stream>>>(
          reinterpret_cast<__half const*>(partials), prefix_accumulator,
          old_scales, block_scales, kRows, block == 0);
      #endif
    #endif
    #if defined(PREFIX_PHASE_DETAIL)
      if (blocks_done != nullptr) {
        check(cudaEventRecord(detail_state_done[block], stream),
              "record detail state done");
      }
    #endif
    }
  #endif
  #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
    check(cudaEventRecord(tail_overlap_prefix_complete, stream),
          "record prefix completion before idle-SM tail join");
    check(cudaStreamWaitEvent(stream, tail_overlap_done, 0),
          "join continuous idle-SM full tail");
  #endif
    if (blocks_done != nullptr) {
      check(cudaEventRecord(blocks_done, stream), "record QK/PV blocks done");
    }
  #if !defined(PREFIX_FULL_ENDPOINT)
    int pairs = kRows * kHeadDim / 2;
    finalize_prefix_accumulator<<<(pairs + 255) / 256, 256, 0, stream>>>(
        prefix_accumulator, prefix_sum, reinterpret_cast<__half*>(output),
        kRows);
  #endif
    if (prefix_done != nullptr) {
      check(cudaEventRecord(prefix_done, stream), "record prefix merge done");
    }
  #if defined(PREFIX_FULL_ENDPOINT)
    #if defined(PREFIX_BATCHED_TRI_TAIL)
      #if !defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
    int batched_tail_rows = kBatchedTailTileRows;
    check(cudaMemcpyToSymbolAsync(g_rows, &batched_tail_rows,
                                  sizeof(batched_tail_rows), 0,
                                  cudaMemcpyHostToDevice, stream),
          "set batched-tail row count");
    float* batched_tail_sum_output = qk_sum;
    check(cudaMemcpyToSymbolAsync(g_row_sum_out, &batched_tail_sum_output,
                                  sizeof(batched_tail_sum_output), 0,
                                  cudaMemcpyHostToDevice, stream),
          "set batched-tail PV sum pointer");
        #if defined(PREFIX_PV_FUSED_PREFIX_SUM)
    check(cudaMemsetAsync(
              qk_sum, 0,
              size_t(kBatchedTailTiles) * kBatchedTailTileRows * sizeof(float),
              stream),
          "reset fused batched-tail row sums");
        #endif
    check(cublasSetStream(qk_cublas, stream), "set batched-tail cuBLAS stream");
    check(cudaEventRecord(batched_tail_qk_start, stream),
          "record batched-tail QK start");
    __half batched_tail_alpha = __float2half(0.0625f);
    __half batched_tail_beta = __float2half(0.0f);
        #if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
    batched_tri_tail_qk_kernel<<<batched_tail_qk_grid, batched_tail_qk_block,
                                 batched_tail_qk_smem_bytes, stream>>>(
        batched_tail_qk_params);
    check(cudaGetLastError(), "launch CUTLASS batched triangular-tail QK");
        #elif defined(PREFIX_BATCHED_TRI_QK_SERIAL)
    for (auto const& task_qk : batched_tail_serial_qk) {
      task_qk->launch(stream);
    }
        #else
    check(cublasGemmBatchedEx(
              qk_cublas, CUBLAS_OP_N, CUBLAS_OP_T, kBatchedTailTileRows,
              kBatchedTailTileTokens, kHeadDim, &batched_tail_alpha,
              reinterpret_cast<void const* const*>(batched_tail_q_ptrs),
              CUDA_R_16F, kRows,
              reinterpret_cast<void const* const*>(batched_tail_k_ptrs),
              CUDA_R_16F, kStoredKV, &batched_tail_beta,
              reinterpret_cast<void* const*>(batched_tail_score_ptrs),
              CUDA_R_16F, kBatchedTailTileRows, kBatchedTailTasks,
              CUBLAS_COMPUTE_16F, batched_tail_qk_algo),
          "launch batched triangular-tail QK");
        #endif
    check(cudaEventRecord(batched_tail_qk_done, stream),
          "record batched-tail QK done");
        #if !defined(PREFIX_BATCHED_TRI_FUSE_CAUSAL_MASK)
    int64_t batched_mask_elements =
        int64_t(kBatchedTailTileRows) *
        std::max(kBatchedTailTileTokens, kBatchedTailPVPadTokens);
    dim3 batched_mask_grid((batched_mask_elements + 255) / 256,
                           kBatchedTailTiles);
    mask_batched_tri_tail_diagonal<<<batched_mask_grid, 256, 0, stream>>>(
        reinterpret_cast<__half*>(scores), kBatchedTailTileRows,
        kBatchedTailTileTokens, kBatchedTailPVPadTokens);
    check(cudaGetLastError(), "mask batched triangular-tail diagonal");
        #endif
    check(cudaEventRecord(batched_tail_mask_done, stream),
          "record batched-tail mask done");
    batched_tri_tail_pv_kernel<<<batched_tail_pv_grid, batched_tail_pv_block,
                                 batched_tail_pv_smem_bytes, stream>>>(
        batched_tail_pv_params
        #if defined(PREFIX_BATCHED_TRI_FUSE_NORMALIZE)
        ,
        reinterpret_cast<__half*>(tail_output), qk_sum, tail_max, tail_sum,
        kBatchedTailTileRows
        #endif
    );
    check(cudaGetLastError(), "launch batched triangular-tail PV");
    check(cudaEventRecord(batched_tail_pv_done, stream),
          "record batched-tail PV done");
        #if defined(PREFIX_BATCHED_TRI_FUSED_PV)
          #if !defined(PREFIX_BATCHED_TRI_FUSE_NORMALIZE)
    finalize_batched_tri_fused_pv<<<kBatchedTailCoveredRows, kHeadDim, 0,
                                    stream>>>(
        reinterpret_cast<__half*>(tail_output), qk_sum, tail_max, tail_sum,
        kBatchedTailTileRows, kBatchedTailCoveredRows);
          #endif
        #else
    finalize_batched_tri_tail<<<kBatchedTailCoveredRows, kHeadDim, 0, stream>>>(
        reinterpret_cast<__half const*>(batched_tail_partials), qk_sum,
        reinterpret_cast<__half*>(tail_output), tail_max, tail_sum,
        kBatchedTailTileRows, kBatchedTailCoveredRows);
        #endif
    check(cudaGetLastError(), "finalize batched triangular tail");
      #else
        #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
    check(cudaEventRecord(batched_tail_qk_start, stream),
          "record fine-grained overlapped-tail phase start");
    check(cudaEventRecord(batched_tail_qk_done, stream),
          "record fine-grained overlapped-tail QK placeholder");
    check(cudaEventRecord(batched_tail_mask_done, stream),
          "record fine-grained overlapped-tail mask placeholder");
    check(cudaEventRecord(batched_tail_pv_done, stream),
          "record fine-grained overlapped-tail PV placeholder");
          #if defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE)
            #if !defined(PREFIX_TAIL_FUSE_STATE_IN_MERGE)
    finalize_round_major_tail_state<<<(kBatchedTailCoveredRows + 255) / 256,
                                      256, 0, stream>>>(
        qk_sum, tail_max, tail_sum, kBatchedTailTileRows,
        kBatchedTailCoveredRows);
            #endif
          #else
    finalize_grouped_batched_tri_tail<<<kBatchedTailCoveredRows, kHeadDim, 0,
                                        stream>>>(
        reinterpret_cast<__half const*>(batched_tail_partials), qk_sum,
        reinterpret_cast<__half*>(tail_output), tail_max, tail_sum,
        kBatchedTailTileRows, kBatchedTailCoveredRows);
          #endif
    check(cudaGetLastError(),
          "finalize fine-grained overlapped triangular tail");
        #else
    // QK was produced on the low-priority stream while prefix PV consumed its
    // full CTA waves. Run fused PV post-prefix only in QK-only mode.
    check(cudaEventRecord(batched_tail_qk_start, stream),
          "record overlapped-tail phase start");
    check(cudaEventRecord(batched_tail_qk_done, stream),
          "record overlapped-tail QK placeholder");
    check(cudaEventRecord(batched_tail_mask_done, stream),
          "record overlapped-tail mask placeholder");
    if (tail_overlap_mode == 0) {
      batched_tri_tail_pv_kernel<<<batched_tail_pv_grid, batched_tail_pv_block,
                                   batched_tail_pv_smem_bytes, stream>>>(
          batched_tail_pv_params, reinterpret_cast<__half*>(tail_output),
          qk_sum, tail_max, tail_sum, kBatchedTailTileRows);
      check(cudaGetLastError(),
            "launch post-prefix fused PV for overlapped tail QK");
    }
    check(cudaEventRecord(batched_tail_pv_done, stream),
          "record post-prefix fused tail PV");
        #endif
      #endif
      #if PREFIX_BATCHED_TAIL_RESIDUAL_TOKENS > 0
    check(tail_raw(query + size_t(kBatchedTailCoveredRows) * kHeadDim,
                   key + size_t(kPrefix) * kHeadDim,
                   value + size_t(kPrefix) * kHeadDim,
                   tail_max + kBatchedTailCoveredRows,
                   tail_sum + kBatchedTailCoveredRows,
                   tail_output + size_t(kBatchedTailCoveredRows) * kHeadDim,
                   kBatchedTailResidualTokens, kTail, 6, 1, 0.0625f, stream),
          "launch exact residual triangular tail");
      #endif
      #if defined(PREFIX_BATCHED_TRI_REPAIR_FIRST_TILE)
    check(tail_raw(query, key + size_t(kPrefix) * kHeadDim,
                   value + size_t(kPrefix) * kHeadDim, tail_max, tail_sum,
                   tail_output, kBatchedTailRepairTokens,
                   kBatchedTailRepairTokens, 6, 1, 0.0625f, stream),
          "repair first batched triangular-tail tile");
      #endif
    check(cudaEventRecord(batched_tail_finalize_done, stream),
          "record batched-tail finalize done");
    #elif defined(PREFIX_CUBLAS_SLICED_TAIL)
    int cublas_tail_rows = kCublasTailSliceRows;
    check(cudaMemcpyToSymbolAsync(g_rows, &cublas_tail_rows,
                                  sizeof(cublas_tail_rows), 0,
                                  cudaMemcpyHostToDevice, stream),
          "set cuBLAS-tail row count");
    float* cublas_tail_sum_output = qk_sum;
    check(cudaMemcpyToSymbolAsync(g_row_sum_out, &cublas_tail_sum_output,
                                  sizeof(cublas_tail_sum_output), 0,
                                  cudaMemcpyHostToDevice, stream),
          "set cuBLAS-tail PV sum pointer");
    for (auto& operation : cublas_tail_operators) {
      operation.qk->launch(stream);
      int64_t mask_elements = int64_t(kCublasTailSliceTokens) * operation.rows;
      mask_sliced_tail_probability<<<(mask_elements + 255) / 256, 256, 0,
                                     stream>>>(
          reinterpret_cast<__half*>(scores), operation.query_token_start,
          kCublasTailSliceTokens, operation.rows, 6);
      check(cudaGetLastError(), "mask cuBLAS sliced-tail probability");
      operation.pv->launch(stream);
      finalize_cublas_sliced_tail<<<operation.rows, kHeadDim, 0, stream>>>(
          reinterpret_cast<__half*>(tail_output), qk_sum, tail_max, tail_sum,
          operation.query_row_start, operation.rows);
      check(cudaGetLastError(), "finalize cuBLAS sliced tail");
    }
    #elif defined(PREFIX_MATERIALIZED_SLICED_TAIL)
    int sliced_tail_rows = kTailSliceRows;
    check(cudaMemcpyToSymbolAsync(g_rows, &sliced_tail_rows,
                                  sizeof(sliced_tail_rows), 0,
                                  cudaMemcpyHostToDevice, stream),
          "set sliced-tail row count");
    float* sliced_tail_sum_output = qk_sum;
    check(cudaMemcpyToSymbolAsync(g_row_sum_out, &sliced_tail_sum_output,
                                  sizeof(sliced_tail_sum_output), 0,
                                  cudaMemcpyHostToDevice, stream),
          "set sliced-tail PV sum pointer");
    for (auto& operation : tail_operators) {
      check(operation.qk(stream), "launch sliced-tail QK");
      int64_t mask_elements = int64_t(kTailSliceTokens) * operation.rows;
      mask_sliced_tail_probability<<<(mask_elements + 255) / 256, 256, 0,
                                     stream>>>(
          reinterpret_cast<__half*>(scores), operation.query_token_start,
          kTailSliceTokens, operation.rows, 6);
      check(cudaGetLastError(), "mask sliced-tail probability");
      operation.pv->launch(stream);
      finalize_sliced_tail<<<operation.rows, kHeadDim, 0, stream>>>(
          tail_numerator, qk_sum, reinterpret_cast<__half*>(tail_output),
          tail_max, tail_sum, operation.query_row_start, operation.rows);
      check(cudaGetLastError(), "finalize sliced tail");
    }
    #elif defined(PREFIX_CONCURRENT_TAIL)
    check(cudaStreamWaitEvent(stream, concurrent_tail_complete, 0),
          "join concurrent causal tail");
    #else
    check(tail_raw(query, key + size_t(kPrefix) * kHeadDim,
                   value + size_t(kPrefix) * kHeadDim, tail_max, tail_sum,
                   tail_output, kTail, kTail, 6, 1, 0.0625f, stream),
          "launch exact causal tail");
    #endif
    if (tail_done != nullptr) {
      check(cudaEventRecord(tail_done, stream), "record causal tail done");
    }
    #if defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE)
      #if defined(PREFIX_TAIL_FUSE_STATE_IN_MERGE)
    merge_prefix_direct_round_major_tail_fused_state<<<kRows, kHeadDim / 2, 0,
                                                       stream>>>(
        reinterpret_cast<__half const*>(partials), prefix_sum,
        reinterpret_cast<__half const*>(tail_output), tail_max, tail_sum,
        qk_sum, reinterpret_cast<__half*>(final_output), kRows,
        kBatchedTailRepairTokens * 6, kBatchedTailTileRows);
      #else
    merge_prefix_direct_round_major_tail<<<kRows, kHeadDim / 2, 0, stream>>>(
        reinterpret_cast<__half const*>(partials), prefix_sum,
        reinterpret_cast<__half const*>(tail_output), tail_max, tail_sum,
        reinterpret_cast<__half*>(final_output), kRows,
        kBatchedTailRepairTokens * 6);
      #endif
    #elif defined(PREFIX_PV_DIRECT_FP16_ACCUMULATE)
    merge_prefix_half_accumulator_tail<<<kRows, kHeadDim, 0, stream>>>(
        reinterpret_cast<__half const*>(partials), prefix_max, prefix_sum,
        reinterpret_cast<__half const*>(tail_output), tail_max, tail_sum,
        reinterpret_cast<__half*>(final_output), kRows);
    #else
    merge_prefix_accumulator_tail<<<kRows, kHeadDim, 0, stream>>>(
        prefix_accumulator, prefix_max, prefix_sum,
        reinterpret_cast<__half const*>(tail_output), tail_max, tail_sum,
        reinterpret_cast<__half*>(final_output), kRows);
    #endif
  #endif
  };

  launch(nullptr, nullptr, nullptr);
  check(cudaDeviceSynchronize(), "warmup synchronize");
  #if defined(PREFIX_DEBUG_PREFIX_STATE)
  return EXIT_SUCCESS;
  #endif
  cudaEvent_t start;
  cudaEvent_t stop;
  check(cudaEventCreate(&start), "create start");
  check(cudaEventCreate(&stop), "create stop");
  check(cudaEventRecord(start, stream), "record start");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    launch(nullptr, nullptr, nullptr);
  }
  check(cudaEventRecord(stop, stream), "record stop");
  check(cudaEventSynchronize(stop), "synchronize stop");
  float elapsed_ms = 0.0f;
  check(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
  float endpoint_ms = elapsed_ms / float(iterations);
  constexpr double kFrozenUsefulTflop = 60.0 * 0.101581;
  #if defined(PREFIX_FULL_ENDPOINT)
  cudaEvent_t phase_start;
  cudaEvent_t blocks_done;
  cudaEvent_t prefix_done;
  cudaEvent_t tail_done;
  cudaEvent_t phase_stop;
  check(cudaEventCreate(&phase_start), "create phase start");
  check(cudaEventCreate(&blocks_done), "create blocks done");
  check(cudaEventCreate(&prefix_done), "create prefix done");
  check(cudaEventCreate(&tail_done), "create tail done");
  check(cudaEventCreate(&phase_stop), "create phase stop");
  check(cudaEventRecord(phase_start, stream), "record phase start");
  launch(blocks_done, prefix_done, tail_done);
  check(cudaEventRecord(phase_stop, stream), "record phase stop");
  check(cudaEventSynchronize(phase_stop), "synchronize phase stop");
  float blocks_ms = 0.0f;
  float prefix_merge_ms = 0.0f;
  float tail_ms = 0.0f;
  float final_merge_ms = 0.0f;
  check(cudaEventElapsedTime(&blocks_ms, phase_start, blocks_done),
        "measure QK/PV blocks");
  check(cudaEventElapsedTime(&prefix_merge_ms, blocks_done, prefix_done),
        "measure prefix merge");
  check(cudaEventElapsedTime(&tail_ms, prefix_done, tail_done),
        "measure causal tail");
  check(cudaEventElapsedTime(&final_merge_ms, tail_done, phase_stop),
        "measure final merge");
    #if defined(PREFIX_DEBUG_GENERIC_PV_STATE)
  std::vector<float> debug_prefix_sum(192);
  std::vector<__half> debug_prefix_numerator(size_t(192) * 256);
  check(cudaMemcpy(debug_prefix_sum.data(), prefix_sum,
                   debug_prefix_sum.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "copy generic PV prefix sums");
  check(cudaMemcpy(debug_prefix_numerator.data(), partials,
                   debug_prefix_numerator.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost),
        "copy generic PV prefix numerators");
  std::cout << "generic_pv_state";
  for (int base : {0, 64, 128}) {
    for (int offset = 0; offset < 16; ++offset) {
      int index = base + offset;
      std::cout << " r" << index << "{sum=" << debug_prefix_sum[index]
                << ",num="
                << __half2float(debug_prefix_numerator[size_t(index) * 256])
                << "}";
    }
  }
  std::cout << "\n";
    #endif
    #if defined(PREFIX_PHASE_DETAIL) && !defined(PREFIX_QK_PV_PIPELINE) && \
        !defined(PREFIX_QK_ASYNC_EXP_PIPELINE) &&                          \
        !defined(PREFIX_PHASE_BATCHED_MATERIALIZATION)
  float detail_qk_ms = 0.0f;
  float detail_pv_ms = 0.0f;
  float detail_state_ms = 0.0f;
  std::cerr << "prefix_block_phases";
  for (int block = 0; block < blocks; ++block) {
    float block_ms = 0.0f;
    float block_qk_ms = 0.0f;
    float block_pv_ms = 0.0f;
    float block_state_ms = 0.0f;
    check(cudaEventElapsedTime(&block_ms, detail_qk_start[block],
                               detail_qk_done[block]),
          "measure detail QK");
    block_qk_ms = block_ms;
    detail_qk_ms += block_ms;
    check(cudaEventElapsedTime(&block_ms, detail_qk_done[block],
                               detail_pv_done[block]),
          "measure detail PV");
    block_pv_ms = block_ms;
    detail_pv_ms += block_ms;
    check(cudaEventElapsedTime(&block_ms, detail_pv_done[block],
                               detail_state_done[block]),
          "measure detail prefix state");
    block_state_ms = block_ms;
    detail_state_ms += block_ms;
    std::cerr << " b" << block << "{qk=" << block_qk_ms << ",pv=" << block_pv_ms
              << ",state=" << block_state_ms << "}";
  }
  std::cerr << "\n";
    #endif
    #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP) &&                            \
        defined(PREFIX_PHASE_DETAIL) && !defined(PREFIX_QK_PV_PIPELINE) && \
        !defined(PREFIX_QK_ASYNC_EXP_PIPELINE) &&                          \
        !defined(PREFIX_PHASE_BATCHED_MATERIALIZATION)
  float tail_start_delay_ms = 0.0f;
  float tail_qk_overlap_ms = 0.0f;
  float tail_pv_overlap_ms = 0.0f;
  float tail_total_overlap_ms = 0.0f;
  float prefix_from_tail_start_ms = 0.0f;
  check(cudaEventElapsedTime(&tail_start_delay_ms, detail_qk_done[0],
                             tail_overlap_start),
        "measure continuous idle-SM tail start delay");
  check(cudaEventElapsedTime(&tail_qk_overlap_ms, tail_overlap_start,
                             tail_overlap_qk_done),
        "measure continuous idle-SM tail QK");
  check(cudaEventElapsedTime(&tail_pv_overlap_ms, tail_overlap_qk_done,
                             tail_overlap_done),
        "measure continuous idle-SM tail PV");
  check(cudaEventElapsedTime(&tail_total_overlap_ms, tail_overlap_start,
                             tail_overlap_done),
        "measure continuous idle-SM full tail");
  check(cudaEventElapsedTime(&prefix_from_tail_start_ms, tail_overlap_start,
                             tail_overlap_prefix_complete),
        "measure prefix progress during continuous idle-SM tail");
  std::cerr << "idle_sm_tail_continuous"
            << " start=" << tail_start_delay_ms << " qk=" << tail_qk_overlap_ms
            << " pv=" << tail_pv_overlap_ms
            << " total=" << tail_total_overlap_ms
            << " prefix=" << prefix_from_tail_start_ms << " overrun="
            << (tail_total_overlap_ms - prefix_from_tail_start_ms) << "\n";
    #endif
    #if defined(PREFIX_BATCHED_TRI_TAIL)
  float batched_tail_qk_ms = 0.0f;
  float batched_tail_mask_ms = 0.0f;
  float batched_tail_pv_ms = 0.0f;
  float batched_tail_finalize_ms = 0.0f;
  check(cudaEventElapsedTime(&batched_tail_qk_ms, batched_tail_qk_start,
                             batched_tail_qk_done),
        "measure batched-tail QK");
  check(cudaEventElapsedTime(&batched_tail_mask_ms, batched_tail_qk_done,
                             batched_tail_mask_done),
        "measure batched-tail mask");
  check(cudaEventElapsedTime(&batched_tail_pv_ms, batched_tail_mask_done,
                             batched_tail_pv_done),
        "measure batched-tail PV");
  check(cudaEventElapsedTime(&batched_tail_finalize_ms, batched_tail_pv_done,
                             batched_tail_finalize_done),
        "measure batched-tail finalize");
  std::cout << "batched_tail_phases qk=" << batched_tail_qk_ms
            << " mask=" << batched_tail_mask_ms << " pv=" << batched_tail_pv_ms
            << " finalize=" << batched_tail_finalize_ms << "\n";
  std::vector<float> debug_batched_sums(size_t(
      #if defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
                                            kFinePVTasks
      #else
                                            kBatchedTailTasks
      #endif
                                            ) *
                                        kBatchedTailTileRows);
  std::vector<float> debug_tail_state(kRows);
  std::vector<float> debug_prefix_state(kRows);
  std::vector<__half> debug_tail_output(size_t(kRows) * kHeadDim);
  check(cudaMemcpy(debug_batched_sums.data(), qk_sum,
                   debug_batched_sums.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "copy batched-tail debug sums");
  check(cudaMemcpy(debug_tail_state.data(), tail_sum,
                   debug_tail_state.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "copy batched-tail debug state");
  check(cudaMemcpy(debug_prefix_state.data(), prefix_sum,
                   debug_prefix_state.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "copy prefix debug state");
  check(cudaMemcpy(debug_tail_output.data(), tail_output,
                   debug_tail_output.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost),
        "copy batched-tail debug output");
  std::cout << "batched_tail_debug";
  for (int query_tile = 0; query_tile < kBatchedTailTiles; ++query_tile) {
    int global_row = query_tile * kBatchedTailTileRows;
    int first_task = query_tile * (query_tile + 1) / 2;
    std::cout << " q" << query_tile
              << "{prefix=" << debug_prefix_state[global_row]
              << ",tail=" << debug_tail_state[global_row] << ",out0="
              << __half2float(debug_tail_output[size_t(global_row) * kHeadDim]);
      #if defined(PREFIX_BATCHED_TRI_FUSED_PV)
    std::cout << ",fused_sum="
              << debug_batched_sums[size_t(query_tile) * kBatchedTailTileRows];
      #elif defined(PREFIX_TAIL_IDLE_SM_FINE_PV)
    int full_groups = query_tile / kFinePVGroupTiles;
    int residual_queries = query_tile % kFinePVGroupTiles;
    int first_grouped_task =
        kFinePVGroupTiles * full_groups * (full_groups + 1) / 2 +
        residual_queries * (full_groups + 1);
    int grouped_tasks =
        (query_tile + 1 + kFinePVGroupTiles - 1) / kFinePVGroupTiles;
    for (int task_offset = 0; task_offset < grouped_tasks; ++task_offset) {
      std::cout << ",g" << task_offset << "="
                << debug_batched_sums[size_t(first_grouped_task + task_offset) *
                                      kBatchedTailTileRows];
    }
      #else
    for (int key_tile = 0; key_tile <= query_tile; ++key_tile) {
      int task = first_task + key_tile;
      std::cout << ",s" << key_tile << "="
                << debug_batched_sums[size_t(task) * kBatchedTailTileRows];
    }
      #endif
    std::cout << "}";
  }
  std::cout << "\n";
  size_t invalid_tile_sums = 0;
  size_t invalid_tail_state = 0;
  size_t invalid_tail_output = 0;
  size_t first_invalid_tile_sum = debug_batched_sums.size();
  size_t first_invalid_tail_state = debug_tail_state.size();
  size_t first_invalid_tail_output = debug_tail_output.size();
  for (size_t index = 0; index < debug_batched_sums.size(); ++index) {
    if (!std::isfinite(debug_batched_sums[index])) {
      first_invalid_tile_sum = std::min(first_invalid_tile_sum, index);
      ++invalid_tile_sums;
    }
  }
  for (size_t index = 0; index < debug_tail_state.size(); ++index) {
    if (!std::isfinite(debug_tail_state[index])) {
      first_invalid_tail_state = std::min(first_invalid_tail_state, index);
      ++invalid_tail_state;
    }
  }
  for (size_t index = 0; index < debug_tail_output.size(); ++index) {
    if (!std::isfinite(__half2float(debug_tail_output[index]))) {
      first_invalid_tail_output = std::min(first_invalid_tail_output, index);
      ++invalid_tail_output;
    }
  }
  std::cout << "batched_tail_invalid tile_sums=" << invalid_tile_sums
            << " first_tile_sum=" << first_invalid_tile_sum
            << " state=" << invalid_tail_state
            << " first_state=" << first_invalid_tail_state
            << " output=" << invalid_tail_output
            << " first_output=" << first_invalid_tail_output << "\n";
    #endif
  check(tail_raw(query, key, value, tail_max, tail_sum, exact_output, kTail,
                 kTotalKV, 6, 1, 0.0625f, stream),
        "launch monolithic exact reference");
  check(cudaDeviceSynchronize(), "synchronize exact reference");
  size_t output_elements = size_t(kRows) * kHeadDim;
  std::vector<__half> candidate_host(output_elements);
  std::vector<__half> reference_host(output_elements);
  check(cudaMemcpy(candidate_host.data(), final_output,
                   output_elements * sizeof(__half), cudaMemcpyDeviceToHost),
        "copy candidate output");
  check(cudaMemcpy(reference_host.data(), exact_output,
                   output_elements * sizeof(__half), cudaMemcpyDeviceToHost),
        "copy exact output");
  double squared_error = 0.0;
  double squared_reference = 0.0;
  double absolute_error = 0.0;
  float max_absolute_error = 0.0f;
  size_t equal_elements = 0;
  for (size_t index = 0; index < output_elements; ++index) {
    float candidate_value = __half2float(candidate_host[index]);
    float reference_value = __half2float(reference_host[index]);
    float difference = candidate_value - reference_value;
    float absolute = std::fabs(difference);
    squared_error += double(difference) * difference;
    squared_reference += double(reference_value) * reference_value;
    absolute_error += absolute;
    max_absolute_error = std::max(max_absolute_error, absolute);
    equal_elements += candidate_value == reference_value;
  }
  double relative_l2 = std::sqrt(squared_error / squared_reference);
  double mean_absolute_error = absolute_error / double(output_elements);
  double measured_tflops = kFrozenUsefulTflop / (endpoint_ms * 1.0e-3);
  std::cout << "{\n"
            << "  \"block_n\": " << block_n << ",\n"
            << "  \"blocks\": " << blocks << ",\n"
            << "  \"full_attention_ms\": " << endpoint_ms << ",\n"
            << "  \"useful_causal_tflops\": " << measured_tflops << ",\n"
            << "  \"prefix_tokens\": " << kPrefix << ",\n"
            << "  \"causal_tail_tokens\": " << kTail << ",\n"
            << "  \"phases_ms\": {\n"
            << "    \"prefix_qk_pv_blocks\": " << blocks_ms << ",\n"
            << "    \"prefix_state_and_partial_merge\": " << prefix_merge_ms
            << ",\n"
            << "    \"exact_causal_tail\": " << tail_ms << ",\n"
            << "    \"prefix_tail_merge\": " << final_merge_ms
    #if defined(PREFIX_PHASE_DETAIL) && !defined(PREFIX_QK_PV_PIPELINE) && \
        !defined(PREFIX_QK_ASYNC_EXP_PIPELINE) &&                          \
        !defined(PREFIX_PHASE_BATCHED_MATERIALIZATION)
            << ",\n"
            << "    \"detail_qk\": " << detail_qk_ms << ",\n"
            << "    \"detail_pv\": " << detail_pv_ms << ",\n"
            << "    \"detail_prefix_state\": " << detail_state_ms << "\n"
    #else
            << "\n"
    #endif
            << "  },\n"
            << "  \"numerical\": {\n"
            << "    \"max_abs_diff\": " << max_absolute_error << ",\n"
            << "    \"mean_abs_diff\": " << mean_absolute_error << ",\n"
            << "    \"relative_l2\": " << relative_l2 << ",\n"
            << "    \"equal_elements\": " << equal_elements << ",\n"
            << "    \"elements\": " << output_elements << "\n"
            << "  },\n"
            << "  \"partial_storage_gib\": "
            << double(partial_elements * sizeof(Element)) / double(1ull << 30)
            << ",\n"
            << "  \"note\": \"measured serial prefix + exact tail state + "
               "final merge endpoint\"\n"
            << "}\n";
  check(cudaEventDestroy(phase_start), "destroy phase start");
  check(cudaEventDestroy(blocks_done), "destroy blocks done");
  check(cudaEventDestroy(prefix_done), "destroy prefix done");
  check(cudaEventDestroy(tail_done), "destroy tail done");
  check(cudaEventDestroy(phase_stop), "destroy phase stop");
    #if defined(PREFIX_PHASE_DETAIL) && !defined(PREFIX_QK_PV_PIPELINE) && \
        !defined(PREFIX_QK_ASYNC_EXP_PIPELINE) &&                          \
        !defined(PREFIX_PHASE_BATCHED_MATERIALIZATION)
  for (int block = 0; block < blocks; ++block) {
    check(cudaEventDestroy(detail_qk_start[block]), "destroy detail QK start");
    check(cudaEventDestroy(detail_qk_done[block]), "destroy detail QK done");
    check(cudaEventDestroy(detail_pv_done[block]), "destroy detail PV done");
    check(cudaEventDestroy(detail_state_done[block]),
          "destroy detail state done");
  }
    #endif
    #if defined(PREFIX_BATCHED_TRI_TAIL)
  check(cudaEventDestroy(batched_tail_qk_start),
        "destroy batched-tail QK start");
  check(cudaEventDestroy(batched_tail_qk_done), "destroy batched-tail QK done");
  check(cudaEventDestroy(batched_tail_mask_done),
        "destroy batched-tail mask done");
  check(cudaEventDestroy(batched_tail_pv_done), "destroy batched-tail PV done");
  check(cudaEventDestroy(batched_tail_finalize_done),
        "destroy batched-tail finalize done");
    #endif
  #else
  double projected_full_ms = endpoint_ms + 4.83;
  double projected_tflops = kFrozenUsefulTflop / (projected_full_ms * 1.0e-3);
  std::cout
      << "{\n"
      << "  \"block_n\": " << block_n << ",\n"
      << "  \"blocks\": " << blocks << ",\n"
      << "  \"prefix_ms_including_merge\": " << endpoint_ms << ",\n"
      << "  \"accepted_tail_ms\": 4.83,\n"
      << "  \"projected_full_ms\": " << projected_full_ms << ",\n"
      << "  \"projected_useful_tflops\": " << projected_tflops << ",\n"
      << "  \"partial_storage_gib\": "
      << double(partial_elements * sizeof(Element)) / double(1ull << 30)
      << ",\n"
      << "  \"note\": \"prefix endpoint; prefix/tail state merge excluded\"\n"
      << "}\n";
  #endif

  check(cudaEventDestroy(start), "destroy start");
  check(cudaEventDestroy(stop), "destroy stop");
  #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  check(cudaEventDestroy(tail_overlap_ready),
        "destroy idle-SM tail-ready event");
  check(cudaEventDestroy(tail_overlap_prefix_pv_ready),
        "destroy idle-SM first-prefix-PV event");
  check(cudaEventDestroy(tail_overlap_start),
        "destroy idle-SM tail-start event");
  check(cudaEventDestroy(tail_overlap_qk_done),
        "destroy idle-SM tail-QK-done event");
  check(cudaEventDestroy(tail_overlap_done), "destroy idle-SM tail-done event");
  check(cudaEventDestroy(tail_overlap_prefix_complete),
        "destroy idle-SM prefix-complete event");
  for (int group = 0; group < 8; ++group) {
    check(cudaEventDestroy(tail_overlap_pv_ready[group]),
          "destroy idle-SM grouped-PV-ready event");
    check(cudaEventDestroy(tail_overlap_pv_done[group]),
          "destroy idle-SM grouped-PV-done event");
  }
  check(cublasDestroy(tail_overlap_cublas),
        "destroy idle-SM tail cuBLAS handle");
  check(cudaStreamDestroy(tail_overlap_stream), "destroy idle-SM tail stream");
  check(cudaStreamDestroy(stream), "destroy high-priority prefix stream");
  #endif
  #if defined(PREFIX_QK_ASYNC_EXP_PIPELINE)
  for (int buffer = 0; buffer < kPipelineBuffers; ++buffer) {
    check(cudaEventDestroy(async_qk_ready[buffer]),
          "destroy async QK-ready event");
    check(cudaEventDestroy(async_exp_ready[buffer]),
          "destroy async exp-ready event");
    check(cudaEventDestroy(async_pv_done[buffer]),
          "destroy async PV-done event");
  }
  check(cudaStreamDestroy(exp_stream), "destroy async exp stream");
  #endif
  #if defined(PREFIX_QK_PV_PIPELINE)
  for (int buffer = 0; buffer < kPipelineBuffers; ++buffer) {
    check(cudaEventDestroy(qk_ready[buffer]), "destroy QK-ready event");
    check(cudaEventDestroy(pv_done[buffer]), "destroy PV-done event");
  }
  check(cudaStreamDestroy(qk_stream), "destroy QK pipeline stream");
  check(cudaStreamDestroy(pv_stream), "destroy PV pipeline stream");
  #endif
  check(cudaFree(query), "free Q");
  check(cudaFree(key), "free K");
  check(cudaFree(value), "free V");
  #if defined(PREFIX_QK_PRETRANSPOSE_INPUTS)
  check(cudaFree(query_transposed), "free transposed Q");
  check(cudaFree(key_transposed), "free transposed prefix K");
  #endif
  check(cudaFree(scores), "free scores");
  check(cudaFree(partials), "free partials");
  check(cudaFree(output), "free output");
  check(cudaFree(qk_norm), "free QK max");
  check(cudaFree(qk_sum), "free QK sum");
  check(cudaFree(prefix_accumulator), "free prefix accumulator");
  check(cudaFree(prefix_max), "free prefix max");
  check(cudaFree(prefix_sum), "free prefix sum");
  check(cudaFree(old_scales), "free old scales");
  check(cudaFree(block_scales), "free block scales");
  #if defined(PREFIX_QK_CUBLAS_RAW)
  check(cublasDestroy(qk_cublas), "destroy cuBLAS QK handle");
  #endif
  #if defined(PREFIX_FULL_ENDPOINT)
    #if defined(PREFIX_CONCURRENT_TAIL)
  check(cudaEventDestroy(concurrent_tail_ready),
        "destroy concurrent tail ready event");
  check(cudaEventDestroy(concurrent_tail_complete),
        "destroy concurrent tail completion event");
  check(cudaStreamDestroy(concurrent_tail_stream),
        "destroy concurrent tail stream");
    #endif
  check(cudaFree(tail_output), "free tail output");
  check(cudaFree(final_output), "free final output");
  check(cudaFree(exact_output), "free exact output");
  check(cudaFree(tail_max), "free tail max");
  check(cudaFree(tail_sum), "free tail sum");
    #if defined(PREFIX_BATCHED_TRI_TAIL)
      #if !defined(PREFIX_BATCHED_TRI_FUSED_PV) && \
          !defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE)
  check(cudaFree(batched_tail_partials),
        "free batched-tail partial numerators");
      #endif
  check(cudaFree(batched_tail_pv_params), "free batched-tail PV params");
      #if defined(PREFIX_TAIL_IDLE_SM_OVERLAP)
  check(cudaFree(tail_overlap_scores), "free idle-SM tail score tile");
        #if defined(PREFIX_TAIL_WAVE_GATED)
  check(cudaFree(tail_wave_completed),
        "free prefix-PV wave completion counters");
  check(cudaFree(tail_wave_ready), "free prefix-PV wave ready flags");
        #endif
      #endif
      #if defined(PREFIX_BATCHED_TRI_CUTLASS_QK)
  check(cudaFree(batched_tail_qk_params), "free batched-tail QK params");
      #endif
  check(cudaFree(batched_tail_q_ptrs), "free batched-tail Q pointer array");
  check(cudaFree(batched_tail_k_ptrs), "free batched-tail K pointer array");
  check(cudaFree(batched_tail_score_ptrs),
        "free batched-tail score pointer array");
    #endif
    #if defined(PREFIX_MATERIALIZED_SLICED_TAIL)
  check(cudaFree(tail_numerator), "free sliced-tail numerator");
    #endif
  #endif
  return 0;
}
#else

extern "C" cudaError_t onecat_sm70_d256_dense_state_raw(
    const void*, const void*, const void*, float*, float*, void*, int, int, int,
    int, float, cudaStream_t);

namespace FLASH_NAMESPACE {

  #if PREFIX_TORCH_QUERY_TOKENS == 8192
extern "C" int64_t onecat_sm70_q8192_accumulation_bits() {
  #else
extern "C" int64_t onecat_sm70_q8000_accumulation_bits() {
  #endif
  #if defined(PREFIX_QK_CUBLAS_FP32_ACCUM) && \
      defined(PREFIX_PV_FP32_MMA_ACCUMULATE)
  return 32;
  #else
  return 16;
  #endif
}

struct Sm70GqaScoreWorkspace {
  at::Tensor scores;
  cudaEvent_t completion = nullptr;
  bool completion_recorded = false;
  std::mutex launch_mutex;

  Sm70GqaScoreWorkspace(const at::Tensor& q, int rows, int block_n)
      : scores(at::empty({rows, block_n}, q.options())) {
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(&completion, cudaEventDisableTiming));
  }

  ~Sm70GqaScoreWorkspace() {
    if (completion != nullptr) {
      cudaEventDestroy(completion);
    }
  }
};

using Sm70GqaScoreWorkspacePtr = std::shared_ptr<Sm70GqaScoreWorkspace>;

std::mutex& sm70_gqa_score_cache_mutex() {
  static std::mutex cache_mutex;
  return cache_mutex;
}

std::map<int, Sm70GqaScoreWorkspacePtr>& sm70_gqa_score_cache() {
  static std::map<int, Sm70GqaScoreWorkspacePtr> cache;
  return cache;
}

Sm70GqaScoreWorkspacePtr get_sm70_gqa_score_workspace(const at::Tensor& q,
                                                      int rows, int block_n) {
  std::lock_guard<std::mutex> lock(sm70_gqa_score_cache_mutex());
  int device = q.get_device();
  auto& cache = sm70_gqa_score_cache();
  auto& workspace = cache[device];
  if (!workspace) {
    workspace = std::make_shared<Sm70GqaScoreWorkspace>(q, rows, block_n);
  }
  return workspace;
}

Sm70GqaScoreWorkspacePtr find_sm70_gqa_score_workspace(int device) {
  std::lock_guard<std::mutex> lock(sm70_gqa_score_cache_mutex());
  auto& cache = sm70_gqa_score_cache();
  auto found = cache.find(device);
  return found == cache.end() ? nullptr : found->second;
}

  #if defined(PREFIX_QK_CUBLAS_RAW) && defined(PREFIX_BATCHED_TRI_TAIL) && \
      defined(PREFIX_TAIL_IDLE_SM_FINE_PV) &&                              \
      defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE)

struct Sm70GqaHalf2Runtime {
  cublasHandle_t prefix_cublas = nullptr;
  cublasHandle_t tail_cublas = nullptr;
  cudaStream_t prefix_stream = nullptr;
  cudaStream_t tail_stream = nullptr;
  cudaEvent_t input_ready = nullptr;
  cudaEvent_t prefix_pv_ready = nullptr;
  cudaEvent_t tail_done = nullptr;
  cudaEvent_t completion = nullptr;

  Sm70GqaHalf2Runtime() {
    try {
      int least_priority = 0;
      int greatest_priority = 0;
      C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority,
                                                      &greatest_priority));
      C10_CUDA_CHECK(cudaStreamCreateWithPriority(
          &prefix_stream, cudaStreamNonBlocking, greatest_priority));
      C10_CUDA_CHECK(cudaStreamCreateWithPriority(
          &tail_stream, cudaStreamNonBlocking, least_priority));
      C10_CUDA_CHECK(
          cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming));
      C10_CUDA_CHECK(
          cudaEventCreateWithFlags(&prefix_pv_ready, cudaEventDisableTiming));
      C10_CUDA_CHECK(
          cudaEventCreateWithFlags(&tail_done, cudaEventDisableTiming));
      C10_CUDA_CHECK(
          cudaEventCreateWithFlags(&completion, cudaEventDisableTiming));
      TORCH_CHECK(cublasCreate(&prefix_cublas) == CUBLAS_STATUS_SUCCESS,
                  "create SM70 half2 prefix cuBLAS handle failed");
      TORCH_CHECK(cublasCreate(&tail_cublas) == CUBLAS_STATUS_SUCCESS,
                  "create SM70 half2 tail cuBLAS handle failed");
      TORCH_CHECK(cublasSetMathMode(prefix_cublas, CUBLAS_TENSOR_OP_MATH) ==
                      CUBLAS_STATUS_SUCCESS,
                  "enable SM70 half2 prefix tensor-op math failed");
      TORCH_CHECK(cublasSetMathMode(tail_cublas, CUBLAS_TENSOR_OP_MATH) ==
                      CUBLAS_STATUS_SUCCESS,
                  "enable SM70 half2 tail tensor-op math failed");
    } catch (...) {
      release();
      throw;
    }
  }

  void release() noexcept {
    if (prefix_cublas != nullptr) {
      cublasDestroy(prefix_cublas);
      prefix_cublas = nullptr;
    }
    if (tail_cublas != nullptr) {
      cublasDestroy(tail_cublas);
      tail_cublas = nullptr;
    }
    if (input_ready != nullptr) {
      cudaEventDestroy(input_ready);
      input_ready = nullptr;
    }
    if (prefix_pv_ready != nullptr) {
      cudaEventDestroy(prefix_pv_ready);
      prefix_pv_ready = nullptr;
    }
    if (tail_done != nullptr) {
      cudaEventDestroy(tail_done);
      tail_done = nullptr;
    }
    if (completion != nullptr) {
      cudaEventDestroy(completion);
      completion = nullptr;
    }
    if (tail_stream != nullptr) {
      cudaStreamDestroy(tail_stream);
      tail_stream = nullptr;
    }
    if (prefix_stream != nullptr) {
      cudaStreamDestroy(prefix_stream);
      prefix_stream = nullptr;
    }
  }

  ~Sm70GqaHalf2Runtime() { release(); }
};

// Kernel arguments are captured by value; stack-backed host-to-symbol copies
// instead leave dangling source addresses when a CUDA Graph is replayed.
__global__ void set_half2_runtime_globals(int rows, float* row_sum,
                                          int tail_rows, float* tail_sum,
                                          const float* row_max,
                                          const float* tail_max,
                                          int* prefix_outliers) {
  g_rows = rows;
  g_row_sum_out = row_sum;
  g_tail_rows = tail_rows;
  g_tail_row_sum_out = tail_sum;
  g_pv_task_base = 0;
    #if defined(PREFIX_TORCH_STABLE_ROWS)
  g_row_max = row_max;
  g_79t_tail_row_max = tail_max;
  g_79t_prefix_outliers = prefix_outliers;
    #endif
}

struct Sm70GqaHalf2Workspace {
  static constexpr int kQuery = PREFIX_TORCH_QUERY_TOKENS;
  static constexpr int kRows = kQuery * 6;
  static constexpr int kHeadDim = 256;
  static constexpr int kMinTotalKV = kQuery;
  static constexpr int kMaxTotalKV = 262144;
  static constexpr int kBlockN = PREFIX_TORCH_BLOCK_N;
  static constexpr int kTailTileTokens = PREFIX_BATCHED_TAIL_TILE_TOKENS;
  static constexpr int kTailTiles = kQuery / kTailTileTokens;
  static constexpr int kTailTileRows = kTailTileTokens * 6;
  static constexpr int kTailTasks = kTailTiles * (kTailTiles + 1) / 2;
  static constexpr int kFinePVGroupTiles = PREFIX_TAIL_FINE_PV_GROUP_TILES;
  static constexpr int kFinePVRounds =
      (kTailTiles + kFinePVGroupTiles - 1) / kFinePVGroupTiles;
  static constexpr int kFinePVTasks =
      kFinePVGroupTiles * (kTailTiles / kFinePVGroupTiles) *
          (kTailTiles / kFinePVGroupTiles + 1) / 2 +
      (kTailTiles % kFinePVGroupTiles) * (kTailTiles / kFinePVGroupTiles + 1);
  static constexpr size_t kTailScoreElements =
      size_t(kTailTileRows) * kTailTileTokens * kTailTasks;
  static constexpr size_t kTailAllocationHeadroom = 2ull << 30;

  std::shared_ptr<onecat_sm70_prefill::ScoreWorkspace> shared_scores;
  Sm70GqaHalf2Runtime runtime;
  cublasHandle_t& prefix_cublas;
  cublasHandle_t& tail_cublas;
  cudaStream_t& prefix_stream;
  cudaStream_t& tail_stream;
  cudaEvent_t& input_ready;
  cudaEvent_t& prefix_pv_ready;
  cudaEvent_t& tail_done;
  cudaEvent_t& completion;
  at::Tensor query_transposed;
  at::Tensor key_transposed;
  at::Tensor scores;
  at::Tensor prefix_numerator;
  at::Tensor prefix_sum;
  at::Tensor tail_scores;
  at::Tensor tail_numerator;
  at::Tensor tail_row_sums;
  at::Tensor tail_max;
  at::Tensor tail_sum;
  at::Tensor tail_q_ptrs;
  at::Tensor tail_k_ptrs;
  at::Tensor tail_score_ptrs;
  at::Tensor tail_pv_params;

    #if defined(PREFIX_TORCH_STABLE_ROWS)
  at::Tensor value_scaled, value_center, value_max, block_sum, block_max,
      prefix_max;
  at::Tensor prefix_accumulator, max_partials, tail_max_partials, prefix_outliers;
    #endif
  // Graph memcpy nodes retain their host source addresses. Keep metadata
  // immutable for each KV length/value buffer, including after later captures.
  struct TailMetadata {
    std::vector<Element*> host_tail_q_ptrs;
    std::vector<Element*> host_tail_k_ptrs;
    std::vector<ScoreElement*> host_tail_score_ptrs;
    std::vector<TailPVKernel::Params> host_tail_pv_params;
    dim3 pv_grid;
    dim3 pv_block;
    int pv_smem_bytes = 0;
  };
  std::map<std::pair<int, const Element*>, TailMetadata> host_tail_metadata;
  bool concurrent_tail_scores = false;
  std::mutex& launch_mutex;

  explicit Sm70GqaHalf2Workspace(const at::Tensor& q)
      : shared_scores(onecat_sm70_prefill::get_score_workspace(q, kBlockN)),
        prefix_cublas(runtime.prefix_cublas),
        tail_cublas(runtime.tail_cublas),
        prefix_stream(runtime.prefix_stream),
        tail_stream(runtime.tail_stream),
        input_ready(runtime.input_ready),
        prefix_pv_ready(runtime.prefix_pv_ready),
        tail_done(runtime.tail_done),
        completion(runtime.completion),
        query_transposed(at::empty({kHeadDim, kRows}, q.options())),
        key_transposed(at::empty({kHeadDim, kMaxTotalKV}, q.options())),
        scores(
            shared_scores->scores.narrow(0, 0, shared_scores->block_n * kRows)
                .view({shared_scores->block_n, kRows})),
        prefix_numerator(at::empty({kRows, kHeadDim},
    #if defined(PREFIX_TORCH_PREFIX_FP32_OUTPUT)
                                   q.options().dtype(at::ScalarType::Float))),
    #else
                                   q.options())),
    #endif
        prefix_sum(
            at::empty({kRows}, q.options().dtype(at::ScalarType::Float))),
        tail_scores(),
        tail_numerator(at::empty({kRows, kHeadDim}, q.options())),
        tail_row_sums(at::empty({kFinePVTasks, kTailTileRows},
                                q.options().dtype(at::ScalarType::Float))),
        tail_max(at::empty({kRows}, q.options().dtype(at::ScalarType::Float))),
        tail_sum(at::empty({kRows}, q.options().dtype(at::ScalarType::Float))),
        tail_q_ptrs(
            at::empty({static_cast<int64_t>(kTailTasks * sizeof(Element*))},
                      q.options().dtype(at::ScalarType::Byte))),
        tail_k_ptrs(
            at::empty({static_cast<int64_t>(kTailTasks * sizeof(Element*))},
                      q.options().dtype(at::ScalarType::Byte))),
        tail_score_ptrs(at::empty(
            {static_cast<int64_t>(kTailTasks * sizeof(ScoreElement*))},
            q.options().dtype(at::ScalarType::Byte))),
        tail_pv_params(at::empty(
            {static_cast<int64_t>(kFinePVTasks * sizeof(TailPVKernel::Params))},
            q.options().dtype(at::ScalarType::Byte))),
        launch_mutex(shared_scores->mutex) {
    static_assert(kQuery % kTailTileTokens == 0);
    static_assert(
        (kQuery == 8000 && kTailTileTokens == 320 && kTailTiles == 25) ||
        (kQuery == 8192 && kTailTileTokens == 256 && kTailTiles == 32));
    static_assert(kFinePVGroupTiles == 4);
    static_assert(kFinePVTasks == (kQuery == 8000 ? 91 : 144));
    static_assert(kBlockN >= 8192 && kBlockN <= 16 * 8192 &&
                  kBlockN % 8192 == 0);
    static_assert(kTailScoreElements <= size_t(kBlockN) * kRows);
    static_assert(PVThreadblockShape::kM == 64 ||
                  PVThreadblockShape::kM == 96 ||
                  PVThreadblockShape::kM == 128);
    static_assert(PVThreadblockShape::kN == 128 ||
                  PVThreadblockShape::kN == 256);
    static_assert(PVDefaultKernel::kThreadCount == 128 ||
                  PVDefaultKernel::kThreadCount == 256 ||
                  PVDefaultKernel::kThreadCount == 384 ||
                  PVDefaultKernel::kThreadCount == 512);

    constexpr int kTailPVSmemBytes =
        int(sizeof(typename TailPVKernel::SharedStorage));
    if (kTailPVSmemBytes >= 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          batched_tri_tail_pv_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kTailPVSmemBytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          cutlass::Kernel<TailPVKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kTailPVSmemBytes));
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t tail_score_bytes = kTailScoreElements * sizeof(ScoreElement);
    const char* serial_tail = std::getenv("PREFIX_TORCH_SERIAL_TAIL");
    bool force_serial_tail =
        serial_tail == nullptr || std::strcmp(serial_tail, "0") != 0;
    concurrent_tail_scores =
        !force_serial_tail &&
        free_bytes >= tail_score_bytes + kTailAllocationHeadroom;
    if (concurrent_tail_scores) {
      tail_scores =
          at::empty({static_cast<int64_t>(kTailScoreElements)}, q.options());
    } else {
      tail_scores = scores;
    }

    #if defined(PREFIX_TORCH_STABLE_ROWS)
    auto fp32 = q.options().dtype(at::ScalarType::Float);
    value_scaled = at::empty({kMaxTotalKV, kHeadDim}, q.options());
    value_center = at::empty({kHeadDim}, fp32);
    value_max = at::empty({1}, fp32);
    block_sum = at::empty({kRows}, fp32);
    block_max = at::empty({kRows}, fp32);
    prefix_max = at::empty({kRows}, fp32);
    prefix_accumulator = at::empty({kRows, kHeadDim}, fp32);
    max_partials = at::empty({16, kRows}, fp32);
    tail_max_partials = at::empty({16, kRows}, fp32);
    prefix_outliers = at::empty(
        {(kRows + PVThreadblockShape::kM - 1) / PVThreadblockShape::kM},
        q.options().dtype(at::ScalarType::Int));
    #endif
  }

  ~Sm70GqaHalf2Workspace() = default;
};

using Sm70GqaHalf2WorkspacePtr = std::shared_ptr<Sm70GqaHalf2Workspace>;

std::mutex& sm70_gqa_half2_cache_mutex() {
  static std::mutex cache_mutex;
  return cache_mutex;
}

std::map<int, Sm70GqaHalf2WorkspacePtr>& sm70_gqa_half2_cache() {
  static std::map<int, Sm70GqaHalf2WorkspacePtr> cache;
  return cache;
}

Sm70GqaHalf2WorkspacePtr get_sm70_gqa_half2_workspace(const at::Tensor& q) {
  std::lock_guard<std::mutex> lock(sm70_gqa_half2_cache_mutex());
  auto& workspace = sm70_gqa_half2_cache()[q.get_device()];
  if (!workspace) {
    workspace = std::make_shared<Sm70GqaHalf2Workspace>(q);
  }
  return workspace;
}

at::Tensor sm70_d256_gqa_half2_family_fwd(const at::Tensor& q,
                                          const at::Tensor& k,
                                          const at::Tensor& v,
                                          at::Tensor& out) {
  using Workspace = Sm70GqaHalf2Workspace;
  const int total_kv = static_cast<int>(k.size(1));
  const int prefix = total_kv - Workspace::kQuery;
  auto workspace = get_sm70_gqa_half2_workspace(q);
  const int block_n = static_cast<int>(workspace->shared_scores->block_n);
  std::unique_lock<std::mutex> launch_lock(workspace->launch_mutex);
  cudaStream_t caller_stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t prefix_stream = workspace->prefix_stream;
  cudaStream_t tail_stream = workspace->tail_stream;
  bool exact_tail_debug = std::getenv("PREFIX_TORCH_EXACT_TAIL") != nullptr;
  bool dump_tail_debug = std::getenv("PREFIX_TORCH_DUMP_TAIL") != nullptr;
  bool direct_tail_debug = std::getenv("PREFIX_TORCH_DIRECT_TAIL") != nullptr;
  bool concurrent_tail_scores = workspace->concurrent_tail_scores;

  auto* query = reinterpret_cast<Element*>(q.data_ptr<at::Half>());
  auto* key = reinterpret_cast<Element*>(k.data_ptr<at::Half>());
  auto* value = reinterpret_cast<Element*>(v.data_ptr<at::Half>());
  auto* output = reinterpret_cast<Element*>(out.data_ptr<at::Half>());
  auto* query_transposed = reinterpret_cast<Element*>(
      workspace->query_transposed.data_ptr<at::Half>());
  auto* key_transposed = reinterpret_cast<Element*>(
      workspace->key_transposed.data_ptr<at::Half>());
  auto* scores =
      reinterpret_cast<ScoreElement*>(workspace->scores.data_ptr<at::Half>());
    #if defined(PREFIX_TORCH_PREFIX_FP32_OUTPUT)
  float* prefix_numerator = workspace->prefix_numerator.data_ptr<float>();
    #else
  auto* prefix_numerator = reinterpret_cast<Element*>(
      workspace->prefix_numerator.data_ptr<at::Half>());
    #endif
  float* prefix_sum = workspace->prefix_sum.data_ptr<float>();
  auto* tail_scores = reinterpret_cast<ScoreElement*>(
      workspace->tail_scores.data_ptr<at::Half>());
  auto* tail_numerator = reinterpret_cast<Element*>(
      workspace->tail_numerator.data_ptr<at::Half>());
  float* tail_row_sums = workspace->tail_row_sums.data_ptr<float>();
  float* tail_max = workspace->tail_max.data_ptr<float>();
  float* tail_sum = workspace->tail_sum.data_ptr<float>();
  auto* tail_pv_params = reinterpret_cast<TailPVKernel::Params*>(
      workspace->tail_pv_params.data_ptr<uint8_t>());

  cudaStreamCaptureStatus capture_status;
  C10_CUDA_CHECK(cudaStreamIsCapturing(caller_stream, &capture_status));
  const bool capturing = capture_status == cudaStreamCaptureStatusActive;
  if (workspace->shared_scores->completion_recorded) {
    // The previous call can belong to another stream or graph. Explicit event
    // nodes preserve that dependency instead of importing uncaptured work.
    C10_CUDA_CHECK(cudaStreamWaitEvent(caller_stream,
                                       workspace->shared_scores->completion,
                                       capturing ? cudaEventWaitExternal : 0));
  }
  C10_CUDA_CHECK(cudaEventRecord(workspace->input_ready, caller_stream));
  C10_CUDA_CHECK(cudaStreamWaitEvent(prefix_stream, workspace->input_ready, 0));
  C10_CUDA_CHECK(cudaMemsetAsync(
      prefix_sum, 0, Workspace::kRows * sizeof(float), prefix_stream));
  C10_CUDA_CHECK(cudaMemsetAsync(tail_row_sums, 0,
                                 size_t(Workspace::kFinePVTasks) *
                                     Workspace::kTailTileRows * sizeof(float),
                                 prefix_stream));

    #if defined(PREFIX_TORCH_STABLE_ROWS)
  float* block_sum = workspace->block_sum.data_ptr<float>();
  float* block_max = workspace->block_max.data_ptr<float>();
  float* prefix_max = workspace->prefix_max.data_ptr<float>();
  float* prefix_accumulator = workspace->prefix_accumulator.data_ptr<float>();
  float* maximum_value = workspace->value_max.data_ptr<float>();
  float* value_center = workspace->value_center.data_ptr<float>();
  auto* scaled_value =
      reinterpret_cast<__half*>(workspace->value_scaled.data_ptr<at::Half>());
  C10_CUDA_CHECK(cudaMemsetAsync(block_sum, 0, Workspace::kRows * sizeof(float),
                                 prefix_stream));
  C10_CUDA_CHECK(
      cudaMemsetAsync(maximum_value, 0, sizeof(float), prefix_stream));
  stable_value_center<<<1, 256, 0, prefix_stream>>>(
      reinterpret_cast<__half const*>(value), value_center, total_kv);
  stable_value_amax<<<1024, 256, 0, prefix_stream>>>(
      reinterpret_cast<__half const*>(value), value_center, maximum_value,
      total_kv * Workspace::kHeadDim);
  stable_scale_values<<<1024, 256, 0, prefix_stream>>>(
      reinterpret_cast<__half const*>(value), scaled_value, value_center,
      maximum_value, total_kv * Workspace::kHeadDim);
  value = reinterpret_cast<Element*>(scaled_value);
    #endif
    #if defined(PREFIX_TORCH_STABLE_ROWS)
  float* prefix_sum_output = block_sum;
  const float* row_max_output = block_max;
    #else
  float* prefix_sum_output = prefix_sum;
  const float* row_max_output = nullptr;
    #endif
  set_half2_runtime_globals<<<1, 1, 0, prefix_stream>>>(
      Workspace::kRows, prefix_sum_output, Workspace::kTailTileRows,
      tail_row_sums, row_max_output, tail_max,
    #if defined(PREFIX_TORCH_STABLE_ROWS)
      workspace->prefix_outliers.data_ptr<int>()
    #else
      nullptr
    #endif
      );

  dim3 transpose_threads(32, 8);
  dim3 query_transpose_grid((Workspace::kHeadDim + 31) / 32,
                            (Workspace::kRows + 31) / 32);
  dim3 key_transpose_grid((Workspace::kHeadDim + 31) / 32,
                          (total_kv + 31) / 32);
  transpose_half_32x32<<<query_transpose_grid, transpose_threads, 0,
                         prefix_stream>>>(
      reinterpret_cast<__half const*>(query),
      reinterpret_cast<__half*>(query_transposed), Workspace::kRows,
      Workspace::kHeadDim);
  transpose_half_32x32<<<key_transpose_grid, transpose_threads, 0,
                         prefix_stream>>>(
      reinterpret_cast<__half const*>(key),
      reinterpret_cast<__half*>(key_transposed), total_kv, Workspace::kHeadDim);

  auto& tail_metadata = workspace->host_tail_metadata[{total_kv, value}];
  if (tail_metadata.host_tail_q_ptrs.empty()) {
    tail_metadata.host_tail_q_ptrs.reserve(Workspace::kTailTasks);
    tail_metadata.host_tail_k_ptrs.reserve(Workspace::kTailTasks);
    tail_metadata.host_tail_score_ptrs.reserve(Workspace::kTailTasks);
    size_t query_score_offset = 0;
    for (int query_tile = 0; query_tile < Workspace::kTailTiles; ++query_tile) {
      auto* query_scores = tail_scores + query_score_offset;
      for (int key_tile = 0; key_tile <= query_tile; ++key_tile) {
        tail_metadata.host_tail_q_ptrs.push_back(
            query_transposed + size_t(query_tile) * Workspace::kTailTileRows);
        tail_metadata.host_tail_k_ptrs.push_back(
            key_transposed + prefix +
            size_t(key_tile) * Workspace::kTailTileTokens);
        tail_metadata.host_tail_score_ptrs.push_back(
            query_scores + size_t(key_tile) * Workspace::kTailTileRows *
                               Workspace::kTailTileTokens);
      }
      query_score_offset += size_t(Workspace::kTailTileRows) *
                            (query_tile + 1) * Workspace::kTailTileTokens;
    }
  }
  C10_CUDA_CHECK(
      cudaMemcpyAsync(workspace->tail_q_ptrs.data_ptr<uint8_t>(),
                      tail_metadata.host_tail_q_ptrs.data(),
                      tail_metadata.host_tail_q_ptrs.size() * sizeof(Element*),
                      cudaMemcpyHostToDevice, prefix_stream));
  C10_CUDA_CHECK(
      cudaMemcpyAsync(workspace->tail_k_ptrs.data_ptr<uint8_t>(),
                      tail_metadata.host_tail_k_ptrs.data(),
                      tail_metadata.host_tail_k_ptrs.size() * sizeof(Element*),
                      cudaMemcpyHostToDevice, prefix_stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      workspace->tail_score_ptrs.data_ptr<uint8_t>(),
      tail_metadata.host_tail_score_ptrs.data(),
      tail_metadata.host_tail_score_ptrs.size() * sizeof(ScoreElement*),
      cudaMemcpyHostToDevice, prefix_stream));

  std::vector<std::unique_ptr<CublasQKLauncher>> prefix_qk;
    #if defined(PREFIX_TORCH_PREFIX_FP32_OUTPUT)
  std::vector<std::unique_ptr<PrefixFloatPVLauncher>> prefix_pv;
    #else
  std::vector<std::unique_ptr<PVLauncher>> prefix_pv;
    #endif
  const int kPrefixBlocks = (prefix + block_n - 1) / block_n;
  prefix_qk.reserve(kPrefixBlocks);
  prefix_pv.reserve(kPrefixBlocks);
  for (int block = 0; block < kPrefixBlocks; ++block) {
    int begin = block * block_n;
    int width = std::min(block_n, prefix - begin);
    prefix_qk.push_back(std::make_unique<CublasQKLauncher>(CublasQKLauncher{
        workspace->prefix_cublas, query_transposed, key_transposed + begin,
        scores, Workspace::kRows, width, Workspace::kRows, total_kv}));
    #if defined(PREFIX_TORCH_PREFIX_FP32_OUTPUT)
    prefix_pv.push_back(std::make_unique<PrefixFloatPVLauncher>(
    #else
    prefix_pv.push_back(std::make_unique<PVLauncher>(
    #endif
        scores, value + size_t(begin) * Workspace::kHeadDim, prefix_numerator,
        Workspace::kRows, width,
    #if defined(PREFIX_TORCH_STABLE_ROWS)
        false));
    #else
        block != 0));
    #endif
  }

  if (tail_metadata.host_tail_pv_params.empty()) {
    tail_metadata.host_tail_pv_params.reserve(Workspace::kFinePVTasks);
    int fine_task = 0;
    for (int round = 0; round < Workspace::kFinePVRounds; ++round) {
      int key_tile_begin = round * Workspace::kFinePVGroupTiles;
      for (int query_tile = key_tile_begin; query_tile < Workspace::kTailTiles;
           ++query_tile) {
        int key_tiles = std::min(Workspace::kFinePVGroupTiles,
                                 query_tile + 1 - key_tile_begin);
        size_t query_score_offset = size_t(Workspace::kTailTileRows) *
                                    size_t(Workspace::kTailTileTokens) *
                                    query_tile * (query_tile + 1) / 2;
        TailPVLauncher task_pv(
            tail_scores + query_score_offset +
                size_t(key_tile_begin) * Workspace::kTailTileRows *
                    Workspace::kTailTileTokens,
            value +
                size_t(prefix + key_tile_begin * Workspace::kTailTileTokens) *
                    Workspace::kHeadDim,
            tail_numerator + size_t(query_tile) * Workspace::kTailTileRows *
                                 Workspace::kHeadDim,
            Workspace::kTailTileRows, key_tiles * Workspace::kTailTileTokens,
            round != 0);
        tail_metadata.host_tail_pv_params.push_back(task_pv.params);
        if (fine_task++ == 0) {
          tail_metadata.pv_grid = task_pv.grid;
          tail_metadata.pv_block = task_pv.block;
          tail_metadata.pv_smem_bytes = task_pv.smem_bytes;
        }
      }
    }
  }
  const dim3 tail_pv_grid = tail_metadata.pv_grid;
  const dim3 tail_pv_block = tail_metadata.pv_block;
  const int tail_pv_smem_bytes = tail_metadata.pv_smem_bytes;
  C10_CUDA_CHECK(cudaMemcpyAsync(
      tail_pv_params, tail_metadata.host_tail_pv_params.data(),
      tail_metadata.host_tail_pv_params.size() * sizeof(TailPVKernel::Params),
      cudaMemcpyHostToDevice, prefix_stream));
  if (dump_tail_debug) {
    auto const& first_params = tail_metadata.host_tail_pv_params.front();
    std::cerr << "half2_tail_host_params"
              << " launch_grid=[" << tail_pv_grid.x << "," << tail_pv_grid.y
              << "," << tail_pv_grid.z << "]"
              << " block=" << tail_pv_block.x << " smem=" << tail_pv_smem_bytes
              << " problem=[" << first_params.problem_size.m() << ","
              << first_params.problem_size.n() << ","
              << first_params.problem_size.k() << "]"
              << " tiled=[" << first_params.grid_tiled_shape.m() << ","
              << first_params.grid_tiled_shape.n() << ","
              << first_params.grid_tiled_shape.k() << "]"
              << " swizzle_log=" << first_params.swizzle_log_tile
              << " gemm_k_size=" << first_params.gemm_k_size
              << " ptrs{a=" << static_cast<void*>(first_params.ref_A.data())
              << ",expected_a=" << static_cast<void*>(tail_scores)
              << ",b=" << static_cast<void*>(first_params.ref_B.data())
              << ",expected_b="
              << static_cast<void*>(value +
                                    size_t(prefix) * Workspace::kHeadDim)
              << ",d=" << static_cast<void*>(first_params.ref_D.data())
              << ",expected_d=" << static_cast<void*>(tail_numerator) << "}\n";
  }

  TORCH_CHECK(cublasSetStream(workspace->tail_cublas, tail_stream) ==
                  CUBLAS_STATUS_SUCCESS,
              "set SM70 half2 tail cuBLAS stream failed");
  auto launch_approximate_tail = [&]() {
    #if defined(PREFIX_QK_CUBLAS_FP32_ACCUM)
    float alpha = 0.0625f;
    float beta = 0.0f;
    constexpr cublasComputeType_t kTorchTailQKComputeType = CUBLAS_COMPUTE_32F;
    #else
    __half alpha = __float2half(0.0625f);
    __half beta = __float2half(0.0f);
    constexpr cublasComputeType_t kTorchTailQKComputeType = CUBLAS_COMPUTE_16F;
    #endif
    TORCH_CHECK(cublasGemmBatchedEx(
                    workspace->tail_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    Workspace::kTailTileRows, Workspace::kTailTileTokens,
                    Workspace::kHeadDim, &alpha,
                    reinterpret_cast<void const* const*>(
                        workspace->tail_q_ptrs.data_ptr<uint8_t>()),
                    CUDA_R_16F, Workspace::kRows,
                    reinterpret_cast<void const* const*>(
                        workspace->tail_k_ptrs.data_ptr<uint8_t>()),
                    CUDA_R_16F, total_kv, &beta,
                    reinterpret_cast<void* const*>(
                        workspace->tail_score_ptrs.data_ptr<uint8_t>()),
                    CUDA_R_16F, Workspace::kTailTileRows, Workspace::kTailTasks,
                    kTorchTailQKComputeType,
                    PREFIX_BATCHED_TAIL_QK_ALGO) == CUBLAS_STATUS_SUCCESS,
                "launch SM70 half2 triangular-tail QK failed");
    int64_t mask_elements =
        int64_t(Workspace::kTailTileRows) * Workspace::kTailTileTokens;
    dim3 mask_grid((mask_elements + 255) / 256, Workspace::kTailTiles);
    mask_batched_tri_tail_diagonal<<<mask_grid, 256, 0, tail_stream>>>(
        reinterpret_cast<__half*>(tail_scores), Workspace::kTailTileRows,
        Workspace::kTailTileTokens, 0);
    #if defined(PREFIX_TORCH_STABLE_ROWS)
    dim3 max_grid((Workspace::kRows + 255) / 256, 1);
    stable_row_max_partials<true><<<max_grid, 128, 0, tail_stream>>>(
        reinterpret_cast<__half const*>(tail_scores),
        workspace->tail_max_partials.data_ptr<float>(), Workspace::kRows,
        Workspace::kQuery);
    stable_finish_max<true><<<(Workspace::kRows + 255) / 256, 256, 0, tail_stream>>>(
        workspace->tail_max_partials.data_ptr<float>(), tail_max,
        Workspace::kRows, 1);
    #endif
    if (direct_tail_debug) {
      set_pv_task_base_kernel<<<1, 1, 0, tail_stream>>>(0);
      dim3 direct_grid = tail_pv_grid;
      direct_grid.z = 1;
      cutlass::Kernel<TailPVKernel>
          <<<direct_grid, tail_pv_block, tail_pv_smem_bytes, tail_stream>>>(
              tail_metadata.host_tail_pv_params[0]);
    } else {
      int direct_task_base = 0;
      for (int round = 0; round < Workspace::kFinePVRounds; ++round) {
        int round_tasks =
            Workspace::kTailTiles - round * Workspace::kFinePVGroupTiles;
        set_pv_task_base_kernel<<<1, 1, 0, tail_stream>>>(direct_task_base);
        dim3 round_grid = tail_pv_grid;
        round_grid.z = unsigned(round_tasks);
        batched_tri_tail_pv_kernel<<<round_grid, tail_pv_block,
                                     tail_pv_smem_bytes, tail_stream>>>(
            tail_pv_params);
        C10_CUDA_CHECK(cudaGetLastError());
        direct_task_base += round_tasks;
      }
    }
    C10_CUDA_CHECK(cudaEventRecord(workspace->tail_done, tail_stream));
  };
  if (kPrefixBlocks == 0 && !exact_tail_debug) {
    C10_CUDA_CHECK(
        cudaMemsetAsync(prefix_numerator, 0,
                        size_t(Workspace::kRows) * Workspace::kHeadDim *
                            sizeof(*prefix_numerator),
                        prefix_stream));
    C10_CUDA_CHECK(cudaEventRecord(workspace->prefix_pv_ready, prefix_stream));
    C10_CUDA_CHECK(
        cudaStreamWaitEvent(tail_stream, workspace->prefix_pv_ready, 0));
    launch_approximate_tail();
  }
  for (int block = 0; block < kPrefixBlocks; ++block) {
    prefix_qk[block]->launch(prefix_stream);
    if (block == 0 && !exact_tail_debug && concurrent_tail_scores) {
      C10_CUDA_CHECK(
          cudaEventRecord(workspace->prefix_pv_ready, prefix_stream));
    }
    #if defined(PREFIX_TORCH_STABLE_ROWS)
    int width = std::min(block_n, prefix - block * block_n);
    int tiles = (width + 8191) / 8192;
    dim3 max_grid((Workspace::kRows + 255) / 256, tiles);
    stable_row_max_partials<false><<<max_grid, 128, 0, prefix_stream>>>(
        reinterpret_cast<__half const*>(scores),
        workspace->max_partials.data_ptr<float>(), Workspace::kRows, width);
    stable_finish_max<false><<<(Workspace::kRows + 255) / 256, 256, 0,
                        prefix_stream>>>(
        workspace->max_partials.data_ptr<float>(), block_max, Workspace::kRows,
        tiles);
    #endif
    prefix_pv[block]->launch(prefix_stream);
    #if defined(PREFIX_TORCH_STABLE_ROWS)
    // No host readback or graph branch: these kernels exit immediately for
    // tiles whose complete PV traversal proved that no exponent was clipped.
    stable_row_max_partials<false, true><<<max_grid, 128, 0, prefix_stream>>>(
        reinterpret_cast<__half const*>(scores),
        workspace->max_partials.data_ptr<float>(), Workspace::kRows, width);
    stable_finish_max<false, true><<<(Workspace::kRows + 255) / 256, 256, 0,
                                    prefix_stream>>>(
        workspace->max_partials.data_ptr<float>(), block_max, Workspace::kRows,
        tiles);
    prefix_pv[block]->repair(prefix_stream);
    stable_merge_prefix<<<(Workspace::kRows + 3) / 4, 256, 0, prefix_stream>>>(
        reinterpret_cast<StablePrefixPartial const*>(prefix_numerator),
        block_sum, block_max, prefix_accumulator, prefix_sum, prefix_max,
        block == 0);
    #endif
    if (block == 0 && !exact_tail_debug && concurrent_tail_scores) {
      C10_CUDA_CHECK(
          cudaStreamWaitEvent(tail_stream, workspace->prefix_pv_ready, 0));
      launch_approximate_tail();
    }
  }
  if (kPrefixBlocks > 0 && !exact_tail_debug && !concurrent_tail_scores) {
    C10_CUDA_CHECK(cudaEventRecord(workspace->prefix_pv_ready, prefix_stream));
    C10_CUDA_CHECK(
        cudaStreamWaitEvent(tail_stream, workspace->prefix_pv_ready, 0));
    launch_approximate_tail();
  }

  int repaired_rows = PREFIX_BATCHED_TRI_REPAIR_TOKENS * 6;
  if (exact_tail_debug) {
    C10_CUDA_CHECK(onecat_sm70_d256_dense_state_raw(
        query, key + size_t(prefix) * Workspace::kHeadDim,
        value + size_t(prefix) * Workspace::kHeadDim, tail_max, tail_sum,
        tail_numerator, Workspace::kQuery, Workspace::kQuery, 6, 1, 0.0625f,
        prefix_stream));
    repaired_rows = Workspace::kRows;
  } else {
    C10_CUDA_CHECK(cudaStreamWaitEvent(prefix_stream, workspace->tail_done, 0));
    finalize_round_major_tail_state<<<(Workspace::kRows + 255) / 256, 256, 0,
                                      prefix_stream>>>(
        tail_row_sums, tail_max, tail_sum, Workspace::kTailTileRows,
        Workspace::kRows);
    if (dump_tail_debug) {
      constexpr int kDebugRows[] = {384, 1920, 12000, 24000, 47994};
      constexpr int kDebugCount = sizeof(kDebugRows) / sizeof(kDebugRows[0]);
      constexpr int kDebugRounds = Workspace::kFinePVRounds;
      float approximate_sum[kDebugCount] = {};
      __half approximate_numerator[kDebugCount] = {};
      float partial_sum[kDebugCount][kDebugRounds] = {};
      int partial_task[kDebugCount][kDebugRounds];
      __half score_samples[4] = {};
      int observed_tail_rows = -1;
      int observed_task_base = -1;
      float* observed_tail_sum_output = nullptr;
      for (int index = 0; index < kDebugCount; ++index) {
        for (int round = 0; round < kDebugRounds; ++round) {
          partial_task[index][round] = -1;
        }
      }
      C10_CUDA_CHECK(cudaStreamSynchronize(prefix_stream));
      size_t query1_offset =
          size_t(Workspace::kTailTileRows) * Workspace::kTailTileTokens;
      constexpr int kLastTailTile = Workspace::kTailTiles - 1;
      size_t query_last_offset = size_t(Workspace::kTailTileRows) *
                                 Workspace::kTailTileTokens * kLastTailTile *
                                 Workspace::kTailTiles / 2;
      size_t query_last_diagonal =
          query_last_offset + size_t(kLastTailTile) * Workspace::kTailTileRows *
                                  Workspace::kTailTileTokens;
      constexpr int kScoreRows = Workspace::kTailTileRows;
      C10_CUDA_CHECK(cudaMemcpy(&score_samples[0], tail_scores, sizeof(__half),
                                cudaMemcpyDeviceToHost));
      C10_CUDA_CHECK(cudaMemcpy(&score_samples[1], tail_scores + query1_offset,
                                sizeof(__half), cudaMemcpyDeviceToHost));
      C10_CUDA_CHECK(cudaMemcpy(
          &score_samples[2],
          tail_scores + query_last_offset + int64_t(123) * kScoreRows + 456,
          sizeof(__half), cudaMemcpyDeviceToHost));
      C10_CUDA_CHECK(
          cudaMemcpy(&score_samples[3],
                     tail_scores + query_last_diagonal +
                         int64_t(Workspace::kTailTileTokens - 1) * kScoreRows +
                         Workspace::kTailTileRows - 6,
                     sizeof(__half), cudaMemcpyDeviceToHost));
      C10_CUDA_CHECK(cudaMemcpyFromSymbol(&observed_tail_rows, g_tail_rows,
                                          sizeof(observed_tail_rows)));
      C10_CUDA_CHECK(cudaMemcpyFromSymbol(&observed_task_base, g_pv_task_base,
                                          sizeof(observed_task_base)));
      C10_CUDA_CHECK(cudaMemcpyFromSymbol(&observed_tail_sum_output,
                                          g_tail_row_sum_out,
                                          sizeof(observed_tail_sum_output)));
      for (int index = 0; index < kDebugCount; ++index) {
        C10_CUDA_CHECK(cudaMemcpy(&approximate_sum[index],
                                  tail_sum + kDebugRows[index], sizeof(float),
                                  cudaMemcpyDeviceToHost));
        C10_CUDA_CHECK(cudaMemcpy(
            &approximate_numerator[index],
            tail_numerator + int64_t(kDebugRows[index]) * Workspace::kHeadDim,
            sizeof(__half), cudaMemcpyDeviceToHost));
        int query_tile = kDebugRows[index] / Workspace::kTailTileRows;
        int local_row =
            kDebugRows[index] - query_tile * Workspace::kTailTileRows;
        int rounds = (query_tile + 1 + Workspace::kFinePVGroupTiles - 1) /
                     Workspace::kFinePVGroupTiles;
        for (int round = 0; round < rounds; ++round) {
          int first_task =
              round * Workspace::kTailTiles -
              Workspace::kFinePVGroupTiles * round * (round - 1) / 2;
          int task =
              first_task + query_tile - round * Workspace::kFinePVGroupTiles;
          partial_task[index][round] = task;
          C10_CUDA_CHECK(cudaMemcpy(
              &partial_sum[index][round],
              tail_row_sums + int64_t(task) * Workspace::kTailTileRows +
                  local_row,
              sizeof(float), cudaMemcpyDeviceToHost));
        }
      }
      std::cerr << "half2_tail_debug"
                << " globals{rows=" << observed_tail_rows
                << ",task_base=" << observed_task_base
                << ",sum_ptr=" << static_cast<void*>(observed_tail_sum_output)
                << ",expected_sum_ptr=" << static_cast<void*>(tail_row_sums)
                << "}"
                << " scores=[" << __half2float(score_samples[0]) << ","
                << __half2float(score_samples[1]) << ","
                << __half2float(score_samples[2]) << ","
                << __half2float(score_samples[3]) << "]";
      for (int index = 0; index < kDebugCount; ++index) {
        std::cerr << " r" << kDebugRows[index]
                  << "{approx_sum=" << approximate_sum[index]
                  << ",raw_num0=" << __half2float(approximate_numerator[index])
                  << ",partials=[";
        for (int round = 0; round < kDebugRounds; ++round) {
          if (partial_task[index][round] < 0) {
            break;
          }
          if (round != 0) {
            std::cerr << ",";
          }
          std::cerr << partial_task[index][round] << ":"
                    << partial_sum[index][round];
        }
        std::cerr << "]}";
      }
      std::cerr << "\n";
    }
    C10_CUDA_CHECK(onecat_sm70_d256_dense_state_raw(
        query, key + size_t(prefix) * Workspace::kHeadDim,
        value + size_t(prefix) * Workspace::kHeadDim, tail_max, tail_sum,
        tail_numerator, PREFIX_BATCHED_TRI_REPAIR_TOKENS,
        PREFIX_BATCHED_TRI_REPAIR_TOKENS, 6, 1, 0.0625f, prefix_stream));
  }
    #if defined(PREFIX_TORCH_STABLE_ROWS)
  stable_merge_final<<<Workspace::kRows, 256, 0, prefix_stream>>>(
      prefix_accumulator, prefix_sum, prefix_max,
      reinterpret_cast<__half const*>(tail_numerator), tail_sum, tail_max,
      value_center, maximum_value, reinterpret_cast<__half*>(output),
      repaired_rows, prefix > 0);
    #else
      #if defined(PREFIX_TORCH_PREFIX_FP32_OUTPUT)
  merge_float_prefix_direct_round_major_tail<<<
      Workspace::kRows, Workspace::kHeadDim / 2, 0, prefix_stream>>>(
      prefix_numerator,
      #else
  merge_prefix_direct_round_major_tail<<<
      Workspace::kRows, Workspace::kHeadDim / 2, 0, prefix_stream>>>(
      reinterpret_cast<__half const*>(prefix_numerator),
      #endif
      prefix_sum, reinterpret_cast<__half const*>(tail_numerator), tail_max,
      tail_sum, reinterpret_cast<__half*>(output), Workspace::kRows,
      repaired_rows);
    #endif
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaEventRecord(workspace->completion, prefix_stream));
  C10_CUDA_CHECK(cudaStreamWaitEvent(caller_stream, workspace->completion, 0));
  C10_CUDA_CHECK(cudaEventRecordWithFlags(
      workspace->shared_scores->completion, caller_stream,
      capturing ? cudaEventRecordExternal : 0));
  workspace->shared_scores->completion_recorded = true;
  return out;
}

  #endif

at::Tensor PREFIX_TORCH_ARCHITECTURE_FUNCTION(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, double softmax_scale, bool causal) {
  constexpr int kQuery = PREFIX_TORCH_QUERY_TOKENS;
  constexpr int kRows = kQuery * 6;
  constexpr int kHeadDim = 256;
  constexpr int kHeadsQ = 6;
  constexpr int kHeadsKV = 1;
  constexpr int kTail = kQuery;
  constexpr int kMinTotalKV = kQuery;
  constexpr int kMaxTotalKV = 262144;
  // The prefix PV Tensor Core mainloop consumes 32 FP16 values per K tile.
  // Smaller remainders can read a partial score tile incorrectly, so keep the
  // accelerated contract aligned to the full tile rather than accepting a
  // merely launchable cuBLAS leading dimension.
  constexpr int kTotalKVAlignment = 32;
  constexpr int kBlockN = 8192;

  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(),
              "SM70 GQA architecture requires CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Half &&
                  k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type() &&
                  out.scalar_type() == q.scalar_type(),
              "SM70 GQA architecture requires FP16 q, k, v, and out");
  TORCH_CHECK(q.sizes() == at::IntArrayRef({1, kQuery, kHeadsQ, kHeadDim}) &&
                  k.dim() == 4 && k.size(0) == 1 && k.size(2) == kHeadsKV &&
                  k.size(3) == kHeadDim && v.sizes() == k.sizes() &&
                  out.sizes() == q.sizes(),
              "SM70 GQA architecture only accepts Q", kQuery,
              "/Hq6/Hkv1/D256 dense tensors");
  const int total_kv = static_cast<int>(k.size(1));
  TORCH_CHECK(total_kv >= kMinTotalKV && total_kv <= kMaxTotalKV &&
                  total_kv % kTotalKVAlignment == 0,
              "SM70 GQA architecture requires KV in [", kMinTotalKV,
              ", 262144] with 32-token alignment, got ", total_kv);
  const int prefix = total_kv - kTail;
  const int blocks = (prefix + kBlockN - 1) / kBlockN;
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  out.is_contiguous(),
              "SM70 GQA architecture requires contiguous tensors");
  TORCH_CHECK(q.get_device() == k.get_device() &&
                  q.get_device() == v.get_device() &&
                  q.get_device() == out.get_device(),
              "SM70 GQA architecture tensors must share one device");
  TORCH_CHECK(causal, "SM70 GQA architecture requires causal attention");
  TORCH_CHECK(std::abs(softmax_scale - 0.0625) < 1.0e-8,
              "SM70 GQA architecture requires D256 softmax scale 1/16");

  const at::cuda::OptionalCUDAGuard device_guard(q.device());
  #if defined(PREFIX_QK_CUBLAS_RAW) && defined(PREFIX_BATCHED_TRI_TAIL) && \
      defined(PREFIX_TAIL_IDLE_SM_FINE_PV) &&                              \
      defined(PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE)
  TORCH_CHECK(total_kv >= Sm70GqaHalf2Workspace::kMinTotalKV &&
                  total_kv <= Sm70GqaHalf2Workspace::kMaxTotalKV,
              "SM70 half2 architecture requires KV in [", kMinTotalKV,
              ", 262144]");
  return sm70_d256_gqa_half2_family_fwd(q, k, v, out);
  #else
  const int qk_tiles_n =
      (kBlockN + QKThreadblockShape::kN - 1) / QKThreadblockShape::kN;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr size_t kScratchAlignment = 256;
  size_t scratch_bytes = 0;
  auto reserve_scratch = [&](size_t bytes) {
    scratch_bytes =
        (scratch_bytes + kScratchAlignment - 1) & ~(kScratchAlignment - 1);
    size_t offset = scratch_bytes;
    scratch_bytes += bytes;
    return offset;
  };
  size_t partial_offset =
      reserve_scratch(size_t(kRows) * kHeadDim * sizeof(Element));
  size_t prefix_accumulator_offset =
      reserve_scratch(size_t(kRows) * kHeadDim * sizeof(float));
  size_t tail_output_offset =
      reserve_scratch(size_t(kRows) * kHeadDim * sizeof(Element));
  size_t qk_norm_offset =
      reserve_scratch(size_t(qk_tiles_n) * kRows * sizeof(float));
  size_t qk_sum_offset =
      reserve_scratch(size_t(qk_tiles_n) * kRows * sizeof(float));
  size_t prefix_max_offset = reserve_scratch(size_t(kRows) * sizeof(float));
  size_t prefix_sum_offset = reserve_scratch(size_t(kRows) * sizeof(float));
  size_t old_scale_offset = reserve_scratch(size_t(kRows) * sizeof(float));
  size_t block_scale_offset = reserve_scratch(size_t(kRows) * sizeof(float));
  size_t tail_max_offset = reserve_scratch(size_t(kRows) * sizeof(float));
  size_t tail_sum_offset = reserve_scratch(size_t(kRows) * sizeof(float));
  int device = q.get_device();
  auto score_workspace = find_sm70_gqa_score_workspace(device);
  if (!score_workspace) {
    constexpr size_t kRequiredPostWorkspaceHeadroom = 128 * 1024 * 1024;
    constexpr size_t kScoreBytes = size_t(kRows) * kBlockN * sizeof(Element);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t required_bytes =
        kScoreBytes + scratch_bytes + kRequiredPostWorkspaceHeadroom;
    TORCH_CHECK_WITH(
        OutOfMemoryError, free_bytes >= required_bytes,
        "SM70 GQA architecture requires ", required_bytes / (1024 * 1024),
        " MiB free before its first workspace allocation, including "
        "128 MiB of downstream headroom, but only ",
        free_bytes / (1024 * 1024), " MiB remains out of ",
        total_bytes / (1024 * 1024), " MiB");
    score_workspace = get_sm70_gqa_score_workspace(q, kRows, kBlockN);
  }
  std::unique_lock<std::mutex> launch_lock(score_workspace->launch_mutex);
  if (score_workspace->completion_recorded) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, score_workspace->completion, 0));
  }
  at::Tensor scratch = at::empty({static_cast<int64_t>(scratch_bytes)},
                                 q.options().dtype(at::ScalarType::Byte));
  auto* scratch_base = scratch.data_ptr<uint8_t>();

  auto* query = reinterpret_cast<Element*>(q.data_ptr<at::Half>());
  auto* key = reinterpret_cast<Element*>(k.data_ptr<at::Half>());
  auto* value = reinterpret_cast<Element*>(v.data_ptr<at::Half>());
  auto* score_ptr =
      reinterpret_cast<Element*>(score_workspace->scores.data_ptr<at::Half>());
  auto* partial_ptr = reinterpret_cast<Element*>(scratch_base + partial_offset);
  float* prefix_accumulator_ptr =
      reinterpret_cast<float*>(scratch_base + prefix_accumulator_offset);
  auto* tail_output_ptr =
      reinterpret_cast<Element*>(scratch_base + tail_output_offset);
  auto* output_ptr = reinterpret_cast<Element*>(out.data_ptr<at::Half>());
  float* qk_norm_ptr = reinterpret_cast<float*>(scratch_base + qk_norm_offset);
  float* qk_sum_ptr = reinterpret_cast<float*>(scratch_base + qk_sum_offset);
  float* prefix_max_ptr =
      reinterpret_cast<float*>(scratch_base + prefix_max_offset);
  float* prefix_sum_ptr =
      reinterpret_cast<float*>(scratch_base + prefix_sum_offset);
  float* old_scale_ptr =
      reinterpret_cast<float*>(scratch_base + old_scale_offset);
  float* block_scale_ptr =
      reinterpret_cast<float*>(scratch_base + block_scale_offset);
  float* tail_max_ptr =
      reinterpret_cast<float*>(scratch_base + tail_max_offset);
  float* tail_sum_ptr =
      reinterpret_cast<float*>(scratch_base + tail_sum_offset);

  C10_CUDA_CHECK(onecat_sm70_d256_dense_state_raw(
      query, key + size_t(prefix) * kHeadDim, value + size_t(prefix) * kHeadDim,
      tail_max_ptr, tail_sum_ptr, tail_output_ptr, kTail, kTail, kHeadsQ,
      kHeadsKV, static_cast<float>(softmax_scale), stream));
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_rows, &kRows, sizeof(kRows), 0,
                                         cudaMemcpyHostToDevice, stream));
  float const* persistent_max_ptr = qk_norm_ptr;
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_row_max, &persistent_max_ptr,
                                         sizeof(persistent_max_ptr), 0,
                                         cudaMemcpyHostToDevice, stream));
    #if defined(PREFIX_QK_FULL_STATS)
  float const* persistent_inv_sum_ptr = qk_sum_ptr;
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_row_inv_sum, &persistent_inv_sum_ptr,
                                         sizeof(persistent_inv_sum_ptr), 0,
                                         cudaMemcpyHostToDevice, stream));
    #else
  float* persistent_sum_ptr = qk_sum_ptr;
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_row_sum_out, &persistent_sum_ptr,
                                         sizeof(persistent_sum_ptr), 0,
                                         cudaMemcpyHostToDevice, stream));
    #endif

  for (int block = 0; block < blocks; ++block) {
    int begin = block * kBlockN;
    int width = std::min(kBlockN, prefix - begin);
    BlockOperators operation;
    operation.width = width;
    typename QKGemm::Arguments arguments(
        {kRows, width, kHeadDim}, 1, {query, QKLayoutA(kHeadDim)},
        {key + size_t(begin) * kHeadDim, QKLayoutB(kHeadDim)},
        {score_ptr, typename QKGemm::LayoutC(width)},
        {score_ptr, typename QKGemm::LayoutC(width)},
        {Element(static_cast<float>(softmax_scale)), Element(0.0f)},
        {qk_norm_ptr, typename QKGemm::LayoutN(kRows)},
        {qk_sum_ptr, typename QKGemm::LayoutS(kRows)},
        {score_ptr, typename QKGemm::LayoutSoft(width)});
    TORCH_CHECK(operation.qk.initialize(arguments) == cutlass::Status::kSuccess,
                "initialize SM70 GQA QK block ", block, " failed");
    operation.pv = std::make_unique<PVLauncher>(
        score_ptr, value + size_t(begin) * kHeadDim,
    #if defined(PREFIX_PV_DIRECT_FP32_ACCUMULATE)
        prefix_accumulator_ptr,
    #else
        partial_ptr,
    #endif
        kRows, width, block != 0);
    TORCH_CHECK(operation.qk(stream) == cutlass::Status::kSuccess,
                "launch SM70 GQA QK block ", block, " failed");
    operation.pv->launch(stream);
    prepare_prefix_update<<<(kRows + 255) / 256, 256, 0, stream>>>(
        qk_norm_ptr, qk_sum_ptr, prefix_max_ptr, prefix_sum_ptr, old_scale_ptr,
        block_scale_ptr, kRows, block == 0);
    #if !defined(PREFIX_PV_DIRECT_FP32_ACCUMULATE)
    constexpr int kPairs = kRows * kHeadDim / 2;
    apply_prefix_update_half2<<<(kPairs + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<__half const*>(partial_ptr), prefix_accumulator_ptr,
        old_scale_ptr, block_scale_ptr, kRows, block == 0);
    #endif
  }
  merge_prefix_accumulator_tail<<<kRows, kHeadDim, 0, stream>>>(
      prefix_accumulator_ptr, prefix_max_ptr, prefix_sum_ptr,
      reinterpret_cast<__half const*>(tail_output_ptr), tail_max_ptr,
      tail_sum_ptr, reinterpret_cast<__half*>(output_ptr), kRows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaEventRecord(score_workspace->completion, stream));
  score_workspace->completion_recorded = true;
  return out;
  #endif
}

}  // namespace FLASH_NAMESPACE

#endif
