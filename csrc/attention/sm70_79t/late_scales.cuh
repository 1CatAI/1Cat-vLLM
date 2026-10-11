// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Included beside the existing 79t QK/PV types. These are operand bindings:
// the family scheduler, score owner, tail tasks and state merge stay shared.
using RowQKLinear =
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>;
using RowQKSwizzle = PVSwizzle;
using RowQKDefault = typename cutlass::gemm::kernel::DefaultGemm<
    Element, cutlass::layout::ColumnMajor, 8, Element,
    cutlass::layout::RowMajor, 8, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70, QKThreadblockShape,
    QKWarpShape, QKInstructionShape, RowQKLinear, RowQKSwizzle, 2, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;

CUTLASS_DEVICE int row_qk_output_row(int sequence, int keys, int rows) {
  using Iterator = typename RowQKDefault::Epilogue::OutputTileIterator;
  using Map = typename Iterator::ThreadMap;
  constexpr int kAccesses =
      Iterator::Fragment::kElements / Iterator::kElementsPerAccess;
  int step = sequence / kAccesses;
  int fragment = sequence % kAccesses;
  Iterator iterator(typename Iterator::Params(cutlass::layout::RowMajor(rows)),
                    nullptr, {keys, rows}, threadIdx.x,
                    {int(blockIdx.x) * QKThreadblockShape::kM,
                     int(blockIdx.y) * QKThreadblockShape::kN});
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Iterator::kIterations; ++i)
    if (i < step) ++iterator;
  return iterator.thread_start().row() + Map::iteration_offset(fragment).row();
}

struct RowQKOutput : RowQKLinear {
  struct Params : RowQKLinear::Params {
    flash_v100::KVScalePlane scales{};
    int key_begin = 0;
    int keys = 0;
    int rows = 0;
    CUTLASS_HOST_DEVICE Params() : RowQKLinear::Params(0.0625f, 0.0f) {}
    CUTLASS_HOST_DEVICE
    Params(flash_v100::KVScalePlane plane, int begin, int width, int query_rows)
        : RowQKLinear::Params(0.0625f, 0.0f),
          scales(plane),
          key_begin(begin),
          keys(width),
          rows(query_rows) {}
  };
  Params args;
  mutable int sequence = 0;
  CUTLASS_HOST_DEVICE explicit RowQKOutput(Params const& params)
      : RowQKLinear(params), args(params) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
    int row = row_qk_output_row(sequence++, args.keys, args.rows);
    float scale = row < args.keys
                      ? args.scales.load(0, args.key_begin + row, 0) * 0.0625f
                      : 0.0f;
    FragmentOutput output;
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < kCount; ++e) output[e] = accumulator[e] * scale;
    return output;
  }
  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator,
                            FragmentSource const&) const {
    return (*this)(accumulator);
  }
};

using RowQKKernel = typename cutlass::gemm::kernel::DefaultGemm<
    Element, cutlass::layout::ColumnMajor, 8, Element,
    cutlass::layout::RowMajor, 8, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70, QKThreadblockShape,
    QKWarpShape, QKInstructionShape, RowQKOutput, RowQKSwizzle, 2, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;

struct RowQKLauncher {
  typename RowQKKernel::Params params;
  dim3 grid, block;
  int smem_bytes;
  RowQKLauncher(Element* query, Element* key, float* output, int rows,
                int width, int query_stride, int key_stride,
                flash_v100::KVScalePlane scales, int begin) {
    cutlass::gemm::GemmCoord problem(width, rows, 256);
    RowQKSwizzle swizzle;
    auto tiles =
        swizzle.get_tiled_shape(problem,
                                {QKThreadblockShape::kM, QKThreadblockShape::kN,
                                 QKThreadblockShape::kK},
                                1);
    params = typename RowQKKernel::Params(
        problem, tiles, {key, cutlass::layout::ColumnMajor(key_stride)},
        {query, cutlass::layout::RowMajor(query_stride)},
        {output, cutlass::layout::RowMajor(rows)},
        {output, cutlass::layout::RowMajor(rows)},
        RowQKOutput::Params(scales, begin, width, rows), nullptr);
    grid = swizzle.get_grid_shape(tiles);
    block = dim3(RowQKKernel::kThreadCount, 1, 1);
    smem_bytes = sizeof(typename RowQKKernel::SharedStorage);
  }
  void launch(cudaStream_t stream) const {
    cutlass::Kernel<RowQKKernel><<<grid, block, smem_bytes, stream>>>(params);
  }
};

__global__ void batched_row_scale_qk_kernel(
    typename RowQKKernel::Params const* params) {
  extern __shared__ int shared_storage_base[];
  auto& shared = *reinterpret_cast<typename RowQKKernel::SharedStorage*>(
      shared_storage_base);
  RowQKKernel operation;
  operation(params[blockIdx.z], shared);
}

using RowPVTransformB = cutlass::NumericArrayConverter<
    typename RowPVDefaultMma::SmemIteratorB::Element,
    typename RowPVDefaultMma::IteratorB::Element,
    RowPVDefaultMma::IteratorB::Fragment::kElements>;

template <bool Tail>
using RowPVMma = cutlass::gemm::threadblock::MmaPipelined79T<
    typename RowPVDefaultMma::Shape, RowScoreIteratorA,
    typename RowPVDefaultMma::SmemIteratorA,
    typename RowPVDefaultMma::IteratorB,
    typename RowPVDefaultMma::SmemIteratorB, PVAccumulator, PVLayoutC,
    typename RowPVDefaultMma::Policy,
    ExpRowSumTransformAImpl<Tail, true, RowScoreIteratorA, true>,
    RowPVTransformB, bool, true>;
template <bool Tail>
using RowPVKernel =
    cutlass::gemm::kernel::Gemm<RowPVMma<Tail>, typename RowPVDefault::Epilogue,
                                PVSwizzle, false>;

template <typename Kernel>
__global__ void repair_row_scale_pv_kernel(typename Kernel::Params params) {
  if (!g_79t_prefix_outliers[blockIdx.x]) return;
  extern __shared__ int shared_storage_base[];
  auto& shared =
      *reinterpret_cast<typename Kernel::SharedStorage*>(shared_storage_base);
  Kernel operation;
  operation(params, shared);
}

template <bool Tail>
struct RowPVLauncher {
  using Kernel = RowPVKernel<Tail>;
  typename Kernel::Params params;
  dim3 grid, block;
  int smem_bytes;
  RowPVLauncher(float* scores, Element* value, float* output, int rows,
                int width, bool accumulate = false) {
    cutlass::gemm::GemmCoord problem(rows, 256, width);
    PVSwizzle swizzle;
    auto tiles = swizzle.get_tiled_shape(
        problem, {RowPVShape::kM, RowPVShape::kN, RowPVShape::kK}, 1);
    params = typename Kernel::Params(
        problem, tiles, {scores, PVLayoutA(rows)}, {value, PVLayoutB(256)},
        {output, PVLayoutC(256)}, {output, PVLayoutC(256)},
        typename PrefixFloatPVOutputOp::Params(1.0f, accumulate ? 1.0f : 0.0f),
        nullptr);
    grid = swizzle.get_grid_shape(tiles);
    block = dim3(Kernel::kThreadCount, 1, 1);
    smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes >= 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_bytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          repair_row_scale_pv_kernel<Kernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
  }
  void launch(cudaStream_t stream) const {
    cutlass::Kernel<Kernel><<<grid, block, smem_bytes, stream>>>(params);
  }
  void repair(cudaStream_t stream) const {
    repair_row_scale_pv_kernel<Kernel>
        <<<grid, block, smem_bytes, stream>>>(params);
  }
};

__global__ void batched_row_scale_pv_kernel(
    typename RowPVKernel<true>::Params const* params) {
  using Kernel = RowPVKernel<true>;
  extern __shared__ int shared_storage_base[];
  auto& shared =
      *reinterpret_cast<typename Kernel::SharedStorage*>(shared_storage_base);
  Kernel operation;
  operation(params[pv_task_index()], shared);
}

__global__ void set_prefix_scale_key_begin(int begin) {
  g_prefix_scale_key_begin = begin;
}

// Both numerators are unnormalized FP32; the denominator never contains V
// scale.
__global__ void merge_row_scale_state(float const* prefix,
                                      float const* prefix_sum,
                                      float const* prefix_max,
                                      float const* tail, float const* tail_sum,
                                      float const* tail_max, __half* output,
                                      bool has_prefix) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  __shared__ float coefficients[3];
  if (d == 0) {
    float pm = has_prefix ? prefix_max[row] : -CUDART_INF_F;
    float tm = tail_max[row];
    float maximum = fmaxf(pm, tm);
    float ps = has_prefix ? expf(pm - maximum) : 0.0f;
    float ts = expf(tm - maximum);
    coefficients[0] = ps;
    coefficients[1] = ts;
    coefficients[2] = 1.0f / ((has_prefix ? prefix_sum[row] * ps : 0.0f) +
                              tail_sum[row] * ts);
  }
  __syncthreads();
  int64_t index = int64_t(row) * 256 + d;
  float p = has_prefix ? prefix[index] * coefficients[0] : 0.0f;
  output[index] =
      __float2half_rn((p + tail[index] * coefficients[1]) * coefficients[2]);
}

__global__ void restore_row_scale_recovered(float const* recovered,
                                            int const* tiles, __half* output) {
  int tile = blockIdx.x;
  if (!tiles[tile]) return;
  for (int local = 0; local < kStableRecoveryRows; ++local) {
    int64_t index =
        int64_t(tile * kStableRecoveryRows + local) * 256 + threadIdx.x;
    output[index] = __float2half_rn(recovered[index]);
  }
}
