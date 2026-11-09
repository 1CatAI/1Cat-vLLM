// SPDX-License-Identifier: BSD-3-Clause
#undef FLASH_NAMESPACE
#define FLASH_NAMESPACE onecat_v37

/***************************************************************************************************
 * Precision-qualified v37 long-prefix attention for V100/SM70.
 *
 * Tile-local probabilities are formed from FP32 logits and statistics before
 * conversion to FP16 Tensor Core operands. QK/PV accumulation, online partials
 * and the exact causal-tail output stay FP32. Prefix and tail overlap without
 * changing their arithmetic. The legacy kernel is a separate rollback route.
 **************************************************************************************************/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/ScopeExit.h>
#include "namespace_config.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"
#include "gemm_with_softmax.h"

namespace FLASH_NAMESPACE {
namespace {

using Element = cutlass::half_t;
using TailOutput = float;

using QKAccumulator = float;

using QKLayoutA = cutlass::layout::RowMajor;
using QKLayoutB = cutlass::layout::ColumnMajor;
using QKThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;
using QKWarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using QKInstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
using QKLinearOutputOp =
    cutlass::epilogue::thread::LinearCombination<Element, 8, QKAccumulator,
                                                 QKAccumulator>;
// A distinct functor type keeps the modified visitor's kernel symbols local
// to this variant, even when baseline and candidate share one extension.
struct QKOutputOp : QKLinearOutputOp {
  CUTLASS_HOST_DEVICE
  explicit QKOutputOp(Params const& params) : QKLinearOutputOp(params) {}
};
using QKGemm =
    cutlass::GemmSoftmaxV37<Element, QKLayoutA, Element, QKLayoutB, Element,
                            QKAccumulator, cutlass::arch::OpClassTensorOp,
                            cutlass::arch::Sm70, QKThreadblockShape,
                            QKWarpShape, QKInstructionShape, QKOutputOp, 2,
                            cutlass::MatrixShape<1, 1024>>;

using PVLayout = cutlass::layout::RowMajor;
using PVAccumulator = float;
using PVOutput = float;
constexpr int kPVOutputAccess = 4;
using PVThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;
using PVWarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using PVInstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
using PVSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using PVLinearOutputOp =
    cutlass::epilogue::thread::LinearCombination<PVOutput, kPVOutputAccess,
                                                 PVAccumulator, float>;
static_assert(std::is_same<PVOutput, float>::value &&
                  std::is_same<PVAccumulator, float>::value,
              "Direct online state requires FP32 output and accumulation");
CUTLASS_DEVICE int pv_output_row(int sequence);

__device__ int g_rows = 0;

struct PVOutputOp : PVLinearOutputOp {
  using Base = PVLinearOutputOp;
  struct Params : Base::Params {
    float const* old_scales;
    float const* block_scales;
    bool initialize;
    CUTLASS_HOST_DEVICE
    Params(float alpha = 1.0f, float beta = 0.0f)
        : Base::Params(alpha, beta),
          old_scales(nullptr),
          block_scales(nullptr),
          initialize(true) {}
    CUTLASS_HOST_DEVICE
    Params(float const* old_, float const* next_, bool init)
        : Base::Params(1.0f, init ? 0.0f : 1.0f),
          old_scales(old_),
          block_scales(next_),
          initialize(init) {}
  };
  float const* old_scales;
  float const* block_scales;
  bool initialize;
  mutable int sequence = 0;
  CUTLASS_HOST_DEVICE
  explicit PVOutputOp(Params const& params)
      : Base(params),
        old_scales(params.old_scales),
        block_scales(params.block_scales),
        initialize(params.initialize) {}
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return !initialize; }
  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accum,
                            FragmentSource const& source) const {
    int row = pv_output_row(sequence++);
    FragmentOutput result;
    float a = row < g_rows ? old_scales[row] : 0.0f;
    float b = row < g_rows ? block_scales[row] : 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i)
      result[i] = fmaf(source[i], a, accum[i] * b);
    return result;
  }
  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accum) const {
    int row = pv_output_row(sequence++);
    FragmentOutput result;
    float b = row < g_rows ? block_scales[row] : 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) result[i] = accum[i] * b;
    return result;
  }
};

constexpr int kPVAlignment = 8;

using PVDefaultKernel = typename cutlass::gemm::kernel::DefaultGemm<
    Element, PVLayout, kPVAlignment, Element, PVLayout, kPVAlignment, PVOutput,
    PVLayout, PVAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, PVThreadblockShape, PVWarpShape, PVInstructionShape,
    PVOutputOp, PVSwizzle, 2, false, cutlass::arch::OpMultiplyAdd>::GemmKernel;

using PVDefaultMma = typename PVDefaultKernel::Mma;
CUTLASS_DEVICE int pv_output_row(int sequence) {
  using Iterator = typename PVDefaultKernel::Epilogue::OutputTileIterator;
  using Map = typename Iterator::ThreadMap;
  constexpr int kAccesses =
      Iterator::Fragment::kElements / Iterator::kElementsPerAccess;
  int step = sequence / kAccesses;
  int fragment = sequence % kAccesses;
  Iterator iterator(typename Iterator::Params(PVLayout(256)), nullptr,
                    {g_rows, 256}, threadIdx.x,
                    {int(blockIdx.x) * PVThreadblockShape::kM,
                     int(blockIdx.y) * PVThreadblockShape::kN});
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Iterator::kIterations; ++i) {
    if (i < step) ++iterator;
  }
  return iterator.thread_start().row() + Map::iteration_offset(fragment).row();
}
using PVIteratorA = typename PVDefaultMma::IteratorA;
using PVIteratorB = typename PVDefaultMma::IteratorB;
using PVSmemIteratorA = typename PVDefaultMma::SmemIteratorA;
using PVSmemIteratorB = typename PVDefaultMma::SmemIteratorB;

__device__ float const* g_row_max = nullptr;
__device__ float const* g_row_inv_sum = nullptr;

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
  int row_index[kStridedIterations];
  int k_offset = 0;
  float* shared_tile_scale;

  CUTLASS_DEVICE
  ExpRowSumTransformA() {
    // Query lengths are multiples of 64, so six packed heads fill M128 tiles.
    constexpr int kScaleCount = (8192 / 256) * 128;
    constexpr int kThreads = 32 * (128 / 64) * (256 / 64);
    __shared__ float tile_scale_smem[kScaleCount];
    shared_tile_scale = tile_scale_smem;
#pragma unroll
    for (int i = threadIdx.x; i < kScaleCount; i += kThreads) {
      int tile = i / 128;
      int row = blockIdx.x * 128 + i % 128;
      int stored_tile = tile == 0 ? 8192 / 256 : tile;
      float delta = g_row_max[stored_tile * g_rows + row] - g_row_max[row];
      shared_tile_scale[i] = exp2f(delta * 1.4426950408889634f);
    }
    __syncthreads();
    auto thread_offset = ThreadMap::initial_offset(threadIdx.x);
#pragma unroll
    for (int s = 0; s < kStridedIterations; ++s) {
      int row = blockIdx.x * PVThreadblockShape::kM + thread_offset.strided() +
                s * ThreadMap::Delta::kStrided;
      row_max[s] = row < g_rows ? g_row_max[row] : 0.0f;
      row_inv_sum[s] = row < g_rows ? g_row_inv_sum[row] : 0.0f;
      row_index[s] = row - blockIdx.x * 128;
    }
  }

  CUTLASS_DEVICE
  OutputFragment operator()(InputFragment const& input) {
    OutputFragment output;
    constexpr float kLog2E = 1.4426950408889634f;
    constexpr int kElementsPerAccess = PVIteratorA::AccessType::kElements;
    auto const* input_access =
        reinterpret_cast<typename PVIteratorA::AccessType const*>(&input);
    auto* output_access =
        reinterpret_cast<typename PVIteratorA::AccessType*>(&output);
#pragma unroll
    for (int s = 0; s < kStridedIterations; ++s) {
      int tile = k_offset / 256;
      int stored_tile = tile == 0 ? 8192 / 256 : tile;
      // The double-buffered pipeline transforms one final masked look-ahead
      // fragment. Its zero values are unused, but its metadata load must stay
      // in bounds as well. Wrap that sentinel tile to the valid first tile.
      int shared_tile = tile % (8192 / 256);
      float probability_scale =
          shared_tile_scale[shared_tile * 128 + row_index[s]];
#pragma unroll
      for (int c = 0; c < kContiguousIterations; ++c) {
#pragma unroll
        for (int v = 0; v < kAccessesPerVector; ++v) {
          int index = v + kAccessesPerVector * (c + s * kContiguousIterations);
          typename PVIteratorA::AccessType transformed;
#pragma unroll
          for (int e = 0; e < kElementsPerAccess; ++e) {
            float value = static_cast<float>(input_access[index][e]);
            float weight = value * probability_scale;
            transformed[e] = Element(weight);
          }
          output_access[index] = transformed;
        }
      }
    }
    k_offset += PVThreadblockShape::kK;
    return output;
  }
};

using PVTransformB =
    cutlass::NumericArrayConverter<typename PVSmemIteratorB::Element,
                                   typename PVIteratorB::Element,
                                   PVIteratorB::Fragment::kElements>;
using PVMma = cutlass::gemm::threadblock::MmaPipelined<
    typename PVDefaultMma::Shape, PVIteratorA, PVSmemIteratorA, PVIteratorB,
    PVSmemIteratorB, PVAccumulator, PVLayout, typename PVDefaultMma::Policy,
    ExpRowSumTransformA, PVTransformB>;
using PVKernel =
    cutlass::gemm::kernel::Gemm<PVMma, typename PVDefaultKernel::Epilogue,
                                PVSwizzle, false>;

void check(cudaError_t result, char const* operation) {
  TORCH_CHECK(result == cudaSuccess, operation, ": ",
              cudaGetErrorString(result));
}

struct PVLauncher {
  typename PVKernel::Params params;
  dim3 grid;
  dim3 block;
  int smem_bytes;

  PVLauncher(Element* scores, Element* value, PVOutput* output, int rows, int k,
             float const* old_scales = nullptr,
             float const* block_scales = nullptr, bool initialize = true) {
    cutlass::gemm::GemmCoord problem(rows, 256, k);
    PVSwizzle swizzle;
    auto tiled_shape =
        swizzle.get_tiled_shape(problem,
                                {PVThreadblockShape::kM, PVThreadblockShape::kN,
                                 PVThreadblockShape::kK},
                                1);
    params = typename PVKernel::Params(
        problem, tiled_shape, {scores, PVLayout(k)}, {value, PVLayout(256)},
        {output, PVLayout(256)}, {output, PVLayout(256)},
        typename PVOutputOp::Params(old_scales, block_scales, initialize),
        nullptr);
    grid = swizzle.get_grid_shape(tiled_shape);
    block = dim3(PVKernel::kThreadCount, 1, 1);
    smem_bytes = int(sizeof(typename PVKernel::SharedStorage));
    if (smem_bytes >= 48 * 1024) {
      check(cudaFuncSetAttribute(cutlass::Kernel<PVKernel>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_bytes),
            "set PV dynamic shared memory");
    }
  }

  void launch(cudaStream_t stream) const {
    cutlass::Kernel<PVKernel><<<grid, block, smem_bytes, stream>>>(params);
  }
};

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
  float next_max = block_max[row];
  float next_mass = 1.0f / block_inv_sum[row];
  if (initialize) {
    old_scales[row] = 0.0f;
    block_scales[row] = 1.0f;
    prefix_max[row] = next_max;
    prefix_sum[row] = next_mass;
    return;
  }
  float old_max = prefix_max[row];
  float global_max = fmaxf(old_max, next_max);
  float old_scale = exp2f((old_max - global_max) * kLog2E);
  float block_scale = next_mass * exp2f((next_max - global_max) * kLog2E);
  old_scales[row] = old_scale;
  block_scales[row] = exp2f((next_max - global_max) * kLog2E);
  prefix_max[row] = global_max;
  prefix_sum[row] = prefix_sum[row] * old_scale + block_scale;
}

__global__ void merge_prefix_accumulator_tail(
    float const* prefix_accumulator, float const* prefix_max,
    float const* prefix_sum, TailOutput const* tail_output,
    float const* tail_max, float const* tail_sum, __half* output, int rows) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  __shared__ float masses[3];
  if (d == 0) {
    constexpr float kLog2E = 1.4426950408889634f;
    float global_max = fmaxf(prefix_max[row], tail_max[row]);
    masses[0] = exp2f((prefix_max[row] - global_max) * kLog2E);
    masses[1] = exp2f((tail_max[row] - global_max) * kLog2E);
    masses[2] =
        1.0f / (prefix_sum[row] * masses[0] + tail_sum[row] * masses[1]);
  }
  __syncthreads();
  int64_t element = int64_t(row) * 256 + d;
  float numerator = prefix_accumulator[element] * masses[0] +
                    float(tail_output[element]) * masses[1];
  output[element] = __float2half_rn(numerator * masses[2]);
}

struct BlockOperators {
  QKGemm qk;
  std::unique_ptr<PVLauncher> pv;
};

}  // namespace
}  // namespace FLASH_NAMESPACE

extern "C" cudaError_t onecat_v37_dense_state_float_raw(
    const void*, const void*, const void*, float*, float*, void*, int, int, int,
    int, float, int, cudaStream_t);

namespace FLASH_NAMESPACE {

struct Sm70GqaScoreWorkspace {
  at::Tensor scores;
  cudaEvent_t completion = nullptr;
  cudaStream_t tail_stream = nullptr;
  cudaEvent_t tail_input_ready = nullptr;
  cudaEvent_t tail_complete = nullptr;
  bool completion_recorded = false;
  std::mutex launch_mutex;

  Sm70GqaScoreWorkspace(const at::Tensor& q, int rows, int block_n)
      : scores(at::empty({rows, block_n}, q.options())) {
    auto cleanup = c10::make_scope_exit([&]() noexcept {
      if (tail_stream) cudaStreamDestroy(tail_stream);
      if (tail_input_ready) cudaEventDestroy(tail_input_ready);
      if (tail_complete) cudaEventDestroy(tail_complete);
      if (completion) cudaEventDestroy(completion);
    });
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(&completion, cudaEventDisableTiming));
    C10_CUDA_CHECK(
        cudaStreamCreateWithFlags(&tail_stream, cudaStreamNonBlocking));
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(&tail_input_ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(&tail_complete, cudaEventDisableTiming));
    cleanup.release();
  }

  ~Sm70GqaScoreWorkspace() {
    if (tail_stream) cudaStreamDestroy(tail_stream);
    if (tail_input_ready) cudaEventDestroy(tail_input_ready);
    if (tail_complete) cudaEventDestroy(tail_complete);
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

at::Tensor sm70_d256_gqa_v37_fwd(const at::Tensor& q, const at::Tensor& k,
                                 const at::Tensor& v, at::Tensor& out,
                                 double softmax_scale, bool causal) {
  constexpr int kMaxQuery = 8192;
  constexpr int kMaxRows = kMaxQuery * 6;
  constexpr int kHeadDim = 256;
  constexpr int kHeadsQ = 6;
  constexpr int kHeadsKV = 1;
  constexpr int kMaxTotalKV = 262144;
  constexpr int kTotalKVStep = 32;
  constexpr int kBlockN = 8192;

  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(),
              "SM70 GQA architecture requires CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Half &&
                  k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type() &&
                  out.scalar_type() == q.scalar_type(),
              "SM70 GQA architecture requires FP16 q, k, v, and out");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(1) >= 64 &&
                  q.size(1) <= kMaxQuery && q.size(1) % 64 == 0 &&
                  q.size(2) == kHeadsQ && q.size(3) == kHeadDim &&
                  k.dim() == 4 && k.size(0) == 1 && k.size(2) == kHeadsKV &&
                  k.size(3) == kHeadDim && v.sizes() == k.sizes() &&
                  out.sizes() == q.sizes(),
              "SM70 GQA architecture only accepts the validated "
              "Q64..8192 (multiple of 64)/Hq6/Hkv1/D256 dense shape family");
  const int kQuery = static_cast<int>(q.size(1));
  const int kRows = kQuery * kHeadsQ;
  const int kTail = kQuery;
  TORCH_CHECK(k.size(1) > kQuery && k.size(1) <= kMaxTotalKV,
              "v37 requires a non-empty prefix and KV <= 262144");
  const int total_kv = static_cast<int>(k.size(1));
  TORCH_CHECK(total_kv % kTotalKVStep == 0,
              "SM70 GQA architecture requires a 32-token KV step, got ",
              total_kv);
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
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "v37 prefill supports SM70 only");
  for (const at::Tensor* tensor :
       {&q, &k, &v, static_cast<const at::Tensor*>(&out)}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16 == 0,
                "v37 prefill requires 16-byte aligned tensors");
  }
  const int qk_tiles_n =
      (kBlockN + QKThreadblockShape::kN - 1) / QKThreadblockShape::kN;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus capture_status;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
  TORCH_CHECK(
      capture_status == cudaStreamCaptureStatusNone,
      "v37 prefill does not support CUDA Graph capture; use exact fallback");

  constexpr size_t kScratchAlignment = 256;
  size_t scratch_bytes = 0;
  auto reserve_scratch = [&](size_t bytes) {
    scratch_bytes =
        (scratch_bytes + kScratchAlignment - 1) & ~(kScratchAlignment - 1);
    size_t offset = scratch_bytes;
    scratch_bytes += bytes;
    return offset;
  };
  size_t prefix_accumulator_offset =
      reserve_scratch(size_t(kRows) * kHeadDim * sizeof(float));
  size_t tail_output_offset =
      reserve_scratch(size_t(kRows) * kHeadDim * sizeof(TailOutput));
  size_t qk_norm_offset =
      reserve_scratch(size_t(qk_tiles_n + 1) * kRows * sizeof(float));
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
    constexpr size_t kScoreBytes = size_t(kMaxRows) * kBlockN * sizeof(Element);
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
    score_workspace = get_sm70_gqa_score_workspace(q, kMaxRows, kBlockN);
  }
  std::unique_lock<std::mutex> launch_lock(score_workspace->launch_mutex);
  if (score_workspace->completion_recorded) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, score_workspace->completion, 0));
  }
  at::Tensor scratch = at::empty({static_cast<int64_t>(scratch_bytes)},
                                 q.options().dtype(at::ScalarType::Byte));
  auto* scratch_base = scratch.data_ptr<uint8_t>();
  // On an exception, the private tail must stop using caller-owned tensors
  // before scratch is released. Also order any queued prefix writes before
  // a subsequent call reuses the shared score/metadata cache.
  auto failed_launch = c10::make_scope_exit([&]() noexcept {
    cudaStreamSynchronize(score_workspace->tail_stream);
    score_workspace->completion_recorded =
        cudaEventRecord(score_workspace->completion, stream) == cudaSuccess;
  });

  auto* query = reinterpret_cast<Element*>(q.data_ptr<at::Half>());
  auto* key = reinterpret_cast<Element*>(k.data_ptr<at::Half>());
  auto* value = reinterpret_cast<Element*>(v.data_ptr<at::Half>());
  auto* score_ptr =
      reinterpret_cast<Element*>(score_workspace->scores.data_ptr<at::Half>());
  float* prefix_accumulator_ptr =
      reinterpret_cast<float*>(scratch_base + prefix_accumulator_offset);
  auto* tail_output_ptr =
      reinterpret_cast<TailOutput*>(scratch_base + tail_output_offset);
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

  auto set_pv_metadata = [&]() {
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_rows, &kRows, sizeof(kRows), 0,
                                           cudaMemcpyHostToDevice, stream));
    float const* persistent_max_ptr = qk_norm_ptr;
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_row_max, &persistent_max_ptr,
                                           sizeof(persistent_max_ptr), 0,
                                           cudaMemcpyHostToDevice, stream));
    float const* persistent_inv_sum_ptr = qk_sum_ptr;
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(
        g_row_inv_sum, &persistent_inv_sum_ptr, sizeof(persistent_inv_sum_ptr),
        0, cudaMemcpyHostToDevice, stream));
  };
  // Set metadata before forking the independent causal tail.
  set_pv_metadata();
  // Fork only after gathered inputs are ready; join before the sole consumer
  // of tail state. Prefix/tail arithmetic and their scratch regions are
  // disjoint.
  C10_CUDA_CHECK(cudaEventRecord(score_workspace->tail_input_ready, stream));
  C10_CUDA_CHECK(cudaStreamWaitEvent(score_workspace->tail_stream,
                                     score_workspace->tail_input_ready, 0));
  cudaStream_t tail_stream = score_workspace->tail_stream;
  C10_CUDA_CHECK(onecat_v37_dense_state_float_raw(
      query, key + size_t(prefix) * kHeadDim, value + size_t(prefix) * kHeadDim,
      tail_max_ptr, tail_sum_ptr, tail_output_ptr, kTail, kTail, kHeadsQ,
      kHeadsKV, static_cast<float>(softmax_scale), 1, tail_stream));
  C10_CUDA_CHECK(cudaEventRecord(score_workspace->tail_complete, tail_stream));

  for (int block = 0; block < blocks; ++block) {
    int begin = block * kBlockN;
    int width = std::min(kBlockN, prefix - begin);
    BlockOperators operation;
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
    TORCH_CHECK(operation.qk(stream) == cutlass::Status::kSuccess,
                "launch SM70 GQA QK block ", block, " failed");
    prepare_prefix_update<<<(kRows + 255) / 256, 256, 0, stream>>>(
        qk_norm_ptr, qk_sum_ptr, prefix_max_ptr, prefix_sum_ptr, old_scale_ptr,
        block_scale_ptr, kRows, block == 0);
    operation.pv = std::make_unique<PVLauncher>(
        score_ptr, value + size_t(begin) * kHeadDim, prefix_accumulator_ptr,
        kRows, width, old_scale_ptr, block_scale_ptr, block == 0);
    operation.pv->launch(stream);
  }
  C10_CUDA_CHECK(
      cudaStreamWaitEvent(stream, score_workspace->tail_complete, 0));
  merge_prefix_accumulator_tail<<<kRows, kHeadDim, 0, stream>>>(
      prefix_accumulator_ptr, prefix_max_ptr, prefix_sum_ptr,
      reinterpret_cast<TailOutput const*>(tail_output_ptr), tail_max_ptr,
      tail_sum_ptr, reinterpret_cast<__half*>(output_ptr), kRows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaEventRecord(score_workspace->completion, stream));
  score_workspace->completion_recorded = true;
  failed_launch.release();
  return out;
}

}  // namespace FLASH_NAMESPACE
