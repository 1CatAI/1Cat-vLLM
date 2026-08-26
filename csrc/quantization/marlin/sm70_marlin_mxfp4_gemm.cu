#include "core/registration.h"
#include "quantization/marlin/dequant.h"
#include "quantization/marlin/sm70_marlin_common.cuh"
#include "quantization/marlin/sm70_marlin_gemm.cuh"
#include "quantization/marlin/sm70_marlin_iterator_utils.cuh"
#include "quantization/marlin/sm70_marlin_splitk.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <torch/library.h>
#include <cstdint>

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

using marlin::sm70::configure_sm70_dynamic_smem;
using marlin::sm70::dispatch_sm70_marlin_fixed_group_geometry;
using marlin::sm70::kQuantTileK;
using marlin::sm70::kQuantTileN;
using marlin::sm70::load_qword_vector;
using marlin::sm70::qword_from_vector;
using marlin::sm70::sm70_active_split_k;
using marlin::sm70::sm70_marlin_cta_grid;
using marlin::sm70::sm70_marlin_dense_auto_params;
using marlin::sm70::sm70_marlin_narrow_f32_to_f16;
using marlin::sm70::sm70_splitk_partition;
using marlin::sm70::Sm70AtomicFp16Epilogue;
using marlin::sm70::Sm70AtomicFp32Epilogue;
using marlin::sm70::Sm70CtaGeometry;
using marlin::sm70::Sm70MarlinGemmTraits;
using marlin::sm70::Sm70SplitKPartition;
using marlin::sm70::u4_packed_macro_n_qweight_offset_from_logical;
using marlin::sm70::validate_sm70_marlin_dense_cta_geometry_supported;
using marlin::sm70::validate_sm70_marlin_dense_cta_n_alignment;

namespace {

constexpr int kMxfp4ValuesPerWord = 8;

CUTLASS_DEVICE
__half2 e8m0x2_to_half2_fast(uint16_t e8m0_x2) {
  // Finite nonzero E8M0 bytes are IEEE FP32 exponent fields for
  // 2^(value - 127); byte zero also rounds to FP16 zero. Let the hardware
  // conversion preserve subnormals and apply round-to-nearest-even.
  uint32_t const bits0 = static_cast<uint32_t>(e8m0_x2 & 0xFF) << 23;
  uint32_t const bits1 = static_cast<uint32_t>(e8m0_x2 >> 8) << 23;
  return __floats2half2_rn(__uint_as_float(bits0), __uint_as_float(bits1));
}

CUTLASS_DEVICE
void dequant_e8m0_scales_to_half2(int q, half2* scale_cache) {
  uint32_t const word = static_cast<uint32_t>(q);

  uint16_t lo_16 = static_cast<uint16_t>(word);
  uint16_t hi_16 = static_cast<uint16_t>(word >> 16);

  __half2 res0 = e8m0x2_to_half2_fast(lo_16);
  __half2 res1 = e8m0x2_to_half2_fast(hi_16);

  uint2 vec_res;
  vec_res.x = *reinterpret_cast<uint32_t*>(&res0);
  vec_res.y = *reinterpret_cast<uint32_t*>(&res1);

  *reinterpret_cast<uint2*>(scale_cache) = vec_res;
}

template <typename Shape_, typename ThreadMap_, int GroupSize_,
          int PackedMacroN_>
class Sm70Mxfp4IteratorB {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  static int const kGroupSize = GroupSize_;
  static int const kPackedMacroN = PackedMacroN_;
  using Element = cutlass::half_t;
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;
  static_assert(Shape::kN == 64 || Shape::kN == 128 || Shape::kN == 256,
                "SM70 Marlin MXFP4 IteratorB expects CTA_N in {64, 128, 256}.");
  static_assert(ThreadMap::Iterations::kContiguous == Shape::kN / kQuantTileN,
                "SM70 Marlin MXFP4 IteratorB expects one contiguous iteration "
                "per 64-column quant tile.");
  static_assert(ThreadMap::Delta::kContiguous == kQuantTileN,
                "SM70 Marlin MXFP4 IteratorB expects 64-column deltas.");
  static_assert(ThreadMap::kElementsPerAccess == kMxfp4ValuesPerWord,
                "SM70 Marlin MXFP4 IteratorB expects one packed FP4 word per "
                "access.");
  static_assert(ThreadMap::Iterations::kStrided >= 1,
                "SM70 Marlin U4-family IteratorB expects one or more strided "
                "iterations.");
  struct Params {
    int size_n;
    int aligned_initial_k_step;

    CUTLASS_HOST_DEVICE
    Params() : size_n(0), aligned_initial_k_step(Shape::kK) {}

    CUTLASS_HOST_DEVICE
    Params(int size_k, int size_n_)
        : size_n(size_n_), aligned_initial_k_step(aligned_k_step(size_k)) {}
  };

 private:
  uint32_t const* qweight_;
  uint8_t const* scales_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int qweight_base_offset_;
  int k_offset_;
  int n_offset_;
  int aligned_k_step_;
  bool mask_enabled_;
  mutable half2 cached_scales_[ThreadMap::Iterations::kCount * 4];

 public:
  CUTLASS_DEVICE
  Sm70Mxfp4IteratorB(Params const& params, uint32_t const* qweight,
                     uint8_t const* scales, int thread_id,
                     cutlass::MatrixCoord const& threadblock_offset)
      : qweight_(qweight),
        scales_(scales),
        params_(params),
        thread_offset_(ThreadMap::initial_offset(thread_id)),
        k_offset_(threadblock_offset.row()),
        n_offset_(threadblock_offset.column()),
        aligned_k_step_(params.aligned_initial_k_step),
        mask_enabled_(true) {
    int const logical_k = threadblock_offset.row() + thread_offset_.strided();
    int const logical_n =
        threadblock_offset.column() + thread_offset_.contiguous();
    qweight_base_offset_ =
        qweight_offset_from_logical(params_, logical_k, logical_n);
  }

  CUTLASS_DEVICE
  Sm70Mxfp4IteratorB& operator++() {
    int const k_advance = aligned_k_step_;
    int const k_advance_qwords =
        (k_advance / kQuantTileK) * (params_.size_n * 2);
    k_offset_ += k_advance;
    aligned_k_step_ = Shape::kK;
    qweight_base_offset_ += k_advance_qwords;
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
  static int aligned_k_step(int size_k) {
    int const k_tail = size_k % Shape::kK;
    return k_tail == 0 ? Shape::kK : k_tail;
  }

  CUTLASS_DEVICE
  int scale_group(int logical_k) const {
    static_assert(kGroupSize == 32,
                  "SM70 Marlin MXFP4 supports only group_size 32.");
    return logical_k / kGroupSize;
  }

  CUTLASS_DEVICE
  static int qweight_offset_from_logical(Params const& params, int logical_k,
                                         int logical_n) {
    return u4_packed_macro_n_qweight_offset_from_logical<kPackedMacroN>(
        params.size_n, logical_k, logical_n);
  }

  CUTLASS_DEVICE
  void cache_metadata_e8m0_scales(int c, int group, int cache_n) const {
    uint2 const scale_words = *reinterpret_cast<uint2 const*>(
        scales_ + group * params_.size_n + cache_n);
    half2* scale_cache = cached_scales_ + c * 4;
    dequant_e8m0_scales_to_half2(
        static_cast<int>(qword_from_vector(scale_words, 0)), scale_cache);
    dequant_e8m0_scales_to_half2(
        static_cast<int>(qword_from_vector(scale_words, 1)), scale_cache + 2);
  }

  CUTLASS_DEVICE
  void cache_current_group_metadata() const {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      int const logical_k =
          k_offset_ + thread_offset_.strided() + s * ThreadMap::Delta::kStrided;
      int const group = scale_group(logical_k);
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int const cache_n = n_offset_ + thread_offset_.contiguous() +
                            c * ThreadMap::Delta::kContiguous;
        int const cache_index = c + s * ThreadMap::Iterations::kContiguous;
        cache_metadata_e8m0_scales(cache_index, group, cache_n);
      }
    }
  }

  CUTLASS_DEVICE
  void load_cta_n_aligned(Fragment& frag) const {
    if constexpr (ThreadMap::Iterations::kStrided == 1) {
      if constexpr (ThreadMap::Iterations::kContiguous == 4) {
        auto const qwords =
            load_qword_vector<4>(qweight_ + qweight_base_offset_);
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = c * kAccess;
          uint32_t const qword = qword_from_vector(qwords, c);
          half2 const* scale_vec = cached_scales_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
              static_cast<int>(qword << 8), deq);
          frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
          frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
          marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
          frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
        }
      } else if constexpr (ThreadMap::Iterations::kContiguous == 2) {
        auto const qwords =
            load_qword_vector<2>(qweight_ + qweight_base_offset_);
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = c * kAccess;
          uint32_t const qword = qword_from_vector(qwords, c);
          half2 const* scale_vec = cached_scales_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
              static_cast<int>(qword << 8), deq);
          frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
          frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
          marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
          frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
        }
      } else {
        static_assert(
            ThreadMap::Iterations::kContiguous == 1,
            "Unsupported SM70 Marlin MXFP4 contiguous iteration count.");
        uint32_t const qword =
            load_qword_vector<1>(qweight_ + qweight_base_offset_);
        half2 const* scale_vec = cached_scales_;

        half2 deq[2];
        half2* frag_vec = reinterpret_cast<half2*>(frag.data());
        marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
            static_cast<int>(qword << 8), deq);
        frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
        frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
        marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
            static_cast<int>(qword), deq);
        frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
        frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
      }
    } else {
      int const logical_n_base = n_offset_ + thread_offset_.contiguous();
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
        int const logical_k_s = k_offset_ + thread_offset_.strided() +
                                s * ThreadMap::Delta::kStrided;
        int const qweight_base_s =
            qweight_offset_from_logical(params_, logical_k_s, logical_n_base);
        if constexpr (ThreadMap::Iterations::kContiguous == 4) {
          auto const qwords = load_qword_vector<4>(qweight_ + qweight_base_s);
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            constexpr int kAccess = ThreadMap::kElementsPerAccess;
            int const frag_base =
                (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
            uint32_t const qword = qword_from_vector(qwords, c);
            half2 const* scale_vec = cached_scales_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
                static_cast<int>(qword << 8), deq);
            frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
            frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
            marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
                static_cast<int>(qword), deq);
            frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
            frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
          }
        } else if constexpr (ThreadMap::Iterations::kContiguous == 2) {
          auto const qwords = load_qword_vector<2>(qweight_ + qweight_base_s);
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            constexpr int kAccess = ThreadMap::kElementsPerAccess;
            int const frag_base =
                (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
            uint32_t const qword = qword_from_vector(qwords, c);
            half2 const* scale_vec = cached_scales_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
                static_cast<int>(qword << 8), deq);
            frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
            frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
            marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
                static_cast<int>(qword), deq);
            frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
            frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
          }
        } else {
          static_assert(
              ThreadMap::Iterations::kContiguous == 1,
              "Unsupported SM70 Marlin MXFP4 contiguous iteration count.");
          uint32_t const qword =
              load_qword_vector<1>(qweight_ + qweight_base_s);
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = s * kAccess;
          half2 const* scale_vec = cached_scales_;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
              static_cast<int>(qword << 8), deq);
          frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
          frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
          marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
          frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
        }
      }
    }
  }

  CUTLASS_DEVICE
  void load(Fragment& frag) const {
    if (!mask_enabled_) {
      return;
    }

    cache_current_group_metadata();
    load_cta_n_aligned(frag);
  }
};

struct Sm70Mxfp4GemmSpec {
  template <typename Shape, typename ThreadMap, int GroupSize, int PackedMacroN>
  using IteratorB =
      Sm70Mxfp4IteratorB<Shape, ThreadMap, GroupSize, PackedMacroN>;
};

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int GroupSize, int PackedMacroN>
using Sm70Mxfp4GemmTraits =
    Sm70MarlinGemmTraits<Sm70Mxfp4GemmSpec, CtaM, CtaN, CtaK, Warps, WarpM,
                         WarpN, WarpK, GroupSize, PackedMacroN>;

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int GroupSize, int PackedMacroN>
__global__ __launch_bounds__(Warps * 32, 1) void sm70_marlin_mxfp4_gemm_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    uint8_t const* __restrict__ b_scales, cutlass::half_t* __restrict__ c,
    int m, int n, int k, int lda) {
  using Traits = Sm70Mxfp4GemmTraits<CtaM, CtaN, CtaK, Warps, WarpM, WarpN,
                                     WarpK, GroupSize, PackedMacroN>;
  using Mma = typename Traits::Mma;
  using Epilogue = typename Traits::Epilogue;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int thread_idx = threadIdx.x;
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_idx = threadIdx.x % 32;

  cutlass::MatrixCoord tb_offset_A{int(blockIdx.x) * CtaM, 0};
  cutlass::MatrixCoord tb_offset_B{0, int(blockIdx.y) * CtaN};
  cutlass::MatrixCoord tb_offset_C{int(blockIdx.x) * CtaM,
                                   int(blockIdx.y) * CtaN};

  typename Traits::LayoutA layout_a(lda);
  typename Traits::LayoutC layout_c(n);

  typename Mma::IteratorA iterator_A(typename Mma::IteratorA::Params(layout_a),
                                     const_cast<cutlass::half_t*>(a),
                                     cutlass::MatrixCoord(m, k), thread_idx,
                                     tb_offset_A);
  typename Mma::IteratorB iterator_B(
      typename Mma::IteratorB::Params(k, n),
      reinterpret_cast<uint32_t const*>(b_q_weight),
      reinterpret_cast<uint8_t const*>(b_scales), thread_idx, tb_offset_B);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  int const gemm_k_iterations = (k + CtaK - 1) / CtaK;
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename Traits::OutputOp output_op({1.0f, 0.0f});
  typename Epilogue::OutputTileIterator iterator_C(
      typename Epilogue::OutputTileIterator::Params(layout_c), c,
      cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);
  typename Epilogue::OutputTileIterator iterator_D(
      typename Epilogue::OutputTileIterator::Params(layout_c), c,
      cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);

  Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(output_op, iterator_D, accumulators, iterator_C);
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int GroupSize, int PackedMacroN>
__global__
__launch_bounds__(Warps * 32, 1) void sm70_marlin_mxfp4_gemm_splitk_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    uint8_t const* __restrict__ b_scales, cutlass::half_t* __restrict__ c,
    float* __restrict__ c32, int m, int n, int k, int lda,
    int requested_split_k) {
  using Traits = Sm70Mxfp4GemmTraits<CtaM, CtaN, CtaK, Warps, WarpM, WarpN,
                                     WarpK, GroupSize, PackedMacroN>;
  using Mma = typename Traits::Mma;
  using AtomicEpilogue = Sm70AtomicFp32Epilogue<Traits>;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int const thread_idx = threadIdx.x;
  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const lane_idx = threadIdx.x % 32;
  Sm70SplitKPartition const partition = sm70_splitk_partition<GroupSize, CtaK>(
      k, requested_split_k, int(blockIdx.z));
  if (partition.partition_k == 0) {
    return;
  }

  cutlass::MatrixCoord tb_offset_A{int(blockIdx.x) * CtaM, partition.k_begin};
  cutlass::MatrixCoord tb_offset_B{partition.k_begin, int(blockIdx.y) * CtaN};
  cutlass::MatrixCoord tb_offset_C{int(blockIdx.x) * CtaM,
                                   int(blockIdx.y) * CtaN};

  typename Traits::LayoutA layout_a(lda);
  typename Traits::LayoutC layout_c(n);

  typename Mma::IteratorA iterator_A(typename Mma::IteratorA::Params(layout_a),
                                     const_cast<cutlass::half_t*>(a),
                                     cutlass::MatrixCoord(m, k), thread_idx,
                                     tb_offset_A);
  typename Mma::IteratorB iterator_B(
      typename Mma::IteratorB::Params(k, n),
      reinterpret_cast<uint32_t const*>(b_q_weight),
      reinterpret_cast<uint8_t const*>(b_scales), thread_idx, tb_offset_B);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  int const gemm_k_iterations = partition.partition_k / CtaK;
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename AtomicEpilogue::OutputTileIterator iterator_D(
      typename AtomicEpilogue::OutputTileIterator::Params(layout_c), c,
      cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);

  AtomicEpilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                          lane_idx);
  epilogue(iterator_D, accumulators, c32, n);
}

}  // namespace

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int GroupSize, int PackedMacroN>
torch::Tensor launch_sm70_marlin_mxfp4_gemm(torch::Tensor& a, torch::Tensor& c,
                                            torch::Tensor& b_q_weight,
                                            torch::Tensor& b_scales,
                                            int64_t size_m, int64_t size_n,
                                            int64_t size_k,
                                            int requested_split_k) {
  auto kernel =
      sm70_marlin_mxfp4_gemm_kernel<CtaM, CtaN, CtaK, Warps, WarpM, WarpN,
                                    WarpK, GroupSize, PackedMacroN>;
  using SharedStorage =
      typename Sm70Mxfp4GemmTraits<CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK,
                                   GroupSize, PackedMacroN>::SharedStorage;
  size_t smem_bytes = configure_sm70_dynamic_smem<SharedStorage>(kernel);

  dim3 block(Warps * 32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  if (requested_split_k == 1) {
    dim3 grid = sm70_marlin_cta_grid(size_m, size_n, CtaM, CtaN);
    kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
        reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
        reinterpret_cast<uint8_t const*>(b_scales.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
        static_cast<int>(size_m), static_cast<int>(size_n),
        static_cast<int>(size_k), static_cast<int>(a.stride(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
  }

  TORCH_CHECK(size_k % int64_t(CtaK) == 0,
              "SM70 Marlin mxfp4 requires K divisible by CTA_K=", CtaK,
              " for requested_split_k > 1. Got K=", size_k,
              ", requested_split_k=", requested_split_k, ".");

  auto split_kernel = sm70_marlin_mxfp4_gemm_splitk_kernel<
      CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK, GroupSize, PackedMacroN>;
  smem_bytes = configure_sm70_dynamic_smem<SharedStorage>(split_kernel);

  int64_t const numel = size_m * size_n;
  // 0009: fp32 split-K scratch (zeroed for the fp32 atomicAdd), M*N floats.
  // Only allocated on the split_k>1 path, which the auto-selector uses ONLY at
  // low batch (M<128), so the scratch is bounded and small.
  auto c32 = torch::zeros({size_m, size_n}, a.options().dtype(at::kFloat));

  dim3 grid = sm70_marlin_cta_grid(size_m, size_n, CtaM, CtaN);
  int const active_split_k =
      sm70_active_split_k(static_cast<int>(size_k), requested_split_k, CtaK);
  grid.z = static_cast<unsigned>(active_split_k);
  split_kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
      reinterpret_cast<uint8_t const*>(b_scales.data_ptr()),
      reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
      c32.data_ptr<float>(), static_cast<int>(size_m), static_cast<int>(size_n),
      static_cast<int>(size_k), static_cast<int>(a.stride(0)),
      requested_split_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 0009: single deterministic-width fp32->fp16 narrow of the scratch.
  int const narrow_block = 256;
  int const narrow_grid =
      static_cast<int>((numel + narrow_block - 1) / narrow_block);
  sm70_marlin_narrow_f32_to_f16<256><<<narrow_grid, narrow_block, 0, stream>>>(
      c32.data_ptr<float>(), reinterpret_cast<half*>(c.data_ptr<at::Half>()),
      numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return c;
}

struct Sm70Mxfp4Launcher {
  torch::Tensor& a;
  torch::Tensor& c;
  torch::Tensor& b_q_weight;
  torch::Tensor& b_scales;
  int64_t size_m;
  int64_t size_n;
  int64_t size_k;
  int requested_split_k;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
            int WarpK, int GroupSize, int PackedMacroN>
  torch::Tensor operator()() const {
    return launch_sm70_marlin_mxfp4_gemm<CtaM, CtaN, CtaK, Warps, WarpM, WarpN,
                                         WarpK, GroupSize, PackedMacroN>(
        a, c, b_q_weight, b_scales, size_m, size_n, size_k, requested_split_k);
  }
};

torch::Tensor sm70_marlin_mxfp4_gemm(torch::Tensor& a, torch::Tensor& c,
                                     torch::Tensor& b_q_weight,
                                     torch::Tensor& b_scales, int64_t size_m,
                                     int64_t size_n, int64_t size_k,
                                     int64_t group_size) {
  c10::cuda::CUDAGuard device_guard(a.device());

  auto const params = sm70_marlin_dense_auto_params("mxfp4", group_size, size_m,
                                                    size_n, size_k);
  Sm70CtaGeometry const geometry = params.geometry;
  validate_sm70_marlin_dense_cta_geometry_supported("SM70 Marlin mxfp4",
                                                    geometry);
  validate_sm70_marlin_dense_cta_n_alignment("SM70 Marlin mxfp4", geometry,
                                             size_n);
  Sm70Mxfp4Launcher const launcher{
      a,      c,      b_q_weight, b_scales,
      size_m, size_n, size_k,     params.requested_split_k};
  return dispatch_sm70_marlin_fixed_group_geometry<32>(
      launcher, geometry, params.packed_macro_n, group_size, "mxfp4");
}
