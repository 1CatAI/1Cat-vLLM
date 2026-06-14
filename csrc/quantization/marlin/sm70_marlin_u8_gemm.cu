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

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

using marlin::sm70::Sm70CtaGeometry;
using marlin::sm70::Sm70AtomicFp16Epilogue;
using marlin::sm70::Sm70MarlinGemmTraits;
using marlin::sm70::Sm70SplitKPartition;
using marlin::sm70::validate_sm70_marlin_dense_cta_geometry_supported;
using marlin::sm70::validate_sm70_marlin_dense_cta_n_alignment;
using marlin::sm70::configure_sm70_dynamic_smem;
using marlin::sm70::dispatch_sm70_marlin_geometry;
using marlin::sm70::kCtaK;
using marlin::sm70::kQuantTileK;
using marlin::sm70::kQuantTileN;
using marlin::sm70::sm70_marlin_dense_auto_cta_geometry;
using marlin::sm70::sm70_marlin_dense_auto_requested_split_k;
using marlin::sm70::load_qword_vector;
using marlin::sm70::qword_from_vector;
using marlin::sm70::sm70_marlin_cta_grid;
using marlin::sm70::sm70_active_split_k;
using marlin::sm70::sm70_splitk_partition;
using marlin::sm70::u8_cta_n_qweight_offset_from_logical;
using marlin::sm70::u8_cta_n_qweight_word_stride_from_logical;

namespace {

constexpr int kU8ValuesPerAccess = 8;

template <typename Shape_, typename ThreadMap_, int GroupSize_>
class Sm70U8ZpIteratorB {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  static int const kGroupSize = GroupSize_;
  using Element = cutlass::half_t;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  static_assert(Shape::kK == kCtaK,
                "SM70 Marlin U8 IteratorB expects CTA_K=32.");
  static_assert(Shape::kN == 64 || Shape::kN == 128 || Shape::kN == 256,
                "SM70 Marlin U8 IteratorB expects CTA_N in {64, 128, 256}.");
  static_assert(ThreadMap::Iterations::kContiguous ==
                    Shape::kN / kQuantTileN,
                "SM70 Marlin U8 IteratorB expects one contiguous iteration per "
                "64-column quant tile.");
  static_assert(ThreadMap::Delta::kContiguous == kQuantTileN,
                "SM70 Marlin U8 IteratorB expects 64-column deltas.");
  static_assert(ThreadMap::kElementsPerAccess == kU8ValuesPerAccess,
                "SM70 Marlin U8 IteratorB expects two packed U8 words per "
                "access.");
  static_assert(ThreadMap::Iterations::kStrided == 1 ||
                    ThreadMap::Iterations::kStrided == 2,
                "SM70 Marlin U8-family IteratorB expects one or two strided "
                "iterations.");
  static constexpr int kQweightWordStrideWords = Shape::kN / kQuantTileN;

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
  half const* scales_;
  half const* zp_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int qweight_base_offset_;
  int k_offset_;
  int n_offset_;
  int aligned_k_step_;
  bool mask_enabled_;
  mutable half2 cached_scales_[ThreadMap::Iterations::kContiguous * 4];
  mutable half2 cached_zp_[ThreadMap::Iterations::kContiguous * 4];

 public:
  CUTLASS_DEVICE
  Sm70U8ZpIteratorB(Params const& params, uint32_t const* qweight,
                    half const* scales, half const* zp, int thread_id,
                    cutlass::MatrixCoord const& threadblock_offset)
      : qweight_(qweight),
        scales_(scales),
        zp_(zp),
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
    if constexpr (kGroupSize == -1) {
      // GroupSize=-1 keeps scale and fp16 zero points stable for
      // every K tile. Cache both planes once before the MMA mainloop.
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int const cache_n =
            n_offset_ + thread_offset_.contiguous() +
            c * ThreadMap::Delta::kContiguous;
        half2 const* scale_vec =
            reinterpret_cast<half2 const*>(scales_ + cache_n);
        half2* scale_cache = cached_scales_ + c * 4;
        scale_cache[0] = scale_vec[0];
        scale_cache[1] = scale_vec[1];
        scale_cache[2] = scale_vec[2];
        scale_cache[3] = scale_vec[3];

        half2 const* zp_vec = reinterpret_cast<half2 const*>(zp_ + cache_n);
        half2* zp_cache = cached_zp_ + c * 4;
        zp_cache[0] = zp_vec[0];
        zp_cache[1] = zp_vec[1];
        zp_cache[2] = zp_vec[2];
        zp_cache[3] = zp_vec[3];
      }
    }
  }

  CUTLASS_DEVICE
  Sm70U8ZpIteratorB& operator++() {
    int const k_advance = aligned_k_step_;
    int const k_advance_qwords =
        (k_advance / kQuantTileK) * (params_.size_n * 4);
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
    if constexpr (kGroupSize == -1) {
      return 0;
    } else {
      static_assert(kGroupSize == 32 || kGroupSize == 64 ||
                        kGroupSize == 128,
                    "SM70 Marlin U8 supports only group sizes "
                    "-1, 32, 64, and 128.");
      return logical_k / kGroupSize;
    }
  }

  CUTLASS_DEVICE
  static int qweight_offset_from_logical(Params const& params, int logical_k,
                                         int logical_n) {
    return u8_cta_n_qweight_offset_from_logical<Shape::kN>(params.size_n, logical_k,
                                                  logical_n);
  }

  CUTLASS_DEVICE
  static int qweight_word_stride_from_logical(Params const&, int logical_n) {
    return u8_cta_n_qweight_word_stride_from_logical<Shape::kN>();
  }

  CUTLASS_DEVICE
  void cache_metadata_lane_vectors(int c, int group, int cache_n) const {
    half2 const* scale_vec = reinterpret_cast<half2 const*>(
        scales_ + group * params_.size_n + cache_n);
    half2* scale_cache = cached_scales_ + c * 4;
    scale_cache[0] = scale_vec[0];
    scale_cache[1] = scale_vec[1];
    scale_cache[2] = scale_vec[2];
    scale_cache[3] = scale_vec[3];

    half2 const* zp_vec = reinterpret_cast<half2 const*>(
        zp_ + group * params_.size_n + cache_n);
    half2* zp_cache = cached_zp_ + c * 4;
    zp_cache[0] = zp_vec[0];
    zp_cache[1] = zp_vec[1];
    zp_cache[2] = zp_vec[2];
    zp_cache[3] = zp_vec[3];
  }

  CUTLASS_DEVICE
  void cache_metadata_vector_words(int c, int group, int cache_n) const {
    uint4 const scale_words = *reinterpret_cast<uint4 const*>(
        scales_ + group * params_.size_n + cache_n);
    half2 const* scale_vec = reinterpret_cast<half2 const*>(&scale_words);
    half2* scale_cache = cached_scales_ + c * 4;
    scale_cache[0] = scale_vec[0];
    scale_cache[1] = scale_vec[1];
    scale_cache[2] = scale_vec[2];
    scale_cache[3] = scale_vec[3];

    uint4 const zp_words = *reinterpret_cast<uint4 const*>(
        zp_ + group * params_.size_n + cache_n);
    half2 const* zp_vec = reinterpret_cast<half2 const*>(&zp_words);
    half2* zp_cache = cached_zp_ + c * 4;
    zp_cache[0] = zp_vec[0];
    zp_cache[1] = zp_vec[1];
    zp_cache[2] = zp_vec[2];
    zp_cache[3] = zp_vec[3];
  }

  CUTLASS_DEVICE
  void cache_current_group_metadata(int group) const {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      int const cache_n =
          n_offset_ + thread_offset_.contiguous() +
          c * ThreadMap::Delta::kContiguous;
      if constexpr (Shape::kN == 256) {
        cache_metadata_vector_words(c, group, cache_n);
      } else {
        cache_metadata_lane_vectors(c, group, cache_n);
      }
    }
  }

  CUTLASS_DEVICE
  void load_cta_n_aligned(Fragment& frag) const {
    if constexpr (ThreadMap::Iterations::kStrided == 1) {
      if constexpr (ThreadMap::Iterations::kContiguous == 4) {
        uint4 const qwords0 =
            load_qword_vector<4>(qweight_ + qweight_base_offset_);
        uint4 const qwords1 = load_qword_vector<4>(
            qweight_ + qweight_base_offset_ + kQweightWordStrideWords);
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = c * kAccess;
          uint32_t const qword0 = qword_from_vector(qwords0, c);
          uint32_t const qword1 = qword_from_vector(qwords1, c);
          half2 const* scale_vec = cached_scales_ + c * 4;
          half2 const* zp_vec = cached_zp_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword0), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword1), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
        }
      } else if constexpr (ThreadMap::Iterations::kContiguous == 2) {
        uint2 const qwords0 =
            load_qword_vector<2>(qweight_ + qweight_base_offset_);
        uint2 const qwords1 = load_qword_vector<2>(
            qweight_ + qweight_base_offset_ + kQweightWordStrideWords);
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = c * kAccess;
          uint32_t const qword0 = qword_from_vector(qwords0, c);
          uint32_t const qword1 = qword_from_vector(qwords1, c);
          half2 const* scale_vec = cached_scales_ + c * 4;
          half2 const* zp_vec = cached_zp_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword0), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword1), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
        }
      } else {
        static_assert(ThreadMap::Iterations::kContiguous == 1,
                      "Unsupported SM70 Marlin U8 contiguous iteration count.");
        uint32_t const qword0 =
            load_qword_vector<1>(qweight_ + qweight_base_offset_);
        uint32_t const qword1 = load_qword_vector<1>(
            qweight_ + qweight_base_offset_ + kQweightWordStrideWords);
        half2 const* scale_vec = cached_scales_;
        half2 const* zp_vec = cached_zp_;

        half2 deq[2];
        half2* frag_vec = reinterpret_cast<half2*>(frag.data());
        marlin::dequant<half2, vllm::kU8.id(), false>(
            static_cast<int>(qword0), deq);
        frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
        frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
        marlin::dequant<half2, vllm::kU8.id(), false>(
            static_cast<int>(qword1), deq);
        frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
        frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
      }
    } else {
      int const qweight_base = qweight_base_offset_;
      constexpr int kStridedQweightDeltaWords =
          64 * kQweightWordStrideWords;
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
        int const qweight_base_s =
            qweight_base + s * kStridedQweightDeltaWords;
        if constexpr (ThreadMap::Iterations::kContiguous == 4) {
          uint4 const qwords0 =
              load_qword_vector<4>(qweight_ + qweight_base_s);
          uint4 const qwords1 = load_qword_vector<4>(
              qweight_ + qweight_base_s + kQweightWordStrideWords);
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            constexpr int kAccess = ThreadMap::kElementsPerAccess;
            int const frag_base =
                (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
            uint32_t const qword0 = qword_from_vector(qwords0, c);
            uint32_t const qword1 = qword_from_vector(qwords1, c);
            half2 const* scale_vec = cached_scales_ + c * 4;
            half2 const* zp_vec = cached_zp_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU8.id(), false>(
                static_cast<int>(qword0), deq);
            frag_vec[0] =
                __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
            frag_vec[1] =
                __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
            marlin::dequant<half2, vllm::kU8.id(), false>(
                static_cast<int>(qword1), deq);
            frag_vec[2] =
                __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
            frag_vec[3] =
                __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
          }
        } else if constexpr (ThreadMap::Iterations::kContiguous == 2) {
          uint2 const qwords0 =
              load_qword_vector<2>(qweight_ + qweight_base_s);
          uint2 const qwords1 = load_qword_vector<2>(
              qweight_ + qweight_base_s + kQweightWordStrideWords);
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            constexpr int kAccess = ThreadMap::kElementsPerAccess;
            int const frag_base =
                (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
            uint32_t const qword0 = qword_from_vector(qwords0, c);
            uint32_t const qword1 = qword_from_vector(qwords1, c);
            half2 const* scale_vec = cached_scales_ + c * 4;
            half2 const* zp_vec = cached_zp_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU8.id(), false>(
                static_cast<int>(qword0), deq);
            frag_vec[0] =
                __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
            frag_vec[1] =
                __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
            marlin::dequant<half2, vllm::kU8.id(), false>(
                static_cast<int>(qword1), deq);
            frag_vec[2] =
                __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
            frag_vec[3] =
                __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
          }
        } else {
          static_assert(ThreadMap::Iterations::kContiguous == 1,
                        "Unsupported SM70 Marlin U8 contiguous iteration count.");
          uint32_t const qword0 =
              load_qword_vector<1>(qweight_ + qweight_base_s);
          uint32_t const qword1 = load_qword_vector<1>(
              qweight_ + qweight_base_s + kQweightWordStrideWords);
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = s * kAccess;
          half2 const* scale_vec = cached_scales_;
          half2 const* zp_vec = cached_zp_;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword0), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword1), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
        }
      }
    }
  }

  CUTLASS_DEVICE
  void load(Fragment& frag) const {
    if (!mask_enabled_) {
      return;
    }

    if constexpr (kGroupSize != -1) {
      int const first_logical_k = k_offset_ + thread_offset_.strided();
      cache_current_group_metadata(scale_group(first_logical_k));
    }

    load_cta_n_aligned(frag);
  }
};

struct Sm70U8ZpGemmSpec {
  template <typename Shape, typename ThreadMap, int GroupSize>
  using IteratorB = Sm70U8ZpIteratorB<Shape, ThreadMap, GroupSize>;
};

template <int CtaM, int CtaN, int Warps, int GroupSize>
using Sm70U8ZpGemmTraits =
    Sm70MarlinGemmTraits<Sm70U8ZpGemmSpec, CtaM, CtaN, Warps, GroupSize>;

template <int CtaM, int CtaN, int Warps, int GroupSize>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_marlin_u8_gemm_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    cutlass::half_t const* __restrict__ b_scales,
    cutlass::half_t const* __restrict__ b_zeros,
    cutlass::half_t* __restrict__ c, int m, int n, int k, int lda) {
  using Traits = Sm70U8ZpGemmTraits<CtaM, CtaN, Warps, GroupSize>;
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

  typename Mma::IteratorA iterator_A(
      typename Mma::IteratorA::Params(layout_a),
      const_cast<cutlass::half_t*>(a), cutlass::MatrixCoord(m, k), thread_idx,
      tb_offset_A);
  typename Mma::IteratorB iterator_B(
      typename Mma::IteratorB::Params(k, n),
      reinterpret_cast<uint32_t const*>(b_q_weight),
      reinterpret_cast<half const*>(b_scales),
      reinterpret_cast<half const*>(b_zeros), thread_idx, tb_offset_B);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  int const gemm_k_iterations = (k + kCtaK - 1) / kCtaK;
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

template <int CtaM, int CtaN, int Warps, int GroupSize>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_marlin_u8_gemm_splitk_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    cutlass::half_t const* __restrict__ b_scales,
    cutlass::half_t const* __restrict__ b_zeros,
    cutlass::half_t* __restrict__ c, int m, int n, int k, int lda, int requested_split_k) {
  using Traits = Sm70U8ZpGemmTraits<CtaM, CtaN, Warps, GroupSize>;
  using Mma = typename Traits::Mma;
  using AtomicEpilogue = Sm70AtomicFp16Epilogue<Traits>;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int const thread_idx = threadIdx.x;
  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const lane_idx = threadIdx.x % 32;
  Sm70SplitKPartition const partition =
      sm70_splitk_partition<GroupSize>(k, requested_split_k, int(blockIdx.z));
  if (partition.partition_k == 0) {
    return;
  }

  cutlass::MatrixCoord tb_offset_A{int(blockIdx.x) * CtaM,
                                   partition.k_begin};
  cutlass::MatrixCoord tb_offset_B{partition.k_begin, int(blockIdx.y) * CtaN};
  cutlass::MatrixCoord tb_offset_C{int(blockIdx.x) * CtaM,
                                   int(blockIdx.y) * CtaN};

  typename Traits::LayoutA layout_a(lda);
  typename Traits::LayoutC layout_c(n);

  typename Mma::IteratorA iterator_A(
      typename Mma::IteratorA::Params(layout_a),
      const_cast<cutlass::half_t*>(a), cutlass::MatrixCoord(m, k), thread_idx,
      tb_offset_A);
  typename Mma::IteratorB iterator_B(
      typename Mma::IteratorB::Params(k, n),
      reinterpret_cast<uint32_t const*>(b_q_weight),
      reinterpret_cast<half const*>(b_scales),
      reinterpret_cast<half const*>(b_zeros), thread_idx, tb_offset_B);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  int const gemm_k_iterations = partition.partition_k / kCtaK;
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename AtomicEpilogue::OutputTileIterator iterator_D(
      typename AtomicEpilogue::OutputTileIterator::Params(layout_c),
      c, cutlass::MatrixCoord(m, n),
      thread_idx, tb_offset_C);

  AtomicEpilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                          lane_idx);
  epilogue(iterator_D, accumulators, c, n);
}

}  // namespace

template <int CtaM, int CtaN, int Warps, int GroupSize>
torch::Tensor launch_sm70_marlin_u8_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zeros, int64_t size_m,
    int64_t size_n, int64_t size_k, int requested_split_k) {
  auto kernel =
      sm70_marlin_u8_gemm_kernel<CtaM, CtaN, Warps, GroupSize>;
  using SharedStorage = typename Sm70U8ZpGemmTraits<
      CtaM, CtaN, Warps, GroupSize>::SharedStorage;
  size_t smem_bytes = configure_sm70_dynamic_smem<SharedStorage>(kernel);

  dim3 block(Warps * 32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  if (requested_split_k == 1) {
    dim3 grid = sm70_marlin_cta_grid(size_m, size_n, CtaM, CtaN);
    kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
        reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
        reinterpret_cast<cutlass::half_t const*>(b_scales.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t const*>(b_zeros.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
        static_cast<int>(size_m), static_cast<int>(size_n),
        static_cast<int>(size_k), static_cast<int>(a.stride(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
  }

  TORCH_CHECK(size_k % int64_t(kCtaK) == 0,
              "SM70 Marlin U8 requires K divisible by 32 for requested_split_k > 1. "
              "Got K=",
              size_k, ", requested_split_k=", requested_split_k, ".");

  auto split_kernel =
      sm70_marlin_u8_gemm_splitk_kernel<CtaM, CtaN, Warps, GroupSize>;
  smem_bytes = configure_sm70_dynamic_smem<SharedStorage>(split_kernel);

  int64_t const numel = size_m * size_n;
  C10_CUDA_CHECK(cudaMemsetAsync(
      c.data_ptr<at::Half>(), 0,
      static_cast<size_t>(numel) * sizeof(at::Half), stream));

  dim3 grid = sm70_marlin_cta_grid(size_m, size_n, CtaM, CtaN);
  int const active_split_k =
      sm70_active_split_k(static_cast<int>(size_k), requested_split_k);
  grid.z = static_cast<unsigned>(active_split_k);
  split_kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
      reinterpret_cast<cutlass::half_t const*>(b_scales.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t const*>(b_zeros.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
      static_cast<int>(size_m), static_cast<int>(size_n),
      static_cast<int>(size_k), static_cast<int>(a.stride(0)), requested_split_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return c;
}

struct Sm70U8Launcher {
  torch::Tensor& a;
  torch::Tensor& c;
  torch::Tensor& b_q_weight;
  torch::Tensor& b_scales;
  torch::Tensor& b_zeros;
  int64_t size_m;
  int64_t size_n;
  int64_t size_k;
  int requested_split_k;

  template <int CtaM, int CtaN, int Warps, int GroupSize>
  torch::Tensor operator()() const {
    return launch_sm70_marlin_u8_gemm<CtaM, CtaN, Warps, GroupSize>(
        a, c, b_q_weight, b_scales, b_zeros, size_m, size_n, size_k, requested_split_k);
  }
};

torch::Tensor sm70_marlin_u8_gemm(torch::Tensor& a, torch::Tensor& c,
                                  torch::Tensor& b_q_weight,
                                  torch::Tensor& b_scales,
                                  torch::Tensor& b_zeros, int64_t size_m,
                                  int64_t size_n, int64_t size_k,
                                  int64_t group_size) {
  c10::cuda::CUDAGuard device_guard(a.device());

  Sm70CtaGeometry const geometry =
      sm70_marlin_dense_auto_cta_geometry(size_m, size_n);
  validate_sm70_marlin_dense_cta_geometry_supported("SM70 Marlin U8", geometry);
  validate_sm70_marlin_dense_cta_n_alignment("SM70 Marlin U8", geometry, size_n);
  int const requested_split_k =
      sm70_marlin_dense_auto_requested_split_k(size_m, size_n, size_k, geometry);
  Sm70U8Launcher const launcher{
      a, c, b_q_weight, b_scales, b_zeros, size_m, size_n, size_k, requested_split_k};
  return dispatch_sm70_marlin_geometry(launcher, geometry, group_size, "U8");
}
