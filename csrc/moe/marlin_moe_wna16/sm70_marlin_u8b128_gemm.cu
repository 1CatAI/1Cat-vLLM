#include "core/registration.h"
#include "moe/marlin_moe_wna16/sm70_marlin_gemm.cuh"
#include "quantization/marlin/dequant.h"
#include "quantization/marlin/sm70_marlin_iterator_utils.cuh"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <torch/library.h>

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

using marlin::sm70::load_qword_vector;
using marlin::sm70::qword_from_vector;
using marlin::sm70::u8_packed_macro_n_qweight_offset_from_logical;

namespace marlin_moe_wna16 {
namespace {

constexpr int kU8B128ValuesPerAccess = 8;

template <typename Shape_, typename ThreadMap_, int GroupSize_,
          int PackedMacroN_, bool UseMetadataVectorWords_ = true>
class Sm70MoeU8B128IteratorB {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  static int const kGroupSize = GroupSize_;
  static int const kPackedMacroN = PackedMacroN_;
  static constexpr bool kUseMetadataVectorWords = UseMetadataVectorWords_;
  using Element = cutlass::half_t;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
    static_assert(Shape::kN == 64 || Shape::kN == 128 || Shape::kN == 256,
                "SM70 Marlin MoE U8B128 IteratorB expects CTA_N in {64, 128, 256}.");
  static_assert(ThreadMap::Iterations::kContiguous ==
                    Shape::kN / kQuantTileN,
                "SM70 Marlin MoE U8B128 IteratorB expects one contiguous iteration per "
                "64-column quant tile.");
  static_assert(ThreadMap::Delta::kContiguous == kQuantTileN,
                "SM70 Marlin MoE U8B128 IteratorB expects 64-column deltas.");
  static_assert(ThreadMap::kElementsPerAccess == kU8B128ValuesPerAccess,
                "SM70 Marlin MoE U8B128 IteratorB expects two packed U8B128 words "
                "per access.");
  static_assert(ThreadMap::Iterations::kStrided >= 1,
                "SM70 Marlin MoE U8-family IteratorB expects one or more strided "
                "iterations.");
  static constexpr int kQweightWordStrideWords = kPackedMacroN / kQuantTileN;

  struct Params {
    int size_k;
    int size_n;
    int num_groups;

    CUTLASS_HOST_DEVICE
    Params() : size_k(0), size_n(0), num_groups(0) {}

    CUTLASS_HOST_DEVICE
    Params(int size_k_, int size_n_)
        : size_k(size_k_),
          size_n(size_n_),
          num_groups(GroupSize_ == -1 ? 1 : size_k_ / GroupSize_) {}
  };

 private:
  uint32_t const* qweight_;
  half const* scales_expert_base_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int qweight_base_offset_;
  int expert_;
  int k_offset_;
  int n_offset_;
  bool mask_enabled_;
  mutable half2 cached_scales_[ThreadMap::Iterations::kContiguous * 4];

 public:
  CUTLASS_DEVICE
  Sm70MoeU8B128IteratorB(Params const& params, uint32_t const* qweight,
                         half const* scales, half const*, int thread_id,
                         int expert, int k_offset, int n_offset)
      : qweight_(qweight),
        scales_expert_base_(scales +
                            (expert >= 0 ? expert : 0) *
                                params.num_groups * params.size_n),
        params_(params),
        thread_offset_(ThreadMap::initial_offset(thread_id)),
        expert_(expert),
        k_offset_(k_offset),
        n_offset_(n_offset),
        mask_enabled_(expert >= 0) {
    int const logical_k = k_offset_ + thread_offset_.strided();
    int const logical_n = n_offset_ + thread_offset_.contiguous();
    qweight_base_offset_ =
        expert_qweight_base_offset() +
        qweight_offset_from_logical(params_, logical_k, logical_n);
    if constexpr (kGroupSize == -1) {
      cache_current_group_metadata(0);
    }
  }

  CUTLASS_DEVICE
  Sm70MoeU8B128IteratorB& operator++() {
    int const k_advance_qwords =
        (Shape::kK / kQuantTileK) * (params_.size_n * 4);
    k_offset_ += Shape::kK;
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
  void enable_mask() { mask_enabled_ = expert_ >= 0; }

  CUTLASS_DEVICE
  int expert_qweight_base_offset() const {
    return expert_ * (params_.size_k / kQuantTileK) * (params_.size_n * 4);
  }

  CUTLASS_DEVICE
  int metadata_group_offset(int group) const {
    return group * params_.size_n;
  }

  CUTLASS_DEVICE
  int scale_group(int logical_k) const {
    if constexpr (kGroupSize == -1) {
      return 0;
    } else {
      static_assert(kGroupSize == 32 || kGroupSize == 64 ||
                        kGroupSize == 128,
                    "SM70 Marlin MoE U8B128 supports only group sizes -1, 32, 64, "
                    "and 128.");
      return logical_k / kGroupSize;
    }
  }

  CUTLASS_DEVICE
  static int qweight_offset_from_logical(Params const& params, int logical_k,
                                         int logical_n) {
    return u8_packed_macro_n_qweight_offset_from_logical<kPackedMacroN>(
        params.size_n, logical_k, logical_n);
  }

  CUTLASS_DEVICE
  void cache_metadata_lane_vectors(int c, int group, int cache_n) const {
    int const metadata_offset = metadata_group_offset(group) + cache_n;
    half2 const* scale_vec = reinterpret_cast<half2 const*>(
        scales_expert_base_ + metadata_offset);
    half2* scale_cache = cached_scales_ + c * 4;
    scale_cache[0] = scale_vec[0];
    scale_cache[1] = scale_vec[1];
    scale_cache[2] = scale_vec[2];
    scale_cache[3] = scale_vec[3];
  }

  CUTLASS_DEVICE
  void cache_metadata_vector_words(int c, int group, int cache_n) const {
    int const metadata_offset = metadata_group_offset(group) + cache_n;
    uint4 const scale_words = *reinterpret_cast<uint4 const*>(
        scales_expert_base_ + metadata_offset);
    half2 const* scale_vec = reinterpret_cast<half2 const*>(&scale_words);
    half2* scale_cache = cached_scales_ + c * 4;
    scale_cache[0] = scale_vec[0];
    scale_cache[1] = scale_vec[1];
    scale_cache[2] = scale_vec[2];
    scale_cache[3] = scale_vec[3];
  }

  CUTLASS_DEVICE
  void cache_current_group_metadata(int group) const {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      int const cache_n =
          n_offset_ + thread_offset_.contiguous() +
          c * ThreadMap::Delta::kContiguous;
      if constexpr (kUseMetadataVectorWords) {
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

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU8B128.id(), false>(
              static_cast<int>(qword0), deq);
          frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
          frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
          marlin::dequant<half2, vllm::kU8B128.id(), false>(
              static_cast<int>(qword1), deq);
          frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
          frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
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

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU8B128.id(), false>(
              static_cast<int>(qword0), deq);
          frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
          frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
          marlin::dequant<half2, vllm::kU8B128.id(), false>(
              static_cast<int>(qword1), deq);
          frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
          frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
        }
      } else {
        static_assert(ThreadMap::Iterations::kContiguous == 1,
                      "Unsupported SM70 Marlin MoE U8B128 contiguous iteration count.");
        uint32_t const qword0 =
            load_qword_vector<1>(qweight_ + qweight_base_offset_);
        uint32_t const qword1 = load_qword_vector<1>(
            qweight_ + qweight_base_offset_ + kQweightWordStrideWords);
        half2 const* scale_vec = cached_scales_;

        half2 deq[2];
        half2* frag_vec = reinterpret_cast<half2*>(frag.data());
        marlin::dequant<half2, vllm::kU8B128.id(), false>(
            static_cast<int>(qword0), deq);
        frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
        frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
        marlin::dequant<half2, vllm::kU8B128.id(), false>(
            static_cast<int>(qword1), deq);
        frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
        frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
      }
    } else {
      int const logical_n_base = n_offset_ + thread_offset_.contiguous();
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
        int const logical_k_s =
            k_offset_ + thread_offset_.strided() +
            s * ThreadMap::Delta::kStrided;
        if constexpr (kGroupSize != -1) {
          cache_current_group_metadata(scale_group(logical_k_s));
        }
        int const qweight_base_s =
            expert_qweight_base_offset() +
            qweight_offset_from_logical(params_, logical_k_s, logical_n_base);
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

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU8B128.id(), false>(
                static_cast<int>(qword0), deq);
            frag_vec[0] =
                __hmul2(deq[0], scale_vec[0]);
            frag_vec[1] =
                __hmul2(deq[1], scale_vec[1]);
            marlin::dequant<half2, vllm::kU8B128.id(), false>(
                static_cast<int>(qword1), deq);
            frag_vec[2] =
                __hmul2(deq[0], scale_vec[2]);
            frag_vec[3] =
                __hmul2(deq[1], scale_vec[3]);
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

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU8B128.id(), false>(
                static_cast<int>(qword0), deq);
            frag_vec[0] =
                __hmul2(deq[0], scale_vec[0]);
            frag_vec[1] =
                __hmul2(deq[1], scale_vec[1]);
            marlin::dequant<half2, vllm::kU8B128.id(), false>(
                static_cast<int>(qword1), deq);
            frag_vec[2] =
                __hmul2(deq[0], scale_vec[2]);
            frag_vec[3] =
                __hmul2(deq[1], scale_vec[3]);
          }
        } else {
          static_assert(ThreadMap::Iterations::kContiguous == 1,
                        "Unsupported SM70 Marlin MoE U8B128 contiguous iteration count.");
          uint32_t const qword0 =
              load_qword_vector<1>(qweight_ + qweight_base_s);
          uint32_t const qword1 = load_qword_vector<1>(
              qweight_ + qweight_base_s + kQweightWordStrideWords);
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = s * kAccess;
          half2 const* scale_vec = cached_scales_;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU8B128.id(), false>(
              static_cast<int>(qword0), deq);
          frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
          frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
          marlin::dequant<half2, vllm::kU8B128.id(), false>(
              static_cast<int>(qword1), deq);
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

    if constexpr (kGroupSize != -1 &&
                  ThreadMap::Iterations::kStrided == 1) {
      int const first_logical_k = k_offset_ + thread_offset_.strided();
      cache_current_group_metadata(scale_group(first_logical_k));
    }

    load_cta_n_aligned(frag);
  }
};

template <bool UseMetadataVectorWords = true>
struct Sm70MoeU8B128GemmSpec {
  using ScaleElement = half;
  using ZeroElement = half;
  static constexpr bool kUsesGlobalScale = false;

  template <typename Shape, typename ThreadMap>
  using IteratorA = Sm70MoeGatherIteratorA<Shape, ThreadMap>;

  template <typename Shape, typename ThreadMap, int GroupSize, int PackedMacroN>
  using IteratorB =
      Sm70MoeU8B128IteratorB<Shape, ThreadMap, GroupSize, PackedMacroN,
                             UseMetadataVectorWords>;
};

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM,
          int WarpN, int WarpK, int GroupSize, int PackedMacroN,
          bool UseMetadataVectorWords = true>
using Sm70MoeU8B128GemmTraits =
    Sm70MarlinMoeGemmTraits<
        Sm70MoeU8B128GemmSpec<UseMetadataVectorWords>, CtaM, CtaN, CtaK,
        Warps, WarpM, WarpN, WarpK, GroupSize, PackedMacroN>;

template <bool UseMetadataVectorWords = true>
struct Sm70MoeU8B128Launcher {
  torch::Tensor& a;
  torch::Tensor& c;
  torch::Tensor& b_q_weight;
  torch::Tensor& b_scales;
  torch::Tensor& b_zeros;
  torch::Tensor& global_scale;
  torch::Tensor& sorted_token_ids;
  torch::Tensor& expert_ids;
  torch::Tensor& num_tokens_past_padded;
  torch::Tensor& topk_weights;
  int64_t moe_block_size;
  int64_t top_k;
  bool mul_topk_weights;
  int64_t size_m;
  int64_t size_n;
  int64_t size_k;
  int requested_split_k;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM,
          int WarpN, int WarpK, int GroupSize, int PackedMacroN>
  torch::Tensor operator()() const {
    using Traits = Sm70MoeU8B128GemmTraits<CtaM, CtaN, CtaK, Warps, WarpM,
                                           WarpN, WarpK, GroupSize, PackedMacroN,
                                           UseMetadataVectorWords>;
    return launch_sm70_marlin_moe_gemm<Traits>(
        a, c, b_q_weight, b_scales, b_zeros, global_scale, sorted_token_ids,
        expert_ids, num_tokens_past_padded, topk_weights, moe_block_size,
        top_k, mul_topk_weights, size_m, size_n, size_k, requested_split_k);
  }
};

}  // namespace

torch::Tensor sm70_marlin_u8b128_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids, torch::Tensor& num_tokens_past_padded,
    torch::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, int64_t size_m, int64_t size_n, int64_t size_k,
    int64_t group_size) {
  c10::cuda::CUDAGuard device_guard(a.device());

  auto const params = sm70_marlin_moe_auto_stage_params(
      "uint8b128", group_size, moe_block_size, top_k, size_m, size_n, size_k);
  Sm70CtaGeometry const geometry = params.geometry;
  validate_sm70_marlin_moe_stage_cta_geometry_supported("SM70 Marlin MoE uint8b128", geometry);
  validate_sm70_marlin_moe_stage_cta_n_alignment("SM70 Marlin MoE uint8b128", geometry,
                                        size_n);
  TORCH_CHECK(size_k % geometry.cta_k == 0,
              "SM70 Marlin MoE uint8b128 requires K divisible by CTA_K=",
              geometry.cta_k, ". Got K=", size_k, ".");

  auto empty_half = torch::empty({0}, b_scales.options().dtype(at::kHalf));
  auto empty_float = torch::empty(
      {0}, torch::TensorOptions().dtype(at::kFloat).device(a.device()));
  if (params.use_metadata_vector_words) {
    Sm70MoeU8B128Launcher<true> const launcher{
        a, c, b_q_weight, b_scales, empty_half, empty_float, sorted_token_ids,
        expert_ids, num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, size_m, size_n, size_k, params.requested_split_k};
    return dispatch_sm70_marlin_moe_geometry_with_group32(
        launcher, geometry, params.packed_macro_n, group_size, "uint8b128");
  }
  Sm70MoeU8B128Launcher<false> const launcher{
      a, c, b_q_weight, b_scales, empty_half, empty_float, sorted_token_ids,
      expert_ids, num_tokens_past_padded, topk_weights, moe_block_size, top_k,
      mul_topk_weights, size_m, size_n, size_k, params.requested_split_k};
  return dispatch_sm70_marlin_moe_geometry_with_group32(
      launcher, geometry, params.packed_macro_n, group_size, "uint8b128");
}

}  // namespace marlin_moe_wna16
