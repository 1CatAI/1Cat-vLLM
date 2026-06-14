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
using marlin::sm70::u4_cta_n_qweight_offset_from_logical;

namespace marlin_moe_wna16 {
namespace {

constexpr int kU4ValuesPerWord = 8;

template <typename Shape_, typename ThreadMap_, int GroupSize_>
class Sm70MoeU4ZpIteratorB {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  static int const kGroupSize = GroupSize_;
  using Element = cutlass::half_t;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  static_assert(Shape::kK == kCtaK,
                "SM70 Marlin MoE U4 IteratorB expects CTA_K=32.");
  static_assert(Shape::kN == 64 || Shape::kN == 128 || Shape::kN == 256,
                "SM70 Marlin MoE U4 IteratorB expects CTA_N in {64, 128, 256}.");
  static_assert(ThreadMap::Iterations::kContiguous ==
                    Shape::kN / kQuantTileN,
                "SM70 Marlin MoE U4 IteratorB expects one contiguous iteration per "
                "64-column quant tile.");
  static_assert(ThreadMap::Delta::kContiguous == kQuantTileN,
                "SM70 Marlin MoE U4 IteratorB expects 64-column deltas.");
  static_assert(ThreadMap::kElementsPerAccess == kU4ValuesPerWord,
                "SM70 Marlin MoE U4 IteratorB expects one packed int4 word per "
                "access.");
  static_assert(ThreadMap::Iterations::kStrided == 1 ||
                    ThreadMap::Iterations::kStrided == 2,
                "SM70 Marlin MoE U4-family IteratorB expects one or two "
                "strided iterations.");
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
  half const* zp_expert_base_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int qweight_base_offset_;
  int expert_;
  int k_offset_;
  int n_offset_;
  bool mask_enabled_;
  mutable half2 cached_scales_[ThreadMap::Iterations::kContiguous * 4];
  mutable half2 cached_zp_[ThreadMap::Iterations::kContiguous * 4];

 public:
  CUTLASS_DEVICE
  Sm70MoeU4ZpIteratorB(Params const& params, uint32_t const* qweight,
                       half const* scales, half const* zp, int thread_id,
                       int expert, int k_offset, int n_offset)
      : qweight_(qweight),
        scales_expert_base_(scales +
                            (expert >= 0 ? expert : 0) *
                                params.num_groups * params.size_n),
        zp_expert_base_(zp + (expert >= 0 ? expert : 0) *
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
  Sm70MoeU4ZpIteratorB& operator++() {
    int const k_advance_qwords =
        (Shape::kK / kQuantTileK) * (params_.size_n * 2);
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
    return expert_ * (params_.size_k / kQuantTileK) * (params_.size_n * 2);
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
                    "SM70 Marlin MoE U4 supports only group sizes -1, 32, 64, "
                    "and 128.");
      return logical_k / kGroupSize;
    }
  }

  CUTLASS_DEVICE
  static int qweight_offset_from_logical(Params const& params, int logical_k,
                                         int logical_n) {
    return u4_cta_n_qweight_offset_from_logical<Shape::kN>(
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

    half2 const* zp_vec = reinterpret_cast<half2 const*>(
        zp_expert_base_ + metadata_offset);
    half2* zp_cache = cached_zp_ + c * 4;
    zp_cache[0] = zp_vec[0];
    zp_cache[1] = zp_vec[1];
    zp_cache[2] = zp_vec[2];
    zp_cache[3] = zp_vec[3];
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

    uint4 const zp_words = *reinterpret_cast<uint4 const*>(
        zp_expert_base_ + metadata_offset);
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
        auto const qwords =
            load_qword_vector<4>(qweight_ + qweight_base_offset_);
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = c * kAccess;
          uint32_t const qword = qword_from_vector(qwords, c);
          half2 const* scale_vec = cached_scales_ + c * 4;
          half2 const* zp_vec = cached_zp_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword >> 8), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
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
          half2 const* zp_vec = cached_zp_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword >> 8), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
        }
      } else {
        static_assert(ThreadMap::Iterations::kContiguous == 1,
                      "Unsupported SM70 Marlin MoE U4 contiguous iteration count.");
        uint32_t const qword = load_qword_vector<1>(qweight_ + qweight_base_offset_);
        half2 const* scale_vec = cached_scales_;
        half2 const* zp_vec = cached_zp_;

        half2 deq[2];
        half2* frag_vec = reinterpret_cast<half2*>(frag.data());
        marlin::dequant<half2, vllm::kU4.id(), false>(
            static_cast<int>(qword), deq);
        frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
        frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
        marlin::dequant<half2, vllm::kU4.id(), false>(
            static_cast<int>(qword >> 8), deq);
        frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
        frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
      }
    } else {
      int const qweight_base = qweight_base_offset_;
      constexpr int kStridedQweightDeltaWords =
          32 * (Shape::kN / kQuantTileN);
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
        int const qweight_base_s =
            qweight_base + s * kStridedQweightDeltaWords;
        if constexpr (ThreadMap::Iterations::kContiguous == 4) {
          auto const qwords =
              load_qword_vector<4>(qweight_ + qweight_base_s);
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            constexpr int kAccess = ThreadMap::kElementsPerAccess;
            int const frag_base =
                (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
            uint32_t const qword = qword_from_vector(qwords, c);
            half2 const* scale_vec = cached_scales_ + c * 4;
            half2 const* zp_vec = cached_zp_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword), deq);
            frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
            frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword >> 8), deq);
            frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
            frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
          }
        } else if constexpr (ThreadMap::Iterations::kContiguous == 2) {
          auto const qwords =
              load_qword_vector<2>(qweight_ + qweight_base_s);
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            constexpr int kAccess = ThreadMap::kElementsPerAccess;
            int const frag_base =
                (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
            uint32_t const qword = qword_from_vector(qwords, c);
            half2 const* scale_vec = cached_scales_ + c * 4;
            half2 const* zp_vec = cached_zp_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword), deq);
            frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
            frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword >> 8), deq);
            frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
            frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
          }
        } else {
          static_assert(ThreadMap::Iterations::kContiguous == 1,
                        "Unsupported SM70 Marlin MoE U4 contiguous iteration count.");
          uint32_t const qword = load_qword_vector<1>(qweight_ + qweight_base_s);
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = s * kAccess;
          half2 const* scale_vec = cached_scales_;
          half2 const* zp_vec = cached_zp_;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword >> 8), deq);
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

struct Sm70MoeU4ZpGemmSpec {
  using ScaleElement = half;
  using ZeroElement = half;
  static constexpr bool kUsesGlobalScale = false;

  template <typename Shape, typename ThreadMap>
  using IteratorA = Sm70MoeGatherIteratorA<Shape, ThreadMap>;

  template <typename Shape, typename ThreadMap, int GroupSize>
  using IteratorB = Sm70MoeU4ZpIteratorB<Shape, ThreadMap, GroupSize>;
};

template <int CtaM, int CtaN, int Warps, int GroupSize>
using Sm70MoeU4ZpGemmTraits =
    Sm70MarlinMoeGemmTraits<Sm70MoeU4ZpGemmSpec, CtaM, CtaN, Warps,
                            GroupSize>;

struct Sm70MoeU4Launcher {
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

  template <int CtaM, int CtaN, int Warps, int GroupSize>
  torch::Tensor operator()() const {
    using Traits = Sm70MoeU4ZpGemmTraits<CtaM, CtaN, Warps, GroupSize>;
    return launch_sm70_marlin_moe_gemm<Traits>(
        a, c, b_q_weight, b_scales, b_zeros, global_scale, sorted_token_ids,
        expert_ids, num_tokens_past_padded, topk_weights, moe_block_size,
        top_k, mul_topk_weights, size_m, size_n, size_k, requested_split_k);
  }
};

}  // namespace

torch::Tensor sm70_marlin_u4_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zeros,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t size_m, int64_t size_n, int64_t size_k, int64_t group_size) {
  c10::cuda::CUDAGuard device_guard(a.device());

  Sm70CtaGeometry const geometry =
      sm70_marlin_moe_u4_zp_auto_stage_cta_geometry(size_m, size_n,
                                                    group_size);
  validate_sm70_marlin_moe_stage_cta_geometry_supported("SM70 Marlin MoE U4", geometry);
  validate_sm70_marlin_moe_stage_cta_n_alignment("SM70 Marlin MoE U4", geometry,
                                        size_n);
  TORCH_CHECK(size_k % kCtaK == 0,
              "SM70 Marlin MoE U4 requires K divisible by 32. Got K=",
              size_k, ".");

  int const requested_split_k = sm70_marlin_moe_auto_stage_requested_split_k(
      size_m, size_n, size_k, top_k, geometry);
  auto empty_float = torch::empty(
      {0}, torch::TensorOptions().dtype(at::kFloat).device(a.device()));
  Sm70MoeU4Launcher const launcher{
      a, c, b_q_weight, b_scales, b_zeros, empty_float, sorted_token_ids,
      expert_ids, num_tokens_past_padded, topk_weights, moe_block_size, top_k,
      mul_topk_weights, size_m, size_n, size_k, requested_split_k};
  return dispatch_sm70_marlin_moe_geometry(launcher, geometry, group_size,
                                           "U4");
}

}  // namespace marlin_moe_wna16
