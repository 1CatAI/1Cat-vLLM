#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/numeric_conversion.h"

namespace marlin::sm70 {

template <
    typename Shape_, typename IteratorA_, typename SmemIteratorA_,
    typename IteratorB_, typename SmemIteratorB_, typename ElementC_,
    typename LayoutC_, typename Policy_,
    typename TransformA_ = cutlass::NumericArrayConverter<
        typename SmemIteratorA_::Element, typename IteratorA_::Element,
        IteratorA_::Fragment::kElements>,
    typename TransformB_ = cutlass::NumericArrayConverter<
        typename SmemIteratorB_::Element, typename IteratorB_::Element,
        IteratorB_::Fragment::kElements>,
    typename Enable = bool>
class Sm70MarlinMmaPipelined
    : public cutlass::gemm::threadblock::MmaBase<Shape_, Policy_, 2> {
 public:
  using Base = cutlass::gemm::threadblock::MmaBase<Shape_, Policy_, 2>;

  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;

  using FragmentA = typename IteratorA::Fragment;
  using FragmentB = typename IteratorB::Fragment;
  using FragmentC = typename Policy::Operator::FragmentC;
  using Operator = typename Policy::Operator;
  using ArchTag = typename Policy::Operator::ArchTag;

  static cutlass::ComplexTransform const kTransformA = Operator::kTransformA;
  static cutlass::ComplexTransform const kTransformB = Operator::kTransformB;

  static_assert(Base::kStages == 2,
                "Sm70MarlinMmaPipelined requires a two-stage pipeline.");

 protected:
  Operator warp_mma;
  SmemIteratorA smem_iterator_A_;
  SmemIteratorB smem_iterator_B_;
  TransformA transform_A_;
  TransformB transform_B_;
  typename Base::TensorRefA warp_operand_A_ref_;
  typename Base::TensorRefB warp_operand_B_ref_;
  int smem_write_stage_idx;
  int lane_idx_;
  int warp_idx_m_;
  int warp_idx_n_;
  int warp_tile_k_group_;

 private:
  static constexpr int kStageKGroups =
      Policy::kPartitionsK * Base::kWarpGemmIterations;
  static constexpr int kCircularKGroups = Base::kStages * kStageKGroups;

  CUTLASS_DEVICE
  static int normalize_warp_tile_k_group(int k_group) {
    int normalized = k_group % kCircularKGroups;
    return normalized < 0 ? normalized + kCircularKGroups : normalized;
  }

  CUTLASS_DEVICE
  void reset_phase_aware_warp_tile_iterators(int k_group) {
    warp_tile_k_group_ = normalize_warp_tile_k_group(k_group);
    this->warp_tile_iterator_A_ =
        typename Operator::IteratorA(warp_operand_A_ref_, lane_idx_);
    this->warp_tile_iterator_B_ =
        typename Operator::IteratorB(warp_operand_B_ref_, lane_idx_);
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m_, 0});
    this->warp_tile_iterator_B_.add_tile_offset({0, warp_idx_n_});

    for (int i = 0; i < warp_tile_k_group_; ++i) {
      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;
    }
  }

  CUTLASS_DEVICE
  void add_initial_warp_tile_offset(int k_group) {
    reset_phase_aware_warp_tile_iterators(k_group);
  }

  CUTLASS_DEVICE
  void add_warp_tile_k_group_offset(int k_group_delta) {
    reset_phase_aware_warp_tile_iterators(warp_tile_k_group_ + k_group_delta);
  }

  CUTLASS_DEVICE
  void advance_warp_tile_iterators() {
    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;
    warp_tile_k_group_ = normalize_warp_tile_k_group(warp_tile_k_group_ + 1);
  }

 public:
  CUTLASS_DEVICE
  Sm70MarlinMmaPipelined(
      typename Base::SharedStorage& shared_storage, int thread_idx,
      int warp_idx, int lane_idx, TransformA transform_A = TransformA(),
      TransformB transform_B = TransformB())
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        transform_A_(transform_A),
        transform_B_(transform_B),
        warp_operand_A_ref_(shared_storage.operand_A_ref()),
        warp_operand_B_ref_(shared_storage.operand_B_ref()),
        smem_write_stage_idx(0),
        lane_idx_(lane_idx),
        warp_idx_m_(0),
        warp_idx_n_(0),
        warp_tile_k_group_(0) {
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    warp_idx_m_ = warp_idx_mn % Base::WarpCount::kM;
    warp_idx_n_ = warp_idx_mn / Base::WarpCount::kM;

    add_initial_warp_tile_offset(Base::kWarpGemmIterations * warp_idx_k);
  }

  CUTLASS_DEVICE
  void advance_smem_write_stage() {
    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    if (smem_write_stage_idx == 1) {
      this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
    }

    smem_write_stage_idx ^= 1;
  }

  CUTLASS_DEVICE
  void advance_smem_stages() {
    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    if (smem_write_stage_idx == 1) {
      this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});

      if constexpr (Policy::kPartitionsK > 1) {
        add_warp_tile_k_group_offset(
            (Policy::kPartitionsK - 1) * Base::kWarpGemmIterations);
      }
    } else {
      add_warp_tile_k_group_offset(
          -(Policy::kPartitionsK + 1) * Base::kWarpGemmIterations);
    }

    smem_write_stage_idx ^= 1;
  }

  CUTLASS_DEVICE
  void prologue(IteratorA& iterator_A, IteratorB& iterator_B,
                int& /*gemm_k_iterations*/) {
    FragmentA tb_frag_A;
    tb_frag_A.clear();
    iterator_A.load(tb_frag_A);
    ++iterator_A;

    FragmentB tb_frag_B;
    tb_frag_B.clear();
    iterator_B.load(tb_frag_B);
    ++iterator_B;

    this->smem_iterator_A_.store(transform_A_(tb_frag_A));
    this->smem_iterator_B_.store(transform_B_(tb_frag_B));

    advance_smem_write_stage();
  }

  CUTLASS_DEVICE
  void gmem_wait() { __syncthreads(); }

  CUTLASS_DEVICE
  void gemm_iters(int gemm_k_iterations, FragmentC& accum,
                  IteratorA& iterator_A, IteratorB& iterator_B) {
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(warp_frag_A[0]);

    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);
    advance_warp_tile_iterators();

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          this->smem_iterator_A_.store(transform_A_(tb_frag_A));
          this->smem_iterator_B_.store(transform_B_(tb_frag_B));

          gmem_wait();
          advance_smem_stages();
        }

        this->warp_tile_iterator_A_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        advance_warp_tile_iterators();

        if (warp_mma_k == 0) {
          tb_frag_A.clear();
          iterator_A.load(tb_frag_A);
          ++iterator_A;

          tb_frag_B.clear();
          iterator_B.load(tb_frag_B);
          ++iterator_B;

          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
        }

        warp_mma(accum, warp_frag_A[warp_mma_k % 2],
                 warp_frag_B[warp_mma_k % 2], accum);
      }
    }
  }

  CUTLASS_DEVICE
  void operator()(int gemm_k_iterations, FragmentC& accum, IteratorA iterator_A,
                  IteratorB iterator_B, FragmentC const& src_accum) {
    prologue(iterator_A, iterator_B, gemm_k_iterations);
    gmem_wait();
    accum = src_accum;
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
  }
};

}  // namespace marlin::sm70
