/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once
#include "gemm/mma_from_smem.h"

namespace cutlass::gemm::threadblock {
// Retain the SM70 software pipeline, accepting the first V tile loaded while
// softmax is computed. This adds no shared-memory or persistent GPU allocation.
template <typename Mma>
class H3PrefetchMma : public Mma {
 public:
  using Mma::Mma;
  using Base = typename Mma::Base;
  using FragmentB = typename Mma::FragmentB;
  using FragmentC = typename Mma::FragmentC;
  using IteratorB = typename Mma::IteratorB;
  using TransformB = typename Mma::TransformB;
  using Operator = typename Mma::Operator;
  using Policy = typename Mma::Policy;
  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;
  using WarpFragmentAScale = typename Mma::WarpIteratorAScale::Fragment;
  using FragmentAScaler =
      FragmentElementwiseScaler<WarpFragmentA, WarpFragmentAScale,
                                Mma::ScaleOperandA>;
  static_assert(!Mma::ScaleOperandA);
  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      int gemm_k_iterations,  ///< number of iterations of the mainloop
      FragmentC& accum,       ///< destination accumulator tile
      // IteratorA iterator_A,                             ///< iterator over A
      // operand in global memory
      IteratorB iterator_B,        ///< iterator over B operand in global memory
      FragmentC const& src_accum,  ///< source accumulator tile
      FragmentB const* prefetched = nullptr,
      // TransformA transform_A = TransformA(),            ///< transformation
      // applied to A fragment
      TransformB transform_B =
          TransformB()) {  ///< transformation applied to B fragment

    //
    // Prologue
    //

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    FragmentB tb_frag_B;

    iterator_B.set_residual_tile(gemm_k_iterations == 1);
    if (prefetched) {
      tb_frag_B = *prefetched;
    } else {
      tb_frag_B.clear();
      iterator_B.load(tb_frag_B);
    }

    ++iterator_B;

    this->smem_iterator_B_.store(transform_B(tb_frag_B));

    ++this->smem_iterator_B_;

    __syncthreads();

    // remember that WarpFragmentAScale and WarpIteratorAScale are empty/no-op
    // if scaling is disabled.

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentAScale warp_frag_A_scale[2];
    WarpFragmentB warp_frag_B[2];
    warp_frag_A[0].clear();
    warp_frag_A_scale[0].clear();
    warp_frag_B[0].clear();

    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_A_scale_.load(warp_frag_A_scale[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_A_scale_;
    ++this->warp_tile_iterator_B_;

    Operator warp_mma;

    int smem_write_stage_idx = 1;

    // Avoid reading out of bounds
    iterator_B.set_residual_tile(gemm_k_iterations == 2);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    // Issue loads during the first warp-level matrix multiply-add *AFTER*
    // issuing shared memory loads (which have the tightest latency
    // requirement).

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.
        bool hasNext = true;

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          if (gemm_k_iterations > 1) {
            // Write fragments to shared memory
            this->smem_iterator_B_.store(transform_B(tb_frag_B));
          }

          __syncthreads();

          ++this->smem_iterator_B_;

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory SMEM: Don't reset iterator A, as
          // we are continuing our iteration at this point
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
          } else {
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
          hasNext = gemm_k_iterations > 1;
        }

        // Only read the next if we need to
        if (hasNext) {
          this->warp_tile_iterator_B_.set_kgroup_index(
              (warp_mma_k + 1) % Base::kWarpGemmIterations);

          this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
          this->warp_tile_iterator_A_scale_.load(
              warp_frag_A_scale[(warp_mma_k + 1) % 2]);
          this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

          ++this->warp_tile_iterator_A_;
          ++this->warp_tile_iterator_A_scale_;
          ++this->warp_tile_iterator_B_;

          if (warp_mma_k == 0) {
            iterator_B.load(tb_frag_B);

            ++iterator_B;

            // Avoid reading out of bounds if this was the last loop iteration
            iterator_B.set_residual_tile(gemm_k_iterations == 3);
            iterator_B.clear_mask(gemm_k_iterations <= 2);
          }
        }

        warp_mma(accum,
                 FragmentAScaler::apply(warp_frag_A[warp_mma_k % 2],
                                        warp_frag_A_scale[warp_mma_k % 2]),
                 warp_frag_B[warp_mma_k % 2], accum);
      }
    }
  }
};
}  // namespace cutlass::gemm::threadblock
