// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/kernels/gemm/kernel_impl.h"

namespace turbomind::gemm {

// Admit only dense M=33..64 shapes; other descriptors keep their established
// kernels. Full-tile iterators require exact tiles.
template <class Gemm, bool FullTiles = false>
class DenseBatchSupplyKernelImpl final : public KernelImpl<Gemm> {
 public:
  DenseBatchSupplyKernelImpl() { this->info_.name += "_sm70_batch_supply"; }

  bool is_feasible(const GemmDesc& desc) const noexcept override {
    if (desc.num != 1 || desc.m <= 32 || desc.m > 64 || desc.n < 2048 ||
        desc.k < 1536) {
      return false;
    }
    if constexpr (FullTiles) {
      const auto tile = this->cta_tile_size();
      if (desc.m != tile.x || desc.n % tile.y != 0 || desc.k % tile.z != 0) {
        return false;
      }
    }
    return KernelImpl<Gemm>::is_feasible(desc);
  }
};

}  // namespace turbomind::gemm
