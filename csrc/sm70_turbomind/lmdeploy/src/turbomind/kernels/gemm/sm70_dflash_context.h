// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cstdlib>

namespace turbomind::gemm {

// The TP4 context projection's small rounding differences can change draft
// acceptance. Keep one measured reduction tree instead of independently
// timing different split-K trees on each rank during graph warmup.
inline bool UseSm70DflashContextFcStableReduction(int m, int n, int k) {
  if (m < 1 || m > 8 || n != 1280 || k != 25600) {
    return false;
  }
  const char* enabled = std::getenv("VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC");
  return enabled && std::atoi(enabled) != 0;
}

}  // namespace turbomind::gemm
