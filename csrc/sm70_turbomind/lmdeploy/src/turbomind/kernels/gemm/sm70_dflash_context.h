// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "sm70_policy.h"

namespace turbomind::gemm {

inline constexpr int kSm70DflashContextStableMaxM = 16;

// The TP4 context projection's small rounding differences can change draft
// acceptance. Keep one measured reduction tree instead of independently
// timing different split-K trees on each rank during graph warmup. Cover the
// entire default F16 tuning range, including batched verification and tails.
// Larger batches keep their established deterministic heuristic dispatch.
inline bool UseSm70DflashContextFcStableReduction(int m, int n, int k) {
  if (m < 1 || m > kSm70DflashContextStableMaxM || n != 1280 || k != 25600) {
    return false;
  }
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::dflash_sharded_context_fc, 0, true) != 0;
}

}  // namespace turbomind::gemm
