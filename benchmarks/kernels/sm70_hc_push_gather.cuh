// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Preserve benchmark aliases while sharing the screened production kernels.
#include "../../csrc/sm70_hc_batch.cuh"

TORCH_LIBRARY_FRAGMENT(_C_custom_ar_flashnext, m) {
  m.def(
      "hc_down_gather(int ptr, Tensor input, Tensor! injection, Tensor! lora) "
      "-> ()");
  m.impl("hc_down_gather", torch::kCUDA, &sm70_hc_batch::run<false>);
  m.def("hc_mix_gather(int ptr, Tensor gate, Tensor x, Tensor! output) -> ()");
  m.impl("hc_mix_gather", torch::kCUDA, &sm70_hc_batch::run<true>);
}
