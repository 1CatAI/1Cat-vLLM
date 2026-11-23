// SPDX-License-Identifier: Apache-2.0
// Alternating 64-byte slots per rank. Entering epoch j+2 requires every
// peer to have published j+1 and therefore finished reading the j slots.
#include <chrono>
#include <thread>
#include <torch/library.h>
#include <ATen/ATen.h>

namespace v41_solo_native {
at::Tensor host_agreement(at::Tensor slots, int64_t rank, int64_t epoch,
                          int64_t original, int64_t padded, bool ubatch,
                          int64_t mode, double timeout_s) {
  TORCH_CHECK(!slots.is_cuda() && slots.scalar_type() == at::kLong &&
              slots.is_contiguous() && slots.numel() == 64);
  TORCH_CHECK(rank >= 0 && rank < 4 && epoch > 0);
  auto result = at::empty({4, 4}, slots.options().dtype(at::kInt));
  auto* output = result.data_ptr<int32_t>();
  auto* base = slots.data_ptr<int64_t>() + (epoch & 1) * 32;
  auto* mine = base + rank * 8;
  mine[1] = original;
  mine[2] = padded;
  mine[3] = ubatch;
  mine[4] = mode;
  __atomic_store_n(mine, epoch, __ATOMIC_RELEASE);
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  for (int peer = 0; peer < 4; ++peer) {
    const auto* slot = base + peer * 8;
    unsigned polls = 0;
    while (__atomic_load_n(slot, __ATOMIC_ACQUIRE) != epoch) {
#if defined(__x86_64__)
      __builtin_ia32_pause();
#endif
      if (++polls == 1024) {
        polls = 0;
        auto elapsed = Clock::now() - start;
        TORCH_CHECK(std::chrono::duration<double>(elapsed).count() < timeout_s,
                    "V4.1 DP metadata exchange timed out waiting for rank ", peer);
        if (elapsed > std::chrono::microseconds(100))
          std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }
    for (int field = 0; field < 4; ++field)
      output[4 * field + peer] = static_cast<int32_t>(slot[field + 1]);
  }
  return result;
}
}  // namespace v41_solo_native

TORCH_LIBRARY_FRAGMENT(v41solo, m) {
  m.def("host_agreement(Tensor(a!) slots, int rank, int epoch, int original, int padded, bool ubatch, int mode, float timeout_s) -> Tensor");
}
TORCH_LIBRARY_IMPL(v41solo, CPU, m) {
  m.impl("host_agreement", v41_solo_native::host_agreement);
}
