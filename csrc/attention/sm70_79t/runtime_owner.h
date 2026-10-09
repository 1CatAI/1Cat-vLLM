// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace onecat_sm70_prefill {
// ABI 1: append only in a new ABI. Values are parsed once by the Python owner.
enum class PolicyField {
  QkOverride,
  QkAlgorithm,
  ScoreBlockTokens,
  SerialTail,
  ExactTail,
  DumpTail,
  DirectTail,
  Count
};
enum class CacheSlot { Scores, Q8000Half2, Q8192Half2, Q8000Score, Q8192Score };

class PrefillOwner : public torch::CustomClassHolder {
 public:
  explicit PrefillOwner(std::vector<int64_t> values);
  ~PrefillOwner() override;
  int64_t value(PolicyField field) const {
    return values_[static_cast<size_t>(field)];
  }
  std::vector<int64_t> values() const { return values_; }
  std::vector<int64_t> observations();
  void close();
  at::Tensor q8000(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
                   double scale, bool causal);
  at::Tensor q8192(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
                   double scale, bool causal);

  template <typename T, typename Factory>
  std::shared_ptr<T> workspace(CacheSlot slot, int device, Factory factory) {
    // A workspace constructor can acquire the owner's shared score buffer.
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    TORCH_CHECK(!closed_, "SM70 prefill runtime has been closed");
    auto& entry = caches_[device][slot];
    if (!entry) entry = factory();
    return std::static_pointer_cast<T>(entry);
  }
  template <typename T>
  std::shared_ptr<T> find(CacheSlot slot, int device) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto dev = caches_.find(device);
    if (dev == caches_.end()) return nullptr;
    auto entry = dev->second.find(slot);
    return entry == dev->second.end()
               ? nullptr
               : std::static_pointer_cast<T>(entry->second);
  }

 private:
  at::Tensor run(bool aligned, at::Tensor q, at::Tensor k, at::Tensor v,
                 at::Tensor out, double scale, bool causal);
  const std::vector<int64_t> values_;
  std::recursive_mutex mutex_;
  bool closed_ = false;
  std::array<int64_t, 2> observations_{};
  std::map<int, std::map<CacheSlot, std::shared_ptr<void>>> caches_;
};

// Host-dispatch scope only. CUDA kernels and replay nodes never borrow this
// pointer: their arguments refer to buffers retained by PrefillOwner.
PrefillOwner* current_owner();
int64_t policy_value(PolicyField field);
}  // namespace onecat_sm70_prefill
