// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <utility>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
enum class CollectivePolicyField {
#define COLLECTIVE_POLICY_FIELD(field, alias) field,
#include "custom_all_reduce_policy_fields.inc"
#undef COLLECTIVE_POLICY_FIELD
  count
};

// Immutable inputs are copied into one communicator, never a process cache.
class CollectivePolicy {
 public:
  static constexpr size_t size =
      static_cast<size_t>(CollectivePolicyField::count);
  explicit CollectivePolicy(const std::vector<std::string>& values) {
    if (values.size() != size)
      throw std::invalid_argument("Custom all-reduce policy ABI size mismatch");
    std::copy(values.begin(), values.end(), values_.begin());
    const auto& sum2_push_any = values_[
        static_cast<size_t>(CollectivePolicyField::sm70_tp4_sum2_push_any)];
    if (sum2_push_any != "\x1f" && sum2_push_any != "0" &&
        sum2_push_any != "1")
      throw std::invalid_argument("VLLM_V41_SUM2_PUSH_ANY must be 0 or 1");
  }
  static CollectivePolicy legacy() {
    std::vector<std::string> values;
#define COLLECTIVE_POLICY_FIELD(field, alias) \
  {                                           \
    const char* raw = std::getenv(alias);     \
    values.emplace_back(raw ? raw : "\x1f");  \
  }
#include "custom_all_reduce_policy_fields.inc"
#undef COLLECTIVE_POLICY_FIELD
    return CollectivePolicy(values);
  }
  const char* raw(CollectivePolicyField field) const {
    const auto& value = values_[static_cast<size_t>(field)];
    return value == "\x1f" ? nullptr : value.c_str();
  }

 private:
  std::array<std::string, size> values_;
};
}  // namespace vllm
