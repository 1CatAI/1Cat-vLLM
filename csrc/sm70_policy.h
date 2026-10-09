// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm::sm70 {

enum class PolicyField {
#define SM70_POLICY_FIELD(field, alias, calculation) field,
#include "sm70_policy_fields.inc"
#undef SM70_POLICY_FIELD
  count
};

inline constexpr size_t policy_size = static_cast<size_t>(PolicyField::count);
inline constexpr const char* policy_names[] = {
#define SM70_POLICY_FIELD(field, alias, calculation) alias,
#include "sm70_policy_fields.inc"
#undef SM70_POLICY_FIELD
};
inline constexpr bool calculation_fields[] = {
#define SM70_POLICY_FIELD(field, alias, calculation) calculation,
#include "sm70_policy_fields.inc"
#undef SM70_POLICY_FIELD
};
inline const char* policy_name(PolicyField field) {
  return policy_names[static_cast<size_t>(field)];
}

// Legacy direct callers have no policy argument. Capture compatibility inputs
// once, without mutating the process environment. Prepared engines supply all
// values explicitly and never consult this fallback.
inline const std::vector<std::string>& legacy_policy() {
  static const auto values = [] {
    std::vector<std::string> result;
    result.reserve(policy_size);
    for (const auto* name : policy_names) {
      const auto* raw = std::getenv(name);
      result.emplace_back(raw ? raw : "\x1f");
    }
    return result;
  }();
  return values;
}

inline thread_local const std::vector<std::string>* active_policy = nullptr;
inline thread_local uint64_t active_policy_key = 0;

inline const char* policy_value(PolicyField field) {
  const auto& values = active_policy ? *active_policy : legacy_policy();
  const auto& value = values[static_cast<size_t>(field)];
  return value == "\x1f" ? nullptr : value.c_str();
}

class PolicyScope {
 public:
  explicit PolicyScope(const std::vector<std::string>& values)
      : previous_(active_policy), previous_key_(active_policy_key) {
    if (values.empty()) return;
    if (values.size() != policy_size) {
      throw std::invalid_argument("SM70 native policy ABI size mismatch");
    }
    active_policy = &values;
    uint64_t key = 14695981039346656037ull;
    for (size_t i = 0; i < values.size(); ++i) {
      if (!calculation_fields[i]) continue;
      const auto& value = values[i];
      for (unsigned char c : value) key = (key ^ c) * 1099511628211ull;
      key = (key ^ 0xff) * 1099511628211ull;
    }
    active_policy_key = key;
  }
  ~PolicyScope() {
    active_policy = previous_;
    active_policy_key = previous_key_;
  }
  PolicyScope(const PolicyScope&) = delete;
  PolicyScope& operator=(const PolicyScope&) = delete;

 private:
  const std::vector<std::string>* previous_;
  uint64_t previous_key_;
};

// This scope is host-only and ends after launch. Capture records the selected
// kernels; graph replay needs neither TLS nor policy storage addresses.
template <typename Result, typename... Args>
auto with_policy(Result (*operation)(Args...)) {
  return [operation](
             Args... args,
             std::optional<std::vector<std::string>> native_policy) -> Result {
    const std::vector<std::string> empty;
    const PolicyScope scope(native_policy ? *native_policy : empty);
    return operation(args...);
  };
}

}  // namespace vllm::sm70
