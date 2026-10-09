// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

inline uint64_t policy_key(const std::vector<std::string>& values) {
  uint64_t key = 14695981039346656037ull;
  for (size_t i = 0; i < values.size(); ++i) {
    if (!calculation_fields[i]) continue;
    for (unsigned char c : values[i]) key = (key ^ c) * 1099511628211ull;
    key = (key ^ 0xff) * 1099511628211ull;
  }
  return key;
}

struct PreparedPolicy {
  std::string token;
  std::vector<std::string> values;
  uint64_t key;
};

// A content token survives AOT serialization; it contains no process address.
// Owners register it at initialization. Calls borrow the parsed values and
// precomputed key, avoiding 55 Python-to-C++ string conversions per launch.
inline const PreparedPolicy& prepared_policy(const std::string& token) {
  thread_local const PreparedPolicy* previous = nullptr;
  if (previous && previous->token == token) return *previous;
  static std::mutex mutex;
  static std::unordered_map<std::string, std::unique_ptr<PreparedPolicy>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = cache.find(token);
  if (found == cache.end()) {
    if (token.compare(0, 7, "sm70:1:") != 0) {
      throw std::invalid_argument("Invalid SM70 policy token");
    }
    auto policy = std::make_unique<PreparedPolicy>();
    policy->token = token;
    size_t pos = 7;
    while (pos < token.size() && policy->values.size() < policy_size) {
      const auto colon = token.find(':', pos);
      if (colon == std::string::npos || colon == pos) {
        throw std::invalid_argument("Invalid SM70 policy field length");
      }
      size_t length = 0;
      for (size_t i = pos; i < colon; ++i) {
        if (token[i] < '0' || token[i] > '9' || length > token.size()) {
          throw std::invalid_argument("Invalid SM70 policy field length");
        }
        length = length * 10 + (token[i] - '0');
      }
      pos = colon + 1;
      if (length > token.size() - pos) {
        throw std::invalid_argument("Truncated SM70 policy field");
      }
      policy->values.emplace_back(token.substr(pos, length));
      pos += length;
    }
    if (pos != token.size() || policy->values.size() != policy_size) {
      throw std::invalid_argument("SM70 native policy ABI size mismatch");
    }
    policy->key = policy_key(policy->values);
    found = cache.emplace(token, std::move(policy)).first;
  }
  previous = found->second.get();
  return *previous;
}

inline void prepare_native_policy(const std::string& token) {
  (void)prepared_policy(token);
}

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
    if (values.size() == 1) {
      const auto& policy = prepared_policy(values.front());
      active_policy = &policy.values;
      active_policy_key = policy.key;
      return;
    }
    if (values.size() != policy_size) {
      throw std::invalid_argument("SM70 native policy ABI size mismatch");
    }
    active_policy = &values;
    active_policy_key = policy_key(values);
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
