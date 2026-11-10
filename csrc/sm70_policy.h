// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sm70_runtime_state.h"
#include "sm70_marlin_policy.h"

#include <array>
#include <algorithm>
#include <sstream>
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

enum class DispatchOverride { Unset, Default, Reuse, Measure, Invalid };

inline DispatchOverride parse_dispatch_override(const std::string& raw) {
  if (raw == "\x1f" || raw.empty()) return DispatchOverride::Unset;
  if (raw == "default") return DispatchOverride::Default;
  if (raw == "reuse") return DispatchOverride::Reuse;
  if (raw == "measure") return DispatchOverride::Measure;
  return DispatchOverride::Invalid;
}

struct GemmTargetPolicy {
  std::string descriptor;
  std::string entry;
  int cta_m{}, cta_n{}, cta_k{}, splits{}, swizzle{}, require_mgroup{};
  std::string name_contains;
  bool valid = false;
};

inline std::vector<GemmTargetPolicy> parse_gemm_targets(
    const std::string& targets) {
  std::vector<GemmTargetPolicy> result;
  size_t begin = 0;
  while (begin <= targets.size()) {
    const size_t end = targets.find(';', begin);
    const auto entry = targets.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    const size_t sep = entry.find('|');
    if (sep != std::string::npos) {
      GemmTargetPolicy target;
      target.descriptor = entry.substr(0, sep);
      target.entry = entry;
      std::string spec = entry.substr(sep + 1);
      if (const size_t name_sep = spec.find('@');
          name_sep != std::string::npos) {
        target.name_contains = spec.substr(name_sep + 1);
        spec.resize(name_sep);
      }
      std::replace(spec.begin(), spec.end(), 'x', ' ');
      std::replace(spec.begin(), spec.end(), ':', ' ');
      std::replace(spec.begin(), spec.end(), ',', ' ');
      std::istringstream input(spec);
      target.valid = static_cast<bool>(input >> target.cta_m >> target.cta_n >>
                                       target.cta_k >> target.splits >>
                                       target.swizzle >> target.require_mgroup);
      if (target.valid && target.name_contains.empty())
        input >> target.name_contains;
      result.emplace_back(std::move(target));
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return result;
}

struct PreparedPolicy {
  std::string token;
  std::vector<std::string> values;
  uint64_t key;
  std::array<int, policy_size> integers{};
  std::array<bool, policy_size> exact_one{};
  std::array<bool, policy_size> present{};
  DispatchOverride dispatch_override = DispatchOverride::Unset;
  std::vector<GemmTargetPolicy> gemm_targets;
  MarlinOverrides marlin_dense, marlin_moe;

  void parse_scalars() {
    dispatch_override = parse_dispatch_override(
        values[static_cast<size_t>(PolicyField::awq_moe_dispatch_policy)]);
    gemm_targets = parse_gemm_targets(
        values[static_cast<size_t>(PolicyField::awq_tp2_fast_targets)]);
    marlin_dense.parse(
        "DENSE",
        values[static_cast<size_t>(PolicyField::marlin_dense_cta_geometry)],
        values[static_cast<size_t>(PolicyField::marlin_dense_split_k)],
        values[static_cast<size_t>(PolicyField::marlin_dense_metadata_cache)]);
    marlin_moe.parse(
        "MOE",
        values[static_cast<size_t>(PolicyField::marlin_moe_cta_geometry)],
        values[static_cast<size_t>(PolicyField::marlin_moe_split_k)],
        values[static_cast<size_t>(PolicyField::marlin_moe_metadata_cache)]);
    for (size_t i = 0; i < policy_size; ++i) {
      present[i] = values[i] != "\x1f";
      integers[i] = present[i] ? std::atoi(values[i].c_str()) : 0;
      exact_one[i] = values[i] == "1";
    }
  }
};
inline thread_local const PreparedPolicy* active_prepared_policy = nullptr;

// A content token survives AOT serialization; it contains no process address.
// Owners register it at initialization. Calls borrow the parsed values and
// precomputed key, avoiding 55 Python-to-C++ string conversions per launch.
inline const PreparedPolicy& prepared_policy(const std::string& token) {
  if (token.compare(0, 10, "sm70:slot:") == 0) {
    if (!active_runtime)
      throw std::runtime_error(
          "Engine SM70 policy requires an active runtime owner");
    const auto it = active_runtime->policies.find(token);
    if (it == active_runtime->policies.end())
      throw std::runtime_error("Unbound SM70 native policy slot: " + token);
    return *it->second;
  }
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
    policy->parse_scalars();
    found = cache.emplace(token, std::move(policy)).first;
  }
  previous = found->second.get();
  return *previous;
}

// Marlin's retained schemas borrow an immutable initialization binding. They
// never resolve compatibility inputs while an engine owns the native scope.
inline const MarlinOverrides* bound_marlin_policy(bool moe) {
  if (!active_runtime) return nullptr;
  const auto& policy = prepared_policy("sm70:slot:kernel_config.sm70_marlin");
  return moe ? &policy.marlin_moe : &policy.marlin_dense;
}

inline void prepare_native_policy(const std::string& token) {
  (void)prepared_policy(token);
}

inline const char* policy_value(PolicyField field) {
  const auto& values = active_policy ? *active_policy : legacy_policy();
  const auto& value = values[static_cast<size_t>(field)];
  return value == "\x1f" ? nullptr : value.c_str();
}

// Prepared calls consume parsed scalars. A standalone compatibility operation
// may retain its historical per-call environment behavior for newly bound
// fields.
inline int policy_atoi(PolicyField field, int fallback = 0,
                       bool dynamic_legacy = false) {
  const auto index = static_cast<size_t>(field);
  if (active_prepared_policy) {
    return active_prepared_policy->present[index]
               ? active_prepared_policy->integers[index]
               : fallback;
  }
  const char* raw =
      dynamic_legacy ? std::getenv(policy_name(field)) : policy_value(field);
  return raw ? std::atoi(raw) : fallback;
}

inline bool policy_exact_one(PolicyField field, bool fallback = false,
                             bool dynamic_legacy = false) {
  const auto index = static_cast<size_t>(field);
  if (active_prepared_policy) {
    return active_prepared_policy->present[index]
               ? active_prepared_policy->exact_one[index]
               : fallback;
  }
  const char* raw =
      dynamic_legacy ? std::getenv(policy_name(field)) : policy_value(field);
  return raw ? raw[0] == '1' && raw[1] == '\0' : fallback;
}

inline DispatchOverride policy_dispatch_override(PolicyField field) {
  if (active_prepared_policy && field == PolicyField::awq_moe_dispatch_policy)
    return active_prepared_policy->dispatch_override;
  const auto* raw = policy_value(field);
  return parse_dispatch_override(raw ? raw : "\x1f");
}

inline const std::vector<GemmTargetPolicy>& policy_gemm_targets() {
  if (active_prepared_policy) return active_prepared_policy->gemm_targets;
  static const auto legacy = parse_gemm_targets(
      legacy_policy()[static_cast<size_t>(PolicyField::awq_tp2_fast_targets)]);
  return legacy;
}

class PolicyScope {
 public:
  explicit PolicyScope(const std::optional<std::string>& token)
      : previous_(active_policy),
        previous_key_(active_policy_key),
        previous_prepared_(active_prepared_policy) {
    if (!token) return;
    const auto& policy = prepared_policy(*token);
    active_policy = &policy.values;
    active_prepared_policy = &policy;
    active_policy_key = policy.key;
  }
  ~PolicyScope() {
    active_policy = previous_;
    active_policy_key = previous_key_;
    active_prepared_policy = previous_prepared_;
  }
  PolicyScope(const PolicyScope&) = delete;
  PolicyScope& operator=(const PolicyScope&) = delete;

 private:
  const std::vector<std::string>* previous_;
  uint64_t previous_key_;
  const PreparedPolicy* previous_prepared_;
};

// This scope is host-only and ends after launch. Capture records the selected
// kernels; graph replay needs neither TLS nor policy storage addresses.
template <typename Result, typename... Args>
auto with_policy(Result (*operation)(Args...)) {
  return [operation](Args... args,
                     std::optional<std::string> native_policy) -> Result {
    const PolicyScope scope(native_policy);
    return operation(args...);
  };
}

}  // namespace vllm::sm70
