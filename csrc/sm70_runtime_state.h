// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vllm::sm70 {
struct PreparedPolicy;

struct RuntimeResource {
  virtual ~RuntimeResource() = default;
  virtual void close() = 0;
};

// Host owners are never serialized into an AOT graph. A reloaded graph resolves
// its stable policy slot under the current engine's forward context.
struct RuntimeState {
  inline static std::atomic<uint64_t> next_id{1};
  const uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
  bool closed = false;
  std::mutex mutex;
  std::unordered_map<std::string, const PreparedPolicy*> policies;
  std::unordered_map<std::string, std::unique_ptr<RuntimeResource>> resources;
  std::unordered_map<std::string, std::atomic<unsigned>> counters;

  void close() {
    std::lock_guard<std::mutex> lock(mutex);
    if (closed) return;
    for (auto& [name, resource] : resources) resource->close();
    resources.clear();
    policies.clear();
    counters.clear();
    closed = true;
  }
};

inline thread_local std::shared_ptr<RuntimeState> active_runtime;
inline thread_local std::vector<std::shared_ptr<RuntimeState>> runtime_stack;

inline void enter_runtime(const std::shared_ptr<RuntimeState>& state) {
  if (state->closed) throw std::runtime_error("SM70 runtime has been closed");
  runtime_stack.push_back(active_runtime);
  active_runtime = state;
}

inline void exit_runtime(const std::shared_ptr<RuntimeState>& state) {
  if (active_runtime != state || runtime_stack.empty())
    throw std::runtime_error("Unbalanced SM70 runtime context");
  active_runtime = runtime_stack.back();
  runtime_stack.pop_back();
}

inline RuntimeState& current_native_runtime() {
  // Independent old no-config calls retain their process-owned compatibility
  // resources. Engine calls enter their own state before any native operation.
  static RuntimeState legacy;
  return active_runtime ? *active_runtime : legacy;
}

inline std::atomic<unsigned>& diagnostic_counter(const std::string& name) {
  auto& state = current_native_runtime();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.counters.try_emplace(name, 0u).first->second;
}
}  // namespace vllm::sm70
