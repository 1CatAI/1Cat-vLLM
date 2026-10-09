// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "sm70_policy.h"
#include <torch/custom_class.h>

namespace vllm::sm70 {
template <int Domain>
class NativeRuntime : public torch::CustomClassHolder {
 public:
  NativeRuntime() : state_(std::make_shared<RuntimeState>()) {}

  void bind(const std::string& slot, const std::string& token) {
    TORCH_CHECK(!state_->closed, "SM70 runtime has been closed");
    const auto& policy = prepared_policy(token);
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto [it, inserted] = state_->policies.emplace(slot, &policy);
    TORCH_CHECK(
        inserted || it->second->token == token,
        "SM70 native policy slot cannot change after initialization: ", slot);
  }
  void enter() { enter_runtime(state_); }
  void exit() { exit_runtime(state_); }
  void close() {
    TORCH_CHECK(active_runtime != state_,
                "Cannot close the active SM70 runtime");
    state_->close();
  }
  int64_t resource_count() const { return state_->resources.size(); }
  int64_t policy_count() const { return state_->policies.size(); }

 private:
  std::shared_ptr<RuntimeState> state_;
};

template <int Domain, typename Library>
void register_native_runtime(Library& library) {
  library.template class_<NativeRuntime<Domain>>("Sm70NativeRuntime")
      .def(torch::init<>())
      .def("bind", &NativeRuntime<Domain>::bind)
      .def("enter", &NativeRuntime<Domain>::enter)
      .def("exit", &NativeRuntime<Domain>::exit)
      .def("close", &NativeRuntime<Domain>::close)
      .def("resource_count", &NativeRuntime<Domain>::resource_count)
      .def("policy_count", &NativeRuntime<Domain>::policy_count);
  library.def("sm70_native_runtime_abi() -> int",
              []() -> int64_t { return 1; });
}
}  // namespace vllm::sm70
