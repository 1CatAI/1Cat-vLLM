// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/custom_class.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include "runtime_owner.h"
#include <ATen/cuda/Exceptions.h>
#include <cstdlib>
#include <map>
#include "shared_workspace.h"

namespace onecat_sm70_prefill {
ExecutionGate::ExecutionGate(int device) : device(device) {
  const c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming));
}
ExecutionGate::~ExecutionGate() {
  // This physical-device gate can outlive Torch's CUDA shutdown. Destruction
  // is best effort; explicit engine close has already drained its work.
  int previous = -1;
  if (cudaGetDevice(&previous) != cudaSuccess) return;
  if (cudaSetDevice(device) != cudaSuccess) return;
  if (completion != nullptr) cudaEventDestroy(completion);
  cudaSetDevice(previous);
}
std::shared_ptr<ExecutionGate> execution_gate(int device) {
  static std::mutex mutex;
  static std::map<int, std::shared_ptr<ExecutionGate>> gates;
  std::lock_guard<std::mutex> lock(mutex);
  auto& gate = gates[device];
  if (!gate) gate = std::make_shared<ExecutionGate>(device);
  return gate;
}
ScoreWorkspace::ScoreWorkspace(const at::Tensor& query, int64_t block_n)
    : block_n(block_n),
      scores(at::empty({block_n * 8192 * 6}, query.options())),
      gate(execution_gate(query.get_device())),
      mutex(gate->mutex),
      completion(gate->completion),
      completion_recorded(gate->completion_recorded) {}
ScoreWorkspace::~ScoreWorkspace() = default;
std::shared_ptr<ScoreWorkspace> get_score_workspace(const at::Tensor& query,
                                                    int64_t block_n) {
  const int64_t configured = policy_value(PolicyField::ScoreBlockTokens);
  if (configured != 0) block_n = configured;
  TORCH_CHECK(block_n >= 8192 && block_n <= 16 * 8192 && block_n % 8192 == 0,
              "VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS must be a "
              "multiple of 8192 between 8192 and 131072");
  if (auto* owner = current_owner()) {
    return owner->workspace<ScoreWorkspace>(
        CacheSlot::Scores, query.get_device(),
        [&] { return std::make_shared<ScoreWorkspace>(query, block_n); });
  }
  static std::mutex mutex;
  static std::map<std::pair<int, int64_t>, std::shared_ptr<ScoreWorkspace>>
      cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto& entry = cache[{query.get_device(), block_n}];
  if (!entry) entry = std::make_shared<ScoreWorkspace>(query, block_n);
  return entry;
}
}  // namespace onecat_sm70_prefill

extern "C" int64_t onecat_sm70_q8000_accumulation_bits();
extern "C" int64_t onecat_sm70_q8192_accumulation_bits();

namespace flash {
at::Tensor sm70_d256_gqa_architecture_fwd(const at::Tensor& q,
                                          const at::Tensor& k,
                                          const at::Tensor& v, at::Tensor& out,
                                          double softmax_scale, bool causal);
}
namespace onecat_79t_q8192 {
at::Tensor sm70_d256_gqa_architecture_q8192_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, double softmax_scale, bool causal);
}

namespace onecat_sm70_prefill {
namespace {
thread_local PrefillOwner* active_owner = nullptr;
struct OwnerScope {
  PrefillOwner* previous = active_owner;
  explicit OwnerScope(PrefillOwner* owner) { active_owner = owner; }
  ~OwnerScope() { active_owner = previous; }
};
}  // namespace
PrefillOwner* current_owner() { return active_owner; }
int64_t policy_value(PolicyField field) {
  if (active_owner) return active_owner->value(field);
  // Independent legacy exports retain their original parsing dialects.
  static constexpr const char* names[] = {
      "PREFIX_QK_CUBLAS_ALGO_RUNTIME",
      "PREFIX_QK_CUBLAS_ALGO_RUNTIME",
      "VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS",
      "PREFIX_TORCH_SERIAL_TAIL",
      "PREFIX_TORCH_EXACT_TAIL",
      "PREFIX_TORCH_DUMP_TAIL",
      "PREFIX_TORCH_DIRECT_TAIL"};
  const char* raw = std::getenv(names[static_cast<size_t>(field)]);
  if (field == PolicyField::QkAlgorithm) return raw ? std::atoi(raw) : 0;
  if (field == PolicyField::SerialTail)
    return !raw || std::strcmp(raw, "0") != 0;
  if (field != PolicyField::ScoreBlockTokens) return raw != nullptr;
  if (!raw) return 0;
  char* end = nullptr;
  const int64_t value = std::strtol(raw, &end, 10);
  return end != raw && *end == '\0' && value >= 8192 && value <= 16 * 8192 &&
                 value % 8192 == 0
             ? value
             : -1;
}
PrefillOwner::PrefillOwner(std::vector<int64_t> values)
    : values_(std::move(values)) {
  TORCH_CHECK(values_.size() == static_cast<size_t>(PolicyField::Count),
              "SM70 prefill policy ABI 1 requires 7 fields");
  for (auto field : {PolicyField::QkOverride, PolicyField::SerialTail,
                     PolicyField::ExactTail, PolicyField::DumpTail,
                     PolicyField::DirectTail}) {
    TORCH_CHECK(value(field) == 0 || value(field) == 1,
                "SM70 prefill policy boolean fields must be 0 or 1");
  }
}
std::vector<int64_t> PrefillOwner::observations() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return {observations_[0], observations_[1]};
}
at::Tensor PrefillOwner::run(bool aligned, at::Tensor q, at::Tensor k,
                             at::Tensor v, at::Tensor out, double scale,
                             bool causal) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  TORCH_CHECK(!closed_, "SM70 prefill runtime has been closed");
  OwnerScope scope(this);
  auto result =
      aligned
          ? onecat_79t_q8192::sm70_d256_gqa_architecture_q8192_fwd(
                q, k, v, out, scale, causal)
          : flash::sm70_d256_gqa_architecture_fwd(q, k, v, out, scale, causal);
  ++observations_[aligned ? 1 : 0];
  return result;
}
at::Tensor PrefillOwner::q8000(at::Tensor q, at::Tensor k, at::Tensor v,
                               at::Tensor out, double scale, bool causal) {
  return run(false, q, k, v, out, scale, causal);
}
at::Tensor PrefillOwner::q8192(at::Tensor q, at::Tensor k, at::Tensor v,
                               at::Tensor out, double scale, bool causal) {
  return run(true, q, k, v, out, scale, causal);
}
void PrefillOwner::close() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (closed_) return;
  for (auto& device : caches_) {
    const c10::cuda::CUDAGuard guard(device.first);
    auto gate = execution_gate(device.first);
    std::lock_guard<std::mutex> execution_lock(gate->mutex);
    // Shutdown only, after graphs are destroyed. This also drains work from a
    // failed dispatch that did not reach the final completion-event record.
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    device.second.clear();
  }
  caches_.clear();
  closed_ = true;
}
PrefillOwner::~PrefillOwner() {
  try {
    close();
  } catch (...) { /* CUDA may already be torn down at exit. */
  }
}
}  // namespace onecat_sm70_prefill

TORCH_LIBRARY_FRAGMENT(_vllm_fa2_C, ops) {
  ops.def("sm70_prefill_policy_abi() -> int", []() -> int64_t { return 1; });
  ops.class_<onecat_sm70_prefill::PrefillOwner>("Sm70PrefillRuntime")
      .def(torch::init<std::vector<int64_t>>())
      .def("q8000", &onecat_sm70_prefill::PrefillOwner::q8000)
      .def("q8192", &onecat_sm70_prefill::PrefillOwner::q8192)
      .def("values", &onecat_sm70_prefill::PrefillOwner::values)
      .def("observations", &onecat_sm70_prefill::PrefillOwner::observations)
      .def("close", &onecat_sm70_prefill::PrefillOwner::close);
  ops.def("sm70_d256_gqa_accumulation_bits() -> int", []() -> int64_t {
    return onecat_sm70_q8000_accumulation_bits() == 32 &&
                   onecat_sm70_q8192_accumulation_bits() == 32
               ? 32
               : 16;
  });
  ops.def(
      "sm70_d256_gqa_architecture_q8192_fwd(Tensor q, Tensor k, Tensor v, "
      "Tensor(a!) out, float softmax_scale, bool causal) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(_vllm_fa2_C, CUDA, ops) {
  ops.impl("sm70_d256_gqa_architecture_q8192_fwd",
           &onecat_79t_q8192::sm70_d256_gqa_architecture_q8192_fwd);
}
