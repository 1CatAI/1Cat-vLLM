// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "gemm_runtime.h"
#include "sm70_policy.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "src/turbomind/kernels/gemm/sm70_dflash_context.h"

namespace vllm::awq_sm70 {
namespace {
struct WorkspaceHolder {
  torch::Tensor barriers;
  torch::Tensor partials;
  torch::Tensor tensormaps;
  torch::Tensor flags;
  turbomind::gemm::Workspace workspace{};
};

struct GemmHolder {
  std::unique_ptr<turbomind::gemm::Gemm> gemm;
};

struct DenseTuneKey {
  TuneKeyKind kind;
  int device;
  int m;
  int n;
  int k;
  int group_size;

  uint64_t policy = vllm::sm70::active_policy_key;

  bool operator==(const DenseTuneKey& other) const {
    return policy == other.policy && kind == other.kind &&
           device == other.device && m == other.m && n == other.n &&
           k == other.k && group_size == other.group_size;
  }
};

struct DenseTuneKeyHash {
  std::size_t operator()(const DenseTuneKey& key) const {
    std::size_t h = std::hash<int>()(static_cast<int>(key.kind));
    h ^= std::hash<int>()(key.device) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.m) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.group_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<uint64_t>()(key.policy) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct MoeTuneKey {
  TuneKeyKind kind;
  int device;
  int total_tokens;
  int n;
  int k;
  int num_experts;
  int group_size;

  uint64_t policy = vllm::sm70::active_policy_key;

  bool operator==(const MoeTuneKey& other) const {
    return policy == other.policy && kind == other.kind &&
           device == other.device && total_tokens == other.total_tokens &&
           n == other.n && k == other.k && num_experts == other.num_experts &&
           group_size == other.group_size;
  }
};

struct MoeTuneKeyHash {
  std::size_t operator()(const MoeTuneKey& key) const {
    std::size_t h = std::hash<int>()(static_cast<int>(key.kind));
    h ^= std::hash<int>()(key.device) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.total_tokens) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.num_experts) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.group_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<uint64_t>()(key.policy) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct Sm70F16WeightCacheKey {
  int device;
  const void* tensor_impl;
  int64_t rows;
  int64_t cols;

  bool operator==(const Sm70F16WeightCacheKey& other) const {
    return device == other.device && tensor_impl == other.tensor_impl &&
           rows == other.rows && cols == other.cols;
  }
};

struct Sm70F16WeightCacheKeyHash {
  std::size_t operator()(const Sm70F16WeightCacheKey& key) const {
    std::size_t h = std::hash<int>()(key.device);
    h ^= std::hash<const void*>()(key.tensor_impl) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<int64_t>()(key.rows) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>()(key.cols) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

// Per-stream workspace management to eliminate mutex contention
struct StreamWorkspaceKey {
  int device;
  cudaStream_t stream;

  bool operator==(const StreamWorkspaceKey& other) const {
    return device == other.device && stream == other.stream;
  }
};

struct StreamWorkspaceKeyHash {
  std::size_t operator()(const StreamWorkspaceKey& k) const {
    return std::hash<int>()(k.device) ^
           (std::hash<cudaStream_t>()(k.stream) << 1);
  }
};

struct TurboMindRuntime : vllm::sm70::RuntimeResource {
  size_t partials_size = turbomind::gemm::Gemm::kPartialsSize;
  std::mutex workspace_mutex;
  std::mutex gemm_mutex;
  std::mutex tune_mutex;
  std::mutex sm70_f16_weight_cache_mutex;
  std::unordered_map<StreamWorkspaceKey, WorkspaceHolder,
                     StreamWorkspaceKeyHash>
      workspace_cache;
  std::map<std::pair<int, uint64_t>, GemmHolder> gemm_cache;
  std::unordered_set<DenseTuneKey, DenseTuneKeyHash> dense_tuned_shapes;
  std::unordered_set<MoeTuneKey, MoeTuneKeyHash> moe_tuned_shapes;
  std::set<std::pair<int, uint64_t>> imported_cache_devices;
  std::unordered_map<Sm70F16WeightCacheKey, Sm70F16WeightCacheEntry,
                     Sm70F16WeightCacheKeyHash>
      sm70_f16_weight_cache;

  void close() override {
    // Engine shutdown drains graphs before releasing owners. Also wait for
    // outstanding eager work before freeing native scratch or packed weights.
    std::set<int> devices;
    for (const auto& [key, holder] : workspace_cache)
      devices.insert(key.device);
    for (const auto& [key, holder] : sm70_f16_weight_cache)
      devices.insert(key.device);
    for (int device : devices) {
      c10::cuda::CUDAGuard guard(device);
      AT_CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
};

TurboMindRuntime& runtime() {
  auto& state = vllm::sm70::current_native_runtime();
  thread_local uint64_t cached_id = 0;
  thread_local TurboMindRuntime* cached = nullptr;
  if (cached_id == state.id) return *cached;
  std::lock_guard<std::mutex> lock(state.mutex);
  auto& resource = state.resources["turbomind"];
  if (!resource) resource = std::make_unique<TurboMindRuntime>();
  cached = static_cast<TurboMindRuntime*>(resource.get());
  cached_id = state.id;
  return *cached;
}

}  // namespace

bool tune_small_shapes_enabled() {
  return vllm::sm70::policy_atoi(vllm::sm70::PolicyField::awq_tune_small_shapes,
                                 1) != 0;
}

bool awq_tune_small_shapes_enabled() {
  return vllm::sm70::policy_atoi(vllm::sm70::PolicyField::awq_tune_small_shapes,
                                 0) != 0;
}

bool fp8_tune_small_shapes_enabled() {
  return vllm::sm70::policy_atoi(vllm::sm70::PolicyField::fp8_tune_small_shapes,
                                 1) != 0;
}

bool mxfp4_tune_small_shapes_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::mxfp4_tune_small_shapes, 1) != 0;
}

bool mxfp4_moe_compact_grouped_decode_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::mxfp4_moe_compact_grouped_decode, 1) != 0;
}

bool mxfp4_moe_broadcast_input_decode_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::mxfp4_moe_broadcast_input_decode, 1) != 0;
}

bool mxfp4_moe_grouped_m8_enabled() {
  return vllm::sm70::policy_atoi(vllm::sm70::PolicyField::mxfp4_moe_grouped_m8,
                                 0) != 0;
}

bool mxfp4_moe_grouped_verifier_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::mxfp4_moe_grouped_verifier, 0) != 0;
}

bool mxfp4_moe_grouped_m8_expert_rows_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::mxfp4_moe_grouped_m8_expert_rows, 0) != 0;
}

bool mxfp4_moe_grouped_m8_fast_selector_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::mxfp4_moe_grouped_m8_fast_selector, 1) !=
         0;
}

bool nvfp4_tune_small_shapes_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::nvfp4_tune_small_shapes, 1) != 0;
}

bool nvfp4_moe_grouped_prefill_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::nvfp4_moe_grouped_prefill, 1) != 0;
}

bool nvfp4_moe_grouped_expert_rows_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::nvfp4_moe_grouped_expert_rows, 0) != 0;
}

bool nvfp4_qwen38_tp4_m1_fast_selector_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::nvfp4_qwen38_tp4_m1_fast_selector, 1) !=
         0;
}

bool fp8_moe_single_token_per_expert_dispatch_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::fp8_moe_single_token_per_expert_dispatch,
             0) != 0;
}

bool fp8_0dot3_dense_selector_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::fp8_0dot3_dense_selector, 0) != 0;
}

bool fp8_safe_fast_selector_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::fp8_safe_fast_selector, 0) != 0;
}

bool fp8_grouped_bmm_decode_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::fp8_grouped_bmm_decode, 1) != 0;
}

bool awq_reuse_imported_cache_enabled() {
  // An imported plan is produced by the coordinated warmup on rank 0.  Keep
  // it as the default source of truth on the other TP ranks; an explicit 0
  // remains available for cache-debugging and old standalone warmups.
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::awq_reuse_imported_cache, 1) != 0;
}

bool fp8_reuse_imported_cache_enabled() {
  // See awq_reuse_imported_cache_enabled().  The Python warmup only marks an
  // imported cache after rank 0 has finished measuring the shape, so this
  // does not change the no-cache path.
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::fp8_reuse_imported_cache, 1) != 0;
}

bool awq_preserve_default_splits_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::awq_preserve_default_splits, 1) != 0;
}

bool awq_preserve_default_splits_only_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::awq_preserve_default_splits_only, 0) != 0;
}

bool fp8_preserve_default_splits_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::fp8_preserve_default_splits, 1) != 0;
}

bool fp8_preserve_default_splits_only_enabled() {
  return vllm::sm70::policy_atoi(
             vllm::sm70::PolicyField::fp8_preserve_default_splits_only, 0) != 0;
}

inline turbomind::gemm::DispatchPolicy maybe_preserve_default_splits(
    turbomind::gemm::DispatchPolicy policy) {
  if (policy == turbomind::gemm::DispatchPolicy::kMeasure ||
      policy == turbomind::gemm::DispatchPolicy::kReuse) {
    if (awq_preserve_default_splits_only_enabled()) {
      return policy |
             turbomind::gemm::DispatchPolicy::kPreserveDefaultSplitCount;
    }
    if (!awq_preserve_default_splits_enabled()) {
      return policy;
    }
    return policy | turbomind::gemm::DispatchPolicy::kPreserveDefaultSplits;
  }
  return policy;
}

inline turbomind::gemm::DispatchPolicy maybe_preserve_fp8_default_splits(
    turbomind::gemm::DispatchPolicy policy) {
  if (policy == turbomind::gemm::DispatchPolicy::kMeasure ||
      policy == turbomind::gemm::DispatchPolicy::kReuse) {
    if (fp8_preserve_default_splits_only_enabled()) {
      return policy |
             turbomind::gemm::DispatchPolicy::kPreserveDefaultSplitCount;
    }
    if (!fp8_preserve_default_splits_enabled()) {
      return policy;
    }
    return policy | turbomind::gemm::DispatchPolicy::kPreserveDefaultSplits;
  }
  return policy;
}

std::optional<turbomind::gemm::DispatchPolicy> dispatch_policy_override(
    vllm::sm70::PolicyField field) {
  switch (vllm::sm70::policy_dispatch_override(field)) {
    case vllm::sm70::DispatchOverride::Unset:
      return std::nullopt;
    case vllm::sm70::DispatchOverride::Default:
      return turbomind::gemm::DispatchPolicy::kDefault;
    case vllm::sm70::DispatchOverride::Reuse:
      return turbomind::gemm::DispatchPolicy::kReuse;
    case vllm::sm70::DispatchOverride::Measure:
      return turbomind::gemm::DispatchPolicy::kMeasure;
    case vllm::sm70::DispatchOverride::Invalid:
      break;
  }
  TORCH_CHECK(false, vllm::sm70::policy_name(field),
              " must be one of: default, reuse, measure.");
  return std::nullopt;
}

std::optional<turbomind::gemm::DispatchPolicy>
awq_moe_dispatch_policy_override() {
  return dispatch_policy_override(
      vllm::sm70::PolicyField::awq_moe_dispatch_policy);
}

int awq_dense_tune_max_m() {
  return std::max(vllm::sm70::policy_atoi(
                      vllm::sm70::PolicyField::awq_dense_tune_max_m, 16),
                  0);
}

int generic_dense_tune_max_m() {
  return std::max(vllm::sm70::policy_atoi(
                      vllm::sm70::PolicyField::f16_dense_tune_max_m, 16),
                  0);
}

int fp8_dense_tune_max_m() {
  return std::max(vllm::sm70::policy_atoi(
                      vllm::sm70::PolicyField::fp8_dense_tune_max_m, 16),
                  0);
}

int mxfp4_dense_tune_max_m() {
  return std::max(vllm::sm70::policy_atoi(
                      vllm::sm70::PolicyField::mxfp4_dense_tune_max_m, 16),
                  0);
}

int nvfp4_dense_tune_max_m() {
  return std::max(vllm::sm70::policy_atoi(
                      vllm::sm70::PolicyField::nvfp4_dense_tune_max_m, 16),
                  0);
}

int moe_tune_max_tokens() {
  return std::max(vllm::sm70::policy_atoi(
                      vllm::sm70::PolicyField::awq_moe_tune_max_tokens, 128),
                  0);
}

int nvfp4_moe_tune_max_tokens() {
  return std::max(vllm::sm70::policy_atoi(
                      vllm::sm70::PolicyField::nvfp4_moe_tune_max_tokens, 128),
                  0);
}

int sm70_f16_dense_max_m() {
  return std::max(
      vllm::sm70::policy_atoi(vllm::sm70::PolicyField::f16_dense_max_m, 64), 0);
}

bool is_stream_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const auto ec = cudaStreamIsCapturing(stream, &status);
  if (ec != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return status != cudaStreamCaptureStatusNone;
}

bool has_imported_cache(int device) {
  std::lock_guard<std::mutex> lock(runtime().tune_mutex);
  return runtime().imported_cache_devices.find(
             {device, vllm::sm70::active_policy_key}) !=
         runtime().imported_cache_devices.end();
}

turbomind::gemm::DispatchPolicy select_dense_dispatch_policy_impl(
    int device, int m, int n, int k, int group_size, cudaStream_t stream,
    TuneKeyKind kind, bool tune_enabled, bool reuse_imported_cache, int max_m) {
  if (m > max_m) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  if (reuse_imported_cache && has_imported_cache(device)) {
    return turbomind::gemm::DispatchPolicy::kReuse;
  }
  if (!tune_enabled) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }

  DenseTuneKey key{kind, device, m, n, k, group_size};
  std::lock_guard<std::mutex> lock(runtime().tune_mutex);
  if (runtime().dense_tuned_shapes.find(key) !=
      runtime().dense_tuned_shapes.end()) {
    return turbomind::gemm::DispatchPolicy::kReuse;
  }
  if (is_stream_capturing(stream)) {
    // runtime().tune_mutex is already held here. Calling has_imported_cache()
    // would acquire it recursively and deadlock on the first uncached graph
    // shape.
    if (runtime().imported_cache_devices.find(
            {device, vllm::sm70::active_policy_key}) !=
        runtime().imported_cache_devices.end()) {
      return turbomind::gemm::DispatchPolicy::kReuse;
    }
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  runtime().dense_tuned_shapes.insert(key);
  return turbomind::gemm::DispatchPolicy::kMeasure;
}

turbomind::gemm::DispatchPolicy select_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  if (group_size == 0 &&
      turbomind::gemm::UseSm70DflashContextFcStableReduction(m, n, k)) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }

  const bool exact_dflash2_rerank =
      (vllm::sm70::policy_atoi(vllm::sm70::PolicyField::dflash2_qpn8_rerank,
                               0) != 0) ||
      (vllm::sm70::policy_atoi(
           vllm::sm70::PolicyField::dflash2_qpn8_rerank_shadow, 0) != 0);
  if (exact_dflash2_rerank && m >= 1 && m <= 8 && n == 62080 && k == 5120 &&
      group_size == 0) {
    // The sparse reranker reproduces this exact split-K contract. Do not let
    // concurrent startup noise choose a numerically different LM-head spec on
    // one TP rank.
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  return select_dense_dispatch_policy_impl(
      device, m, n, k, group_size, stream, TuneKeyKind::kGenericDense,
      tune_small_shapes_enabled(), false, generic_dense_tune_max_m());
}

turbomind::gemm::DispatchPolicy select_awq_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  return maybe_preserve_default_splits(select_dense_dispatch_policy_impl(
      device, m, n, k, group_size, stream, TuneKeyKind::kAwqDense,
      awq_tune_small_shapes_enabled(), awq_reuse_imported_cache_enabled(),
      awq_dense_tune_max_m()));
}

turbomind::gemm::DispatchPolicy select_fp8_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  if (fp8_0dot3_dense_selector_enabled()) {
    return select_dense_dispatch_policy_impl(
        device, m, n, k, group_size, stream, TuneKeyKind::kGenericDense,
        tune_small_shapes_enabled(), fp8_reuse_imported_cache_enabled(),
        generic_dense_tune_max_m());
  }
  auto policy = select_dense_dispatch_policy_impl(
      device, m, n, k, group_size, stream, TuneKeyKind::kFp8Dense,
      fp8_tune_small_shapes_enabled(), fp8_reuse_imported_cache_enabled(),
      fp8_dense_tune_max_m());
  if (!fp8_safe_fast_selector_enabled()) {
    return policy;
  }
  return maybe_preserve_fp8_default_splits(policy);
}

turbomind::gemm::DispatchPolicy select_mxfp4_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  return select_dense_dispatch_policy_impl(
      device, m, n, k, group_size, stream, TuneKeyKind::kMxfp4Dense,
      mxfp4_tune_small_shapes_enabled(), true, mxfp4_dense_tune_max_m());
}

turbomind::gemm::DispatchPolicy select_nvfp4_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  if (nvfp4_qwen38_tp4_m1_fast_selector_enabled() && m == 1 &&
      group_size == 16 &&
      ((n == 8704 && k == 5120) || (n == 5120 && k == 4352))) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  return select_dense_dispatch_policy_impl(
      device, m, n, k, group_size, stream, TuneKeyKind::kNvfp4Dense,
      nvfp4_tune_small_shapes_enabled(), true, nvfp4_dense_tune_max_m());
}

turbomind::gemm::DispatchPolicy select_moe_dispatch_policy_impl(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream, TuneKeyKind kind, bool tune_enabled,
    int max_tune_tokens) {
  const int tune_limit =
      max_tune_tokens >= 0 ? max_tune_tokens : moe_tune_max_tokens();
  if (!tune_enabled || total_tokens > tune_limit) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }

  MoeTuneKey key{kind, device, total_tokens, n, k, num_experts, group_size};
  std::lock_guard<std::mutex> lock(runtime().tune_mutex);
  if (runtime().moe_tuned_shapes.find(key) !=
      runtime().moe_tuned_shapes.end()) {
    return turbomind::gemm::DispatchPolicy::kReuse;
  }
  if (is_stream_capturing(stream)) {
    if (runtime().imported_cache_devices.find(
            {device, vllm::sm70::active_policy_key}) !=
        runtime().imported_cache_devices.end()) {
      return turbomind::gemm::DispatchPolicy::kReuse;
    }
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  runtime().moe_tuned_shapes.insert(key);
  return turbomind::gemm::DispatchPolicy::kMeasure;
}

turbomind::gemm::DispatchPolicy select_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream) {
  return select_moe_dispatch_policy_impl(
      device, total_tokens, n, k, num_experts, group_size, stream,
      TuneKeyKind::kGenericMoe, tune_small_shapes_enabled());
}

turbomind::gemm::DispatchPolicy select_fp8_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream) {
  if (fp8_grouped_bmm_decode_enabled() && total_tokens == 2 && n == 1024 &&
      k == 4096 && num_experts == 2 && group_size == 128) {
    // The matching dense WO-A projection uses the fixed launch spec selected
    // in gemm.cu. Do not let measurement replace its accumulation tree.
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  return select_moe_dispatch_policy_impl(
      device, total_tokens, n, k, num_experts, group_size, stream,
      TuneKeyKind::kFp8Moe, fp8_tune_small_shapes_enabled());
}

turbomind::gemm::DispatchPolicy select_mxfp4_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream) {
  const bool exact_grouped_m8 =
      total_tokens == 48 && num_experts == 48 && group_size == 32 &&
      ((n == 512 && k == 4096) || (n == 4096 && k == 256));
  if (mxfp4_moe_grouped_m8_enabled() &&
      !mxfp4_moe_grouped_m8_expert_rows_enabled() &&
      mxfp4_moe_grouped_m8_fast_selector_enabled() && exact_grouped_m8) {
    return turbomind::gemm::DispatchPolicy::kMxfp4MoeGroupedM8Fast;
  }
  return select_moe_dispatch_policy_impl(
      device, total_tokens, n, k, num_experts, group_size, stream,
      TuneKeyKind::kMxfp4Moe, mxfp4_tune_small_shapes_enabled());
}

turbomind::gemm::DispatchPolicy select_nvfp4_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream) {
  return select_moe_dispatch_policy_impl(
      device, total_tokens, n, k, num_experts, group_size, stream,
      TuneKeyKind::kNvfp4Moe, nvfp4_tune_small_shapes_enabled(),
      nvfp4_moe_tune_max_tokens());
}

static WorkspaceHolder& get_workspace(int device, cudaStream_t stream) {
  thread_local int cached_device = -1;
  thread_local uint64_t cached_owner = 0;
  const auto owner = vllm::sm70::current_native_runtime().id;
  thread_local cudaStream_t cached_stream = nullptr;
  thread_local WorkspaceHolder* cached_holder = nullptr;
  if (cached_holder != nullptr && cached_owner == owner &&
      cached_device == device && cached_stream == stream) {
    return *cached_holder;
  }

  StreamWorkspaceKey key{device, stream};

  // Fast path: check if workspace exists without lock
  {
    std::lock_guard<std::mutex> lock(runtime().workspace_mutex);
    auto it = runtime().workspace_cache.find(key);
    if (it != runtime().workspace_cache.end()) {
      cached_owner = owner;
      cached_device = device;
      cached_stream = stream;
      cached_holder = &it->second;
      return it->second;
    }
  }

  // Slow path: create new workspace
  WorkspaceHolder holder;
  auto byte_opts = torch::TensorOptions()
                       .device(torch::Device(torch::kCUDA, device))
                       .dtype(torch::kUInt8);
  auto int_opts = torch::TensorOptions()
                      .device(torch::Device(torch::kCUDA, device))
                      .dtype(torch::kInt32);

  holder.barriers = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kBarriersSize}, byte_opts);
  holder.partials =
      torch::zeros({(long long)runtime().partials_size}, byte_opts);
  // Keep same tensormap size as TurboMind LlamaLinear.
  holder.tensormaps = torch::empty({(long long)(8192 * 128)}, byte_opts);
  holder.flags = torch::zeros({1}, int_opts);

  holder.workspace.barriers = holder.barriers.data_ptr();
  holder.workspace.barriers_size = holder.barriers.numel();
  holder.workspace.partials = holder.partials.data_ptr();
  holder.workspace.partials_size = holder.partials.numel();
  holder.workspace.tensormaps = holder.tensormaps.data_ptr();
  holder.workspace.tensormaps_size = holder.tensormaps.numel();
  holder.workspace.flags = holder.flags.data_ptr<int>();

  std::lock_guard<std::mutex> lock(runtime().workspace_mutex);
  auto [insert_it, _] =
      runtime().workspace_cache.emplace(key, std::move(holder));
  cached_owner = owner;
  cached_device = device;
  cached_stream = stream;
  cached_holder = &insert_it->second;
  return insert_it->second;
}

turbomind::gemm::Gemm& get_gemm(int device) {
  thread_local int cached_device = -1;
  thread_local uint64_t cached_owner = 0;
  const auto owner = vllm::sm70::current_native_runtime().id;
  thread_local uint64_t cached_policy = 0;
  const auto policy = vllm::sm70::active_policy_key;
  thread_local turbomind::gemm::Gemm* cached_gemm = nullptr;
  if (cached_gemm != nullptr && cached_owner == owner &&
      cached_device == device && cached_policy == policy) {
    return *cached_gemm;
  }

  std::lock_guard<std::mutex> lock(runtime().gemm_mutex);
  auto it = runtime().gemm_cache.find({device, policy});
  if (it != runtime().gemm_cache.end()) {
    cached_owner = owner;
    cached_device = device;
    cached_policy = policy;
    cached_gemm = it->second.gemm.get();
    return *it->second.gemm;
  }
  GemmHolder holder;
  holder.gemm = std::make_unique<turbomind::gemm::Gemm>();
  auto [insert_it, _] = runtime().gemm_cache.emplace(
      std::make_pair(device, policy), std::move(holder));
  cached_owner = owner;
  cached_device = device;
  cached_policy = policy;
  cached_gemm = insert_it->second.gemm.get();
  return *insert_it->second.gemm;
}

const turbomind::gemm::Workspace& workspace_for(int device,
                                                cudaStream_t stream) {
  return get_workspace(device, stream).workspace;
}

Sm70F16WeightCacheKey make_sm70_f16_weight_cache_key(
    const torch::Tensor& weight) {
  return Sm70F16WeightCacheKey{
      weight.get_device(),
      static_cast<const void*>(weight.unsafeGetTensorImpl()),
      weight.size(0),
      weight.size(1),
  };
}

static Sm70F16WeightCacheEntry prepare_sm70_f16_weight(torch::Tensor weight,
                                                       cudaStream_t stream) {
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  TORCH_CHECK(conv_w, "sm70_f16_prepare: no compatible TurboMind converter.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::kHalf;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty_like(weight);
  TORCH_CHECK(conv_w->Convert(weight.data_ptr(), w_desc, tm_weight.data_ptr(),
                              k_desc, stream) == 0,
              "sm70_f16_prepare: weight conversion failed.");

  return {std::move(tm_weight), static_cast<int64_t>(k_desc.ld)};
}

void configure_gemm_workspace(int64_t bytes) {
  TORCH_CHECK(vllm::sm70::active_runtime,
              "TurboMind scratch must be configured under an engine owner");
  TORCH_CHECK(bytes >= (1 << 20) &&
                  bytes <= turbomind::gemm::Gemm::kPartialsSize &&
                  bytes % 256 == 0,
              "TurboMind scratch must be an aligned 1..32 MiB budget");
  auto& state = runtime();
  std::lock_guard<std::mutex> lock(state.workspace_mutex);
  TORCH_CHECK(
      state.workspace_cache.empty() || state.partials_size == bytes,
      "TurboMind scratch cannot change after a stream workspace exists");
  state.partials_size = bytes;
}

std::vector<int64_t> gemm_workspace_storage() {
  auto& state = runtime();
  std::lock_guard<std::mutex> lock(state.workspace_mutex);
  int64_t total = 0;
  for (const auto& [key, holder] : state.workspace_cache) {
    total += holder.barriers.nbytes() + holder.partials.nbytes() +
             holder.tensormaps.nbytes() + holder.flags.nbytes();
  }
  return {static_cast<int64_t>(state.workspace_cache.size()),
          static_cast<int64_t>(state.partials_size), total};
}

Sm70F16WeightCacheEntry get_sm70_f16_cached_weight(torch::Tensor weight,
                                                   cudaStream_t stream) {
  weight = weight.contiguous();
  const auto key = make_sm70_f16_weight_cache_key(weight);

  {
    std::lock_guard<std::mutex> lock(runtime().sm70_f16_weight_cache_mutex);
    auto it = runtime().sm70_f16_weight_cache.find(key);
    if (it != runtime().sm70_f16_weight_cache.end()) {
      return it->second;
    }
  }

  TORCH_CHECK(!is_stream_capturing(stream),
              "sm70_f16_prepare: cache miss during CUDA graph capture.");

  auto entry = prepare_sm70_f16_weight(weight, stream);

  std::lock_guard<std::mutex> lock(runtime().sm70_f16_weight_cache_mutex);
  auto [it, _] = runtime().sm70_f16_weight_cache.emplace(key, entry);
  return it->second;
}

}  // namespace vllm::awq_sm70

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def("sm70_gemm_configure_workspace(int bytes) -> ()",
        &vllm::awq_sm70::configure_gemm_workspace);
  m.def("sm70_gemm_workspace_storage() -> int[]",
        &vllm::awq_sm70::gemm_workspace_storage);
}

int64_t sm70_gemm_import_cache(torch::Tensor device_hint,
                               const std::string& path) {
  TORCH_CHECK(device_hint.is_cuda(),
              "sm70_gemm_import_cache: device_hint must be CUDA.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(device_hint));
  const int device = device_hint.get_device();

  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.good()) {
    return 0;
  }

  auto& gemm = vllm::awq_sm70::get_gemm(device);
  const int64_t imported = gemm.Import(ifs);
  if (imported > 0) {
    std::lock_guard<std::mutex> lock(vllm::awq_sm70::runtime().tune_mutex);
    vllm::awq_sm70::runtime().imported_cache_devices.insert(
        {device, vllm::sm70::active_policy_key});
  }
  return imported;
}

int64_t sm70_gemm_export_cache(torch::Tensor device_hint,
                               const std::string& path) {
  TORCH_CHECK(device_hint.is_cuda(),
              "sm70_gemm_export_cache: device_hint must be CUDA.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(device_hint));
  const int device = device_hint.get_device();

  try {
    const std::filesystem::path fs_path(path);
    if (fs_path.has_parent_path()) {
      std::filesystem::create_directories(fs_path.parent_path());
    }
  } catch (const std::exception& e) {
    TORCH_CHECK(false,
                "sm70_gemm_export_cache: failed to create parent "
                "directory for ",
                path, " (", e.what(), ").");
  }

  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  TORCH_CHECK(ofs.good(), "sm70_gemm_export_cache: failed to open ", path,
              " for writing.");

  auto& gemm = vllm::awq_sm70::get_gemm(device);
  return gemm.Export(ofs);
}
