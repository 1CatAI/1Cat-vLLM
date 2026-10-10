// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <torch/all.h>
#include <cuda_runtime_api.h>
#include <optional>
#include "src/turbomind/kernels/gemm/gemm.h"

// The sole runtime owns packed-weight caches, tuner decisions and scratch.
// Consumers receive views; no resource maps or environment parsing live here.
namespace vllm::awq_sm70 {
enum class TuneKeyKind : int {
  kGenericDense = 0,
  kAwqDense = 1,
  kFp8Dense = 2,
  kGenericMoe = 3,
  kAwqMoe = 4,
  kFp8Moe = 5,
  kMxfp4Dense = 6,
  kNvfp4Dense = 7,
  kMxfp4Moe = 8,
  kNvfp4Moe = 9,
  kGgufAffineU4 = 10,
  kGgufAffineU8 = 11,
  kGgufAffineU2 = 12,
  kGgufBitPlane3 = 13,
  kGgufBitPlane5 = 14,
  kGgufBitPlane6 = 15,
  kGgufLut4IQ = 16,
  kGgufLut4E2M1 = 17,
};

struct Sm70F16WeightCacheEntry {
  torch::Tensor tm_weight;
  int64_t k_ld;
};

const turbomind::gemm::Workspace& workspace_for(int device,
                                                cudaStream_t stream);
bool awq_tune_small_shapes_enabled();

bool mxfp4_moe_compact_grouped_decode_enabled();

bool mxfp4_moe_broadcast_input_decode_enabled();

bool mxfp4_moe_grouped_m8_enabled();

bool mxfp4_moe_grouped_verifier_enabled();

bool mxfp4_moe_grouped_m8_expert_rows_enabled();

bool nvfp4_moe_grouped_prefill_enabled();

bool nvfp4_moe_grouped_expert_rows_enabled();

bool fp8_moe_single_token_per_expert_dispatch_enabled();

std::optional<turbomind::gemm::DispatchPolicy>
awq_moe_dispatch_policy_override();

int sm70_f16_dense_max_m();

turbomind::gemm::DispatchPolicy select_dense_dispatch_policy_impl(
    int device, int m, int n, int k, int group_size, cudaStream_t stream,
    TuneKeyKind kind, bool tune_enabled, bool reuse_imported_cache, int max_m);

turbomind::gemm::DispatchPolicy select_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);

turbomind::gemm::DispatchPolicy select_awq_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);

turbomind::gemm::DispatchPolicy select_fp8_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);

turbomind::gemm::DispatchPolicy select_mxfp4_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);

turbomind::gemm::DispatchPolicy select_nvfp4_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);

turbomind::gemm::DispatchPolicy select_moe_dispatch_policy_impl(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream, TuneKeyKind kind, bool tune_enabled,
    int max_tune_tokens = -1);

turbomind::gemm::DispatchPolicy select_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream);

turbomind::gemm::DispatchPolicy select_fp8_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream);

turbomind::gemm::DispatchPolicy select_mxfp4_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream);

turbomind::gemm::DispatchPolicy select_nvfp4_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream);

turbomind::gemm::Gemm& get_gemm(int device);

Sm70F16WeightCacheEntry get_sm70_f16_cached_weight(torch::Tensor weight,
                                                   cudaStream_t stream);

}  // namespace vllm::awq_sm70
