#pragma once

#include <optional>
#include <string>
#include <torch/library.h>
#include <tuple>

#include "core/scalar_type.hpp"

#include <vector>

torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
  // Ensure tensor is on CUDA
  if (!tensor.is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA device");
  }

  // Get the raw data pointer
  void* data_ptr = tensor.data_ptr();

  // Get tensor sizes and strides
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  // Get tensor options (dtype, device)
  auto options = tensor.options();

  // Create a new tensor from the raw data pointer
  auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}

// rms_norm and fused_add_rms_norm declarations also exist in
// csrc/libtorch_stable/ops.h (torch::stable ABI for CUDA). They remain here
// because the CPU build still uses these torch::Tensor declarations.
void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

torch::Tensor fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    torch::Tensor const& q_in, torch::Tensor const& kv, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, int64_t q_head_padded, double eps,
    int64_t cache_block_size);

void silu_and_mul_per_block_quant(torch::Tensor& out,
                                  torch::Tensor const& input,
                                  torch::Tensor& scales, int64_t group_size,
                                  std::optional<torch::Tensor> scale_ub,
                                  bool is_scale_transposed);

// rotary_embedding also exist in csrc/libtorch_stable/ops.h (torch::stable
// ABI for CUDA). It remains here because the CPU build still uses these
// torch::Tensor declarations.
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void silu_and_mul_clamp(torch::Tensor& out, torch::Tensor& input, double limit);

void silu_and_mul_quant(torch::Tensor& out, torch::Tensor& input,
                        torch::Tensor& scale);

void persistent_masked_m_silu_mul_quant(
    const at::Tensor& input,   // (E, T, 2*H)
    const at::Tensor& counts,  // (E)
    at::Tensor& y_q,           // (E, T, H) [OUT]
    at::Tensor& y_s,           // (E, T, H//group_size) [OUT]
    bool use_ue8m0);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void cutlass_mla_decode(torch::Tensor const& out, torch::Tensor const& q_nope,
                        torch::Tensor const& q_pe,
                        torch::Tensor const& kv_c_and_k_pe_cache,
                        torch::Tensor const& seq_lens,
                        torch::Tensor const& page_table, double scale);

torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);

#if !defined(USE_ROCM) && defined(ENABLE_SM70_TURBOMIND)
std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor _kernel,
                                            torch::Tensor _scaling_factors,
                                            torch::Tensor _zeros,
                                            int64_t group_size,
                                            bool interleave_gated_silu);

std::vector<torch::Tensor> fp8_sm70_prepare(torch::Tensor _kernel,
                                            torch::Tensor _scaling_factors,
                                            int64_t group_size,
                                            bool interleave_gated_silu);

std::vector<torch::Tensor> sm70_f16_prepare(torch::Tensor _kernel);

torch::Tensor awq_gemm_sm70(torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors,
                            int64_t group_size,
                            int64_t k_ld,
                            int64_t q_ld);

torch::Tensor sm70_f16_gemm(torch::Tensor _in_feats, torch::Tensor _kernel);

void awq_gemm_sm70_out(torch::Tensor out,
                       torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       torch::Tensor _scaling_factors,
                       int64_t group_size,
                       int64_t k_ld,
                       int64_t q_ld,
                       bool gated_silu);

void fp8_gemm_sm70_out(torch::Tensor out,
                       torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       torch::Tensor _scaling_factors,
                       int64_t group_size,
                       int64_t k_ld,
                       int64_t q_ld,
                       bool gated_silu);

void fp8_gemm_sm70_out_auto(torch::Tensor out,
                            torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors);

void fp8_gemm_sm70_out_meta(torch::Tensor out,
                            torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors,
                            torch::Tensor _meta,
                            bool gated_silu);

void sm70_f16_gemm_out(torch::Tensor out,
                       torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       int64_t k_ld,
                       bool gated_silu);

void sm70_f16_lm_head_top1_out(torch::Tensor values_out,
                               torch::Tensor indices_out,
                               torch::Tensor _in_feats,
                               torch::Tensor _kernel,
                               int64_t k_ld,
                               int64_t vocab_start_index,
                               int64_t num_vocab_padding);

void sm70_f16_lm_head_top1_tc_out(torch::Tensor values_out,
                                  torch::Tensor indices_out,
                                  torch::Tensor _in_feats,
                                  torch::Tensor _kernel,
                                  int64_t k_ld,
                                  int64_t vocab_start_index,
                                  int64_t num_vocab_padding);

void sm70_f16_gate_mul_out(torch::Tensor out,
                           torch::Tensor _in_feats,
                           torch::Tensor _gate_weight);

int64_t sm70_gemm_import_cache(torch::Tensor device_hint,
                               const std::string& path);

int64_t sm70_gemm_export_cache(torch::Tensor device_hint,
                               const std::string& path);

std::vector<torch::Tensor> awq_moe_build_strided_ptrs(
    torch::Tensor tm_weights,
    torch::Tensor tm_scales,
    int64_t k_ld,
    int64_t q_ld,
    int64_t num_experts);

void awq_moe_gemm_sm70_out(torch::Tensor out,
                           torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s,
                           int64_t num_experts,
                           int64_t k,
                           int64_t n,
                           int64_t group_size,
                           bool gated_silu);

void awq_moe_gemm_sm70_per_expert_dispatch_out(torch::Tensor out,
                                               torch::Tensor sorted_input,
                                               torch::Tensor expert_offsets,
                                               torch::Tensor strided_ptrs_w,
                                               torch::Tensor strided_ptrs_s,
                                               int64_t num_experts,
                                               int64_t k,
                                               int64_t n,
                                               int64_t group_size,
                                               bool gated_silu);

void awq_moe_dense_stage_sm70_out(torch::Tensor out,
                                  torch::Tensor input,
                                  torch::Tensor expert_offsets,
                                  torch::Tensor dense_expert_ids,
                                  torch::Tensor ptrs_w,
                                  torch::Tensor ptrs_s,
                                  int64_t num_experts,
                                  int64_t k,
                                  int64_t n,
                                  int64_t group_size);

void awq_moe_active_dense_stage_sm70_out(
    torch::Tensor out,
    torch::Tensor input,
    torch::Tensor permuted_experts_id,
    torch::Tensor active_expert_offsets,
    torch::Tensor active_expert_ids,
    torch::Tensor ptrs_w,
    torch::Tensor ptrs_s,
    int64_t total_slots,
    int64_t k,
    int64_t n,
    int64_t group_size);

void awq_moe_single_token_dense_stage_sm70_out(
    torch::Tensor out,
    torch::Tensor input,
    torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids,
    torch::Tensor ptrs_w,
    torch::Tensor ptrs_s,
    int64_t top_k,
    int64_t k,
    int64_t n,
    int64_t group_size);

void awq_moe_single_token_indexed_dense_stage_sm70_out(
    torch::Tensor out,
    torch::Tensor input,
    torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids,
    torch::Tensor ptrs_w,
    torch::Tensor ptrs_s,
    int64_t top_k,
    int64_t k,
    int64_t n,
    int64_t group_size);

void awq_moe_single_token_dense_w13_sm70_out(
    torch::Tensor gate_up,
    torch::Tensor compact_input,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor w13_ptrs_w,
    torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets,
    torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx,
    torch::Tensor sorted_expert_ids,
    int64_t w13_k,
    int64_t w13_n,
    int64_t group_size,
    int64_t hidden_logical_size);

void awq_moe_single_token_indexed_dense_w13_sm70_out(
    torch::Tensor gate_up,
    torch::Tensor compact_input,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor w13_ptrs_w,
    torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets,
    torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx,
    torch::Tensor sorted_expert_ids,
    int64_t w13_k,
    int64_t w13_n,
    int64_t group_size,
    int64_t hidden_logical_size);

void awq_moe_single_token_compact_dense_w13_sm70_out(
    torch::Tensor gate_up,
    torch::Tensor compact_input,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor w13_ptrs_w,
    torch::Tensor w13_ptrs_s,
    torch::Tensor compact_w13_ptrs_w,
    torch::Tensor compact_w13_ptrs_s,
    torch::Tensor expert_offsets,
    torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx,
    torch::Tensor sorted_expert_ids,
    int64_t w13_k,
    int64_t w13_n,
    int64_t group_size,
    int64_t hidden_logical_size);

void awq_moe_single_token_exact_layout_prepare(torch::Tensor topk_ids,
                                               torch::Tensor x,
                                               torch::Tensor compact_input,
                                               torch::Tensor expert_offsets,
                                               torch::Tensor expert_offsets64,
                                               torch::Tensor inv_permuted_idx,
                                               int64_t num_experts);

void awq_moe_single_token_weighted_reduce_out(torch::Tensor sorted_output,
                                              torch::Tensor topk_weights,
                                              torch::Tensor inv_permuted_idx,
                                              torch::Tensor out,
                                              int64_t top_k,
                                              int64_t hidden_logical_size);

void awq_moe_single_token_sm70_out(
    torch::Tensor out,
    torch::Tensor x,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor src_w13_ptrs_w_rows,
    torch::Tensor src_w13_ptrs_s_rows,
    torch::Tensor src_w2_ptrs_w_rows,
    torch::Tensor src_w2_ptrs_s_rows,
    torch::Tensor compact_input,
    torch::Tensor intermediate,
    torch::Tensor sorted_output,
    torch::Tensor dst_w13_ptrs_w_rows,
    torch::Tensor dst_w13_ptrs_s_rows,
    torch::Tensor dst_w2_ptrs_w_rows,
    torch::Tensor dst_w2_ptrs_s_rows,
    torch::Tensor expert_offsets,
    torch::Tensor inv_permuted_idx,
    int64_t w13_k,
    int64_t w13_n,
    int64_t w2_k,
    int64_t w2_n,
    int64_t group_size,
    int64_t hidden_logical_size);

void fp8_moe_gemm_sm70_out(torch::Tensor out,
                           torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s,
                           int64_t num_experts,
                           int64_t k,
                           int64_t n,
                           int64_t group_size,
                           bool gated_silu);

void fp8_moe_gemm_sm70_per_expert_dispatch_out(torch::Tensor out,
                                               torch::Tensor sorted_input,
                                               torch::Tensor expert_offsets,
                                               torch::Tensor strided_ptrs_w,
                                               torch::Tensor strided_ptrs_s,
                                               int64_t num_experts,
                                               int64_t k,
                                               int64_t n,
                                               int64_t group_size,
                                               bool gated_silu);

void fp8_moe_dense_stage_sm70_out(torch::Tensor out,
                                  torch::Tensor input,
                                  torch::Tensor expert_offsets,
                                  torch::Tensor dense_expert_ids,
                                  torch::Tensor ptrs_w,
                                  torch::Tensor ptrs_s,
                                  int64_t num_experts,
                                  int64_t k,
                                  int64_t n,
                                  int64_t group_size);

void fp8_moe_single_token_dense_stage_sm70_out(
    torch::Tensor out,
    torch::Tensor input,
    torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids,
    torch::Tensor ptrs_w,
    torch::Tensor ptrs_s,
    int64_t top_k,
    int64_t k,
    int64_t n,
    int64_t group_size);

void fp8_moe_single_token_indexed_dense_stage_sm70_out(
    torch::Tensor out,
    torch::Tensor input,
    torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids,
    torch::Tensor ptrs_w,
    torch::Tensor ptrs_s,
    int64_t top_k,
    int64_t k,
    int64_t n,
    int64_t group_size);

void fp8_moe_single_token_dense_w13_sm70_out(
    torch::Tensor gate_up,
    torch::Tensor compact_input,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor w13_ptrs_w,
    torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets,
    torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx,
    torch::Tensor sorted_expert_ids,
    int64_t w13_k,
    int64_t w13_n,
    int64_t group_size,
    int64_t hidden_logical_size);

void fp8_moe_single_token_indexed_dense_w13_sm70_out(
    torch::Tensor gate_up,
    torch::Tensor compact_input,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor w13_ptrs_w,
    torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets,
    torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx,
    torch::Tensor sorted_expert_ids,
    int64_t w13_k,
    int64_t w13_n,
    int64_t group_size,
    int64_t hidden_logical_size);

void fp8_moe_single_token_compact_dense_w13_sm70_out(
    torch::Tensor gate_up,
    torch::Tensor compact_input,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor w13_ptrs_w,
    torch::Tensor w13_ptrs_s,
    torch::Tensor compact_w13_ptrs_w,
    torch::Tensor compact_w13_ptrs_s,
    torch::Tensor expert_offsets,
    torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx,
    torch::Tensor sorted_expert_ids,
    int64_t w13_k,
    int64_t w13_n,
    int64_t group_size,
    int64_t hidden_logical_size);

void fp8_moe_single_token_sm70_out(
    torch::Tensor out,
    torch::Tensor x,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor src_w13_ptrs_w_rows,
    torch::Tensor src_w13_ptrs_s_rows,
    torch::Tensor src_w2_ptrs_w_rows,
    torch::Tensor src_w2_ptrs_s_rows,
    torch::Tensor compact_input,
    torch::Tensor gate_up,
    torch::Tensor intermediate,
    torch::Tensor sorted_output,
    torch::Tensor sorted_weights,
    torch::Tensor dst_w13_ptrs_w_rows,
    torch::Tensor dst_w13_ptrs_s_rows,
    torch::Tensor dst_w2_ptrs_w_rows,
    torch::Tensor dst_w2_ptrs_s_rows,
    torch::Tensor expert_offsets,
    torch::Tensor inv_permuted_idx,
    torch::Tensor sorted_expert_ids,
    torch::Tensor broadcast_input_indices,
    torch::Tensor w2_raw_weight,
    torch::Tensor w2_raw_scale_inv,
    int64_t w13_k,
    int64_t w13_n,
    int64_t w2_k,
    int64_t w2_n,
    int64_t group_size,
    int64_t hidden_logical_size,
    bool fused_gated_silu,
    bool fused_weighted_reduce,
    bool broadcast_input,
    bool w2_direct_reduce,
    bool indexed_expert_ptrs,
    bool exact_per_route);
#endif

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              std::optional<torch::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               std::optional<torch::Tensor> const& azp);

torch::Tensor dynamic_4bit_int_moe_cpu(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_packed, torch::Tensor w2_packed, int64_t H, int64_t I,
    int64_t I2, int64_t group_size, bool apply_router_weight_on_input,
    int64_t activation_kind);

using fptr_t = int64_t;
fptr_t init_custom_ar(const std::vector<int64_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t reg_buffer, int64_t reg_buffer_sz_bytes);
void all_reduce_sum2(fptr_t _fa, torch::Tensor& inp_a, torch::Tensor& inp_b,
                     torch::Tensor& out);
void top1_argmax(fptr_t _fa, torch::Tensor& input_pair, torch::Tensor& output,
                 fptr_t reg_buffer, int64_t reg_buffer_sz_bytes);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, const std::vector<int64_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
std::tuple<int64_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size);
int64_t open_mem_handle(torch::Tensor& mem_handle);
void free_shared_buffer(int64_t buffer);

#ifdef USE_ROCM
fptr_t init_custom_qr(int64_t rank, int64_t world_size,
                      std::optional<int64_t> qr_max_size = std::nullopt);
void qr_destroy(fptr_t _fa);
torch::Tensor qr_get_handle(fptr_t _fa);
void qr_open_handles(fptr_t _fa, const std::vector<torch::Tensor>& handles);
void qr_all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                   int64_t quant_level, bool cast_bf2half = false);
int64_t qr_max_size();
#endif

#ifndef USE_ROCM
torch::Tensor minimax_allreduce_rms(torch::Tensor const& input,
                                    torch::Tensor const& norm_weight,
                                    torch::Tensor workspace, int64_t const rank,
                                    int64_t const nranks, double const eps);
std::tuple<torch::Tensor, torch::Tensor> minimax_allreduce_rms_qk(
    torch::Tensor qkv, torch::Tensor const& norm_weight_q,
    torch::Tensor const& norm_weight_k, torch::Tensor workspace,
    int64_t const q_size, int64_t const kv_size, int64_t const rank,
    int64_t const nranks, double const eps);
#endif
