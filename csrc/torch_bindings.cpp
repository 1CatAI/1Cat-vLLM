// Provides torch::Tensor for ops.h (previously included transitively via
// cache.h, which is no longer included here after cache ops moved to
// _C_stable_libtorch).
#include <torch/all.h>
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"
#include <torch/library.h>
#include <torch/version.h>

namespace {

bool sm70_marlin_available() {
#ifdef ENABLE_SM70_MARLIN
  return true;
#else
  return false;
#endif
}

}  // namespace

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops
  //

  ops.def(
      "persistent_masked_m_silu_mul_quant(Tensor input, Tensor counts, Tensor! "
      "y_q, Tensor! y_s,"
      "bool use_ue8m0) -> ()");
  ops.impl("persistent_masked_m_silu_mul_quant", torch::kCUDA,
           &persistent_masked_m_silu_mul_quant);

  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);

  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
  ops.impl("get_cuda_view_from_cpu_tensor", torch::kCPU,
           &get_cuda_view_from_cpu_tensor);

  // Activation ops (quantized only — basic ops moved to _C_stable_libtorch)
#ifdef VLLM_REGISTER_BASIC_ACTIVATION_IN_C
  // Compatibility path for local builds where _C_stable_libtorch is not
  // available. Keep disabled by default to avoid duplicate upstream schemas.
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);
#endif

  ops.def(
      "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()");
  ops.impl("silu_and_mul_quant", torch::kCUDA, &silu_and_mul_quant);

  // Fused SiLU+Mul + per-block quantization
  ops.def(
      "silu_and_mul_per_block_quant("
      "Tensor! out, "
      "Tensor input, "
      "Tensor! scales, "
      "int group_size, "
      "Tensor? scale_ub=None, "
      "bool is_scale_transposed=False) -> ()");
  ops.impl("silu_and_mul_per_block_quant", torch::kCUDA,
           &silu_and_mul_per_block_quant);

  // Horizontally-fused DeepseekV4-MLA: per-head RMSNorm + GPT-J RoPE for Q, and
  // GPT-J RoPE + UE8M0 FP8 quant + paged cache insert for KV, all in one
  // kernel launch.
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert("
      "Tensor q_in, Tensor kv, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "int q_head_padded, float eps, int cache_block_size) -> Tensor");
  ops.impl("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert", torch::kCUDA,
           &fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert);

  // Quantization ops
#ifndef USE_ROCM

  // Note about marlin kernel 'workspace' arguments:
  // Technically these should be mutable since they are modified by the kernel.
  // But since they are set back to zero once the kernel is finished we can
  // hand wave and say that they have no net effect.
  //
  // The reason to mark 'workspace' as immutable is so that they don't interfere
  // with using ScalarType arguments in the ops. If they are marked as mutable,
  // pytorch throws an assert in
  // 'torch._higher_order_ops._register_effectful_op' that prevents these
  // kernels from being torch.compile'd.
  // See the following document for more info on custom types and ops that use
  // custom types:
  // https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA

  // Machete (Dense) Optimized Mixed Precision GEMM for Hopper.
  ops.def(
      "machete_supported_schedules("
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? maybe_group_scales_type,"
      "   ScalarType? maybe_group_zeros_type,"
      "   ScalarType? maybe_channel_scales_type,"
      "   ScalarType? maybe_token_scales_type,"
      "   ScalarType? maybe_out_type"
      ") -> str[]");
  ops.def(
      "machete_mm("
      "   Tensor A,"
      "   Tensor B,"
      "   int b_type,"
      "   ScalarType? out_type,"
      "   Tensor? group_scales,"
      "   Tensor? group_zeros,"
      "   int?    group_size,"
      "   Tensor? channel_scales,"
      "   Tensor? token_scales,"
      "   str?    schedule"
      ") -> Tensor");
  ops.def(
      "machete_prepack_B("
      "   Tensor B,"
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? group_scales_type"
      ") -> Tensor");
  // conditionally compiled so impl registration is in source file

  // Marlin Optimized Quantized GEMM (supports GPTQ, AWQ, FP8, NVFP4, MXFP4).
  ops.def("sm70_marlin_available() -> bool");
  ops.impl("sm70_marlin_available", &sm70_marlin_available);

  ops.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none,Tensor b_scales, "
      "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, "
      "Tensor? "
      "g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // gptq_marlin repack from GPTQ.
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // awq_marlin repack from AWQ.
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // preprocess W-int4A-fp8 weight for marlin kernel
  ops.def(
      "marlin_int4_fp8_preprocess(Tensor qweight, "
      "Tensor? qzeros_or_none, bool inplace) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  #ifdef ENABLE_SM70_TURBOMIND
  ops.def("silu_and_mul_interleaved(Tensor! result, Tensor input) -> ()");
  ops.impl("silu_and_mul_interleaved", torch::kCUDA, &silu_and_mul_interleaved);

  ops.def(
      "awq_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, Tensor "
      "_zeros, "
      "int group_size, bool interleave_gated_silu) -> Tensor[]");
  ops.impl("awq_sm70_prepare", torch::kCUDA, &awq_sm70_prepare);

  ops.def(
      "awq_sm70_prepare_compact(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, int group_size, bool interleave_gated_silu) -> "
      "Tensor[]");
  ops.impl("awq_sm70_prepare_compact", torch::kCUDA, &awq_sm70_prepare_compact);

  ops.def(
      "awq_sm70_dequantize_out(Tensor(a!) out, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size) -> ()");
  ops.impl("awq_sm70_dequantize_out", torch::kCUDA, &awq_sm70_dequantize_out);

  ops.def(
      "uint4_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, int group_size, bool interleave_gated_silu) -> Tensor[]");
  ops.impl("uint4_sm70_prepare", torch::kCUDA, &uint4_sm70_prepare);

  ops.def(
      "fp8_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, "
      "int group_size, bool interleave_gated_silu) -> Tensor[]");
  ops.impl("fp8_sm70_prepare", torch::kCUDA, &fp8_sm70_prepare);

  ops.def(
      "fp8_sm70_dequantize_out(Tensor(a!) out, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size) -> ()");
  ops.impl("fp8_sm70_dequantize_out", torch::kCUDA, &fp8_sm70_dequantize_out);

  ops.def(
      "mxfp4_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, "
      "int group_size, bool interleave_gated_silu) -> Tensor[]");
  ops.impl("mxfp4_sm70_prepare", torch::kCUDA, &mxfp4_sm70_prepare);

  ops.def(
      "nvfp4_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, "
      "int group_size, bool interleave_gated_silu) -> Tensor[]");
  ops.impl("nvfp4_sm70_prepare", torch::kCUDA, &nvfp4_sm70_prepare);

  ops.def("sm70_f16_prepare(Tensor _kernel) -> Tensor[]");
  ops.impl("sm70_f16_prepare", torch::kCUDA, &sm70_f16_prepare);

  ops.def(
      "sm70_glm53_tp8_cublaslt_out("
      "Tensor(a!) out, Tensor input, Tensor weight) -> ()");
  ops.impl("sm70_glm53_tp8_cublaslt_out", torch::kCUDA,
           &sm70_glm53_tp8_cublaslt_out);

  ops.def(
      "awq_gemm_sm70(Tensor _in_feats, Tensor _kernel, Tensor "
      "_scaling_factors, int group_size, int k_ld, int q_ld) -> Tensor");
  ops.impl("awq_gemm_sm70", torch::kCUDA, &awq_gemm_sm70);

  ops.def("sm70_f16_gemm(Tensor _in_feats, Tensor _kernel) -> Tensor");
  ops.impl("sm70_f16_gemm", torch::kCUDA, &sm70_f16_gemm);

  ops.def(
      "awq_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu) -> ()");
  ops.impl("awq_gemm_sm70_out", torch::kCUDA, &awq_gemm_sm70_out);

  ops.def(
      "awq_gemm_sm70_out_tile_reduce(Tensor(a!) out, Tensor(b!) staging, "
      "Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, "
      "int group_size, int k_ld, int q_ld, int fa_ptr, int tile_numel, "
      "int reducer_blocks, int kernel_reducer_blocks, bool overlap) -> ()");
  ops.impl("awq_gemm_sm70_out_tile_reduce", torch::kCUDA,
           &awq_gemm_sm70_out_tile_reduce);

  ops.def(
      "fp8_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu) -> ()");
  ops.impl("fp8_gemm_sm70_out", torch::kCUDA, &fp8_gemm_sm70_out);

  ops.def("fp8_qpn8_prepare_sm70(Tensor qweight, Tensor scales) -> Tensor[]");
  ops.impl("fp8_qpn8_prepare_sm70", torch::kCUDA, &fp8_qpn8_prepare_sm70);

  ops.def(
      "fp8_qpn8_dequantize_sm70_out(Tensor(a!) out, Tensor codes, "
      "Tensor group_scales) -> ()");
  ops.impl("fp8_qpn8_dequantize_sm70_out", torch::kCUDA,
           &fp8_qpn8_dequantize_sm70_out);

  ops.def(
      "fp8_qpn8_prefill_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor group_scales, bool gated_silu) -> "
      "()");
  ops.impl("fp8_qpn8_prefill_sm70_out", torch::kCUDA,
           &fp8_qpn8_prefill_sm70_out);

  ops.def(
      "fp8_qpn8_dispatch_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor group_scales, int split_k, "
      "int accumulator_chains, bool prefetch_codes, bool gated_silu) -> ()");
  ops.impl("fp8_qpn8_dispatch_sm70_out", torch::kCUDA,
           &fp8_qpn8_dispatch_sm70_out);

  ops.def(
      "fp8_qpn8_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor group_scales, int split_k, int accumulator_chains, "
      "bool fast_decoder, bool prefetch_codes) -> ()");
  ops.impl("fp8_qpn8_gemm_sm70_out", torch::kCUDA, &fp8_qpn8_gemm_sm70_out);

  ops.def(
      "fp8_qpn8_gemm_ba_split_sm70_out(Tensor(a!) qkv_out, Tensor(b!) "
      "z_out, Tensor(c!) b_out, Tensor(d!) a_out, Tensor input, Tensor codes, "
      "Tensor group_scales, Tensor ba_weight) -> ()");
  ops.impl("fp8_qpn8_gemm_ba_split_sm70_out", torch::kCUDA,
           &fp8_qpn8_gemm_ba_split_sm70_out);

  ops.def(
      "fp8_qpn8_dispatch_ba_split_sm70_out(Tensor(a!) qkv_out, Tensor(b!) "
      "z_out, Tensor(c!) b_out, Tensor(d!) a_out, Tensor(e!) qkvz_staging, "
      "Tensor(f!) ba_staging, int dense_weight_ptr, Tensor input, Tensor "
      "codes, Tensor group_scales, Tensor ba_weight) -> ()");
  ops.impl("fp8_qpn8_dispatch_ba_split_sm70_out", torch::kCUDA,
           &fp8_qpn8_dispatch_ba_split_sm70_out);

  ops.def(
      "fp8_qpn8_gated_pair_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor group_scales, int split_k, "
      "int accumulator_chains, bool fast_decoder, bool prefetch_codes) -> ()");
  ops.impl("fp8_qpn8_gated_pair_sm70_out", torch::kCUDA,
           &fp8_qpn8_gated_pair_sm70_out);

  ops.def(
      "fp8_qpn8_hc_dispatch_sm70_out(Tensor(a!) block_out, Tensor(b!) "
      "injection_out, Tensor(c!) down_staging, Tensor(d!) lora_staging, "
      "Tensor(e!) gate_staging, Tensor(f!) partials, int dense_weight_ptr, "
      "Tensor xn, Tensor down_codes, Tensor down_scales, Tensor up_codes, "
      "Tensor up_scales) -> ()");
  ops.impl("fp8_qpn8_hc_dispatch_sm70_out", torch::kCUDA,
           &fp8_qpn8_hc_dispatch_sm70_out);

  ops.def("nvfp4_qpn4_prepare_sm70(Tensor qweight, Tensor scales) -> Tensor[]");
  ops.impl("nvfp4_qpn4_prepare_sm70", torch::kCUDA, &nvfp4_qpn4_prepare_sm70);

  ops.def(
      "nvfp4_qpn4_prepare_scale_code_sm70(Tensor qweight, Tensor "
      "scale_codes) -> Tensor[]");
  ops.impl("nvfp4_qpn4_prepare_scale_code_sm70", torch::kCUDA,
           &nvfp4_qpn4_prepare_scale_code_sm70);

  ops.def(
      "nvfp4_qpn4_dequantize_sm70_out(Tensor(a!) out, Tensor codes, Tensor "
      "scales, float global_scale, bool use_scale_code) -> ()");
  ops.impl("nvfp4_qpn4_dequantize_sm70_out", torch::kCUDA,
           &nvfp4_qpn4_dequantize_sm70_out);

  ops.def(
      "nvfp4_qpn4_prefill_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor scales, float global_scale, bool "
      "use_scale_code, bool gated_silu) -> ()");
  ops.impl("nvfp4_qpn4_prefill_sm70_out", torch::kCUDA,
           &nvfp4_qpn4_prefill_sm70_out);

  ops.def(
      "nvfp4_qpn4_dispatch_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor scales, float global_scale, bool "
      "use_scale_code, bool gated_silu) -> ()");
  ops.impl("nvfp4_qpn4_dispatch_sm70_out", torch::kCUDA,
           &nvfp4_qpn4_dispatch_sm70_out);

  ops.def(
      "fp8_gemm_sm70_prefill_prescaled_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _prescaled_factors, int group_size, int k_ld, "
      "int q_ld) -> ()");
  ops.impl("fp8_gemm_sm70_prefill_prescaled_out", torch::kCUDA,
           &fp8_gemm_sm70_prefill_prescaled_out);

  // A distinct schema is also a capability marker: older extensions expose
  // only the 8K-prefill contract and must not receive the new M=1 shapes.
  ops.def(
      "fp8_gemm_sm70_prescaled_m1_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _prescaled_factors, int group_size, int k_ld, "
      "int q_ld) -> ()");
  ops.impl("fp8_gemm_sm70_prescaled_m1_out", torch::kCUDA,
           &fp8_gemm_sm70_prescaled_m1_out);

  ops.def(
      "nvfp4_qpn2_prepare_sm70(Tensor weight_packed, Tensor weight_scale) -> "
      "Tensor[]");
  ops.impl("nvfp4_qpn2_prepare_sm70", torch::kCUDA, &nvfp4_qpn2_prepare_sm70);

  ops.def(
      "nvfp4_qpn2_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains) -> ()");
  ops.impl("nvfp4_qpn2_gemm_sm70_out", torch::kCUDA, &nvfp4_qpn2_gemm_sm70_out);

  ops.def(
      "nvfp4_qpn2_gated_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains) -> ()");
  ops.impl("nvfp4_qpn2_gated_sm70_out", torch::kCUDA,
           &nvfp4_qpn2_gated_sm70_out);

  ops.def(
      "nvfp4_qpn2_dispatch_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains, Tensor tm_weight, Tensor tm_scales, "
      "int tm_group_size, int tm_k_ld, int tm_q_ld, bool gated_silu) -> ()");
  ops.impl("nvfp4_qpn2_dispatch_sm70_out", torch::kCUDA,
           &nvfp4_qpn2_dispatch_sm70_out);

  ops.def(
      "nvfp4_qpn2_prefill_dispatch_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains, Tensor tm_weight, Tensor tm_scales, "
      "int tm_group_size, int tm_k_ld, int tm_q_ld, bool gated_silu, "
      "int min_prefill_m) -> ()");
  ops.impl("nvfp4_qpn2_prefill_dispatch_sm70_out", torch::kCUDA,
           &nvfp4_qpn2_prefill_dispatch_sm70_out);

  ops.def(
      "fp8_gemm_sm70_prefill_dispatch_out(Tensor(a!) out, "
      "int dense_weight_ptr, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu, int min_prefill_m) -> ()");
  ops.impl("fp8_gemm_sm70_prefill_dispatch_out", torch::kCUDA,
           &fp8_gemm_sm70_prefill_dispatch_out);

  ops.def(
      "mxfp4_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu) -> ()");
  ops.impl("mxfp4_gemm_sm70_out", torch::kCUDA, &mxfp4_gemm_sm70_out);

  ops.def(
      "nvfp4_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu) -> ()");
  ops.impl("nvfp4_gemm_sm70_out", torch::kCUDA, &nvfp4_gemm_sm70_out);

  ops.def(
      "nvfp4_gemv_sm70_raw_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _scaling_factors, Tensor(b!) partials, "
      "int group_size, int split_k) -> ()");
  ops.impl("nvfp4_gemv_sm70_raw_out", torch::kCUDA, &nvfp4_gemv_sm70_raw_out);

  ops.def(
      "nvfp4_gemv_sm70_warp_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _scaling_factors, int group_size) -> ()");
  ops.impl("nvfp4_gemv_sm70_warp_out", torch::kCUDA, &nvfp4_gemv_sm70_warp_out);

  ops.def(
      "nvfp4_gemv_sm70_h2_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _scaling_factors, Tensor(b!) partials, "
      "int group_size, int split_k) -> ()");
  ops.impl("nvfp4_gemv_sm70_h2_out", torch::kCUDA, &nvfp4_gemv_sm70_h2_out);

  ops.def(
      "fp8_gemm_sm70_out_auto(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _scaling_factors) -> ()");
  ops.impl("fp8_gemm_sm70_out_auto", torch::kCUDA, &fp8_gemm_sm70_out_auto);

  ops.def(
      "fp8_gemm_sm70_out_meta(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _scaling_factors, Tensor _meta, "
      "bool gated_silu) -> ()");
  ops.impl("fp8_gemm_sm70_out_meta", torch::kCUDA, &fp8_gemm_sm70_out_meta);

  ops.def(
      "sm70_f16_gemm_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "int k_ld, bool gated_silu) -> ()");
  ops.impl("sm70_f16_gemm_out", torch::kCUDA, &sm70_f16_gemm_out);

  ops.def(
      "sm70_glm_mhc_pre_norm_out("
      "Tensor gemm_mul, Tensor gemm_sqrsum, Tensor hc_scale, Tensor hc_base, "
      "Tensor residual, Tensor(a!) post_mix, Tensor(b!) comb_mix, "
      "Tensor(c!) layer_input, Tensor norm_weight, float rms_eps, "
      "float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult, "
      "int sinkhorn_repeat, float norm_eps) -> ()");
  ops.impl("sm70_glm_mhc_pre_norm_out", torch::kCUDA,
           &sm70_glm_mhc_pre_norm_out);

  ops.def(
      "sm70_glm_mhc_post_dot_q8_out("
      "Tensor(a!) residual_out, Tensor(b!) gemm_mul, "
      "Tensor(c!) gemm_sqrsum, Tensor comb_mix, Tensor residual, "
      "Tensor post_mix, Tensor x, Tensor weight, int tile_n) -> ()");
  ops.impl("sm70_glm_mhc_post_dot_q8_out", torch::kCUDA,
           &sm70_glm_mhc_post_dot_q8_out);

  ops.def(
      "sm70_glm_kda_fg_b_out(Tensor(a!) f_out, Tensor(b!) g_out, "
      "Tensor f_input, Tensor g_input, Tensor f_weight, Tensor g_weight) -> "
      "()");
  ops.impl("sm70_glm_kda_fg_b_out", torch::kCUDA, &sm70_glm_kda_fg_b_out);

  ops.def(
      "sm70_glm53_fp16_gemv_out(Tensor(a!) output, Tensor input, Tensor "
      "weight) -> ()");
  ops.impl("sm70_glm53_fp16_gemv_out", torch::kCUDA, &sm70_glm53_fp16_gemv_out);

  ops.def(
      "sm70_glm53_moe_permute_q8_out("
      "Tensor input, Tensor topk_ids, Tensor(a!) permuted_input, "
      "Tensor(b!) sorted_row_idx, Tensor(c!) inv_permuted_idx, "
      "Tensor(d!) compact_offsets, Tensor(e!) active_expert_ids) -> ()");
  ops.impl("sm70_glm53_moe_permute_q8_out", torch::kCUDA,
           &sm70_glm53_moe_permute_q8_out);

  ops.def(
      "sm70_f16_indexed_rerank_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor candidate_ids, Tensor(b!) selected_raw, "
      "Tensor(c!) selected_packed, Tensor(d!) expanded, Tensor(e!) partials, "
      "Tensor(f!) barriers, int cta_n, int split_k) -> ()");
  ops.impl("sm70_f16_indexed_rerank_out", torch::kCUDA,
           &sm70_f16_indexed_rerank_out);

  ops.def(
      "sm70_f16_indexed_rerank_packed_out(Tensor(a!) out, "
      "Tensor _in_feats, Tensor _packed_kernel, Tensor candidate_ids, "
      "Tensor(b!) selected_packed, Tensor(c!) expanded, "
      "Tensor(d!) partials, Tensor(e!) barriers, int cta_n, int split_k) "
      "-> ()");
  ops.impl("sm70_f16_indexed_rerank_packed_out", torch::kCUDA,
           &sm70_f16_indexed_rerank_packed_out);

  ops.def(
      "sm70_f16_rerank_keys_out(Tensor(a!) keys, Tensor logits, "
      "Tensor candidate_ids) -> ()");
  ops.impl("sm70_f16_rerank_keys_out", torch::kCUDA, &sm70_f16_rerank_keys_out);

  ops.def(
      "sm70_f16_rerank_topk_out(Tensor(a!) values_out, "
      "Tensor(b!) ids_out, Tensor logits, Tensor candidate_ids, "
      "int vocab_start_index) -> ()");
  ops.impl("sm70_f16_rerank_topk_out", torch::kCUDA, &sm70_f16_rerank_topk_out);

  ops.def(
      "sm70_f16_lm_head_top1_out(Tensor(a!) values_out, "
      "Tensor(b!) indices_out, Tensor _in_feats, Tensor _kernel, int k_ld, "
      "int vocab_start_index, int num_vocab_padding) -> ()");
  ops.impl("sm70_f16_lm_head_top1_out", torch::kCUDA,
           &sm70_f16_lm_head_top1_out);

  ops.def(
      "sm70_f16_lm_head_top1_tc_out(Tensor(a!) values_out, "
      "Tensor(b!) indices_out, Tensor _in_feats, Tensor _kernel, int k_ld, "
      "int vocab_start_index, int num_vocab_padding) -> ()");
  ops.impl("sm70_f16_lm_head_top1_tc_out", torch::kCUDA,
           &sm70_f16_lm_head_top1_tc_out);

  ops.def(
      "sm70_f16_lm_head_top20_tc_out(Tensor(a!) values_out, "
      "Tensor(b!) indices_out, Tensor _in_feats, Tensor _kernel, int k_ld, "
      "int vocab_start_index, int num_vocab_padding) -> ()");
  ops.impl("sm70_f16_lm_head_top20_tc_out", torch::kCUDA,
           &sm70_f16_lm_head_top20_tc_out);

  ops.def(
      "sm70_merge_tail_top20_pack_out(Tensor(a!) pairs_out, "
      "Tensor base_values, Tensor base_indices, Tensor base_token_id_map, "
      "Tensor tail_logits, Tensor tail_token_ids, int tail_row_start) -> ()");
  ops.impl("sm70_merge_tail_top20_pack_out", torch::kCUDA,
           &sm70_merge_tail_top20_pack_out);

  ops.def(
      "sm70_sample_packed_top20_out(Tensor(a!) sampled_token_out, "
      "Tensor(b!) sparse_ids_out, Tensor(c!) sparse_probs_out, "
      "Tensor gathered_pairs, Tensor exponential, float top_p) -> ()");
  ops.impl("sm70_sample_packed_top20_out", torch::kCUDA,
           &sm70_sample_packed_top20_out);

  ops.def(
      "sm70_dynamic_draft_vocab_update_tail_out(Tensor(a!) lru_token_ids, "
      "Tensor(b!) local_tail_token_ids, Tensor(c!) source_row_indices, "
      "Tensor observed_output_ids, Tensor target_candidate_ids, "
      "Tensor base_token_mask, int full_vocab_size, int local_shard_start, "
      "int local_shard_end) -> ()");
  ops.impl("sm70_dynamic_draft_vocab_update_tail_out", torch::kCUDA,
           &sm70_dynamic_draft_vocab_update_tail_out);

  ops.def(
      "sm70_dynamic_draft_vocab_refresh_tail_weight_out("
      "Tensor(a!) local_tail_weight, Tensor source_weight, "
      "Tensor source_row_indices) -> ()");
  ops.impl("sm70_dynamic_draft_vocab_refresh_tail_weight_out", torch::kCUDA,
           &sm70_dynamic_draft_vocab_refresh_tail_weight_out);

  ops.def(
      "sm70_f16_gate_mul_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _gate_weight) -> ()");
  ops.impl("sm70_f16_gate_mul_out", torch::kCUDA, &sm70_f16_gate_mul_out);

  ops.def(
      "qwen38_shared_gate_exact_out(Tensor(a!) out, Tensor input, "
      "Tensor weight) -> ()");
  ops.impl("qwen38_shared_gate_exact_out", torch::kCUDA,
           &qwen38_shared_gate_exact_out);

  ops.def("sm70_gemm_import_cache(Tensor device_hint, str path) -> int");
  ops.impl("sm70_gemm_import_cache", torch::kCUDA, &sm70_gemm_import_cache);

  ops.def("sm70_gemm_export_cache(Tensor device_hint, str path) -> int");
  ops.impl("sm70_gemm_export_cache", torch::kCUDA, &sm70_gemm_export_cache);

  ops.def(
      "awq_moe_build_strided_ptrs(Tensor tm_weights, Tensor tm_scales, "
      "int k_ld, int q_ld, int num_experts) -> Tensor[]");
  ops.impl("awq_moe_build_strided_ptrs", torch::kCUDA,
           &awq_moe_build_strided_ptrs);

  ops.def(
      "awq_moe_gemm_sm70_out(Tensor(a!) out, Tensor sorted_input, "
      "Tensor expert_offsets, Tensor strided_ptrs_w, Tensor strided_ptrs_s, "
      "int num_experts, int k, int n, int group_size, bool gated_silu) -> ()");
  ops.impl("awq_moe_gemm_sm70_out", torch::kCUDA, &awq_moe_gemm_sm70_out);

  ops.def(
      "awq_moe_gemm_sm70_per_expert_dispatch_out("
      "Tensor(a!) out, Tensor sorted_input, Tensor expert_offsets, "
      "Tensor strided_ptrs_w, Tensor strided_ptrs_s, int num_experts, "
      "int k, int n, int group_size, bool gated_silu) -> ()");
  ops.impl("awq_moe_gemm_sm70_per_expert_dispatch_out", torch::kCUDA,
           &awq_moe_gemm_sm70_per_expert_dispatch_out);

  ops.def(
      "awq_moe_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor dense_expert_ids, Tensor ptrs_w, Tensor ptrs_s, "
      "int num_experts, int k, int n, int group_size) -> ()");
  ops.impl("awq_moe_dense_stage_sm70_out", torch::kCUDA,
           &awq_moe_dense_stage_sm70_out);

  ops.def(
      "awq_moe_indexed_dense_w13_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor input_row_indices, "
      "Tensor expert_offsets, Tensor dense_expert_ids, Tensor ptrs_w, "
      "Tensor ptrs_s, int num_experts, int k, int n, int group_size) -> ()");
  ops.impl("awq_moe_indexed_dense_w13_sm70_out", torch::kCUDA,
           &awq_moe_indexed_dense_w13_sm70_out);

  ops.def(
      "awq_moe_active_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor permuted_experts_id, "
      "Tensor active_expert_offsets, Tensor active_expert_ids, Tensor ptrs_w, "
      "Tensor ptrs_s, int total_slots, int k, int n, int group_size) -> ()");
  ops.impl("awq_moe_active_dense_stage_sm70_out", torch::kCUDA,
           &awq_moe_active_dense_stage_sm70_out);

  ops.def(
      "awq_moe_single_token_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor sorted_expert_ids, Tensor ptrs_w, Tensor ptrs_s, int top_k, "
      "int k, int n, int group_size) -> ()");
  ops.impl("awq_moe_single_token_dense_stage_sm70_out", torch::kCUDA,
           &awq_moe_single_token_dense_stage_sm70_out);

  ops.def(
      "awq_moe_single_token_indexed_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor sorted_expert_ids, Tensor ptrs_w, Tensor ptrs_s, int top_k, "
      "int k, int n, int group_size) -> ()");
  ops.impl("awq_moe_single_token_indexed_dense_stage_sm70_out", torch::kCUDA,
           &awq_moe_single_token_indexed_dense_stage_sm70_out);

  ops.def(
      "awq_moe_single_token_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) expert_offsets, Tensor(d!) expert_offsets64, "
      "Tensor(e!) inv_permuted_idx, Tensor(f!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size) -> ()");
  ops.impl("awq_moe_single_token_dense_w13_sm70_out", torch::kCUDA,
           &awq_moe_single_token_dense_w13_sm70_out);

  ops.def(
      "awq_moe_single_token_indexed_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) expert_offsets, Tensor(d!) expert_offsets64, "
      "Tensor(e!) inv_permuted_idx, Tensor(f!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size) -> ()");
  ops.impl("awq_moe_single_token_indexed_dense_w13_sm70_out", torch::kCUDA,
           &awq_moe_single_token_indexed_dense_w13_sm70_out);

  ops.def(
      "awq_moe_single_token_compact_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) compact_w13_ptrs_w, Tensor(d!) compact_w13_ptrs_s, "
      "Tensor(e!) expert_offsets, Tensor(f!) expert_offsets64, "
      "Tensor(g!) inv_permuted_idx, Tensor(h!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size) -> ()");
  ops.impl("awq_moe_single_token_compact_dense_w13_sm70_out", torch::kCUDA,
           &awq_moe_single_token_compact_dense_w13_sm70_out);

  ops.def(
      "awq_moe_single_token_exact_layout_prepare("
      "Tensor topk_ids, Tensor x, Tensor(a!) compact_input, "
      "Tensor(b!) expert_offsets, Tensor(c!) expert_offsets64, "
      "Tensor(d!) inv_permuted_idx, int num_experts) -> ()");
  ops.impl("awq_moe_single_token_exact_layout_prepare", torch::kCUDA,
           &awq_moe_single_token_exact_layout_prepare);

  ops.def(
      "awq_moe_single_token_weighted_reduce_out("
      "Tensor sorted_output, Tensor topk_weights, Tensor inv_permuted_idx, "
      "Tensor(a!) out, int top_k, int hidden_logical_size) -> ()");
  ops.impl("awq_moe_single_token_weighted_reduce_out", torch::kCUDA,
           &awq_moe_single_token_weighted_reduce_out);

  ops.def(
      "awq_moe_single_token_sm70_out("
      "Tensor(a!) out, Tensor x, Tensor topk_weights, Tensor topk_ids, "
      "Tensor src_w13_ptrs_w_rows, Tensor src_w13_ptrs_s_rows, "
      "Tensor src_w2_ptrs_w_rows, Tensor src_w2_ptrs_s_rows, "
      "Tensor(b!) compact_input, Tensor(c!) intermediate, "
      "Tensor(d!) sorted_output, Tensor(e!) sorted_weights, "
      "Tensor(f!) dst_w13_ptrs_w_rows, Tensor(g!) dst_w13_ptrs_s_rows, "
      "Tensor(h!) dst_w2_ptrs_w_rows, Tensor(i!) dst_w2_ptrs_s_rows, "
      "Tensor(j!) expert_offsets, Tensor(k!) inv_permuted_idx, "
      "int w13_k, int w13_n, int w2_k, int w2_n, int group_size, "
      "int hidden_logical_size) -> ()");
  ops.impl("awq_moe_single_token_sm70_out", torch::kCUDA,
           &awq_moe_single_token_sm70_out);

  ops.def(
      "fp8_moe_gemm_sm70_out(Tensor(a!) out, Tensor sorted_input, "
      "Tensor expert_offsets, Tensor strided_ptrs_w, Tensor strided_ptrs_s, "
      "int num_experts, int k, int n, int group_size, bool gated_silu) -> ()");
  ops.impl("fp8_moe_gemm_sm70_out", torch::kCUDA, &fp8_moe_gemm_sm70_out);

  ops.def(
      "fp8_moe_gemm_sm70_per_expert_dispatch_out("
      "Tensor(a!) out, Tensor sorted_input, Tensor expert_offsets, "
      "Tensor strided_ptrs_w, Tensor strided_ptrs_s, int num_experts, "
      "int k, int n, int group_size, bool gated_silu) -> ()");
  ops.impl("fp8_moe_gemm_sm70_per_expert_dispatch_out", torch::kCUDA,
           &fp8_moe_gemm_sm70_per_expert_dispatch_out);

  ops.def(
      "fp8_moe_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor dense_expert_ids, Tensor ptrs_w, Tensor ptrs_s, "
      "int num_experts, int k, int n, int group_size) -> ()");
  ops.impl("fp8_moe_dense_stage_sm70_out", torch::kCUDA,
           &fp8_moe_dense_stage_sm70_out);

  ops.def(
      "mxfp4_moe_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor dense_expert_ids, Tensor ptrs_w, Tensor ptrs_s, "
      "int num_experts, int k, int n, int group_size) -> ()");
  ops.impl("mxfp4_moe_dense_stage_sm70_out", torch::kCUDA,
           &mxfp4_moe_dense_stage_sm70_out);

  ops.def(
      "mxfp4_moe_qpn_m1_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, bool broadcast_input) -> ()");
  ops.impl("mxfp4_moe_qpn_m1_sm70_out", torch::kCUDA,
           &mxfp4_moe_qpn_m1_sm70_out);

  ops.def(
      "nvfp4_moe_qpn_m1_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, bool broadcast_input, int split_k) -> ()");
  ops.impl("nvfp4_moe_qpn_m1_sm70_out", torch::kCUDA,
           &nvfp4_moe_qpn_m1_sm70_out);

  ops.def(
      "nvfp4_expand_raw_scales_sm70_out("
      "Tensor(a!) out, Tensor scale_codes, Tensor global_scales, "
      "bool interleaved_w13, bool fast_decode_rounding) -> ()");
  ops.impl("nvfp4_expand_raw_scales_sm70_out", torch::kCUDA,
           &nvfp4_expand_raw_scales_sm70_out);

  ops.def(
      "nvfp4_moe_qpn_raw_scale_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scale_codes, "
      "Tensor global_scales, Tensor expert_ids, bool broadcast_input, "
      "bool interleaved_w13, int split_k) -> ()");
  ops.impl("nvfp4_moe_qpn_raw_scale_sm70_out", torch::kCUDA,
           &nvfp4_moe_qpn_raw_scale_sm70_out);

  ops.def(
      "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, bool interleaved) -> ()");
  ops.impl("nvfp4_moe_qpn_w13_swiglu_batch_sm70_out", torch::kCUDA,
           &nvfp4_moe_qpn_w13_swiglu_batch_sm70_out);

  ops.def(
      "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scale_codes, "
      "Tensor global_scales, Tensor expert_ids, bool interleaved) -> ()");
  ops.impl("nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out", torch::kCUDA,
           &nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out);

  ops.def(
      "nvfp4_moe_qpn_w2_reduce_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, Tensor topk_weights) -> ()");
  ops.impl("nvfp4_moe_qpn_w2_reduce_sm70_out", torch::kCUDA,
           &nvfp4_moe_qpn_w2_reduce_sm70_out);

  ops.def(
      "nvfp4_moe_qpn_raw_w2_reduce_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scale_codes, "
      "Tensor global_scales, Tensor expert_ids, Tensor topk_weights) -> ()");
  ops.impl("nvfp4_moe_qpn_raw_w2_reduce_sm70_out", torch::kCUDA,
           &nvfp4_moe_qpn_raw_w2_reduce_sm70_out);

  ops.def(
      "nvfp4_qwen38_w2_direct_reduce_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, Tensor topk_weights) -> ()");
  ops.impl("nvfp4_qwen38_w2_direct_reduce_out", torch::kCUDA,
           &nvfp4_qwen38_w2_direct_reduce_out);

  ops.def(
      "nvfp4_qwen38_w13_fused_swiglu_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids) -> ()");
  ops.impl("nvfp4_qwen38_w13_fused_swiglu_out", torch::kCUDA,
           &nvfp4_qwen38_w13_fused_swiglu_out);

  // Keep the five-row verifier on a distinct schema so an old extension that
  // only supports the ten-route M=1 contract cannot be selected accidentally.
  ops.def(
      "nvfp4_moe_qpn_mtp5_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, bool broadcast_input, int split_k) -> ()");
  ops.impl("nvfp4_moe_qpn_mtp5_sm70_out", torch::kCUDA,
           &nvfp4_moe_qpn_mtp5_sm70_out);

  ops.def(
      "nvfp4_glm53_moe_q8_qpn_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, Tensor sorted_row_idx, bool w13) -> ()");
  ops.impl("nvfp4_glm53_moe_q8_qpn_sm70_out", torch::kCUDA,
           &nvfp4_glm53_moe_q8_qpn_sm70_out);

  ops.def(
      "nvfp4_moe_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor dense_expert_ids, Tensor ptrs_w, Tensor ptrs_s, "
      "int num_experts, int k, int n, int group_size) -> ()");
  ops.impl("nvfp4_moe_dense_stage_sm70_out", torch::kCUDA,
           &nvfp4_moe_dense_stage_sm70_out);

  ops.def(
      "nvfp4_moe_indexed_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor input_row_indices, "
      "Tensor expert_offsets, Tensor dense_expert_ids, Tensor ptrs_w, "
      "Tensor ptrs_s, int num_experts, int k, int n, int group_size) -> ()");
  ops.impl("nvfp4_moe_indexed_dense_stage_sm70_out", torch::kCUDA,
           &nvfp4_moe_indexed_dense_stage_sm70_out);

  ops.def(
      "nvfp4_moe_indexed_fused_swiglu_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor input_row_indices, "
      "Tensor expert_offsets, Tensor dense_expert_ids, Tensor ptrs_w, "
      "Tensor ptrs_s, int num_experts, int k, int n, int group_size) -> ()");
  ops.impl("nvfp4_moe_indexed_fused_swiglu_sm70_out", torch::kCUDA,
           &nvfp4_moe_indexed_fused_swiglu_sm70_out);

  ops.def(
      "mxfp4_moe_single_token_prepare_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) expert_offsets, Tensor(d!) inv_permuted_idx, "
      "Tensor(e!) sorted_expert_ids, int w13_k, int w13_n, "
      "int group_size, int hidden_logical_size) -> ()");
  ops.impl("mxfp4_moe_single_token_prepare_w13_sm70_out", torch::kCUDA,
           &mxfp4_moe_single_token_prepare_w13_sm70_out);

  ops.def(
      "fp8_moe_single_token_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor sorted_expert_ids, Tensor ptrs_w, Tensor ptrs_s, int top_k, "
      "int k, int n, int group_size) -> ()");
  ops.impl("fp8_moe_single_token_dense_stage_sm70_out", torch::kCUDA,
           &fp8_moe_single_token_dense_stage_sm70_out);

  ops.def(
      "fp8_moe_single_token_indexed_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor sorted_expert_ids, Tensor ptrs_w, Tensor ptrs_s, int top_k, "
      "int k, int n, int group_size) -> ()");
  ops.impl("fp8_moe_single_token_indexed_dense_stage_sm70_out", torch::kCUDA,
           &fp8_moe_single_token_indexed_dense_stage_sm70_out);

  ops.def(
      "fp8_moe_single_token_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) expert_offsets, Tensor(d!) expert_offsets64, "
      "Tensor(e!) inv_permuted_idx, Tensor(f!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size) -> ()");
  ops.impl("fp8_moe_single_token_dense_w13_sm70_out", torch::kCUDA,
           &fp8_moe_single_token_dense_w13_sm70_out);

  ops.def(
      "fp8_moe_single_token_indexed_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) expert_offsets, Tensor(d!) expert_offsets64, "
      "Tensor(e!) inv_permuted_idx, Tensor(f!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size) -> ()");
  ops.impl("fp8_moe_single_token_indexed_dense_w13_sm70_out", torch::kCUDA,
           &fp8_moe_single_token_indexed_dense_w13_sm70_out);

  ops.def(
      "fp8_moe_single_token_compact_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) compact_w13_ptrs_w, Tensor(d!) compact_w13_ptrs_s, "
      "Tensor(e!) expert_offsets, Tensor(f!) expert_offsets64, "
      "Tensor(g!) inv_permuted_idx, Tensor(h!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size) -> ()");
  ops.impl("fp8_moe_single_token_compact_dense_w13_sm70_out", torch::kCUDA,
           &fp8_moe_single_token_compact_dense_w13_sm70_out);

  ops.def(
      "fp8_moe_single_token_sm70_out("
      "Tensor(a!) out, Tensor x, Tensor topk_weights, Tensor topk_ids, "
      "Tensor src_w13_ptrs_w_rows, Tensor src_w13_ptrs_s_rows, "
      "Tensor src_w2_ptrs_w_rows, Tensor src_w2_ptrs_s_rows, "
      "Tensor(b!) compact_input, Tensor(c!) gate_up, Tensor(d!) intermediate, "
      "Tensor(e!) sorted_output, Tensor(f!) sorted_weights, "
      "Tensor(g!) dst_w13_ptrs_w_rows, Tensor(h!) dst_w13_ptrs_s_rows, "
      "Tensor(i!) dst_w2_ptrs_w_rows, Tensor(j!) dst_w2_ptrs_s_rows, "
      "Tensor(k!) expert_offsets, Tensor(l!) inv_permuted_idx, "
      "Tensor(m!) sorted_expert_ids, Tensor broadcast_input_indices, "
      "Tensor w2_raw_weight, Tensor w2_raw_scale_inv, "
      "int w13_k, int w13_n, int w2_k, int w2_n, int group_size, "
      "int hidden_logical_size, bool fused_gated_silu, "
      "bool fused_weighted_reduce, bool broadcast_input, "
      "bool w2_direct_reduce, bool indexed_expert_ptrs, "
      "bool exact_per_route) -> ()");
  ops.impl("fp8_moe_single_token_sm70_out", torch::kCUDA,
           &fp8_moe_single_token_sm70_out);
  #endif

#endif

#ifndef USE_ROCM
  // Expert-specialization mxfp8 blockscaled grouped quantization (SM100+).
  ops.def(
      "mxfp8_experts_quant("
      " Tensor input, Tensor problem_sizes, Tensor expert_offsets,"
      " Tensor blockscale_offsets, Tensor! quant_output, Tensor! scale_factor)"
      " -> ()");
  // conditionally compiled so impl registration is in source file

  // Expert-specialization mxfp8 blockscaled grouped GEMM (SM100+).
  ops.def(
      "cutlass_mxfp8_grouped_mm("
      " Tensor a, Tensor b, Tensor sfa, Tensor sfb, Tensor! out,"
      " Tensor problem_sizes, Tensor expert_offsets, Tensor blockscale_offsets)"
      " -> ()");
  // conditionally compiled so impl registration is in source file

#endif

#ifndef USE_ROCM
  ops.def(
      "minimax_allreduce_rms("
      "Tensor input,"
      "Tensor norm_weight,"
      "Tensor workspace,"
      "int rank,"
      "int nranks,"
      "float eps) -> Tensor");
  ops.impl("minimax_allreduce_rms", torch::kCUDA, &minimax_allreduce_rms);
  ops.def(
      "minimax_allreduce_rms_qk("
      "Tensor qkv,"
      "Tensor norm_weight_q,"
      "Tensor norm_weight_k,"
      "Tensor workspace,"
      "int q_size,"
      "int kv_size,"
      "int rank,"
      "int nranks,"
      "float eps) -> (Tensor, Tensor)");
  ops.impl("minimax_allreduce_rms_qk", torch::kCUDA, &minimax_allreduce_rms_qk);

  //  conditionally compiled so impl in source file
#endif
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // Cuda utils

  // Gets the specified device attribute.
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.impl("get_device_attribute", &get_device_attribute);

  // Gets the maximum shared memory per block device attribute.
  cuda_utils.def(
      "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  &get_max_shared_memory_per_block_device_attribute);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Custom all-reduce kernels
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.impl("all_reduce", torch::kCUDA, &all_reduce);
  custom_ar.def(
      "sm70_tp2_all_reduce_gemma_rms_norm(int fa, Tensor inp, Tensor "
      "residual, Tensor weight, Tensor! normalized_out, Tensor! residual_out, "
      "int reg_buffer, int reg_buffer_sz_bytes, float epsilon) -> ()");
  custom_ar.impl("sm70_tp2_all_reduce_gemma_rms_norm", torch::kCUDA,
                 &sm70_tp2_all_reduce_gemma_rms_norm);
  custom_ar.def(
      "sm70_tp4_all_reduce_gemma_rms_norm(int fa, Tensor inp, Tensor "
      "residual, Tensor weight, Tensor! normalized_out, Tensor! residual_out, "
      "int reg_buffer, int reg_buffer_sz_bytes, float epsilon) -> ()");
  custom_ar.impl("sm70_tp4_all_reduce_gemma_rms_norm", torch::kCUDA,
                 &sm70_tp4_all_reduce_gemma_rms_norm);
  custom_ar.def(
      "sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather(int fa, Tensor inp, "
      "Tensor residual, Tensor weight, Tensor! normalized_out, Tensor! "
      "residual_out, int reg_input_buffer, int reg_output_buffer, int "
      "reg_buffer_sz_bytes, float epsilon) -> ()");
  custom_ar.impl("sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather",
                 torch::kCUDA,
                 &sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather);
  custom_ar.def(
      "all_reduce_sum2(int fa, Tensor inp_a, Tensor inp_b, Tensor! out) -> ()");
  custom_ar.impl("all_reduce_sum2", torch::kCUDA, &all_reduce_sum2);
  custom_ar.def(
      "sm70_qwen38_hc_down_allgather(int fa, Tensor inp, Tensor! out) -> ()");
  custom_ar.impl("sm70_qwen38_hc_down_allgather", torch::kCUDA,
                 &sm70_qwen38_hc_down_allgather);
  custom_ar.def(
      "sm70_qwen38_hc_gate_mix(int fa, Tensor local_gate, Tensor branches, "
      "Tensor! out) -> ()");
  custom_ar.impl("sm70_qwen38_hc_gate_mix", torch::kCUDA,
                 &sm70_qwen38_hc_gate_mix);
  custom_ar.def(
      "sm70_qwen38_hc_output_allgather(int fa, Tensor local_block, "
      "Tensor! out) -> ()");
  custom_ar.impl("sm70_qwen38_hc_output_allgather", torch::kCUDA,
                 &sm70_qwen38_hc_output_allgather);
  custom_ar.def(
      "sm70_qwen38_hc_up_mix_allgather(int fa, Tensor lora, Tensor weight, "
      "Tensor branches, Tensor! out) -> ()");
  custom_ar.impl("sm70_qwen38_hc_up_mix_allgather", torch::kCUDA,
                 &sm70_qwen38_hc_up_mix_allgather);
  custom_ar.def(
      "top1_argmax(int fa, Tensor input_pair, Tensor! output, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.impl("top1_argmax", torch::kCUDA, &top1_argmax);
  custom_ar.def(
      "tile_runtime_all_reduce(int fa, Tensor inp, Tensor! out, int "
      "reg_buffer, int reg_buffer_sz_bytes, int tile_numel, int "
      "engine_blocks, int compute_iters) -> ()");
  custom_ar.impl("tile_runtime_all_reduce", torch::kCUDA,
                 &tile_runtime_all_reduce);
  custom_ar.def(
      "tile_runtime_all_reduce_engine(int fa, Tensor inp, Tensor! out, int "
      "reg_buffer, int reg_buffer_sz_bytes, int tile_numel, int "
      "producer_blocks, int reducer_blocks, int compute_iters) -> ()");
  custom_ar.impl("tile_runtime_all_reduce_engine", torch::kCUDA,
                 &tile_runtime_all_reduce_engine);
  custom_ar.def(
      "tile_runtime_wait_reduce(int fa, Tensor staging, Tensor! out, "
      "int tile_numel, int reducer_blocks) -> ()");
  custom_ar.impl("tile_runtime_wait_reduce", torch::kCUDA,
                 &tile_runtime_wait_reduce);

  custom_ar.def("dispose", &dispose);
  custom_ar.def("meta_size", &meta_size);
  custom_ar.def("sm70_tp4_push_allreduce_buffer_size",
                &sm70_tp4_push_allreduce_buffer_size);
  custom_ar.def("sm70_tp8_hierarchical_push_allreduce_buffer_size",
                &sm70_tp8_hierarchical_push_allreduce_buffer_size);

  custom_ar.def("register_buffer", &register_buffer);
  custom_ar.def("register_sm70_tp4_push_allreduce_buffer",
                &register_sm70_tp4_push_allreduce_buffer);
  custom_ar.def("register_sm70_tp8_hierarchical_push_allreduce_buffer",
                &register_sm70_tp8_hierarchical_push_allreduce_buffer);
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  custom_ar.def("register_graph_buffers", &register_graph_buffers);

  custom_ar.def("allocate_shared_buffer_and_handle",
                &allocate_shared_buffer_and_handle);
  custom_ar.def("open_mem_handle(Tensor mem_handle) -> int", &open_mem_handle);
  custom_ar.impl("open_mem_handle", torch::kCPU, &open_mem_handle);

  custom_ar.def("free_shared_buffer", &free_shared_buffer);
#ifdef USE_ROCM
  // Quick Reduce all-reduce kernels
  custom_ar.def(
      "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
      "cast_bf2half) -> ()");
  custom_ar.impl("qr_all_reduce", torch::kCUDA, &qr_all_reduce);

  custom_ar.def("init_custom_qr", &init_custom_qr);
  custom_ar.def("qr_destroy", &qr_destroy);

  custom_ar.def("qr_get_handle", &qr_get_handle);

  custom_ar.def("qr_open_handles(int _fa, Tensor[](b!) handles) -> ()");
  custom_ar.impl("qr_open_handles", torch::kCPU, &qr_open_handles);

  // Max input size in bytes
  custom_ar.def("qr_max_size", &qr_max_size);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
