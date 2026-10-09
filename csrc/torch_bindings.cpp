#include "sm70_policy.h"
// Provides torch::Tensor for ops.h (previously included transitively via
// cache.h, which is no longer included here after cache ops moved to
// _C_stable_libtorch).
#include <torch/all.h>
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"
#include <torch/library.h>
#include <torch/version.h>

#ifdef ENABLE_SM70_TURBOMIND
void gguf_dmv_restore_iq2_sm70_out(torch::Tensor weight, torch::Tensor stats,
                                   torch::Tensor codes, torch::Tensor meta,
                                   torch::Tensor reverse, int64_t kind,
                                   int64_t k, int64_t n);
#endif

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
  ops.def("sm70_native_policy_abi() -> int",
          []() -> int64_t { return vllm::sm70::policy_size; });
  ops.def("sm70_prepare_native_policy_token(str token) -> ()",
          &vllm::sm70::prepare_native_policy);
  // vLLM custom ops
  //

#ifdef ENABLE_SM70_TURBOMIND
  ops.def(
      "gguf_canonical_linear_n64_sm70_out(Tensor(a!) output, Tensor input, "
      "Tensor weight, Tensor stats, Tensor(b!) partials, Tensor(c!) counters, "
      "int bits, int group_size) -> ()");
  ops.impl("gguf_canonical_linear_n64_sm70_out", torch::kCUDA,
           &gguf_canonical_linear_n64_sm70_out);
  ops.def(
      "gguf_native_linear_n64_sm70_out(Tensor(a!) out, Tensor input, Tensor "
      "weight, "
      "Tensor(b!) partials, Tensor(c!) counters, int source_type) -> ()");
  ops.impl("gguf_native_linear_n64_sm70_out", torch::kCUDA,
           &gguf_native_linear_n64_sm70_out);
  ops.def(
      "gguf_native_linear_sm70_out(Tensor(a!) out, Tensor input, Tensor "
      "weight, "
      "int source_type) -> ()");
  ops.def(
      "gguf_dmvq_sm70_out(Tensor(a!) output, Tensor input, Tensor weight, "
      "Tensor(b!) workspace, Tensor(c!) counters, Tensor table, int type, "
      "int kw, int split) -> ()");
  ops.impl("gguf_dmvq_sm70_out", torch::kCUDA, &gguf_dmvq_sm70_out);
  ops.def("gguf_dmvq_book_sm70_out(Tensor(a!) table, int type) -> ()");
  ops.impl("gguf_dmvq_book_sm70_out", torch::kCUDA, &gguf_dmvq_book_sm70_out);
  ops.def(
      "gguf_dmv_sm70_out(Tensor input, Tensor[] codes, Tensor[] high, "
      "Tensor[] scale, Tensor(a!)[] outputs, int[] formats, int[] widths, "
      "int k, int split, int kw, Tensor(b!) workspace, Tensor(c!) counters, "
      "int tn, Tensor? sigmoid_gate, Tensor? table, Tensor? floating_weight, "
      "Tensor(d!)? floating_output, Tensor(e!)? pair_output, bool gdn_heads) "
      "-> ()");
  ops.impl("gguf_dmv_sm70_out", torch::kCUDA, &gguf_dmv_sm70_out);
  ops.def("gguf_dmv_gdn_heads_sm70_supported(int k) -> bool",
          [](int64_t k) { return k == 1536 || k == 3072; });
  ops.def(
      "gguf_dmv_sm70_clocked_out(Tensor(f!) timestamps, Tensor input, Tensor[] "
      "codes, Tensor[] high, "
      "Tensor[] scale, Tensor(a!)[] outputs, int[] formats, int[] widths, "
      "int k, int split, int kw, Tensor(b!) workspace, Tensor(c!) counters, "
      "int tn, Tensor? sigmoid_gate, Tensor? table, Tensor? floating_weight, "
      "Tensor(d!)? floating_output, Tensor(e!)? pair_output, bool gdn_heads) "
      "-> ()");
  ops.impl("gguf_dmv_sm70_clocked_out", torch::kCUDA,
           &gguf_dmv_sm70_clocked_out);
  ops.def("gguf_dmv_three_formats_sm70_supported() -> bool",
          []() { return true; });
  ops.def(
      "sm70_hcx_out(Tensor p0, Tensor? p1, Tensor res, Tensor inj, Tensor nw,"
      " float eps, Tensor wd, Tensor wu, Tensor(a!) res_out, Tensor(b!) blk_o"
      "ut, Tensor(c!) inj_out, Tensor(d!) xn, Tensor(e!) sq, Tensor(f!) dpart"
      ", Tensor(g!) bar, Tensor(h!) seq, int[] ar, int[] lora, int[] hb, int "
      "rank, Tensor? dbg, int full, Tensor? ox, Tensor? ocodes, Tensor? ohigh"
      ", Tensor? oscale, int ofmt, Tensor? gz, Tensor? gw, float geps, Tensor"
      "(i!)? gscr, bool local_schedule=False) -> ()");
  ops.impl("sm70_hcx_out", torch::kCUDA, &sm70_hcx_out);
  ops.def(
      "sm70_dmv13_out(Tensor x, Tensor[] codes, Tensor[] high, Tensor[] scale, "
      "Tensor(a!)[] out, int[] fmt, int[] n, int K, int split, int warps, "
      "Tensor(b!) ws, Tensor(c!) cnt, int tp, Tensor? extra_weight, "
      "Tensor(d!)? extra_out) -> ()");
  ops.impl("sm70_dmv13_out", torch::kCUDA, &sm70_dmv13_out);
  ops.def(
      "qsa_prep_sm70_out(Tensor qkv, Tensor positions, Tensor cos_sin, "
      "Tensor q_norm_weight, Tensor k_norm_weight, float eps, Tensor(a!) "
      "query, "
      "Tensor(b!) key_cache, Tensor(c!) value_cache, Tensor slot_mapping) -> "
      "()");
  ops.impl("qsa_prep_sm70_out", torch::kCUDA, &qsa_prep_sm70_out);
  ops.def(
      "sm70_top1x_out(Tensor(a!) out, Tensor pairs, int[] buffers, "
      "Tensor(b!) seq, int rank) -> ()");
  ops.impl("sm70_top1x_out", torch::kCUDA, &sm70_top1x_out);
  ops.def(
      "sm70_gdn_verify_out(Tensor qkv, Tensor a, Tensor b, Tensor A_log, "
      "Tensor dt_bias, Tensor(a!) state, Tensor(b!) o, Tensor cu, Tensor idx, "
      "Tensor? nacc, int H, int HV, float scale, int cfg, Tensor? rbuf, "
      "Tensor? rn, Tensor? ctr, int tmax) -> ()");
  ops.impl("sm70_gdn_verify_out", torch::kCUDA, &sm70_gdn_verify_out);
  ops.def(
      "qsa_dense_decode_sm70_out(Tensor(a!) out, Tensor query, Tensor "
      "key_cache, "
      "Tensor value_cache, Tensor block_table, Tensor token_to_req, Tensor "
      "positions, "
      "Tensor? gate, Tensor(b!) ws_o, Tensor(c!) ws_ml, int num_requests, int "
      "splits, "
      "int warps) -> ()");
  ops.impl("qsa_dense_decode_sm70_out", torch::kCUDA,
           &qsa_dense_decode_sm70_out);
  ops.def(
      "gguf_moe_gate_up_sm70_out(Tensor(a!) hidden, Tensor input, Tensor ids, "
      "Tensor gate_codes, Tensor gate_scale, Tensor up_codes, Tensor up_scale, "
      "int format, Tensor table, int kw) -> ()");
  ops.impl("gguf_moe_gate_up_sm70_out", torch::kCUDA,
           &gguf_moe_gate_up_sm70_out);
  ops.def(
      "gguf_dmv_restore_iq2_sm70_out(Tensor(a!) weight, Tensor(b!) stats, "
      "Tensor codes, Tensor meta, Tensor reverse, int type, int k, int n) -> "
      "()");
  ops.impl("gguf_dmv_restore_iq2_sm70_out", torch::kCUDA,
           &gguf_dmv_restore_iq2_sm70_out);
  ops.def(
      "gguf_dmv_restore_sm70_out(Tensor(a!) weight, Tensor(b!) stats, "
      "Tensor codes, Tensor scale, int format, int k, int n) -> ()");
  ops.impl("gguf_dmv_restore_sm70_out", torch::kCUDA,
           &gguf_dmv_restore_sm70_out);
  ops.def(
      "gguf_qkvz_sm70_out(Tensor(a!) output, Tensor input, "
      "Tensor[] weights, Tensor[] stats, int[] types, "
      "Tensor(b!) partials, Tensor(c!) counters) -> ()");
  ops.impl("gguf_qkvz_sm70_out", torch::kCUDA, &gguf_qkvz_sm70_out);
  ops.def(
      "gguf_qkv_sm70_out(Tensor(a!) output, Tensor input, "
      "Tensor[] weights, Tensor[] stats, int[] types, "
      "Tensor(b!) partials, Tensor(c!) counters) -> ()");
  ops.impl("gguf_qkv_sm70_out", torch::kCUDA, &gguf_qkv_sm70_out);
  ops.def(
      "gguf_small_output_sm70_out(Tensor(a!) output, Tensor input, Tensor "
      "weight, "
      "Tensor(b!) partials, Tensor(c!) counters, int type, int splits, bool "
      "gdn_head_tiling) -> ()");
  ops.impl("gguf_small_output_sm70_out", torch::kCUDA,
           &gguf_small_output_sm70_out);
  ops.impl("gguf_native_linear_sm70_out", torch::kCUDA,
           &gguf_native_linear_sm70_out);
  ops.def(
      "gguf_native_pair_sm70_out(Tensor(a!) out, Tensor input, Tensor gate, "
      "Tensor up, int gate_type, int up_type) -> ()");
  ops.impl("gguf_native_pair_sm70_out", torch::kCUDA,
           &gguf_native_pair_sm70_out);
  ops.def(
      "gguf_iq3_gated_sm70_out(Tensor(a!) out, Tensor input, Tensor gate, "
      "Tensor up) -> ()");
  ops.impl("gguf_iq3_gated_sm70_out", torch::kCUDA, &gguf_iq3_gated_sm70_out);
#endif
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

  ops.def(
      "ple_disk_gather_u8(Tensor ids, Tensor pointers, int shard_size, "
      "int num_rows, int row_bytes, Tensor(a!) out) -> ()");
  ops.impl("ple_disk_gather_u8", torch::kCPU, &ple_disk_gather_u8);

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
      ") -> str?");
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
  ops.def(
      "gguf_affine_sm70_prepare(Tensor codes, Tensor scales, Tensor mins, "
      "int bits, int group_size=32) -> Tensor[]");
  ops.impl("gguf_affine_sm70_prepare", torch::kCUDA, &gguf_affine_sm70_prepare);
  ops.def(
      "gguf_affine_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor weight, "
      "Tensor stats, int bits, int k_ld, int q_ld, int group_size=32, str? "
      "native_policy=None) -> ()");
  ops.impl("gguf_affine_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&gguf_affine_gemm_sm70_out));
  ops.def(
      "gguf_affine_grouped_gemm_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor offsets, Tensor weight_ptrs, Tensor stats_ptrs, "
      "int bits, int num_experts, int group_size=32, str? "
      "native_policy=None) "
      "-> ()");
  ops.impl("gguf_affine_grouped_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&gguf_affine_grouped_gemm_sm70_out));
  ops.def(
      "gguf_affine_dequantize_sm70_out(Tensor(a!) out, Tensor weight, "
      "Tensor stats, int bits, int group_size) -> ()");
  ops.impl("gguf_affine_dequantize_sm70_out", torch::kCUDA,
           &gguf_affine_dequantize_sm70_out);
  ops.def(
      "gguf_affine_blas_sm70_out(Tensor(a!) out, Tensor input, Tensor weight, "
      "Tensor stats, int bits, Tensor(b!) scratch, int group_size) -> ()");
  ops.impl("gguf_affine_blas_sm70_out", torch::kCUDA,
           &gguf_affine_blas_sm70_out);
  ops.def(
      "gguf_small_grouped_vec_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor offsets, Tensor weight_ptrs, Tensor stats_ptrs, int source_type, "
      "int num_experts, int group_size) -> ()");
  ops.impl("gguf_small_grouped_vec_sm70_out", torch::kCUDA,
           &gguf_small_grouped_vec_sm70_out);
  ops.def(
      "gguf_dp4a_down_unroute_sm70_out(Tensor(a!) out, Tensor input, Tensor "
      "ids, Tensor route_weights, Tensor weight_ptrs, Tensor stats_ptrs, int "
      "source_type, int num_experts) -> ()");
  ops.impl("gguf_dp4a_down_unroute_sm70_out", torch::kCUDA,
           &gguf_dp4a_down_unroute_sm70_out);
  ops.def(
      "gguf_dense_restore_canonical_sm70_out(Tensor(a!) weight, Tensor(b!) "
      "stats, "
      "Tensor codes, Tensor high, Tensor scale, int fmt, int k, int n) -> ()");
  ops.impl("gguf_dense_restore_canonical_sm70_out", torch::kCUDA,
           &gguf_dense_restore_canonical_sm70_out);
  ops.def(
      "gguf_dense_segments_sm70_out(Tensor x, Tensor[] codes, Tensor[] high, "
      "Tensor[] scale, Tensor(a!)[] out, int[] fmt, int[] n, int k, int split, "
      "int warps, Tensor(b!) ws, Tensor(c!) cnt, Tensor? sgate=None) -> ()");
  ops.impl("gguf_dense_segments_sm70_out", torch::kCUDA,
           &gguf_dense_segments_sm70_out);
  ops.def(
      "gguf_shared_gate_up_sm70_out(Tensor x, Tensor[] gate, Tensor[] up, "
      "int[] fmts, Tensor wg, Tensor(a!) h, Tensor(b!) sg, Tensor(c!) ws, "
      "Tensor(d!) cnt, int split, int warps) -> ()");
  ops.impl("gguf_shared_gate_up_sm70_out", torch::kCUDA,
           &gguf_shared_gate_up_sm70_out);

  ops.def(
      "sm70_hc_ll_down_out(Tensor x, Tensor wd, Tensor(a!) part, Tensor(b!) "
      "cnt, "
      "int[] ll, Tensor(c!) seq, int rank, int variant, bool "
      "optimized_loads=True) -> ()");
  ops.impl("sm70_hc_ll_down_out", torch::kCUDA, &sm70_hc_ll_down_out);
  ops.def(
      "sm70_hc_ll_up_out(int ll_lora, Tensor wu, Tensor x, Tensor(a!) cnt, "
      "int[] ll, Tensor(b!) seq, Tensor down_seq, int rank, Tensor(c!) out, "
      "Tensor(d!) lora_out, Tensor(e!) inj_out, int warps, bool "
      "optimized_loads=True) -> ()");
  ops.impl("sm70_hc_ll_up_out", torch::kCUDA, &sm70_hc_ll_up_out);

  ops.def("gguf_quantize_q8_1_sm70_out(Tensor(a!) out, Tensor input) -> ()");
  ops.impl("gguf_quantize_q8_1_sm70_out", torch::kCUDA,
           &gguf_quantize_q8_1_sm70_out);
  ops.def(
      "gguf_dp4a_scalar_lut_gate_up_sm70_out(Tensor(a!) out, Tensor "
      "activation, "
      "Tensor ids, Tensor gate, Tensor up, int source_type, bool activated) -> "
      "()");
  ops.impl("gguf_dp4a_scalar_lut_gate_up_sm70_out", torch::kCUDA,
           &gguf_dp4a_scalar_lut_gate_up_sm70_out);
  ops.def(
      "gguf_dp4a_gate_up_sm70_out(Tensor(a!) out, Tensor activation, Tensor "
      "ids, Tensor gate, Tensor up, int source_type, bool activated, "
      "int lanes_per_row=16, bool bank_aware=False) -> ()");
  ops.impl("gguf_dp4a_gate_up_sm70_out", torch::kCUDA,
           &gguf_dp4a_gate_up_sm70_out);
  ops.def(
      "gguf_dp4a_lut4_gate_up_sm70_out(Tensor(a!) out, Tensor activation, "
      "Tensor ids, Tensor gate, Tensor gate_stats, Tensor up, Tensor up_stats, "
      "int num_experts, int lanes_per_row=16) -> ()");
  ops.impl("gguf_dp4a_lut4_gate_up_sm70_out", torch::kCUDA,
           &gguf_dp4a_lut4_gate_up_sm70_out);
  ops.def(
      "gguf_lattice_raw_dequantize_sm70_out(Tensor(a!) out, Tensor weight, int "
      "source_type) -> ()");
  ops.impl("gguf_lattice_raw_dequantize_sm70_out", torch::kCUDA,
           &gguf_lattice_raw_dequantize_sm70_out);
  ops.def(
      "gguf_lattice_raw_grouped_gate_up_sm70_out(Tensor(a!) gate, Tensor(b!) "
      "up, Tensor input, Tensor gate_weights, Tensor up_weights, Tensor "
      "offsets, "
      "Tensor ids, int source_type, int top_k) -> ()");
  ops.impl("gguf_lattice_raw_grouped_gate_up_sm70_out", torch::kCUDA,
           &gguf_lattice_raw_grouped_gate_up_sm70_out);
  ops.def(
      "gguf_lattice_dequantize_sm70_out(Tensor(a!) out, Tensor weight, "
      "Tensor stats, int source_type, int group_size) -> ()");
  ops.impl("gguf_lattice_dequantize_sm70_out", torch::kCUDA,
           &gguf_lattice_dequantize_sm70_out);
  ops.def(
      "gguf_lattice_blas_sm70_out(Tensor(a!) out, Tensor input, Tensor weight, "
      "Tensor stats, int source_type, Tensor(b!) scratch, int group_size) -> "
      "()");
  ops.impl("gguf_lattice_blas_sm70_out", torch::kCUDA,
           &gguf_lattice_blas_sm70_out);
  ops.def(
      "gguf_lut4_sm70_prepare(Tensor codes, Tensor scales, int lut_id, "
      "int group_size) -> Tensor[]");
  ops.impl("gguf_lut4_sm70_prepare", torch::kCUDA, &gguf_lut4_sm70_prepare);
  ops.def(
      "gguf_lut4_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor weight, "
      "Tensor stats, int lut_id, int k_ld, int q_ld, int group_size, str? "
      "native_policy=None) -> ()");
  ops.impl("gguf_lut4_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&gguf_lut4_gemm_sm70_out));
  ops.def(
      "gguf_lut4_grouped_gemm_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor offsets, Tensor weight_ptrs, Tensor stats_ptrs, int lut_id, "
      "int num_experts, int group_size, str? native_policy=None) -> ()");
  ops.impl("gguf_lut4_grouped_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&gguf_lut4_grouped_gemm_sm70_out));
  ops.def(
      "gguf_lattice_sm70_prepare(Tensor codes, Tensor scales, int source_type, "
      "int group_size) -> Tensor[]");
  ops.impl("gguf_lattice_sm70_prepare", torch::kCUDA,
           &gguf_lattice_sm70_prepare);
  ops.def(
      "gguf_lattice_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor weight, "
      "Tensor stats, int source_type, int k_ld, int q_ld, int group_size, "
      "str? native_policy=None) -> "
      "()");
  ops.impl("gguf_lattice_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&gguf_lattice_gemm_sm70_out));
  ops.def(
      "gguf_lattice_grouped_gemm_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor offsets, Tensor weight_ptrs, Tensor stats_ptrs, int source_type, "
      "int num_experts, int group_size, str? native_policy=None) -> ()");
  ops.impl("gguf_lattice_grouped_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&gguf_lattice_grouped_gemm_sm70_out));
  ops.def(
      "gguf_lattice_grouped_vec_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor offsets, Tensor weight_ptrs, Tensor stats_ptrs, int source_type, "
      "int num_experts, int group_size) -> ()");
  ops.impl("gguf_lattice_grouped_vec_sm70_out", torch::kCUDA,
           &gguf_lattice_grouped_vec_sm70_out);
  ops.def("silu_and_mul_interleaved(Tensor! result, Tensor input) -> ()");
  ops.impl("silu_and_mul_interleaved", torch::kCUDA, &silu_and_mul_interleaved);

  ops.def(
      "awq_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, Tensor "
      "_zeros, "
      "int group_size, bool interleave_gated_silu, str? native_policy=None ) "
      "-> "
      "Tensor[]");
  ops.impl("awq_sm70_prepare", torch::kCUDA,
           vllm::sm70::with_policy(&awq_sm70_prepare));

  ops.def(
      "awq_sm70_prepare_compact(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, int group_size, bool interleave_gated_silu, str? "
      "native_policy=None ) -> "
      "Tensor[]");
  ops.impl("awq_sm70_prepare_compact", torch::kCUDA,
           vllm::sm70::with_policy(&awq_sm70_prepare_compact));

  ops.def(
      "awq_sm70_dequantize_out(Tensor(a!) out, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, str? native_policy=None ) -> "
      "()");
  ops.impl("awq_sm70_dequantize_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_sm70_dequantize_out));

  ops.def(
      "uint4_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, int group_size, bool interleave_gated_silu, str? "
      "native_policy=None ) -> Tensor[]");
  ops.impl("uint4_sm70_prepare", torch::kCUDA,
           vllm::sm70::with_policy(&uint4_sm70_prepare));

  ops.def(
      "fp8_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, "
      "int group_size, bool interleave_gated_silu, str? native_policy=None ) "
      "-> "
      "Tensor[]");
  ops.impl("fp8_sm70_prepare", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_sm70_prepare));

  ops.def(
      "fp8_sm70_dequantize_out(Tensor(a!) out, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, str? native_policy=None ) -> "
      "()");
  ops.impl("fp8_sm70_dequantize_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_sm70_dequantize_out));

  ops.def(
      "mxfp4_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, "
      "int group_size, bool interleave_gated_silu, str? native_policy=None ) "
      "-> "
      "Tensor[]");
  ops.impl("mxfp4_sm70_prepare", torch::kCUDA,
           vllm::sm70::with_policy(&mxfp4_sm70_prepare));

  ops.def(
      "nvfp4_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, "
      "int group_size, bool interleave_gated_silu, str? native_policy=None ) "
      "-> "
      "Tensor[]");
  ops.impl("nvfp4_sm70_prepare", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_sm70_prepare));

  ops.def(
      "sm70_f16_prepare(Tensor _kernel, str? native_policy=None ) -> "
      "Tensor[]");
  ops.impl("sm70_f16_prepare", torch::kCUDA,
           vllm::sm70::with_policy(&sm70_f16_prepare));

  ops.def(
      "sm70_glm53_tp8_cublaslt_out("
      "Tensor(a!) out, Tensor input, Tensor weight, str? native_policy=None "
      ") "
      "-> ()");
  ops.impl("sm70_glm53_tp8_cublaslt_out", torch::kCUDA,
           vllm::sm70::with_policy(&sm70_glm53_tp8_cublaslt_out));

  ops.def(
      "awq_gemm_sm70(Tensor _in_feats, Tensor _kernel, Tensor "
      "_scaling_factors, int group_size, int k_ld, int q_ld, str? "
      "native_policy=None ) -> Tensor");
  ops.impl("awq_gemm_sm70", torch::kCUDA,
           vllm::sm70::with_policy(&awq_gemm_sm70));

  ops.def(
      "sm70_f16_gemm(Tensor _in_feats, Tensor _kernel, str? "
      "native_policy=None "
      ") -> Tensor");
  ops.impl("sm70_f16_gemm", torch::kCUDA,
           vllm::sm70::with_policy(&sm70_f16_gemm));

  ops.def(
      "awq_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu, str? native_policy=None ) -> ()");
  ops.impl("awq_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_gemm_sm70_out));

  ops.def(
      "awq_gemm_sm70_out_tile_reduce(Tensor(a!) out, Tensor(b!) staging, "
      "Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, "
      "int group_size, int k_ld, int q_ld, int fa_ptr, int tile_numel, "
      "int reducer_blocks, int kernel_reducer_blocks, bool overlap, str? "
      "native_policy=None ) -> ()");
  ops.impl("awq_gemm_sm70_out_tile_reduce", torch::kCUDA,
           vllm::sm70::with_policy(&awq_gemm_sm70_out_tile_reduce));

  ops.def(
      "fp8_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu, bool preserve_default_partition=False, str? "
      "native_policy=None ) -> ()");
  ops.impl("fp8_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_gemm_sm70_out));

  ops.def(
      "sm70_dflash2_fp16_m8_out(Tensor(a!) output, Tensor input, Tensor "
      "packed, int tile, int warps) -> ()");
  ops.impl("sm70_dflash2_fp16_m8_out", torch::kCUDA, &sm70_dflash2_fp16_m8_out);
  ops.def(
      "sm70_dflash2_fp16_dispatch_out(Tensor(a!) output, Tensor input, Tensor "
      "packed, Tensor weight, int tile, int warps) -> ()");
  ops.impl("sm70_dflash2_fp16_dispatch_out", torch::kCUDA,
           &sm70_dflash2_fp16_dispatch_out);

  ops.def(
      "fp8_qpn8_prepare_sm70(Tensor qweight, Tensor scales, str? "
      "native_policy=None ) -> Tensor[]");
  ops.impl("fp8_qpn8_prepare_sm70", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_qpn8_prepare_sm70));

  ops.def(
      "fp8_qpn8_dequantize_sm70_out(Tensor(a!) out, Tensor codes, "
      "Tensor group_scales, str? native_policy=None ) -> ()");
  ops.impl("fp8_qpn8_dequantize_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_qpn8_dequantize_sm70_out));

  ops.def(
      "fp8_qpn8_prefill_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor group_scales, bool gated_silu, "
      "str? "
      "native_policy=None ) -> "
      "()");
  ops.impl("fp8_qpn8_prefill_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_qpn8_prefill_sm70_out));

  ops.def(
      "fp8_qpn8_dispatch_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor group_scales, int split_k, "
      "int accumulator_chains, bool prefetch_codes, bool gated_silu, str? "
      "native_policy=None ) -> ()");
  ops.impl("fp8_qpn8_dispatch_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_qpn8_dispatch_sm70_out));

  ops.def(
      "fp8_qpn8_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor group_scales, int split_k, int accumulator_chains, "
      "bool fast_decoder, bool prefetch_codes, str? native_policy=None ) -> "
      "()");
  ops.impl("fp8_qpn8_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_qpn8_gemm_sm70_out));

  ops.def(
      "fp8_qpn8_gemm_ba_split_sm70_out(Tensor(a!) qkv_out, Tensor(b!) "
      "z_out, Tensor(c!) b_out, Tensor(d!) a_out, Tensor input, Tensor codes, "
      "Tensor group_scales, Tensor ba_weight, str? native_policy=None ) -> "
      "()");
  ops.impl("fp8_qpn8_gemm_ba_split_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_qpn8_gemm_ba_split_sm70_out));

  ops.def(
      "fp8_qpn8_dispatch_ba_split_sm70_out(Tensor(a!) qkv_out, Tensor(b!) "
      "z_out, Tensor(c!) b_out, Tensor(d!) a_out, Tensor(e!) qkvz_staging, "
      "Tensor(f!) ba_staging, int dense_weight_ptr, Tensor input, Tensor "
      "codes, Tensor group_scales, Tensor ba_weight, str? native_policy=None "
      ") "
      "-> ()");
  ops.impl("fp8_qpn8_dispatch_ba_split_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_qpn8_dispatch_ba_split_sm70_out));

  ops.def(
      "fp8_qpn8_gated_pair_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor group_scales, int split_k, "
      "int accumulator_chains, bool fast_decoder, bool prefetch_codes, str? "
      "native_policy=None ) -> ()");
  ops.impl("fp8_qpn8_gated_pair_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_qpn8_gated_pair_sm70_out));

  ops.def(
      "fp8_qpn8_hc_dispatch_sm70_out(Tensor(a!) block_out, Tensor(b!) "
      "injection_out, Tensor(c!) down_staging, Tensor(d!) lora_staging, "
      "Tensor(e!) gate_staging, Tensor(f!) partials, int dense_weight_ptr, "
      "Tensor xn, Tensor down_codes, Tensor down_scales, Tensor up_codes, "
      "Tensor up_scales, str? native_policy=None ) -> ()");
  ops.impl("fp8_qpn8_hc_dispatch_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_qpn8_hc_dispatch_sm70_out));

  ops.def(
      "nvfp4_qpn4_prepare_sm70(Tensor qweight, Tensor scales, str? "
      "native_policy=None ) -> Tensor[]");
  ops.impl("nvfp4_qpn4_prepare_sm70", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn4_prepare_sm70));

  ops.def(
      "nvfp4_qpn4_prepare_scale_code_sm70(Tensor qweight, Tensor "
      "scale_codes, str? native_policy=None ) -> Tensor[]");
  ops.impl("nvfp4_qpn4_prepare_scale_code_sm70", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn4_prepare_scale_code_sm70));

  ops.def(
      "nvfp4_qpn4_dequantize_sm70_out(Tensor(a!) out, Tensor codes, Tensor "
      "scales, float global_scale, bool use_scale_code, str? "
      "native_policy=None "
      ") -> ()");
  ops.impl("nvfp4_qpn4_dequantize_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn4_dequantize_sm70_out));

  ops.def(
      "nvfp4_qpn4_prefill_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor scales, float global_scale, bool "
      "use_scale_code, bool gated_silu, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_qpn4_prefill_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn4_prefill_sm70_out));

  ops.def(
      "nvfp4_qpn4_dispatch_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor scales, float global_scale, bool "
      "use_scale_code, bool gated_silu, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_qpn4_dispatch_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn4_dispatch_sm70_out));

  ops.def(
      "fp8_gemm_sm70_prefill_prescaled_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _prescaled_factors, int group_size, int k_ld, "
      "int q_ld, str? native_policy=None ) -> ()");
  ops.impl("fp8_gemm_sm70_prefill_prescaled_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_gemm_sm70_prefill_prescaled_out));

  // A distinct schema is also a capability marker: older extensions expose
  // only the 8K-prefill contract and must not receive the new M=1 shapes.
  ops.def(
      "fp8_gemm_sm70_prescaled_m1_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _prescaled_factors, int group_size, int k_ld, "
      "int q_ld, str? native_policy=None ) -> ()");
  ops.impl("fp8_gemm_sm70_prescaled_m1_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_gemm_sm70_prescaled_m1_out));

  ops.def(
      "nvfp4_qpn2_prepare_sm70(Tensor weight_packed, Tensor weight_scale, "
      "str? native_policy=None ) -> "
      "Tensor[]");
  ops.impl("nvfp4_qpn2_prepare_sm70", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn2_prepare_sm70));
  ops.def(
      "nvfp4_qpn2_bundle_sm70(Tensor codes, Tensor scales, str? "
      "native_policy=None ) -> Tensor[]");
  ops.impl("nvfp4_qpn2_bundle_sm70", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn2_bundle_sm70));

  ops.def(
      "nvfp4_qpn2_prepare_scales_sm70(Tensor weight_scale, str? "
      "native_policy=None ) -> Tensor");
  ops.def(
      "nvfp4_qpn2_restore_tm_scales_sm70_out(Tensor(a!) out, "
      "Tensor scales, float global_scale) -> ()");
  ops.impl("nvfp4_qpn2_restore_tm_scales_sm70_out", torch::kCUDA,
           &nvfp4_qpn2_restore_tm_scales_sm70_out);
  // Python may be newer than the loaded extension. Only this version retains
  // compact-scale scratch across layers in larger CUDA graphs.
  ops.def("nvfp4_qpn2_compact_scales_version_sm70() -> int",
          []() -> int64_t { return 1; });
  ops.def(
      "nvfp4_qpn2_compact_tm_gemm_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor weight, Tensor scales, float global_scale, int k_ld, "
      "int q_ld, bool gated_silu, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_qpn2_compact_tm_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn2_compact_tm_gemm_sm70_out));
  ops.impl("nvfp4_qpn2_prepare_scales_sm70", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn2_prepare_scales_sm70));
  ops.def(
      "nvfp4_qpn2_tm_dispatch_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor tm_weight, Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains, Tensor tm_scales, int tm_group_size, "
      "int tm_k_ld, int tm_q_ld, bool gated_silu, int min_prefill_m, "
      "bool prescaled_scales=False, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_qpn2_tm_dispatch_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn2_tm_dispatch_sm70_out));

  ops.def(
      "nvfp4_qpn2_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_qpn2_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn2_gemm_sm70_out));

  // Skinny QPN GEMM and grouped NVFP4/MXFP4 MoE (SM70/SM75), weights
  // prepacked in mma.m8n8k4 fragment order.
  ops.def(
      "skinny_qpn_gemm_sm70(Tensor x, Tensor qcodes, Tensor qscales, "
      "float gscale, int n) -> Tensor");
  ops.impl("skinny_qpn_gemm_sm70", torch::kCUDA, &skinny_qpn_gemm_sm70);

  ops.def(
      "skinny_moe_qpn_sm70(Tensor x, Tensor qcodes, Tensor qscales, "
      "Tensor gscales, Tensor perm, Tensor gids, Tensor goff, int topk, "
      "Tensor(a!) y_slots, bool x_slot_major, int num_tokens, int splitk, "
      "int nacc, int scale_mode) -> ()");
  ops.impl("skinny_moe_qpn_sm70", torch::kCUDA, &skinny_moe_qpn_sm70);

  ops.def(
      "nvfp4_qpn2_gated_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_qpn2_gated_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn2_gated_sm70_out));

  ops.def(
      "nvfp4_qpn2_dispatch_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains, Tensor tm_weight, Tensor tm_scales, "
      "int tm_group_size, int tm_k_ld, int tm_q_ld, bool gated_silu, str? "
      "native_policy=None ) -> ()");
  ops.impl("nvfp4_qpn2_dispatch_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn2_dispatch_sm70_out));

  ops.def(
      "nvfp4_qpn2_prefill_dispatch_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains, Tensor tm_weight, Tensor tm_scales, "
      "int tm_group_size, int tm_k_ld, int tm_q_ld, bool gated_silu, "
      "int min_prefill_m, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_qpn2_prefill_dispatch_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qpn2_prefill_dispatch_sm70_out));

  ops.def(
      "fp8_gemm_sm70_prefill_dispatch_out(Tensor(a!) out, "
      "int dense_weight_ptr, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu, int min_prefill_m, str? native_policy=None ) -> ()");
  ops.impl("fp8_gemm_sm70_prefill_dispatch_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_gemm_sm70_prefill_dispatch_out));

  ops.def(
      "mxfp4_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu, str? native_policy=None ) -> ()");
  ops.impl("mxfp4_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&mxfp4_gemm_sm70_out));

  ops.def(
      "nvfp4_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_gemm_sm70_out));

  ops.def(
      "nvfp4_gemm_sm70_prescaled_out(Tensor(a!) out, Tensor input, "
      "Tensor weight, Tensor scales, int group_size, int k_ld, int q_ld, "
      "bool gated_silu, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_gemm_sm70_prescaled_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_gemm_sm70_prescaled_out));

  ops.def(
      "nvfp4_gemv_sm70_raw_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _scaling_factors, Tensor(b!) partials, "
      "int group_size, int split_k, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_gemv_sm70_raw_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_gemv_sm70_raw_out));

  ops.def(
      "nvfp4_gemv_sm70_warp_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _scaling_factors, int group_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("nvfp4_gemv_sm70_warp_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_gemv_sm70_warp_out));

  ops.def(
      "nvfp4_gemv_sm70_h2_out(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _scaling_factors, Tensor(b!) partials, "
      "int group_size, int split_k, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_gemv_sm70_h2_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_gemv_sm70_h2_out));

  ops.def(
      "fp8_gemm_sm70_out_auto(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _scaling_factors, str? native_policy=None ) -> "
      "()");
  ops.impl("fp8_gemm_sm70_out_auto", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_gemm_sm70_out_auto));

  ops.def(
      "fp8_gemm_sm70_out_meta(Tensor(a!) out, Tensor _in_feats, "
      "Tensor _kernel, Tensor _scaling_factors, Tensor _meta, "
      "bool gated_silu, str? native_policy=None ) -> ()");
  ops.impl("fp8_gemm_sm70_out_meta", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_gemm_sm70_out_meta));

  ops.def(
      "sm70_f16_gemm_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "int k_ld, bool gated_silu, str? native_policy=None ) -> ()");
  ops.impl("sm70_f16_gemm_out", torch::kCUDA,
           vllm::sm70::with_policy(&sm70_f16_gemm_out));

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
      "weight, str? native_policy=None ) -> ()");
  ops.impl("sm70_glm53_fp16_gemv_out", torch::kCUDA,
           vllm::sm70::with_policy(&sm70_glm53_fp16_gemv_out));

  ops.def(
      "sm70_glm53_moe_permute_q8_out("
      "Tensor input, Tensor topk_ids, Tensor(a!) permuted_input, "
      "Tensor(b!) sorted_row_idx, Tensor(c!) inv_permuted_idx, "
      "Tensor(d!) compact_offsets, Tensor(e!) active_expert_ids, str? "
      "native_policy=None ) -> ()");
  ops.impl("sm70_glm53_moe_permute_q8_out", torch::kCUDA,
           vllm::sm70::with_policy(&sm70_glm53_moe_permute_q8_out));

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
  ops.def(
      "qwen38_shared_gate_sigmoid_mul_out(Tensor(a!) out, Tensor logits) -> "
      "()");
  ops.impl("qwen38_shared_gate_sigmoid_mul_out", torch::kCUDA,
           &qwen38_shared_gate_sigmoid_mul_out);

  ops.def(
      "sm70_gemm_import_cache(Tensor device_hint, str path, str? "
      "native_policy=None ) -> int");
  ops.impl("sm70_gemm_import_cache", torch::kCUDA,
           vllm::sm70::with_policy(&sm70_gemm_import_cache));

  ops.def(
      "sm70_gemm_export_cache(Tensor device_hint, str path, str? "
      "native_policy=None ) -> int");
  ops.impl("sm70_gemm_export_cache", torch::kCUDA,
           vllm::sm70::with_policy(&sm70_gemm_export_cache));

  ops.def(
      "awq_moe_build_strided_ptrs(Tensor tm_weights, Tensor tm_scales, "
      "int k_ld, int q_ld, int num_experts, str? native_policy=None ) -> "
      "Tensor[]");
  ops.impl("awq_moe_build_strided_ptrs", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_build_strided_ptrs));

  ops.def(
      "awq_moe_gemm_sm70_out(Tensor(a!) out, Tensor sorted_input, "
      "Tensor expert_offsets, Tensor strided_ptrs_w, Tensor strided_ptrs_s, "
      "int num_experts, int k, int n, int group_size, bool gated_silu, str? "
      "native_policy=None ) -> ()");
  ops.impl("awq_moe_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_gemm_sm70_out));

  ops.def(
      "awq_moe_gemm_sm70_per_expert_dispatch_out("
      "Tensor(a!) out, Tensor sorted_input, Tensor expert_offsets, "
      "Tensor strided_ptrs_w, Tensor strided_ptrs_s, int num_experts, "
      "int k, int n, int group_size, bool gated_silu, str? "
      "native_policy=None ) "
      "-> ()");
  ops.impl("awq_moe_gemm_sm70_per_expert_dispatch_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_gemm_sm70_per_expert_dispatch_out));

  ops.def(
      "awq_moe_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor dense_expert_ids, Tensor ptrs_w, Tensor ptrs_s, "
      "int num_experts, int k, int n, int group_size, str? "
      "native_policy=None ) "
      "-> ()");
  ops.impl("awq_moe_dense_stage_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_dense_stage_sm70_out));

  ops.def(
      "awq_moe_indexed_dense_w13_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor input_row_indices, "
      "Tensor expert_offsets, Tensor dense_expert_ids, Tensor ptrs_w, "
      "Tensor ptrs_s, int num_experts, int k, int n, int group_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("awq_moe_indexed_dense_w13_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_indexed_dense_w13_sm70_out));

  ops.def(
      "awq_moe_active_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor permuted_experts_id, "
      "Tensor(b!) active_expert_offsets, Tensor(c!) active_expert_ids, Tensor "
      "ptrs_w, "
      "Tensor ptrs_s, int total_slots, int k, int n, int group_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("awq_moe_active_dense_stage_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_active_dense_stage_sm70_out));

  ops.def(
      "awq_moe_chunked_w2_sm70_out("
      "Tensor(a!) out, Tensor(b!) chunk_output, Tensor input, "
      "Tensor expert_offsets, Tensor permuted_idx, Tensor topk_weights, "
      "Tensor(c!) chunk_expert_offsets, Tensor(d!) chunk_range_begin, "
      "Tensor(e!) chunk_range_end, Tensor(f!) chunk_a_indices, "
      "Tensor(g!) chunk_inv_permuted_idx, Tensor ptrs_w, Tensor ptrs_s, "
      "int num_tokens, int top_k, int num_experts, int k, int n, "
      "int hidden_logical_size, int group_size, int chunk_tokens, str? "
      "native_policy=None ) -> ()");
  ops.impl("awq_moe_chunked_w2_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_chunked_w2_sm70_out));

  ops.def(
      "awq_moe_single_token_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor sorted_expert_ids, Tensor ptrs_w, Tensor ptrs_s, int top_k, "
      "int k, int n, int group_size, str? native_policy=None ) -> ()");
  ops.impl("awq_moe_single_token_dense_stage_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_single_token_dense_stage_sm70_out));

  ops.def(
      "awq_moe_single_token_indexed_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor sorted_expert_ids, Tensor ptrs_w, Tensor ptrs_s, int top_k, "
      "int k, int n, int group_size, str? native_policy=None ) -> ()");
  ops.impl("awq_moe_single_token_indexed_dense_stage_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(
               &awq_moe_single_token_indexed_dense_stage_sm70_out));

  ops.def(
      "awq_moe_single_token_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) expert_offsets, Tensor(d!) expert_offsets64, "
      "Tensor(e!) inv_permuted_idx, Tensor(f!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("awq_moe_single_token_dense_w13_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_single_token_dense_w13_sm70_out));

  ops.def(
      "awq_moe_single_token_indexed_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) expert_offsets, Tensor(d!) expert_offsets64, "
      "Tensor(e!) inv_permuted_idx, Tensor(f!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("awq_moe_single_token_indexed_dense_w13_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(
               &awq_moe_single_token_indexed_dense_w13_sm70_out));

  ops.def(
      "awq_moe_single_token_compact_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) compact_w13_ptrs_w, Tensor(d!) compact_w13_ptrs_s, "
      "Tensor(e!) expert_offsets, Tensor(f!) expert_offsets64, "
      "Tensor(g!) inv_permuted_idx, Tensor(h!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("awq_moe_single_token_compact_dense_w13_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(
               &awq_moe_single_token_compact_dense_w13_sm70_out));

  ops.def(
      "awq_moe_single_token_exact_layout_prepare("
      "Tensor topk_ids, Tensor x, Tensor(a!) compact_input, "
      "Tensor(b!) expert_offsets, Tensor(c!) expert_offsets64, "
      "Tensor(d!) inv_permuted_idx, int num_experts, str? native_policy=None "
      ") "
      "-> ()");
  ops.impl("awq_moe_single_token_exact_layout_prepare", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_single_token_exact_layout_prepare));

  ops.def(
      "awq_moe_single_token_weighted_reduce_out("
      "Tensor sorted_output, Tensor topk_weights, Tensor inv_permuted_idx, "
      "Tensor(a!) out, int top_k, int hidden_logical_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("awq_moe_single_token_weighted_reduce_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_single_token_weighted_reduce_out));

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
      "int hidden_logical_size, str? native_policy=None ) -> ()");
  ops.impl("awq_moe_single_token_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_single_token_sm70_out));

  ops.def(
      "awq_moe_qpn_m1_sm70_out(Tensor(a!) out, Tensor(b!) intermediate, "
      "Tensor input, Tensor w13, Tensor s13, Tensor w2, Tensor s2, "
      "Tensor ids, Tensor topk, str? native_policy=None ) -> ()");
  ops.impl("awq_moe_qpn_m1_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&awq_moe_qpn_m1_sm70_out));

  ops.def(
      "fp8_moe_gemm_sm70_out(Tensor(a!) out, Tensor sorted_input, "
      "Tensor expert_offsets, Tensor strided_ptrs_w, Tensor strided_ptrs_s, "
      "int num_experts, int k, int n, int group_size, bool gated_silu, str? "
      "native_policy=None ) -> ()");
  ops.impl("fp8_moe_gemm_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_moe_gemm_sm70_out));

  ops.def(
      "fp8_moe_gemm_sm70_per_expert_dispatch_out("
      "Tensor(a!) out, Tensor sorted_input, Tensor expert_offsets, "
      "Tensor strided_ptrs_w, Tensor strided_ptrs_s, int num_experts, "
      "int k, int n, int group_size, bool gated_silu, str? "
      "native_policy=None ) "
      "-> ()");
  ops.impl("fp8_moe_gemm_sm70_per_expert_dispatch_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_moe_gemm_sm70_per_expert_dispatch_out));

  ops.def(
      "fp8_moe_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor dense_expert_ids, Tensor ptrs_w, Tensor ptrs_s, "
      "int num_experts, int k, int n, int group_size, str? "
      "native_policy=None ) "
      "-> ()");
  ops.impl("fp8_moe_dense_stage_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_moe_dense_stage_sm70_out));

  ops.def(
      "mxfp4_moe_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor dense_expert_ids, Tensor ptrs_w, Tensor ptrs_s, "
      "int num_experts, int k, int n, int group_size, str? "
      "native_policy=None ) "
      "-> ()");
  ops.impl("mxfp4_moe_dense_stage_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&mxfp4_moe_dense_stage_sm70_out));

  ops.def(
      "mxfp4_moe_qpn_m1_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, bool broadcast_input, str? native_policy=None ) -> "
      "()");
  ops.impl("mxfp4_moe_qpn_m1_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&mxfp4_moe_qpn_m1_sm70_out));

  ops.def(
      "nvfp4_moe_qpn_m1_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, bool broadcast_input, int split_k, str? "
      "native_policy=None ) -> ()");
  ops.impl("nvfp4_moe_qpn_m1_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_moe_qpn_m1_sm70_out));

  ops.def(
      "nvfp4_expand_raw_scales_sm70_out("
      "Tensor(a!) out, Tensor scale_codes, Tensor global_scales, "
      "bool interleaved_w13, bool fast_decode_rounding, str? "
      "native_policy=None "
      ") -> ()");
  ops.impl("nvfp4_expand_raw_scales_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_expand_raw_scales_sm70_out));

  ops.def(
      "nvfp4_moe_qpn_raw_scale_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scale_codes, "
      "Tensor global_scales, Tensor expert_ids, bool broadcast_input, "
      "bool interleaved_w13, int split_k, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_moe_qpn_raw_scale_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_moe_qpn_raw_scale_sm70_out));

  ops.def(
      "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, bool interleaved, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_moe_qpn_w13_swiglu_batch_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_moe_qpn_w13_swiglu_batch_sm70_out));

  ops.def(
      "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scale_codes, "
      "Tensor global_scales, Tensor expert_ids, bool interleaved, str? "
      "native_policy=None ) -> ()");
  ops.impl(
      "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out", torch::kCUDA,
      vllm::sm70::with_policy(&nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out));

  ops.def(
      "nvfp4_moe_qpn_w2_reduce_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, Tensor topk_weights, str? native_policy=None ) -> "
      "()");
  ops.impl("nvfp4_moe_qpn_w2_reduce_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_moe_qpn_w2_reduce_sm70_out));

  ops.def(
      "nvfp4_moe_qpn_raw_w2_reduce_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scale_codes, "
      "Tensor global_scales, Tensor expert_ids, Tensor topk_weights, str? "
      "native_policy=None ) -> ()");
  ops.impl("nvfp4_moe_qpn_raw_w2_reduce_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_moe_qpn_raw_w2_reduce_sm70_out));

  ops.def(
      "nvfp4_qwen38_w2_direct_reduce_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, Tensor topk_weights, str? native_policy=None ) -> "
      "()");
  ops.impl("nvfp4_qwen38_w2_direct_reduce_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qwen38_w2_direct_reduce_out));

  ops.def(
      "nvfp4_qwen38_w13_fused_swiglu_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, str? native_policy=None ) -> ()");
  ops.impl("nvfp4_qwen38_w13_fused_swiglu_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_qwen38_w13_fused_swiglu_out));

  // Keep the five-row verifier on a distinct schema so an old extension that
  // only supports the ten-route M=1 contract cannot be selected accidentally.
  ops.def(
      "nvfp4_moe_qpn_mtp5_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, bool broadcast_input, int split_k, str? "
      "native_policy=None ) -> ()");
  ops.impl("nvfp4_moe_qpn_mtp5_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_moe_qpn_mtp5_sm70_out));

  ops.def(
      "nvfp4_glm53_moe_q8_qpn_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor expert_ids, Tensor sorted_row_idx, bool w13, str? "
      "native_policy=None ) -> ()");
  ops.impl("nvfp4_glm53_moe_q8_qpn_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_glm53_moe_q8_qpn_sm70_out));

  ops.def(
      "nvfp4_moe_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor dense_expert_ids, Tensor ptrs_w, Tensor ptrs_s, "
      "int num_experts, int k, int n, int group_size, str? "
      "native_policy=None ) "
      "-> ()");
  ops.impl("nvfp4_moe_dense_stage_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_moe_dense_stage_sm70_out));

  ops.def(
      "nvfp4_moe_indexed_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor input_row_indices, "
      "Tensor expert_offsets, Tensor dense_expert_ids, Tensor ptrs_w, "
      "Tensor ptrs_s, int num_experts, int k, int n, int group_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("nvfp4_moe_indexed_dense_stage_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_moe_indexed_dense_stage_sm70_out));

  ops.def(
      "nvfp4_moe_indexed_fused_swiglu_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor input_row_indices, "
      "Tensor expert_offsets, Tensor dense_expert_ids, Tensor ptrs_w, "
      "Tensor ptrs_s, int num_experts, int k, int n, int group_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("nvfp4_moe_indexed_fused_swiglu_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&nvfp4_moe_indexed_fused_swiglu_sm70_out));

  ops.def(
      "mxfp4_moe_single_token_prepare_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) expert_offsets, Tensor(d!) inv_permuted_idx, "
      "Tensor(e!) sorted_expert_ids, int w13_k, int w13_n, "
      "int group_size, int hidden_logical_size, str? native_policy=None ) -> "
      "()");
  ops.impl(
      "mxfp4_moe_single_token_prepare_w13_sm70_out", torch::kCUDA,
      vllm::sm70::with_policy(&mxfp4_moe_single_token_prepare_w13_sm70_out));

  ops.def(
      "fp8_moe_single_token_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor sorted_expert_ids, Tensor ptrs_w, Tensor ptrs_s, int top_k, "
      "int k, int n, int group_size, str? native_policy=None ) -> ()");
  ops.impl("fp8_moe_single_token_dense_stage_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_moe_single_token_dense_stage_sm70_out));

  ops.def(
      "fp8_moe_single_token_indexed_dense_stage_sm70_out("
      "Tensor(a!) out, Tensor input, Tensor expert_offsets, "
      "Tensor sorted_expert_ids, Tensor ptrs_w, Tensor ptrs_s, int top_k, "
      "int k, int n, int group_size, str? native_policy=None ) -> ()");
  ops.impl("fp8_moe_single_token_indexed_dense_stage_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(
               &fp8_moe_single_token_indexed_dense_stage_sm70_out));

  ops.def(
      "fp8_moe_single_token_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) expert_offsets, Tensor(d!) expert_offsets64, "
      "Tensor(e!) inv_permuted_idx, Tensor(f!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("fp8_moe_single_token_dense_w13_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_moe_single_token_dense_w13_sm70_out));

  ops.def(
      "fp8_moe_single_token_indexed_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) expert_offsets, Tensor(d!) expert_offsets64, "
      "Tensor(e!) inv_permuted_idx, Tensor(f!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("fp8_moe_single_token_indexed_dense_w13_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(
               &fp8_moe_single_token_indexed_dense_w13_sm70_out));

  ops.def(
      "fp8_moe_single_token_compact_dense_w13_sm70_out("
      "Tensor(a!) gate_up, Tensor(b!) compact_input, Tensor x, "
      "Tensor topk_ids, Tensor w13_ptrs_w, Tensor w13_ptrs_s, "
      "Tensor(c!) compact_w13_ptrs_w, Tensor(d!) compact_w13_ptrs_s, "
      "Tensor(e!) expert_offsets, Tensor(f!) expert_offsets64, "
      "Tensor(g!) inv_permuted_idx, Tensor(h!) sorted_expert_ids, "
      "int w13_k, int w13_n, int group_size, int hidden_logical_size, str? "
      "native_policy=None ) -> ()");
  ops.impl("fp8_moe_single_token_compact_dense_w13_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(
               &fp8_moe_single_token_compact_dense_w13_sm70_out));

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
      "bool exact_per_route, str? native_policy=None ) -> ()");
  ops.impl("fp8_moe_single_token_sm70_out", torch::kCUDA,
           vllm::sm70::with_policy(&fp8_moe_single_token_sm70_out));
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
      "sm70_tp4_all_reduce_gemma_rms_norm_reference(int fa, Tensor inp, Tensor "
      "residual, Tensor weight, Tensor! normalized_out, Tensor! residual_out, "
      "int reg_buffer, int reg_buffer_sz_bytes, float epsilon) -> ()");
  custom_ar.impl("sm70_tp4_all_reduce_gemma_rms_norm_reference", torch::kCUDA,
                 &sm70_tp4_all_reduce_gemma_rms_norm_reference);
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
      "sm70_qwen38_hc_batch(int fa, Tensor input, Tensor packed_down, "
      "Tensor packed_up, Tensor(a!) partials, Tensor(b!) lora, "
      "Tensor(c!) local_output, Tensor(d!) output, Tensor(e!) injection, "
      "bool round_down_partials=False, bool cooperative=False, bool "
      "full_unroll=False, bool fused_chain=False, int cta_split_warps=0) -> "
      "()");
  custom_ar.impl("sm70_qwen38_hc_batch", torch::kCUDA, &sm70_qwen38_hc_batch);
  custom_ar.def(
      "sm70_qwen38_hc_down_local(Tensor input, Tensor packed, "
      "Tensor(a!) partials, Tensor(b!) output, int rank) -> ()");
  custom_ar.impl("sm70_qwen38_hc_down_local", torch::kCUDA,
                 &sm70_qwen38_hc_down_local);
  custom_ar.def(
      "sm70_qwen38_hc_up_local(Tensor lora, Tensor packed, Tensor branches, "
      "Tensor(a!) output, int rank) -> ()");
  custom_ar.impl("sm70_qwen38_hc_up_local", torch::kCUDA,
                 &sm70_qwen38_hc_up_local);
  custom_ar.def(
      "sm70_qwen38_hc_replicated(Tensor input, Tensor packed_down, "
      "Tensor packed_up, Tensor(a!) partials, Tensor(b!) lora, "
      "Tensor(c!) output, Tensor(d!) injection) -> ()");
  custom_ar.impl("sm70_qwen38_hc_replicated", torch::kCUDA,
                 &sm70_qwen38_hc_replicated);
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
