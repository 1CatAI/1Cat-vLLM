#ifndef FUSED_MHA_H
#define FUSED_MHA_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <torch/extension.h>
#include <ATen/ATen.h>

std::vector<at::Tensor> flash_attention_forward(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor>& out_,
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_
);

at::Tensor flash_attention_qk_scores(
    const at::Tensor& q,
    const at::Tensor& k,
    const float softmax_scale,
    const bool is_causal
);

at::Tensor flash_attention_decode_paged(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const float softmax_scale,
    const int partition_size,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale,
    const int window_size_left,
    const int window_size_right
);

at::Tensor flash_attention_decode_paged_wmma(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const float softmax_scale,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale
);

at::Tensor flash_attention_decode_qk_scores(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const float softmax_scale,
    const int partition_size,
    const std::string& kv_cache_dtype,
    const float k_scale
);

at::Tensor flash_attention_turboquant_decode_paged(
    const at::Tensor& q_rot,
    const at::Tensor& kv_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const at::Tensor& centroids,
    const float softmax_scale,
    const int partition_size,
    const int mse_bits,
    const int value_quant_bits,
    const bool norm_correction
);

at::Tensor flash_attention_prefill_paged(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const float softmax_scale,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale,
    const bool is_causal,
    const int window_size_left,
    const int window_size_right
);

std::vector<at::Tensor> flash_attention_backward(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor>& dq_,
    std::optional<at::Tensor>& dk_,
    std::optional<at::Tensor>& dv_,
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool deterministic,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor>& rng_state
);

#endif
