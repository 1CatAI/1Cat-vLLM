# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 auxiliary bindings and their fake implementations."""

import torch

from .common import _op, _qwen38_qpn8_op, register_fake


def silu_and_mul_interleaved(out: torch.Tensor, input: torch.Tensor) -> None:
    _op("silu_and_mul_interleaved")(out, input)


if hasattr(torch.ops._C, "silu_and_mul_interleaved"):

    @register_fake("_C::silu_and_mul_interleaved")
    def _silu_and_mul_interleaved_fake(out: torch.Tensor, input: torch.Tensor) -> None:
        del out, input
        return None


def qwen38_shared_gate_exact_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    _qwen38_qpn8_op("qwen38_shared_gate_exact_out")(out, input, weight)


if hasattr(torch.ops._C, "qwen38_shared_gate_exact_out"):

    @register_fake("_C::qwen38_shared_gate_exact_out")
    def _qwen38_shared_gate_exact_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        return None


if hasattr(torch.ops._C_qwen38, "qwen38_shared_gate_exact_out"):

    @register_fake("_C_qwen38::qwen38_shared_gate_exact_out")
    def _qwen38_shared_gate_exact_out_sidecar_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        return None


def qwen38_shared_gate_sigmoid_mul_out(out: torch.Tensor, logits: torch.Tensor) -> None:
    torch.ops._C.qwen38_shared_gate_sigmoid_mul_out(out, logits)


if hasattr(torch.ops._C, "qwen38_shared_gate_sigmoid_mul_out"):

    @register_fake("_C::qwen38_shared_gate_sigmoid_mul_out")
    def _qwen38_shared_gate_sigmoid_mul_out_fake(
        out: torch.Tensor, logits: torch.Tensor
    ) -> None:
        return None


def sm70_glm_mhc_pre_norm_out(
    gemm_mul: torch.Tensor,
    gemm_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult: float,
    sinkhorn_repeat: int,
    norm_eps: float,
) -> None:
    _op("sm70_glm_mhc_pre_norm_out")(
        gemm_mul,
        gemm_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult,
        sinkhorn_repeat,
        norm_eps,
    )


if hasattr(torch.ops._C, "sm70_glm_mhc_pre_norm_out"):

    @register_fake("_C::sm70_glm_mhc_pre_norm_out")
    def _sm70_glm_mhc_pre_norm_out_fake(
        gemm_mul: torch.Tensor,
        gemm_sqrsum: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        layer_input: torch.Tensor,
        norm_weight: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult: float,
        sinkhorn_repeat: int,
        norm_eps: float,
    ) -> None:
        return None


def sm70_glm_mhc_post_dot_q8_out(
    residual_out: torch.Tensor,
    gemm_mul: torch.Tensor,
    gemm_sqrsum: torch.Tensor,
    comb_mix: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    tile_n: int,
) -> None:
    _op("sm70_glm_mhc_post_dot_q8_out")(
        residual_out,
        gemm_mul,
        gemm_sqrsum,
        comb_mix,
        residual,
        post_mix,
        x,
        weight,
        tile_n,
    )


if hasattr(torch.ops._C, "sm70_glm_mhc_post_dot_q8_out"):

    @register_fake("_C::sm70_glm_mhc_post_dot_q8_out")
    def _sm70_glm_mhc_post_dot_q8_out_fake(
        residual_out: torch.Tensor,
        gemm_mul: torch.Tensor,
        gemm_sqrsum: torch.Tensor,
        comb_mix: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        tile_n: int,
    ) -> None:
        return None


def sm70_f16_indexed_rerank_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    candidate_ids: torch.Tensor,
    selected_raw: torch.Tensor,
    selected_packed: torch.Tensor,
    expanded: torch.Tensor,
    partials: torch.Tensor,
    barriers: torch.Tensor,
    cta_n: int,
    split_k: int,
) -> None:
    _op("sm70_f16_indexed_rerank_out")(
        out,
        input,
        weight,
        candidate_ids,
        selected_raw,
        selected_packed,
        expanded,
        partials,
        barriers,
        cta_n,
        split_k,
    )


if hasattr(torch.ops._C, "sm70_f16_indexed_rerank_out"):

    @register_fake("_C::sm70_f16_indexed_rerank_out")
    def _sm70_f16_indexed_rerank_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        candidate_ids: torch.Tensor,
        selected_raw: torch.Tensor,
        selected_packed: torch.Tensor,
        expanded: torch.Tensor,
        partials: torch.Tensor,
        barriers: torch.Tensor,
        cta_n: int,
        split_k: int,
    ) -> None:
        return None


def sm70_glm_kda_fg_b_out(
    f_out: torch.Tensor,
    g_out: torch.Tensor,
    f_input: torch.Tensor,
    g_input: torch.Tensor,
    f_weight: torch.Tensor,
    g_weight: torch.Tensor,
) -> None:
    _op("sm70_glm_kda_fg_b_out")(f_out, g_out, f_input, g_input, f_weight, g_weight)


if hasattr(torch.ops._C, "sm70_glm_kda_fg_b_out"):

    @register_fake("_C::sm70_glm_kda_fg_b_out")
    def _sm70_glm_kda_fg_b_out_fake(
        f_out: torch.Tensor,
        g_out: torch.Tensor,
        f_input: torch.Tensor,
        g_input: torch.Tensor,
        f_weight: torch.Tensor,
        g_weight: torch.Tensor,
    ) -> None:
        return None


def sm70_f16_indexed_rerank_packed_out(
    out: torch.Tensor,
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    candidate_ids: torch.Tensor,
    selected_packed: torch.Tensor,
    expanded: torch.Tensor,
    partials: torch.Tensor,
    barriers: torch.Tensor,
    cta_n: int,
    split_k: int,
) -> None:
    _op("sm70_f16_indexed_rerank_packed_out")(
        out,
        input,
        packed_weight,
        candidate_ids,
        selected_packed,
        expanded,
        partials,
        barriers,
        cta_n,
        split_k,
    )


if hasattr(torch.ops._C, "sm70_f16_indexed_rerank_packed_out"):

    @register_fake("_C::sm70_f16_indexed_rerank_packed_out")
    def _sm70_f16_indexed_rerank_packed_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        candidate_ids: torch.Tensor,
        selected_packed: torch.Tensor,
        expanded: torch.Tensor,
        partials: torch.Tensor,
        barriers: torch.Tensor,
        cta_n: int,
        split_k: int,
    ) -> None:
        return None


def sm70_f16_rerank_keys_out(
    keys: torch.Tensor,
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> None:
    _op("sm70_f16_rerank_keys_out")(keys, logits, candidate_ids)


if hasattr(torch.ops._C, "sm70_f16_rerank_keys_out"):

    @register_fake("_C::sm70_f16_rerank_keys_out")
    def _sm70_f16_rerank_keys_out_fake(
        keys: torch.Tensor,
        logits: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> None:
        return None


def sm70_f16_rerank_topk_out(
    values_out: torch.Tensor,
    ids_out: torch.Tensor,
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    vocab_start_index: int,
) -> None:
    _op("sm70_f16_rerank_topk_out")(
        values_out,
        ids_out,
        logits,
        candidate_ids,
        vocab_start_index,
    )


if hasattr(torch.ops._C, "sm70_f16_rerank_topk_out"):

    @register_fake("_C::sm70_f16_rerank_topk_out")
    def _sm70_f16_rerank_topk_out_fake(
        values_out: torch.Tensor,
        ids_out: torch.Tensor,
        logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        vocab_start_index: int,
    ) -> None:
        return None


def sm70_f16_lm_head_top1_out(
    values_out: torch.Tensor,
    indices_out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    vocab_start_index: int,
    num_vocab_padding: int,
) -> None:
    _op("sm70_f16_lm_head_top1_out")(
        values_out,
        indices_out,
        input,
        weight,
        k_ld,
        vocab_start_index,
        num_vocab_padding,
    )


if hasattr(torch.ops._C, "sm70_f16_lm_head_top1_out"):

    @register_fake("_C::sm70_f16_lm_head_top1_out")
    def _sm70_f16_lm_head_top1_out_fake(
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        vocab_start_index: int,
        num_vocab_padding: int,
    ) -> None:
        return None


def sm70_f16_lm_head_top1_tc_out(
    values_out: torch.Tensor,
    indices_out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    vocab_start_index: int,
    num_vocab_padding: int,
) -> None:
    _op("sm70_f16_lm_head_top1_tc_out")(
        values_out,
        indices_out,
        input,
        weight,
        k_ld,
        vocab_start_index,
        num_vocab_padding,
    )


if hasattr(torch.ops._C, "sm70_f16_lm_head_top1_tc_out"):

    @register_fake("_C::sm70_f16_lm_head_top1_tc_out")
    def _sm70_f16_lm_head_top1_tc_out_fake(
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        vocab_start_index: int,
        num_vocab_padding: int,
    ) -> None:
        return None


def sm70_f16_lm_head_top20_tc_out(
    values_out: torch.Tensor,
    indices_out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    vocab_start_index: int,
    num_vocab_padding: int,
) -> None:
    _op("sm70_f16_lm_head_top20_tc_out")(
        values_out,
        indices_out,
        input,
        weight,
        k_ld,
        vocab_start_index,
        num_vocab_padding,
    )


if hasattr(torch.ops._C, "sm70_f16_lm_head_top20_tc_out"):

    @register_fake("_C::sm70_f16_lm_head_top20_tc_out")
    def _sm70_f16_lm_head_top20_tc_out_fake(
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        vocab_start_index: int,
        num_vocab_padding: int,
    ) -> None:
        return None


def sm70_merge_tail_top20_pack_out(
    pairs_out: torch.Tensor,
    base_values: torch.Tensor,
    base_indices: torch.Tensor,
    base_token_id_map: torch.Tensor,
    tail_logits: torch.Tensor,
    tail_token_ids: torch.Tensor,
    tail_row_start: int,
) -> None:
    _op("sm70_merge_tail_top20_pack_out")(
        pairs_out,
        base_values,
        base_indices,
        base_token_id_map,
        tail_logits,
        tail_token_ids,
        tail_row_start,
    )


if hasattr(torch.ops._C, "sm70_merge_tail_top20_pack_out"):

    @register_fake("_C::sm70_merge_tail_top20_pack_out")
    def _sm70_merge_tail_top20_pack_out_fake(
        pairs_out: torch.Tensor,
        base_values: torch.Tensor,
        base_indices: torch.Tensor,
        base_token_id_map: torch.Tensor,
        tail_logits: torch.Tensor,
        tail_token_ids: torch.Tensor,
        tail_row_start: int,
    ) -> None:
        return None


def sm70_sample_packed_top20_out(
    sampled_token_out: torch.Tensor,
    sparse_ids_out: torch.Tensor,
    sparse_probs_out: torch.Tensor,
    gathered_pairs: torch.Tensor,
    exponential: torch.Tensor,
    top_p: float,
) -> None:
    _op("sm70_sample_packed_top20_out")(
        sampled_token_out,
        sparse_ids_out,
        sparse_probs_out,
        gathered_pairs,
        exponential,
        top_p,
    )


if hasattr(torch.ops._C, "sm70_sample_packed_top20_out"):

    @register_fake("_C::sm70_sample_packed_top20_out")
    def _sm70_sample_packed_top20_out_fake(
        sampled_token_out: torch.Tensor,
        sparse_ids_out: torch.Tensor,
        sparse_probs_out: torch.Tensor,
        gathered_pairs: torch.Tensor,
        exponential: torch.Tensor,
        top_p: float,
    ) -> None:
        return None


def sm70_sample_sorted_top20_philox_out(
    sampled_token_out: torch.Tensor,
    sparse_ids_out: torch.Tensor,
    sparse_probs_out: torch.Tensor,
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    generator: torch.Generator | None,
    vocab_size: int,
    top_p: float,
) -> None:
    _op("sm70_sample_sorted_top20_philox_out")(
        sampled_token_out,
        sparse_ids_out,
        sparse_probs_out,
        top_values,
        top_indices,
        generator,
        vocab_size,
        top_p,
    )


if hasattr(torch.ops._C, "sm70_sample_sorted_top20_philox_out"):

    @register_fake("_C::sm70_sample_sorted_top20_philox_out")
    def _sm70_sample_sorted_top20_philox_out_fake(
        sampled_token_out: torch.Tensor,
        sparse_ids_out: torch.Tensor,
        sparse_probs_out: torch.Tensor,
        top_values: torch.Tensor,
        top_indices: torch.Tensor,
        generator: torch.Generator | None,
        vocab_size: int,
        top_p: float,
    ) -> None:
        return None


def sm70_sample_sorted_top20_philox_token_out(
    sampled_token_out: torch.Tensor,
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    generator: torch.Generator | None,
    vocab_size: int,
    top_p: float,
) -> None:
    _op("sm70_sample_sorted_top20_philox_token_out")(
        sampled_token_out,
        top_values,
        top_indices,
        generator,
        vocab_size,
        top_p,
    )


if hasattr(torch.ops._C, "sm70_sample_sorted_top20_philox_token_out"):

    @register_fake("_C::sm70_sample_sorted_top20_philox_token_out")
    def _sm70_sample_sorted_top20_philox_token_out_fake(
        sampled_token_out: torch.Tensor,
        top_values: torch.Tensor,
        top_indices: torch.Tensor,
        generator: torch.Generator | None,
        vocab_size: int,
        top_p: float,
    ) -> None:
        return None


def sm70_sample_chunked_top20_philox_token_out(
    sampled_token_out: torch.Tensor,
    global_values: torch.Tensor,
    local_indices: torch.Tensor,
    global_positions: torch.Tensor,
    generator: torch.Generator | None,
    vocab_size: int,
    top_p: float,
    chunk_size: int,
) -> None:
    _op("sm70_sample_chunked_top20_philox_token_out")(
        sampled_token_out,
        global_values,
        local_indices,
        global_positions,
        generator,
        vocab_size,
        top_p,
        chunk_size,
    )


if hasattr(torch.ops._C, "sm70_sample_chunked_top20_philox_token_out"):

    @register_fake("_C::sm70_sample_chunked_top20_philox_token_out")
    def _sm70_sample_chunked_top20_philox_token_out_fake(
        sampled_token_out: torch.Tensor,
        global_values: torch.Tensor,
        local_indices: torch.Tensor,
        global_positions: torch.Tensor,
        generator: torch.Generator | None,
        vocab_size: int,
        top_p: float,
        chunk_size: int,
    ) -> None:
        return None


def sm70_dynamic_draft_vocab_update_tail_out(
    lru_token_ids: torch.Tensor,
    local_tail_token_ids: torch.Tensor,
    source_row_indices: torch.Tensor,
    observed_output_ids: torch.Tensor,
    target_candidate_ids: torch.Tensor,
    base_token_mask: torch.Tensor,
    full_vocab_size: int,
    local_shard_start: int,
    local_shard_end: int,
) -> None:
    _op("sm70_dynamic_draft_vocab_update_tail_out")(
        lru_token_ids,
        local_tail_token_ids,
        source_row_indices,
        observed_output_ids,
        target_candidate_ids,
        base_token_mask,
        full_vocab_size,
        local_shard_start,
        local_shard_end,
    )


if hasattr(torch.ops._C, "sm70_dynamic_draft_vocab_update_tail_out"):

    @register_fake("_C::sm70_dynamic_draft_vocab_update_tail_out")
    def _sm70_dynamic_draft_vocab_update_tail_out_fake(
        lru_token_ids: torch.Tensor,
        local_tail_token_ids: torch.Tensor,
        source_row_indices: torch.Tensor,
        observed_output_ids: torch.Tensor,
        target_candidate_ids: torch.Tensor,
        base_token_mask: torch.Tensor,
        full_vocab_size: int,
        local_shard_start: int,
        local_shard_end: int,
    ) -> None:
        return None


def sm70_dynamic_draft_vocab_refresh_tail_weight_out(
    local_tail_weight: torch.Tensor,
    source_weight: torch.Tensor,
    source_row_indices: torch.Tensor,
) -> None:
    _op("sm70_dynamic_draft_vocab_refresh_tail_weight_out")(
        local_tail_weight,
        source_weight,
        source_row_indices,
    )


if hasattr(torch.ops._C, "sm70_dynamic_draft_vocab_refresh_tail_weight_out"):

    @register_fake("_C::sm70_dynamic_draft_vocab_refresh_tail_weight_out")
    def _sm70_dynamic_draft_vocab_refresh_tail_weight_out_fake(
        local_tail_weight: torch.Tensor,
        source_weight: torch.Tensor,
        source_row_indices: torch.Tensor,
    ) -> None:
        return None


def sm70_f16_gate_mul_out(
    out: torch.Tensor,
    input: torch.Tensor,
    gate_weight: torch.Tensor,
) -> None:
    _op("sm70_f16_gate_mul_out")(out, input, gate_weight)


if hasattr(torch.ops._C, "sm70_f16_gate_mul_out"):

    @register_fake("_C::sm70_f16_gate_mul_out")
    def _sm70_f16_gate_mul_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        gate_weight: torch.Tensor,
    ) -> None:
        return None
