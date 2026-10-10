# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auxiliary platform warmup, exact sampler fallback and dispatch diagnostics."""

from __future__ import annotations

import torch

from vllm import envs
from vllm.config.compilation import CUDAGraphMode
from vllm.config.sm70_dflash2 import capture_sm70_dflash2_config, sm70_dflash2_enabled
from vllm.model_executor.layers import sm70_fuse47 as _fuse47
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.sm70_greedy_verify import greedy_verify


def warmup_smallq_metadata(self, logger) -> bool:
    """Pre-compile the grouped small-query verifier metadata Triton kernel.

    ``_sm70_prepare_grouped_smallq_decode_metadata_kernel`` is launched once
    per verify step by ``prepare_dflash2_smallq_group_metadata``. Its
    ``BLOCK_COLS`` constexpr is fixed by the full-attention block-table
    width and its ``REQ_BLOCK`` constexpr follows ``num_reqs``. The kernel is
    skipped during CUDA-graph capture (capture builds no grouped small-query
    metadata), so the first real verify of each ``num_reqs`` otherwise pays a
    multi-hundred-ms Triton JIT spike (jit_monitor warns
    ``_sm70_prepare_grouped_smallq_decode_metadata_kernel``). Compile every
    ``num_reqs`` in 1..max_num_seqs (q = decode_query_len) ahead of time.
    """
    policy = capture_sm70_dflash2_config(self.vllm_config)
    if not (
        sm70_dflash2_enabled("fused_smallq_metadata", policy)
        and sm70_dflash2_enabled("grouped_smallq_metadata", policy)
    ):
        return False
    attn_groups = getattr(self, "attn_groups", None)
    block_tables = getattr(self, "block_tables", None)
    if not attn_groups or block_tables is None:
        return False
    try:
        from vllm.v1.attention.backends.flash_v100 import metadata
        from vllm.v1.attention.backends.flash_v100.spec import smallq_metadata

        builder_class = metadata.FlashAttnV100MetadataBuilder
        prepare_metadata = smallq_metadata._sm70_prepare_grouped_smallq_decode_metadata

        group_block_tables = block_tables.block_tables
        block_cols_set: set[int] = set()
        for kv_cache_group_id, groups in enumerate(attn_groups):
            if kv_cache_group_id >= len(group_block_tables):
                continue
            for group in groups:
                builder = group.get_metadata_builder(0)
                if not isinstance(builder, builder_class):
                    continue
                width = int(group_block_tables[kv_cache_group_id].gpu.shape[1])
                if width > 0:
                    block_cols_set.add(width)
        if not block_cols_set:
            return False
        q = int(self.decode_query_len)
        device = self.device
        for block_cols in sorted(block_cols_set):
            for num_reqs in range(1, self.max_num_reqs + 1):
                num_query_tokens = num_reqs * q
                real_variants = {num_query_tokens}
                if num_reqs >= 2:
                    real_variants.add(q)
                for real_num_query_tokens in sorted(real_variants):
                    out_bt = torch.zeros(
                        (num_query_tokens, block_cols), dtype=torch.int32, device=device
                    )
                    out_sl = torch.zeros(
                        (num_query_tokens,), dtype=torch.int32, device=device
                    )
                    out_qsl = torch.zeros(
                        (num_reqs + 1,), dtype=torch.int32, device=device
                    )
                    in_bt = torch.zeros(
                        (num_reqs, block_cols), dtype=torch.int32, device=device
                    )
                    seq_lens = torch.full(
                        (num_reqs,), max(q, 1), dtype=torch.int32, device=device
                    )
                    query_start_loc = torch.arange(
                        0, (num_reqs + 1) * q, q, dtype=torch.int32, device=device
                    )
                    prepare_metadata(
                        [out_bt],
                        [out_sl],
                        [out_qsl],
                        [in_bt],
                        seq_lens,
                        query_start_loc,
                        num_reqs=num_reqs,
                        num_query_tokens=num_query_tokens,
                        real_num_query_tokens=real_num_query_tokens,
                    )
        torch.accelerator.synchronize()
        return True
    except Exception as exc:
        logger.warning_once(
            "SM70 DFlash2 grouped small-query metadata warmup skipped: %s", exc
        )
        return False


def try_target_sample(runner, hidden_states, input_batch, grammar_output):
    """Try exact greedy verification after upstream sampler routes decline."""
    if not _fuse47.unit_enabled("s1"):
        return None
    reason = _fuse47.s1_block_reason(runner, input_batch, grammar_output)
    if reason is not None:
        _fuse47.note_route("s1", "fallback:" + reason)
        return None
    assert runner.rejection_sampler is not None
    _fuse47.note_route("s1", "fused")
    top = _fuse47.tp_local_top1(runner.model, hidden_states)
    sampled, num_sampled = greedy_verify(
        top,
        input_batch.input_ids[input_batch.logits_indices],
        input_batch.cu_num_logits,
        runner.rejection_sampler.num_speculative_steps,
    )
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
    )


def note_dispatch(
    runner, num_reqs, num_toks, max_query_len, uniform_tok_count, batch_desc, logger
):
    """Log bounded dispatch diagnostics without changing graph selection."""
    if not envs.VLLM_SM70_CG_DISPATCH_DEBUG or max_query_len <= 1:
        return
    n = getattr(runner, "_cg_dispatch_debug_count", 0) + 1
    runner._cg_dispatch_debug_count = n
    if n <= 2000 or batch_desc.cg_mode != CUDAGraphMode.FULL:
        logger.info(
            "SM70 CG dispatch: num_reqs=%d num_tokens=%d "
            "max_query_len=%d uniform_tok_count=%s cg_mode=%s",
            num_reqs,
            num_toks,
            max_query_len,
            uniform_tok_count,
            batch_desc.cg_mode.name,
        )
