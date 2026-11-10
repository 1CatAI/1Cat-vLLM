# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-specific prefix admission, accounting and diagnostics."""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100.plan import routing as _plan

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def is_draft_layer(layer):
    return bool(getattr(layer, "is_dflash_draft_attn", False))


def reject_tree_anchor():
    raise RuntimeError(
        "FLASH_ATTN_V100 anchored decode-window mask does not "
        "support ddtree drafting metadata."
    )


def noncausal_batch(config, ops, request, record: _plan.RecordRoute):
    shape = (
        request.num_seqs,
        request.max_query_len,
        request.query.shape[1],
        request.head_dim,
    )
    ops.log_noncausal(
        config,
        request.num_seqs,
        request.max_query_len,
        request.block_size,
    )
    record(_routing.ROUTE_SPECS["prefill_prefix_dflash_noncausal_batch"].name)
    ops.run_paged(
        route="prefill_prefix_dflash_noncausal_batch",
        q_len=request.max_query_len,
        seq_len=int(request.seq_lens.max().item()),
        heads_q=request.query.shape[1],
        heads_kv=request.num_kv_heads,
        head_dim=request.head_dim,
        block_size=request.block_size,
        fn=lambda: ops.paged(
            request.query.reshape(shape),
            request.key_cache,
            request.value_cache,
            request.attn_metadata.block_table[: request.num_seqs],
            request.attn_metadata.seq_lens[: request.num_seqs],
            out=request.out_view.view(shape),
            softmax_scale=config.scale,
            kv_cache_dtype=config.kv_cache_dtype,
            k_scale=float(request.layer._k_scale_float),
            v_scale=float(request.layer._v_scale_float),
            causal=False,
            window_size=request.window_size,
        ),
    )


def log_noncausal(config, num_seqs, max_query_len, block_size):
    logger.info_once(
        "FLASH_ATTN_V100 DFlash uniform noncausal paged batch route "
        "active (batch=%d, q=%d, page=%d).",
        num_seqs,
        max_query_len,
        block_size,
    )
