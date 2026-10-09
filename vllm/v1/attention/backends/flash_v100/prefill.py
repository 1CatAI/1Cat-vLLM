# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 prefill methods, bound by impl."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import debug as _debug
from vllm.v1.attention.backends.flash_v100 import dense_prefill as _dense_prefill
from vllm.v1.attention.backends.flash_v100 import impl as _impl
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import masks as _masks
from vllm.v1.attention.backends.flash_v100 import metadata as _metadata
from vllm.v1.attention.backends.flash_v100 import ops as _ops
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100 import state as _state
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP8_E5M2,
    FP16,
)
from vllm.v1.attention.ops.sm70_grouped import (
    MAX_GROUPS_PER_CALL,
    grouped_e4m3_fp32_groups_allowed,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def _flash_v100_prefill(
    self: _impl.FlashAttnV100Impl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Prefill path for no-prefix case (query_len == seq_len per sequence)."""
    causal = getattr(attn_metadata, "causal", True)
    window_size = self._flash_v100_window_size(causal)
    num_actual_tokens = attn_metadata.num_actual_tokens
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    query_start_loc = (
        query_start_loc_cpu
        if query_start_loc_cpu is not None
        else attn_metadata.query_start_loc
    )
    return _dense_prefill.flash_v100_dense_prefill(
        query=query,
        key=key,
        value=value,
        output=output,
        query_start_loc=query_start_loc,
        num_actual_tokens=num_actual_tokens,
        softmax_scale=self.scale,
        causal=causal,
        window_size=window_size,
        query_start_loc_device=attn_metadata.query_start_loc,
    )


def _should_use_fp8_prefill_bridge(
    self: _impl.FlashAttnV100Impl,
    *,
    q_len: int,
    head_dim: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    causal: bool,
    window_size: tuple[int, int],
) -> bool:
    # Eight-byte input loads and 16-byte output stores. Keep layouts
    # outside the native bridge contract on their existing fallback.
    if self.kv_codec is FP8_E4M3 and not all(
        tensor.ndim == 4
        and tensor.stride(-1) == 1
        and tensor.data_ptr() % 16 == 0
        and all(stride % 8 == 0 for stride in tensor.stride()[:3])
        for tensor in (key_cache, value_cache)
    ):
        return False
    return (
        self.use_fp8_prefill_bridge
        and self.use_flash_v100_prefill_paged
        and self.kv_codec in (FP8_E4M3, FP8_E5M2)
        and self.kv_codec.stores(key_cache, value_cache)
        and key_cache.shape == value_cache.shape
        and head_dim == 256
        and q_len >= 32
        and causal
        and window_size == (-1, -1)
    )


def _run_fp8_prefill_bridge(
    self: _impl.FlashAttnV100Impl,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len: int,
    k_scale: float,
    v_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    out: torch.Tensor,
) -> tuple[torch.Tensor, bool] | None:
    if block_table.shape[0] != 1:
        return None
    input_block_size = int(key_cache.shape[1])
    active_input_blocks = min(
        int(block_table.shape[1]),
        _masks._cdiv_int(seq_len, input_block_size),
    )
    if active_input_blocks <= 0:
        return None
    active_block_table = block_table[:, :active_input_blocks]
    input_capacity = active_input_blocks * input_block_size
    required_blocks = _masks._cdiv_int(
        input_capacity,
        _dense_prefill._FP8_PREFILL_BRIDGE_PAGE_SIZE,
    )
    workspace = _dense_prefill._get_fp8_prefill_bridge_workspace(
        key_cache,
        required_blocks,
    )
    if workspace is None:
        return None
    key_out, value_out, output_block_table = workspace
    bridge = (
        self.fp8_e4m3_paged_kv_to_fp16
        if self.kv_codec is FP8_E4M3
        else self.fp8_e5m2_paged_kv_to_fp16
    )
    if bridge is None:
        return None
    bridge(
        key_cache,
        value_cache,
        active_block_table,
        seq_lens,
        key_out,
        value_out,
        k_scale,
        v_scale,
    )
    q_len = int(query.shape[1])
    exact_query = query
    exact_out = out
    tail_prefix = 0
    if q_len % 64 != 0 and seq_len % 32 == 0:
        padded_q_len = _masks._cdiv_int(q_len, 64) * 64
        if padded_q_len <= seq_len:
            tail_workspace = _dense_prefill._get_fp8_prefill_bridge_tail_workspace(
                query,
                padded_q_len,
            )
            if tail_workspace is not None:
                exact_query, exact_out = tail_workspace
                tail_prefix = padded_q_len - q_len
                exact_query[:, :tail_prefix].zero_()
                exact_query[:, tail_prefix:].copy_(query)
    cu_q, cu_k = _dense_prefill._uniform_cu_seqlens(
        exact_query,
        batch_size=1,
        query_len=int(exact_query.shape[1]),
        kv_len=seq_len,
    )
    key_dense = key_out.flatten(0, 1)[:seq_len].unsqueeze(0)
    value_dense = value_out.flatten(0, 1)[:seq_len].unsqueeze(0)
    exact_result = _dense_prefill._try_sm70_fa2_d256_prefill(
        exact_query,
        key_dense,
        value_dense,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=int(exact_query.shape[1]),
        max_seqlen_k=seq_len,
        softmax_scale=self.scale,
        causal=causal,
        window_size=window_size,
        out=exact_out,
    )
    if exact_result is not None:
        if tail_prefix:
            out.copy_(exact_result[:, tail_prefix:])
            _routing._record_route(
                _routing.ROUTE_SPECS[
                    "prefill_prefix_fp8_bridge_exact_dense_d256_tailpad"
                ].name
            )
            return out, True
        _routing._record_route(
            _routing.ROUTE_SPECS["prefill_prefix_fp8_bridge_exact_dense_d256"].name
        )
        return exact_result, True
    exact_result = _dense_prefill._try_sm70_fa2_d256_prefill(
        exact_query,
        key_out,
        value_out,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=None,
        max_seqlen_q=int(exact_query.shape[1]),
        max_seqlen_k=seq_len,
        softmax_scale=self.scale,
        causal=causal,
        window_size=window_size,
        out=exact_out,
        seqused_k=seq_lens,
        block_table=output_block_table,
    )
    if exact_result is not None:
        if tail_prefix:
            out.copy_(exact_result[:, tail_prefix:])
            _routing._record_route(
                _routing.ROUTE_SPECS[
                    "prefill_prefix_fp8_bridge_exact_d256_tailpad"
                ].name
            )
            return out, True
        _routing._record_route(
            _routing.ROUTE_SPECS["prefill_prefix_fp8_bridge_exact_d256"].name
        )
        return exact_result, True
    paged_result = self.flash_attn_prefill_paged(
        query,
        key_out,
        value_out,
        output_block_table,
        seq_lens,
        softmax_scale=self.scale,
        kv_cache_dtype=FP16.name,
        k_scale=1.0,
        v_scale=1.0,
        causal=causal,
        window_size=window_size,
    )
    return paged_result, False


def _should_use_prefill_splitkv(
    self: _impl.FlashAttnV100Impl,
    *,
    q_len: int,
    seq_len: int,
    head_dim: int,
    key_cache: torch.Tensor,
    causal: bool,
) -> bool:
    if not self.use_flash_v100_prefill_splitkv:
        return False
    if self.flash_attn_prefill_paged_splitkv is None:
        return False
    if not causal:
        return False
    if head_dim != 256:
        return False
    if key_cache.dtype != torch.float16:
        return False
    if q_len < self.prefill_split_kv_min_q:
        return False
    if self.prefill_split_kv_max_q > 0 and q_len > self.prefill_split_kv_max_q:
        return False
    if seq_len < self.prefill_split_kv_min_kv:
        return False
    return seq_len > self.prefill_split_kv_tokens


def _should_use_prefill_bfla(
    self: _impl.FlashAttnV100Impl,
    *,
    q_len: int,
    seq_len: int,
    head_dim: int,
    key_cache: torch.Tensor,
    causal: bool,
    window_size: tuple[int, int],
) -> bool:
    if not self.use_flash_v100_prefill_bfla:
        return False
    if self.flash_attn_prefill_paged_bfla is None:
        return False
    if not causal or window_size != (-1, -1):
        return False
    if head_dim != 256:
        return False
    if key_cache.dtype != torch.float16:
        return False
    if q_len < self.prefill_bfla_min_q:
        return False
    if seq_len < self.prefill_bfla_min_kv:
        return False
    return self.prefill_bfla_mask_block_n > 0


def _should_use_prefill_contig_dense(
    self: _impl.FlashAttnV100Impl,
    *,
    q_len: int,
    seq_len: int,
    head_dim: int,
    key_cache: torch.Tensor,
    causal: bool,
    window_size: tuple[int, int],
) -> bool:
    if not self.use_flash_v100_prefill_contig_dense:
        return False
    if not causal or window_size != (-1, -1):
        return False
    if head_dim != 256:
        return False
    if key_cache.dtype != torch.float16:
        return False
    if q_len < self.prefill_contig_dense_min_q:
        return False
    return seq_len >= self.prefill_contig_dense_min_kv


def _should_use_prefill_gather_dense(
    self: _impl.FlashAttnV100Impl,
    *,
    q_len: int,
    seq_len: int,
    head_dim: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    causal: bool,
    window_size: tuple[int, int],
    num_seqs: int,
) -> bool:
    graph_capture = _routing._is_cuda_graph_capturing(key_cache)
    q8192_family = (
        not envs.VLLM_FLASH_V100_PREFILL_D256_GQA_V37
        and _dense_prefill._SM70_79T_CORE_QUERY_LEN
        <= q_len
        <= _dense_prefill._SM70_79T_MAX_QUERY_LEN
    )
    aligned_shape = (
        q8192_family and seq_len % _dense_prefill._SM70_79T_KV_ALIGNMENT == 0
    ) or (
        q_len % _dense_prefill._SM70_79T_EXACT_QUERY_ALIGNMENT == 0
        and seq_len % _dense_prefill._SM70_SPLITD_KV_ALIGNMENT == 0
    )
    eligible = (
        self.use_flash_v100_prefill_gather_dense
        and q_len >= self.prefill_gather_dense_min_q
        and seq_len >= self.prefill_gather_dense_min_kv
        and seq_len >= q_len
        and aligned_shape
        and head_dim == 256
        and causal
        and window_size == (-1, -1)
        and key_cache.dtype == torch.float16
        and value_cache.dtype == torch.float16
        and key_cache.shape == value_cache.shape
        and not graph_capture
    )
    _debug._sm70_profile_trace(
        "prefill gather-dense policy: eligible=%s gate=%s q=%d min_q=%d "
        "kv=%d min_kv=%d num_seqs=%d head_dim=%d causal=%s window=%s "
        "key_dtype=%s value_dtype=%s same_shape=%s graph_capture=%s",
        eligible,
        self.use_flash_v100_prefill_gather_dense,
        q_len,
        self.prefill_gather_dense_min_q,
        seq_len,
        self.prefill_gather_dense_min_kv,
        num_seqs,
        head_dim,
        causal,
        window_size,
        key_cache.dtype,
        value_cache.dtype,
        key_cache.shape == value_cache.shape,
        graph_capture,
    )
    return eligible


def _prefill_prefix_decode_rows_allowed(
    self: _impl.FlashAttnV100Impl,
    *,
    causal: bool,
    anchor_lens: torch.Tensor | None,
    num_seqs: int,
    query: torch.Tensor,
    window_size: tuple[int, int],
) -> bool:
    return (
        envs.VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS
        and causal
        and anchor_lens is None
        and num_seqs > 1
        and self.use_flash_v100_decode
        and self.use_flash_v100_prefill_paged
        and not self.use_decode_paged_prefill
        and not self.use_decode_dense_cache
        and not self.use_decode_dense_reference
        and window_size == (-1, -1)
        and not _routing._is_cuda_graph_capturing(query)
    )


def _run_mixed_rows_grouped_e4m3(
    self: _impl.FlashAttnV100Impl,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    out_view: torch.Tensor,
    plan: _metadata._MixedDecodeRowsPlan,
) -> bool:
    """Run the resident rows of a mixed batch on the grouped E4M3 operator.

    This is the route a uniform verification batch already takes: eight
    query rows share one pass over the request's KV, accumulate in FP32 and
    keep the explicit per-row causal length. Without it a DFlash2 target in
    a mixed batch reads the whole KV once per query token through the
    scalar decoder, and the cost grows with the context. Returns ``False``
    when the operator or the layout is not admitted; nothing has been
    written to ``out_view`` in that case.
    """
    grouped_op = getattr(self, "flash_attn_grouped_e4m3_fp32_paged", None)
    if grouped_op is None:
        return False
    table = plan.group_table(attn_metadata.block_table)
    lengths = plan.group_lengths(attn_metadata.seq_lens)
    total_rows = plan.num_groups * _metadata._MIXED_ROWS_GROUP
    q_pad = query.new_zeros((total_rows, query.shape[1], query.shape[2]))
    out_pad = torch.empty_like(q_pad)
    chunks = [
        (g0, min(g0 + MAX_GROUPS_PER_CALL, plan.num_groups))
        for g0 in range(0, plan.num_groups, MAX_GROUPS_PER_CALL)
    ]
    for g0, g1 in chunks:
        r0, r1 = g0 * _metadata._MIXED_ROWS_GROUP, g1 * _metadata._MIXED_ROWS_GROUP
        if not grouped_e4m3_fp32_groups_allowed(
            self,
            q_pad[r0:r1],
            key_cache,
            value_cache,
            table[g0:g1],
            lengths[r0:r1],
            causal=bool(getattr(attn_metadata, "causal", True)),
            out=out_pad[r0:r1],
        ):
            return False
    q_pad.index_copy_(0, plan.dst_idx, query.index_select(0, plan.src_idx))
    k_scale = float(layer._k_scale_float)
    v_scale = float(layer._v_scale_float)
    for g0, g1 in chunks:
        r0, r1 = g0 * _metadata._MIXED_ROWS_GROUP, g1 * _metadata._MIXED_ROWS_GROUP
        # Row lengths are authoritative: padding rows have length zero and
        # produce zero output, so no row can read an unwritten KV entry.
        grouped_op(
            q_pad[r0:r1],
            key_cache,
            value_cache,
            table[g0:g1],
            lengths[r0:r1],
            out=out_pad[r0:r1],
            softmax_scale=self.scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    out_view.index_copy_(0, plan.src_idx, out_pad.index_select(0, plan.dst_idx))
    if not _state._logged_prefill_prefix_decode_rows_grouped:
        logger.info(
            "FLASH_ATTN_V100 mixed-batch small-query rows take the grouped "
            "E4M3 FP32 route (requests=%d, groups=%d, max_q=%d, "
            "max_seq_len=%d).",
            len(plan.rows),
            plan.num_groups,
            plan.max_query_len,
            plan.max_seq_len_hint,
        )
        _state._logged_prefill_prefix_decode_rows_grouped = True
    _routing._log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "grouped_fp32")
    _routing._record_route(
        _routing.ROUTE_SPECS["prefill_prefix_decode_rows_e4m3_grouped_fp32"].name
    )
    return True


def _run_prefill_prefix_decode_rows(
    self: _impl.FlashAttnV100Impl,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    out_view: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    window_size: tuple[int, int],
) -> set[int]:
    """Run the small-query rows of a mixed batch as one paged-decode batch.

    Inside a chunked-prefill batch every row takes the prefill route, and
    ``prefill_paged_fwd`` gives a small-q row one CTA per query head for
    the whole context (kernel/fused_mha_api.cpp launches ``grid(ceil(q/BM),
    1, B*H)``). A resident decoder at 240K pays ~58 ms per layer that way
    versus ~1.8 ms on the partitioned decode kernel (1CatAI/1Cat-vLLM#490),
    and an MTP/DFlash verify row (q = K+1) has the same grid. The rows are
    therefore pulled out of the prefill batch and run on a decode operator.
    Every query token of a selected row becomes one decode row whose visible
    KV length grows by one, the expansion _flash_v100_small_query_prefill_as_decode
    uses for the verifier, so the causal mask is preserved. A DFlash2 E4M3
    target takes the grouped FP32 operator like its uniform verifier does;
    other layouts use XQA or the scalar decoder. Returns the row indices
    consumed here; the caller's per-sequence loop skips them.
    """
    plan = _metadata._mixed_decode_rows_plan(
        attn_metadata,
        query_start_loc,
        seq_lens,
        max(1, int(self.smallq_decode_max_query_len)),
        query.device,
    )
    if plan is None:
        return set()
    num_rows = int(plan.src_idx.numel())
    max_seq_len_hint = plan.max_seq_len_hint
    max_query_len_rows = plan.max_query_len
    num_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[2])
    xqa_codec = self._xqa_kv_codec(key_cache, value_cache, attn_metadata)
    # Same selection as the uniform-decode path (_flash_v100_decode), with
    # the sequence hint taken from this batch's rows because build() only
    # attaches decode shape hints when max_query_len == 1.
    use_xqa = (
        _routing.select_route(
            _routing.RouteContext(
                stage="mixed_decode",
                codec=xqa_codec,
                shape=_routing.RouteShape(
                    num_rows,
                    num_heads,
                    num_kv_heads,
                    int(query.shape[2]),
                    int(key_cache.shape[1]),
                ),
                enabled=self.use_decode_xqa,
                available=self.flash_attn_decode_paged_xqa is not None,
                max_seq_len_hint=max_seq_len_hint,
            ),
            ("prefill_prefix_decode_rows_xqa",),
        )
        is not None
    )
    if (
        self.kv_codec is FP8_E4M3
        and not use_xqa
        and self._run_mixed_rows_grouped_e4m3(
            layer,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            out_view,
            plan,
        )
    ):
        return set(plan.rows)

    q_rows = query.index_select(0, plan.src_idx)
    out_rows = torch.empty_like(q_rows)
    block_table = attn_metadata.block_table.index_select(0, plan.token_req)
    seq_lens_rows = plan.token_lengths(attn_metadata.seq_lens)
    partition_size_hint = (
        _routing._g6_aligned_page_partition_size_hint(
            q_rows, key_cache, value_cache, self.kv_cache_dtype
        )
        if use_xqa
        else None
    )
    k_scale = float(layer._k_scale_float)
    v_scale = float(layer._v_scale_float)
    if use_xqa:
        route = "prefill_prefix_decode_rows_xqa"

        def run() -> torch.Tensor:
            self.flash_attn_decode_paged_xqa(
                q_rows,
                key_cache,
                value_cache,
                block_table,
                seq_lens_rows,
                softmax_scale=self.scale,
                out=out_rows,
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
                window_size=window_size,
                max_seq_len_hint=max_seq_len_hint,
                partition_size_hint=partition_size_hint,
                # This path runs outside a decode graph, so the live
                # context length can safely select the same optimized
                # batch/long-context routes used by uniform decode.
                batch_context_routing=True,
            )
            return out_rows
    else:
        route = "prefill_prefix_decode_rows_scalar"

        def run() -> torch.Tensor:
            self._call_flash_attn_decode_paged(
                q_rows,
                key_cache,
                value_cache,
                block_table,
                seq_lens_rows,
                softmax_scale=self.scale,
                out=out_rows,
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
                window_size=window_size,
                max_seq_len_hint=max_seq_len_hint,
            )
            return out_rows

    if not _state._logged_prefill_prefix_decode_rows:
        logger.info(
            "FLASH_ATTN_V100 mixed-batch small-query rows take the paged "
            "decode route (%s, rows=%d of %d, max_q=%d, max_seq_len=%d).",
            route,
            len(plan.rows),
            len(query_start_loc) - 1,
            max_query_len_rows,
            max_seq_len_hint,
        )
        _state._logged_prefill_prefix_decode_rows = True
    self._run_prefill_paged_call(
        route=route,
        q_len=max_query_len_rows,
        seq_len=max_seq_len_hint,
        heads_q=num_heads,
        heads_kv=num_kv_heads,
        head_dim=int(query.shape[2]),
        block_size=int(key_cache.shape[1]),
        fn=run,
    )
    _routing._log_fp8_kv_cache_route(
        "decode",
        self.kv_cache_dtype,
        "xqa_paged" if use_xqa else "scalar_paged",
    )
    _routing._record_route(route)
    out_view.index_copy_(0, plan.src_idx, out_rows)
    return set(plan.rows)


def _run_prefill_paged_call(
    self: _impl.FlashAttnV100Impl,
    *,
    route: str,
    q_len: int,
    seq_len: int,
    heads_q: int,
    heads_kv: int,
    head_dim: int,
    block_size: int,
    fn: Callable[[], torch.Tensor],
) -> torch.Tensor:
    if not envs.VLLM_FLASH_V100_PREFILL_CHUNK_PROFILE:
        return fn()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    out = fn()
    end_event.record()
    torch.accelerator.synchronize()
    logger.info(
        "FLASH_ATTN_V100 prefill chunk profile: route=%s q_len=%d "
        "seq_len=%d heads_q=%d heads_kv=%d head_dim=%d block_size=%d "
        "elapsed_ms=%.3f",
        route,
        q_len,
        seq_len,
        heads_q,
        heads_kv,
        head_dim,
        block_size,
        float(start_event.elapsed_time(end_event)),
    )
    return out


def _flash_v100_prefill_with_prefix(
    self: _impl.FlashAttnV100Impl,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    kv_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Prefill path for prefix/chunked context via gathered contiguous KV."""
    causal = getattr(attn_metadata, "causal", True)
    window_size = self._flash_v100_window_size(causal)
    if self.prefix_anchored_decode_window is None:
        anchor_lens, anchored_window = None, 0
    else:
        anchor_lens, anchored_window = self._anchored_swa_params(attn_metadata)
    if anchor_lens is not None:
        # Fail closed: with the anchored decode-window mask active the
        # KV cache manager evicts gap blocks, so running any unmasked
        # prefill route would silently produce wrong output.
        if not self.use_flash_v100_prefill_paged:
            raise RuntimeError(
                "FLASH_ATTN_V100 anchored decode-window mask requires "
                "the paged prefill kernel; it is disabled or unavailable."
            )
        if not self._flash_prefill_paged_supports_anchor:
            raise RuntimeError(
                "FLASH_ATTN_V100 prefill op does not support the "
                "anchored decode-window mask with this extension build; "
                "rebuild flash_attn_v100."
            )
    num_actual_tokens = attn_metadata.num_actual_tokens
    query = query[:num_actual_tokens]
    out_view = output[:num_actual_tokens]

    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    query_start_loc = (
        query_start_loc_cpu
        if query_start_loc_cpu is not None
        else attn_metadata.query_start_loc
    )
    query_start_loc = _kv_layout._normalize_query_start_loc_for_available_tokens(
        query_start_loc,
        int(query.shape[0]),
    )
    seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
    seq_lens = seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
    num_seqs = len(query_start_loc) - 1

    key_cache, value_cache = _kv_layout._split_paged_kv_cache(kv_cache)
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    debug_compare = os.getenv("VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE", "0") == "1"
    dflash_dump = (
        _debug._dflash_prefix_dump_enabled()
        and not _state._logged_dflash_prefix_dump
        and bool(getattr(layer, "is_dflash_draft_attn", False))
    )

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    max_query_len = int(query_lens.max().item()) if num_seqs > 0 else 0
    if (
        self.use_flash_v100_prefill_paged
        and not causal
        and bool(getattr(layer, "is_dflash_draft_attn", False))
        and anchor_lens is None
        and (
            num_seqs > 1
            or (
                num_seqs == 1
                and (
                    (
                        self._flash_prefill_paged_supports_dflash2_bmhd
                        and block_size in (1024, 2048)
                    )
                    or block_size in self._flash_prefill_paged_dflash2_split_pages
                )
                and max_query_len == 8
                and query.shape[1:] == (8, 128)
                and query.dtype == torch.float16
                and key_cache.dtype == value_cache.dtype == torch.float16
                and window_size == (2047, 2047)
            )
        )
        and 0 < max_query_len <= 16
        and query.shape[0] == num_seqs * max_query_len
        and bool(torch.all(query_lens == max_query_len).item())
        and out_view.is_contiguous()
        and not debug_compare
        and not dflash_dump
    ):
        # The native paged kernel already has a batch grid dimension.
        # Keep all uniform draft rows in one launch instead of capturing
        # one small attention kernel per request. Only the query shape is
        # static: live GPU sequence lengths and block tables must remain
        # inputs so replay can advance or replace individual requests.
        shape = (num_seqs, max_query_len, query.shape[1], head_dim)
        logger.info_once(
            "FLASH_ATTN_V100 DFlash uniform noncausal paged batch route "
            "active (batch=%d, q=%d, page=%d).",
            num_seqs,
            max_query_len,
            block_size,
        )
        _routing._record_route(
            _routing.ROUTE_SPECS["prefill_prefix_dflash_noncausal_batch"].name
        )
        self._run_prefill_paged_call(
            route="prefill_prefix_dflash_noncausal_batch",
            q_len=max_query_len,
            seq_len=int(seq_lens.max().item()),
            heads_q=query.shape[1],
            heads_kv=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            fn=lambda: self.flash_attn_prefill_paged(
                query.reshape(shape),
                key_cache,
                value_cache,
                attn_metadata.block_table[:num_seqs],
                attn_metadata.seq_lens[:num_seqs],
                out=out_view.view(shape),
                softmax_scale=self.scale,
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=float(layer._k_scale_float),
                v_scale=float(layer._v_scale_float),
                causal=False,
                window_size=window_size,
            ),
        )
        return output
    if causal and _masks._ddtree_parent_metadata_requires_branch(
        attn_metadata,
        query_start_loc,
    ):
        if anchor_lens is not None:
            raise RuntimeError(
                "FLASH_ATTN_V100 anchored decode-window mask does not "
                "support ddtree drafting metadata."
            )
        return self._flash_v100_ddtree_small_query_prefill_dense(
            layer,
            query,
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata,
            output,
            query_start_loc,
            seq_lens,
        )

    if (
        causal
        and anchor_lens is None
        and self.use_flash_v100_decode
        and self.smallq_decode_max_query_len > 0
        and max_query_len <= self.smallq_decode_max_query_len
        and (
            self.smallq_decode_max_model_len <= 0
            or getattr(attn_metadata, "max_model_len", 0)
            <= self.smallq_decode_max_model_len
        )
        and not self.use_decode_paged_prefill
    ):
        if not _state._logged_prefill_smallq_decode:
            logger.info(
                "FLASH_ATTN_V100 prefix prefill small-query path active "
                "(paged decode verifier, max_query_len<=%d).",
                self.smallq_decode_max_query_len,
            )
            _state._logged_prefill_smallq_decode = True
        return self._flash_v100_small_query_prefill_as_decode(
            layer,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            output,
            query_start_loc,
            seq_lens,
        )

    decode_rows: set[int] = set()
    if self._prefill_prefix_decode_rows_allowed(
        causal=causal,
        anchor_lens=anchor_lens,
        num_seqs=num_seqs,
        query=query,
        window_size=window_size,
    ):
        decode_rows = self._run_prefill_prefix_decode_rows(
            layer,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            out_view,
            query_start_loc,
            seq_lens,
            window_size,
        )

    for i in range(num_seqs):
        if i in decode_rows:
            continue
        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        if end <= start:
            continue
        out_is_destination = False

        if self.use_flash_v100_prefill_paged:
            q_len = end - start
            seq_len = int(seq_lens[i].item())
            q_seq = query[start:end].unsqueeze(0)
            if anchor_lens is not None:
                # Anchored decode-window mask: single masked paged
                # prefill route; every unmasked fast path is bypassed.
                _routing._record_route(
                    _routing.ROUTE_SPECS["prefill_prefix_paged_anchored"].name
                )
                out_seq = self._run_prefill_paged_call(
                    route="prefill_prefix_paged_anchored",
                    q_len=q_len,
                    seq_len=seq_len,
                    heads_q=query.shape[1],
                    heads_kv=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    fn=lambda q_seq=q_seq, i=i: self.flash_attn_prefill_paged(  # type: ignore[misc]
                        q_seq,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[i : i + 1],
                        attn_metadata.seq_lens[i : i + 1],
                        softmax_scale=self.scale,
                        kv_cache_dtype=self.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                        causal=causal,
                        window_size=window_size,
                        anchor_lens=anchor_lens[i : i + 1],
                        anchored_window=anchored_window,
                    ),
                )
                out_view[start:end].copy_(out_seq.squeeze(0))
                continue
            bfla_block_mask = None
            use_bfla = self._should_use_prefill_bfla(
                q_len=q_len,
                seq_len=seq_len,
                head_dim=head_dim,
                key_cache=key_cache,
                causal=causal,
                window_size=window_size,
            )
            if use_bfla:
                bfla_block_mask = _masks._build_bfla_block_mask_for_seq(
                    q_seq,
                    key_cache,
                    attn_metadata.block_table[i],
                    seq_len=seq_len,
                    block_size=block_size,
                    mask_block_n=self.prefill_bfla_mask_block_n,
                    softmax_scale=self.scale,
                )
            fa2_paged_out = None
            fa2_route = None
            if (
                bfla_block_mask is None
                and envs.VLLM_FLASH_V100_FA2_D256_PREFILL
                and key_cache.dtype == torch.float16
                and value_cache.dtype == torch.float16
                and q_len >= 1024
                and head_dim == 256
                and causal
                and window_size == (-1, -1)
            ):
                cu_q, cu_k = _dense_prefill._uniform_cu_seqlens(
                    q_seq,
                    batch_size=1,
                    query_len=q_len,
                    kv_len=seq_len,
                )
                fa2_out_dest = out_view[start:end].unsqueeze(0)
                fa2_dense_kv = _kv_layout._contiguous_paged_kv_view(
                    key_cache,
                    value_cache,
                    attn_metadata.block_table[i],
                    seq_len,
                    block_size,
                    attn_metadata,
                    i,
                    False,
                )
                fa2_dense_route = "prefill_prefix_contig_splitd_d256"
                if (
                    fa2_dense_kv is None
                    and self._should_use_prefill_gather_dense(
                        q_len=q_len,
                        seq_len=seq_len,
                        head_dim=head_dim,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        causal=causal,
                        window_size=window_size,
                        num_seqs=num_seqs,
                    )
                    and _ops._get_sm70_splitd_d256_ops() is not None
                ):
                    fa2_dense_kv = _kv_layout._gather_paged_kv_to_exact_dense(
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[i],
                        seq_len,
                    )
                    fa2_dense_route = "prefill_prefix_gather_splitd_d256"
                if fa2_dense_kv is not None:
                    fa2_route = fa2_dense_route
                    fa2_key, fa2_value = fa2_dense_kv
                    fa2_paged_out = self._run_prefill_paged_call(
                        route=fa2_route,
                        q_len=q_len,
                        seq_len=seq_len,
                        heads_q=query.shape[1],
                        heads_kv=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        fn=lambda q_seq=q_seq,  # type: ignore[misc]
                        fa2_key=fa2_key,
                        fa2_value=fa2_value,
                        cu_q=cu_q,
                        cu_k=cu_k,
                        q_len=q_len,
                        seq_len=seq_len,
                        out_dest=fa2_out_dest: (
                            _dense_prefill._try_sm70_fa2_d256_prefill(
                                q_seq,
                                fa2_key,
                                fa2_value,
                                cu_seqlens_q=cu_q,
                                cu_seqlens_k=cu_k,
                                max_seqlen_q=q_len,
                                max_seqlen_k=seq_len,
                                softmax_scale=self.scale,
                                causal=causal,
                                window_size=window_size,
                                out=out_dest,
                            )
                        ),
                    )
                else:
                    fa2_route = "prefill_prefix_paged_splitd_d256"
                    fa2_paged_out = self._run_prefill_paged_call(
                        route=fa2_route,
                        q_len=q_len,
                        seq_len=seq_len,
                        heads_q=query.shape[1],
                        heads_kv=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        fn=lambda q_seq=q_seq,  # type: ignore[misc]
                        key_cache=key_cache,
                        value_cache=value_cache,
                        cu_q=cu_q,
                        q_len=q_len,
                        seq_len=seq_len,
                        out_dest=fa2_out_dest,
                        i=i: _dense_prefill._try_sm70_fa2_d256_prefill(
                            q_seq,
                            key_cache,
                            value_cache,
                            cu_seqlens_q=cu_q,
                            cu_seqlens_k=None,
                            max_seqlen_q=q_len,
                            max_seqlen_k=seq_len,
                            softmax_scale=self.scale,
                            causal=causal,
                            window_size=window_size,
                            out=out_dest,
                            seqused_k=attn_metadata.seq_lens[i : i + 1],
                            block_table=attn_metadata.block_table[i : i + 1],
                        ),
                    )
            contig_dense_kv = None
            contig_dense_kv_bhmd = None
            if (
                bfla_block_mask is None
                and fa2_paged_out is None
                and self._should_use_prefill_contig_dense(
                    q_len=q_len,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    key_cache=key_cache,
                    causal=causal,
                    window_size=window_size,
                )
            ):
                if (
                    self.prefill_contig_dense_allow_copy
                    and self.flash_attn_bhmd_func is not None
                ):
                    contig_dense_kv_bhmd = _kv_layout._contiguous_paged_kv_bhmd(
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[i],
                        seq_len,
                        block_size,
                        attn_metadata,
                        i,
                    )
                if contig_dense_kv_bhmd is None:
                    contig_dense_kv = _kv_layout._contiguous_paged_kv_view(
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[i],
                        seq_len,
                        block_size,
                        attn_metadata,
                        i,
                        self.prefill_contig_dense_allow_copy,
                    )
            use_splitkv = self._should_use_prefill_splitkv(
                q_len=q_len,
                seq_len=seq_len,
                head_dim=head_dim,
                key_cache=key_cache,
                causal=causal,
            )
            use_fp8_bridge = self._should_use_fp8_prefill_bridge(
                q_len=q_len,
                head_dim=head_dim,
                key_cache=key_cache,
                value_cache=value_cache,
                causal=causal,
                window_size=window_size,
            )
            if bfla_block_mask is not None:
                if not _state._logged_prefill_prefix_bfla:
                    logger.info(
                        "FLASH_ATTN_V100 prefix prefill BFLA sparse path "
                        "active (min_q=%d min_kv=%d mask_block_n=%d "
                        "keep_mass=%.4f local_blocks=%d pool=%s).",
                        self.prefill_bfla_min_q,
                        self.prefill_bfla_min_kv,
                        self.prefill_bfla_mask_block_n,
                        envs.VLLM_FLASH_V100_BFLA_KEEP_MASS,
                        envs.VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS,
                        envs.VLLM_FLASH_V100_BFLA_POOL,
                    )
                    _state._logged_prefill_prefix_bfla = True
                _routing._record_route(_routing.ROUTE_SPECS["prefill_prefix_bfla"].name)
                out_seq = self._run_prefill_paged_call(
                    route="prefill_prefix_bfla",
                    q_len=q_len,
                    seq_len=seq_len,
                    heads_q=query.shape[1],
                    heads_kv=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    fn=partial(
                        self.flash_attn_prefill_paged_bfla,
                        q_seq,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[i : i + 1],
                        attn_metadata.seq_lens[i : i + 1],
                        bfla_block_mask,
                        self.prefill_bfla_mask_block_n,
                        softmax_scale=self.scale,
                        kv_cache_dtype=self.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                        causal=causal,
                        window_size=window_size,
                    ),
                )
            elif fa2_paged_out is not None:
                if not _dense_prefill._logged_prefill_fa2_d256:
                    logger.info(
                        "FLASH_ATTN_V100 SM70 Split-D D256 software-pipelined "
                        "prefill path active (route=%s).",
                        fa2_route,
                    )
                    _dense_prefill._logged_prefill_fa2_d256 = True
                _routing._record_route(fa2_route or "prefill_prefix_splitd_d256")
                out_seq = fa2_paged_out
                out_is_destination = True
            elif contig_dense_kv_bhmd is not None:
                if not _state._logged_prefill_prefix_contig_dense:
                    logger.info(
                        "FLASH_ATTN_V100 prefix prefill contiguous dense "
                        "BHMD path active (min_q=%d min_kv=%d allow_copy=%s).",
                        self.prefill_contig_dense_min_q,
                        self.prefill_contig_dense_min_kv,
                        str(self.prefill_contig_dense_allow_copy),
                    )
                    _state._logged_prefill_prefix_contig_dense = True
                k_bhmd, v_bhmd = contig_dense_kv_bhmd
                q_bhmd = q_seq.permute(0, 2, 1, 3).contiguous()
                _routing._record_route(
                    _routing.ROUTE_SPECS["prefill_prefix_contig_dense_bhmd"].name
                )
                out_bhmd = self._run_prefill_paged_call(
                    route="prefill_prefix_contig_dense_bhmd",
                    q_len=q_len,
                    seq_len=seq_len,
                    heads_q=query.shape[1],
                    heads_kv=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    fn=lambda q_bhmd=q_bhmd,  # type: ignore[misc]
                    k_bhmd=k_bhmd,
                    v_bhmd=v_bhmd: self.flash_attn_bhmd_func(
                        q_bhmd,
                        k_bhmd,
                        v_bhmd,
                        causal=causal,
                        softmax_scale=self.scale,
                        window_size=window_size,
                    ),
                )
                out_view[start:end].copy_(out_bhmd.squeeze(0).permute(1, 0, 2))
                continue
            elif contig_dense_kv is not None:
                if not _state._logged_prefill_prefix_contig_dense:
                    logger.info(
                        "FLASH_ATTN_V100 prefix prefill contiguous dense "
                        "path active (min_q=%d min_kv=%d).",
                        self.prefill_contig_dense_min_q,
                        self.prefill_contig_dense_min_kv,
                    )
                    _state._logged_prefill_prefix_contig_dense = True
                k_dense, v_dense = contig_dense_kv
                fa2_out = None
                if envs.VLLM_FLASH_V100_FA2_D256_PREFILL:
                    cu_q, cu_k = _dense_prefill._uniform_cu_seqlens(
                        q_seq,
                        batch_size=1,
                        query_len=q_len,
                        kv_len=seq_len,
                    )
                    fa2_out_dest = out_view[start:end].unsqueeze(0)
                    fa2_out = self._run_prefill_paged_call(
                        route="prefill_prefix_contig_dense_fa2_d256",
                        q_len=q_len,
                        seq_len=seq_len,
                        heads_q=query.shape[1],
                        heads_kv=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        fn=lambda q_seq=q_seq,  # type: ignore[misc]
                        k_dense=k_dense,
                        v_dense=v_dense,
                        cu_q=cu_q,
                        cu_k=cu_k,
                        q_len=q_len,
                        seq_len=seq_len,
                        out_dest=fa2_out_dest: (
                            _dense_prefill._try_sm70_fa2_d256_prefill(
                                q_seq,
                                k_dense,
                                v_dense,
                                cu_seqlens_q=cu_q,
                                cu_seqlens_k=cu_k,
                                max_seqlen_q=q_len,
                                max_seqlen_k=seq_len,
                                softmax_scale=self.scale,
                                causal=causal,
                                window_size=window_size,
                                out=out_dest,
                            )
                        ),
                    )
                if fa2_out is not None:
                    if not _dense_prefill._logged_prefill_fa2_d256:
                        logger.info(
                            "FLASH_ATTN_V100 SM70 FA2 D256 "
                            "software-pipelined dense prefill path active."
                        )
                        _dense_prefill._logged_prefill_fa2_d256 = True
                    _routing._record_route(
                        _routing.ROUTE_SPECS[
                            "prefill_prefix_contig_dense_fa2_d256"
                        ].name
                    )
                    out_seq = fa2_out
                    out_is_destination = True
                else:
                    _routing._record_route(
                        _routing.ROUTE_SPECS["prefill_prefix_contig_dense"].name
                    )
                    out_seq = self._run_prefill_paged_call(
                        route="prefill_prefix_contig_dense",
                        q_len=q_len,
                        seq_len=seq_len,
                        heads_q=query.shape[1],
                        heads_kv=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        fn=lambda q_seq=q_seq,  # type: ignore[misc]
                        k_dense=k_dense,
                        v_dense=v_dense: self.flash_attn_func(
                            q_seq,
                            k_dense,
                            v_dense,
                            causal=causal,
                            softmax_scale=self.scale,
                            window_size=window_size,
                        ),
                    )
            elif use_fp8_bridge:
                bridge_result = self._run_fp8_prefill_bridge(
                    query=q_seq,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=attn_metadata.block_table[i : i + 1],
                    seq_lens=attn_metadata.seq_lens[i : i + 1],
                    seq_len=seq_len,
                    k_scale=float(layer._k_scale_float),
                    v_scale=float(layer._v_scale_float),
                    causal=causal,
                    window_size=window_size,
                    out=out_view[start:end].unsqueeze(0),
                )
                if bridge_result is not None:
                    out_seq, out_is_destination = bridge_result
                    if not _state._logged_fp8_prefill_bridge:
                        logger.info(
                            "FLASH_ATTN_V100 %s prefill bridge "
                            "active (one-pass dequant, shared FP16 page-%d "
                            "workspace).",
                            self.kv_cache_dtype,
                            _dense_prefill._FP8_PREFILL_BRIDGE_PAGE_SIZE,
                        )
                        _state._logged_fp8_prefill_bridge = True
                    _routing._record_route(
                        "prefill_prefix_fp8_e4m3_bridge"
                        if self.kv_codec is FP8_E4M3
                        else "prefill_prefix_fp8_e5m2_bridge"
                    )
                else:
                    out_seq = self.flash_attn_prefill_paged(
                        q_seq,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[i : i + 1],
                        attn_metadata.seq_lens[i : i + 1],
                        softmax_scale=self.scale,
                        kv_cache_dtype=self.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                        causal=causal,
                        window_size=window_size,
                    )
            elif use_splitkv:
                if not _state._logged_prefill_prefix_splitkv:
                    logger.info(
                        "FLASH_ATTN_V100 prefix prefill split-KV path active "
                        "(split_kv_tokens=%d min_q=%d max_q=%d min_kv=%d).",
                        self.prefill_split_kv_tokens,
                        self.prefill_split_kv_min_q,
                        self.prefill_split_kv_max_q,
                        self.prefill_split_kv_min_kv,
                    )
                    _state._logged_prefill_prefix_splitkv = True
                _routing._record_route(
                    _routing.ROUTE_SPECS["prefill_prefix_splitkv"].name
                )
                out_seq = self._run_prefill_paged_call(
                    route="prefill_prefix_splitkv",
                    q_len=q_len,
                    seq_len=seq_len,
                    heads_q=query.shape[1],
                    heads_kv=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    fn=lambda q_seq=q_seq,  # type: ignore[misc]
                    i=i,
                    seq_len=seq_len: self.flash_attn_prefill_paged_splitkv(
                        q_seq,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[i : i + 1],
                        attn_metadata.seq_lens[i : i + 1],
                        softmax_scale=self.scale,
                        kv_cache_dtype=self.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                        causal=causal,
                        window_size=window_size,
                        split_kv_tokens=self.prefill_split_kv_tokens,
                        max_seq_len_hint=seq_len,
                    ),
                )
            else:
                out_seq = self._run_prefill_paged_call(
                    route="prefill_prefix_paged",
                    q_len=q_len,
                    seq_len=seq_len,
                    heads_q=query.shape[1],
                    heads_kv=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    fn=lambda q_seq=q_seq, i=i: self.flash_attn_prefill_paged(  # type: ignore[misc]
                        q_seq,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[i : i + 1],
                        attn_metadata.seq_lens[i : i + 1],
                        softmax_scale=self.scale,
                        kv_cache_dtype=self.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                        causal=causal,
                        window_size=window_size,
                    ),
                )
            need_dense_debug = (
                debug_compare and not _state._logged_prefill_compare
            ) or dflash_dump
            if need_dense_debug:
                k_cont, v_cont = _kv_layout._extract_contiguous_kv_from_paged_cache(
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table[i : i + 1],
                    seq_lens=attn_metadata.seq_lens[i : i + 1],
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    total_tokens=seq_len,
                )
                k_cont, v_cont = _kv_layout._dequantize_fp8_contiguous_kv(
                    k_cont,
                    v_cont,
                    self.kv_cache_dtype,
                    float(layer._k_scale_float),
                    float(layer._v_scale_float),
                )
                if bool(getattr(layer, "is_dflash_draft_attn", False)):
                    ref_out = _masks._torch_attention_reference(
                        query[start:end],
                        k_cont,
                        v_cont,
                        causal=causal,
                        softmax_scale=self.scale,
                        window_size=window_size,
                    )
                else:
                    ref_out = self.flash_attn_func(
                        query[start:end].unsqueeze(0),
                        k_cont.unsqueeze(0),
                        v_cont.unsqueeze(0),
                        causal=causal,
                        softmax_scale=self.scale,
                        window_size=window_size,
                    )
                diff = (out_seq - ref_out).abs()
                nan_count = int(torch.isnan(out_seq).sum().item())
                if debug_compare and not _state._logged_prefill_compare:
                    logger.warning(
                        "FLASH_ATTN_V100 debug prefix compare: "
                        "query_len=%d seq_len=%d max_diff=%.8f mean_diff=%.8f "
                        "nan_count=%d q_absmax=%.6f k_absmax=%.6f "
                        "v_absmax=%.6f kv_cache_shape=%s key_shape=%s "
                        "key_stride=%s value_stride=%s key_contig=%s "
                        "value_contig=%s",
                        end - start,
                        seq_len,
                        float(diff.max().item()),
                        float(diff.mean().item()),
                        nan_count,
                        float(query[start:end].abs().max().item()),
                        float(k_cont.abs().max().item()),
                        float(v_cont.abs().max().item()),
                        tuple(kv_cache.shape),
                        tuple(key_cache.shape),
                        tuple(key_cache.stride()),
                        tuple(value_cache.stride()),
                        str(key_cache.is_contiguous()),
                        str(value_cache.is_contiguous()),
                    )
                if dflash_dump:
                    slot_mapping = getattr(attn_metadata, "slot_mapping", None)
                    slot_slice = None
                    cache_k_by_slot = None
                    cache_v_by_slot = None
                    key_input = None
                    value_input = None
                    slot_k_diff = None
                    slot_v_diff = None
                    tail_k_diff = None
                    tail_v_diff = None
                    if (
                        slot_mapping is not None
                        and key is not None
                        and value is not None
                        and key_cache.dtype != torch.uint8
                    ):
                        slot_slice = slot_mapping[start:end].to(torch.long)
                        valid_slots = slot_slice >= 0
                        if bool(valid_slots.all().item()):
                            slot_blocks = torch.div(
                                slot_slice,
                                block_size,
                                rounding_mode="floor",
                            )
                            slot_offsets = torch.remainder(slot_slice, block_size)
                            cache_k_by_slot = key_cache[slot_blocks, slot_offsets]
                            cache_v_by_slot = value_cache[
                                slot_blocks,
                                slot_offsets,
                            ]
                            cache_k_by_slot, cache_v_by_slot = (
                                _kv_layout._dequantize_fp8_contiguous_kv(
                                    cache_k_by_slot,
                                    cache_v_by_slot,
                                    self.kv_cache_dtype,
                                    float(layer._k_scale_float),
                                    float(layer._v_scale_float),
                                )
                            )
                            key_input = key[start:end]
                            value_input = value[start:end]
                            slot_k_diff = (cache_k_by_slot - key_input).abs()
                            slot_v_diff = (cache_v_by_slot - value_input).abs()
                            tail_start = max(0, seq_len - (end - start))
                            tail_k = k_cont[tail_start:seq_len]
                            tail_v = v_cont[tail_start:seq_len]
                            if tail_k.shape == key_input.shape:
                                tail_k_diff = (tail_k - key_input).abs()
                                tail_v_diff = (tail_v - value_input).abs()

                    dump_path = (
                        f"/tmp/flash_v100_dflash_prefix_dump_pid{os.getpid()}_seq{i}.pt"
                    )
                    torch.save(
                        {
                            "layer_name": self._layer_debug_info(layer).get(
                                "layer_name"
                            ),
                            "causal": causal,
                            "window_size": window_size,
                            "query_start_loc": query_start_loc.detach().cpu(),
                            "seq_lens": seq_lens.detach().cpu(),
                            "attn_seq_lens": attn_metadata.seq_lens.detach().cpu(),
                            "block_table": attn_metadata.block_table[i : i + 1]
                            .detach()
                            .cpu(),
                            "slot_mapping": None
                            if slot_slice is None
                            else slot_slice.detach().cpu(),
                            "query": query[start:end].detach().cpu(),
                            "key_input": None
                            if key_input is None
                            else key_input.detach().cpu(),
                            "value_input": None
                            if value_input is None
                            else value_input.detach().cpu(),
                            "cache_k_by_slot": None
                            if cache_k_by_slot is None
                            else cache_k_by_slot.detach().cpu(),
                            "cache_v_by_slot": None
                            if cache_v_by_slot is None
                            else cache_v_by_slot.detach().cpu(),
                            "k_cont_tail": k_cont[
                                max(0, seq_len - (end - start)) : seq_len
                            ]
                            .detach()
                            .cpu(),
                            "v_cont_tail": v_cont[
                                max(0, seq_len - (end - start)) : seq_len
                            ]
                            .detach()
                            .cpu(),
                            "k_cont": k_cont.detach().cpu(),
                            "v_cont": v_cont.detach().cpu(),
                            "out_seq": out_seq.detach().cpu(),
                            "ref_out": ref_out.detach().cpu(),
                            "paged_vs_dense_max": float(diff.max().item()),
                            "paged_vs_dense_mean": float(diff.mean().item()),
                            "slot_k_max": None
                            if slot_k_diff is None
                            else float(slot_k_diff.max().item()),
                            "slot_v_max": None
                            if slot_v_diff is None
                            else float(slot_v_diff.max().item()),
                            "tail_k_max": None
                            if tail_k_diff is None
                            else float(tail_k_diff.max().item()),
                            "tail_v_max": None
                            if tail_v_diff is None
                            else float(tail_v_diff.max().item()),
                            "kv_cache_shape": tuple(kv_cache.shape),
                            "key_cache_shape": tuple(key_cache.shape),
                            "key_cache_stride": tuple(key_cache.stride()),
                            "value_cache_stride": tuple(value_cache.stride()),
                        },
                        dump_path,
                    )
                    logger.warning(
                        "FLASH_ATTN_V100 saved DFlash prefix dump to %s "
                        "(paged_vs_dense_max=%.8f slot_k_max=%s tail_k_max=%s)",
                        dump_path,
                        float(diff.max().item()),
                        "n/a"
                        if slot_k_diff is None
                        else f"{float(slot_k_diff.max().item()):.8f}",
                        "n/a"
                        if tail_k_diff is None
                        else f"{float(tail_k_diff.max().item()):.8f}",
                    )
                    _state._logged_dflash_prefix_dump = True
                if (
                    debug_compare
                    and not _state._logged_prefill_compare
                    and nan_count > 0
                ):
                    dump_path = f"/tmp/flash_v100_prefill_nan_dump_pid{os.getpid()}.pt"
                    torch.save(
                        {
                            "query": query[start:end].detach().cpu(),
                            "key_cache": key_cache.detach().cpu(),
                            "value_cache": value_cache.detach().cpu(),
                            "block_table": attn_metadata.block_table[i : i + 1]
                            .detach()
                            .cpu(),
                            "seq_lens": attn_metadata.seq_lens[i : i + 1]
                            .detach()
                            .cpu(),
                            "k_cont": k_cont.detach().cpu(),
                            "v_cont": v_cont.detach().cpu(),
                            "out_seq": out_seq.detach().cpu(),
                            "ref_out": ref_out.detach().cpu(),
                        },
                        dump_path,
                    )
                    logger.warning(
                        "FLASH_ATTN_V100 saved failing prefix prefill dump to %s",
                        dump_path,
                    )
                if debug_compare and not _state._logged_prefill_compare:
                    _state._logged_prefill_compare = True
        else:
            seq_len = int(seq_lens[i].item())
            k_cont, v_cont = _kv_layout._extract_contiguous_kv_from_paged_cache(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table[i : i + 1],
                seq_lens=attn_metadata.seq_lens[i : i + 1],
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=seq_len,
            )
            k_cont, v_cont = _kv_layout._dequantize_fp8_contiguous_kv(
                k_cont,
                v_cont,
                self.kv_cache_dtype,
                float(layer._k_scale_float),
                float(layer._v_scale_float),
            )

            out_seq = self.flash_attn_func(
                query[start:end].unsqueeze(0),
                k_cont.unsqueeze(0),
                v_cont.unsqueeze(0),
                causal=causal,
                softmax_scale=self.scale,
                window_size=window_size,
            )
        if not out_is_destination:
            out_view[start:end].copy_(out_seq.squeeze(0))

    return output
