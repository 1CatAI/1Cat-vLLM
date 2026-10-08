# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 verify methods, bound by impl."""

from __future__ import annotations

import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import debug as _debug
from vllm.v1.attention.backends.flash_v100 import impl as _impl
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import masks as _masks
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100 import state as _state
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)
from vllm.v1.attention.kv_codecs import (
    FP8_E5M2,
)
from vllm.v1.attention.ops.sm70_grouped import (
    grouped_e4m3_fp32_allowed,
    grouped_fp16_fp32_reason,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def _validate_dflash_attention_contract(
    self: _impl.FlashAttnV100Impl,
    layer: torch.nn.Module,
    attn_metadata: TritonAttentionMetadata,
) -> None:
    if not getattr(layer, "is_dflash_draft_attn", False):
        return

    actual_causal = bool(getattr(attn_metadata, "causal", True))
    expected_causal = getattr(layer, "dflash_expected_causal", None)
    if expected_causal is None:
        raise RuntimeError(
            "FLASH_ATTN_V100 DFlash attention is missing its declared "
            "causality contract."
        )
    expected_causal = bool(expected_causal)
    if actual_causal != expected_causal:
        raise RuntimeError(
            "FLASH_ATTN_V100 DFlash causality mismatch: "
            f"model={expected_causal} metadata={actual_causal}."
        )

    declared_window = getattr(layer, "dflash_expected_sliding_window", None)
    expected_window = (
        (-1, -1)
        if declared_window is None
        else (
            int(declared_window) - 1,
            0 if expected_causal else int(declared_window) - 1,
        )
    )
    actual_window = self._flash_v100_window_size(actual_causal)
    if actual_window != expected_window:
        raise RuntimeError(
            "FLASH_ATTN_V100 DFlash sliding-window mismatch: "
            f"model={expected_window} backend={actual_window}."
        )

    signature = (
        getattr(layer, "layer_name", None),
        actual_causal,
        actual_window,
        getattr(layer, "dflash_rope_is_neox_style", None),
    )
    if signature not in _state._logged_dflash_attention_contracts:
        _state._logged_dflash_attention_contracts.add(signature)
        logger.info(
            "FLASH_ATTN_V100 DFlash attention contract: layer=%s "
            "causal=%s window=%s rope_neox=%s.",
            signature[0],
            actual_causal,
            actual_window,
            signature[3],
        )


def _dflash2_grouped_verify_allowed(
    self: _impl.FlashAttnV100Impl,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    *,
    num_query_tokens: int,
) -> bool:
    """Gate the verifier on its hardware and tensor-layout contract."""
    block_table = getattr(attn_metadata, "block_table", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    num_reqs = int(
        getattr(
            attn_metadata,
            "num_reqs",
            0 if block_table is None else block_table.shape[0],
        )
    )
    max_query_len = int(
        getattr(
            attn_metadata,
            "max_query_len",
            num_query_tokens if num_reqs == 1 else 0,
        )
    )
    single_request_shape = bool(
        num_reqs == 1
        and num_query_tokens in (8, 16)
        and num_query_tokens <= self.dflash2_grouped_verify_max_query_tokens
    )
    batched_request_shape = bool(
        self.use_dflash2_batched_grouped_verify
        and self.dflash2_grouped_verify_request_major_abi_version >= 1
        and num_reqs in (2, 4, 8)
        and max_query_len == 8
        and num_query_tokens == num_reqs * 8
    )
    allowed = bool(
        self.use_dflash2_grouped_verify
        and (single_request_shape or batched_request_shape)
        and self.flash_attn_grouped_verify_paged is not None
        and getattr(attn_metadata, "is_dflash_selector_target", False)
        and getattr(attn_metadata, "max_model_len", 0)
        >= self.dflash2_grouped_verify_min_model_len
        and getattr(attn_metadata, "causal", True)
        and self._flash_v100_window_size(causal=True) == (-1, -1)
        and tuple(query.shape) == (num_query_tokens, 6, 256)
        and query.dtype == torch.float16
        and query.is_contiguous()
        and key_cache.ndim == 4
        and value_cache.ndim == 4
        and key_cache.device == query.device
        and value_cache.device == query.device
        # q15 LABD increases the aligned hybrid-cache page from the
        # block-8 service's 1648/3296 layout to 1728/3456. The grouped
        # operator's runtime-stride implementation is exact for both.
        and key_cache.shape[1] in (1648, 1728, 3296, 3456)
        and tuple(key_cache.shape[2:]) == (1, 256)
        and tuple(value_cache.shape) == tuple(key_cache.shape)
        and key_cache.dtype == torch.uint8
        and value_cache.dtype == torch.uint8
        and key_cache.stride(-1) == 1
        and value_cache.stride(-1) == 1
        # This legacy verifier stores normalized partials in FP16.
        # E4M3 must reach the repaired FP32 path below, including when
        # the old native entry advertises E4M3 byte-format support.
        and self.kv_codec is FP8_E5M2
        and block_table is not None
        and block_table.ndim == 2
        and block_table.shape[0] == num_reqs
        and block_table.device == query.device
        and block_table.dtype == torch.int32
        and block_table.is_contiguous()
        and seq_lens is not None
        and seq_lens.ndim == 1
        and seq_lens.shape[0] == num_reqs
        and seq_lens.device == query.device
        and seq_lens.dtype == torch.int32
        and seq_lens.is_contiguous()
    )
    if (
        self.use_dflash2_grouped_verify
        and not allowed
        and not _state._logged_prefill_smallq_grouped_verify_gate
    ):
        logger.info(
            "FLASH_ATTN_V100 DFlash2 grouped verifier gate rejected: "
            "op=%s marker=%s max_model_len=%s min_model_len=%s "
            "causal=%s window=%s reqs=%d max_q=%d actual=%d "
            "native_max_q=%d q=%s/%s "
            "k=%s/%s v=%s/%s kv_dtype=%s block_table=%s/%s "
            "seq_lens=%s/%s.",
            self.flash_attn_grouped_verify_paged is not None,
            getattr(attn_metadata, "is_dflash_selector_target", False),
            getattr(attn_metadata, "max_model_len", None),
            self.dflash2_grouped_verify_min_model_len,
            getattr(attn_metadata, "causal", True),
            self._flash_v100_window_size(causal=True),
            num_reqs,
            max_query_len,
            num_query_tokens,
            self.dflash2_grouped_verify_max_query_tokens,
            tuple(query.shape),
            query.dtype,
            tuple(key_cache.shape),
            key_cache.dtype,
            tuple(value_cache.shape),
            value_cache.dtype,
            self.kv_cache_dtype,
            None if block_table is None else tuple(block_table.shape),
            None if block_table is None else block_table.dtype,
            None if seq_lens is None else tuple(seq_lens.shape),
            None if seq_lens is None else seq_lens.dtype,
        )
        _state._logged_prefill_smallq_grouped_verify_gate = True
    return allowed


def _call_dflash2_grouped_verify(
    self: _impl.FlashAttnV100Impl,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    *,
    out: torch.Tensor,
) -> None:
    num_reqs = int(attn_metadata.block_table.shape[0])
    if not _state._logged_prefill_smallq_grouped_verify:
        logger.info(
            "FLASH_ATTN_V100 DFlash2 exact grouped verifier active "
            "(request-major B%d/q%d/H6/Hkv1/D256, %s KV, one-pass).",
            num_reqs,
            query.shape[0] // num_reqs,
            self.kv_cache_dtype,
        )
        _state._logged_prefill_smallq_grouped_verify = True
    self.flash_attn_grouped_verify_paged(
        query,
        key_cache,
        value_cache,
        attn_metadata.block_table[:num_reqs],
        attn_metadata.seq_lens[:num_reqs],
        softmax_scale=self.scale,
        out=out,
        kv_cache_dtype=self.kv_cache_dtype,
        k_scale=float(layer._k_scale_float),
        v_scale=float(layer._v_scale_float),
        one_pass=True,
    )
    _routing._log_fp8_kv_cache_route(
        "decode", self.kv_cache_dtype, "dflash2_grouped_verify"
    )
    _routing._record_route(
        _routing.ROUTE_SPECS["prefill_smallq_dflash2_grouped_verify"].name
    )


def _smallq_decode_xqa_allowed(
    self: _impl.FlashAttnV100Impl,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    *,
    window_size: tuple[int, int],
    max_seq_len_hint: int | None,
    workspace_seq_capacity_hint: int | None,
    partition_size_hint: int | None,
) -> bool:
    context = _routing.RouteContext(
        stage="verify",
        codec=self._xqa_kv_codec(key_cache, value_cache, attn_metadata),
        shape=_routing.RouteShape(
            query.shape[0],
            query.shape[1],
            key_cache.shape[2],
            query.shape[2],
            key_cache.shape[1],
        ),
        enabled=self.use_smallq_decode_xqa,
        available=self.flash_attn_decode_paged_xqa is not None,
        query=query,
        metadata=attn_metadata,
        seq_rows=seq_lens.shape[0],
        max_seq_len_hint=max_seq_len_hint,
        workspace_seq_capacity_hint=workspace_seq_capacity_hint,
        partition_size_hint=partition_size_hint,
        window_size=window_size,
    )
    return (
        _routing.route_reason(
            _routing.ROUTE_SPECS["prefill_smallq_decode_xqa"], context
        )
        is None
    )


def _call_flash_attn_smallq_decode_paged(
    self: _impl.FlashAttnV100Impl,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    *,
    out: torch.Tensor,
    max_seq_len_hint: int | None,
    workspace_seq_capacity_hint: int | None,
    partition_size_hint: int | None,
) -> None:
    fp16_grouped = getattr(self, "flash_attn_grouped_fp16_fp32_paged", None)
    if (
        fp16_grouped is not None
        and grouped_fp16_fp32_reason(
            self,
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            attn_metadata,
            out=out,
            partition_size_hint=partition_size_hint,
        )
        is None
    ):
        fp16_grouped(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            seq_lens,
            out=out,
            softmax_scale=self.scale,
        )
        logger.info_once(
            "FLASH_ATTN_V100 FP16 KV grouped verifier active "
            "(FP32 probability/PV/numerator/max/sum, page=%d).",
            key_cache.shape[1],
            scope="process",
        )
        _routing._record_route(
            _routing.ROUTE_SPECS["prefill_smallq_fp16_grouped_fp32"].name
        )
        return
    grouped_op = getattr(self, "flash_attn_grouped_e4m3_fp32_paged", None)
    if grouped_op is not None and grouped_e4m3_fp32_allowed(
        self,
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        attn_metadata,
        out=out,
        partition_size_hint=partition_size_hint,
    ):
        # Preserve the builder's device row lengths. In particular, padded
        # rows must not move the causal boundary of the preceding queries.
        grouped_op(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            seq_lens,
            out=out,
            softmax_scale=self.scale,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
        )
        logger.info_once(
            "FLASH_ATTN_V100 E4M3 grouped FP32 route selected "
            "(rows=%d, page=%d, FP32 numerator/max/sum, explicit row lengths).",
            query.shape[0],
            key_cache.shape[1],
            scope="process",
        )
        _routing._log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "grouped_fp32")
        _routing._record_route(
            _routing.ROUTE_SPECS["prefill_smallq_e4m3_grouped_fp32"].name
        )
        return
    window_size = self._flash_v100_window_size(causal=True)
    if self._smallq_decode_xqa_allowed(
        query,
        key_cache,
        value_cache,
        seq_lens,
        attn_metadata,
        window_size=window_size,
        max_seq_len_hint=max_seq_len_hint,
        workspace_seq_capacity_hint=workspace_seq_capacity_hint,
        partition_size_hint=partition_size_hint,
    ):
        verifier_partition_size_hint = (
            _routing._mtp5_xqa_dual_cta_partition_size_hint()
            if (
                query.shape[0] == 5
                and query.shape[2] == 256
                and key_cache.shape[1] == 1616
                and key_cache.shape[2] > 0
                and query.shape[1] == 6 * key_cache.shape[2]
                and self.kv_codec is FP8_E5M2
                and FP8_E5M2.stores(key_cache, value_cache)
            )
            else None
        )
        if not _state._logged_prefill_smallq_decode_xqa:
            logger.info(
                "FLASH_ATTN_V100 MTP verifier XQA path active "
                "(rows=%d, q_per_kv=%d, partition_hint=%s, "
                "mtp5_dual_cta=%s).",
                int(query.shape[0]),
                int(query.shape[1] // key_cache.shape[2]),
                verifier_partition_size_hint,
                verifier_partition_size_hint is not None,
            )
            _state._logged_prefill_smallq_decode_xqa = True
        _routing._log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "xqa_paged")
        self.flash_attn_decode_paged_xqa(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=self.scale,
            out=out,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            window_size=window_size,
            max_seq_len_hint=max_seq_len_hint,
            workspace_seq_capacity_hint=workspace_seq_capacity_hint,
            partition_size_hint=verifier_partition_size_hint,
            batch_context_routing=bool(
                getattr(
                    attn_metadata,
                    "flash_v100_batch_context_routing",
                    False,
                )
            ),
        )
        _routing._record_route(_routing.ROUTE_SPECS["prefill_smallq_decode_xqa"].name)
        return

    self._call_flash_attn_decode_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale=self.scale,
        out=out,
        kv_cache_dtype=self.kv_cache_dtype,
        k_scale=float(layer._k_scale_float),
        v_scale=float(layer._v_scale_float),
        window_size=window_size,
        max_seq_len_hint=max_seq_len_hint,
        workspace_seq_capacity_hint=workspace_seq_capacity_hint,
        partition_size_hint=partition_size_hint,
    )
    _routing._record_route(_routing.ROUTE_SPECS["prefill_smallq_decode_scalar"].name)


def _small_query_decode_enabled(
    self: _impl.FlashAttnV100Impl,
    attn_metadata: TritonAttentionMetadata,
) -> bool:
    if (
        not getattr(attn_metadata, "causal", True)
        or not self.use_flash_v100_decode
        or self.smallq_decode_max_query_len <= 0
    ):
        return False
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    query_start_loc = (
        query_start_loc_cpu
        if query_start_loc_cpu is not None
        else attn_metadata.query_start_loc
    )
    if len(query_start_loc) <= 1:
        return False

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    max_query_len = int(query_lens.max().item())
    max_model_len = getattr(attn_metadata, "max_model_len", 0)
    model_len_supported = (
        self.smallq_decode_max_model_len <= 0
        or max_model_len <= self.smallq_decode_max_model_len
    )
    return max_query_len <= self.smallq_decode_max_query_len and model_len_supported


def _flash_v100_ddtree_small_query_prefill_dense(
    self: _impl.FlashAttnV100Impl,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Correctness bridge for branched DDTree verifier attention."""

    is_capturing = _routing._is_cuda_graph_capturing(query)
    parent_ids = getattr(attn_metadata, "ddtree_parent_ids", None)
    num_tree_tokens_cpu = getattr(attn_metadata, "ddtree_num_tree_tokens_cpu", None)
    num_reqs = min(
        max(0, len(query_start_loc) - 1),
        int(parent_ids.shape[0]) if parent_ids is not None else 0,
        int(num_tree_tokens_cpu.numel()) if num_tree_tokens_cpu is not None else 0,
    )
    window_size = self._flash_v100_window_size(causal=True)
    if (
        _debug._dflash_ddtree_triton_branch_attn_enabled()
        and parent_ids is not None
        and _masks._ddtree_triton_seq_lens_match(
            attn_metadata,
            seq_lens,
            num_reqs,
        )
        and _masks._ddtree_triton_query_start_loc_match(
            attn_metadata,
            query_start_loc,
            num_reqs,
        )
    ):
        triton_parent_ids = _masks._ddtree_triton_parent_ids_for_query(
            parent_ids,
            num_tree_tokens_cpu,
            query_start_loc,
            is_capturing=is_capturing,
        )
        if triton_parent_ids is not None:
            try:
                from vllm.v1.attention.backends.ddtree_branch_triton import (
                    ddtree_branch_attention_correction,
                )

                ddtree_branch_attention_correction(
                    impl=self,
                    query=query,
                    key=key,
                    value=value,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    output=output,
                    attn_metadata=attn_metadata,
                    parent_ids=triton_parent_ids,
                    window_size=window_size,
                )
            except Exception:
                if is_capturing or _debug._dflash_ddtree_triton_branch_attn_strict():
                    raise
                if _routing._ddtree_trace_enabled():
                    _routing._ddtree_trace_event(
                        "flash_ddtree_attention_route",
                        {
                            "route": "triton_exception_fallback",
                            "num_reqs": num_reqs,
                            "num_actual_tokens": int(
                                getattr(attn_metadata, "num_actual_tokens", 0)
                            ),
                            "query_start_loc": query_start_loc.detach().cpu().tolist(),
                            "seq_lens": seq_lens.detach().cpu().tolist(),
                            "tree_tokens": (
                                num_tree_tokens_cpu.detach().cpu().tolist()
                                if num_tree_tokens_cpu is not None
                                else None
                            ),
                        },
                    )
                if not _state._logged_prefill_ddtree_triton_fallback:
                    logger.exception(
                        "FLASH_ATTN_V100 DDTree Triton verifier failed; "
                        "falling back to dense masked verifier."
                    )
                    _state._logged_prefill_ddtree_triton_fallback = True
            else:
                if not _state._logged_prefill_ddtree_triton:
                    logger.info(
                        "FLASH_ATTN_V100 DDTree branched verifier path active "
                        "(Triton paged-KV ancestor mask)."
                    )
                _state._logged_prefill_ddtree_triton = True
                _routing._record_route(
                    _routing.ROUTE_SPECS["prefill_ddtree_triton"].name
                )
                if _routing._ddtree_trace_enabled():
                    _routing._ddtree_trace_event(
                        "flash_ddtree_attention_route",
                        {
                            "route": "triton",
                            "num_reqs": num_reqs,
                            "num_actual_tokens": int(
                                getattr(attn_metadata, "num_actual_tokens", 0)
                            ),
                            "query_start_loc": query_start_loc.detach().cpu().tolist(),
                            "seq_lens": seq_lens.detach().cpu().tolist(),
                            "tree_tokens": (
                                num_tree_tokens_cpu.detach().cpu().tolist()
                                if num_tree_tokens_cpu is not None
                                else None
                            ),
                        },
                    )
                return output

    if is_capturing:
        raise RuntimeError(
            "FLASH_ATTN_V100 DDTree dense verifier fallback is not "
            "CUDA-graph safe and the Triton branch verifier is disabled "
            "or unavailable."
        )

    parent_ids_cpu = _masks._ddtree_parent_ids_cpu(attn_metadata)
    if parent_ids_cpu is None or num_tree_tokens_cpu is None:
        raise RuntimeError("DDTree dense verifier fallback requires parent metadata")

    if not _state._logged_prefill_ddtree_dense:
        logger.info(
            "FLASH_ATTN_V100 DDTree branched verifier path active "
            "(dense masked small-query fallback)."
        )
        _state._logged_prefill_ddtree_dense = True

    _routing._record_route(_routing.ROUTE_SPECS["prefill_ddtree_dense"].name)
    if _routing._ddtree_trace_enabled():
        _routing._ddtree_trace_event(
            "flash_ddtree_attention_route",
            {
                "route": "dense",
                "num_reqs": num_reqs,
                "num_actual_tokens": int(
                    getattr(attn_metadata, "num_actual_tokens", 0)
                ),
                "query_start_loc": query_start_loc.detach().cpu().tolist(),
                "seq_lens": seq_lens.detach().cpu().tolist(),
                "tree_tokens": num_tree_tokens_cpu.detach().cpu().tolist(),
            },
        )
    trace_kv_diff = os.getenv("VLLM_DFLASH_DDTREE_TRACE_KV_CACHE_DIFF", "0") == "1"
    profile_enabled = envs.VLLM_FLASH_V100_PREFILL_CHUNK_PROFILE
    profile_start: torch.cuda.Event | None = None
    profile_end: torch.cuda.Event | None = None
    if profile_enabled:
        profile_start = torch.cuda.Event(enable_timing=True)
        profile_end = torch.cuda.Event(enable_timing=True)
        profile_start.record()
    num_seqs = len(query_start_loc) - 1
    total_query_tokens = 0
    total_tree_tokens = 0
    max_seq_len = 0
    out_view = output[: attn_metadata.num_actual_tokens]
    for req_idx in range(num_seqs):
        start = int(query_start_loc[req_idx].item())
        end = int(query_start_loc[req_idx + 1].item())
        q_len = end - start
        if q_len <= 0:
            continue

        seq_len = int(seq_lens[req_idx].item())
        if seq_len <= 0:
            continue
        total_query_tokens += q_len
        max_seq_len = max(max_seq_len, seq_len)
        prefix_len = max(seq_len - q_len, 0)
        tree_len = (
            int(num_tree_tokens_cpu[req_idx].item())
            if req_idx < int(num_tree_tokens_cpu.numel())
            else 0
        )
        total_tree_tokens += max(tree_len, 0)
        parent_row = (
            parent_ids_cpu[req_idx] if req_idx < int(parent_ids_cpu.shape[0]) else None
        )

        if trace_kv_diff:
            slot_mapping = getattr(attn_metadata, "slot_mapping", None)
            if (
                slot_mapping is not None
                and key is not None
                and value is not None
                and end <= int(slot_mapping.numel())
            ):
                slot_slice = slot_mapping[start:end].to(torch.long)
                valid_slots = slot_slice >= 0
                if bool(valid_slots.all().item()):
                    slot_blocks = torch.div(
                        slot_slice,
                        key_cache.shape[1],
                        rounding_mode="floor",
                    )
                    slot_offsets = torch.remainder(slot_slice, key_cache.shape[1])
                    cache_k_by_slot = key_cache[slot_blocks, slot_offsets]
                    cache_v_by_slot = value_cache[slot_blocks, slot_offsets]
                    cache_k_by_slot, cache_v_by_slot = (
                        _kv_layout._dequantize_fp8_contiguous_kv(
                            cache_k_by_slot,
                            cache_v_by_slot,
                            self.kv_cache_dtype,
                            float(layer._k_scale_float),
                            float(layer._v_scale_float),
                        )
                    )
                    key_diff = (cache_k_by_slot - key[start:end]).abs()
                    value_diff = (cache_v_by_slot - value[start:end]).abs()
                    _routing._ddtree_trace_event(
                        "flash_ddtree_kv_cache_diff",
                        {
                            "layer": str(
                                self._layer_debug_info(layer).get("layer_name")
                            ),
                            "req_idx": req_idx,
                            "query_start": start,
                            "query_end": end,
                            "seq_len": seq_len,
                            "prefix_len": prefix_len,
                            "tree_len": tree_len,
                            "key_max_diff": float(key_diff.max().item()),
                            "key_mean_diff": float(key_diff.mean().item()),
                            "value_max_diff": float(value_diff.max().item()),
                            "value_mean_diff": float(value_diff.mean().item()),
                        },
                    )

        k_cont, v_cont = _kv_layout._extract_contiguous_kv_from_paged_cache(
            (key_cache, value_cache),
            attn_metadata.block_table[req_idx : req_idx + 1],
            attn_metadata.seq_lens[req_idx : req_idx + 1],
            key_cache.shape[2],
            key_cache.shape[3],
            key_cache.shape[1],
            total_tokens=seq_len,
        )
        k_cont, v_cont = _kv_layout._dequantize_fp8_contiguous_kv(
            k_cont,
            v_cont,
            self.kv_cache_dtype,
            float(layer._k_scale_float),
            float(layer._v_scale_float),
        )
        if prefix_len + q_len <= k_cont.shape[0]:
            k_cont[prefix_len : prefix_len + q_len].copy_(key[start:end])
            v_cont[prefix_len : prefix_len + q_len].copy_(value[start:end])

        q_seq = query[start:end]
        q_f = q_seq.float()
        k_f = k_cont.float()
        v_f = v_cont.float()
        if q_f.shape[1] % k_f.shape[1] != 0:
            raise ValueError(
                "DDTree dense verifier requires Q heads divisible by KV heads, "
                f"got {q_f.shape[1]} and {k_f.shape[1]}"
            )
        if q_f.shape[1] != k_f.shape[1]:
            repeat = q_f.shape[1] // k_f.shape[1]
            k_f = k_f.repeat_interleave(repeat, dim=1)
            v_f = v_f.repeat_interleave(repeat, dim=1)

        scores = torch.einsum("mhd,nhd->hmn", q_f, k_f) * self.scale
        visible = _masks._build_ddtree_visibility_mask(
            q_len=q_len,
            seq_len=seq_len,
            prefix_len=prefix_len,
            tree_len=tree_len,
            parent_row=parent_row,
            device=query.device,
            window_size=window_size,
        )
        scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out_seq = torch.einsum("hmn,nhd->mhd", probs, v_f)
        out_view[start:end].copy_(out_seq.to(dtype=query.dtype))

    if profile_start is not None and profile_end is not None:
        profile_end.record()
        torch.accelerator.synchronize()
        logger.info(
            "FLASH_ATTN_V100 prefill chunk profile: route=%s layer=%s "
            "elapsed_ms=%.3f query_tokens=%d tree_tokens=%d max_seq_len=%d "
            "heads_q=%d heads_kv=%d head_dim=%d",
            "prefill_ddtree_dense",
            self._layer_debug_info(layer).get("layer_name"),
            float(profile_start.elapsed_time(profile_end)),
            total_query_tokens,
            total_tree_tokens,
            max_seq_len,
            int(query.shape[1]),
            int(key_cache.shape[2]),
            int(key_cache.shape[3]),
        )

    return output


def _flash_v100_small_query_prefill_as_decode(
    self: _impl.FlashAttnV100Impl,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    _seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Run small causal prefix-prefill queries through paged decode.

    MTP verification presents a tiny query span over a long KV prefix. The
    paged prefill kernel is correct, but its work scheduling is much more
    expensive for this shape and exceeds SM70 shared-memory limits at very
    long contexts. Treating every query token as an independent decode row
    with an increasing seq_len preserves the causal mask without exposing
    future draft tokens.
    """
    device = attn_metadata.seq_lens.device
    dtype = attn_metadata.seq_lens.dtype

    num_query_tokens = min(
        int(attn_metadata.num_actual_tokens),
        int(query.shape[0]),
        int(output.shape[0]),
    )
    if self.use_dflash2_grouped_verify and self._dflash2_grouped_verify_allowed(
        query,
        key_cache,
        value_cache,
        attn_metadata,
        num_query_tokens=num_query_tokens,
    ):
        query = query[:num_query_tokens]
        out_view = output[:num_query_tokens]
        self._call_dflash2_grouped_verify(
            layer,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            out=out_view,
        )
        return output

    persistent_decode_block_table = getattr(
        attn_metadata,
        "smallq_decode_block_table",
        None,
    )
    persistent_decode_seq_lens = getattr(
        attn_metadata,
        "smallq_decode_seq_lens",
        None,
    )
    persistent_query_start_loc = getattr(
        attn_metadata,
        "smallq_query_start_loc",
        None,
    )
    if (
        persistent_decode_block_table is not None
        and persistent_decode_seq_lens is not None
        and persistent_query_start_loc is not None
        and int(persistent_decode_seq_lens.shape[0]) >= num_query_tokens
        and int(persistent_decode_block_table.shape[0]) >= num_query_tokens
        and not _kv_layout._metadata_expects_more_query_tokens_than_available(
            attn_metadata,
            num_query_tokens,
        )
    ):
        query = query[:num_query_tokens]
        out_view = output[:num_query_tokens]
        if _debug._draft_graph_debug_enabled():
            _debug._graph_metadata_debug_log(
                "smallq_call",
                "layer=%s num_query_tokens=%s %s %s %s %s %s",
                self._layer_debug_info(layer).get("layer_name"),
                num_query_tokens,
                _debug._format_tensor_debug(query, "query"),
                _debug._format_tensor_debug(out_view, "out"),
                _debug._format_tensor_debug(
                    persistent_decode_block_table[:num_query_tokens],
                    "smallq_bt",
                ),
                _debug._format_tensor_debug(
                    persistent_decode_seq_lens[:num_query_tokens],
                    "smallq_seq",
                ),
                _debug._format_tensor_debug(persistent_query_start_loc, "smallq_qsl"),
            )
        self._call_flash_attn_smallq_decode_paged(
            layer,
            query,
            key_cache,
            value_cache,
            persistent_decode_block_table[:num_query_tokens],
            persistent_decode_seq_lens[:num_query_tokens],
            attn_metadata,
            out=out_view,
            max_seq_len_hint=getattr(
                attn_metadata,
                "smallq_decode_max_seq_len_hint",
                None,
            ),
            workspace_seq_capacity_hint=getattr(
                attn_metadata,
                "smallq_decode_workspace_seq_capacity_hint",
                None,
            ),
            partition_size_hint=getattr(
                attn_metadata,
                "smallq_decode_partition_size_hint",
                None,
            ),
        )
        return output

    if _routing._is_cuda_graph_capturing(query):
        raise RuntimeError(
            "FLASH_ATTN_V100 small-query prefix prefill entered CUDA graph "
            "capture without persistent smallq decode metadata. The "
            "metadata builder must attach smallq_decode_block_table and "
            "smallq_decode_seq_lens so replay does not capture transient "
            "derived tensors."
        )

    query_start_loc_norm = _kv_layout._normalize_query_start_loc_for_available_tokens(
        query_start_loc,
        num_query_tokens,
    )
    query_start_loc_gpu = query_start_loc_norm.to(
        device=device,
        dtype=attn_metadata.query_start_loc.dtype,
    )
    query = query[:num_query_tokens]
    out_view = output[:num_query_tokens]
    query_lens_gpu = query_start_loc_gpu[1:] - query_start_loc_gpu[:-1]
    real_query_lens_gpu = query_lens_gpu
    real_num_query_tokens = query_start_loc_gpu[-1]
    num_seqs = query_lens_gpu.numel()
    if num_seqs > 0:
        # FULL CUDA graph replay may pad a 3-request MTP verifier batch
        # from 15 tokens to 20 tokens while query_start_loc still marks
        # only the 15 live tokens. Give the padded tail a dummy query span
        # so repeat_interleave keeps the captured graph shape. The padded
        # rows are masked below and must not read real KV cache entries.
        padding_tokens = torch.clamp(
            num_query_tokens - real_num_query_tokens,
            min=0,
        )
        query_lens_gpu = query_lens_gpu.clone()
        query_lens_gpu[-1] += padding_tokens

    seq_lens = _seq_lens[:num_seqs].to(
        device=device,
        dtype=attn_metadata.seq_lens.dtype,
    )
    effective_seq_lens = torch.maximum(
        seq_lens,
        real_query_lens_gpu.to(dtype=attn_metadata.seq_lens.dtype),
    )
    block_table = attn_metadata.block_table[:num_seqs].clamp_min(0)
    decode_block_table = torch.repeat_interleave(
        block_table,
        query_lens_gpu,
        dim=0,
        output_size=num_query_tokens,
    ).contiguous()
    seq_lens_rep = torch.repeat_interleave(
        effective_seq_lens,
        query_lens_gpu,
        output_size=num_query_tokens,
    )
    query_lens_rep = torch.repeat_interleave(
        real_query_lens_gpu.to(dtype=dtype),
        query_lens_gpu,
        output_size=num_query_tokens,
    )
    start_locs_rep = torch.repeat_interleave(
        query_start_loc_gpu[:-1].to(dtype=dtype),
        query_lens_gpu,
        output_size=num_query_tokens,
    )
    token_indices = torch.arange(
        num_query_tokens,
        device=device,
        dtype=dtype,
    )
    offsets = token_indices - start_locs_rep + 1
    decode_seq_lens = (seq_lens_rep - query_lens_rep + offsets).contiguous()
    padding_mask = token_indices >= real_num_query_tokens
    decode_seq_lens = torch.where(
        padding_mask,
        torch.zeros_like(decode_seq_lens),
        decode_seq_lens,
    ).contiguous()
    decode_block_table = torch.where(
        padding_mask[:, None],
        torch.zeros_like(decode_block_table),
        decode_block_table,
    ).contiguous()
    # EAGER fallback branch (persistent smallq metadata absent). Cap the
    # workspace/launch grid to the runtime max_seq_len instead of passing the
    # raw block-table capacity (== max_model_len worth of blocks), which would
    # over-launch ceil(max_model_len/ps) partitions where only
    # ceil(max_seq_len/ps) do work. eager_max_seq_len is computed once (single
    # device->host sync, was already paid for max_seq_len_hint) and reused; the
    # interface floors the hint at effective max_seq_len (_get_decode_plan:
    # 165-169), so the cap can never under-cover the runtime sequences.
    if num_seqs > 0:
        eager_max_seq_len = int(seq_lens.max().item())
        eager_workspace_seq_capacity_hint = min(
            int(block_table.shape[1]) * int(key_cache.shape[1]),
            eager_max_seq_len,
        )
    else:
        eager_max_seq_len = None
        eager_workspace_seq_capacity_hint = None
    self._call_flash_attn_smallq_decode_paged(
        layer,
        query,
        key_cache,
        value_cache,
        decode_block_table,
        decode_seq_lens,
        attn_metadata,
        out=out_view,
        max_seq_len_hint=eager_max_seq_len,
        workspace_seq_capacity_hint=eager_workspace_seq_capacity_hint,
        partition_size_hint=None,
    )
    return output
