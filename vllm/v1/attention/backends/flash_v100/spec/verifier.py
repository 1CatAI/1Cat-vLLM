# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifier calculations with explicit policy and operator dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger, log_once_seen, set_log_once_state
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)
from vllm.v1.attention.kv_codecs import (
    FP8_E5M2,
    KVCodec,
    resolve_kv_codec,
)
from vllm.v1.attention.ops.sm70_grouped import (
    grouped_e4m3_fp32_allowed,
    grouped_fp16_fp32_reason,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


@dataclass(frozen=True)
class VerificationConfig:
    policy: _config.V100AttnConfig
    scale: float
    kv_cache_dtype: str
    alibi_slopes: Any
    logits_soft_cap: float
    grouped_enabled: bool
    grouped_batch_enabled: bool
    grouped_max_query: int
    grouped_request_major_abi: int
    grouped_min_model_len: int

    @property
    def kv_codec(self) -> KVCodec | None:
        return resolve_kv_codec(self.kv_cache_dtype)


@dataclass(frozen=True)
class VerificationOps:
    grouped: Any
    fp16_grouped: Any
    e4m3_grouped: Any
    xqa: Any
    window_size: Any
    layer_info: Any
    xqa_codec: Any
    decode: Any
    tree_seq_lens_match: Any = None
    tree_query_start_match: Any = None
    tree_parent_ids: Any = None
    tree_visibility: Any = None
    admit_grouped_override: Any = None
    run_grouped_override: Any = None
    admit_xqa_override: Any = None
    run_smallq_override: Any = None
    validate_contract: Any = None
    partition_hint: Any = None
    branch_enabled: Any = None
    branch_strict: Any = None
    tree_trace_enabled: Any = None
    tree_trace_event: Any = None
    parent_ids_cpu: Any = None
    draft_debug_enabled: Any = None
    metadata_debug_log: Any = None
    format_debug: Any = None


@dataclass(frozen=True)
class GroupedAdmission:
    """Exact legacy grouped-kernel input contract, without a backend receiver."""

    kv_cache_dtype: str
    use_smallq_decode_xqa: bool
    flash_attn_grouped_fp16_fp32_paged: Any
    flash_attn_grouped_e4m3_fp32_paged: Any
    _flash_v100_window_size: Any


class VerificationExecutor:
    def __init__(self, config: VerificationConfig, ops: VerificationOps):
        self.config = config
        self.ops = ops
        self.grouped_admission = GroupedAdmission(
            config.kv_cache_dtype,
            getattr(config.policy, "use_smallq_decode_xqa", False),
            ops.fp16_grouped,
            ops.e4m3_grouped,
            ops.window_size,
        )
        self.admit_grouped = (
            self.grouped_verify_allowed
            if ops.admit_grouped_override is None
            else ops.admit_grouped_override
        )
        self.run_grouped = (
            self.call_grouped_verify
            if ops.run_grouped_override is None
            else ops.run_grouped_override
        )
        self.admit_xqa = (
            self.smallq_xqa_allowed
            if ops.admit_xqa_override is None
            else ops.admit_xqa_override
        )
        self.run_smallq = (
            self.call_smallq_decode_paged
            if ops.run_smallq_override is None
            else ops.run_smallq_override
        )

    def validate_contract(self, layer, attn_metadata) -> None:
        self.ops.validate_contract(layer, attn_metadata, self.ops.window_size)

    def grouped_verify_allowed(
        self,
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
            and num_query_tokens <= self.config.grouped_max_query
        )
        batched_request_shape = bool(
            self.config.grouped_batch_enabled
            and self.config.grouped_request_major_abi >= 1
            and num_reqs in (2, 4, 8)
            and max_query_len == 8
            and num_query_tokens == num_reqs * 8
        )
        allowed = bool(
            self.config.grouped_enabled
            and (single_request_shape or batched_request_shape)
            and self.ops.grouped is not None
            and getattr(attn_metadata, "is_dflash_selector_target", False)
            and getattr(attn_metadata, "max_model_len", 0)
            >= self.config.grouped_min_model_len
            and getattr(attn_metadata, "causal", True)
            and self.ops.window_size(causal=True) == (-1, -1)
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
            and self.config.kv_codec is FP8_E5M2
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
            self.config.grouped_enabled
            and not allowed
            and not log_once_seen(
                "flash_v100._logged_prefill_smallq_grouped_verify_gate"
            )
        ):
            logger.info_once(
                "FLASH_ATTN_V100 DFlash2 grouped verifier gate rejected: "
                "op=%s marker=%s max_model_len=%s min_model_len=%s "
                "causal=%s window=%s reqs=%d max_q=%d actual=%d "
                "native_max_q=%d q=%s/%s "
                "k=%s/%s v=%s/%s kv_dtype=%s block_table=%s/%s "
                "seq_lens=%s/%s.",
                self.ops.grouped is not None,
                getattr(attn_metadata, "is_dflash_selector_target", False),
                getattr(attn_metadata, "max_model_len", None),
                self.config.grouped_min_model_len,
                getattr(attn_metadata, "causal", True),
                self.ops.window_size(causal=True),
                num_reqs,
                max_query_len,
                num_query_tokens,
                self.config.grouped_max_query,
                tuple(query.shape),
                query.dtype,
                tuple(key_cache.shape),
                key_cache.dtype,
                tuple(value_cache.shape),
                value_cache.dtype,
                self.config.kv_cache_dtype,
                None if block_table is None else tuple(block_table.shape),
                None if block_table is None else block_table.dtype,
                None if seq_lens is None else tuple(seq_lens.shape),
                None if seq_lens is None else seq_lens.dtype,
                scope="process",
                key="flash_v100._logged_prefill_smallq_grouped_verify_gate",
            )
            set_log_once_state(
                "flash_v100._logged_prefill_smallq_grouped_verify_gate", True
            )
        return allowed

    def call_grouped_verify(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        *,
        out: torch.Tensor,
    ) -> None:
        num_reqs = int(attn_metadata.block_table.shape[0])
        if not log_once_seen("flash_v100._logged_prefill_smallq_grouped_verify"):
            logger.info_once(
                "FLASH_ATTN_V100 DFlash2 exact grouped verifier active "
                "(request-major B%d/q%d/H6/Hkv1/D256, %s KV, one-pass).",
                num_reqs,
                query.shape[0] // num_reqs,
                self.config.kv_cache_dtype,
                scope="process",
                key="flash_v100._logged_prefill_smallq_grouped_verify",
            )
            set_log_once_state("flash_v100._logged_prefill_smallq_grouped_verify", True)
        self.ops.grouped(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table[:num_reqs],
            attn_metadata.seq_lens[:num_reqs],
            softmax_scale=self.config.scale,
            out=out,
            kv_cache_dtype=self.config.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            one_pass=True,
        )
        _routing.log_fp8_kv_cache_route(
            "decode", self.config.kv_cache_dtype, "dflash2_grouped_verify"
        )
        _routing.record_route(
            _routing.ROUTE_SPECS["prefill_smallq_dflash2_grouped_verify"].name
        )

    def smallq_xqa_allowed(
        self,
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
            codec=self.ops.xqa_codec(key_cache, value_cache, attn_metadata),
            shape=_routing.RouteShape(
                query.shape[0],
                query.shape[1],
                key_cache.shape[2],
                query.shape[2],
                key_cache.shape[1],
            ),
            enabled=self.config.policy.use_smallq_decode_xqa,
            available=self.ops.xqa is not None,
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

    def call_smallq_decode_paged(
        self,
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
        fp16_grouped = self.ops.fp16_grouped
        if (
            fp16_grouped is not None
            and grouped_fp16_fp32_reason(
                self.grouped_admission,
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
                softmax_scale=self.config.scale,
            )
            logger.info_once(
                "FLASH_ATTN_V100 FP16 KV grouped verifier active "
                "(FP32 probability/PV/numerator/max/sum, page=%d).",
                key_cache.shape[1],
                scope="process",
            )
            _routing.record_route(
                _routing.ROUTE_SPECS["prefill_smallq_fp16_grouped_fp32"].name
            )
            return
        grouped_op = self.ops.e4m3_grouped
        if grouped_op is not None and grouped_e4m3_fp32_allowed(
            self.grouped_admission,
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
                softmax_scale=self.config.scale,
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
            _routing.log_fp8_kv_cache_route(
                "decode", self.config.kv_cache_dtype, "grouped_fp32"
            )
            _routing.record_route(
                _routing.ROUTE_SPECS["prefill_smallq_e4m3_grouped_fp32"].name
            )
            return
        window_size = self.ops.window_size(causal=True)
        if self.admit_xqa(
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
                self.ops.partition_hint()
                if (
                    query.shape[0] == 5
                    and query.shape[2] == 256
                    and key_cache.shape[1] == 1616
                    and key_cache.shape[2] > 0
                    and query.shape[1] == 6 * key_cache.shape[2]
                    and self.config.kv_codec is FP8_E5M2
                    and FP8_E5M2.stores(key_cache, value_cache)
                )
                else None
            )
            if not log_once_seen("flash_v100._logged_prefill_smallq_decode_xqa"):
                logger.info_once(
                    "FLASH_ATTN_V100 MTP verifier XQA path active "
                    "(rows=%d, q_per_kv=%d, partition_hint=%s, "
                    "mtp5_dual_cta=%s).",
                    int(query.shape[0]),
                    int(query.shape[1] // key_cache.shape[2]),
                    verifier_partition_size_hint,
                    verifier_partition_size_hint is not None,
                    scope="process",
                    key="flash_v100._logged_prefill_smallq_decode_xqa",
                )
                set_log_once_state("flash_v100._logged_prefill_smallq_decode_xqa", True)
            _routing.log_fp8_kv_cache_route(
                "decode", self.config.kv_cache_dtype, "xqa_paged"
            )
            self.ops.xqa(
                query,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale=self.config.scale,
                out=out,
                kv_cache_dtype=self.config.kv_cache_dtype,
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
            _routing.record_route(
                _routing.ROUTE_SPECS["prefill_smallq_decode_xqa"].name
            )
            return

        self.ops.decode(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=self.config.scale,
            out=out,
            kv_cache_dtype=self.config.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            window_size=window_size,
            max_seq_len_hint=max_seq_len_hint,
            workspace_seq_capacity_hint=workspace_seq_capacity_hint,
            partition_size_hint=partition_size_hint,
        )
        _routing.record_route(_routing.ROUTE_SPECS["prefill_smallq_decode_scalar"].name)

    def small_query_enabled(
        self,
        attn_metadata: TritonAttentionMetadata,
    ) -> bool:
        if (
            not getattr(attn_metadata, "causal", True)
            or not self.config.policy.use_flash_v100_decode
            or self.config.policy.smallq_decode_max_query_len <= 0
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
            self.config.policy.smallq_decode_max_model_len <= 0
            or max_model_len <= self.config.policy.smallq_decode_max_model_len
        )
        return (
            max_query_len <= self.config.policy.smallq_decode_max_query_len
            and model_len_supported
        )

    def tree_prefill(
        self,
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

        is_capturing = _routing.is_cuda_graph_capturing(query)
        parent_ids = getattr(attn_metadata, "ddtree_parent_ids", None)
        num_tree_tokens_cpu = getattr(attn_metadata, "ddtree_num_tree_tokens_cpu", None)
        num_reqs = min(
            max(0, len(query_start_loc) - 1),
            int(parent_ids.shape[0]) if parent_ids is not None else 0,
            int(num_tree_tokens_cpu.numel()) if num_tree_tokens_cpu is not None else 0,
        )
        window_size = self.ops.window_size(causal=True)
        if (
            self.ops.branch_enabled()
            and parent_ids is not None
            and self.ops.tree_seq_lens_match(
                attn_metadata,
                seq_lens,
                num_reqs,
            )
            and self.ops.tree_query_start_match(
                attn_metadata,
                query_start_loc,
                num_reqs,
            )
        ):
            triton_parent_ids = self.ops.tree_parent_ids(
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
                        impl=self.config,
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
                    if is_capturing or self.ops.branch_strict():
                        raise
                    if self.ops.tree_trace_enabled():
                        self.ops.tree_trace_event(
                            "flash_ddtree_attention_route",
                            {
                                "route": "triton_exception_fallback",
                                "num_reqs": num_reqs,
                                "num_actual_tokens": int(
                                    getattr(attn_metadata, "num_actual_tokens", 0)
                                ),
                                "query_start_loc": query_start_loc.detach()
                                .cpu()
                                .tolist(),
                                "seq_lens": seq_lens.detach().cpu().tolist(),
                                "tree_tokens": (
                                    num_tree_tokens_cpu.detach().cpu().tolist()
                                    if num_tree_tokens_cpu is not None
                                    else None
                                ),
                            },
                        )
                    if not log_once_seen("flash_v100.tree_fallback"):
                        logger.exception_once(
                            "FLASH_ATTN_V100 DDTree Triton verifier failed; "
                            "falling back to dense masked verifier.",
                            scope="process",
                            key="flash_v100.tree_fallback",
                        )
                        set_log_once_state("flash_v100.tree_fallback", True)
                else:
                    if not log_once_seen("flash_v100.tree_paged"):
                        logger.info_once(
                            "FLASH_ATTN_V100 DDTree branched verifier path active "
                            "(Triton paged-KV ancestor mask).",
                            scope="process",
                            key="flash_v100.tree_paged",
                        )
                    set_log_once_state("flash_v100.tree_paged", True)
                    _routing.record_route(
                        _routing.ROUTE_SPECS["prefill_ddtree_triton"].name
                    )
                    if self.ops.tree_trace_enabled():
                        self.ops.tree_trace_event(
                            "flash_ddtree_attention_route",
                            {
                                "route": "triton",
                                "num_reqs": num_reqs,
                                "num_actual_tokens": int(
                                    getattr(attn_metadata, "num_actual_tokens", 0)
                                ),
                                "query_start_loc": query_start_loc.detach()
                                .cpu()
                                .tolist(),
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

        parent_ids_cpu = self.ops.parent_ids_cpu(attn_metadata)
        if parent_ids_cpu is None or num_tree_tokens_cpu is None:
            raise RuntimeError(
                "DDTree dense verifier fallback requires parent metadata"
            )

        if not log_once_seen("flash_v100.tree_dense"):
            logger.info_once(
                "FLASH_ATTN_V100 DDTree branched verifier path active "
                "(dense masked small-query fallback).",
                scope="process",
                key="flash_v100.tree_dense",
            )
            set_log_once_state("flash_v100.tree_dense", True)

        _routing.record_route(_routing.ROUTE_SPECS["prefill_ddtree_dense"].name)
        if self.ops.tree_trace_enabled():
            self.ops.tree_trace_event(
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
        trace_kv_diff = (
            _config.raw("VLLM_DFLASH_DDTREE_TRACE_KV_CACHE_DIFF", "0") == "1"
        )
        profile_enabled = _config.trace().flash_v100.value("prefill_chunk_profile")
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
                parent_ids_cpu[req_idx]
                if req_idx < int(parent_ids_cpu.shape[0])
                else None
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
                            _kv_layout.dequantize_fp8_contiguous_kv(
                                cache_k_by_slot,
                                cache_v_by_slot,
                                self.config.kv_cache_dtype,
                                float(layer._k_scale_float),
                                float(layer._v_scale_float),
                            )
                        )
                        key_diff = (cache_k_by_slot - key[start:end]).abs()
                        value_diff = (cache_v_by_slot - value[start:end]).abs()
                        self.ops.tree_trace_event(
                            "flash_ddtree_kv_cache_diff",
                            {
                                "layer": str(
                                    self.ops.layer_info(layer).get("layer_name")
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

            k_cont, v_cont = _kv_layout.extract_contiguous_kv_from_paged_cache(
                (key_cache, value_cache),
                attn_metadata.block_table[req_idx : req_idx + 1],
                attn_metadata.seq_lens[req_idx : req_idx + 1],
                key_cache.shape[2],
                key_cache.shape[3],
                key_cache.shape[1],
                total_tokens=seq_len,
            )
            k_cont, v_cont = _kv_layout.dequantize_fp8_contiguous_kv(
                k_cont,
                v_cont,
                self.config.kv_cache_dtype,
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

            scores = torch.einsum("mhd,nhd->hmn", q_f, k_f) * self.config.scale
            visible = self.ops.tree_visibility(
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
                self.ops.layer_info(layer).get("layer_name"),
                float(profile_start.elapsed_time(profile_end)),
                total_query_tokens,
                total_tree_tokens,
                max_seq_len,
                int(query.shape[1]),
                int(key_cache.shape[2]),
                int(key_cache.shape[3]),
            )

        return output

    def small_query_prefill(
        self,
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
        if self.config.grouped_enabled and self.admit_grouped(
            query,
            key_cache,
            value_cache,
            attn_metadata,
            num_query_tokens=num_query_tokens,
        ):
            query = query[:num_query_tokens]
            out_view = output[:num_query_tokens]
            self.run_grouped(
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
            and not _kv_layout.metadata_expects_more_query_tokens_than_available(
                attn_metadata,
                num_query_tokens,
            )
        ):
            query = query[:num_query_tokens]
            out_view = output[:num_query_tokens]
            if self.ops.draft_debug_enabled():
                self.ops.metadata_debug_log(
                    "smallq_call",
                    "layer=%s num_query_tokens=%s %s %s %s %s %s",
                    self.ops.layer_info(layer).get("layer_name"),
                    num_query_tokens,
                    self.ops.format_debug(query, "query"),
                    self.ops.format_debug(out_view, "out"),
                    self.ops.format_debug(
                        persistent_decode_block_table[:num_query_tokens],
                        "smallq_bt",
                    ),
                    self.ops.format_debug(
                        persistent_decode_seq_lens[:num_query_tokens],
                        "smallq_seq",
                    ),
                    self.ops.format_debug(persistent_query_start_loc, "smallq_qsl"),
                )
            self.run_smallq(
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

        if _routing.is_cuda_graph_capturing(query):
            raise RuntimeError(
                "FLASH_ATTN_V100 small-query prefix prefill entered CUDA graph "
                "capture without persistent smallq decode metadata. The "
                "metadata builder must attach smallq_decode_block_table and "
                "smallq_decode_seq_lens so replay does not capture transient "
                "derived tensors."
            )

        return self._eager_small_query_prefill(
            layer,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            output,
            query_start_loc,
            _seq_lens,
            num_query_tokens,
            device,
            dtype,
        )

    def _eager_small_query_prefill(
        self,
        layer,
        query,
        key_cache,
        value_cache,
        attn_metadata,
        output,
        query_start_loc,
        _seq_lens,
        num_query_tokens,
        device,
        dtype,
    ):
        query_start_loc_norm = (
            _kv_layout.normalize_query_start_loc_for_available_tokens(
                query_start_loc,
                num_query_tokens,
            )
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
        self.run_smallq(
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
