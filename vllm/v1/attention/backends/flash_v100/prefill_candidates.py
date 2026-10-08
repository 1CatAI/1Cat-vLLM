# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered per-sequence prefill with explicit operators and layer policy."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import masks as _masks
from vllm.v1.attention.backends.flash_v100 import ops as _ops
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100.plan import routing as _plan
from vllm.v1.attention.backends.flash_v100.workspace import V100Workspace
from vllm.v1.attention.kv_codecs import FP8_E4M3, KVCodec, resolve_kv_codec


@dataclass(frozen=True)
class PrefillConfig:
    policy: _config.V100AttnConfig
    scale: float
    kv_cache_dtype: str

    @property
    def kv_codec(self) -> KVCodec | None:
        return resolve_kv_codec(self.kv_cache_dtype)


@dataclass(frozen=True)
class PrefillOps:
    """Explicit native, admission and diagnostic injection points."""

    bridge: Any
    run_paged: Any
    should_bridge: Any
    should_bfla: Any
    should_contig: Any
    should_gather: Any
    should_split: Any
    bhmd: Any
    dense: Any
    paged: Any
    bfla: Any
    splitkv: Any
    uniform: Any
    try_fa2: Any
    log_bfla: Any
    log_fa2: Any
    log_contiguous_bhmd: Any
    log_contiguous_dense: Any
    log_dense_fa2: Any
    log_fp8_bridge: Any
    log_splitkv: Any


@dataclass
class PrefillRequest:
    layer: torch.nn.Module
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    attn_metadata: Any
    out_view: torch.Tensor
    i: int
    start: int
    end: int
    q_len: int
    seq_len: int
    q_seq: torch.Tensor
    num_seqs: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    causal: bool
    window_size: tuple[int, int]
    contig_allowed: bool = False
    use_splitkv: bool = False
    use_fp8_bridge: bool = False


@dataclass(frozen=True)
class PrefillResult:
    output: Any
    is_destination: bool
    skip_debug: bool


class PrefillExecutor:
    def __init__(
        self, config: PrefillConfig, ops: PrefillOps, workspace: V100Workspace
    ):
        self.config = config
        self.ops = ops
        self.workspace = workspace

    def sequence(self, request: PrefillRequest) -> PrefillResult:
        return _plan.execute(request, self.candidates(request))

    def candidates(self, request: PrefillRequest):
        for candidate in PREPARED_CANDIDATES:
            yield candidate(self)
        self.prepare_fallback(request)
        for fallback in FALLBACK_CANDIDATES:
            yield fallback(self)

    def prepare_fallback(self, request: PrefillRequest) -> None:
        # These policy reads happened even when a higher-priority route won.
        request.use_splitkv = self.ops.should_split(
            q_len=request.q_len,
            seq_len=request.seq_len,
            head_dim=request.head_dim,
            key_cache=request.key_cache,
            causal=request.causal,
        )
        request.use_fp8_bridge = self.ops.should_bridge(
            q_len=request.q_len,
            head_dim=request.head_dim,
            key_cache=request.key_cache,
            value_cache=request.value_cache,
            causal=request.causal,
            window_size=request.window_size,
        )


class SequenceCandidate(_plan.Candidate[PrefillRequest, PrefillResult]):
    def __init__(self, executor: PrefillExecutor):
        self.executor = executor


class BflaPrefill(SequenceCandidate):
    def admit(self, request: PrefillRequest) -> bool:
        return self.executor.ops.should_bfla(
            q_len=request.q_len,
            seq_len=request.seq_len,
            head_dim=request.head_dim,
            key_cache=request.key_cache,
            causal=request.causal,
            window_size=request.window_size,
        )

    def run(
        self, request: PrefillRequest, record: _plan.RecordRoute
    ) -> PrefillResult | None:
        out_is_destination = False
        bfla_block_mask = _masks._build_bfla_block_mask_for_seq(
            request.q_seq,
            request.key_cache,
            request.attn_metadata.block_table[request.i],
            seq_len=request.seq_len,
            block_size=request.block_size,
            mask_block_n=self.executor.config.policy.prefill_bfla_mask_block_n,
            softmax_scale=self.executor.config.scale,
        )
        if bfla_block_mask is None:
            return None
        self.executor.prepare_fallback(request)
        self.executor.ops.log_bfla(self.executor.config)
        record(_routing.ROUTE_SPECS["prefill_prefix_bfla"].name)
        out_seq = self.executor.ops.run_paged(
            route="prefill_prefix_bfla",
            q_len=request.q_len,
            seq_len=request.seq_len,
            heads_q=request.query.shape[1],
            heads_kv=request.num_kv_heads,
            head_dim=request.head_dim,
            block_size=request.block_size,
            fn=partial(
                self.executor.ops.bfla,
                request.q_seq,
                request.key_cache,
                request.value_cache,
                request.attn_metadata.block_table[request.i : request.i + 1],
                request.attn_metadata.seq_lens[request.i : request.i + 1],
                bfla_block_mask,
                self.executor.config.policy.prefill_bfla_mask_block_n,
                softmax_scale=self.executor.config.scale,
                kv_cache_dtype=self.executor.config.kv_cache_dtype,
                k_scale=float(request.layer._k_scale_float),
                v_scale=float(request.layer._v_scale_float),
                causal=request.causal,
                window_size=request.window_size,
            ),
        )
        return PrefillResult(out_seq, out_is_destination, False)


class Fa2Prefill(SequenceCandidate):
    def admit(self, request: PrefillRequest) -> bool:
        return (
            _config.registered("VLLM_FLASH_V100_FA2_D256_PREFILL")
            and request.key_cache.dtype == torch.float16
            and (request.value_cache.dtype == torch.float16)
            and (request.q_len >= 1024)
            and (request.head_dim == 256)
            and request.causal
            and (request.window_size == (-1, -1))
        )

    def run(
        self, request: PrefillRequest, record: _plan.RecordRoute
    ) -> PrefillResult | None:
        out_is_destination = False
        cu_q, cu_k = self.executor.ops.uniform(
            request.q_seq, batch_size=1, query_len=request.q_len, kv_len=request.seq_len
        )
        fa2_out_dest = request.out_view[request.start : request.end].unsqueeze(0)
        fa2_dense_kv = _kv_layout._contiguous_paged_kv_view(
            request.key_cache,
            request.value_cache,
            request.attn_metadata.block_table[request.i],
            request.seq_len,
            request.block_size,
            request.attn_metadata,
            request.i,
            False,
        )
        fa2_dense_route = "prefill_prefix_contig_splitd_d256"
        if (
            fa2_dense_kv is None
            and self.executor.ops.should_gather(
                q_len=request.q_len,
                seq_len=request.seq_len,
                head_dim=request.head_dim,
                key_cache=request.key_cache,
                value_cache=request.value_cache,
                causal=request.causal,
                window_size=request.window_size,
                num_seqs=request.num_seqs,
            )
            and (_ops._get_sm70_splitd_d256_ops() is not None)
        ):
            fa2_dense_kv = _kv_layout._gather_paged_kv_to_exact_dense(
                request.key_cache,
                request.value_cache,
                request.attn_metadata.block_table[request.i],
                request.seq_len,
            )
            fa2_dense_route = "prefill_prefix_gather_splitd_d256"
        if fa2_dense_kv is not None:
            fa2_route = fa2_dense_route
            fa2_key, fa2_value = fa2_dense_kv
            fa2_paged_out = self.executor.ops.run_paged(
                route=fa2_route,
                q_len=request.q_len,
                seq_len=request.seq_len,
                heads_q=request.query.shape[1],
                heads_kv=request.num_kv_heads,
                head_dim=request.head_dim,
                block_size=request.block_size,
                fn=lambda q_seq=request.q_seq,
                fa2_key=fa2_key,
                fa2_value=fa2_value,
                cu_q=cu_q,
                cu_k=cu_k,
                q_len=request.q_len,
                seq_len=request.seq_len,
                out_dest=fa2_out_dest: self.executor.ops.try_fa2(
                    request.q_seq,
                    fa2_key,
                    fa2_value,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=request.q_len,
                    max_seqlen_k=request.seq_len,
                    softmax_scale=self.executor.config.scale,
                    causal=request.causal,
                    window_size=request.window_size,
                    out=out_dest,
                ),
            )
        else:
            fa2_route = "prefill_prefix_paged_splitd_d256"
            fa2_paged_out = self.executor.ops.run_paged(
                route=fa2_route,
                q_len=request.q_len,
                seq_len=request.seq_len,
                heads_q=request.query.shape[1],
                heads_kv=request.num_kv_heads,
                head_dim=request.head_dim,
                block_size=request.block_size,
                fn=lambda q_seq=request.q_seq,
                key_cache=request.key_cache,
                value_cache=request.value_cache,
                cu_q=cu_q,
                q_len=request.q_len,
                seq_len=request.seq_len,
                out_dest=fa2_out_dest,
                i=request.i: self.executor.ops.try_fa2(
                    request.q_seq,
                    request.key_cache,
                    request.value_cache,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=None,
                    max_seqlen_q=request.q_len,
                    max_seqlen_k=request.seq_len,
                    softmax_scale=self.executor.config.scale,
                    causal=request.causal,
                    window_size=request.window_size,
                    out=out_dest,
                    seqused_k=request.attn_metadata.seq_lens[request.i : request.i + 1],
                    block_table=request.attn_metadata.block_table[
                        request.i : request.i + 1
                    ],
                ),
            )
        if fa2_paged_out is None:
            return None
        self.executor.prepare_fallback(request)
        self.executor.ops.log_fa2(self.executor.config, fa2_route)
        record(fa2_route or "prefill_prefix_splitd_d256")
        out_seq = fa2_paged_out
        out_is_destination = True
        return PrefillResult(out_seq, out_is_destination, False)


class ContiguousBhmdPrefill(SequenceCandidate):
    def admit(self, request: PrefillRequest) -> bool:
        return True

    def run(
        self, request: PrefillRequest, record: _plan.RecordRoute
    ) -> PrefillResult | None:
        request.contig_allowed = self.executor.ops.should_contig(
            q_len=request.q_len,
            seq_len=request.seq_len,
            head_dim=request.head_dim,
            key_cache=request.key_cache,
            causal=request.causal,
            window_size=request.window_size,
        )
        contig_dense_kv_bhmd = None
        if (
            request.contig_allowed
            and self.executor.config.policy.prefill_contig_dense_allow_copy
            and self.executor.ops.bhmd is not None
        ):
            contig_dense_kv_bhmd = _kv_layout._contiguous_paged_kv_bhmd(
                request.key_cache,
                request.value_cache,
                request.attn_metadata.block_table[request.i],
                request.seq_len,
                request.block_size,
                request.attn_metadata,
                request.i,
            )
        if contig_dense_kv_bhmd is None:
            return None
        self.executor.prepare_fallback(request)
        self.executor.ops.log_contiguous_bhmd(self.executor.config)
        k_bhmd, v_bhmd = contig_dense_kv_bhmd
        q_bhmd = request.q_seq.permute(0, 2, 1, 3).contiguous()
        record(_routing.ROUTE_SPECS["prefill_prefix_contig_dense_bhmd"].name)
        out_bhmd = self.executor.ops.run_paged(
            route="prefill_prefix_contig_dense_bhmd",
            q_len=request.q_len,
            seq_len=request.seq_len,
            heads_q=request.query.shape[1],
            heads_kv=request.num_kv_heads,
            head_dim=request.head_dim,
            block_size=request.block_size,
            fn=lambda q_bhmd=q_bhmd,
            k_bhmd=k_bhmd,
            v_bhmd=v_bhmd: self.executor.ops.bhmd(
                q_bhmd,
                k_bhmd,
                v_bhmd,
                causal=request.causal,
                softmax_scale=self.executor.config.scale,
                window_size=request.window_size,
            ),
        )
        request.out_view[request.start : request.end].copy_(
            out_bhmd.squeeze(0).permute(1, 0, 2)
        )
        return PrefillResult(None, False, True)


class ContiguousDensePrefill(SequenceCandidate):
    def admit(self, request: PrefillRequest) -> bool:
        return request.contig_allowed

    def run(
        self, request: PrefillRequest, record: _plan.RecordRoute
    ) -> PrefillResult | None:
        out_is_destination = False
        contig_dense_kv = _kv_layout._contiguous_paged_kv_view(
            request.key_cache,
            request.value_cache,
            request.attn_metadata.block_table[request.i],
            request.seq_len,
            request.block_size,
            request.attn_metadata,
            request.i,
            self.executor.config.policy.prefill_contig_dense_allow_copy,
        )
        if contig_dense_kv is None:
            return None
        self.executor.prepare_fallback(request)
        self.executor.ops.log_contiguous_dense(self.executor.config)
        k_dense, v_dense = contig_dense_kv
        fa2_out = None
        if _config.registered("VLLM_FLASH_V100_FA2_D256_PREFILL"):
            cu_q, cu_k = self.executor.ops.uniform(
                request.q_seq,
                batch_size=1,
                query_len=request.q_len,
                kv_len=request.seq_len,
            )
            fa2_out_dest = request.out_view[request.start : request.end].unsqueeze(0)
            fa2_out = self.executor.ops.run_paged(
                route="prefill_prefix_contig_dense_fa2_d256",
                q_len=request.q_len,
                seq_len=request.seq_len,
                heads_q=request.query.shape[1],
                heads_kv=request.num_kv_heads,
                head_dim=request.head_dim,
                block_size=request.block_size,
                fn=lambda q_seq=request.q_seq,
                k_dense=k_dense,
                v_dense=v_dense,
                cu_q=cu_q,
                cu_k=cu_k,
                q_len=request.q_len,
                seq_len=request.seq_len,
                out_dest=fa2_out_dest: self.executor.ops.try_fa2(
                    request.q_seq,
                    k_dense,
                    v_dense,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=request.q_len,
                    max_seqlen_k=request.seq_len,
                    softmax_scale=self.executor.config.scale,
                    causal=request.causal,
                    window_size=request.window_size,
                    out=out_dest,
                ),
            )
        if fa2_out is not None:
            self.executor.ops.log_dense_fa2(self.executor.config)
            record(_routing.ROUTE_SPECS["prefill_prefix_contig_dense_fa2_d256"].name)
            out_seq = fa2_out
            out_is_destination = True
        else:
            record(_routing.ROUTE_SPECS["prefill_prefix_contig_dense"].name)
            out_seq = self.executor.ops.run_paged(
                route="prefill_prefix_contig_dense",
                q_len=request.q_len,
                seq_len=request.seq_len,
                heads_q=request.query.shape[1],
                heads_kv=request.num_kv_heads,
                head_dim=request.head_dim,
                block_size=request.block_size,
                fn=lambda q_seq=request.q_seq,
                k_dense=k_dense,
                v_dense=v_dense: self.executor.ops.dense(
                    request.q_seq,
                    k_dense,
                    v_dense,
                    causal=request.causal,
                    softmax_scale=self.executor.config.scale,
                    window_size=request.window_size,
                ),
            )
        return PrefillResult(out_seq, out_is_destination, False)


class Fp8BridgePrefill(SequenceCandidate):
    def admit(self, request: PrefillRequest) -> bool:
        return request.use_fp8_bridge

    def run(
        self, request: PrefillRequest, record: _plan.RecordRoute
    ) -> PrefillResult | None:
        out_is_destination = False
        bridge_result = self.executor.ops.bridge(
            query=request.q_seq,
            key_cache=request.key_cache,
            value_cache=request.value_cache,
            block_table=request.attn_metadata.block_table[request.i : request.i + 1],
            seq_lens=request.attn_metadata.seq_lens[request.i : request.i + 1],
            seq_len=request.seq_len,
            k_scale=float(request.layer._k_scale_float),
            v_scale=float(request.layer._v_scale_float),
            causal=request.causal,
            window_size=request.window_size,
            out=request.out_view[request.start : request.end].unsqueeze(0),
        )
        if bridge_result is not None:
            out_seq, out_is_destination = bridge_result
            self.executor.ops.log_fp8_bridge(self.executor.config)
            record(
                "prefill_prefix_fp8_e4m3_bridge"
                if self.executor.config.kv_codec is FP8_E4M3
                else "prefill_prefix_fp8_e5m2_bridge"
            )
        else:
            out_seq = self.executor.ops.paged(
                request.q_seq,
                request.key_cache,
                request.value_cache,
                request.attn_metadata.block_table[request.i : request.i + 1],
                request.attn_metadata.seq_lens[request.i : request.i + 1],
                softmax_scale=self.executor.config.scale,
                kv_cache_dtype=self.executor.config.kv_cache_dtype,
                k_scale=float(request.layer._k_scale_float),
                v_scale=float(request.layer._v_scale_float),
                causal=request.causal,
                window_size=request.window_size,
            )
        return PrefillResult(out_seq, out_is_destination, False)


class SplitKvPrefill(SequenceCandidate):
    def admit(self, request: PrefillRequest) -> bool:
        return request.use_splitkv

    def run(
        self, request: PrefillRequest, record: _plan.RecordRoute
    ) -> PrefillResult | None:
        out_is_destination = False
        self.executor.ops.log_splitkv(self.executor.config)
        record(_routing.ROUTE_SPECS["prefill_prefix_splitkv"].name)
        out_seq = self.executor.ops.run_paged(
            route="prefill_prefix_splitkv",
            q_len=request.q_len,
            seq_len=request.seq_len,
            heads_q=request.query.shape[1],
            heads_kv=request.num_kv_heads,
            head_dim=request.head_dim,
            block_size=request.block_size,
            fn=lambda q_seq=request.q_seq,
            i=request.i,
            seq_len=request.seq_len: self.executor.ops.splitkv(
                request.q_seq,
                request.key_cache,
                request.value_cache,
                request.attn_metadata.block_table[request.i : request.i + 1],
                request.attn_metadata.seq_lens[request.i : request.i + 1],
                softmax_scale=self.executor.config.scale,
                kv_cache_dtype=self.executor.config.kv_cache_dtype,
                k_scale=float(request.layer._k_scale_float),
                v_scale=float(request.layer._v_scale_float),
                causal=request.causal,
                window_size=request.window_size,
                split_kv_tokens=self.executor.config.policy.prefill_split_kv_tokens,
                max_seq_len_hint=request.seq_len,
            ),
        )
        return PrefillResult(out_seq, out_is_destination, False)


class PagedPrefill(SequenceCandidate):
    def admit(self, request: PrefillRequest) -> bool:
        return True

    def run(
        self, request: PrefillRequest, record: _plan.RecordRoute
    ) -> PrefillResult | None:
        out_is_destination = False
        out_seq = self.executor.ops.run_paged(
            route="prefill_prefix_paged",
            q_len=request.q_len,
            seq_len=request.seq_len,
            heads_q=request.query.shape[1],
            heads_kv=request.num_kv_heads,
            head_dim=request.head_dim,
            block_size=request.block_size,
            fn=lambda q_seq=request.q_seq, i=request.i: self.executor.ops.paged(
                request.q_seq,
                request.key_cache,
                request.value_cache,
                request.attn_metadata.block_table[request.i : request.i + 1],
                request.attn_metadata.seq_lens[request.i : request.i + 1],
                softmax_scale=self.executor.config.scale,
                kv_cache_dtype=self.executor.config.kv_cache_dtype,
                k_scale=float(request.layer._k_scale_float),
                v_scale=float(request.layer._v_scale_float),
                causal=request.causal,
                window_size=request.window_size,
            ),
        )
        return PrefillResult(out_seq, out_is_destination, False)


PREPARED_CANDIDATES = (
    BflaPrefill,
    Fa2Prefill,
    ContiguousBhmdPrefill,
    ContiguousDensePrefill,
)
FALLBACK_CANDIDATES = (Fp8BridgePrefill, SplitKvPrefill, PagedPrefill)
