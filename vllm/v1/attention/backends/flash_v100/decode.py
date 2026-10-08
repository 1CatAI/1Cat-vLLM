# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode execution over explicit policy, operators and per-layer workspace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100 import state as _state
from vllm.v1.attention.backends.flash_v100.plan import routing as _plan
from vllm.v1.attention.backends.flash_v100.workspace import (
    V100Workspace as V100Workspace,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    KVCodec,
    resolve_kv_codec,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


@dataclass(frozen=True)
class DecodeConfig:
    """Layer geometry and the frozen policy used by decode execution."""

    policy: _config.V100AttnConfig
    scale: float
    kv_cache_dtype: str
    attn_type: str
    sliding_window: tuple[int, int] | None

    @property
    def kv_codec(self) -> KVCodec | None:
        return resolve_kv_codec(self.kv_cache_dtype)


@dataclass(frozen=True)
class DecodeOps:
    """Native ABI callables; no backend receiver or mutable tensor storage.

    Diagnostic callbacks remain explicit while the debug observer is extracted.
    ``scalar_override`` preserves the old private injection point during migration.
    Native optional signatures are inspected by the loader, so they are typed Any.
    """

    dense: Any
    paged: Any
    xqa: Any
    wmma: Any
    prefill: Any
    prefill_bhmd: Any
    paged_keywords: set[str]
    scalar_tail: Any
    reject_xqa: Callable[[KVCodec, TritonAttentionMetadata], bool]
    reserve_bhmd_compare: Callable[[], int | None]
    write_bhmd_compare: Callable[..., None]
    compare_bhmd: Callable[..., None]
    scalar_override: Callable[..., None] | None = None


@dataclass(frozen=True)
class NativeDecodeRequest:
    layer: torch.nn.Module
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    attn_metadata: TritonAttentionMetadata
    out_view: torch.Tensor
    output: torch.Tensor
    window_size: tuple[int, int]
    xqa_codec: KVCodec | None
    anchor_lens: torch.Tensor | None
    anchored_window: int
    selection: _routing.RouteSpec | None


class DecodeExecutor:
    """Execute decode without importing or receiving the attention Impl."""

    def __init__(self, config: DecodeConfig, ops: DecodeOps, workspace: V100Workspace):
        self.config = config
        self.ops = ops
        self.workspace = workspace

    def _flash_v100_window_size(self, causal: bool) -> tuple[int, int]:
        window = self.config.sliding_window
        if window is None or tuple(window) == (-1, -1):
            return (-1, -1)
        left, right = tuple(window)
        left, right = int(left), int(right)
        if not causal and left >= 0 and right == 0:
            right = left
        return (left, right)

    def _xqa_kv_codec(self, key_cache, value_cache, attn_metadata) -> KVCodec | None:
        codec = self.config.kv_codec
        if codec not in (FP16, FP8_E4M3, FP8_E5M2) or not codec.stores(
            key_cache, value_cache
        ):
            return None
        if self.ops.reject_xqa(codec, attn_metadata):
            return None
        return codec

    def _call_flash_attn_decode_paged(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        softmax_scale: float,
        out: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        window_size: tuple[int, int] = (-1, -1),
        max_seq_len_hint: int | None = None,
        workspace_seq_capacity_hint: int | None = None,
        active_num_partitions: int | None = None,
        partition_size_hint: int | None = None,
        anchor_lens: torch.Tensor | None = None,
        anchored_window: int = 0,
        record: _plan.RecordRoute = _plan.record_legacy,
    ) -> None:
        scalar_tail = self.ops.scalar_tail
        if scalar_tail is not None and scalar_tail(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            out=out,
            softmax_scale=softmax_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_cache_dtype=kv_cache_dtype,
            window_size=window_size,
            max_seq_len_hint=max_seq_len_hint,
            partition_size_hint=partition_size_hint,
            anchor_lens=anchor_lens,
            anchored_window=anchored_window,
        ):
            record(_routing.ROUTE_SPECS["decode_e4m3_compact_scalar_tail"].name)
            return
        kwargs: dict[str, object] = {
            "softmax_scale": softmax_scale,
            "out": out,
            "kv_cache_dtype": kv_cache_dtype,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }
        if "window_size" in self.ops.paged_keywords:
            kwargs["window_size"] = window_size
        elif tuple(window_size) != (-1, -1):
            raise RuntimeError(
                "FLASH_ATTN_V100 decode op does not support sliding-window "
                "attention with this extension build."
            )
        if anchor_lens is not None and anchored_window > 0:
            if "anchor_lens" not in self.ops.paged_keywords:
                raise RuntimeError(
                    "FLASH_ATTN_V100 decode op does not support the anchored "
                    "decode-window mask with this extension build; rebuild "
                    "flash_attn_v100."
                )
            kwargs["anchor_lens"] = anchor_lens
            kwargs["anchored_window"] = anchored_window
        optional_kwargs = {
            "max_seq_len_hint": max_seq_len_hint,
            "workspace_seq_capacity_hint": workspace_seq_capacity_hint,
            "active_num_partitions": active_num_partitions,
            "partition_size_hint": partition_size_hint,
        }
        for name, value in optional_kwargs.items():
            if name in self.ops.paged_keywords:
                kwargs[name] = value
        self.ops.paged(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            **kwargs,
        )

    def _anchored_swa_params(
        self,
        attn_metadata: TritonAttentionMetadata,
    ) -> tuple[torch.Tensor | None, int]:
        """Anchored decode-window mask parameters, when active.

        Returns ``(prefix_anchor_lens, decode_sliding_window)`` when this
        decoder cache group carries the engine's prefix-anchored spec and
        per-request prompt lengths; otherwise ``(None, 0)``.
        """
        window = self.config.policy.prefix_anchored_decode_window
        if window is None:
            return None, 0

        metadata_window = getattr(attn_metadata, "decode_sliding_window", None)
        anchor_lens = getattr(attn_metadata, "prefix_anchor_lens", None)
        if (
            self.config.attn_type != AttentionType.DECODER
            or self.config.kv_codec is not FP16
            or metadata_window != window
            or anchor_lens is None
        ):
            raise RuntimeError(
                "FLASH_ATTN_V100 prefix-anchored SWA metadata does not match "
                "the enabled decoder-layer contract"
            )
        return anchor_lens, int(window)

    def _flash_v100_decode_as_paged_prefill(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode through the paged prefill WMMA kernel.

        This opt-in path keeps the paged KV layout but uses the same compute
        order as dense/paged prefill. It is a strictness bridge while the
        scalar paged decode kernel is brought to bitwise parity.
        """
        if not _state._logged_decode_paged_prefill:
            logger.warning(
                "FLASH_ATTN_V100 decode-as-paged-prefill path active. This is "
                "for strict debugging and may be slower than paged decode."
            )
            _state._logged_decode_paged_prefill = True

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]
        if query.shape[0] == 0:
            return output

        key_cache, value_cache = _kv_layout._split_paged_kv_cache(kv_cache)

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens_host = (
            seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        )
        num_seqs = min(len(query_start_loc) - 1, len(seq_lens_host))
        if num_seqs > 0:
            query_lens = query_start_loc[1 : num_seqs + 1] - query_start_loc[:num_seqs]
            first_query_len = int(query_lens[0].item())
            total_query_tokens = first_query_len * num_seqs
            can_batch_decode = (
                first_query_len > 0
                and bool(torch.all(query_lens == first_query_len).item())
                and int(query_start_loc[0].item()) == 0
                and int(query_start_loc[num_seqs].item()) == total_query_tokens
                and total_query_tokens <= query.shape[0]
            )
            if can_batch_decode:
                q_batch = query[:total_query_tokens].reshape(
                    num_seqs,
                    first_query_len,
                    query.shape[1],
                    query.shape[2],
                )
                out_batch_view = out_view[:total_query_tokens].reshape(
                    num_seqs,
                    first_query_len,
                    query.shape[1],
                    query.shape[2],
                )
                q_bhmd = q_batch.permute(0, 2, 1, 3)
                out_bhmd = out_batch_view.permute(0, 2, 1, 3)
                if (
                    first_query_len == 1
                    and self.config.policy.use_decode_wmma_wrapper
                    and self.ops.wmma is not None
                ):
                    if not _state._logged_decode_wmma_wrapper:
                        logger.info(
                            "FLASH_ATTN_V100 decode WMMA wrapper path active "
                            "(experimental exactness bridge)."
                        )
                        _state._logged_decode_wmma_wrapper = True
                    q_wmma = q_batch[:, 0].contiguous()
                    out_wmma = out_batch_view[:, 0]
                    self.ops.wmma(
                        q_wmma,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[:num_seqs],
                        attn_metadata.seq_lens[:num_seqs],
                        softmax_scale=self.config.scale,
                        out=out_wmma,
                        kv_cache_dtype=self.config.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                    )
                    return output
                if (
                    first_query_len == 1
                    and self.config.policy.use_decode_paged_prefill_bhmd_out
                    and self.ops.prefill_bhmd is not None
                    and q_bhmd.is_contiguous()
                    and out_bhmd.is_contiguous()
                ):
                    if not _state._logged_decode_paged_prefill_bhmd:
                        logger.info(
                            "FLASH_ATTN_V100 decode-as-paged-prefill "
                            "BHMD out path active."
                        )
                        _state._logged_decode_paged_prefill_bhmd = True
                    compare_call_idx = self.ops.reserve_bhmd_compare()
                    safe_bmhd = None
                    if compare_call_idx is not None:
                        safe_bmhd = self.ops.prefill(
                            q_batch,
                            key_cache,
                            value_cache,
                            attn_metadata.block_table[:num_seqs],
                            attn_metadata.seq_lens[:num_seqs],
                            softmax_scale=self.config.scale,
                            kv_cache_dtype=self.config.kv_cache_dtype,
                            k_scale=float(layer._k_scale_float),
                            v_scale=float(layer._v_scale_float),
                            causal=True,
                        )
                    raw_q_bhmd = q_bhmd
                    q_out_same_storage = _routing._same_storage(raw_q_bhmd, out_bhmd)
                    if q_out_same_storage:
                        if not _state._logged_decode_paged_prefill_bhmd_q_clone:
                            logger.info(
                                "FLASH_ATTN_V100 BHMD out path cloned Q to "
                                "avoid input/output storage aliasing."
                            )
                            _state._logged_decode_paged_prefill_bhmd_q_clone = True
                        raw_q_bhmd = q_bhmd.clone()
                    self.ops.prefill_bhmd(
                        raw_q_bhmd,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[:num_seqs],
                        attn_metadata.seq_lens[:num_seqs],
                        softmax_scale=self.config.scale,
                        out=out_bhmd,
                        kv_cache_dtype=self.config.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                        causal=True,
                    )
                    if safe_bmhd is not None:
                        assert compare_call_idx is not None
                        self.ops.write_bhmd_compare(
                            out_batch_view,
                            safe_bmhd,
                            compare_call_idx,
                            "direct_out_vs_safe",
                            {
                                "q_bhmd_stride": list(q_bhmd.stride()),
                                "out_bhmd_stride": list(out_bhmd.stride()),
                                "out_bhmd_contiguous": out_bhmd.is_contiguous(),
                                "q_out_same_storage": q_out_same_storage,
                            },
                        )
                    return output
                out_batch = self.ops.prefill(
                    q_batch,
                    key_cache,
                    value_cache,
                    attn_metadata.block_table[:num_seqs],
                    attn_metadata.seq_lens[:num_seqs],
                    softmax_scale=self.config.scale,
                    kv_cache_dtype=self.config.kv_cache_dtype,
                    k_scale=float(layer._k_scale_float),
                    v_scale=float(layer._v_scale_float),
                    causal=True,
                )
                if first_query_len == 1 and q_bhmd.is_contiguous():
                    self.ops.compare_bhmd(
                        layer,
                        q_bhmd,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[:num_seqs],
                        attn_metadata.seq_lens[:num_seqs],
                        out_batch,
                    )
                out_view[:total_query_tokens].copy_(
                    out_batch.reshape(
                        total_query_tokens,
                        query.shape[1],
                        query.shape[2],
                    )
                )
                return output

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            out_seq = self.ops.prefill(
                query[start:end].unsqueeze(0),
                key_cache,
                value_cache,
                attn_metadata.block_table[i : i + 1],
                attn_metadata.seq_lens[i : i + 1],
                softmax_scale=self.config.scale,
                kv_cache_dtype=self.config.kv_cache_dtype,
                k_scale=float(layer._k_scale_float),
                v_scale=float(layer._v_scale_float),
                causal=True,
            )
            out_view[start:end].copy_(out_seq.squeeze(0))

        return output

    def _flash_v100_decode_dense_cache(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode through dense Flash-V100 with an incremental single-seq KV cache.

        This is a strict single-concurrency bridge for no-MTP experiments. It
        avoids full paged-KV gather after the first step, but it is still an
        oracle path rather than the final paged decode kernel.
        """
        if _routing._uses_fp8_kv_cache(self.config.kv_cache_dtype):
            if self.config.policy.use_flash_v100_prefill_paged:
                return self._flash_v100_decode_as_paged_prefill(
                    layer,
                    query,
                    kv_cache,
                    attn_metadata,
                    output,
                )
            return self._flash_v100_decode_dense_reference(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )
        if not _state._logged_decode_dense_cache:
            logger.warning(
                "FLASH_ATTN_V100 decode dense-cache path active. This is "
                "single-sequence strict debugging and may be slower than paged decode."
            )
            _state._logged_decode_dense_cache = True

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]
        if query.shape[0] == 0:
            return output

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens_host = (
            seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        )
        num_seqs = min(len(query_start_loc) - 1, len(seq_lens_host))
        if num_seqs != 1:
            if self.config.policy.use_flash_v100_prefill_paged:
                return self._flash_v100_decode_as_paged_prefill(
                    layer,
                    query,
                    kv_cache,
                    attn_metadata,
                    output,
                )
            return self._flash_v100_decode_dense_reference(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )

        key_cache, _ = _kv_layout._split_paged_kv_cache(kv_cache)
        block_size = key_cache.shape[1]
        head_dim = key_cache.shape[3]
        seq_len = int(seq_lens_host[0].item())
        k_cont, v_cont = self.workspace.decode_cache.get_kv_single_seq(
            key,
            value,
            kv_cache,
            attn_metadata,
            attn_metadata.seq_lens[:1],
            block_size,
            head_dim,
            extract=_kv_layout._extract_contiguous_kv_from_paged_cache,
        )
        out_seq = self.ops.dense(
            query.unsqueeze(0),
            k_cont[:seq_len].unsqueeze(0),
            v_cont[:seq_len].unsqueeze(0),
            causal=True,
            softmax_scale=self.config.scale,
        )
        out_view.copy_(out_seq.squeeze(0))
        return output

    def _flash_v100_decode_dense_reference(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode through dense Flash-V100 over gathered KV.

        This is an opt-in strict-debug path, not a speed path. It gives us a
        dense Flash-V100 oracle while the paged decode kernel is brought to
        bitwise parity.
        """
        if not _state._logged_decode_dense_reference:
            logger.warning(
                "FLASH_ATTN_V100 decode dense-reference path active. This is "
                "for strict debugging and is expected to be slower than paged decode."
            )
            _state._logged_decode_dense_reference = True

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]
        if query.shape[0] == 0:
            return output

        key_cache, value_cache = _kv_layout._split_paged_kv_cache(kv_cache)
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens_host = (
            seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        )
        num_seqs = min(len(query_start_loc) - 1, len(seq_lens_host))

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            seq_len = int(seq_lens_host[i].item())
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
                self.config.kv_cache_dtype,
                float(layer._k_scale_float),
                float(layer._v_scale_float),
            )
            out_seq = self.ops.dense(
                query[start:end].unsqueeze(0),
                k_cont.unsqueeze(0),
                v_cont.unsqueeze(0),
                causal=True,
                softmax_scale=self.config.scale,
            )
            out_view[start:end].copy_(out_seq.squeeze(0))
        return output

    def _flash_v100_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode path using Flash V100 directly over paged KV cache."""
        window_size = self._flash_v100_window_size(causal=True)
        if self.config.policy.prefix_anchored_decode_window is None:
            anchor_lens, anchored_window = None, 0
        else:
            anchor_lens, anchored_window = self._anchored_swa_params(attn_metadata)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if query.shape[0] == 0:
            return output

        key_cache, value_cache = _kv_layout._split_paged_kv_cache(kv_cache)
        xqa_codec = self._xqa_kv_codec(key_cache, value_cache, attn_metadata)

        # FP8 G4 XQA had no end-to-end gain on 35B-A3B TP4 and has no accepted
        # sampled-quality advantage. Keep that shape on scalar decode.
        selection = _routing.select_route(
            _routing.RouteContext(
                stage="decode",
                codec=xqa_codec,
                shape=_routing.RouteShape(
                    query.shape[0],
                    query.shape[1],
                    key_cache.shape[2],
                    query.shape[2],
                    key_cache.shape[1],
                ),
                enabled=self.config.policy.use_decode_xqa,
                available=self.ops.xqa is not None,
                query=query,
                metadata=attn_metadata,
                seq_rows=attn_metadata.seq_lens.shape[0],
                window_size=window_size,
            ),
            ("decode_xqa_paged",),
            fallback="decode_scalar_paged",
        )
        request = NativeDecodeRequest(
            layer,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            out_view,
            output,
            window_size,
            xqa_codec,
            anchor_lens,
            anchored_window,
            selection,
        )
        return _plan.execute(request, (XqaDecode(self), ScalarDecode(self)))

    def xqa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        out_view: torch.Tensor,
        output: torch.Tensor,
        window_size: tuple[int, int],
        xqa_codec: KVCodec | None,
        *,
        record: _plan.RecordRoute = _plan.record_legacy,
    ) -> torch.Tensor:
        _routing._log_fp8_kv_cache_route(
            "decode", self.config.kv_cache_dtype, "xqa_paged", record=record
        )
        _routing._trace_decode_active(
            route="decode_xqa_paged",
            query=query,
            key_cache=key_cache,
            seq_lens=attn_metadata.seq_lens,
            attn_metadata=attn_metadata,
            window_size=window_size,
        )
        partition_size_hint = _routing._g6_aligned_page_partition_size_hint(
            query,
            key_cache,
            value_cache,
            self.config.kv_cache_dtype,
            strategy=getattr(self.config.policy, "decode_strategy", "legacy"),
        )
        if partition_size_hint is not None:
            if (
                xqa_codec is FP8_E4M3
                and getattr(self.config.policy, "decode_strategy", "legacy") == "legacy"
                and query.shape[0] == 1
                and _config.raw("VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO", "1") != "0"
            ):
                record(f"decode_xqa_e4m3_dynamic_page{key_cache.shape[1]}")
            else:
                record(f"decode_xqa_p{partition_size_hint}_page{key_cache.shape[1]}")
        self.ops.xqa(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            softmax_scale=self.config.scale,
            out=out_view,
            kv_cache_dtype=self.config.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            window_size=window_size,
            max_seq_len_hint=getattr(
                attn_metadata,
                "flash_v100_decode_max_seq_len_hint",
                None,
            ),
            workspace_seq_capacity_hint=getattr(
                attn_metadata,
                "flash_v100_decode_workspace_seq_capacity_hint",
                None,
            ),
            active_num_partitions=getattr(
                attn_metadata,
                "flash_v100_decode_active_num_partitions",
                None,
            ),
            partition_size_hint=partition_size_hint,
            batch_context_routing=bool(
                getattr(
                    attn_metadata,
                    "flash_v100_batch_context_routing",
                    False,
                )
            ),
        )
        record(_routing.ROUTE_SPECS["decode_xqa_paged"].name)
        return output

    def scalar(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        out_view: torch.Tensor,
        output: torch.Tensor,
        window_size: tuple[int, int],
        anchor_lens: torch.Tensor | None,
        anchored_window: int,
        *,
        record: _plan.RecordRoute = _plan.record_legacy,
    ) -> torch.Tensor:
        _routing._log_fp8_kv_cache_route(
            "decode", self.config.kv_cache_dtype, "scalar_paged", record=record
        )
        _routing._trace_decode_active(
            route="decode_scalar_paged",
            query=query,
            key_cache=key_cache,
            seq_lens=attn_metadata.seq_lens,
            attn_metadata=attn_metadata,
            window_size=window_size,
        )
        (
            self.ops.scalar_override
            or partial(self._call_flash_attn_decode_paged, record=record)
        )(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            softmax_scale=self.config.scale,
            out=out_view,
            kv_cache_dtype=self.config.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            window_size=window_size,
            max_seq_len_hint=getattr(
                attn_metadata,
                "flash_v100_decode_max_seq_len_hint",
                None,
            ),
            workspace_seq_capacity_hint=getattr(
                attn_metadata,
                "flash_v100_decode_workspace_seq_capacity_hint",
                None,
            ),
            active_num_partitions=getattr(
                attn_metadata,
                "flash_v100_decode_active_num_partitions",
                None,
            ),
            anchor_lens=anchor_lens,
            anchored_window=anchored_window,
        )
        record(_routing.ROUTE_SPECS["decode_scalar_paged"].name)
        return output


class XqaDecode:
    def __init__(self, executor: DecodeExecutor):
        self.executor = executor

    def admit(self, request: NativeDecodeRequest) -> bool:
        return request.selection is _routing.ROUTE_SPECS["decode_xqa_paged"]

    def run(
        self, request: NativeDecodeRequest, record: _plan.RecordRoute
    ) -> torch.Tensor:
        return self.executor.xqa(
            request.layer,
            request.query,
            request.key_cache,
            request.value_cache,
            request.attn_metadata,
            request.out_view,
            request.output,
            request.window_size,
            request.xqa_codec,
            record=record,
        )


class ScalarDecode:
    def __init__(self, executor: DecodeExecutor):
        self.executor = executor

    def admit(self, request: NativeDecodeRequest) -> bool:
        return True

    def run(
        self, request: NativeDecodeRequest, record: _plan.RecordRoute
    ) -> torch.Tensor:
        return self.executor.scalar(
            request.layer,
            request.query,
            request.key_cache,
            request.value_cache,
            request.attn_metadata,
            request.out_view,
            request.output,
            request.window_size,
            request.anchor_lens,
            request.anchored_window,
            record=record,
        )
