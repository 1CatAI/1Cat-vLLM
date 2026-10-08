# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 attention implementation (forward and route bodies)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from functools import partial

import torch

import vllm.envs as envs
from vllm.config.sm70_dflash2 import (
    capture_sm70_dflash2_config,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_v100 import debug as _debug
from vllm.v1.attention.backends.flash_v100 import dense_prefill as _dense_prefill
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import masks as _masks
from vllm.v1.attention.backends.flash_v100 import metadata as _metadata
from vllm.v1.attention.backends.flash_v100 import ops as _ops
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    KVCodec,
    resolve_kv_codec,
)
from vllm.v1.attention.ops.sm70_e4m3_grouped import (
    MAX_GROUPS_PER_CALL,
    grouped_e4m3_fp32_allowed,
    grouped_e4m3_fp32_groups_allowed,
    load_grouped_e4m3_fp32,
)
from vllm.v1.attention.ops.sm70_fp16_grouped import (
    grouped_fp16_fp32_reason,
    load_grouped_fp16_fp32,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")
_warned_feature_fallback = False
_warned_decode_fallback = False
_warned_decode_strict_fallback = False
_logged_prefill_flash = False
_logged_prefill_prefix_flash = False
_logged_prefill_prefix_contig_dense = False
_logged_prefill_prefix_bfla = False
_logged_prefill_prefix_splitkv = False
_logged_prefill_paged_cache = False
_logged_prefill_smallq_decode = False
_logged_prefill_prefix_decode_rows = False
_logged_prefill_prefix_decode_rows_grouped = False
_logged_prefill_smallq_decode_xqa = False
_logged_prefill_smallq_grouped_verify = False
_logged_prefill_smallq_grouped_verify_gate = False
_logged_prefill_triton_safe = False
_logged_decode_flash = False
_logged_decode_dense_reference = False
_logged_decode_dense_cache = False
_logged_decode_paged_prefill = False
_logged_decode_paged_prefill_bhmd = False
_logged_decode_paged_prefill_bhmd_q_clone = False
_logged_decode_wmma_wrapper = False
_logged_fp8_prefill_bridge = False
_logged_prefill_compare = False
_logged_dflash_prefix_dump = False
_logged_prefill_ddtree_dense = False
_logged_prefill_ddtree_triton = False
_logged_prefill_ddtree_triton_fallback = False
_logged_dflash_attention_contracts: set[tuple[object, ...]] = set()


class FlashAttnV100Impl(TritonAttentionImpl):
    """Flash Attention V100 implementation with explicit fallback policy."""

    def __init__(self, *args, **kwargs):
        self.prefix_anchored_decode_window = kwargs.pop(
            "prefix_anchored_decode_window", None
        )
        super().__init__(*args, **kwargs)
        _routing._log_kv_dtype_contract(self.kv_cache_dtype)
        self.kv_cache_dtype = _routing._normalize_flash_v100_kv_cache_dtype(
            self.kv_cache_dtype
        )
        (
            self.flash_attn_func,
            self.flash_attn_bhmd_func,
            self.flash_attn_decode_paged,
            self.flash_attn_decode_paged_xqa,
            self.flash_attn_decode_paged_wmma,
            self.flash_attn_prefill_paged,
            self.flash_attn_prefill_paged_bhmd,
            self.flash_attn_prefill_paged_bfla,
            self.flash_attn_prefill_paged_splitkv,
        ) = _ops._get_flash_ops()
        self.flash_attn_grouped_verify_paged = _ops._get_flash_grouped_verify_op()
        use_e4m3_fp32 = (
            envs.VLLM_FLASH_V100_E4M3_GROUPED_FP32
            and self.kv_codec is FP8_E4M3
            and current_platform.is_device_capability(70)
        )
        self.flash_attn_grouped_e4m3_fp32_paged = (
            load_grouped_e4m3_fp32() if use_e4m3_fp32 else None
        )
        self.flash_attn_grouped_fp16_fp32_paged = (
            load_grouped_fp16_fp32()
            if self.kv_codec is FP16 and current_platform.is_device_capability(70)
            else None
        )
        self._sm70_scalar_tail_attention = None
        from vllm.v1.attention.ops.sm70_e4m3_scalar import (
            load_scalar_tail_attention,
            scalar_tail_attention_available,
        )

        if (
            use_e4m3_fp32
            and envs.VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS
            and (
                envs.VLLM_SM70_DFLASH2_SCALAR_ATTENTION_MANIFEST
                or scalar_tail_attention_available()
            )
            and not os.environ.get("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
        ):
            # An empty name selects the operator compiled into this extension;
            # a manifest name keeps the explicit experimental override.
            self._sm70_scalar_tail_attention = load_scalar_tail_attention(
                envs.VLLM_SM70_DFLASH2_SCALAR_ATTENTION_MANIFEST or "",
                torch.device("cuda", torch.accelerator.current_device_index()),
            )
        if use_e4m3_fp32 and self.flash_attn_grouped_e4m3_fp32_paged is None:
            logger.warning_once(
                "E4M3 grouped FP32 requires Flash-V100 precision revision 4; "
                "the E4M3 scalar fallback also requires this revision for "
                "FP32 partial storage. Rebuild the extension and restart workers.",
                scope="process",
            )
        self.dflash2_grouped_verify_max_query_tokens = (
            _ops._flash_attn_grouped_verify_max_query_tokens
        )
        self.dflash2_grouped_verify_request_major_abi_version = (
            _ops._flash_attn_grouped_verify_request_major_abi_version
        )
        self.fp8_e5m2_paged_kv_to_fp16 = _ops._get_fp8_e5m2_paged_kv_bridge_op()
        self.fp8_e4m3_paged_kv_to_fp16 = (
            _ops._get_sm70_v37_e4m3_bridge_op() if self.kv_codec is FP8_E4M3 else None
        )
        # V100 FA2 kernels consume fp16 Q. FP8 KV cache support is implemented
        # as storage compression only, with K/V dequantized inside FA2 kernels.
        self.supports_quant_query_input = False
        self.use_flash_v100 = self.flash_attn_func is not None
        self.use_flash_v100_decode = self.flash_attn_decode_paged is not None
        self._flash_decode_paged_kwargs = {
            name
            for name in (
                "window_size",
                "max_seq_len_hint",
                "workspace_seq_capacity_hint",
                "active_num_partitions",
                "partition_size_hint",
                "anchor_lens",
                "anchored_window",
            )
            if self.flash_attn_decode_paged is not None
            and _ops._callable_accepts_keyword(self.flash_attn_decode_paged, name)
        }
        self._flash_prefill_paged_supports_anchor = (
            self.flash_attn_prefill_paged is not None
            and _ops._callable_accepts_keyword(
                self.flash_attn_prefill_paged, "anchor_lens"
            )
        )
        self._flash_prefill_paged_supports_dflash2_bmhd = bool(
            getattr(self.flash_attn_prefill_paged, "_sm70_dflash2_direct_bmhd", False)
        )
        self._flash_prefill_paged_dflash2_split_pages = getattr(
            self.flash_attn_prefill_paged, "_sm70_dflash2_split_pages", ()
        )
        split_enabled = getattr(
            capture_sm70_dflash2_config(), "draft_window_split", True
        )
        if not split_enabled:
            self._flash_prefill_paged_dflash2_split_pages = ()
        if self.flash_attn_prefill_paged is not None and _ops._callable_accepts_keyword(
            self.flash_attn_prefill_paged, "dflash2_window_split"
        ):
            from functools import partial

            self.flash_attn_prefill_paged = partial(
                self.flash_attn_prefill_paged, dflash2_window_split=split_enabled
            )
        logger.info_once(
            "FLASH_ATTN_V100 DFlash single-request window split pages=%s; "
            "page832 policy=%s; dtype/query/window guards apply at dispatch.",
            self._flash_prefill_paged_dflash2_split_pages,
            "enabled" if split_enabled else "disabled_by_configuration",
        )
        paged_prefill_enable = os.getenv("VLLM_FLASH_V100_ENABLE_PAGED_PREFILL")
        paged_prefill_disable = (
            os.getenv("VLLM_FLASH_V100_DISABLE_PAGED_PREFILL", "0") == "1"
        )
        self.use_flash_v100_prefill_paged = (
            self.flash_attn_prefill_paged is not None
            and paged_prefill_enable != "0"
            and not paged_prefill_disable
        )
        self.use_fp8_prefill_bridge = (
            self.fp8_e4m3_paged_kv_to_fp16 is not None
            if self.kv_codec is FP8_E4M3
            else self.fp8_e5m2_paged_kv_to_fp16 is not None
        ) and os.getenv("VLLM_FLASH_V100_FP8_PREFILL_BRIDGE", "1") != "0"
        self.use_flash_v100_prefill_splitkv = (
            self.flash_attn_prefill_paged_splitkv is not None
            and envs.VLLM_FLASH_V100_PREFILL_SPLIT_KV
            and self.use_flash_v100_prefill_paged
        )
        self.use_flash_v100_prefill_bfla = (
            self.flash_attn_prefill_paged_bfla is not None
            and envs.VLLM_FLASH_V100_BFLA_PREFILL
            and self.use_flash_v100_prefill_paged
        )
        self.use_flash_v100_prefill_contig_dense = (
            self.flash_attn_func is not None
            and self.use_flash_v100_prefill_paged
            and envs.VLLM_FLASH_V100_PREFILL_CONTIG_DENSE
        )
        self.prefill_contig_dense_min_q = (
            envs.VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_Q
        )
        self.prefill_contig_dense_min_kv = (
            envs.VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_KV
        )
        self.prefill_contig_dense_allow_copy = (
            envs.VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_ALLOW_COPY
        )
        self.use_flash_v100_prefill_gather_dense = (
            self.use_flash_v100_prefill_paged
            and envs.VLLM_FLASH_V100_PREFILL_GATHER_DENSE
        )
        self.prefill_gather_dense_min_q = (
            envs.VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_Q
        )
        self.prefill_gather_dense_min_kv = (
            envs.VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_KV
        )
        self.prefill_split_kv_tokens = envs.VLLM_FLASH_V100_PREFILL_SPLIT_KV_TOKENS
        self.prefill_split_kv_min_q = envs.VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_Q
        self.prefill_split_kv_max_q = envs.VLLM_FLASH_V100_PREFILL_SPLIT_KV_MAX_Q
        self.prefill_split_kv_min_kv = envs.VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_KV
        self.prefill_bfla_min_q = envs.VLLM_FLASH_V100_BFLA_MIN_Q
        self.prefill_bfla_min_kv = envs.VLLM_FLASH_V100_BFLA_MIN_KV
        self.prefill_bfla_mask_block_n = envs.VLLM_FLASH_V100_BFLA_MASK_BLOCK_N
        self.use_prefill_paged_cache = (
            os.getenv("VLLM_FLASH_V100_PREFILL_USE_PAGED_CACHE", "0") == "1"
        )
        # Explicit diagnostic fallback only. The production migration target is
        # a complete Flash-V100 backend, so selected Flash routes should not
        # hide Flash prefill issues behind Triton by default.
        self.use_triton_prefill = (
            os.getenv("VLLM_FLASH_V100_PREFILL_USE_TRITON", "0") != "0"
        )
        self.allow_triton_fallback = (
            os.getenv("VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK", "0") == "1"
        )
        self.smallq_decode_max_query_len = int(
            os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16")
        )
        self.smallq_decode_max_model_len = int(
            os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN", "0")
        )
        self.use_decode_dense_reference = (
            os.getenv("VLLM_FLASH_V100_DECODE_DENSE_REFERENCE", "0") == "1"
        )
        self.use_decode_dense_cache = (
            os.getenv("VLLM_FLASH_V100_DECODE_DENSE_CACHE", "0") == "1"
        )
        # Classified quality rule: long q=1 scalar paged decode is a Type-B
        # reduction-order path, not a Type-A layout bug. Keep it as the
        # production Flash decode default so an explicit FLASH_ATTN_V100
        # selection does not silently become Triton during CUDA graph capture.
        decode_paged_prefill_env = os.getenv("VLLM_FLASH_V100_DECODE_USE_PAGED_PREFILL")
        self.use_decode_paged_prefill = decode_paged_prefill_env == "1"
        decode_bhmd_out_env = os.getenv("VLLM_FLASH_V100_DECODE_USE_BHMD_OUT")
        self.use_decode_paged_prefill_bhmd_out = decode_bhmd_out_env != "0"
        self.use_decode_wmma_wrapper = (
            os.getenv("VLLM_FLASH_V100_DECODE_USE_WMMA_WRAPPER", "0") == "1"
        )
        self.use_decode_xqa = os.getenv("VLLM_FLASH_V100_DECODE_USE_XQA", "1") == "1"
        self.use_smallq_decode_xqa = (
            self.use_decode_xqa
            and os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_USE_XQA", "1") == "1"
        )
        self.use_dflash2_grouped_verify = (
            self.flash_attn_grouped_verify_paged is not None
            and envs.VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY
            and current_platform.is_device_capability(70)
        )
        self.use_dflash2_batched_grouped_verify = (
            self.use_dflash2_grouped_verify
            and envs.VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY
        )
        self.dflash2_grouped_verify_min_model_len = (
            envs.VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY_MIN_MODEL_LEN
        )
        if self.dflash2_grouped_verify_min_model_len < 1:
            raise ValueError(
                "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY_MIN_MODEL_LEN must be positive"
            )
        decode_scalar_paged_env = os.getenv("VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED")
        self.use_decode_scalar_paged = decode_scalar_paged_env != "0"
        self.compare_bhmd_out_dir = os.getenv("VLLM_FLASH_V100_COMPARE_BHMD_OUT_DIR")
        self.compare_bhmd_out_max_calls = int(
            os.getenv("VLLM_FLASH_V100_COMPARE_BHMD_OUT_MAX_CALLS", "0")
        )
        self._compare_bhmd_out_calls = 0
        self.compare_triton_out_dir = os.getenv(
            "VLLM_FLASH_V100_COMPARE_TRITON_OUT_DIR"
        )
        self.compare_triton_out_max_calls = int(
            os.getenv("VLLM_FLASH_V100_COMPARE_TRITON_OUT_MAX_CALLS", "0")
        )
        self.compare_triton_tensor_dump_dir = os.getenv(
            "VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_DIR"
        )
        self.compare_triton_tensor_dump_max_tokens = int(
            os.getenv("VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_MAX_TOKENS", "64")
        )
        self._compare_triton_out_calls = 0
        self._decode_cache_k: torch.Tensor | None = None
        self._decode_cache_v: torch.Tensor | None = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0

        if self.prefix_anchored_decode_window is not None:
            if (
                self.prefix_anchored_decode_window <= 0
                or self.attn_type != AttentionType.DECODER
                or self.kv_codec is not FP16
            ):
                raise ValueError(
                    "prefix-anchored SWA requires a positive window, causal "
                    "decoder attention, and an fp16 KV cache"
                )
            if self.use_triton_prefill:
                raise ValueError(
                    "prefix-anchored SWA cannot use the Triton prefill fallback"
                )
            if (
                not self.use_flash_v100_decode
                or not self.use_decode_scalar_paged
                or not {"anchor_lens", "anchored_window"}
                <= self._flash_decode_paged_kwargs
            ):
                raise RuntimeError(
                    "prefix-anchored SWA requires the masked scalar paged "
                    "decode extension"
                )
            if (
                not self.use_flash_v100_prefill_paged
                or not self._flash_prefill_paged_supports_anchor
            ):
                raise RuntimeError(
                    "prefix-anchored SWA requires the masked paged prefill extension"
                )

            # Select the only two routes that carry the anchored mask once at
            # construction time. The default-off hot path therefore retains
            # its existing route predicates without extra metadata parsing.
            self.smallq_decode_max_query_len = 0
            self.use_decode_paged_prefill = False
            self.use_decode_dense_cache = False
            self.use_decode_dense_reference = False
            self.use_decode_xqa = False
            self.use_smallq_decode_xqa = False
            self.use_flash_v100_prefill_splitkv = False
            self.use_flash_v100_prefill_bfla = False
            self.use_flash_v100_prefill_contig_dense = False
            self.use_flash_v100_prefill_gather_dense = False

    def _reset_decode_cache(self) -> None:
        self._decode_cache_k = None
        self._decode_cache_v = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0

    def _ensure_decode_cache_capacity(
        self,
        required_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and self._decode_cache_capacity >= required_len
            and self._decode_cache_k.shape[1] == num_kv_heads
            and self._decode_cache_k.shape[2] == head_dim
            and self._decode_cache_k.dtype == dtype
            and self._decode_cache_k.device == device
        ):
            return

        new_capacity = max(required_len, max(16, self._decode_cache_capacity * 2))
        new_k = torch.empty(
            (new_capacity, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        new_v = torch.empty(
            (new_capacity, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        if (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and self._decode_cache_len > 0
        ):
            new_k[: self._decode_cache_len].copy_(
                self._decode_cache_k[: self._decode_cache_len]
            )
            new_v[: self._decode_cache_len].copy_(
                self._decode_cache_v[: self._decode_cache_len]
            )

        self._decode_cache_k = new_k
        self._decode_cache_v = new_v
        self._decode_cache_capacity = new_capacity

    def _get_decode_kv_single_seq(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        seq_lens_cpu: torch.Tensor,
        block_size: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(seq_lens_cpu[0])
        q_len = int(attn_metadata.num_actual_tokens)
        num_kv_heads = key.shape[1]

        cache_hit = (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and seq_len > self._decode_cache_len
            and seq_len - q_len == self._decode_cache_len
        )

        if not cache_hit:
            k_cont, v_cont = _kv_layout._extract_contiguous_kv_from_paged_cache(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=seq_len,
            )
            self._ensure_decode_cache_capacity(
                seq_len,
                num_kv_heads,
                head_dim,
                k_cont.dtype,
                k_cont.device,
            )
            assert self._decode_cache_k is not None
            assert self._decode_cache_v is not None
            self._decode_cache_k[:seq_len].copy_(k_cont)
            self._decode_cache_v[:seq_len].copy_(v_cont)
            self._decode_cache_len = seq_len
            return (
                self._decode_cache_k[:seq_len],
                self._decode_cache_v[:seq_len],
            )

        self._ensure_decode_cache_capacity(
            seq_len,
            num_kv_heads,
            head_dim,
            key.dtype,
            key.device,
        )
        assert self._decode_cache_k is not None
        assert self._decode_cache_v is not None
        self._decode_cache_k[self._decode_cache_len : seq_len].copy_(key[:q_len])
        self._decode_cache_v[self._decode_cache_len : seq_len].copy_(value[:q_len])
        self._decode_cache_len = seq_len
        return (
            self._decode_cache_k[:seq_len],
            self._decode_cache_v[:seq_len],
        )

    def _maybe_compare_bhmd_out(
        self,
        layer: torch.nn.Module,
        q_bhmd: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        safe_bmhd: torch.Tensor,
    ) -> None:
        call_idx = self._reserve_bhmd_compare_call()
        if call_idx is None or self.flash_attn_prefill_paged_bhmd is None:
            return

        raw_bmhd = torch.empty_like(safe_bmhd)
        raw_bhmd = raw_bmhd.permute(0, 2, 1, 3)
        self.flash_attn_prefill_paged_bhmd(
            q_bhmd,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=self.scale,
            out=raw_bhmd,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            causal=True,
        )
        self._write_bhmd_compare_report(
            raw_bmhd,
            safe_bmhd,
            call_idx,
            "scratch_raw_vs_safe",
            {
                "q_bhmd_stride": list(q_bhmd.stride()),
                "raw_bhmd_stride": list(raw_bhmd.stride()),
                "raw_bhmd_contiguous": raw_bhmd.is_contiguous(),
            },
        )

    def _reserve_bhmd_compare_call(self) -> int | None:
        if (
            not self.compare_bhmd_out_dir
            or self.compare_bhmd_out_max_calls <= 0
            or self._compare_bhmd_out_calls >= self.compare_bhmd_out_max_calls
        ):
            return None

        call_idx = self._compare_bhmd_out_calls
        self._compare_bhmd_out_calls += 1
        return call_idx

    def _reserve_triton_compare_call(self) -> int | None:
        if (
            not self.compare_triton_out_dir
            or self.compare_triton_out_max_calls <= 0
            or self._compare_triton_out_calls >= self.compare_triton_out_max_calls
        ):
            return None

        call_idx = self._compare_triton_out_calls
        self._compare_triton_out_calls += 1
        return call_idx

    def _write_bhmd_compare_report(
        self,
        candidate_bmhd: torch.Tensor,
        reference_bmhd: torch.Tensor,
        call_idx: int,
        mode: str,
        extra: dict[str, object],
    ) -> None:
        assert self.compare_bhmd_out_dir is not None
        diff = candidate_bmhd - reference_bmhd
        report = {
            "call_idx": call_idx,
            "mode": mode,
            "equal": bool(torch.equal(candidate_bmhd, reference_bmhd)),
            "max_diff": float(diff.abs().max().item()),
            "mean_diff": float(diff.abs().float().mean().item()),
            "num_different": int((candidate_bmhd != reference_bmhd).sum().item()),
            "shape_bmhd": list(reference_bmhd.shape),
            "pid": os.getpid(),
        }
        report.update(extra)

        os.makedirs(self.compare_bhmd_out_dir, exist_ok=True)
        file_name = (
            f"bhmd_compare_pid{os.getpid()}_call{call_idx}_{time.time_ns()}.json"
        )
        path = os.path.join(self.compare_bhmd_out_dir, file_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")

    def _write_triton_compare_report(
        self,
        candidate: torch.Tensor,
        reference: torch.Tensor,
        call_idx: int,
        stage: str,
        extra: dict[str, object],
    ) -> None:
        assert self.compare_triton_out_dir is not None
        diff = candidate.float() - reference.float()
        abs_diff = diff.abs()
        report = {
            "call_idx": call_idx,
            "stage": stage,
            "equal": bool(torch.equal(candidate, reference)),
            "max_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
            "mean_diff": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
            "num_different": int((candidate != reference).sum().item()),
            "shape": list(candidate.shape),
            "dtype": str(candidate.dtype),
            "candidate_nan_count": int(torch.isnan(candidate).sum().item()),
            "reference_nan_count": int(torch.isnan(reference).sum().item()),
            "pid": os.getpid(),
        }
        report.update(extra)

        os.makedirs(self.compare_triton_out_dir, exist_ok=True)
        file_name = (
            f"triton_out_compare_pid{os.getpid()}_call{call_idx}_"
            f"{stage}_{time.time_ns()}.json"
        )
        path = os.path.join(self.compare_triton_out_dir, file_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")

    def _maybe_write_triton_tensor_dump(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        candidate: torch.Tensor,
        reference: torch.Tensor,
        call_idx: int,
        stage: str,
        num_actual_tokens: int,
    ) -> dict[str, object]:
        if not self.compare_triton_tensor_dump_dir:
            return {}
        if num_actual_tokens > self.compare_triton_tensor_dump_max_tokens:
            return {
                "tensor_dump_skipped": "num_actual_tokens_exceeds_limit",
                "tensor_dump_max_tokens": self.compare_triton_tensor_dump_max_tokens,
            }

        payload: dict[str, object] = {
            "call_idx": call_idx,
            "stage": stage,
            "num_actual_tokens": num_actual_tokens,
            "scale": self.scale,
            "kv_cache_dtype": self.kv_cache_dtype,
            "layer": self._layer_debug_info(layer),
            "query": query[:num_actual_tokens].detach().cpu(),
            "raw_key": key[:num_actual_tokens].detach().cpu(),
            "raw_value": value[:num_actual_tokens].detach().cpu(),
            "candidate_output": candidate[:num_actual_tokens].detach().cpu(),
            "triton_reference_output": reference[:num_actual_tokens].detach().cpu(),
            "query_start_loc": attn_metadata.query_start_loc.detach().cpu(),
            "seq_lens": attn_metadata.seq_lens.detach().cpu(),
            "block_table": attn_metadata.block_table.detach().cpu(),
        }

        if stage in ("prefill_no_prefix", "prefill_no_prefix_paged_cache"):
            key_cache, _ = _kv_layout._split_paged_kv_cache(kv_cache)
            block_size = key_cache.shape[1]
            num_kv_heads = key_cache.shape[2]
            head_dim = key_cache.shape[3]
            query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
            query_start_loc = (
                query_start_loc_cpu
                if query_start_loc_cpu is not None
                else attn_metadata.query_start_loc
            )
            num_seqs = len(query_start_loc) - 1
            k_cont, v_cont = _kv_layout._extract_contiguous_kv_from_paged_cache(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table[:num_seqs],
                seq_lens=attn_metadata.seq_lens[:num_seqs],
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=num_actual_tokens,
            )
            k_cont, v_cont = _kv_layout._dequantize_fp8_contiguous_kv(
                k_cont,
                v_cont,
                self.kv_cache_dtype,
                float(layer._k_scale_float),
                float(layer._v_scale_float),
            )
            payload["cache_key"] = k_cont.detach().cpu()
            payload["cache_value"] = v_cont.detach().cpu()

        os.makedirs(self.compare_triton_tensor_dump_dir, exist_ok=True)
        file_name = (
            f"triton_tensor_dump_pid{os.getpid()}_call{call_idx}_"
            f"{stage}_{time.time_ns()}.pt"
        )
        path = os.path.join(self.compare_triton_tensor_dump_dir, file_name)
        torch.save(payload, path)
        return {"tensor_dump_path": path}

    @staticmethod
    def _small_tensor_list(
        tensor: torch.Tensor | None,
        limit: int = 32,
    ) -> list[int] | None:
        if tensor is None:
            return None
        flat = tensor.detach().cpu().reshape(-1)
        return [int(x) for x in flat[:limit].tolist()]

    @staticmethod
    def _layer_debug_info(layer: torch.nn.Module) -> dict[str, object]:
        return {
            "layer_name": getattr(layer, "layer_name", None),
            "is_dflash_draft_attn": getattr(layer, "is_dflash_draft_attn", False),
            "kv_sharing_target_layer_name": getattr(
                layer, "kv_sharing_target_layer_name", None
            ),
            "impl_kv_sharing_target_layer_name": getattr(
                getattr(layer, "impl", None), "kv_sharing_target_layer_name", None
            ),
        }

    @staticmethod
    def _tensor_compare_stats(
        candidate: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, object]:
        if candidate.shape != reference.shape:
            return {
                "shape_mismatch": True,
                "candidate_shape": list(candidate.shape),
                "reference_shape": list(reference.shape),
            }

        diff = candidate.float() - reference.float()
        abs_diff = diff.abs()
        return {
            "shape_mismatch": False,
            "equal": bool(torch.equal(candidate, reference)),
            "max_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
            "mean_diff": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
            "num_different": int((candidate != reference).sum().item()),
            "candidate_dtype": str(candidate.dtype),
            "reference_dtype": str(reference.dtype),
            "candidate_abs_max": float(candidate.float().abs().max().item())
            if candidate.numel()
            else 0.0,
            "reference_abs_max": float(reference.float().abs().max().item())
            if reference.numel()
            else 0.0,
            "candidate_mean": float(candidate.float().mean().item())
            if candidate.numel()
            else 0.0,
            "reference_mean": float(reference.float().mean().item())
            if reference.numel()
            else 0.0,
            "shape": list(candidate.shape),
        }

    def _prefill_raw_kv_cache_compare_stats(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        num_actual_tokens: int,
    ) -> dict[str, object]:
        key_cache, _ = _kv_layout._split_paged_kv_cache(kv_cache)

        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        num_seqs = len(query_start_loc) - 1
        k_cont, v_cont = _kv_layout._extract_contiguous_kv_from_paged_cache(
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table[:num_seqs],
            seq_lens=attn_metadata.seq_lens[:num_seqs],
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            total_tokens=num_actual_tokens,
        )
        k_cont, v_cont = _kv_layout._dequantize_fp8_contiguous_kv(
            k_cont,
            v_cont,
            self.kv_cache_dtype,
            float(layer._k_scale_float),
            float(layer._v_scale_float),
        )
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        return {
            "raw_key_vs_cache": self._tensor_compare_stats(
                key[:num_actual_tokens], k_cont
            ),
            "raw_value_vs_cache": self._tensor_compare_stats(
                value[:num_actual_tokens], v_cont
            ),
            "kv_cache_dtype": str(kv_cache.dtype),
            "kv_cache_shape": list(kv_cache.shape),
            "query_start_loc": self._small_tensor_list(attn_metadata.query_start_loc),
            "query_start_loc_cpu": self._small_tensor_list(query_start_loc_cpu),
            "seq_lens": self._small_tensor_list(attn_metadata.seq_lens),
            "seq_lens_cpu": self._small_tensor_list(seq_lens_cpu),
            "block_table_shape": list(attn_metadata.block_table.shape),
            "block_table_first_row": self._small_tensor_list(
                attn_metadata.block_table[:1]
            ),
        }

    def _maybe_compare_triton_output(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
        stage: str,
    ) -> None:
        call_idx = self._reserve_triton_compare_call()
        if call_idx is None:
            return
        if query.is_cuda and torch.cuda.is_current_stream_capturing():
            return

        reference = torch.empty_like(output)
        super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            reference,
            output_scale,
            output_block_scale,
        )
        if query.is_cuda:
            # Diagnostic-only: the Triton reference path may update KV cache
            # asynchronously before this hook extracts raw-vs-cache tensors.
            torch.accelerator.synchronize(query.device)
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        extra = {
            "num_actual_tokens": num_actual_tokens,
            "max_query_len": int(attn_metadata.max_query_len),
            "max_seq_len": int(attn_metadata.max_seq_len),
            "layer_type": type(layer).__name__,
        }
        extra.update(self._layer_debug_info(layer))
        if stage in ("prefill_no_prefix", "prefill_no_prefix_paged_cache"):
            extra.update(
                self._prefill_raw_kv_cache_compare_stats(
                    layer,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    num_actual_tokens,
                )
            )
        extra.update(
            self._maybe_write_triton_tensor_dump(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                reference,
                call_idx,
                stage,
                num_actual_tokens,
            )
        )
        self._write_triton_compare_report(
            output[:num_actual_tokens],
            reference[:num_actual_tokens],
            call_idx,
            stage,
            extra,
        )

    @property
    def kv_codec(self) -> KVCodec | None:
        """Storage codec of this layer's KV cache."""
        return resolve_kv_codec(self.kv_cache_dtype)

    def _xqa_kv_codec(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> KVCodec | None:
        """The codec when the XQA operator reads this cache natively."""
        codec = self.kv_codec
        if codec not in (FP16, FP8_E4M3, FP8_E5M2) or not codec.stores(
            key_cache, value_cache
        ):
            return None
        # The E4M3 XQA wave route retains half partials. A DFlash2 selector
        # target keeps its dedicated grouped FP32 route instead.
        if codec is FP8_E4M3 and getattr(
            attn_metadata, "is_dflash_selector_target", False
        ):
            return None
        return codec

    def _supports_flash_v100_path(self) -> bool:
        """Check whether current layer/config can run Flash V100 safely."""
        supported_kv_dtype = not _routing._uses_fp8_kv_cache(
            self.kv_cache_dtype
        ) or self.kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2")
        return (
            self.use_flash_v100
            and self.attn_type == AttentionType.DECODER
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and supported_kv_dtype
        )

    def _flash_v100_has_sliding_window(self) -> bool:
        sliding_window = self.sliding_window
        if sliding_window is None:
            return False
        return tuple(sliding_window) != (-1, -1)

    def _flash_v100_window_size(self, causal: bool) -> tuple[int, int]:
        if not self._flash_v100_has_sliding_window():
            return (-1, -1)
        left, right = tuple(self.sliding_window)
        left = int(left)
        right = int(right)
        if not causal and left >= 0 and right == 0:
            right = left
        return (left, right)

    def _validate_dflash_attention_contract(
        self,
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
        if signature not in _logged_dflash_attention_contracts:
            _logged_dflash_attention_contracts.add(signature)
            logger.info(
                "FLASH_ATTN_V100 DFlash attention contract: layer=%s "
                "causal=%s window=%s rope_neox=%s.",
                signature[0],
                actual_causal,
                actual_window,
                signature[3],
            )

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
    ) -> None:
        scalar_tail = getattr(self, "_sm70_scalar_tail_attention", None)
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
            _routing._record_route(
                _routing.ROUTE_SPECS["decode_e4m3_compact_scalar_tail"].name
            )
            return
        kwargs: dict[str, object] = {
            "softmax_scale": softmax_scale,
            "out": out,
            "kv_cache_dtype": kv_cache_dtype,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }
        if "window_size" in self._flash_decode_paged_kwargs:
            kwargs["window_size"] = window_size
        elif tuple(window_size) != (-1, -1):
            raise RuntimeError(
                "FLASH_ATTN_V100 decode op does not support sliding-window "
                "attention with this extension build."
            )
        if anchor_lens is not None and anchored_window > 0:
            if "anchor_lens" not in self._flash_decode_paged_kwargs:
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
            if name in self._flash_decode_paged_kwargs:
                kwargs[name] = value
        self.flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            **kwargs,
        )

    def _dflash2_grouped_verify_allowed(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        *,
        num_query_tokens: int,
    ) -> bool:
        """Gate the verifier on its hardware and tensor-layout contract."""
        global _logged_prefill_smallq_grouped_verify_gate
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
            and not _logged_prefill_smallq_grouped_verify_gate
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
            _logged_prefill_smallq_grouped_verify_gate = True
        return allowed

    def _call_dflash2_grouped_verify(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        *,
        out: torch.Tensor,
    ) -> None:
        global _logged_prefill_smallq_grouped_verify
        num_reqs = int(attn_metadata.block_table.shape[0])
        if not _logged_prefill_smallq_grouped_verify:
            logger.info(
                "FLASH_ATTN_V100 DFlash2 exact grouped verifier active "
                "(request-major B%d/q%d/H6/Hkv1/D256, %s KV, one-pass).",
                num_reqs,
                query.shape[0] // num_reqs,
                self.kv_cache_dtype,
            )
            _logged_prefill_smallq_grouped_verify = True
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
        global _logged_prefill_smallq_decode_xqa
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
            _routing._log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "grouped_fp32"
            )
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
            if not _logged_prefill_smallq_decode_xqa:
                logger.info(
                    "FLASH_ATTN_V100 MTP verifier XQA path active "
                    "(rows=%d, q_per_kv=%d, partition_hint=%s, "
                    "mtp5_dual_cta=%s).",
                    int(query.shape[0]),
                    int(query.shape[1] // key_cache.shape[2]),
                    verifier_partition_size_hint,
                    verifier_partition_size_hint is not None,
                )
                _logged_prefill_smallq_decode_xqa = True
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
            _routing._record_route(
                _routing.ROUTE_SPECS["prefill_smallq_decode_xqa"].name
            )
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
        _routing._record_route(
            _routing.ROUTE_SPECS["prefill_smallq_decode_scalar"].name
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
        window = self.prefix_anchored_decode_window
        if window is None:
            return None, 0

        metadata_window = getattr(attn_metadata, "decode_sliding_window", None)
        anchor_lens = getattr(attn_metadata, "prefix_anchor_lens", None)
        if (
            self.attn_type != AttentionType.DECODER
            or self.kv_codec is not FP16
            or metadata_window != window
            or anchor_lens is None
        ):
            raise RuntimeError(
                "FLASH_ATTN_V100 prefix-anchored SWA metadata does not match "
                "the enabled decoder-layer contract"
            )
        return anchor_lens, int(window)

    def _small_query_decode_enabled(
        self,
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

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward path.

        - Prefill: use Flash-V100 by default. Triton prefill is an explicit
          diagnostic fallback only.
        - Decode: use scalar paged Flash-V100 by default, including CUDA graph
          capture/replay, so selecting this backend is not a no-op in
          production decode. Mixed Triton/Flash routes are never silent.
        """
        global _logged_decode_flash, _logged_prefill_flash
        global _logged_prefill_paged_cache
        global _logged_prefill_prefix_flash
        global _logged_prefill_triton_safe
        global _warned_decode_fallback
        global _warned_decode_strict_fallback, _warned_feature_fallback

        if attn_metadata is None:
            assert output is not None
            if (
                self.attn_type == AttentionType.DECODER
                and self.sliding_window == (-1, -1)
                and self.alibi_slopes is None
                and not self.logits_soft_cap
                and self.sinks is None
                and abs(self.scale - 0.0625) <= 1.0e-8
            ):
                _dense_prefill._profile_sm70_prefill_workspace(query, self.num_kv_heads)
            _routing._record_route(
                _routing.ROUTE_SPECS["metadata_none_zero_output"].name
            )
            return output.fill_(0)

        self._validate_dflash_attention_contract(layer, attn_metadata)

        if not self._supports_flash_v100_path():
            layer_info = self._layer_debug_info(layer)
            is_dflash_draft_attn = bool(layer_info.get("is_dflash_draft_attn"))
            message = (
                "FLASH_ATTN_V100 cannot run this layer/config because a required "
                "Flash op is unavailable or the attention features/KV cache dtype "
                "are unsupported. Select TRITON_ATTN for a full Triton route, or "
                "set VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1 for explicit "
                "diagnostic fallback. "
                f"Details: layer={layer_info.get('layer_name')!r}, "
                f"flash_ops_available={self.use_flash_v100}, "
                f"attn_type={self.attn_type!r}, "
                f"has_alibi={self.alibi_slopes is not None}, "
                f"logits_soft_cap={self.logits_soft_cap!r}, "
                f"has_sinks={self.sinks is not None}, "
                f"kv_cache_dtype={self.kv_cache_dtype!r}."
            )
            if not (self.allow_triton_fallback or is_dflash_draft_attn):
                raise RuntimeError(message)
            if self.use_flash_v100 and not _warned_feature_fallback:
                if is_dflash_draft_attn:
                    logger.warning(
                        "FLASH_ATTN_V100 falling back to Triton for D-Flash "
                        "draft attention layer %s because the SM70 Flash-V100 "
                        "backend does not yet support this layer/config.",
                        layer_info.get("layer_name"),
                    )
                else:
                    logger.warning("%s", message)
                _warned_feature_fallback = True
            _routing._record_route(
                "dflash_draft_triton_fallback"
                if is_dflash_draft_attn
                else "unsupported_triton_fallback"
            )
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        assert output is not None
        is_prefill = attn_metadata.max_query_len > 1
        is_capturing = _routing._is_cuda_graph_capturing(query)
        layer_name = self._layer_debug_info(layer).get("layer_name")
        if _debug._draft_graph_debug_enabled():
            _debug._draft_graph_debug_log(
                "forward:enter",
                "layer=%s is_prefill=%s is_capturing=%s max_query_len=%s "
                "max_seq_len=%s num_actual_tokens=%s %s %s %s %s %s %s",
                layer_name,
                is_prefill,
                is_capturing,
                int(attn_metadata.max_query_len),
                int(attn_metadata.max_seq_len),
                int(attn_metadata.num_actual_tokens),
                _debug._format_tensor_debug(query, "query"),
                _debug._format_tensor_debug(output, "output"),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "query_start_loc", None),
                    "attn_qsl",
                ),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "seq_lens", None),
                    "attn_seq",
                ),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "block_table", None),
                    "attn_bt",
                ),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "smallq_decode_seq_lens", None),
                    "smallq_seq",
                ),
            )
        _debug._sm70_profile_trace(
            "forward enter layer=%s q_shape=%s k_shape=%s v_shape=%s "
            "kv_shape=%s is_prefill=%s is_capturing=%s max_query_len=%s "
            "max_seq_len=%s num_actual_tokens=%s use_decode_scalar=%s "
            "use_decode_paged_prefill=%s use_prefill_paged=%s "
            "use_triton_prefill=%s",
            layer_name,
            tuple(query.shape),
            tuple(key.shape),
            tuple(value.shape),
            tuple(kv_cache.shape) if hasattr(kv_cache, "shape") else None,
            is_prefill,
            is_capturing,
            int(attn_metadata.max_query_len),
            int(attn_metadata.max_seq_len),
            int(attn_metadata.num_actual_tokens),
            self.use_decode_scalar_paged,
            self.use_decode_paged_prefill,
            self.use_flash_v100_prefill_paged,
            self.use_triton_prefill,
        )

        if is_prefill:
            available_query_tokens = min(
                int(query.shape[0]),
                int(key.shape[0]),
                int(value.shape[0]),
                int(output.shape[0]),
            )
            metadata_live_token_mismatch = (
                _kv_layout._metadata_expects_more_query_tokens_than_available(
                    attn_metadata,
                    available_query_tokens,
                )
            )
            if self.use_triton_prefill:
                if not _logged_prefill_triton_safe:
                    logger.info(
                        "FLASH_ATTN_V100 prefill uses explicit Triton diagnostic "
                        "fallback because VLLM_FLASH_V100_PREFILL_USE_TRITON=1; "
                        "this mixed route is not a final performance path."
                    )
                    _logged_prefill_triton_safe = True
                _debug._sm70_profile_trace(
                    "forward branch=prefill_triton_safe layer=%s",
                    layer_name,
                )
                self._reset_decode_cache()
                _routing._record_route(_routing.ROUTE_SPECS["prefill_triton_safe"].name)
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            if is_capturing:
                # CUDA graph capture uses dummy metadata whose seq_lens can
                # look like no-prefix prefill, while replayed MTP verification
                # is a uniform small-query decode over an existing KV prefix.
                # Capture the same small-query kernel branch that replay needs.
                is_dflash_non_causal = bool(
                    getattr(layer, "is_dflash_draft_attn", False)
                ) and not bool(getattr(attn_metadata, "causal", True))
                if is_dflash_non_causal:
                    # DFlash pre-inserts target context K/V before replay. Its
                    # dummy capture has seq_len == query_len and would
                    # otherwise freeze the no-prefix dense branch into the
                    # graph. Bind directly to the non-causal paged-prefix
                    # kernel; runtime updates its persistent sequence and
                    # block-table buffers before every replay.
                    _routing._record_route(
                        _routing.ROUTE_SPECS[
                            "prefill_capture_dflash_noncausal_paged"
                        ].name
                    )
                    return self._flash_v100_prefill_with_prefix(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                    )
                smallq_decode = self._small_query_decode_enabled(attn_metadata)
                if smallq_decode:
                    if _debug._draft_graph_debug_enabled():
                        _debug._draft_graph_debug_log(
                            "forward:prefill_capture_smallq",
                            "layer=%s %s %s %s",
                            layer_name,
                            _debug._format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_decode_block_table",
                                    None,
                                ),
                                "smallq_bt",
                            ),
                            _debug._format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_decode_seq_lens",
                                    None,
                                ),
                                "smallq_seq",
                            ),
                            _debug._format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_query_start_loc",
                                    None,
                                ),
                                "smallq_qsl",
                            ),
                        )
                    _debug._sm70_profile_trace(
                        "forward branch=prefill_capture_smallq layer=%s",
                        layer_name,
                    )
                    _routing._record_route(
                        _routing.ROUTE_SPECS["prefill_capture_smallq"].name
                    )
                    if getattr(attn_metadata, "ddtree_parent_ids", None) is None:
                        _routing._record_route(
                            _routing.ROUTE_SPECS[
                                "prefill_capture_smallq_no_ddtree_metadata"
                            ].name
                        )
                    else:
                        _routing._record_route(
                            _routing.ROUTE_SPECS[
                                "prefill_capture_smallq_ddtree_metadata"
                            ].name
                        )
                    return self._flash_v100_prefill_with_prefix(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                    )
                _debug._sm70_profile_trace(
                    "forward branch=prefill_capture_full_flash layer=%s",
                    layer_name,
                )
            has_prefix_context = (
                metadata_live_token_mismatch
                or _kv_layout._has_prefix_context(attn_metadata)
            )
            smallq_decode = has_prefix_context and self._small_query_decode_enabled(
                attn_metadata
            )
            if has_prefix_context:
                if _debug._draft_graph_debug_enabled():
                    _debug._draft_graph_debug_log(
                        "forward:prefill_prefix",
                        "layer=%s smallq=%s metadata_live_token_mismatch=%s %s %s %s",
                        layer_name,
                        smallq_decode,
                        metadata_live_token_mismatch,
                        _debug._format_tensor_debug(
                            getattr(attn_metadata, "smallq_decode_block_table", None),
                            "smallq_bt",
                        ),
                        _debug._format_tensor_debug(
                            getattr(attn_metadata, "smallq_decode_seq_lens", None),
                            "smallq_seq",
                        ),
                        _debug._format_tensor_debug(
                            getattr(attn_metadata, "smallq_query_start_loc", None),
                            "smallq_qsl",
                        ),
                    )
                _debug._sm70_profile_trace(
                    "forward branch=prefill_prefix layer=%s smallq=%s",
                    layer_name,
                    smallq_decode,
                )
                if not _logged_prefill_prefix_flash:
                    if smallq_decode:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via small-query paged decode)."
                        )
                    elif self.use_flash_v100_prefill_paged:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via direct paged prefill kernel)."
                        )
                    else:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via paged-KV gather)."
                        )
                    _logged_prefill_prefix_flash = True
                if metadata_live_token_mismatch:
                    logger.info(
                        "FLASH_ATTN_V100 prefill switched to prefix/live-token "
                        "path because layer QKV tokens (%d) are shorter than "
                        "query metadata span.",
                        available_query_tokens,
                    )
                _routing._log_fp8_kv_cache_route(
                    "prefill", self.kv_cache_dtype, "prefix"
                )
                self._reset_decode_cache()
                result = self._flash_v100_prefill_with_prefix(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                )
                self._maybe_compare_triton_output(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                    "prefill_prefix",
                )
                _routing._record_route(
                    _routing.ROUTE_SPECS["prefill_prefix_flash"].name
                )
                return result
            if not _logged_prefill_flash:
                logger.info(
                    "FLASH_ATTN_V100 prefill path active (no prefix/chunked context)."
                )
                _logged_prefill_flash = True
            self._reset_decode_cache()
            if self.use_prefill_paged_cache and self.use_flash_v100_prefill_paged:
                _debug._sm70_profile_trace(
                    "forward branch=prefill_no_prefix_paged_cache layer=%s",
                    layer_name,
                )
                if not _logged_prefill_paged_cache:
                    logger.warning(
                        "FLASH_ATTN_V100 no-prefix prefill is reading paged "
                        "KV cache for strict input-source diagnostics. This "
                        "may be slower than dense raw-KV prefill."
                    )
                    _logged_prefill_paged_cache = True
                _routing._log_fp8_kv_cache_route(
                    "prefill", self.kv_cache_dtype, "no_prefix_paged_cache"
                )
                result = self._flash_v100_prefill_with_prefix(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                )
                self._maybe_compare_triton_output(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                    "prefill_no_prefix_paged_cache",
                )
                _routing._record_route(
                    _routing.ROUTE_SPECS["prefill_no_prefix_paged_cache_flash"].name
                )
                return result
            _debug._sm70_profile_trace(
                "forward branch=prefill_no_prefix_dense layer=%s",
                layer_name,
            )
            result = self._flash_v100_prefill(query, key, value, attn_metadata, output)
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "prefill_no_prefix",
            )
            _routing._record_route(
                _routing.ROUTE_SPECS["prefill_no_prefix_dense_flash"].name
            )
            return result

        if not self.use_flash_v100_decode:
            message = (
                "FLASH_ATTN_V100 decode cannot run because the paged decode op "
                "is unavailable. Select TRITON_ATTN for a full Triton route, or "
                "set VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1 for explicit "
                "diagnostic fallback."
            )
            if not self.allow_triton_fallback:
                raise RuntimeError(message)
            if self.use_flash_v100 and not _warned_decode_fallback:
                logger.warning("%s", message)
                _warned_decode_fallback = True
            _debug._sm70_profile_trace(
                "forward branch=decode_triton_no_flash_decode layer=%s",
                layer_name,
            )
            _routing._record_route(
                _routing.ROUTE_SPECS["decode_triton_no_flash_decode"].name
            )
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if (
            self.use_decode_paged_prefill
            and self.use_flash_v100_prefill_paged
            and not is_capturing
        ):
            _routing._log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "decode_as_paged_prefill"
            )
            _debug._sm70_profile_trace(
                "forward branch=decode_paged_prefill layer=%s",
                layer_name,
            )
            result = self._flash_v100_decode_as_paged_prefill(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "decode_paged_prefill",
            )
            _routing._record_route(_routing.ROUTE_SPECS["decode_paged_prefill"].name)
            return result
        if self.use_decode_dense_cache and not is_capturing:
            _routing._log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "dense_cache_bridge"
            )
            _debug._sm70_profile_trace(
                "forward branch=decode_dense_cache layer=%s",
                layer_name,
            )
            result = self._flash_v100_decode_dense_cache(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
            )
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "decode_dense_cache",
            )
            _routing._record_route(_routing.ROUTE_SPECS["decode_dense_cache"].name)
            return result
        if self.use_decode_dense_reference and not is_capturing:
            _routing._log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "dense_reference_bridge"
            )
            _debug._sm70_profile_trace(
                "forward branch=decode_dense_reference layer=%s",
                layer_name,
            )
            result = self._flash_v100_decode_dense_reference(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "decode_dense_reference",
            )
            _routing._record_route(_routing.ROUTE_SPECS["decode_dense_reference"].name)
            return result
        if not self.use_decode_scalar_paged:
            message = (
                "FLASH_ATTN_V100 decode has no enabled Flash route: scalar "
                "paged decode is disabled and the strict paged-prefill bridge "
                "is unavailable or CUDA graph capture is active. Re-enable "
                "VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED=1, select TRITON_ATTN "
                "for a full Triton route, or set "
                "VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1 for explicit "
                "diagnostic fallback."
            )
            if not self.allow_triton_fallback:
                raise RuntimeError(message)
            if not _warned_decode_strict_fallback:
                logger.warning("%s", message)
                _warned_decode_strict_fallback = True
            _debug._sm70_profile_trace(
                "forward branch=decode_triton_scalar_disabled layer=%s",
                layer_name,
            )
            _routing._record_route(
                _routing.ROUTE_SPECS["decode_triton_scalar_disabled"].name
            )
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if not _logged_decode_flash:
            logger.info(
                "FLASH_ATTN_V100 decode path active (paged KV, "
                "CUDA-graph safe; selected route is reported separately)."
            )
            _logged_decode_flash = True
        if _debug._draft_graph_debug_enabled():
            _debug._draft_graph_debug_log(
                "forward:decode",
                "layer=%s %s %s %s",
                layer_name,
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "query_start_loc", None),
                    "attn_qsl",
                ),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "seq_lens", None),
                    "attn_seq",
                ),
                _debug._format_tensor_debug(
                    getattr(attn_metadata, "block_table", None),
                    "attn_bt",
                ),
            )
        _debug._sm70_profile_trace(
            "forward branch=decode_scalar_paged layer=%s",
            layer_name,
        )
        result = self._flash_v100_decode(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )
        self._maybe_compare_triton_output(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
            "decode_scalar_paged",
        )
        return result

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
        global _logged_decode_paged_prefill
        global _logged_decode_paged_prefill_bhmd
        global _logged_decode_paged_prefill_bhmd_q_clone
        global _logged_decode_wmma_wrapper
        if not _logged_decode_paged_prefill:
            logger.warning(
                "FLASH_ATTN_V100 decode-as-paged-prefill path active. This is "
                "for strict debugging and may be slower than paged decode."
            )
            _logged_decode_paged_prefill = True

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
                    and self.use_decode_wmma_wrapper
                    and self.flash_attn_decode_paged_wmma is not None
                ):
                    if not _logged_decode_wmma_wrapper:
                        logger.info(
                            "FLASH_ATTN_V100 decode WMMA wrapper path active "
                            "(experimental exactness bridge)."
                        )
                        _logged_decode_wmma_wrapper = True
                    q_wmma = q_batch[:, 0].contiguous()
                    out_wmma = out_batch_view[:, 0]
                    self.flash_attn_decode_paged_wmma(
                        q_wmma,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[:num_seqs],
                        attn_metadata.seq_lens[:num_seqs],
                        softmax_scale=self.scale,
                        out=out_wmma,
                        kv_cache_dtype=self.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                    )
                    return output
                if (
                    first_query_len == 1
                    and self.use_decode_paged_prefill_bhmd_out
                    and self.flash_attn_prefill_paged_bhmd is not None
                    and q_bhmd.is_contiguous()
                    and out_bhmd.is_contiguous()
                ):
                    if not _logged_decode_paged_prefill_bhmd:
                        logger.info(
                            "FLASH_ATTN_V100 decode-as-paged-prefill BHMD "
                            "out path active."
                        )
                        _logged_decode_paged_prefill_bhmd = True
                    compare_call_idx = self._reserve_bhmd_compare_call()
                    safe_bmhd = None
                    if compare_call_idx is not None:
                        safe_bmhd = self.flash_attn_prefill_paged(
                            q_batch,
                            key_cache,
                            value_cache,
                            attn_metadata.block_table[:num_seqs],
                            attn_metadata.seq_lens[:num_seqs],
                            softmax_scale=self.scale,
                            kv_cache_dtype=self.kv_cache_dtype,
                            k_scale=float(layer._k_scale_float),
                            v_scale=float(layer._v_scale_float),
                            causal=True,
                        )
                    raw_q_bhmd = q_bhmd
                    q_out_same_storage = _routing._same_storage(raw_q_bhmd, out_bhmd)
                    if q_out_same_storage:
                        if not _logged_decode_paged_prefill_bhmd_q_clone:
                            logger.info(
                                "FLASH_ATTN_V100 BHMD out path cloned Q to "
                                "avoid input/output storage aliasing."
                            )
                            _logged_decode_paged_prefill_bhmd_q_clone = True
                        raw_q_bhmd = q_bhmd.clone()
                    self.flash_attn_prefill_paged_bhmd(
                        raw_q_bhmd,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[:num_seqs],
                        attn_metadata.seq_lens[:num_seqs],
                        softmax_scale=self.scale,
                        out=out_bhmd,
                        kv_cache_dtype=self.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                        causal=True,
                    )
                    if safe_bmhd is not None:
                        assert compare_call_idx is not None
                        self._write_bhmd_compare_report(
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
                out_batch = self.flash_attn_prefill_paged(
                    q_batch,
                    key_cache,
                    value_cache,
                    attn_metadata.block_table[:num_seqs],
                    attn_metadata.seq_lens[:num_seqs],
                    softmax_scale=self.scale,
                    kv_cache_dtype=self.kv_cache_dtype,
                    k_scale=float(layer._k_scale_float),
                    v_scale=float(layer._v_scale_float),
                    causal=True,
                )
                if first_query_len == 1 and q_bhmd.is_contiguous():
                    self._maybe_compare_bhmd_out(
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
            out_seq = self.flash_attn_prefill_paged(
                query[start:end].unsqueeze(0),
                key_cache,
                value_cache,
                attn_metadata.block_table[i : i + 1],
                attn_metadata.seq_lens[i : i + 1],
                softmax_scale=self.scale,
                kv_cache_dtype=self.kv_cache_dtype,
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
        global _logged_decode_dense_cache
        if _routing._uses_fp8_kv_cache(self.kv_cache_dtype):
            if self.use_flash_v100_prefill_paged:
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
        if not _logged_decode_dense_cache:
            logger.warning(
                "FLASH_ATTN_V100 decode dense-cache path active. This is "
                "single-sequence strict debugging and may be slower than paged decode."
            )
            _logged_decode_dense_cache = True

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
            if self.use_flash_v100_prefill_paged:
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
        k_cont, v_cont = self._get_decode_kv_single_seq(
            key,
            value,
            kv_cache,
            attn_metadata,
            attn_metadata.seq_lens[:1],
            block_size,
            head_dim,
        )
        out_seq = self.flash_attn_func(
            query.unsqueeze(0),
            k_cont[:seq_len].unsqueeze(0),
            v_cont[:seq_len].unsqueeze(0),
            causal=True,
            softmax_scale=self.scale,
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
        global _logged_decode_dense_reference
        if not _logged_decode_dense_reference:
            logger.warning(
                "FLASH_ATTN_V100 decode dense-reference path active. This is "
                "for strict debugging and is expected to be slower than paged decode."
            )
            _logged_decode_dense_reference = True

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
                self.kv_cache_dtype,
                float(layer._k_scale_float),
                float(layer._v_scale_float),
            )
            out_seq = self.flash_attn_func(
                query[start:end].unsqueeze(0),
                k_cont.unsqueeze(0),
                v_cont.unsqueeze(0),
                causal=True,
                softmax_scale=self.scale,
            )
            out_view[start:end].copy_(out_seq.squeeze(0))
        return output

    def _flash_v100_prefill(
        self,
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
        if self.prefix_anchored_decode_window is None:
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
                enabled=self.use_decode_xqa,
                available=self.flash_attn_decode_paged_xqa is not None,
                query=query,
                metadata=attn_metadata,
                seq_rows=attn_metadata.seq_lens.shape[0],
                window_size=window_size,
            ),
            ("decode_xqa_paged",),
            fallback="decode_scalar_paged",
        )
        if selection is _routing.ROUTE_SPECS["decode_xqa_paged"]:
            _routing._log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "xqa_paged")
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
                self.kv_cache_dtype,
            )
            if partition_size_hint is not None:
                if (
                    xqa_codec is FP8_E4M3
                    and query.shape[0] == 1
                    and os.getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO", "1")
                    != "0"
                ):
                    _routing._record_route(
                        f"decode_xqa_e4m3_dynamic_page{key_cache.shape[1]}"
                    )
                else:
                    _routing._record_route(
                        f"decode_xqa_p{partition_size_hint}_page{key_cache.shape[1]}"
                    )
            self.flash_attn_decode_paged_xqa(
                query,
                key_cache,
                value_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                softmax_scale=self.scale,
                out=out_view,
                kv_cache_dtype=self.kv_cache_dtype,
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
            _routing._record_route(_routing.ROUTE_SPECS["decode_xqa_paged"].name)
            return output

        _routing._log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "scalar_paged")
        _routing._trace_decode_active(
            route="decode_scalar_paged",
            query=query,
            key_cache=key_cache,
            seq_lens=attn_metadata.seq_lens,
            attn_metadata=attn_metadata,
            window_size=window_size,
        )
        self._call_flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            softmax_scale=self.scale,
            out=out_view,
            kv_cache_dtype=self.kv_cache_dtype,
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
        _routing._record_route(_routing.ROUTE_SPECS["decode_scalar_paged"].name)
        return output

    def _flash_v100_ddtree_small_query_prefill_dense(
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
        global _logged_prefill_ddtree_dense
        global _logged_prefill_ddtree_triton
        global _logged_prefill_ddtree_triton_fallback

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
                    if (
                        is_capturing
                        or _debug._dflash_ddtree_triton_branch_attn_strict()
                    ):
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
                    if not _logged_prefill_ddtree_triton_fallback:
                        logger.exception(
                            "FLASH_ATTN_V100 DDTree Triton verifier failed; "
                            "falling back to dense masked verifier."
                        )
                        _logged_prefill_ddtree_triton_fallback = True
                else:
                    if not _logged_prefill_ddtree_triton:
                        logger.info(
                            "FLASH_ATTN_V100 DDTree branched verifier path active "
                            "(Triton paged-KV ancestor mask)."
                        )
                    _logged_prefill_ddtree_triton = True
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

        parent_ids_cpu = _masks._ddtree_parent_ids_cpu(attn_metadata)
        if parent_ids_cpu is None or num_tree_tokens_cpu is None:
            raise RuntimeError(
                "DDTree dense verifier fallback requires parent metadata"
            )

        if not _logged_prefill_ddtree_dense:
            logger.info(
                "FLASH_ATTN_V100 DDTree branched verifier path active "
                "(dense masked small-query fallback)."
            )
            _logged_prefill_ddtree_dense = True

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
                    _debug._format_tensor_debug(
                        persistent_query_start_loc, "smallq_qsl"
                    ),
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

        query_start_loc_norm = (
            _kv_layout._normalize_query_start_loc_for_available_tokens(
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

    def _should_use_fp8_prefill_bridge(
        self,
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
        self,
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
        self,
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
        self,
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
        self,
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
        self,
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
        self,
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
        self,
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
        global _logged_prefill_prefix_decode_rows_grouped
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
        if not _logged_prefill_prefix_decode_rows_grouped:
            logger.info(
                "FLASH_ATTN_V100 mixed-batch small-query rows take the grouped "
                "E4M3 FP32 route (requests=%d, groups=%d, max_q=%d, "
                "max_seq_len=%d).",
                len(plan.rows),
                plan.num_groups,
                plan.max_query_len,
                plan.max_seq_len_hint,
            )
            _logged_prefill_prefix_decode_rows_grouped = True
        _routing._log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "grouped_fp32")
        _routing._record_route(
            _routing.ROUTE_SPECS["prefill_prefix_decode_rows_e4m3_grouped_fp32"].name
        )
        return True

    def _run_prefill_prefix_decode_rows(
        self,
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
        global _logged_prefill_prefix_decode_rows
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

        if not _logged_prefill_prefix_decode_rows:
            logger.info(
                "FLASH_ATTN_V100 mixed-batch small-query rows take the paged "
                "decode route (%s, rows=%d of %d, max_q=%d, max_seq_len=%d).",
                route,
                len(plan.rows),
                len(query_start_loc) - 1,
                max_query_len_rows,
                max_seq_len_hint,
            )
            _logged_prefill_prefix_decode_rows = True
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
        self,
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
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path for prefix/chunked context via gathered contiguous KV."""
        global _logged_dflash_prefix_dump
        global _logged_prefill_prefix_bfla
        global _logged_prefill_prefix_contig_dense
        global _logged_prefill_prefix_splitkv
        global _logged_fp8_prefill_bridge
        global _logged_prefill_compare, _logged_prefill_smallq_decode
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
            and not _logged_dflash_prefix_dump
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
            if not _logged_prefill_smallq_decode:
                logger.info(
                    "FLASH_ATTN_V100 prefix prefill small-query path active "
                    "(paged decode verifier, max_query_len<=%d).",
                    self.smallq_decode_max_query_len,
                )
                _logged_prefill_smallq_decode = True
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
                    if not _logged_prefill_prefix_bfla:
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
                        _logged_prefill_prefix_bfla = True
                    _routing._record_route(
                        _routing.ROUTE_SPECS["prefill_prefix_bfla"].name
                    )
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
                    if not _logged_prefill_prefix_contig_dense:
                        logger.info(
                            "FLASH_ATTN_V100 prefix prefill contiguous dense "
                            "BHMD path active (min_q=%d min_kv=%d allow_copy=%s).",
                            self.prefill_contig_dense_min_q,
                            self.prefill_contig_dense_min_kv,
                            str(self.prefill_contig_dense_allow_copy),
                        )
                        _logged_prefill_prefix_contig_dense = True
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
                    if not _logged_prefill_prefix_contig_dense:
                        logger.info(
                            "FLASH_ATTN_V100 prefix prefill contiguous dense "
                            "path active (min_q=%d min_kv=%d).",
                            self.prefill_contig_dense_min_q,
                            self.prefill_contig_dense_min_kv,
                        )
                        _logged_prefill_prefix_contig_dense = True
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
                        if not _logged_fp8_prefill_bridge:
                            logger.info(
                                "FLASH_ATTN_V100 %s prefill bridge "
                                "active (one-pass dequant, shared FP16 page-%d "
                                "workspace).",
                                self.kv_cache_dtype,
                                _dense_prefill._FP8_PREFILL_BRIDGE_PAGE_SIZE,
                            )
                            _logged_fp8_prefill_bridge = True
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
                    if not _logged_prefill_prefix_splitkv:
                        logger.info(
                            "FLASH_ATTN_V100 prefix prefill split-KV path active "
                            "(split_kv_tokens=%d min_q=%d max_q=%d min_kv=%d).",
                            self.prefill_split_kv_tokens,
                            self.prefill_split_kv_min_q,
                            self.prefill_split_kv_max_q,
                            self.prefill_split_kv_min_kv,
                        )
                        _logged_prefill_prefix_splitkv = True
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
                    debug_compare and not _logged_prefill_compare
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
                    if debug_compare and not _logged_prefill_compare:
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
                            f"/tmp/flash_v100_dflash_prefix_dump_pid{os.getpid()}"
                            f"_seq{i}.pt"
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
                        _logged_dflash_prefix_dump = True
                    if debug_compare and not _logged_prefill_compare and nan_count > 0:
                        dump_path = (
                            f"/tmp/flash_v100_prefill_nan_dump_pid{os.getpid()}.pt"
                        )
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
                    if debug_compare and not _logged_prefill_compare:
                        _logged_prefill_compare = True
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
