# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Attention V100 backend for SM70.

Prefill uses the dense Flash V100 kernel for strict no-prefix cases.
Decode uses a paged Flash V100 kernel that reads vLLM's KV cache directly.
"""

from __future__ import annotations

import os
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

logger = init_logger(__name__)

# Lazy imports: only resolve optional CUDA extensions when needed.
_flash_attn_func = None
_flash_attn_decode_paged = None
_flash_attn_prefill_paged = None
_paged_kv_utils = None
_warned_prefill_fallback = False
_warned_feature_fallback = False
_warned_decode_fallback = False
_logged_prefill_flash = False
_logged_prefill_prefix_flash = False
_logged_prefill_smallq_decode = False
_logged_decode_flash = False
_logged_prefill_compare = False


def _get_flash_ops():
    """Lazy-load flash_attn_v100 ops if available."""
    global _flash_attn_func, _flash_attn_decode_paged, _flash_attn_prefill_paged
    if (_flash_attn_func is None or _flash_attn_decode_paged is None
            or _flash_attn_prefill_paged is None):
        try:
            from flash_attn_v100 import (flash_attn_decode_paged,
                                         flash_attn_func,
                                         flash_attn_prefill_paged)

            _flash_attn_func = flash_attn_func
            _flash_attn_decode_paged = flash_attn_decode_paged
            _flash_attn_prefill_paged = flash_attn_prefill_paged
        except ImportError:
            _flash_attn_func = None
            _flash_attn_decode_paged = None
            _flash_attn_prefill_paged = None
    return _flash_attn_func, _flash_attn_decode_paged, _flash_attn_prefill_paged


def _get_paged_kv_utils():
    """Lazy-load paged KV extraction CUDA extension."""
    global _paged_kv_utils
    if _paged_kv_utils is None:
        try:
            import paged_kv_utils

            _paged_kv_utils = paged_kv_utils
        except ImportError:
            _paged_kv_utils = None
    return _paged_kv_utils


def _has_prefix_context(attn_metadata: TritonAttentionMetadata) -> bool:
    """Return True if any sequence has KV context before current query tokens."""
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
    if query_start_loc_cpu is not None and seq_lens_cpu is not None:
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        return bool(torch.any(query_lens != seq_lens_cpu).item())

    query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
    return not torch.equal(query_lens, attn_metadata.seq_lens)


def _extract_contiguous_kv_from_paged_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    total_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract contiguous K/V from paged KV cache.

    Uses the CUDA extension when available and falls back to a Python path.
    """

    paged_kv_utils = _get_paged_kv_utils()

    if isinstance(kv_cache, (list, tuple)):
        key_cache, value_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        elif kv_cache.shape[1] == 2:
            key_cache, value_cache = kv_cache.unbind(1)
        else:
            raise ValueError(
                f"Unexpected KV cache shape {tuple(kv_cache.shape)}; "
                "expected dimension 2 at axis 0 or 1"
            )

    if paged_kv_utils is not None and key_cache.dtype != torch.uint8:
        if hasattr(paged_kv_utils, "paged_kv_to_contiguous"):
            k_cont, v_cont = paged_kv_utils.paged_kv_to_contiguous(
                key_cache, value_cache, block_table, seq_lens)
        else:
            k_cont = paged_kv_utils.paged_to_contiguous(key_cache, block_table,
                                                        seq_lens)
            v_cont = paged_kv_utils.paged_to_contiguous(value_cache, block_table,
                                                        seq_lens)
        if total_tokens is None:
            total_tokens = int(seq_lens.sum().item())
        return k_cont[:total_tokens], v_cont[:total_tokens]

    # Slow Python fallback.
    batch_size = block_table.shape[0]
    if total_tokens is None:
        total_tokens = int(seq_lens.sum().item())

    k_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=key_cache.dtype,
        device=key_cache.device,
    )
    v_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=value_cache.dtype,
        device=value_cache.device,
    )

    token_offset = 0
    for batch_idx in range(batch_size):
        seq_len = int(seq_lens[batch_idx].item())
        if seq_len == 0:
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            physical_block_idx = int(block_table[batch_idx, block_idx].item())
            start_token = block_idx * block_size
            end_token = min(start_token + block_size, seq_len)
            n = end_token - start_token

            k_cont[token_offset:token_offset + n] = key_cache[physical_block_idx, :n]
            v_cont[token_offset:token_offset + n] = value_cache[physical_block_idx, :n]
            token_offset += n

    return k_cont, v_cont


def _fp8_dtype_from_cache_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FLASH_ATTN_V100 fp8 dtype: {kv_cache_dtype}")


def _dequantize_fp8_contiguous_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not kv_cache_dtype.startswith("fp8"):
        return key, value
    fp8_dtype = _fp8_dtype_from_cache_dtype(kv_cache_dtype)
    key = key.view(fp8_dtype).to(torch.float16) * k_scale
    value = value.view(fp8_dtype).to(torch.float16) * v_scale
    return key, value


class FlashAttnV100MetadataBuilder(TritonAttentionMetadataBuilder):
    """Attach CPU metadata for the dense prefill path."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def build_for_cudagraph_capture(self, common_attn_metadata):
        attn_metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        attn_metadata.max_model_len = self.vllm_config.model_config.max_model_len

        # The Triton builder shortens capture seq_lens to 1 so full graph
        # capture stays cheap. That is valid for single-token decode, but the
        # FA2 small-query MTP verifier replays a tiny causal prefill as paged
        # decode. Capturing that branch with seq_len < query_len creates
        # negative per-token decode lengths and can poison long-context graph
        # replay. Keep capture cheap while preserving a valid verifier shape.
        max_query_len = getattr(attn_metadata, "max_query_len", 1)
        if max_query_len > 1:
            attn_metadata.seq_lens.fill_(max_query_len)

        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        attn_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        attn_metadata.causal = common_attn_metadata.causal
        attn_metadata.max_model_len = self.vllm_config.model_config.max_model_len
        return attn_metadata


class FlashAttnV100Impl(TritonAttentionImpl):
    """Flash Attention V100 implementation with strict fallback policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (self.flash_attn_func, self.flash_attn_decode_paged,
         self.flash_attn_prefill_paged) = _get_flash_ops()
        # V100 FA2 kernels consume fp16 Q. FP8 KV cache support is implemented
        # as storage compression only, with K/V dequantized inside FA2 kernels.
        self.supports_quant_query_input = False
        self.use_flash_v100 = self.flash_attn_func is not None
        self.use_flash_v100_decode = self.flash_attn_decode_paged is not None
        paged_prefill_enable = os.getenv("VLLM_FLASH_V100_ENABLE_PAGED_PREFILL")
        paged_prefill_disable = (
            os.getenv("VLLM_FLASH_V100_DISABLE_PAGED_PREFILL", "0") == "1")
        self.use_flash_v100_prefill_paged = (
            self.flash_attn_prefill_paged is not None
            and paged_prefill_enable != "0"
            and not paged_prefill_disable)
        self.smallq_decode_max_query_len = int(
            os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16"))
        self.smallq_decode_max_model_len = int(
            os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN",
                      "0"))
        self._warned_prefill_fallback = False
        self._warned_feature_fallback = False
        self._warned_decode_fallback = False
        self._logged_prefill_flash = False
        self._logged_prefill_prefix_flash = False
        self._logged_prefill_smallq_decode = False
        self._logged_decode_flash = False
        self._logged_prefill_compare = False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(query)
        if attn_metadata is None:
            return output

        if attn_metadata.max_query_len > 1:
            return self._forward_prefill(
                layer, query, key, value, kv_cache, attn_metadata, output)
        return self._forward_decode(
            layer, query, key, value, kv_cache, attn_metadata, output)

    def _forward_prefill(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if _has_prefix_context(attn_metadata):
            return self._flash_v100_prefill_with_prefix(
                layer, query, kv_cache, attn_metadata, output)
        return self._flash_v100_prefill_no_prefix(
            layer, query, key, value, attn_metadata, output)

    def _forward_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_flash_v100_decode:
            return self._flash_v100_decode(
                layer, query, kv_cache, attn_metadata, output)
        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata, output)

    def _flash_v100_prefill_no_prefix(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        global _logged_prefill_flash, _warned_prefill_fallback
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        out_view = output[:num_actual_tokens]
        causal = getattr(attn_metadata, "causal", True)

        if self.use_flash_v100:
            if not _logged_prefill_flash:
                logger.info("FLASH_ATTN_V100 dense prefill path active.")
                _logged_prefill_flash = True
            out = self.flash_attn_func(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                causal=causal,
                softmax_scale=self.scale,
            )
            out_view.copy_(out.squeeze(0))
            return output

        if not _warned_prefill_fallback:
            logger.warning(
                "FLASH_ATTN_V100 dense prefill unavailable; falling back to Triton.")
            _warned_prefill_fallback = True
        return super().forward(
            layer, query, key, value, None, attn_metadata, output)

    def _flash_v100_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        global _logged_decode_flash, _warned_decode_fallback
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)

        if not _logged_decode_flash:
            logger.info("FLASH_ATTN_V100 paged decode path active.")
            _logged_decode_flash = True

        decode_seq_lens = attn_metadata.seq_lens
        decode_block_table = attn_metadata.block_table
        padding_mask = decode_seq_lens == 0
        num_query_tokens = query.shape[0]
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
        self.flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            decode_block_table,
            decode_seq_lens,
            softmax_scale=self.scale,
            out=out_view,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
        )
        return output

    def _flash_v100_prefill_with_prefix(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path for prefix/chunked context via gathered contiguous KV."""
        global _logged_prefill_compare, _logged_prefill_smallq_decode
        causal = getattr(attn_metadata, "causal", True)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens = seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        num_seqs = len(query_start_loc) - 1

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]
        debug_compare = (os.getenv("VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE", "0")
                         == "1")

        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        max_query_len = int(query_lens.max().item()) if num_seqs > 0 else 0
        if (causal and self.use_flash_v100_decode
                and self.smallq_decode_max_query_len > 0
                and max_query_len <= self.smallq_decode_max_query_len
                and (self.smallq_decode_max_model_len <= 0
                     or getattr(attn_metadata, "max_model_len", 0)
                     <= self.smallq_decode_max_model_len)):
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

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue

            if self.use_flash_v100_prefill_paged:
                out_seq = self.flash_attn_prefill_paged(
                    query[start:end].unsqueeze(0),
                    key_cache,
                    value_cache,
                    attn_metadata.block_table[i:i + 1],
                    attn_metadata.seq_lens[i:i + 1],
                    softmax_scale=self.scale,
                    kv_cache_dtype=self.kv_cache_dtype,
                    k_scale=float(layer._k_scale_float),
                    v_scale=float(layer._v_scale_float),
                    causal=causal,
                )
                if debug_compare and not _logged_prefill_compare:
                    seq_len = int(seq_lens[i].item())
                    k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                        kv_cache=kv_cache,
                        block_table=attn_metadata.block_table[i:i + 1],
                        seq_lens=attn_metadata.seq_lens[i:i + 1],
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        total_tokens=seq_len,
                    )
                    k_cont, v_cont = _dequantize_fp8_contiguous_kv(
                        k_cont,
                        v_cont,
                        self.kv_cache_dtype,
                        float(layer._k_scale_float),
                        float(layer._v_scale_float),
                    )
                    ref_out = self.flash_attn_func(
                        query[start:end].unsqueeze(0),
                        k_cont.unsqueeze(0),
                        v_cont.unsqueeze(0),
                        causal=causal,
                        softmax_scale=self.scale,
                    )
                    diff = (out_seq - ref_out).abs()
                    nan_count = int(torch.isnan(out_seq).sum().item())
                    logger.warning(
                        "FLASH_ATTN_V100 debug prefix compare: "
                        "query_len=%d seq_len=%d max_diff=%.8f mean_diff=%.8f "
                        "nan_count=%d q_absmax=%.6f k_absmax=%.6f v_absmax=%.6f "
                        "kv_cache_shape=%s key_shape=%s key_stride=%s "
                        "value_stride=%s key_contig=%s value_contig=%s",
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
                    if nan_count > 0:
                        dump_path = (
                            f"/tmp/flash_v100_prefill_nan_dump_pid{os.getpid()}.pt"
                        )
                        torch.save(
                            {
                                "query": query[start:end].detach().cpu(),
                                "key_cache": key_cache.detach().cpu(),
                                "value_cache": value_cache.detach().cpu(),
                                "block_table": attn_metadata.block_table[
                                    i:i + 1].detach().cpu(),
                                "seq_lens": attn_metadata.seq_lens[
                                    i:i + 1].detach().cpu(),
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
                    _logged_prefill_compare = True
            else:
                seq_len = int(seq_lens[i].item())
                k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table[i:i + 1],
                    seq_lens=attn_metadata.seq_lens[i:i + 1],
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    total_tokens=seq_len,
                )
                k_cont, v_cont = _dequantize_fp8_contiguous_kv(
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
                )
            out_view[start:end].copy_(out_seq.squeeze(0))

        return output


class FlashAttnV100Backend(TritonAttentionBackend):
    """Flash Attention V100 Backend."""

    # Keep vLLM unified KV cache update path.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_impl_cls():
        return FlashAttnV100Impl

    @staticmethod
    def get_builder_cls():
        return FlashAttnV100MetadataBuilder

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_V100"

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # Keep this aligned with the dense prefill kernel dispatch table.
        return [64, 128, 256]
