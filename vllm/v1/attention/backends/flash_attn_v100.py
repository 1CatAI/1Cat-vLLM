# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Attention V100 backend for SM70.

Selecting this backend keeps both prefill and decode on Flash-V100 by default.
The upstream Triton prefill path is still available as an explicit diagnostic
fallback with VLLM_FLASH_V100_PREFILL_USE_TRITON=1, but mixed Triton-prefill
plus Flash-decode runs do not count as the final SM70 FlashAttention route.
"""

from __future__ import annotations

import atexit
import json
import os
import time

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

logger = init_logger(__name__)


def _sm70_profile_trace(message: str, *args: object) -> None:
    if envs.VLLM_SM70_PROFILE_TRACE:
        if args:
            message = message % args
        logger.info("SM70 Flash-V100 trace: %s", message)


# Lazy imports: only resolve optional CUDA extensions when needed.
_flash_attn_func = None
_flash_attn_decode_paged = None
_flash_attn_decode_paged_wmma = None
_flash_attn_prefill_paged = None
_flash_attn_prefill_paged_bhmd = None
_flash_attn_turboquant_decode_paged = None
_flash_attn_turboquant_decode_checked = False
_paged_kv_utils = None
_warned_feature_fallback = False
_warned_decode_fallback = False
_warned_decode_strict_fallback = False
_logged_prefill_flash = False
_logged_prefill_prefix_flash = False
_logged_prefill_paged_cache = False
_logged_prefill_smallq_decode = False
_logged_prefill_triton_safe = False
_logged_decode_flash = False
_logged_decode_dense_reference = False
_logged_decode_dense_cache = False
_logged_decode_paged_prefill = False
_logged_decode_paged_prefill_bhmd = False
_logged_decode_paged_prefill_bhmd_q_clone = False
_logged_decode_wmma_wrapper = False
_logged_fp8_kv_prefill = False
_logged_fp8_kv_decode = False
_logged_prefill_compare = False
_logged_dflash_prefix_dump = False
_route_summary_registered = False
_route_counts: dict[str, int] = {}
_draft_graph_debug_counts: dict[str, int] = {}


def _draft_graph_debug_enabled() -> bool:
    return os.getenv("VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG", "0") == "1"


def _draft_graph_debug_limit() -> int:
    return int(os.getenv("VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG_LIMIT", "12"))


def _dflash_prefix_dump_enabled() -> bool:
    return os.getenv("VLLM_FLASH_V100_DFLASH_PREFIX_DUMP", "0") == "1"


def _format_tensor_debug(tensor: torch.Tensor | None, name: str) -> str:
    if tensor is None:
        return f"{name}=None"

    values = ""
    if tensor.numel() > 0 and not (
        tensor.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        try:
            flat = tensor.detach().reshape(-1)[: min(8, tensor.numel())]
            values = f" vals={flat.cpu().tolist()}"
        except Exception as exc:  # pragma: no cover - diagnostic only.
            values = f" vals=<unavailable:{type(exc).__name__}>"

    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"ptr=0x{tensor.data_ptr():x} storage=0x"
        f"{tensor.untyped_storage().data_ptr():x} "
        f"offset={tensor.storage_offset()}{values}"
    )


def _draft_graph_debug_log(key: str, message: str, *args: object) -> None:
    if not _draft_graph_debug_enabled():
        return
    count = _draft_graph_debug_counts.get(key, 0)
    if count >= _draft_graph_debug_limit():
        return
    _draft_graph_debug_counts[key] = count + 1
    if args:
        message = message % args
    logger.info(
        "FLASH_ATTN_V100 draft graph debug[%s#%d]: %s",
        key,
        count,
        message,
    )


def _graph_metadata_debug_log(key: str, message: str, *args: object) -> None:
    if not _draft_graph_debug_enabled():
        return
    count = _draft_graph_debug_counts.get(key, 0)
    if count >= _draft_graph_debug_limit():
        return
    _draft_graph_debug_counts[key] = count + 1
    if args:
        message = message % args
    logger.info(
        "FLASH_ATTN_V100 graph metadata debug[%s#%d]: %s",
        key,
        count,
        message,
    )


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())


def _route_summary_enabled() -> bool:
    return os.getenv("VLLM_FLASH_V100_ROUTE_SUMMARY", "0") == "1"


def _log_route_summary() -> None:
    if _route_counts:
        logger.info(
            "FLASH_ATTN_V100 route summary: %s",
            json.dumps(_route_counts, sort_keys=True),
        )


def _record_route(route: str) -> None:
    global _route_summary_registered
    if not _route_summary_enabled():
        return
    _route_counts[route] = _route_counts.get(route, 0) + 1
    if not _route_summary_registered:
        atexit.register(_log_route_summary)
        _route_summary_registered = True


def _uses_fp8_kv_cache(kv_cache_dtype: str) -> bool:
    return isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8")


def _log_fp8_kv_cache_route(stage: str, kv_cache_dtype: str, route: str) -> None:
    global _logged_fp8_kv_decode, _logged_fp8_kv_prefill

    if not _uses_fp8_kv_cache(kv_cache_dtype):
        return
    if stage == "prefill":
        if _logged_fp8_kv_prefill:
            return
        logger.info(
            "FLASH_ATTN_V100 FP8 KV cache prefill path active "
            "(kv_cache_dtype=%s, route=%s).",
            kv_cache_dtype,
            route,
        )
        _logged_fp8_kv_prefill = True
        _record_route("fp8_kv_prefill")
        return
    if stage == "decode":
        if _logged_fp8_kv_decode:
            return
        logger.info(
            "FLASH_ATTN_V100 FP8 KV cache decode path active "
            "(kv_cache_dtype=%s, route=%s).",
            kv_cache_dtype,
            route,
        )
        _logged_fp8_kv_decode = True
        _record_route("fp8_kv_decode")
        return
    raise ValueError(f"Unsupported FP8 KV cache route stage: {stage}")


def _get_flash_ops():
    """Lazy-load flash_attn_v100 ops if available."""
    global _flash_attn_func, _flash_attn_decode_paged, _flash_attn_prefill_paged
    global _flash_attn_decode_paged_wmma, _flash_attn_prefill_paged_bhmd
    if (
        _flash_attn_func is None
        or _flash_attn_decode_paged is None
        or _flash_attn_prefill_paged is None
    ):
        try:
            from flash_attn_v100 import (
                flash_attn_decode_paged,
                flash_attn_func,
                flash_attn_prefill_paged,
            )

            _flash_attn_func = flash_attn_func
            _flash_attn_decode_paged = flash_attn_decode_paged
            _flash_attn_prefill_paged = flash_attn_prefill_paged
            try:
                from flash_attn_v100 import flash_attn_decode_paged_wmma

                _flash_attn_decode_paged_wmma = flash_attn_decode_paged_wmma
            except ImportError:
                _flash_attn_decode_paged_wmma = None
            try:
                from flash_attn_v100 import flash_attn_prefill_paged_bhmd

                _flash_attn_prefill_paged_bhmd = flash_attn_prefill_paged_bhmd
            except ImportError:
                _flash_attn_prefill_paged_bhmd = None
        except ImportError:
            _flash_attn_func = None
            _flash_attn_decode_paged = None
            _flash_attn_decode_paged_wmma = None
            _flash_attn_prefill_paged = None
            _flash_attn_prefill_paged_bhmd = None
    return (
        _flash_attn_func,
        _flash_attn_decode_paged,
        _flash_attn_decode_paged_wmma,
        _flash_attn_prefill_paged,
        _flash_attn_prefill_paged_bhmd,
    )


def flash_v100_dense_prefill_available() -> bool:
    flash_attn_func, _, _, _, _ = _get_flash_ops()
    return flash_attn_func is not None


def flash_v100_dense_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_actual_tokens: int,
    softmax_scale: float,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """Run Flash-V100 dense raw-QKV prefill without backend metadata coupling."""
    flash_attn_func, _, _, _, _ = _get_flash_ops()
    if flash_attn_func is None:
        raise RuntimeError("flash_attn_v100 dense prefill op is unavailable")

    query = query[:num_actual_tokens]
    key = key[:num_actual_tokens]
    value = value[:num_actual_tokens]
    out_view = output[:num_actual_tokens]

    num_seqs = len(query_start_loc) - 1
    if num_seqs == 0:
        return output

    seq_lens = query_start_loc[1:] - query_start_loc[:-1]
    run_start = 0
    while run_start < num_seqs:
        run_seq_len = int(seq_lens[run_start].item())
        run_end = run_start + 1
        while run_end < num_seqs and int(seq_lens[run_end].item()) == run_seq_len:
            run_end += 1

        if run_seq_len > 0:
            tok_start = int(query_start_loc[run_start].item())
            tok_end = int(query_start_loc[run_end].item())
            batch_size = run_end - run_start

            q_batch = query[tok_start:tok_end].view(
                batch_size, run_seq_len, query.shape[1], query.shape[2]
            )
            k_batch = key[tok_start:tok_end].view(
                batch_size, run_seq_len, key.shape[1], key.shape[2]
            )
            v_batch = value[tok_start:tok_end].view(
                batch_size, run_seq_len, value.shape[1], value.shape[2]
            )

            out_batch = flash_attn_func(
                q_batch,
                k_batch,
                v_batch,
                causal=causal,
                softmax_scale=softmax_scale,
                window_size=window_size,
            )
            out_view[tok_start:tok_end].copy_(
                out_batch.view(tok_end - tok_start, out_batch.shape[2], out_batch.shape[3])
            )

        run_start = run_end

    return output


def _get_flash_turboquant_decode_op():
    """Lazy-load the optional TurboQuant decode op without touching base ops."""
    global _flash_attn_turboquant_decode_checked
    global _flash_attn_turboquant_decode_paged
    if not _flash_attn_turboquant_decode_checked:
        try:
            from flash_attn_v100 import (
                flash_attn_turboquant_decode_paged,
                flash_attn_turboquant_decode_paged_available,
            )

            if flash_attn_turboquant_decode_paged_available():
                _flash_attn_turboquant_decode_paged = flash_attn_turboquant_decode_paged
            else:
                _flash_attn_turboquant_decode_paged = None
        except ImportError:
            _flash_attn_turboquant_decode_paged = None
        _flash_attn_turboquant_decode_checked = True
    return _flash_attn_turboquant_decode_paged


def flash_v100_turboquant_decode_available() -> bool:
    return _get_flash_turboquant_decode_op() is not None


def flash_v100_turboquant_decode(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    softmax_scale: float,
    mse_bits: int,
    value_quant_bits: int,
    norm_correction: bool,
    num_kv_splits: int,
) -> torch.Tensor:
    """Run Flash-V100 decode directly over TurboQuant packed paged cache."""
    op = _get_flash_turboquant_decode_op()
    if op is None:
        raise RuntimeError("flash_attn_v100 TurboQuant decode op is unavailable")
    return op(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        softmax_scale=softmax_scale,
        out=output,
        mse_bits=mse_bits,
        value_quant_bits=value_quant_bits,
        norm_correction=norm_correction,
        num_kv_splits=num_kv_splits,
    )


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


def _metadata_expects_more_query_tokens_than_available(
    attn_metadata: TritonAttentionMetadata,
    available_query_tokens: int,
) -> bool:
    """Return True when per-layer Q/K/V tensors are shorter than query metadata.

    Hybrid Qwen3.5/3.6 routes can feed a full-attention layer only the live
    query-token subset while the batch-level metadata still describes the
    wider request span. That shape is not a dense raw-QKV prefill; it must use
    the prefix/live-token compatible path.
    """
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    query_start_loc = (
        query_start_loc_cpu
        if query_start_loc_cpu is not None
        else attn_metadata.query_start_loc
    )
    if len(query_start_loc) <= 1:
        return False
    expected_query_tokens = int(query_start_loc[-1].item())
    return available_query_tokens < expected_query_tokens


def _normalize_query_start_loc_for_available_tokens(
    query_start_loc: torch.Tensor,
    available_query_tokens: int,
) -> torch.Tensor:
    """Project metadata query spans onto the tokens actually present in Q/K/V.

    This is only needed when a hybrid/model-specific path feeds a full-attention
    layer a live-token subset instead of the full batch span described by the
    shared metadata.
    """
    num_seqs = len(query_start_loc) - 1
    if num_seqs <= 0:
        return query_start_loc

    expected_query_tokens = int(query_start_loc[-1].item())
    if available_query_tokens >= expected_query_tokens:
        return query_start_loc

    if available_query_tokens <= 0:
        return query_start_loc.new_zeros(query_start_loc.shape)

    if num_seqs == 1:
        return query_start_loc.new_tensor([0, available_query_tokens])

    if available_query_tokens == num_seqs:
        return torch.arange(
            num_seqs + 1,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )

    if available_query_tokens % num_seqs == 0:
        q_per_seq = available_query_tokens // num_seqs
        orig_query_lens = query_start_loc[1:] - query_start_loc[:-1]
        if int(orig_query_lens.min().item()) >= q_per_seq:
            return torch.arange(
                0,
                available_query_tokens + 1,
                q_per_seq,
                dtype=query_start_loc.dtype,
                device=query_start_loc.device,
            )

    raise RuntimeError(
        "FLASH_ATTN_V100 received fewer layer query tokens than query metadata "
        "describes, and the per-sequence live-token layout could not be "
        "reconstructed safely."
    )


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
                key_cache, value_cache, block_table, seq_lens
            )
        else:
            k_cont = paged_kv_utils.paged_to_contiguous(
                key_cache, block_table, seq_lens
            )
            v_cont = paged_kv_utils.paged_to_contiguous(
                value_cache, block_table, seq_lens
            )
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

            k_cont[token_offset : token_offset + n] = key_cache[physical_block_idx, :n]
            v_cont[token_offset : token_offset + n] = value_cache[
                physical_block_idx, :n
            ]
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
    if not _uses_fp8_kv_cache(kv_cache_dtype):
        return key, value
    fp8_dtype = _fp8_dtype_from_cache_dtype(kv_cache_dtype)
    key = key.view(fp8_dtype).to(torch.float16) * k_scale
    value = value.view(fp8_dtype).to(torch.float16) * v_scale
    return key, value


def _torch_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    window_size: tuple[int, int],
    softmax_scale: float,
) -> torch.Tensor:
    """Small debug-only fp32 attention reference for prefix/paged checks."""
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()

    num_q_heads = query_f.shape[1]
    num_kv_heads = key_f.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "num attention heads must be divisible by num KV heads for debug "
            f"reference, got {num_q_heads=} {num_kv_heads=}"
        )
    if num_q_heads != num_kv_heads:
        repeat = num_q_heads // num_kv_heads
        key_f = key_f.repeat_interleave(repeat, dim=1)
        value_f = value_f.repeat_interleave(repeat, dim=1)

    # [H, M, N]
    scores = torch.einsum("mhd,nhd->hmn", query_f, key_f) * softmax_scale
    q_len = query_f.shape[0]
    k_len = key_f.shape[0]
    q_pos = torch.arange(q_len, device=query.device) + max(k_len - q_len, 0)
    k_pos = torch.arange(k_len, device=query.device)
    valid = torch.ones((q_len, k_len), device=query.device, dtype=torch.bool)
    if causal:
        valid &= k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    window_left, window_right = window_size
    if window_left >= 0:
        valid &= k_pos.unsqueeze(0) >= q_pos.unsqueeze(1) - window_left
    if window_right >= 0:
        valid &= k_pos.unsqueeze(0) <= q_pos.unsqueeze(1) + window_right
    scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hmn,nhd->mhd", probs, value_f)
    return out.to(dtype=query.dtype).unsqueeze(0)


class FlashAttnV100MetadataBuilder(TritonAttentionMetadataBuilder):
    """Attach CPU metadata for the dense prefill path."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        self._is_speculative_draft_model = (
            spec_config is not None
            and getattr(spec_config, "draft_model_config", None)
            is self.vllm_config.model_config
        )
        self._draft_block_table: torch.Tensor | None = None
        self._draft_seq_lens: torch.Tensor | None = None
        self._draft_query_start_loc: torch.Tensor | None = None
        self._draft_buffer_shape: tuple[int, int] | None = None
        self._smallq_decode_block_table: torch.Tensor | None = None
        self._smallq_decode_seq_lens: torch.Tensor | None = None
        self._smallq_query_start_loc: torch.Tensor | None = None
        self._smallq_token_indices: torch.Tensor | None = None
        self._smallq_buffer_shape: tuple[int, int, int] | None = None

    def _attach_common_flash_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        attn_metadata.causal = common_attn_metadata.causal
        attn_metadata.max_model_len = self.vllm_config.model_config.max_model_len

    def _debug_draft_metadata(
        self,
        stage: str,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        if (
            not self._is_speculative_draft_model
            or not _draft_graph_debug_enabled()
        ):
            return
        _draft_graph_debug_log(
            f"builder:{stage}",
            "num_reqs=%s num_actual_tokens=%s max_query_len=%s max_seq_len=%s "
            "common_qsl_cpu=%s common_seq_cpu=%s %s %s %s %s %s %s %s",
            getattr(common_attn_metadata, "num_reqs", None),
            getattr(common_attn_metadata, "num_actual_tokens", None),
            getattr(common_attn_metadata, "max_query_len", None),
            getattr(common_attn_metadata, "max_seq_len", None),
            getattr(common_attn_metadata, "query_start_loc_cpu", None),
            getattr(common_attn_metadata, "seq_lens_cpu", None),
            _format_tensor_debug(
                getattr(common_attn_metadata, "query_start_loc", None),
                "common_qsl",
            ),
            _format_tensor_debug(
                getattr(common_attn_metadata, "seq_lens", None),
                "common_seq",
            ),
            _format_tensor_debug(
                getattr(common_attn_metadata, "block_table_tensor", None),
                "common_bt",
            ),
            _format_tensor_debug(
                getattr(attn_metadata, "query_start_loc", None),
                "attn_qsl",
            ),
            _format_tensor_debug(getattr(attn_metadata, "seq_lens", None), "attn_seq"),
            _format_tensor_debug(
                getattr(attn_metadata, "block_table", None),
                "attn_bt",
            ),
            _format_tensor_debug(
                getattr(attn_metadata, "smallq_decode_seq_lens", None),
                "smallq_seq",
            ),
        )

    def _ensure_draft_graph_buffers(
        self,
        required_reqs: int,
        block_table: torch.Tensor,
    ) -> bool:
        req_capacity = max(
            int(self.vllm_config.scheduler_config.max_num_seqs),
            int(required_reqs),
            1,
        )
        block_cols = int(block_table.shape[1])
        shape = (req_capacity, block_cols)
        if self._draft_buffer_shape == shape:
            return True

        if self._draft_buffer_shape is not None:
            old_reqs, old_block_cols = self._draft_buffer_shape
            if required_reqs <= old_reqs and block_cols == old_block_cols:
                return True
            return False

        self._draft_block_table = torch.empty(
            (req_capacity, block_cols),
            dtype=torch.int32,
            device=self.device,
        )
        self._draft_seq_lens = torch.empty(
            (req_capacity,),
            dtype=torch.int32,
            device=self.device,
        )
        self._draft_query_start_loc = torch.empty(
            (req_capacity + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self._draft_buffer_shape = shape
        return True

    def _stabilize_draft_graph_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        num_reqs = int(common_attn_metadata.num_reqs)
        if num_reqs <= 0:
            return

        block_table = attn_metadata.block_table[:num_reqs]
        if not self._ensure_draft_graph_buffers(num_reqs, block_table):
            assert self._draft_buffer_shape is not None
            req_capacity, block_cols = self._draft_buffer_shape
            raise RuntimeError(
                "FLASH_ATTN_V100 draft CUDA graph metadata shape exceeds "
                "the captured persistent buffer capacity: "
                f"required_reqs={num_reqs}, "
                f"required_block_cols={int(block_table.shape[1])}, "
                f"capacity_reqs={req_capacity}, "
                f"capacity_block_cols={block_cols}. "
                "Replay would otherwise read stale draft metadata."
            )

        assert self._draft_block_table is not None
        assert self._draft_seq_lens is not None
        assert self._draft_query_start_loc is not None

        self._draft_block_table[:num_reqs].copy_(block_table, non_blocking=True)
        self._draft_seq_lens[:num_reqs].copy_(
            attn_metadata.seq_lens[:num_reqs],
            non_blocking=True,
        )
        self._draft_query_start_loc[: num_reqs + 1].copy_(
            attn_metadata.query_start_loc[: num_reqs + 1],
            non_blocking=True,
        )

        attn_metadata.block_table = self._draft_block_table[:num_reqs]
        attn_metadata.seq_lens = self._draft_seq_lens[:num_reqs]
        attn_metadata.query_start_loc = self._draft_query_start_loc[
            : num_reqs + 1
        ]

    def _configured_smallq_max_query_len(self) -> int:
        return int(os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16"))

    def _configured_smallq_max_model_len(self) -> int:
        return int(os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN", "0"))

    def _smallq_buffer_token_capacity(self, required_tokens: int) -> int:
        compilation_config = self.vllm_config.compilation_config
        graph_tokens = compilation_config.max_cudagraph_capture_size
        if graph_tokens is None and compilation_config.cudagraph_capture_sizes:
            graph_tokens = max(compilation_config.cudagraph_capture_sizes)
        if graph_tokens is None or graph_tokens <= 0:
            graph_tokens = required_tokens
        return max(int(graph_tokens), int(required_tokens), 1)

    def _ensure_smallq_decode_buffers(
        self,
        required_tokens: int,
        required_reqs: int,
        block_table: torch.Tensor,
    ) -> bool:
        token_capacity = self._smallq_buffer_token_capacity(required_tokens)
        req_capacity = max(
            min(
                int(self.vllm_config.scheduler_config.max_num_seqs),
                token_capacity,
            ),
            int(required_reqs),
            1,
        )
        block_cols = int(block_table.shape[1])
        shape = (token_capacity, req_capacity, block_cols)
        if self._smallq_buffer_shape == shape:
            return True

        if self._smallq_buffer_shape is not None:
            old_tokens, old_reqs, old_block_cols = self._smallq_buffer_shape
            if (
                required_tokens <= old_tokens
                and required_reqs <= old_reqs
                and block_cols == old_block_cols
            ):
                return True
            return False

        self._smallq_decode_block_table = torch.empty(
            (token_capacity, block_cols),
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_decode_seq_lens = torch.empty(
            (token_capacity,),
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_query_start_loc = torch.empty(
            (req_capacity + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_token_indices = torch.arange(
            token_capacity,
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_buffer_shape = shape
        return True

    def _clear_smallq_decode_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
    ) -> None:
        attn_metadata.smallq_decode_block_table = None
        attn_metadata.smallq_decode_seq_lens = None
        attn_metadata.smallq_query_start_loc = None

    def _update_smallq_decode_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
        *,
        force: bool = False,
    ) -> None:
        self._clear_smallq_decode_metadata(attn_metadata)

        max_query_len = int(getattr(attn_metadata, "max_query_len", 1))
        smallq_max_query_len = self._configured_smallq_max_query_len()
        if (
            smallq_max_query_len <= 0
            or max_query_len <= 1
            or max_query_len > smallq_max_query_len
        ):
            return

        smallq_max_model_len = self._configured_smallq_max_model_len()
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        if smallq_max_model_len > 0 and max_model_len > smallq_max_model_len:
            return

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        has_prefix_context = bool(torch.any(query_lens_cpu != seq_lens_cpu).item())
        if (
            not force
            and not has_prefix_context
            and self._smallq_buffer_shape is None
        ):
            return

        num_query_tokens = int(attn_metadata.num_actual_tokens)
        num_reqs = int(common_attn_metadata.num_reqs)
        if num_query_tokens <= 0 or num_reqs <= 0:
            return

        block_table = attn_metadata.block_table[:num_reqs]
        if not self._ensure_smallq_decode_buffers(
            num_query_tokens,
            num_reqs,
            block_table,
        ):
            assert self._smallq_buffer_shape is not None
            token_capacity, req_capacity, block_cols = self._smallq_buffer_shape
            raise RuntimeError(
                "FLASH_ATTN_V100 small-query CUDA graph metadata shape exceeds "
                "the captured persistent buffer capacity: "
                f"required_tokens={num_query_tokens}, "
                f"required_reqs={num_reqs}, "
                f"required_block_cols={int(block_table.shape[1])}, "
                f"capacity_tokens={token_capacity}, "
                f"capacity_reqs={req_capacity}, "
                f"capacity_block_cols={block_cols}. "
                "Replay would otherwise use stale captured metadata."
            )
        assert self._smallq_decode_block_table is not None
        assert self._smallq_decode_seq_lens is not None
        assert self._smallq_query_start_loc is not None
        assert self._smallq_token_indices is not None

        query_start_loc = attn_metadata.query_start_loc[: num_reqs + 1]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        real_query_lens = query_lens
        real_num_query_tokens = int(query_start_loc_cpu[-1].item())
        if real_num_query_tokens > num_query_tokens:
            return

        repeat_query_lens = query_lens
        padding_tokens = num_query_tokens - real_num_query_tokens
        if padding_tokens > 0:
            repeat_query_lens = query_lens.clone()
            repeat_query_lens[-1] += padding_tokens

        seq_lens = attn_metadata.seq_lens[:num_reqs]
        effective_seq_lens = torch.maximum(
            seq_lens,
            real_query_lens.to(dtype=seq_lens.dtype),
        )
        block_table = block_table.clamp_min(0)
        decode_block_table = torch.repeat_interleave(
            block_table,
            repeat_query_lens,
            dim=0,
            output_size=num_query_tokens,
        ).contiguous()
        seq_lens_rep = torch.repeat_interleave(
            effective_seq_lens,
            repeat_query_lens,
            output_size=num_query_tokens,
        )
        query_lens_rep = torch.repeat_interleave(
            real_query_lens.to(dtype=seq_lens.dtype),
            repeat_query_lens,
            output_size=num_query_tokens,
        )
        start_locs_rep = torch.repeat_interleave(
            query_start_loc[:-1].to(dtype=seq_lens.dtype),
            repeat_query_lens,
            output_size=num_query_tokens,
        )
        token_indices = self._smallq_token_indices[:num_query_tokens].to(
            dtype=seq_lens.dtype
        )
        offsets = token_indices - start_locs_rep + 1
        decode_seq_lens = (seq_lens_rep - query_lens_rep + offsets).contiguous()
        if padding_tokens > 0:
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

        self._smallq_decode_block_table[:num_query_tokens].copy_(
            decode_block_table,
            non_blocking=True,
        )
        self._smallq_decode_seq_lens[:num_query_tokens].copy_(
            decode_seq_lens,
            non_blocking=True,
        )
        self._smallq_query_start_loc[: num_reqs + 1].copy_(
            query_start_loc,
            non_blocking=True,
        )

        attn_metadata.smallq_decode_block_table = (
            self._smallq_decode_block_table[:num_query_tokens]
        )
        attn_metadata.smallq_decode_seq_lens = (
            self._smallq_decode_seq_lens[:num_query_tokens]
        )
        attn_metadata.smallq_query_start_loc = (
            self._smallq_query_start_loc[: num_reqs + 1]
        )
        if _draft_graph_debug_enabled():
            _graph_metadata_debug_log(
                "smallq_update",
                "draft=%s force=%s num_reqs=%s num_query_tokens=%s "
                "real_num_query_tokens=%s padding_tokens=%s max_query_len=%s "
                "common_qsl_cpu=%s common_seq_cpu=%s %s %s %s %s %s %s",
                self._is_speculative_draft_model,
                force,
                num_reqs,
                num_query_tokens,
                real_num_query_tokens,
                padding_tokens,
                max_query_len,
                query_start_loc_cpu,
                seq_lens_cpu,
                _format_tensor_debug(attn_metadata.query_start_loc, "attn_qsl"),
                _format_tensor_debug(attn_metadata.seq_lens, "attn_seq"),
                _format_tensor_debug(attn_metadata.block_table, "attn_bt"),
                _format_tensor_debug(
                    attn_metadata.smallq_decode_block_table,
                    "smallq_bt",
                ),
                _format_tensor_debug(
                    attn_metadata.smallq_decode_seq_lens,
                    "smallq_seq",
                ),
                _format_tensor_debug(
                    attn_metadata.smallq_query_start_loc,
                    "smallq_qsl",
                ),
            )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        attn_metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        self._attach_common_flash_metadata(attn_metadata, common_attn_metadata)

        # The Triton builder shortens capture seq_lens to 1 so full graph
        # capture stays cheap. That is valid for single-token decode, but the
        # FA2 small-query MTP verifier replays a tiny causal prefill as paged
        # decode. Capturing that branch with seq_len < query_len creates
        # negative per-token decode lengths and can poison long-context graph
        # replay. Keep capture cheap while preserving a valid verifier shape.
        max_query_len = getattr(attn_metadata, "max_query_len", 1)
        if max_query_len > 1:
            attn_metadata.seq_lens.fill_(max_query_len)
            self._update_smallq_decode_metadata(
                attn_metadata,
                common_attn_metadata,
                force=True,
            )
        else:
            # PIECEWISE graph replay captures the q=1 decode kernel arguments
            # during metadata warmup. Runtime drafting updates the persistent
            # draft metadata buffers in build_for_drafting(), so capture must
            # bind the graph to the same buffers instead of transient dummy
            # capture tensors.
            self._stabilize_draft_graph_metadata(
                attn_metadata,
                common_attn_metadata,
            )
        self._debug_draft_metadata(
            "capture",
            attn_metadata,
            common_attn_metadata,
        )

        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        attn_metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        self._attach_common_flash_metadata(attn_metadata, common_attn_metadata)
        self._update_smallq_decode_metadata(attn_metadata, common_attn_metadata)
        self._debug_draft_metadata("build", attn_metadata, common_attn_metadata)
        return attn_metadata

    def build_for_drafting(self, common_attn_metadata, draft_index: int):
        attn_metadata = super().build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )
        self._attach_common_flash_metadata(attn_metadata, common_attn_metadata)
        self._stabilize_draft_graph_metadata(attn_metadata, common_attn_metadata)
        self._update_smallq_decode_metadata(attn_metadata, common_attn_metadata)
        self._debug_draft_metadata(
            f"draft{draft_index}",
            attn_metadata,
            common_attn_metadata,
        )
        return attn_metadata


class FlashAttnV100Impl(TritonAttentionImpl):
    """Flash Attention V100 implementation with explicit fallback policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (
            self.flash_attn_func,
            self.flash_attn_decode_paged,
            self.flash_attn_decode_paged_wmma,
            self.flash_attn_prefill_paged,
            self.flash_attn_prefill_paged_bhmd,
        ) = _get_flash_ops()
        # V100 FA2 kernels consume fp16 Q. FP8 KV cache support is implemented
        # as storage compression only, with K/V dequantized inside FA2 kernels.
        self.supports_quant_query_input = False
        self.use_flash_v100 = self.flash_attn_func is not None
        self.use_flash_v100_decode = self.flash_attn_decode_paged is not None
        paged_prefill_enable = os.getenv("VLLM_FLASH_V100_ENABLE_PAGED_PREFILL")
        paged_prefill_disable = (
            os.getenv("VLLM_FLASH_V100_DISABLE_PAGED_PREFILL", "0") == "1"
        )
        self.use_flash_v100_prefill_paged = (
            self.flash_attn_prefill_paged is not None
            and paged_prefill_enable != "0"
            and not paged_prefill_disable
        )
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
        decode_paged_prefill_env = os.getenv(
            "VLLM_FLASH_V100_DECODE_USE_PAGED_PREFILL"
        )
        self.use_decode_paged_prefill = decode_paged_prefill_env == "1"
        decode_bhmd_out_env = os.getenv("VLLM_FLASH_V100_DECODE_USE_BHMD_OUT")
        self.use_decode_paged_prefill_bhmd_out = decode_bhmd_out_env != "0"
        self.use_decode_wmma_wrapper = (
            os.getenv("VLLM_FLASH_V100_DECODE_USE_WMMA_WRAPPER", "0") == "1"
        )
        decode_scalar_paged_env = os.getenv("VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED")
        self.use_decode_scalar_paged = decode_scalar_paged_env != "0"
        self.compare_bhmd_out_dir = os.getenv(
            "VLLM_FLASH_V100_COMPARE_BHMD_OUT_DIR"
        )
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
            k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
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
            f"bhmd_compare_pid{os.getpid()}_call{call_idx}_"
            f"{time.time_ns()}.json"
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
            if kv_cache.shape[0] == 2:
                key_cache, _ = kv_cache.unbind(0)
            else:
                key_cache, _ = kv_cache.unbind(1)
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
            k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table[:num_seqs],
                seq_lens=attn_metadata.seq_lens[:num_seqs],
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=num_actual_tokens,
            )
            k_cont, v_cont = _dequantize_fp8_contiguous_kv(
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
        if kv_cache.shape[0] == 2:
            key_cache, _ = kv_cache.unbind(0)
        else:
            key_cache, _ = kv_cache.unbind(1)

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
        k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table[:num_seqs],
            seq_lens=attn_metadata.seq_lens[:num_seqs],
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            total_tokens=num_actual_tokens,
        )
        k_cont, v_cont = _dequantize_fp8_contiguous_kv(
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
            torch.cuda.synchronize(query.device)
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

    def _supports_flash_v100_path(self) -> bool:
        """Check whether current layer/config can run Flash V100 safely."""
        supported_kv_dtype = not _uses_fp8_kv_cache(
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
        attn_metadata: TritonAttentionMetadata,
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
            _record_route("metadata_none_zero_output")
            return output.fill_(0)

        if not self._supports_flash_v100_path():
            layer_info = self._layer_debug_info(layer)
            is_dflash_draft_attn = bool(
                layer_info.get("is_dflash_draft_attn")
            )
            message = (
                "FLASH_ATTN_V100 cannot run this layer/config because a required "
                "Flash op is unavailable or the attention features/KV cache dtype "
                "are unsupported. Select TRITON_ATTN for a full Triton route, or "
                "set VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1 for explicit "
                "diagnostic fallback."
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
            _record_route(
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

        is_prefill = attn_metadata.max_query_len > 1
        is_capturing = _is_cuda_graph_capturing(query)
        layer_name = self._layer_debug_info(layer).get("layer_name")
        if _draft_graph_debug_enabled():
            _draft_graph_debug_log(
                "forward:enter",
                "layer=%s is_prefill=%s is_capturing=%s max_query_len=%s "
                "max_seq_len=%s num_actual_tokens=%s %s %s %s %s %s %s",
                layer_name,
                is_prefill,
                is_capturing,
                int(attn_metadata.max_query_len),
                int(attn_metadata.max_seq_len),
                int(attn_metadata.num_actual_tokens),
                _format_tensor_debug(query, "query"),
                _format_tensor_debug(output, "output"),
                _format_tensor_debug(
                    getattr(attn_metadata, "query_start_loc", None),
                    "attn_qsl",
                ),
                _format_tensor_debug(
                    getattr(attn_metadata, "seq_lens", None),
                    "attn_seq",
                ),
                _format_tensor_debug(
                    getattr(attn_metadata, "block_table", None),
                    "attn_bt",
                ),
                _format_tensor_debug(
                    getattr(attn_metadata, "smallq_decode_seq_lens", None),
                    "smallq_seq",
                ),
            )
        _sm70_profile_trace(
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
                _metadata_expects_more_query_tokens_than_available(
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
                _sm70_profile_trace(
                    "forward branch=prefill_triton_safe layer=%s",
                    layer_name,
                )
                self._reset_decode_cache()
                _record_route("prefill_triton_safe")
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
                smallq_decode = self._small_query_decode_enabled(attn_metadata)
                if smallq_decode:
                    if _draft_graph_debug_enabled():
                        _draft_graph_debug_log(
                            "forward:prefill_capture_smallq",
                            "layer=%s %s %s %s",
                            layer_name,
                            _format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_decode_block_table",
                                    None,
                                ),
                                "smallq_bt",
                            ),
                            _format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_decode_seq_lens",
                                    None,
                                ),
                                "smallq_seq",
                            ),
                            _format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_query_start_loc",
                                    None,
                                ),
                                "smallq_qsl",
                            ),
                        )
                    _sm70_profile_trace(
                        "forward branch=prefill_capture_smallq layer=%s",
                        layer_name,
                    )
                    _record_route("prefill_capture_smallq")
                    return self._flash_v100_prefill_with_prefix(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                )
                _sm70_profile_trace(
                    "forward branch=prefill_capture_full_flash layer=%s",
                    layer_name,
                )
            has_prefix_context = (
                metadata_live_token_mismatch or _has_prefix_context(attn_metadata)
            )
            smallq_decode = has_prefix_context and self._small_query_decode_enabled(
                attn_metadata
            )
            if has_prefix_context:
                if _draft_graph_debug_enabled():
                    _draft_graph_debug_log(
                        "forward:prefill_prefix",
                        "layer=%s smallq=%s metadata_live_token_mismatch=%s "
                        "%s %s %s",
                        layer_name,
                        smallq_decode,
                        metadata_live_token_mismatch,
                        _format_tensor_debug(
                            getattr(attn_metadata, "smallq_decode_block_table", None),
                            "smallq_bt",
                        ),
                        _format_tensor_debug(
                            getattr(attn_metadata, "smallq_decode_seq_lens", None),
                            "smallq_seq",
                        ),
                        _format_tensor_debug(
                            getattr(attn_metadata, "smallq_query_start_loc", None),
                            "smallq_qsl",
                        ),
                    )
                _sm70_profile_trace(
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
                _log_fp8_kv_cache_route("prefill", self.kv_cache_dtype, "prefix")
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
                _record_route("prefill_prefix_flash")
                return result
            if not _logged_prefill_flash:
                logger.info(
                    "FLASH_ATTN_V100 prefill path active (no prefix/chunked context)."
                )
                _logged_prefill_flash = True
            self._reset_decode_cache()
            if self.use_prefill_paged_cache and self.use_flash_v100_prefill_paged:
                _sm70_profile_trace(
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
                _log_fp8_kv_cache_route(
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
                _record_route("prefill_no_prefix_paged_cache_flash")
                return result
            _sm70_profile_trace(
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
            _record_route("prefill_no_prefix_dense_flash")
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
            _sm70_profile_trace(
                "forward branch=decode_triton_no_flash_decode layer=%s",
                layer_name,
            )
            _record_route("decode_triton_no_flash_decode")
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
            _log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "decode_as_paged_prefill"
            )
            _sm70_profile_trace(
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
            _record_route("decode_paged_prefill")
            return result
        if self.use_decode_dense_cache and not is_capturing:
            _log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "dense_cache_bridge"
            )
            _sm70_profile_trace(
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
            _record_route("decode_dense_cache")
            return result
        if self.use_decode_dense_reference and not is_capturing:
            _log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "dense_reference_bridge"
            )
            _sm70_profile_trace(
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
            _record_route("decode_dense_reference")
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
            _sm70_profile_trace(
                "forward branch=decode_triton_scalar_disabled layer=%s",
                layer_name,
            )
            _record_route("decode_triton_scalar_disabled")
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
                "FLASH_ATTN_V100 decode path active (scalar paged KV kernel, "
                "CUDA-graph safe, classified Type-B for long q=1 decode)."
            )
            _logged_decode_flash = True
        _log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "scalar_paged")
        if _draft_graph_debug_enabled():
            _draft_graph_debug_log(
                "forward:decode",
                "layer=%s %s %s %s",
                layer_name,
                _format_tensor_debug(
                    getattr(attn_metadata, "query_start_loc", None),
                    "attn_qsl",
                ),
                _format_tensor_debug(
                    getattr(attn_metadata, "seq_lens", None),
                    "attn_seq",
                ),
                _format_tensor_debug(
                    getattr(attn_metadata, "block_table", None),
                    "attn_bt",
                ),
            )
        _sm70_profile_trace(
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
        _record_route("decode_scalar_paged")
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

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)

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
                    q_out_same_storage = _same_storage(raw_q_bhmd, out_bhmd)
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
        if _uses_fp8_kv_cache(self.kv_cache_dtype):
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

        key_cache = kv_cache[0] if kv_cache.shape[0] == 2 else kv_cache[:, 0]
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

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)
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
            k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table[i : i + 1],
                seq_lens=attn_metadata.seq_lens[i : i + 1],
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
        return flash_v100_dense_prefill(
            query=query,
            key=key,
            value=value,
            output=output,
            query_start_loc=query_start_loc,
            num_actual_tokens=num_actual_tokens,
            softmax_scale=self.scale,
            causal=causal,
            window_size=window_size,
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
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if query.shape[0] == 0:
            return output

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)

        self.flash_attn_decode_paged(
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
            and not _metadata_expects_more_query_tokens_than_available(
                attn_metadata,
                num_query_tokens,
            )
        ):
            query = query[:num_query_tokens]
            out_view = output[:num_query_tokens]
            if _draft_graph_debug_enabled():
                _graph_metadata_debug_log(
                    "smallq_call",
                    "layer=%s num_query_tokens=%s %s %s %s %s %s",
                    self._layer_debug_info(layer).get("layer_name"),
                    num_query_tokens,
                    _format_tensor_debug(query, "query"),
                    _format_tensor_debug(out_view, "out"),
                    _format_tensor_debug(
                        persistent_decode_block_table[:num_query_tokens],
                        "smallq_bt",
                    ),
                    _format_tensor_debug(
                        persistent_decode_seq_lens[:num_query_tokens],
                        "smallq_seq",
                    ),
                    _format_tensor_debug(persistent_query_start_loc, "smallq_qsl"),
                )
            self.flash_attn_decode_paged(
                query,
                key_cache,
                value_cache,
                persistent_decode_block_table[:num_query_tokens],
                persistent_decode_seq_lens[:num_query_tokens],
                softmax_scale=self.scale,
                out=out_view,
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=float(layer._k_scale_float),
                v_scale=float(layer._v_scale_float),
                window_size=self._flash_v100_window_size(causal=True),
            )
            return output

        if _is_cuda_graph_capturing(query):
            raise RuntimeError(
                "FLASH_ATTN_V100 small-query prefix prefill entered CUDA graph "
                "capture without persistent smallq decode metadata. The "
                "metadata builder must attach smallq_decode_block_table and "
                "smallq_decode_seq_lens so replay does not capture transient "
                "derived tensors."
            )

        query_start_loc_norm = _normalize_query_start_loc_for_available_tokens(
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
            window_size=self._flash_v100_window_size(causal=True),
        )
        return output

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
        global _logged_prefill_compare, _logged_prefill_smallq_decode
        causal = getattr(attn_metadata, "causal", True)
        window_size = self._flash_v100_window_size(causal)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        query_start_loc = _normalize_query_start_loc_for_available_tokens(
            query_start_loc,
            int(query.shape[0]),
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
        debug_compare = os.getenv("VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE", "0") == "1"
        dflash_dump = (
            _dflash_prefix_dump_enabled()
            and not _logged_dflash_prefix_dump
            and bool(getattr(layer, "is_dflash_draft_attn", False))
        )

        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        max_query_len = int(query_lens.max().item()) if num_seqs > 0 else 0
        if (
            causal
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
                    attn_metadata.block_table[i : i + 1],
                    attn_metadata.seq_lens[i : i + 1],
                    softmax_scale=self.scale,
                    kv_cache_dtype=self.kv_cache_dtype,
                    k_scale=float(layer._k_scale_float),
                    v_scale=float(layer._v_scale_float),
                    causal=causal,
                    window_size=window_size,
                )
                need_dense_debug = (
                    (debug_compare and not _logged_prefill_compare) or dflash_dump
                )
                if need_dense_debug:
                    seq_len = int(seq_lens[i].item())
                    k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                        kv_cache=kv_cache,
                        block_table=attn_metadata.block_table[i : i + 1],
                        seq_lens=attn_metadata.seq_lens[i : i + 1],
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
                    if bool(getattr(layer, "is_dflash_draft_attn", False)):
                        ref_out = _torch_attention_reference(
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
                                    _dequantize_fp8_contiguous_kv(
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
                k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table[i : i + 1],
                    seq_lens=attn_metadata.seq_lens[i : i + 1],
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
                    window_size=window_size,
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

    @classmethod
    def supports_non_causal(cls) -> bool:
        # D-Flash uses non-causal decoder attention over the draft query
        # tokens. The V100 backend handles this in the prefill paths by
        # forwarding attn_metadata.causal to FA2/Triton-compatible kernels.
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # Keep this aligned with the dense prefill kernel dispatch table.
        return [64, 128, 256]
