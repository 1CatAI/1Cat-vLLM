# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense D256 prefill dispatch (FA2, 79T, split-D, split-KV) and its workspaces."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import torch

from vllm.logger import init_logger, log_once_seen, set_log_once_state
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import masks as _masks
from vllm.v1.attention.backends.flash_v100 import ops as _ops
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100 import workspace as _workspace
from vllm.v1.attention.ops.sm70_grouped import (
    clear_grouped_fp16_workspaces,
)
from vllm.v1.attention.ops.sm70_workspaces import retain_for_capture, workspace_cache

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")
LEGACY_OBSERVATIONS: dict[str, tuple[ModuleType, str]]
_sm70_fa2_cu_seqlens_cache: dict[
    tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}


_FP8_PREFILL_BRIDGE_PAGE_SIZE = 784
_SM70_79T_CORE_QUERY_LEN = 8000
_SM70_79T_MAX_QUERY_LEN = 8192
_SM70_79T_EXACT_QUERY_ALIGNMENT = 64
_SM70_79T_KV_ALIGNMENT = 32
_SM70_SPLITD_KV_ALIGNMENT = 32
_fp8_prefill_bridge_workspaces: dict[
    tuple[int, int, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_fp8_prefill_bridge_tail_workspaces: dict[
    tuple[int, int, torch.dtype, int, int],
    tuple[torch.Tensor, torch.Tensor],
] = {}
_prefill_dense_splitkv3_workspaces: dict[
    tuple[int, int, torch.dtype],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_sm70_79t_q8192_padding_workspaces: dict[
    tuple[int, int, torch.dtype, int, int, int],
    tuple[torch.Tensor, torch.Tensor],
] = {}


def clear_flash_attn_v100_workspaces(config=None) -> None:
    """Release this engine, or the independent legacy caller when unconfigured."""
    if config is not None:
        resources = getattr(config, "_runtime_resources", {})
        owner = resources.pop("attention_workspaces", None)
        if owner is not None:
            owner.close()
        return
    clear_grouped_fp16_workspaces()
    _sm70_fa2_cu_seqlens_cache.clear()
    _fp8_prefill_bridge_workspaces.clear()
    _fp8_prefill_bridge_tail_workspaces.clear()
    _kv_layout._prefill_gather_dense_workspaces.clear()
    _prefill_dense_splitkv3_workspaces.clear()
    _sm70_79t_q8192_padding_workspaces.clear()


def uniform_cu_seqlens(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    query_len: int,
    kv_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = workspace_cache("sm70_fa2_cu_seqlens_cache", _sm70_fa2_cu_seqlens_cache)
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    cache_key = (device_index, batch_size, query_len, kv_len)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    cu_q = torch.arange(
        0,
        (batch_size + 1) * query_len,
        query_len,
        dtype=torch.int32,
        device=tensor.device,
    )
    cu_k = torch.arange(
        0,
        (batch_size + 1) * kv_len,
        kv_len,
        dtype=torch.int32,
        device=tensor.device,
    )
    cache[cache_key] = (cu_q, cu_k)
    return cu_q, cu_k


def _get_prefill_dense_splitkv3_workspace(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    cache = workspace_cache(
        "prefill_dense_splitkv3_workspaces", _prefill_dense_splitkv3_workspaces
    )
    if _routing.is_cuda_graph_capturing(query):
        return None
    device_index = query.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index() if query.is_cuda else -1
    stream_id = (
        int(torch.cuda.current_stream(query.device).cuda_stream) if query.is_cuda else 0
    )
    cache_key = (device_index, stream_id, query.dtype)
    expected_out_shape = (3, *query.shape)
    expected_stats_shape = (3, *query.shape[:-1])
    workspace = cache.get(cache_key)
    if (
        workspace is not None
        and workspace[0].shape == expected_out_shape
        and workspace[1].shape == expected_stats_shape
    ):
        return workspace

    cache.pop(cache_key, None)
    workspace = None
    try:
        partial_out = torch.empty(
            expected_out_shape,
            dtype=torch.float32,
            device=query.device,
        )
        partial_max = torch.empty(
            expected_stats_shape,
            dtype=torch.float32,
            device=query.device,
        )
        partial_sum = torch.empty_like(partial_max)
    except torch.OutOfMemoryError:
        if not log_once_seen("flash_v100._warned_prefill_dense_splitkv3_oom"):
            logger.warning_once(
                "Insufficient memory for the long-prefill split-KV3 FP32 "
                "workspace; falling back to the exact dense kernel.",
                scope="process",
                key="flash_v100._warned_prefill_dense_splitkv3_oom",
            )
            set_log_once_state("flash_v100._warned_prefill_dense_splitkv3_oom", True)
        return None
    workspace = (partial_out, partial_max, partial_sum)
    cache[cache_key] = workspace
    return workspace


def _should_use_prefill_dense_splitkv3(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    splitkv3_op: Callable[..., torch.Tensor] | None,
) -> bool:
    return (
        _config.options().value("prefill_dense_splitkv3")
        and splitkv3_op is not None
        and (
            query.shape == (1, 4096, 6, 256)
            or (
                _config.options().value("prefill_dense_splitkv3_q8000_experimental")
                and query.shape == (1, 8000, 6, 256)
            )
        )
        and key.ndim == 4
        and key.shape[0] == 1
        and key.shape[1] == max_seqlen_k
        and key.shape[2:] == (1, 256)
        and max_seqlen_q == query.shape[1]
        and max_seqlen_k >= _config.options().value("prefill_dense_splitkv3_min_kv")
        and max_seqlen_k > max_seqlen_q
        and not _routing.is_cuda_graph_capturing(query)
    )


def _should_use_prefill_d256_gqa_architecture(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    architecture_op: Callable[..., torch.Tensor] | None,
) -> bool:
    """Use the v37 family or the Q8000-core long-prefill dispatcher."""
    if _config.options().value("prefill_d256_gqa_v37"):
        shape_allowed = (
            64 <= max_seqlen_q <= 8192
            and max_seqlen_q % 64 == 0
            and max_seqlen_q < max_seqlen_k <= 262144
            and max_seqlen_k % 32 == 0
        )
    else:
        shape_allowed = (
            _SM70_79T_CORE_QUERY_LEN <= max_seqlen_q <= _SM70_79T_MAX_QUERY_LEN
            and max_seqlen_q <= max_seqlen_k <= 262144
            and max_seqlen_k % _SM70_79T_KV_ALIGNMENT == 0
        )
    return (
        _config.options().value("prefill_d256_gqa_arch_128k_experimental")
        and architecture_op is not None
        and shape_allowed
        and query.ndim == 4
        and query.shape[0] > 0
        and query.shape[1] == max_seqlen_q
        and query.shape[3] == 256
        and key.ndim == 4
        and key.shape[0] == query.shape[0]
        and key.shape[2] > 0
        and key.shape[3] == 256
        and query.shape[2] == 6 * key.shape[2]
        and value.shape == key.shape
        and max_seqlen_k == key.shape[1]
        and query.dtype == torch.float16
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.device == key.device
        and query.device == value.device
        and query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and abs(softmax_scale - 0.0625) <= 1.0e-8
        and not _routing.is_cuda_graph_capturing(query)
    )


# Native prefill storage is shared by layers in an engine. Initialize it
# during memory profiling so the KV allocator does not consume its budget.
_sm70_prefill_profiled_workspaces: set[tuple[torch.device, int]] = set()


def profile_sm70_prefill_workspace(query: torch.Tensor, num_kv_heads: int) -> None:
    if (
        not _config.options().value("prefill_d256_gqa_arch_128k_experimental")
        or not _config.options().value("fa2_d256_prefill")
        or _config.options().value("prefill_d256_gqa_v37")
        or not query.is_cuda
        or query.dtype != torch.float16
        or query.ndim != 3
        or query.shape[2] != 256
        or query.shape[1] != 6 * num_kv_heads
        or query.shape[0] < _SM70_79T_CORE_QUERY_LEN
        or not current_platform.is_device_capability(70)
        or torch.cuda.is_current_stream_capturing()
    ):
        return
    q_len = (
        _SM70_79T_CORE_QUERY_LEN
        if query.shape[0] == _SM70_79T_CORE_QUERY_LEN
        else _SM70_79T_MAX_QUERY_LEN
    )
    cache_key = (query.device, q_len)
    profiled = workspace_cache("prefill_profiled", _sm70_prefill_profiled_workspaces)
    if cache_key in profiled:
        return
    op = (
        _ops.get_sm70_d256_gqa_architecture_op()
        if q_len == _SM70_79T_CORE_QUERY_LEN
        else _ops.get_sm70_d256_gqa_architecture_q8192_op()
    )
    if op is None:
        return
    q = torch.zeros((1, q_len, 6, 256), device=query.device, dtype=query.dtype)
    kv = torch.zeros((1, q_len, 1, 256), device=query.device, dtype=query.dtype)
    op(q, kv, kv, torch.empty_like(q), 0.0625, True)
    profiled.add(cache_key)
    logger.info_once("SM70 Q%d prefill workspace included in memory profiling.", q_len)


def _run_sm70_gqa_groups(
    op: Callable[..., torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Apply the native GQA6 core independently to each local KV head."""
    if query.shape[0] == 1 and key.shape[2] == 1:
        return op(query, key, value, out, softmax_scale, causal)
    for batch in range(query.shape[0]):
        for head in range(key.shape[2]):
            group_q = query[batch : batch + 1, :, head * 6 : (head + 1) * 6]
            group_out = torch.empty_like(group_q, memory_format=torch.contiguous_format)
            op(
                group_q.contiguous(),
                key[batch : batch + 1, :, head : head + 1].contiguous(),
                value[batch : batch + 1, :, head : head + 1].contiguous(),
                group_out,
                softmax_scale,
                causal,
            )
            out[batch : batch + 1, :, head * 6 : (head + 1) * 6].copy_(group_out)
    return out


def _run_sm70_d256_gqa_79t_dispatch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    *,
    softmax_scale: float,
    architecture_op: Callable[..., torch.Tensor],
    dense_op: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Run Q8000 directly and preserve it as the core of Q8001..Q8192.

    For a Q8000+R causal chunk, the leading R rows attend K[:KV-8000].
    The remaining 8000 rows have the same causal alignment as the qualified
    Q8000 operator against the full K/V tensors.  Padding only the small
    leading fringe to 64 rows keeps the exact Split-D contract without adding
    work to the Q8000 core.
    """
    query_len = int(query.shape[1])
    fringe_len = query_len - _SM70_79T_CORE_QUERY_LEN
    if fringe_len < 0 or query_len > _SM70_79T_MAX_QUERY_LEN:
        raise ValueError(f"unsupported SM70 79T query length {query_len}")
    if fringe_len == 0:
        return _run_sm70_gqa_groups(
            architecture_op,
            query,
            key,
            value,
            out,
            softmax_scale,
            True,
        )

    core_query = query[:, fringe_len:]
    core_out = out[:, fringe_len:]
    _run_sm70_gqa_groups(
        architecture_op,
        core_query,
        key,
        value,
        core_out,
        softmax_scale,
        True,
    )

    fringe_kv_len = int(key.shape[1]) - _SM70_79T_CORE_QUERY_LEN
    padded_fringe_len = (
        _masks.cdiv_int(fringe_len, _SM70_79T_EXACT_QUERY_ALIGNMENT)
        * _SM70_79T_EXACT_QUERY_ALIGNMENT
    )
    if padded_fringe_len == fringe_len:
        fringe_query = query[:, :fringe_len]
        fringe_out = out[:, :fringe_len]
        fringe_prefix = 0
    else:
        fringe_query = torch.zeros(
            (query.shape[0], padded_fringe_len, *query.shape[2:]),
            dtype=query.dtype,
            device=query.device,
        )
        fringe_out = torch.empty_like(fringe_query)
        fringe_prefix = padded_fringe_len - fringe_len
        fringe_query[:, fringe_prefix:].copy_(query[:, :fringe_len])

    dense_op(
        fringe_query,
        key[:, :fringe_kv_len],
        value[:, :fringe_kv_len],
        fringe_out,
        softmax_scale,
        True,
    )
    if fringe_prefix:
        out[:, :fringe_len].copy_(fringe_out[:, fringe_prefix:])
    return out


def _get_sm70_79t_q8192_padding_workspace(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = workspace_cache(
        "sm70_79t_q8192_padding_workspaces", _sm70_79t_q8192_padding_workspaces
    )
    if not query.is_cuda:
        padded_query = torch.empty(
            (query.shape[0], _SM70_79T_MAX_QUERY_LEN, *query.shape[2:]),
            dtype=query.dtype,
            device=query.device,
        )
        return padded_query, torch.empty_like(padded_query)

    device_index = query.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    stream_id = int(torch.cuda.current_stream(query.device).cuda_stream)
    cache_key = (
        device_index,
        stream_id,
        query.dtype,
        int(query.shape[0]),
        int(query.shape[2]),
        int(query.shape[3]),
    )
    workspace = cache.get(cache_key)
    if workspace is None:
        shape = (query.shape[0], _SM70_79T_MAX_QUERY_LEN, *query.shape[2:])
        padded_query = torch.empty(shape, dtype=query.dtype, device=query.device)
        workspace = padded_query, torch.empty_like(padded_query)
        cache[cache_key] = workspace
    return workspace


def _run_sm70_d256_gqa_79t_q8192_dispatch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    *,
    softmax_scale: float,
    architecture_q8192_op: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Run Q8001..Q8192 through the native Q8192 specialization."""
    query_len = int(query.shape[1])
    if not _SM70_79T_CORE_QUERY_LEN < query_len <= _SM70_79T_MAX_QUERY_LEN:
        raise ValueError(f"unsupported SM70 Q8192 dispatch length {query_len}")
    if query_len == _SM70_79T_MAX_QUERY_LEN:
        return _run_sm70_gqa_groups(
            architecture_q8192_op,
            query,
            key,
            value,
            out,
            softmax_scale,
            True,
        )

    padded_query, padded_out = _get_sm70_79t_q8192_padding_workspace(query)
    leading_padding = _SM70_79T_MAX_QUERY_LEN - query_len
    # A scheduler can split the first chunk below 8192 while other requests
    # decode. If KV is also shorter, pad only future keys and shift the real
    # query slice back by the same amount. For original query i, the last
    # visible key remains KV - Q + i; padded keys are always masked out.
    kv_padding = max(0, _SM70_79T_MAX_QUERY_LEN - int(key.shape[1]))
    if kv_padding:
        key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, kv_padding))
        value = torch.nn.functional.pad(value, (0, 0, 0, 0, 0, kv_padding))
        leading_padding -= kv_padding
    padded_query.zero_()
    padded_query[:, leading_padding : leading_padding + query_len].copy_(query)
    _run_sm70_gqa_groups(
        architecture_q8192_op,
        padded_query,
        key,
        value,
        padded_out,
        softmax_scale,
        True,
    )
    out.copy_(padded_out[:, leading_padding : leading_padding + query_len])
    return out


def try_sm70_fa2_d256_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    out: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor | None:
    int32_max = torch.iinfo(torch.int32).max
    if not _config.options().value("fa2_d256_prefill"):
        return None
    if (
        query.device.type != "cuda"
        or query.dtype != torch.float16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or query.stride(-1) != 1
        or key.stride(-1) != 1
        or value.stride(-1) != 1
        or any(stride > int32_max for stride in query.stride()[:-1])
        or any(stride > int32_max for stride in key.stride()[:-1])
        or (out is not None and out.stride(-1) != 1)
        or (out is not None and not out.is_contiguous())
        or query.shape[-1] != 256
        or key.shape[-1] != 256
        or value.shape[-1] != 256
        or max_seqlen_q < 64
        or not causal
        or window_size != (-1, -1)
        or cu_seqlens_q.device != query.device
        or cu_seqlens_q.dtype != torch.int32
        or not cu_seqlens_q.is_contiguous()
    ):
        return None
    paged_kv = block_table is not None
    if max_seqlen_q < 1024:
        if paged_kv or not _config.options().value("prefill_d256_gqa_v37"):
            return None
        if not _should_use_prefill_d256_gqa_architecture(
            query,
            key,
            value,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            architecture_op=_ops.get_sm70_d256_gqa_architecture_op(),
        ):
            return None
    if block_table is not None:
        if (
            seqused_k is None
            or cu_seqlens_k is not None
            or key.ndim != 4
            or value.ndim != 4
            or key.shape[1] % 16 != 0
            or block_table.device != query.device
            or block_table.dtype != torch.int32
            or block_table.stride(-1) != 1
            or seqused_k.device != query.device
            or seqused_k.dtype != torch.int32
            or not seqused_k.is_contiguous()
        ):
            return None
    elif (
        cu_seqlens_k is None
        or seqused_k is not None
        or cu_seqlens_k.device != query.device
        or cu_seqlens_k.dtype != torch.int32
        or not cu_seqlens_k.is_contiguous()
    ):
        return None
    device_index = query.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    device_capability = current_platform.get_device_capability(device_index)
    if device_capability is None or (
        device_capability.major,
        device_capability.minor,
    ) != (7, 0):
        return None

    splitd_ops, splitd_eligible = _splitd_admission(
        query, paged_kv, max_seqlen_q, max_seqlen_k
    )

    if splitd_eligible:
        dense_op, paged_op, splitkv3_op = splitd_ops
        splitd_result = None
        if paged_kv:
            splitd_eligible = (
                query.shape[0] == 1
                and block_table is not None
                and block_table.shape[0] == 1
                and key.shape[1] % 4 == 0
                and max_seqlen_k <= block_table.shape[1] * key.shape[1]
            )
            if splitd_eligible:
                splitd_out = out if out is not None else torch.empty_like(query)
                splitd_result = paged_op(
                    query,
                    key,
                    value,
                    block_table,
                    splitd_out,
                    max_seqlen_k,
                    softmax_scale,
                    True,
                )
        else:
            splitd_eligible = (
                key.ndim == 4
                and value.ndim == 4
                and key.shape[0] == query.shape[0]
                and key.shape[1] == max_seqlen_k
            )
            if splitd_eligible:
                splitd_out = out if out is not None else torch.empty_like(query)
                splitd_result = _try_dense_architecture(
                    query,
                    key,
                    value,
                    splitd_out,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale,
                    dense_op,
                )

                if splitd_result is None and _should_use_prefill_dense_splitkv3(
                    query,
                    key,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    splitkv3_op=splitkv3_op,
                ):
                    workspace = _get_prefill_dense_splitkv3_workspace(query)
                    if workspace is not None:
                        partial_out, partial_max, partial_sum = workspace
                        splitd_result = splitkv3_op(
                            query,
                            key,
                            value,
                            partial_out,
                            partial_max,
                            partial_sum,
                            splitd_out,
                            softmax_scale,
                            True,
                        )

                        if not log_once_seen(
                            "flash_v100._logged_prefill_dense_splitkv3"
                        ):
                            logger.info_once(
                                "FLASH_ATTN_V100 SM70 exact dense split-KV3 "
                                "long-prefill route active (q=%d kv=%d).",
                                max_seqlen_q,
                                max_seqlen_k,
                                scope="process",
                                key="flash_v100._logged_prefill_dense_splitkv3",
                            )
                            set_log_once_state(
                                "flash_v100._logged_prefill_dense_splitkv3", True
                            )
                        _routing.record_route(
                            _routing.ROUTE_SPECS[
                                "prefill_dense_splitd_d256_splitkv3_kernel"
                            ].name
                        )
                if (
                    splitd_result is None
                    and max_seqlen_q % _SM70_79T_EXACT_QUERY_ALIGNMENT == 0
                    and max_seqlen_k % _SM70_SPLITD_KV_ALIGNMENT == 0
                ):
                    splitd_result = dense_op(
                        query, key, value, splitd_out, softmax_scale, True
                    )
        if splitd_result is not None:
            result = splitd_result.reshape(query.shape)
            if out is not None:
                return out.reshape(query.shape)
            return result
    return None


def get_fp8_prefill_bridge_workspace(
    key_cache: torch.Tensor,
    required_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    cache = workspace_cache(
        "fp8_prefill_bridge_workspaces", _fp8_prefill_bridge_workspaces
    )
    device_index = (
        key_cache.device.index
        if key_cache.device.index is not None
        else (torch.accelerator.current_device_index() if key_cache.is_cuda else -1)
    )
    stream_id = (
        int(torch.cuda.current_stream(key_cache.device).cuda_stream)
        if key_cache.is_cuda
        else 0
    )
    cache_key = (
        device_index,
        stream_id,
        int(key_cache.shape[2]),
        int(key_cache.shape[3]),
    )
    workspace = cache.get(cache_key)
    if workspace is not None and workspace[0].shape[0] >= required_blocks:
        retain_for_capture(cache, workspace, key_cache)
        return (
            workspace[0][:required_blocks],
            workspace[1][:required_blocks],
            workspace[2][:, :required_blocks],
        )

    if _routing.is_cuda_graph_capturing(key_cache):
        return None

    previous_capacity = workspace[0].shape[0] if workspace is not None else 0
    capacity = max(required_blocks, previous_capacity * 2)
    shape = (
        capacity,
        _FP8_PREFILL_BRIDGE_PAGE_SIZE,
        key_cache.shape[2],
        key_cache.shape[3],
    )

    def _allocate() -> tuple[torch.Tensor, ...]:
        key_out = torch.empty(shape, dtype=torch.float16, device=key_cache.device)
        value_out = torch.empty_like(key_out)
        block_table = torch.arange(
            capacity,
            dtype=torch.int32,
            device=key_cache.device,
        ).unsqueeze(0)
        return key_out, value_out, block_table

    # Drop the old buffers before allocating the grown ones; see
    # _allocate_growing_workspace. The stale entry is not restored on failure
    # so that a later, smaller request re-allocates from scratch instead of
    # inheriting a doubled capacity that already failed once.
    workspace = None
    cache.pop(cache_key, None)
    allocated = _workspace.allocate_growing_workspace(
        _allocate,
        on_cuda=key_cache.is_cuda,
    )
    if allocated is None:
        return None
    key_out, value_out, block_table = allocated
    cache[cache_key] = (
        key_out,
        value_out,
        block_table,
    )
    return (
        key_out[:required_blocks],
        value_out[:required_blocks],
        block_table[:, :required_blocks],
    )


def get_fp8_prefill_bridge_tail_workspace(
    query: torch.Tensor,
    padded_query_len: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    cache = workspace_cache(
        "fp8_prefill_bridge_tail_workspaces", _fp8_prefill_bridge_tail_workspaces
    )
    device_index = (
        query.device.index
        if query.device.index is not None
        else (torch.accelerator.current_device_index() if query.is_cuda else -1)
    )
    stream_id = (
        int(torch.cuda.current_stream(query.device).cuda_stream) if query.is_cuda else 0
    )
    cache_key = (
        device_index,
        stream_id,
        query.dtype,
        int(query.shape[2]),
        int(query.shape[3]),
    )
    workspace = cache.get(cache_key)
    if workspace is not None and workspace[0].shape[1] >= padded_query_len:
        retain_for_capture(cache, workspace, query)
        return (
            workspace[0][:, :padded_query_len],
            workspace[1][:, :padded_query_len],
        )

    if _routing.is_cuda_graph_capturing(query):
        return None

    previous_capacity = workspace[0].shape[1] if workspace is not None else 0
    capacity = max(padded_query_len, previous_capacity * 2)
    shape = (1, capacity, query.shape[2], query.shape[3])

    def _allocate() -> tuple[torch.Tensor, ...]:
        padded_query = torch.empty(shape, dtype=query.dtype, device=query.device)
        return padded_query, torch.empty_like(padded_query)

    workspace = None
    cache.pop(cache_key, None)
    allocated = _workspace.allocate_growing_workspace(_allocate, on_cuda=query.is_cuda)
    if allocated is None:
        return None
    padded_query, padded_output = allocated
    cache[cache_key] = (
        padded_query,
        padded_output,
    )
    return (
        padded_query[:, :padded_query_len],
        padded_output[:, :padded_query_len],
    )


def flash_v100_dense_prefill_available() -> bool:
    flash_attn_func, _, _, _, _, _, _, _, _ = _ops.get_flash_ops()
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
    query_start_loc_device: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Flash-V100 dense raw-QKV prefill without backend metadata coupling."""
    flash_attn_func, _, _, _, _, _, _, _, _ = _ops.get_flash_ops()
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
    min_seq_len = int(seq_lens.min().item())
    max_seq_len = int(seq_lens.max().item())
    if min_seq_len >= 1024 and query_start_loc_device is not None:
        splitd_query = query
        splitd_key = key
        splitd_value = value
        splitd_out = out_view
        if (
            query.ndim == 3
            and min_seq_len == max_seq_len
            and num_actual_tokens == num_seqs * max_seq_len
        ):
            splitd_query = query.view(num_seqs, max_seq_len, *query.shape[1:])
            splitd_key = key.view(num_seqs, max_seq_len, *key.shape[1:])
            splitd_value = value.view(num_seqs, max_seq_len, *value.shape[1:])
            splitd_out = out_view.view(num_seqs, max_seq_len, *out_view.shape[1:])

        fa2_out = try_sm70_fa2_d256_prefill(
            splitd_query,
            splitd_key,
            splitd_value,
            cu_seqlens_q=query_start_loc_device,
            cu_seqlens_k=query_start_loc_device,
            max_seqlen_q=max_seq_len,
            max_seqlen_k=max_seq_len,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            out=splitd_out,
        )
        if fa2_out is not None:
            if not log_once_seen("flash_v100._logged_prefill_fa2_d256"):
                logger.info_once(
                    "FLASH_ATTN_V100 SM70 exact Split-D D256 "
                    "software-pipelined dense prefill path active.",
                    scope="process",
                    key="flash_v100._logged_prefill_fa2_d256",
                )
                set_log_once_state("flash_v100._logged_prefill_fa2_d256", True)
            _routing.record_route(
                _routing.ROUTE_SPECS["prefill_dense_splitd_d256"].name
            )
            return output

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
                out_batch.view(
                    tok_end - tok_start, out_batch.shape[2], out_batch.shape[3]
                )
            )

        run_start = run_end

    return output


def flash_v100_dense_prefill_lse_available() -> bool:
    return _ops.get_flash_dense_forward() is not None


def flash_v100_dense_prefill_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_actual_tokens: int,
    softmax_scale: float,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash-V100 dense varlen prefill that also emits the softmax LSE.

    Inputs use packed ``[tokens, heads, dim]`` layout and ``softmax_lse`` uses
    the FA2 ``[query_heads, total_query_tokens]`` convention. Empty key segments
    produce zero output and ``-inf`` LSE, making them neutral when attention
    states are merged. Causal calls require equal Q/K lengths for every sequence.
    """
    fwd = _ops.get_flash_dense_forward()
    if fwd is None:
        raise RuntimeError("flash_attn_v100 dense LSE prefill op is unavailable")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("dense LSE prefill expects packed [T, H, D] Q/K/V")
    if (
        query.dtype != torch.float16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        raise TypeError("dense LSE prefill requires fp16 Q/K/V")
    if query.device != key.device or query.device != value.device:
        raise ValueError("dense LSE prefill Q/K/V must share a device")
    if key.shape != value.shape:
        raise ValueError("dense LSE prefill K/V shapes must match")
    if query.shape[2] != key.shape[2]:
        raise ValueError("dense LSE prefill Q/K/V head dimensions must match")
    if key.shape[1] <= 0 or query.shape[1] % key.shape[1] != 0:
        raise ValueError("dense LSE prefill has an invalid Q/K head mapping")
    if output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError("dense LSE prefill output must match the query")
    if output.device != query.device:
        raise ValueError("dense LSE prefill output must share the query device")
    if softmax_lse.shape != (query.shape[1], query.shape[0]):
        raise ValueError("dense LSE prefill LSE must have shape [Hq, Tq]")
    if softmax_lse.dtype != torch.float32 or softmax_lse.device != query.device:
        raise ValueError("dense LSE prefill LSE must be fp32 on the query device")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("dense LSE prefill sequence metadata must be one-dimensional")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("Q/K sequence metadata must describe the same batch")
    if num_actual_tokens < 0 or num_actual_tokens > query.shape[0]:
        raise ValueError("num_actual_tokens is outside the query bounds")

    # A single device-to-host transfer replaces per-sequence scalar syncs.
    q_off = cu_seqlens_q.tolist()
    k_off = cu_seqlens_k.tolist()
    if not q_off or q_off[0] != 0 or k_off[0] != 0:
        raise ValueError("Q/K sequence offsets must start at zero")
    if any(a > b for a, b in zip(q_off, q_off[1:])) or any(
        a > b for a, b in zip(k_off, k_off[1:])
    ):
        raise ValueError("Q/K sequence offsets must be nondecreasing")
    if q_off[-1] != num_actual_tokens:
        raise ValueError("Q sequence offsets must cover every actual query token")
    if k_off[-1] != key.shape[0]:
        raise ValueError("K sequence offsets must cover every packed key/value token")

    query = query[:num_actual_tokens]
    out_view = output[:num_actual_tokens]
    lse_view = softmax_lse[:, :num_actual_tokens]
    num_seqs = len(q_off) - 1
    if num_seqs == 0:
        return output, softmax_lse

    window_size_left, window_size_right = window_size
    num_q_heads, head_dim = query.shape[1], query.shape[2]
    num_kv_heads = key.shape[1]

    run_start = 0
    while run_start < num_seqs:
        q_len = q_off[run_start + 1] - q_off[run_start]
        k_len = k_off[run_start + 1] - k_off[run_start]
        # Batch the maximal run sharing BOTH lengths — the kernel takes a
        # rectangular [B,H,M,D] x [B,H,N,D] batch.
        run_end = run_start + 1
        while (
            run_end < num_seqs
            and q_off[run_end + 1] - q_off[run_end] == q_len
            and k_off[run_end + 1] - k_off[run_end] == k_len
        ):
            run_end += 1

        if q_len > 0:
            qt0, qt1 = q_off[run_start], q_off[run_end]
            batch_size = run_end - run_start

            if k_len == 0:
                # This is the neutral element expected by merge_attn_states.
                out_view[qt0:qt1].zero_()
                lse_view[:, qt0:qt1].fill_(float("-inf"))
            else:
                if causal and q_len != k_len:
                    raise RuntimeError(
                        "flash_v100_dense_prefill_lse: causal=True requires "
                        f"q_len == k_len (got {q_len} vs {k_len})"
                    )
                kt0, kt1 = k_off[run_start], k_off[run_end]

                # [T,H,D] -> [B,M,H,D] -> [B,H,M,D].
                q_batch = (
                    query[qt0:qt1]
                    .view(batch_size, q_len, num_q_heads, head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                k_batch = (
                    key[kt0:kt1]
                    .view(batch_size, k_len, num_kv_heads, head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                v_batch = (
                    value[kt0:kt1]
                    .view(batch_size, k_len, num_kv_heads, head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )

                out_batch, lse_batch, _, _ = fwd(
                    q_batch,
                    k_batch,
                    v_batch,
                    None,
                    0.0,
                    softmax_scale,
                    causal,
                    window_size_left,
                    window_size_right,
                    0.0,
                    None,
                    False,
                )

                # [B,H,M,D] -> [B,M,H,D] -> packed [B*M,H,D].
                out_view[qt0:qt1].copy_(
                    out_batch.permute(0, 2, 1, 3).reshape(-1, num_q_heads, head_dim)
                )
                # [B,H,M] -> [H,B*M], matching FA2's [num_heads, total_q].
                lse_view[:, qt0:qt1].copy_(
                    lse_batch.permute(1, 0, 2).reshape(num_q_heads, -1)
                )

        run_start = run_end

    return output, softmax_lse


# Public owner operations; legacy bindings are installed by package assembly.
LEGACY_ALIASES = {
    "_get_fp8_prefill_bridge_tail_workspace": "get_fp8_prefill_bridge_tail_workspace",
    "_try_sm70_fa2_d256_prefill": "try_sm70_fa2_d256_prefill",
    "_uniform_cu_seqlens": "uniform_cu_seqlens",
    "_get_fp8_prefill_bridge_workspace": "get_fp8_prefill_bridge_workspace",
    "_profile_sm70_prefill_workspace": "profile_sm70_prefill_workspace",
}


def _try_dense_architecture(
    query, key, value, splitd_out, max_seqlen_q, max_seqlen_k, softmax_scale, dense_op
):
    splitd_result = None
    architecture_op = (
        _ops.get_sm70_d256_gqa_architecture_op()
        if _config.options().value("prefill_d256_gqa_arch_128k_experimental")
        else None
    )
    architecture_q8192_op = (
        _ops.get_sm70_d256_gqa_architecture_q8192_op()
        if architecture_op is not None
        and not _config.options().value("prefill_d256_gqa_v37")
        and max_seqlen_q > _SM70_79T_CORE_QUERY_LEN
        else None
    )
    if _should_use_prefill_d256_gqa_architecture(
        query,
        key,
        value,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        architecture_op=architecture_op,
    ):
        assert architecture_op is not None
        try:
            if _config.options().value("prefill_d256_gqa_v37"):
                splitd_result = _run_sm70_gqa_groups(
                    architecture_op,
                    query,
                    key,
                    value,
                    splitd_out,
                    softmax_scale,
                    True,
                )
            elif architecture_q8192_op is not None:
                splitd_result = _run_sm70_d256_gqa_79t_q8192_dispatch(
                    query,
                    key,
                    value,
                    splitd_out,
                    softmax_scale=softmax_scale,
                    architecture_q8192_op=architecture_q8192_op,
                )
            elif (
                max_seqlen_q == _SM70_79T_CORE_QUERY_LEN
                or max_seqlen_k % _SM70_SPLITD_KV_ALIGNMENT == 0
            ):
                splitd_result = _run_sm70_d256_gqa_79t_dispatch(
                    query,
                    key,
                    value,
                    splitd_out,
                    softmax_scale=softmax_scale,
                    architecture_op=architecture_op,
                    dense_op=dense_op,
                )
        except torch.OutOfMemoryError:
            if not log_once_seen(
                "flash_v100._warned_prefill_d256_gqa_architecture_oom"
            ):
                logger.warning_once(
                    "Insufficient memory for the default-on "
                    "SM70 D256 GQA long-prefill architecture; "
                    "falling back to the exact dense kernel.",
                    scope="process",
                    key="flash_v100._warned_prefill_d256_gqa_architecture_oom",
                )
                set_log_once_state(
                    "flash_v100._warned_prefill_d256_gqa_architecture_oom",
                    True,
                )
        if splitd_result is not None:
            if not log_once_seen("flash_v100._logged_prefill_d256_gqa_architecture"):
                logger.info_once(
                    "FLASH_ATTN_V100 SM70 D256 GQA "
                    "long-prefill architecture route active (%s).",
                    "v37 FP32"
                    if _config.options().value("prefill_d256_gqa_v37")
                    else "Q8000 core / Q8192 QK+PV FP32 dispatch",
                    scope="process",
                    key="flash_v100._logged_prefill_d256_gqa_architecture",
                )
                set_log_once_state(
                    "flash_v100._logged_prefill_d256_gqa_architecture", True
                )
            _routing.record_route(
                _routing.ROUTE_SPECS["prefill_dense_d256_gqa_arch_long"].name
            )
            if _config.options().value("prefill_d256_gqa_v37"):
                _routing.record_route(
                    _routing.ROUTE_SPECS["prefill_dense_d256_gqa_v37"].name
                )
            else:
                _routing.record_route(
                    _routing.ROUTE_SPECS["prefill_dense_d256_gqa_79t_fp32"].name
                )
                if max_seqlen_q > _SM70_79T_CORE_QUERY_LEN:
                    if architecture_q8192_op is not None:
                        _routing.record_route(
                            _routing.ROUTE_SPECS[
                                "prefill_dense_d256_gqa_79t_fp32_q8192"
                            ].name
                        )
                        if max_seqlen_q < _SM70_79T_MAX_QUERY_LEN:
                            _routing.record_route(
                                _routing.ROUTE_SPECS[
                                    "prefill_dense_d256_gqa_79t_fp32_q8192_pad"
                                ].name
                            )
                    else:
                        _routing.record_route(
                            _routing.ROUTE_SPECS[
                                "prefill_dense_d256_gqa_79t_fp32_fringe_fallback"
                            ].name
                        )
    return splitd_result


def _splitd_admission(query, paged_kv, max_seqlen_q, max_seqlen_k):
    splitd_ops = _ops.get_sm70_splitd_d256_ops()
    q8000_core_dispatch_eligible = (
        not paged_kv
        and not _config.options().value("prefill_d256_gqa_v37")
        and _SM70_79T_CORE_QUERY_LEN <= max_seqlen_q <= _SM70_79T_MAX_QUERY_LEN
    )
    architecture_kv_eligible = (
        q8000_core_dispatch_eligible and max_seqlen_k % _SM70_79T_KV_ALIGNMENT == 0
    )
    exact_splitd_shape_eligible = (
        max_seqlen_q % _SM70_79T_EXACT_QUERY_ALIGNMENT == 0
        and max_seqlen_k % _SM70_SPLITD_KV_ALIGNMENT == 0
    )
    splitd_eligible = (
        splitd_ops is not None
        and query.ndim == 4
        and query.shape[1] == max_seqlen_q
        and (architecture_kv_eligible or exact_splitd_shape_eligible)
    )
    return splitd_ops, splitd_eligible
