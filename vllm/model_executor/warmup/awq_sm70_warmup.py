# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up SM70 AWQ kernels before CUDA graph capture.

This primes the TurboMind small-shape autotuner using representative decode
shapes so CUDA graph capture replays the tuned kernels instead of the default
dispatch path.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _warmup_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_AWQ_WARMUP")
    if raw is None:
        return True
    return raw not in ("0", "false", "False")


def _single_token_compact_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_AWQ_ENABLE_SINGLE_TOKEN_COMPACT")
    if raw is None:
        enabled = True
    else:
        enabled = raw == "1"
    return enabled and hasattr(torch.ops._C, "awq_moe_single_token_sm70_out")


def _get_lut_template() -> str | None:
    return os.getenv("VLLM_SM70_GEMM_LUT_PATH")


def _resolve_lut_path(device: torch.device) -> str | None:
    template = _get_lut_template()
    if not template:
        return None
    device_idx = 0 if device.index is None else int(device.index)
    return (
        template.replace("{device}", str(device_idx))
        .replace("{arch}", "sm70")
        .replace("{rank}", str(device_idx))
    )


def _load_lut_cache(device: torch.device) -> int:
    path = _resolve_lut_path(device)
    if path is None or not hasattr(torch.ops._C, "sm70_gemm_import_cache"):
        return 0
    if not Path(path).exists():
        return 0
    device_hint = torch.empty(0, dtype=torch.uint8, device=device)
    try:
        return int(ops.sm70_gemm_import_cache(device_hint, path))
    except Exception as exc:
        logger.warning("SM70 GEMM LUT import failed from %s (%s).", path, exc)
        return 0


def _save_lut_cache(device: torch.device) -> int:
    path = _resolve_lut_path(device)
    if path is None or not hasattr(torch.ops._C, "sm70_gemm_export_cache"):
        return 0
    device_hint = torch.empty(0, dtype=torch.uint8, device=device)
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return int(ops.sm70_gemm_export_cache(device_hint, path))
    except Exception as exc:
        logger.warning("SM70 GEMM LUT export failed to %s (%s).", path, exc)
        return 0


def _get_decode_m_values(worker: "Worker") -> list[int]:
    max_dense_m = _parse_positive_int_env("VLLM_SM70_AWQ_WARMUP_MAX_M", 8)
    sizes = {1, 2, 4, 8}
    capture_sizes = worker.vllm_config.compilation_config.cudagraph_capture_sizes
    if capture_sizes is not None:
        sizes.update(
            int(size) for size in capture_sizes if 0 < int(size) <= max_dense_m
        )
    return sorted(size for size in sizes if size <= max_dense_m)


def _get_moe_token_counts(worker: "Worker") -> list[int]:
    max_tokens = _parse_positive_int_env(
        "VLLM_SM70_AWQ_WARMUP_MAX_MOE_TOKENS", 8
    )
    return [m for m in _get_decode_m_values(worker) if m <= max_tokens]


def _build_balanced_offsets(
    total_tokens: int, num_experts: int, device: torch.device
) -> torch.Tensor:
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    used_experts = min(total_tokens, num_experts)
    if used_experts > 0:
        base = total_tokens // used_experts
        rem = total_tokens % used_experts
        counts[:used_experts] = base
        if rem > 0:
            counts[:rem] += 1
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets


def _group_size_from_tm_scales(k_dim: int, tm_scales: torch.Tensor) -> int:
    num_groups = int(tm_scales.shape[0])
    return k_dim // num_groups


def _iter_unique_dense_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    seen: set[tuple[int, int, int]] = set()
    for layer in model.modules():
        if not getattr(layer, "_awq_sm70_prepared", False):
            continue
        k_dim = int(layer._awq_sm70_weight.shape[0])
        n_dim = int(layer._awq_sm70_weight.shape[1] * 8)
        group_size = _group_size_from_tm_scales(k_dim, layer._awq_sm70_scales)
        key = (k_dim, n_dim, group_size)
        if key in seen:
            continue
        seen.add(key)
        yield layer


def _iter_unique_moe_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for layer in model.modules():
        if not getattr(layer, "sm70_batched_ready", False):
            continue
        group_size = _group_size_from_tm_scales(
            int(layer.sm70_w13_k_dim), layer.w13_tm_scales[0]
        )
        key = (
            int(layer.sm70_w13_k_dim),
            int(layer.sm70_w13_n_dim),
            int(layer.sm70_w2_k_dim),
            int(layer.sm70_w2_n_dim),
            int(layer.sm70_num_experts),
            group_size,
        )
        if key in seen:
            continue
        seen.add(key)
        yield layer


def _warmup_dense_layers(
    dense_layers: list[torch.nn.Module],
    m_values: list[int],
) -> int:
    calls = 0
    for layer in dense_layers:
        device = layer._awq_sm70_weight.device
        k_dim = int(layer._awq_sm70_weight.shape[0])
        n_dim = int(layer._awq_sm70_weight.shape[1] * 8)
        group_size = _group_size_from_tm_scales(k_dim, layer._awq_sm70_scales)
        for m_dim in m_values:
            x = torch.empty((m_dim, k_dim), dtype=torch.float16, device=device)
            out = torch.empty((m_dim, n_dim), dtype=torch.float16, device=device)
            ops.awq_gemm_sm70_out(
                out,
                x,
                layer._awq_sm70_weight,
                layer._awq_sm70_scales,
                group_size,
                layer._awq_sm70_k_ld,
                layer._awq_sm70_q_ld,
                False,
            )
            calls += 1
    return calls


def _warmup_moe_layers(
    moe_layers: list[torch.nn.Module],
    token_counts: list[int],
) -> int:
    calls = 0
    for layer in moe_layers:
        device = layer.w13_tm_weight.device
        top_k = int(layer._buf_top_k)
        group_size = _group_size_from_tm_scales(
            int(layer.sm70_w13_k_dim), layer.w13_tm_scales[0]
        )
        for num_tokens in token_counts:
            total_slots = num_tokens * top_k
            expert_offsets = _build_balanced_offsets(
                total_slots, int(layer.sm70_num_experts), device
            )
            permuted_input = torch.empty(
                (total_slots, int(layer.sm70_w13_k_dim)),
                dtype=torch.float16,
                device=device,
            )
            intermediate = torch.empty(
                (total_slots, int(layer.sm70_intermediate_size)),
                dtype=torch.float16,
                device=device,
            )
            sorted_output = torch.empty(
                (total_slots, int(layer.sm70_w2_n_dim)),
                dtype=torch.float16,
                device=device,
            )

            ops.awq_moe_gemm_sm70_out(
                intermediate,
                permuted_input,
                expert_offsets,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                int(layer.sm70_num_experts),
                int(layer.sm70_w13_k_dim),
                int(layer.sm70_w13_n_dim),
                group_size,
                True,
            )
            ops.awq_moe_gemm_sm70_out(
                sorted_output,
                intermediate,
                expert_offsets,
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                int(layer.sm70_num_experts),
                int(layer.sm70_w2_k_dim),
                int(layer.sm70_w2_n_dim),
                group_size,
            )
            calls += 2
    return calls


def _warmup_single_token_compact_layers(
    moe_layers: list[torch.nn.Module],
) -> int:
    calls = 0
    if not _single_token_compact_enabled():
        return calls

    for layer in moe_layers:
        if not getattr(layer, "sm70_batched_ready", False):
            continue
        device = layer.w13_tm_weight.device
        top_k = int(layer._buf_top_k)
        hidden_size = int(layer.sm70_hidden_logical_size)

        x = torch.empty((1, hidden_size), dtype=torch.float16, device=device)
        topk_weights = torch.full(
            (1, top_k),
            1.0 / max(top_k, 1),
            dtype=torch.float32,
            device=device,
        )
        topk_ids = torch.arange(top_k, dtype=torch.int64, device=device)
        output = layer._buf_output[:1]
        compact_input = layer._buf_permuted_input[:top_k]
        intermediate = layer._buf_intermediate[:top_k]
        sorted_output = layer._buf_sorted_output[:top_k]

        ops.awq_moe_single_token_sm70_out(
            output,
            x,
            topk_weights,
            topk_ids,
            layer.w13_strided_ptrs_w_rows,
            layer.w13_strided_ptrs_s_rows,
            layer.w2_strided_ptrs_w_rows,
            layer.w2_strided_ptrs_s_rows,
            compact_input,
            intermediate,
            sorted_output,
            layer._buf_single_w13_ptrs_w,
            layer._buf_single_w13_ptrs_s,
            layer._buf_single_w2_ptrs_w,
            layer._buf_single_w2_ptrs_s,
            layer._buf_single_expert_offsets,
            layer._buf_single_inv_permuted_idx,
            int(layer.sm70_w13_k_dim),
            int(layer.sm70_w13_n_dim),
            int(layer.sm70_w2_k_dim),
            int(layer.sm70_w2_n_dim),
            int(_group_size_from_tm_scales(
                int(layer.sm70_w13_k_dim), layer.w13_tm_scales[0]
            )),
            hidden_size,
        )
        calls += 1
    return calls


def sm70_awq_warmup(worker: "Worker") -> None:
    if not _warmup_enabled() or not hasattr(torch.ops._C, "awq_gemm_sm70_out"):
        return

    device = worker.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 0):
        return

    model = worker.get_model()
    dense_layers = list(_iter_unique_dense_layers(model))
    moe_layers = list(_iter_unique_moe_layers(model))
    if not dense_layers and not moe_layers:
        return

    imported_records = _load_lut_cache(device)
    if imported_records > 0:
        logger.info(
            "Loaded SM70 GEMM LUT (%d records) for device %s.",
            imported_records,
            device,
        )

    m_values = _get_decode_m_values(worker)
    moe_token_counts = _get_moe_token_counts(worker)

    logger.info(
        "Warming up SM70 AWQ kernels (%d dense shapes, %d MoE shapes).",
        len(dense_layers),
        len(moe_layers),
    )
    with torch.inference_mode():
        dense_calls = _warmup_dense_layers(dense_layers, m_values)
        moe_calls = _warmup_moe_layers(moe_layers, moe_token_counts)
        compact_calls = _warmup_single_token_compact_layers(moe_layers)
    torch.cuda.synchronize(device)
    logger.info(
        "SM70 AWQ warmup finished (%d dense calls, %d MoE calls, "
        "%d compact MoE calls).",
        dense_calls,
        moe_calls,
        compact_calls,
    )
    exported_records = _save_lut_cache(device)
    if exported_records > 0:
        logger.info(
            "Saved SM70 GEMM LUT (%d records) for device %s.",
            exported_records,
            device,
        )
