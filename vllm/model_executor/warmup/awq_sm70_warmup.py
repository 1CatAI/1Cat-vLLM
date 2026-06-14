# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up accepted SM70 TurboMind routes before CUDA graph capture."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm import _sm70_ops as sm70_ops
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _group_size_from_tm_scales(k_dim: int, tm_scales: torch.Tensor) -> int:
    num_groups = int(tm_scales.shape[0])
    return k_dim // num_groups


def _resolve_lut_path(device: torch.device) -> str | None:
    template = envs.VLLM_SM70_GEMM_LUT_PATH
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
        logger.info(
            "No SM70 GEMM LUT cache found at %s for device %s.", path, device
        )
        return 0
    device_hint = torch.empty(0, dtype=torch.uint8, device=device)
    try:
        return int(sm70_ops.sm70_gemm_import_cache(device_hint, path))
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
        return int(sm70_ops.sm70_gemm_export_cache(device_hint, path))
    except Exception as exc:
        logger.warning("SM70 GEMM LUT export failed to %s (%s).", path, exc)
        return 0


def _spec_decode_query_len(worker: Worker) -> int:
    spec_config = worker.vllm_config.speculative_config
    if spec_config is None:
        return 1
    return max(1, 1 + int(spec_config.num_speculative_tokens))


def _get_decode_m_values(worker: Worker) -> list[int]:
    spec_query_len = _spec_decode_query_len(worker)
    max_dense_m = max(1, int(envs.VLLM_SM70_AWQ_WARMUP_MAX_M), spec_query_len)
    sizes = {1, 2, 4, 8}
    sizes.add(spec_query_len)
    capture_sizes = worker.vllm_config.compilation_config.cudagraph_capture_sizes
    if capture_sizes is not None:
        sizes.update(
            int(size) for size in capture_sizes if 0 < int(size) <= max_dense_m
        )
    return sorted(size for size in sizes if size <= max_dense_m)


def _get_moe_token_counts(worker: Worker) -> list[int]:
    max_tokens = max(
        1,
        int(envs.VLLM_SM70_AWQ_WARMUP_MAX_MOE_TOKENS),
        _spec_decode_query_len(worker),
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


def _iter_unique_fp8_dense_layers(
    model: torch.nn.Module,
) -> Iterable[tuple[torch.nn.Module, bool]]:
    seen: set[tuple[int, int, bool]] = set()
    for layer in model.modules():
        if not getattr(layer, "sm70_fp8_turbomind", False):
            continue

        k_dim = int(layer.weight.shape[0])
        n_dim = int(layer.output_size_per_partition)
        key = (k_dim, n_dim, False)
        if key not in seen:
            seen.add(key)
            yield layer, False

        if not getattr(layer, "sm70_fp8_gated_silu", False):
            continue
        gated_n_dim = int(layer.sm70_fp8_gated_silu_weight.shape[1])
        key = (k_dim, gated_n_dim, True)
        if key in seen:
            continue
        seen.add(key)
        yield layer, True


def _is_awq_moe_sm70_layer(layer: torch.nn.Module) -> bool:
    return all(
        hasattr(layer, name)
        for name in (
            "w13_strided_ptrs_w",
            "w13_strided_ptrs_s",
            "w2_strided_ptrs_w",
            "w2_strided_ptrs_s",
            "w13_tm_scales",
            "sm70_w13_k_dim",
            "sm70_w13_n_dim",
            "sm70_w2_k_dim",
            "sm70_w2_n_dim",
            "sm70_num_experts",
            "_awq_moe_buf_top_k",
        )
    )


def _iter_unique_moe_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for layer in model.modules():
        if not _is_awq_moe_sm70_layer(layer):
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
            sm70_ops.awq_gemm_sm70_out(
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


def _warmup_fp8_dense_layers(
    dense_layers: list[tuple[torch.nn.Module, bool]],
    m_values: list[int],
) -> int:
    if not hasattr(torch.ops._C, "fp8_gemm_sm70_out_meta"):
        return 0

    calls = 0
    for layer, gated_silu in dense_layers:
        if gated_silu:
            weight = layer.sm70_fp8_gated_silu_weight
            scales = layer.sm70_fp8_gated_silu_scales
            k_ld = int(layer.sm70_fp8_gated_silu_k_ld)
            q_ld = int(layer.sm70_fp8_gated_silu_q_ld)
            n_dim = int(weight.shape[1]) // 2
        else:
            weight = layer.weight
            scales = layer.weight_scale_inv
            k_ld = int(layer.sm70_fp8_k_ld)
            q_ld = int(layer.sm70_fp8_q_ld)
            n_dim = int(layer.output_size_per_partition)

        device = weight.device
        k_dim = int(weight.shape[0])
        for m_dim in m_values:
            x = torch.empty((m_dim, k_dim), dtype=torch.float16, device=device)
            out = torch.empty((m_dim, n_dim), dtype=torch.float16, device=device)
            sm70_ops.fp8_gemm_sm70_out(
                out, x, weight, scales, 128, k_ld, q_ld, gated_silu
            )
            calls += 1
    return calls


def _warmup_moe_dense_stage_layers(
    moe_layers: list[torch.nn.Module],
    token_counts: list[int],
) -> int:
    if not hasattr(torch.ops._C, "awq_moe_dense_stage_sm70_out"):
        return 0

    calls = 0
    for layer in moe_layers:
        device = layer.w13_tm_scales.device
        top_k = int(layer._awq_moe_buf_top_k)
        num_experts = int(layer.sm70_num_experts)
        group_size = _group_size_from_tm_scales(
            int(layer.sm70_w13_k_dim), layer.w13_tm_scales[0]
        )
        dense_expert_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
        for num_tokens in token_counts:
            total_slots = num_tokens * top_k
            expert_offsets = _build_balanced_offsets(
                total_slots, num_experts, device
            )
            permuted_input = torch.empty(
                (total_slots, int(layer.sm70_w13_k_dim)),
                dtype=torch.float16,
                device=device,
            )
            gate_up = torch.empty(
                (total_slots, int(layer.sm70_w13_n_dim)),
                dtype=torch.float16,
                device=device,
            )
            intermediate = torch.empty(
                (total_slots, int(layer.sm70_w2_k_dim)),
                dtype=torch.float16,
                device=device,
            )
            sorted_output = torch.empty(
                (total_slots, int(layer.sm70_w2_n_dim)),
                dtype=torch.float16,
                device=device,
            )

            sm70_ops.awq_moe_dense_stage_sm70_out(
                gate_up,
                permuted_input,
                expert_offsets,
                dense_expert_ids,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                num_experts,
                int(layer.sm70_w13_k_dim),
                int(layer.sm70_w13_n_dim),
                group_size,
            )
            torch.ops._C.silu_and_mul(intermediate, gate_up)
            sm70_ops.awq_moe_dense_stage_sm70_out(
                sorted_output,
                intermediate,
                expert_offsets,
                dense_expert_ids,
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                num_experts,
                int(layer.sm70_w2_k_dim),
                int(layer.sm70_w2_n_dim),
                group_size,
            )
            calls += 2
    return calls


def _warmup_moe_single_token_layers(moe_layers: list[torch.nn.Module]) -> int:
    if not (
        hasattr(torch.ops._C, "awq_moe_single_token_dense_w13_sm70_out")
        and hasattr(torch.ops._C, "awq_moe_single_token_dense_stage_sm70_out")
    ):
        return 0

    calls = 0
    for layer in moe_layers:
        device = layer.w13_tm_scales.device
        top_k = int(layer._awq_moe_buf_top_k)
        num_experts = int(layer.sm70_num_experts)
        hidden_size = int(
            getattr(layer, "sm70_hidden_logical_size", layer.sm70_w13_k_dim)
        )
        group_size = _group_size_from_tm_scales(
            int(layer.sm70_w13_k_dim), layer.w13_tm_scales[0]
        )

        x = torch.empty((1, hidden_size), dtype=torch.float16, device=device)
        topk_ids = torch.arange(top_k, dtype=torch.int32, device=device)
        topk_ids.remainder_(max(num_experts, 1))
        compact_input = torch.empty(
            (top_k, hidden_size), dtype=torch.float16, device=device
        )
        gate_up = torch.empty(
            (top_k, int(layer.sm70_w13_n_dim)),
            dtype=torch.float16,
            device=device,
        )
        intermediate = torch.empty(
            (top_k, int(layer.sm70_w2_k_dim)),
            dtype=torch.float16,
            device=device,
        )
        sorted_output = torch.empty(
            (top_k, int(layer.sm70_w2_n_dim)),
            dtype=torch.float16,
            device=device,
        )
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        expert_offsets64 = torch.empty(
            num_experts + 1, dtype=torch.int64, device=device
        )
        inv_permuted_idx = torch.empty(top_k, dtype=torch.int32, device=device)
        sorted_expert_ids = torch.empty(top_k, dtype=torch.int32, device=device)

        sm70_ops.awq_moe_single_token_dense_w13_sm70_out(
            gate_up,
            compact_input,
            x,
            topk_ids,
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            expert_offsets,
            expert_offsets64,
            inv_permuted_idx,
            sorted_expert_ids,
            int(layer.sm70_w13_k_dim),
            int(layer.sm70_w13_n_dim),
            group_size,
            hidden_size,
        )
        torch.ops._C.silu_and_mul(intermediate, gate_up)
        sm70_ops.awq_moe_single_token_dense_stage_sm70_out(
            sorted_output,
            intermediate,
            expert_offsets,
            sorted_expert_ids,
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            top_k,
            int(layer.sm70_w2_k_dim),
            int(layer.sm70_w2_n_dim),
            group_size,
        )
        calls += 2

        if (
            envs.VLLM_SM70_AWQ_MOE_LEGACY_SINGLE_TOKEN_COMPACT
            and hasattr(torch.ops._C, "awq_moe_single_token_sm70_out")
            and hasattr(layer, "w13_legacy_strided_ptrs_w_rows")
            and hasattr(layer, "w2_strided_ptrs_w_rows")
        ):
            topk_weights = torch.empty(top_k, dtype=torch.float32, device=device)
            topk_weights.fill_(1.0 / max(top_k, 1))
            ptr_row_bytes = int(layer.sm70_ptr_row_bytes)
            legacy_w13_ptrs_w = torch.empty(
                top_k, ptr_row_bytes, dtype=torch.uint8, device=device
            )
            legacy_w13_ptrs_s = torch.empty(
                top_k, ptr_row_bytes, dtype=torch.uint8, device=device
            )
            legacy_w2_ptrs_w = torch.empty(
                top_k, ptr_row_bytes, dtype=torch.uint8, device=device
            )
            legacy_w2_ptrs_s = torch.empty(
                top_k, ptr_row_bytes, dtype=torch.uint8, device=device
            )
            legacy_output = torch.empty(
                (1, hidden_size), dtype=torch.float16, device=device
            )
            legacy_offsets = torch.empty(
                top_k + 1, dtype=torch.int32, device=device
            )
            sm70_ops.awq_moe_single_token_sm70_out(
                legacy_output,
                x,
                topk_weights,
                topk_ids,
                layer.w13_legacy_strided_ptrs_w_rows,
                layer.w13_legacy_strided_ptrs_s_rows,
                layer.w2_strided_ptrs_w_rows,
                layer.w2_strided_ptrs_s_rows,
                compact_input,
                intermediate,
                sorted_output,
                legacy_w13_ptrs_w,
                legacy_w13_ptrs_s,
                legacy_w2_ptrs_w,
                legacy_w2_ptrs_s,
                legacy_offsets,
                inv_permuted_idx,
                int(layer.sm70_w13_k_dim),
                int(layer.sm70_w13_n_dim),
                int(layer.sm70_w2_k_dim),
                int(layer.sm70_w2_n_dim),
                group_size,
                hidden_size,
            )
            calls += 1
    return calls


def sm70_awq_warmup(worker: Worker) -> None:
    if not envs.VLLM_SM70_AWQ_WARMUP:
        return
    if not hasattr(torch.ops._C, "awq_gemm_sm70_out"):
        return

    device = worker.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 0):
        return

    model = worker.get_model()
    dense_layers = list(_iter_unique_dense_layers(model))
    fp8_dense_layers = list(_iter_unique_fp8_dense_layers(model))
    moe_layers = list(_iter_unique_moe_layers(model))
    if not dense_layers and not fp8_dense_layers and not moe_layers:
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
    lut_path = _resolve_lut_path(device)

    logger.info(
        "Warming up SM70 AWQ/FP8 accepted routes (%d AWQ dense layer shapes, "
        "%d FP8 dense layer shapes, %d MoE layer shapes, dense_m=%s, "
        "moe_tokens=%s, lut_path=%s).",
        len(dense_layers),
        len(fp8_dense_layers),
        len(moe_layers),
        m_values,
        moe_token_counts,
        lut_path,
    )
    with torch.inference_mode():
        dense_calls = _warmup_dense_layers(dense_layers, m_values)
        fp8_dense_calls = _warmup_fp8_dense_layers(fp8_dense_layers, m_values)
        moe_stage_calls = _warmup_moe_dense_stage_layers(
            moe_layers, moe_token_counts
        )
        single_token_calls = _warmup_moe_single_token_layers(moe_layers)
    torch.cuda.synchronize(device)
    logger.info(
        "SM70 AWQ warmup finished (%d AWQ dense calls, %d FP8 dense calls, "
        "%d MoE stage calls, %d single-token active-expert calls).",
        dense_calls,
        fp8_dense_calls,
        moe_stage_calls,
        single_token_calls,
    )
    exported_records = _save_lut_cache(device)
    if exported_records > 0:
        logger.info(
            "Saved SM70 GEMM LUT (%d records) for device %s.",
            exported_records,
            device,
        )
