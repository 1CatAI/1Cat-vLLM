# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up accepted SM70 TurboMind routes before CUDA graph capture."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm import _sm70_ops as sm70_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    _prepare_compact_slot_groups,
)

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


def _lut_cache_disabled_for_dynamic_quant_dispatch(
    has_awq_dense: bool,
    has_fp8_dense: bool,
    fp4_kinds: set[str],
    has_mxfp4_moe: bool = False,
    has_nvfp4_moe: bool = False,
) -> bool:
    awq_no_preserve = (
        has_awq_dense
        and envs.VLLM_SM70_AWQ_TUNE_SMALL_SHAPES
        and not envs.VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS
        and not envs.VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY
    )
    fp8_dynamic = (
        has_fp8_dense
        and envs.VLLM_SM70_FP8_TUNE_SMALL_SHAPES
        and not envs.VLLM_SM70_FP8_REUSE_IMPORTED_CACHE
    )
    mxfp4_dynamic = (
        "mxfp4" in fp4_kinds or has_mxfp4_moe
    ) and envs.VLLM_SM70_MXFP4_TUNE_SMALL_SHAPES
    nvfp4_dynamic = (
        "nvfp4" in fp4_kinds or has_nvfp4_moe
    ) and envs.VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES
    return awq_no_preserve or fp8_dynamic or mxfp4_dynamic or nvfp4_dynamic


def _load_lut_cache(device: torch.device, skip_import: bool = False) -> int:
    path = _resolve_lut_path(device)
    if path is None or not hasattr(torch.ops._C, "sm70_gemm_import_cache"):
        return 0
    if skip_import:
        logger.info(
            "Skipping SM70 GEMM LUT import for dynamic quant dispatch "
            "(path=%s, device=%s).",
            path,
            device,
        )
        return 0
    if not Path(path).exists():
        logger.info("No SM70 GEMM LUT cache found at %s for device %s.", path, device)
        return 0
    device_hint = torch.empty(0, dtype=torch.uint8, device=device)
    try:
        return int(sm70_ops.sm70_gemm_import_cache(device_hint, path))
    except Exception as exc:
        logger.warning("SM70 GEMM LUT import failed from %s (%s).", path, exc)
        return 0


def _save_lut_cache(device: torch.device, skip_export: bool = False) -> int:
    path = _resolve_lut_path(device)
    if path is None or not hasattr(torch.ops._C, "sm70_gemm_export_cache"):
        return 0
    if skip_export:
        logger.info(
            "Skipping SM70 GEMM LUT export for dynamic quant dispatch "
            "(path=%s, device=%s).",
            path,
            device,
        )
        return 0
    device_hint = torch.empty(0, dtype=torch.uint8, device=device)
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return int(sm70_ops.sm70_gemm_export_cache(device_hint, path))
    except Exception as exc:
        logger.warning("SM70 GEMM LUT export failed to %s (%s).", path, exc)
        return 0


def _export_lut_bytes(device: torch.device) -> tuple[bytes | None, int]:
    if not hasattr(torch.ops._C, "sm70_gemm_export_cache"):
        return None, 0
    device_hint = torch.empty(0, dtype=torch.uint8, device=device)
    with tempfile.TemporaryDirectory(prefix="vllm-sm70-gemm-export-") as tmpdir:
        path = Path(tmpdir) / "gemm-lut.bin"
        records = int(sm70_ops.sm70_gemm_export_cache(device_hint, str(path)))
        if records <= 0 or not path.exists():
            return None, records
        return path.read_bytes(), records


def _import_lut_bytes(device: torch.device, payload: bytes) -> int:
    if not hasattr(torch.ops._C, "sm70_gemm_import_cache"):
        return 0
    device_hint = torch.empty(0, dtype=torch.uint8, device=device)
    with tempfile.TemporaryDirectory(prefix="vllm-sm70-gemm-import-") as tmpdir:
        path = Path(tmpdir) / "gemm-lut.bin"
        path.write_bytes(payload)
        return int(sm70_ops.sm70_gemm_import_cache(device_hint, str(path)))


def _warmup_fp8_dense_layers_coordinated(
    layers: list[tuple[torch.nn.Module, bool]],
    m_values: list[int],
    device: torch.device,
) -> int:
    if not layers:
        return 0
    if (
        not envs.VLLM_SM70_FP8_TUNE_SMALL_SHAPES
        or not envs.VLLM_SM70_FP8_COORDINATED_TUNING
    ):
        return _warmup_fp8_dense_layers(layers, m_values)

    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    if tp_group.world_size <= 1:
        return _warmup_fp8_dense_layers(layers, m_values)

    is_leader = tp_group.rank_in_group == 0
    if is_leader:
        _warmup_fp8_dense_layers(layers, m_values)
        torch.accelerator.synchronize(device)
        payload, exported_records = _export_lut_bytes(device)
    else:
        payload = None
        exported_records = 0

    payload = tp_group.broadcast_object(payload, src=0)
    if payload is None:
        logger.warning(
            "SM70 FP8 coordinated tuning produced no LUT; rank %d is "
            "falling back to local tuning.",
            tp_group.rank_in_group,
        )
        return _warmup_fp8_dense_layers(layers, m_values)

    imported_records = (
        exported_records if is_leader else _import_lut_bytes(device, payload)
    )
    tp_group.barrier()
    calls = _warmup_fp8_dense_layers(layers, m_values)
    torch.accelerator.synchronize(device)
    logger.info(
        "SM70 FP8 coordinated tuning loaded %d rank0 LUT records on rank %d.",
        imported_records,
        tp_group.rank_in_group,
    )
    return calls


def _silu_and_mul_w13(
    layer: torch.nn.Module, out: torch.Tensor, gate_up: torch.Tensor
) -> None:
    if getattr(layer, "sm70_awq_moe_w13_interleaved", False):
        sm70_ops.silu_and_mul_interleaved(out, gate_up)
    else:
        torch.ops._C.silu_and_mul(out, gate_up)


def _spec_decode_query_len(worker: Worker) -> int:
    spec_config = worker.vllm_config.speculative_config
    if spec_config is None:
        return 1
    return max(1, 1 + int(spec_config.num_speculative_tokens))


def _get_decode_m_values(worker: Worker) -> list[int]:
    spec_query_len = _spec_decode_query_len(worker)
    max_dense_m = max(1, int(envs.VLLM_SM70_AWQ_WARMUP_MAX_M), spec_query_len)
    sizes = {1, 2, 4, 8}
    pow2 = 16
    while pow2 <= max_dense_m:
        sizes.add(pow2)
        pow2 *= 2
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
        if not (
            getattr(layer, "sm70_fp8_turbomind", False)
            or getattr(layer, "sm70_modelopt_fp8_turbomind", False)
        ):
            continue
        # QPN8 has a static source-selected dispatch and does not populate the
        # TurboMind LUT warmed below. CUDA graph capture exercises its admitted
        # M<=8 kernels separately.
        if getattr(layer, "sm70_fp8_qpn8", False):
            continue

        if getattr(layer, "sm70_fp8_bmm", False):
            k_dim = int(layer.weight.shape[1])
            n_dim = int(layer.sm70_fp8_bmm_output_size)
            key = (k_dim, n_dim, False)
            if key not in seen:
                seen.add(key)
                yield layer, False
            continue

        k_dim = int(layer.weight.shape[0])
        n_dim = int(layer.output_size_per_partition)
        if not getattr(layer, "sm70_fp8_gated_silu_primary", False):
            key = (k_dim, n_dim, False)
            if key not in seen:
                seen.add(key)
                yield layer, False

        if not getattr(layer, "sm70_fp8_gated_silu", False):
            continue
        if getattr(layer, "sm70_fp8_gated_silu_primary", False):
            gated_n_dim = int(layer.weight.shape[1])
        else:
            gated_n_dim = int(layer.sm70_fp8_gated_silu_weight.shape[1])
        key = (k_dim, gated_n_dim, True)
        if key in seen:
            continue
        seen.add(key)
        yield layer, True


def _iter_unique_fp4_dense_layers(model: torch.nn.Module) -> Iterable[Any]:
    seen: set[tuple[str, int, int, int, bool]] = set()
    for layer in model.modules():
        state = getattr(layer, sm70_tm.STATE_ATTR, None)
        if state is None or state.op_kind not in ("mxfp4", "nvfp4"):
            continue
        k_dim = int(state.weight.shape[0])
        n_dim = int(state.output_size)
        key = (
            state.op_kind,
            k_dim,
            n_dim,
            int(state.group_size),
            bool(state.gated_silu),
        )
        if key in seen:
            continue
        seen.add(key)
        yield state


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


def _is_mxfp4_moe_sm70_layer(layer: torch.nn.Module) -> bool:
    return getattr(layer, "sm70_mxfp4_moe", False) and all(
        hasattr(layer, name)
        for name in (
            "w13_strided_ptrs_w",
            "w13_strided_ptrs_s",
            "w2_strided_ptrs_w",
            "w2_strided_ptrs_s",
            "sm70_mxfp4_num_experts",
            "sm70_mxfp4_w13_k_dim",
            "sm70_mxfp4_w13_n_dim",
            "sm70_mxfp4_w2_k_dim",
            "sm70_mxfp4_w2_n_dim",
            "sm70_mxfp4_group_size",
            "_mxfp4_sm70_buf_dense_expert_ids",
        )
    )


def _iter_unique_mxfp4_moe_layers(
    model: torch.nn.Module,
) -> Iterable[torch.nn.Module]:
    seen: set[tuple[int, int, int, int, int]] = set()
    for layer in model.modules():
        if not _is_mxfp4_moe_sm70_layer(layer):
            continue
        key = (
            int(layer.sm70_mxfp4_w13_k_dim),
            int(layer.sm70_mxfp4_w13_n_dim),
            int(layer.sm70_mxfp4_w2_k_dim),
            int(layer.sm70_mxfp4_w2_n_dim),
            int(layer.sm70_mxfp4_num_experts),
        )
        if key in seen:
            continue
        seen.add(key)
        yield layer


def _is_nvfp4_moe_sm70_layer(layer: torch.nn.Module) -> bool:
    return getattr(layer, "sm70_nvfp4_moe", False) and all(
        hasattr(layer, name)
        for name in (
            "w13_strided_ptrs_w",
            "w13_strided_ptrs_s",
            "w2_strided_ptrs_w",
            "w2_strided_ptrs_s",
            "sm70_nvfp4_num_experts",
            "sm70_nvfp4_w13_k_dim",
            "sm70_nvfp4_w13_n_dim",
            "sm70_nvfp4_w2_k_dim",
            "sm70_nvfp4_w2_n_dim",
            "sm70_nvfp4_group_size",
            "_nvfp4_sm70_dense_expert_ids",
        )
    )


def _iter_unique_nvfp4_moe_layers(
    model: torch.nn.Module,
) -> Iterable[torch.nn.Module]:
    seen: set[tuple[int, int, int, int, int]] = set()
    for layer in model.modules():
        if not _is_nvfp4_moe_sm70_layer(layer):
            continue
        key = (
            int(layer.sm70_nvfp4_w13_k_dim),
            int(layer.sm70_nvfp4_w13_n_dim),
            int(layer.sm70_nvfp4_w2_k_dim),
            int(layer.sm70_nvfp4_w2_n_dim),
            int(layer.sm70_nvfp4_num_experts),
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
        is_modelopt = getattr(layer, "sm70_modelopt_fp8_turbomind", False)
        if getattr(layer, "sm70_fp8_bmm", False):
            assert not gated_silu
            weight = layer.weight[0]
            scales = layer.weight_scale_inv[0]
            k_ld = int(layer.sm70_fp8_k_ld)
            q_ld = int(layer.sm70_fp8_q_ld)
            n_dim = int(layer.sm70_fp8_bmm_output_size)
        elif gated_silu:
            if getattr(layer, "sm70_fp8_gated_silu_primary", False):
                weight = layer.weight
                scales = layer.weight_scale_inv
                k_ld = int(layer.sm70_fp8_k_ld)
                q_ld = int(layer.sm70_fp8_q_ld)
            else:
                weight = layer.sm70_fp8_gated_silu_weight
                scales = layer.sm70_fp8_gated_silu_scales
                k_ld = int(layer.sm70_fp8_gated_silu_k_ld)
                q_ld = int(layer.sm70_fp8_gated_silu_q_ld)
            n_dim = int(weight.shape[1]) // 2
        else:
            weight = layer.weight
            if is_modelopt:
                scales = layer.weight_scale
                k_ld = int(layer.sm70_modelopt_fp8_k_ld)
                q_ld = int(layer.sm70_modelopt_fp8_q_ld)
            else:
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
        if getattr(layer, "sm70_fp8_bmm_grouped_decode", False):
            group_count = int(layer.sm70_fp8_bmm_groups)
            grouped_x = torch.empty(
                (group_count, k_dim), dtype=torch.float16, device=device
            )
            grouped_out = torch.empty(
                (group_count, n_dim), dtype=torch.float16, device=device
            )
            sm70_ops.fp8_moe_gemm_sm70_per_expert_dispatch_out(
                grouped_out,
                grouped_x,
                layer.sm70_fp8_bmm_grouped_offsets,
                layer.sm70_fp8_bmm_grouped_ptrs_w,
                layer.sm70_fp8_bmm_grouped_ptrs_s,
                group_count,
                k_dim,
                n_dim,
                128,
                False,
            )
            calls += 1
    return calls


def _warmup_fp4_dense_layers(
    dense_layers: list[Any],
    m_values: list[int],
) -> int:
    calls = 0
    for state in dense_layers:
        device = state.weight.device
        k_dim = int(state.weight.shape[0])
        gated_silu = bool(state.gated_silu)
        n_dim = int(state.output_size) // (2 if gated_silu else 1)
        for m_dim in m_values:
            x = torch.empty((m_dim, k_dim), dtype=torch.float16, device=device)
            out = torch.empty((m_dim, n_dim), dtype=torch.float16, device=device)
            if state.op_kind == "mxfp4":
                if not hasattr(torch.ops._C, "mxfp4_gemm_sm70_out"):
                    continue
                sm70_ops.mxfp4_gemm_sm70_out(
                    out,
                    x,
                    state.weight,
                    state.scales,
                    int(state.group_size),
                    int(state.k_ld),
                    int(state.q_ld),
                    gated_silu,
                )
            elif state.op_kind == "nvfp4":
                if not hasattr(torch.ops._C, "nvfp4_gemm_sm70_out"):
                    continue
                sm70_ops.nvfp4_gemm_sm70_out(
                    out,
                    x,
                    state.weight,
                    state.scales,
                    int(state.group_size),
                    int(state.k_ld),
                    int(state.q_ld),
                    gated_silu,
                )
            else:
                continue
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
            expert_offsets = _build_balanced_offsets(total_slots, num_experts, device)
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
            _silu_and_mul_w13(layer, intermediate, gate_up)
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


def _warmup_mxfp4_moe_b1_layers(moe_layers: list[torch.nn.Module]) -> int:
    """Initialize MXFP4 MoE's exact-B1 TM dispatch before graph capture."""
    if not hasattr(torch.ops._C, "mxfp4_moe_dense_stage_sm70_out"):
        return 0

    calls = 0
    for layer in moe_layers:
        device = layer.w13_tm_weight.device
        top_k = int(layer.moe_config.experts_per_token)
        num_experts = int(layer.sm70_mxfp4_num_experts)
        total_slots = top_k
        expert_offsets = _build_balanced_offsets(total_slots, num_experts, device)
        permuted_input = torch.empty(
            total_slots,
            int(layer.sm70_mxfp4_w13_k_dim),
            dtype=torch.float16,
            device=device,
        )
        gate_up = torch.empty(
            total_slots,
            int(layer.sm70_mxfp4_w13_n_dim),
            dtype=torch.float16,
            device=device,
        )
        intermediate = torch.empty(
            total_slots,
            int(layer.sm70_mxfp4_w2_k_dim),
            dtype=torch.float16,
            device=device,
        )
        sorted_output = torch.empty(
            total_slots,
            int(layer.sm70_mxfp4_w2_n_dim),
            dtype=torch.float16,
            device=device,
        )
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            gate_up,
            permuted_input,
            expert_offsets,
            layer._mxfp4_sm70_buf_dense_expert_ids,
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            num_experts,
            int(layer.sm70_mxfp4_w13_k_dim),
            int(layer.sm70_mxfp4_w13_n_dim),
            int(layer.sm70_mxfp4_group_size),
        )
        if layer.swiglu_limit is None:
            torch.ops._C.silu_and_mul(intermediate, gate_up)
        else:
            torch.ops._C.silu_and_mul_with_clamp(
                intermediate, gate_up, float(layer.swiglu_limit)
            )
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            sorted_output,
            intermediate,
            expert_offsets,
            layer._mxfp4_sm70_buf_dense_expert_ids,
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            num_experts,
            int(layer.sm70_mxfp4_w2_k_dim),
            int(layer.sm70_mxfp4_w2_n_dim),
            int(layer.sm70_mxfp4_group_size),
        )
        calls += 2
    return calls


def _warmup_nvfp4_moe_decode_layers(
    moe_layers: list[torch.nn.Module], token_counts: list[int]
) -> int:
    """Tune compact and full-group NVFP4 graph shapes."""
    if not hasattr(torch.ops._C, "nvfp4_moe_dense_stage_sm70_out"):
        return 0

    calls = 0
    for layer in moe_layers:
        device = layer.w13_tm_weight.device
        top_k = int(layer.moe_config.experts_per_token)
        max_experts = int(layer.sm70_nvfp4_num_experts)
        compact_max_tokens = int(
            getattr(layer, "sm70_nvfp4_compact_grouped_max_tokens", 8)
        )
        for num_tokens in token_counts:
            total_slots = num_tokens * top_k
            if num_tokens <= compact_max_tokens:
                stage_experts = total_slots
                sorted_expert_ids = layer._nvfp4_sm70_dense_expert_ids[:total_slots]
                expert_offsets = torch.empty(
                    total_slots + 1, dtype=torch.int32, device=device
                )
                active_expert_ids = torch.empty(
                    total_slots, dtype=torch.int32, device=device
                )
                _prepare_compact_slot_groups(
                    sorted_expert_ids,
                    expert_offsets,
                    active_expert_ids,
                )
            else:
                stage_experts = max_experts
                expert_offsets = _build_balanced_offsets(
                    total_slots, max_experts, device
                )
                active_expert_ids = layer._nvfp4_sm70_dense_expert_ids
            permuted_input = torch.empty(
                total_slots,
                int(layer.sm70_nvfp4_w13_k_dim),
                dtype=torch.float16,
                device=device,
            )
            gate_up = torch.empty(
                total_slots,
                int(layer.sm70_nvfp4_w13_n_dim),
                dtype=torch.float16,
                device=device,
            )
            intermediate = torch.empty(
                total_slots,
                int(layer.sm70_nvfp4_w2_k_dim),
                dtype=torch.float16,
                device=device,
            )
            sorted_output = torch.empty(
                total_slots,
                int(layer.sm70_nvfp4_w2_n_dim),
                dtype=torch.float16,
                device=device,
            )
            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                gate_up,
                permuted_input,
                expert_offsets,
                active_expert_ids,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                stage_experts,
                int(layer.sm70_nvfp4_w13_k_dim),
                int(layer.sm70_nvfp4_w13_n_dim),
                int(layer.sm70_nvfp4_group_size),
            )
            if layer.swiglu_limit is None:
                torch.ops._C.silu_and_mul(intermediate, gate_up)
            else:
                torch.ops._C.silu_and_mul_with_clamp(
                    intermediate, gate_up, float(layer.swiglu_limit)
                )
            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                sorted_output,
                intermediate,
                expert_offsets,
                active_expert_ids,
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                stage_experts,
                int(layer.sm70_nvfp4_w2_k_dim),
                int(layer.sm70_nvfp4_w2_n_dim),
                int(layer.sm70_nvfp4_group_size),
            )
            calls += 2
    return calls


def _get_nvfp4_moe_token_counts(
    worker: Worker,
    moe_layers: list[torch.nn.Module],
    base_token_counts: list[int],
) -> list[int]:
    """Include graph-safe and tuned NVFP4 CUDA-graph token widths."""
    if not moe_layers:
        return base_token_counts
    graph_safe_max_tokens = max(
        int(getattr(layer, "sm70_nvfp4_graph_safe_max_tokens", 8))
        for layer in moe_layers
    )
    compact_max_tokens = max(
        int(getattr(layer, "sm70_nvfp4_compact_grouped_max_tokens", 8))
        for layer in moe_layers
    )
    max_top_k = max(int(layer.moe_config.experts_per_token) for layer in moe_layers)
    tuned_max_tokens = max(0, int(envs.VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS)) // max(
        max_top_k, 1
    )
    max_tokens = max(graph_safe_max_tokens, tuned_max_tokens)
    compilation_config = getattr(
        getattr(worker, "vllm_config", None), "compilation_config", None
    )
    capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None) or []
    return sorted(
        set(base_token_counts)
        | set(range(1, compact_max_tokens + 1))
        | {int(size) for size in capture_sizes if 0 < int(size) <= max_tokens}
    )


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
        _silu_and_mul_w13(layer, intermediate, gate_up)
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
            and getattr(layer, "sm70_awq_moe_legacy_single_token_compact", False)
            and hasattr(layer, "w2_strided_ptrs_w_rows")
        ):
            topk_weights = torch.empty(top_k, dtype=torch.float32, device=device)
            topk_weights.fill_(1.0 / max(top_k, 1))
            sorted_weights = torch.empty(top_k, dtype=torch.float32, device=device)
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
            legacy_output.zero_()
            legacy_offsets = torch.empty(top_k + 1, dtype=torch.int32, device=device)
            sm70_ops.awq_moe_single_token_sm70_out(
                legacy_output,
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
                sorted_weights,
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
    if device is None or device.type != "cuda":
        return
    if torch.cuda.get_device_capability(device) != (7, 0):
        return

    model = worker.get_model()
    dense_layers = list(_iter_unique_dense_layers(model))
    fp8_dense_layers = list(_iter_unique_fp8_dense_layers(model))
    fp4_dense_layers = list(_iter_unique_fp4_dense_layers(model))
    moe_layers = list(_iter_unique_moe_layers(model))
    mxfp4_moe_layers = list(_iter_unique_mxfp4_moe_layers(model))
    nvfp4_moe_layers = list(_iter_unique_nvfp4_moe_layers(model))
    if not (
        dense_layers
        or fp8_dense_layers
        or fp4_dense_layers
        or moe_layers
        or mxfp4_moe_layers
        or nvfp4_moe_layers
    ):
        return

    fp4_kinds = {str(state.op_kind) for state in fp4_dense_layers}
    skip_lut_cache = _lut_cache_disabled_for_dynamic_quant_dispatch(
        bool(dense_layers),
        bool(fp8_dense_layers),
        fp4_kinds,
        bool(mxfp4_moe_layers),
        bool(nvfp4_moe_layers),
    )
    imported_records = _load_lut_cache(device, skip_import=skip_lut_cache)
    if imported_records > 0:
        logger.info(
            "Loaded SM70 GEMM LUT (%d records) for device %s.",
            imported_records,
            device,
        )

    m_values = _get_decode_m_values(worker)
    moe_token_counts = _get_moe_token_counts(worker)
    nvfp4_moe_token_counts = _get_nvfp4_moe_token_counts(
        worker, nvfp4_moe_layers, moe_token_counts
    )
    lut_path = _resolve_lut_path(device)

    logger.info(
        "Warming up SM70 TurboMind accepted routes (%d AWQ dense layer "
        "shapes, %d FP8 dense layer shapes, %d FP4 dense layer shapes, "
        "%d MoE layer shapes, %d MXFP4 MoE B1 layer shapes, "
        "%d NVFP4 MoE layer shapes, dense_m=%s, "
        "moe_tokens=%s, nvfp4_moe_tokens=%s, lut_path=%s).",
        len(dense_layers),
        len(fp8_dense_layers),
        len(fp4_dense_layers),
        len(moe_layers),
        len(mxfp4_moe_layers),
        len(nvfp4_moe_layers),
        m_values,
        moe_token_counts,
        nvfp4_moe_token_counts,
        lut_path,
    )
    with torch.inference_mode():
        dense_calls = _warmup_dense_layers(dense_layers, m_values)
        fp8_dense_calls = _warmup_fp8_dense_layers_coordinated(
            fp8_dense_layers,
            m_values,
            device,
        )
        fp4_dense_calls = _warmup_fp4_dense_layers(fp4_dense_layers, m_values)
        moe_stage_calls = _warmup_moe_dense_stage_layers(moe_layers, moe_token_counts)
        single_token_calls = _warmup_moe_single_token_layers(moe_layers)
        mxfp4_moe_stage_calls = _warmup_mxfp4_moe_b1_layers(mxfp4_moe_layers)
        nvfp4_moe_stage_calls = _warmup_nvfp4_moe_decode_layers(
            nvfp4_moe_layers, nvfp4_moe_token_counts
        )
    torch.accelerator.synchronize(device)
    logger.info(
        "SM70 TurboMind warmup finished (%d AWQ dense calls, %d FP8 dense "
        "calls, %d FP4 dense calls, %d MoE stage calls, "
        "%d single-token active-expert calls, %d MXFP4 MoE B1 stage calls, "
        "%d NVFP4 MoE stage calls).",
        dense_calls,
        fp8_dense_calls,
        fp4_dense_calls,
        moe_stage_calls,
        single_token_calls,
        mxfp4_moe_stage_calls,
        nvfp4_moe_stage_calls,
    )
    exported_records = _save_lut_cache(device, skip_export=skip_lut_cache)
    if exported_records > 0:
        logger.info(
            "Saved SM70 GEMM LUT (%d records) for device %s.",
            exported_records,
            device,
        )
