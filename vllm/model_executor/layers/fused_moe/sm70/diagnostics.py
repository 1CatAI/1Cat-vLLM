# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AWQ reference comparisons and historical buffer observations.

These diagnostic oracles deliberately retain their original arithmetic and
stage positions. The production executor never calls the old format module.
"""

import json
import os
from typing import Any

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.config.sm70_moe import capture_sm70_moe_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.sm70_moe_router import Sm70MoeStageRoute

logger = init_logger(__name__)
RoutedExperts = Any


def _silu_and_mul_w13(
    layer: RoutedExperts, out: torch.Tensor, gate_up: torch.Tensor
) -> None:
    if getattr(layer, "sm70_awq_moe_w13_interleaved", False):
        sm70_ops.silu_and_mul_interleaved(out, gate_up)
    else:
        torch.ops._C.silu_and_mul(out, gate_up)


def _parse_layer_id_filter(raw: str | None, env_name: str) -> set[int] | None:
    if raw is None:
        return None
    layer_ids: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(f"{env_name} has invalid layer range: {item}") from exc
            if start < 0 or end < start:
                raise ValueError(f"{env_name} has invalid layer range: {item}")
            layer_ids.update(range(start, end + 1))
            continue
        try:
            layer_id = int(item)
        except ValueError as exc:
            raise ValueError(f"{env_name} has invalid layer id: {item}") from exc
        if layer_id < 0:
            raise ValueError(f"{env_name} has invalid layer id: {item}")
        layer_ids.add(layer_id)
    return layer_ids


def _get_layer_id(layer: RoutedExperts) -> int | None:
    try:
        return int(layer.layer_id)
    except (AttributeError, AssertionError, TypeError, ValueError):
        pass
    layer_name = getattr(layer, "layer_name", "")
    if not layer_name:
        return None
    parts = str(layer_name).split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    ids = []
    for part in parts:
        try:
            ids.append(int(part))
        except ValueError:
            continue
    if len(ids) == 1:
        return ids[0]
    return None


def _diagnostics(layer):
    # Legacy helper users can omit a prepared layer; engines always bind it once.
    policy = getattr(layer, "sm70_moe_diagnostics", None)
    return policy if policy is not None else capture_sm70_moe_config("awq").diagnostics


def _dump_awq_moe_buffer_requested(layer: RoutedExperts, label: str) -> bool:
    policy = _diagnostics(layer)
    if not policy.dump_buffers:
        return False
    if not policy.dump_dir:
        return False

    raw_layer_ids = policy.dump_layers
    if raw_layer_ids.strip().lower() not in {"*", "all"}:
        try:
            layer_ids = _parse_layer_id_filter(
                raw_layer_ids, "VLLM_SM70_DUMP_QWEN_LAYER_IDS"
            )
        except ValueError:
            layer_ids = {0, 1}
        layer_id = _get_layer_id(layer)
        if layer_id is None:
            layer_id = getattr(layer, "sm70_awq_moe_layer_id", None)
        if layer_id is None or layer_id not in (layer_ids or {0, 1}):
            return False

    raw_labels = policy.dump_labels
    labels = {item.strip() for item in raw_labels.split(",") if item.strip()}
    return not labels or label in labels


def _dump_awq_moe_buffer(
    layer: RoutedExperts,
    tensor: torch.Tensor,
    label: str,
) -> torch.Tensor:
    if not _dump_awq_moe_buffer_requested(layer, label):
        return tensor
    layer_id = _get_layer_id(layer)
    if layer_id is None:
        layer_id = getattr(layer, "sm70_awq_moe_layer_id", None)
    if layer_id is None:
        layer_id = -1
    return torch.ops.vllm.sm70_moe_runner_dump(tensor, f"awq_{label}", layer_id)


def _compare_dense_base_enabled(layer: RoutedExperts) -> bool:
    policy = _diagnostics(layer)
    if not policy.compare_dir:
        return False
    enable_file = policy.compare_enable_file
    if enable_file and not os.path.exists(enable_file):
        return False
    raw_layer_ids = policy.compare_layers
    if raw_layer_ids is not None and raw_layer_ids.strip().lower() in {"*", "all"}:
        return True
    layer_ids = _parse_layer_id_filter(
        raw_layer_ids,
        "VLLM_SM70_AWQ_MOE_COMPARE_DENSE_LAYER_IDS",
    )
    if layer_ids is None:
        return True
    layer_id = _get_layer_id(layer)
    if layer_id is None:
        layer_id = getattr(layer, "sm70_awq_moe_layer_id", None)
    return layer_id is not None and layer_id in layer_ids


def _compare_dense_decode_step(layer: RoutedExperts) -> int | None:
    policy = _diagnostics(layer)
    if not _compare_dense_base_enabled(layer):
        return None
    step = int(getattr(layer, "_awq_moe_compare_dense_decode_step", 0))
    layer._awq_moe_compare_dense_decode_step = step + 1
    steps = _parse_layer_id_filter(
        policy.compare_steps,
        "VLLM_SM70_AWQ_MOE_COMPARE_DENSE_STEPS",
    )
    if steps is not None and step not in steps:
        return None
    reports = int(getattr(layer, "_awq_moe_compare_dense_reports", 0))
    max_reports = policy.compare_max_reports
    if max_reports > 0 and reports >= max_reports:
        return None
    layer._awq_moe_compare_dense_reports = reports + 1
    return step


def _diff_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    diff = (left - right).abs()
    if diff.numel() == 0:
        return {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "left_abs_max": 0.0,
            "right_abs_max": 0.0,
            "max_index": -1,
        }
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.float().mean().item()),
        "left_abs_max": float(left.abs().max().item()),
        "right_abs_max": float(right.abs().max().item()),
        "max_index": int(diff.argmax().item()),
    }


def _write_compare_dense_record(record: dict[str, object], policy=None) -> None:
    policy = (
        policy if policy is not None else capture_sm70_moe_config("awq").diagnostics
    )
    out_dir = policy.compare_dir
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    device = (
        torch.accelerator.current_device_index() if torch.cuda.is_available() else "cpu"
    )
    path = os.path.join(
        out_dir,
        f"awq_moe_dense_compare_pid{os.getpid()}_cuda{device}.jsonl",
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _expert_offset_ranges(offsets: torch.Tensor) -> list[tuple[int, int, int]]:
    values = offsets.detach().cpu().tolist()
    return [
        (expert, int(start), int(end))
        for expert, (start, end) in enumerate(zip(values, values[1:]))
        if start != end
    ]


class AwqStageObserver:
    def __init__(
        self,
        *,
        layer,
        x,
        topk_weights,
        ids,
        plan,
        group_size,
        policy,
        indexed_w13,
        active_grouped,
        compare_step,
        weighted_reduce,
        operators=sm70_ops,
    ):
        self.layer = layer
        self.operators = operators
        self.x = x
        self.topk_weights = topk_weights
        self.topk_ids_i32 = ids
        self.top_k = ids.shape[1]
        self.num_tokens = x.shape[0]
        self.total_slots = ids.numel()
        self.route_plan = plan
        self.group_size = group_size
        self.sm70_moe_policy = policy
        self.weighted_reduce = weighted_reduce
        self.indexed_w13 = indexed_w13
        self.use_active_exact_small_batched_moe = active_grouped
        self.use_batched_active_exact_w2 = plan.use_batched_active_exact_w2
        self.compare_dense_step = compare_step
        self.compare_dense_w13_stats = None
        self.compare_dense_w2_stats = None
        self.compare_dense_full_w2_stats = None
        self.compare_dense_full_output = None
        self.compare_strict_output = None
        self.compare_strict_stats = None
        self.compare_route_state = None
        self.dense_gate_up = None

    def after_w13(self, buffers):
        if self.indexed_w13:
            pass
        elif self.use_active_exact_small_batched_moe:
            if self.compare_dense_step is not None:
                self.dense_gate_up = torch.empty_like(buffers["gate_up"])
                self.operators.awq_moe_dense_stage_sm70_out(
                    self.dense_gate_up,
                    buffers["permuted_input"],
                    buffers["expert_offsets"],
                    self.layer._awq_moe_buf_dense_expert_ids,
                    self.layer.w13_strided_ptrs_w,
                    self.layer.w13_strided_ptrs_s,
                    self.layer.sm70_num_experts,
                    self.layer.sm70_w13_k_dim,
                    self.layer.sm70_w13_n_dim,
                    self.group_size,
                )
                self.compare_dense_w13_stats = _diff_stats(
                    buffers["gate_up"], self.dense_gate_up
                )
        elif self.route_plan.w13 == Sm70MoeStageRoute.PER_EXPERT_DISPATCH:
            buffers["gate_up"] = _dump_awq_moe_buffer(
                self.layer, buffers["gate_up"], "w13_batched_out"
            )
            if self.compare_dense_step is not None:
                self.dense_gate_up = torch.empty_like(buffers["gate_up"])
                self.operators.awq_moe_dense_stage_sm70_out(
                    self.dense_gate_up,
                    buffers["permuted_input"],
                    buffers["expert_offsets"],
                    self.layer._awq_moe_buf_dense_expert_ids,
                    self.layer.w13_strided_ptrs_w,
                    self.layer.w13_strided_ptrs_s,
                    self.layer.sm70_num_experts,
                    self.layer.sm70_w13_k_dim,
                    self.layer.sm70_w13_n_dim,
                    self.group_size,
                )
                self.compare_dense_w13_stats = _diff_stats(
                    buffers["gate_up"], self.dense_gate_up
                )
        else:
            buffers["gate_up"] = _dump_awq_moe_buffer(
                self.layer, buffers["gate_up"], "w13_dense_out"
            )

    def after_activation(self, buffers):
        buffers["intermediate"] = _dump_awq_moe_buffer(
            self.layer, buffers["intermediate"], "silu_out"
        )

    def after_w2(self, buffers):
        output = buffers["output"]
        if self.use_active_exact_small_batched_moe or self.use_batched_active_exact_w2:
            if self.compare_dense_step is not None:
                dense_sorted_output = torch.empty_like(buffers["sorted_output"])
                self.operators.awq_moe_dense_stage_sm70_out(
                    dense_sorted_output,
                    buffers["intermediate"],
                    buffers["expert_offsets"],
                    self.layer._awq_moe_buf_dense_expert_ids,
                    self.layer.w2_strided_ptrs_w,
                    self.layer.w2_strided_ptrs_s,
                    self.layer.sm70_num_experts,
                    self.layer.sm70_w2_k_dim,
                    self.layer.sm70_w2_n_dim,
                    self.group_size,
                )
                self.compare_dense_w2_stats = _diff_stats(
                    buffers["sorted_output"], dense_sorted_output
                )
                self.compare_dense_full_output = torch.empty_like(output)
                self.compare_dense_full_output.zero_()
                dense_full_sorted_output = dense_sorted_output
                if self.dense_gate_up is not None:
                    dense_intermediate = torch.empty_like(buffers["intermediate"])
                    dense_full_sorted_output = torch.empty_like(
                        buffers["sorted_output"]
                    )
                    _silu_and_mul_w13(
                        self.layer, dense_intermediate, self.dense_gate_up
                    )
                    self.operators.awq_moe_dense_stage_sm70_out(
                        dense_full_sorted_output,
                        dense_intermediate,
                        buffers["expert_offsets"],
                        self.layer._awq_moe_buf_dense_expert_ids,
                        self.layer.w2_strided_ptrs_w,
                        self.layer.w2_strided_ptrs_s,
                        self.layer.sm70_num_experts,
                        self.layer.sm70_w2_k_dim,
                        self.layer.sm70_w2_n_dim,
                        self.group_size,
                    )
                    self.compare_dense_full_w2_stats = _diff_stats(
                        buffers["sorted_output"], dense_full_sorted_output
                    )
                dense_full_sorted_output_logical = dense_full_sorted_output[
                    :, : self.layer.sm70_hidden_logical_size
                ]
                self.operators.moe_unpermute(
                    dense_full_sorted_output_logical,
                    self.topk_weights,
                    buffers["inv_permuted_idx"],
                    buffers["expert_offsets64"],
                    self.top_k,
                    self.compare_dense_full_output,
                )
                self.compare_route_state = {
                    "expert_ranges": _expert_offset_ranges(buffers["expert_offsets"]),
                    "active_expert_offsets": buffers["active_expert_offsets"]
                    .detach()
                    .cpu()
                    .tolist(),
                    "active_expert_ids": buffers["sorted_expert_ids"]
                    .detach()
                    .cpu()
                    .tolist(),
                    "permuted_experts_id": buffers["permuted_experts_id"]
                    .detach()
                    .cpu()
                    .tolist(),
                }
        elif self.route_plan.w2 == Sm70MoeStageRoute.PER_EXPERT_DISPATCH:
            buffers["sorted_output"] = _dump_awq_moe_buffer(
                self.layer, buffers["sorted_output"], "w2_batched_out"
            )
            if self.compare_dense_step is not None:
                dense_sorted_output = torch.empty_like(buffers["sorted_output"])
                self.operators.awq_moe_dense_stage_sm70_out(
                    dense_sorted_output,
                    buffers["intermediate"],
                    buffers["expert_offsets"],
                    self.layer._awq_moe_buf_dense_expert_ids,
                    self.layer.w2_strided_ptrs_w,
                    self.layer.w2_strided_ptrs_s,
                    self.layer.sm70_num_experts,
                    self.layer.sm70_w2_k_dim,
                    self.layer.sm70_w2_n_dim,
                    self.group_size,
                )
                self.compare_dense_w2_stats = _diff_stats(
                    buffers["sorted_output"], dense_sorted_output
                )
                dense_intermediate = torch.empty_like(buffers["intermediate"])
                dense_full_sorted_output = torch.empty_like(buffers["sorted_output"])
                self.compare_dense_full_output = torch.empty_like(output)
                self.compare_dense_full_output.zero_()
                _silu_and_mul_w13(self.layer, dense_intermediate, self.dense_gate_up)
                self.operators.awq_moe_dense_stage_sm70_out(
                    dense_full_sorted_output,
                    dense_intermediate,
                    buffers["expert_offsets"],
                    self.layer._awq_moe_buf_dense_expert_ids,
                    self.layer.w2_strided_ptrs_w,
                    self.layer.w2_strided_ptrs_s,
                    self.layer.sm70_num_experts,
                    self.layer.sm70_w2_k_dim,
                    self.layer.sm70_w2_n_dim,
                    self.group_size,
                )
                self.compare_dense_full_w2_stats = _diff_stats(
                    buffers["sorted_output"], dense_full_sorted_output
                )
                dense_full_sorted_output_logical = dense_full_sorted_output[
                    :, : self.layer.sm70_hidden_logical_size
                ]
                self.operators.moe_unpermute(
                    dense_full_sorted_output_logical,
                    self.topk_weights,
                    buffers["inv_permuted_idx"],
                    buffers["expert_offsets64"],
                    self.top_k,
                    self.compare_dense_full_output,
                )
                if self.num_tokens == 1:
                    strict_gate_up = torch.empty_like(buffers["gate_up"])
                    strict_compact_input = torch.empty_like(buffers["permuted_input"])
                    strict_intermediate = torch.empty_like(buffers["intermediate"])
                    strict_sorted_output = torch.empty_like(buffers["sorted_output"])
                    strict_expert_offsets = torch.empty_like(buffers["expert_offsets"])
                    strict_expert_offsets64 = torch.empty_like(
                        buffers["expert_offsets64"]
                    )
                    strict_inv_permuted_idx = torch.empty_like(
                        buffers["inv_permuted_idx"]
                    )
                    strict_sorted_expert_ids = torch.empty_like(
                        buffers["sorted_expert_ids"]
                    )
                    self.compare_strict_output = torch.empty_like(output)
                    self.compare_strict_output.zero_()
                    self.operators.awq_moe_single_token_dense_w13_sm70_out(
                        strict_gate_up,
                        strict_compact_input,
                        self.x,
                        self.topk_ids_i32,
                        self.layer.w13_strided_ptrs_w,
                        self.layer.w13_strided_ptrs_s,
                        strict_expert_offsets,
                        strict_expert_offsets64,
                        strict_inv_permuted_idx,
                        strict_sorted_expert_ids,
                        self.layer.sm70_w13_k_dim,
                        self.layer.sm70_w13_n_dim,
                        self.group_size,
                        self.layer.sm70_hidden_logical_size,
                    )
                    _silu_and_mul_w13(self.layer, strict_intermediate, strict_gate_up)
                    self.operators.awq_moe_single_token_dense_stage_sm70_out(
                        strict_sorted_output,
                        strict_intermediate,
                        strict_expert_offsets,
                        strict_sorted_expert_ids,
                        self.layer.w2_strided_ptrs_w,
                        self.layer.w2_strided_ptrs_s,
                        self.top_k,
                        self.layer.sm70_w2_k_dim,
                        self.layer.sm70_w2_n_dim,
                        self.group_size,
                    )
                    strict_sorted_output_logical = strict_sorted_output[
                        :, : self.layer.sm70_hidden_logical_size
                    ]
                    if self.weighted_reduce:
                        self.operators.awq_moe_single_token_weighted_reduce_out(
                            strict_sorted_output_logical,
                            self.topk_weights,
                            strict_inv_permuted_idx,
                            self.compare_strict_output,
                            self.top_k,
                            self.layer.sm70_hidden_logical_size,
                        )
                    else:
                        self.operators.moe_unpermute(
                            strict_sorted_output_logical,
                            self.topk_weights,
                            strict_inv_permuted_idx,
                            strict_expert_offsets64[: self.top_k + 1],
                            self.top_k,
                            self.compare_strict_output,
                        )
                    self.compare_strict_stats = {
                        "w13_batched_vs_strict": _diff_stats(
                            buffers["gate_up"], strict_gate_up
                        ),
                        "silu_batched_vs_strict": _diff_stats(
                            buffers["intermediate"], strict_intermediate
                        ),
                        "w2_batched_vs_strict": _diff_stats(
                            buffers["sorted_output"], strict_sorted_output
                        ),
                        "batched_inv_permuted_idx": buffers["inv_permuted_idx"]
                        .detach()
                        .cpu()
                        .tolist(),
                        "strict_inv_permuted_idx": strict_inv_permuted_idx.detach()
                        .cpu()
                        .tolist(),
                        "strict_sorted_expert_ids": strict_sorted_expert_ids.detach()
                        .cpu()
                        .tolist(),
                        "strict_expert_offsets_prefix": strict_expert_offsets[
                            : self.top_k + 1
                        ]
                        .detach()
                        .cpu()
                        .tolist(),
                    }
        else:
            buffers["sorted_output"] = _dump_awq_moe_buffer(
                self.layer, buffers["sorted_output"], "w2_dense_out"
            )

    def finish(self, output, *, chunked=False):
        if chunked:
            return _dump_awq_moe_buffer(self.layer, output, "chunked_w2_output")
        if self.compare_dense_step is not None:
            record = {
                "decode_step": self.compare_dense_step,
                "device": int(torch.accelerator.current_device_index())
                if torch.cuda.is_available()
                else None,
                "layer_id": getattr(self.layer, "sm70_awq_moe_layer_id", None),
                "layer_name": str(getattr(self.layer, "layer_name", "")),
                "num_tokens": int(self.num_tokens),
                "pid": int(os.getpid()),
                "topk_ids": self.topk_ids_i32.detach().cpu().tolist(),
                "topk_weights": self.topk_weights.detach().float().cpu().tolist(),
                "w13_batched_vs_dense": self.compare_dense_w13_stats,
                "w2_batched_vs_dense_same_intermediate": self.compare_dense_w2_stats,
                "w2_batched_vs_dense_full_pipeline": self.compare_dense_full_w2_stats,
                "route_state": self.compare_route_state,
                "strict_reference": self.compare_strict_stats,
            }
            if self.compare_dense_full_output is not None:
                record["output_batched_vs_dense_full_pipeline"] = _diff_stats(
                    output, self.compare_dense_full_output
                )
            if self.compare_strict_output is not None:
                record["output_batched_vs_strict"] = _diff_stats(
                    output, self.compare_strict_output
                )
            _write_compare_dense_record(record, _diagnostics(self.layer))
        return _dump_awq_moe_buffer(self.layer, output, "output")


def fp8_stage_reference(codec, layer, x, weights, ids, buffers, group_size, *, compact):
    """Historical FP8 reference layouts, using the production stage executor."""
    from dataclasses import replace

    from vllm.model_executor.layers.fused_moe.sm70.stages import execute_routed
    from vllm.model_executor.layers.quantization.sm70_moe_router import (
        select_sm70_quantized_moe_route,
    )

    names = (
        "permuted_input",
        "gate_up",
        "intermediate",
        "sorted_output",
        "output",
        "expert_offsets",
        "expert_offsets64",
        "inv_permuted_idx",
    )
    ref = {name: torch.empty_like(buffers[name]) for name in names}
    ref["output"].zero_()
    top_k = weights.shape[1]
    if compact:
        codec.operators.awq_moe_single_token_exact_layout_prepare(
            ids,
            x,
            ref["permuted_input"],
            ref["expert_offsets"],
            ref["expert_offsets64"],
            ref["inv_permuted_idx"],
            layer.sm70_num_experts,
        )
        plan = select_sm70_quantized_moe_route(
            batched_enabled=True, num_tokens=1, total_slots=top_k
        )
    else:
        permuted_idx = torch.empty_like(buffers["permuted_idx"])
        args = (
            x,
            ids,
            buffers["token_expert_indices"],
            layer.expert_map,
            layer.global_num_experts,
            layer.local_num_experts,
            top_k,
            ref["permuted_input"],
            ref["expert_offsets64"],
            ref["inv_permuted_idx"],
            permuted_idx,
        )
        if layer.sm70_fp8_moe_permute_with_scratch:
            permuted_idx.fill_(x.shape[0] * top_k)
            codec.operators.moe_permute_with_scratch(
                *args,
                *(
                    torch.empty_like(buffers[name])
                    for name in (
                        "sort_workspace",
                        "permuted_experts_id",
                        "sorted_row_idx",
                        "topk_ids_for_sort",
                    )
                ),
            )
        else:
            codec.operators.moe_permute(*args)
        ref["expert_offsets"].copy_(ref["expert_offsets64"], non_blocking=True)
        plan = select_sm70_quantized_moe_route(
            batched_enabled=layer.sm70_fp8_moe_batched_gemm,
            num_tokens=x.shape[0],
            total_slots=x.shape[0] * top_k,
            w13_per_expert_dispatch=layer.sm70_fp8_moe_batched_w13_per_expert_dispatch,
            w2_per_expert_dispatch=layer.sm70_fp8_moe_batched_w2_per_expert_dispatch,
        )
    execute_routed(
        replace(codec, diagnostic=True),
        plan,
        layer,
        ref,
        x,
        weights,
        group_size,
        layer._fp8_buf_dense_expert_ids,
    )
    return ref


def report_stage_compare(
    logger,
    layer_name,
    prefix,
    report_index,
    reference_tensors,
    actual_tensors,
    topk_ids_i32,
):
    """Common comparison observer, preserving the historical FP8 fields."""

    def _max_diff(name: str) -> float:
        actual = actual_tensors[name]
        expected = reference_tensors[name]
        return float((actual - expected).abs().max().item())

    logger.warning(
        "SM70 FP8 %s compare: layer=%s report=%d "
        "perm=%g off_eq=%s off64_eq=%s inv_eq=%s "
        "w13=%g silu=%g w2=%g out=%g topk_ids=%s",
        prefix,
        layer_name,
        report_index,
        _max_diff("permuted_input"),
        torch.equal(
            actual_tensors["expert_offsets"],
            reference_tensors["expert_offsets"],
        ),
        torch.equal(
            actual_tensors["expert_offsets64"],
            reference_tensors["expert_offsets64"],
        ),
        torch.equal(
            actual_tensors["inv_permuted_idx"],
            reference_tensors["inv_permuted_idx"],
        ),
        _max_diff("gate_up"),
        _max_diff("intermediate"),
        _max_diff("sorted_output"),
        _max_diff("output"),
        topk_ids_i32.detach().cpu().view(-1).tolist(),
    )
