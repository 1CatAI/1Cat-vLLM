# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 AWQ MoE method backed by TurboMind GEMM kernels."""

import json
import os

import torch
from torch.nn import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    Sm70MoeStageRoute,
    select_sm70_quantized_moe_route,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

_DEFAULT_PERSISTENT_MAX_TOKENS = 32


def _log_runtime_route_once(message: str, *args) -> None:
    if torch.compiler.is_compiling():
        return
    logger.info_once(message, *args)


def _use_temporary_buffers_for_dummy_or_capture() -> bool:
    if is_forward_context_available() and get_forward_context().is_dummy_run:
        return True
    # CUDA graph replay is address-fixed. Use the per-layer persistent buffers
    # during capture too, so the captured indexed MoE scratch/output lifetimes
    # do not depend on graph-pool temporary allocation analysis.
    return False


def _single_token_weighted_reduce_enabled() -> bool:
    if not (
        envs.VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH
        or envs.VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH
    ):
        return False
    return hasattr(torch.ops._C, "awq_moe_single_token_weighted_reduce_out")


def _single_token_indexed_w13_enabled() -> bool:
    if not (
        envs.VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH
        or envs.VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W13_FASTPATH
    ):
        return False
    return hasattr(torch.ops._C, "awq_moe_single_token_indexed_dense_w13_sm70_out")


def _single_token_compact_w13_enabled() -> bool:
    if not envs.VLLM_SM70_MOE_SINGLE_TOKEN_COMPACT_W13_FASTPATH:
        return False
    return hasattr(torch.ops._C, "awq_moe_single_token_compact_dense_w13_sm70_out")


def _single_token_indexed_w2_enabled() -> bool:
    if not (
        envs.VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH
        or envs.VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH
    ):
        return False
    return hasattr(torch.ops._C, "awq_moe_single_token_indexed_dense_stage_sm70_out")


def _legacy_single_token_compact_enabled() -> bool:
    if not envs.VLLM_SM70_AWQ_MOE_LEGACY_SINGLE_TOKEN_COMPACT:
        return False
    return hasattr(torch.ops._C, "awq_moe_single_token_sm70_out")


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


def _dump_awq_moe_buffer_requested(layer: RoutedExperts, label: str) -> bool:
    if os.getenv("VLLM_SM70_DUMP_AWQ_MOE_BUFFERS") != "1":
        return False
    if not os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_DIR"):
        return False

    raw_layer_ids = os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_IDS", "0,1")
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

    raw_labels = os.getenv("VLLM_SM70_DUMP_AWQ_MOE_LABELS", "")
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


def _batched_gemm_enabled_for_layer(layer: RoutedExperts, default: bool) -> bool:
    if not default:
        return False
    allowlist = _parse_layer_id_filter(
        envs.VLLM_SM70_AWQ_MOE_BATCHED_LAYER_ALLOWLIST,
        "VLLM_SM70_AWQ_MOE_BATCHED_LAYER_ALLOWLIST",
    )
    denylist = _parse_layer_id_filter(
        envs.VLLM_SM70_AWQ_MOE_BATCHED_LAYER_DENYLIST,
        "VLLM_SM70_AWQ_MOE_BATCHED_LAYER_DENYLIST",
    )
    if allowlist is None and denylist is None:
        return True

    layer_id = _get_layer_id(layer)
    if layer_id is None:
        logger.warning_once(
            "SM70 AWQ MoE batched layer filter is set, but layer id could "
            "not be extracted from %r; keeping batched path enabled.",
            getattr(layer, "layer_name", None),
        )
        return True
    if allowlist is not None and layer_id not in allowlist:
        return False
    return not (denylist is not None and layer_id in denylist)


def _compare_dense_base_enabled(layer: RoutedExperts) -> bool:
    if not envs.VLLM_SM70_AWQ_MOE_COMPARE_DENSE_DIR:
        return False
    enable_file = envs.VLLM_SM70_AWQ_MOE_COMPARE_DENSE_ENABLE_FILE
    if enable_file and not os.path.exists(enable_file):
        return False
    raw_layer_ids = envs.VLLM_SM70_AWQ_MOE_COMPARE_DENSE_LAYER_IDS
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
    if not _compare_dense_base_enabled(layer):
        return None
    step = int(getattr(layer, "_awq_moe_compare_dense_decode_step", 0))
    layer._awq_moe_compare_dense_decode_step = step + 1
    steps = _parse_layer_id_filter(
        envs.VLLM_SM70_AWQ_MOE_COMPARE_DENSE_STEPS,
        "VLLM_SM70_AWQ_MOE_COMPARE_DENSE_STEPS",
    )
    if steps is not None and step not in steps:
        return None
    reports = int(getattr(layer, "_awq_moe_compare_dense_reports", 0))
    max_reports = envs.VLLM_SM70_AWQ_MOE_COMPARE_DENSE_MAX_REPORTS
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


def _write_compare_dense_record(record: dict[str, object]) -> None:
    out_dir = envs.VLLM_SM70_AWQ_MOE_COMPARE_DENSE_DIR
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
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


def _round_up(value: int, align: int) -> int:
    if align <= 0:
        return value
    return ((value + align - 1) // align) * align


def _pad_last_dim(tensor: torch.Tensor, pad_elems: int) -> torch.Tensor:
    if pad_elems <= 0:
        return tensor
    pad = torch.zeros(
        (*tensor.shape[:-1], pad_elems),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, pad), dim=-1)


def _pad_penultimate_dim(tensor: torch.Tensor, pad_elems: int) -> torch.Tensor:
    if pad_elems <= 0:
        return tensor
    pad = torch.zeros(
        (*tensor.shape[:-2], pad_elems, tensor.shape[-1]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, pad), dim=-2)


def _set_parameter(
    layer: torch.nn.Module,
    name: str,
    value: torch.Tensor,
) -> None:
    param = value if isinstance(value, Parameter) else Parameter(value)
    param.requires_grad_(False)
    setattr(layer, name, param)


def _align_awq_output_dim(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    pack_factor: int,
    align: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    old_n = int(qweight.shape[-1]) * pack_factor
    new_n = _round_up(old_n, align)
    if new_n == old_n:
        return qweight, scales, qzeros, old_n
    pad_n = new_n - old_n
    if pad_n % pack_factor != 0:
        raise ValueError("SM70 AWQ MoE output padding must preserve pack factor.")
    qweight = _pad_last_dim(qweight, pad_n // pack_factor)
    qzeros = _pad_last_dim(qzeros, pad_n // pack_factor)
    scales = _pad_last_dim(scales, pad_n)
    return qweight, scales, qzeros, new_n


def _align_awq_input_dim(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    align: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    old_k = int(qweight.shape[-2])
    new_k = _round_up(old_k, align)
    if new_k == old_k:
        return qweight, scales, qzeros, old_k
    if new_k % group_size != 0:
        raise ValueError("SM70 AWQ MoE input padding must preserve groups.")
    old_groups = int(scales.shape[-2])
    new_groups = new_k // group_size
    qweight = _pad_penultimate_dim(qweight, new_k - old_k)
    qzeros = _pad_penultimate_dim(qzeros, new_groups - old_groups)
    scales = _pad_penultimate_dim(scales, new_groups - old_groups)
    return qweight, scales, qzeros, new_k


class AWQSM70MoEMethod(FusedMoEMethodBase):
    """SM70 AWQ MoE path backed by TurboMind kernels.

    The source default matches the 0.0.3 V100 throughput baseline and uses the
    grouped/batched MoE GEMM. Set VLLM_SM70_AWQ_MOE_BATCHED_GEMM=0 to force the
    per-expert dense TurboMind bridge for strict exactness diagnostics.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        layer: RoutedExperts,
    ) -> None:
        super().__init__(layer.moe_config)
        if weight_bits != 4:
            raise ValueError(
                f"AWQSM70MoEMethod only supports 4-bit, got {weight_bits}."
            )
        if group_size not in (32, 64, 128):
            raise ValueError(
                "AWQSM70MoEMethod supports group_size=32/64/128, "
                f"got {group_size}."
            )
        if not zero_point:
            raise ValueError("AWQSM70MoEMethod currently requires AWQ zero points.")
        if self.moe.has_bias:
            raise NotImplementedError("SM70 AWQ MoE does not support bias yet.")
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.pack_factor = 32 // weight_bits
        self.use_batched_gemm = envs.VLLM_SM70_AWQ_MOE_BATCHED_GEMM

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )
        extra_weight_attrs.pop("intermediate_size_full", None)

        w13_qweight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size

        w13_scales = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        align = self.group_size
        hidden_logical_size = int(layer.w13_qweight.shape[1])
        w13_logical_out = int(layer.w13_scales.shape[-1])
        intermediate_logical_size = w13_logical_out // 2
        batched_gemm = _batched_gemm_enabled_for_layer(layer, self.use_batched_gemm)

        w13_qweight, w13_scales, w13_qzeros, w13_aligned_out = (
            _align_awq_output_dim(
                layer.w13_qweight,
                layer.w13_scales,
                layer.w13_qzeros,
                self.pack_factor,
                align * 2,
            )
        )
        _set_parameter(layer, "w13_qweight", w13_qweight)
        _set_parameter(layer, "w13_scales", w13_scales)
        _set_parameter(layer, "w13_qzeros", w13_qzeros)
        aligned_intermediate_size = w13_aligned_out // 2

        w2_qweight, w2_scales, w2_qzeros, _ = _align_awq_input_dim(
            layer.w2_qweight,
            layer.w2_scales,
            layer.w2_qzeros,
            self.group_size,
            align,
        )
        w2_qweight, w2_scales, w2_qzeros, hidden_aligned_size = (
            _align_awq_output_dim(
                w2_qweight,
                w2_scales,
                w2_qzeros,
                self.pack_factor,
                align,
            )
        )
        _set_parameter(layer, "w2_qweight", w2_qweight)
        _set_parameter(layer, "w2_scales", w2_scales)
        _set_parameter(layer, "w2_qzeros", w2_qzeros)

        layer.sm70_hidden_logical_size = hidden_logical_size
        layer.sm70_hidden_aligned_size = hidden_aligned_size
        layer.sm70_intermediate_logical_size = intermediate_logical_size
        layer.sm70_intermediate_aligned_size = aligned_intermediate_size
        if (
            aligned_intermediate_size != intermediate_logical_size
            or hidden_aligned_size != hidden_logical_size
        ):
            logger.info_once(
                "SM70 AWQ MoE alignment hidden=%d->%d inter=%d->%d",
                hidden_logical_size,
                hidden_aligned_size,
                intermediate_logical_size,
                aligned_intermediate_size,
            )

        num_experts = int(layer.w13_qweight.shape[0])
        w13_tm_weights, w13_tm_scales, w13_meta = [], [], []
        w13_legacy_tm_weights, w13_legacy_tm_scales, w13_legacy_meta = [], [], []
        w2_tm_weights, w2_tm_scales, w2_meta = [], [], []
        build_legacy_w13 = (
            batched_gemm
            and envs.VLLM_SM70_AWQ_MOE_LEGACY_SINGLE_TOKEN_COMPACT
            and hasattr(torch.ops._C, "awq_moe_single_token_sm70_out")
        )
        for expert_id in range(num_experts):
            r13 = sm70_ops.awq_sm70_prepare(
                layer.w13_qweight[expert_id],
                layer.w13_scales[expert_id],
                layer.w13_qzeros[expert_id],
                self.group_size,
                False,
            )
            w13_tm_weights.append(r13[0])
            w13_tm_scales.append(r13[1])
            w13_meta.append(r13[2])
            if build_legacy_w13:
                r13_legacy = sm70_ops.awq_sm70_prepare(
                    layer.w13_qweight[expert_id],
                    layer.w13_scales[expert_id],
                    layer.w13_qzeros[expert_id],
                    self.group_size,
                    True,
                )
                w13_legacy_tm_weights.append(r13_legacy[0])
                w13_legacy_tm_scales.append(r13_legacy[1])
                w13_legacy_meta.append(r13_legacy[2])

            r2 = sm70_ops.awq_sm70_prepare(
                layer.w2_qweight[expert_id],
                layer.w2_scales[expert_id],
                layer.w2_qzeros[expert_id],
                self.group_size,
                False,
            )
            w2_tm_weights.append(r2[0])
            w2_tm_scales.append(r2[1])
            w2_meta.append(r2[2])

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False
        )
        layer.w13_tm_scales = Parameter(
            torch.stack(w13_tm_scales), requires_grad=False
        )
        if build_legacy_w13:
            layer.w13_legacy_tm_weight = Parameter(
                torch.stack(w13_legacy_tm_weights), requires_grad=False
            )
            layer.w13_legacy_tm_scales = Parameter(
                torch.stack(w13_legacy_tm_scales), requires_grad=False
            )
        layer.w2_tm_weight = Parameter(torch.stack(w2_tm_weights), requires_grad=False)
        layer.w2_tm_scales = Parameter(torch.stack(w2_tm_scales), requires_grad=False)

        w13_k_ld, w13_q_ld = int(w13_meta[0][0].item()), int(w13_meta[0][1].item())
        if build_legacy_w13:
            w13_legacy_k_ld = int(w13_legacy_meta[0][0].item())
            w13_legacy_q_ld = int(w13_legacy_meta[0][1].item())
        w2_k_ld, w2_q_ld = int(w2_meta[0][0].item()), int(w2_meta[0][1].item())
        w13_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w13_tm_weight,
            layer.w13_tm_scales,
            w13_k_ld,
            w13_q_ld,
            num_experts,
        )
        if build_legacy_w13:
            w13_legacy_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
                layer.w13_legacy_tm_weight,
                layer.w13_legacy_tm_scales,
                w13_legacy_k_ld,
                w13_legacy_q_ld,
                num_experts,
            )
        w2_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w2_tm_weight,
            layer.w2_tm_scales,
            w2_k_ld,
            w2_q_ld,
            num_experts,
        )
        layer.w13_strided_ptrs_w = Parameter(w13_ptrs[0], requires_grad=False)
        layer.w13_strided_ptrs_s = Parameter(w13_ptrs[1], requires_grad=False)
        if build_legacy_w13:
            layer.w13_legacy_strided_ptrs_w = Parameter(
                w13_legacy_ptrs[0], requires_grad=False
            )
            layer.w13_legacy_strided_ptrs_s = Parameter(
                w13_legacy_ptrs[1], requires_grad=False
            )
        layer.w2_strided_ptrs_w = Parameter(w2_ptrs[0], requires_grad=False)
        layer.w2_strided_ptrs_s = Parameter(w2_ptrs[1], requires_grad=False)
        ptr_row_bytes = int(layer.w13_strided_ptrs_w.numel() // num_experts)
        layer.sm70_ptr_row_bytes = ptr_row_bytes
        layer.w13_strided_ptrs_w_rows = layer.w13_strided_ptrs_w.view(
            num_experts, ptr_row_bytes
        )
        layer.w13_strided_ptrs_s_rows = layer.w13_strided_ptrs_s.view(
            num_experts, ptr_row_bytes
        )
        if build_legacy_w13:
            layer.w13_legacy_strided_ptrs_w_rows = (
                layer.w13_legacy_strided_ptrs_w.view(num_experts, ptr_row_bytes)
            )
            layer.w13_legacy_strided_ptrs_s_rows = (
                layer.w13_legacy_strided_ptrs_s.view(num_experts, ptr_row_bytes)
            )
        layer.w2_strided_ptrs_w_rows = layer.w2_strided_ptrs_w.view(
            num_experts, ptr_row_bytes
        )
        layer.w2_strided_ptrs_s_rows = layer.w2_strided_ptrs_s.view(
            num_experts, ptr_row_bytes
        )

        layer.sm70_num_experts = num_experts
        layer.sm70_w13_k_dim = int(layer.w13_tm_weight.shape[1])
        layer.sm70_w13_n_dim = int(layer.w13_tm_weight.shape[2]) * self.pack_factor
        layer.sm70_w2_k_dim = int(layer.w2_tm_weight.shape[1])
        layer.sm70_w2_n_dim = int(layer.w2_tm_weight.shape[2]) * self.pack_factor
        layer.sm70_w13_k_ld = w13_k_ld
        layer.sm70_w13_q_ld = w13_q_ld
        layer.sm70_w2_k_ld = w2_k_ld
        layer.sm70_w2_q_ld = w2_q_ld
        layer.sm70_intermediate_size = layer.sm70_w2_k_dim
        layer.sm70_awq_moe_batched_gemm = batched_gemm
        layer.sm70_awq_moe_layer_id = _get_layer_id(layer)

        self._allocate_buffers(layer)
        del layer.w13_qweight, layer.w13_scales, layer.w13_qzeros
        del layer.w2_qweight, layer.w2_scales, layer.w2_qzeros
        if (
            envs.VLLM_SM70_AWQ_MOE_BATCHED_LAYER_ALLOWLIST is not None
            or envs.VLLM_SM70_AWQ_MOE_BATCHED_LAYER_DENYLIST is not None
        ):
            logger.info_once(
                "SM70 AWQ MoE batched layer filter active allow=%r deny=%r.",
                envs.VLLM_SM70_AWQ_MOE_BATCHED_LAYER_ALLOWLIST,
                envs.VLLM_SM70_AWQ_MOE_BATCHED_LAYER_DENYLIST,
            )
        logger.info_once(
            "SM70 AWQ MoE TurboMind %s path enabled (%d experts).",
            "batched" if batched_gemm else "per-expert dense",
            num_experts,
        )

    def _allocate_buffers(self, layer: RoutedExperts) -> None:
        device = layer.w13_tm_weight.device
        top_k = self.moe.experts_per_token
        persistent_tokens = _DEFAULT_PERSISTENT_MAX_TOKENS
        max_slots = persistent_tokens * top_k
        layer._awq_moe_buf_max_tokens = persistent_tokens
        layer._awq_moe_buf_max_slots = max_slots
        layer._awq_moe_buf_top_k = top_k
        layer._awq_moe_buf_output = torch.empty(
            persistent_tokens,
            layer.sm70_hidden_logical_size,
            dtype=torch.float16,
            device=device,
        )
        layer._awq_moe_buf_permuted_input = torch.empty(
            max_slots,
            layer.sm70_hidden_logical_size,
            dtype=torch.float16,
            device=device,
        )
        layer._awq_moe_buf_gate_up = torch.empty(
            max_slots, layer.sm70_w13_n_dim, dtype=torch.float16, device=device
        )
        layer._awq_moe_buf_intermediate = torch.empty(
            max_slots,
            layer.sm70_intermediate_size,
            dtype=torch.float16,
            device=device,
        )
        layer._awq_moe_buf_sorted_output = torch.empty(
            max_slots, layer.sm70_w2_n_dim, dtype=torch.float16, device=device
        )
        layer._awq_moe_buf_expert_offsets = torch.empty(
            layer.sm70_num_experts + 1, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_expert_offsets64 = torch.empty(
            layer.sm70_num_experts + 1, dtype=torch.int64, device=device
        )
        layer._awq_moe_buf_inv_permuted_idx = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_topk_ids = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(persistent_tokens, top_k)
        layer._awq_moe_buf_permuted_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_sorted_expert_ids = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        sort_workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            max_slots, layer.global_num_experts
        )
        layer._awq_moe_buf_sort_workspace = torch.empty(
            sort_workspace_size, dtype=torch.int8, device=device
        )
        layer._awq_moe_buf_permuted_experts_id = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_sorted_row_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_topk_ids_for_sort = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_strict_expert_offsets = torch.empty(
            layer.sm70_num_experts + 1, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_strict_expert_offsets64 = torch.empty(
            layer.sm70_num_experts + 1, dtype=torch.int64, device=device
        )
        layer._awq_moe_buf_strict_inv_permuted_idx = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_strict_sorted_expert_ids = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_dense_expert_ids = torch.arange(
            layer.sm70_num_experts, dtype=torch.int32, device=device
        )
        layer._awq_moe_buf_active_expert_offsets = torch.arange(
            max_slots + 1, dtype=torch.int32, device=device
        )
        ptr_row_bytes = int(layer.sm70_ptr_row_bytes)
        layer._awq_moe_buf_compact_w13_ptrs_w = torch.empty(
            top_k * ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._awq_moe_buf_compact_w13_ptrs_s = torch.empty(
            top_k * ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._awq_moe_buf_legacy_w13_ptrs_w = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._awq_moe_buf_legacy_w13_ptrs_s = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._awq_moe_buf_legacy_w2_ptrs_w = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._awq_moe_buf_legacy_w2_ptrs_s = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )

    def _get_buffers(
        self,
        layer: RoutedExperts,
        total_slots: int,
        num_tokens: int,
    ) -> dict[str, torch.Tensor]:
        use_temporary_buffers = _use_temporary_buffers_for_dummy_or_capture()
        if (
            not use_temporary_buffers
            and
            total_slots <= layer._awq_moe_buf_max_slots
            and num_tokens <= layer._awq_moe_buf_max_tokens
        ):
            return {
                "output": layer._awq_moe_buf_output[:num_tokens],
                "permuted_input": layer._awq_moe_buf_permuted_input[:total_slots],
                "gate_up": layer._awq_moe_buf_gate_up[:total_slots],
                "intermediate": layer._awq_moe_buf_intermediate[:total_slots],
                "sorted_output": layer._awq_moe_buf_sorted_output[:total_slots],
                "expert_offsets": layer._awq_moe_buf_expert_offsets,
                "expert_offsets64": layer._awq_moe_buf_expert_offsets64,
                "inv_permuted_idx": layer._awq_moe_buf_inv_permuted_idx[:num_tokens],
                "topk_ids": layer._awq_moe_buf_topk_ids[:num_tokens],
                "token_expert_indices": layer._awq_moe_buf_token_expert_indices[
                    :num_tokens
                ],
                "permuted_idx": layer._awq_moe_buf_permuted_idx[:total_slots],
                "sorted_expert_ids": layer._awq_moe_buf_sorted_expert_ids[
                    :total_slots
                ],
                "sort_workspace": layer._awq_moe_buf_sort_workspace,
                "permuted_experts_id": layer._awq_moe_buf_permuted_experts_id[
                    :total_slots
                ],
                "sorted_row_idx": layer._awq_moe_buf_sorted_row_idx[:total_slots],
                "topk_ids_for_sort": layer._awq_moe_buf_topk_ids_for_sort[
                    :total_slots
                ],
                "active_expert_offsets": (
                    layer._awq_moe_buf_active_expert_offsets[: total_slots + 1]
                ),
                "strict_expert_offsets": layer._awq_moe_buf_strict_expert_offsets,
                "strict_expert_offsets64": (
                    layer._awq_moe_buf_strict_expert_offsets64
                ),
                "strict_inv_permuted_idx": (
                    layer._awq_moe_buf_strict_inv_permuted_idx[:num_tokens]
                ),
                "strict_sorted_expert_ids": (
                    layer._awq_moe_buf_strict_sorted_expert_ids[:total_slots]
                ),
                "compact_w13_ptrs_w": layer._awq_moe_buf_compact_w13_ptrs_w,
                "compact_w13_ptrs_s": layer._awq_moe_buf_compact_w13_ptrs_s,
                "legacy_w13_ptrs_w": layer._awq_moe_buf_legacy_w13_ptrs_w,
                "legacy_w13_ptrs_s": layer._awq_moe_buf_legacy_w13_ptrs_s,
                "legacy_w2_ptrs_w": layer._awq_moe_buf_legacy_w2_ptrs_w,
                "legacy_w2_ptrs_s": layer._awq_moe_buf_legacy_w2_ptrs_s,
            }

        device = layer._awq_moe_buf_output.device
        top_k = layer._awq_moe_buf_top_k
        compact_w13_ptrs_w = layer._awq_moe_buf_compact_w13_ptrs_w
        compact_w13_ptrs_s = layer._awq_moe_buf_compact_w13_ptrs_s
        legacy_w13_ptrs_w = layer._awq_moe_buf_legacy_w13_ptrs_w
        legacy_w13_ptrs_s = layer._awq_moe_buf_legacy_w13_ptrs_s
        legacy_w2_ptrs_w = layer._awq_moe_buf_legacy_w2_ptrs_w
        legacy_w2_ptrs_s = layer._awq_moe_buf_legacy_w2_ptrs_s
        sort_workspace = layer._awq_moe_buf_sort_workspace
        if use_temporary_buffers:
            compact_w13_ptrs_w = torch.empty_like(compact_w13_ptrs_w)
            compact_w13_ptrs_s = torch.empty_like(compact_w13_ptrs_s)
            legacy_w13_ptrs_w = torch.empty_like(legacy_w13_ptrs_w)
            legacy_w13_ptrs_s = torch.empty_like(legacy_w13_ptrs_s)
            legacy_w2_ptrs_w = torch.empty_like(legacy_w2_ptrs_w)
            legacy_w2_ptrs_s = torch.empty_like(legacy_w2_ptrs_s)
            sort_workspace = torch.empty_like(sort_workspace)
        if total_slots > layer._awq_moe_buf_max_slots:
            sort_workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
                total_slots, layer.global_num_experts
            )
            sort_workspace = torch.empty(
                sort_workspace_size, dtype=torch.int8, device=device
            )
            active_expert_offsets = torch.arange(
                total_slots + 1, dtype=torch.int32, device=device
            )
        else:
            active_expert_offsets = layer._awq_moe_buf_active_expert_offsets[
                : total_slots + 1
            ]
        return {
            "output": torch.empty(
                num_tokens,
                layer.sm70_hidden_logical_size,
                dtype=torch.float16,
                device=device,
            ),
            "permuted_input": torch.empty(
                total_slots,
                layer.sm70_hidden_logical_size,
                dtype=torch.float16,
                device=device,
            ),
            "gate_up": torch.empty(
                total_slots,
                layer.sm70_w13_n_dim,
                dtype=torch.float16,
                device=device,
            ),
            "intermediate": torch.empty(
                total_slots,
                layer.sm70_intermediate_size,
                dtype=torch.float16,
                device=device,
            ),
            "sorted_output": torch.empty(
                total_slots,
                layer.sm70_w2_n_dim,
                dtype=torch.float16,
                device=device,
            ),
            "expert_offsets": torch.empty(
                layer.sm70_num_experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                layer.sm70_num_experts + 1, dtype=torch.int64, device=device
            ),
            "inv_permuted_idx": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "topk_ids": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "token_expert_indices": torch.arange(
                total_slots, dtype=torch.int32, device=device
            ).view(num_tokens, top_k),
            "permuted_idx": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "sorted_expert_ids": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "sort_workspace": sort_workspace,
            "permuted_experts_id": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "sorted_row_idx": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "topk_ids_for_sort": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "active_expert_offsets": active_expert_offsets,
            "strict_expert_offsets": torch.empty(
                layer.sm70_num_experts + 1, dtype=torch.int32, device=device
            ),
            "strict_expert_offsets64": torch.empty(
                layer.sm70_num_experts + 1, dtype=torch.int64, device=device
            ),
            "strict_inv_permuted_idx": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "strict_sorted_expert_ids": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "compact_w13_ptrs_w": compact_w13_ptrs_w,
            "compact_w13_ptrs_s": compact_w13_ptrs_s,
            "legacy_w13_ptrs_w": legacy_w13_ptrs_w,
            "legacy_w13_ptrs_s": legacy_w13_ptrs_s,
            "legacy_w2_ptrs_w": legacy_w2_ptrs_w,
            "legacy_w2_ptrs_s": legacy_w2_ptrs_s,
        }

    def _apply_legacy_single_token_compact(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids_i32: torch.Tensor,
        buffers: dict[str, torch.Tensor],
        top_k: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        _log_runtime_route_once(
            "SM70 AWQ MoE legacy single-token monolithic compact path enabled "
            "(top_k=%d, experts=%d).",
            top_k,
            layer.sm70_num_experts,
        )
        sm70_ops.awq_moe_single_token_sm70_out(
            output,
            x,
            topk_weights,
            topk_ids_i32,
            layer.w13_legacy_strided_ptrs_w_rows,
            layer.w13_legacy_strided_ptrs_s_rows,
            layer.w2_strided_ptrs_w_rows,
            layer.w2_strided_ptrs_s_rows,
            buffers["permuted_input"],
            buffers["intermediate"],
            buffers["sorted_output"],
            buffers["legacy_w13_ptrs_w"],
            buffers["legacy_w13_ptrs_s"],
            buffers["legacy_w2_ptrs_w"],
            buffers["legacy_w2_ptrs_s"],
            buffers["active_expert_offsets"],
            buffers["inv_permuted_idx"],
            layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim,
            layer.sm70_w2_k_dim,
            layer.sm70_w2_n_dim,
            self.group_size,
            layer.sm70_hidden_logical_size,
        )
        return output

    @property
    def supports_eplb(self) -> bool:
        return False

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "SM70 AWQ MoE does not support apply_router_weight_on_input yet."
            )

        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        total_slots = num_tokens * top_k
        buffers = self._get_buffers(layer, total_slots, num_tokens)
        output = buffers["output"]
        output.zero_()
        if total_slots == 0:
            return output

        topk_ids_i32 = buffers["topk_ids"]
        topk_ids_i32.copy_(topk_ids, non_blocking=True)
        x = _dump_awq_moe_buffer(layer, x, "input")
        topk_weights = _dump_awq_moe_buffer(layer, topk_weights, "topk_weights")
        topk_ids_i32 = _dump_awq_moe_buffer(layer, topk_ids_i32, "topk_ids_i32")
        if (
            num_tokens == 1
            and layer.sm70_awq_moe_batched_gemm
            and _legacy_single_token_compact_enabled()
            and hasattr(layer, "w13_legacy_strided_ptrs_w_rows")
        ):
            return self._apply_legacy_single_token_compact(
                layer, x, topk_weights, topk_ids_i32, buffers, top_k, output
            )
        use_batched_single_token_strict = (
            num_tokens == 1
            and layer.sm70_awq_moe_batched_gemm
            and envs.VLLM_SM70_AWQ_MOE_BATCHED_SINGLE_TOKEN_DENSE_W13
        )
        use_batched_single_token_indexed = (
            num_tokens == 1
            and layer.sm70_awq_moe_batched_gemm
            and not use_batched_single_token_strict
            and _single_token_indexed_w13_enabled()
            and _single_token_indexed_w2_enabled()
        )
        if num_tokens == 1 and (
            not layer.sm70_awq_moe_batched_gemm
            or use_batched_single_token_strict
            or use_batched_single_token_indexed
        ):
            use_compact_w13 = _single_token_compact_w13_enabled()
            use_indexed_w13 = (
                not use_compact_w13
                and not use_batched_single_token_strict
                and _single_token_indexed_w13_enabled()
            )
            use_indexed_w2 = (
                not use_batched_single_token_strict
                and _single_token_indexed_w2_enabled()
            )
            _log_runtime_route_once(
                "SM70 AWQ MoE single-token active-expert dense path enabled "
                "(top_k=%d, experts=%d).",
                top_k,
                layer.sm70_num_experts,
            )
            if use_batched_single_token_strict:
                _log_runtime_route_once(
                    "SM70 AWQ MoE batched path using strict single-token "
                    "decode route (top_k=%d, experts=%d).",
                    top_k,
                    layer.sm70_num_experts,
                )
            if use_batched_single_token_indexed:
                _log_runtime_route_once(
                    "SM70 AWQ MoE batched path using single-token indexed "
                    "dense-stage route (top_k=%d, experts=%d).",
                    top_k,
                    layer.sm70_num_experts,
                )
            if use_indexed_w13 or use_indexed_w2:
                _log_runtime_route_once(
                    "SM70 AWQ MoE single-token indexed dense-stage path "
                    "enabled (top_k=%d, w13=%s, w2=%s).",
                    top_k,
                    use_indexed_w13,
                    use_indexed_w2,
                )
            if use_compact_w13:
                _log_runtime_route_once(
                    "SM70 AWQ MoE single-token compact grouped W13 path "
                    "enabled (top_k=%d).",
                    top_k,
                )
                sm70_ops.awq_moe_single_token_compact_dense_w13_sm70_out(
                    buffers["gate_up"],
                    buffers["permuted_input"],
                    x,
                    topk_ids_i32,
                    layer.w13_strided_ptrs_w,
                    layer.w13_strided_ptrs_s,
                    buffers["compact_w13_ptrs_w"],
                    buffers["compact_w13_ptrs_s"],
                    buffers["expert_offsets"],
                    buffers["expert_offsets64"],
                    buffers["inv_permuted_idx"],
                    buffers["sorted_expert_ids"],
                    layer.sm70_w13_k_dim,
                    layer.sm70_w13_n_dim,
                    self.group_size,
                    layer.sm70_hidden_logical_size,
                )
            elif use_indexed_w13:
                sm70_ops.awq_moe_single_token_indexed_dense_w13_sm70_out(
                    buffers["gate_up"],
                    buffers["permuted_input"],
                    x,
                    topk_ids_i32,
                    layer.w13_strided_ptrs_w,
                    layer.w13_strided_ptrs_s,
                    buffers["expert_offsets"],
                    buffers["expert_offsets64"],
                    buffers["inv_permuted_idx"],
                    buffers["sorted_expert_ids"],
                    layer.sm70_w13_k_dim,
                    layer.sm70_w13_n_dim,
                    self.group_size,
                    layer.sm70_hidden_logical_size,
                )
            else:
                sm70_ops.awq_moe_single_token_dense_w13_sm70_out(
                    buffers["gate_up"],
                    buffers["permuted_input"],
                    x,
                    topk_ids_i32,
                    layer.w13_strided_ptrs_w,
                    layer.w13_strided_ptrs_s,
                    buffers["expert_offsets"],
                    buffers["expert_offsets64"],
                    buffers["inv_permuted_idx"],
                    buffers["sorted_expert_ids"],
                    layer.sm70_w13_k_dim,
                    layer.sm70_w13_n_dim,
                    self.group_size,
                    layer.sm70_hidden_logical_size,
                )
            buffers["expert_offsets"] = _dump_awq_moe_buffer(
                layer, buffers["expert_offsets"], "st_expert_offsets"
            )
            buffers["sorted_expert_ids"] = _dump_awq_moe_buffer(
                layer, buffers["sorted_expert_ids"], "st_sorted_expert_ids"
            )
            buffers["inv_permuted_idx"] = _dump_awq_moe_buffer(
                layer, buffers["inv_permuted_idx"], "st_inv_permuted_idx"
            )
            buffers["gate_up"] = _dump_awq_moe_buffer(
                layer, buffers["gate_up"], "st_w13_out"
            )
            torch.ops._C.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
            buffers["intermediate"] = _dump_awq_moe_buffer(
                layer, buffers["intermediate"], "st_silu_out"
            )
            if use_indexed_w2:
                sm70_ops.awq_moe_single_token_indexed_dense_stage_sm70_out(
                    buffers["sorted_output"],
                    buffers["intermediate"],
                    buffers["expert_offsets"],
                    buffers["sorted_expert_ids"],
                    layer.w2_strided_ptrs_w,
                    layer.w2_strided_ptrs_s,
                    top_k,
                    layer.sm70_w2_k_dim,
                    layer.sm70_w2_n_dim,
                    self.group_size,
                )
            else:
                sm70_ops.awq_moe_single_token_dense_stage_sm70_out(
                    buffers["sorted_output"],
                    buffers["intermediate"],
                    buffers["expert_offsets"],
                    buffers["sorted_expert_ids"],
                    layer.w2_strided_ptrs_w,
                    layer.w2_strided_ptrs_s,
                    top_k,
                    layer.sm70_w2_k_dim,
                    layer.sm70_w2_n_dim,
                    self.group_size,
                )
            buffers["sorted_output"] = _dump_awq_moe_buffer(
                layer, buffers["sorted_output"], "st_w2_out"
            )
            sorted_output = buffers["sorted_output"][
                :, : layer.sm70_hidden_logical_size
            ]
            if _single_token_weighted_reduce_enabled():
                _log_runtime_route_once(
                    "SM70 AWQ MoE single-token weighted-reduce path enabled "
                    "(top_k=%d).",
                    top_k,
                )
                sm70_ops.awq_moe_single_token_weighted_reduce_out(
                    sorted_output,
                    topk_weights,
                    buffers["inv_permuted_idx"],
                    output,
                    top_k,
                    layer.sm70_hidden_logical_size,
                )
            else:
                torch.ops._moe_C.moe_unpermute(
                    sorted_output,
                    topk_weights,
                    buffers["inv_permuted_idx"],
                    buffers["expert_offsets64"][: top_k + 1],
                    top_k,
                    output,
                )
            output = _dump_awq_moe_buffer(layer, output, "st_output")
            return output
        torch.ops._moe_C.moe_permute_with_scratch(
            x,
            topk_ids_i32,
            buffers["token_expert_indices"],
            layer.expert_map,
            layer.global_num_experts,
            layer.local_num_experts,
            top_k,
            buffers["permuted_input"],
            buffers["expert_offsets64"],
            buffers["inv_permuted_idx"],
            buffers["permuted_idx"],
            buffers["sort_workspace"],
            buffers["permuted_experts_id"],
            buffers["sorted_row_idx"],
            buffers["topk_ids_for_sort"],
        )
        buffers["expert_offsets"].copy_(buffers["expert_offsets64"], non_blocking=True)
        buffers["expert_offsets"] = _dump_awq_moe_buffer(
            layer, buffers["expert_offsets"], "expert_offsets"
        )
        buffers["expert_offsets64"] = _dump_awq_moe_buffer(
            layer, buffers["expert_offsets64"], "expert_offsets64"
        )
        buffers["inv_permuted_idx"] = _dump_awq_moe_buffer(
            layer, buffers["inv_permuted_idx"], "inv_permuted_idx"
        )
        buffers["permuted_experts_id"] = _dump_awq_moe_buffer(
            layer, buffers["permuted_experts_id"], "permuted_experts_id"
        )
        buffers["sorted_expert_ids"] = _dump_awq_moe_buffer(
            layer, buffers["sorted_expert_ids"], "sorted_expert_ids"
        )
        route_plan = select_sm70_quantized_moe_route(
            batched_enabled=layer.sm70_awq_moe_batched_gemm,
            num_tokens=num_tokens,
            total_slots=total_slots,
            batched_decode_max_tokens=(
                envs.VLLM_SM70_AWQ_MOE_BATCHED_DECODE_MAX_TOKENS
            ),
            strict_dense_w13=(
                envs.VLLM_SM70_AWQ_MOE_BATCHED_SINGLE_TOKEN_DENSE_W13
            ),
            exact_w2=envs.VLLM_SM70_AWQ_MOE_BATCHED_EXACT_W2,
            active_exact_w2=envs.VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2,
            w13_per_expert_dispatch=True,
            w2_per_expert_dispatch=True,
        )
        use_batched_strict_moe = route_plan.use_batched_strict_w13
        use_batched_moe_gemm = route_plan.use_batched_moe_gemm
        use_batched_active_exact_w2 = route_plan.use_batched_active_exact_w2
        use_batched_exact_w2 = route_plan.use_batched_exact_w2
        use_active_exact_small_batched_moe = False
        compare_dense_step = None
        compare_dense_w13_stats = None
        compare_dense_w2_stats = None
        compare_dense_full_w2_stats = None
        compare_dense_full_output = None
        compare_strict_output = None
        compare_strict_stats = None
        compare_route_state = None
        dense_gate_up = None
        if num_tokens <= 8 and use_batched_moe_gemm:
            compare_dense_step = _compare_dense_decode_step(layer)

        if use_active_exact_small_batched_moe:
            _log_runtime_route_once(
                "SM70 AWQ MoE batched path using active-route exact "
                "dense-stage route (tokens=%d, routes=%d).",
                num_tokens,
                total_slots,
            )
            sm70_ops.awq_moe_single_token_dense_stage_sm70_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                buffers["active_expert_offsets"],
                buffers["permuted_experts_id"],
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                total_slots,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
            )
        elif route_plan.w13 == Sm70MoeStageRoute.PER_EXPERT_DISPATCH:
            _log_runtime_route_once(
                "SM70 AWQ MoE batched W13 using per-expert dispatch "
                "selection (experts=%d).",
                layer.sm70_num_experts,
            )
            sm70_ops.awq_moe_gemm_sm70_per_expert_dispatch_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                buffers["expert_offsets"],
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
                False,
            )
            buffers["gate_up"] = _dump_awq_moe_buffer(
                layer, buffers["gate_up"], "w13_batched_out"
            )
            if compare_dense_step is not None:
                dense_gate_up = torch.empty_like(buffers["gate_up"])
                sm70_ops.awq_moe_dense_stage_sm70_out(
                    dense_gate_up,
                    buffers["permuted_input"],
                    buffers["expert_offsets"],
                    layer._awq_moe_buf_dense_expert_ids,
                    layer.w13_strided_ptrs_w,
                    layer.w13_strided_ptrs_s,
                    layer.sm70_num_experts,
                    layer.sm70_w13_k_dim,
                    layer.sm70_w13_n_dim,
                    self.group_size,
                )
                compare_dense_w13_stats = _diff_stats(
                    buffers["gate_up"], dense_gate_up
                )
        else:
            if use_batched_strict_moe:
                _log_runtime_route_once(
                    "SM70 AWQ MoE batched path using strict dense-stage "
                    "for multi-token shapes (experts=%d).",
                    layer.sm70_num_experts,
                )
            _log_runtime_route_once(
                "SM70 AWQ MoE CUDA-graph-safe dense-stage path enabled "
                "(experts=%d).",
                layer.sm70_num_experts,
            )
            sm70_ops.awq_moe_dense_stage_sm70_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                buffers["expert_offsets"],
                layer._awq_moe_buf_dense_expert_ids,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
            )
            buffers["gate_up"] = _dump_awq_moe_buffer(
                layer, buffers["gate_up"], "w13_dense_out"
            )
        torch.ops._C.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
        buffers["intermediate"] = _dump_awq_moe_buffer(
            layer, buffers["intermediate"], "silu_out"
        )
        if use_active_exact_small_batched_moe:
            sm70_ops.awq_moe_single_token_dense_stage_sm70_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["active_expert_offsets"],
                buffers["permuted_experts_id"],
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                total_slots,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
            )
        elif use_batched_active_exact_w2:
            _log_runtime_route_once(
                "SM70 AWQ MoE batched path using grouped-active exact W2 "
                "(routes=%d).",
                total_slots,
            )
            sm70_ops.awq_moe_active_dense_stage_sm70_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["permuted_experts_id"],
                buffers["active_expert_offsets"],
                buffers["sorted_expert_ids"],
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                total_slots,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
            )
            if compare_dense_step is not None:
                dense_sorted_output = torch.empty_like(buffers["sorted_output"])
                sm70_ops.awq_moe_dense_stage_sm70_out(
                    dense_sorted_output,
                    buffers["intermediate"],
                    buffers["expert_offsets"],
                    layer._awq_moe_buf_dense_expert_ids,
                    layer.w2_strided_ptrs_w,
                    layer.w2_strided_ptrs_s,
                    layer.sm70_num_experts,
                    layer.sm70_w2_k_dim,
                    layer.sm70_w2_n_dim,
                    self.group_size,
                )
                compare_dense_w2_stats = _diff_stats(
                    buffers["sorted_output"], dense_sorted_output
                )
                compare_dense_full_output = torch.empty_like(output)
                compare_dense_full_output.zero_()
                dense_full_sorted_output = dense_sorted_output
                if dense_gate_up is not None:
                    dense_intermediate = torch.empty_like(buffers["intermediate"])
                    dense_full_sorted_output = torch.empty_like(
                        buffers["sorted_output"]
                    )
                    torch.ops._C.silu_and_mul(dense_intermediate, dense_gate_up)
                    sm70_ops.awq_moe_dense_stage_sm70_out(
                        dense_full_sorted_output,
                        dense_intermediate,
                        buffers["expert_offsets"],
                        layer._awq_moe_buf_dense_expert_ids,
                        layer.w2_strided_ptrs_w,
                        layer.w2_strided_ptrs_s,
                        layer.sm70_num_experts,
                        layer.sm70_w2_k_dim,
                        layer.sm70_w2_n_dim,
                        self.group_size,
                    )
                    compare_dense_full_w2_stats = _diff_stats(
                        buffers["sorted_output"], dense_full_sorted_output
                    )
                dense_full_sorted_output_logical = dense_full_sorted_output[
                    :, : layer.sm70_hidden_logical_size
                ]
                torch.ops._moe_C.moe_unpermute(
                    dense_full_sorted_output_logical,
                    topk_weights,
                    buffers["inv_permuted_idx"],
                    buffers["expert_offsets64"],
                    top_k,
                    compare_dense_full_output,
                )
                compare_route_state = {
                    "expert_ranges": _expert_offset_ranges(
                        buffers["expert_offsets"]
                    ),
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
        elif route_plan.w2 == Sm70MoeStageRoute.PER_EXPERT_DISPATCH:
            _log_runtime_route_once(
                "SM70 AWQ MoE batched W2 using per-expert dispatch "
                "selection (experts=%d).",
                layer.sm70_num_experts,
            )
            sm70_ops.awq_moe_gemm_sm70_per_expert_dispatch_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["expert_offsets"],
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
                False,
            )
            buffers["sorted_output"] = _dump_awq_moe_buffer(
                layer, buffers["sorted_output"], "w2_batched_out"
            )
            if compare_dense_step is not None:
                dense_sorted_output = torch.empty_like(buffers["sorted_output"])
                sm70_ops.awq_moe_dense_stage_sm70_out(
                    dense_sorted_output,
                    buffers["intermediate"],
                    buffers["expert_offsets"],
                    layer._awq_moe_buf_dense_expert_ids,
                    layer.w2_strided_ptrs_w,
                    layer.w2_strided_ptrs_s,
                    layer.sm70_num_experts,
                    layer.sm70_w2_k_dim,
                    layer.sm70_w2_n_dim,
                    self.group_size,
                )
                compare_dense_w2_stats = _diff_stats(
                    buffers["sorted_output"], dense_sorted_output
                )
                dense_intermediate = torch.empty_like(buffers["intermediate"])
                dense_full_sorted_output = torch.empty_like(buffers["sorted_output"])
                compare_dense_full_output = torch.empty_like(output)
                compare_dense_full_output.zero_()
                torch.ops._C.silu_and_mul(dense_intermediate, dense_gate_up)
                sm70_ops.awq_moe_dense_stage_sm70_out(
                    dense_full_sorted_output,
                    dense_intermediate,
                    buffers["expert_offsets"],
                    layer._awq_moe_buf_dense_expert_ids,
                    layer.w2_strided_ptrs_w,
                    layer.w2_strided_ptrs_s,
                    layer.sm70_num_experts,
                    layer.sm70_w2_k_dim,
                    layer.sm70_w2_n_dim,
                    self.group_size,
                )
                compare_dense_full_w2_stats = _diff_stats(
                    buffers["sorted_output"], dense_full_sorted_output
                )
                dense_full_sorted_output_logical = dense_full_sorted_output[
                    :, : layer.sm70_hidden_logical_size
                ]
                torch.ops._moe_C.moe_unpermute(
                    dense_full_sorted_output_logical,
                    topk_weights,
                    buffers["inv_permuted_idx"],
                    buffers["expert_offsets64"],
                    top_k,
                    compare_dense_full_output,
                )
                if num_tokens == 1:
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
                    compare_strict_output = torch.empty_like(output)
                    compare_strict_output.zero_()
                    sm70_ops.awq_moe_single_token_dense_w13_sm70_out(
                        strict_gate_up,
                        strict_compact_input,
                        x,
                        topk_ids_i32,
                        layer.w13_strided_ptrs_w,
                        layer.w13_strided_ptrs_s,
                        strict_expert_offsets,
                        strict_expert_offsets64,
                        strict_inv_permuted_idx,
                        strict_sorted_expert_ids,
                        layer.sm70_w13_k_dim,
                        layer.sm70_w13_n_dim,
                        self.group_size,
                        layer.sm70_hidden_logical_size,
                    )
                    torch.ops._C.silu_and_mul(strict_intermediate, strict_gate_up)
                    sm70_ops.awq_moe_single_token_dense_stage_sm70_out(
                        strict_sorted_output,
                        strict_intermediate,
                        strict_expert_offsets,
                        strict_sorted_expert_ids,
                        layer.w2_strided_ptrs_w,
                        layer.w2_strided_ptrs_s,
                        top_k,
                        layer.sm70_w2_k_dim,
                        layer.sm70_w2_n_dim,
                        self.group_size,
                    )
                    strict_sorted_output_logical = strict_sorted_output[
                        :, : layer.sm70_hidden_logical_size
                    ]
                    if _single_token_weighted_reduce_enabled():
                        sm70_ops.awq_moe_single_token_weighted_reduce_out(
                            strict_sorted_output_logical,
                            topk_weights,
                            strict_inv_permuted_idx,
                            compare_strict_output,
                            top_k,
                            layer.sm70_hidden_logical_size,
                        )
                    else:
                        torch.ops._moe_C.moe_unpermute(
                            strict_sorted_output_logical,
                            topk_weights,
                            strict_inv_permuted_idx,
                            strict_expert_offsets64[: top_k + 1],
                            top_k,
                            compare_strict_output,
                        )
                    compare_strict_stats = {
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
                            : top_k + 1
                        ]
                        .detach()
                        .cpu()
                        .tolist(),
                    }
        else:
            if use_batched_exact_w2:
                _log_runtime_route_once(
                    "SM70 AWQ MoE batched path using exact dense-stage W2 "
                    "(experts=%d).",
                    layer.sm70_num_experts,
                )
            sm70_ops.awq_moe_dense_stage_sm70_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["expert_offsets"],
                layer._awq_moe_buf_dense_expert_ids,
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
            )
            buffers["sorted_output"] = _dump_awq_moe_buffer(
                layer, buffers["sorted_output"], "w2_dense_out"
            )
        sorted_output = buffers["sorted_output"][:, : layer.sm70_hidden_logical_size]
        torch.ops._moe_C.moe_unpermute(
            sorted_output,
            topk_weights,
            buffers["inv_permuted_idx"],
            buffers["expert_offsets64"],
            top_k,
            output,
        )
        if compare_dense_step is not None:
            record = {
                "decode_step": compare_dense_step,
                "device": int(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None,
                "layer_id": getattr(layer, "sm70_awq_moe_layer_id", None),
                "layer_name": str(getattr(layer, "layer_name", "")),
                "num_tokens": int(num_tokens),
                "pid": int(os.getpid()),
                "topk_ids": topk_ids_i32.detach().cpu().tolist(),
                "topk_weights": topk_weights.detach().float().cpu().tolist(),
                "w13_batched_vs_dense": compare_dense_w13_stats,
                "w2_batched_vs_dense_same_intermediate": compare_dense_w2_stats,
                "w2_batched_vs_dense_full_pipeline": compare_dense_full_w2_stats,
                "route_state": compare_route_state,
                "strict_reference": compare_strict_stats,
            }
            if compare_dense_full_output is not None:
                record["output_batched_vs_dense_full_pipeline"] = _diff_stats(
                    output, compare_dense_full_output
                )
            if compare_strict_output is not None:
                record["output_batched_vs_strict"] = _diff_stats(
                    output, compare_strict_output
                )
            _write_compare_dense_record(record)
        output = _dump_awq_moe_buffer(layer, output, "output")
        return output

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, x, router_logits, input_ids
        raise NotImplementedError("SM70 AWQ MoE base path is not monolithic.")

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None
