# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 AWQ MoE method backed by TurboMind GEMM kernels."""

from dataclasses import replace
from typing import Final

import torch
from torch.nn import Parameter

from vllm.config import get_current_vllm_config_or_none
from vllm.config.sm70_moe import Sm70MoEFormatConfig, capture_sm70_moe_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    AwqStageObserver,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    _compare_dense_base_enabled as _compare_dense_base_enabled,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    _compare_dense_decode_step as _compare_dense_decode_step,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    _diff_stats as _diff_stats,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    _dump_awq_moe_buffer as _dump_awq_moe_buffer,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    _dump_awq_moe_buffer_requested as _dump_awq_moe_buffer_requested,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    _expert_offset_ranges as _expert_offset_ranges,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    _get_layer_id as _get_layer_id,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    _parse_layer_id_filter as _parse_layer_id_filter,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    _silu_and_mul_w13 as _silu_and_mul_w13,
)
from vllm.model_executor.layers.fused_moe.sm70.diagnostics import (
    _write_compare_dense_record as _write_compare_dense_record,
)
from vllm.model_executor.layers.fused_moe.sm70.method import Sm70MoEMethodBase
from vllm.model_executor.layers.fused_moe.sm70.single_token import execute_single_token
from vllm.model_executor.layers.fused_moe.sm70.stages import execute_routed
from vllm.model_executor.layers.quantization.awq_qpn_sm70 import (
    initialize_qpn_m1,
    use_qpn_m1,
)
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    Sm70MoeStageRoute,
    select_sm70_quantized_moe_route,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

_DEFAULT_PERSISTENT_MAX_TOKENS = 32
_QWEN38_INDEXED_PREFILL_MIN_TOKENS: Final = 128
_QWEN38_CHUNKED_W2_SUPPORTED_TOKENS: Final = (4096, 6144)


def _resolve_persistent_max_tokens(
    max_num_seqs: int,
    verifier_width: int = 1,
    override: int = 0,
) -> int:
    """Size resident decode/verifier scratch up to the legacy ceiling."""
    scheduler_cap = max(1, int(max_num_seqs)) * max(1, int(verifier_width))
    requested_cap = int(override)
    if requested_cap <= 0:
        return min(scheduler_cap, _DEFAULT_PERSISTENT_MAX_TOKENS)
    return min(requested_cap, _DEFAULT_PERSISTENT_MAX_TOKENS)


def _persistent_max_tokens_for_runtime(
    policy: Sm70MoEFormatConfig | None = None,
) -> int:
    policy = policy if policy is not None else capture_sm70_moe_config("awq")
    vllm_config = get_current_vllm_config_or_none()
    scheduler_config = None if vllm_config is None else vllm_config.scheduler_config
    max_num_seqs = (
        _DEFAULT_PERSISTENT_MAX_TOKENS
        if scheduler_config is None
        else scheduler_config.max_num_seqs
    )
    speculative_config = None if vllm_config is None else vllm_config.speculative_config
    verifier_width = (
        speculative_config.num_speculative_state_tokens() + 1
        if speculative_config is not None and speculative_config.method == "mtp"
        else 1
    )
    return _resolve_persistent_max_tokens(
        max_num_seqs,
        verifier_width,
        policy.persistent_tokens or 0,
    )


_QWEN38_TP4_NUM_EXPERTS = 512
_QWEN38_TP4_W13_QWEIGHT_SHAPE = (2560, 40)
_QWEN38_TP4_W2_QWEIGHT_SHAPE = (160, 320)


def _log_runtime_route_once(message: str, *args) -> None:
    if torch.compiler.is_compiling():
        return
    logger.info_once(message, *args)


def _qwen38_active_grouped_layer_contract(
    layer: RoutedExperts, group_size: int
) -> bool:
    return bool(
        int(layer.moe_config.tp_size) == 4
        and layer.sm70_num_experts == 512
        and group_size == 32
        and layer.sm70_hidden_logical_size == layer.sm70_w13_k_dim == 2560
        and layer.sm70_w13_n_dim == 320
        and layer.sm70_w2_k_dim == 160
        and layer.sm70_w2_n_dim == 2560
    )


def _use_qwen38_active_grouped_decode(
    layer: RoutedExperts,
    num_tokens: int,
    top_k: int,
    policy: Sm70MoEFormatConfig | None = None,
) -> bool:
    """Share the runtime admission policy with pre-capture warmup."""
    if policy is None:
        policy = getattr(layer, "sm70_moe_policy", None)
    policy = policy if policy is not None else capture_sm70_moe_config("awq")
    max_tokens = policy.max_batched_tokens or 0
    return bool(
        getattr(layer, "sm70_awq_qwen38_active_grouped_decode", False)
        and layer.sm70_awq_moe_batched_gemm
        and 2 <= num_tokens <= 8
        and top_k == 10
        and (max_tokens <= 0 or num_tokens <= max_tokens)
        and not policy.strict_w13
        and not policy.exact_w2
        and not policy.active_exact_w2
    )


def _use_temporary_buffers_for_dummy_or_capture() -> bool:
    # Dummy/profile and CUDA graph capture allocate temporary tensors. Captured
    # addresses subsequently remain fixed in the graph pool; normal eager
    # decode can reuse the smaller per-layer resident buffers below.
    return is_forward_context_available() and get_forward_context().is_dummy_run


def _use_qwen38_indexed_prefill(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    """Admit only the exact Qwen3.8 TP4 g32 W13 prefill contract."""
    return bool(
        getattr(layer, "sm70_awq_qwen38_indexed_prefill", False)
        and getattr(layer, "sm70_awq_moe_batched_gemm", False)
        and x.ndim == 2
        and x.shape[0] >= _QWEN38_INDEXED_PREFILL_MIN_TOKENS
        and x.shape[1] == 2560
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.shape == (x.shape[0], 10)
        and int(layer.moe_config.tp_size) == 4
        and int(layer.sm70_num_experts) == 512
        and int(layer.sm70_hidden_logical_size) == 2560
        and int(layer.sm70_intermediate_size) == 160
        and int(layer.sm70_w13_k_dim) == 2560
        and int(layer.sm70_w13_n_dim) == 320
        and int(layer.sm70_awq_checkpoint_group_size) == 32
        and int(layer.sm70_awq_group_size) == 32
    )


def _use_qwen38_chunked_w2(
    layer: RoutedExperts,
    num_tokens: int,
    indexed_w13: bool,
) -> bool:
    chunk_tokens = int(layer.sm70_awq_qwen38_w2_chunk_tokens)
    if (
        not indexed_w13
        or chunk_tokens not in _QWEN38_CHUNKED_W2_SUPPORTED_TOKENS
        or chunk_tokens >= num_tokens
    ):
        return False

    # The native loop rebalances short tails, avoiding a small tail that would
    # otherwise force a larger-than-profiled full-output allocation.
    top_k = int(layer._awq_moe_buf_top_k)
    chunk_slots = chunk_tokens * top_k
    full_scratch_bytes = num_tokens * top_k * int(layer.sm70_w2_n_dim) * 2
    chunk_scratch_bytes = (
        chunk_slots * (int(layer.sm70_w2_n_dim) * 2 + 2 * 4)
        + (3 * int(layer.sm70_num_experts) + 1) * 4
    )
    return chunk_scratch_bytes < full_scratch_bytes


def _single_token_weighted_reduce_enabled(
    policy: Sm70MoEFormatConfig | None = None,
) -> bool:
    policy = policy if policy is not None else capture_sm70_moe_config("awq")
    return bool(policy.single_token_reduce == "weighted") and hasattr(
        torch.ops._C, "awq_moe_single_token_weighted_reduce_out"
    )


def _single_token_indexed_w13_enabled(
    policy: Sm70MoEFormatConfig | None = None,
) -> bool:
    policy = policy if policy is not None else capture_sm70_moe_config("awq")
    return bool("indexed" in (policy.single_token_w13 or ())) and hasattr(
        torch.ops._C, "awq_moe_single_token_indexed_dense_w13_sm70_out"
    )


def _single_token_compact_w13_enabled(
    policy: Sm70MoEFormatConfig | None = None,
) -> bool:
    policy = policy if policy is not None else capture_sm70_moe_config("awq")
    return bool("compact" in (policy.single_token_w13 or ())) and hasattr(
        torch.ops._C, "awq_moe_single_token_compact_dense_w13_sm70_out"
    )


def _single_token_indexed_w2_enabled(policy: Sm70MoEFormatConfig | None = None) -> bool:
    policy = policy if policy is not None else capture_sm70_moe_config("awq")
    return bool(policy.single_token_w2 == "indexed") and hasattr(
        torch.ops._C, "awq_moe_single_token_indexed_dense_stage_sm70_out"
    )


def _legacy_single_token_compact_enabled(
    policy: Sm70MoEFormatConfig | None = None,
) -> bool:
    policy = policy if policy is not None else capture_sm70_moe_config("awq")
    return bool(policy.legacy_compact) and hasattr(
        torch.ops._C, "awq_moe_single_token_sm70_out"
    )


def _is_qwen38_tp4_compact_metadata_shape(
    layer: RoutedExperts, group_size: int
) -> bool:
    """Return whether the loaded tensors match the validated compact lane."""
    return (
        group_size == 32
        and tuple(layer.w13_qweight.shape)
        == (_QWEN38_TP4_NUM_EXPERTS, *_QWEN38_TP4_W13_QWEIGHT_SHAPE)
        and tuple(layer.w2_qweight.shape)
        == (_QWEN38_TP4_NUM_EXPERTS, *_QWEN38_TP4_W2_QWEIGHT_SHAPE)
    )


def _resolve_compact_metadata(
    *, requested: bool, explicit: bool, native_available: bool, shape_ok: bool
) -> bool:
    """Decide the prepared metadata layout.

    The default-on compact layout silently falls back to the 4-byte layout
    when the build or the layer shape does not support it. An explicit
    VLLM_SM70_AWQ_MOE_COMPACT_METADATA=1 fails closed instead.
    """
    if not requested:
        return False
    if native_available and shape_ok:
        return True
    if explicit:
        if not native_available:
            raise RuntimeError(
                "VLLM_SM70_AWQ_MOE_COMPACT_METADATA=1 requires an SM70 "
                "build with awq_sm70_prepare_compact."
            )
        raise RuntimeError(
            "VLLM_SM70_AWQ_MOE_COMPACT_METADATA=1 currently requires "
            "the exact Qwen3.8 TP4 E512 native-g32 W13/W2 shapes."
        )
    logger.info_once(
        "SM70 AWQ MoE compact metadata default skipped (%s); "
        "using the 4-byte scale/bias layout.",
        "unsupported layer shape" if native_available else "no compact prepare op",
    )
    return False


def _batched_gemm_enabled_for_layer(
    layer: RoutedExperts, default: bool, policy: Sm70MoEFormatConfig | None = None
) -> bool:
    policy = policy if policy is not None else capture_sm70_moe_config("awq")
    if not default:
        return False
    allowlist = _parse_layer_id_filter(
        policy.layer_allowlist,
        "VLLM_SM70_AWQ_MOE_BATCHED_LAYER_ALLOWLIST",
    )
    denylist = _parse_layer_id_filter(
        policy.layer_denylist,
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
    # Required by int32 torch.empty(int32) buffers: the Parameter ctor itself
    # refuses or warns when an int tensor is registered with requires_grad=True
    # (the default). Pass the flag in directly so the constructor sees it from
    # line one. The trailing param.requires_grad_(False) stays as a no-op
    # idempotent guard for the already-Parameter path.
    param = (
        value if isinstance(value, Parameter) else Parameter(value, requires_grad=False)
    )
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


class AWQSM70MoEMethod(Sm70MoEMethodBase):
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
                f"AWQSM70MoEMethod supports group_size=32/64/128, got {group_size}."
            )
        if not zero_point:
            raise ValueError("AWQSM70MoEMethod currently requires AWQ zero points.")
        if self.moe.has_bias:
            raise NotImplementedError("SM70 AWQ MoE does not support bias yet.")
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.pack_factor = 32 // weight_bits
        self._initialize_sm70_policy("awq", layer, logger)

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
        batched_gemm = _batched_gemm_enabled_for_layer(
            layer, self.use_batched_gemm, self.sm70_moe_policy
        )

        w13_qweight, w13_scales, w13_qzeros, w13_aligned_out = _align_awq_output_dim(
            layer.w13_qweight,
            layer.w13_scales,
            layer.w13_qzeros,
            self.pack_factor,
            align * 2,
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
        w2_qweight, w2_scales, w2_qzeros, hidden_aligned_size = _align_awq_output_dim(
            w2_qweight,
            w2_scales,
            w2_qzeros,
            self.pack_factor,
            align,
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
        compact_metadata = _resolve_compact_metadata(
            requested=bool(self.sm70_moe_policy.compact_metadata),
            explicit="compact_metadata" in self.sm70_moe_policy.explicit_fields,
            native_available=hasattr(torch.ops._C, "awq_sm70_prepare_compact"),
            shape_ok=_is_qwen38_tp4_compact_metadata_shape(layer, self.group_size),
        )
        build_legacy_w13 = (
            batched_gemm
            and bool(self.sm70_moe_policy.legacy_compact)
            and hasattr(torch.ops._C, "awq_moe_single_token_sm70_out")
        )
        # Use one interleaved W13 layout for both batched W13 and the legacy
        # single-token compact op. This keeps the compact speed path without
        # carrying a second per-expert W13 TurboMind copy.
        w13_interleaved = build_legacy_w13
        (
            (w13_tm_weights, w13_tm_scales, w13_meta),
            (w2_tm_weights, w2_tm_scales, w2_meta),
        ) = self.weight_codec.prepare_weights(
            (layer.w13_qweight, layer.w13_scales, layer.w13_qzeros),
            (layer.w2_qweight, layer.w2_scales, layer.w2_qzeros),
            self.group_size,
            compact_metadata=compact_metadata,
            w13_interleaved=w13_interleaved,
        )

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False
        )
        layer.w13_tm_scales = Parameter(torch.stack(w13_tm_scales), requires_grad=False)
        layer.w2_tm_weight = Parameter(torch.stack(w2_tm_weights), requires_grad=False)
        layer.w2_tm_scales = Parameter(torch.stack(w2_tm_scales), requires_grad=False)

        w13_k_ld, w13_q_ld = int(w13_meta[0][0].item()), int(w13_meta[0][1].item())
        w2_k_ld, w2_q_ld = int(w2_meta[0][0].item()), int(w2_meta[0][1].item())
        w13_ptrs = self.native_ops.awq_moe_build_strided_ptrs(
            layer.w13_tm_weight,
            layer.w13_tm_scales,
            w13_k_ld,
            w13_q_ld,
            num_experts,
        )
        w2_ptrs = self.native_ops.awq_moe_build_strided_ptrs(
            layer.w2_tm_weight,
            layer.w2_tm_scales,
            w2_k_ld,
            w2_q_ld,
            num_experts,
        )
        layer.w13_strided_ptrs_w = Parameter(w13_ptrs[0], requires_grad=False)
        layer.w13_strided_ptrs_s = Parameter(w13_ptrs[1], requires_grad=False)
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
            layer.w13_legacy_strided_ptrs_w_rows = layer.w13_strided_ptrs_w_rows
            layer.w13_legacy_strided_ptrs_s_rows = layer.w13_strided_ptrs_s_rows
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
        checkpoint_group_size = int(
            getattr(self, "checkpoint_group_size", self.group_size)
        )
        layer.sm70_awq_checkpoint_group_size = checkpoint_group_size
        layer.sm70_awq_group_size = self.group_size
        layer.sm70_awq_moe_layer_id = _get_layer_id(layer)
        layer.sm70_awq_moe_w13_interleaved = w13_interleaved
        layer.sm70_awq_moe_legacy_single_token_compact = build_legacy_w13
        layer.sm70_awq_moe_compact_metadata = compact_metadata

        indexed_prefill_contract = bool(
            batched_gemm
            and self.group_size == 32
            and int(layer.moe_config.tp_size) == 4
            and self.moe.experts_per_token == 10
            and num_experts == 512
            and hidden_logical_size == 2560
            and intermediate_logical_size == 160
            and layer.sm70_w13_k_dim == 2560
            and layer.sm70_w13_n_dim == 320
            and checkpoint_group_size == 32
        )
        indexed_prefill_requested = bool(self.sm70_moe_policy.indexed_prefill)
        indexed_prefill_ops = {
            "awq_moe_indexed_dense_w13_sm70_out": hasattr(
                torch.ops._C, "awq_moe_indexed_dense_w13_sm70_out"
            ),
            "moe_permute_metadata_with_scratch": hasattr(
                torch.ops._moe_C, "moe_permute_metadata_with_scratch"
            ),
        }
        indexed_prefill_available = all(indexed_prefill_ops.values())
        indexed_prefill_explicit = (
            "indexed_prefill" in self.sm70_moe_policy.explicit_fields
        )
        if (
            indexed_prefill_contract
            and indexed_prefill_requested
            and not indexed_prefill_available
        ):
            missing = [
                name for name, available in indexed_prefill_ops.items() if not available
            ]
            if indexed_prefill_explicit:
                raise RuntimeError(
                    "The explicit SM70 Qwen3.8 AWQ indexed-A prefill route "
                    "requires " + ", ".join(missing) + "."
                )
            logger.warning_once(
                "The default SM70 Qwen3.8 AWQ indexed-A prefill route is not "
                "present in the loaded extension; falling back to the "
                "materialized-input route. Explicitly setting "
                "VLLM_SM70_AWQ_QWEN38_MOE_INDEXED_PREFILL=1 fails closed."
            )
        layer.sm70_awq_qwen38_indexed_prefill = bool(
            indexed_prefill_contract
            and indexed_prefill_requested
            and indexed_prefill_available
        )
        layer.sm70_awq_qwen38_active_grouped_decode = bool(
            self.sm70_moe_policy.active_grouped_decode
            and _qwen38_active_grouped_layer_contract(layer, self.group_size)
        )
        layer.sm70_awq_qwen38_qpn_m1 = initialize_qpn_m1(
            layer, _qwen38_active_grouped_layer_contract(layer, self.group_size)
        )
        w2_chunk_tokens = int(self.sm70_moe_policy.w2_chunk_tokens or 0)
        if w2_chunk_tokens not in (0, *_QWEN38_CHUNKED_W2_SUPPORTED_TOKENS):
            raise ValueError(
                "VLLM_SM70_AWQ_QWEN38_MOE_W2_CHUNK_TOKENS must be one of "
                "0, 4096, or 6144."
            )
        chunked_w2_available = hasattr(torch.ops._C, "awq_moe_chunked_w2_sm70_out")
        if (
            w2_chunk_tokens > 0
            and layer.sm70_awq_qwen38_indexed_prefill
            and not chunked_w2_available
        ):
            raise RuntimeError(
                "The SM70 Qwen3.8 AWQ chunked W2 path requires its CUDA extension."
            )
        layer.sm70_awq_qwen38_w2_chunk_tokens = int(
            w2_chunk_tokens
            if layer.sm70_awq_qwen38_indexed_prefill and chunked_w2_available
            else 0
        )

        self._allocate_buffers(layer)
        del layer.w13_qweight, layer.w13_scales, layer.w13_qzeros
        del layer.w2_qweight, layer.w2_scales, layer.w2_qzeros
        # Release unused conversion blocks between layers instead of retaining
        # their high-water mark throughout model loading. Live prepared weights
        # and inference buffers are unaffected; this is not an inference hook.
        torch.accelerator.empty_cache()
        if (
            self.sm70_moe_policy.layer_allowlist is not None
            or self.sm70_moe_policy.layer_denylist is not None
        ):
            logger.info_once(
                "SM70 AWQ MoE batched layer filter active allow=%r deny=%r.",
                self.sm70_moe_policy.layer_allowlist,
                self.sm70_moe_policy.layer_denylist,
            )
        logger.info_once(
            "SM70 AWQ MoE TurboMind %s path enabled (%d experts).",
            "batched" if batched_gemm else "per-expert dense",
            num_experts,
        )
        if compact_metadata:
            logger.info_once(
                "SM70 Qwen3.8 TP4 AWQ compact scale/zero metadata enabled."
            )

    def _allocate_buffers(self, layer: RoutedExperts) -> None:
        device = layer.w13_tm_weight.device
        top_k = self.moe.experts_per_token
        persistent_tokens = _persistent_max_tokens_for_runtime(self.sm70_moe_policy)
        max_slots = persistent_tokens * top_k
        layer._awq_moe_buf_max_tokens = persistent_tokens
        layer._awq_moe_buf_max_slots = max_slots
        layer._awq_moe_buf_top_k = top_k
        logger.info_once(
            "SM70 AWQ MoE persistent scratch cap=%d tokens (legacy ceiling=%d).",
            persistent_tokens,
            _DEFAULT_PERSISTENT_MAX_TOKENS,
        )
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
        layer._awq_moe_buf_input_row_indices = torch.empty(
            max_slots, dtype=torch.int32, device=device
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
        layer._awq_moe_buf_sorted_weights = torch.empty(
            persistent_tokens, top_k, dtype=torch.float32, device=device
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
        indexed_w13: bool,
    ) -> dict[str, torch.Tensor]:
        w2_chunk_tokens = int(layer.sm70_awq_qwen38_w2_chunk_tokens)
        chunked_w2 = _use_qwen38_chunked_w2(layer, num_tokens, indexed_w13)
        max_chunk_slots = min(num_tokens, w2_chunk_tokens) * layer._awq_moe_buf_top_k
        use_temporary_buffers = _use_temporary_buffers_for_dummy_or_capture()
        if (
            not use_temporary_buffers
            and total_slots <= layer._awq_moe_buf_max_slots
            and num_tokens <= layer._awq_moe_buf_max_tokens
        ):
            assert not chunked_w2, (
                "Qwen3.8 indexed prefill starts above the persistent AWQ MoE "
                "buffer cap."
            )
            return {
                "output": layer._awq_moe_buf_output[:num_tokens],
                "permuted_input": layer._awq_moe_buf_permuted_input[:total_slots],
                "input_row_indices": layer._awq_moe_buf_input_row_indices[:total_slots],
                "gate_up": layer._awq_moe_buf_gate_up[:total_slots],
                "intermediate": layer._awq_moe_buf_intermediate[:total_slots],
                "sorted_output": layer._awq_moe_buf_sorted_output[:total_slots],
                "expert_offsets": layer._awq_moe_buf_expert_offsets,
                "expert_offsets64": layer._awq_moe_buf_expert_offsets64,
                "inv_permuted_idx": layer._awq_moe_buf_inv_permuted_idx[:num_tokens],
                "topk_ids": layer._awq_moe_buf_topk_ids[:num_tokens],
                "sorted_weights": layer._awq_moe_buf_sorted_weights[:num_tokens],
                "token_expert_indices": layer._awq_moe_buf_token_expert_indices[
                    :num_tokens
                ],
                "permuted_idx": layer._awq_moe_buf_permuted_idx[:total_slots],
                "sorted_expert_ids": layer._awq_moe_buf_sorted_expert_ids[:total_slots],
                "sort_workspace": layer._awq_moe_buf_sort_workspace,
                "permuted_experts_id": layer._awq_moe_buf_permuted_experts_id[
                    :total_slots
                ],
                "sorted_row_idx": layer._awq_moe_buf_sorted_row_idx[:total_slots],
                "topk_ids_for_sort": layer._awq_moe_buf_topk_ids_for_sort[:total_slots],
                "active_expert_offsets": (
                    layer._awq_moe_buf_active_expert_offsets[: total_slots + 1]
                ),
                "strict_expert_offsets": layer._awq_moe_buf_strict_expert_offsets,
                "strict_expert_offsets64": (layer._awq_moe_buf_strict_expert_offsets64),
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
            "permuted_input": (
                torch.empty(
                    0,
                    layer.sm70_hidden_logical_size,
                    dtype=torch.float16,
                    device=device,
                )
                if indexed_w13
                else torch.empty(
                    total_slots,
                    layer.sm70_hidden_logical_size,
                    dtype=torch.float16,
                    device=device,
                )
            ),
            "input_row_indices": (
                torch.empty(total_slots, dtype=torch.int32, device=device)
                if indexed_w13
                else torch.empty(0, dtype=torch.int32, device=device)
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
                max_chunk_slots if chunked_w2 else total_slots,
                layer.sm70_w2_n_dim,
                dtype=torch.float16,
                device=device,
            ),
            "chunk_expert_offsets": torch.empty(
                layer.sm70_num_experts + 1 if chunked_w2 else 0,
                dtype=torch.int32,
                device=device,
            ),
            "chunk_range_begin": torch.empty(
                layer.sm70_num_experts if chunked_w2 else 0,
                dtype=torch.int32,
                device=device,
            ),
            "chunk_range_end": torch.empty(
                layer.sm70_num_experts if chunked_w2 else 0,
                dtype=torch.int32,
                device=device,
            ),
            "chunk_a_indices": torch.empty(
                max_chunk_slots if chunked_w2 else 0,
                dtype=torch.int32,
                device=device,
            ),
            "chunk_inv_permuted_idx": torch.empty(
                max_chunk_slots if chunked_w2 else 0,
                dtype=torch.int32,
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
            "sorted_weights": torch.empty(
                num_tokens, top_k, dtype=torch.float32, device=device
            ),
            "token_expert_indices": torch.arange(
                total_slots, dtype=torch.int32, device=device
            ).view(num_tokens, top_k),
            "permuted_idx": torch.empty(total_slots, dtype=torch.int32, device=device),
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
        if use_qpn_m1(layer, x, topk_weights, topk_ids_i32):
            _log_runtime_route_once(
                "SM70 AWQ Qwen3.8 QPN M1 W13/W2 enabled "
                "(existing prepared banks, direct route order)."
            )
            self.native_ops.awq_moe_qpn_m1_sm70_out(
                output,
                buffers["intermediate"],
                x,
                layer.w13_tm_weight,
                layer.w13_tm_scales,
                layer.w2_tm_weight,
                layer.w2_tm_scales,
                topk_ids_i32,
                topk_weights,
            )
            return output
        _log_runtime_route_once(
            "SM70 AWQ MoE legacy single-token monolithic compact path enabled "
            "(top_k=%d, experts=%d).",
            top_k,
            layer.sm70_num_experts,
        )
        self.native_ops.awq_moe_single_token_sm70_out(
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
            buffers["sorted_weights"].view(-1),
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
        indexed_w13 = _use_qwen38_indexed_prefill(layer, x, topk_ids)
        buffers = self._get_buffers(layer, total_slots, num_tokens, indexed_w13)
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
            and self.legacy_single_token
            and layer.sm70_awq_moe_legacy_single_token_compact
        ):
            return self._apply_legacy_single_token_compact(
                layer, x, topk_weights, topk_ids_i32, buffers, top_k, output
            )
        use_batched_single_token_strict = (
            num_tokens == 1
            and layer.sm70_awq_moe_batched_gemm
            and self.sm70_moe_policy.strict_w13
        )
        use_batched_single_token_indexed = (
            num_tokens == 1
            and layer.sm70_awq_moe_batched_gemm
            and not use_batched_single_token_strict
            and self.single_token_indexed
        )
        if num_tokens == 1 and (
            not layer.sm70_awq_moe_batched_gemm
            or use_batched_single_token_strict
            or use_batched_single_token_indexed
        ):
            plan = (
                self.strict_single_token_plan
                if use_batched_single_token_strict
                else self.batched_single_token_plan
                if use_batched_single_token_indexed
                else self.single_token_plan
            )
            return execute_single_token(
                self.weight_codec,
                plan,
                layer,
                x,
                topk_weights,
                topk_ids_i32,
                buffers,
                self.group_size,
                activation=_silu_and_mul_w13,
                observe=_dump_awq_moe_buffer,
                trim_output=True,
            )
        if indexed_w13:
            self.native_ops.moe_permute_metadata_with_scratch(
                x,
                topk_ids_i32,
                buffers["token_expert_indices"],
                layer.expert_map,
                layer.global_num_experts,
                layer.local_num_experts,
                top_k,
                buffers["expert_offsets64"],
                buffers["inv_permuted_idx"],
                buffers["permuted_idx"],
                buffers["input_row_indices"],
                buffers["sort_workspace"],
                buffers["permuted_experts_id"],
                buffers["sorted_row_idx"],
                buffers["topk_ids_for_sort"],
            )
        else:
            self.native_ops.moe_permute_with_scratch(
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
            batched_decode_max_tokens=self.sm70_moe_policy.max_batched_tokens or 0,
            strict_dense_w13=bool(self.sm70_moe_policy.strict_w13),
            exact_w2=bool(self.sm70_moe_policy.exact_w2),
            active_exact_w2=bool(self.sm70_moe_policy.active_exact_w2),
            w13_per_expert_dispatch=True,
            w2_per_expert_dispatch=True,
        )
        use_batched_moe_gemm = route_plan.use_batched_moe_gemm
        use_batched_active_exact_w2 = route_plan.use_batched_active_exact_w2
        use_active_exact_small_batched_moe = _use_qwen38_active_grouped_decode(
            layer, num_tokens, top_k, self.sm70_moe_policy
        )
        compare_step = (
            _compare_dense_decode_step(layer)
            if num_tokens <= 8 and use_batched_moe_gemm
            else None
        )
        if indexed_w13:
            _log_runtime_route_once(
                "SM70 Qwen3.8 AWQ indexed-A W13 prefill enabled "
                "(tokens=%d, routes=%d).",
                num_tokens,
                total_slots,
            )
            route_plan = replace(route_plan, w13=Sm70MoeStageRoute.INDEXED_PREFILL)
        elif use_active_exact_small_batched_moe:
            _log_runtime_route_once(
                "SM70 Qwen3.8 AWQ active grouped decode (tokens=%d, routed_slots=%d).",
                num_tokens,
                total_slots,
            )
            route_plan = replace(route_plan, w13=Sm70MoeStageRoute.ACTIVE_GROUPED)
        if use_active_exact_small_batched_moe or use_batched_active_exact_w2:
            route_plan = replace(route_plan, w2=Sm70MoeStageRoute.ACTIVE_GROUPED)
        if _use_qwen38_chunked_w2(layer, num_tokens, indexed_w13):
            chunk = int(layer.sm70_awq_qwen38_w2_chunk_tokens)
            _log_runtime_route_once(
                "SM70 Qwen3.8 AWQ chunked W2 enabled "
                "(tokens=%d, routes=%d, chunk_tokens=%d).",
                num_tokens,
                total_slots,
                chunk,
            )
            route_plan = replace(
                route_plan, w2=Sm70MoeStageRoute.CHUNKED, chunk_tokens=chunk
            )
        observer = None
        if compare_step is not None or self.sm70_moe_policy.diagnostics.dump_buffers:
            observer = AwqStageObserver(
                operators=self.native_ops,
                layer=layer,
                x=x,
                topk_weights=topk_weights,
                ids=topk_ids_i32,
                plan=route_plan,
                group_size=self.group_size,
                policy=self.sm70_moe_policy,
                indexed_w13=indexed_w13,
                active_grouped=use_active_exact_small_batched_moe,
                compare_step=compare_step,
                weighted_reduce=_single_token_weighted_reduce_enabled(
                    self.sm70_moe_policy
                ),
            )
        return execute_routed(
            self.weight_codec,
            route_plan,
            layer,
            buffers,
            x,
            topk_weights,
            self.group_size,
            layer._awq_moe_buf_dense_expert_ids,
            activation=_silu_and_mul_w13,
            observer=observer,
            trim_output=True,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, x, router_logits, input_ids
        raise NotImplementedError("SM70 AWQ MoE base path is not monolithic.")
