# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM70 TurboMind NVFP4 MoE for validated expert shapes.

The route keeps ModelOpt W4A16_NVFP4 expert weights packed. It combines the
checkpoint's FP8 block scales with its explicit ModelOpt global scales once at
load time, repacks both tensors for TurboMind, and never materializes an FP16
expert-weight copy.
"""

from __future__ import annotations

import os
from typing import Final

import torch
from torch.nn import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    MoEActivation,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    NVFP4_GROUP_SIZE,
    is_exact_sm70_cuda,
    unpack_mxfp4_weight,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_DEBUG_DFLASH_NVFP4_TRACE = bool(
    int(os.getenv("VLLM_DFLASH_DEBUG_TARGET_LAYER_TRACE", "0"))
)
_DFLASH_NVFP4_TRACE_ARMED = False


def arm_dflash_nvfp4_trace() -> None:
    global _DFLASH_NVFP4_TRACE_ARMED
    if _DEBUG_DFLASH_NVFP4_TRACE:
        _DFLASH_NVFP4_TRACE_ARMED = True


_SUPPORTED_CONTRACTS: Final = {
    # (hidden size, global expert intermediate size, experts, top-k)
    (2048, 512, 256, 8),  # Qwen3.6-35B-A3B
    (2560, 640, 512, 10),  # Qwen3.8-Flash-Next
    (4096, 2048, 288, 8),  # GLM-5.3-Flash
}
_SUPPORTED_TP_SIZES: Final = (1, 2, 4)
_GRAPH_SAFE_MAX_TOKENS: Final = 18
_COMPACT_GROUPED_MAX_TOKENS: Final = 10
_MAX_SUPPORTED_TOP_K: Final = max(contract[3] for contract in _SUPPORTED_CONTRACTS)
_QWEN38_QPN_M1_W13_SPLIT_K: Final = 8
_QWEN38_QPN_M1_W2_SPLIT_K: Final = 1
_QWEN38_INDEXED_PREFILL_MIN_TOKENS: Final = 128


def _use_qwen38_qpn_m1_decode(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    """Admit only the exact validated Qwen3.8 TP4 single-token route."""
    return bool(
        envs.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE
        and x.shape == (1, 2560)
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.shape == (1, 10)
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and int(layer.moe_config.tp_size) == 4
        and int(layer.sm70_nvfp4_num_experts) == 512
        and int(layer.sm70_nvfp4_hidden_size) == 2560
        and int(layer.sm70_nvfp4_intermediate_size) == 160
        and int(layer.sm70_nvfp4_top_k) == 10
    )


def _use_glm53_qpn_w13_q8(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    fused_permute: bool,
) -> bool:
    """Admit the exact GLM-5.3 TP8 eight-token W13 route."""
    return bool(
        fused_permute
        and getattr(layer, "sm70_glm53_qpn_w13_q8", False)
        and x.shape == (8, 4096)
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.shape == (8, 8)
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and int(layer.moe_config.tp_size) == 8
        and int(layer.sm70_nvfp4_num_experts) == 288
        and int(layer.sm70_nvfp4_hidden_size) == 4096
        and int(layer.sm70_nvfp4_intermediate_size) == 256
        and int(layer.sm70_nvfp4_top_k) == 8
    )


def _use_qwen38_indexed_prefill(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    """Admit only long exact Qwen3.8 TP4 W13 prefill batches."""
    return bool(
        getattr(
            layer,
            "sm70_nvfp4_qwen38_indexed_prefill",
            envs.VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL,
        )
        and envs.VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL
        and x.ndim == 2
        and x.shape[0] >= _QWEN38_INDEXED_PREFILL_MIN_TOKENS
        and x.shape[1] == 2560
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.shape == (x.shape[0], 10)
        and int(layer.moe_config.tp_size) == 4
        and int(layer.sm70_nvfp4_num_experts) == 512
        and int(layer.sm70_nvfp4_hidden_size) == 2560
        and int(layer.sm70_nvfp4_intermediate_size) == 160
        and int(layer.sm70_nvfp4_top_k) == 10
    )


@triton.jit
def _prepare_single_token_slots_kernel(
    input_ptr,
    topk_ids_ptr,
    expanded_input_ptr,
    active_expert_ids_ptr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(expanded_input_ptr + slot * HIDDEN + offsets, values, mask=mask)
    expert_id = tl.load(topk_ids_ptr + slot)
    tl.store(active_expert_ids_ptr + slot, expert_id.to(tl.int32))


def _prepare_single_token_slots(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    expanded_input: torch.Tensor,
    active_expert_ids: torch.Tensor,
) -> None:
    top_k = topk_ids.numel()
    hidden = x.shape[1]
    if x.shape[0] != 1 or tuple(topk_ids.shape) != (1, top_k):
        raise ValueError("SM70 NVFP4 direct routing requires one input token.")
    if tuple(expanded_input.shape) != (top_k, hidden):
        raise ValueError("SM70 NVFP4 direct routing buffer shape mismatch.")
    if active_expert_ids.numel() != top_k:
        raise ValueError("SM70 NVFP4 direct expert-ID buffer shape mismatch.")
    _prepare_single_token_slots_kernel[(top_k,)](
        x,
        topk_ids,
        expanded_input,
        active_expert_ids,
        HIDDEN=hidden,
        BLOCK=triton.next_power_of_2(hidden),
        num_warps=8,
    )


@triton.jit
def _single_token_weighted_reduce_kernel(
    expert_output_ptr,
    topk_weights_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(0, TOP_K):
        values = tl.load(
            expert_output_ptr + slot * HIDDEN + offsets,
            mask=mask,
            other=0.0,
        )
        weight = tl.load(topk_weights_ptr + slot)
        acc += values.to(tl.float32) * weight
    tl.store(output_ptr + offsets, acc, mask=mask)


def _single_token_weighted_reduce(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    top_k, hidden = expert_output.shape
    if tuple(topk_weights.shape) != (1, top_k) or tuple(output.shape) != (1, hidden):
        raise ValueError("SM70 NVFP4 direct weighted-reduce shape mismatch.")
    block = 256
    _single_token_weighted_reduce_kernel[(triton.cdiv(hidden, block),)](
        expert_output,
        topk_weights,
        output,
        HIDDEN=hidden,
        TOP_K=top_k,
        BLOCK=block,
        num_warps=4,
    )


@triton.jit
def _prepare_compact_slot_groups_kernel(
    sorted_expert_ids_ptr,
    compact_offsets_ptr,
    active_expert_ids_ptr,
    TOTAL_SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < TOTAL_SLOTS
    expert_ids = tl.load(
        sorted_expert_ids_ptr + offsets,
        mask=valid,
        other=-1,
    )
    tl.store(
        compact_offsets_ptr + offsets,
        offsets,
        mask=offsets <= TOTAL_SLOTS,
    )
    tl.store(
        active_expert_ids_ptr + offsets,
        expert_ids,
        mask=valid,
    )


def _prepare_compact_slot_groups(
    sorted_expert_ids: torch.Tensor,
    compact_offsets: torch.Tensor,
    active_expert_ids: torch.Tensor,
) -> None:
    total_slots = sorted_expert_ids.numel()
    max_slots = _COMPACT_GROUPED_MAX_TOKENS * _MAX_SUPPORTED_TOP_K
    if not (0 < total_slots <= max_slots):
        raise ValueError(f"Unsupported SM70 NVFP4 active-expert slots: {total_slots}")
    block = triton.next_power_of_2(total_slots + 1)
    # TurboMind's compact grouped dispatch forces one row per group. Keep each
    # routed slot independent even when adjacent slots select the same expert;
    # coalescing duplicate expert IDs would make the forced one-row scheduler
    # silently skip or miscompute the additional rows.
    _prepare_compact_slot_groups_kernel[(1,)](
        sorted_expert_ids,
        compact_offsets,
        active_expert_ids,
        TOTAL_SLOTS=total_slots,
        BLOCK=block,
        num_warps=1,
    )


@triton.jit
def _prepare_compact_expert_groups_kernel(
    sorted_expert_ids_ptr,
    compact_offsets_ptr,
    active_expert_ids_ptr,
    TOTAL_SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < TOTAL_SLOTS
    expert_ids = tl.load(
        sorted_expert_ids_ptr + offsets,
        mask=valid,
        other=-1,
    )
    previous_ids = tl.load(
        sorted_expert_ids_ptr + offsets - 1,
        mask=valid & (offsets > 0),
        other=-2,
    )
    is_boundary = valid & ((offsets == 0) | (expert_ids != previous_ids))
    active_indices = tl.cumsum(is_boundary.to(tl.int32), axis=0) - 1

    tl.store(
        compact_offsets_ptr + offsets,
        TOTAL_SLOTS,
        mask=offsets <= TOTAL_SLOTS,
    )
    tl.store(
        active_expert_ids_ptr + offsets,
        0,
        mask=valid,
    )
    tl.store(
        compact_offsets_ptr + active_indices,
        offsets,
        mask=is_boundary,
    )
    tl.store(
        active_expert_ids_ptr + active_indices,
        expert_ids,
        mask=is_boundary,
    )


def _prepare_compact_expert_groups(
    sorted_expert_ids: torch.Tensor,
    compact_offsets: torch.Tensor,
    active_expert_ids: torch.Tensor,
) -> None:
    total_slots = sorted_expert_ids.numel()
    max_slots = _COMPACT_GROUPED_MAX_TOKENS * _MAX_SUPPORTED_TOP_K
    if not (0 < total_slots <= max_slots):
        raise ValueError(f"Unsupported SM70 NVFP4 active-expert slots: {total_slots}")
    block = triton.next_power_of_2(total_slots + 1)
    _prepare_compact_expert_groups_kernel[(1,)](
        sorted_expert_ids,
        compact_offsets,
        active_expert_ids,
        TOTAL_SLOTS=total_slots,
        BLOCK=block,
        num_warps=1,
    )


def _use_glm53_grouped_expert_rows(
    layer: RoutedExperts,
    num_tokens: int,
) -> bool:
    return bool(
        envs.VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS
        and 1 < num_tokens <= _COMPACT_GROUPED_MAX_TOKENS
        and int(layer.moe_config.tp_size) in (4, 8)
        and int(layer.sm70_nvfp4_num_experts) == 288
        and int(layer.sm70_nvfp4_hidden_size) == 4096
        and int(layer.sm70_nvfp4_intermediate_size) in (256, 512)
        and int(layer.sm70_nvfp4_top_k) == 8
    )


def _use_glm53_fused_permute_q8(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    return bool(
        getattr(layer, "sm70_glm53_fused_permute_q8", False)
        and _use_glm53_grouped_expert_rows(layer, x.shape[0])
        and tuple(x.shape) == (8, 4096)
        and x.is_contiguous()
        and tuple(topk_ids.shape) == (8, 8)
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and topk_ids.device == x.device
    )


def validate_nvfp4_sm70_moe_contract(moe: FusedMoEConfig) -> None:
    """Reject every topology outside the validated SM70 NVFP4 contract."""
    local_intermediate = moe.intermediate_size_per_partition
    global_intermediate = local_intermediate * max(moe.tp_size, 1)
    contract = (
        moe.hidden_dim,
        global_intermediate,
        moe.num_experts,
        moe.experts_per_token,
    )
    glm53_tp8 = moe.tp_size == 8 and contract == (4096, 2048, 288, 8)
    if moe.tp_size not in _SUPPORTED_TP_SIZES and not glm53_tp8:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE currently supports tensor parallel "
            f"sizes {_SUPPORTED_TP_SIZES}, plus GLM-5.3 TP8; got {moe.tp_size}."
        )
    if local_intermediate <= 0 or local_intermediate % NVFP4_GROUP_SIZE:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE requires a positive local intermediate "
            f"size divisible by {NVFP4_GROUP_SIZE}, got {local_intermediate}."
        )
    if contract not in _SUPPORTED_CONTRACTS:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE shape is not validated: "
            f"hidden={moe.hidden_dim}, intermediate={global_intermediate}, "
            f"experts={moe.num_experts}, top_k={moe.experts_per_token}. "
            f"Validated contracts: {sorted(_SUPPORTED_CONTRACTS)}."
        )
    if moe.moe_parallel_config.use_all2all_kernels:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE does not support DP+EP all-to-all."
        )


def _validate_weight_layout(layer: RoutedExperts) -> None:
    local_experts = int(layer.local_num_experts)
    hidden = int(layer.moe_config.hidden_dim)
    intermediate = int(layer.moe_config.intermediate_size_per_partition)
    expected = {
        "w13_weight": (local_experts, 2 * intermediate, hidden // 2),
        "w13_weight_scale": (
            local_experts,
            2 * intermediate,
            hidden // NVFP4_GROUP_SIZE,
        ),
        "w13_weight_scale_2": (local_experts, 2),
        "w2_weight": (local_experts, hidden, intermediate // 2),
        "w2_weight_scale": (
            local_experts,
            hidden,
            intermediate // NVFP4_GROUP_SIZE,
        ),
        "w2_weight_scale_2": (local_experts,),
    }
    tensors = {name: getattr(layer, name) for name in expected}
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(
                f"SM70 NVFP4 MoE layout mismatch for {name}: "
                f"expected {shape}, got {tuple(tensors[name].shape)}."
            )
    if layer.w13_weight.dtype != torch.uint8 or layer.w2_weight.dtype != torch.uint8:
        raise TypeError("SM70 NVFP4 MoE requires packed uint8 expert weights.")
    if (
        layer.w13_weight_scale.dtype != torch.float8_e4m3fn
        or layer.w2_weight_scale.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("SM70 NVFP4 MoE requires FP8 E4M3 block scales.")


class ModelOptNvFp4SM70MoEMethod(ModelOptNvFp4FusedMoE):
    """ModelOpt NVFP4 experts with FP16 activations on native SM70."""

    def __init__(
        self,
        quant_config: ModelOptNvFp4Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        FusedMoEMethodBase.__init__(self, moe_config)
        if quant_config.quant_method not in {"NVFP4", "W4A16_NVFP4"}:
            raise NotImplementedError(
                "SM70 TurboMind ModelOpt NVFP4 MoE requires NVFP4-family "
                f"checkpoint weights, got {quant_config.quant_method}."
            )
        self.quant_config = quant_config
        self.use_a16 = True
        self.use_global_sf = False
        validate_nvfp4_sm70_moe_contract(moe_config)

    @property
    def supports_eplb(self) -> bool:
        return False

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        # This method owns routing, TurboMind expert GEMMs, and unpermutation.
        # Do not wrap it in the generic ModelOpt modular-kernel path.
        del routing_tables
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        required_ops = (
            "nvfp4_sm70_prepare",
            "nvfp4_moe_dense_stage_sm70_out",
            "awq_moe_build_strided_ptrs",
        )
        missing = [name for name in required_ops if not hasattr(torch.ops._C, name)]
        if (
            envs.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE
            and not sm70_ops.has_nvfp4_qpn_m1_dispatch()
        ):
            missing.append("nvfp4_moe_qpn_m1_sm70_out")
        indexed_prefill_ops = {
            "nvfp4_moe_indexed_dense_stage_sm70_out": hasattr(
                torch.ops._C, "nvfp4_moe_indexed_dense_stage_sm70_out"
            ),
            "moe_permute_metadata_with_scratch": hasattr(
                torch.ops._moe_C, "moe_permute_metadata_with_scratch"
            ),
        }
        indexed_prefill_requested = bool(
            envs.VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL
        )
        indexed_prefill_available = all(indexed_prefill_ops.values())
        indexed_prefill_explicit = (
            "VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL" in os.environ
        )
        if (
            indexed_prefill_requested
            and not indexed_prefill_available
            and indexed_prefill_explicit
        ):
            missing.extend(
                name for name, available in indexed_prefill_ops.items() if not available
            )
        elif indexed_prefill_requested and not indexed_prefill_available:
            logger.warning_once(
                "The default SM70 Qwen3.8 indexed-A prefill route is not "
                "present in the loaded extension; falling back to the "
                "materialized-input route. Explicitly setting "
                "VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL=1 fails closed."
            )
        if missing:
            raise RuntimeError(
                "SM70 NVFP4 MoE requires the TurboMind extension "
                "with " + ", ".join(missing) + "."
            )
        if not hasattr(torch.ops._moe_C, "moe_permute_with_scratch"):
            raise RuntimeError("SM70 NVFP4 MoE requires graph-safe MoE permute ops.")
        if self.moe.has_bias:
            raise NotImplementedError("SM70 NVFP4 MoE does not support expert bias.")
        if layer.activation != MoEActivation.SILU:
            raise NotImplementedError(
                "SM70 NVFP4 MoE currently supports SwiGLU/SILU only."
            )
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "SM70 NVFP4 MoE does not support router weights on input."
            )
        if layer.expert_map is not None:
            raise NotImplementedError(
                "SM70 NVFP4 MoE currently requires fully replicated experts."
            )
        if layer.local_num_experts != layer.global_num_experts:
            raise NotImplementedError(
                "SM70 NVFP4 MoE currently requires local and global experts to match."
            )

        validate_nvfp4_sm70_moe_contract(layer.moe_config)
        _validate_weight_layout(layer)
        num_experts = int(layer.local_num_experts)
        hidden = int(layer.moe_config.hidden_dim)
        intermediate = int(layer.moe_config.intermediate_size_per_partition)
        fused_swiglu_requested = bool(
            envs.VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL
            and int(layer.moe_config.tp_size) == 4
            and num_experts == 512
            and hidden == 2560
            and intermediate == 160
            and int(layer.moe_config.experts_per_token) == 10
            and layer.swiglu_limit is None
        )
        fused_swiglu_available = hasattr(
            torch.ops._C, "nvfp4_moe_indexed_fused_swiglu_sm70_out"
        )
        fused_swiglu_explicit = (
            "VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL" in os.environ
        )
        if (
            fused_swiglu_requested
            and not fused_swiglu_available
            and fused_swiglu_explicit
        ):
            raise RuntimeError(
                "SM70 Qwen3.8 fused-SwiGLU prefill requires the TurboMind "
                "extension with nvfp4_moe_indexed_fused_swiglu_sm70_out."
            )
        if fused_swiglu_requested and not fused_swiglu_available:
            logger.warning_once(
                "The default SM70 Qwen3.8 fused-SwiGLU prefill op is absent "
                "from the loaded extension; retaining the standalone "
                "activation route. Explicit opt-in fails closed."
            )
        fused_swiglu_prefill = bool(fused_swiglu_requested and fused_swiglu_available)
        fast_prefill = bool(
            fused_swiglu_prefill and envs.VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL
        )
        glm53_fused_permute_requested = bool(
            envs.VLLM_SM70_GLM53_MOE_FUSED_PERMUTE_Q8
            and int(layer.moe_config.tp_size) in (4, 8)
            and num_experts == 288
            and hidden == 4096
            and intermediate in (256, 512)
            and int(layer.moe_config.experts_per_token) == 8
        )
        glm53_fused_permute_available = hasattr(
            torch.ops._C, "sm70_glm53_moe_permute_q8_out"
        )
        glm53_fused_permute_explicit = (
            "VLLM_SM70_GLM53_MOE_FUSED_PERMUTE_Q8" in os.environ
        )
        if (
            glm53_fused_permute_requested
            and not glm53_fused_permute_available
            and glm53_fused_permute_explicit
        ):
            raise RuntimeError(
                "SM70 GLM-5.3 fused q8 MoE permute requires the TurboMind "
                "extension with sm70_glm53_moe_permute_q8_out."
            )
        if glm53_fused_permute_requested and not glm53_fused_permute_available:
            logger.warning_once(
                "The default SM70 GLM-5.3 fused q8 MoE permute op is absent "
                "from the loaded extension; retaining the generic graph-safe "
                "permute route. Explicit opt-in fails closed."
            )
        glm53_fused_permute_q8 = bool(
            glm53_fused_permute_requested and glm53_fused_permute_available
        )
        glm53_qpn_w13_requested = bool(
            envs.VLLM_SM70_GLM53_MOE_QPN_W13_Q8
            and glm53_fused_permute_q8
            and int(layer.moe_config.tp_size) == 8
            and num_experts == 288
            and hidden == 4096
            and intermediate == 256
            and int(layer.moe_config.experts_per_token) == 8
        )
        glm53_qpn_w13_available = hasattr(
            torch.ops._C, "nvfp4_glm53_moe_q8_qpn_sm70_out"
        )
        glm53_qpn_w13_explicit = "VLLM_SM70_GLM53_MOE_QPN_W13_Q8" in os.environ
        if (
            glm53_qpn_w13_requested
            and not glm53_qpn_w13_available
            and glm53_qpn_w13_explicit
        ):
            raise RuntimeError(
                "SM70 GLM-5.3 TP8 q8 W13 QPN requires the TurboMind extension "
                "with nvfp4_glm53_moe_q8_qpn_sm70_out."
            )
        if glm53_qpn_w13_requested and not glm53_qpn_w13_available:
            logger.warning_once(
                "The default SM70 GLM-5.3 TP8 q8 W13 QPN op is absent; "
                "retaining the exact TurboMind split-3 path. Explicit opt-in "
                "fails closed."
            )
        glm53_qpn_w13_q8 = bool(glm53_qpn_w13_requested and glm53_qpn_w13_available)

        w13_tm_weights: list[torch.Tensor] = []
        w13_tm_scales: list[torch.Tensor] = []
        w13_meta: list[torch.Tensor] = []
        w2_tm_weights: list[torch.Tensor] = []
        w2_tm_scales: list[torch.Tensor] = []
        w2_meta: list[torch.Tensor] = []
        for expert_id in range(num_experts):
            w13_packed = unpack_mxfp4_weight(layer.w13_weight[expert_id].data)
            w13_scales = layer.w13_weight_scale[expert_id].float().clone()
            w13_global = layer.w13_weight_scale_2[expert_id].float()
            w13_scales[:intermediate].mul_(w13_global[0])
            w13_scales[intermediate:].mul_(w13_global[1])
            prepared_w13 = sm70_ops.nvfp4_sm70_prepare(
                w13_packed,
                w13_scales.half().t().contiguous(),
                NVFP4_GROUP_SIZE,
                interleave_gated_silu=fused_swiglu_prefill,
            )
            w13_tm_weights.append(prepared_w13[0])
            w13_tm_scales.append(prepared_w13[1])
            w13_meta.append(prepared_w13[2])

            w2_packed = unpack_mxfp4_weight(layer.w2_weight[expert_id].data)
            w2_scales = (
                layer.w2_weight_scale[expert_id].float()
                * layer.w2_weight_scale_2[expert_id].float()
            )
            prepared_w2 = sm70_ops.nvfp4_sm70_prepare(
                w2_packed,
                w2_scales.half().t().contiguous(),
                NVFP4_GROUP_SIZE,
            )
            w2_tm_weights.append(prepared_w2[0])
            w2_tm_scales.append(prepared_w2[1])
            w2_meta.append(prepared_w2[2])

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False
        )
        layer.w13_tm_scales = Parameter(torch.stack(w13_tm_scales), requires_grad=False)
        layer.w2_tm_weight = Parameter(torch.stack(w2_tm_weights), requires_grad=False)
        layer.w2_tm_scales = Parameter(torch.stack(w2_tm_scales), requires_grad=False)

        w13_k_ld = int(w13_meta[0][0].item())
        w13_q_ld = int(w13_meta[0][1].item())
        w2_k_ld = int(w2_meta[0][0].item())
        w2_q_ld = int(w2_meta[0][1].item())
        w13_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w13_tm_weight,
            layer.w13_tm_scales,
            w13_k_ld,
            w13_q_ld,
            num_experts,
        )
        w2_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w2_tm_weight,
            layer.w2_tm_scales,
            w2_k_ld,
            w2_q_ld,
            num_experts,
        )
        if fast_prefill:
            # TurboMind packs this exact N320 weight as five contiguous N64
            # tiles per expert. The N256/N64 views therefore partition the
            # existing allocation without copying it. Scales remain a strided
            # N320 view, so both subprojections retain the original q_ld.
            w13_flat = layer.w13_tm_weight.view(num_experts, -1)
            head_words = hidden * 256 // 8
            w13_head_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
                w13_flat[:, :head_words],
                layer.w13_tm_scales[:, :, :256],
                w13_k_ld,
                w13_q_ld,
                num_experts,
            )
            w13_tail_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
                w13_flat[:, head_words:],
                layer.w13_tm_scales[:, :, 256:],
                w13_k_ld,
                w13_q_ld,
                num_experts,
            )
            layer.w13_head_strided_ptrs_w = Parameter(
                w13_head_ptrs[0], requires_grad=False
            )
            layer.w13_head_strided_ptrs_s = Parameter(
                w13_head_ptrs[1], requires_grad=False
            )
            layer.w13_tail_strided_ptrs_w = Parameter(
                w13_tail_ptrs[0], requires_grad=False
            )
            layer.w13_tail_strided_ptrs_s = Parameter(
                w13_tail_ptrs[1], requires_grad=False
            )
        layer.w13_strided_ptrs_w = Parameter(w13_ptrs[0], requires_grad=False)
        layer.w13_strided_ptrs_s = Parameter(w13_ptrs[1], requires_grad=False)
        layer.w2_strided_ptrs_w = Parameter(w2_ptrs[0], requires_grad=False)
        layer.w2_strided_ptrs_s = Parameter(w2_ptrs[1], requires_grad=False)

        layer.sm70_nvfp4_moe = True
        layer.sm70_nvfp4_num_experts = num_experts
        layer.sm70_nvfp4_hidden_size = hidden
        layer.sm70_nvfp4_intermediate_size = intermediate
        layer.sm70_nvfp4_top_k = int(layer.moe_config.experts_per_token)
        layer.sm70_nvfp4_w13_k_dim = hidden
        layer.sm70_nvfp4_w13_n_dim = 2 * intermediate
        layer.sm70_nvfp4_w2_k_dim = intermediate
        layer.sm70_nvfp4_w2_n_dim = hidden
        layer.sm70_nvfp4_group_size = NVFP4_GROUP_SIZE
        layer.sm70_nvfp4_qwen38_indexed_prefill = bool(
            indexed_prefill_requested and indexed_prefill_available
        )
        layer.sm70_nvfp4_qwen38_fused_swiglu_prefill = fused_swiglu_prefill
        layer.sm70_nvfp4_qwen38_fast_prefill = fast_prefill
        layer.sm70_glm53_fused_permute_q8 = glm53_fused_permute_q8
        layer.sm70_glm53_qpn_w13_q8 = glm53_qpn_w13_q8
        layer.sm70_nvfp4_graph_safe_max_tokens = _GRAPH_SAFE_MAX_TOKENS
        layer.sm70_nvfp4_compact_grouped_max_tokens = _COMPACT_GROUPED_MAX_TOKENS
        self._allocate_graph_safe_decode_buffers(layer)

        del layer.w13_weight
        del layer.w13_weight_scale
        del layer.w13_weight_scale_2
        del layer.w13_input_scale
        del layer.w2_weight
        del layer.w2_weight_scale
        del layer.w2_weight_scale_2
        del layer.w2_input_scale
        logger.info_once(
            "SM70 ModelOpt NVFP4 TurboMind MoE path enabled "
            "(hidden=%d, local_intermediate=%d, local_experts=%d, top_k=%d, "
            "graph_safe_decode=B1-B%d, compact_grouped_decode=B1-B%d).",
            hidden,
            intermediate,
            num_experts,
            layer.sm70_nvfp4_top_k,
            _GRAPH_SAFE_MAX_TOKENS,
            _COMPACT_GROUPED_MAX_TOKENS,
        )
        if fused_swiglu_prefill:
            logger.info_once(
                "SM70 Qwen3.8 indexed-A fused-SwiGLU prefill candidate "
                "enabled (interleaved W13, exact FP16 epilogue arithmetic)."
            )
        if fast_prefill:
            logger.info_once(
                "SM70 Qwen3.8 NVFP4 fast grouped prefill enabled "
                "(zero-copy W13 N256+N64, cached-B W2)."
            )
        if glm53_fused_permute_q8:
            logger.info_once(
                "SM70 GLM-5.3 exact fused q8 MoE permute enabled "
                "(M8/K8/E288 stable sort and materialized expert rows)."
            )

    def _allocate_graph_safe_decode_buffers(self, layer: RoutedExperts) -> None:
        device = layer.w13_tm_weight.device
        top_k = int(layer.sm70_nvfp4_top_k)
        max_slots = _GRAPH_SAFE_MAX_TOKENS * top_k
        experts = int(layer.sm70_nvfp4_num_experts)
        hidden = int(layer.sm70_nvfp4_hidden_size)
        intermediate = int(layer.sm70_nvfp4_intermediate_size)

        layer._nvfp4_sm70_output = torch.empty(
            _GRAPH_SAFE_MAX_TOKENS, hidden, dtype=torch.float16, device=device
        )
        layer._nvfp4_sm70_permuted_input = torch.empty(
            max_slots, hidden, dtype=torch.float16, device=device
        )
        layer._nvfp4_sm70_input_row_indices = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_gate_up = torch.empty(
            max_slots, 2 * intermediate, dtype=torch.float16, device=device
        )
        layer._nvfp4_sm70_intermediate = torch.empty(
            max_slots, intermediate, dtype=torch.float16, device=device
        )
        layer._nvfp4_sm70_sorted_output = torch.empty(
            max_slots, hidden, dtype=torch.float16, device=device
        )
        layer._nvfp4_sm70_expert_offsets = torch.empty(
            experts + 1, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_expert_offsets64 = torch.empty(
            experts + 1, dtype=torch.int64, device=device
        )
        layer._nvfp4_sm70_inv_permuted_idx = torch.empty(
            _GRAPH_SAFE_MAX_TOKENS,
            top_k,
            dtype=torch.int32,
            device=device,
        )
        layer._nvfp4_sm70_topk_ids = torch.empty(
            _GRAPH_SAFE_MAX_TOKENS,
            top_k,
            dtype=torch.int32,
            device=device,
        )
        layer._nvfp4_sm70_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(_GRAPH_SAFE_MAX_TOKENS, top_k)
        layer._nvfp4_sm70_permuted_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_permuted_experts_id = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_sorted_row_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_topk_ids_for_sort = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            max_slots, layer.global_num_experts
        )
        layer._nvfp4_sm70_sort_workspace = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
        layer._nvfp4_sm70_dense_expert_ids = torch.arange(
            experts, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_compact_offsets = torch.arange(
            max_slots + 1, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_active_expert_ids = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )

    @staticmethod
    def _persistent_buffers(
        layer: RoutedExperts, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        slots = num_tokens * int(layer.sm70_nvfp4_top_k)
        return {
            "output": layer._nvfp4_sm70_output[:num_tokens],
            "permuted_input": layer._nvfp4_sm70_permuted_input[:slots],
            "input_row_indices": layer._nvfp4_sm70_input_row_indices[:slots],
            "gate_up": layer._nvfp4_sm70_gate_up[:slots],
            "intermediate": layer._nvfp4_sm70_intermediate[:slots],
            "sorted_output": layer._nvfp4_sm70_sorted_output[:slots],
            "expert_offsets": layer._nvfp4_sm70_expert_offsets,
            "expert_offsets64": layer._nvfp4_sm70_expert_offsets64,
            "inv_permuted_idx": layer._nvfp4_sm70_inv_permuted_idx[:num_tokens],
            "topk_ids": layer._nvfp4_sm70_topk_ids[:num_tokens],
            "token_expert_indices": (
                layer._nvfp4_sm70_token_expert_indices[:num_tokens]
            ),
            "permuted_idx": layer._nvfp4_sm70_permuted_idx[:slots],
            "sort_workspace": layer._nvfp4_sm70_sort_workspace,
            "permuted_experts_id": layer._nvfp4_sm70_permuted_experts_id[:slots],
            "sorted_row_idx": layer._nvfp4_sm70_sorted_row_idx[:slots],
            "topk_ids_for_sort": layer._nvfp4_sm70_topk_ids_for_sort[:slots],
            "dense_expert_ids": layer._nvfp4_sm70_dense_expert_ids,
            "compact_offsets": layer._nvfp4_sm70_compact_offsets[: slots + 1],
            "active_expert_ids": layer._nvfp4_sm70_active_expert_ids[:slots],
        }

    @staticmethod
    def _eager_buffers(
        layer: RoutedExperts, num_tokens: int, indexed_w13: bool
    ) -> dict[str, torch.Tensor]:
        device = layer.w13_tm_weight.device
        top_k = int(layer.sm70_nvfp4_top_k)
        slots = num_tokens * top_k
        experts = int(layer.sm70_nvfp4_num_experts)
        hidden = int(layer.sm70_nvfp4_hidden_size)
        intermediate = int(layer.sm70_nvfp4_intermediate_size)
        workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            slots, layer.global_num_experts
        )
        return {
            "output": torch.empty(
                num_tokens, hidden, dtype=torch.float16, device=device
            ),
            "permuted_input": (
                torch.empty(0, hidden, dtype=torch.float16, device=device)
                if indexed_w13
                else torch.empty(slots, hidden, dtype=torch.float16, device=device)
            ),
            "input_row_indices": (
                torch.empty(slots, dtype=torch.int32, device=device)
                if indexed_w13
                else torch.empty(0, dtype=torch.int32, device=device)
            ),
            "gate_up": torch.empty(
                slots, 2 * intermediate, dtype=torch.float16, device=device
            ),
            "intermediate": torch.empty(
                slots, intermediate, dtype=torch.float16, device=device
            ),
            "sorted_output": torch.empty(
                slots, hidden, dtype=torch.float16, device=device
            ),
            "expert_offsets": torch.empty(
                experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                experts + 1, dtype=torch.int64, device=device
            ),
            "inv_permuted_idx": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "topk_ids": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "token_expert_indices": torch.arange(
                slots, dtype=torch.int32, device=device
            ).view(num_tokens, top_k),
            "permuted_idx": torch.empty(slots, dtype=torch.int32, device=device),
            "sort_workspace": torch.empty(
                workspace_size, dtype=torch.int8, device=device
            ),
            "permuted_experts_id": torch.empty(slots, dtype=torch.int32, device=device),
            "sorted_row_idx": torch.empty(slots, dtype=torch.int32, device=device),
            "topk_ids_for_sort": torch.empty(slots, dtype=torch.int32, device=device),
            "dense_expert_ids": layer._nvfp4_sm70_dense_expert_ids,
            "compact_offsets": torch.arange(
                slots + 1, dtype=torch.int32, device=device
            ),
            "active_expert_ids": torch.empty(slots, dtype=torch.int32, device=device),
        }

    def _get_buffers(
        self, layer: RoutedExperts, num_tokens: int, indexed_w13: bool
    ) -> dict[str, torch.Tensor]:
        if 0 < num_tokens <= _GRAPH_SAFE_MAX_TOKENS:
            return self._persistent_buffers(layer, num_tokens)
        return self._eager_buffers(layer, num_tokens, indexed_w13)

    @staticmethod
    def _apply_swiglu(
        layer: RoutedExperts,
        out: torch.Tensor,
        gate_up: torch.Tensor,
        *,
        interleaved: bool = False,
    ) -> None:
        if interleaved:
            if layer.swiglu_limit is not None:
                raise RuntimeError(
                    "Interleaved SM70 NVFP4 SwiGLU does not support clamping."
                )
            torch.ops._C.silu_and_mul_interleaved(out, gate_up)
            return
        if layer.swiglu_limit is None:
            torch.ops._C.silu_and_mul(out, gate_up)
        else:
            torch.ops._C.silu_and_mul_with_clamp(
                out, gate_up, float(layer.swiglu_limit)
            )

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
        if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2:
            raise TypeError("SM70 NVFP4 MoE requires CUDA FP16 activations [M, H].")
        if not is_exact_sm70_cuda(x, enabled=True):
            raise RuntimeError("SM70 NVFP4 MoE dispatch is restricted to CUDA SM70.")
        hidden = int(layer.sm70_nvfp4_hidden_size)
        top_k = int(layer.sm70_nvfp4_top_k)
        if x.shape[1] != hidden:
            raise ValueError(
                "SM70 NVFP4 MoE activation hidden size mismatch: expected "
                f"{hidden}, got {x.shape[1]}."
            )
        if tuple(topk_ids.shape) != (x.shape[0], top_k):
            raise ValueError(
                "SM70 NVFP4 MoE top-k ID shape mismatch: expected "
                f"{(x.shape[0], top_k)}, got {tuple(topk_ids.shape)}."
            )
        if tuple(topk_weights.shape) != tuple(topk_ids.shape):
            raise ValueError("SM70 NVFP4 MoE top-k weights and IDs must share shape.")
        if topk_weights.dtype != torch.float32:
            raise TypeError("SM70 NVFP4 MoE requires float32 top-k weights.")

        num_tokens = x.shape[0]
        if num_tokens == 0:
            return x.new_empty((0, hidden))
        indexed_w13 = _use_qwen38_indexed_prefill(layer, x, topk_ids)
        interleaved_w13 = bool(
            getattr(layer, "sm70_nvfp4_qwen38_fused_swiglu_prefill", False)
        )
        fused_indexed_w13 = indexed_w13 and interleaved_w13
        split_fused_indexed_w13 = fused_indexed_w13 and bool(
            getattr(layer, "sm70_nvfp4_qwen38_fast_prefill", False)
        )
        glm53_fused_permute_q8 = _use_glm53_fused_permute_q8(layer, x, topk_ids)
        glm53_qpn_w13_q8 = _use_glm53_qpn_w13_q8(
            layer,
            x,
            topk_ids,
            fused_permute=glm53_fused_permute_q8,
        )
        buffers = self._get_buffers(layer, num_tokens, indexed_w13)
        output = buffers["output"]
        slots = num_tokens * top_k
        direct_single_token = num_tokens == 1
        if _use_qwen38_qpn_m1_decode(layer, x, topk_ids):
            logger.info_once(
                "SM70 Qwen3.8 NVFP4 direct QPN-M1 expert path enabled "
                "(TP4, E512/K10, W13 split8, W2 split1)."
            )
            route_ids = topk_ids.view(-1)
            sm70_ops.nvfp4_moe_qpn_m1_sm70_out(
                buffers["gate_up"],
                x,
                layer.w13_tm_weight,
                layer.w13_tm_scales,
                route_ids,
                True,
                _QWEN38_QPN_M1_W13_SPLIT_K,
            )
            self._apply_swiglu(
                layer,
                buffers["intermediate"],
                buffers["gate_up"],
                interleaved=interleaved_w13,
            )
            sm70_ops.nvfp4_moe_qpn_m1_sm70_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                layer.w2_tm_weight,
                layer.w2_tm_scales,
                route_ids,
                False,
                _QWEN38_QPN_M1_W2_SPLIT_K,
            )
            _single_token_weighted_reduce(
                buffers["sorted_output"], topk_weights, output
            )
            return output
        if direct_single_token:
            _prepare_single_token_slots(
                x,
                topk_ids,
                buffers["permuted_input"],
                buffers["active_expert_ids"],
            )
            stage_offsets = buffers["compact_offsets"]
            stage_expert_ids = buffers["active_expert_ids"]
            stage_experts = top_k
        elif glm53_fused_permute_q8:
            sm70_ops.sm70_glm53_moe_permute_q8_out(
                x,
                topk_ids,
                buffers["permuted_input"],
                buffers["sorted_row_idx"],
                buffers["inv_permuted_idx"],
                buffers["compact_offsets"],
                buffers["active_expert_ids"],
            )
            stage_offsets = buffers["compact_offsets"]
            stage_expert_ids = buffers["active_expert_ids"]
            stage_experts = slots
        else:
            output.zero_()
            topk_ids_i32 = buffers["topk_ids"]
            topk_ids_i32.copy_(topk_ids, non_blocking=True)
            buffers["permuted_idx"].fill_(slots)
            if indexed_w13:
                torch.ops._moe_C.moe_permute_metadata_with_scratch(
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
            buffers["expert_offsets"].copy_(
                buffers["expert_offsets64"], non_blocking=True
            )

        if (
            not direct_single_token
            and not glm53_fused_permute_q8
            and num_tokens <= _COMPACT_GROUPED_MAX_TOKENS
        ):
            prepare_groups = (
                _prepare_compact_expert_groups
                if _use_glm53_grouped_expert_rows(layer, num_tokens)
                else _prepare_compact_slot_groups
            )
            prepare_groups(
                buffers["permuted_experts_id"],
                buffers["compact_offsets"],
                buffers["active_expert_ids"],
            )
            stage_offsets = buffers["compact_offsets"]
            stage_expert_ids = buffers["active_expert_ids"]
            stage_experts = slots
        elif not direct_single_token and not glm53_fused_permute_q8:
            stage_offsets = buffers["expert_offsets"]
            stage_expert_ids = buffers["dense_expert_ids"]
            stage_experts = int(layer.sm70_nvfp4_num_experts)

        if split_fused_indexed_w13:
            logger.info_once(
                "SM70 Qwen3.8 NVFP4 indexed-A fused-SwiGLU split-W13 "
                "prefill route enabled (N256+N64)."
            )
            for intermediate, ptrs_w, ptrs_s, n in (
                (
                    buffers["intermediate"][:, :128],
                    layer.w13_head_strided_ptrs_w,
                    layer.w13_head_strided_ptrs_s,
                    256,
                ),
                (
                    buffers["intermediate"][:, 128:],
                    layer.w13_tail_strided_ptrs_w,
                    layer.w13_tail_strided_ptrs_s,
                    64,
                ),
            ):
                sm70_ops.nvfp4_moe_indexed_fused_swiglu_sm70_out(
                    intermediate,
                    x,
                    buffers["input_row_indices"],
                    stage_offsets,
                    stage_expert_ids,
                    ptrs_w,
                    ptrs_s,
                    stage_experts,
                    layer.sm70_nvfp4_w13_k_dim,
                    n,
                    layer.sm70_nvfp4_group_size,
                )
        elif fused_indexed_w13:
            logger.info_once(
                "SM70 Qwen3.8 NVFP4 indexed-A fused-SwiGLU W13 prefill "
                "candidate enabled."
            )
            sm70_ops.nvfp4_moe_indexed_fused_swiglu_sm70_out(
                buffers["intermediate"],
                x,
                buffers["input_row_indices"],
                stage_offsets,
                stage_expert_ids,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                stage_experts,
                layer.sm70_nvfp4_w13_k_dim,
                layer.sm70_nvfp4_w13_n_dim,
                layer.sm70_nvfp4_group_size,
            )
        elif indexed_w13:
            logger.info_once(
                "SM70 Qwen3.8 NVFP4 indexed-A W13 prefill route enabled "
                "(TP4, E512/K10, materialized input rows skipped)."
            )
            sm70_ops.nvfp4_moe_indexed_dense_stage_sm70_out(
                buffers["gate_up"],
                x,
                buffers["input_row_indices"],
                stage_offsets,
                stage_expert_ids,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                stage_experts,
                layer.sm70_nvfp4_w13_k_dim,
                layer.sm70_nvfp4_w13_n_dim,
                layer.sm70_nvfp4_group_size,
            )
        elif glm53_qpn_w13_q8:
            logger.info_once(
                "SM70 GLM-5.3 TP8 q8 exact W13 QPN path enabled "
                "(CTA-K32 split-3 accumulation tree)."
            )
            sm70_ops.nvfp4_glm53_moe_q8_qpn_sm70_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                layer.w13_tm_weight,
                layer.w13_tm_scales,
                topk_ids.view(-1),
                buffers["sorted_row_idx"],
                True,
            )
        else:
            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                stage_offsets,
                stage_expert_ids,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                stage_experts,
                layer.sm70_nvfp4_w13_k_dim,
                layer.sm70_nvfp4_w13_n_dim,
                layer.sm70_nvfp4_group_size,
            )
        if not fused_indexed_w13:
            self._apply_swiglu(
                layer,
                buffers["intermediate"],
                buffers["gate_up"],
                interleaved=interleaved_w13,
            )
        sm70_ops.nvfp4_moe_dense_stage_sm70_out(
            buffers["sorted_output"],
            buffers["intermediate"],
            stage_offsets,
            stage_expert_ids,
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            stage_experts,
            layer.sm70_nvfp4_w2_k_dim,
            layer.sm70_nvfp4_w2_n_dim,
            layer.sm70_nvfp4_group_size,
        )
        if direct_single_token:
            _single_token_weighted_reduce(
                buffers["sorted_output"], topk_weights, output
            )
        else:
            torch.ops._moe_C.moe_unpermute(
                buffers["sorted_output"],
                topk_weights,
                buffers["inv_permuted_idx"],
                None if glm53_fused_permute_q8 else buffers["expert_offsets64"],
                top_k,
                output,
            )
        global _DFLASH_NVFP4_TRACE_ARMED
        if _DFLASH_NVFP4_TRACE_ARMED and num_tokens > 1:
            _DFLASH_NVFP4_TRACE_ARMED = False
            actual = output.clone()
            reference_rows = []
            for token_idx in range(num_tokens):
                reference_rows.append(
                    self.apply(
                        layer,
                        x[token_idx : token_idx + 1],
                        topk_weights[token_idx : token_idx + 1],
                        topk_ids[token_idx : token_idx + 1],
                        None,
                        None,
                    ).clone()
                )
            reference = torch.cat(reference_rows, dim=0)
            delta = (actual - reference).float().reshape(num_tokens, -1)
            logger.warning(
                "DFLASH_TARGET_NVFP4_SEQUENCE_DELTA tokens=%d "
                "max_by_token=%s mean_by_token=%s sqsum_by_token=%s "
                "routes=%s",
                num_tokens,
                delta.abs().amax(dim=1).cpu().tolist(),
                delta.abs().mean(dim=1).cpu().tolist(),
                (delta * delta).sum(dim=1).cpu().tolist(),
                topk_ids.detach().cpu().tolist(),
            )
            output.copy_(actual)
        return output

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, x, router_logits, input_ids
        raise NotImplementedError("SM70 NVFP4 MoE is not a monolithic route.")

    def get_fused_moe_quant_config(  # type: ignore[override]
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None
