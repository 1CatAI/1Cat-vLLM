# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM70 TurboMind NVFP4 MoE for validated expert shapes.

The route keeps ModelOpt W4A16_NVFP4 expert weights packed. It combines the
checkpoint's FP8 block scales with its explicit ModelOpt global scales once at
load time, repacks both tensors for TurboMind, and never materializes an FP16
expert-weight copy.
"""

from __future__ import annotations

from typing import Final

import torch
from torch.nn import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm._sm70.policy import NativeBindings
from vllm.config.sm70_moe import Sm70NvFp4MoEConfig, capture_nvfp4_moe_config
from vllm.config.sm70_runtime import capture_runtime_trace
from vllm.diagnostics import diagnostic_history
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers import sm70_fuse47 as _fuse47
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    MoEActivation,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.sm70.fp4_codec import Fp4MoECodec
from vllm.model_executor.layers.fused_moe.sm70.fp4_stages import (
    apply_swiglu,
    execute_fp4,
)
from vllm.model_executor.layers.fused_moe.sm70.fp4_workspace import (
    NvFp4MoEWorkspace,
    _get_qwen38_raw_scale_workspace,
)
from vllm.model_executor.layers.fused_moe.sm70.fp4_workspace import (
    clear_sm70_nvfp4_moe_workspaces as clear_sm70_nvfp4_moe_workspaces,
)
from vllm.model_executor.layers.fused_moe.sm70.reduction import (
    _mtp_weighted_reduce as _mtp_weighted_reduce,
)
from vllm.model_executor.layers.fused_moe.sm70.reduction import (
    _single_token_weighted_reduce as _single_token_weighted_reduce,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    Sm70MoeStageRoute as Stage,
)
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    select_fp4_stage_plan,
)
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    NVFP4_GROUP_SIZE,
    is_exact_sm70_cuda,
    unpack_mxfp4_weight,
)
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    LayerWorkspaceView,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


def arm_dflash_nvfp4_trace() -> None:
    if capture_runtime_trace().dflash.value("target_layer_trace"):
        diagnostic_history("nvfp4_dflash")["armed"] = True


_SUPPORTED_CONTRACTS: Final = {
    # (hidden size, global expert intermediate size, experts, top-k)
    (2048, 512, 256, 8),  # Qwen3.6-35B-A3B
    (2560, 640, 512, 10),  # Qwen3.8-Flash-Next
    (4096, 2048, 288, 8),  # GLM-5.3-Flash
}
_SUPPORTED_TP_SIZES: Final = (1, 2, 4)
_GRAPH_SAFE_MAX_TOKENS: Final = 18
_COMPACT_GROUPED_MAX_SLOTS: Final = 80
# V100 real-shape M=1 tuning favors 16 warps. This retains checkpoint NVFP4,
# FP32 MMA accumulation, and the FP16 W13 output boundary; only the order in
# which the FP32 K partitions are joined changes from the former split-8 plan.
_QWEN38_QPN_M1_W13_SPLIT_K: Final = 16
_QWEN38_QPN_M1_W2_SPLIT_K: Final = 1
_QWEN38_INDEXED_PREFILL_MIN_TOKENS: Final = 128
_QWEN38_QPN_MTP5_W13_SPLIT_K: Final = 4
# Split choices retained by the per-width performance screen. The direct
# kernel is deterministic, but TurboMind's grouped baseline can autotune to a
# different reduction order in a new process; quality admission therefore
# uses an independent FP32 oracle and endpoint datasets, not baseline bitwise
# identity alone.
_QWEN38_QPN_BATCH_W13_SPLIT_K: Final = {2: 10, 4: 5, 8: 4, 16: 1}
_QWEN38_DYNAMIC_QPN_BATCH_W13_SPLIT_K: Final = {
    2: 10,
    3: 8,
    4: 5,
    5: 4,
    6: 8,
    7: 8,
    8: 4,
    9: 4,
    10: 4,
    11: 4,
    12: 4,
    13: 5,
    14: 5,
    15: 4,
    16: 1,
}
_QWEN38_QPN_BATCH_FUSED_W13_TOKENS: Final = frozenset((4, 8, 16))


def _raw_scales_match_prepared(
    workspace: torch.Tensor,
    codes: torch.Tensor,
    globals_: torch.Tensor,
    prepared: torch.Tensor,
    interleaved: bool,
) -> bool:
    """Admit both prefill scales and the effective QPN decode scales."""
    sm70_ops.nvfp4_expand_raw_scales_sm70_out(
        workspace, codes, globals_, interleaved, False
    )
    if not torch.equal(workspace, prepared):
        return False
    sm70_ops.nvfp4_expand_raw_scales_sm70_out(
        workspace, codes, globals_, interleaved, True
    )
    return torch.equal(workspace, prepared * 16384.0)


def _nvfp4_policy(layer) -> Sm70NvFp4MoEConfig:
    policy = getattr(layer, "sm70_moe_policy", None)
    # Compatibility for direct helper callers; loaded layers always own a policy.
    return policy if policy is not None else capture_nvfp4_moe_config()


def _use_qwen38_qpn_m1_decode(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    """Admit only the exact validated Qwen3.8 TP4 single-token route."""
    return bool(
        _nvfp4_policy(layer).qpn_m1
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
            _nvfp4_policy(layer).indexed_prefill,
        )
        and _nvfp4_policy(layer).grouped_prefill
        and x.ndim == 2
        and x.shape[0] >= _QWEN38_INDEXED_PREFILL_MIN_TOKENS
        and x.shape[1] == 2560
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.shape == (x.shape[0], 10)
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and int(layer.moe_config.tp_size) == 4
        and int(layer.sm70_nvfp4_num_experts) == 512
        and int(layer.sm70_nvfp4_hidden_size) == 2560
        and int(layer.sm70_nvfp4_intermediate_size) == 160
        and int(layer.sm70_nvfp4_top_k) == 10
    )


def _use_qwen38_qpn_batch_decode(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    """Admit the screened Qwen3.8 TP4 no-MTP CUDA Graph batch widths."""
    tokens = x.shape[0]
    split_table = (
        _QWEN38_DYNAMIC_QPN_BATCH_W13_SPLIT_K
        if _nvfp4_policy(layer).qpn_dynamic
        else _QWEN38_QPN_BATCH_W13_SPLIT_K
    )
    return bool(
        _nvfp4_policy(layer).qpn_batch
        and tokens in split_table
        and x.shape == (tokens, 2560)
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.shape == (tokens, 10)
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and int(layer.moe_config.tp_size) == 4
        and int(layer.sm70_nvfp4_num_experts) == 512
        and int(layer.sm70_nvfp4_hidden_size) == 2560
        and int(layer.sm70_nvfp4_intermediate_size) == 160
        and int(layer.sm70_nvfp4_top_k) == 10
    )


def _nvfp4_batch_capability(layer, stage: str) -> bool:
    capabilities = getattr(layer, "sm70_nvfp4_batch_capabilities", None)
    if capabilities is not None:
        return capabilities[stage]
    # Compatibility helper callers have no load lifecycle.
    return bool(
        getattr(
            sm70_ops,
            "has_nvfp4_qpn_"
            + ("w13_swiglu_batch_dispatch" if stage == "w13" else "w2_reduce_dispatch"),
        )()
    )


def _use_qwen38_qpn_batch_fused_w13(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    """Admit the retained split-preserving W13+SwiGLU fusion widths."""
    return bool(
        _nvfp4_policy(layer).fused_batch_w13
        and x.shape[0] in _QWEN38_QPN_BATCH_FUSED_W13_TOKENS
        and layer.swiglu_limit is None
        and _nvfp4_batch_capability(layer, "w13")
        and _use_qwen38_qpn_batch_decode(layer, x, topk_ids)
    )


def _grouped_decode_context_ok() -> bool:
    """Use CPU metadata, never GPU readback, to exclude prefill and verify."""
    if not is_forward_context_available():
        return False
    context = get_forward_context()
    metadata = context.attn_metadata
    # DBO/list metadata needs per-microbatch ownership, not a shared decision.
    if not isinstance(metadata, dict) or not metadata:
        return False
    key = "sm70_grouped_moe_decode"
    if key not in context.additional_kwargs:
        seen_decode = False
        allowed = True
        for meta in metadata.values():
            prefills = getattr(meta, "num_prefills", 0)
            prefill_tokens = getattr(meta, "num_prefill_tokens", 0)
            if (
                not isinstance(prefills, int)
                or not isinstance(prefill_tokens, int)
                or prefills != 0
                or prefill_tokens != 0
            ):
                allowed = False
                break
            max_query = getattr(meta, "max_query_len", None)
            if max_query is not None:
                if not isinstance(max_query, int) or max_query != 1:
                    allowed = False
                    break
                seen_decode = True
            num_decodes = getattr(meta, "num_decodes", None)
            if num_decodes is not None:
                decode_tokens = getattr(meta, "num_decode_tokens", None)
                if (
                    not isinstance(num_decodes, int)
                    or not isinstance(decode_tokens, int)
                    or decode_tokens != num_decodes
                ):
                    allowed = False
                    break
                seen_decode |= num_decodes > 0
        context.additional_kwargs[key] = bool(allowed and seen_decode)
    return bool(context.additional_kwargs[key])


def _use_grouped_decode(layer, x: torch.Tensor, topk_ids: torch.Tensor) -> bool:
    """Local operator contract only; no TP/KV/scheduler/model-name binding."""
    return bool(
        getattr(layer, "sm70_nvfp4_grouped_decode", False)
        and x.ndim == 2
        and x.shape[0] in (8, 16)
        and x.shape[1] == 2560
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.shape == (x.shape[0], 10)
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and _grouped_decode_context_ok()
    )


def _use_qwen38_qpn_batch_fused_w2(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    """Admit the fixed-order parallel W2 reduction for direct batch QPN."""
    return bool(
        _nvfp4_policy(layer).fused_batch_w2
        and _nvfp4_batch_capability(layer, "w2")
        and _use_qwen38_qpn_batch_decode(layer, x, topk_ids)
    )


def _use_qwen38_qpn_mtp5_decode(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    """Admit only the exact Qwen3.8 TP4 MTP4 verifier route."""
    return bool(
        _nvfp4_policy(layer).qpn_mtp5
        and x.shape == (5, 2560)
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.shape == (5, 10)
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and int(layer.moe_config.tp_size) == 4
        and int(layer.sm70_nvfp4_num_experts) == 512
        and int(layer.sm70_nvfp4_hidden_size) == 2560
        and int(layer.sm70_nvfp4_intermediate_size) == 160
        and int(layer.sm70_nvfp4_top_k) == 10
    )


def _use_grouped_mtp5(layer, x: torch.Tensor, topk_ids: torch.Tensor) -> bool:
    """Keep MTP's admitted split4 arithmetic while sharing expert weights."""
    return bool(
        getattr(layer, "sm70_nvfp4_grouped_mtp5", False)
        and _use_qwen38_qpn_mtp5_decode(layer, x, topk_ids)
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
    if not (0 < total_slots <= _COMPACT_GROUPED_MAX_SLOTS):
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
    max_slots = _COMPACT_GROUPED_MAX_SLOTS
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
        _nvfp4_policy(layer).grouped_expert_rows
        and num_tokens > 1
        and _use_compact_grouped(num_tokens, int(layer.sm70_nvfp4_top_k))
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


def _use_compact_grouped(num_tokens: int, top_k: int) -> bool:
    """Bound compact dispatch by its routed-row workload, not batch size."""
    total_slots = num_tokens * top_k
    return 0 < total_slots <= _COMPACT_GROUPED_MAX_SLOTS


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
        self.sm70_moe_policy = capture_nvfp4_moe_config()
        self.native_ops = NativeBindings(self.sm70_moe_policy.native.values)
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
        layer.sm70_moe_policy = self.sm70_moe_policy
        layer.sm70_nvfp4_batch_capabilities = {
            "w13": self.native_ops.has_nvfp4_qpn_w13_swiglu_batch_dispatch(),
            "w2": self.native_ops.has_nvfp4_qpn_w2_reduce_dispatch(),
        }
        required_ops = (
            "nvfp4_sm70_prepare",
            "nvfp4_moe_dense_stage_sm70_out",
            "awq_moe_build_strided_ptrs",
        )
        missing = [name for name in required_ops if not hasattr(torch.ops._C, name)]
        if (
            _nvfp4_policy(layer).qpn_m1 or _nvfp4_policy(layer).qpn_batch
        ) and not self.native_ops.has_nvfp4_qpn_m1_dispatch():
            missing.append("nvfp4_moe_qpn_m1_sm70_out")
        w2_direct_reduce_requested = bool(_nvfp4_policy(layer).w2_direct_reduce)
        w2_direct_reduce_available = self.native_ops.has_nvfp4_qwen38_w2_direct_reduce()
        w2_direct_reduce_explicit = (
            "w2_direct_reduce" in _nvfp4_policy(layer).explicit_fields
        )
        if (
            w2_direct_reduce_requested
            and not w2_direct_reduce_available
            and w2_direct_reduce_explicit
        ):
            missing.append("nvfp4_qwen38_w2_direct_reduce_out")
        elif w2_direct_reduce_requested and not w2_direct_reduce_available:
            logger.warning_once(
                "The default SM70 Qwen3.8 W2 direct-reduce op is absent from "
                "the loaded extension; retaining separate W2 and weighted "
                "reduce kernels. Explicit opt-in fails closed."
            )
        if (
            _nvfp4_policy(layer).qpn_mtp5
            and not self.native_ops.has_nvfp4_qpn_mtp5_dispatch()
        ):
            missing.append("nvfp4_moe_qpn_mtp5_sm70_out")
        indexed_prefill_ops = {
            "nvfp4_moe_indexed_dense_stage_sm70_out": hasattr(
                torch.ops._C, "nvfp4_moe_indexed_dense_stage_sm70_out"
            ),
            "moe_permute_metadata_with_scratch": hasattr(
                torch.ops._moe_C, "moe_permute_metadata_with_scratch"
            ),
        }
        indexed_prefill_requested = bool(_nvfp4_policy(layer).indexed_prefill)
        indexed_prefill_available = all(indexed_prefill_ops.values())
        indexed_prefill_explicit = (
            "indexed_prefill" in _nvfp4_policy(layer).explicit_fields
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
            _nvfp4_policy(layer).fused_swiglu_prefill
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
            "fused_swiglu_prefill" in _nvfp4_policy(layer).explicit_fields
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
        fused_swiglu_decode = bool(
            _nvfp4_policy(layer).qpn_m1
            and fused_swiglu_prefill
            and self.native_ops.has_nvfp4_qwen38_w13_fused_swiglu()
        )
        if (
            _nvfp4_policy(layer).qpn_m1
            and fused_swiglu_prefill
            and not fused_swiglu_decode
        ):
            logger.warning_once(
                "The SM70 Qwen3.8 fused W13/SwiGLU decode op is absent; "
                "retaining separate exact W13 and activation kernels."
            )
        fast_prefill = bool(fused_swiglu_prefill and _nvfp4_policy(layer).fast_prefill)
        raw_scale_requested = bool(
            _nvfp4_policy(layer).raw_scale
            and int(layer.moe_config.tp_size) == 4
            and num_experts == 512
            and hidden == 2560
            and intermediate == 160
            and int(layer.moe_config.experts_per_token) == 10
        )
        raw_scale_available = self.native_ops.has_nvfp4_qpn_raw_scale_dispatch()
        raw_scale_explicit = "raw_scale" in _nvfp4_policy(layer).explicit_fields
        if raw_scale_requested and not raw_scale_available and raw_scale_explicit:
            raise RuntimeError(
                "SM70 Qwen3.8 raw E4M3 scale storage requires the matching "
                "QPN decode and prefill-expansion operators."
            )
        if raw_scale_requested and not raw_scale_available:
            logger.warning_once(
                "The default SM70 Qwen3.8 raw E4M3 scale operators are not "
                "present; retaining persistent FP16 prepared scales. An "
                "explicit opt-in fails closed."
            )
        raw_scale = bool(raw_scale_requested and raw_scale_available)

        glm53_fused_permute_requested = bool(
            _nvfp4_policy(layer).glm53_fused_permute
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
            "glm53_fused_permute" in _nvfp4_policy(layer).explicit_fields
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
            _nvfp4_policy(layer).glm53_qpn_w13
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
        glm53_qpn_w13_explicit = "glm53_qpn_w13" in _nvfp4_policy(layer).explicit_fields
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
        w13_raw_scale_codes: list[torch.Tensor] = []
        w13_raw_global_scales: list[torch.Tensor] = []
        w13_meta: list[torch.Tensor] = []
        w2_tm_weights: list[torch.Tensor] = []
        w2_tm_scales: list[torch.Tensor] = []
        w2_raw_scale_codes: list[torch.Tensor] = []
        w2_raw_global_scales: list[torch.Tensor] = []
        w2_meta: list[torch.Tensor] = []
        for expert_id in range(num_experts):
            w13_packed = unpack_mxfp4_weight(layer.w13_weight[expert_id].data)
            w13_scales = layer.w13_weight_scale[expert_id].float().clone()
            w13_global = layer.w13_weight_scale_2[expert_id].float()
            w13_scales[:intermediate].mul_(w13_global[0])
            w13_scales[intermediate:].mul_(w13_global[1])
            prepared_w13 = Fp4MoECodec.prepare_weights(
                "nvfp4",
                w13_packed,
                w13_scales.half().t().contiguous(),
                NVFP4_GROUP_SIZE,
                interleave_gated_silu=fused_swiglu_prefill,
            )
            w13_tm_weights.append(prepared_w13[0])
            w13_tm_scales.append(prepared_w13[1])
            if raw_scale:
                if fused_swiglu_prefill:
                    physical_global = w13_global.repeat(
                        intermediate // 32
                    ).repeat_interleave(32)
                else:
                    physical_global = w13_global.repeat_interleave(intermediate)
                w13_raw_scale_codes.append(
                    (prepared_w13[1].float() / physical_global[None, :].float())
                    .to(torch.float8_e4m3fn)
                    .view(torch.uint8)
                    .contiguous()
                )
                w13_raw_global_scales.append(w13_global.contiguous())
            w13_meta.append(prepared_w13[2])

            w2_packed = unpack_mxfp4_weight(layer.w2_weight[expert_id].data)
            w2_global = layer.w2_weight_scale_2[expert_id].float().reshape(())
            w2_scales = layer.w2_weight_scale[expert_id].float() * w2_global
            prepared_w2 = Fp4MoECodec.prepare_weights(
                "nvfp4",
                w2_packed,
                w2_scales.half().t().contiguous(),
                NVFP4_GROUP_SIZE,
            )
            w2_tm_weights.append(prepared_w2[0])
            w2_tm_scales.append(prepared_w2[1])
            if raw_scale:
                w2_raw_scale_codes.append(
                    (prepared_w2[1].float() / w2_global)
                    .to(torch.float8_e4m3fn)
                    .view(torch.uint8)
                    .contiguous()
                )
                w2_raw_global_scales.append(w2_global.reshape(1))
            w2_meta.append(prepared_w2[2])

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False
        )
        layer.w2_tm_weight = Parameter(torch.stack(w2_tm_weights), requires_grad=False)
        prepared_w13_scales = torch.stack(w13_tm_scales)
        prepared_w2_scales = torch.stack(w2_tm_scales)
        if raw_scale:
            raw_w13_codes = torch.stack(w13_raw_scale_codes)
            raw_w13_globals = torch.stack(w13_raw_global_scales).float()
            raw_w2_codes = torch.stack(w2_raw_scale_codes)
            raw_w2_globals = torch.stack(w2_raw_global_scales).float()
            scale_workspace = _get_qwen38_raw_scale_workspace(
                layer.w13_tm_weight.device
            )
            w13_workspace = scale_workspace.view(512, 160, 320)
            w2_workspace = scale_workspace[: 512 * 10 * 2560].view(512, 10, 2560)
            w13_exact = _raw_scales_match_prepared(
                w13_workspace,
                raw_w13_codes,
                raw_w13_globals,
                prepared_w13_scales,
                fused_swiglu_prefill,
            )
            w2_exact = _raw_scales_match_prepared(
                w2_workspace,
                raw_w2_codes,
                raw_w2_globals,
                prepared_w2_scales,
                False,
            )
            if not (w13_exact and w2_exact):
                if raw_scale_explicit:
                    raise RuntimeError(
                        "SM70 Qwen3.8 raw E4M3 scale reconstruction is not "
                        "bitwise equal to the prepared FP16 checkpoint scales."
                    )
                logger.warning_once(
                    "SM70 Qwen3.8 raw E4M3 scale reconstruction did not match "
                    "the prepared checkpoint scales; retaining the FP16 scale "
                    "representation."
                )
                raw_scale = False

        if raw_scale:
            layer.w13_raw_scale_codes = Parameter(raw_w13_codes, requires_grad=False)
            layer.w13_raw_global_scales = Parameter(
                raw_w13_globals, requires_grad=False
            )
            layer.w2_raw_scale_codes = Parameter(raw_w2_codes, requires_grad=False)
            layer.w2_raw_global_scales = Parameter(raw_w2_globals, requires_grad=False)
            layer.w13_tm_scales = w13_workspace
            layer.w2_tm_scales = w2_workspace
        else:
            layer.w13_tm_scales = Parameter(prepared_w13_scales, requires_grad=False)
            layer.w2_tm_scales = Parameter(prepared_w2_scales, requires_grad=False)

        w13_k_ld = int(w13_meta[0][0].item())
        w13_q_ld = int(w13_meta[0][1].item())
        w2_k_ld = int(w2_meta[0][0].item())
        w2_q_ld = int(w2_meta[0][1].item())
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
        if fast_prefill:
            # TurboMind packs this exact N320 weight as five contiguous N64
            # tiles per expert. The N256/N64 views therefore partition the
            # existing allocation without copying it. Scales remain a strided
            # N320 view, so both subprojections retain the original q_ld.
            w13_flat = layer.w13_tm_weight.view(num_experts, -1)
            head_words = hidden * 256 // 8
            w13_head_ptrs = self.native_ops.awq_moe_build_strided_ptrs(
                w13_flat[:, :head_words],
                layer.w13_tm_scales[:, :, :256],
                w13_k_ld,
                w13_q_ld,
                num_experts,
            )
            w13_tail_ptrs = self.native_ops.awq_moe_build_strided_ptrs(
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
        layer.sm70_nvfp4_qwen38_fused_swiglu_decode = fused_swiglu_decode
        layer.sm70_nvfp4_qwen38_fast_prefill = fast_prefill
        layer.sm70_nvfp4_qwen38_raw_scale = raw_scale
        layer.sm70_nvfp4_qwen38_w2_direct_reduce = bool(
            w2_direct_reduce_requested and w2_direct_reduce_available
        )

        layer.sm70_glm53_fused_permute_q8 = glm53_fused_permute_q8
        layer.sm70_glm53_qpn_w13_q8 = glm53_qpn_w13_q8
        layer.sm70_nvfp4_graph_safe_max_tokens = _GRAPH_SAFE_MAX_TOKENS
        layer.sm70_nvfp4_compact_grouped_max_slots = _COMPACT_GROUPED_MAX_SLOTS
        grouped_supported = bool(
            (num_experts, hidden, intermediate, layer.sm70_nvfp4_top_k)
            == (512, 2560, 160, 10)
            and not raw_scale
            and layer.swiglu_limit is None
        )
        grouped_requested = bool(
            _nvfp4_policy(layer).grouped_decode and grouped_supported
        )
        grouped_mtp5 = bool(
            _nvfp4_policy(layer).grouped_mtp5
            and _nvfp4_policy(layer).qpn_mtp5
            and int(layer.moe_config.tp_size) == 4
            and grouped_supported
        )
        if (grouped_requested or grouped_mtp5) and (
            not self.native_ops.has_nvfp4_grouped_decode_dispatch()
            or (
                grouped_mtp5
                and not self.native_ops.has_nvfp4_grouped_batch_reduce_dispatch()
            )
        ):
            raise RuntimeError(
                "SM70 grouped MoE decode requires a matching native build."
            )
        layer.sm70_nvfp4_grouped_decode = grouped_requested
        layer.sm70_nvfp4_grouped_mtp5 = grouped_mtp5
        if grouped_requested or grouped_mtp5:
            # Layer-owned metadata; W2 reuses sorted_output. No process-global
            # cache or additional activation buffer inside graph capture.
            device = layer.w13_tm_weight.device
            layer._nvfp4_grouped_rows = torch.empty(
                160, 8, dtype=torch.int32, device=device
            )
            layer._nvfp4_grouped_experts = torch.empty(
                160, dtype=torch.int32, device=device
            )
            layer._nvfp4_grouped_sizes = torch.empty(
                160, dtype=torch.int32, device=device
            )
            layer._nvfp4_grouped_total = torch.empty(
                1, dtype=torch.int32, device=device
            )
        layer.sm70_fp4_codec = Fp4MoECodec(
            "nvfp4",
            LayerWorkspaceView(layer, ""),
            LayerWorkspaceView(layer, "sm70_nvfp4_"),
            raw_scale=bool(getattr(layer, "sm70_nvfp4_qwen38_raw_scale", False)),
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            bindings=self.native_ops,
        )
        self._allocate_graph_safe_decode_buffers(layer)

        del layer.w13_weight
        del layer.w13_weight_scale
        del layer.w13_weight_scale_2
        del layer.w13_input_scale
        del layer.w2_weight
        del layer.w2_weight_scale
        del layer.w2_weight_scale_2
        del layer.w2_input_scale
        # Release unused conversion blocks between layers instead of retaining
        # their high-water mark throughout model loading. Live prepared weights
        # and inference buffers are unaffected; this is not an inference hook.
        torch.accelerator.empty_cache()
        logger.info_once(
            "SM70 ModelOpt NVFP4 TurboMind MoE path enabled "
            "(hidden=%d, local_intermediate=%d, local_experts=%d, top_k=%d, "
            "graph_safe_decode=B1-B%d, compact_grouped_decode<=%d routed rows).",
            hidden,
            intermediate,
            num_experts,
            layer.sm70_nvfp4_top_k,
            _GRAPH_SAFE_MAX_TOKENS,
            _COMPACT_GROUPED_MAX_SLOTS,
        )
        if raw_scale:
            logger.info_once(
                "SM70 Qwen3.8 raw E4M3 expert-scale storage enabled; "
                "generic and prefill routes share one FP16 expansion workspace."
            )
        if fused_swiglu_prefill:
            logger.info_once(
                "SM70 Qwen3.8 indexed-A fused-SwiGLU prefill candidate "
                "enabled (interleaved W13, exact FP16 epilogue arithmetic)."
            )
        if fused_swiglu_decode:
            logger.info_once(
                "SM70 Qwen3.8 fused W13/SwiGLU decode route enabled "
                "(split16, exact FP16 rounding and activation arithmetic)."
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

    def _allocate_graph_safe_decode_buffers(self, layer):
        NvFp4MoEWorkspace.allocate(layer)

    def _get_buffers(self, layer, num_tokens, indexed_w13):
        return NvFp4MoEWorkspace.get(layer, num_tokens, indexed_w13)

    @staticmethod
    def _apply_swiglu(layer, out, gate_up, *, interleaved=False) -> None:
        apply_swiglu(out, gate_up, layer.swiglu_limit, interleaved=interleaved)

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
        grouped_mtp5 = _use_grouped_mtp5(layer, x, topk_ids)
        if grouped_mtp5 or _use_grouped_decode(layer, x, topk_ids):
            plan = select_fp4_stage_plan(
                Stage.ACTIVE_GROUPED,
                Stage.GROUPED_BATCH_REDUCE if grouped_mtp5 else Stage.ACTIVE_GROUPED,
                w13_split_k=4 if grouped_mtp5 or num_tokens == 8 else 8,
                interleaved=interleaved_w13,
            )
            execute_fp4(layer.sm70_fp4_codec, plan, buffers, x, topk_ids, topk_weights)
            logger.info_once(
                "Experimental SM70 grouped native-NVFP4 decode selected "
                "(tokens=%d, W13/W2 share route groups, MTP split4/reduce=%s).",
                num_tokens,
                grouped_mtp5,
            )
            return output
        direct_single_token = num_tokens == 1
        direct_qpn_m1 = _use_qwen38_qpn_m1_decode(layer, x, topk_ids)
        direct_qpn_batch = _use_qwen38_qpn_batch_decode(layer, x, topk_ids)
        direct_qpn_mtp5 = _use_qwen38_qpn_mtp5_decode(layer, x, topk_ids)
        raw_scale = bool(getattr(layer, "sm70_nvfp4_qwen38_raw_scale", False))
        if _nvfp4_policy(layer).route_debug and num_tokens <= 16:
            logger.warning_once(
                "SM70 Qwen3.8 QPN route debug: tokens=%d x_shape=%s "
                "x_stride=%s x_dtype=%s x_contiguous=%s ids_shape=%s "
                "ids_stride=%s ids_dtype=%s ids_contiguous=%s tp=%s "
                "experts=%s hidden=%s intermediate=%s top_k=%s "
                "m1=%s batch=%s mtp5=%s compiling=%s capturing=%s.",
                num_tokens,
                tuple(x.shape),
                tuple(x.stride()),
                x.dtype,
                x.is_contiguous(),
                tuple(topk_ids.shape),
                tuple(topk_ids.stride()),
                topk_ids.dtype,
                topk_ids.is_contiguous(),
                layer.moe_config.tp_size,
                layer.sm70_nvfp4_num_experts,
                layer.sm70_nvfp4_hidden_size,
                layer.sm70_nvfp4_intermediate_size,
                layer.sm70_nvfp4_top_k,
                direct_qpn_m1,
                direct_qpn_batch,
                direct_qpn_mtp5,
                torch.compiler.is_compiling(),
                torch.cuda.is_current_stream_capturing(),
            )
        if direct_qpn_m1 or direct_qpn_batch or direct_qpn_mtp5:
            if direct_qpn_m1:
                w13_split_k = _QWEN38_QPN_M1_W13_SPLIT_K
            elif direct_qpn_batch:
                split_table = (
                    _QWEN38_DYNAMIC_QPN_BATCH_W13_SPLIT_K
                    if _nvfp4_policy(layer).qpn_dynamic
                    else _QWEN38_QPN_BATCH_W13_SPLIT_K
                )
                w13_split_k = split_table[num_tokens]
            else:
                w13_split_k = _QWEN38_QPN_MTP5_W13_SPLIT_K
            logger.info_once(
                "SM70 Qwen3.8 NVFP4 direct expert path enabled "
                "(TP4, E512/K10, tokens=%d, W13 split%d, W2 split1).",
                num_tokens,
                w13_split_k,
            )
            fused_batch_w13 = _use_qwen38_qpn_batch_fused_w13(layer, x, topk_ids)
            fused_w13_decode = bool(
                direct_qpn_m1
                and not raw_scale
                and interleaved_w13
                and getattr(layer, "sm70_nvfp4_qwen38_fused_swiglu_decode", False)
            )
            w13 = (
                Stage.FUSED_QPN
                if fused_w13_decode
                else Stage.FUSED_BATCH_QPN
                if fused_batch_w13
                else Stage.QPN
            )
            direct_reduce = bool(
                direct_qpn_m1
                and not raw_scale
                and getattr(layer, "sm70_nvfp4_qwen38_w2_direct_reduce", False)
            )
            w2 = (
                Stage.DIRECT_REDUCE
                if direct_reduce
                else Stage.BATCH_REDUCE
                if _use_qwen38_qpn_batch_fused_w2(layer, x, topk_ids)
                else Stage.QPN
            )
            if w13 == Stage.FUSED_BATCH_QPN:
                logger.info_once(
                    "SM70 Qwen3.8 NVFP4 direct W13+SwiGLU fusion enabled (tokens=%d).",
                    num_tokens,
                )
            if w2 == Stage.BATCH_REDUCE:
                logger.info_once(
                    "SM70 Qwen3.8 NVFP4 direct W2+weighted-reduce fusion "
                    "enabled (tokens=%d).",
                    num_tokens,
                )
            plan = select_fp4_stage_plan(
                w13,
                w2,
                w13_split_k=w13_split_k,
                qpn_mtp=direct_qpn_mtp5,
                reduction="triton_single" if direct_qpn_m1 else "triton_batch",
                interleaved=interleaved_w13,
            )
            return execute_fp4(
                layer.sm70_fp4_codec, plan, buffers, x, topk_ids, topk_weights
            )
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
            self.native_ops.sm70_glm53_moe_permute_q8_out(
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
        elif (
            not indexed_w13
            and not _use_compact_grouped(num_tokens, top_k)
            and _fuse47.unit_enabled("h2")
            and _fuse47.h2_supported(layer, x, topk_ids, buffers)
        ):
            # STEP-47 h2: output zero fill, top-k id copy, radix sort, expert
            # offsets (int64 and int32) and row expansion in one kernel.
            _fuse47.note_route("h2", "fused")
            _fuse47.moe_permute_fused(layer, x, topk_ids, buffers)
        else:
            output.zero_()
            topk_ids_i32 = buffers["topk_ids"]
            topk_ids_i32.copy_(topk_ids, non_blocking=True)
            buffers["permuted_idx"].fill_(slots)
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
            buffers["expert_offsets"].copy_(
                buffers["expert_offsets64"], non_blocking=True
            )

        if (
            not direct_single_token
            and not glm53_fused_permute_q8
            and _use_compact_grouped(num_tokens, top_k)
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
            w13 = Stage.INDEXED_SPLIT_FUSED
            logger.info_once(
                "SM70 Qwen3.8 NVFP4 indexed-A fused-SwiGLU split-W13 "
                "prefill route enabled (N256+N64)."
            )
        elif fused_indexed_w13:
            w13 = Stage.INDEXED_FUSED
            logger.info_once(
                "SM70 Qwen3.8 NVFP4 indexed-A fused-SwiGLU W13 prefill "
                "candidate enabled."
            )
        elif indexed_w13:
            w13 = Stage.INDEXED_PREFILL
            logger.info_once(
                "SM70 Qwen3.8 NVFP4 indexed-A W13 prefill route enabled "
                "(TP4, E512/K10, materialized input rows skipped)."
            )
        elif glm53_qpn_w13_q8:
            w13 = Stage.GLM_QPN
            logger.info_once(
                "SM70 GLM-5.3 TP8 q8 exact W13 QPN path enabled "
                "(CTA-K32 split-3 accumulation tree)."
            )
        else:
            w13 = Stage.DENSE
        plan = select_fp4_stage_plan(
            w13,
            Stage.DENSE,
            interleaved=interleaved_w13,
            reduction="triton_single" if direct_single_token else "unpermute",
        )
        execute_fp4(
            layer.sm70_fp4_codec,
            plan,
            buffers,
            x if indexed_w13 else buffers["permuted_input"],
            topk_ids,
            topk_weights,
            offsets=stage_offsets,
            expert_ids=stage_expert_ids,
            expert_count=stage_experts,
            unpermute_offsets=None
            if glm53_fused_permute_q8
            else buffers["expert_offsets64"],
        )
        if diagnostic_history("nvfp4_dflash").get("armed", False) and num_tokens > 1:
            diagnostic_history("nvfp4_dflash")["armed"] = False
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
