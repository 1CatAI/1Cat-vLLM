# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM70 TurboMind MXFP4 MoE for DeepSeek-V4-Flash.

This is deliberately a narrow route: DeepSeek-V4-Flash's packed MXFP4 expert
weights are converted once into TurboMind's packed e2m1 layout and retain their
UE8M0 scales. It never materializes an FP16/BF16 expert-weight copy.
"""

from __future__ import annotations

from typing import Final

import torch
from torch.nn import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm.config.sm70_moe import Sm70MxFp4MoEConfig, capture_mxfp4_moe_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEParallelConfig,
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
from vllm.model_executor.layers.fused_moe.sm70.fp4_workspace import MxFp4MoEWorkspace
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    Sm70MoeStageRoute as Stage,
)
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    select_fp4_stage_plan,
)
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    MXFP4_GROUP_SIZE,
    is_exact_sm70_cuda,
    unpack_mxfp4_weight,
)
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    LayerWorkspaceView,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_DEEPSEEK_V4_FLASH_HIDDEN_SIZE: Final = 4096
_DEEPSEEK_V4_FLASH_INTERMEDIATE_SIZE: Final = 2048
_DEEPSEEK_V4_FLASH_NUM_EXPERTS: Final = 256
_DEEPSEEK_V4_FLASH_TOP_K: Final = 6
_GRAPH_SAFE_MAX_TOKENS: Final = 8
_MXFP4_QPN_M1_ENV: Final = "VLLM_SM70_MXFP4_MOE_QPN_M1_DECODE"
_MXFP4_QPN_M1_OP: Final = "mxfp4_moe_qpn_m1_sm70_out"


def _mxfp4_policy(layer) -> Sm70MxFp4MoEConfig:
    policy = getattr(layer, "sm70_moe_policy", None)
    return policy if policy is not None else capture_mxfp4_moe_config()


def _mxfp4_qpn_m1_op_available() -> bool:
    return hasattr(torch.ops._C, _MXFP4_QPN_M1_OP)


def _mxfp4_qpn_m1_extension_enabled(policy: Sm70MxFp4MoEConfig | None = None) -> bool:
    """Resolve the default-on route without breaking an older extension."""
    policy = policy if policy is not None else capture_mxfp4_moe_config()
    if not policy.qpn_m1:
        return False
    if _mxfp4_qpn_m1_op_available():
        return True
    if "qpn_m1" in policy.explicit_fields:
        raise RuntimeError(
            "The explicitly enabled SM70 MXFP4 QPN M1 route requires the "
            f"source-built {_MXFP4_QPN_M1_OP} operator."
        )
    logger.warning_once(
        "The default SM70 MXFP4 QPN M1 route is unavailable in the loaded "
        "vllm._C; retaining the TurboMind dense-stage path."
    )
    return False


def _mxfp4_active_expert_b1_enabled(policy: Sm70MxFp4MoEConfig | None = None) -> bool:
    policy = policy if policy is not None else capture_mxfp4_moe_config()
    return bool(
        policy.active_experts
        and not policy.single_token_fastpath
        and not policy.single_token_permute
    )


def _mxfp4_active_expert_max_tokens(policy: Sm70MxFp4MoEConfig | None = None) -> int:
    policy = policy if policy is not None else capture_mxfp4_moe_config()
    if not _mxfp4_active_expert_b1_enabled(policy):
        return 0
    return min(
        int(policy.active_expert_max_tokens or 0),
        _GRAPH_SAFE_MAX_TOKENS,
    )


def _mxfp4_grouped_m8_enabled(policy: Sm70MxFp4MoEConfig | None = None) -> bool:
    policy = policy if policy is not None else capture_mxfp4_moe_config()
    return bool(policy.grouped_m8)


def _mxfp4_grouped_verifier_enabled(policy: Sm70MxFp4MoEConfig | None = None) -> bool:
    policy = policy if policy is not None else capture_mxfp4_moe_config()
    return bool(policy.grouped_verifier)


def _mxfp4_grouped_verifier_for_tokens(
    num_tokens: int, policy: Sm70MxFp4MoEConfig | None = None
) -> bool:
    policy = policy if policy is not None else capture_mxfp4_moe_config()
    return bool(
        (num_tokens == _GRAPH_SAFE_MAX_TOKENS and _mxfp4_grouped_m8_enabled(policy))
        or (
            1 < num_tokens <= _GRAPH_SAFE_MAX_TOKENS
            and _mxfp4_grouped_verifier_enabled(policy)
        )
    )


def _mxfp4_grouped_m8_expert_rows_enabled(
    policy: Sm70MxFp4MoEConfig | None = None,
) -> bool:
    policy = policy if policy is not None else capture_mxfp4_moe_config()
    return bool(
        (_mxfp4_grouped_m8_enabled(policy) or _mxfp4_grouped_verifier_enabled(policy))
        and policy.grouped_expert_rows
    )


@triton.jit
def _compact_sorted_experts_kernel(
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


def _compact_mxfp4_active_experts(
    sorted_expert_ids: torch.Tensor,
    compact_offsets: torch.Tensor,
    active_expert_ids: torch.Tensor,
) -> None:
    total_slots = sorted_expert_ids.numel()
    if not (0 < total_slots <= _GRAPH_SAFE_MAX_TOKENS * _DEEPSEEK_V4_FLASH_TOP_K):
        raise ValueError(f"Unsupported SM70 MXFP4 active-expert slots: {total_slots}")
    block = triton.next_power_of_2(total_slots + 1)
    _compact_sorted_experts_kernel[(1,)](
        sorted_expert_ids,
        compact_offsets,
        active_expert_ids,
        TOTAL_SLOTS=total_slots,
        BLOCK=block,
        num_warps=1,
    )


def _select_mxfp4_stage_dispatch(
    buffers: dict[str, torch.Tensor],
    *,
    num_tokens: int,
    num_experts: int,
    fully_replicated_experts: bool,
    policy: Sm70MxFp4MoEConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if (
        0 < num_tokens <= _mxfp4_active_expert_max_tokens(policy)
        and fully_replicated_experts
    ):
        # Keep the graph launch count fixed. The compactor represents unused
        # tail entries as zero-row experts, avoiding a host readback of the
        # dynamic unique-expert count.
        graph_expert_slots = num_tokens * _DEEPSEEK_V4_FLASH_TOP_K
        if _mxfp4_grouped_verifier_for_tokens(num_tokens, policy):
            if _mxfp4_grouped_m8_expert_rows_enabled(policy):
                return (
                    buffers["compact_expert_offsets"],
                    buffers["active_expert_ids"],
                    graph_expert_slots,
                )
            return (
                buffers["slot_expert_offsets"],
                buffers["permuted_experts_id"],
                graph_expert_slots,
            )
        return (
            buffers["compact_expert_offsets"],
            (
                buffers["permuted_experts_id"]
                if num_tokens == 1
                else buffers["active_expert_ids"]
            ),
            graph_expert_slots,
        )
    return buffers["expert_offsets"], buffers["dense_expert_ids"], num_experts


def _select_mxfp4_direct_order_offsets(
    buffers: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return immutable one-row offsets for B1 direct-order decode.

    Active-expert compaction overwrites ``compact_expert_offsets`` during
    M2--M8 warmup and verifier calls. Direct-order B1 has one row per route,
    so it must use the separately maintained slot offsets instead.
    """
    return buffers["slot_expert_offsets"]


def _mxfp4_qpn_m1_decode_contract(layer: RoutedExperts, *, direct_order: bool) -> bool:
    """Admit only the measured TP4 six-route W13/W2 tensor pair."""
    return bool(
        _mxfp4_policy(layer).qpn_m1
        and getattr(layer, "sm70_mxfp4_qpn_m1_available", False)
        and direct_order
        and int(layer.moe_config.tp_size) == 4
        and int(layer.local_num_experts) == 256
        and int(layer.global_num_experts) == 256
        and int(layer.sm70_mxfp4_w13_k_dim) == 4096
        and int(layer.sm70_mxfp4_w13_n_dim) == 1024
        and int(layer.sm70_mxfp4_w2_k_dim) == 512
        and int(layer.sm70_mxfp4_w2_n_dim) == 4096
        and int(layer.sm70_mxfp4_group_size) == 32
        and tuple(layer.w13_tm_weight.shape) == (256, 4096, 128)
        and tuple(layer.w13_tm_scales.shape) == (256, 128, 1024)
        and tuple(layer.w2_tm_weight.shape) == (256, 512, 512)
        and tuple(layer.w2_tm_scales.shape) == (256, 16, 4096)
    )


def validate_mxfp4_sm70_moe_contract(
    *,
    global_num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    tp_size: int,
) -> None:
    """Reject shapes outside the exact V4-Flash SM70 implementation contract."""
    if global_num_experts != _DEEPSEEK_V4_FLASH_NUM_EXPERTS:
        raise NotImplementedError(
            "SM70 TurboMind MXFP4 MoE currently supports DeepSeek-V4-Flash "
            f"with {_DEEPSEEK_V4_FLASH_NUM_EXPERTS} global experts, got "
            f"{global_num_experts}."
        )
    if top_k != _DEEPSEEK_V4_FLASH_TOP_K:
        raise NotImplementedError(
            "SM70 TurboMind MXFP4 MoE currently supports DeepSeek-V4-Flash "
            f"top-k={_DEEPSEEK_V4_FLASH_TOP_K}, got {top_k}."
        )
    if hidden_size != _DEEPSEEK_V4_FLASH_HIDDEN_SIZE:
        raise NotImplementedError(
            "SM70 TurboMind MXFP4 MoE currently supports hidden size "
            f"{_DEEPSEEK_V4_FLASH_HIDDEN_SIZE}, got {hidden_size}."
        )
    if intermediate_size_per_partition <= 0 or (
        intermediate_size_per_partition % MXFP4_GROUP_SIZE
    ):
        raise NotImplementedError(
            "SM70 TurboMind MXFP4 MoE requires a positive local intermediate "
            f"size divisible by {MXFP4_GROUP_SIZE}, got "
            f"{intermediate_size_per_partition}."
        )
    if intermediate_size_per_partition * max(tp_size, 1) != (
        _DEEPSEEK_V4_FLASH_INTERMEDIATE_SIZE
    ):
        raise NotImplementedError(
            "SM70 TurboMind MXFP4 MoE currently supports DeepSeek-V4-Flash "
            f"intermediate size {_DEEPSEEK_V4_FLASH_INTERMEDIATE_SIZE}; got "
            f"local={intermediate_size_per_partition}, tp_size={tp_size}."
        )


def validate_mxfp4_sm70_moe_weight_layout(
    *,
    local_num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> None:
    """Validate the checkpoint's packed MXFP4/UE8M0 tensors without unpacking."""
    expected_shapes = {
        "w13_weight": (
            local_num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // 2,
        ),
        "w13_weight_scale": (
            local_num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // MXFP4_GROUP_SIZE,
        ),
        "w2_weight": (
            local_num_experts,
            hidden_size,
            intermediate_size_per_partition // 2,
        ),
        "w2_weight_scale": (
            local_num_experts,
            hidden_size,
            intermediate_size_per_partition // MXFP4_GROUP_SIZE,
        ),
    }
    actual = {
        "w13_weight": w13_weight,
        "w13_weight_scale": w13_weight_scale,
        "w2_weight": w2_weight,
        "w2_weight_scale": w2_weight_scale,
    }
    for name, tensor in actual.items():
        if tensor.dtype != torch.uint8:
            raise TypeError(
                "SM70 TurboMind MXFP4 MoE requires packed uint8 "
                f"{name}, got {tensor.dtype}."
            )
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                "SM70 TurboMind MXFP4 MoE packed layout mismatch for "
                f"{name}: expected {expected_shapes[name]}, got "
                f"{tuple(tensor.shape)}."
            )


def _prepare_mxfp4_sm70_experts(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> list[torch.Tensor]:
    """Repack one projection's experts into stacked TurboMind tensors.

    Each prepared expert is copied straight into its slot, so the repack holds
    the checkpoint projection and one stacked copy instead of a per-expert
    list next to its stack. The converter only repacks nibbles and scales; it
    does not dequantize or materialize a full-precision expert weight.
    """
    num_experts = weight.shape[0]
    stacks: list[torch.Tensor] = []
    for expert_id in range(num_experts):
        prepared = Fp4MoECodec.prepare_weights(
            "mxfp4",
            unpack_mxfp4_weight(weight[expert_id].data),
            weight_scale[expert_id].data.t().contiguous(),
            MXFP4_GROUP_SIZE,
        )
        if not stacks:
            stacks = [
                torch.empty(
                    (num_experts, *part.shape), dtype=part.dtype, device=part.device
                )
                for part in prepared
            ]
        for stack, part in zip(stacks, prepared):
            stack[expert_id].copy_(part)
    return stacks


class Mxfp4SM70MoEMethod(Mxfp4MoEMethod):
    """Exact-SM70 V4-Flash MXFP4 MoE using TurboMind packed GEMMs.

    ``Mxfp4MoEMethod`` owns the checkpoint parameter layout. Its generic
    Oracle backends are intentionally bypassed here because they are not an
    SM70 implementation and may select Marlin or a weight-emulation route.
    """

    def __init__(self, moe: FusedMoEConfig):
        FusedMoEMethodBase.__init__(self, moe)
        self.sm70_moe_policy = capture_mxfp4_moe_config()
        self.weight_dtype = "mxfp4"
        if moe.moe_parallel_config.use_all2all_kernels:
            raise NotImplementedError(
                "SM70 MXFP4 MoE does not support DP+EP all-to-all routing yet."
            )
        validate_mxfp4_sm70_moe_contract(
            global_num_experts=moe.num_experts,
            top_k=moe.experts_per_token,
            hidden_size=moe.hidden_dim,
            intermediate_size_per_partition=moe.intermediate_size_per_partition,
            tp_size=moe.tp_size,
        )

    @property
    def skip_forward_padding(self) -> bool:
        # The generic MXFP4 implementation keys this on its Oracle backend;
        # this native SM70 implementation has no Oracle backend.
        return False

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = (
            FusedMoEMethodBase.maybe_roundup_sizes(
                self,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size_per_partition,
                act_dtype=act_dtype,
                moe_parallel_config=moe_parallel_config,
            )
        )
        validate_mxfp4_sm70_moe_contract(
            global_num_experts=self.moe.num_experts,
            top_k=self.moe.experts_per_token,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            tp_size=moe_parallel_config.tp_size,
        )
        return hidden_size, intermediate_size_per_partition

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        layer.sm70_moe_policy = self.sm70_moe_policy
        required_ops: tuple[str, ...] = (
            "mxfp4_sm70_prepare",
            "mxfp4_moe_dense_stage_sm70_out",
            "awq_moe_build_strided_ptrs",
        )
        if self.sm70_moe_policy.direct_top6:
            required_ops += ("mxfp4_moe_single_token_prepare_w13_sm70_out",)
        if self.sm70_moe_policy.direct_top6 and self.sm70_moe_policy.direct_order:
            required_ops += ("awq_moe_single_token_weighted_reduce_out",)
        missing_ops = [name for name in required_ops if not hasattr(torch.ops._C, name)]
        if missing_ops:
            raise RuntimeError(
                "DeepSeek-V4 MXFP4 MoE on SM70 requires the TurboMind CUDA "
                "extension with " + ", ".join(missing_ops) + "."
            )
        layer.sm70_mxfp4_qpn_m1_available = _mxfp4_qpn_m1_extension_enabled(
            self.sm70_moe_policy
        )
        if not hasattr(torch.ops._moe_C, "moe_permute_with_scratch"):
            raise RuntimeError(
                "DeepSeek-V4 MXFP4 MoE graph-safe B1 requires "
                "_moe_C.moe_permute_with_scratch."
            )
        if self.moe.has_bias:
            raise NotImplementedError("SM70 MXFP4 MoE does not support expert bias.")
        if layer.activation != MoEActivation.SILU:
            raise NotImplementedError(
                "SM70 MXFP4 MoE only supports the DeepSeek-V4 SwiGLU activation."
            )
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "SM70 MXFP4 MoE does not support applying router weights to input."
            )

        num_experts = int(layer.local_num_experts)
        hidden_size = int(layer.moe_config.hidden_dim)
        intermediate_size = int(layer.moe_config.intermediate_size_per_partition)
        validate_mxfp4_sm70_moe_contract(
            global_num_experts=int(layer.global_num_experts),
            top_k=int(layer.top_k),
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            tp_size=layer.moe_config.tp_size,
        )
        validate_mxfp4_sm70_moe_weight_layout(
            local_num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            w13_weight=layer.w13_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight=layer.w2_weight,
            w2_weight_scale=layer.w2_weight_scale,
        )

        # post-load processing runs after every layer of the stage is
        # resident, so each projection's checkpoint copy is released as soon
        # as its TurboMind stack exists.
        w13_tm_weight, w13_tm_scales, w13_meta = _prepare_mxfp4_sm70_experts(
            layer.w13_weight, layer.w13_weight_scale
        )
        del layer.w13_weight
        del layer.w13_weight_scale
        w2_tm_weight, w2_tm_scales, w2_meta = _prepare_mxfp4_sm70_experts(
            layer.w2_weight, layer.w2_weight_scale
        )
        del layer.w2_weight
        del layer.w2_weight_scale
        # Return the released checkpoint blocks before the pointer tables
        # allocate, so V100 does not exhaust driver memory while the CUDA
        # caching allocator retains them.
        torch.accelerator.empty_cache()

        layer.w13_tm_weight = Parameter(w13_tm_weight, requires_grad=False)
        layer.w13_tm_scales = Parameter(w13_tm_scales, requires_grad=False)
        layer.w13_tm_meta = Parameter(w13_meta, requires_grad=False)
        layer.w2_tm_weight = Parameter(w2_tm_weight, requires_grad=False)
        layer.w2_tm_scales = Parameter(w2_tm_scales, requires_grad=False)
        layer.w2_tm_meta = Parameter(w2_meta, requires_grad=False)

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
        layer.w13_strided_ptrs_w = Parameter(w13_ptrs[0], requires_grad=False)
        layer.w13_strided_ptrs_s = Parameter(w13_ptrs[1], requires_grad=False)
        layer.w2_strided_ptrs_w = Parameter(w2_ptrs[0], requires_grad=False)
        layer.w2_strided_ptrs_s = Parameter(w2_ptrs[1], requires_grad=False)

        layer.sm70_mxfp4_moe = True
        layer.sm70_mxfp4_num_experts = num_experts
        layer.sm70_mxfp4_hidden_size = hidden_size
        layer.sm70_mxfp4_intermediate_size = intermediate_size
        layer.sm70_mxfp4_w13_k_dim = hidden_size
        layer.sm70_mxfp4_w13_n_dim = 2 * intermediate_size
        layer.sm70_mxfp4_w2_k_dim = intermediate_size
        layer.sm70_mxfp4_w2_n_dim = hidden_size
        layer.sm70_mxfp4_group_size = MXFP4_GROUP_SIZE
        layer.sm70_fp4_codec = Fp4MoECodec(
            "mxfp4",
            LayerWorkspaceView(layer, ""),
            LayerWorkspaceView(layer, "sm70_mxfp4_"),
            raw_scale=False,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
        )
        self._allocate_graph_safe_decode_buffers(layer)

        logger.info_once(
            "SM70 TurboMind MXFP4 MoE enabled for DeepSeek-V4-Flash "
            "(local_experts=%d, graph_safe_decode=B1-B%d, "
            "active_expert_max_tokens=%d).",
            num_experts,
            _GRAPH_SAFE_MAX_TOKENS,
            _mxfp4_active_expert_max_tokens(self.sm70_moe_policy),
        )

    def _allocate_graph_safe_decode_buffers(self, layer):
        MxFp4MoEWorkspace.allocate(layer)

    def _get_buffers(self, layer, num_tokens):
        return MxFp4MoEWorkspace.get(layer, num_tokens)

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
            raise TypeError("SM70 MXFP4 MoE requires CUDA FP16 activations [M, H].")
        if not is_exact_sm70_cuda(x, enabled=True):
            raise RuntimeError("SM70 MXFP4 MoE dispatch is restricted to CUDA SM70.")
        if x.shape[1] != _DEEPSEEK_V4_FLASH_HIDDEN_SIZE:
            raise ValueError(
                "SM70 MXFP4 MoE activation hidden size mismatch: expected "
                f"{_DEEPSEEK_V4_FLASH_HIDDEN_SIZE}, got {x.shape[1]}."
            )
        if tuple(topk_ids.shape) != (
            x.shape[0],
            _DEEPSEEK_V4_FLASH_TOP_K,
        ):
            raise ValueError("SM70 MXFP4 MoE requires top-k IDs with shape [M, 6].")
        if tuple(topk_weights.shape) != tuple(topk_ids.shape):
            raise ValueError("SM70 MXFP4 MoE top-k weights and IDs must share shape.")
        if topk_weights.dtype != torch.float32:
            raise TypeError("SM70 MXFP4 MoE requires float32 top-k weights.")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "SM70 MXFP4 MoE does not support applying router weights to input."
            )

        num_tokens = x.shape[0]
        if num_tokens == 0:
            return x.new_empty((0, _DEEPSEEK_V4_FLASH_HIDDEN_SIZE))
        buffers = self._get_buffers(layer, num_tokens)
        output = buffers["output"]

        direct_top6 = (
            num_tokens == 1
            and self.sm70_moe_policy.direct_top6
            and layer.expert_map is None
            and layer.local_num_experts == layer.global_num_experts
        )
        direct_order = bool(
            direct_top6
            and self.sm70_moe_policy.direct_order
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
        )
        if _mxfp4_qpn_m1_decode_contract(layer, direct_order=direct_order):
            plan = select_fp4_stage_plan(
                Stage.QPN, Stage.QPN, reduction="native_weighted"
            )
            execute_fp4(layer.sm70_fp4_codec, plan, buffers, x, topk_ids, topk_weights)
            logger.info_once(
                "Default SM70 MXFP4 QPN M1 route enabled for the exact "
                "TP4 six-route W13/W2 tensor contract."
            )
            return output
        if direct_order:
            plan = select_fp4_stage_plan(
                Stage.DENSE, Stage.DENSE, reduction="native_weighted"
            )
            return execute_fp4(
                layer.sm70_fp4_codec,
                plan,
                buffers,
                x,
                topk_ids,
                topk_weights,
                offsets=_select_mxfp4_direct_order_offsets(buffers),
                expert_ids=topk_ids.view(-1),
                expert_count=_DEEPSEEK_V4_FLASH_TOP_K,
            )
        if direct_top6:
            plan = select_fp4_stage_plan(Stage.PREPARE_W13, Stage.DENSE)
            return execute_fp4(
                layer.sm70_fp4_codec,
                plan,
                buffers,
                x,
                topk_ids,
                topk_weights,
                offsets=buffers["compact_expert_offsets"],
                expert_ids=buffers["permuted_experts_id"],
                expert_count=_DEEPSEEK_V4_FLASH_TOP_K,
                unpermute_offsets=buffers["compact_expert_offsets64"],
            )

        output.zero_()

        total_slots = num_tokens * _DEEPSEEK_V4_FLASH_TOP_K
        topk_ids_i32 = buffers["topk_ids"]
        topk_ids_i32.copy_(topk_ids, non_blocking=True)
        buffers["permuted_idx"].fill_(total_slots)
        torch.ops._moe_C.moe_permute_with_scratch(
            x,
            topk_ids_i32,
            buffers["token_expert_indices"],
            layer.expert_map,
            layer.global_num_experts,
            layer.local_num_experts,
            _DEEPSEEK_V4_FLASH_TOP_K,
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

        if (
            num_tokens > 1
            and num_tokens <= _mxfp4_active_expert_max_tokens(self.sm70_moe_policy)
            and not (
                _mxfp4_grouped_verifier_for_tokens(num_tokens, self.sm70_moe_policy)
                and not _mxfp4_grouped_m8_expert_rows_enabled(self.sm70_moe_policy)
            )
            and layer.expert_map is None
            and layer.local_num_experts == layer.global_num_experts
        ):
            _compact_mxfp4_active_experts(
                buffers["permuted_experts_id"],
                buffers["compact_expert_offsets"],
                buffers["active_expert_ids"],
            )

        stage_offsets, stage_expert_ids, stage_expert_count = (
            _select_mxfp4_stage_dispatch(
                buffers,
                policy=self.sm70_moe_policy,
                num_tokens=num_tokens,
                num_experts=layer.sm70_mxfp4_num_experts,
                fully_replicated_experts=(
                    layer.expert_map is None
                    and layer.local_num_experts == layer.global_num_experts
                ),
            )
        )

        return execute_fp4(
            layer.sm70_fp4_codec,
            select_fp4_stage_plan(Stage.DENSE, Stage.DENSE),
            buffers,
            buffers["permuted_input"],
            topk_ids,
            topk_weights,
            offsets=stage_offsets,
            expert_ids=stage_expert_ids,
            expert_count=stage_expert_count,
            unpermute_offsets=buffers["expert_offsets64"],
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, x, router_logits, input_ids
        raise NotImplementedError("SM70 MXFP4 MoE is not a monolithic route.")

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None
