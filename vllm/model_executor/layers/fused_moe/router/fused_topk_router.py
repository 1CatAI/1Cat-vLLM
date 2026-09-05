# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm._custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _sm70_qwen38_router_topk_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    token_expert_indices_ptr,
    E: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    PACKED_HALF_KEY: tl.constexpr = False,
) -> None:
    """Sort one exact Qwen3.8 decode or MTP verifier row per program."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_E)
    valid = offsets < E
    logits = tl.load(
        gating_ptr + row * E + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    max_logit = tl.max(logits, axis=0)

    # Match topk_softmax's degenerate-row behavior: NaN, +Inf, or all -Inf
    # produces the first K expert IDs with zero weights.
    has_nan = tl.max((logits != logits).to(tl.int32), axis=0) != 0
    invalid_row = has_nan | (max_logit == float("inf")) | (max_logit == -float("inf"))
    sort_logits = tl.where(invalid_row, -offsets.to(tl.float32), logits)
    # Numeric ties use the lower expert ID in the generic CUDA op. Canonicalize
    # signed zero before bit packing so -0.0 and +0.0 remain one tie class.
    sort_logits = tl.where(sort_logits == 0.0, 0.0, sort_logits)

    # Transform float32 into an ascending-sortable key. Packing the expert ID
    # into the low bits preserves the generic kernel's lower-ID tie break.
    if PACKED_HALF_KEY:
        # FP16 -> FP32 above is exact. Sort the original 16-bit values plus
        # nine expert-ID bits in one int32, without quantizing any logits.
        # Degenerate rows use -offsets (0..511), also exactly representable.
        tl.static_assert(E == 512 and BLOCK_E == 512)
        bits = sort_logits.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
        key = tl.where(bits < 0, bits ^ 0x8000, bits ^ 0xFFFF) & 0xFFFF
        # Original int64 sort is signed: flip the key sign bit when moving
        # to a positive 25-bit key so positive logits still precede negatives.
        packed = ((key ^ 0x8000) << 9) | offsets
        sorted_packed = tl.sort(packed, descending=False)
        sorted_keys = (sorted_packed >> 9) ^ 0x8000
        sorted_ids = sorted_packed & 0x1FF
        sorted_bits = tl.where(
            (sorted_keys & 0x8000) != 0,
            sorted_keys ^ 0xFFFF,
            sorted_keys ^ 0x8000,
        ).to(tl.uint16)
        sorted_logits = sorted_bits.to(tl.float16, bitcast=True).to(tl.float32)
    else:
        min_i32: tl.constexpr = -2147483648
        logit_bits = sort_logits.to(tl.int32, bitcast=True)
        sign = logit_bits >> 31
        key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ min_i32)
        key = tl.where(valid, key, 0x7FFFFFFF)
        packed = ((key.to(tl.int64) & 0xFFFFFFFF) << 32) | offsets.to(tl.int64)
        sorted_packed = tl.sort(packed, descending=False)

        sorted_keys = ((sorted_packed >> 32) & 0xFFFFFFFF).to(tl.int32)
        sorted_ids = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
        sorted_sign = sorted_keys >> 31
        sorted_bits = tl.where(sorted_sign < 0, sorted_keys ^ -1, sorted_keys ^ min_i32)
        sorted_logits = sorted_bits.to(tl.float32, bitcast=True)

    raw_weights = tl.math.exp2((sorted_logits - max_logit) * 1.4426950408889634)
    raw_weights = tl.where(invalid_row, 0.0, raw_weights)
    top_mask = offsets < K
    denominator = tl.sum(tl.where(top_mask, raw_weights, 0.0), axis=0)
    denominator = tl.where(denominator > 0.0, denominator, 1.0)
    weights = raw_weights / denominator

    output_offsets = row * K + offsets
    tl.store(topk_ids_ptr + output_offsets, sorted_ids, mask=top_mask)
    tl.store(topk_weights_ptr + output_offsets, weights, mask=top_mask)
    # Match topkGating's rank-major source-row convention.
    tl.store(
        token_expert_indices_ptr + output_offsets, offsets * M + row, mask=top_mask
    )


def _sm70_qwen38_router_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
) -> None:
    num_tokens = gating_output.shape[0]
    _sm70_qwen38_router_topk_kernel[(num_tokens,)](
        gating_output,
        topk_weights,
        topk_ids,
        token_expert_indices,
        E=512,
        K=10,
        M=num_tokens,
        BLOCK_E=512,
        PACKED_HALF_KEY=(gating_output.dtype == torch.float16 and num_tokens == 1),
        num_warps=8,
    )


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_sigmoid
    return vllm_topk_sigmoid


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        if (
            envs.VLLM_SM70_QWEN38_ROUTER_TOPK
            and 1 <= M <= 16
            and gating_output.shape == (M, 512)
            and gating_output.dtype == torch.float16
            and gating_output.is_contiguous()
            and topk == 10
            and renormalize
            and topk_ids.dtype == torch.int32
            and current_platform.is_device_capability(70)
        ):
            logger.info_once(
                "SM70 Qwen3.8 E512/K10 router top-k path enabled for M=%d.", M
            )
            _sm70_qwen38_router_topk(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
            )
            return topk_weights, topk_ids, token_expert_indices

        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        eplb_state: EplbLayerState | None = None,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            indices_type_getter=indices_type_getter,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
        )

        return topk_weights, topk_ids
