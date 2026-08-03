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
from vllm import envs
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
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    MXFP4_GROUP_SIZE,
    is_exact_sm70_cuda,
    unpack_mxfp4_weight,
)

logger = init_logger(__name__)

_DEEPSEEK_V4_FLASH_HIDDEN_SIZE: Final = 4096
_DEEPSEEK_V4_FLASH_INTERMEDIATE_SIZE: Final = 2048
_DEEPSEEK_V4_FLASH_NUM_EXPERTS: Final = 256
_DEEPSEEK_V4_FLASH_TOP_K: Final = 6
_GRAPH_SAFE_MAX_TOKENS: Final = 1


def _mxfp4_active_expert_b1_enabled() -> bool:
    return bool(
        envs.VLLM_SM70_MXFP4_MOE_ACTIVE_EXPERT_B1
        and not envs.VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH
        and not envs.VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH
    )


def _single_token_weighted_reduce_enabled() -> bool:
    if not (
        envs.VLLM_SM70_MXFP4_MOE_FUSED_SHARED_REDUCE
        and (
            envs.VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH
            or envs.VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH
        )
    ):
        return False
    return hasattr(torch.ops._C, "awq_moe_single_token_weighted_reduce_out")


def _select_mxfp4_stage_dispatch(
    buffers: dict[str, torch.Tensor],
    *,
    num_tokens: int,
    num_experts: int,
    fully_replicated_experts: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if (
        num_tokens == 1
        and fully_replicated_experts
        and _mxfp4_active_expert_b1_enabled()
    ):
        return (
            buffers["compact_expert_offsets"],
            buffers["permuted_experts_id"],
            _DEEPSEEK_V4_FLASH_TOP_K,
        )
    return buffers["expert_offsets"], buffers["dense_expert_ids"], num_experts


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


class Mxfp4SM70MoEMethod(Mxfp4MoEMethod):
    """Exact-SM70 V4-Flash MXFP4 MoE using TurboMind packed GEMMs.

    ``Mxfp4MoEMethod`` owns the checkpoint parameter layout. Its generic
    Oracle backends are intentionally bypassed here because they are not an
    SM70 implementation and may select Marlin or a weight-emulation route.
    """

    def __init__(self, moe: FusedMoEConfig):
        FusedMoEMethodBase.__init__(self, moe)
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
        required_ops = (
            "mxfp4_sm70_prepare",
            "mxfp4_moe_dense_stage_sm70_out",
            "awq_moe_build_strided_ptrs",
        )
        if envs.VLLM_SM70_MXFP4_MOE_DIRECT_TOP6_DECODE:
            required_ops += ("mxfp4_moe_single_token_prepare_w13_sm70_out",)
        missing_ops = [name for name in required_ops if not hasattr(torch.ops._C, name)]
        if missing_ops:
            raise RuntimeError(
                "DeepSeek-V4 MXFP4 MoE on SM70 requires the TurboMind CUDA "
                "extension with " + ", ".join(missing_ops) + "."
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

        w13_tm_weights: list[torch.Tensor] = []
        w13_tm_scales: list[torch.Tensor] = []
        w13_meta: list[torch.Tensor] = []
        w2_tm_weights: list[torch.Tensor] = []
        w2_tm_scales: list[torch.Tensor] = []
        w2_meta: list[torch.Tensor] = []
        for expert_id in range(num_experts):
            # The converter only repacks nibbles/scales for TurboMind; it does
            # not dequantize or materialize a full-precision expert weight.
            w13_packed = unpack_mxfp4_weight(layer.w13_weight[expert_id].data)
            w13_scales = layer.w13_weight_scale[expert_id].data.t().contiguous()
            prepared_w13 = sm70_ops.mxfp4_sm70_prepare(
                w13_packed, w13_scales, MXFP4_GROUP_SIZE
            )
            w13_tm_weights.append(prepared_w13[0])
            w13_tm_scales.append(prepared_w13[1])
            w13_meta.append(prepared_w13[2])

            w2_packed = unpack_mxfp4_weight(layer.w2_weight[expert_id].data)
            w2_scales = layer.w2_weight_scale[expert_id].data.t().contiguous()
            prepared_w2 = sm70_ops.mxfp4_sm70_prepare(
                w2_packed, w2_scales, MXFP4_GROUP_SIZE
            )
            w2_tm_weights.append(prepared_w2[0])
            w2_tm_scales.append(prepared_w2[1])
            w2_meta.append(prepared_w2[2])

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False
        )
        layer.w13_tm_scales = Parameter(torch.stack(w13_tm_scales), requires_grad=False)
        layer.w13_tm_meta = Parameter(torch.stack(w13_meta), requires_grad=False)
        layer.w2_tm_weight = Parameter(torch.stack(w2_tm_weights), requires_grad=False)
        layer.w2_tm_scales = Parameter(torch.stack(w2_tm_scales), requires_grad=False)
        layer.w2_tm_meta = Parameter(torch.stack(w2_meta), requires_grad=False)

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
        self._allocate_graph_safe_b1_buffers(layer)

        # Raw checkpoint tensors are replaced by the equivalent TurboMind
        # packed e2m1/UE8M0 representation, never by dequantized weights.
        del layer.w13_weight
        del layer.w13_weight_scale
        del layer.w2_weight
        del layer.w2_weight_scale
        logger.info_once(
            "SM70 TurboMind MXFP4 MoE enabled for DeepSeek-V4-Flash "
            "(local_experts=%d, graph_safe_decode=B1, active_expert_b1=%s).",
            num_experts,
            _mxfp4_active_expert_b1_enabled(),
        )

    def _allocate_graph_safe_b1_buffers(self, layer: RoutedExperts) -> None:
        device = layer.w13_tm_weight.device
        top_k = _DEEPSEEK_V4_FLASH_TOP_K
        max_slots = _GRAPH_SAFE_MAX_TOKENS * top_k
        num_experts = int(layer.sm70_mxfp4_num_experts)
        hidden_size = int(layer.sm70_mxfp4_hidden_size)
        intermediate_size = int(layer.sm70_mxfp4_intermediate_size)

        layer._mxfp4_sm70_buf_output = torch.empty(
            _GRAPH_SAFE_MAX_TOKENS,
            hidden_size,
            dtype=torch.float16,
            device=device,
        )
        layer._mxfp4_sm70_buf_permuted_input = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        layer._mxfp4_sm70_buf_gate_up = torch.empty(
            max_slots,
            int(layer.sm70_mxfp4_w13_n_dim),
            dtype=torch.float16,
            device=device,
        )
        layer._mxfp4_sm70_buf_intermediate = torch.empty(
            max_slots, intermediate_size, dtype=torch.float16, device=device
        )
        layer._mxfp4_sm70_buf_sorted_output = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        layer._mxfp4_sm70_buf_expert_offsets = torch.empty(
            num_experts + 1, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_expert_offsets64 = torch.empty(
            num_experts + 1, dtype=torch.int64, device=device
        )
        layer._mxfp4_sm70_buf_inv_permuted_idx = torch.empty(
            _GRAPH_SAFE_MAX_TOKENS, top_k, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_topk_ids = torch.empty(
            _GRAPH_SAFE_MAX_TOKENS, top_k, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(_GRAPH_SAFE_MAX_TOKENS, top_k)
        layer._mxfp4_sm70_buf_permuted_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_permuted_experts_id = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_sorted_row_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_topk_ids_for_sort = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        sort_workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            max_slots, layer.global_num_experts
        )
        layer._mxfp4_sm70_buf_sort_workspace = torch.empty(
            sort_workspace_size, dtype=torch.int8, device=device
        )
        layer._mxfp4_sm70_buf_dense_expert_ids = torch.arange(
            num_experts, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_compact_expert_offsets = torch.arange(
            top_k + 1, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_compact_expert_offsets64 = torch.arange(
            top_k + 1, dtype=torch.int64, device=device
        )

    @staticmethod
    def _persistent_b1_buffers(layer: RoutedExperts) -> dict[str, torch.Tensor]:
        return {
            "output": layer._mxfp4_sm70_buf_output,
            "permuted_input": layer._mxfp4_sm70_buf_permuted_input,
            "gate_up": layer._mxfp4_sm70_buf_gate_up,
            "intermediate": layer._mxfp4_sm70_buf_intermediate,
            "sorted_output": layer._mxfp4_sm70_buf_sorted_output,
            "expert_offsets": layer._mxfp4_sm70_buf_expert_offsets,
            "expert_offsets64": layer._mxfp4_sm70_buf_expert_offsets64,
            "inv_permuted_idx": layer._mxfp4_sm70_buf_inv_permuted_idx,
            "topk_ids": layer._mxfp4_sm70_buf_topk_ids,
            "token_expert_indices": layer._mxfp4_sm70_buf_token_expert_indices,
            "permuted_idx": layer._mxfp4_sm70_buf_permuted_idx,
            "sort_workspace": layer._mxfp4_sm70_buf_sort_workspace,
            "permuted_experts_id": layer._mxfp4_sm70_buf_permuted_experts_id,
            "sorted_row_idx": layer._mxfp4_sm70_buf_sorted_row_idx,
            "topk_ids_for_sort": layer._mxfp4_sm70_buf_topk_ids_for_sort,
            "dense_expert_ids": layer._mxfp4_sm70_buf_dense_expert_ids,
            "compact_expert_offsets": (layer._mxfp4_sm70_buf_compact_expert_offsets),
            "compact_expert_offsets64": (
                layer._mxfp4_sm70_buf_compact_expert_offsets64
            ),
        }

    @staticmethod
    def _eager_buffers(
        layer: RoutedExperts, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        device = layer.w13_tm_weight.device
        top_k = _DEEPSEEK_V4_FLASH_TOP_K
        total_slots = num_tokens * top_k
        num_experts = int(layer.sm70_mxfp4_num_experts)
        hidden_size = int(layer.sm70_mxfp4_hidden_size)
        intermediate_size = int(layer.sm70_mxfp4_intermediate_size)
        sort_workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            total_slots, layer.global_num_experts
        )
        return {
            "output": torch.empty(
                num_tokens, hidden_size, dtype=torch.float16, device=device
            ),
            "permuted_input": torch.empty(
                total_slots, hidden_size, dtype=torch.float16, device=device
            ),
            "gate_up": torch.empty(
                total_slots,
                int(layer.sm70_mxfp4_w13_n_dim),
                dtype=torch.float16,
                device=device,
            ),
            "intermediate": torch.empty(
                total_slots, intermediate_size, dtype=torch.float16, device=device
            ),
            "sorted_output": torch.empty(
                total_slots, hidden_size, dtype=torch.float16, device=device
            ),
            "expert_offsets": torch.empty(
                num_experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                num_experts + 1, dtype=torch.int64, device=device
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
            "permuted_idx": torch.empty(total_slots, dtype=torch.int32, device=device),
            "sort_workspace": torch.empty(
                sort_workspace_size, dtype=torch.int8, device=device
            ),
            "permuted_experts_id": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "sorted_row_idx": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "topk_ids_for_sort": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "dense_expert_ids": layer._mxfp4_sm70_buf_dense_expert_ids,
            "compact_expert_offsets": (layer._mxfp4_sm70_buf_compact_expert_offsets),
            "compact_expert_offsets64": (
                layer._mxfp4_sm70_buf_compact_expert_offsets64
            ),
        }

    def _get_buffers(
        self, layer: RoutedExperts, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        if num_tokens == _GRAPH_SAFE_MAX_TOKENS:
            return self._persistent_b1_buffers(layer)
        # B1 owns fixed buffers before capture. Larger M remains a functional
        # eager/prefill route; during a fixed-shape graph capture PyTorch owns
        # these allocations in that graph's pool. Its replay stability is a
        # separate GPU gate and is not claimed by the B1 contract.
        return self._eager_buffers(layer, num_tokens)

    @staticmethod
    def _apply_swiglu(
        layer: RoutedExperts, out: torch.Tensor, gate_up: torch.Tensor
    ) -> None:
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
        del shared_experts_input
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
        self._sm70_mxfp4_shared_output_finalized = False

        direct_top6 = (
            num_tokens == 1
            and envs.VLLM_SM70_MXFP4_MOE_DIRECT_TOP6_DECODE
            and layer.expert_map is None
            and layer.local_num_experts == layer.global_num_experts
        )
        fused_shared_reduce = bool(
            direct_top6
            and shared_experts is not None
            and envs.VLLM_SM70_MXFP4_MOE_FUSED_SHARED_REDUCE
            and hasattr(
                torch.ops._C,
                "sm70_moe_single_token_weighted_reduce_add_out",
            )
        )
        layer._sm70_mxfp4_pending_shared_reduce = False
        if direct_top6:
            sm70_ops.mxfp4_moe_single_token_prepare_w13_sm70_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                x,
                topk_ids,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                buffers["compact_expert_offsets"],
                buffers["inv_permuted_idx"],
                buffers["permuted_experts_id"],
                layer.sm70_mxfp4_w13_k_dim,
                layer.sm70_mxfp4_w13_n_dim,
                layer.sm70_mxfp4_group_size,
                layer.sm70_mxfp4_hidden_size,
            )
            self._apply_swiglu(layer, buffers["intermediate"], buffers["gate_up"])
            sm70_ops.mxfp4_moe_dense_stage_sm70_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["compact_expert_offsets"],
                buffers["permuted_experts_id"],
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                _DEEPSEEK_V4_FLASH_TOP_K,
                layer.sm70_mxfp4_w2_k_dim,
                layer.sm70_mxfp4_w2_n_dim,
                layer.sm70_mxfp4_group_size,
            )
            if fused_shared_reduce:
                layer._sm70_mxfp4_pending_shared_reduce = True
                layer._sm70_mxfp4_pending_topk_weights = topk_weights
            elif _single_token_weighted_reduce_enabled():
                if not torch.compiler.is_compiling():
                    logger.info_once(
                        "SM70 MXFP4 MoE single-token weighted-reduce path "
                        "enabled (top_k=%d).",
                        _DEEPSEEK_V4_FLASH_TOP_K,
                    )
                sm70_ops.awq_moe_single_token_weighted_reduce_out(
                    buffers["sorted_output"],
                    topk_weights,
                    buffers["inv_permuted_idx"],
                    output,
                    _DEEPSEEK_V4_FLASH_TOP_K,
                    layer.sm70_mxfp4_hidden_size,
                )
            else:
                torch.ops._moe_C.moe_unpermute(
                    buffers["sorted_output"],
                    topk_weights,
                    buffers["inv_permuted_idx"],
                    buffers["compact_expert_offsets64"],
                    _DEEPSEEK_V4_FLASH_TOP_K,
                    output,
                )
            return output

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

        stage_offsets, stage_expert_ids, stage_expert_count = (
            _select_mxfp4_stage_dispatch(
                buffers,
                num_tokens=num_tokens,
                num_experts=layer.sm70_mxfp4_num_experts,
                fully_replicated_experts=(
                    layer.expert_map is None
                    and layer.local_num_experts == layer.global_num_experts
                ),
            )
        )

        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            buffers["gate_up"],
            buffers["permuted_input"],
            stage_offsets,
            stage_expert_ids,
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            stage_expert_count,
            layer.sm70_mxfp4_w13_k_dim,
            layer.sm70_mxfp4_w13_n_dim,
            layer.sm70_mxfp4_group_size,
        )
        self._apply_swiglu(layer, buffers["intermediate"], buffers["gate_up"])
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            buffers["sorted_output"],
            buffers["intermediate"],
            stage_offsets,
            stage_expert_ids,
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            stage_expert_count,
            layer.sm70_mxfp4_w2_k_dim,
            layer.sm70_mxfp4_w2_n_dim,
            layer.sm70_mxfp4_group_size,
        )
        torch.ops._moe_C.moe_unpermute(
            buffers["sorted_output"],
            topk_weights,
            buffers["inv_permuted_idx"],
            buffers["expert_offsets64"],
            _DEEPSEEK_V4_FLASH_TOP_K,
            output,
        )
        return output

    def finalize_shared_expert_output(
        self,
        layer: RoutedExperts,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if not getattr(layer, "_sm70_mxfp4_pending_shared_reduce", False):
            return shared_output, fused_output
        if shared_output is None:
            raise RuntimeError(
                "SM70 MXFP4 fused shared reduce requires shared-expert output."
            )
        topk_weights = layer._sm70_mxfp4_pending_topk_weights
        buffers = self._persistent_b1_buffers(layer)
        shared_scale = 1.0 / routed_scaling_factor
        sm70_ops.sm70_moe_single_token_weighted_reduce_add_out(
            buffers["sorted_output"],
            topk_weights,
            buffers["inv_permuted_idx"],
            shared_output,
            fused_output,
            shared_scale,
            _DEEPSEEK_V4_FLASH_TOP_K,
            layer.sm70_mxfp4_hidden_size,
        )
        layer._sm70_mxfp4_pending_shared_reduce = False
        layer._sm70_mxfp4_pending_topk_weights = None
        self._sm70_mxfp4_shared_output_finalized = True
        if not torch.compiler.is_compiling():
            logger.info_once(
                "SM70 MXFP4 MoE weighted-reduce + shared-add fusion enabled."
            )
        # The shared-MoE custom-op schema requires two Tensor outputs. The
        # outer runner discards this already-consumed tensor after unpacking.
        return shared_output, fused_output

    def is_shared_expert_output_finalized(self) -> bool:
        return getattr(self, "_sm70_mxfp4_shared_output_finalized", False)

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
