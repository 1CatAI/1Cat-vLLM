# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SM70 AWQ MoE method using TurboMind s884h GEMM kernels.

Pre-allocates all intermediate buffers (lmdeploy-style) for CUDA graph
compatibility. Uses batched Gemm::Run via StridedPtr arrays, zero CUDA syncs.
Falls back to per-expert loop if batched GEMM is unavailable.
"""

import os

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.linear import set_weight_attrs

logger = init_logger(__name__)

# Keep a small persistent workspace for decode / cudagraph capture.
# Larger MoE batches allocate temporary workspaces on demand so we do not
# permanently reserve hundreds of MiB per layer.
_DEFAULT_PERSISTENT_MAX_TOKENS = 32


def _single_token_compact_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_AWQ_ENABLE_SINGLE_TOKEN_COMPACT")
    if raw is None:
        return True
    return raw == "1"


def _single_token_compact_compare_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_AWQ_COMPACT_COMPARE")
    return raw == "1"


def _round_up(value: int, align: int) -> int:
    if align <= 0:
        return value
    return ((value + align - 1) // align) * align


def _pad_last_dim(t: torch.Tensor, pad_elems: int) -> torch.Tensor:
    if pad_elems <= 0:
        return t
    pad_shape = (*t.shape[:-1], pad_elems)
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat((t, pad), dim=-1)


def _pad_penultimate_dim(t: torch.Tensor, pad_elems: int) -> torch.Tensor:
    if pad_elems <= 0:
        return t
    pad_shape = (*t.shape[:-2], pad_elems, t.shape[-1])
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat((t, pad), dim=-2)


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
    assert pad_n % pack_factor == 0
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
    assert new_k % group_size == 0
    pad_k = new_k - old_k
    old_groups = int(scales.shape[-2])
    new_groups = new_k // group_size
    qweight = _pad_penultimate_dim(qweight, pad_k)
    qzeros = _pad_penultimate_dim(qzeros, new_groups - old_groups)
    scales = _pad_penultimate_dim(scales, new_groups - old_groups)
    return qweight, scales, qzeros, new_k


class AWQSM70MoEMethod(FusedMoEMethodBase):
    """AWQ MoE method for SM70 (V100) using TurboMind GEMM kernels.

    Only supports group_size=32/64/128, float16, 4-bit weights.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        if weight_bits != 4:
            raise ValueError(
                f"AWQSM70MoEMethod only supports 4-bit, got {weight_bits}."
            )
        if group_size not in (32, 64, 128):
            raise ValueError(
                f"AWQSM70MoEMethod supports group_size=32/64/128, got {group_size}."
            )
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.pack_factor = 32 // weight_bits  # 8

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )
        extra_weight_attrs.pop("intermediate_size_full", None)

        w13_qweight = Parameter(
            torch.empty(num_experts, hidden_size,
                        2 * intermediate_size_per_partition // self.pack_factor,
                        dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(
            torch.empty(num_experts, intermediate_size_per_partition,
                        hidden_size // self.pack_factor, dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size

        w13_scales = Parameter(
            torch.empty(num_experts, num_groups_w13,
                        intermediate_size_per_partition * 2, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(
            torch.empty(num_experts, num_groups_w2, hidden_size,
                        dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_qzeros = Parameter(
            torch.empty(num_experts, num_groups_w13,
                        2 * intermediate_size_per_partition // self.pack_factor,
                        dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = Parameter(
            torch.empty(num_experts, num_groups_w2,
                        hidden_size // self.pack_factor, dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert AWQ weights to TurboMind format and pre-allocate buffers."""
        align = self.group_size
        hidden_logical_size = int(layer.w13_qweight.shape[1])
        w13_logical_out = int(layer.w13_scales.shape[-1])
        intermediate_logical_size = w13_logical_out // 2
        w2_logical_out = int(layer.w2_scales.shape[-1])

        layer.w13_qweight, layer.w13_scales, layer.w13_qzeros, w13_aligned_out = (
            _align_awq_output_dim(
                layer.w13_qweight,
                layer.w13_scales,
                layer.w13_qzeros,
                self.pack_factor,
                align * 2,
            )
        )
        aligned_intermediate_size = w13_aligned_out // 2

        layer.w2_qweight, layer.w2_scales, layer.w2_qzeros, _ = (
            _align_awq_input_dim(
                layer.w2_qweight,
                layer.w2_scales,
                layer.w2_qzeros,
                self.group_size,
                align,
            )
        )
        layer.w2_qweight, layer.w2_scales, layer.w2_qzeros, hidden_aligned_size = (
            _align_awq_output_dim(
                layer.w2_qweight,
                layer.w2_scales,
                layer.w2_qzeros,
                self.pack_factor,
                align,
            )
        )

        layer.sm70_hidden_logical_size = hidden_logical_size
        layer.sm70_hidden_aligned_size = hidden_aligned_size
        layer.sm70_intermediate_logical_size = intermediate_logical_size
        layer.sm70_intermediate_aligned_size = aligned_intermediate_size
        if (
            aligned_intermediate_size != intermediate_logical_size
            or hidden_aligned_size != hidden_logical_size
            or w2_logical_out != hidden_logical_size
        ):
            logger.info_once(
                "SM70 MoE alignment layer=%s hidden=%d->%d inter=%d->%d",
                getattr(layer, "layer_name", "<unknown>"),
                hidden_logical_size,
                hidden_aligned_size,
                intermediate_logical_size,
                aligned_intermediate_size,
            )

        num_experts = layer.w13_qweight.shape[0]
        device = layer.w13_qweight.device

        # --- Prepare TurboMind weights per expert ---
        w13_tm_weights, w13_tm_scales, w13_meta = [], [], []
        w2_tm_weights, w2_tm_scales, w2_meta = [], [], []

        for e in range(num_experts):
            r13 = ops.awq_sm70_prepare(
                layer.w13_qweight[e], layer.w13_scales[e],
                layer.w13_qzeros[e], self.group_size,
                interleave_gated_silu=True)
            w13_tm_weights.append(r13[0])
            w13_tm_scales.append(r13[1])
            w13_meta.append(r13[2])

            r2 = ops.awq_sm70_prepare(
                layer.w2_qweight[e], layer.w2_scales[e],
                layer.w2_qzeros[e], self.group_size)
            w2_tm_weights.append(r2[0])
            w2_tm_scales.append(r2[1])
            w2_meta.append(r2[2])

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False)
        layer.w13_tm_scales = Parameter(
            torch.stack(w13_tm_scales), requires_grad=False)
        layer.w2_tm_weight = Parameter(
            torch.stack(w2_tm_weights), requires_grad=False)
        layer.w2_tm_scales = Parameter(
            torch.stack(w2_tm_scales), requires_grad=False)

        # Cache meta as CPU ints (zero-cost at inference)
        layer.w13_meta_list = [
            (int(w13_meta[i][0].item()), int(w13_meta[i][1].item()))
            for i in range(num_experts)
        ]
        layer.w2_meta_list = [
            (int(w2_meta[i][0].item()), int(w2_meta[i][1].item()))
            for i in range(num_experts)
        ]
        layer.sm70_num_experts = num_experts

        # Dimensions for batched GEMM
        layer.sm70_w13_k_dim = layer.w13_tm_weight.shape[1]
        layer.sm70_w13_n_dim = layer.w13_tm_weight.shape[2] * 8
        layer.sm70_w2_k_dim = layer.w2_tm_weight.shape[1]
        layer.sm70_w2_n_dim = layer.w2_tm_weight.shape[2] * 8
        intermediate_size = layer.sm70_w2_k_dim
        hidden_size = layer.sm70_hidden_logical_size
        layer.sm70_intermediate_size = intermediate_size

        # --- Build StridedPtr arrays for batched GEMM ---
        w13_k_ld, w13_q_ld = layer.w13_meta_list[0]
        w2_k_ld, w2_q_ld = layer.w2_meta_list[0]
        try:
            w13_ptrs = ops.awq_moe_build_strided_ptrs(
                layer.w13_tm_weight, layer.w13_tm_scales,
                w13_k_ld, w13_q_ld, num_experts)
            w2_ptrs = ops.awq_moe_build_strided_ptrs(
                layer.w2_tm_weight, layer.w2_tm_scales,
                w2_k_ld, w2_q_ld, num_experts)
            layer.w13_strided_ptrs_w = Parameter(
                w13_ptrs[0], requires_grad=False)
            layer.w13_strided_ptrs_s = Parameter(
                w13_ptrs[1], requires_grad=False)
            layer.w2_strided_ptrs_w = Parameter(
                w2_ptrs[0], requires_grad=False)
            layer.w2_strided_ptrs_s = Parameter(
                w2_ptrs[1], requires_grad=False)
            layer.w13_strided_ptrs_w_rows = layer.w13_strided_ptrs_w.view(
                num_experts, -1)
            layer.w13_strided_ptrs_s_rows = layer.w13_strided_ptrs_s.view(
                num_experts, -1)
            layer.w2_strided_ptrs_w_rows = layer.w2_strided_ptrs_w.view(
                num_experts, -1)
            layer.w2_strided_ptrs_s_rows = layer.w2_strided_ptrs_s.view(
                num_experts, -1)
            layer.sm70_ptr_row_bytes = layer.w13_strided_ptrs_w_rows.shape[1]
            layer.sm70_batched_ready = True
            logger.info_once("SM70 MoE: batched GEMM enabled (%d experts)",
                             num_experts)
        except Exception as e:
            layer.sm70_batched_ready = False
            logger.warning("SM70 MoE: batched GEMM unavailable (%s), "
                           "using per-expert loop fallback.", e)

        # --- Pre-allocate a small persistent decode workspace ---
        top_k = self.moe.experts_per_token
        persistent_tokens = _DEFAULT_PERSISTENT_MAX_TOKENS
        max_slots = persistent_tokens * top_k
        layer._buf_max_tokens = persistent_tokens
        layer._buf_max_slots = max_slots
        layer._buf_top_k = top_k
        layer._buf_expert_counts = torch.empty(
            num_experts, dtype=torch.int32, device=device)
        layer._buf_expert_offsets = torch.empty(
            num_experts + 1, dtype=torch.int32, device=device)
        layer._buf_expert_offsets64 = torch.empty(
            num_experts + 1, dtype=torch.int64, device=device)
        layer._buf_gate_up = torch.empty(
            max_slots, layer.sm70_w13_n_dim, dtype=torch.float16, device=device)
        layer._buf_intermediate = torch.empty(
            max_slots, intermediate_size, dtype=torch.float16, device=device)
        layer._buf_permuted_input = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device)
        layer._buf_sorted_output = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device)
        layer._buf_inv_permuted_idx = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device)
        layer._buf_topk_ids_i32 = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device)
        layer._buf_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device).view(
                persistent_tokens, top_k)
        layer._buf_permuted_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device)
        layer._buf_m_indices = torch.empty(
            max_slots, dtype=torch.int32, device=device)
        layer._buf_output = torch.empty(
            persistent_tokens, hidden_size, dtype=torch.float16,
            device=device)
        layer._buf_single_topk_ids_i64 = torch.empty(
            top_k, dtype=torch.int64, device=device)
        layer._buf_single_w13_ptrs_w = torch.empty(
            top_k, layer.sm70_ptr_row_bytes, dtype=torch.uint8, device=device)
        layer._buf_single_w13_ptrs_s = torch.empty(
            top_k, layer.sm70_ptr_row_bytes, dtype=torch.uint8, device=device)
        layer._buf_single_w2_ptrs_w = torch.empty(
            top_k, layer.sm70_ptr_row_bytes, dtype=torch.uint8, device=device)
        layer._buf_single_w2_ptrs_s = torch.empty(
            top_k, layer.sm70_ptr_row_bytes, dtype=torch.uint8, device=device)
        layer._buf_single_expert_offsets = torch.arange(
            top_k + 1, dtype=torch.int32, device=device)
        layer._buf_single_expert_offsets64 = torch.arange(
            top_k + 1, dtype=torch.int64, device=device)
        layer._buf_single_inv_permuted_idx = torch.arange(
            top_k, dtype=torch.int32, device=device).view(1, top_k)

        # Free original weights
        del layer.w13_qweight, layer.w13_scales, layer.w13_qzeros
        del layer.w2_qweight, layer.w2_scales, layer.w2_qzeros

    def _get_buffers(self, layer: torch.nn.Module, total_slots: int,
                     num_tokens: int):
        """Use persistent decode buffers when they fit, otherwise temp ones."""
        if (total_slots <= layer._buf_max_slots
                and num_tokens <= layer._buf_max_tokens):
            return {
                "output": layer._buf_output[:num_tokens],
                "permuted_input": layer._buf_permuted_input[:total_slots],
                "sorted_output": layer._buf_sorted_output[:total_slots],
                "gate_up": layer._buf_gate_up[:total_slots],
                "intermediate": layer._buf_intermediate[:total_slots],
                "expert_offsets": layer._buf_expert_offsets,
                "expert_offsets64": layer._buf_expert_offsets64,
                "inv_permuted_idx": layer._buf_inv_permuted_idx[:num_tokens],
                "topk_ids_i32": layer._buf_topk_ids_i32[:num_tokens],
                "token_expert_indices":
                layer._buf_token_expert_indices[:num_tokens],
                "permuted_idx": layer._buf_permuted_idx[:total_slots],
                "m_indices": layer._buf_m_indices[:total_slots],
            }

        device = layer._buf_output.device
        top_k = layer._buf_top_k
        hidden_size = layer.sm70_hidden_logical_size
        return {
            "output": torch.empty(num_tokens,
                                   hidden_size,
                                   dtype=torch.float16,
                                   device=device),
            "permuted_input": torch.empty(total_slots,
                                           hidden_size,
                                           dtype=torch.float16,
                                           device=device),
            "sorted_output": torch.empty(total_slots,
                                          hidden_size,
                                          dtype=torch.float16,
                                          device=device),
            "gate_up": torch.empty(total_slots,
                                    layer.sm70_w13_n_dim,
                                    dtype=torch.float16,
                                    device=device),
            "intermediate": torch.empty(total_slots,
                                         layer.sm70_intermediate_size,
                                         dtype=torch.float16,
                                         device=device),
            "expert_offsets": torch.empty(layer.sm70_num_experts + 1,
                                           dtype=torch.int32,
                                           device=device),
            "expert_offsets64": torch.empty(layer.sm70_num_experts + 1,
                                             dtype=torch.int64,
                                             device=device),
            "inv_permuted_idx": torch.empty(num_tokens,
                                             top_k,
                                             dtype=torch.int32,
                                             device=device),
            "topk_ids_i32": torch.empty(num_tokens,
                                         top_k,
                                         dtype=torch.int32,
                                         device=device),
            "token_expert_indices": torch.arange(
                total_slots, dtype=torch.int32, device=device).view(
                    num_tokens, top_k),
            "permuted_idx": torch.empty(total_slots,
                                         dtype=torch.int32,
                                         device=device),
            "m_indices": torch.empty(total_slots,
                                      dtype=torch.int32,
                                      device=device),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """MoE forward: batched GEMM (preferred) or sorted-loop fallback."""
        if (
            getattr(layer, "sm70_batched_ready", False)
            and x.shape[0] == 1
            and _single_token_compact_enabled()
        ):
            if (
                _single_token_compact_compare_enabled()
                and not torch.cuda.is_current_stream_capturing()
            ):
                compact_out = self._apply_single_token_compact(
                    layer, x, topk_weights, topk_ids
                )
                batched_out = self._apply_batched(
                    layer, x, topk_weights, topk_ids
                )
                diff = (compact_out - batched_out).abs()
                max_diff = float(diff.max().item())
                if max_diff != 0.0:
                    logger.warning(
                        "SM70 compact mismatch layer=%s max_diff=%.6f mean_diff=%.6f "
                        "topk_ids=%s local_experts=%d",
                        getattr(layer, "layer_name", "<unknown>"),
                        max_diff,
                        float(diff.float().mean().item()),
                        topk_ids.view(-1).tolist(),
                        int(layer.sm70_num_experts),
                    )
                else:
                    logger.info_once(
                        "SM70 compact compare matched for layer=%s",
                        getattr(layer, "layer_name", "<unknown>"),
                    )
                return batched_out
            if _single_token_compact_compare_enabled():
                logger.info_once(
                    "SM70 compact compare is skipped during CUDA graph capture."
                )
            return self._apply_single_token_compact(
                layer, x, topk_weights, topk_ids)
        if getattr(layer, "sm70_batched_ready", False):
            return self._apply_batched(layer, x, topk_weights, topk_ids)
        return self._apply_sorted_loop(layer, x, topk_weights, topk_ids)

    def _get_single_token_active_ids(
        self,
        layer: torch.nn.Module,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        active_ids = topk_ids.view(-1)
        if active_ids.dtype == torch.int64:
            return active_ids
        single_topk_ids_i64 = layer._buf_single_topk_ids_i64
        single_topk_ids_i64.copy_(active_ids, non_blocking=True)
        return single_topk_ids_i64

    def _apply_single_token_compact(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token decode fast path with compact active experts."""
        top_k = topk_ids.shape[1]
        buffers = self._get_buffers(layer, top_k, 1)
        output = buffers["output"]
        compact_input = buffers["permuted_input"][:top_k]
        intermediate = buffers["intermediate"][:top_k]
        sorted_output = buffers["sorted_output"][:top_k]
        active_ids = self._get_single_token_active_ids(layer, topk_ids)
        inv_permuted_idx = layer._buf_single_inv_permuted_idx

        ops.awq_moe_single_token_sm70_out(
            output,
            x,
            topk_weights,
            active_ids,
            layer.w13_strided_ptrs_w_rows,
            layer.w13_strided_ptrs_s_rows,
            layer.w2_strided_ptrs_w_rows,
            layer.w2_strided_ptrs_s_rows,
            compact_input,
            intermediate,
            sorted_output,
            layer._buf_single_w13_ptrs_w,
            layer._buf_single_w13_ptrs_s,
            layer._buf_single_w2_ptrs_w,
            layer._buf_single_w2_ptrs_s,
            layer._buf_single_expert_offsets,
            inv_permuted_idx,
            layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim,
            layer.sm70_w2_k_dim,
            layer.sm70_w2_n_dim,
            self.group_size,
            layer.sm70_hidden_logical_size,
        )
        return output

    def _permute_tokens_by_expert(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        buffers: dict[str, torch.Tensor],
    ):
        """Permute tokens by expert using the native MoE CUDA kernels."""
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]

        permuted_input = buffers["permuted_input"]
        expert_offsets64 = buffers["expert_offsets64"]
        inv_permuted_idx = buffers["inv_permuted_idx"]
        permuted_idx = buffers["permuted_idx"]
        m_indices = buffers["m_indices"]
        topk_ids_i32 = buffers["topk_ids_i32"]
        token_expert_indices = buffers["token_expert_indices"]

        topk_ids_i32.copy_(topk_ids, non_blocking=True)
        torch.ops._moe_C.moe_permute(
            x,
            topk_ids_i32,
            token_expert_indices,
            None,
            num_experts,
            num_experts,
            top_k,
            None,
            permuted_input,
            expert_offsets64,
            inv_permuted_idx,
            permuted_idx,
            m_indices,
        )
        buffers["expert_offsets"].copy_(expert_offsets64, non_blocking=True)
        return permuted_input, expert_offsets64, inv_permuted_idx

    def _apply_batched(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Batched GEMM via TurboMind. Zero CUDA syncs, graph safe."""
        num_tokens, hidden_size = x.shape
        top_k = topk_ids.shape[1]
        num_experts = layer.sm70_num_experts
        total_slots = num_tokens * top_k

        buffers = self._get_buffers(layer, total_slots, num_tokens)
        output = buffers["output"]
        output.zero_()
        if total_slots == 0:
            return output

        (permuted_input, expert_offsets64,
         inv_permuted_idx) = self._permute_tokens_by_expert(
            layer, x, topk_ids, num_experts, buffers)
        expert_offsets = buffers["expert_offsets"]
        intermediate = buffers["intermediate"]
        sorted_output = buffers["sorted_output"]

        # Batched w13 GEMM (gate+up) — write into a pre-allocated buffer.
        ops.awq_moe_gemm_sm70_out(
            intermediate,
            permuted_input, expert_offsets,
            layer.w13_strided_ptrs_w, layer.w13_strided_ptrs_s,
            num_experts, layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim, self.group_size,
            True,
        )

        # Batched w2 GEMM (down projection) — write into a pre-allocated buffer.
        ops.awq_moe_gemm_sm70_out(
            sorted_output,
            intermediate, expert_offsets,
            layer.w2_strided_ptrs_w, layer.w2_strided_ptrs_s,
            num_experts, layer.sm70_w2_k_dim,
            layer.sm70_w2_n_dim, self.group_size,
        )
        sorted_output_logical = sorted_output[:, : layer.sm70_hidden_logical_size]
        torch.ops._moe_C.moe_unpermute(
            sorted_output_logical,
            topk_weights,
            inv_permuted_idx,
            expert_offsets64,
            top_k,
            output,
        )
        return output

    def _apply_sorted_loop(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback: sort + single CPU sync + per-expert GEMM loop."""
        num_tokens, hidden_size = x.shape
        top_k = topk_ids.shape[1]
        num_experts = layer.sm70_num_experts
        total_slots = num_tokens * top_k

        buffers = self._get_buffers(layer, total_slots, num_tokens)
        output = buffers["output"]
        output.zero_()
        if total_slots == 0:
            return output

        flat_ids = topk_ids.view(-1)
        flat_weights = topk_weights.view(-1)
        token_origin = (
            torch.arange(num_tokens, device=x.device, dtype=torch.int64)
            .unsqueeze(1).expand(num_tokens, top_k).reshape(-1))

        sorted_order = torch.argsort(flat_ids.long(), stable=True)
        sorted_token_origin = token_origin[sorted_order]
        sorted_weights = flat_weights[sorted_order]
        sorted_input = x[sorted_token_origin]

        sorted_expert_ids = flat_ids[sorted_order]
        expert_counts = torch.bincount(
            sorted_expert_ids.long(), minlength=num_experts)
        expert_offsets = torch.zeros(
            num_experts + 1, dtype=torch.int64, device=x.device)
        torch.cumsum(expert_counts, dim=0, out=expert_offsets[1:])

        # Optimization: Single CPU sync for all offsets at once
        # This is much faster than per-expert sync in the loop
        h_offsets = expert_offsets.cpu().numpy()
        h_counts = expert_counts.cpu().numpy()

        sorted_output = torch.empty(
            total_slots, layer.sm70_w2_n_dim, dtype=x.dtype, device=x.device)

        for e in range(num_experts):
            # Use pre-synced numpy arrays (no additional sync)
            if h_counts[e] == 0:
                continue

            start, end = int(h_offsets[e]), int(h_offsets[e + 1])
            expert_input = sorted_input[start:end]

            w13_k_ld, w13_q_ld = layer.w13_meta_list[e]
            intermediate = torch.empty(
                end - start, layer.sm70_intermediate_size,
                dtype=x.dtype, device=x.device)
            ops.awq_gemm_sm70_out(
                intermediate,
                expert_input,
                layer.w13_tm_weight[e],
                layer.w13_tm_scales[e],
                self.group_size,
                w13_k_ld,
                w13_q_ld,
                True,
            )

            w2_k_ld, w2_q_ld = layer.w2_meta_list[e]
            expert_output = ops.awq_gemm_sm70(
                intermediate,
                layer.w2_tm_weight[e],
                layer.w2_tm_scales[e],
                self.group_size,
                w2_k_ld,
                w2_q_ld,
            )
            sorted_output[start:end] = expert_output

        sorted_output_logical = sorted_output[:, : layer.sm70_hidden_logical_size]
        weighted = sorted_output_logical * sorted_weights.unsqueeze(1).to(x.dtype)
        output.index_add_(0, sorted_token_origin, weighted)
        return output

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None
