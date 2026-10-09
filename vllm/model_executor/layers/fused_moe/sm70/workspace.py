# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared SM70 MoE buffer lifecycle; format-specific stages migrate separately."""

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    LayerWorkspaceView,
)


class Fp8MoEWorkspace:
    """Persistent and overflow buffers shared independently of weight encoding.

    Legacy attributes stay on the layer, so graph addresses and patches keep
    their original owner. FP8 is the first consumer; stage kernels, compare
    and other formats are migrated in subsequent review scopes.
    """

    def __init__(self, top_k: int, use_permute_with_scratch: bool):
        self.top_k = top_k
        self.use_permute_with_scratch = use_permute_with_scratch

    def allocate(
        self,
        layer: RoutedExperts,
        *,
        buffer_prefix: str,
        persistent_max_tokens: int,
        empty_weight_dtype: torch.dtype,
        empty_scale_dtype: torch.dtype,
    ) -> None:
        buffers = LayerWorkspaceView(layer, buffer_prefix)
        device = layer.w13_tm_weight.device
        top_k = self.top_k
        persistent_tokens = persistent_max_tokens
        max_slots = persistent_tokens * top_k
        hidden_size = layer.sm70_hidden_logical_size
        num_experts = layer.sm70_num_experts
        buffers.max_tokens = persistent_tokens
        buffers.max_slots = max_slots
        buffers.top_k = top_k
        buffers.output = torch.empty(
            persistent_tokens, hidden_size, dtype=torch.float16, device=device
        )
        buffers.permuted_input = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        buffers.intermediate = torch.empty(
            max_slots, layer.sm70_intermediate_size, dtype=torch.float16, device=device
        )
        buffers.gate_up = torch.empty(
            max_slots, layer.sm70_w13_n_dim, dtype=torch.float16, device=device
        )
        buffers.sorted_output = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        buffers.expert_offsets = torch.empty(
            num_experts + 1, dtype=torch.int32, device=device
        )
        buffers.expert_offsets64 = torch.empty(
            num_experts + 1, dtype=torch.int64, device=device
        )
        buffers.inv_permuted_idx = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        buffers.topk_ids = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        buffers.token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(persistent_tokens, top_k)
        buffers.permuted_idx = torch.empty(max_slots, dtype=torch.int32, device=device)
        buffers.sorted_expert_ids = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        if self.use_permute_with_scratch:
            sort_workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
                max_slots, layer.global_num_experts
            )
        else:
            sort_workspace_size = 0
        buffers.sort_workspace = torch.empty(
            sort_workspace_size, dtype=torch.int8, device=device
        )
        buffers.permuted_experts_id = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        buffers.sorted_row_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        buffers.topk_ids_for_sort = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        buffers.active_expert_offsets = torch.arange(
            max_slots + 1, dtype=torch.int32, device=device
        )
        buffers.sorted_weights = torch.empty(top_k, dtype=torch.float32, device=device)
        buffers.broadcast_input_indices = torch.empty(
            top_k, dtype=torch.int32, device=device
        )
        buffers.dense_expert_ids = torch.arange(
            num_experts, dtype=torch.int32, device=device
        )
        ptr_row_bytes = int(layer.sm70_ptr_row_bytes)
        buffers.compact_w13_ptrs_w = torch.empty(
            top_k * ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.compact_w13_ptrs_s = torch.empty(
            top_k * ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.legacy_w13_ptrs_w = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.legacy_w13_ptrs_s = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.legacy_w2_ptrs_w = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.legacy_w2_ptrs_s = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.empty_weight = torch.empty(0, dtype=empty_weight_dtype, device=device)
        buffers.empty_scale = torch.empty(0, dtype=empty_scale_dtype, device=device)

    def get(
        self,
        layer: RoutedExperts,
        total_slots: int,
        num_tokens: int,
        *,
        buffer_prefix: str,
    ) -> dict[str, torch.Tensor]:
        buffers = LayerWorkspaceView(layer, buffer_prefix)
        if total_slots <= buffers.max_slots and num_tokens <= buffers.max_tokens:
            return {
                "output": buffers.output[:num_tokens],
                "permuted_input": buffers.permuted_input[:total_slots],
                "intermediate": buffers.intermediate[:total_slots],
                "gate_up": buffers.gate_up[:total_slots],
                "sorted_output": buffers.sorted_output[:total_slots],
                "expert_offsets": buffers.expert_offsets,
                "expert_offsets64": buffers.expert_offsets64,
                "inv_permuted_idx": buffers.inv_permuted_idx[:num_tokens],
                "topk_ids": buffers.topk_ids[:num_tokens],
                "token_expert_indices": buffers.token_expert_indices[:num_tokens],
                "permuted_idx": buffers.permuted_idx[:total_slots],
                "sorted_expert_ids": buffers.sorted_expert_ids[:total_slots],
                "sort_workspace": buffers.sort_workspace,
                "permuted_experts_id": buffers.permuted_experts_id[:total_slots],
                "sorted_row_idx": buffers.sorted_row_idx[:total_slots],
                "topk_ids_for_sort": buffers.topk_ids_for_sort[:total_slots],
                "active_expert_offsets": (
                    buffers.active_expert_offsets[: total_slots + 1]
                ),
                "sorted_weights": buffers.sorted_weights,
                "broadcast_input_indices": buffers.broadcast_input_indices,
                "compact_w13_ptrs_w": buffers.compact_w13_ptrs_w,
                "compact_w13_ptrs_s": buffers.compact_w13_ptrs_s,
                "legacy_w13_ptrs_w": buffers.legacy_w13_ptrs_w,
                "legacy_w13_ptrs_s": buffers.legacy_w13_ptrs_s,
                "legacy_w2_ptrs_w": buffers.legacy_w2_ptrs_w,
                "legacy_w2_ptrs_s": buffers.legacy_w2_ptrs_s,
                "empty_weight": buffers.empty_weight,
                "empty_scale": buffers.empty_scale,
            }

        device = buffers.output.device
        top_k = buffers.top_k
        hidden_size = layer.sm70_hidden_logical_size
        if self.use_permute_with_scratch:
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
            sort_workspace = buffers.sort_workspace
            active_expert_offsets = buffers.active_expert_offsets[: total_slots + 1]
        return {
            "output": torch.empty(
                num_tokens, hidden_size, dtype=torch.float16, device=device
            ),
            "permuted_input": torch.empty(
                total_slots, hidden_size, dtype=torch.float16, device=device
            ),
            "intermediate": torch.empty(
                total_slots,
                layer.sm70_intermediate_size,
                dtype=torch.float16,
                device=device,
            ),
            "gate_up": torch.empty(
                total_slots,
                layer.sm70_w13_n_dim,
                dtype=torch.float16,
                device=device,
            ),
            "sorted_output": torch.empty(
                total_slots, hidden_size, dtype=torch.float16, device=device
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
            "sorted_weights": buffers.sorted_weights,
            "broadcast_input_indices": buffers.broadcast_input_indices,
            "compact_w13_ptrs_w": buffers.compact_w13_ptrs_w,
            "compact_w13_ptrs_s": buffers.compact_w13_ptrs_s,
            "legacy_w13_ptrs_w": buffers.legacy_w13_ptrs_w,
            "legacy_w13_ptrs_s": buffers.legacy_w13_ptrs_s,
            "legacy_w2_ptrs_w": buffers.legacy_w2_ptrs_w,
            "legacy_w2_ptrs_s": buffers.legacy_w2_ptrs_s,
            "empty_weight": buffers.empty_weight,
            "empty_scale": buffers.empty_scale,
        }
