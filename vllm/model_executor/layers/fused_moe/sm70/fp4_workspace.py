# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 workspace lifetimes. Layout differences remain explicit.

Buffers stay on their original layer attributes; no tensor copying, persistent
address change, or alternate pointer registry is introduced by these owners.
"""

from typing import Final

import torch

from vllm.config.vllm import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe import RoutedExperts


class NvFp4MoEWorkspace:
    @classmethod
    def allocate(cls, layer: RoutedExperts) -> None:
        device = layer.w13_tm_weight.device
        top_k = int(layer.sm70_nvfp4_top_k)
        max_slots = 18 * top_k
        experts = int(layer.sm70_nvfp4_num_experts)
        hidden = int(layer.sm70_nvfp4_hidden_size)
        intermediate = int(layer.sm70_nvfp4_intermediate_size)

        layer._nvfp4_sm70_output = torch.empty(
            18, hidden, dtype=torch.float16, device=device
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
            18,
            top_k,
            dtype=torch.int32,
            device=device,
        )
        layer._nvfp4_sm70_topk_ids = torch.empty(
            18,
            top_k,
            dtype=torch.int32,
            device=device,
        )
        layer._nvfp4_sm70_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(18, top_k)
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

    @classmethod
    def get(
        cls, layer: RoutedExperts, num_tokens: int, indexed_w13: bool
    ) -> dict[str, torch.Tensor]:
        if 0 < num_tokens <= 18:
            return cls._persistent_buffers(layer, num_tokens)
        return cls._eager_buffers(layer, num_tokens, indexed_w13)


class MxFp4MoEWorkspace:
    @classmethod
    def allocate(cls, layer: RoutedExperts) -> None:
        device = layer.w13_tm_weight.device
        top_k = 6
        max_slots = 8 * top_k
        num_experts = int(layer.sm70_mxfp4_num_experts)
        hidden_size = int(layer.sm70_mxfp4_hidden_size)
        intermediate_size = int(layer.sm70_mxfp4_intermediate_size)

        layer._mxfp4_sm70_buf_output = torch.empty(
            8,
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
            8, top_k, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_topk_ids = torch.empty(
            8, top_k, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(8, top_k)
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
            max_slots + 1, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_compact_expert_offsets64 = torch.arange(
            max_slots + 1, dtype=torch.int64, device=device
        )
        layer._mxfp4_sm70_buf_slot_expert_offsets = torch.arange(
            max_slots + 1, dtype=torch.int32, device=device
        )
        layer._mxfp4_sm70_buf_active_expert_ids = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )

    @staticmethod
    def _persistent_decode_buffers(
        layer: RoutedExperts, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        total_slots = num_tokens * 6
        return {
            "output": layer._mxfp4_sm70_buf_output[:num_tokens],
            "permuted_input": layer._mxfp4_sm70_buf_permuted_input[:total_slots],
            "gate_up": layer._mxfp4_sm70_buf_gate_up[:total_slots],
            "intermediate": layer._mxfp4_sm70_buf_intermediate[:total_slots],
            "sorted_output": layer._mxfp4_sm70_buf_sorted_output[:total_slots],
            "expert_offsets": layer._mxfp4_sm70_buf_expert_offsets,
            "expert_offsets64": layer._mxfp4_sm70_buf_expert_offsets64,
            "inv_permuted_idx": layer._mxfp4_sm70_buf_inv_permuted_idx[:num_tokens],
            "topk_ids": layer._mxfp4_sm70_buf_topk_ids[:num_tokens],
            "token_expert_indices": (
                layer._mxfp4_sm70_buf_token_expert_indices[:num_tokens]
            ),
            "permuted_idx": layer._mxfp4_sm70_buf_permuted_idx[:total_slots],
            "sort_workspace": layer._mxfp4_sm70_buf_sort_workspace,
            "permuted_experts_id": (
                layer._mxfp4_sm70_buf_permuted_experts_id[:total_slots]
            ),
            "sorted_row_idx": layer._mxfp4_sm70_buf_sorted_row_idx[:total_slots],
            "topk_ids_for_sort": (
                layer._mxfp4_sm70_buf_topk_ids_for_sort[:total_slots]
            ),
            "dense_expert_ids": layer._mxfp4_sm70_buf_dense_expert_ids,
            "compact_expert_offsets": (
                layer._mxfp4_sm70_buf_compact_expert_offsets[: total_slots + 1]
            ),
            "compact_expert_offsets64": (
                layer._mxfp4_sm70_buf_compact_expert_offsets64[: total_slots + 1]
            ),
            "slot_expert_offsets": (
                layer._mxfp4_sm70_buf_slot_expert_offsets[: total_slots + 1]
            ),
            "active_expert_ids": (
                layer._mxfp4_sm70_buf_active_expert_ids[:total_slots]
            ),
        }

    @staticmethod
    def _eager_buffers(
        layer: RoutedExperts, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        device = layer.w13_tm_weight.device
        top_k = 6
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
            "slot_expert_offsets": torch.arange(
                total_slots + 1, dtype=torch.int32, device=device
            ),
            "active_expert_ids": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
        }

    @classmethod
    def get(cls, layer: RoutedExperts, num_tokens: int) -> dict[str, torch.Tensor]:
        if 0 < num_tokens <= 8:
            return cls._persistent_decode_buffers(layer, num_tokens)
        return cls._eager_buffers(layer, num_tokens)


_QWEN38_RAW_SCALE_WORKSPACE_ELEMENTS: Final = 512 * 160 * 320
_qwen38_raw_scale_workspaces: dict[int, torch.Tensor] = {}


def clear_sm70_nvfp4_moe_workspaces() -> None:
    """Release process-global Qwen3.8 raw-scale expansion workspaces."""
    _qwen38_raw_scale_workspaces.clear()


def _get_qwen38_raw_scale_workspace(device: torch.device) -> torch.Tensor:
    # The persistent views below share one expansion buffer across layers.
    # Concurrent microbatches could overwrite it before a GEMM consumes it.
    # Reject at load time, without adding synchronization to decode.
    config = get_current_vllm_config_or_none()
    if config is not None and config.parallel_config.use_ubatching:
        raise NotImplementedError(
            "SM70 raw-scale storage uses a shared expansion workspace and "
            "cannot be combined with DBO or microbatching. Disable "
            "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE to use prepared scales."
        )
    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    workspace = _qwen38_raw_scale_workspaces.get(device_index)
    if workspace is None:
        workspace = torch.empty(
            _QWEN38_RAW_SCALE_WORKSPACE_ELEMENTS,
            dtype=torch.float16,
            device=device,
        )
        _qwen38_raw_scale_workspaces[device_index] = workspace
    return workspace
