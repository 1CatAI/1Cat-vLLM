# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contiguous allocation of original expert banks, preserving TP block guards."""

import math

import torch

from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
from vllm.model_executor.layers.quantization.gguf_native import (
    NATIVE_TYPES,
    packed_tp_span,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_size


def original_expert_bank(
    shape, weight_type, shard, hidden, intermediate, tp_size, tp_rank
):
    """Return the local byte shape and the upstream final-row safety tail."""
    k, rows, experts = map(int, shape)
    block, size = quant_size(weight_type)
    if shard in ("w1", "w3") and k == hidden and rows == intermediate * tp_size:
        physical_k = k
        local_rows = intermediate
        packed_k = k // block * size
    elif shard == "w2" and k == intermediate * tp_size and rows == hidden:
        first, last, _, physical_k = packed_tp_span(k, weight_type, tp_size, tp_rank)
        local_rows = hidden
        packed_k = last - first
    else:
        raise ValueError(f"Original GGUF expert shape {(k, rows, experts)} is invalid")
    if k % block or (-physical_k) % 512 % block:
        raise ValueError("Original GGUF expert bank is not quantization-block aligned")
    tail = ((-physical_k) % 512) // block * size
    return (experts, local_rows, packed_k), tail


def prepare_original_expert_arena(model, name_map, tensors, device):
    """Reserve immutable expert banks before replaceable dense loading storage."""
    shards = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    banks = []
    total = 0
    for raw, name in name_map.items():
        if ".experts." not in name or not name.endswith(".weight"):
            continue
        prefix, projection, _ = name.rsplit(".", 2)
        if projection not in shards:
            continue
        layer = model.get_submodule(prefix)
        method = layer.quant_method
        tensor = tensors[raw]
        value = int(tensor.tensor_type)
        if not isinstance(method, GGUFNativeMoEMethod) or value not in NATIVE_TYPES:
            continue
        shape, tail = original_expert_bank(
            tensor.shape,
            value,
            shards[projection],
            method.hidden_size,
            method.intermediate_size,
            layer.tp_size,
            layer.tp_rank,
        )
        if shape[0] != method.num_experts:
            raise ValueError("GGUF expert count does not match the model")
        total = (total + 255) // 256 * 256
        elements = math.prod(shape)
        banks.append((layer, method, shards[projection], shape, total, elements, tail))
        total += elements + tail
    if not banks:
        return
    arena = torch.empty(total, dtype=torch.uint8, device=device)
    for layer, method, shard, shape, offset, elements, tail in banks:
        arena[offset + elements : offset + elements + tail].zero_()
        layer.register_buffer(
            "gguf_" + shard,
            arena[offset : offset + elements].view(shape),
            persistent=False,
        )
        method.guarded_shards.add(shard)
