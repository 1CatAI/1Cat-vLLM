# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Protocol

import torch


class GGUFLinearLayout(Protocol):
    """A layout transform shared by GGUF linear inputs and weights."""

    def input_to_gguf(self, x: torch.Tensor) -> torch.Tensor: ...

    def weight_to_vllm(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        head_dim: int | None = None,
    ) -> torch.Tensor: ...

    def shard_weight(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        logical_size: int,
        block_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class GGUFHeadTilingLayout:
    """Describe grouped heads stored as head-major tiles by GGML."""

    heads_per_group: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.heads_per_group <= 1:
            raise ValueError("heads_per_group must be greater than one")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")

    def input_to_gguf(self, x: torch.Tensor) -> torch.Tensor:
        num_heads, remainder = divmod(x.shape[-1], self.head_dim)
        if remainder or num_heads % self.heads_per_group:
            raise ValueError(
                "Cannot reorder linear input shape "
                f"{tuple(x.shape)} with heads_per_group={self.heads_per_group}, "
                f"head_dim={self.head_dim}"
            )
        num_groups = num_heads // self.heads_per_group
        shape = (*x.shape[:-1], num_groups, self.heads_per_group, self.head_dim)
        return x.reshape(shape).transpose(-3, -2).reshape(x.shape).contiguous()

    def weight_to_vllm(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        head_dim: int | None = None,
    ) -> torch.Tensor:
        """Restore a GGML head-tiled dimension to the vLLM head order."""
        if dim < 0:
            dim += weight.ndim
        if not 0 <= dim < weight.ndim:
            raise ValueError(f"Invalid dimension {dim} for shape {tuple(weight.shape)}")

        head_dim = self.head_dim if head_dim is None else head_dim
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        num_heads, remainder = divmod(weight.shape[dim], head_dim)
        num_groups, group_remainder = divmod(num_heads, self.heads_per_group)
        if remainder or group_remainder:
            raise ValueError(
                "Cannot restore GGML weight shape "
                f"{tuple(weight.shape)} along dim={dim} with "
                f"heads_per_group={self.heads_per_group}, head_dim={head_dim}"
            )

        shape = list(weight.shape)
        tiled_shape = (
            *shape[:dim],
            self.heads_per_group,
            num_groups,
            head_dim,
            *shape[dim + 1 :],
        )
        return (
            weight.reshape(tiled_shape)
            .transpose(dim, dim + 1)
            .reshape(shape)
            .contiguous()
        )

    def shard_weight(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        logical_size: int,
        block_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> torch.Tensor:
        """Select this rank's groups from every stored head tile.

        The weight may contain scalar values or packed GGML blocks. Sharding
        is done on a view tiled as (heads, groups, packed group span), so
        quantized weights stay quantized while being sharded.
        """
        if tp_size == 1:
            return weight

        num_groups, remainder = divmod(
            logical_size, self.head_dim * self.heads_per_group
        )
        if remainder:
            raise ValueError(
                f"Cannot shard logical input size {logical_size} with "
                f"heads_per_group={self.heads_per_group}, "
                f"head_dim={self.head_dim}"
            )
        local_groups, remainder = divmod(num_groups, tp_size)
        if remainder:
            raise ValueError(
                f"Cannot divide {num_groups} head groups across TP size {tp_size}"
            )
        if (local_groups * self.head_dim) % block_size:
            raise ValueError(
                f"TP size {tp_size} splits a stored head tile at "
                f"{local_groups * self.head_dim} logical elements, which is "
                f"not aligned to GGML block size {block_size}"
            )

        total_groups = self.heads_per_group * num_groups
        group_span, remainder = divmod(weight.shape[dim], total_groups)
        if remainder:
            raise ValueError(
                f"Packed weight dimension {weight.shape[dim]} is not "
                f"divisible into {total_groups} head groups"
            )

        moved = weight.movedim(dim, 0)
        tiled = moved.reshape(
            self.heads_per_group, num_groups, group_span, *moved.shape[1:]
        )
        shard = tiled[:, tp_rank * local_groups : (tp_rank + 1) * local_groups]
        return shard.reshape(-1, *moved.shape[1:]).movedim(0, dim).contiguous()
