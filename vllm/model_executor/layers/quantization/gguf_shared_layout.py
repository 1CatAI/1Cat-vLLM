# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-preserving views between a shared original expert and its TP banks.

The shared host representation contains each original block once. Device
banks retain complete boundary blocks, so the existing readers can keep their
logical K offset. During eviction only one rank writes each original block.
"""

import math
from dataclasses import dataclass
from typing import Literal

import gguf
import numpy as np

import vllm.envs as envs

# Row-split down: each TP rank owns hidden/tp output rows with the full K, so
# no original block is duplicated across ranks (no K boundary blocks).
DOWN_ROWS = envs.VLLM_V41_DOWN_ROWS == "1"


@dataclass(frozen=True)
class SharedExpertBudget:
    """Capacity measured in original cold experts, independently of GPU spares.

    ``expert_bytes`` contains one original expert's three projections for each
    layer. Exchange space is group-wide and must cover all outstanding M6
    routes plus spare operations at the largest layer, without reusing inputs.
    """

    num_experts: int
    offload_fraction: float
    expert_bytes: tuple[int, ...]
    spare_slots: int = 8
    max_routes: int = 36
    exchange_bytes: int = 512 * 1024**2

    def __post_init__(self):
        if (
            self.num_experts <= 1
            or not math.isfinite(self.offload_fraction)
            or not 0 < self.offload_fraction < 1
            or not self.expert_bytes
            or min(self.expert_bytes) <= 0
            or self.spare_slots <= 0
            or self.max_routes <= 0
        ):
            raise ValueError("Invalid shared expert cache dimensions")
        if not 0 < self.cold_experts < self.num_experts:
            raise ValueError("Offload fraction rounds to an empty cold or hot set")
        if self.gpu_slots > self.num_experts:
            raise ValueError("Offload fraction cannot accommodate the spare slots")
        if self.max_routes > self.hot_experts:
            raise ValueError("Hot capacity cannot hold the maximum routed set")
        if self.exchange_bytes < self.required_exchange_bytes:
            raise ValueError("Shared exchange reserve cannot hold the maximum plan")

    @property
    def cold_experts(self) -> int:
        return round(self.offload_fraction * self.num_experts)

    @property
    def hot_experts(self) -> int:
        return self.num_experts - self.cold_experts

    @property
    def gpu_slots(self) -> int:
        return self.hot_experts + self.spare_slots

    @property
    def cold_bytes(self) -> int:
        return self.cold_experts * sum(self.expert_bytes)

    @property
    def required_exchange_bytes(self) -> int:
        return (self.max_routes + self.spare_slots) * max(self.expert_bytes)

    @property
    def arena_bytes(self) -> int:
        return self.cold_bytes + self.exchange_bytes


@dataclass(frozen=True)
class SharedExpertLayout:
    hidden: int
    intermediate: int
    tp_size: int
    tp_rank: int
    role: Literal["w1", "w3", "w2"]
    block_size: int
    block_bytes: int

    def __post_init__(self):
        if self.role not in ("w1", "w3", "w2"):
            raise ValueError("Unknown expert projection")
        if (
            min(
                self.hidden,
                self.intermediate,
                self.tp_size,
                self.block_size,
                self.block_bytes,
            )
            <= 0
            or not 0 <= self.tp_rank < self.tp_size
        ):
            raise ValueError("Invalid expert or TP dimensions")
        if (
            self.intermediate % self.tp_size
            or self.hidden % self.tp_size
            or self.source_k % self.block_size
        ):
            raise ValueError("Expert geometry does not cover complete source blocks")
        if (-self.physical_k) % 512 % self.block_size:
            raise ValueError("Device bank guard is not block aligned")

    @classmethod
    def for_type(cls, hidden, intermediate, tp_size, tp_rank, role, weight_type):
        block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(weight_type)]
        return cls(hidden, intermediate, tp_size, tp_rank, role, block, size)

    @property
    def source_k(self) -> int:
        return self.intermediate if self.role == "w2" else self.hidden

    @property
    def source_rows(self) -> int:
        return self.hidden if self.role == "w2" else self.intermediate

    @property
    def local_intermediate(self) -> int:
        return self.intermediate // self.tp_size

    @property
    def first_block(self) -> int:
        if self.role != "w2" or DOWN_ROWS:
            return 0
        return self.tp_rank * self.local_intermediate // self.block_size

    @property
    def end_block(self) -> int:
        if self.role != "w2" or DOWN_ROWS:
            return self.source_k // self.block_size
        end = (self.tp_rank + 1) * self.local_intermediate
        return (end + self.block_size - 1) // self.block_size

    @property
    def physical_k(self) -> int:
        return (self.end_block - self.first_block) * self.block_size

    @property
    def logical_k_offset(self) -> int:
        if self.role != "w2" or DOWN_ROWS:
            return 0
        return self.tp_rank * self.local_intermediate % self.block_size

    @property
    def source_row_bytes(self) -> int:
        return self.source_k // self.block_size * self.block_bytes

    @property
    def device_row_bytes(self) -> int:
        return (self.end_block - self.first_block) * self.block_bytes

    @property
    def device_rows(self) -> int:
        if self.role == "w2":
            return self.hidden // self.tp_size if DOWN_ROWS else self.hidden
        return self.local_intermediate

    @property
    def guard_bytes(self) -> int:
        """One tail for the whole bank, not one tail per expert."""
        return (-self.physical_k) % 512 // self.block_size * self.block_bytes

    @property
    def write_end_block(self) -> int:
        if self.role != "w2" or DOWN_ROWS or self.tp_rank == self.tp_size - 1:
            return self.end_block
        return (self.tp_rank + 1) * self.local_intermediate // self.block_size

    @property
    def source_row_start(self) -> int:
        if self.role == "w2":
            return self.tp_rank * self.device_rows if DOWN_ROWS else 0
        return self.tp_rank * self.local_intermediate

    def pack(self, original: np.ndarray) -> np.ndarray:
        """CPU reference for gathering the bytes a device bank needs."""
        self._check(original, (self.source_rows, self.source_row_bytes))
        start = self.first_block * self.block_bytes
        rows = slice(self.source_row_start, self.source_row_start + self.device_rows)
        return original[rows, start : start + self.device_row_bytes].copy()

    def writeback(self, original: np.ndarray, device: np.ndarray) -> None:
        """CPU reference for this rank's exclusive canonical write ownership."""
        self._check(original, (self.source_rows, self.source_row_bytes))
        self._check(device, (self.device_rows, self.device_row_bytes))
        width = (self.write_end_block - self.first_block) * self.block_bytes
        rows = slice(self.source_row_start, self.source_row_start + self.device_rows)
        start = self.first_block * self.block_bytes
        original[rows, start : start + width] = device[:, :width]

    @staticmethod
    def _check(values: np.ndarray, shape: tuple[int, int]) -> None:
        if values.dtype != np.uint8 or values.shape != shape:
            raise ValueError(f"Expected original uint8 byte matrix with shape {shape}")
