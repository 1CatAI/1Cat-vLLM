# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-message pipeline handover for fixed-shape decode and verify steps.

The intermediate-tensor dict of a stage boundary is sent as one NCCL message
instead of one message per tensor. Both sides derive the same byte layout from
the tensors' names, dtypes and per-row shapes. The sender's FULL CUDA graph
writes its outputs straight into the packed block of its row count, and the
receiver gets the block into the buffers the graph for that row count was
captured on, so neither side runs a pack or unpack copy.

A step takes this path exactly when :func:`packs_handover` accepts its batch
descriptor. Sender and receiver dispatch the same scheduler output, so they
agree; every other step keeps the metadata-carrying per-tensor transfer.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

# Segment starts are aligned so every view is valid for vectorized kernel loads
# of any dtype and matches the allocator's own block alignment.
SEGMENT_ALIGNMENT_BYTES = 256

# Name under which the flat byte block crosses the group's static transfer.
PACKED_KEY = "packed"


def pp_pack_enabled(vllm_config: VllmConfig) -> bool:
    """Admission for the packed handover.

    The packed block is sent whole, so tensor-parallel slicing of the
    transfer (``all_gather_group``) is not modelled; data-parallel padding can
    change a stage's descriptor independently of its neighbour.
    """
    parallel = vllm_config.parallel_config
    return (
        vllm_config.kernel_config.pp_pack_handover
        and parallel.pipeline_parallel_size > 1
        and parallel.tensor_parallel_size == 1
        and parallel.data_parallel_size == 1
    )


def packs_handover(desc: "BatchExecutionDescriptor") -> bool:
    """Whether a step's shape is fixed by its FULL decode graph."""
    return desc.cg_mode == CUDAGraphMode.FULL and desc.uniform_token_count is not None


@dataclass(frozen=True)
class PackedSegment:
    name: str
    dtype: torch.dtype
    row_shape: tuple[int, ...]

    @property
    def row_bytes(self) -> int:
        return math.prod(self.row_shape) * self.dtype.itemsize


def _align(offset: int) -> int:
    return -(-offset // SEGMENT_ALIGNMENT_BYTES) * SEGMENT_ALIGNMENT_BYTES


class PackedHandoverLayout:
    """Byte layout of one stage boundary, a function of names, dtypes and shapes."""

    def __init__(self, segments: tuple[PackedSegment, ...]):
        self.segments = segments

    @classmethod
    def from_tensors(
        cls, tensors: Mapping[str, torch.Tensor]
    ) -> "PackedHandoverLayout":
        # Sorted, so sender and receiver agree whatever their dict order is.
        return cls(
            tuple(
                PackedSegment(name, tensor.dtype, tuple(tensor.shape[1:]))
                for name, tensor in sorted(tensors.items())
            )
        )

    @property
    def signature(self) -> tuple[tuple[str, torch.dtype, tuple[int, ...]], ...]:
        return tuple((s.name, s.dtype, s.row_shape) for s in self.segments)

    def offsets(self, rows: int) -> list[int]:
        offsets: list[int] = []
        end = 0
        for segment in self.segments:
            start = _align(end)
            offsets.append(start)
            end = start + rows * segment.row_bytes
        return offsets

    def nbytes(self, rows: int) -> int:
        if not self.segments:
            return 0
        return _align(self.offsets(rows)[-1] + rows * self.segments[-1].row_bytes)


@dataclass(frozen=True)
class PackedBlock:
    """One contiguous byte buffer and the per-tensor views into it."""

    flat: torch.Tensor
    tensors: dict[str, torch.Tensor]


class PackedHandoverBuffers:
    """Persistent packed blocks, one per row count (graph size)."""

    def __init__(self, layout: PackedHandoverLayout, device: torch.device):
        self.layout = layout
        self.device = device
        self._blocks: dict[int, PackedBlock] = {}

    def block(self, rows: int) -> PackedBlock:
        block = self._blocks.get(rows)
        if block is None:
            block = self._allocate(rows)
            self._blocks[rows] = block
        return block

    def _allocate(self, rows: int) -> PackedBlock:
        flat = torch.zeros(
            self.layout.nbytes(rows), dtype=torch.uint8, device=self.device
        )
        tensors = {}
        for segment, start in zip(self.layout.segments, self.layout.offsets(rows)):
            raw = flat[start : start + rows * segment.row_bytes]
            tensors[segment.name] = raw.view(segment.dtype).view(
                rows, *segment.row_shape
            )
        return PackedBlock(flat, tensors)


class PackedIntermediateTensors(IntermediateTensors):
    """Stage output whose tensors are views of one contiguous send block.

    Not sliceable: it is only built by the CUDA graph manager and consumed by
    the worker's send.
    """

    def __init__(self, block: PackedBlock):
        super().__init__(block.tensors)
        self.packed = block.flat


def handover_inputs(
    desc: "BatchExecutionDescriptor",
    standard: IntermediateTensors,
    buffers: PackedHandoverBuffers | None,
) -> dict[str, torch.Tensor]:
    """The tensors the graph for ``desc`` reads its stage inputs from.

    SSOT for capture and replay: the packed receive block when the step takes
    the packed path, the persistent standard buffers otherwise.
    """
    if buffers is not None and packs_handover(desc):
        return buffers.block(desc.num_tokens).tensors
    return {key: value[: desc.num_tokens] for key, value in standard.tensors.items()}
