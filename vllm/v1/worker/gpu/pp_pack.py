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
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

logger = init_logger(__name__)

# Segment starts are aligned so every view is valid for vectorized kernel loads
# of any dtype and matches the allocator's own block alignment.
SEGMENT_ALIGNMENT_BYTES = 256

# Name under which the flat byte block crosses the group's static transfer.
PACKED_KEY = "packed"


def _is_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


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

    @property
    def row_nbytes(self) -> int:
        return sum(segment.row_bytes for segment in self.segments)

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

    @property
    def captured_rows(self) -> tuple[int, ...]:
        return tuple(sorted(self._blocks))

    def block(self, rows: int) -> PackedBlock:
        block = self._blocks.get(rows)
        if block is None:
            # Inside a graph capture the zero fill would become a graph node
            # and the block would live in the graph pool. Every graph's eager
            # warmup runs before its capture and allocates the block there.
            assert not _is_capturing(self.device), (
                f"packed handover block for {rows} rows first used while "
                "capturing a CUDA graph"
            )
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


def _describe(buffers: PackedHandoverBuffers | None) -> dict[str, Any] | None:
    if buffers is None or not buffers.captured_rows:
        return None
    layout = buffers.layout
    return {
        "signature": layout.signature,
        "row_nbytes": layout.row_nbytes,
        "nbytes": {rows: layout.nbytes(rows) for rows in buffers.captured_rows},
    }


def verify_handover_layouts(
    group: "GroupCoordinator",
    recv: PackedHandoverBuffers | None,
    send: PackedHandoverBuffers | None,
) -> None:
    """Fail loudly at startup when two neighbours disagree on the packed layout.

    A different dtype, shape or graph size makes the two ends of the one NCCL
    message differ in size, which hangs instead of failing. Every rank first
    receives its predecessor's send layout, then sends its own, so the chain
    over the CPU group cannot deadlock. Call after the graphs are captured.
    """
    mine_recv, mine_send = _describe(recv), _describe(send)
    logger.info(
        "PP packed handover layout rank %d: recv %s, send %s",
        group.rank_in_group,
        mine_recv,
        mine_send,
    )
    if group.rank_in_group > 0:
        theirs = group.recv_object(src=group.rank_in_group - 1)
        if theirs != mine_recv:
            raise RuntimeError(
                "PP packed handover layout mismatch: stage "
                f"{group.rank_in_group - 1} sends {theirs}, stage "
                f"{group.rank_in_group} receives {mine_recv}"
            )
    if group.rank_in_group < group.world_size - 1:
        group.send_object(mine_send, dst=group.rank_in_group + 1)
