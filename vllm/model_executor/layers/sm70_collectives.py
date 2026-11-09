# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, calibrated FP32 local-row reduction for SM70 TP4 callers.

This experimental interface has no automatic dispatch. Prepare collectively
outside CUDA graphs, use one bound stream, consume each returned view before
calling again, and close collectively before destroying the process group.
Calibration depends on the communicator and shape, never on model values.
"""

import sys
from functools import lru_cache
from importlib import import_module
from operator import index
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _layout(shape):
    if len(shape) != 2 or any(isinstance(value, bool) for value in shape):
        raise ValueError("Exact row reduction requires two integer dimensions")
    rows, columns = map(index, shape)
    if rows <= 0 or columns <= 0 or rows % 4:
        raise ValueError("Exact row reduction requires positive TP4-aligned rows")
    count = rows * columns
    if count > (2**63 - 1) // 4:
        raise ValueError("Exact row reduction allocation size overflows int64")
    local = count // 4
    raw = count * 4 + 2 * 80 * 4 * 4
    resident = raw + local * 5
    calibration_peak = resident + count * 8 + local * 2 + 256
    return (rows, columns), raw, resident, calibration_peak


@lru_cache(maxsize=1)
def _extension():
    try:
        return import_module("vllm._sm70_exact_reduce_C")
    except ImportError:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parents[3]
        source = root / "csrc/sm70_turbomind/ops/exact_row_reduce.cu"
        if not source.is_file():
            raise RuntimeError(
                "SM70 exact row reduction requires the source build"
            ) from None
        extension = load(
            name="onecat_sm70_exact_reduce",
            sources=[str(source)],
            extra_cuda_cflags=[
                "-O3",
                "--fmad=false",
                "-gencode=arch=compute_70,code=sm_70",
            ],
            verbose=False,
        )
        sys.modules["onecat_sm70_exact_reduce"] = extension
        return extension


class SM70ExactRowReductionPlan:
    """Own one shape's IPC buffers and calibrated native FP32 addition order.

    ``group`` is a vLLM TP group with four ranks and a CPU process group.
    ``memory_budget_bytes`` must cover persistent buffers and calibration
    scratch. Callers retain their ordinary collective when this explicit plan
    is unsuitable. Returned tensors alias plan storage. CUDA graphs and use
    from another stream/device are rejected rather than silently changing
    synchronization semantics.
    """

    @staticmethod
    def required_memory_bytes(shape):
        """Conservative explicit budget for setup scratch and persistent buffers."""
        return _layout(shape)[3]

    def __init__(self, group, shape, *, memory_budget_bytes):
        error = None
        try:
            self.shape, self.raw_ipc_bytes, self.resident_bytes, peak = _layout(shape)
            budget = index(memory_budget_bytes)
            if isinstance(memory_budget_bytes, bool) or budget < peak:
                error = "Exact row reduction exceeds the explicit memory budget"
        except (TypeError, ValueError, OverflowError) as exc:
            self.shape, peak = None, 0
            error = str(exc)
        self.calibration_peak_bytes = peak
        self.group = group
        self.rank = group.rank_in_group
        self.device = torch.accelerator.current_device_index()
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        self._closed = False
        self._buffers = []
        self._epoch = 0
        self.calls = 0
        properties = torch.cuda.get_device_properties(self.device)
        if group.world_size != 4 or not 0 <= self.rank < 4:
            error = "Exact row reduction requires TP4"
        elif (properties.major, properties.minor) != (7, 0):
            error = "Exact row reduction requires SM70"
        elif properties.multi_processor_count < 80:
            error = "Exact row reduction requires at least 80 SMs"
        elif torch.cuda.is_current_stream_capturing():
            error = "Prepare exact row reduction outside CUDA graphs"
        metadata: list[Any] = [None] * group.world_size
        with torch.inference_mode(False):
            dist.all_gather_object(
                metadata,
                (self.shape, self.device, error, str(properties.uuid)),
                group=group.cpu_group,
            )
        errors = [item[2] for item in metadata if item[2]]
        if errors or any(item[0] != self.shape for item in metadata):
            raise ValueError(errors or "Ranks requested different reduction shapes")
        if len({item[3] for item in metadata}) != 4 or any(
            not 0 <= item[1] < torch.accelerator.device_count()
            or str(torch.cuda.get_device_properties(item[1]).uuid) != item[3]
            for item in metadata
        ):
            error = (
                "Exact row reduction requires one host with "
                "consistent CUDA device visibility"
            )
        elif any(
            peer != self.rank
            and not torch.cuda.can_device_access_peer(self.device, item[1])
            for peer, item in enumerate(metadata)
        ):
            error = "Exact row reduction requires peer access to every TP rank"
        self._agree(error)
        self.ops = _extension()
        self.output = None
        self.codes = None
        try:
            with torch.inference_mode(False):
                count = self.shape[0] * self.shape[1]
                self.pointers = self._shared(count * 4)
                self.flags = self._shared(2 * 80 * 4 * 4)
                self.output = torch.empty(
                    (self.shape[0] // 4, self.shape[1]),
                    device=self.device,
                    dtype=torch.float32,
                )
                self.codes = self._calibrate(count)
            dist.barrier(group=group.cpu_group)
        except BaseException:
            self.close()
            raise

    def _agree(self, error):
        errors = [None] * self.group.world_size
        with torch.inference_mode(False):
            dist.all_gather_object(errors, error, group=self.group.cpu_group)
        if any(errors):
            raise RuntimeError(f"Exact row reduction setup failed: {errors}")

    def _shared(self, size):
        pointer, handle, error = 0, None, None
        try:
            pointer, handle = self.ops.allocate(size)
            torch.accelerator.synchronize()
        except RuntimeError as exc:
            error = str(exc)
        handles: list[Any] = [None] * 4
        dist.all_gather_object(handles, (handle, error), group=self.group.cpu_group)
        if any(item[1] for item in handles):
            if pointer:
                self.ops.release(pointer, True)
            raise RuntimeError(f"Exact row reduction IPC allocation failed: {handles}")
        pointers = [0] * 4
        pointers[self.rank] = pointer
        self._buffers.append(pointers)
        for peer, (handle, _) in enumerate(handles):
            if peer != self.rank:
                try:
                    pointers[peer] = self.ops.open_handle(handle)
                except RuntimeError as exc:
                    error = str(exc)
                    break
        self._agree(error)
        return pointers

    def _calibrate(self, count):
        from vllm.triton_utils import triton

        from .sm70_collective_calibration import PROBES, decode_mask, update_mask

        n = count // 4
        masks = torch.full((n,), 0x7FFF, device=self.device, dtype=torch.int16)
        codes = torch.empty(n, device=self.device, dtype=torch.uint8)
        stats = torch.zeros(2, device=self.device, dtype=torch.int64)
        value = torch.empty(self.shape, device=self.device, dtype=torch.float32)
        for inputs, expected_bits in PROBES:
            value.fill_(inputs[self.rank])
            reference = self.group.all_reduce(value)
            expected = torch.tensor(
                expected_bits, device=self.device, dtype=torch.uint32
            )
            update_mask[(triton.cdiv(n, 256),)](
                reference, masks, expected, n, n * self.rank, 256
            )
            del reference, expected
        decode_mask[(triton.cdiv(n, 256),)](masks, codes, stats, n, 256)
        self._agree(
            None
            if stats.tolist() == [0, 0]
            else "Native FP32 addition order cannot be classified uniquely"
        )
        return codes

    def reduce(self, value):
        """Return this rank's local rows; adapters must already be included."""
        if self._closed:
            raise RuntimeError("Exact row reduction plan is closed")
        if (
            not value.is_cuda
            or value.device.index != self.device
            or value.dtype != torch.float32
            or tuple(value.shape) != self.shape
            or not value.is_contiguous()
            or value.requires_grad
        ):
            raise ValueError("Exact row reduction requires the prepared FP32 layout")
        if (
            torch.accelerator.current_device_index() != self.device
            or torch.cuda.current_stream(self.device).cuda_stream != self.stream
            or torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "Exact row reduction requires its original uncaptured stream"
            )
        if self._epoch == 2**32 - 1:
            raise RuntimeError(
                "Exact row reduction epoch exhausted; prepare a new plan"
            )
        self._epoch += 1
        self.calls += 1
        self.ops.run(
            value,
            self.codes,
            self.output,
            self.pointers,
            self.flags,
            self.rank,
            self._epoch,
        )
        return self.output

    def close(self):
        """Collectively release peer handles before freeing their owners."""
        if self._closed:
            return
        with torch.accelerator.device_index(self.device):
            torch.accelerator.synchronize()
            dist.barrier(group=self.group.cpu_group)
            for pointers in self._buffers:
                for peer, pointer in enumerate(pointers):
                    if pointer and peer != self.rank:
                        self.ops.release(pointer, False)
            dist.barrier(group=self.group.cpu_group)
            for pointers in self._buffers:
                if pointers[self.rank]:
                    self.ops.release(pointers[self.rank], True)
            self._buffers.clear()
            self.output = self.codes = None
            self._closed = True
