# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, graph-visible routing records for untimed decode cost analysis."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

CAPACITY, MAX_ROWS, TOP_K = 256, 20, 10
_RECORDERS: dict[str, RouteRecorder] = {}


@triton.jit
def _record_routes(
    Ids,
    History,
    Rows,
    Counter,
    M: tl.constexpr,
    BLOCK: tl.constexpr,
    CAP: tl.constexpr,
    MAX_M: tl.constexpr,
    K: tl.constexpr,
):
    step = tl.load(Counter)
    slot = step % CAP
    offsets = tl.arange(0, BLOCK)
    ids = tl.load(Ids + offsets, offsets < M * K, other=-1)
    tl.store(
        History + slot * MAX_M * K + offsets,
        ids,
        offsets < MAX_M * K,
    )
    tl.store(Rows + slot, M)
    tl.store(Counter, step + 1)


class RouteRecorder:
    def __init__(self, name: str, device="cuda"):
        self.name = name
        self.ids = torch.full(
            (CAPACITY, MAX_ROWS, TOP_K), -1, dtype=torch.int32, device=device
        )
        self.rows = torch.zeros(CAPACITY, dtype=torch.int32, device=device)
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)

    def capture(self, ids):
        rows = ids.shape[0]
        if rows > MAX_ROWS or rows == 0:
            return
        if ids.shape[1] != TOP_K or not ids.is_contiguous():
            raise ValueError("Cost recorder requires contiguous top-10 routes")
        _record_routes[(1,)](
            ids,
            self.ids,
            self.rows,
            self.counter,
            M=rows,
            BLOCK=256,
            CAP=CAPACITY,
            MAX_M=MAX_ROWS,
            K=TOP_K,
            num_warps=1,
        )


def ordered_route_records(ids, rows, total: int):
    """Recover a bounded ring in execution order, explicitly reporting loss."""
    capacity = len(rows)
    if total < 0 or capacity == 0 or len(ids) != capacity:
        raise ValueError("Invalid routing ring")
    first = max(0, total - capacity)
    records = []
    for ordinal in range(first, total):
        slot = ordinal % capacity
        count = int(rows[slot])
        if not 0 < count <= MAX_ROWS:
            raise ValueError("Routing ring contains an unwritten record")
        values = ids[slot][:count].tolist()
        records.append(
            {
                "ordinal": ordinal,
                "rows": count,
                "ids": values,
                "unique_experts": len({expert for row in values for expert in row}),
            }
        )
    return {"total": total, "dropped": first, "records": records}


def attach_route_recorder(router, name, device="cuda"):
    if name in _RECORDERS or router.capture_fn is not None:
        raise ValueError(f"Routing capture already owned: {name}")
    recorder = RouteRecorder(name, device)
    _RECORDERS[name] = recorder
    router.set_capture_fn(recorder.capture)


def reset_routes():
    if not _RECORDERS:
        raise RuntimeError("No routing recorders were attached before graph capture")
    for recorder in _RECORDERS.values():
        recorder.counter.zero_()
    torch.accelerator.synchronize()
    return {"layers": sorted(_RECORDERS), "capacity": CAPACITY}


def read_routes():
    torch.accelerator.synchronize()
    return {
        name: ordered_route_records(
            recorder.ids.cpu(), recorder.rows.cpu(), int(recorder.counter.item())
        )
        for name, recorder in sorted(_RECORDERS.items())
    }


def tensor_inventory(model):
    """Describe loaded storage without copying weights or counting aliases twice.

    Kept source/canonical banks are both listed; the cost analyzer must select
    the actual dispatched operands rather than sum resident model storage.
    """
    records = []
    seen = set()
    tensors = [
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ]
    # HCX packed weights and several graph workspaces are ordinary tensor
    # attributes. Omitting these would describe checkpoint matrices rather
    # than all of the loaded operands selected by the native kernels.
    for prefix, module in model.named_modules():
        tensors.extend(
            (f"{prefix}.{name}" if prefix else name, value)
            for name, value in vars(module).items()
            if isinstance(value, torch.Tensor)
        )
    for name, tensor in tensors:
        if tensor.device.type == "meta":
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        records.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "logical_bytes": tensor.numel() * tensor.element_size(),
                "storage_bytes": storage.nbytes(),
                "storage_offset": tensor.storage_offset(),
                "storage_id": list(key),
                "storage_alias": key in seen,
            }
        )
        seen.add(key)
    return records
