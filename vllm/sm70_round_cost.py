# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, graph-visible routing records for untimed decode cost analysis."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

CAPACITY, MAX_ROWS, TOP_K = 256, 20, 10
_RECORDERS: dict[str, RouteRecorder] = {}
_SELECTIONS: dict[str, SelectionRecorder] = {}
SELECTION_CAPACITY = 64


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
    Requests=None,
    RequestHistory=None,
    Positions=None,
    PositionHistory=None,
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
    if RequestHistory is not None:
        requests = tl.load(Requests + offsets, offsets < M, other=-1)
        positions = tl.load(Positions + offsets, offsets < M, other=-1)
        tl.store(RequestHistory + slot * MAX_M + offsets, requests, offsets < MAX_M)
        tl.store(PositionHistory + slot * MAX_M + offsets, positions, offsets < MAX_M)
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


@triton.jit
def _record_cache_totals(
    Stats, History, Counter, N: tl.constexpr, BLOCK: tl.constexpr, CAP: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(Stats + offsets, offsets < N, other=0)
    slot = (tl.load(Counter) - 1) % CAP
    for column in tl.static_range(3):
        total = tl.sum(tl.where(offsets % 3 == column, values, 0), 0)
        tl.store(History + slot * 3 + column, total)


class SelectionRecorder:
    def __init__(self, name, width, device, owner=None):
        self.name, self.width, self.owner = name, width, owner
        self.ids = torch.full(
            (SELECTION_CAPACITY, MAX_ROWS, width), -1, dtype=torch.int32, device=device
        )
        self.rows = torch.zeros(SELECTION_CAPACITY, dtype=torch.int32, device=device)
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self.requests = torch.full(
            (SELECTION_CAPACITY, MAX_ROWS), -1, dtype=torch.int32, device=device
        )
        self.positions = torch.full_like(self.requests, -1)
        self.cache_totals = torch.zeros(
            SELECTION_CAPACITY, 3, dtype=torch.int64, device=device
        )

    def capture(self, ids, requests, positions):
        rows = ids.shape[0]
        if not 0 < rows <= MAX_ROWS:
            return
        if ids.shape[1] != self.width or not ids.is_contiguous():
            raise ValueError(
                "Selection recorder requires contiguous fixed-width indices"
            )
        _record_routes[(1,)](
            ids,
            self.ids,
            self.rows,
            self.counter,
            M=rows,
            BLOCK=triton.next_power_of_2(MAX_ROWS * self.width),
            CAP=SELECTION_CAPACITY,
            MAX_M=MAX_ROWS,
            K=self.width,
            Requests=requests,
            RequestHistory=self.requests,
            Positions=positions,
            PositionHistory=self.positions,
            num_warps=4,
        )

    def capture_cache(self, stats, rows):
        if 0 < rows <= MAX_ROWS:
            _record_cache_totals[(1,)](
                stats,
                self.cache_totals,
                self.counter,
                N=stats.numel(),
                BLOCK=triton.next_power_of_2(stats.numel()),
                CAP=SELECTION_CAPACITY,
                num_warps=4,
            )


def ordered_selection_records(ids, rows, total, cache_totals, requests, positions):
    """Return logical selected-byte counts, not a sum of cache allocations.

    Across-row unions are valid for one request only. Multi-request records
    retain per-row counts and explicitly omit a cross-request union.
    Cache totals are cumulative; a truncated prefix has an unknown first delta.
    """
    capacity = len(rows)
    if total < 0 or len(ids) != capacity or len(cache_totals) != capacity:
        raise ValueError("Invalid selection ring")
    first = max(0, total - capacity)
    previous = None if first else torch.zeros(3, dtype=torch.int64)
    records = []
    for ordinal in range(first, total):
        slot = ordinal % capacity
        count = int(rows[slot])
        if not 0 < count <= MAX_ROWS:
            raise ValueError("Selection ring contains an unwritten record")
        values = ids[slot, :count]
        reqs = requests[slot, :count]
        pos = positions[slot, :count]
        valid = (values >= 0) & (values <= pos[:, None]) & (reqs[:, None] >= 0)
        single = bool((reqs >= 0).all()) and int(reqs.unique().numel()) == 1
        cumulative = cache_totals[slot]
        delta = None if previous is None else (cumulative - previous).tolist()
        if delta is not None and any(value < 0 for value in delta):
            raise ValueError("Cache counters must increase monotonically")
        records.append(
            {
                "ordinal": ordinal,
                "rows": count,
                "valid_per_row": valid.sum(1).tolist(),
                "unique_per_row": [
                    int(row[mask].unique().numel()) for row, mask in zip(values, valid)
                ],
                "logical_union_single_request": int(values[valid].unique().numel())
                if single
                else None,
                "cache_hit_miss_contention_delta": delta,
            }
        )
        previous = cumulative
    return {
        "total": total,
        "dropped": first,
        "width": ids.shape[-1],
        "records": records,
        "union_contract": "Logical IDs with captured causal positions and requests; "
        "union only for single-request records. Not measured HBM bytes.",
    }


def attach_selection_recorder(name, width, device, owner):
    if name in _SELECTIONS:
        raise ValueError(f"Selection capture already owned: {name}")
    recorder = SelectionRecorder(name, width, device, owner)
    _SELECTIONS[name] = recorder
    return recorder


def read_selections():
    torch.accelerator.synchronize()
    return {
        name: ordered_selection_records(
            recorder.ids.cpu(),
            recorder.rows.cpu(),
            int(recorder.counter.item()),
            recorder.cache_totals.cpu(),
            recorder.requests.cpu(),
            recorder.positions.cpu(),
        )
        | {
            "cache_counters_available": getattr(recorder.owner, "host_kv", None)
            is not None
        }
        for name, recorder in sorted(_SELECTIONS.items())
    }


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
    for recorder in _SELECTIONS.values():
        recorder.counter.zero_()
        state = getattr(recorder.owner, "host_kv", None)
        if state is not None:
            state._stats.zero_()
    torch.accelerator.synchronize()
    return {
        "layers": sorted(_RECORDERS),
        "capacity": CAPACITY,
        "selection_layers": sorted(_SELECTIONS),
        "selection_capacity": SELECTION_CAPACITY,
    }


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
