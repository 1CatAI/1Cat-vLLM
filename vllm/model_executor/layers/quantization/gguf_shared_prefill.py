# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded V4.1 Tensor Core prefill over shared original GGUF experts.

There is one reusable gate/up and down canonical workspace, never a persistent
FP16 expert bank. Cold rows are read from the single mapped original pool;
hot rows come from the rank's raw GPU carriers. Preparation and GEMMs use the
model stream, so every rank finishes all cold reads before its next decode
request. The coordinator's four-request barrier is also the read fence.
"""

from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import torch

import vllm.envs as envs
from vllm.triton_utils import tl, triton

from .gguf_expert_cache import TurboMindPrefill, _canonical
from .gguf_shared_layout import DOWN_ROWS
from .gguf_shared_protocol import _fail_if
from .gguf_shared_runtime import SHARDS


@triton.jit
def _check_transcode(error):
    _fail_if(tl.load(error))


@triton.jit
def _partition_tile(
    table,
    routed,
    hot,
    cold,
    hot_count,
    cold_count,
    statistics,
    FIRST: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.arange(0, BLOCK)
    slots = tl.load(table + FIRST + local, local < COUNT, -1)
    # Experts without routes in this pass need no transcode (or staging).
    used = tl.load(routed + FIRST + local, local < COUNT, 0) != 0
    resident = (local < COUNT) & (slots >= 0) & used
    offloaded = (local < COUNT) & (slots < 0) & used
    hot_index = tl.cumsum(resident.to(tl.int32)) - 1
    cold_index = tl.cumsum(offloaded.to(tl.int32)) - 1
    tl.store(hot + hot_index, local, resident)
    tl.store(cold + cold_index, local, offloaded)
    tl.store(hot_count, tl.sum(resident.to(tl.int32)))
    tl.store(cold_count, tl.sum(offloaded.to(tl.int32)))
    tl.store(statistics, tl.load(statistics) + tl.sum(offloaded.to(tl.int64)))


@triton.jit(do_not_specialize=["ROUTES"])
def _v41_prefill_activation(
    projection,
    probabilities,
    order,
    hidden,
    bounds,
    ROUTES,
    EXPERTS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (route < ROUTES) & (col < WIDTH)
    valid &= (route >= tl.load(bounds)) & (route < tl.load(bounds + EXPERTS))
    gate = tl.load(projection + route * 2 * WIDTH + 2 * col, valid, 0).to(tl.float32)
    up = tl.load(projection + route * 2 * WIDTH + 2 * col + 1, valid, 0).to(tl.float32)
    gate = tl.minimum(gate, 10.0)
    up = tl.minimum(tl.maximum(up, -10.0), 10.0)
    original = tl.load(order + route)
    probability = tl.load(probabilities + original)
    value = (gate / (1.0 + tl.exp(-gate))) * up * probability
    tl.store(hidden + route * WIDTH + col, value, valid)


@triton.jit
def _v41_prefill_accumulate(
    down,
    inverse,
    bounds,
    acc,
    TOKENS,
    EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """acc[t] += sum of down[inverse[t, j]] over routes j inside this tile."""
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    low = tl.load(bounds)
    high = tl.load(bounds + EXPERTS)
    value = tl.zeros((BLOCK,), dtype=tl.float32)
    hit = token < 0
    for j in tl.static_range(TOPK):
        row = tl.load(inverse + token * TOPK + j)
        inside = (row >= low) & (row < high)
        hit |= inside
        value += tl.load(
            down + row.to(tl.int64) * H + cols, mask=inside & (cols < H), other=0.0
        ).to(tl.float32)
    mask = hit & (cols < H) & (token < TOKENS)
    where = acc + token.to(tl.int64) * H + cols
    tl.store(where, tl.load(where, mask=mask, other=0.0) + value, mask=mask)


@triton.jit
def _v41_prefill_activation_rows(
    projection,
    probabilities,
    order,
    out0,
    out1,
    out2,
    out3,
    bounds,
    ROUTES,
    EXPERTS: tl.constexpr,
    WIDTH: tl.constexpr,
    FULL: tl.constexpr,
    COLUMN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Row-split down: write this rank's h columns into all four full rows."""
    route = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (route < ROUTES) & (col < WIDTH)
    valid &= (route >= tl.load(bounds)) & (route < tl.load(bounds + EXPERTS))
    gate = tl.load(projection + route * 2 * WIDTH + 2 * col, valid, 0).to(tl.float32)
    up = tl.load(projection + route * 2 * WIDTH + 2 * col + 1, valid, 0).to(tl.float32)
    gate = tl.minimum(gate, 10.0)
    up = tl.minimum(tl.maximum(up, -10.0), 10.0)
    original = tl.load(order + route)
    probability = tl.load(probabilities + original)
    value = ((gate / (1.0 + tl.exp(-gate))) * up * probability).to(tl.float16)
    at = route.to(tl.int64) * FULL + COLUMN + col
    tl.store(out0.to(tl.pointer_type(tl.float16)) + at, value, valid)
    tl.store(out1.to(tl.pointer_type(tl.float16)) + at, value, valid)
    tl.store(out2.to(tl.pointer_type(tl.float16)) + at, value, valid)
    tl.store(out3.to(tl.pointer_type(tl.float16)) + at, value, valid)


@triton.jit
def _v41_prefill_activation_rows_p(
    projection,
    probabilities,
    order,
    out0,
    out1,
    out2,
    out3,
    bounds,
    EXPERTS: tl.constexpr,
    WIDTH: tl.constexpr,
    FULL: tl.constexpr,
    COLUMN: tl.constexpr,
    BLOCK: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    """Persistent form: a fixed grid strides over [bounds[0], bounds[-1])."""
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = col < WIDTH
    low = tl.load(bounds)
    high = tl.load(bounds + EXPERTS)
    for route in range(low + tl.program_id(0), high, PROGRAMS):
        gate = tl.load(projection + route * 2 * WIDTH + 2 * col, inside, 0).to(
            tl.float32
        )
        up = tl.load(projection + route * 2 * WIDTH + 2 * col + 1, inside, 0).to(
            tl.float32
        )
        gate = tl.minimum(gate, 10.0)
        up = tl.minimum(tl.maximum(up, -10.0), 10.0)
        probability = tl.load(probabilities + tl.load(order + route))
        value = ((gate / (1.0 + tl.exp(-gate))) * up * probability).to(tl.float16)
        at = route.to(tl.int64) * FULL + COLUMN + col  # type: ignore[attr-defined]
        tl.store(out0.to(tl.pointer_type(tl.float16)) + at, value, inside)
        tl.store(out1.to(tl.pointer_type(tl.float16)) + at, value, inside)
        tl.store(out2.to(tl.pointer_type(tl.float16)) + at, value, inside)
        tl.store(out3.to(tl.pointer_type(tl.float16)) + at, value, inside)


_ACT_PROGRAMS = int(envs.VLLM_V41_ACT_PROGRAMS)
# Size each tile's GEMMs to its own routes (host offsets) instead of the group.
_EXACT_GEMM = envs.VLLM_V41_EXACT_GEMM == "1"
# Measure TurboMind kernel configs during the first prefill pass(es) (the
# startup profile run); later passes reuse the measured family.
_TUNE_PASSES = int(envs.VLLM_V41_TUNE_PREFILL_PASSES)


class _DeviceView:
    """Zero-copy torch view of a raw device allocation."""

    def __init__(self, pointer, shape, typestr):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": typestr,
            "data": (int(pointer), False),
            "version": 3,
        }


_ROUTED_MAX = int(envs.VLLM_V41_ROUTED_PREFILL_MAX)


def tiles_for_layer(layer, spec, tile):
    return [
        (layer, f, min(tile, spec.experts - f)) for f in range(0, spec.experts, tile)
    ]


class SharedOriginalPrefill:
    """Serial preparation first: no outstanding host read at decode publish."""

    # Routing is independent of the old rank-local host layout.
    _route = cast(Any, TurboMindPrefill._route)

    def __init__(self, cache, tile_experts=32):
        self.cache = cache
        # One reusable bank, with room left for the target and draft graphs.
        self.tile_experts = min(tile_experts, cache.experts)
        if self.tile_experts < 1:
            raise ValueError("Canonical expert tile must have positive capacity")
        self.expert_ids = torch.arange(
            cache.experts + 1, dtype=torch.int32, device=cache.device
        )
        self.info = []
        maximum = {"gu": 0, "dn": 0}
        for spec in cache.specs:
            if spec.kinds[0] != spec.kinds[1]:
                raise ValueError("V4.1 prefill requires matching gate and up formats")
            rows = spec.intermediate // 4
            gu = _canonical(spec.kinds[0], 2 * rows, spec.hidden)
            # Row-split down: N = hidden / 4 output rows, full K.
            dn_n, dn_k = (
                (spec.hidden // 4, spec.intermediate)
                if DOWN_ROWS
                else (spec.hidden, rows)
            )
            dn = _canonical(spec.kinds[2], dn_n, dn_k)
            info = {
                "gu": gu,
                "dn": dn,
                "rows": rows,
                "hidden": spec.hidden,
                "dn_n": dn_n,
                "dn_k": dn_k,
            }
            self.info.append(info)
            for key in maximum:
                maximum[key] = max(maximum[key], self.tile_experts * sum(info[key][3:]))
        self.storage = {
            key: torch.empty(size, dtype=torch.uint8, device=cache.device)
            for key, size in maximum.items()
        }
        self.tables = []
        sequence = torch.arange(
            self.tile_experts, device=cache.device, dtype=torch.int64
        )
        for info in self.info:
            tables = {}
            for key, k, n in (
                ("gu", info["hidden"], 2 * info["rows"]),
                ("dn", info["dn_k"], info["dn_n"]),
            ):
                fmt = info[key]
                weight, stats = self.bank(key, fmt)
                pair = []
                for base, stride, ld in ((weight, fmt[3], 32 * k), (stats, fmt[4], n)):
                    table = torch.empty(
                        self.tile_experts, 2, dtype=torch.int64, device=cache.device
                    )
                    table[:, 0] = base.data_ptr() + sequence * stride
                    table[:, 1] = ld
                    pair.append(table.view(torch.uint8).flatten())
                tables[key] = pair
            self.tables.append(tables)
        self.error = torch.zeros(1, dtype=torch.int32, device=cache.device)
        self.hot_ids = torch.empty(
            self.tile_experts, dtype=torch.int32, device=cache.device
        )
        self.cold_ids = torch.empty_like(self.hot_ids)
        self.routed = torch.ones(cache.experts, dtype=torch.int8, device=cache.device)
        self.tuned_passes = 0
        self.hot_count = torch.empty(1, dtype=torch.int32, device=cache.device)
        self.cold_count = torch.empty_like(self.hot_count)
        self.cold_transcodes = torch.zeros(
            len(cache.specs), dtype=torch.int64, device=cache.device
        )
        # Decode graphs must wait for the latest prefill record, including
        # work issued outside their capture stream on a later request.
        self.done = torch.cuda.Event(external=True)
        self.used = False
        # Cold experts: DMA of each rank's original slice into a GPU staging
        # bank (device-carrier layout), double buffered and prefetched one
        # tile ahead on a copy stream, instead of mapped zero-copy reads.
        self.dma = envs.VLLM_V41_PREFILL_DMA == "1"
        self.timing = envs.VLLM_V41_PREFILL_TIMING == "1" and cache.rank == 0
        self._t = []  # (layer, kind, event) of the current pass
        self._s = []  # (kind, event) GEMM-loop sub-steps of the current pass
        if self.dma:
            self._init_staging()
        self.rows_mode = DOWN_ROWS
        if self.rows_mode:
            self._init_rows()

    def _init_rows(self):
        """Two parity banks of full-width h rows, mapped on all four ranks."""
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        cache = self.cache
        group = cache.group
        self.rows_full = max(s.intermediate for s in cache.specs)
        self.rows_max = cache.prefill_tokens * 6
        size = self.rows_max * self.rows_full * 2
        self.rows_pointers = [
            CustomAllreduce.create_shared_buffer(size, group.cpu_group)
            for _ in range(2)
        ]
        self.rows_local = [
            torch.as_tensor(
                _DeviceView(p[cache.rank], (self.rows_max, self.rows_full), "<f2"),
                device=cache.device,
            )
            for p in self.rows_pointers
        ]
        self.rows_nccl = group.device_communicator.pynccl_comm
        self.rows_fence = torch.zeros(1, dtype=torch.int32, device=cache.device)
        self.rows_parity = 0
        from . import gguf_v41_moe as moe

        self.rows_peer = CustomAllreduce.create_shared_buffer(
            moe.ROWS_BYTES, group.cpu_group
        )
        own = torch.as_tensor(
            _DeviceView(self.rows_peer[cache.rank], (moe.ROWS_BYTES,), "|u1"),
            device=cache.device,
        )
        moe.register_rows_peer(cache.device, cache.rank, self.rows_peer, own)
        if envs.VLLM_V41_WARM_TORCH_NCCL == "1":
            # Optional: create the torch NCCL communicator now (its lazy
            # cudaMalloc can land after a long chunk filled the allocator).
            # all_gather uses pynccl by default, so it is usually never made.
            probe = torch.zeros(4, device=cache.device)
            torch.distributed.all_gather_into_tensor(
                torch.empty(16, device=cache.device), probe, group=group.device_group
            )
        torch.accelerator.synchronize(cache.device)
        self.cache._barrier()  # every rank's flags are zero before any gather

    def _rows_barrier(self):
        # Every rank's P2P h writes precede this NCCL turn in stream order.
        self.rows_nccl.all_reduce(self.rows_fence)

    def _init_staging(self):
        cache = self.cache
        self.slice_bytes = []  # [layer][shard]
        largest = [0, 0, 0]
        guard = [0, 0, 0]
        for spec in cache.specs:
            sizes = []
            for s, geometry in enumerate(spec.layouts(cache.rank)):
                size = geometry.device_rows * geometry.device_row_bytes
                sizes.append(size)
                largest[s] = max(largest[s], size)
                guard[s] = max(guard[s], geometry.guard_bytes)
            self.slice_bytes.append(sizes)
        self.ring = envs.VLLM_V41_STAGE_RING == "1"
        if self.ring:
            self._init_ring(largest, guard)
            return
        self.staging = [
            [
                torch.empty(
                    self.tile_experts * largest[s] + guard[s] + 512,
                    dtype=torch.uint8,
                    device=cache.device,
                )
                for s in range(3)
            ]
            for _ in range(2)
        ]
        local = torch.arange(self.tile_experts, device=cache.device, dtype=torch.int64)
        # pointers[layer][buffer][shard]: per-local-expert staging addresses.
        self.stage_pointers = [
            [
                [
                    self.staging[b][s].data_ptr() + local * self.slice_bytes[l][s]
                    for s in range(3)
                ]
                for b in range(2)
            ]
            for l in range(len(cache.specs))  # noqa: E741
        ]
        self.copy_stream = torch.cuda.Stream(device=cache.device)
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.free = [torch.cuda.Event() for _ in range(2)]
        for event in self.free:
            event.record(torch.cuda.current_stream(cache.device))
        self.issued = {}
        self.host_table = self.host_cold = None
        from .gguf_expert_cache import _native

        self.native = _native()

    def _init_ring(self, largest, guard):
        """Cold expert slots in one ring per shard; entries are tiles in issue order."""
        cache = self.cache
        slots = int(envs.VLLM_V41_STAGE_SLOTS)
        if slots < self.tile_experts:
            raise ValueError("Staging ring must hold at least one full tile")
        self.ring_slots, self.ring_bytes = slots, list(largest)
        self.ring_bufs = [
            torch.empty(
                slots * largest[s] + guard[s] + 512,
                dtype=torch.uint8,
                device=cache.device,
            )
            for s in range(3)
        ]
        self.ring_rows = 64
        self.ring_ptr_dev = torch.zeros(
            self.ring_rows, 3, self.tile_experts, dtype=torch.int64, device=cache.device
        )
        self.ring_ptr_host = torch.zeros(
            self.ring_rows, 3, self.tile_experts, dtype=torch.int64, pin_memory=True
        )
        self.ring_row_ready = [None] * self.ring_rows
        self.ring_fifo: Any = deque()
        self.ring_entries = {}
        self.ring_pending: Any = deque()
        self.ring_head = 0
        self.ring_free = slots
        self.ring_short = False
        self.ring_routed = {}
        self.ring_row = 0
        self.copy_stream = torch.cuda.Stream(device=cache.device)
        self.issued = {}
        self.host_table = self.host_cold = None
        from .gguf_expert_cache import _native

        self.native = _native()

    def _ring_start(self, short=False):
        """A new prefill pass: refresh residency, queue every tile of every layer.

        Short passes queue nothing ahead: each layer queues its own tiles with
        only the routed cold experts once its routes are known (G6)."""
        self._refresh_host_tables()
        self.ring_entries = {}
        self.ring_short = short
        self.ring_routed = {}
        self.ring_pending = (
            deque()
            if short
            else deque(
                (layer, f, min(self.tile_experts, spec.experts - f))
                for layer, spec in enumerate(self.cache.specs)
                for f in range(0, spec.experts, self.tile_experts)
            )
        )

    def _ring_pump(self, block=False):
        """Issue pending tiles while ring slots and pointer rows are available."""
        cache, runtime = self.cache, self.cache.runtime
        stream = self.copy_stream
        base = runtime.host.data_ptr()
        while self.ring_pending:
            layer, first, count = self.ring_pending[0]
            table, cold = self.host_table[layer], self.host_cold[layer]  # type: ignore[index]
            used = self.ring_routed.get(layer)
            local = [
                i
                for i in range(count)
                if table[first + i] < 0 and (used is None or used[first + i])
            ]
            need = len(local)
            fifo = self.ring_fifo
            while (
                (need > self.ring_free or len(fifo) >= self.ring_rows)
                and fifo
                and fifo[0].consumed
            ):
                done = fifo.popleft()
                stream.wait_event(done.free)
                self.ring_free += done.nslots
            if need > self.ring_free or len(fifo) >= self.ring_rows:
                return
            row = self.ring_row
            previous = self.ring_row_ready[row]
            if previous is not None and not previous.query():
                # Its pointer H2D still reads this pinned row: do not block the
                # host (that stalls compute issue) unless a tile needs it now.
                if not block:
                    return
                previous.synchronize()
            self.ring_row = (row + 1) % self.ring_rows
            host = self.ring_ptr_host[row]
            table_row = [[0] * self.tile_experts for _ in range(3)]
            spec = cache.specs[layer]
            geometries = spec.layouts(cache.rank)
            dst, src, dpitch, spitch, width, height = [], [], [], [], [], []
            for i in local:
                slot = self.ring_head % self.ring_slots
                self.ring_head += 1
                at = cold[first + i]
                if at < 0:
                    raise RuntimeError(
                        "Shared prefill found an expert neither hot nor cold"
                    )
                for sh, (geometry, size) in enumerate(
                    zip(geometries, spec.projection_bytes)
                ):
                    address = self.ring_bufs[sh].data_ptr() + slot * self.ring_bytes[sh]
                    table_row[sh][i] = address
                    src.append(
                        base
                        + at
                        + geometry.source_row_start * geometry.source_row_bytes
                        + geometry.first_block * geometry.block_bytes
                    )
                    dst.append(address)
                    dpitch.append(geometry.device_row_bytes)
                    spitch.append(geometry.source_row_bytes)
                    width.append(geometry.device_row_bytes)
                    height.append(geometry.device_rows)
                    at += size
            self.ring_free -= need
            host.copy_(torch.tensor(table_row, dtype=torch.int64))
            self._mark(layer, "copy0", stream)
            if dst:
                self.native.copy_h2d_2d(
                    dst, src, dpitch, spitch, width, height, stream.cuda_stream
                )
            with torch.cuda.stream(stream):
                self.ring_ptr_dev[row].copy_(host, non_blocking=True)
            self._mark(layer, "copy1", stream)
            ready = torch.cuda.Event()
            ready.record(stream)
            self.ring_row_ready[row] = ready
            entry = SimpleNamespace(
                nslots=need,
                ready=ready,
                free=torch.cuda.Event(),
                consumed=False,
                pointers=[self.ring_ptr_dev[row, sh] for sh in range(3)],
            )
            fifo.append(entry)
            self.ring_entries[layer, first] = entry
            self.ring_pending.popleft()

    def _ring_take(self, layer, first):
        entry = self.ring_entries.pop((layer, first), None)
        if entry is None:
            self._ring_pump(block=True)
            entry = self.ring_entries.pop((layer, first), None)
        if entry is None:
            raise RuntimeError("Prefill staging ring could not issue the next tile")
        return entry

    def _refresh_host_tables(self):
        runtime = self.cache.runtime
        # Residency changes only in decode commits; one read per prefill pass.
        self.host_table = runtime.table.cpu().tolist()
        if hasattr(runtime, "wait_jobs"):
            runtime.wait_jobs()  # evicted experts' host copies must be complete
        self.host_cold = runtime.cold_offsets.cpu().tolist()

    def _issue(self, layer, first, count, buffer):
        """Queue DMA of this tile's cold expert slices into staging[buffer]."""
        cache, runtime = self.cache, self.cache.runtime
        spec = cache.specs[layer]
        stream = self.copy_stream
        stream.wait_event(self.free[buffer])
        base = runtime.host.data_ptr()
        table, cold = self.host_table[layer], self.host_cold[layer]  # type: ignore[index]
        dst, src, dpitch, spitch, width, height = [], [], [], [], [], []
        geometries = spec.layouts(cache.rank)
        for i in range(count):
            expert = first + i
            if table[expert] >= 0:
                continue
            at = cold[expert]
            if at < 0:
                raise RuntimeError(
                    "Shared prefill found an expert neither hot nor cold"
                )
            for s, (geometry, size) in enumerate(
                zip(geometries, spec.projection_bytes)
            ):
                src.append(
                    base
                    + at
                    + geometry.source_row_start * geometry.source_row_bytes
                    + geometry.first_block * geometry.block_bytes
                )
                dst.append(
                    self.staging[buffer][s].data_ptr() + i * self.slice_bytes[layer][s]
                )
                dpitch.append(geometry.device_row_bytes)
                spitch.append(geometry.source_row_bytes)
                width.append(geometry.device_row_bytes)
                height.append(geometry.device_rows)
                at += size
        self._mark(layer, "copy0", stream)
        if dst:
            self.native.copy_h2d_2d(
                dst, src, dpitch, spitch, width, height, stream.cuda_stream
            )
        self._mark(layer, "copy1", stream)
        self.ready[buffer].record(stream)
        self.issued[layer, first] = buffer

    def _sub(self, kind):
        if self.timing:
            e = torch.cuda.Event(enable_timing=True)
            e.record(torch.cuda.current_stream(self.cache.device))
            self._s.append((kind, e))

    def _mark(self, layer, kind, stream=None):
        if self.timing:
            e = torch.cuda.Event(enable_timing=True)
            e.record(stream or torch.cuda.current_stream(self.cache.device))
            self._t.append((layer, kind, e))

    def _report(self, tokens):
        torch.accelerator.synchronize(self.cache.device)
        marks = self._t
        self._t = []
        first: Any = {}
        for layer, kind, e in marks:
            first.setdefault((layer, kind), []).append(e)
        layers = sorted({l for l, _, _ in marks})  # noqa: E741
        tot = dict(nonmoe=0.0, moe=0.0, wait=0.0, transcode=0.0, gemm=0.0, copy=0.0)
        prev_end = None
        for l in layers:  # noqa: E741
            start, end = first[(l, "start")][0], first[(l, "end")][0]
            if prev_end is not None:
                tot["nonmoe"] += prev_end.elapsed_time(start)
            tot["moe"] += start.elapsed_time(end)
            for a, b in zip(
                first.get((l, "tile0"), []), first.get((l, "prepared"), [])
            ):
                pass
            t0, w0, w1, pr, ge = (
                first.get((l, k), [])
                for k in ("tile0", "wait0", "wait1", "prepared", "gemm")
            )
            for i in range(len(t0)):
                wait = w0[i].elapsed_time(w1[i]) if i < len(w0) and i < len(w1) else 0.0
                tot["wait"] += wait
                tot["transcode"] += t0[i].elapsed_time(pr[i]) - wait
                tot["gemm"] += pr[i].elapsed_time(ge[i])
            c0, c1 = first.get((l, "copy0"), []), first.get((l, "copy1"), [])
            for a, b in zip(c0, c1):
                tot["copy"] += a.elapsed_time(b)
            prev_end = end
        span = first[(layers[0], "start")][0].elapsed_time(
            first[(layers[-1], "end")][0]
        )
        sub, subs = {}, self._s  # type: ignore[var-annotated]
        self._s = []
        for (a, ea), (b, eb) in zip(subs, subs[1:]):
            if a != "tile_end":
                sub[a] = sub.get(a, 0.0) + ea.elapsed_time(eb)
        if sub:
            print(
                "[prefill gemm] " + " ".join(f"{k}={v:.0f}" for k, v in sub.items()),
                flush=True,
            )
        print(
            "[prefill timing] tokens %d layers %d span %.0f ms | non-MoE %.0f | MoE %.0f = wait %.0f + "  # noqa: E501, UP031
            "transcode %.0f + gemm %.0f (+unroute/other) | copy-stream busy %.0f ms"
            % (
                tokens,
                len(layers),
                span,
                tot["nonmoe"],
                tot["moe"],
                tot["wait"],
                tot["transcode"],
                tot["gemm"],
                tot["copy"],
            ),
            flush=True,
        )

    def bank(self, key, fmt):
        size = self.tile_experts * fmt[3]
        return (
            self.storage[key][:size],
            self.storage[key][size : size + self.tile_experts * fmt[4]],
        )

    def storage_bytes(self):
        return sum(t.numel() for t in self.storage.values())

    def wait_decode(self):
        if self.used:
            torch.cuda.current_stream(self.cache.device).wait_event(self.done)

    def prepare(self, layer, first, count, buffer=None):
        cache, runtime = self.cache, self.cache.runtime
        spec, info = cache.specs[layer], self.info[layer]
        self.wait_decode()  # Also fences reuse of the single canonical workspace.
        _partition_tile[(1,)](
            runtime.table[layer],
            self.routed,
            self.hot_ids,
            self.cold_ids,
            self.hot_count,
            self.cold_count,
            self.cold_transcodes[layer : layer + 1],
            FIRST=first,
            COUNT=count,
            BLOCK=triton.next_power_of_2(self.tile_experts),
        )
        slots = runtime.table[layer, first : first + count]
        cold = runtime.cold_offsets[layer, first : first + count]
        hot_sources, cold_sources = [], []
        staged = buffer is not None
        ring = staged and not isinstance(buffer, int)
        if staged:
            cold_sources = (
                buffer.pointers if ring else self.stage_pointers[layer][buffer]
            )
        at = 0
        for shard, geometry, size in zip(
            SHARDS, spec.layouts(cache.rank), spec.projection_bytes
        ):
            bank = runtime.banks[layer, shard]
            hot_sources.append(bank.data_ptr() + slots.long() * bank[0].numel())
            if not staged:
                cold_sources.append(
                    runtime.mapped.data_ptr()
                    + cold
                    + at
                    + geometry.source_row_start * geometry.source_row_bytes
                )
            at += size
        for key, source, n, k, kind, geometry in (
            (
                "gu",
                0,
                2 * info["rows"],
                spec.hidden,
                spec.kinds[0],
                spec.layouts(cache.rank)[0],
            ),
            (
                "dn",
                2,
                info["dn_n"],
                info["dn_k"],
                spec.kinds[2],
                spec.layouts(cache.rank)[2],
            ),
        ):
            fmt = info[key]
            weights, stats = self.bank(key, fmt)
            for resident, ids, count, pointers in (
                (True, self.hot_ids, self.hot_count, hot_sources),
                (False, self.cold_ids, self.cold_count, cold_sources),
            ):
                if not resident and staged and key == "gu":
                    self._mark(layer, "wait0")
                    torch.cuda.current_stream(cache.device).wait_event(
                        buffer.ready if ring else self.ready[buffer]
                    )
                    self._mark(layer, "wait1")
                # Staged cold slices already have the device-carrier layout.
                carrier = resident or staged
                # Cold down pointers start at column zero of the full source
                # rows. The absolute TP K offset and full row stride must stay
                # paired; hot carriers instead use the intra-block offset.
                offset = (
                    geometry.logical_k_offset
                    if carrier
                    else cache.rank * info["rows"]
                    if key == "dn"
                    else 0
                )
                stride = (
                    geometry.device_row_bytes if carrier else geometry.source_row_bytes
                )
                torch.ops._C.gguf_prefill_transcode_sm70(
                    weights,
                    stats,
                    fmt[3],
                    fmt[4],
                    pointers[source],
                    pointers[1] if key == "gu" else None,
                    ids,
                    count,
                    128 if carrier else 8,
                    kind,
                    n,
                    k,
                    offset,
                    stride,
                    self.error,
                )
        if ring:
            buffer.free.record(torch.cuda.current_stream(cache.device))
            buffer.consumed = True
            self._ring_pump()
        elif staged:
            self.free[buffer].record(torch.cuda.current_stream(cache.device))
        _check_transcode[(1,)](self.error)

    def forward(self, layer, x, ids, probabilities):
        spec, info, tables = (
            self.cache.specs[layer],
            self.info[layer],
            self.tables[layer],
        )
        if (
            x.dtype != torch.float16
            or x.shape[1] != spec.hidden
            or ids.shape != (x.shape[0], 6)
            or probabilities.shape != ids.shape
            or probabilities.dtype != torch.float32
        ):
            raise ValueError("V4.1 shared prefill requires FP16, top-6 and FP32 routes")
        self._mark(layer, "start")
        routes = []
        for first in range(0, x.shape[0], self.cache.prefill_tokens):
            last = min(first + self.cache.prefill_tokens, x.shape[0])
            sub_ids = ids[first:last].contiguous()
            offsets, order, rows, top_k, inverse, _ = self._route(
                sub_ids, first, spec.experts
            )
            routes.append((first, last, offsets, order, rows, top_k, inverse))
        if not routes:
            return torch.empty_like(x)
        maximum = min(x.shape[0], self.cache.prefill_tokens) * 6
        projection = torch.empty(
            maximum, 2 * info["rows"], dtype=x.dtype, device=x.device
        )
        out_n = info["dn_n"]
        hidden = (
            None
            if self.rows_mode
            else torch.empty(maximum, info["rows"], dtype=x.dtype, device=x.device)
        )
        # Down results of one route group live only for one tile: each tile
        # adds its routes into an FP32 token accumulator (tile order, then
        # top-k order inside a tile), so memory no longer scales with 6x tokens.
        down = torch.empty(maximum, out_n, dtype=x.dtype, device=x.device)
        acc = torch.zeros(x.shape[0], out_n, dtype=torch.float32, device=x.device)
        tiles = [
            (f, min(self.tile_experts, spec.experts - f))
            for f in range(0, spec.experts, self.tile_experts)
        ]
        host_offsets = [o.tolist() for _, _, o, *_ in routes] if _EXACT_GEMM else None
        tune = self.tuned_passes < _TUNE_PASSES
        ring = self.dma and self.ring
        self.routed = torch.zeros(spec.experts, dtype=torch.int8, device=x.device)
        self.routed.index_fill_(0, ids.reshape(-1).long(), 1)
        if ring:
            if layer == 0 or self.host_table is None:
                self._ring_start(short=x.shape[0] <= _ROUTED_MAX)
            if self.ring_short:
                self.ring_routed[layer] = self.routed.tolist()  # one sync per layer
                self.ring_pending.extend(
                    tiles_for_layer(layer, spec, self.tile_experts)
                )
            self._ring_pump()
        elif self.dma:
            if layer == 0 or self.host_table is None:
                self.issued.clear()
                self._refresh_host_tables()
            if (layer, 0) not in self.issued:
                self._issue(layer, 0, tiles[0][1], 0)
        for index, (expert_first, expert_count) in enumerate(tiles):
            buffer = None
            if ring:
                buffer = self._ring_take(layer, expert_first)
            elif self.dma:
                buffer = self.issued.pop((layer, expert_first))
                if index + 1 < len(tiles):
                    self._issue(layer, *tiles[index + 1], 1 - buffer)
                elif layer + 1 < len(self.cache.specs):
                    nxt = self.cache.specs[layer + 1]
                    self._issue(
                        layer + 1, 0, min(self.tile_experts, nxt.experts), 1 - buffer
                    )
            self._mark(layer, "tile0")
            self.prepare(layer, expert_first, expert_count, buffer)
            self._mark(layer, "prepared")
            for group, (first, last, offsets, order, rows, top_k, inverse) in enumerate(
                routes
            ):
                count = (last - first) * 6
                bounds = offsets[expert_first : expert_first + expert_count + 1]
                if host_offsets is not None:
                    lo = host_offsets[group][expert_first]
                    hi = host_offsets[group][expert_first + expert_count]
                    if hi == lo:
                        continue  # no route of this group in the tile
                    rel = bounds - lo
                else:
                    lo, hi, rel = 0, count, bounds
                fmt = info["gu"]
                self._sub("gu")
                torch.ops._C.gguf_grouped_prefill_gemm_sm70_out(
                    projection[lo:hi],
                    x,
                    rel,
                    *(t[: expert_count * 16] for t in tables["gu"]),
                    fmt[0],
                    fmt[1],
                    2 * info["rows"],
                    expert_count,
                    fmt[2],
                    False,
                    tune,
                    rows[lo:hi],
                )
                self._sub("act")
                if self.rows_mode:
                    parity = self.rows_parity
                    self.rows_parity ^= 1
                    if _ACT_PROGRAMS:
                        programs = min(count, _ACT_PROGRAMS)
                        _v41_prefill_activation_rows_p[
                            (programs, triton.cdiv(info["rows"], 256))
                        ](
                            projection,
                            probabilities[first:last].contiguous(),
                            order,
                            *self.rows_pointers[parity],
                            bounds,
                            EXPERTS=expert_count,
                            WIDTH=info["rows"],
                            FULL=self.rows_full,
                            COLUMN=self.cache.rank * info["rows"],
                            BLOCK=256,
                            PROGRAMS=_ACT_PROGRAMS,  # fixed stride: no recompiles
                            enable_fp_fusion=False,
                        )
                    else:
                        _v41_prefill_activation_rows[
                            (count, triton.cdiv(info["rows"], 256))
                        ](
                            projection,
                            probabilities[first:last].contiguous(),
                            order,
                            *self.rows_pointers[parity],
                            bounds,
                            ROUTES=count,
                            EXPERTS=expert_count,
                            WIDTH=info["rows"],
                            FULL=self.rows_full,
                            COLUMN=self.cache.rank * info["rows"],
                            BLOCK=256,
                            enable_fp_fusion=False,
                        )
                    self._sub("barrier")
                    self._rows_barrier()
                    h = self.rows_local[parity][:count, : info["dn_k"]]
                else:
                    _v41_prefill_activation[(count, triton.cdiv(info["rows"], 256))](
                        projection,
                        probabilities[first:last].contiguous(),
                        order,
                        hidden,
                        bounds,
                        ROUTES=count,
                        EXPERTS=expert_count,
                        WIDTH=info["rows"],
                        BLOCK=256,
                        enable_fp_fusion=False,
                    )
                    h = hidden[:count]  # type: ignore[index]
                fmt = info["dn"]
                self._sub("dn")
                torch.ops._C.gguf_grouped_prefill_gemm_sm70_out(
                    down[lo:hi],
                    h[lo:hi],
                    rel,
                    *(t[: expert_count * 16] for t in tables["dn"]),
                    fmt[0],
                    fmt[1],
                    out_n,
                    expert_count,
                    fmt[2],
                    False,
                    tune,
                    None,
                )
                # Route probabilities were applied before the FP16 down input.
                self._sub("accumulate")
                _v41_prefill_accumulate[(last - first, triton.cdiv(out_n, 1024))](
                    down,
                    inverse,
                    bounds,
                    acc[first:last],
                    last - first,
                    EXPERTS=expert_count,
                    TOPK=top_k,
                    H=out_n,
                    BLOCK=1024,
                )
            self._sub("tile_end")
            self._mark(layer, "gemm")
        if self.rows_mode:
            # Other ranks' rows stay zero; the TP all-reduce assembles them.
            result = torch.zeros_like(x)
            column = self.cache.rank * out_n
            result[:, column : column + out_n].copy_(acc)
        else:
            result = acc.to(x.dtype)
        del acc
        self.done.record(torch.cuda.current_stream(self.cache.device))
        self._mark(layer, "end")
        if layer == len(self.cache.specs) - 1:
            self.tuned_passes += 1
        if self.timing and layer == len(self.cache.specs) - 1:
            self._report(x.shape[0])
        self.used = True
        return result
