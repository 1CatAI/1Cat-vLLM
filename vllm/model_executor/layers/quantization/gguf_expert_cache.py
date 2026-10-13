# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-expert GPU cache: a fraction of each layer's experts in GPU slots.

Each MoE layer keeps ``slots`` experts per projection in a slot bank
``[slots, rows, bytes]``; every other expert lives in a pinned host arena.
The GGUF expert kernels address ``ids * rows * stride`` from the bank base,
so routing ids remapped to slots run unchanged.

Decode (graph-safe): one kernel looks the routed ids up in the device slot
table and publishes them to a host-mapped ring. On a miss it spins until the
native lane (``csrc/gguf_expert_cache.cpp``) answers with a copy plan in
host-mapped memory. A second kernel performs the plan (write-backs, then
loads, read from pinned host memory in-kernel) and a third updates the slot
table and remaps the ids. The lane itself issues no GPU work: a release that
went through a CUDA stream could queue behind the wait it releases.

Prefill: the lane assembles the layer's complete bank (slot bank + cold
experts) into one of two buffers, prefetching the next layer while this one
computes, and the routed ids are remapped into that bank.
"""

import os
import threading
import time
from typing import Any

import regex as re
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

SHARDS = ("w1", "w3", "w2")
_ENTRY_HEADER = 8
_RING_ENTRIES = 1024
_MAX_OPS = 256
_COPY_PROGRAMS = (16, 8)
_LAYER = re.compile(r"layers\.(\d+)\.")
_MEMLOG = envs.VLLM_EXPERT_CACHE_MEMLOG


@triton.jit(do_not_specialize=["layer", "count"])
def _lookup_publish(
    ids,
    table,
    counter,
    ring,
    published,
    reply,
    plan,
    layer,
    count,
    EXPERTS: tl.constexpr,
    RING: tl.constexpr,
    ENTRY: tl.constexpr,
    BLOCK: tl.constexpr,
    PLAN: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    mask = lane < count
    expert = tl.load(ids + lane, mask=mask, other=0)
    slot = tl.load(table + layer * EXPERTS + expert, mask=mask, other=0)
    miss = tl.max(tl.where(mask & (slot < 0), 1, 0), axis=0)
    seq = tl.load(counter)
    entry = ring + (seq % RING) * ENTRY
    tl.store(entry + 8 + lane, expert, mask=mask)
    tl.store(entry + 1, layer.to(tl.int32))
    tl.store(entry + 2, count.to(tl.int32))
    tl.store(entry + 3, miss.to(tl.int32))
    tl.debug_barrier()
    tl.atomic_xchg(entry, seq.to(tl.int32), sem="release", scope="sys")
    tl.atomic_xchg(published, seq + 1, sem="release", scope="sys")
    tl.store(counter, seq + 1)
    ops = 0
    if miss > 0:
        # The lane answers with this request's sequence number.
        # A failed lane answers every request with no copies.
        tag = tl.load(reply, volatile=True)
        failed = tl.load(reply + 2, volatile=True)
        while (tag != seq.to(tl.int32)) & (failed == 0):
            tag = tl.load(reply, volatile=True)
            failed = tl.load(reply + 2, volatile=True)
        tag = tl.atomic_add(reply, 0, sem="acquire", scope="sys")
        ops = tl.maximum(tl.load(reply + 1, volatile=True), 0)
        ops = tl.where(failed == 0, ops, 0)
        words = tl.arange(0, PLAN)
        body = words < ops * 5
        tl.store(
            plan + 8 + words,
            tl.load(reply + 8 + words, mask=body, other=0, volatile=True),
            mask=body,
        )
    tl.store(plan, ops)


@triton.jit(do_not_specialize=["layer"])
def _apply_plan(
    plan,
    arena,
    host,
    device_offsets,
    host_offsets,
    words,
    layer,
    OPS_PROGRAMS: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write back, then load, the planned experts (int64 words).

    Both sides of a word are read before either is written, so an op may swap
    the evicted expert into the host slot of the expert it loads.
    """
    ops = tl.load(plan)
    lane = tl.arange(0, BLOCK)
    for op in range(tl.program_id(0), ops, OPS_PROGRAMS):
        slot = tl.load(plan + 8 + op * 5).to(tl.int64)
        load = tl.load(plan + 9 + op * 5).to(tl.int64)
        back = tl.load(plan + 10 + op * 5).to(tl.int64)
        for shard in tl.static_range(3):
            n = tl.load(words + layer * 3 + shard)
            device = tl.load(device_offsets + layer * 3 + shard) + slot * n
            host_base = tl.load(host_offsets + layer * 3 + shard)
            for begin in range(tl.program_id(1) * BLOCK, n, SPLIT * BLOCK):
                offsets = begin + lane
                mask = offsets < n
                old = tl.load(arena + device + offsets, mask=mask & (back >= 0))
                new = tl.load(
                    host + host_base + load * n + offsets, mask=mask & (load >= 0)
                )
                tl.store(
                    host + host_base + back * n + offsets, old, mask=mask & (back >= 0)
                )
                tl.store(arena + device + offsets, new, mask=mask & (load >= 0))


@triton.jit(do_not_specialize=["layer", "count"])
def _remap(
    ids,
    table,
    plan,
    out,
    layer,
    count,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    OPS: tl.constexpr,
):
    ops = tl.load(plan)
    op = tl.arange(0, OPS)
    valid = op < ops
    row = table + layer * EXPERTS
    slot = tl.load(plan + 8 + op * 5, mask=valid, other=-1)
    added = tl.load(plan + 11 + op * 5, mask=valid, other=-1)
    evicted = tl.load(plan + 12 + op * 5, mask=valid, other=-1)
    tl.store(row + evicted, -1, mask=valid & (evicted >= 0))
    tl.debug_barrier()
    tl.store(row + added, slot, mask=valid & (added >= 0))
    tl.debug_barrier()
    lane = tl.arange(0, BLOCK)
    mask = lane < count
    expert = tl.load(ids + lane, mask=mask, other=0)
    slot = tl.maximum(tl.load(row + expert, mask=mask, other=0), 0)
    tl.store(out + lane, slot, mask=mask)


@triton.jit
def _map_ids(ids, table, out, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    expert = tl.load(ids + offsets, mask=mask, other=0)
    tl.store(out + offsets, tl.load(table + expert, mask=mask, other=0), mask=mask)


_NATIVE = None


def _native():
    global _NATIVE
    if _NATIVE is None:
        from torch.utils.cpp_extension import load

        source = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "csrc",
            "gguf_expert_cache.cpp",
        )
        _NATIVE = load(
            name="gguf_expert_cache_lane",
            sources=[os.path.abspath(source)],
            extra_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    return _NATIVE


def layer_index(prefix: str) -> int:
    match = _LAYER.search(prefix)
    return int(match.group(1)) if match else -1


class ExpertCache:
    """Per-rank storage and residency of every cached MoE layer."""

    def __init__(
        self,
        banks,
        num_experts,
        fraction,
        spare,
        host_slack,
        profile,
        device,
        prefill_mode="mmq",
        prefill_tokens=4096,
    ):
        """``banks``: [(layer_index, shard, (experts, rows, bytes), tail,
        (gguf_type, logical_k, k_offset))]."""
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.accelerator.current_device_index())
        self.device = device
        self.experts = num_experts
        self.spare = spare
        self.slots = max(
            min(int(round(fraction * num_experts)), num_experts), spare + 32
        )
        self.resident0 = self.slots - spare
        self.host_slots = (
            num_experts - self.resident0 + int(round(host_slack * num_experts))
        )
        self.layers = sorted({layer for layer, *_ in banks})
        self.index = {layer: i for i, layer in enumerate(self.layers)}
        shapes = {}
        tails = {}
        self.sources = {}
        for layer, shard, shape, tail, source in banks:
            if shape[0] != num_experts:
                raise ValueError("Expert cache needs complete expert banks")
            shapes[layer, shard] = tuple(shape[1:])
            tails[layer, shard] = tail
            self.sources[layer, shard] = source
        layers = len(self.layers)
        size = {key: shape[0] * shape[1] for key, shape in shapes.items()}
        self.slot_bytes = [size[layer, s] for layer in self.layers for s in SHARDS]

        # Device: per layer and projection a slot bank plus its zero tail.
        offsets = {}
        total = 0
        for layer in self.layers:
            for shard in SHARDS:
                total = (total + 255) // 256 * 256
                offsets[layer, shard] = total
                total += self.slots * size[layer, shard] + tails[layer, shard]
        self.arena = torch.zeros((total + 7) // 8 * 8, dtype=torch.uint8, device=device)
        self.banks = {
            key: self.arena[off : off + self.slots * size[key]].view(
                self.slots, *shapes[key]
            )
            for key, off in offsets.items()
        }
        self.shapes, self.sizes = shapes, size
        self.prefill_mode = prefill_mode
        self.prefill_tokens = prefill_tokens
        self.prefill_rows = num_experts + spare
        self.prefill_storage = []
        if prefill_mode == "turbomind":
            self.tm: TurboMindPrefill | None = TurboMindPrefill(self)
        else:
            # Two complete original banks (slots + cold experts) per
            # projection, sized for the largest layer and viewed per layer.
            self.tm = None
            for _ in range(2):
                group = {}
                for shard in SHARDS:
                    largest = max(
                        self.prefill_rows * size[layer, shard] + tails[layer, shard]
                        for layer in self.layers
                    )
                    group[shard] = torch.zeros(
                        largest, dtype=torch.uint8, device=device
                    )
                self.prefill_storage.append(group)
        self.prefill_maps = torch.zeros(
            (2, num_experts), dtype=torch.int32, device=device
        )

        # Host: one registered arena, per layer and projection [host_slots].
        host_bytes = sum(self.host_slots * b for b in self.slot_bytes)
        # Page-aligned and page-sized, so registrations never share a page.
        page = 4096
        host_bytes = (host_bytes + page - 1) // page * page
        backing = torch.empty(host_bytes + page, dtype=torch.uint8)
        start = (-backing.data_ptr()) % page
        self.host_arena = backing[start : start + host_bytes]
        self.host = {}
        offset = 0
        for layer in self.layers:
            for shard in SHARDS:
                nbytes = self.host_slots * size[layer, shard]
                self.host[layer, shard] = self.host_arena[
                    offset : offset + nbytes
                ].view(self.host_slots, *shapes[layer, shard])
                offset += nbytes
        _native().host_register(self.host_arena.data_ptr(), host_bytes)
        self.host_view = (
            _native().mapped_view(self.host_arena, device.index).view(torch.int64)
        )
        self.arena_words = self.arena.view(torch.int64)
        if any(b % 8 for b in self.slot_bytes):
            raise ValueError("Expert cache slots must be whole 8-byte words")
        arena_base = self.arena.data_ptr()
        host_base = self.host_arena.data_ptr()
        self.device_offsets = torch.tensor(
            [
                (self.banks[layer, s].data_ptr() - arena_base) // 8
                for layer in self.layers
                for s in SHARDS
            ],
            dtype=torch.int64,
            device=device,
        )
        self.host_offsets = torch.tensor(
            [
                (self.host[layer, s].data_ptr() - host_base) // 8
                for layer in self.layers
                for s in SHARDS
            ],
            dtype=torch.int64,
            device=device,
        )
        self.words = torch.tensor(
            [b // 8 for b in self.slot_bytes], dtype=torch.int64, device=device
        )

        # Initial residency: the profile's most routed experts, else low ids.
        ranking = None
        if profile:
            counts = torch.load(profile, map_location="cpu")
            if (
                counts.dim() == 2
                and counts.shape[1] == num_experts
                and (counts.shape[0] > max(self.layers))
            ):
                ranking = torch.argsort(
                    counts.double(), dim=1, descending=True, stable=True
                )
            else:
                logger.warning(
                    "Ignoring expert cache profile of shape %s", tuple(counts.shape)
                )
        self.slot_of = torch.full((layers, num_experts), -1, dtype=torch.int32)
        self.host_slot_of = torch.full((layers, num_experts), -1, dtype=torch.int32)
        for i, layer in enumerate(self.layers):
            order = ranking[layer] if ranking is not None else torch.arange(num_experts)
            resident = order[: self.resident0]
            self.slot_of[i, resident] = torch.arange(self.resident0, dtype=torch.int32)
            cold = order[self.resident0 :]
            self.host_slot_of[i, cold] = torch.arange(len(cold), dtype=torch.int32)
        self.table = self.slot_of.to(device)
        self.counter = torch.zeros(1, dtype=torch.int64, device=device)
        self.ring = torch.zeros((_RING_ENTRIES, 256), dtype=torch.int32).pin_memory()
        self.published = torch.zeros(8, dtype=torch.int64).pin_memory()
        self.ring_view = torch.ops._C.get_cuda_view_from_cpu_tensor(self.ring)
        self.published_view = torch.ops._C.get_cuda_view_from_cpu_tensor(self.published)
        # The lane's copy plan for the current miss, and its device copy.
        self.reply = torch.full((8 + 5 * _MAX_OPS,), -1, dtype=torch.int32).pin_memory()
        self.reply[1:8] = 0
        self.reply_view = torch.ops._C.get_cuda_view_from_cpu_tensor(self.reply)
        self.plan = torch.zeros(8 + 5 * _MAX_OPS, dtype=torch.int32, device=device)
        # Started lazily by start() on the first forward.
        self.lane: Any = None
        self.copy_stream: Any = None
        self.ready: list[Any] = [None, None]
        self.done: list[Any] = [None, None]
        self.prefetched = {}
        self.loaded = 0
        if self.tm is not None:
            self.tm.bind_host()
        prefill_bytes = (
            self.tm.storage_bytes()
            if self.tm is not None
            else sum(b.numel() for g in self.prefill_storage for b in g.values())
        )
        logger.info(
            "GGUF expert cache: %d/%d experts per layer on GPU (%d spare), "
            "%d host slots; %.2f GiB GPU slots, %.2f GiB GPU %s prefill banks, "
            "%.2f GiB pinned host per rank",
            self.resident0,
            num_experts,
            spare,
            self.host_slots,
            self.arena.numel() / 2**30,
            prefill_bytes / 2**30,
            prefill_mode,
            host_bytes / 2**30,
        )

    # -- loading ---------------------------------------------------------
    def bank(self, layer: int, shard: str) -> torch.Tensor:
        return self.banks[layer, shard]

    def prefill_bank(self, buffer: int, layer: int, shard: str) -> torch.Tensor:
        rows = self.prefill_rows * self.sizes[layer, shard]
        return self.prefill_storage[buffer][shard][:rows].view(
            self.prefill_rows, *self.shapes[layer, shard]
        )

    def store(self, layer: int, shard: str, expert: int, value: torch.Tensor):
        i = self.index[layer]
        slot = int(self.slot_of[i, expert])
        if slot >= 0:
            self.banks[layer, shard][slot].copy_(value)
        else:
            self.host[layer, shard][int(self.host_slot_of[i, expert])].copy_(value)

    def store_bank(self, layer: int, shard: str, values: torch.Tensor):
        """Store every expert of one projection ([experts, rows, bytes])."""
        i = self.index[layer]
        slot, host = self.slot_of[i].long(), self.host_slot_of[i].long()
        resident = (slot >= 0).nonzero().flatten()
        bank = self.banks[layer, shard]
        if resident.numel():
            # One copy of the whole (pinned) bank, then a device gather.
            staged = values.to(self.device, non_blocking=True)
            bank[slot[resident].to(self.device)] = staged[resident.to(self.device)]
            del staged
        cold = (host >= 0).nonzero().flatten()
        if cold.numel():
            self.host[layer, shard][host[cold]] = values[cold]
        torch.cuda.current_stream(self.device).synchronize()

    def start(self):
        if self.lane is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Expert cache must run eagerly before graph capture")
        torch.accelerator.synchronize(self.device)
        self.lane = _native().ExpertCacheLane(
            self.device.index,
            len(self.layers),
            self.experts,
            self.slots,
            self.host_slots,
            self.spare,
            self.slot_bytes,
            [self.banks[layer, s].data_ptr() for layer in self.layers for s in SHARDS],
            [self.host[layer, s].data_ptr() for layer in self.layers for s in SHARDS],
            self.ring.data_ptr(),
            _RING_ENTRIES,
            self.ring.shape[1],
            self.published.data_ptr(),
            self.reply.data_ptr(),
            _MAX_OPS,
            self.slot_of,
            self.host_slot_of,
        )
        # High priority: prefill transcode CTAs are dispatched ahead of the
        # model stream's pending CTAs, so the next layer's bank is ready in
        # time (the GPU waits on it otherwise).
        self.copy_stream = torch.cuda.Stream(self.device, priority=-1)
        if self.tm is not None:
            self.tm.start()
        if envs.VLLM_EXPERT_CACHE_STACKS:
            import faulthandler
            import signal

            faulthandler.register(signal.SIGUSR2, all_threads=True)
        threading.Thread(
            target=self._report, name="expert-cache-report", daemon=True
        ).start()
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.done = [torch.cuda.Event(), torch.cuda.Event()]
        for event in self.done:
            event.record()

    def _report(self, period_s: float = 30.0):
        last: dict[str, int] = {}
        while self.lane is not None:
            time.sleep(period_s)
            stats = self.stats()
            if not stats or stats == last:
                continue
            delta = {k: v - last.get(k, 0) for k, v in stats.items()}
            last = stats
            requests = max(delta["requests"], 1)
            if self.tm is not None and int(self.tm.error_host[0]):
                stats["errors"] += 1
                logger.error(
                    "GGUF expert cache: TurboMind prefill transcode "
                    "met coefficients outside FP16"
                )
            log = logger.error if stats["errors"] else logger.info
            log(
                "GGUF expert cache: %d entries, %d miss requests (%.1f us mean, "
                "%.1f us max), %d loaded, %d evicted, %d written back, "
                "%d prefill banks, skipped %d, errors %d, published %d, consumed %d",
                delta["entries"],
                delta["requests"],
                delta["serve_ns"] / requests / 1e3,
                stats["max_serve_ns"] / 1e3,
                delta["loaded"],
                delta["evicted"],
                delta["writebacks"],
                delta["prefills"],
                stats["skipped"],
                stats["errors"],
                stats["published"],
                stats["consumed"],
            )

    def close(self):
        if self.lane is not None:
            self.lane.stop()
            self.lane = None
        if self.host_arena is not None:
            _native().host_unregister(self.host_arena.data_ptr())
            self.host_arena = None

    def stats(self) -> dict[str, int]:
        names = (
            "entries",
            "requests",
            "loaded",
            "evicted",
            "writebacks",
            "serve_ns",
            "max_serve_ns",
            "skipped",
            "errors",
            "prefills",
            "published",
            "consumed",
        )
        return dict(zip(names, self.lane.stats())) if self.lane else {}

    # -- decode ----------------------------------------------------------
    def resolve(self, layer: int, ids: torch.Tensor) -> torch.Tensor:
        """Slot ids for ``ids`` once every routed expert is resident."""
        self.start()
        i = self.index[layer]
        count = ids.numel()
        if count > min(self.resident0, self.ring.shape[1] - _ENTRY_HEADER):
            # One request must fit an entry and the evictable slots.
            raise ValueError(
                f"Expert cache cannot resolve {count} routed ids "
                f"with {self.resident0} resident slots"
            )
        block = max(triton.next_power_of_2(count), 32)
        _lookup_publish[(1,)](
            ids,
            self.table,
            self.counter,
            self.ring_view,
            self.published_view,
            self.reply_view,
            self.plan,
            i,
            count,
            EXPERTS=self.experts,
            RING=_RING_ENTRIES,
            ENTRY=self.ring.shape[1],
            BLOCK=block,
            PLAN=triton.next_power_of_2(5 * _MAX_OPS),
        )
        _apply_plan[_COPY_PROGRAMS](
            self.plan,
            self.arena_words,
            self.host_view,
            self.device_offsets,
            self.host_offsets,
            self.words,
            i,
            OPS_PROGRAMS=_COPY_PROGRAMS[0],
            SPLIT=_COPY_PROGRAMS[1],
            BLOCK=1024,
            num_warps=8,
        )
        out = torch.empty_like(ids)
        _remap[(1,)](
            ids,
            self.table,
            self.plan,
            out,
            i,
            count,
            EXPERTS=self.experts,
            BLOCK=block,
            OPS=_MAX_OPS,
        )
        return out

    # -- prefill ---------------------------------------------------------
    def prefill_moe(
        self, layer: int, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Routed-expert output of a prefill batch on TurboMind banks."""
        assert self.tm is not None
        return self.tm.forward(self.index[layer], x, ids, weights)

    def _issue(self, i: int):
        buffer = i % 2
        stream = self.copy_stream
        # Decode steps change slots, the table and host copies on the model
        # stream; everything they submitted precedes this copy.
        stream.wait_stream(torch.cuda.current_stream())
        stream.wait_event(self.done[buffer])
        group = self.prefill_storage[buffer]
        self.lane.prefill(
            i,
            [group[s].data_ptr() for s in SHARDS],
            self.prefill_rows,
            self.prefill_maps[buffer].data_ptr(),
            stream.cuda_stream,
        )
        self.ready[buffer].record(stream)
        self.prefetched[buffer] = i

    def prefill(self, layer: int, ids: torch.Tensor):
        """(w1, w3, w2, ids) of the complete bank of ``layer``."""
        self.start()
        i = self.index[layer]
        buffer = i % 2
        if self.prefetched.get(buffer) != i:
            self._issue(i)
        self.prefetched.pop(buffer, None)
        torch.cuda.current_stream().wait_event(self.ready[buffer])
        if i + 1 < len(self.layers):
            self._issue(i + 1)
        mapped = torch.empty_like(ids)
        count = ids.numel()
        _map_ids[(triton.cdiv(count, 1024),)](
            ids, self.prefill_maps[buffer], mapped, count, BLOCK=1024
        )
        layer_banks = [self.prefill_bank(buffer, layer, s) for s in SHARDS]
        return (*layer_banks, mapped)

    def prefill_done(self, layer: int):
        self.done[self.index[layer] % 2].record()


@triton.jit
def _unroute(
    down, weights, inverse, out, hidden, TOPK: tl.constexpr, BLOCK: tl.constexpr
):
    """out[t] = sum_j weights[t, j] * down[inverse[t * TOPK + j]] (FP32)."""
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        row = tl.load(inverse + token * TOPK + j).to(tl.int64)
        w = tl.load(weights + token * TOPK + j).to(tl.float32)
        value = tl.load(down + row * hidden + cols, mask=mask, other=0.0)
        acc += w * value.to(tl.float32)
    tl.store(
        out + token.to(tl.int64) * hidden + cols,
        acc.to(out.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _locate(residency, bases, sources, lists, counts, experts, BLOCK: tl.constexpr):
    """Per-expert transcode sources and resident / cold work lists.

    residency: [2, E] (slot or -1, host slot); bases: [3, 3] per shard
    (slot bank, host arena, bytes per expert). sources: [3, E] addresses;
    lists[0, :counts[0]] resident experts, lists[1, :counts[1]] cold ones.
    """
    e = tl.arange(0, BLOCK)
    mask = e < experts
    slot = tl.load(residency + e, mask=mask, other=-1)
    host = tl.load(residency + experts + e, mask=mask, other=0)
    resident = slot >= 0
    where = tl.where(resident, slot, host).to(tl.int64)
    for s in tl.static_range(3):
        device_base = tl.load(bases + s * 3)
        host_base = tl.load(bases + s * 3 + 1)
        size = tl.load(bases + s * 3 + 2)
        source = tl.where(resident, device_base, host_base) + where * size
        tl.store(sources + s * experts + e, source, mask=mask)
    hot = (resident & mask).to(tl.int32)
    cold = ((~resident) & mask).to(tl.int32)
    tl.store(lists + tl.cumsum(hot, 0) - 1, e, mask=hot != 0)
    tl.store(lists + experts + tl.cumsum(cold, 0) - 1, e, mask=cold != 0)
    tl.store(counts, tl.sum(hot, 0))
    tl.store(counts + 1, tl.sum(cold, 0))


@triton.jit
def _route_rows(
    order,
    rows,
    inverse,
    gather,
    outer,
    routes,
    t0,
    TOPK: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """rows[p] = token of sorted route p; inverse[order[p]] = p.

    GATHER: gather[p] = outer[t0 * TOPK + order[p]], the position of the
    route in an enclosing routing's sorted order (outer is its inverse).
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < routes
    o = tl.load(order + p, mask=mask, other=0)
    tl.store(rows + p, (o // TOPK + t0).to(tl.int32), mask=mask)
    tl.store(inverse + o, p.to(tl.int32), mask=mask)
    if GATHER:
        tl.store(gather + p, tl.load(outer + t0 * TOPK + o, mask=mask), mask=mask)


@triton.jit
def _sub_bounds(offsets, rows, start, count, t0, t1, BLOCK: tl.constexpr):
    """Per expert: the routes of tokens [t0, t1) within its segment of an
    all-token routing (tokens ascend within a segment): first and count."""
    e = tl.program_id(0)
    lo = tl.load(offsets + e)
    hi = tl.load(offsets + e + 1)
    below0 = 0
    below1 = 0
    for base in range(lo, hi, BLOCK):
        p = base + tl.arange(0, BLOCK)
        r = tl.load(rows + p, mask=p < hi, other=2147483647)
        below0 += tl.sum((r < t0).to(tl.int32), 0)
        below1 += tl.sum((r < t1).to(tl.int32), 0)
    tl.store(start + e, lo + below0)
    tl.store(count + e, below1 - below0)


@triton.jit
def _sub_fill(
    start, count, sub_offsets, order, gather, inverse, base_route, BLOCK: tl.constexpr
):
    """gather[q] = all-token position of sub route q; inverse[local flat
    route index] = q (the sub routing's sorted position)."""
    e = tl.program_id(0)
    first = tl.load(start + e)
    n = tl.load(count + e)
    out = tl.load(sub_offsets + e)
    for i in range(0, n, BLOCK):
        k = i + tl.arange(0, BLOCK)
        mask = k < n
        p = first + k
        q = out + k
        tl.store(gather + q, p.to(tl.int32), mask=mask)
        flat = tl.load(order + p, mask=mask, other=0)
        tl.store(inverse + (flat - base_route), q.to(tl.int32), mask=mask)


_LATTICE = (18, 21, 22)
_LUT4 = (20, 23)
_AFFINE = (42,)


def _canonical(gguf_type: int, n: int, k: int):
    """(family, decoder, group, weight bytes, stats bytes) of one expert."""
    if gguf_type in _LATTICE:
        group = 16 if gguf_type == 22 else 32
        stat = 8 if gguf_type in (18, 21) else 4
        return 0, gguf_type, group, n * k // 4, k // group * n * stat
    if gguf_type in _LUT4:
        return 1, 0, 32, n * k // 2, k // 32 * n * 2
    if gguf_type in _AFFINE:
        return 2, 2, 32, n * k // 4, k // 32 * n * 4
    raise ValueError(f"No TurboMind prefill bank for GGUF type {gguf_type}")


class TurboMindPrefill:
    """Prefill MoE on TurboMind grouped GEMMs over canonical banks.

    Experts stay in their original encoding (GPU slots + pinned host arena).
    Per layer and prefill chunk, a fused kernel transcodes all experts into
    canonical banks: resident experts from the slot banks with many CTAs,
    cold experts from a device staging area that the copy engine fills from
    the host arena. (Transcoding them straight from mapped host memory, even
    with 8 CTAs, slowed the concurrent model-stream kernels up to 2x: HC
    down 0.59 -> 1.2 ms.) Gate and up are interleaved into one bank whose
    GEMM epilogue applies SiLU.

    SLOTS banks per projection, by layer: as soon as layer l's gate|up GEMMs
    are queued, layer l + SLOTS's gate|up transcode is queued behind them on
    side streams (likewise for down). With 2 slots the cold PCIe stream runs
    a full layer ahead, but the second bank set (~0.33 GB per GPU) does not
    fit next to today's prefill activation peak on 16 GB cards: the
    allocator then thrashes (32K TTFT 8.6 s -> 13+ s).
    """

    # Cold experts: staged by DMA (default) or read zero-copy with few CTAs.
    COLD_DMA = envs.VLLM_TM_COLD_DMA
    COLD_CTAS = envs.VLLM_TM_COLD_CTAS or (64 if COLD_DMA else 8)
    KEYS = ("gu", "dn")
    SLOTS = 1
    VIEWS = 4

    def __init__(self, cache: "ExpertCache"):
        self.cache = cache
        experts, device = cache.experts, cache.device
        self.layer = {}
        sizes = {"gu": 0, "dn": 0}
        for layer in cache.layers:
            gate, up, down = (cache.sources[layer, s] for s in SHARDS)
            if gate[0] != up[0] or gate[1:] != up[1:]:
                raise ValueError("TurboMind prefill needs matching gate/up banks")
            rows = cache.shapes[layer, "w1"][0]
            hidden = cache.shapes[layer, "w2"][0]
            gu = _canonical(gate[0], 2 * rows, gate[1])
            dn = _canonical(down[0], hidden, down[1])
            self.layer[layer] = {
                "gate": gate,
                "down": down,
                "rows": rows,
                "hidden": hidden,
                "gu": gu,
                "dn": dn,
            }
            for key, spec in (("gu", gu), ("dn", dn)):
                sizes[key] = max(sizes[key], experts * (spec[3] + spec[4]))
        # Canonical banks per projection and slot, sized for the largest
        # layer; a layer's stats follow its weights (formats split bytes
        # differently).
        self.storage = {
            key: [
                torch.empty(size, dtype=torch.uint8, device=device)
                for _ in range(self.SLOTS)
            ]
            for key, size in sizes.items()
        }
        # Per layer: TurboMind StridedPtr tables (pointer, ld).
        self.tables = {}
        ids = torch.arange(experts, dtype=torch.int64, device=device)
        for i, (layer, info) in enumerate(self.layer.items()):
            gk, dk = info["gate"][1], info["down"][1]
            tables = {}
            for key, ld_w, ld_s in (
                ("gu", 32 * gk, 2 * info["rows"]),
                ("dn", 32 * dk, info["hidden"]),
            ):
                spec = info[key]
                weight, stats = self.bank(key, spec, i % self.SLOTS)
                pair = []
                for base, stride, ld in (
                    (weight, spec[3], ld_w),
                    (stats, spec[4], ld_s),
                ):
                    table = torch.empty(experts, 2, dtype=torch.int64, device=device)
                    table[:, 0] = base.data_ptr() + ids * stride
                    table[:, 1] = ld
                    pair.append(table.view(torch.uint8).flatten())
                tables[key] = pair
            self.tables[layer] = tables
        # Residency-derived transcode inputs, in a ring by layer: layers
        # l + 1 and l + 2 are located while layer l's down transcode may
        # still read layer l's.
        ring = self.VIEWS
        self.residency = torch.zeros(
            (ring, 2, experts), dtype=torch.int32, device=device
        )
        self.sources = torch.zeros((ring, 3, experts), dtype=torch.int64, device=device)
        self.lists = torch.zeros((ring, 2, experts), dtype=torch.int32, device=device)
        self.counts = torch.zeros((ring, 2), dtype=torch.int32, device=device)
        # Views per ring slot, built once: the per-layer path stays
        # launch-only.
        self.views = [
            {
                "residency": self.residency[p],
                "sources": self.sources[p],
                "lists": self.lists[p],
                "counts": self.counts[p],
                "src": [self.sources[p, s] for s in range(3)],
                "list": [self.lists[p, w] for w in range(2)],
                "count": [self.counts[p, w : w + 1] for w in range(2)],
            }
            for p in range(ring)
        ]
        self.expert_ids = torch.arange(experts + 1, dtype=torch.int32, device=device)
        self.error = torch.zeros(1, dtype=torch.int32, device=device)
        self.error_host = torch.zeros(1, dtype=torch.int32).pin_memory()
        self.prefetched: dict[tuple[str, int], bool] = {}
        self.tuned: set[Any] = set()
        # Cold-expert staging, per shard sized for the largest layer's host
        # slots; [0, extent) host slots are copied per layer.
        self.stage = None
        self.extent = [0] * ring
        if self.COLD_DMA:
            spans = [
                max(
                    cache.host_slots * cache.sizes[layer, shard]
                    for layer in cache.layers
                )
                for shard in SHARDS
            ]
            self.stage = torch.empty(sum(spans), dtype=torch.uint8, device=device)
            self.stage_at = [
                self.stage.data_ptr() + sum(spans[:s]) for s in range(len(SHARDS))
            ]

    def bank(self, key: str, spec, slot: int):
        """(weights, stats) views of a projection bank for one layer format."""
        buffer = self.storage[key][slot]
        split = self.cache.experts * spec[3]
        return buffer[:split], buffer[split : split + self.cache.experts * spec[4]]

    def storage_bytes(self) -> int:
        return sum(t.numel() for ts in self.storage.values() for t in ts) + (
            self.stage.numel() if self.stage is not None else 0
        )

    def bind_host(self):
        """Device addresses of every (layer, shard) slot bank / host arena."""
        cache = self.cache
        host_base = cache.host_view.data_ptr()
        arena_base = cache.host_arena.data_ptr()
        rows = []
        for layer in cache.layers:
            for shard in SHARDS:
                host = (
                    self.stage_at[SHARDS.index(shard)]
                    if self.COLD_DMA
                    else host_base + cache.host[layer, shard].data_ptr() - arena_base
                )
                rows.append(
                    (
                        cache.banks[layer, shard].data_ptr(),
                        host,
                        cache.sizes[layer, shard],
                    )
                )
        self.bases = torch.tensor(rows, dtype=torch.int64, device=cache.device)
        self.layer_bases = [
            self.bases[i * 3 : i * 3 + 3] for i in range(len(cache.layers))
        ]

    def start(self):
        self.resident_stream = self.cache.copy_stream
        self.cold_stream = torch.cuda.Stream(self.cache.device, priority=-1)
        pairs = [(key, slot) for key in self.KEYS for slot in range(self.SLOTS)]
        self.ready = {pair: torch.cuda.Event() for pair in pairs}
        self.cold_done = {pair: torch.cuda.Event() for pair in pairs}
        self.done = {pair: torch.cuda.Event() for pair in pairs}
        for event in self.done.values():
            event.record()
        self.fresh = torch.cuda.Event()
        self.fresh.record()

    # -- transcode ------------------------------------------------------
    def _locate(self, i: int):
        """Snapshot layer i's residency and derive each expert's source."""
        cache, view = self.cache, self.views[i % self.VIEWS]
        stream = self.resident_stream
        with torch.cuda.stream(stream):
            self.extent[i % self.VIEWS] = cache.lane.residency(
                i, view["residency"].data_ptr(), stream.cuda_stream
            )
            experts = cache.experts
            _locate[(1,)](
                view["residency"],
                self.layer_bases[i],
                view["sources"],
                view["lists"],
                view["counts"],
                experts,
                BLOCK=triton.next_power_of_2(experts),
            )

    def _issue(self, key: str, i: int):
        """Queue layer i's transcode into the `key` bank."""
        cache = self.cache
        layer = cache.layers[i]
        info, p = self.layer[layer], i % self.VIEWS
        slot = (key, i % self.SLOTS)
        main = torch.cuda.current_stream()
        stream, cold = self.resident_stream, self.cold_stream
        # Decode steps changed slots, tables and host copies on the model
        # stream before this point; the bank's previous GEMMs must finish.
        # Layer i's views are rewritten by layer i + VIEWS's locate, queued
        # on the resident stream after layer i's down (which waits for its
        # cold part below).
        stream.wait_stream(main)
        stream.wait_event(self.done[slot])
        if key == "gu":
            self._locate(i)
            s0, s1, source, n_out, shape = (
                0,
                1,
                info["gate"],
                2 * info["rows"],
                cache.shapes[layer, "w1"],
            )
        else:
            s0, s1, source, n_out, shape = (
                2,
                None,
                info["down"],
                info["hidden"],
                cache.shapes[layer, "w2"],
            )
        if self.COLD_DMA:
            # The copy needs only the host arena as of this prefill step
            # (decode write-backs precede it) and the previous layer's cold
            # transcode (stream order), not the bank: it runs ahead.
            cold.wait_event(self.fresh)
            rows = self.extent[p]
            for s in (0, 1) if key == "gu" else (2,):
                shard = SHARDS[s]
                nbytes = rows * cache.sizes[layer, shard]
                if nbytes:
                    _native().copy_h2d(
                        self.stage_at[s],
                        cache.host[layer, shard].data_ptr(),
                        nbytes,
                        cold.cuda_stream,
                    )
        cold.wait_stream(stream)
        spec = info[key]
        weight, stats = self.bank(key, spec, slot[1])
        view = self.views[p]
        for which, ctas, target in ((0, 1024, stream), (1, self.COLD_CTAS, cold)):
            with torch.cuda.stream(target):
                torch.ops._C.gguf_prefill_transcode_sm70(
                    weight,
                    stats,
                    spec[3],
                    spec[4],
                    view["src"][s0],
                    None if s1 is None else view["src"][s1],
                    view["list"][which],
                    view["count"][which],
                    ctas,
                    source[0],
                    n_out,
                    source[1],
                    source[2],
                    shape[1],
                    self.error,
                )
        self.cold_done[slot].record(cold)
        stream.wait_event(self.cold_done[slot])
        if key == "dn":
            with torch.cuda.stream(stream):
                self.error_host.copy_(self.error, non_blocking=True)
        self.ready[slot].record(stream)
        self.prefetched[key, i] = True

    def _prefetch(self, key: str, i: int):
        if i < len(self.cache.layers) and (key, i) not in self.prefetched:
            self._issue(key, i)

    def _acquire(self, key: str, i: int):
        # In layer order: i .. i + SLOTS - 1, one per slot (already queued
        # unless this chunk starts here). A chunk starts at layer 0, so any
        # entry left then is from an interrupted pass.
        if i == 0:
            self.fresh.record()
            for stale in [k for k in self.prefetched if k[0] == key]:
                del self.prefetched[stale]
        for j in range(i, i + self.SLOTS):
            self._prefetch(key, j)
        self.prefetched.pop((key, i), None)
        torch.cuda.current_stream().wait_event(self.ready[key, i % self.SLOTS])

    def _release(self, key: str, i: int):
        self.done[key, i % self.SLOTS].record()
        self._prefetch(key, i + self.SLOTS)

    # -- compute --------------------------------------------------------
    def forward(
        self, i: int, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        cache = self.cache
        cache.start()
        layer = cache.layers[i]
        info = self.layer[layer]
        tables = self.tables[layer]
        experts = cache.experts
        tokens, top_k = ids.shape
        out = torch.empty_like(x)
        weights = weights.float().contiguous()
        step = cache.prefill_tokens
        self._acquire("gu", i)
        self._tune(layer, info, tables, x)
        if _MEMLOG and i in (0, len(cache.layers) - 1):
            if i == 0:
                torch.accelerator.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            reserved = torch.accelerator.memory_reserved()
            logger.info(
                "TurboMind prefill layer %d: %d tokens, allocated "
                "%.3f GiB, peak since layer 0 %.3f GiB, reserved %.3f "
                "GiB, device used %.3f GiB (non-torch %.3f GiB)",
                i,
                x.shape[0],
                torch.accelerator.memory_allocated() / 2**30,
                torch.accelerator.max_memory_allocated() / 2**30,
                reserved / 2**30,
                (total - free) / 2**30,
                (total - free - reserved) / 2**30,
            )
        # Gate|up over the whole chunk (its SiLU output is small): experts
        # get full-size row groups. Then the bank is released.
        route = self._route(ids, 0, experts)
        hidden = self._gate_up(info, tables, x, route, False)
        self._release("gu", i)
        self._acquire("dn", i)
        # Down's output is wide: token sub-batches, each reading its routes'
        # gate|up rows from the chunk-wide order.
        if tokens <= step:
            self._down(info, tables, hidden, route, weights, out, False)
        else:
            for t0 in range(0, tokens, step):
                t1 = min(tokens, t0 + step)
                sub = self._subroute(route, t0, t1)
                self._down(info, tables, hidden, sub, weights[t0:t1], out[t0:t1], False)
        self._release("dn", i)
        return out

    def _route(self, ids, t0, experts, outer=None):
        """Sort ids' routes by expert. With `outer` (the inverse of a routing
        of all tokens), also return each route's row in that routing."""
        top_k = ids.shape[1]
        flat = ids.reshape(-1)
        sorted_ids, order = flat.sort(stable=True)
        offsets = torch.searchsorted(
            sorted_ids, self.expert_ids.to(sorted_ids.dtype), out_int32=True
        )
        routes = order.numel()
        rows = torch.empty(routes, dtype=torch.int32, device=ids.device)
        inverse = torch.empty(routes, dtype=torch.int32, device=ids.device)
        gather = (
            torch.empty(routes, dtype=torch.int32, device=ids.device)
            if outer is not None
            else None
        )
        block = 1024
        _route_rows[(triton.cdiv(routes, block),)](
            order,
            rows,
            inverse,
            gather if gather is not None else rows,
            outer if outer is not None else inverse,
            routes,
            t0,
            TOPK=top_k,
            GATHER=outer is not None,
            BLOCK=block,
        )
        return offsets, order, rows, top_k, inverse, gather

    def _subroute(self, route, t0, t1):
        """The routing of tokens [t0, t1) derived from the all-token routing
        `route` without sorting: the same routes in the same order as sorting
        them alone, plus each route's row in the all-token order."""
        offsets, order, rows, top_k, _, _ = route
        experts = offsets.numel() - 1
        device = offsets.device
        start = torch.empty(experts, dtype=torch.int32, device=device)
        count = torch.empty(experts, dtype=torch.int32, device=device)
        _sub_bounds[(experts,)](offsets, rows, start, count, t0, t1, BLOCK=256)
        sub_offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
        torch.cumsum(count, 0, out=sub_offsets[1:])
        routes = (t1 - t0) * top_k
        gather = torch.empty(routes, dtype=torch.int32, device=device)
        inverse = torch.empty(routes, dtype=torch.int32, device=device)
        _sub_fill[(experts,)](
            start, count, sub_offsets, order, gather, inverse, t0 * top_k, BLOCK=128
        )
        return sub_offsets, inverse, None, top_k, inverse, gather

    def _gate_up(self, info, tables, x, route, tune):
        offsets, _, rows, _, _, _ = route
        gu = info["gu"]
        hidden = torch.empty(
            rows.numel(), info["rows"], dtype=torch.float16, device=x.device
        )
        torch.ops._C.gguf_grouped_prefill_gemm_sm70_out(
            hidden,
            x,
            offsets,
            *tables["gu"],
            gu[0],
            gu[1],
            2 * info["rows"],
            self.cache.experts,
            gu[2],
            True,
            tune,
            rows,
        )
        return hidden

    def _down(self, info, tables, hidden, route, weights, out, tune):
        offsets, order, _, top_k, inverse, gather = route
        dn = info["dn"]
        routes = order.numel()
        down = torch.empty(
            routes, info["hidden"], dtype=torch.float16, device=hidden.device
        )
        torch.ops._C.gguf_grouped_prefill_gemm_sm70_out(
            down,
            hidden,
            offsets,
            *tables["dn"],
            dn[0],
            dn[1],
            info["hidden"],
            self.cache.experts,
            dn[2],
            False,
            tune,
            gather,
        )
        if out is None:
            return
        block = 1024
        _unroute[(out.shape[0], triton.cdiv(info["hidden"], block))](
            down, weights, inverse, out, info["hidden"], TOPK=top_k, BLOCK=block
        )

    def _tune(self, layer, info, tables, x):
        """Measure each GEMM descriptor family once, at the prefill sizes.

        Called with this layer's gate|up bank ready; the down bank is only
        written after it, so down is measured on whatever it holds (timing
        does not depend on the weight values).
        """
        key = (info["gate"][0], info["down"][0], info["rows"], info["hidden"])
        if key in self.tuned or torch.cuda.is_current_stream_capturing():
            return
        self.tuned.add(key)
        step, top_k = self.cache.prefill_tokens, 10
        experts = self.cache.experts
        generator = torch.Generator(device=x.device).manual_seed(0)
        chunk = max(step, x.shape[0])
        up_sizes = sorted({max(512, chunk >> s) for s in range(5)}, reverse=True)
        down_sizes = sorted({max(512, step >> s) for s in range(4)}, reverse=True)
        probe = torch.zeros(chunk, x.shape[1], dtype=x.dtype, device=x.device)
        for tokens in up_sizes:
            ids = (
                torch.randn(tokens, experts, device=x.device, generator=generator)
                .topk(top_k, dim=1)
                .indices
            )
            route = self._route(ids, 0, experts)
            hidden = self._gate_up(info, tables, probe, route, True)
            del hidden
        ids = (
            torch.randn(step, experts, device=x.device, generator=generator)
            .topk(top_k, dim=1)
            .indices
        )
        route = self._route(ids, 0, experts)
        hidden = self._gate_up(info, tables, probe, route, False)
        for tokens in down_sizes:
            sub = self._route(ids[:tokens], 0, experts, outer=route[4])
            self._down(info, tables, hidden, sub, None, None, True)
        logger.info(
            "Tuned TurboMind prefill GEMMs for gate/up type %d at %s "
            "tokens, down type %d at %s tokens",
            key[0],
            up_sizes,
            key[1],
            down_sizes,
        )


_CACHE: ExpertCache | None = None


def current_cache() -> ExpertCache | None:
    return _CACHE


def create_cache(banks, num_experts, config, device) -> ExpertCache:
    global _CACHE
    if _CACHE is not None:
        _CACHE.close()
    _CACHE = ExpertCache(
        banks,
        num_experts,
        config.expert_cache_fraction,
        config.expert_cache_spare_slots,
        config.expert_cache_host_slack,
        config.expert_cache_profile,
        device,
        prefill_mode=config.expert_cache_prefill,
        prefill_tokens=config.expert_cache_prefill_tokens,
    )
    return _CACHE
