# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared original expert storage and graph-safe TP4 decode primitives.

The shared_original loader selects these building blocks for the V4.1 profile.
The owner must outlive all rank registrations and captured graphs. They hold
raw GGUF bytes and bounded copy plans, never a full dequantized expert bank.
"""

import hashlib
import os
import socket
import struct
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.utils.shared_host import SharedHostArena

from .gguf_expert_cache import _native
from .gguf_shared_copy import swap_original_experts, swap_original_experts3
from .gguf_shared_layout import SharedExpertLayout
from .gguf_shared_protocol import (
    commit_local_routes,
    commit_shared_routes,
    publish_hints,
    publish_round,
    publish_shared_routes,
    resolve_resident_routes,
    stash_plan,
    substitute_dropped,
)

SHARDS = ("w1", "w3", "w2")

# Fresh-text decode defaults (release 1, lossless): free-slot swap, exact
# truncation, 2 empty slots per layer with background eviction. The lossy
# cold-route drop stays opt-in (VLLM_V41_SKIP_BELOW=0.2: ~+23% tok/s, about -2
# points on the mixed-240 accuracy set). The native planner reads these too,
# so they are set before any coordinator.
os.environ.setdefault("VLLM_V41_SWAP_FREE", "1")
os.environ.setdefault("VLLM_V41_TRUNC", "1")
os.environ.setdefault("VLLM_V41_ASYNC_EVICT", "2")
os.environ.setdefault("VLLM_V41_SKIP_BELOW", "0")


def triton_next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


@dataclass(frozen=True)
class SharedExpertLayerSpec:
    hidden: int
    intermediate: int
    experts: int
    resident: int
    kinds: tuple[int, int, int]

    def __post_init__(self):
        object.__setattr__(self, "kinds", tuple(self.kinds))
        if not 0 < self.resident < self.experts or len(self.kinds) != 3:
            raise ValueError("Invalid shared expert layer specification")
        for rank in range(4):
            self.layouts(rank)

    def layouts(self, rank):
        return tuple(
            SharedExpertLayout.for_type(
                self.hidden, self.intermediate, 4, rank, role, kind
            )
            for role, kind in zip(SHARDS, self.kinds)
        )

    @property
    def projection_bytes(self):
        return tuple(
            layout.source_rows * layout.source_row_bytes for layout in self.layouts(0)
        )

    @property
    def expert_bytes(self):
        return sum(self.projection_bytes)


def _spec_fingerprint(specs):
    digest = hashlib.sha256()
    for spec in specs:
        digest.update(
            struct.pack(
                "!7Q",
                spec.hidden,
                spec.intermediate,
                spec.experts,
                spec.resident,
                *spec.kinds,
            )
        )
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


class SharedExpertOwner:
    """Exactly one cold allocation and one native coordinator per TP group."""

    def __init__(
        self,
        specs: tuple[SharedExpertLayerSpec, ...],
        exchange_bytes=512 * 1024**2,
        max_routes=36,
        timeout_ns=60_000_000_000,
    ):
        self.specs = tuple(specs)
        if not specs:
            raise ValueError("Shared expert pool requires layers")
        if (envs.VLLM_V41_SWAP_FREE or "0") == "1":
            per_class = int(envs.VLLM_V41_FREE_SLOTS or "12")
            empty = int(envs.VLLM_V41_ASYNC_EVICT or "0")
            # As the native planner carves it: per size class, per_class plus
            # one host allocation per kept-empty GPU slot of each layer.
            exchange_bytes = max(
                exchange_bytes,
                per_class * sum({s.expert_bytes for s in specs})
                + empty * sum(s.expert_bytes for s in specs),
            )
        self.data = SharedHostArena.create(
            sum((s.experts - s.resident) * s.expert_bytes for s in specs)
            + exchange_bytes
        )
        self.control = None
        self.coordinator = None
        try:
            native = _native()
            self.control = SharedHostArena.create(
                native.shared_expert_hit_layout()[4] * 8, "vllm-expert-control"
            )
            self.coordinator = native.SharedExpertCoordinator(
                self.data.address,
                self.data.size,
                self.control.address,
                [
                    native.SharedExpertLayer(s.expert_bytes, s.experts, s.resident)
                    for s in specs
                ],
                4,
                max_routes,
                exchange_bytes,
                timeout_ns,
            )
            struct.pack_into("=q", self.control.buffer, 7 * 8, _spec_fingerprint(specs))
        except BaseException:
            self.close()
            raise

    def send(self, connection: socket.socket):
        if self.control is None:
            raise RuntimeError("Shared expert owner is closed")
        self.data.send(connection)
        self.control.send(connection)

    def close(self):
        if self.coordinator is not None:
            self.coordinator.stop()
            self.coordinator = None
        if self.control is not None:
            self.control.close()
            self.control = None
        self.data.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_INPLACE_RANKS: Any = []
_ALL_RANKS: Any = []


def set_trunc_cutoff(cutoff: int):
    """Per step: verify positions >= cutoff never wait for a cold expert
    (0 disables). Every rank of the group writes the same value."""
    for rank in _ALL_RANKS:
        if rank.trunc_enabled and rank.control_host is not None:
            rank.control_host[5] = int(cutoff)


def stash_router(logits, bias, scale):
    """Decode only: keep this layer's router logits and bias for the
    post-commit substitution of dropped cold routes (VLLM_V41_SKIP_SUBST)."""
    if logits.shape[0] > 8:
        return
    for rank in _ALL_RANKS:
        if rank.device == logits.device and rank.router_logits is not None:
            if logits.shape[1] != rank.router_logits.shape[1]:
                return
            rank.router_logits[: logits.shape[0]].copy_(logits)
            rank.router_bias.copy_(bias)
            rank.router_scale = float(scale)
            rank.router_ready = True
            return


def publish_prefetch_hints(gates, layer, inputs):
    """Decode only: post this rank's prefetch hints for layers layer+1 and
    layer+2 (gates[j] = (weight, bias) of layer j's router)."""
    if inputs.shape[0] > 8:
        return
    for rank in _ALL_RANKS:
        if rank.prefetch and rank.device == inputs.device:
            rank.hint(gates, layer, inputs)
            return


def current_trunc(device):
    """Device tensor: first verify position not fully computed this step."""
    for rank in _ALL_RANKS:
        if rank.trunc_enabled and rank.device == device:
            return rank.trunc
    return None


_CAPTURE_PATCHED = False


def _thread_local_graph_capture():
    """The writeback lane issues copies from its own thread while the model
    captures CUDA graphs. In torch's default global capture mode, those copies
    block until the capture ends, while the capture waits on GPU work that
    waits on the lane. thread_local restricts only the capturing thread."""
    global _CAPTURE_PATCHED
    if _CAPTURE_PATCHED:
        return
    original = torch.cuda.CUDAGraph.capture_begin

    def capture_begin(self, pool=None, capture_error_mode="global"):
        if capture_error_mode == "global":
            capture_error_mode = "thread_local"
        return original(self, pool=pool, capture_error_mode=capture_error_mode)

    torch.cuda.CUDAGraph.capture_begin = capture_begin
    _CAPTURE_PATCHED = True


def join_writebacks():
    """Join each rank's side-stream writeback into its current stream (call
    once at the end of a decode forward, inside the captured region)."""
    for rank in _INPLACE_RANKS:
        if (
            rank.swap_dma is None
            and not rank.free_swap
            and rank.device.index == torch.accelerator.current_device_index()
        ):
            torch.cuda.current_stream(rank.device).wait_stream(rank.swap_side)


class SharedExpertRank:
    """Per-rank CUDA aliases, raw GPU banks and bounded decode plan buffers.

    ``banks[(layer, shard)]`` must include only this rank's physical carrier;
    the loader owns its bank-tail guards. Call warmup before publishing routes
    so module compilation cannot queue behind a waiting graph kernel.
    """

    def __init__(
        self,
        data: SharedHostArena,
        control: SharedHostArena,
        rank: int,
        specs: tuple[SharedExpertLayerSpec, ...],
        banks: dict,
    ):
        if (
            not specs
            or rank not in range(4)
            or torch.accelerator.current_device_index() != rank
        ):
            raise ValueError("Select this TP rank's CUDA device before registration")
        self.data, self.control, self.rank = data, control, rank
        self.specs, self.banks = specs, banks
        self.geometries = [spec.layouts(rank) for spec in specs]
        self.device = torch.device("cuda", rank)
        self.experts = max(s.experts for s in specs)
        self.native = _native()
        layout = self.native.shared_expert_control_layout()
        if layout != [512, 232, 64, 8, 5, 44] or control.size < layout[0] * 8:
            raise ValueError("Unsupported shared expert control ABI")
        self.meta = dict(
            RANK=rank,
            RANK_START=layout[1],
            RANK_WORDS=layout[2],
            OP_START=layout[3],
            OP_WORDS=layout[4],
            BLOCK=64,
        )
        self.max_ops = layout[5]
        hit = self.native.shared_expert_hit_layout()
        if control.size < hit[4] * 8:
            raise ValueError("Shared expert control mapping lacks the hit log")
        self.hit_meta = dict(
            HIT_START=hit[0], HIT_HEADER=hit[1], HIT_ENTRIES=hit[2], HIT_WORDS=hit[3]
        )
        # All-resident decode layers resolve on the GPU (no CPU round trip).
        self.fast_hits = envs.VLLM_V41_EXPERT_FAST_HIT == "1"
        self._registered = []
        self.host = self.control_host = self.mapped = self.control_view = None
        self.ready = False
        try:
            self.host = torch.frombuffer(data.buffer, dtype=torch.uint8)
            self.control_host = torch.frombuffer(control.buffer, dtype=torch.int64)
            if int(self.control_host[7]) != _spec_fingerprint(specs):
                raise ValueError("Shared expert TP ranks have different layer metadata")
            self.max_routes = int(self.control_host[4])
            if not 0 < self.max_routes <= 36:
                raise ValueError("Invalid native shared route capacity")
            for tensor in (self.host, self.control_host):
                self.native.host_register(
                    tensor.data_ptr(), tensor.numel() * tensor.element_size()
                )
                self._registered.append(tensor.data_ptr())
            self.mapped = self.native.mapped_view(self.host, rank)
            self.control_view = self.native.mapped_view(self.control_host, rank)
            table = torch.full((len(specs), self.experts), -1, dtype=torch.int32)
            cold = torch.full((len(specs), self.experts), -1, dtype=torch.int64)
            cold_base = 0
            for i, spec in enumerate(specs):
                table[i, : spec.resident] = torch.arange(
                    spec.resident, dtype=torch.int32
                )
                cold[i, spec.resident : spec.experts] = (
                    cold_base
                    + torch.arange(spec.experts - spec.resident, dtype=torch.int64)
                    * spec.expert_bytes
                )
                cold_base += (spec.experts - spec.resident) * spec.expert_bytes
                for shard, geometry in zip(SHARDS, self.geometries[i]):
                    bank = banks[i, shard]
                    if (
                        bank.device != self.device
                        or bank.dtype != torch.uint8
                        or bank.ndim != 3
                        or bank.shape[0] < spec.resident
                        or bank.shape[1:]
                        != (geometry.device_rows, geometry.device_row_bytes)
                    ):
                        raise ValueError("GPU bank does not match the shared layer")
            self.table = table.to(self.device)
            self.cold_offsets = cold.to(self.device)
            self.counter = torch.ones(1, dtype=torch.int64, device=self.device)
            # Per-layer decode requests and unique misses. Each rank observes
            # the same plan; group reports use one copy of these counters.
            self.statistics = torch.zeros(
                len(specs), 2, dtype=torch.int64, device=self.device
            )
            self.slots = torch.zeros(
                self.max_ops, dtype=torch.int32, device=self.device
            )
            self.loads = torch.full(
                (self.max_ops,), -1, dtype=torch.int64, device=self.device
            )
            self.saves = torch.full_like(self.loads, -1)
            self.count = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.fast_flag = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.hit_counter = torch.zeros(1, dtype=torch.int64, device=self.device)
            self.trunc = torch.full(
                (1,), 1 << 20, dtype=torch.int32, device=self.device
            )
            self.trunc_enabled = envs.VLLM_V41_TRUNC == "1"
            self.round = torch.zeros(1, dtype=torch.int64, device=self.device)
            # Predictive prefetch hints (VLLM_V41_PREFETCH, needs async eviction).
            self.prefetch = (
                envs.VLLM_V41_PREFETCH == "1"
                and (envs.VLLM_V41_SWAP_FREE or "0") == "1"
                and int(envs.VLLM_V41_ASYNC_EVICT or "0") > 0
            )
            self.hint_counter = torch.zeros(1, dtype=torch.int64, device=self.device)
            # Lossy cold-route drop threshold on the raw gate weight (0 = off).
            self.skip_below = float(envs.VLLM_V41_SKIP_BELOW)
            # Later verify positions are accepted less often: optionally drop
            # more there (VLLM_V41_SKIP_LATE at positions >= _SKIP_LATE_FROM).
            self.skip_late = float(envs.VLLM_V41_SKIP_LATE) or self.skip_below
            self.skip_late_from = int(envs.VLLM_V41_SKIP_LATE_FROM)
            # Skip v2: never drop a token's top min_rank routes; drops visit
            # the coordinator (adopt background loads, queue new ones); still
            # dropped routes are substituted by the best resident expert and
            # each token is renormalized like the router.
            self.skip_min_rank = int(envs.VLLM_V41_SKIP_MIN_RANK)
            self.skip_defer = envs.VLLM_V41_SKIP_DEFER == "1"
            self.skip_subst = envs.VLLM_V41_SKIP_SUBST == "1"
            self.skip_renorm = envs.VLLM_V41_SKIP_RENORM == "1"
            self.router_logits = self.router_bias = None
            self.router_scale, self.router_ready = 1.5, False
            if (
                self.skip_below > 0
                and self.skip_defer
                and (self.skip_subst or self.skip_renorm)
            ):
                n = max(s.experts for s in specs)
                self.router_logits = torch.zeros(
                    8, n, dtype=torch.float32, device=self.device
                )
                self.router_bias = torch.zeros(
                    n, dtype=torch.float32, device=self.device
                )
            self.skip_mask = torch.zeros(64, dtype=torch.int32, device=self.device)
            self.miss_hist = (
                torch.zeros(32, dtype=torch.int64, device=self.device)
                if envs.VLLM_V41_MISS_HIST == "1"
                else None
            )
            self.hint_weights: Any = {}
            _ALL_RANKS.append(self)
            self.free_swap = (envs.VLLM_V41_SWAP_FREE or "0") == "1"
            self.inplace = self.free_swap or (envs.VLLM_V41_SWAP_INPLACE or "0") == "1"
            if self.inplace:
                self._init_inplace()
        except BaseException:
            self.close()
            raise

    def _init_inplace(self):
        from .gguf_v41_moe import default_v41k_library

        torch.ops.load_library(default_v41k_library())
        self.swap_layouts = []
        for spec, layouts in zip(self.specs, self.geometries):
            offsets, offset = [], 0
            for size in spec.projection_bytes:
                offsets.append(offset)
                offset += size
            lay = []
            for layout, projection in zip(layouts, offsets):
                lay += [
                    layout.source_row_bytes,
                    layout.source_row_start,
                    layout.first_block * layout.block_bytes,
                    layout.device_row_bytes,
                    (layout.write_end_block - layout.first_block) * layout.block_bytes,
                    layout.device_rows,
                    projection,
                ]
            self.swap_layouts.append(lay)
        per_op = max(torch.ops.v41k.swap_scratch_per_op(l) for l in self.swap_layouts)  # noqa: E741
        # Free mode needs no device scratch (victims go straight to host).
        self.swap_scratch = torch.empty(
            0 if self.free_swap else per_op * self.max_ops,
            dtype=torch.uint8,
            device=self.device,
        )
        self.swap_wb = torch.zeros(2, dtype=torch.int64, device=self.device)
        # publish copies the plan words for the local commit (one host read pass)
        self.stash_in_publish = envs.VLLM_V41_STASH_IN_PUBLISH == "1"
        # Async eviction (VLLM_V41_ASYNC_EVICT=<empty slots per layer>): the
        # coordinator drops LRU victims in plans; this rank's job lane saves
        # them with copy-engine DMA in the background.
        self.async_evict = self.free_swap and int(envs.VLLM_V41_ASYNC_EVICT or "0") > 0
        self.plan_evicts = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.job_lane = None
        if self.async_evict:
            banks = []
            for layer in range(len(self.specs)):
                tensors = [self.banks[layer, shard] for shard in SHARDS]
                banks.append(
                    [t.data_ptr() for t in tensors]
                    + [t.stride(0) * t.element_size() for t in tensors]
                )
            self.job_lane = self.native.SharedExpertJobLane(
                self.control_host.data_ptr(),  # type: ignore[union-attr]
                self.host.data_ptr(),  # type: ignore[union-attr]
                self.device.index,
                self.rank,
                banks,
                self.swap_layouts,
                60_000_000_000,
            )
        self.plan_incoming = torch.zeros(
            self.max_ops, dtype=torch.int64, device=self.device
        )
        self.plan_victims = torch.zeros(
            self.max_ops, dtype=torch.int64, device=self.device
        )
        self.plan_trunc = torch.full(
            (1,), 1 << 20, dtype=torch.int32, device=self.device
        )
        self.swap_done = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.swap_side = torch.cuda.Stream(device=self.device)
        # Copy-engine writeback lane (default): SM zero-copy stores throttle
        # every concurrent kernel's memory traffic.
        self.swap_dma = None
        if envs.VLLM_V41_SWAP_DMA == "1":
            _thread_local_graph_capture()
            start, words = self.meta["RANK_START"], self.meta["RANK_WORDS"]
            self.swap_dma = self.native.SharedExpertDmaLane(
                self.control_host.data_ptr(),  # type: ignore[union-attr]
                self.host.data_ptr(),  # type: ignore[union-attr]
                self.swap_scratch.data_ptr(),
                self.device.index,
                self.rank,
                4,
                self.swap_layouts,
                per_op,
                start,
                words,
                self.meta["OP_START"],
                self.meta["OP_WORDS"],
                40,
                60_000_000_000,
            )
        start, words = self.meta["RANK_START"], self.meta["RANK_WORDS"]
        self.swap_flag0, self.swap_stride = start + 40, words
        self.swap_ack = start + self.rank * words + 1
        _INPLACE_RANKS.append(self)

    def hint(self, gates, layer, inputs):
        experts = self.specs[0].experts
        quarter = experts // 4
        if layer not in self.hint_weights:
            # This rank's quarter of the next two routers, FP16 (only ranks
            # candidates; the exact FP32 router still decides routing).
            weight = torch.zeros(
                2 * quarter, inputs.shape[1], dtype=torch.float16, device=self.device
            )
            bias = torch.zeros(2 * quarter, dtype=torch.float32, device=self.device)
            first = self.rank * quarter
            for d in range(2):
                j = layer + 1 + d
                if j < len(gates):
                    w, b = gates[j]
                    weight[d * quarter : (d + 1) * quarter] = w[
                        first : first + quarter
                    ].to(torch.float16)
                    bias[d * quarter : (d + 1) * quarter] = b[first : first + quarter]
            self.hint_weights[layer] = (weight, bias)
        weight, bias = self.hint_weights[layer]
        logits = torch.mm(inputs, weight.t())
        start, header, entries, words = self.native.shared_expert_hint_layout()[:4]
        publish_hints[(1,)](
            logits,
            bias,
            self.control_view,
            self.hint_counter,
            self.trunc,
            layer,
            inputs.shape[0],
            BASE=start + self.rank * (header + entries * words),
            ENTRIES=entries,
            WORDS=words,
            EQ=quarter,
            FIRST=self.rank * quarter,
            TOK=8,
            CAND=8,
            BLOCK=triton_next_pow2(quarter),
            TRUNC=self.trunc_enabled,
            num_warps=4,
        )

    def wait_jobs(self, timeout=30.0):
        """Block (CPU only) until the coordinator finished every plan this
        rank published and all ranks' lanes completed the resulting jobs, so
        host cold copies of evicted experts are complete (prefill reads them)."""
        if not self.async_evict:
            return
        import time

        start, header = self.native.shared_expert_job_layout()[:2]
        request = self.meta["RANK_START"] + self.rank * self.meta["RANK_WORDS"]
        control = self.control_host
        begin = time.monotonic()
        while True:
            if int(control[1]) >= int(control[request]):  # type: ignore[index]
                head = int(control[start])  # type: ignore[index]
                # per rank: saves done (word 1 + r) + loads done (word 5 + r)
                if all(
                    int(control[start + 1 + r]) + int(control[start + 5 + r]) >= head  # type: ignore[index]
                    for r in range(4)
                ):
                    return
            if int(control[2]) != 0:  # type: ignore[index]
                raise RuntimeError("Shared expert coordinator failed")
            if time.monotonic() - begin > timeout:
                raise RuntimeError("Shared expert job lanes did not drain")
            time.sleep(0.0002)

    def _inplace_swap(self, layer, ids, out):
        if not self.stash_in_publish:
            stash_plan[(1,)](
                self.control_view,
                self.fast_flag,
                self.plan_incoming,
                self.plan_victims,
                self.plan_trunc,
                OP_START=self.meta["OP_START"],
                OP_WORDS=self.meta["OP_WORDS"],
                MAX_OPS=self.max_ops,
                BLOCK=64,
            )
        banks = [self.banks[layer, shard] for shard in SHARDS]
        lay = self.swap_layouts[layer]
        if self.free_swap:
            torch.ops.v41k.swap_combined(
                banks[0],
                banks[1],
                banks[2],
                self.mapped,
                self.slots,
                self.loads,
                self.saves,
                self.count,
                self.fast_flag,
                lay,
                int(envs.VLLM_V41_SWAP_CTAS),
            )
            return
        torch.ops.v41k.swap_stage(
            banks[0],
            banks[1],
            banks[2],
            self.mapped,
            self.swap_scratch,
            self.slots,
            self.loads,
            self.count,
            self.fast_flag,
            lay,
        )
        torch.ops.v41k.swap_signal(
            self.control_view,
            self.counter,
            self.swap_wb,
            self.count,
            self.fast_flag,
            self.swap_flag0 + self.rank * self.swap_stride,
        )
        if self.swap_dma is not None:
            return
        main = torch.cuda.current_stream(self.device)
        self.swap_side.wait_stream(main)
        with torch.cuda.stream(self.swap_side):
            torch.ops.v41k.swap_writeback(
                self.mapped,
                self.swap_scratch,
                self.loads,
                self.count,
                self.fast_flag,
                self.swap_wb,
                self.control_view,
                self.swap_flag0,
                self.swap_stride,
                4,
                self.swap_ack,
                self.swap_done,
                lay,
                int(envs.VLLM_V41_SWAP_WB_CTAS),
            )

    def _copies(self, layer):
        spec = self.specs[layer]
        if envs.VLLM_V41_SWAP_FUSED == "1":
            offsets, offset = [], 0
            for size in spec.projection_bytes:
                offsets.append(offset)
                offset += size
            swap_original_experts3(
                [self.banks[layer, shard] for shard in SHARDS],
                self.mapped,
                self.slots,
                self.loads,
                self.saves,
                self.count,
                self.geometries[layer],
                offsets,
            )
            return
        offset = 0
        for shard, geometry, size in zip(
            SHARDS, self.geometries[layer], spec.projection_bytes
        ):
            swap_original_experts(
                self.banks[layer, shard],
                self.mapped,
                self.slots,
                self.loads,
                self.saves,
                self.count,
                geometry,
                projection_offset=offset,
            )
            offset += size

    def warmup(self):
        if self.ready:
            return
        if not self._registered:
            raise RuntimeError("Shared rank registration is closed")
        for dtype in (torch.int32, torch.int64):
            ids = torch.zeros(1, dtype=dtype, device=self.device)
            out = torch.empty_like(ids)
            publish_shared_routes.warmup(
                ids,
                self.counter,
                self.control_view,
                self.slots,
                self.loads,
                self.saves,
                self.count,
                self.fast_flag,
                0,
                1,
                MAX_OPS=self.max_ops,
                **self.meta,
                grid=(1,),
            )
            commit_shared_routes.warmup(
                ids,
                self.table,
                self.cold_offsets,
                self.counter,
                self.control_view,
                out,
                self.statistics,
                self.fast_flag,
                0,
                1,
                EXPERTS=self.experts,
                **self.meta,
                grid=(1,),
            )
            if self.fast_hits:
                resolve_resident_routes.warmup(
                    ids,
                    self.table,
                    out,
                    self.fast_flag,
                    self.count,
                    self.statistics,
                    self.hit_counter,
                    self.control_view,
                    0,
                    1,
                    EXPERTS=self.experts,
                    BLOCK=64,
                    LOG=self.rank == 0,
                    **self.hit_meta,
                    grid=(1,),
                )
        for layer in range(len(self.specs)):
            self._copies(layer)  # count zero: compilation only, no data copies.
        torch.accelerator.synchronize(self.device)
        self.ready = True

    def resolve(self, layer: int, ids: torch.Tensor, weights=None):
        if not self.ready:
            raise RuntimeError("Warm up shared expert kernels before resolving routes")
        if (
            not 0 <= layer < len(self.specs)
            or ids.device != self.device
            or ids.dtype not in (torch.int32, torch.int64)
            or not ids.is_contiguous()
            or not 0 < ids.numel() <= min(self.max_routes, self.specs[layer].resident)
        ):
            raise ValueError("Shared decode supports at most M6 top-6 routes")
        out = torch.empty_like(ids)
        gate = None
        if weights is not None and (self.miss_hist is not None or self.skip_below > 0):
            gate = weights.float().contiguous()
        skip = (
            self.skip_below > 0 and gate is not None and self.fast_hits and self.inplace
        )
        if self.trunc_enabled and layer == 0:
            self.trunc.fill_(1 << 20)
            self.round.add_(1)
        if self.fast_hits:
            resolve_resident_routes[(1,)](
                ids,
                self.table,
                out,
                self.fast_flag,
                self.count,
                self.statistics,
                self.hit_counter,
                self.control_view,
                layer,
                ids.numel(),
                EXPERTS=self.experts,
                BLOCK=64,
                LOG=self.rank == 0,
                RELAXED=envs.VLLM_V41_HITLOG_RELAXED == "1",
                trunc=self.trunc,
                TRUNC=self.trunc_enabled,
                weights=gate,
                hist=self.miss_hist,
                HIST=self.miss_hist is not None and gate is not None,
                skip_mask=self.skip_mask,
                skip_below=self.skip_below,
                skip_late=self.skip_late,
                late_from=self.skip_late_from,
                SKIP=skip,
                min_rank=self.skip_min_rank,
                DEFER=skip and self.skip_defer,
                **self.hit_meta,
            )
        if self.trunc_enabled:
            publish_round[(1,)](
                self.control_view,
                self.round,
                self.meta["RANK_START"] + self.rank * self.meta["RANK_WORDS"] + 41,
            )
        publish_shared_routes[(1,)](
            ids,
            self.counter,
            self.control_view,
            self.slots,
            self.loads,
            self.saves,
            self.count,
            self.fast_flag,
            layer,
            ids.numel(),
            MAX_OPS=self.max_ops,
            **self.meta,
            plan_evicts=self.plan_evicts if self.inplace else None,
            EVICT=self.inplace and self.async_evict,
            skip_mask=self.skip_mask,
            SKIP=skip,
            DEFER=skip and self.skip_defer,
            plan_incoming=self.plan_incoming if self.inplace else None,
            plan_victims=self.plan_victims if self.inplace else None,
            plan_trunc=self.plan_trunc if self.inplace else None,
            STASH=self.inplace and self.stash_in_publish,
        )
        if self.inplace:
            self._inplace_swap(layer, ids, out)
            commit_local_routes[(1,)](
                ids,
                self.table,
                self.cold_offsets,
                self.counter,
                self.slots,
                self.loads,
                self.count,
                self.plan_incoming,
                self.plan_victims,
                self.plan_trunc,
                out,
                self.statistics,
                self.fast_flag,
                layer,
                ids.numel(),
                EXPERTS=self.experts,
                BLOCK=64,
                trunc=self.trunc,
                TRUNC=self.trunc_enabled,
                plan_saves=self.saves,
                control=self.control_view,
                ack_word=self.meta["RANK_START"]
                + self.rank * self.meta["RANK_WORDS"]
                + 1,
                FREE=self.free_swap,
                plan_evicts=self.plan_evicts,
                EVICT=self.async_evict,
                skip_mask=self.skip_mask,
                SKIP=skip,
            )
            if (
                skip
                and self.skip_defer
                and self.router_ready
                and weights is not None
                and weights.dtype == torch.float32
                and weights.is_contiguous()
            ):
                # weights is the MoE's own probability tensor: rewritten in place.
                substitute_dropped[(1,)](
                    out,
                    ids,
                    weights,
                    self.skip_mask,
                    self.fast_flag,
                    self.table[layer],
                    self.router_logits,
                    self.router_bias,
                    self.trunc,
                    ids.numel(),
                    EXPERTS=self.router_logits.shape[1],  # type: ignore[union-attr]
                    BLOCK_E=triton_next_pow2(self.router_logits.shape[1]),  # type: ignore[union-attr]
                    SCALE=self.router_scale,
                    SUBST=self.skip_subst,
                    TRUNC=self.trunc_enabled,
                )
        else:
            self._copies(layer)
            commit_shared_routes[(1,)](
                ids,
                self.table,
                self.cold_offsets,
                self.counter,
                self.control_view,
                out,
                self.statistics,
                self.fast_flag,
                layer,
                ids.numel(),
                EXPERTS=self.experts,
                **self.meta,
            )
        return out

    def close(self):
        lane = getattr(self, "swap_dma", None)
        if lane is not None:
            lane.stop()
            self.swap_dma = None
        lane = getattr(self, "job_lane", None)
        if lane is not None:
            lane.stop()
            self.job_lane = None
        try:
            if self._registered:
                torch.accelerator.synchronize(self.device)
        finally:
            self.mapped = self.control_view = None
            for pointer in reversed(self._registered):
                self.native.host_unregister(pointer)
            self._registered.clear()
            self.host = self.control_host = None
            self.ready = False
