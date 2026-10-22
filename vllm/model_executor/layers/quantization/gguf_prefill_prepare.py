# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact predictive preparation owned by one TurboMind expert cache.

Masks are performance hints, never semantic state. A stale or empty prediction
is safe: every currently routed miss is converted before its bank is consumed.
Completed CPU snapshots are adopted only at the next layer-zero boundary.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _predicted_lists(
    residency, predicted, lists, counts, E: tl.constexpr, B: tl.constexpr
):
    e = tl.arange(0, B)
    needed = tl.load(predicted + e, e < E, 0) != 0
    slot = tl.load(residency + e, e < E, -1)
    hot = (needed & (slot >= 0) & (e < E)).to(tl.int32)
    cold = (needed & (slot < 0) & (e < E)).to(tl.int32)
    tl.store(lists + tl.cumsum(hot, 0) - 1, e, hot != 0)
    tl.store(lists + E + tl.cumsum(cold, 0) - 1, e, cold != 0)
    tl.store(counts, tl.sum(hot, 0))
    tl.store(counts + 1, tl.sum(cold, 0))


@triton.jit
def _missing_sources(
    residency,
    bases,
    offsets,
    predicted,
    observed,
    sources,
    lists,
    counts,
    E: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.arange(0, B)
    slot = tl.load(residency + e, e < E, -1)
    host_slot = tl.load(residency + E + e, e < E, -1)
    needed = tl.load(offsets + e + 1, e < E, 0) > tl.load(offsets + e, e < E, 0)
    tl.store(observed + e, needed, e < E)
    missing = needed & (tl.load(predicted + e, e < E, 0) == 0) & (e < E)
    for shard in tl.static_range(3):
        bank = tl.load(bases + shard * 3)
        host = tl.load(bases + shard * 3 + 1)
        stride = tl.load(bases + shard * 3 + 2)
        pointer = tl.where(
            slot >= 0,
            bank + slot.to(tl.int64) * stride,
            host + host_slot.to(tl.int64) * stride,
        )
        tl.store(sources + shard * E + e, pointer, e < E)
    hot = (missing & (slot >= 0)).to(tl.int32)
    cold = (missing & (slot < 0)).to(tl.int32)
    tl.store(lists + tl.cumsum(hot, 0) - 1, e, hot != 0)
    tl.store(lists + E + tl.cumsum(cold, 0) - 1, e, cold != 0)
    tl.store(counts, tl.sum(hot, 0))
    tl.store(counts + 1, tl.sum(cold, 0))


class PredictivePrefillPreparation:
    def __init__(self, tm):
        from .gguf_expert_cache import SHARDS

        self.tm, self.cache = tm, tm.cache
        cache, rows = self.cache, []
        for layer in cache.layers:
            for shard in SHARDS:
                host = (
                    cache.host_view.data_ptr()
                    + cache.host[layer, shard].data_ptr()
                    - cache.host_arena.data_ptr()
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
        shape = (len(cache.layers), cache.experts)
        self.host_hints = torch.ones((2, *shape), dtype=torch.bool, pin_memory=True)
        self.predicted = torch.ones(shape, dtype=torch.bool, device=cache.device)
        self.observed = torch.ones_like(self.predicted)
        self.predicted_rows = list(self.predicted.unbind())
        self.observed_rows = list(self.observed.unbind())
        self.host_rows = [list(t.unbind()) for t in self.host_hints.unbind()]
        self.sources = torch.empty_like(tm.sources)
        self.lists = torch.empty_like(tm.lists)
        self.counts = torch.empty_like(tm.counts)
        self.views = [
            {
                "sources": self.sources[p],
                "lists": self.lists[p],
                "counts": self.counts[p],
                "src": [self.sources[p, s] for s in range(3)],
                "list": [self.lists[p, w] for w in range(2)],
                "count": [self.counts[p, w : w + 1] for w in range(2)],
            }
            for p in range(tm.VIEWS)
        ]
        self.active_hint = 0
        self.pending = None
        self.pass_active = False
        self.started = False
        self.admit_recent = cache.prefill_prepare == "predict_resident"
        if self.admit_recent:
            self.host_plans = torch.empty(
                (len(cache.layers), cache.plan.numel()),
                dtype=torch.int32,
                pin_memory=True,
            )
            self.device_plan = torch.zeros_like(cache.plan)
            # Eight independent three-shard swaps share three stream-ordered
            # DMA batches, with incoming bytes saved before any write-back.
            self.swap_temporary = torch.empty(
                8
                * max(
                    sum(cache.sizes[layer, s] for s in SHARDS) for layer in cache.layers
                ),
                dtype=torch.uint8,
                device=cache.device,
            )

    def start(self):
        if not hasattr(self.cache.lane, "stage_cold_selected"):
            raise RuntimeError(
                "Predictive prefill requires the source-matched "
                "expert-cache native helper"
            )
        self.miss_stream = torch.cuda.Stream(self.cache.device, priority=-1)
        self.done = {key: torch.cuda.Event() for key in self.tm.KEYS}
        self.ready = {key: torch.cuda.Event() for key in self.tm.KEYS}
        self.hint_ready = [torch.cuda.Event(), torch.cuda.Event()]
        if self.admit_recent:
            if not hasattr(self.cache.lane, "admit_prefill"):
                raise RuntimeError("Resident prefill requires the source-matched lane")
            self.admission_stream = torch.cuda.Stream(self.cache.device)
            self.decode_fence_stream = torch.cuda.Stream(self.cache.device)
            # Decode graph replays must wait for the latest eager admission,
            # including an interrupted prefill. Keep an external event node.
            self.admission_done = torch.cuda.Event(external=True)
            self.decode_ready = torch.cuda.Event(external=True)
            self._commit_slots(0)
            self.admission_done.record()
            self.decode_ready.record()
        self.started = True
        logger.info_once(
            "GGUF TurboMind prefill predicts active experts for %d layers "
            "and completes every route miss.",
            len(self.cache.layers),
        )

    def begin(self, i, x, ids):
        qualified = (
            4096 <= ids.shape[0] <= 8192
            and x.dtype == torch.float16
            and x.is_contiguous()
            and not torch.cuda.is_current_stream_capturing()
        )
        if i == 0:
            self.pass_active = qualified
            if self.admit_recent:
                if torch.cuda.is_current_stream_capturing():
                    torch.cuda.current_stream().wait_event(self.decode_ready)
                else:
                    self.decode_ready.synchronize()
            if qualified:
                # Interrupted passes may still have speculative preparations in
                # flight. Retire their reads before rewriting the shared hint.
                main = torch.cuda.current_stream()
                for stream in (
                    self.tm.resident_stream,
                    self.tm.cold_stream,
                    self.miss_stream,
                ):
                    main.wait_stream(stream)
                if self.admit_recent:
                    # CPU LRU planning must follow earlier GPU-published decode
                    # entries, including graph replay. Never poll from a lane
                    # that also submits the CUDA work releasing that poll.
                    main.synchronize()
                if self.pending is not None and self.pending[1].query():
                    self.active_hint = self.pending[0]
                    self.pending = None
                    logger.info_once(
                        "GGUF predictive prefill adopted a completed routing hint."
                    )
                self.predicted.copy_(
                    self.host_hints[self.active_hint], non_blocking=True
                )
        return qualified and self.pass_active

    def filter(self, i):
        view = self.tm.views[i % self.tm.VIEWS]
        _predicted_lists[(1,)](
            view["residency"],
            self.predicted_rows[i],
            view["lists"],
            view["counts"],
            self.cache.experts,
            triton.next_power_of_2(self.cache.experts),
        )

    def stage(self, i, shard, destination, stream):
        self.cache.lane.stage_cold_selected(
            i,
            shard,
            destination,
            stream,
            self.host_rows[self.active_hint][i],
        )

    def complete(self, key, i, offsets):
        tm, cache = self.tm, self.cache
        stream, cold = tm.resident_stream, self.miss_stream
        stream.wait_stream(torch.cuda.current_stream())
        p, info = i % tm.VIEWS, tm.layer[cache.layers[i]]
        view = self.views[p]
        if key == "gu":
            with torch.cuda.stream(stream):
                _missing_sources[(1,)](
                    tm.views[p]["residency"],
                    self.layer_bases[i],
                    offsets,
                    self.predicted_rows[i],
                    self.observed_rows[i],
                    view["sources"],
                    view["lists"],
                    view["counts"],
                    cache.experts,
                    triton.next_power_of_2(cache.experts),
                )
            s0, s1, source = 0, 1, info["gate"]
            n_out, shape = 2 * info["rows"], cache.shapes[cache.layers[i], "w1"]
        else:
            s0, s1, source = 2, None, info["down"]
            n_out, shape = info["hidden"], cache.shapes[cache.layers[i], "w2"]
        cold.wait_stream(stream)
        spec = info[key]
        weight, stats = tm.bank(key, spec, i % tm.SLOTS)
        for which, ctas, target in ((0, 1024, stream), (1, 8, cold)):
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
                    tm.error,
                )
        self.done[key].record(cold)
        stream.wait_event(self.done[key])
        self.ready[key].record(stream)
        torch.cuda.current_stream().wait_event(self.ready[key])

    def finish(self, i):
        if self.admit_recent:
            selected = (
                torch.nonzero(self.host_rows[self.active_hint][i], as_tuple=False)
                .flatten()
                .to(torch.int32)
            )
            if 0 < len(selected) <= self.cache.resident0:
                main = torch.cuda.current_stream()
                self.admission_stream.wait_stream(main)
                with torch.cuda.stream(self.admission_stream):
                    admitted = self.cache.lane.admit_prefill(
                        i,
                        selected,
                        self.host_plans[i],
                        self.device_plan,
                        self.swap_temporary,
                        self.admission_stream.cuda_stream,
                    )
                    self._commit_slots(i)
                    self.admission_done.record()
                if admitted:
                    logger.info_once(
                        "GGUF prefill admits completed routing hints through "
                        "the existing LRU planner and DMA."
                    )
            # An interrupted pass can leave next-layer raw readers in flight.
            # Join them for decode without delaying the admission-copy stream.
            for stream in (
                self.tm.resident_stream,
                self.tm.cold_stream,
                self.miss_stream,
                self.admission_stream,
            ):
                self.decode_fence_stream.wait_stream(stream)
            self.decode_ready.record(self.decode_fence_stream)
        if i == len(self.cache.layers) - 1 and self.pending is None:
            target = 1 - self.active_hint
            self.host_hints[target].copy_(self.observed, non_blocking=True)
            self.hint_ready[target].record()
            self.pending = (target, self.hint_ready[target])

    def _commit_slots(self, i):
        from .gguf_expert_cache import _MAX_OPS, _remap

        # Zero output count: reuse the ordinary table commit without a routed
        # slot output. The owner keeps both the input and unused output alive.
        _remap[(1,)](
            self.cache.table,
            self.cache.table,
            self.device_plan,
            self.cache.table,
            i,
            0,
            EXPERTS=self.cache.experts,
            BLOCK=32,
            OPS=_MAX_OPS,
        )

    def wait_decode(self):
        if self.admit_recent:
            torch.cuda.current_stream().wait_event(self.decode_ready)

    def storage_bytes(self):
        ordinary = sum(
            t.numel() * t.element_size()
            for t in (
                self.bases,
                self.predicted,
                self.observed,
                self.sources,
                self.lists,
                self.counts,
            )
        )
        if self.admit_recent:
            ordinary += self.device_plan.numel() * self.device_plan.element_size()
            ordinary += self.swap_temporary.numel()
        return ordinary

    def close(self):
        if self.started:
            # Mapped-host reads and pinned hint transfers must finish before
            # either arena is released. This runs only during cache teardown.
            torch.accelerator.synchronize(self.cache.device)
