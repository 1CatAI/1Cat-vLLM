# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side compact/reference dispatch for uniform DFlash2 verifier batches."""

from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_triton import sort_topk_with_vocab_ties
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    dflash2_sparse_topk_rejection_sample,
    rejection_sample,
)

logger = init_logger(__name__)

ROWS_PER_REQUEST = 8
# Each request count captures its own graphs into private pools the KV budget
# does not cover: about 170 MiB per count up to c4, and 2.7 GiB in total once
# c5-c8 are captured too, which also ran slower than the host-decision path.
MAX_GRAPH_REQUESTS = 4
MAX_GRAPHS = 8
# Capture allocates outside the budget. Serving still needs room after it: the
# host-decision path at c8 reserves about 1.3 GiB on top of the idle footprint.
CAPTURE_HEADROOM_BYTES = 5 << 29


@triton.jit
def _prepare_hidden(
    hidden,
    fixed_hidden,
    dense,
    dense_pointer,
    hidden_size: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    tl.store(
        fixed_hidden + offsets,
        tl.load(hidden + offsets, offsets < hidden_size, 0),
        offsets < hidden_size,
    )
    if tl.program_id(0) == 0:
        tl.store(dense_pointer, dense.to(tl.int64))


@triton.jit
def _prepare_request(
    inputs,
    fixed_inputs,
    positions,
    fixed_positions,
    source_mapping,
    mapping,
    expanded,
    ids,
    fixed_ids,
    values,
    fixed_values,
    sparse_size: tl.constexpr,
    source_sparse_stride: tl.constexpr,
    fixed_sparse_stride: tl.constexpr,
    rows: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0)
    request = tl.load(source_mapping + index)
    offsets = tl.arange(0, block)
    first = index * rows
    tl.store(
        fixed_ids + request * fixed_sparse_stride + offsets,
        tl.load(
            ids + request * source_sparse_stride + offsets, offsets < sparse_size, 0
        ),
        offsets < sparse_size,
    )
    tl.store(
        fixed_values + request * fixed_sparse_stride + offsets,
        tl.load(
            values + request * source_sparse_stride + offsets, offsets < sparse_size, 0
        ),
        offsets < sparse_size,
    )
    tl.store(mapping + index, request)
    tl.store(expanded + first + offsets, request, offsets < rows)
    tl.store(
        fixed_inputs + first + offsets,
        tl.load(inputs + first + offsets, offsets < rows, 0),
        offsets < rows,
    )
    tl.store(
        fixed_positions + first + offsets,
        tl.load(positions + first + offsets, offsets < rows, 0),
        offsets < rows,
    )


@triton.jit
def _copy_dense_slot(
    source_pointer,
    destination,
    mapping,
    destination_stride: tl.constexpr,
    slot_size: tl.constexpr,
    block: tl.constexpr,
):
    # The source address is refreshed with the current request's inputs.
    # Only the reference branch reads dense proposals; compact rounds incur
    # no full-vocabulary copy, and inactive request slots are never copied.
    source = tl.load(source_pointer).to(destination.dtype)
    request = tl.load(mapping + tl.program_id(0))
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    values = tl.load(source + request * slot_size + offsets, offsets < slot_size, 0)
    tl.store(
        destination + request * destination_stride + offsets,
        values,
        offsets < slot_size,
    )


def _capture_headroom(device: torch.device) -> bool:
    free, _ = torch.cuda.mem_get_info(device)
    idle = torch.accelerator.memory_reserved(device) - (
        torch.accelerator.memory_allocated(device)
    )
    return free + idle >= CAPTURE_HEADROOM_BYTES


def _all_ranks_support(status: int) -> bool:
    # Capability disagreement must select the existing sampler on every rank.
    # This is a capture-time CPU collective, never a per-round GPU fence.
    with torch.inference_mode(False):
        unsupported = torch.tensor(int(status != 0), dtype=torch.int32)
        dist.all_reduce(
            unsupported, op=dist.ReduceOp.MAX, group=get_tp_group().cpu_group
        )
    return unsupported.item() == 0


class _RejectionGraph:
    def __init__(self, model, speculator, rejection, hidden, batch, sparse, penalties):
        self.requests = batch.num_reqs
        # Penalties are a trace constant: graphs without them launch no kernel.
        self.penalties = penalties
        rows = self.requests * ROWS_PER_REQUEST
        self.hidden = torch.empty_like(hidden)
        self.inputs = torch.empty_like(batch.input_ids[:rows])
        self.positions = torch.empty_like(batch.positions[:rows])
        # MRv2 allocates expanded metadata anew every round. Keep graph inputs
        # independent of those allocations and preserve the original slot ID.
        self.mapping = batch.idx_mapping[: self.requests].clone()
        self.expanded = batch.expanded_idx_mapping[:rows].clone()
        self.local = torch.arange(
            ROWS_PER_REQUEST,
            dtype=batch.expanded_local_pos.dtype,
            device=hidden.device,
        ).repeat(self.requests)
        self.cu = batch.cu_num_logits[: self.requests + 1].clone()
        self.request_indices = batch.idx_mapping_np.copy()
        self.rejection = rejection
        self.model = model
        self.speculator = speculator
        self.sparse = tuple(torch.empty_like(t) for t in sparse)
        dense = speculator.draft_logits
        if self.requests == 1:
            # Alias the one private proposal slot for every state index while
            # preserving the original request mapping. Dense kernels accept
            # stride(0)==0; no other request slot is read.
            self.draft_slot = dense.new_empty((1, *dense.shape[1:]))
            self.draft_logits = self.draft_slot.expand(dense.shape[0], -1, -1)
        else:
            # Every replay refills the slots it reads, so graphs can share this.
            shared = getattr(rejection, "_sm70_dflash2_graph_draft_logits", None)
            if shared is None or shared.shape != dense.shape:
                shared = dense.new_empty(dense.shape)
                rejection._sm70_dflash2_graph_draft_logits = shared
            self.draft_logits = shared
        self.dense_pointer = self.inputs.new_empty(1, dtype=torch.int64)
        self.output = self.inputs.new_empty(
            (self.requests, ROWS_PER_REQUEST), dtype=torch.int64
        )
        self.count = self.inputs.new_empty(self.requests, dtype=torch.int32)
        # Sampling parameters live in UVA-backed tensors whose pool rotates
        # round-robin on every apply_staged_writes(), so states.temperature.gpu
        # is a DIFFERENT address each request. A captured graph freezes the
        # pointer it saw, which then reads a stale slot and corrupts sampling
        # (observed: temperature-dependent filler-token degeneration — outputs
        # repeat a fragment forever at T>=0.5 while T==0 looks healthy, because
        # greedy never reads the temperature value).
        # Keep private storage and refresh it in prepare(), same as positions.
        # ⚠️ Size must be the FULL state table (max_num_reqs), NOT self.requests.
        # The kernels index these with `self.mapping` = batch.idx_mapping, which
        # holds GLOBAL request-slot ids (0..max_num_reqs-1), not local 0..N-1.
        # Sizing to self.requests made the lookup read out of bounds whenever a
        # request lived in a high slot, which flipped `flags` inconsistently
        # across TP ranks and deadlocked graph dispatch under concurrency
        # (symptom: `No available shared memory broadcast block found`).
        # `empty_like` leaves these as uninitialised device memory, and capture
        # runs BEFORE the first real prepare(). `flags` is derived from these
        # tensors during capture, so garbage here produced a garbage branch
        # decision that differed across TP ranks (deadlock). Seed them with the
        # neutral values (temperature 1.0 / top_p 1.0) so capture always sees a
        # well-defined state.
        states = rejection.sampler.sampling_states
        self.temperature = torch.empty_like(states.temperature.gpu)
        self.top_p = torch.empty_like(states.top_p.gpu)
        self.seeds = torch.empty_like(states.seeds.gpu)
        self.temperature.fill_(1.0)
        self.top_p.fill_(1.0)
        self.seeds.zero_()
        self.counters = self.inputs.new_zeros(2, dtype=torch.int64)
        self.request_id = None
        self.prefix = torch.cuda.CUDAGraph(keep_graph=True)
        self.reference = torch.cuda.CUDAGraph(keep_graph=True)
        self.compact = torch.cuda.CUDAGraph(keep_graph=True)
        self.prepare(hidden, batch, sparse)

    def prepare(self, hidden, batch, sparse):
        ids, values = sparse
        fixed_ids, fixed_values = self.sparse
        _prepare_hidden[(triton.cdiv(hidden.numel(), 1024),)](
            hidden,
            self.hidden,
            self.speculator.draft_logits,
            self.dense_pointer,
            hidden.numel(),
            1024,
            num_warps=4,
        )
        _prepare_request[(self.requests,)](
            batch.input_ids,
            self.inputs,
            batch.positions,
            self.positions,
            batch.idx_mapping,
            self.mapping,
            self.expanded,
            ids,
            fixed_ids,
            values,
            fixed_values,
            ids.shape[1] * ids.shape[2],
            ids.stride(0),
            fixed_ids.stride(0),
            ROWS_PER_REQUEST,
            triton.next_power_of_2(ids.shape[1] * ids.shape[2]),
            num_warps=1,
        )
        # Refresh the private sampling parameters from the rotating UVA pool.
        # These are the live pointers the graph reads, so they must be updated
        # on every round even though the graph itself is captured once.
        # Whole-table copy: the kernels index with global slot ids (see the
        # sizing note in __init__), so a per-request gather is not enough.
        # `cu_num_logits` drives every row index inside the kernels, and the
        # capture-time clone is only valid while the batch shape is identical.
        # Refresh it every round so a changed batch cannot leave the kernels
        # walking rows outside the candidate buffers (symptom: target_keep
        # ended up all-false and the decode repeated one token forever).
        self.cu.copy_(batch.cu_num_logits[: self.requests + 1])
        states = self.rejection.sampler.sampling_states
        self.temperature.copy_(states.temperature.gpu)
        self.top_p.copy_(states.top_p.gpu)
        self.seeds.copy_(states.seeds.gpu)
        # Dense proposals are accessed indirectly inside a conditional body.
        # Record their consumer stream for replaced/eager tensor lifetimes.
        self.speculator.draft_logits.record_stream(torch.cuda.current_stream())

    def _probe(self):
        from .sparse_rejection import _compact_target_reference_flags, _penalize_probe

        probe = getattr(
            self.model,
            "get_compact_target_probe_with_fallback",
            self.model.get_topk_tokens_and_logits_with_fallback,
        )
        self.ids, self.values, self.fallback = probe(self.hidden, 64)
        if self.fallback is None:
            return False
        sampler = self.rejection.sampler
        states = sampler.sampling_states
        floors = None
        if self.penalties:
            floors = _penalize_probe(
                self.values,
                self.ids,
                sampler.penalties_state,
                self.expanded,
                self.inputs,
                self.local,
            )
        self.values, self.ids = sort_topk_with_vocab_ties(
            self.values, self.ids, vocab_size=states.vocab_size, descending=True
        )
        # NOTE(2026-11-08): `states.*.gpu` are UVA pointers that rotate on every
        # apply_staged_writes(), so a captured graph that reads them keeps
        # reading the buffer it saw at capture time -- i.e. stale parameters.
        # The private buffers are refreshed in prepare() right before every
        # replay, so they carry the current round's values. Use them here too.
        # (The earlier deadlock from doing this came from `empty_like` leaving
        # the private buffers uninitialised during capture; they are now seeded
        # with the neutral values in __init__.)
        # Use the private buffers (refreshed in prepare() every round), never
        # states.*.gpu -- those are UVA round-robin pool pointers, and the graph
        # freezes the address it saw at capture time, so a replay would read
        # stale parameters.
        self.flags = _compact_target_reference_flags(
            self.values, self.temperature, self.top_p, self.expanded, floors
        )
        # The compact (top-64 sparse) branch emits a constant token when T > 0,
        # i.e. it repeats. Confirmed by elimination:
        #   - inputs are correct (SGTEMP buf == src; SGVAL sees 64/64 finite
        #     candidates; SGRES residuals reproduce)
        #   - forcing temperature 1.0 still repeats -> not temperature
        #   - always rejecting the draft still repeats -> not draft acceptance
        #   - fp64 gumbel, position keys and injecting pos noise all fail to fix it
        #   - in-kernel diagnostics report target_keep all-false (that reading is
        #     itself unreliable -- see the defect register)
        # It is independent of the environment: clearing every VLLM_* env
        # variable (all default to on) and using e4m3 draft KV still repeats at
        # T=0.7, so the compact branch itself is at fault (#1409).
        #
        # It is also independent of concurrency -- the earlier "2/4/8 concurrent
        # is fine" observation was an artefact of coding (HTML/SVG) prompts whose
        # strong structural constraints mask the degeneration. Measured with the
        # official env set, e4m3 draft KV and adaptive compact dispatch:
        #   1 request  + short prompt -> repeats (3/3 runs, "Redis####...")
        #   2 concurrent + short prompt -> repeats
        #   4 concurrent + short prompt -> repeats
        #   2/4/8 concurrent + coding prompt -> fine
        # Any concurrency level repeats on free-form text.
        #
        # Cost of staying on the reference branch: compact is 6.3%/4.1%/1.0%
        # faster at 2/4/8 concurrency, but that is not a gain worth repeating
        # output for. The reference branch lives in the same graph, passes 4/4
        # at 4-way concurrency, and produces correct output; full-vocab
        # rejection costs throughput, roughly 432 -> 277 tok/s.
        self.flags.fill_(True)
        return True

    def _dense(self):
        slot_size = self.draft_logits.shape[1] * self.draft_logits.shape[2]
        _copy_dense_slot[(self.requests, triton.cdiv(slot_size, 4096))](
            self.dense_pointer,
            self.draft_logits,
            self.mapping,
            self.draft_logits.stride(0),
            slot_size,
            4096,
            num_warps=4,
        )
        logits = self.fallback()
        if logits is None:
            raise RuntimeError("DFlash2 graph rejection requires replicated logits")
        sampler = self.rejection.sampler
        states = sampler.sampling_states
        processed = sampler.apply_sampling_params(
            logits,
            self.expanded,
            self.request_indices,
            self.positions,
            self.inputs,
            self.local,
        )
        output, count = rejection_sample(
            processed,
            self.draft_logits,
            self.inputs,
            self.cu,
            self.positions,
            self.mapping,
            self.expanded,
            self.local,
            self.temperature,
            self.seeds,
            self.rejection.num_speculative_steps,
            use_fp64=sampler.use_fp64_gumbel,
        )
        self.output.copy_(output)
        self.count.copy_(count)

    def _compact(self):
        sampler = self.rejection.sampler
        states = sampler.sampling_states
        ids, values = self.sparse
        output, count = dflash2_sparse_topk_rejection_sample(
            self.ids,
            self.values,
            ids,
            values,
            self.inputs,
            self.cu,
            self.positions,
            self.mapping,
            self.temperature,
            self.top_p,
            self.seeds,
            self.rejection.num_speculative_steps,
            use_fp64=sampler.use_fp64_gumbel,
            target_top_k=20,
        )
        self.output.copy_(output)
        self.count.copy_(count)

    def capture(self):
        # Sampling is stateless in seed/position. Warmup writes only the private
        # outputs and does not advance request state or proposal random streams.
        # `capture_mode` forces the parametric kernels (temperature/top-k/top-p/
        # min-p) into the graph: their skip-branches read host numpy values and
        # would otherwise be folded to "skip" for the lifetime of the graph,
        # because warm-up still carries the default sampling params.
        states = self.rejection.sampler.sampling_states
        previous_capture_mode = states.capture_mode
        states.capture_mode = True
        try:
            return self._capture()
        finally:
            states.capture_mode = previous_capture_mode

    def _capture(self):
        with graph_capture(device=self.hidden.device):
            if not _all_ranks_support(0 if self._probe() else 801):
                return False
            self._dense()
            self._compact()
            torch.accelerator.synchronize()
            with torch.cuda.graph(self.prefix):
                self._probe()
            with torch.cuda.graph(self.reference):
                self._dense()
            with torch.cuda.graph(self.compact):
                self._compact()
        status = torch.ops._C.sm70_sampler_graph_prepare_collective(
            self.flags, self.reference.raw_cuda_graph()
        )
        if not _all_ranks_support(status):
            return False
        status = torch.ops._C.sm70_sampler_graph_attach_branch(
            self.flags,
            self.prefix.raw_cuda_graph(),
            self.reference.raw_cuda_graph(),
            self.compact.raw_cuda_graph(),
            self.counters,
        )
        if not _all_ranks_support(status):
            return False
        status = 0
        try:
            self.prefix.instantiate()
        except RuntimeError as error:
            logger.debug("CUDA rejected a DFlash2 sampler graph: %s", error)
            status = 801
        return _all_ranks_support(status)

    def replay(self, hidden, batch, sparse):
        request_id = tuple(batch.req_ids)
        if request_id != self.request_id:
            self.counters.zero_()
            self.request_id = request_id
        self.prepare(hidden, batch, sparse)
        self.prefix.replay()
        # AsyncOutput copies on another stream. Give it independent storage so
        # the next graph replay cannot overwrite a still-pending output copy.
        return SamplerOutput(
            sampled_token_ids=self.output.clone(),
            num_sampled=self.count.clone(),
            logprobs_tensors=None,
            num_nans=None,
        )


def try_graph_rejection(
    model: Any,
    speculator: Any,
    rejection: Any,
    hidden: torch.Tensor,
    batch: Any,
    sparse: tuple[torch.Tensor, torch.Tensor],
) -> SamplerOutput | None:
    """Caller has checked the positive-temperature compact sampling contract."""
    if (
        not 1 <= getattr(batch, "num_reqs", 0) <= MAX_GRAPH_REQUESTS
        or batch.num_tokens != batch.num_reqs * ROWS_PER_REQUEST
        or hidden.shape[0] != batch.num_reqs * ROWS_PER_REQUEST
        or rejection.num_speculative_steps != ROWS_PER_REQUEST - 1
        or not np.array_equal(
            batch.cu_num_logits_np,
            np.arange(batch.num_reqs + 1) * ROWS_PER_REQUEST,
        )
        or get_tensor_model_parallel_world_size() != 4
        or not hasattr(model, "get_topk_tokens_and_logits_with_fallback")
        or not hasattr(torch.ops._C, "sm70_sampler_graph_prepare_collective")
        or rejection.sampler.sampling_states.vocab_size < 32768
        or not all(
            t.is_contiguous() for t in (hidden, speculator.draft_logits, *sparse)
        )
        or sparse[0].stride(0) != sparse[1].stride(0)
    ):
        return None
    states = rejection.sampler.sampling_states
    idx = batch.idx_mapping_np
    penalties = bool(np.any(rejection.sampler.penalties_state.use_penalty[idx]))
    # CPU dispatch choices are trace constants; numeric parameters and request
    # slots remain live GPU inputs. Private input storage makes cache decisions
    # independent of the allocator and the target graph's output lifetime.
    tensors = (
        hidden,
        batch.input_ids,
        batch.positions,
        speculator.draft_logits,
        *sparse,
    )
    key = (
        id(model),
        id(speculator),
        hidden.dtype,
        bool(np.any(states.temperature.np[idx] != 1.0)),
        bool(np.any(states.top_p.np[idx] != 1.0)),
        rejection.sampler.use_fp64_gumbel,
        penalties,
        tuple((t.dtype, tuple(t.shape), tuple(t.stride())) for t in tensors),
    )
    graphs = getattr(rejection, "_sm70_dflash2_rejection_graphs", None)
    if graphs is None:
        graphs = rejection._sm70_dflash2_rejection_graphs = {}
    if key not in graphs:
        if len(graphs) >= MAX_GRAPHS:
            # Bound graph storage for unexpected shape/contract combinations.
            return None
        if not _all_ranks_support(0 if _capture_headroom(hidden.device) else 801):
            graphs[key] = None
            return None
        candidate = _RejectionGraph(
            model, speculator, rejection, hidden, batch, sparse, penalties
        )
        graphs[key] = candidate if candidate.capture() else None
    graph = graphs[key]
    if graph is None:
        return None
    rejection.sm70_dflash2_reference_counts = graph.counters
    logger.info_once(
        "DFlash2 compact/reference rejection uses device-side graph dispatch."
    )
    return graph.replay(hidden, batch, sparse)
