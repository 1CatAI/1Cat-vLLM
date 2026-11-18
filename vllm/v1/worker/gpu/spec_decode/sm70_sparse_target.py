# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact top-k target sampling for argmax (greedy) drafts on SM70.

With top-k 20 sampling, the target distribution is zero outside its global
top-20. Each TP rank therefore only contributes its local candidates instead
of all-gathering full-vocabulary logits, and rejection of a one-hot proposal
runs on those candidates. The DFlash2 compact-target contract is reused as
is: identical sampling transforms (temperature, top-k, top-p), the same
per-request random streams, and a full-vocabulary fallback for every request
whose cutoff is ambiguous (ties at the top-k boundary, or a top-p boundary
within rounding of a candidate).
"""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger
from vllm.model_executor.layers.sm70_topk_gather import _pack, _unpack
from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.ops.topk_topp_triton import sort_topk_with_vocab_ties
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
    _TARGET_PROBE_K,
    _compact_target_reference_flags,
    _compact_target_reference_rows,
    _compact_target_reference_rows_gpu,
    _penalize_probe,
    _sample_reference_requests,
    _supports_sparse_sampling_contract,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    greedy_draft_sparse_topk_rejection_sample,
    rejection_sample,
)
from vllm.v1.worker.gpu.spec_decode.target_sampling import ComputedTargetLogits

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

logger = init_logger(__name__)

_WARMED_SPARSE_TARGET_KEYS: set[tuple[int, int, int]] = set()

# Up to 4 requests share one (prefix, reference, compact) CUDA graph memory
# pool triple (~170 MiB total). Larger batches are compute-bound and use the
# per-request host-decision route so a single flagged row never sorts all rows.
_MAX_GRAPH_REQUESTS = 4
_MAX_GRAPHS = 16
_CAPTURE_HEADROOM_BYTES = 1 << 30


@triton.jit
def _prepare_greedy_graph_inputs(
    src_hidden,
    dst_hidden,
    hidden_numel,
    src_inputs,
    dst_inputs,
    src_positions,
    dst_positions,
    src_mapping,
    dst_mapping,
    dst_expanded,
    src_cu,
    dst_cu,
    src_temp,
    dst_temp,
    src_top_k,
    dst_top_k,
    src_top_p,
    dst_top_p,
    src_min_p,
    dst_min_p,
    src_seeds,
    dst_seeds,
    src_rep,
    dst_rep,
    src_freq,
    dst_freq,
    src_pres,
    dst_pres,
    state_size,
    requests: tl.constexpr,
    width: tl.constexpr,
    HAS_PENALTIES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        base = 0
        while base < state_size:
            idx = base + offs
            mask = idx < state_size
            tl.store(dst_temp + idx, tl.load(src_temp + idx, mask, 0.0), mask)
            tl.store(dst_top_k + idx, tl.load(src_top_k + idx, mask, 0), mask)
            tl.store(dst_top_p + idx, tl.load(src_top_p + idx, mask, 0.0), mask)
            tl.store(dst_min_p + idx, tl.load(src_min_p + idx, mask, 0.0), mask)
            tl.store(dst_seeds + idx, tl.load(src_seeds + idx, mask, 0), mask)
            if HAS_PENALTIES:
                tl.store(dst_rep + idx, tl.load(src_rep + idx, mask, 1.0), mask)
                tl.store(dst_freq + idx, tl.load(src_freq + idx, mask, 0.0), mask)
                tl.store(dst_pres + idx, tl.load(src_pres + idx, mask, 0.0), mask)
            base += BLOCK

        req_mask = offs < requests
        req_slot = tl.load(src_mapping + offs, req_mask, 0)
        tl.store(dst_mapping + offs, req_slot, req_mask)

        cu_mask = offs < (requests + 1)
        tl.store(dst_cu + offs, tl.load(src_cu + offs, cu_mask, 0), cu_mask)

        rows: tl.constexpr = requests * width
        row_mask = offs < rows
        row_req = offs // width
        row_slot = tl.load(src_mapping + row_req, row_mask, 0)
        tl.store(dst_expanded + offs, row_slot, row_mask)
        tl.store(dst_inputs + offs, tl.load(src_inputs + offs, row_mask, 0), row_mask)
        tl.store(
            dst_positions + offs,
            tl.load(src_positions + offs, row_mask, 0),
            row_mask,
        )
    else:
        h_offs = (pid - 1) * BLOCK + offs
        h_mask = h_offs < hidden_numel
        tl.store(dst_hidden + h_offs, tl.load(src_hidden + h_offs, h_mask, 0), h_mask)


def _capture_headroom(device: torch.device) -> bool:
    free, _ = torch.cuda.mem_get_info(device)
    idle = torch.accelerator.memory_reserved(device) - (
        torch.accelerator.memory_allocated(device)
    )
    return free + idle >= _CAPTURE_HEADROOM_BYTES


def _all_ranks_support(status: int) -> bool:
    if get_tensor_model_parallel_world_size() <= 1:
        return status == 0
    with torch.inference_mode(False):
        unsupported = torch.tensor(int(status != 0), dtype=torch.int32)
        dist.all_reduce(
            unsupported, op=dist.ReduceOp.MAX, group=get_tp_group().cpu_group
        )
    return unsupported.item() == 0


def _snapshot_sampling_inputs(
    sampler: Any,
) -> tuple[Any, dict[str, tuple[Any, torch.Tensor]]]:
    """Give a graph fixed GPU parameter addresses without mutating eager state."""
    view = copy(sampler)
    copies: dict[str, tuple[Any, torch.Tensor]] = {}
    for state_name, names in (
        ("sampling_states", ("temperature", "top_k", "top_p", "min_p", "seeds")),
        (
            "penalties_state",
            ("repetition_penalty", "frequency_penalty", "presence_penalty"),
        ),
    ):
        source_state = getattr(sampler, state_name)
        state = copy(source_state)
        for name in names:
            source = getattr(source_state, name, None)
            if source is None:
                continue
            tensor = source.gpu.clone()
            field = copy(source)
            field.gpu = tensor
            setattr(state, name, field)
            copies[name] = (source, tensor)
        setattr(view, state_name, state)
    return view, copies


class _GreedyDraftRejectionGraph:
    def __init__(
        self,
        model: Any,
        rejection: Any,
        hidden: torch.Tensor,
        batch: Any,
        *,
        top_k: int,
        penalties: bool,
        pools: tuple[Any, Any, Any],
    ) -> None:
        self.requests = batch.num_reqs
        self.width = rejection.num_speculative_steps + 1
        self.top_k = top_k
        self.penalties = penalties
        self.pools = pools
        rows = self.requests * self.width
        self.hidden = torch.empty_like(hidden)
        self.inputs = torch.empty_like(batch.input_ids[:rows])
        self.positions = torch.empty_like(batch.positions[:rows])
        self.mapping = batch.idx_mapping[: self.requests].clone()
        self.expanded = batch.expanded_idx_mapping[:rows].clone()
        self.local = torch.arange(
            self.width,
            dtype=batch.expanded_local_pos.dtype,
            device=hidden.device,
        ).repeat(self.requests)
        self.cu = batch.cu_num_logits[: self.requests + 1].clone()
        self.request_indices = batch.idx_mapping_np[: self.requests].copy()
        self.rejection = rejection
        self.sampler, self._parameter_copies = _snapshot_sampling_inputs(
            rejection.sampler
        )
        self.model = model
        # Allocate branch flags outside the shared prefix graph pool so
        # multiple request-count graphs never alias conditional branch inputs.
        self.flags = torch.empty(rows, dtype=torch.bool, device=hidden.device)
        self.output = self.inputs.new_empty(
            (self.requests, self.width), dtype=torch.int64
        )
        self.count = self.inputs.new_empty(self.requests, dtype=torch.int32)
        self.counters = self.inputs.new_zeros(2, dtype=torch.int64)
        self.request_id: tuple[str, ...] | None = None
        self.ids: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.fallback: Any = None
        self.prefix = torch.cuda.CUDAGraph(keep_graph=True)
        self.reference = torch.cuda.CUDAGraph(keep_graph=True)
        self.compact = torch.cuda.CUDAGraph(keep_graph=True)
        self.sampler.sampling_states.capture_mode = True
        self.prepare(hidden, batch)

    def _clear_transient(self) -> None:
        self.ids = None
        self.values = None
        self.fallback = None

    def prepare(self, hidden: torch.Tensor, batch: Any) -> None:
        copies = self._parameter_copies
        src_temp, dst_temp = copies["temperature"]
        src_top_k, dst_top_k = copies["top_k"]
        src_top_p, dst_top_p = copies["top_p"]
        src_min_p, dst_min_p = copies["min_p"]
        src_seeds, dst_seeds = copies["seeds"]
        if self.penalties:
            src_rep, dst_rep = copies["repetition_penalty"]
            src_freq, dst_freq = copies["frequency_penalty"]
            src_pres, dst_pres = copies["presence_penalty"]
            rep_s, rep_d = src_rep.gpu, dst_rep
            freq_s, freq_d = src_freq.gpu, dst_freq
            pres_s, pres_d = src_pres.gpu, dst_pres
        else:
            rep_s = rep_d = dst_temp
            freq_s = freq_d = dst_temp
            pres_s = pres_d = dst_temp

        hidden_numel = hidden.numel()
        grid = (1 + triton.cdiv(hidden_numel, 1024),)
        _prepare_greedy_graph_inputs[grid](
            hidden,
            self.hidden,
            hidden_numel,
            batch.input_ids,
            self.inputs,
            batch.positions,
            self.positions,
            batch.idx_mapping,
            self.mapping,
            self.expanded,
            batch.cu_num_logits,
            self.cu,
            src_temp.gpu,
            dst_temp,
            src_top_k.gpu,
            dst_top_k,
            src_top_p.gpu,
            dst_top_p,
            src_min_p.gpu,
            dst_min_p,
            src_seeds.gpu,
            dst_seeds,
            rep_s,
            rep_d,
            freq_s,
            freq_d,
            pres_s,
            pres_d,
            int(dst_temp.numel()),
            self.requests,
            self.width,
            self.penalties,
            1024,
            num_warps=4,
        )

    def _probe(self) -> bool:
        probe_fn = getattr(
            self.model,
            "get_compact_target_probe_with_fallback",
            getattr(self.model, "get_topk_tokens_and_logits_with_fallback", None),
        )
        if probe_fn is None:
            return False
        ids, values, fallback = probe_fn(self.hidden, _TARGET_PROBE_K)
        if fallback is None:
            return False
        sampler = self.sampler
        states = sampler.sampling_states
        floors = None
        if self.penalties:
            floors = _penalize_probe(
                values,
                ids,
                sampler.penalties_state,
                self.expanded,
                self.inputs,
                self.local,
            )
        values, ids = sort_topk_with_vocab_ties(
            values,
            ids,
            vocab_size=states.vocab_size,
            descending=True,
        )
        _compact_target_reference_flags(
            values,
            states.temperature.gpu,
            states.top_p.gpu,
            self.expanded,
            floors,
            target_top_k=self.top_k,
            out=self.flags,
        )
        self.ids = ids
        self.values = values
        self.fallback = fallback
        return True

    def _dense(self) -> None:
        assert self.fallback is not None
        logits = self.fallback()
        if logits is None:
            raise RuntimeError(
                "Greedy compact target graph rejection requires replicated logits"
            )
        sampler = self.sampler
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
            None,
            self.inputs,
            self.cu,
            self.positions,
            self.mapping,
            self.expanded,
            self.local,
            states.temperature.gpu,
            states.seeds.gpu,
            self.rejection.num_speculative_steps,
            use_fp64=sampler.use_fp64_gumbel,
        )
        self.output.copy_(output)
        self.count.copy_(count)

    def _compact(self) -> None:
        assert self.ids is not None and self.values is not None
        sampler = self.sampler
        states = sampler.sampling_states
        output, count = greedy_draft_sparse_topk_rejection_sample(
            self.ids,
            self.values,
            self.inputs,
            self.cu,
            self.positions,
            self.mapping,
            states.temperature.gpu,
            states.top_p.gpu,
            states.seeds.gpu,
            self.rejection.num_speculative_steps,
            use_fp64=sampler.use_fp64_gumbel,
            target_top_k=self.top_k,
        )
        self.output.copy_(output)
        self.count.copy_(count)

    def capture(self) -> bool:
        pool_prefix, pool_reference, pool_compact = self.pools
        with graph_capture(device=self.hidden.device):
            if not _all_ranks_support(0 if self._probe() else 801):
                self._clear_transient()
                return False
            self._dense()
            self._compact()
            self._clear_transient()
            torch.accelerator.synchronize(self.hidden.device)
            torch.accelerator.empty_cache()
            with torch.cuda.graph(self.prefix, pool=pool_prefix):
                self._probe()
            with torch.cuda.graph(self.reference, pool=pool_reference):
                self._dense()
            with torch.cuda.graph(self.compact, pool=pool_compact):
                self._compact()
            self._clear_transient()
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
            logger.debug("CUDA rejected a greedy target sampler graph: %s", error)
            status = 801
        return _all_ranks_support(status)

    def replay(self, hidden: torch.Tensor, batch: Any) -> SamplerOutput:
        request_id = tuple(batch.req_ids)
        if request_id != self.request_id:
            self.counters.zero_()
            self.request_id = request_id
        self.prepare(hidden, batch)
        self.prefix.replay()
        return SamplerOutput(
            sampled_token_ids=self.output.clone(),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=self.count.clone(),
        )


def try_graph_greedy_draft_rejection(
    model: Any,
    rejection: Any,
    hidden: torch.Tensor,
    batch: Any,
    *,
    batch_top_k: int,
    pre_capture_only: bool = False,
) -> SamplerOutput | None:
    """Run compact/reference greedy-draft rejection on GPU with no host sync."""
    width = rejection.num_speculative_steps + 1
    requests = getattr(batch, "num_reqs", 0)
    rows = requests * width
    if (
        not 2 <= width <= 16
        or not 1 <= requests <= _MAX_GRAPH_REQUESTS
        or rows > 64
        or batch.num_tokens != rows
        or hidden.shape[0] != rows
        or not np.array_equal(batch.cu_num_logits_np, np.arange(requests + 1) * width)
        or get_tensor_model_parallel_world_size() not in (1, 2, 4, 8)
        or not (
            hasattr(model, "get_compact_target_probe_with_fallback")
            or hasattr(model, "get_topk_tokens_and_logits_with_fallback")
        )
        or not hasattr(torch.ops._C, "sm70_sampler_graph_prepare_collective")
        or rejection.sampler.sampling_states.vocab_size < 32768
        or not all(
            t.is_contiguous()
            for t in (hidden, batch.input_ids[:rows], batch.positions[:rows])
        )
    ):
        return None
    states = rejection.sampler.sampling_states
    idx = batch.idx_mapping_np
    penalties = bool(np.any(rejection.sampler.penalties_state.use_penalty[idx]))
    tensors = (
        hidden,
        batch.input_ids[:rows],
        batch.positions[:rows],
    )
    key = (
        id(model),
        hidden.dtype,
        batch_top_k,
        bool(np.any(states.top_p.np[idx] != 1.0)),
        rejection.sampler.use_fp64_gumbel,
        penalties,
        tuple((t.dtype, tuple(t.shape), tuple(t.stride())) for t in tensors),
    )
    graphs = getattr(rejection, "_sm70_greedy_target_rejection_graphs", None)
    if graphs is None:
        graphs = rejection._sm70_greedy_target_rejection_graphs = {}
    pools = getattr(rejection, "_sm70_greedy_target_graph_pools", None)
    if pools is None:
        pool_branch = torch.cuda.graph_pool_handle()
        pools = rejection._sm70_greedy_target_graph_pools = (
            torch.cuda.graph_pool_handle(),
            pool_branch,
            pool_branch,
        )
    if key not in graphs:
        if len(graphs) >= _MAX_GRAPHS:
            return None
        if not _all_ranks_support(0 if _capture_headroom(hidden.device) else 801):
            graphs[key] = None
            return None
        candidate = _GreedyDraftRejectionGraph(
            model,
            rejection,
            hidden,
            batch,
            top_k=batch_top_k,
            penalties=penalties,
            pools=pools,
        )
        graphs[key] = candidate if candidate.capture() else None
    graph = graphs[key]
    if graph is None or pre_capture_only:
        return None
    rejection.sm70_greedy_target_reference_counts = graph.counters
    logger.info_once(
        "SM70 greedy-draft compact/reference rejection uses device-side graph dispatch."
    )
    return graph.replay(hidden, batch)


def _sparse_target_reference_check_enabled(speculator: Any) -> bool:
    ref_check = getattr(
        speculator.vllm_config.kernel_config,
        "sm70_sparse_target_reference_check",
        None,
    )
    return True if ref_check is None else bool(ref_check)


@torch.inference_mode()
def maybe_capture_sparse_target_graphs(
    speculator: Any,
    model: Any,
    rejection_sampler: Any,
) -> None:
    """Pre-capture common compact/reference rejection graphs during startup."""
    if (
        rejection_sampler is None
        or speculator.draft_logits is not None
        or speculator.device.type != "cuda"
        or torch.cuda.get_device_capability(speculator.device) != (7, 0)
        or not getattr(
            speculator.vllm_config.kernel_config, "sm70_sparse_target_sampling", False
        )
        or not _sparse_target_reference_check_enabled(speculator)
        or not (
            hasattr(model, "get_compact_target_probe_with_fallback")
            or hasattr(model, "get_topk_tokens_and_logits_with_fallback")
        )
    ):
        return

    device = speculator.device
    width = int(rejection_sampler.num_speculative_steps) + 1
    states = rejection_sampler.sampler.sampling_states
    penalties = rejection_sampler.sampler.penalties_state
    vocab_size = int(getattr(states, "vocab_size", 0))
    precapture_limit = _MAX_GRAPH_REQUESTS
    max_reqs = min(
        precapture_limit,
        int(getattr(speculator, "max_num_reqs", 0)),
        64 // max(width, 1),
    )
    if max_reqs < 1 or not 2 <= width <= 16:
        return

    if vocab_size < 32768 or states.max_num_reqs < max_reqs:
        return

    hidden_size = int(speculator.vllm_config.model_config.get_hidden_size())
    dtype = speculator.vllm_config.model_config.dtype
    input_buffers = getattr(speculator, "input_buffers", None)

    saved_temp = states.temperature.np[:max_reqs].copy()
    saved_top_k = states.top_k.np[:max_reqs].copy()
    saved_top_p = states.top_p.np[:max_reqs].copy()
    saved_min_p = states.min_p.np[:max_reqs].copy()
    saved_seeds = states.seeds.np[:max_reqs].copy()
    saved_pres = penalties.presence_penalty.np[:max_reqs].copy()
    saved_use_pen = penalties.use_penalty[:max_reqs].copy()
    try:
        for use_pen, temp, top_p, pres in (
            (False, 1.0, 0.95, 0.0),
            (True, 0.7, 0.80, 1.5),
        ):
            states.temperature.np[:max_reqs] = temp
            states.top_k.np[:max_reqs] = 20
            states.top_p.np[:max_reqs] = top_p
            states.min_p.np[:max_reqs] = 0.0
            states.seeds.np[:max_reqs] = 2026
            states.apply_staged_writes()
            penalties.presence_penalty.np[:max_reqs] = pres
            penalties.presence_penalty.copy_to_uva()
            penalties.use_penalty[:max_reqs] = use_pen

            # Capture from largest batch down to 1 so the shared pools allocate the
            # high-water mark once and smaller request counts reuse the same blocks.
            for requests in range(max_reqs, 0, -1):
                rows = requests * width
                if (
                    input_buffers is not None
                    and input_buffers.input_ids.shape[0] >= rows
                    and input_buffers.positions.shape[0] >= rows
                ):
                    input_ids = input_buffers.input_ids[:rows]
                    positions = input_buffers.positions[:rows]
                else:
                    input_ids = torch.zeros(rows, dtype=torch.int32, device=device)
                    positions = torch.zeros(rows, dtype=torch.int64, device=device)
                mapping_np = np.arange(requests, dtype=np.int32)
                mapping = torch.arange(requests, dtype=torch.int32, device=device)
                cu_np = np.arange(requests + 1, dtype=np.int32) * width
                dummy_batch = SimpleNamespace(
                    num_reqs=requests,
                    num_tokens=rows,
                    req_ids=[f"_sm70_sparse_warmup_{r}" for r in range(requests)],
                    idx_mapping_np=mapping_np,
                    idx_mapping=mapping,
                    expanded_idx_mapping=mapping.repeat_interleave(width),
                    expanded_local_pos=torch.arange(
                        width, dtype=torch.int32, device=device
                    ).repeat(requests),
                    input_ids=input_ids,
                    positions=positions,
                    cu_num_logits_np=cu_np,
                    cu_num_logits=torch.tensor(cu_np, dtype=torch.int32, device=device),
                )
                dummy_hidden = torch.zeros(
                    (rows, hidden_size), dtype=dtype, device=device
                )
                try_graph_greedy_draft_rejection(
                    model,
                    rejection_sampler,
                    dummy_hidden,
                    dummy_batch,
                    batch_top_k=20,
                    pre_capture_only=True,
                )
    finally:
        states.temperature.np[:max_reqs] = saved_temp
        states.top_k.np[:max_reqs] = saved_top_k
        states.top_p.np[:max_reqs] = saved_top_p
        states.min_p.np[:max_reqs] = saved_min_p
        states.seeds.np[:max_reqs] = saved_seeds
        states.apply_staged_writes()
        penalties.presence_penalty.np[:max_reqs] = saved_pres
        penalties.presence_penalty.copy_to_uva()
        penalties.use_penalty[:max_reqs] = saved_use_pen
        torch.accelerator.synchronize(device)
        torch.accelerator.empty_cache()


@torch.inference_mode()
def warmup_sm70_sparse_target(
    device: torch.device,
    vocab_size: int,
    num_speculative_steps: int,
) -> None:
    """Pre-compile Triton sampling and compact rejection kernels during load."""
    if device.type != "cuda" or vocab_size <= 0 or num_speculative_steps <= 0:
        return
    if torch.cuda.get_device_capability(device) != (7, 0):
        return
    key = (int(device.index or 0), int(vocab_size), int(num_speculative_steps))
    if key in _WARMED_SPARSE_TARGET_KEYS:
        return
    _WARMED_SPARSE_TARGET_KEYS.add(key)

    rows = int(num_speculative_steps) + 1
    probe_w = min(int(_TARGET_PROBE_K), int(vocab_size))
    top_k = min(20, probe_w)
    probe_logits = (
        torch.linspace(4.0, 0.0, probe_w, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(rows, -1)
        .contiguous()
    )
    probe_ids = (
        torch.arange(probe_w, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(rows, -1)
        .contiguous()
    )
    sort_topk_with_vocab_ties(
        probe_logits,
        probe_ids,
        vocab_size=vocab_size,
        descending=True,
    )
    wide_w = min(256, int(vocab_size))
    if wide_w > probe_w:
        wide_logits = (
            torch.linspace(4.0, 0.0, wide_w, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(rows, -1)
            .contiguous()
        )
        wide_ids = (
            torch.arange(wide_w, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(rows, -1)
            .contiguous()
        )
        sort_topk_with_vocab_ties(
            wide_logits,
            wide_ids,
            vocab_size=vocab_size,
            descending=True,
            top_k=probe_w,
        )
    wire = torch.empty(probe_logits.numel() * 2, dtype=torch.int32, device=device)
    _pack[(triton.cdiv(probe_logits.numel(), 256),)](
        probe_logits, probe_ids, wire, probe_logits.numel(), False, 256
    )
    _unpack[(triton.cdiv(probe_logits.numel(), 256),)](
        wire, probe_logits, probe_ids, probe_logits.numel(), False, 256
    )
    temperatures = torch.full((16,), 0.6, device=device, dtype=torch.float32)
    top_ps = torch.full((16,), 0.95, device=device, dtype=torch.float32)
    min_ps = torch.zeros(16, device=device, dtype=torch.float32)
    top_ks = torch.full((16,), top_k, device=device, dtype=torch.int32)
    seeds = torch.full((16,), 2026, device=device, dtype=torch.int64)
    row_to_req = torch.zeros(rows, device=device, dtype=torch.int32)
    if probe_w == _TARGET_PROBE_K and top_k == 20:
        _compact_target_reference_flags(
            probe_logits,
            temperatures,
            top_ps,
            row_to_req,
            target_top_k=top_k,
        )
    draft_sampled = torch.zeros(rows, device=device, dtype=torch.int32)
    cu_num_logits = torch.tensor([0, rows], device=device, dtype=torch.int32)
    pos = torch.arange(rows, device=device, dtype=torch.int64)
    idx_mapping = torch.zeros(4, device=device, dtype=torch.int32)
    dummy_hidden = torch.zeros((rows, 64), device=device, dtype=torch.float16)
    for state_sz in (8, 16):
        _prepare_greedy_graph_inputs[(2,)](
            dummy_hidden,
            dummy_hidden,
            dummy_hidden.numel(),
            draft_sampled,
            draft_sampled,
            pos,
            pos,
            idx_mapping,
            idx_mapping,
            row_to_req,
            cu_num_logits,
            cu_num_logits,
            temperatures,
            temperatures,
            top_ks,
            top_ks,
            top_ps,
            top_ps,
            min_ps,
            min_ps,
            seeds,
            seeds,
            temperatures,
            temperatures,
            temperatures,
            temperatures,
            temperatures,
            temperatures,
            state_sz,
            1,
            rows,
            False,
            1024,
            num_warps=4,
        )
    greedy_draft_sparse_topk_rejection_sample(
        probe_ids,
        probe_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping[:1],
        temperatures,
        top_ps,
        seeds,
        num_speculative_steps,
        use_fp64=False,
        target_top_k=top_k,
    )
    cu_single = torch.tensor([0, 1], device=device, dtype=torch.int32)
    greedy_draft_sparse_topk_rejection_sample(
        probe_ids[:1, :top_k].contiguous(),
        probe_logits[:1, :top_k].contiguous(),
        draft_sampled[:1],
        cu_single,
        pos[:1],
        idx_mapping[:1],
        temperatures,
        top_ps,
        seeds,
        num_speculative_steps,
        use_fp64=False,
        target_top_k=top_k,
    )

    # Also pre-compile non-greedy prefill and dense reference-fallback kernels.
    dense_rows = max(rows, 8)
    dense_logits = torch.zeros(
        (dense_rows, vocab_size), device=device, dtype=torch.float32
    )
    dense_req_map = torch.zeros(dense_rows, device=device, dtype=torch.int32)
    apply_temperature(dense_logits[:rows], row_to_req, temperatures)
    if vocab_size >= 32768:
        top_k_vec = torch.full((dense_rows,), top_k, device=device, dtype=torch.int32)
        top_p_vec = torch.full((dense_rows,), 0.95, device=device, dtype=torch.float32)
        apply_top_k_top_p(dense_logits[:2], top_k_vec[:2], top_p_vec[:2])
        if rows not in (2, 8):
            apply_top_k_top_p(dense_logits[:rows], top_k_vec[:rows], top_p_vec[:rows])
        apply_top_k_top_p(dense_logits[:8], top_k_vec[:8], top_p_vec[:8])
    gumbel_sample(
        dense_logits[:1],
        dense_req_map[:1],
        temperatures,
        seeds,
        pos[:1],
        apply_temperature=False,
        is_drafting=False,
        use_fp64=False,
    )
    expanded_local_pos = torch.arange(rows, device=device, dtype=torch.int32)
    rejection_sample(
        dense_logits[:rows],
        None,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping[:1],
        row_to_req,
        expanded_local_pos,
        temperatures,
        seeds,
        num_speculative_steps,
        use_fp64=False,
    )
    rejection_sample(
        dense_logits[:rows],
        None,
        draft_sampled,
        cu_num_logits.to(torch.int64),
        pos,
        idx_mapping[:1],
        row_to_req,
        expanded_local_pos,
        temperatures,
        seeds,
        num_speculative_steps,
        use_fp64=False,
    )


def maybe_warmup_sparse_target(speculator: Any) -> None:
    """Warm compact target sampling kernels for an Eagle/MTP speculator."""
    if speculator.draft_logits is not None or speculator.device.type != "cuda":
        return
    if not getattr(
        speculator.vllm_config.kernel_config, "sm70_sparse_target_sampling", False
    ):
        return
    warmup_sm70_sparse_target(
        device=speculator.device,
        vocab_size=speculator.vllm_config.model_config.get_vocab_size(),
        num_speculative_steps=speculator.num_speculative_steps,
    )


def maybe_sample_sparse_target(
    speculator: Any,
    model: Any,
    rejection_sampler: RejectionSampler,
    sample_hidden_states: torch.Tensor,
    input_batch: InputBatch,
    grammar_output: GrammarOutput | None,
    *,
    allow_graph: bool = True,
) -> SamplerOutput | ComputedTargetLogits | None:
    """Dispatch compact target sampling for an Eagle/MTP speculator."""
    if speculator.draft_logits is not None or not getattr(
        speculator.vllm_config.kernel_config, "sm70_sparse_target_sampling", False
    ):
        return None
    return try_greedy_draft_sparse_rejection(
        model,
        rejection_sampler,
        sample_hidden_states,
        input_batch,
        grammar_output,
        reference_check=_sparse_target_reference_check_enabled(speculator),
        allow_graph=allow_graph,
    )


def try_greedy_draft_sparse_rejection(
    model: Any,
    rejection_sampler: RejectionSampler,
    sample_hidden_states: torch.Tensor,
    input_batch: InputBatch,
    grammar_output: GrammarOutput | None,
    reference_check: bool = True,
    *,
    allow_graph: bool = True,
) -> SamplerOutput | ComputedTargetLogits | None:
    """Sample verified tokens from compact target supports, or return None
    (or the computed dense logits) to keep the full-vocabulary sampler."""
    if grammar_output is not None or input_batch.has_structured_output_reqs:
        return None
    if sample_hidden_states.device.type != "cuda":
        return None
    if torch.cuda.get_device_capability(sample_hidden_states.device) != (7, 0):
        return None
    if not hasattr(model, "get_topk_tokens_and_logits"):
        return None
    if not _supports_sparse_sampling_contract(
        rejection_sampler,
        input_batch,
        allow_penalties=reference_check,
        max_top_k=48,
    ):
        return None

    states = rejection_sampler.sampler.sampling_states
    idx = input_batch.idx_mapping_np
    batch_top_k = int(states.top_k.np[idx[0]])
    if allow_graph and reference_check:
        result = try_graph_greedy_draft_rejection(
            model,
            rejection_sampler,
            sample_hidden_states,
            input_batch,
            batch_top_k=batch_top_k,
        )
        if result is not None:
            return result

    # The reference sampler dispatches on logit rows: a verifier with two or
    # more rows over a large vocabulary uses the radix path's tie order, so
    # keep every cutoff tie (see the DFlash2 contract).
    retain_ties = (
        int(input_batch.cu_num_logits_np[-1]) >= 2
        and getattr(states, "vocab_size", 0) >= 32768
    )
    if not reference_check:
        # No host decision: keep cutoff ties in vocabulary order and decide
        # every row on the compact candidates.
        retain_ties = True
    probe_k = _TARGET_PROBE_K if retain_ties else batch_top_k + 1
    fallback = None
    if not reference_check:
        if retain_ties and hasattr(model, "get_compact_target_probe_with_fallback"):
            target_ids, target_logits, _ = model.get_compact_target_probe_with_fallback(
                sample_hidden_states, probe_k
            )
        else:
            target_ids, target_logits = model.get_topk_tokens_and_logits(
                sample_hidden_states, probe_k
            )
    elif retain_ties and hasattr(model, "get_compact_target_probe_with_fallback"):
        target_ids, target_logits, fallback = (
            model.get_compact_target_probe_with_fallback(sample_hidden_states, probe_k)
        )
    elif hasattr(model, "get_topk_tokens_and_logits_with_fallback"):
        target_ids, target_logits, fallback = (
            model.get_topk_tokens_and_logits_with_fallback(
                sample_hidden_states, probe_k
            )
        )
    else:
        target_ids, target_logits = model.get_topk_tokens_and_logits(
            sample_hidden_states, probe_k
        )
    floors = None
    penalties = rejection_sampler.sampler.penalties_state
    if reference_check and np.any(penalties.use_penalty[idx]):
        floors = _penalize_probe(
            target_logits,
            target_ids,
            penalties,
            input_batch.expanded_idx_mapping,
            input_batch.input_ids[input_batch.logits_indices],
            input_batch.expanded_local_pos,
        )
    if retain_ties:
        target_logits, target_ids = sort_topk_with_vocab_ties(
            target_logits,
            target_ids,
            vocab_size=states.vocab_size,
            descending=True,
        )
    if not reference_check:
        reference_reqs = np.empty(0, dtype=np.int64)
    elif retain_ties:
        reference_rows = _compact_target_reference_rows_gpu(
            target_logits,
            states.temperature.gpu,
            states.top_p.gpu,
            input_batch.expanded_idx_mapping,
            floors,
            target_top_k=batch_top_k,
        )
    else:
        num_logits = np.diff(input_batch.cu_num_logits_np)
        reference_rows = _compact_target_reference_rows(
            target_logits,
            np.repeat(states.temperature.np[idx], num_logits),
            np.repeat(states.top_p.np[idx], num_logits),
            target_top_k=batch_top_k,
        )
    if reference_check:
        reference_reqs = np.flatnonzero(
            np.logical_or.reduceat(reference_rows, input_batch.cu_num_logits_np[:-1])
        )
    reference_logits = None
    if reference_reqs.size:
        if fallback is None:
            return None
        reference_logits = fallback()
        if reference_reqs.size == idx.size or reference_logits is None:
            return ComputedTargetLogits(reference_logits)
    if not retain_ties:
        target_ids = target_ids[:, :batch_top_k]
        target_logits = target_logits[:, :batch_top_k]
    rows = target_ids.shape[0]
    if input_batch.num_tokens == rows:
        draft_sampled = input_batch.input_ids[:rows]
        pos = input_batch.positions[:rows]
    else:
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
    sampled, num_sampled = greedy_draft_sparse_topk_rejection_sample(
        target_ids,
        target_logits,
        draft_sampled,
        input_batch.cu_num_logits,
        pos,
        input_batch.idx_mapping,
        states.temperature.gpu,
        states.top_p.gpu,
        states.seeds.gpu,
        rejection_sampler.num_speculative_steps,
        use_fp64=rejection_sampler.sampler.use_fp64_gumbel,
        target_top_k=batch_top_k,
    )
    if reference_logits is not None:
        reqs, dense_sampled, dense_num_sampled = _sample_reference_requests(
            reference_logits,
            reference_reqs,
            input_batch,
            rejection_sampler,
            None,
        )
        sampled.index_copy_(0, reqs, dense_sampled)
        num_sampled.index_copy_(0, reqs, dense_num_sampled)
    logger.info_once("Using SM70 compact top-k target sampling for greedy drafts.")
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
    )
