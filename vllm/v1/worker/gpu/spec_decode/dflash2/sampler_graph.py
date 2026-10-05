# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side compact/reference dispatch for a uniform DFlash2 verifier."""

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


@triton.jit
def _prepare_inputs(
    hidden,
    fixed_hidden,
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
    dense,
    dense_pointer,
    hidden_size: tl.constexpr,
    sparse_size: tl.constexpr,
    source_sparse_stride: tl.constexpr,
    fixed_sparse_stride: tl.constexpr,
    block: tl.constexpr,
):
    request = tl.load(source_mapping)
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    tl.store(
        fixed_hidden + offsets,
        tl.load(hidden + offsets, offsets < hidden_size, 0),
        offsets < hidden_size,
    )
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
    if tl.program_id(0) == 0:
        tl.store(mapping, request)
        tl.store(expanded + offsets, request, offsets < 8)
        tl.store(
            fixed_inputs + offsets,
            tl.load(inputs + offsets, offsets < 8, 0),
            offsets < 8,
        )
        tl.store(
            fixed_positions + offsets,
            tl.load(positions + offsets, offsets < 8, 0),
            offsets < 8,
        )
        tl.store(dense_pointer, dense.to(tl.int64))


@triton.jit
def _copy_dense_slot(
    source_pointer, destination, mapping, slot_size: tl.constexpr, block: tl.constexpr
):
    # The source address is refreshed with the current request's inputs.
    # Only the reference branch reads dense proposals; compact rounds incur
    # no full-vocabulary copy, and inactive request slots are never copied.
    source = tl.load(source_pointer).to(destination.dtype)
    request = tl.load(mapping)
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(source + request * slot_size + offsets, offsets < slot_size, 0)
    tl.store(destination + offsets, values, offsets < slot_size)


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
    def __init__(self, model, speculator, rejection, hidden, batch, sparse):
        self.hidden = torch.empty_like(hidden)
        self.inputs = torch.empty_like(batch.input_ids[:8])
        self.positions = torch.empty_like(batch.positions[:8])
        # MRv2 allocates expanded metadata anew every round. Keep graph inputs
        # independent of those allocations and preserve the original slot ID.
        self.mapping = batch.idx_mapping[:1].clone()
        self.expanded = batch.expanded_idx_mapping[:8].clone()
        self.local = torch.arange(
            8, dtype=batch.expanded_local_pos.dtype, device=hidden.device
        )
        self.cu = batch.cu_num_logits[:2].clone()
        self.request_indices = batch.idx_mapping_np.copy()
        self.rejection = rejection
        self.model = model
        self.speculator = speculator
        self.sparse = tuple(torch.empty_like(t) for t in sparse)
        dense = speculator.draft_logits
        # Only one request is captured. Alias its private proposal slot for
        # every state index while preserving the original request mapping.
        # Dense kernels accept stride(0)==0; no other request slot is read.
        self.draft_slot = dense.new_empty((1, *dense.shape[1:]))
        self.draft_logits = self.draft_slot.expand(dense.shape[0], -1, -1)
        self.dense_pointer = self.inputs.new_empty(1, dtype=torch.int64)
        self.output = self.inputs.new_empty((1, 8), dtype=torch.int64)
        self.count = self.inputs.new_empty(1, dtype=torch.int32)
        self.counters = self.inputs.new_zeros(2, dtype=torch.int64)
        self.request_id = None
        self.prefix = torch.cuda.CUDAGraph(keep_graph=True)
        self.reference = torch.cuda.CUDAGraph(keep_graph=True)
        self.compact = torch.cuda.CUDAGraph(keep_graph=True)
        self.prepare(hidden, batch, sparse)

    def prepare(self, hidden, batch, sparse):
        ids, values = sparse
        fixed_ids, fixed_values = self.sparse
        _prepare_inputs[(triton.cdiv(hidden.numel(), 1024),)](
            hidden,
            self.hidden,
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
            self.speculator.draft_logits,
            self.dense_pointer,
            hidden.numel(),
            ids.shape[1] * ids.shape[2],
            ids.stride(0),
            fixed_ids.stride(0),
            1024,
            num_warps=4,
        )
        # Dense proposals are accessed indirectly inside a conditional body.
        # Record their consumer stream for replaced/eager tensor lifetimes.
        self.speculator.draft_logits.record_stream(torch.cuda.current_stream())

    def _probe(self):
        from .sparse_rejection import _compact_target_reference_flags

        probe = getattr(
            self.model,
            "get_compact_target_probe_with_fallback",
            self.model.get_topk_tokens_and_logits_with_fallback,
        )
        self.ids, self.values, self.fallback = probe(self.hidden, 64)
        if self.fallback is None:
            return False
        states = self.rejection.sampler.sampling_states
        self.values, self.ids = sort_topk_with_vocab_ties(
            self.values, self.ids, vocab_size=states.vocab_size, descending=True
        )
        self.flags = _compact_target_reference_flags(
            self.values, states.temperature.gpu, states.top_p.gpu, self.expanded
        )
        return True

    def _dense(self):
        slot_size = self.draft_logits.shape[1] * self.draft_logits.shape[2]
        _copy_dense_slot[(triton.cdiv(slot_size, 4096),)](
            self.dense_pointer,
            self.draft_logits,
            self.mapping,
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
            states.temperature.gpu,
            states.seeds.gpu,
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
            states.temperature.gpu,
            states.top_p.gpu,
            states.seeds.gpu,
            self.rejection.num_speculative_steps,
            use_fp64=sampler.use_fp64_gumbel,
            target_top_k=20,
        )
        self.output.copy_(output)
        self.count.copy_(count)

    def capture(self):
        # Sampling is stateless in seed/position. Warmup writes only the private
        # outputs and does not advance request state or proposal random streams.
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
        request_id = batch.req_ids[0]
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
        getattr(batch, "num_reqs", 0) != 1
        or batch.num_tokens != 8
        or hidden.shape[0] != 8
        or rejection.num_speculative_steps != 7
        or not np.array_equal(batch.cu_num_logits_np, [0, 8])
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
        tuple((t.dtype, tuple(t.shape), tuple(t.stride())) for t in tensors),
    )
    graphs = getattr(rejection, "_sm70_dflash2_rejection_graphs", None)
    if graphs is None:
        graphs = rejection._sm70_dflash2_rejection_graphs = {}
    if key not in graphs:
        if len(graphs) >= 8:
            # Bound graph storage for unexpected shape/contract combinations.
            return None
        candidate = _RejectionGraph(model, speculator, rejection, hidden, batch, sparse)
        graphs[key] = candidate if candidate.capture() else None
    graph = graphs[key]
    if graph is None:
        return None
    rejection.sm70_dflash2_reference_counts = graph.counters
    logger.info_once(
        "DFlash2 compact/reference rejection uses device-side graph dispatch."
    )
    return graph.replay(hidden, batch, sparse)
