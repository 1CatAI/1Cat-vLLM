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

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.ops.topk_topp_triton import sort_topk_with_vocab_ties
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
    _TARGET_PROBE_K,
    _compact_target_reference_flags,
    _compact_target_reference_rows,
    _compact_target_reference_rows_gpu,
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
    temperatures = torch.tensor([0.6], device=device, dtype=torch.float32)
    top_ps = torch.tensor([0.95], device=device, dtype=torch.float32)
    seeds = torch.tensor([2026], device=device, dtype=torch.int64)
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
    idx_mapping = torch.zeros(1, device=device, dtype=torch.int32)
    greedy_draft_sparse_topk_rejection_sample(
        probe_ids,
        probe_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
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
        idx_mapping,
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
        idx_mapping,
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
        idx_mapping,
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
        reference_check=speculator.vllm_config.kernel_config.sm70_sparse_target_reference_check,
    )


def try_greedy_draft_sparse_rejection(
    model: Any,
    rejection_sampler: RejectionSampler,
    sample_hidden_states: torch.Tensor,
    input_batch: InputBatch,
    grammar_output: GrammarOutput | None,
    reference_check: bool = True,
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
        rejection_sampler, input_batch, max_top_k=48
    ):
        return None

    states = rejection_sampler.sampler.sampling_states
    idx = input_batch.idx_mapping_np
    batch_top_k = int(states.top_k.np[idx[0]])
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
