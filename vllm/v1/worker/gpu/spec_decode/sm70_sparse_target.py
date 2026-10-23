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
from vllm.v1.sample.ops.topk_topp_triton import sort_topk_with_vocab_ties
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
    _TARGET_PROBE_K,
    _TARGET_TOP_K,
    _compact_target_reference_rows,
    _compact_target_reference_rows_gpu,
    _sample_reference_requests,
    _supports_sparse_sampling_contract,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    greedy_draft_sparse_topk_rejection_sample,
)
from vllm.v1.worker.gpu.spec_decode.target_sampling import ComputedTargetLogits

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

logger = init_logger(__name__)


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
    if not _supports_sparse_sampling_contract(rejection_sampler, input_batch):
        return None

    states = rejection_sampler.sampler.sampling_states
    idx = input_batch.idx_mapping_np
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
    probe_k = _TARGET_PROBE_K if retain_ties else _TARGET_TOP_K + 1
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
        )
    else:
        num_logits = np.diff(input_batch.cu_num_logits_np)
        reference_rows = _compact_target_reference_rows(
            target_logits,
            np.repeat(states.temperature.np[idx], num_logits),
            np.repeat(states.top_p.np[idx], num_logits),
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
        target_ids = target_ids[:, :_TARGET_TOP_K]
        target_logits = target_logits[:, :_TARGET_TOP_K]
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
        target_top_k=_TARGET_TOP_K,
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
