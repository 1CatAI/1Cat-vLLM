# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strictly gated compact target rejection for DFlash2 on SM70."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    dflash2_sparse_topk_rejection_sample,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

logger = init_logger(__name__)

_TARGET_TOP_K = 20
_SELECTOR_ALIGNMENT_DUMP_COUNT = 0
_SELECTOR_ALIGNMENT_STEP = 0


def _parse_alignment_steps(raw_steps: str | None) -> set[int] | None:
    if not raw_steps:
        return None
    steps: set[int] = set()
    try:
        for item in raw_steps.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if start < 0 or end < start:
                    return set()
                steps.update(range(start, end + 1))
            else:
                step = int(item)
                if step < 0:
                    return set()
                steps.add(step)
    except ValueError:
        return set()
    return steps


def _safe_dump_tag(raw_tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw_tag)


def _diagnostic_rank() -> int:
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))


def _maybe_dump_selector_alignment(
    *,
    speculator: DFlash2Speculator,
    rejection_sampler: RejectionSampler,
    input_batch: InputBatch,
    target_topk_ids: torch.Tensor,
    target_topk_logits: torch.Tensor,
    draft_topk_ids: torch.Tensor,
    draft_topk_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
) -> None:
    """Dump one exact B1 selector/target alignment record when requested."""
    if not envs.VLLM_SPEC_DUMP_ALIGNMENT:
        return
    if _diagnostic_rank() != 0:
        return

    global _SELECTOR_ALIGNMENT_DUMP_COUNT, _SELECTOR_ALIGNMENT_STEP
    _SELECTOR_ALIGNMENT_STEP += 1
    if _SELECTOR_ALIGNMENT_DUMP_COUNT >= envs.VLLM_SPEC_DUMP_ALIGNMENT_LIMIT:
        return
    selected_steps = _parse_alignment_steps(envs.VLLM_SPEC_DUMP_ALIGNMENT_STEPS)
    if selected_steps is not None and _SELECTOR_ALIGNMENT_STEP not in selected_steps:
        return

    shadow = speculator.get_selector_alignment_shadow()
    if shadow is None:
        return
    shadow_ids, unary_logits, lattice_scores = shadow
    req_state = int(input_batch.idx_mapping_np[0])
    packed_row = 0
    sampling_states = rejection_sampler.sampler.sampling_states

    with torch.no_grad():
        draft_sampled_cpu = draft_sampled.detach().cpu()
        if bool(torch.all(draft_sampled_cpu == 0).item()):
            return
        payload = {
            "format": "dflash2_selector_alignment_v1",
            "rank": _diagnostic_rank(),
            "step": _SELECTOR_ALIGNMENT_STEP,
            "request_state": req_state,
            "selector_top_k": speculator.selector_top_k,
            "num_speculative_steps": speculator.num_speculative_steps,
            "target_topk_ids": target_topk_ids.detach().cpu(),
            "target_topk_logits": target_topk_logits.detach().float().cpu(),
            "draft_candidate_ids": draft_topk_ids[req_state].detach().cpu(),
            "draft_realized_logits": (
                draft_topk_logits[req_state].detach().float().cpu()
            ),
            "selector_candidate_ids": shadow_ids[packed_row].detach().cpu(),
            "selector_unary_logits": (unary_logits[packed_row].detach().float().cpu()),
            "selector_lattice_scores": (
                lattice_scores[packed_row].detach().float().cpu()
            ),
            "draft_sampled": draft_sampled_cpu,
            "positions": pos.detach().cpu(),
            "cu_num_logits": input_batch.cu_num_logits.detach().cpu(),
            "idx_mapping": input_batch.idx_mapping.detach().cpu(),
            "temperature": float(sampling_states.temperature.np[req_state]),
            "top_p": float(sampling_states.top_p.np[req_state]),
            "top_k": int(sampling_states.top_k.np[req_state]),
            "sampled_token_ids": sampled.detach().cpu(),
            "num_sampled": num_sampled.detach().cpu(),
        }
        _SELECTOR_ALIGNMENT_DUMP_COUNT += 1
        dump_dir = os.getenv("VLLM_SPEC_DUMP_ALIGNMENT_DIR", "/tmp")
        os.makedirs(dump_dir, exist_ok=True)
        tag = _safe_dump_tag(os.getenv("VLLM_SPEC_DUMP_ALIGNMENT_TAG", ""))
        tag_part = f"{tag}_" if tag else ""
        dump_path = os.path.join(
            dump_dir,
            f"spec_alignment_dflash2_selector_{tag_part}pid{os.getpid()}_"
            f"step{_SELECTOR_ALIGNMENT_STEP:06d}_"
            f"{_SELECTOR_ALIGNMENT_DUMP_COUNT}.pt",
        )
        torch.save(payload, dump_path)
        logger.warning("Dumped DFlash2 selector alignment diagnostics to %s", dump_path)


def _supports_sparse_sampling_contract(
    rejection_sampler: RejectionSampler,
    input_batch: InputBatch,
) -> bool:
    """Whether compact logits preserve every requested sampling transform."""
    if rejection_sampler.rejection_sample_method != "standard":
        return False
    # Start with the single-request path used by the latency target. The
    # kernel supports batches, but mixed-request graph validation is a
    # separate promotion gate.
    if input_batch.num_reqs != 1 or np.any(input_batch.is_prefilling_np):
        return False

    sampler = rejection_sampler.sampler
    idx = input_batch.idx_mapping_np
    states = sampler.sampling_states
    temperatures = states.temperature.np[idx]
    top_k = states.top_k.np[idx]
    top_p = states.top_p.np[idx]
    if np.any(temperatures <= 0.0):
        return False
    if np.any(top_k != _TARGET_TOP_K):
        return False
    if np.any((top_p <= 0.0) | (top_p > 1.0)):
        return False
    if np.any(states.min_p.np[idx] != 0.0):
        return False

    if np.any(sampler.penalties_state.use_penalty[idx]):
        return False
    if np.any(sampler.logit_bias_state.use_logit_bias[idx]):
        return False
    if np.any(sampler.bad_words_state.num_bad_words.np[idx] != 0):
        return False
    if states.max_num_logprobs(idx) != NO_LOGPROBS:
        return False
    if sampler.logprob_token_ids_state.max_num_token_ids(idx) != 0:
        return False
    return not sampler.compute_nans


def try_dflash2_sparse_target_rejection(
    model: Any,
    speculator: Any,
    rejection_sampler: RejectionSampler,
    sample_hidden_states: torch.Tensor,
    input_batch: InputBatch,
    grammar_output: GrammarOutput | None,
) -> SamplerOutput | None:
    """Sample from compact target/draft supports, or return ``None`` safely."""
    if not envs.VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION:
        return None
    if not isinstance(speculator, DFlash2Speculator):
        return None
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

    sparse_draft_logits = speculator.get_sparse_draft_logits()
    if sparse_draft_logits is None:
        return None
    draft_topk_ids, draft_topk_logits = sparse_draft_logits
    target_topk_ids, target_topk_logits = model.get_topk_tokens_and_logits(
        sample_hidden_states,
        _TARGET_TOP_K,
    )
    draft_sampled = input_batch.input_ids[input_batch.logits_indices]
    pos = input_batch.positions[input_batch.logits_indices]
    sampled, num_sampled = dflash2_sparse_topk_rejection_sample(
        target_topk_ids,
        target_topk_logits,
        draft_topk_ids,
        draft_topk_logits,
        draft_sampled,
        input_batch.cu_num_logits,
        pos,
        input_batch.idx_mapping,
        rejection_sampler.sampler.sampling_states.temperature.gpu,
        rejection_sampler.sampler.sampling_states.top_p.gpu,
        rejection_sampler.sampler.sampling_states.seeds.gpu,
        rejection_sampler.num_speculative_steps,
        use_fp64=rejection_sampler.sampler.use_fp64_gumbel,
    )
    if envs.VLLM_SPEC_DUMP_ALIGNMENT:
        _maybe_dump_selector_alignment(
            speculator=speculator,
            rejection_sampler=rejection_sampler,
            input_batch=input_batch,
            target_topk_ids=target_topk_ids,
            target_topk_logits=target_topk_logits,
            draft_topk_ids=draft_topk_ids,
            draft_topk_logits=draft_topk_logits,
            draft_sampled=draft_sampled,
            pos=pos,
            sampled=sampled,
            num_sampled=num_sampled,
        )
    logger.info_once("Using SM70 DFlash2 compact target top-k rejection sampling.")
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
    )
