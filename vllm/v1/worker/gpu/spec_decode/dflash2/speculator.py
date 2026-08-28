# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any

import numpy as np
import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import (
    _point_mass_draft_logits_kernel,
    fuse_draft,
    suffix_lookup,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.ngram_assist import DFlash2NgramAssist

logger = init_logger(__name__)


def _requires_sm70_tail(device: torch.device, num_steps: int) -> bool:
    """Whether the final dependent selector slot needs its own kernel."""
    return (
        device.type == "cuda"
        and num_steps > 1
        and torch.cuda.get_device_capability(device) == (7, 0)
    )


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    path_state_ptr,
    num_steps: tl.constexpr,
    walk_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(walk_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        if SAMPLE_PROBABILISTIC and temperature != 0.0:
            # Cache the exact temperature-applied proposal scores expected by
            # the shared rejection sampler. This keeps Eagle/MTP's established
            # contract unchanged while matching the DFlash2 selector draw.
            scores = scores / temperature
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # Candidate token IDs key the noise, matching the target sampler.
        position = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            position,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=False,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index

    if walk_steps < num_steps:
        tl.store(path_state_ptr + row, previous, mask=valid)


@triton.jit
def _selector_walk_tail_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    path_state_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """Write the final dependent slot separately on SM70.

    Triton can drop the seventh store of a fully unrolled selector walk during
    CUDA Graph replay on Volta. The first six slots remain fused; this tiny tail
    consumes their persistent path state and guarantees the seventh write.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = tl.load(path_state_ptr + row, mask=valid, other=0)
    step: tl.constexpr = num_steps - 1
    flat = row * num_steps + step
    score_base = (flat * top_k + previous) * top_k
    scores = tl.load(
        scores_ptr + score_base + offsets,
        mask=mask & valid,
        other=float("-inf"),
    ).to(tl.float64 if USE_FP64 else tl.float32)
    if SAMPLE_PROBABILISTIC and temperature != 0.0:
        scores = scores / temperature
    candidate_base = flat * top_k
    candidates = tl.load(
        candidate_ptr + candidate_base + offsets,
        mask=mask & valid,
        other=0,
    )
    position = tl.load(sample_pos_ptr + flat) - 1
    _, index = gumbel_noised_argmax(
        scores,
        candidates,
        mask & valid,
        seed,
        position,
        temperature if SAMPLE_PROBABILISTIC else 0.0,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    tl.store(
        realized_scores_ptr + candidate_base + offsets,
        scores,
        mask=mask & valid,
    )
    token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
    tl.store(tokens_ptr + flat, token, mask=valid)


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    cached_score_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    cache_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_SCORES: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * cache_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)
    if CACHE_SCORES:
        tl.store(cached_score_ptr + cache_base + offsets, scores, mask=mask)


@triton.jit
def _apply_ngram_draft_kernel(
    ngram_tokens_ptr,
    ngram_lengths_ptr,
    sample_req_state_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cached_candidate_ptr,
    cached_score_ptr,
    cache_stride_0,
    cache_stride_1,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_DRAFT_LOGITS: tl.constexpr,
    CACHE_SCORES: tl.constexpr,
):
    flat = tl.program_id(0)
    batch_idx = flat // num_steps
    step = flat % num_steps
    req_state = tl.load(sample_req_state_ptr + flat)
    valid = (req_state >= 0) & (tl.load(ngram_lengths_ptr + batch_idx) == num_steps)
    token = tl.load(ngram_tokens_ptr + flat, mask=valid, other=0).to(tl.int64)
    tl.store(
        draft_tokens_ptr + batch_idx * draft_tokens_stride + step,
        token,
        mask=valid,
    )

    if CACHE_DRAFT_LOGITS:
        offsets = tl.arange(0, BLOCK_K)
        topk_mask = valid & (offsets < top_k)
        cache_base = (
            cached_candidate_ptr + req_state * cache_stride_0 + step * cache_stride_1
        )
        old_ids = tl.load(cache_base + offsets, mask=topk_mask, other=0)
        logits_base = (
            draft_logits_ptr
            + req_state * draft_logits_stride_0
            + step * draft_logits_stride_1
        )
        tl.store(logits_base + old_ids, -float("inf"), mask=topk_mask)

        is_proposal = offsets == 0
        new_ids = tl.where(is_proposal, token, 0)
        new_scores = tl.where(is_proposal, 0.0, -float("inf"))
        tl.store(cache_base + offsets, new_ids, mask=topk_mask)
        if CACHE_SCORES:
            score_base = (
                cached_score_ptr + req_state * cache_stride_0 + step * cache_stride_1
            )
            tl.store(score_base + offsets, new_scores, mask=topk_mask)
        tl.store(logits_base + token, 0.0, mask=valid)


@triton.jit
def _prepare_lookup_controller_flags_kernel(
    take_flags_ptr,
    emitted_ptr,
    out_ptr,
    full_emitted,
    num_reqs,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_reqs
    take = tl.load(take_flags_ptr + offsets, mask=mask, other=0) > 0
    emitted = tl.load(emitted_ptr + offsets, mask=mask, other=0)
    wants_long = take & (emitted >= full_emitted)
    tl.store(out_ptr + offsets, wants_long.to(tl.int32), mask=mask)


@triton.jit
def _prepare_chain_entry_flags_kernel(
    match_len_ptr,
    out_ptr,
    min_match,
    num_reqs,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_reqs
    match_len = tl.load(match_len_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, (match_len >= min_match).to(tl.int32), mask=mask)


def _advance_lookup_controller(
    *,
    want: bool | None,
    num_reqs: int,
    entry_streak: int,
    sticky_steps: int,
    last_want: bool,
    want_streak: int,
    sticky_remaining: int,
    long_active: bool,
) -> tuple[bool, int, int, bool]:
    """Advance the host-only adaptive q8/q16 controller.

    ``want=None`` means the asynchronous flag copy has not landed. Preserve
    the prior decision in that case instead of synchronizing the decode
    stream merely to choose a verification width.
    """
    if want is None:
        return last_want, want_streak, sticky_remaining, long_active

    if want:
        want_streak = want_streak + 1 if last_want else 1
        if want_streak >= max(entry_streak, 1):
            long_active = True
            sticky_remaining = max(sticky_steps, 0) if num_reqs == 1 else 0
    else:
        want_streak = 0
        if num_reqs == 1 and long_active and sticky_remaining > 0:
            sticky_remaining -= 1
        else:
            long_active = False
            sticky_remaining = 0
    return want, want_streak, sticky_remaining, long_active


def _advance_chain_controller(
    *,
    active: bool,
    previous_was_chain: bool,
    rejected: bool | None,
    entry_evidence: bool | None,
) -> tuple[bool, bool, bool]:
    """Advance a single-request drafter-free chain.

    A missing asynchronous verdict preserves an active chain. After the first
    rejection, one normal draft step is required before fresh lookup evidence
    may start another chain.
    """
    if active:
        if rejected:
            return False, True, False
        return True, True, True
    if previous_was_chain:
        return False, False, False
    if entry_evidence:
        return True, True, True
    return False, False, False


def _chain_rejection_for_controller(
    verdict: tuple[bool, bool] | None,
) -> bool | None:
    """Filter an async rejection verdict by the proposal that produced it.

    The D2H copy completes one host decision later. A rejection from the
    normal neural-draft step that admitted a chain must not be applied to the
    first drafter-free proposal. Only a verdict tagged as coming from a chain
    proposal may terminate an active chain.
    """
    if verdict is None:
        return None
    rejected, proposal_was_chain = verdict
    return rejected if proposal_was_chain else None


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._selector_tokens = torch.empty(
            self.max_num_reqs,
            self.draft_block,
            dtype=self.draft_tokens.dtype,
            device=device,
        )
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.draft_block,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.int64,
            device=device,
        )
        self._cached_candidate_scores = None
        if (
            self.draft_logits is not None
            and envs.VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION
        ):
            self._cached_candidate_scores = torch.full(
                self._cached_candidate_ids.shape,
                -float("inf"),
                dtype=torch.float32,
                device=device,
            )
        self._selector_path_state = torch.empty(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self._alignment_candidate_ids: torch.Tensor | None = None
        self._alignment_unary_logits: torch.Tensor | None = None
        self._alignment_lattice_scores: torch.Tensor | None = None
        if (
            envs.VLLM_SPEC_DUMP_ALIGNMENT
            and envs.VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION
        ):
            packed_shape = (
                self.max_num_reqs,
                self.draft_block,
                self.selector_top_k,
            )
            self._alignment_candidate_ids = torch.empty(
                packed_shape, dtype=torch.int64, device=device
            )
            self._alignment_unary_logits = torch.empty(
                packed_shape, dtype=torch.float32, device=device
            )
            self._alignment_lattice_scores = torch.empty(
                (*packed_shape, self.selector_top_k),
                dtype=torch.float32,
                device=device,
            )
        self._use_sm70_tail = _requires_sm70_tail(device, self.draft_block)
        if self.draft_logits is not None:
            # The cache kernel writes only K columns; all other vocabulary
            # columns must remain impossible.
            self.draft_logits.fill_(-float("inf"))

        self._ngram_assist: DFlash2NgramAssist | None = None
        self._ngram_num_hits = 0
        self._ngram_rounds = 0
        self._ngram_skipped_rounds = 0
        speculative_config = getattr(self, "speculative_config", None)
        ngram_assist = bool(
            speculative_config is not None
            and getattr(speculative_config, "ngram_assist", False)
        )
        if (
            speculative_config is not None
            and ngram_assist
            and self.draft_block == self.num_speculative_steps
        ):
            min_ngram = speculative_config.prompt_lookup_min
            max_ngram = speculative_config.prompt_lookup_max
            assert min_ngram is not None and max_ngram is not None
            self._ngram_assist = DFlash2NgramAssist(
                min_ngram=min_ngram,
                max_ngram=max_ngram,
                num_draft_tokens=self.num_speculative_steps,
                max_model_len=self.max_model_len,
            )
            self._ngram_tokens_cpu_tensor = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            self._ngram_lengths_cpu_tensor = torch.zeros(
                self.max_num_reqs,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            self._ngram_tokens_cpu = self._ngram_tokens_cpu_tensor.numpy()
            self._ngram_lengths_cpu = self._ngram_lengths_cpu_tensor.numpy()
            self._ngram_tokens = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.int64,
                device=device,
            )
            self._ngram_lengths = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device
            )
            logger.info(
                "Enabled DFlash2 ngram assist with prompt lookup [%d, %d] "
                "and draft width %d.",
                min_ngram,
                max_ngram,
                self.num_speculative_steps,
            )

        self._lookup_enabled = bool(
            ngram_assist and self.draft_block < self.num_speculative_steps
        )
        self._req_states = None
        self._lookup_current_req_key: tuple[str, ...] = ()
        self._lookup_current_eligible = False
        self._lookup_last_emitted: torch.Tensor | None = None
        self._sampling_states = None
        self._chain_enabled = False
        self._chain_active = False
        self._chain_previous_was_chain = False
        self._chain_req_key: tuple[str, ...] = ()
        self._chain_steps_total = 0
        self._chain_steps_engaged = 0
        if self._lookup_enabled:
            if device.type != "cuda":
                raise ValueError("Lookup-augmented DFlash2 requires a CUDA device")
            assert speculative_config is not None
            assert speculative_config.prompt_lookup_min is not None
            assert speculative_config.prompt_lookup_max is not None
            self._lookup_nmin = int(speculative_config.prompt_lookup_min)
            self._lookup_nmax = int(speculative_config.prompt_lookup_max)
            self._lookup_nstrong = envs.VLLM_DFLASH2_LOOKUP_NSTRONG
            self._lookup_agree = envs.VLLM_DFLASH2_LOOKUP_AGREE
            self._lookup_nmin_tail = envs.VLLM_DFLASH2_LOOKUP_NMIN_TAIL
            self._lookup_long_min = envs.VLLM_DFLASH2_LOOKUP_LONG_MIN
            self._lookup_search = envs.VLLM_DFLASH2_LOOKUP_SEARCH
            self._lookup_adaptive = envs.VLLM_DFLASH2_LOOKUP_ADAPTIVE
            self._lookup_entry_streak = envs.VLLM_DFLASH2_LOOKUP_ENTRY_STREAK
            self._lookup_sticky_steps = envs.VLLM_DFLASH2_LOOKUP_STICKY
            self._lookup_cheap_context = envs.VLLM_DFLASH2_LOOKUP_CHEAP_CONTEXT
            self._lookup_tokens = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.int32,
                device=device,
            )
            self._lookup_match_len = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device
            )
            self._lookup_valid = torch.zeros_like(self._lookup_match_len)
            self._lookup_eligible = torch.zeros_like(self._lookup_match_len)
            self._lookup_use = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.int32,
                device=device,
            )
            self._lookup_take_flags = torch.zeros_like(self._lookup_match_len)
            self._lookup_controller_flags = torch.zeros_like(self._lookup_match_len)
            self._lookup_hits = torch.zeros((), dtype=torch.int64, device=device)
            self._lookup_copy_stream = torch.cuda.Stream(device=device)
            self._lookup_copy_event = torch.cuda.Event()
            self._lookup_flags_cpu = torch.zeros(
                self.max_num_reqs,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            self._chain_enabled = envs.VLLM_DFLASH2_CHAIN
            if self._chain_enabled:
                self._chain_min_match = envs.VLLM_DFLASH2_CHAIN_MINMATCH
                self._chain_greedy_only = envs.VLLM_DFLASH2_CHAIN_GREEDY_ONLY
                self._chain_log_sec = envs.VLLM_DFLASH2_CHAIN_LOG_SEC
                self._chain_entry_flags = torch.zeros_like(self._lookup_match_len)
                self._chain_entry_flags_cpu = torch.zeros(
                    self.max_num_reqs,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
                self._chain_rejected = torch.zeros_like(self._lookup_match_len)
                self._chain_rejected_cpu = torch.zeros(
                    self.max_num_reqs,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
                self._chain_reject_stream = torch.cuda.Stream(device=device)
                self._chain_reject_event = torch.cuda.Event()
                self._chain_reject_pending = False
                self._chain_reject_req_key: tuple[str, ...] = ()
                self._chain_reject_pending_was_chain = False
                self._chain_last_proposal_was_chain = False
                self._chain_last_log = time.monotonic()
            self._lookup_copy_pending = False
            self._lookup_pending_req_key: tuple[str, ...] = ()
            self._lookup_pending_num_reqs = 0
            self._lookup_last_want = False
            self._lookup_want_streak = 0
            self._lookup_sticky_remaining = 0
            self._lookup_long_active = False
            self._lookup_controller_req_key: tuple[str, ...] = ()
            self._lookup_last_verify_tokens = 0
            self._lookup_q8_rounds = 0
            self._lookup_q16_rounds = 0
            logger.info(
                "Enabled GPU lookup-augmented DFlash2: model drafts=%d, "
                "max target drafts=%d, ngram=[%d,%d], adaptive=%s.",
                self.draft_block,
                self.num_speculative_steps,
                self._lookup_nmin,
                self._lookup_nmax,
                self._lookup_adaptive,
            )
            if self._chain_enabled:
                logger.info(
                    "Enabled drafter-free DFlash2 chains "
                    "(min_match=%d, greedy_only=%s).",
                    self._chain_min_match,
                    self._chain_greedy_only,
                )
        elif envs.VLLM_DFLASH2_CHAIN:
            logger.warning(
                "VLLM_DFLASH2_CHAIN=1 requires lookup-augmented DFlash2 "
                "(ngram_assist=true and a verifier wider than the checkpoint); "
                "chains are disabled."
            )

    @property
    def requires_host_token_state(self) -> bool:
        """Whether the runner must expose async samples and request history."""
        return self._ngram_assist is not None

    def set_req_states(self, req_states) -> None:
        """Expose the request token history used by the device lookup."""
        self._req_states = req_states

    def set_sampling_states(self, sampling_states) -> None:
        """Expose the host sampling view for the zero-sync chain gate."""
        self._sampling_states = sampling_states

    def _reset_lookup_controller(self, req_key: tuple[str, ...]) -> None:
        self._lookup_controller_req_key = req_key
        self._lookup_last_want = False
        self._lookup_want_streak = 0
        self._lookup_sticky_remaining = 0
        self._lookup_long_active = False

    def _record_lookup_width(self, draft_tokens: int, reason: str) -> int:
        """Record the asynchronously selected target verification width."""
        verify_tokens = 1 + draft_tokens
        if draft_tokens == self.draft_block:
            self._lookup_q8_rounds += 1
        else:
            self._lookup_q16_rounds += 1
        if (
            envs.VLLM_DFLASH_PROFILE
            and verify_tokens != self._lookup_last_verify_tokens
        ):
            logger.info(
                "DFlash2 lookup target verifier selected q%d (%s); "
                "q8_rounds=%d, q16_rounds=%d.",
                verify_tokens,
                reason,
                self._lookup_q8_rounds,
                self._lookup_q16_rounds,
            )
        self._lookup_last_verify_tokens = verify_tokens
        return draft_tokens

    def _prepare_proposal_runtime(
        self,
        input_batch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        """Refresh lookup eligibility and controller inputs before replay."""
        del num_rejected
        if not self._lookup_enabled:
            return

        num_reqs = input_batch.num_reqs
        # A request-state slot is reused after a request finishes.  Keying the
        # asynchronous lookup/chain controllers by slot alone lets a completed
        # request's pending flags admit a drafter-free proposal for the next
        # request that lands in the same slot.  Request IDs are stable across
        # TP ranks and uniquely identify the controller lifetime.
        req_key = tuple(input_batch.req_ids[:num_reqs])
        self._lookup_current_req_key = req_key
        self._lookup_last_emitted = num_sampled[:num_reqs]

        # Grammar validation keeps the checkpoint-native q8 scheduler
        # contract. Prefill rows have no stable generated suffix to extend.
        self._lookup_eligible.zero_()
        eligible = np.logical_not(input_batch.is_prefilling_np[:num_reqs])
        if input_batch.has_structured_output_reqs:
            eligible.fill(False)
        self._lookup_current_eligible = bool(num_reqs and eligible.all())
        eligible_cpu = torch.from_numpy(eligible.astype(np.int32, copy=False))
        self._lookup_eligible[:num_reqs].copy_(eligible_cpu, non_blocking=True)

        if req_key != self._lookup_controller_req_key:
            self._reset_lookup_controller(req_key)

    def _consume_lookup_flags(self) -> bool | None:
        """Return the last completed batch-wide lookup decision, if ready."""
        if not self._lookup_copy_pending or not self._lookup_copy_event.query():
            return None

        pending_key = self._lookup_pending_req_key
        pending_num_reqs = self._lookup_pending_num_reqs
        self._lookup_copy_pending = False
        if pending_key != self._lookup_current_req_key or pending_num_reqs <= 0:
            return False
        return bool(self._lookup_flags_cpu[:pending_num_reqs].numpy().all())

    def _queue_lookup_flags(self, num_reqs: int) -> None:
        """Asynchronously copy the current q16 signal into pinned host memory."""
        if self._lookup_copy_pending or self._lookup_last_emitted is None:
            return

        block = triton.next_power_of_2(max(num_reqs, 1))
        _prepare_lookup_controller_flags_kernel[(1,)](
            self._lookup_take_flags,
            self._lookup_last_emitted,
            self._lookup_controller_flags,
            1 + self.draft_block,
            num_reqs,
            BLOCK=block,
            num_warps=1,
        )
        if self._chain_enabled:
            _prepare_chain_entry_flags_kernel[(1,)](
                self._lookup_match_len,
                self._chain_entry_flags,
                self._chain_min_match,
                num_reqs,
                BLOCK=block,
                num_warps=1,
            )
        current_stream = torch.cuda.current_stream(self.device)
        self._lookup_copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._lookup_copy_stream):
            self._lookup_flags_cpu[:num_reqs].copy_(
                self._lookup_controller_flags[:num_reqs], non_blocking=True
            )
            if self._chain_enabled:
                self._chain_entry_flags_cpu[:num_reqs].copy_(
                    self._chain_entry_flags[:num_reqs], non_blocking=True
                )
            self._lookup_copy_event.record(self._lookup_copy_stream)
        self._lookup_pending_req_key = self._lookup_current_req_key
        self._lookup_pending_num_reqs = num_reqs
        self._lookup_copy_pending = True

    def next_num_draft_tokens(self) -> int:
        """Choose q8 or q16 for the next target verification step."""
        if not self._lookup_enabled:
            return self.num_speculative_steps

        if getattr(self, "_chain_enabled", False) and self._chain_active:
            # Clear the normal-step feedback that admitted the chain. Chain
            # steps do not run the neural lookup graph and therefore publish
            # no new adaptive q8/q16 feedback.
            self._consume_lookup_flags()
            return self._record_lookup_width(
                self.num_speculative_steps, "drafter-free-chain"
            )

        num_reqs = len(self._lookup_current_req_key)
        if num_reqs == 0 or not self._lookup_current_eligible:
            self._reset_lookup_controller(self._lookup_current_req_key)
            return self._record_lookup_width(self.draft_block, "ineligible")

        want = self._consume_lookup_flags()
        self._queue_lookup_flags(num_reqs)
        if not self._lookup_adaptive:
            return self._record_lookup_width(
                self.num_speculative_steps, "adaptive-disabled"
            )
        if (
            self._lookup_cheap_context > 0
            and self.draft_max_seq_len <= self._lookup_cheap_context
        ):
            return self._record_lookup_width(
                self.num_speculative_steps, "cheap-context"
            )

        (
            self._lookup_last_want,
            self._lookup_want_streak,
            self._lookup_sticky_remaining,
            self._lookup_long_active,
        ) = _advance_lookup_controller(
            want=want,
            num_reqs=num_reqs,
            entry_streak=self._lookup_entry_streak,
            sticky_steps=self._lookup_sticky_steps,
            last_want=self._lookup_last_want,
            want_streak=self._lookup_want_streak,
            sticky_remaining=self._lookup_sticky_remaining,
            long_active=self._lookup_long_active,
        )
        width = (
            self.num_speculative_steps if self._lookup_long_active else self.draft_block
        )
        reason = "strong-copy" if self._lookup_long_active else "adaptive-default"
        return self._record_lookup_width(width, reason)

    def _chain_entry_evidence(self) -> bool | None:
        if not self._lookup_copy_pending:
            return None
        # Every TP rank must make the same host-side branch.  Merely querying
        # an asynchronous D2H event can return ready on one rank and pending on
        # another, causing only a subset of ranks to skip the draft graph and
        # corrupting its TP collectives.  The copy was queued in the preceding
        # proposal step, so this is normally already complete and the fence is
        # only a correctness backstop for the drafter-free path.
        if not self._lookup_copy_event.query():
            self._lookup_copy_event.synchronize()
        if (
            self._lookup_pending_req_key != self._lookup_current_req_key
            or self._lookup_pending_num_reqs != 1
        ):
            return False
        return bool(self._chain_entry_flags_cpu[0])

    def _consume_chain_rejection(self) -> tuple[bool, bool] | None:
        if not self._chain_reject_pending:
            return None
        # See _chain_entry_evidence: a TP-local event readiness race must not
        # choose different draft/skip collective sequences across ranks.
        if not self._chain_reject_event.query():
            self._chain_reject_event.synchronize()
        self._chain_reject_pending = False
        if self._chain_reject_req_key != self._lookup_current_req_key:
            return None
        return (
            bool(self._chain_rejected_cpu[0]),
            self._chain_reject_pending_was_chain,
        )

    def _queue_chain_rejection(
        self,
        num_rejected: torch.Tensor,
        *,
        proposal_was_chain: bool,
    ) -> None:
        if self._chain_reject_pending:
            return
        self._chain_rejected[:1].copy_(num_rejected[:1])
        current_stream = torch.cuda.current_stream(self.device)
        self._chain_reject_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._chain_reject_stream):
            self._chain_rejected_cpu[:1].copy_(
                self._chain_rejected[:1], non_blocking=True
            )
            self._chain_reject_event.record(self._chain_reject_stream)
        self._chain_reject_req_key = self._lookup_current_req_key
        self._chain_reject_pending_was_chain = proposal_was_chain
        self._chain_reject_pending = True

    def _reset_chain(self, req_key: tuple[str, ...]) -> None:
        self._chain_req_key = req_key
        self._chain_active = False
        self._chain_previous_was_chain = False
        self._chain_last_proposal_was_chain = False

    def _chain_is_eligible(self, input_batch) -> bool:
        if (
            not self._chain_enabled
            or self.dp_size != 1
            or input_batch.num_reqs != 1
            or not self._lookup_current_eligible
            or self._req_states is None
        ):
            return False
        if not self._chain_greedy_only:
            return True
        if self._sampling_states is None:
            return False
        req_state = int(input_batch.idx_mapping_np[0])
        return float(self._sampling_states.temperature.np[req_state]) == 0.0

    def _prepare_draftless_proposal(
        self,
        input_batch,
        num_rejected: torch.Tensor,
        *,
        dummy_run: bool,
        is_profile: bool,
    ) -> bool:
        if not self._chain_enabled:
            return False

        req_key = self._lookup_current_req_key
        if req_key != self._chain_req_key:
            self._reset_chain(req_key)
        if dummy_run or is_profile or not self._chain_is_eligible(input_batch):
            self._reset_chain(req_key)
            return False

        self._chain_steps_total += 1
        verdict = self._consume_chain_rejection()
        self._queue_chain_rejection(
            num_rejected,
            proposal_was_chain=self._chain_last_proposal_was_chain,
        )
        rejected = _chain_rejection_for_controller(verdict)
        (
            self._chain_active,
            self._chain_previous_was_chain,
            engage,
        ) = _advance_chain_controller(
            active=self._chain_active,
            previous_was_chain=self._chain_previous_was_chain,
            rejected=rejected,
            entry_evidence=self._chain_entry_evidence(),
        )
        self._chain_last_proposal_was_chain = engage
        if not engage:
            return False

        self._chain_steps_engaged += 1
        self._generate_chain_proposal(input_batch.num_reqs)
        now = time.monotonic()
        if (
            self._chain_log_sec > 0
            and now - self._chain_last_log >= self._chain_log_sec
        ):
            logger.info(
                "DFlash2 chain engaged %d/%d eligible steps.",
                self._chain_steps_engaged,
                self._chain_steps_total,
            )
            self._chain_last_log = now
        return True

    def _generate_chain_proposal(self, num_reqs: int) -> None:
        """Fill the verifier block from request history without draft forward."""
        assert self._req_states is not None
        self._lookup_tokens[:num_reqs].zero_()
        tokens, _, _ = suffix_lookup(
            self._req_states.all_token_ids.gpu,
            self._req_states.total_len.gpu,
            self.sample_idx_mapping,
            self._lookup_eligible,
            num_reqs,
            self.num_speculative_steps,
            idx_mapping_stride=self.draft_block,
            nmax=self._lookup_nmax,
            nmin=self._lookup_nmin,
            search_max=self._lookup_search,
            out_tokens=self._lookup_tokens,
            out_len=self._lookup_match_len,
            out_valid=self._lookup_valid,
        )
        # A missing continuation becomes token zero and is rejected by the
        # target. That exact miss is the exit signal for the next host step.
        self.draft_tokens[:num_reqs].copy_(tokens[:num_reqs])
        self._lookup_use[:num_reqs].fill_(1)
        self._rewrite_lookup_point_masses(num_reqs)

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # The selector walk and rejection sampler must consume identical scores.
        # BF16 rounding measurably changes candidate order, so keep this FP32.
        return torch.float32, -float("inf")

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        # The SM70 tail must consume the exact same packed lattice as the
        # prefix walk. Keep one persistent view instead of creating temporary
        # contiguous inputs only for the first launch.
        scores = scores.contiguous()
        candidate_ids = candidate_ids.contiguous()
        block_k = triton.next_power_of_2(self.selector_top_k)
        walk_steps = self.draft_block - 1 if self._use_sm70_tail else self.draft_block
        _selector_walk_kernel[(num_reqs,)](
            scores,
            candidate_ids,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self._selector_tokens,
            self._selector_scores,
            self._selector_path_state,
            num_steps=self.draft_block,
            walk_steps=walk_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )
        if self._use_sm70_tail:
            _selector_walk_tail_kernel[(num_reqs,)](
                scores,
                candidate_ids,
                self.sample_pos,
                self.sample_idx_mapping,
                self.temperature,
                self.seeds,
                self._selector_tokens,
                self._selector_scores,
                self._selector_path_state,
                num_steps=self.draft_block,
                top_k=self.selector_top_k,
                BLOCK_K=block_k,
                SAMPLE_PROBABILISTIC=self.draft_logits is not None,
                USE_FP64=self.use_fp64_gumbel,
                num_warps=1,
            )

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        draft_logits = self.draft_logits
        assert draft_logits is not None
        cached_scores = self._cached_candidate_scores
        block_k = triton.next_power_of_2(self.selector_top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self._cached_candidate_ids,
            self._selector_scores if cached_scores is None else cached_scores,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.draft_block,
            cache_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            CACHE_SCORES=cached_scores is not None,
            num_warps=1,
        )

    def get_sparse_draft_logits(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return request-slot proposal candidates for sparse rejection."""
        if self.draft_logits is None or self._cached_candidate_scores is None:
            return None
        return self._cached_candidate_ids, self._cached_candidate_scores

    def get_selector_alignment_shadow(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return packed selector tensors for an explicitly enabled diagnostic."""
        if self._alignment_candidate_ids is None:
            return None
        assert self._alignment_unary_logits is not None
        assert self._alignment_lattice_scores is not None
        return (
            self._alignment_candidate_ids,
            self._alignment_unary_logits,
            self._alignment_lattice_scores,
        )

    def _prepare_ngram_assist(
        self,
        input_batch,
        output_copy_event: torch.cuda.Event | None,
        sampled_token_ids_cpu: np.ndarray | None,
        num_sampled_tokens_cpu: np.ndarray | None,
        all_token_ids_cpu: np.ndarray | None,
    ) -> bool:
        assist = self._ngram_assist
        self._ngram_num_hits = 0
        if (
            assist is None
            or input_batch.has_structured_output_reqs
            or output_copy_event is None
            or sampled_token_ids_cpu is None
            or num_sampled_tokens_cpu is None
            or all_token_ids_cpu is None
        ):
            return False

        # The copy stream only depends on target sampling. The main stream can
        # materialize DFlash context K/V while the host waits here, so lookup
        # does not serialize the context projection.
        output_copy_event.synchronize()
        num_reqs = input_batch.num_reqs
        num_draft_tokens = input_batch.num_draft_tokens_per_req
        if num_draft_tokens is None:
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        prior_lengths = (
            input_batch.seq_lens_cpu_upper_bound[:num_reqs].numpy() - num_draft_tokens
        )
        eligible = (~input_batch.is_prefilling_np[:num_reqs]) & (
            num_sampled_tokens_cpu[:num_reqs] > 0
        )
        full_hits = assist.propose(
            all_token_ids_cpu,
            input_batch.idx_mapping_np,
            prior_lengths,
            sampled_token_ids_cpu,
            num_sampled_tokens_cpu,
            eligible,
            self._ngram_tokens_cpu,
            self._ngram_lengths_cpu,
        )
        self._ngram_rounds += 1
        skip_query = num_reqs > 0 and full_hits == num_reqs
        self._ngram_num_hits = full_hits if skip_query else 0
        if skip_query:
            self._ngram_tokens[:num_reqs].copy_(
                self._ngram_tokens_cpu_tensor[:num_reqs], non_blocking=True
            )
            self._ngram_lengths[:num_reqs].copy_(
                self._ngram_lengths_cpu_tensor[:num_reqs], non_blocking=True
            )

        self._ngram_skipped_rounds += int(skip_query)
        if (
            envs.VLLM_DFLASH_PROFILE
            and self._ngram_rounds % envs.VLLM_DFLASH_PROFILE_LOG_INTERVAL == 0
        ):
            eligible_count = max(assist.num_eligible, 1)
            logger.info(
                "DFLASH2_NGRAM_PROFILE rounds=%d eligible=%d full_hits=%d "
                "hit_rate=%.4f skipped_query_rounds=%d lookup_avg_ms=%.4f",
                self._ngram_rounds,
                assist.num_eligible,
                assist.num_full_hits,
                assist.num_full_hits / eligible_count,
                self._ngram_skipped_rounds,
                assist.lookup_seconds * 1000.0 / self._ngram_rounds,
            )
        return skip_query

    def _apply_ngram_assist(self, num_reqs: int) -> None:
        if self._ngram_assist is None or self._ngram_num_hits == 0:
            return
        draft_logits = self.draft_logits
        cached_scores = self._cached_candidate_scores
        block_k = triton.next_power_of_2(self.selector_top_k)
        _apply_ngram_draft_kernel[(num_reqs * self.num_speculative_steps,)](
            self._ngram_tokens,
            self._ngram_lengths,
            self.sample_idx_mapping,
            self.draft_tokens,
            self.draft_tokens.stride(0),
            self._cached_candidate_ids,
            self._selector_scores if cached_scores is None else cached_scores,
            self._cached_candidate_ids.stride(0),
            self._cached_candidate_ids.stride(1),
            self._selector_scores if draft_logits is None else draft_logits,
            0 if draft_logits is None else draft_logits.stride(0),
            0 if draft_logits is None else draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            CACHE_DRAFT_LOGITS=draft_logits is not None,
            CACHE_SCORES=cached_scores is not None,
            num_warps=1,
        )

    def _apply_lookup(self, num_reqs: int) -> None:
        """Fuse a history continuation into the DFlash2 proposal exactly."""
        if not self._lookup_enabled or self._req_states is None:
            return

        tokens, match_len, valid = suffix_lookup(
            self._req_states.all_token_ids.gpu,
            self._req_states.total_len.gpu,
            self.sample_idx_mapping,
            self._lookup_eligible,
            num_reqs,
            self.num_speculative_steps,
            idx_mapping_stride=self.draft_block,
            nmax=self._lookup_nmax,
            nmin=self._lookup_nmin,
            search_max=self._lookup_search,
            out_tokens=self._lookup_tokens,
            out_len=self._lookup_match_len,
            out_valid=self._lookup_valid,
        )
        fuse_draft(
            self.draft_tokens,
            tokens,
            match_len,
            valid,
            self._lookup_use,
            self.sample_idx_mapping,
            self._lookup_hits,
            num_reqs,
            self.num_speculative_steps,
            draft_block=self.draft_block,
            idx_mapping_stride=self.draft_block,
            nmin=self._lookup_nmin,
            nstrong=self._lookup_nstrong,
            agree_min=self._lookup_agree,
            nmin_tail=self._lookup_nmin_tail,
            long_min=self._lookup_long_min,
            take_flags=self._lookup_take_flags,
        )

        self._rewrite_lookup_point_masses(num_reqs)

    def _rewrite_lookup_point_masses(self, num_reqs: int) -> None:
        draft_logits = self.draft_logits
        if draft_logits is None:
            return
        cached_scores = self._cached_candidate_scores
        block_k = triton.next_power_of_2(self.selector_top_k)
        _point_mass_draft_logits_kernel[(num_reqs * self.num_speculative_steps,)](
            draft_logits,
            self._cached_candidate_ids,
            self._selector_scores if cached_scores is None else cached_scores,
            self.draft_tokens,
            self.draft_tokens.stride(0),
            self._lookup_use,
            self.sample_idx_mapping,
            self.draft_block,
            self._cached_candidate_ids.stride(0),
            self._cached_candidate_ids.stride(1),
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            CACHE_SCORES=cached_scores is not None,
            num_warps=1,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.draft_block
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.draft_block, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.draft_block, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        if self._alignment_candidate_ids is not None:
            assert self._alignment_unary_logits is not None
            assert self._alignment_lattice_scores is not None
            self._alignment_candidate_ids[:num_reqs].copy_(candidate_ids)
            self._alignment_unary_logits[:num_reqs].copy_(unary_logits)
            self._alignment_lattice_scores[:num_reqs].copy_(scores)
        self._sample_path(candidate_ids, scores, num_reqs)
        self.draft_tokens[:num_reqs, : self.draft_block].copy_(
            self._selector_tokens[:num_reqs]
        )
        if self.draft_block < self.num_speculative_steps:
            self.draft_tokens[:num_reqs, self.draft_block :].zero_()
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)
        self._apply_lookup(num_reqs)
