# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
def _proposal_nucleus_logits(
    scores,
    mask,
    top_p: tl.constexpr,
):
    sorted_scores = tl.sort(tl.where(mask, scores, float("-inf")), descending=True)
    row_max = tl.max(sorted_scores, axis=0)
    unnormalized = tl.exp(sorted_scores - row_max)
    probs = unnormalized / tl.sum(unnormalized, axis=0)
    cumulative_before = tl.cumsum(probs, axis=0) - probs
    keep_sorted = cumulative_before < top_p
    cutoff = tl.min(tl.where(keep_sorted, sorted_scores, float("inf")), axis=0)
    return tl.where(mask & (scores >= cutoff), scores, float("-inf"))


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
    PROPOSAL_TEMPERATURE_SCALE: tl.constexpr,
    PROPOSAL_TOP_P: tl.constexpr,
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
        ).to(tl.float32)
        if SAMPLE_PROBABILISTIC and temperature != 0.0:
            # Cache the exact temperature-applied proposal scores expected by
            # the shared rejection sampler. This keeps Eagle/MTP's established
            # contract unchanged while matching the DFlash2 selector draw.
            scores = scores / (temperature * PROPOSAL_TEMPERATURE_SCALE)
            if PROPOSAL_TOP_P < 1.0:
                scores = _proposal_nucleus_logits(scores, mask, PROPOSAL_TOP_P)
        scores = scores.to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # Candidate token IDs key an independent draft-noise stream.
        position = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            position,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            IS_DRAFTING=True,
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
    PROPOSAL_TEMPERATURE_SCALE: tl.constexpr,
    PROPOSAL_TOP_P: tl.constexpr,
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
    ).to(tl.float32)
    if SAMPLE_PROBABILISTIC and temperature != 0.0:
        scores = scores / (temperature * PROPOSAL_TEMPERATURE_SCALE)
        if PROPOSAL_TOP_P < 1.0:
            scores = _proposal_nucleus_logits(scores, mask, PROPOSAL_TOP_P)
    scores = scores.to(tl.float64 if USE_FP64 else tl.float32)
    candidate_base = flat * top_k
    candidates = tl.load(
        candidate_ptr + candidate_base + offsets,
        mask=mask & valid,
        other=0,
    )
    # Candidate token IDs key an independent draft-noise stream.
    position = tl.load(sample_pos_ptr + flat) - 1
    _, index = gumbel_noised_argmax(
        scores,
        candidates,
        mask & valid,
        seed,
        position,
        temperature if SAMPLE_PROBABILISTIC else 0.0,
        IS_DRAFTING=True,
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


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._debug_token_dump_count = 0
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self.proposal_temperature_scale = (
            envs.VLLM_SM70_DFLASH2_PROPOSAL_TEMPERATURE_SCALE
        )
        self.proposal_top_p = envs.VLLM_SM70_DFLASH2_PROPOSAL_TOP_P
        if self.proposal_temperature_scale <= 0.0:
            raise ValueError(
                "VLLM_SM70_DFLASH2_PROPOSAL_TEMPERATURE_SCALE must be positive"
            )
        if not 0.0 < self.proposal_top_p <= 1.0:
            raise ValueError("VLLM_SM70_DFLASH2_PROPOSAL_TOP_P must be in (0, 1]")
        if self.proposal_temperature_scale != 1.0 or self.proposal_top_p != 1.0:
            logger.info_once(
                "Using DFlash2 proposal calibration: temperature_scale=%.3f, "
                "top_p=%.3f. Cached q logits preserve exact rejection sampling.",
                self.proposal_temperature_scale,
                self.proposal_top_p,
            )
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
            self._draft_logits_init is not None
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
        self._debug_backbone_hidden_states: torch.Tensor | None = None
        self._debug_candidate_ids: torch.Tensor | None = None
        self._debug_unary_logits: torch.Tensor | None = None
        self._debug_lattice_scores: torch.Tensor | None = None
        if getattr(self, "_debug_tensor_dump_dir", ""):
            packed_shape = (
                self.max_num_reqs,
                self.draft_block,
                self.selector_top_k,
            )
            self._debug_backbone_hidden_states = torch.empty(
                self.max_num_reqs,
                self.draft_block,
                self.hidden_size,
                dtype=self.dtype,
                device=device,
            )
            self._debug_candidate_ids = torch.empty(
                packed_shape, dtype=torch.int64, device=device
            )
            self._debug_unary_logits = torch.empty(
                packed_shape, dtype=torch.float32, device=device
            )
            self._debug_lattice_scores = torch.empty(
                (*packed_shape, self.selector_top_k),
                dtype=torch.float32,
                device=device,
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
        self._lookup_current_req_key: tuple[int, ...] = ()
        self._lookup_current_eligible = False
        self._lookup_last_emitted: torch.Tensor | None = None
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
            self._lookup_copy_pending = False
            self._lookup_pending_req_key: tuple[int, ...] = ()
            self._lookup_pending_num_reqs = 0
            self._lookup_last_want = False
            self._lookup_want_streak = 0
            self._lookup_sticky_remaining = 0
            self._lookup_long_active = False
            self._lookup_controller_req_key: tuple[int, ...] = ()
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

    @property
    def requires_host_token_state(self) -> bool:
        """Whether the runner must expose async samples and request history."""
        return self._ngram_assist is not None

    def set_req_states(self, req_states) -> None:
        """Expose the request token history used by the device lookup."""
        self._req_states = req_states

    def _reset_lookup_controller(self, req_key: tuple[int, ...]) -> None:
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
        req_key = tuple(int(index) for index in input_batch.idx_mapping_np[:num_reqs])
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
        current_stream = torch.cuda.current_stream(self.device)
        self._lookup_copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._lookup_copy_stream):
            self._lookup_flags_cpu[:num_reqs].copy_(
                self._lookup_controller_flags[:num_reqs], non_blocking=True
            )
            self._lookup_copy_event.record(self._lookup_copy_stream)
        self._lookup_pending_req_key = self._lookup_current_req_key
        self._lookup_pending_num_reqs = num_reqs
        self._lookup_copy_pending = True

    def next_num_draft_tokens(self) -> int:
        """Choose q8 or q16 for the next target verification step."""
        if not self._lookup_enabled:
            return self.num_speculative_steps

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
            PROPOSAL_TEMPERATURE_SCALE=self.proposal_temperature_scale,
            PROPOSAL_TOP_P=self.proposal_top_p,
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
                PROPOSAL_TEMPERATURE_SCALE=self.proposal_temperature_scale,
                PROPOSAL_TOP_P=self.proposal_top_p,
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
        if self._debug_candidate_ids is not None:
            assert self._debug_backbone_hidden_states is not None
            assert self._debug_unary_logits is not None
            assert self._debug_lattice_scores is not None
            self._debug_backbone_hidden_states[:num_reqs].copy_(hidden_states)
            self._debug_candidate_ids[:num_reqs].copy_(candidate_ids)
            self._debug_unary_logits[:num_reqs].copy_(unary_logits)
            self._debug_lattice_scores[:num_reqs].copy_(scores)
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
        if (
            getattr(self, "_debug_proposal_stages", False)
            and getattr(self, "_debug_real_proposal", False)
            and self._debug_token_dump_count < 2
        ):
            first_scores = scores[0, 0, 0]
            first_unary = unary_logits[0, 0]
            first_bilinear = first_scores - first_unary
            greedy_path = []
            predecessor = 0
            for step in range(self.draft_block):
                score_row = scores[0, step, predecessor]
                successor = int(score_row.argmax().item())
                greedy_path.append(
                    (
                        successor,
                        int(candidate_ids[0, step, successor].item()),
                        float(score_row[successor].item()),
                    )
                )
                predecessor = successor
            logger.info(
                "DFlash2 token diagnostic: finite_hidden=%s max_abs_hidden=%s "
                "query_input_ids=%s anchor_token_id=%s candidate_ids=%s "
                "unary_logits=%s first_bilinear=%s first_total=%s "
                "greedy_path=%s draft_tokens=%s",
                bool(torch.isfinite(hidden_states).all().item()),
                float(hidden_states.abs().max().item()),
                self.input_buffers.input_ids[: self.num_query_per_req].tolist(),
                int(anchor_token_ids[0].item()),
                candidate_ids[0].tolist(),
                unary_logits[0].tolist(),
                first_bilinear.tolist(),
                first_scores.tolist(),
                greedy_path,
                self.draft_tokens[0, : self.draft_block].tolist(),
            )
            self._debug_token_dump_count += 1
        if self.draft_block < self.num_speculative_steps:
            self.draft_tokens[:num_reqs, self.draft_block :].zero_()
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)
        self._apply_lookup(num_reqs)
