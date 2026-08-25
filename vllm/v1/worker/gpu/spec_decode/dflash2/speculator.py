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
    cache_base = (req_state * num_steps + step) * top_k
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
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self._selector_scores.shape, dtype=torch.int64, device=device
        )
        self._cached_candidate_scores = None
        if (
            self.draft_logits is not None
            and envs.VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION
        ):
            self._cached_candidate_scores = torch.empty_like(self._selector_scores)
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
                self.num_speculative_steps,
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
        self._use_sm70_tail = _requires_sm70_tail(device, self.num_speculative_steps)
        if self.draft_logits is not None:
            # The cache kernel writes only K columns; all other vocabulary
            # columns must remain impossible.
            self.draft_logits.fill_(-float("inf"))

        self._ngram_assist: DFlash2NgramAssist | None = None
        self._ngram_num_hits = 0
        self._ngram_rounds = 0
        self._ngram_skipped_rounds = 0
        speculative_config = getattr(self, "speculative_config", None)
        if speculative_config is not None and getattr(
            speculative_config, "ngram_assist", False
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

    @property
    def requires_host_token_state(self) -> bool:
        """Whether the runner must expose async samples and request history."""
        return self._ngram_assist is not None

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
        walk_steps = (
            self.num_speculative_steps - 1
            if self._use_sm70_tail
            else self.num_speculative_steps
        )
        _selector_walk_kernel[(num_reqs,)](
            scores,
            candidate_ids,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            self._selector_path_state,
            num_steps=self.num_speculative_steps,
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
                self.draft_tokens,
                self._selector_scores,
                self._selector_path_state,
                num_steps=self.num_speculative_steps,
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
            num_steps=self.num_speculative_steps,
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
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
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
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)
