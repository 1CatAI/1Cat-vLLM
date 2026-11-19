# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.kernel import adaptive_mtp_min_k, is_adaptive_mtp_enabled
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Template for speculative decoding placeholders. Each request must get
        # its own list instance because scheduler outputs and worker feedback
        # mutate spec-token rows in later phases.
        self._spec_token_placeholders: tuple[int, ...] = (-1,) * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size
        speculative_config = self.vllm_config.speculative_config
        draft_model_config = (
            speculative_config.draft_model_config
            if speculative_config is not None
            else None
        )
        self._drafter_max_model_len = (
            draft_model_config.max_model_len
            if draft_model_config is not None
            and draft_model_config.max_model_len is not None
            else self.max_model_len
        )
        self._adaptive_mtp = self.use_v2_model_runner and is_adaptive_mtp_enabled(
            speculative_config, self.num_spec_tokens
        )
        self._adaptive_mtp_min_k = adaptive_mtp_min_k(
            self.num_spec_tokens, self.vllm_config.speculative_config
        )
        # req_id -> [ema_successes, ema_trials, cur_k]
        self._adaptive_mtp_state: dict[str, list[float]] = {}
        if self._adaptive_mtp:
            logger.info(
                "Adaptive MTP (EntMTP-Async) enabled: k in [%d, %d].",
                self._adaptive_mtp_min_k,
                self.num_spec_tokens,
            )

    def _on_spec_decode_feedback(
        self,
        req_id: str,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> None:
        if not self._adaptive_mtp or num_draft_tokens <= 0:
            return
        k_min = self._adaptive_mtp_min_k
        k_max = self.num_spec_tokens
        state = self._adaptive_mtp_state.get(req_id)
        if state is None:
            # Neutral-optimistic prior: p_hat = 0.80 -> starts at k_max.
            state = [4.0, 5.0, float(k_max)]
            self._adaptive_mtp_state[req_id] = state
        # Geometric-MLE EMA: p_hat = E[accepted] / E[min(accepted + 1, k)].
        # Invariant to current depth k, so low k never traps the estimate.
        alpha = 0.28
        trials = min(num_accepted_tokens + 1, num_draft_tokens)
        s = (1.0 - alpha) * state[0] + alpha * float(num_accepted_tokens)
        n = (1.0 - alpha) * state[1] + alpha * float(trials)
        p_hat = s / max(n, 1e-6)
        cur_k = int(state[2])

        # Instant ramp-up when the current depth saturates (100% accepted).
        if num_accepted_tokens >= num_draft_tokens and cur_k < k_max:
            new_k = min(k_max, cur_k + 1)
        else:
            # Measured throughput-optimal k*(p) with hysteresis (+/-0.025).
            # Boundaries: k=2 (<0.62), k=3 (0.62..0.74), k=4 (0.74..0.82),
            #             k=5 (0.82..0.89), k=6 (>=0.89).
            if p_hat < 0.62:
                target_k = 2
            elif p_hat < 0.74:
                target_k = 3
            elif p_hat < 0.82:
                target_k = 4
            elif p_hat < 0.89:
                target_k = 5
            else:
                target_k = 6
            target_k = max(k_min, min(k_max, target_k))
            # Apply 0.025 downward hysteresis to prevent step-to-step chatter.
            down_thresh = {3: 0.595, 4: 0.715, 5: 0.795, 6: 0.865}
            if target_k < cur_k:
                if p_hat < down_thresh.get(cur_k, 0.0):
                    new_k = max(k_min, cur_k - 1)
                else:
                    new_k = cur_k
            elif target_k > cur_k:
                new_k = min(k_max, cur_k + 1)
            else:
                new_k = cur_k

        state[0] = s
        state[1] = n
        state[2] = float(new_k)

    def _select_adaptive_batch_k(self, decode_req_ids: list[str]) -> int:
        if not self._adaptive_mtp or not decode_req_ids:
            return self.num_spec_tokens
        if len(decode_req_ids) > 1:
            # Concurrent batches (c >= 2) have tiny marginal verify cost per
            # token (~2-4%) and require uniform qlen for pre-captured graphs.
            return self.num_spec_tokens
        state = self._adaptive_mtp_state.get(decode_req_ids[0])
        if state is None:
            return self.num_spec_tokens
        return max(self._adaptive_mtp_min_k, min(self.num_spec_tokens, int(state[2])))

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        if self._adaptive_mtp:
            if scheduler_output.finished_req_ids:
                for finished_id in scheduler_output.finished_req_ids:
                    self._adaptive_mtp_state.pop(finished_id, None)
            if len(self._adaptive_mtp_state) > len(self.requests):
                for stale_id in list(self._adaptive_mtp_state):
                    if stale_id not in self.requests:
                        del self._adaptive_mtp_state[stale_id]
        decode_req_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if not self.requests[req_id].is_prefill_chunk
        ]
        next_k = self._select_adaptive_batch_k(decode_req_ids)
        if self._adaptive_mtp and decode_req_ids:
            scheduler_output.next_spec_steps = next_k
        placeholders = (
            self._spec_token_placeholders
            if next_k == self.num_spec_tokens
            else (-1,) * next_k
        )
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            if request.num_computed_tokens + next_k <= self._drafter_max_model_len:
                request.spec_token_ids = list(placeholders)
            else:
                request.spec_token_ids = []

            if self.use_v2_model_runner:
                request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.async_tokens_to_discard > 0:
            # The request was force-preempted in reset_prefix_cache; drop one
            # stale in-flight async output frame per call until the counter
            # is drained.
            request.async_tokens_to_discard -= 1
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
