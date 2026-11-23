# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.kernel import resolve_adaptive_verify
from vllm.logger import init_logger
from vllm.v1.core.sched.adaptive_verify import (
    VerifyLengthController,
    VerifyState,
    build_verify_controller,
)
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
        self._verify_controller: VerifyLengthController | None = None
        # req_id -> controller state of that request
        self._verify_states: dict[str, VerifyState] = {}
        plan = resolve_adaptive_verify(
            speculative_config, self.vllm_config.kernel_config, self.num_spec_tokens
        )
        # Inkling2 DSpark: draft width (verified tokens) by decode batch size.
        self._dspark_widths = None
        if (
            self.use_v2_model_runner
            and speculative_config is not None
            and speculative_config.method == "dspark"
            and getattr(draft_model_config.hf_config, "model_type", None)
            == "inkling2_dspark"
        ):
            from vllm.v1.worker.gpu.spec_decode.dspark.widths import (
                DSparkAdaptive,
                DSparkWidths,
            )

            self._dspark_widths = DSparkWidths(self.num_spec_tokens)
            # A lone decode request gets its width from its own acceptance.
            self._dspark_adaptive = DSparkAdaptive(self.num_spec_tokens)
            self._dspark_adaptive.widths = self._dspark_widths
            self._dspark_widths.adaptive_obj = self._dspark_adaptive
            logger.info(
                "Inkling2 DSpark batch-aware draft widths (B=1,2,...): %s",
                self._dspark_widths.table,
            )
        if plan is not None and not self.use_v2_model_runner:
            # Only the V2 runner honours a per-round verify length, so the
            # automatic depth stays fixed here; the one method an explicit
            # adaptive_verify setting admits runs on the V2 runner only.
            plan = None
        if plan is not None:
            self._verify_controller = build_verify_controller(plan)
            logger.info(
                "Adaptive verify length (%s) enabled: n in [%d, %d], first round %d.",
                plan.rule,
                plan.min_tokens,
                plan.max_tokens,
                plan.initial_tokens,
            )

    def _on_spec_decode_feedback(
        self,
        req_id: str,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> None:
        if self._dspark_widths is not None:
            self._dspark_adaptive.feedback(
                req_id, num_draft_tokens, num_accepted_tokens
            )
            return
        controller = self._verify_controller
        if controller is None or num_draft_tokens <= 0:
            return
        state = self._verify_states.get(req_id)
        if state is None:
            state = self._verify_states[req_id] = controller.new_state()
        controller.update(state, num_draft_tokens, num_accepted_tokens)

    def _select_adaptive_batch_k(self, decode_req_ids: list[str]) -> int:
        if self._dspark_widths is not None and decode_req_ids:
            width = self._dspark_widths.width(len(decode_req_ids))
            if len(decode_req_ids) == 1 and width > 0:
                width = self._dspark_adaptive.width(decode_req_ids[0], width)
            return width
        controller = self._verify_controller
        if controller is None or not decode_req_ids:
            return self.num_spec_tokens
        if len(decode_req_ids) > 1:
            # Concurrent batches (c >= 2) have tiny marginal verify cost per
            # token (~2-4%) and require uniform qlen for pre-captured graphs.
            return controller.max_length
        state = self._verify_states.get(decode_req_ids[0])
        if state is None:
            return controller.initial_length
        return state.length

    def schedule(self) -> SchedulerOutput:
        if self._dspark_widths is not None:
            self._uniform_dspark_drafts()
        return super().schedule()

    def _uniform_dspark_drafts(self) -> None:
        """Trim the draft tokens of the running decode requests to one common
        length: at most the width of this batch size, and no more than the
        shortest request holds. A batch of equal query lengths replays a
        captured target graph; mixed lengths (a request that drafted when the
        batch was smaller) would run eagerly."""
        decodes = [
            r for r in self.running if r.spec_token_ids and not r.is_prefill_chunk
        ]
        if not decodes:
            return
        if len(decodes) < 2 and self._dspark_adaptive.enabled:
            # A lone request keeps the width it was proposed with (adaptive).
            return
        width = min(
            self._dspark_widths.width(len(decodes)),
            min(len(r.spec_token_ids) for r in decodes),
        )
        for request in decodes:
            if len(request.spec_token_ids) > width:
                request.spec_token_ids = request.spec_token_ids[:width]

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        if self._verify_controller is not None:
            for finished_id in scheduler_output.finished_req_ids:
                self._verify_states.pop(finished_id, None)
            if len(self._verify_states) > len(self.requests):
                for stale_id in list(self._verify_states):
                    if stale_id not in self.requests:
                        del self._verify_states[stale_id]
        decode_req_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if not self.requests[req_id].is_prefill_chunk
        ]
        if self._dspark_widths is not None:
            self._dspark_adaptive.forget(self.requests)
        next_k = self._select_adaptive_batch_k(decode_req_ids)
        if (
            self._verify_controller is not None or self._dspark_widths is not None
        ) and decode_req_ids:
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
