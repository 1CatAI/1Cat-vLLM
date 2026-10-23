# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nonblocking, batch-bound PP latency feedback, including queued work."""

import time

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import MixedPrefillTiming, ModelRunnerOutput

logger = init_logger(__name__)
_last_prefill_log: float | None = None


def start_pp_mixed_prefill_timing(output: SchedulerOutput) -> None:
    if output.mixed_prefill_is_pp and output.mixed_prefill_tokens > 0:
        output.mixed_prefill_started_at = time.perf_counter()


def finish_pp_mixed_prefill_timing(
    output: SchedulerOutput, model_output: ModelRunnerOutput
) -> None:
    global _last_prefill_log
    started = output.mixed_prefill_started_at
    if started is None:
        return
    # Each queued/deferred batch owns its timestamp and token counts. Never
    # substitute the latest submitted batch, or time only future.result(): a
    # result may already be ready when the engine starts consuming it.
    output.mixed_prefill_started_at = None
    now = time.perf_counter()
    sample = MixedPrefillTiming(
        prefill_tokens=output.mixed_prefill_tokens,
        decode_tokens=output.mixed_decode_tokens,
        budget_tokens=output.mixed_prefill_budget,
        elapsed_ms=(now - started) * 1000.0,
        is_pipeline=True,
    )
    model_output.mixed_prefill_timing = sample
    # One aggregate batch sample per ten seconds, without prompt text, request
    # identifiers, CUDA events or device synchronization. In particular this
    # makes progress visible before a long prefill emits its first token.
    if (
        _last_prefill_log is None
        or now < _last_prefill_log
        or now - _last_prefill_log >= 10.0
    ):
        _last_prefill_log = now
        logger.info(
            "PP prefill batch: mode=%s, prefill=%d tokens, decode=%d tokens, "
            "batch latency=%.1f ms, mixed budget=%d tokens",
            "mixed" if sample.budget_tokens > 0 else "standalone",
            sample.prefill_tokens,
            sample.decode_tokens,
            sample.elapsed_ms,
            sample.budget_tokens,
        )
