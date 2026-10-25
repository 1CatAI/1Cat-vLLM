# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feedback control of prefill work sharing a step with resident decoders."""

import math

from vllm.v1.outputs import MixedPrefillTiming


class MixedPrefillBudget:
    """Learn a token budget from completed execution, without additional waits.

    The first mixed step uses a conservative seed. Subsequent samples adapt to
    the actual model, devices, parallelism and context. This is a latency target,
    not a hard deadline: even a decode-only step can exceed the target.
    """

    def __init__(
        self,
        max_tokens: int,
        target_ms: float,
        min_tokens: int = 0,
        *,
        initial_tokens: int = 512,
    ) -> None:
        self.max_tokens = max_tokens
        self.target_ms = target_ms
        self.min_tokens = min(min_tokens, max_tokens)
        self.tokens = max(min(initial_tokens, max_tokens), self.min_tokens)
        self.ms_per_token: float | None = None

    def update(self, sample: MixedPrefillTiming | None) -> None:
        if (
            sample is None
            or self.target_ms <= 0
            or sample.prefill_tokens <= 0
            or sample.budget_tokens <= 0
            or (sample.decode_tokens <= 0 and not sample.is_pipeline)
            or not math.isfinite(sample.elapsed_ms)
            or sample.elapsed_ms <= 0
        ):
            return
        # A tiny boundary/tail chunk is dominated by fixed decode overhead.
        # It must not teach the controller that a full chunk is equally costly.
        if sample.prefill_tokens < sample.budget_tokens * 0.75 and not (
            sample.is_pipeline and sample.elapsed_ms > self.target_ms
        ):
            return
        # A PP batch capped by another scheduling limit must still teach us
        # about stalls. Otherwise e.g. 192 tokens under a 384-token budget can
        # never shrink that budget. Short fast tails still cannot grow it.
        cost = sample.elapsed_ms / sample.prefill_tokens
        self.ms_per_token = (
            cost if self.ms_per_token is None else 0.5 * self.ms_per_token + 0.5 * cost
        )
        desired = self.target_ms / self.ms_per_token
        # Grow slowly on fast devices; react promptly to a long-context stall.
        quantum = min(16, self.max_tokens)
        # Permit one quantum of growth at small budgets: rounding 16 * 1.25
        # back to 16 would otherwise permanently trap the controller there.
        desired = min(desired, max(self.tokens * 1.25, self.tokens + quantum))
        self.tokens = max(
            quantum,
            self.min_tokens,
            min(self.max_tokens, int(desired) // quantum * quantum),
        )
