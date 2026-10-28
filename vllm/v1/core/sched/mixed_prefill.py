# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feedback control of prefill work sharing a step with resident decoders."""

import math

from vllm.v1.outputs import MixedPrefillTiming


class MixedPrefillBudget:
    """Learn a token budget from completed GPU steps, without host-side waits.

    The first mixed step uses a conservative seed. Subsequent samples adapt to
    the actual model, device, TP size and context. This is a latency target,
    not a hard deadline: even a decode-only step can exceed the target.
    """

    # Weight of earlier samples in the step-cost fit.
    DECAY = 0.75

    def __init__(
        self,
        max_tokens: int,
        target_ms: float,
        min_tokens: int = 0,
    ) -> None:
        self.max_tokens = max_tokens
        self.target_ms = target_ms
        self.min_tokens = min(min_tokens, max_tokens)
        self.tokens = max(min(512, max_tokens), self.min_tokens)
        self.ms_per_token: float | None = None
        # Decayed sums for the fit elapsed = fixed + per_token * tokens:
        # weight, x, y, x*x and x*y.
        self._fit = [0.0] * 5

    def _affine_budget(self) -> float | None:
        """Budget from the fixed-plus-linear step cost, once chunks differ.

        A mixed step pays a fixed cost (the decode rows plus any launches
        that do not scale with prefill) on top of the prefill work. Charging
        all of it to prefill tokens shrinks the budget whenever the fixed
        part alone reaches the target, and the smaller chunks then make each
        token look even more expensive. Fitting both terms removes that
        feedback. A chunk smaller than the prefill that costs as much as the
        fixed part spends most of its step on overhead and only multiplies the
        number of mixed steps, so the budget never drops below that amount.
        Without this floor a fixed cost just under the target would leave a
        budget of a few tokens.
        """
        w, sx, sy, sxx, sxy = self._fit
        spread = w * sxx - sx * sx
        # Require a relative spread of the chunk sizes of about 10%.
        if w < 2 or spread <= 0.01 * sx * sx:
            return None
        per_token = (w * sxy - sx * sy) / spread
        fixed = max(0.0, (sy - per_token * sx) / w)
        if per_token <= 0:
            return float(self.max_tokens)
        return max(self.target_ms - fixed, fixed) / per_token

    def update(self, sample: MixedPrefillTiming | None) -> None:
        if (
            sample is None
            or self.target_ms <= 0
            or sample.prefill_tokens <= 0
            or sample.decode_tokens <= 0
            or not math.isfinite(sample.elapsed_ms)
            or sample.elapsed_ms <= 0
        ):
            return
        # A tiny boundary/tail chunk is dominated by fixed decode overhead.
        # It must not teach the controller that a full chunk is equally costly.
        if sample.prefill_tokens < sample.budget_tokens * 0.75:
            return
        # A short prefill that fits a captured graph skips most of the eager
        # fixed cost. Fitting it with eager steps would make prefill tokens
        # look expensive and split longer prompts into more eager steps. It
        # still lets the budget move toward the eager fit, so a budget that
        # fell into the graph range can recover.
        if not sample.cuda_graph:
            x, y = float(sample.prefill_tokens), float(sample.elapsed_ms)
            w, sx, sy, sxx, sxy = (v * self.DECAY for v in self._fit)
            self._fit = [w + 1, sx + x, sy + y, sxx + x * x, sxy + x * y]
            cost = sample.elapsed_ms / sample.prefill_tokens
            self.ms_per_token = (
                cost
                if self.ms_per_token is None
                else 0.5 * self.ms_per_token + 0.5 * cost
            )
        affine = self._affine_budget()
        if affine is not None:
            desired = affine
        elif sample.cuda_graph or self.ms_per_token is None:
            return
        else:
            desired = self.target_ms / self.ms_per_token
        quantum = min(16, self.max_tokens)
        # Grow gradually, by at least one quantum so that a budget at the
        # quantum can recover; a fitted cost model may double per step. React
        # promptly to a long-context stall.
        growth = 2.0 if affine is not None else 1.25
        desired = min(desired, max(self.tokens * growth, self.tokens + quantum))
        self.tokens = max(
            quantum,
            self.min_tokens,
            min(self.max_tokens, int(desired) // quantum * quantum),
        )
