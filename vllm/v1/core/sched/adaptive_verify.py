# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request verification length of speculative decoding rounds.

The async scheduler asks one controller, chosen by build_verify_controller from
the resolved plan, how many draft tokens the next round of a request verifies.
The method decides the rule: MTP keeps 1Cat's measured acceptance bands,
block drafters maximise expected tokens per second from per-position
acceptance.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from vllm.config.kernel import (
    ADAPTIVE_MTP_FLOOR_K,
    AdaptiveVerifyPlan,
    required_verify_cost_ratio,
)


class VerifyState(Protocol):
    length: int


class VerifyLengthController(Protocol):
    initial_length: int
    max_length: int

    def new_state(self) -> VerifyState: ...

    def update(
        self, state: Any, num_draft_tokens: int, num_accepted_tokens: int
    ) -> None:
        """Fold one finished round into state and choose the next length.

        state is whatever new_state returned; each rule keeps its own fields
        next to the shared `length`."""


# 1Cat's EntMTP constants as measured by 1Cat; kept bit for bit.
_MTP_EMA_ALPHA = 0.28
_MTP_PRIOR_SUCCESSES = 4.0
_MTP_PRIOR_TRIALS = 5.0
# Lower band edge of each target depth (k=2 below the first edge).
_MTP_BAND_EDGES = ((0.62, 3), (0.74, 4), (0.82, 5), (0.89, 6))
# Acceptance a depth must fall under before the depth is lowered (0.025 below
# its band edge), against step-to-step chatter.
_MTP_DOWN_THRESHOLDS = {3: 0.595, 4: 0.715, 5: 0.795, 6: 0.865}


@dataclass
class MtpBandState:
    ema_successes: float
    ema_trials: float
    length: int


class MtpBandController:
    """1Cat's geometric-MLE acceptance estimate with fixed depth bands."""

    def __init__(self, plan: AdaptiveVerifyPlan) -> None:
        self._k_min = plan.min_tokens
        self._k_max = plan.max_tokens
        self.initial_length = plan.initial_tokens
        self.max_length = plan.max_tokens

    def new_state(self) -> MtpBandState:
        # Neutral-optimistic prior: p_hat = 0.80 -> starts at k_max.
        return MtpBandState(_MTP_PRIOR_SUCCESSES, _MTP_PRIOR_TRIALS, self._k_max)

    def update(
        self,
        state: MtpBandState,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> None:
        k_min, k_max = self._k_min, self._k_max
        # Geometric-MLE EMA: p_hat = E[accepted] / E[min(accepted + 1, k)].
        # Invariant to current depth k, so low k never traps the estimate.
        alpha = _MTP_EMA_ALPHA
        trials = min(num_accepted_tokens + 1, num_draft_tokens)
        successes = (1.0 - alpha) * state.ema_successes + alpha * float(
            num_accepted_tokens
        )
        total = (1.0 - alpha) * state.ema_trials + alpha * float(trials)
        p_hat = successes / max(total, 1e-6)
        cur_k = state.length

        # Instant ramp-up when the current depth saturates (100% accepted).
        if num_accepted_tokens >= num_draft_tokens and cur_k < k_max:
            new_k = min(k_max, cur_k + 1)
        else:
            target_k = ADAPTIVE_MTP_FLOOR_K
            for edge, depth in _MTP_BAND_EDGES:
                if p_hat >= edge:
                    target_k = depth
            target_k = max(k_min, min(k_max, target_k))
            if target_k < cur_k:
                if p_hat < _MTP_DOWN_THRESHOLDS.get(cur_k, 0.0):
                    new_k = max(k_min, cur_k - 1)
                else:
                    new_k = cur_k
            elif target_k > cur_k:
                new_k = min(k_max, cur_k + 1)
            else:
                new_k = cur_k

        state.ema_successes = successes
        state.ema_trials = total
        state.length = new_k


def expected_throughput(
    cumulative_acceptance: Sequence[float], n: int, cost_ratio: float
) -> float:
    """Expected tokens per round over round cost, in units of the fixed cost.

    cumulative_acceptance[j - 1] is A_j, the probability that at least j draft
    tokens are accepted. E(n) = 1 + sum_{j<=n} A_j; the round costs
    t0 * (1 + cost_ratio * n).
    """
    return (1.0 + sum(cumulative_acceptance[:n])) / (1.0 + cost_ratio * n)


def best_verify_length(
    cumulative_acceptance: Sequence[float],
    cost_ratio: float,
    min_tokens: int,
    max_tokens: int,
) -> int:
    """argmax_n E(n) / (t0 + c * n); the shorter length wins a tie."""
    return max(
        range(min_tokens, max_tokens + 1),
        key=lambda n: (
            expected_throughput(cumulative_acceptance, n, cost_ratio),
            -n,
        ),
    )


@dataclass
class ThroughputState:
    successes: list[float]
    trials: list[float]
    length: int


class ThroughputController:
    """Throughput-optimal verification length from per-position acceptance.

    Each draft position j keeps an EMA of its conditional acceptance
    c_j = P(position j accepted | positions < j accepted); A_j = prod_{i<=j} c_i.
    A round that verified n drafts says nothing about positions beyond n, so
    those continue with the last observed conditional acceptance.
    """

    def __init__(self, plan: AdaptiveVerifyPlan) -> None:
        self._plan = plan
        self._settings = plan.settings
        self._cost_ratio = required_verify_cost_ratio(plan.settings)
        self.initial_length = plan.initial_tokens
        self.max_length = plan.max_tokens

    def new_state(self) -> ThroughputState:
        settings = self._settings
        count = self._plan.max_tokens
        return ThroughputState(
            successes=[settings.prior_conditional_acceptance * settings.prior_weight]
            * count,
            trials=[settings.prior_weight] * count,
            length=self.initial_length,
        )

    def _cumulative_acceptance(
        self, state: ThroughputState, observed_tokens: int
    ) -> list[float]:
        conditional = [s / t for s, t in zip(state.successes, state.trials)]
        carried = conditional[observed_tokens - 1]
        cumulative: list[float] = []
        product = 1.0
        for position, value in enumerate(conditional, start=1):
            product *= value if position <= observed_tokens else carried
            cumulative.append(product)
        return cumulative

    def update(
        self,
        state: ThroughputState,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> None:
        plan, settings = self._plan, self._settings
        alpha = settings.ema_alpha
        observed_tokens = min(num_draft_tokens, plan.max_tokens)
        # Position j is tried only if all earlier drafts were accepted.
        for position in range(1, observed_tokens + 1):
            if num_accepted_tokens < position - 1:
                break
            accepted = 1.0 if num_accepted_tokens >= position else 0.0
            index = position - 1
            state.trials[index] = (1.0 - alpha) * state.trials[index] + alpha
            state.successes[index] = (1.0 - alpha) * state.successes[
                index
            ] + alpha * accepted

        # Like 1Cat: a fully accepted round probes one draft further, so the
        # estimate for the next position is observed instead of extrapolated.
        if num_accepted_tokens >= num_draft_tokens and state.length < plan.max_tokens:
            state.length += 1
            return
        cumulative = self._cumulative_acceptance(state, observed_tokens)
        cost_ratio = self._cost_ratio
        best = best_verify_length(
            cumulative, cost_ratio, plan.min_tokens, plan.max_tokens
        )
        if expected_throughput(cumulative, best, cost_ratio) > (
            1.0 + settings.switch_margin
        ) * expected_throughput(cumulative, state.length, cost_ratio):
            state.length = best


@dataclass
class FixedState:
    length: int


class FixedLengthController:
    """Calibration: every round verifies the same number of draft tokens."""

    def __init__(self, plan: AdaptiveVerifyPlan) -> None:
        self.initial_length = plan.initial_tokens
        self.max_length = plan.max_tokens

    def new_state(self) -> FixedState:
        return FixedState(self.initial_length)

    def update(
        self, state: FixedState, num_draft_tokens: int, num_accepted_tokens: int
    ) -> None:
        pass


def build_verify_controller(plan: AdaptiveVerifyPlan) -> VerifyLengthController:
    """The one dispatch point from the resolved rule to its controller."""
    if plan.rule == "mtp_bands":
        return MtpBandController(plan)
    if plan.rule == "throughput":
        return ThroughputController(plan)
    return FixedLengthController(plan)
