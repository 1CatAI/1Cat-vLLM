# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in wall-time credits for separate resident prefill/decode steps.

Only phase-pure steps train cost estimates. Waiting requests restored from a
prefix or preemption may produce a mixed recovery step; normal cache recovery
takes precedence, and that step is excluded from the feedback loop.
"""

import math
from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


@dataclass(frozen=True)
class PhaseConfig:
    decode_time_share: float = 1 / 3
    max_decode_wait_ms: float = 700.0
    min_prefill_tokens: int = 1024
    max_prefill_tokens: int = 8192
    block_size: int = 1
    max_feedback_ms: float = 3000.0
    credit_cap_steps: float = 8.0
    credit_cap_floor_ms: float = 1500.0
    min_prefill_tokens_for_ema: int = 256
    ema_alpha: float = 0.3
    log_interval_steps: int = 0
    decode_sweep_interval_steps: int = 8
    decode_fair_share_on_sweep: bool = True

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.decode_time_share)
            or not 0 <= self.decode_time_share < 1
        ):
            raise ValueError("decode_time_share must be finite and in [0, 1)")
        for name in ("max_decode_wait_ms", "max_feedback_ms", "credit_cap_floor_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.credit_cap_steps) or self.credit_cap_steps < 0:
            raise ValueError("credit_cap_steps must be finite and non-negative")
        if not math.isfinite(self.ema_alpha) or not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if (
            min(
                self.block_size,
                self.min_prefill_tokens,
                self.min_prefill_tokens_for_ema,
            )
            <= 0
        ):
            raise ValueError("block size and token floors must be positive")
        if self.log_interval_steps < 0 or self.decode_sweep_interval_steps < 0:
            raise ValueError("step intervals must be non-negative")
        if self.chunk_floor > self.chunk_ceiling:
            raise ValueError(
                "prefill bounds must contain at least one full cache block"
            )

    @property
    def earn_ratio(self) -> float:
        return self.decode_time_share / (1 - self.decode_time_share)

    @property
    def chunk_floor(self) -> int:
        return (
            (self.min_prefill_tokens + self.block_size - 1) // self.block_size
        ) * self.block_size

    @property
    def chunk_ceiling(self) -> int:
        return (self.max_prefill_tokens // self.block_size) * self.block_size


class PhaseCreditAllocator:
    def __init__(self, config: PhaseConfig | None = None) -> None:
        self.config = config or PhaseConfig()
        self._starved: dict[str, int] = {}
        self._sweep_due = False
        self.credit_ms = 0.0
        self._prev_phase: Phase | None = None
        self._prev_ts: float | None = None
        self._prev_did_work = False
        self._prev_prefill_tokens = 0
        self._last_decode_ts: float | None = None
        self._force_flip: Phase | None = None
        self.decode_steps = self.prefill_steps = self.mixed_steps = 0
        self.decode_step_ms: float | None = None
        self.prefill_ms_per_token: float | None = None

    def next_phase(
        self,
        *,
        has_decode: bool,
        has_prefill: bool,
        now: float,
        fresh_decode: bool = False,
        has_waiting: bool | None = None,
    ) -> Phase | None:
        if has_waiting is None:
            has_waiting = has_prefill
        prefill_available = has_prefill or has_waiting
        if not has_decode or not prefill_available:
            self._force_flip = None
            return (
                Phase.DECODE
                if has_decode
                else Phase.PREFILL
                if prefill_available
                else None
            )
        # An empty attempt precedes a still-fresh decoder to prevent spinning.
        if self._force_flip is not None:
            forced = Phase.PREFILL if self._force_flip is Phase.DECODE else Phase.DECODE
            self._force_flip = None
            return forced
        if fresh_decode:
            return Phase.DECODE
        if (
            self._last_decode_ts is not None
            and (now - self._last_decode_ts) * 1000 >= self.config.max_decode_wait_ms
        ):
            return Phase.DECODE
        return Phase.DECODE if self.credit_ms >= 0 else Phase.PREFILL

    def due_for_sweep(self) -> bool:
        return self._sweep_due

    def mark_pending_decoders(self, pending: list[str], served: set[str]) -> None:
        interval = self.config.decode_sweep_interval_steps
        self._starved = {
            rid: self._starved.get(rid, 0) + 1 for rid in pending if rid not in served
        }
        self._sweep_due = interval > 0 and any(
            n >= interval for n in self._starved.values()
        )

    def sweep_share(self, pending: list[str], budget: int) -> int:
        if (
            not self._sweep_due
            or not pending
            or not self.config.decode_fair_share_on_sweep
        ):
            return 0
        # Scheduler admission guarantees at least one token per resident slot.
        return budget // len(pending)

    def clear_sweep(self) -> None:
        self._sweep_due = False

    def note_empty_phase_step(self, phase: Phase) -> None:
        self._force_flip = phase

    def charge_elapsed(self, now: float) -> None:
        """Charge before choosing a phase so credit is never a step stale."""
        if (
            self._prev_phase is not None
            and self._prev_ts is not None
            and self._prev_did_work
        ):
            dt_ms = (now - self._prev_ts) * 1000
            if 0 < dt_ms <= self.config.max_feedback_ms:
                self._apply_feedback(self._prev_phase, dt_ms)
        self._prev_ts = now
        self._prev_did_work = False
        self._prev_prefill_tokens = 0

    def begin_phase(self, phase: Phase, now: float) -> None:
        self._prev_phase = phase
        self._prev_ts = now
        if self._last_decode_ts is None:
            self._last_decode_ts = now

    def observe_step_work(self, *, scheduled_tokens: int, prefill_tokens: int) -> None:
        if not 0 <= prefill_tokens <= scheduled_tokens:
            raise ValueError("prefill work must be a subset of scheduled work")
        self._prev_did_work = scheduled_tokens > 0
        self._prev_prefill_tokens = prefill_tokens
        if scheduled_tokens == 0:
            return
        if prefill_tokens < scheduled_tokens:
            self._last_decode_ts = self._prev_ts
        if prefill_tokens == scheduled_tokens:
            self._prev_phase = Phase.PREFILL
            self.prefill_steps += 1
        elif prefill_tokens == 0:
            self._prev_phase = Phase.DECODE
            self.decode_steps += 1
        else:
            self.mixed_steps += 1
            self._prev_did_work = False

    def _apply_feedback(self, phase: Phase, dt_ms: float) -> None:
        cfg = self.config
        if phase is Phase.DECODE:
            self.credit_ms -= dt_ms
            self.decode_step_ms = self._ema(self.decode_step_ms, dt_ms)
        else:
            self.credit_ms += cfg.earn_ratio * dt_ms
            if self._prev_prefill_tokens >= cfg.min_prefill_tokens_for_ema:
                self.prefill_ms_per_token = self._ema(
                    self.prefill_ms_per_token, dt_ms / self._prev_prefill_tokens
                )
        measured = cfg.credit_cap_steps * (self.decode_step_ms or 0)
        self.credit_ms = min(
            max(measured, cfg.credit_cap_floor_ms),
            max(-2 * cfg.max_decode_wait_ms, self.credit_ms),
        )

    def _ema(self, current: float | None, sample: float) -> float:
        return (
            sample
            if current is None
            else (1 - self.config.ema_alpha) * current + self.config.ema_alpha * sample
        )

    def prefill_chunk_cap(self) -> int:
        cfg = self.config
        if not self.prefill_ms_per_token:
            return cfg.chunk_ceiling
        fits = int(cfg.max_decode_wait_ms / self.prefill_ms_per_token)
        aligned = (fits // cfg.block_size) * cfg.block_size
        return max(cfg.chunk_floor, min(cfg.chunk_ceiling, aligned))

    def stats(self) -> str:
        return (
            f"D:P:mixed={self.decode_steps}:{self.prefill_steps}:{self.mixed_steps} "
            f"credit={self.credit_ms:.1f}ms dec_step={self.decode_step_ms}ms "
            f"pre_ms_per_tok={self.prefill_ms_per_token} "
            f"chunk_cap={self.prefill_chunk_cap()}"
        )
