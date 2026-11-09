# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time-credit phase allocator: keep decode and prefill in separate phases.

Purely a scheduler-layer time-credit allocator; it makes no attempt to
partition SMs between phases.

An earlier design partitioned SMs with CUDA Green Contexts so prefill and
decode could run concurrently. That is not viable on multi-GPU tensor parallel:
a Green Context is an independent CUDA context and cannot see the model weights
or participate in the all-reduce, so on TP4 the two phases cannot coexist at
all. This file therefore contains only the scheduling half.

Why fence *time* rather than *tokens*
-------------------------------------
A mixed batch (decoders + a prefill chunk) can never replay a FULL CUDA graph:
``get_uniform_decode_token_count`` returns ``None`` as soon as ``has_prefill``
is true, so the handful of decode tokens in that step falls back to eager
execution. Measured on this deployment (TP4, 1024-token chunks):

    pure prefill, 1 req, 1024 tok : cpu 232 ms / gpu 268 ms
    mixed,        3 req, 1040 tok : cpu 343 ms / gpu 350 ms
                                    (+108 ms host, +82 ms GPU for 16 decode tokens)

Fencing the *token* budget cannot fix this, because a prefill step costs almost
the same whether it carries 20 or 1024 tokens -- there is a fixed ~230 ms host
dispatch floor. Earlier attempts (the "fence v1-v4" series) all fenced tokens
and all failed for exactly this reason. Fencing *time* -- running whole steps as
one phase or the other -- lets the decode phase replay its FULL graph (measured
23 ms cpu for 16 tokens, vs 343 ms when a prefill shares the step).

This module owns only the bookkeeping. It never touches the model, the GPU, or
the scheduler's data structures, so it is unit-testable in isolation.

Guardrails against the MixedPrefillBudget failure mode
------------------------------------------------------
``MixedPrefillBudget`` can collapse to its 16-token floor and never recover.
Three specific mistakes are designed out here:

1. **Only the phase's own elapsed time is fed back.** Its predecessor divided a
   whole step that also contained concurrent decode work by the prefill tokens
   alone, inflating the apparent per-token cost roughly 10x.
2. **The decision compares against an absolute zero**, never against a quantity
   derived from the controller's own output. The predecessor's gate
   ``prefill_tokens < budget * 0.75`` compared against its *current* budget, so
   once it reached the floor the gate always passed and the collapse became
   self-sustaining.
3. **Growth and shrink are symmetric, and the floor is an absolute constant.**
   The predecessor capped growth at ``tokens * 1.25`` but left shrink unbounded,
   making the collapse a one-way ratchet.

The chunk size is deliberately *not* configured here. It follows from the Mamba
state-block retention setting (``--prefix-cache-retention-interval``), so this
controller cannot change the batch geometry underneath itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


@dataclass(frozen=True)
class PhaseConfig:
    """Knobs that are safe to tune; everything else is measured at runtime."""

    # Target share of wall-clock time given to the decode phase (0..1).
    #
    # NOTE this is a *share*, not the earn rate. The credit balance is
    # stationary when ``earn_ratio * prefill_ms == decode_ms``, and
    # ``decode_ms / (decode_ms + prefill_ms) == share`` implies
    # ``earn_ratio == share / (1 - share)``. Storing the share (and converting
    # below) keeps this knob in the same units the operator reasons about, and
    # makes the 1/3-share case a 0.5 earn rate rather than a 1/3 one -- using
    # 1/3 directly would settle at a 25% share, not 33%.
    decode_time_share: float = 1.0 / 3.0
    # Hard ceiling on how long a decoder may go unserved, in milliseconds.
    #
    # NOTE this bounds the prefill *streak*, but the wait a decoder actually
    # observes is `max_decode_wait_ms + one prefill step`: the guard can only
    # be evaluated at a step boundary, so a prefill step already in flight has
    # to finish first. Measured on this deployment: 700 + 386 = 1086 ms
    # predicted, 1071 ms observed. Size this against the prefill step, not
    # against the stall target on its own.
    max_decode_wait_ms: float = 700.0
    # Per-step budget for a prefill chunk that runs *alongside* decoders.
    # Retained for `prefill_chunk_cap(coexist=...)`; PD-mini never passes True
    # in the 150 build, so this is inert.
    coexist_chunk_ms: float = 60.0
    # Absolute token floor/ceiling for a prefill chunk. Constants on purpose --
    # deriving them from the controller's own output invites that collapse.
    min_prefill_tokens: int = 1024
    max_prefill_tokens: int = 8192
    # Actual attention block size, injected by the scheduler. The chunk floor
    # must never fall below one block: a sub-block chunk makes the scheduler
    # submit half a block per step, and every such step re-aligns downward and
    # pays the block's fixed cost for half its tokens. On a hybrid Mamba model
    # the platform raises block size to 2048 (see the G29 note), which is above
    # the 1024 default floor, so the floor has to follow the real value rather
    # than stay at 1024.
    block_size: int = 1
    # Drop a feedback interval longer than this: it means the engine sat idle
    # between steps, which says nothing about phase cost.
    max_feedback_ms: float = 3000.0
    # Cap on banked credit, expressed as a multiple of a measured decode step.
    credit_cap_steps: float = 8.0
    # Absolute floor for that cap. Deliberately *independent* of
    # max_decode_wait_ms: when the cap was derived from it, lowering the wait
    # budget to tighten the stall bound also throttled the earn rate and the
    # realised share fell below the configured one. The cap and the stall
    # bound are different concerns and must not share a knob.
    credit_cap_floor_ms: float = 1500.0
    # Below this many prefill tokens a step is dominated by the fixed ~250 ms
    # host dispatch floor, so dividing its wall-clock by its token count gives
    # a meaningless per-token cost (observed: 161 ms/token on a 3-token replay).
    # Such samples are skipped rather than allowed to poison the average.
    min_prefill_tokens_for_ema: int = 256
    ema_alpha: float = 0.3
    # Emit a phase-pattern line every N steps. 0 = silent (the default): this
    # line is diagnostic, and the step profiler is the wrong tool for the same
    # question (it samples 1 step in 40, and asking it for every step forces a
    # device sync that perturbs the cadence being measured).
    log_interval_steps: int = 0
    # How often (in steps) the decode side re-examines *every* pending decoder
    # rather than only the ones FCFS already reached.
    #
    # The problem this solves: schedulers visit `self.running` in arrival order
    # and spend the token budget as they go. A request that just finished its
    # prefill sits at the back of that list, so while a long-generation request
    # at the front keeps consuming the budget, the newcomer is scheduled for
    # zero tokens step after step. To the client it looks like the request is
    # stuck: its prefill is done but no token ever comes out.
    #
    # A periodic sweep re-admits every pending decoder so the newcomer joins the
    # same decode step as the long-running request instead of waiting for it to
    # finish. 0 disables the sweep (pure FCFS), 1 sweeps every step.
    decode_sweep_interval_steps: int = 8
    # Fair share of the decode budget per request.
    #
    # A sweep must *equalise*, not merely admit. Handing the newcomer one token
    # while the incumbent keeps the whole remaining budget still starves it.
    # In practice decode needs ~8 tokens per request (1 + draft), so the budget
    # is almost never the binding constraint for decode -- see `sweep_share`.
    decode_fair_share_on_sweep: bool = True

    @property
    def earn_ratio(self) -> float:
        """Decode-milliseconds earned per prefill-millisecond."""
        share = min(max(self.decode_time_share, 0.0), 0.95)
        return share / (1.0 - share)


class PhaseCreditAllocator:
    """Decide which phase the next step runs, and bookkeep what it cost."""

    def __init__(self, config: PhaseConfig | None = None) -> None:
        self.config = config or PhaseConfig()
        # Decode-side liveness bookkeeping (see decode_sweep_interval_steps).
        # Maps req_id -> number of consecutive decode steps it went unserved.
        self._starved: dict[str, int] = {}
        self._sweep_due = False
        self.credit_ms = 0.0
        # Wall-clock bookkeeping for the feedback loop (seconds, monotonic).
        self._prev_phase: Phase | None = None
        self._prev_ts: float | None = None
        self._prev_did_work = False
        self._prev_prefill_tokens = 0
        # When the decode phase last actually ran, for the starvation guard.
        self._last_decode_ts: float | None = None
        # Set when a chosen phase scheduled nothing; see note_empty_phase_step.
        self._force_flip: Phase | None = None
        # Realised step counts, for tuning. The configured knob is a *time*
        # share, but operators reason in "N decode steps per prefill step", so
        # these are what make the two views reconcilable. Counted here rather
        # than via the step profiler: the profiler samples one step in forty
        # (too coarse to see the pattern) and forces a device sync when asked
        # for every step, which would perturb the very cadence being measured.
        self.decode_steps = 0
        self.prefill_steps = 0
        # Measured, never fitted. Diagnostics and the chunk cap read these.
        self.decode_step_ms: float | None = None
        self.prefill_ms_per_token: float | None = None

    # ---------------- decision ----------------

    def next_phase(
        self,
        *,
        has_decode: bool,
        has_prefill: bool,
        now: float,
        fresh_decode: bool = False,
        has_waiting: bool | None = None,
    ) -> Phase | None:
        """Pick the phase for the step about to be scheduled.

        ``None`` means there is nothing to do. Rule order matters:

        * Emptiness is checked before credit, so a deeply negative credit with
          no prefill work available cannot spin an empty step.
        * Starvation is checked before credit, so a decoder cannot be held off
          indefinitely by a long prefill streak.

        ``has_prefill`` means "running already holds work with prefill left".
        ``has_waiting`` means "requests are queued and would enter if admitted".
        The two are kept apart here even though PD-mini is not ported: the
        distinction is also what tells `next_phase` whether there is any work
        for a future prefill phase to do.
        """
        if has_waiting is None:
            has_waiting = has_prefill
        if not has_decode and not has_prefill and not has_waiting:
            return None
        # Nothing to decode: prefill must run, even with credit exhausted.
        if not has_decode:
            return Phase.PREFILL
        # Nothing to prefill: decode must run, even with credit in debt.
        if not has_prefill and not has_waiting:
            return Phase.DECODE
        # A request has just finished its prefill and is waiting to produce its
        # first token. Every step spent on prefill is a step where that client
        # sees nothing, and a long prefill streak can hold it off for the whole
        # streak even though no phase is starved. Serve decode now.
        #
        # This does not fight the credit loop: it fires once per newly-ready
        # request, then the request joins the normal decode set and the credit
        # takes over again. Repeated firing would mean the request is not being
        # scheduled even in decode steps -- that is the budget-order problem,
        # handled by the sweep, not here.
        if fresh_decode:
            return Phase.DECODE
        # The phase we chose last step scheduled nothing (e.g. every decoder is
        # momentarily ineligible). Flip rather than retry, or the engine spins
        # empty steps while real work waits -- and with no work scheduled the
        # credit never moves, so the spin would be permanent.
        if self._force_flip is not None:
            forced = (
                Phase.PREFILL if self._force_flip is Phase.DECODE else Phase.DECODE
            )
            self._force_flip = None
            return forced
        # A decoder has waited at least max_decode_wait_ms: force decode.
        if (
            self._last_decode_ts is not None
            and (now - self._last_decode_ts) * 1000.0 >= self.config.max_decode_wait_ms
        ):
            return Phase.DECODE
        # Otherwise credit decides. Zero is the tie-break in favour of decode,
        # so an idle-credit system alternates rather than starving either side.
        return Phase.DECODE if self.credit_ms >= 0.0 else Phase.PREFILL

    # ---------------- decode liveness ----------------

    def due_for_sweep(self) -> bool:
        """Whether this decode step should admit every pending decoder.

        Independent of the phase decision: the FCFS budget walk happens inside
        one phase, and that is where a freshly-prefilled request gets skipped.
        """
        return self._sweep_due

    def mark_pending_decoders(self, pending: list[str], served: set[str]) -> None:
        """Update who has been waiting, and arm the next sweep.

        ``pending`` is every request that could decode this step; ``served`` is
        the subset the scheduler actually gave tokens to. Anything pending but
        unserved counts a strike -- after ``decode_sweep_interval_steps``
        strikes the next step opens with a sweep, so a request cannot be
        starved indefinitely by budget order alone.
        """
        interval = self.config.decode_sweep_interval_steps
        alive = set(pending)
        for rid in list(self._starved):
            if rid not in alive:
                # Finished or preempted; forget it rather than leak.
                del self._starved[rid]
        for rid in pending:
            if rid in served:
                self._starved.pop(rid, None)
            else:
                self._starved[rid] = self._starved.get(rid, 0) + 1
        if interval <= 0:
            self._sweep_due = False
        else:
            self._sweep_due = any(n >= interval for n in self._starved.values())

    def sweep_share(self, pending: list[str], budget: int) -> int:
        """Per-request token allowance for an even decode split.

        Returns 0 when this is not a sweep step (caller keeps FCFS behaviour),
        otherwise ``budget // len(pending)`` -- the same channel width every
        pending decoder gets, regardless of who arrived first.

        NOTE: token budget is almost never the binding constraint for decode --
        a spec-decode step asks for 1 + num_draft tokens (~8), so even at 16
        concurrent requests the whole batch needs ~128 of the 2048 budget. The
        real blocker is the phase filter, which drops every decoder from a
        prefill step. This quota only matters for a very wide batch; the phase
        handling is what actually governs liveness.
        """
        if not self._sweep_due or not pending or budget <= 0:
            return 0
        if not self.config.decode_fair_share_on_sweep:
            return 0
        return max(1, budget // len(pending))

    def clear_sweep(self) -> None:
        """Called once a sweep step has been scheduled."""
        self._sweep_due = False

    def note_empty_phase_step(self, phase: Phase) -> None:
        """Record that ``phase`` was chosen but scheduled no tokens.

        The scheduler calls this only for a step that really ran (not an idle
        engine), so it is safe to treat as "this phase has nothing to do now".
        """
        self._force_flip = phase

    # ---------------- feedback ----------------

    def charge_elapsed(self, now: float) -> None:
        """Charge the previous step's wall-clock to the phase that ran it.

        Must be called *before* :meth:`next_phase`. Charging afterwards leaves
        the credit one decision stale, and a prefill step's earnings then only
        become visible two decisions later -- which makes a credit balance that
        has just crossed zero need two consecutive prefill steps to recover,
        burning a whole prefill step per cycle (measured: a 23% decode share
        instead of the requested 33%).
        """
        if self._prev_phase is not None and self._prev_ts is not None:
            if self._prev_did_work:
                dt_ms = (now - self._prev_ts) * 1000.0
                # Idle gaps say nothing about phase cost -- drop them.
                if 0.0 < dt_ms <= self.config.max_feedback_ms:
                    self._apply_feedback(self._prev_phase, dt_ms)
        self._prev_ts = now
        self._prev_did_work = False
        self._prev_prefill_tokens = 0

    def begin_phase(self, phase: Phase, now: float) -> None:
        """Record the phase the step about to run belongs to."""
        self._prev_phase = phase
        if phase is Phase.DECODE or self._last_decode_ts is None:
            # Stamp the first step whatever its phase. Without this the
            # starvation guard stays inert until decode happens to run once,
            # so a burst of prefill steps right after startup could hold
            # decoders off for far longer than max_decode_wait_ms.
            self._last_decode_ts = now

    def observe_step_work(self, *, scheduled_tokens: int, prefill_tokens: int) -> None:
        """Record what the step actually did, once the batch is known."""
        self._prev_did_work = scheduled_tokens > 0
        self._prev_prefill_tokens = prefill_tokens
        if scheduled_tokens > 0:
            if prefill_tokens > 0:
                self.prefill_steps += 1
            else:
                self.decode_steps += 1

    def _apply_feedback(self, phase: Phase, dt_ms: float) -> None:
        cfg = self.config
        if phase is Phase.DECODE:
            self.credit_ms -= dt_ms
            self.decode_step_ms = self._ema(self.decode_step_ms, dt_ms)
        else:
            self.credit_ms += cfg.earn_ratio * dt_ms
            # Only prefill tokens divide prefill time. This is guardrail #1:
            # the step was phase-pure, so its whole duration belongs to them.
            if self._prev_prefill_tokens >= cfg.min_prefill_tokens_for_ema:
                self.prefill_ms_per_token = self._ema(
                    self.prefill_ms_per_token, dt_ms / self._prev_prefill_tokens
                )
        # Symmetric clamp (guardrail #3). The upper bound is how much decode
        # time may be banked while only prefill work exists, which is also the
        # longest decode burst that can follow -- so it must never fall below
        # one wait budget, or the cap silently throttles the configured share.
        # (With a bare 8 * decode_step_ms the cap measured 320 ms, just under
        # the ~326 ms a 653 ms prefill step earns at a 1/3 share, and the
        # realised share collapsed to 27%.) The lower bound is how long a
        # prefill streak may run before decode is forced back in.
        measured = (
            cfg.credit_cap_steps * self.decode_step_ms
            if self.decode_step_ms is not None
            else 0.0
        )
        upper = max(measured, cfg.credit_cap_floor_ms)
        lower = -2.0 * cfg.max_decode_wait_ms
        self.credit_ms = min(upper, max(lower, self.credit_ms))

    def _ema(self, current: float | None, sample: float) -> float:
        if current is None:
            return sample
        a = self.config.ema_alpha
        return (1.0 - a) * current + a * sample

    # ---------------- derived ----------------

    def _chunk_floor(self) -> int:
        """Smallest chunk we may submit: one full attention block."""
        return max(self.config.min_prefill_tokens, self.config.block_size)

    def prefill_chunk_cap(self) -> int:
        """Largest prefill chunk that still fits inside one phase.

        Derived from *measured* ms/token, so it shrinks automatically as the
        context grows (per-token cost rises with KV length). A fixed token cap
        would overrun the phase budget at long context; this cannot.
        """
        cfg = self.config
        if not self.prefill_ms_per_token:
            return cfg.max_prefill_tokens
        budget_ms = cfg.max_decode_wait_ms
        fits = budget_ms / self.prefill_ms_per_token
        floor = self._chunk_floor()
        # Never exceed the configured ceiling even if the floor is above it.
        return max(floor, min(max(cfg.max_prefill_tokens, floor), int(fits)))

    def steps_per_prefill(self) -> float:
        """Realised decode steps per prefill step (0 until both have run)."""
        if not self.prefill_steps:
            return 0.0
        return self.decode_steps / self.prefill_steps

    def stats(self) -> str:
        def fmt(v: float | None, nd: int = 2) -> str:
            return "n/a" if v is None else f"{round(v, nd)}"

        return (
            f"D:P={self.decode_steps}:{self.prefill_steps} "
            f"({self.steps_per_prefill():.2f} dec/pre) "
            f"credit={self.credit_ms:.1f}ms "
            f"dec_step={fmt(self.decode_step_ms)}ms "
            f"pre_ms_per_tok={fmt(self.prefill_ms_per_token, 4)} "
            f"chunk_cap={self.prefill_chunk_cap()}"
        )
