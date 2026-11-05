# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_PREFILL_SKIP_ROWS: skip the layers after the last KV source in
early prefill chunks.

Layers 21-39 of DeepSeek-V4.1 own no compressed KV and no index keys: layer
20 is the last ``kv_source_layer_ids`` entry, and layers 21-39 read its keys
(the reindex layers 24/28/32/36 score layer 20's index keys with their own
queries). The only per-token state of layers 21-39 is a 256-row window ring
each, and layer 39's output is read only at the sampled rows. So a prefill
chunk that ends at least K rows before the end of the prompt needs layers
0-20 (their KV) and layer 20's attention (its compressed KV, index keys,
window cache and anchor ring), and nothing after it.

The cone. Let s be the first position at which layers 21+ run again after
skipped chunks (the first executed chunk's start; the rings of layers 21+
are zeroed on every skipped step, so rows before s read zeros). Layer 20 is
exact everywhere. Layer L's attention reads a 128-row window (the row and
127 before it), so the inputs of layer L (21 <= L <= 39) are exact for
positions p >= s + 127 * (L - 21), and layer 39's output for p >= s + 2413.
With s <= N - K (N = prompt length) and K >= 2560:

- the last prompt row N - 1 (the sampled row) is exact (K - 1 >= 2413);
- the window keys that decode reads, positions >= N - 127 at layers up to
  39 (inputs exact from s + 2286), are exact;
- a same-session hit within the ring reach R (``_V41_RING_REACH`` in
  kv_cache_manager.py, at most 128) of the last processed position reads
  ring rows >= processed - (R + 127) >= processed - 255, exact when the
  request ended at least 2560 rows after the skip boundary (see
  ``session_record_ok``);
- the DSpark drafter attends a 128 + n window of its own context rows, built
  from the aux taps at layers 37-39 (exact from s + 2032 at the latest).

"Exact" means the same inputs. Every executed kernel keeps its shape (the
chunk still has all its rows), so a row's result is bitwise the baseline's
when the kernels compute rows independently; the GPU greedy check confirms
it (and the near-tie gate applies if it does not hold).

Skipped steps return zeros for the hidden rows and the aux taps; a partial
prefill's sample is discarded.

Where the state of layers 21-39 lives. Only in ``window_ring`` (one
[256, 512] FP16 GPU buffer per layer, global to the stage, not per request
and not in paged blocks). Layer 20's paged window cache, its packed store
(host compressed latents, GPU hot ring, GPU index keys) and the drafter's
paged window cache are written on every step, skipped or not. Prefix
caching on V4.1 requires VLLM_V41_SESSION_PREFIX=1 (model init refuses it
otherwise), and the scheduler admits a hit only if it continues the last
freed sequence within the ring reach R rows of its last processed
position (``KVCacheManager._v41_session_hit``); ``restore_prefix`` and the
hot refill rewind nothing in the rings. So a later request whose prompt
shares only a shorter prefix P << N with a skipped prompt gets no hit and
recomputes from 0. The one new case is a request freed inside or just past
its skipped chunks (abort, preemption): ``session_record_ok`` drops its
record, so no hit is admitted against its rings.

Fail closed: the skip never runs for pooling models, speculative methods
other than DSpark (whose drafter reads only a 128 + n window), KV/EC
transfer connectors, prompt logprobs (also behind echo with logprobs) or
routed-expert capture; each reason is logged once.

``K = 3072`` matches philbert440/deepseek-v4.1-v100 ``VLLM_DS41_REPLAY_ROWS``.
"""

from __future__ import annotations

from collections.abc import Iterable

import vllm.envs as envs

KNOB = "VLLM_V41_PREFILL_SKIP_ROWS"
# 20 window reaches of 128 rows (layers 21..39 plus the ring margin).
MIN_ROWS = 2560


def skip_rows() -> int:
    """K from the environment; 0 = off. A value from 1 to MIN_ROWS - 1 is an
    error, not a silent fallback, so a typo cannot leave a wrong cone."""
    value = (envs.VLLM_V41_PREFILL_SKIP_ROWS or "0").strip()
    try:
        rows = int(value)
    except ValueError:
        raise ValueError(f"{KNOB} must be an integer, got {value!r}") from None
    if rows < 0 or 0 < rows < MIN_ROWS:
        raise ValueError(f"{KNOB} must be 0 (off) or >= {MIN_ROWS}, got {rows}")
    return rows


def skip_step(
    num_computed: Iterable[int],
    num_scheduled: Iterable[int],
    num_prompt: Iterable[int],
    rows: int,
) -> bool:
    """Whether this step may skip the layers after the last KV source.

    Every request with scheduled rows must end this step at least ``rows``
    rows before the end of its prompt (c + n <= p - rows). A step with no
    scheduled rows never skips.
    """
    if rows <= 0:
        return False
    any_rows = False
    for c, n, p in zip(num_computed, num_scheduled, num_prompt):
        if n <= 0:
            continue
        any_rows = True
        if c + n > p - rows:
            return False
    return any_rows


# Speculative methods whose drafter reads only a window of context rows
# (DSpark: 128 + n rows of its own paged window cache). EAGLE-style drafters
# and hidden-state extraction read every prompt row, so they disable the skip.
WINDOWED_SPEC_METHODS = (None, "dspark")


def static_block_reasons(
    *,
    pooling: bool,
    spec_method: str | None,
    kv_transfer: bool,
    ec_transfer: bool,
) -> list[str]:
    """Configuration-level reasons the skip may never run (fail closed):
    every one of these needs the hidden rows of layers 21-39 or the head at
    every prompt position, or exports state the skip leaves unwritten."""
    reasons = []
    if pooling:
        reasons.append("pooling model (all-row hidden states)")
    if spec_method not in WINDOWED_SPEC_METHODS:
        reasons.append(f"speculative method {spec_method!r} reads every context row")
    if kv_transfer:
        reasons.append("KV transfer connector")
    if ec_transfer:
        reasons.append("EC transfer connector")
    return reasons


def step_block_reasons(*, prompt_logprobs: bool, routed_experts: bool) -> list[str]:
    """Per-step reasons (fail closed): prompt logprobs (also behind
    echo=True with logprobs) need logits at every prompt row; routed-expert
    capture records routing of every row at every layer."""
    reasons = []
    if prompt_logprobs:
        reasons.append("prompt logprobs requested")
    if routed_experts:
        reasons.append("routed-expert capture")
    return reasons


class SkipCounter:
    """Per-request count of skipped prefill chunks (one line per request
    that skipped any): reported at the request's first executed chunk, or
    when it leaves the batch while still skipping."""

    def __init__(self):
        self.counts: dict[str, list[int]] = {}

    def step(self, skipped: bool, req_ids, num_scheduled) -> list[str]:
        lines = []
        for req_id, n in zip(req_ids, num_scheduled):
            if n <= 0:
                continue
            if skipped:
                entry = self.counts.setdefault(req_id, [0, 0])
                entry[0] += 1
                entry[1] += n
            elif req_id in self.counts:
                chunks, rows = self.counts.pop(req_id)
                lines.append(
                    f"request {req_id}: skipped layers 21-39 for {chunks} "
                    f"prefill chunks ({rows} rows)"
                )
        present = set(req_ids)
        for req_id in [r for r in self.counts if r not in present]:
            chunks, rows = self.counts.pop(req_id)
            lines.append(
                f"request {req_id}: left while skipping after {chunks} "
                f"prefill chunks ({rows} rows)"
            )
        return lines


def session_record_ok(num_computed: int, num_prompt: int, rows: int) -> bool:
    """Whether a freed request's window rings may serve a same-session hit.

    With the knob on, the rings of layers 21+ hold exact rows only when the
    request processed at least MIN_ROWS rows past its skip boundary
    (num_prompt - rows); a request that finished its prompt always did,
    since rows >= MIN_ROWS. Conservative: a request that never skipped but
    stopped early in a short prompt is also refused (a cache miss only).
    """
    if rows <= 0:
        return True
    return num_computed >= num_prompt or num_computed >= num_prompt - rows + MIN_ROWS
