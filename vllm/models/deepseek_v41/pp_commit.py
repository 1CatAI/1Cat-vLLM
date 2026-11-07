# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When the first V4.1 pipeline stage commits its per-step state.

The target's committed state (Engram hash history, ratio-2 compressor states)
lives on the first stage. Every forward that hashes is committed exactly once,
before the next forward hashes. The device kernels trap a missing commit, a
double commit and a wrong count (``_hash_request`` / ``_commit_history`` in
``layers/deepseek_v41_engram.py``: the next forward must start at the
committed position).

On one stage the runner commits after sampling. A first stage that is not the
last never samples, so this class decides when it commits:

- a prefill forward commits at once (the device selects ``pending_length``),
  so the next chunk can enter the stage while this one is still downstream;
- a decode forward is armed, then committed by whichever comes first:
  - the sampled counts that the last stage broadcasts under async scheduling
    with drafts (``take_sampled``, through the runner's sampled-count hook);
  - at the start of the next step, the request's ``num_computed_tokens``
    delta, which is the accepted drafts plus the bonus token
    (``before_forward``).

A request that ends, or is replaced, while a forward is armed drops it: the
next request starts at position 0 or after ``restore_prefix``, and both reset
the device protocol. Pure Python; the state is one request.
"""


class StageCommit:
    def __init__(self) -> None:
        # (request id, num_computed_tokens of the armed decode forward)
        self._armed: tuple[str, int] | None = None

    @property
    def armed(self) -> bool:
        return self._armed is not None

    def after_forward(self, req_ids, num_computed, num_prompt) -> bool:
        """After a forward. True: commit it now (a prefill forward).

        Uses the runner's prefill rule for the attention metadata
        (``is_prefilling = num_computed < num_prompt``), so the device selector
        ``pending_prefill`` agrees.
        """
        if len(req_ids) != 1:
            raise ValueError("V4.1 stage commit requires exactly one request")
        if self._armed is not None:
            raise RuntimeError(
                "V4.1 PP stage 1: the previous decode forward was never committed"
            )
        if num_computed[0] < num_prompt[0]:
            return True
        self._armed = (req_ids[0], int(num_computed[0]))
        return False

    def take_sampled(self) -> bool:
        """Sampled counts arrived. True if they commit an armed forward."""
        armed = self._armed is not None
        self._armed = None
        return armed

    def before_forward(self, req_ids, num_computed) -> int | None:
        """Before the next forward: the deferred count, or None."""
        if self._armed is None:
            return None
        req_id, before = self._armed
        self._armed = None
        if len(req_ids) != 1 or req_ids[0] != req_id:
            return None
        count = int(num_computed[0]) - before
        if count < 1:
            raise RuntimeError(
                f"V4.1 PP stage 1: request {req_id} moved from {before} to "
                f"{num_computed[0]} computed tokens; a decode step commits >= 1"
            )
        return count

    def reset(self) -> None:
        self._armed = None
