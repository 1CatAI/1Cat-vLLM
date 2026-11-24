# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Batch-aware draft width of Inkling2 DSpark (light module: imported by the
scheduler, the CUDA graph manager and the speculator).

The target verifies `batch x (1 + width)` tokens per step, which its graphs and
TurboMind MoE paths bound (compact grouped path: <= 10 tokens; graph-safe:
<= 18), and whose cost grows with the verified tokens. The draft always
proposes the checkpoint's 7 tokens; verifying a prefix of them is exact.

`VLLM_INKLING2_SM70_DSPARK_WIDTHS` is a comma list of widths for batches of
1, 2, 3, ... decode requests; the last entry covers larger batches. Default
"3,3,0" (was "4,3,0"; re-measured with the one-launch kernels of 23 Nov, on
TP4, one stream, 23 prompts of code / chat: 3,3,0 196 tok/s, 4,3,0 190, 2,2,0 190, 5,3,0 184; T=1 182 / 174 / 177 /
166): the verify step stays cheap up to ~8 tokens
and falls off a cliff beyond (compact MoE path / graph shapes), so B=1 verifies
4 tokens, B=2 verifies 8 (width 3 + 1 each) and larger batches decode without drafts (their draft
step is skipped; only the draft KV is kept current). "4,3,1,1,0" is faster in
steady batches of 3-4 (+20-35%) but lost 7% in a churning 20-request test.
`VLLM_INKLING2_SM70_DSPARK_WIDTHS_FILE` (benchmark aid) re-reads the table from
a file at every scheduling step.
"""

import os

DEFAULT_WIDTHS = "3,3,0"


def parse_dspark_widths(max_width: int, text: str | None = None) -> list[int]:
    if text is None:
        text = os.environ.get("VLLM_INKLING2_SM70_DSPARK_WIDTHS", DEFAULT_WIDTHS)
    widths = [min(max(int(w), 0), max_width) for w in text.split(",") if w.strip()]
    return widths or [max_width]


class DSparkWidths:
    def __init__(self, max_width: int) -> None:
        self.max_width = max_width
        self.table = parse_dspark_widths(max_width)
        self.path = os.environ.get("VLLM_INKLING2_SM70_DSPARK_WIDTHS_FILE")
        # Benchmark aid, set from the widths file: "a[;key=value...]" switches the per-request adaptive
        # width on (keys: cost, decay, n, min, probe; see DSparkAdaptive), "a;3,3,0" gives its table.
        self.live_adaptive = False
        self.adaptive_obj = (
            None  # the scheduler's DSparkAdaptive, to configure from the file
        )
        self._file_text: str | None = None

    def _read_file(self) -> None:
        try:
            with open(self.path) as f:
                text = f.read().strip()
        except OSError:
            return
        if text == self._file_text:
            return
        self._file_text = text
        try:
            parts = [p.strip() for p in text.split(";") if p.strip()]
            self.live_adaptive = bool(parts) and parts[0].lower().startswith("a")
            table = [p for p in parts if not p.lower().startswith("a") and "=" not in p]
            if parts and parts[0].lower().startswith("a") and len(parts[0]) > 1:
                table.insert(0, parts[0][1:])
            if table:
                self.table = parse_dspark_widths(self.max_width, table[0])
            elif self.live_adaptive:
                self.table = parse_dspark_widths(self.max_width, DEFAULT_WIDTHS)
            else:
                self.table = parse_dspark_widths(self.max_width, text)
            if self.adaptive_obj is not None:
                self.adaptive_obj.configure(
                    dict(p.split("=", 1) for p in parts if "=" in p)
                )
        except ValueError:
            pass

    def width(self, num_decode_reqs: int) -> int:
        if self.path:
            self._read_file()
        return self.table[min(max(num_decode_reqs, 1), len(self.table)) - 1]


# ms of one step that verifies 1 + k tokens of one request, k = 0..7 (TP4, V100, one stream; spd2-adapt measured
# with rbench.py, server mode, round 3 merged kernels; see work/spd2-adapt/REPORT.md). Only the ratios matter.
DEFAULT_STEP_MS = "6.1,7.96,8.84,9.44,10.5,11.3,11.75,12.2"

# Per-position acceptance prior (P[draft j accepted | drafts 1..j-1 accepted]) of the 23-prompt mix, TP4.
PRIOR_Q = (0.65, 0.73, 0.74, 0.62, 0.62, 0.62, 0.62)


class DSparkAdaptive:
    """Draft width of a lone decode request from its own acceptance history.

    The width that pays depends on the text: copying a file back (code edits,
    tool calls quoting their results) accepts 6+ drafts in a row, free prose
    about 2. Per request the hazard of every draft position j is kept,
    q_j = P(draft j accepted | 1..j-1 accepted), as exponentially decayed counts
    that respect censoring (a step of width k tells about positions 1..min(a+1, k)
    only, so a small width cannot trap the estimate). A position that was never
    reached takes its predecessor's q scaled by the prior's own ratio, so a
    request that accepts everything at width 3 is tried at 5 and then 7. The width
    k maximising (1 + sum_{j<=k} q_1..q_j) / step_ms[k] is used, with a 3 %
    margin against chatter. At width 0 nothing is observed, so every `probe`
    steps one draft is tried. Exact: only how many drafted tokens the target
    verifies changes.

    `VLLM_INKLING2_SM70_DSPARK_ADAPTIVE=1` turns it on (default 1; 0: the
    batch-size table only), `..._DSPARK_STEP_MS` (the cost table), `..._DSPARK_PROBE`,
    `..._DSPARK_ADAPT_DECAY` (per step, 0.95), `..._DSPARK_ADAPT_PRIOR_N` (weight of
    the prior in steps, 5), `..._DSPARK_ADAPT_HYST` (margin before a switch, 0.08), `..._DSPARK_ADAPT_MIN` (smallest width, 1: width 0 gives no feedback and a streak of misses once parked a request there for good).
    """

    def __init__(self, max_width: int) -> None:
        env = os.environ.get
        self.env_enabled = env("VLLM_INKLING2_SM70_DSPARK_ADAPTIVE", "1") == "1"
        self.widths = None  # DSparkWidths: its file may switch the adaptive width on
        self.max_width = max_width
        text = env("VLLM_INKLING2_SM70_DSPARK_STEP_MS", DEFAULT_STEP_MS)
        cost = [float(c) for c in text.split(",") if c.strip()]
        while len(cost) < max_width + 1:
            cost.append(cost[-1] * 1.04)
        self.cost = cost[: max_width + 1]
        self.probe = max(int(env("VLLM_INKLING2_SM70_DSPARK_PROBE", "24")), 1)
        self.decay = float(env("VLLM_INKLING2_SM70_DSPARK_ADAPT_DECAY", "0.95"))
        self.prior_n = float(env("VLLM_INKLING2_SM70_DSPARK_ADAPT_PRIOR_N", "5.0"))
        self.min_width = max(int(env("VLLM_INKLING2_SM70_DSPARK_ADAPT_MIN", "1")), 0)
        self.hyst = float(env("VLLM_INKLING2_SM70_DSPARK_ADAPT_HYST", "0.08"))
        prior = list(PRIOR_Q) + [PRIOR_Q[-1]] * max(max_width - len(PRIOR_Q), 0)
        self._prior = prior[:max_width]
        self._defaults = (
            list(self.cost),
            self.decay,
            self.prior_n,
            self.min_width,
            self.probe,
            self.hyst,
        )
        # request id -> [reach[1..K], accepted[1..K], width, steps since a draft]
        self.state: dict[str, list] = {}

    @property
    def enabled(self) -> bool:
        # A widths file (benchmark aid) overrides the environment: "3,3,0" is the fixed table, "a..." adaptive.
        if self.widths is not None and self.widths._file_text is not None:
            return self.widths.live_adaptive
        return self.env_enabled

    def configure(self, params: dict[str, str]) -> None:
        """Benchmark aid (widths file): cost / decay / n / min / probe; unnamed ones go back to their defaults."""
        self.cost, self.decay, self.prior_n = (
            list(self._defaults[0]),
            self._defaults[1],
            self._defaults[2],
        )
        self.min_width, self.probe, self.hyst = (
            self._defaults[3],
            self._defaults[4],
            self._defaults[5],
        )
        try:
            if "cost" in params:
                cost = [float(c) for c in params["cost"].split(",") if c.strip()]
                while len(cost) < self.max_width + 1:
                    cost.append(cost[-1] * 1.04)
                self.cost = cost[: self.max_width + 1]
            if "decay" in params:
                self.decay = float(params["decay"])
            if "n" in params:
                self.prior_n = float(params["n"])
            if "min" in params:
                self.min_width = max(int(params["min"]), 0)
            if "h" in params:
                self.hyst = float(params["h"])
            if "probe" in params:
                self.probe = max(int(params["probe"]), 1)
        except ValueError:
            pass

    def hazards(self, reach: list[float], acc: list[float]) -> list[float]:
        q: list[float] = []
        prev, prev_prior = 1.0, 1.0
        for j in range(self.max_width):
            ext = min(prev * self._prior[j] / prev_prior, 0.98)
            qj = (acc[j] + self.prior_n * ext) / (reach[j] + self.prior_n)
            qj = min(qj, 0.99)
            q.append(qj)
            prev, prev_prior = qj, self._prior[j]
        return q

    def rates(self, q: list[float]) -> list[float]:
        """Expected tokens per ms at width k = 0..max_width."""
        tokens, term = 1.0, 1.0
        out = [tokens / self.cost[0]]
        for k in range(1, self.max_width + 1):
            term *= q[k - 1]
            tokens += term
            out.append(tokens / self.cost[k])
        return out

    def best(self, q: list[float]) -> int:
        rates = self.rates(q)
        lo = self.min_width
        return max(range(lo, self.max_width + 1), key=lambda k: rates[k])

    def feedback(
        self, req_id: str, num_draft_tokens: int, num_accepted_tokens: int
    ) -> None:
        if not self.enabled or num_draft_tokens <= 0:
            return
        state = self.state.get(req_id)
        if state is None:
            n = self.max_width
            state = self.state[req_id] = [
                [0.0] * n,
                [0.0] * n,
                float(num_draft_tokens),
                0.0,
            ]
        reach, acc = state[0], state[1]
        decay = self.decay
        for j in range(self.max_width):
            reach[j] *= decay
            acc[j] *= decay
        k = min(num_draft_tokens, self.max_width)
        a = min(num_accepted_tokens, k)
        for j in range(min(a + 1, k)):
            reach[j] += 1.0
            if j < a:
                acc[j] += 1.0
        q = self.hazards(reach, acc)
        rates = self.rates(q)
        cur = int(state[2])
        best = self.best(q)
        if best != cur and rates[best] > (1.0 + self.hyst) * rates[cur]:
            state[2] = float(best)
        state[3] = 0.0

    def width(self, req_id: str, default: int) -> int:
        """Width of the next proposal of a lone decode request."""
        state = self.state.get(req_id) if self.enabled else None
        if state is None:
            return default
        k = int(state[2])
        if k == 0:
            state[3] += 1.0
            if state[3] >= self.probe:
                state[3] = 0.0
                return 1
        return k

    def forget(self, live_req_ids) -> None:
        if len(self.state) > len(live_req_ids):
            for req_id in list(self.state):
                if req_id not in live_req_ids:
                    del self.state[req_id]
