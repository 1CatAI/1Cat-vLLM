# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.metrics.stats import RequestStateStats


def get_request_usage_timings(
    metrics: RequestStateStats | None, *, num_choices: int = 1
) -> dict[str, float] | None:
    """Expose complete single-choice engine timings without mixing clocks.

    Parent outputs currently carry one child's stats, so they do not describe
    a whole multi-choice request. Missing or reordered timestamps likewise
    cannot establish a measured phase duration and are omitted.
    """
    if metrics is None or num_choices != 1:
        return None
    queued = metrics.queued_ts
    scheduled = metrics.scheduled_ts
    first = metrics.first_token_ts
    last = metrics.last_token_ts
    ttft = metrics.first_token_latency
    if not all(math.isfinite(t) for t in (queued, scheduled, first, last, ttft)):
        return None
    if not (0 < queued <= scheduled <= first <= last) or ttft < 0:
        return None
    return {
        "queue_s": scheduled - queued,
        "prefill_s": first - scheduled,
        "decode_s": last - first,
        "inference_s": last - scheduled,
        "ttft_s": ttft,
    }
